import argparse
import json
import time
from collections import Counter
from pathlib import Path

from codes_data.client import ChatClient
from codes_data.io import append_jsonl, read_jsonl, stable_key


PARAPHRASE_SYSTEM = """你是中文Text-to-SQL数据标注员。根据原始问题和SQL，改写出语义完全等价的新问题。
必须保留SQL中的所有查询字段、过滤条件、数值边界、AND/OR作用域、聚合、分组、排序和数量限制。
不得增加、删除或改变任何条件；不得在问题中出现SQL、表名或下划线字段名。
只输出JSON：{"query":"改写后的问题"}。
""".strip()

REVIEW_SYSTEM = """你是Text-to-SQL语义质检员。检查候选问题是否与原始问题和SQL严格等价。
逐项检查SELECT字段、WHERE条件和值、AND/OR作用域、聚合、分组、排序和LIMIT。
只要有任何信息遗漏、新增或含义改变，consistent必须为false。
只输出JSON：{"consistent":true或false,"issues":["问题"]}。
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Create two reviewed paraphrases per held-out validation sample")
    parser.add_argument("--validation", default="outputs/augmentation_v2/manual_validation.jsonl", type=Path)
    parser.add_argument("--base-train", default="outputs/query_only_codes3b_v2/hard_train.jsonl", type=Path)
    parser.add_argument("--output-dir", default="outputs/validation_augmentation_v3", type=Path)
    parser.add_argument("--per-sample", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:6006/v1")
    parser.add_argument("--model", default="Qwen3-14B")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-attempts-per-sample", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.per_sample != 2:
        parser.error("this stage requires exactly two paraphrases per validation sample")

    validation = read_jsonl(args.validation)
    base_train = read_jsonl(args.base_train)
    if len(validation) != 160:
        parser.error(f"expected 160 validation samples, got {len(validation)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = args.output_dir / "validation_augmented_320.jsonl"
    combined_path = args.output_dir / "hard_train_plus_validation_augmented.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    if not args.resume and (augmented_path.exists() or failures_path.exists()):
        parser.error("output exists; choose a new directory or pass --resume")

    completed = read_jsonl(augmented_path) if augmented_path.exists() else []
    by_parent = Counter(str(row["parent_id"]) for row in completed)
    seen_queries = {str(row["query"]).strip() for row in validation + base_train + completed}
    client = ChatClient(args.base_url, args.model, args.timeout)
    started = time.perf_counter()
    attempts = 0
    for position, seed in enumerate(validation):
        parent_id = str(seed.get("id", position))
        seed_attempts = 0
        while by_parent[parent_id] < args.per_sample:
            if seed_attempts >= args.max_attempts_per_sample:
                break
            attempts += 1
            seed_attempts += 1
            try:
                response = client.ask_json(
                    PARAPHRASE_SYSTEM,
                    f"原始问题：{seed['query']}\nSQL：{seed['sql']}",
                    max_tokens=700,
                    temperature=0.8,
                )
                query = str(response["query"]).strip()
                if len(query) < 8 or query in seen_queries or "SELECT " in query.upper():
                    raise ValueError("invalid, copied, or duplicate query")
                review = client.ask_json(
                    REVIEW_SYSTEM,
                    f"原始问题：{seed['query']}\n候选问题：{query}\nSQL：{seed['sql']}",
                    max_tokens=400,
                    temperature=0,
                )
                if review.get("consistent") is not True:
                    raise ValueError(f"semantic review rejected: {review.get('issues', [])}")
                item = {
                    "augmentation_id": stable_key("validation_v3", parent_id, by_parent[parent_id], query),
                    "parent_id": seed.get("id", position),
                    "variant": by_parent[parent_id] + 1,
                    "query": query,
                    "sql": seed["sql"],
                    "table": seed.get("table"),
                    "source": "sql_to_question",
                    "origin": "held_out_validation_paraphrase",
                    "semantic_review": review,
                }
                append_jsonl(augmented_path, item)
                completed.append(item)
                by_parent[parent_id] += 1
                seen_queries.add(query)
                print(f"PROGRESS {len(completed)}/{len(validation) * args.per_sample} parent={parent_id}", flush=True)
            except Exception as exc:
                append_jsonl(failures_path, {
                    "parent_id": seed.get("id", position),
                    "attempt": attempts,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    expected = len(validation) * args.per_sample
    complete = len(completed) == expected and all(by_parent[str(row.get("id", i))] == 2 for i, row in enumerate(validation))
    if complete:
        with combined_path.open("w", encoding="utf-8") as handle:
            for row in base_train + completed:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "validation_seeds": len(validation),
        "paraphrases_per_seed": args.per_sample,
        "expected_augmentation": expected,
        "generated_augmentation": len(completed),
        "base_training_samples": len(base_train),
        "combined_training_samples": len(base_train) + len(completed) if complete else 0,
        "attempts_this_run": attempts,
        "seconds_this_run": round(time.perf_counter() - started, 3),
        "validation_leakage": True,
        "complete": complete,
        "augmentation_file": str(augmented_path),
        "combined_file": str(combined_path) if complete else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit("target not reached; inspect failures.jsonl and resume")


if __name__ == "__main__":
    main()
