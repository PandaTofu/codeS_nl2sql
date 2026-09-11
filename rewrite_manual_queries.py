import argparse
import json
import time
from collections import Counter
from pathlib import Path

from codes_data.client import ChatClient
from codes_data.io import append_jsonl, read_jsonl, stable_key


REWRITE_SYSTEM = """你是中文数据标注员。对用户问题做一次轻量同义改写，语义必须与原问题完全一致。
可以替换同义词、调整语序或句式，但必须原样保留品牌、型号、专有名词、枚举值、数字、日期、单位、比较边界、查询字段、AND/OR关系、聚合、分组、排序和数量限制。
不得增加或删除任何要求，不得在问题中出现SQL、表名或下划线字段名。改写后不得与原文完全相同。
只输出JSON：{"query":"改写后的问题"}。
""".strip()

REVIEW_SYSTEM = """你是中文语义质检员。只判断改写问题与原始问题是否严格等价，不要推测或补充原问题没有明说的需求。
任何查询内容、条件、值、比较边界、AND/OR作用域、聚合、分组、排序或数量限制被遗漏、新增或改变时，consistent必须为false。
“详细信息”、“全部配置”等概括表达在改写后保持原意即可，不得因为它没有枚举具体字段而判错。
只输出JSON：{"consistent":true或false,"issues":["问题"]}。
""".strip()


def pair_key(query, sql):
    return str(query).strip(), str(sql).strip()


def main():
    parser = argparse.ArgumentParser(description="Rewrite manual queries and replace manual rows in a combined SFT dataset")
    parser.add_argument("--manual", default="data/manual_raw_train.jsonl", type=Path)
    parser.add_argument("--combined", default="data/codes_sft_train.jsonl", type=Path)
    parser.add_argument("--output-dir", default="outputs/manual_query_rewrite_v1", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:6006/v1")
    parser.add_argument("--model", default="Qwen3-14B")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-attempts-per-sample", type=int, default=10)
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manual = read_jsonl(args.manual)
    combined = read_jsonl(args.combined)
    ids = [str(row["id"]) for row in manual]
    if len(ids) != len(set(ids)):
        parser.error("manual dataset contains duplicate ids")
    original_by_id = {str(row["id"]): row for row in manual}
    ids_by_pair = {}
    for row in manual:
        key = pair_key(row["query"], row["sql"])
        if key in ids_by_pair:
            parser.error("manual dataset contains duplicate query+sql pairs")
        ids_by_pair[key] = str(row["id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rewritten_path = args.output_dir / "manual_raw_train_query_augmented.jsonl"
    replaced_path = args.output_dir / "codes_sft_train_query_replaced.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    if not args.resume and (rewritten_path.exists() or failures_path.exists()):
        parser.error("output exists; choose a new directory or pass --resume")
    completed = read_jsonl(rewritten_path) if rewritten_path.exists() else []
    rewritten_by_id = {str(row["id"]): row for row in completed}
    unknown_ids = set(rewritten_by_id) - set(original_by_id)
    if unknown_ids:
        parser.error(f"rewrite output contains unknown ids: {sorted(unknown_ids)[:5]}")
    seen_queries = {str(row["query"]).strip() for row in manual + completed}
    client = ChatClient(args.base_url, args.model, args.timeout)
    started = time.perf_counter()
    attempts = 0

    for seed in manual:
        sample_id = str(seed["id"])
        if sample_id in rewritten_by_id:
            continue
        for seed_attempt in range(1, args.max_attempts_per_sample + 1):
            attempts += 1
            try:
                response = client.ask_json(
                    REWRITE_SYSTEM,
                    f"原始问题：{seed['query']}",
                    max_tokens=700,
                    temperature=0.7,
                )
                query = str(response["query"]).strip()
                if len(query) < 8 or query == str(seed["query"]).strip() or query in seen_queries:
                    raise ValueError("empty, unchanged, or duplicate rewritten query")
                if "SELECT " in query.upper():
                    raise ValueError("rewritten query contains SQL")
                review = {"consistent": None, "issues": [], "skipped": True}
                if not args.skip_review:
                    review = client.ask_json(
                        REVIEW_SYSTEM,
                        f"原始问题：{seed['query']}\n改写问题：{query}",
                        max_tokens=400,
                        temperature=0,
                    )
                    if review.get("consistent") is not True:
                        raise ValueError(f"semantic review rejected: {review.get('issues', [])}")
                row = dict(seed)
                row.update({
                    "query": query,
                    "original_query": seed["query"],
                    "augmentation_id": stable_key("manual_query_rewrite_v1", sample_id, query),
                    "augmentation": "query_synonym_rewrite",
                    "semantic_review": review,
                })
                if row["sql"] != seed["sql"] or str(row["id"]) != sample_id:
                    raise ValueError("id or SQL changed during rewrite")
                append_jsonl(rewritten_path, row)
                completed.append(row)
                rewritten_by_id[sample_id] = row
                seen_queries.add(query)
                print(f"PROGRESS {len(completed)}/{len(manual)} id={sample_id}", flush=True)
                break
            except Exception as exc:
                append_jsonl(failures_path, {
                    "id": seed["id"],
                    "attempt": seed_attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    complete = len(rewritten_by_id) == len(manual)
    replaced = 0
    inferred_ids = 0
    manual_rows = 0
    result = []
    if complete:
        for source_row in combined:
            row = dict(source_row)
            if row.get("source") == "manual_original":
                manual_rows += 1
                sample_id = str(row["id"]) if row.get("id") is not None else ids_by_pair.get(pair_key(row["query"], row["sql"]))
                if sample_id is None or sample_id not in rewritten_by_id:
                    parser.error("cannot map a manual_original row back to manual dataset")
                if row.get("id") is None:
                    inferred_ids += 1
                rewrite = rewritten_by_id[sample_id]
                if str(row["sql"]).strip() != str(rewrite["sql"]).strip():
                    parser.error(f"SQL mismatch while replacing id={sample_id}")
                row["id"] = rewrite["id"]
                row["query"] = rewrite["query"]
                row["query_rewrite_id"] = rewrite["augmentation_id"]
                replaced += 1
            result.append(row)
        if manual_rows != 840 or replaced != 840:
            parser.error(f"expected 840 manual_original rows, got manual_rows={manual_rows}, replaced={replaced}")
        with replaced_path.open("w", encoding="utf-8") as handle:
            for row in result:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "manual_samples": len(manual),
        "rewritten_samples": len(rewritten_by_id),
        "combined_samples": len(combined),
        "manual_original_rows": manual_rows,
        "replaced_rows": replaced,
        "ids_inferred_from_query_sql": inferred_ids,
        "source_distribution": dict(Counter(row.get("source", "missing") for row in combined)),
        "attempts_this_run": attempts,
        "seconds_this_run": round(time.perf_counter() - started, 3),
        "sql_changed": False,
        "complete": complete and replaced == 840,
        "rewritten_manual_file": str(rewritten_path),
        "replaced_training_file": str(replaced_path) if complete else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise SystemExit("rewrite incomplete; inspect failures.jsonl and resume")


if __name__ == "__main__":
    main()
