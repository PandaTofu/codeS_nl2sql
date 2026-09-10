import argparse
import json
import random
import time
from pathlib import Path

from codes_data.client import ChatClient
from codes_data.io import append_jsonl, read_jsonl, split_name, stable_key
from codes_data.sql_tools import mutate_sql, schema_text, validate_sql


SQL_TO_QUESTION_SYSTEM = """你是中文Text-to-SQL训练数据标注员。根据给定Schema和SQL，写出一条自然、完整、无歧义的中文用户问题。
问题必须表达SQL中的每个SELECT字段、过滤条件、逻辑关系、聚合、分组、排序和数量限制；不得增加SQL没有的条件。
不得出现SQL、表名、下划线字段名或“根据上述”等提示语。只输出JSON：{"query":"..."}。"""

QUESTION_TO_SQL_SYSTEM = """你是中文Text-to-SQL训练数据生成员。参考原始样本的语言风格，为同一张表创造一个不同意图的新问题和匹配的SQLite SQL。
只能使用Schema中的字段和值；问题与SQL必须逐项对应；不要复制原问题；只允许单表SELECT；不得使用子查询或JOIN。
只输出JSON：{"query":"...","sql":"SELECT ..."}。"""


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--input", default="data/manual_raw_train.jsonl")
    result.add_argument("--schema", default="data/schema_catalog.json")
    result.add_argument("--output-dir", default="outputs/augmentation_v1")
    result.add_argument("--target", type=int, default=2000)
    result.add_argument("--sql-driven-ratio", type=float, default=0.8)
    result.add_argument("--validation-ratio", type=float, default=0.15)
    result.add_argument("--base-url", default="http://127.0.0.1:6006/v1")
    result.add_argument("--model", default="Qwen3-14B")
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--seed", type=int, default=20260910)
    result.add_argument("--max-attempts", type=int, default=10000)
    result.add_argument("--resume", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if args.target <= 0 or not 0 <= args.sql_driven_ratio <= 1 or not 0 < args.validation_ratio < 1:
        raise SystemExit("invalid target or ratio")
    rows = read_jsonl(args.input)
    catalog = json.loads(Path(args.schema).read_text(encoding="utf-8-sig"))["tables"]
    train = [row for row in rows if split_name(row["id"], args.validation_ratio) == "train"]
    validation = [row for row in rows if split_name(row["id"], args.validation_ratio) == "validation"]
    eligible = []
    for row in train:
        table = row.get("table")
        if table not in catalog:
            continue
        try:
            validate_sql(row["sql"], catalog, table)
            eligible.append(row)
        except Exception:
            continue
    if not eligible:
        raise SystemExit("no valid single-table training seeds")
    out = Path(args.output_dir)
    augmentation_file = out / "augmentation.jsonl"
    failure_file = out / "failures.jsonl"
    if not args.resume and (augmentation_file.exists() or failure_file.exists()):
        raise SystemExit("output exists; choose a new directory or pass --resume")
    completed = read_jsonl(augmentation_file) if augmentation_file.exists() else []
    known = {row["augmentation_id"] for row in completed}
    seen_queries = {row["query"].strip() for row in rows}
    seen_queries.update(row["query"].strip() for row in completed)
    seen_sql = set()
    for row in rows + completed:
        try:
            seen_sql.add(validate_sql(row["sql"], catalog, row.get("table")))
        except Exception:
            pass
    client = ChatClient(args.base_url, args.model, args.timeout)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    attempts = 0
    while len(completed) < args.target and attempts < args.max_attempts:
        attempts += 1
        seed = rng.choice(eligible)
        table = seed.get("table")
        if table not in catalog:
            append_jsonl(failure_file, {"seed_id": seed.get("id"), "error": "unknown seed table"})
            continue
        direction = "sql_to_question" if rng.random() < args.sql_driven_ratio else "question_to_sql"
        augmentation_id = stable_key(args.seed, attempts, seed["id"], direction)
        if augmentation_id in known:
            continue
        try:
            if direction == "sql_to_question":
                sql, mutation = mutate_sql(seed["sql"], table, catalog[table], rng)
                sql = validate_sql(sql, catalog, table)
                response = client.ask_json(
                    SQL_TO_QUESTION_SYSTEM,
                    schema_text(table, catalog[table], sql) + f"\nSQL：{sql}",
                )
                query = str(response["query"]).strip()
            else:
                mutation = {"kind": "teacher_generated_pair"}
                response = client.ask_json(
                    QUESTION_TO_SQL_SYSTEM,
                    schema_text(table, catalog[table])
                    + f"\n原始问题：{seed['query']}\n原始SQL：{seed['sql']}",
                )
                query, sql = str(response["query"]).strip(), str(response["sql"]).strip()
                sql = validate_sql(sql, catalog, table)
            if len(query) < 8 or query == seed["query"] or "SELECT " in query.upper():
                raise ValueError("invalid or copied query")
            if query in seen_queries:
                raise ValueError("duplicate query")
            if sql in seen_sql:
                raise ValueError("duplicate SQL")
            row = {
                "augmentation_id": augmentation_id,
                "query": query,
                "sql": sql,
                "table": table,
                "source": direction,
                "parent_id": seed["id"],
                "mutation": mutation,
                "validated": {"sqlglot": True, "schema": True, "database_execution": False},
            }
            append_jsonl(augmentation_file, row)
            completed.append(row)
            known.add(augmentation_id)
            seen_queries.add(query)
            seen_sql.add(sql)
            print(f"[{len(completed)}/{args.target}] {direction} parent={seed['id']}", flush=True)
        except Exception as exc:
            append_jsonl(failure_file, {
                "attempt": attempts, "seed_id": seed.get("id"), "direction": direction,
                "error": f"{type(exc).__name__}: {exc}",
            })
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manual_train.jsonl").open("w", encoding="utf-8") as handle:
        for row in train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "manual_validation.jsonl").open("w", encoding="utf-8") as handle:
        for row in validation:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "codes_sft_train.jsonl").open("w", encoding="utf-8") as handle:
        for row in train:
            handle.write(json.dumps({"query": row["query"], "sql": row["sql"], "table": row["table"], "source": "manual_original"}, ensure_ascii=False) + "\n")
        for row in completed[:args.target]:
            handle.write(json.dumps({key: row[key] for key in ("query", "sql", "table", "source")}, ensure_ascii=False) + "\n")
    summary = {
        "manual_samples": len(rows), "manual_train": len(train), "manual_validation": len(validation),
        "eligible_single_table_seeds": len(eligible),
        "augmentation_target": args.target, "augmentation_generated": min(len(completed), args.target),
        "attempts_this_run": attempts, "seconds_this_run": round(time.perf_counter() - started, 3),
        "sql_driven_ratio_requested": args.sql_driven_ratio,
        "database_execution_checked": False,
        "complete": len(completed) >= args.target,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise SystemExit("target not reached; inspect failures.jsonl and resume")


if __name__ == "__main__":
    main()
