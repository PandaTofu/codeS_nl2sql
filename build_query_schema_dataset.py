from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sqlglot
from sqlglot import exp


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将陈老师整理的Schema领域语料与Query-SQL样本组合为Query+Schema监督数据"
    )
    parser.add_argument("--input", action="append", type=Path, required=True,
                        help="Query-SQL JSONL，可重复指定")
    parser.add_argument("--schema-corpus", type=Path, required=True,
                        help="陈老师方案的lora1_domain_corpus.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dialect", default="mysql")
    parser.add_argument("--include-used-field-constraints", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="可选附加Gold SQL已用字段的详细约束；默认关闭以避免目标泄漏")
    return parser.parse_args()


def load_corpus(path: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    table_records: dict[str, list[dict]] = defaultdict(list)
    field_records: dict[str, list[dict]] = defaultdict(list)
    allowed = {"table_overview", "schema_chunk", "field_definition", "field_constraint"}
    for _, row in read_jsonl(path):
        kind = str(row.get("type") or row.get("kind") or "")
        table = str(row.get("table") or "")
        text = str(row.get("text") or "").strip()
        if kind not in allowed or not table or not text:
            continue
        item = {**row, "_kind": kind}
        if kind in {"table_overview", "schema_chunk"}:
            table_records[table].append(item)
        else:
            field_records[table].append(item)
    if not table_records:
        raise ValueError(f"Schema语料中没有可用的表概述或字段分块: {path}")
    return table_records, field_records


def sql_structure(sql: str, dialect: str) -> tuple[list[str], set[str]]:
    statements = sqlglot.parse(sql, read=dialect)
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
        raise ValueError("只支持单条SELECT/UNION")
    tree = statements[0]
    tables = list(dict.fromkeys(node.name for node in tree.find_all(exp.Table) if node.name))
    fields = {node.name for node in tree.find_all(exp.Column) if node.name}
    if not tables:
        raise ValueError("SQL没有数据表")
    return tables, fields


def related_field_record(row: dict, used_fields: set[str]) -> bool:
    field = str(row.get("field") or "")
    if field:
        return field in used_fields
    text = str(row.get("text") or "")
    return any(field_name in text for field_name in used_fields)


def schema_context(
    tables: list[str],
    used_fields: set[str],
    table_records: dict[str, list[dict]],
    field_records: dict[str, list[dict]],
    include_constraints: bool,
) -> str:
    sections = []
    for table in tables:
        base = table_records.get(table)
        if not base:
            raise ValueError(f"Schema语料缺少表: {table}")
        texts = [str(row["text"]).strip() for row in base]
        if include_constraints:
            texts.extend(
                str(row["text"]).strip()
                for row in field_records.get(table, [])
                if related_field_record(row, used_fields)
            )
        sections.append("\n".join(dict.fromkeys(texts)))
    return "\n\n".join(sections)


def main() -> None:
    args = parse_arguments()
    table_records, field_records = load_corpus(args.schema_corpus)
    output_rows = []
    failures = []
    seen_queries = set()
    for source in args.input:
        for line_number, row in read_jsonl(source):
            query = str(row.get("query") or "").strip()
            sql = str(row.get("sql") or row.get("predicted_sql") or "").strip().rstrip(";")
            if not query or not sql:
                failures.append({"source": str(source), "line": line_number,
                                 "reason": "empty_query_or_sql"})
                continue
            if query in seen_queries:
                failures.append({"source": str(source), "line": line_number,
                                 "reason": "duplicate_query"})
                continue
            try:
                tables, fields = sql_structure(sql, args.dialect)
                context = schema_context(
                    tables, fields, table_records, field_records,
                    args.include_used_field_constraints,
                )
            except Exception as error:
                failures.append({"source": str(source), "line": line_number,
                                 "reason": f"{type(error).__name__}: {error}"})
                continue
            seen_queries.add(query)
            output_rows.append({
                "id": row.get("id", f"{source.stem}:{line_number}"),
                "query": query,
                "schema": context,
                "sql": sql,
                "tables": tables,
                "fields": sorted(fields),
                "source": row.get("source", source.stem),
            })
    write_jsonl(args.output, output_rows)
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "inputs": [str(path) for path in args.input],
        "schema_corpus": str(args.schema_corpus),
        "output": str(args.output),
        "samples": len(output_rows),
        "failures": len(failures),
        "include_used_field_constraints": args.include_used_field_constraints,
        "failure_details": failures,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failure_details"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
