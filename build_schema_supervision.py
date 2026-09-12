from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import sqlglot
from sqlglot import exp


ROLES = ("select", "filter", "group", "having", "order", "join")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从query+SQL生成表路由、字段角色和字段候选对监督数据"
    )
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--validation-input", action="append", type=Path, default=[])
    parser.add_argument("--schema", type=Path, default=Path("data/schema_catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/schema_supervision_v1"))
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--hard-negatives", type=int, default=2)
    parser.add_argument("--random-negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def stable_id(query: str, sql: str) -> str:
    return hashlib.sha1(f"{query}\n{sql}".encode("utf-8")).hexdigest()[:16]


def split_bucket(query: str, ratio: float) -> str:
    digest = int(hashlib.sha1(query.encode("utf-8")).hexdigest()[:12], 16)
    return "validation" if digest / float(16**12) < ratio else "train"


def read_rows(paths: list[Path], requested_split: str | None) -> list[dict]:
    rows = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                rows.append({"error": f"JSON错误: {error}", "source": str(path), "line": line_number})
                continue
            rows.append(
                {
                    "source": path.name,
                    "line": line_number,
                    "query": normalized_text(item.get("query", item.get("question", ""))),
                    "sql": normalized_text(item.get("sql", item.get("predicted_sql", ""))).rstrip(";"),
                    "original_id": item.get("id"),
                    "requested_split": requested_split,
                }
            )
    return rows


def parse_query(sql: str) -> exp.Expression:
    errors = []
    for dialect in ("mysql", "sqlite"):
        try:
            statements = sqlglot.parse(sql, read=dialect)
            if len(statements) == 1 and isinstance(statements[0], (exp.Select, exp.Union)):
                return statements[0]
        except Exception as error:
            errors.append(f"{dialect}: {type(error).__name__}: {error}")
    raise ValueError("; ".join(errors) or "SQL不是单条SELECT/UNION")


def physical_tables(tree: exp.Expression, catalog: dict) -> tuple[list[str], dict[str, str]]:
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = []
    aliases = {}
    for node in tree.find_all(exp.Table):
        name = node.name
        if name.lower() in cte_names or name not in catalog:
            continue
        if name not in tables:
            tables.append(name)
        aliases[name.lower()] = name
        aliases[(node.alias_or_name or name).lower()] = name
    return tables, aliases


def column_role(column: exp.Column) -> str:
    current = column.parent
    while current is not None:
        if isinstance(current, exp.Where):
            return "filter"
        if isinstance(current, exp.Having):
            return "having"
        if isinstance(current, exp.Group):
            return "group"
        if isinstance(current, exp.Order):
            return "order"
        if isinstance(current, exp.Join):
            return "join"
        if isinstance(current, exp.Select):
            return "select"
        current = current.parent
    return "select"


def resolve_column_tables(
    column: exp.Column,
    tables: list[str],
    aliases: dict[str, str],
    catalog: dict,
) -> list[str]:
    if column.table:
        table = aliases.get(column.table.lower())
        return [table] if table and column.name in catalog[table] else []
    candidates = [table for table in tables if column.name in catalog[table]]
    if len(candidates) == 1:
        return candidates
    if len(tables) == 1 and column.name in catalog[tables[0]]:
        return tables
    return candidates


def extract_labels(tree: exp.Expression, catalog: dict) -> dict:
    tables, aliases = physical_tables(tree, catalog)
    roles = {role: set() for role in ROLES}
    unresolved = []
    for column in tree.find_all(exp.Column):
        if column.name == "*":
            continue
        matched_tables = resolve_column_tables(column, tables, aliases, catalog)
        if not matched_tables:
            unresolved.append(column.sql())
            continue
        role = column_role(column)
        for table in matched_tables:
            roles[role].add(f"{table}::{column.name}")
    select_all = any(isinstance(node, exp.Star) for node in tree.find_all(exp.Star))
    return {
        "tables": tables,
        "roles": {role: sorted(values) for role, values in roles.items()},
        "select_all": select_all,
        "unresolved_columns": sorted(set(unresolved)),
    }


def compact_field(table: str, field: str, definition: dict) -> dict:
    return {
        "table": table,
        "field": field,
        "type": definition.get("type"),
        "sql_type": definition.get("sql_type"),
        "description": definition.get("description", ""),
        "values": (definition.get("values") or [])[:20],
    }


def relevance(query: str, field: str, definition: dict) -> float:
    query_lower = query.lower()
    field_lower = field.lower()
    direct = 2.0 if field_lower in query_lower else 0.0
    description = str(definition.get("description") or "")[:160].lower()
    values = " ".join(map(str, (definition.get("values") or [])[:20])).lower()
    value_hit = 1.0 if any(value and value.lower() in query_lower for value in map(str, (definition.get("values") or [])[:20])) else 0.0
    similarity = max(
        SequenceMatcher(None, query_lower, field_lower).ratio(),
        SequenceMatcher(None, query_lower, description).ratio() if description else 0.0,
        SequenceMatcher(None, query_lower, values).ratio() if values else 0.0,
    )
    return direct + value_hit + similarity


def candidate_pairs(
    record: dict,
    catalog: dict,
    hard_count: int,
    random_count: int,
    seed: int,
) -> list[dict]:
    positive_roles = defaultdict(set)
    for role, labels in record["field_roles"].items():
        for label in labels:
            positive_roles[label].add(role)
    pairs = []
    rng = random.Random(f"{seed}:{record['sample_id']}")
    for table in record["tables"]:
        fields = catalog[table]
        positive_names = {
            label.split("::", 1)[1]
            for label in positive_roles
            if label.startswith(f"{table}::")
        }
        negatives = [field for field in fields if field not in positive_names]
        ranked = sorted(
            negatives,
            key=lambda field: (-relevance(record["query"], field, fields[field]), field),
        )
        chosen = ranked[:hard_count]
        remaining = [field for field in negatives if field not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[:random_count])
        for field in sorted(positive_names):
            label = f"{table}::{field}"
            pairs.append(
                {
                    "sample_id": record["sample_id"],
                    "query": record["query"],
                    **compact_field(table, field, fields[field]),
                    "label": 1,
                    "roles": sorted(positive_roles[label]),
                    "negative_type": None,
                }
            )
        hard_names = set(ranked[:hard_count])
        for field in chosen:
            pairs.append(
                {
                    "sample_id": record["sample_id"],
                    "query": record["query"],
                    **compact_field(table, field, fields[field]),
                    "label": 0,
                    "roles": [],
                    "negative_type": "hard" if field in hard_names else "random",
                }
            )
    return pairs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = arguments()
    if not 0 <= args.validation_ratio < 1:
        raise SystemExit("--validation-ratio必须在[0, 1)范围内")
    catalog = json.loads(args.schema.read_text(encoding="utf-8-sig"))["tables"]
    source_rows = read_rows(args.input, None) + read_rows(args.validation_input, "validation")
    errors = [row for row in source_rows if row.get("error")]
    valid_rows = [row for row in source_rows if not row.get("error")]

    deduplicated = {}
    conflicts = defaultdict(set)
    for row in valid_rows:
        if not row["query"] or not row["sql"]:
            errors.append({**row, "error": "query或sql为空"})
            continue
        conflicts[row["query"]].add(row["sql"])
        key = (row["query"], row["sql"])
        previous = deduplicated.get(key)
        if previous is None or row["requested_split"] == "validation":
            deduplicated[key] = row

    conflicting_queries = {query for query, sqls in conflicts.items() if len(sqls) > 1}
    records = []
    for row in deduplicated.values():
        if row["query"] in conflicting_queries:
            errors.append({**row, "error": "同一query对应多个不同SQL"})
            continue
        try:
            labels = extract_labels(parse_query(row["sql"]), catalog)
        except Exception as error:
            errors.append({**row, "error": f"SQL解析失败: {type(error).__name__}: {error}"})
            continue
        if not labels["tables"]:
            errors.append({**row, "error": "未解析到Schema中的物理表"})
            continue
        sample_id = stable_id(row["query"], row["sql"])
        split = row["requested_split"] or split_bucket(row["query"], args.validation_ratio)
        records.append(
            {
                "sample_id": sample_id,
                "original_id": row["original_id"],
                "source": row["source"],
                "split": split,
                "query": row["query"],
                "sql": row["sql"],
                "tables": labels["tables"],
                "field_roles": labels["roles"],
                "select_all": labels["select_all"],
                "unresolved_columns": labels["unresolved_columns"],
            }
        )

    # Any query explicitly assigned to validation is removed from training aliases.
    validation_queries = {record["query"] for record in records if record["split"] == "validation"}
    for record in records:
        if record["query"] in validation_queries:
            record["split"] = "validation"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "inputs": [str(path) for path in args.input],
        "validation_inputs": [str(path) for path in args.validation_input],
        "schema": str(args.schema),
        "source_rows": len(source_rows),
        "accepted_samples": len(records),
        "train_samples": sum(record["split"] == "train" for record in records),
        "validation_samples": sum(record["split"] == "validation" for record in records),
        "duplicate_pairs_removed": len(valid_rows) - len(deduplicated),
        "conflicting_queries": len(conflicting_queries),
        "parse_or_format_errors": len(errors),
        "samples_with_unresolved_columns": sum(bool(record["unresolved_columns"]) for record in records),
        "table_counts": dict(sorted(Counter(table for record in records for table in record["tables"]).items())),
        "role_label_counts": {
            role: sum(len(record["field_roles"][role]) for record in records)
            for role in ROLES
        },
        "hard_negatives_per_table": args.hard_negatives,
        "random_negatives_per_table": args.random_negatives,
        "seed": args.seed,
    }

    for split in ("train", "validation"):
        selected = [record for record in records if record["split"] == split]
        router = [
            {
                "sample_id": record["sample_id"],
                "query": record["query"],
                "tables": record["tables"],
                "multi_table": len(record["tables"]) > 1,
            }
            for record in selected
        ]
        roles = [
            {
                "sample_id": record["sample_id"],
                "query": record["query"],
                "tables": record["tables"],
                "field_roles": record["field_roles"],
                "select_all": record["select_all"],
                "unresolved_columns": record["unresolved_columns"],
            }
            for record in selected
        ]
        pairs = [
            pair
            for record in selected
            for pair in candidate_pairs(
                record,
                catalog,
                args.hard_negatives,
                args.random_negatives,
                args.seed,
            )
        ]
        write_jsonl(args.output_dir / f"table_router_{split}.jsonl", router)
        write_jsonl(args.output_dir / f"field_roles_{split}.jsonl", roles)
        write_jsonl(args.output_dir / f"field_pairs_{split}.jsonl", pairs)

    write_jsonl(args.output_dir / "rejected.jsonl", errors)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        print(f"警告：{len(errors)}条记录被拒绝，详见{args.output_dir / 'rejected.jsonl'}")


if __name__ == "__main__":
    main()
