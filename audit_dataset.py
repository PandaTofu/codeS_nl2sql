import argparse
import json
from collections import Counter
from pathlib import Path

from codes_data.io import read_jsonl
from codes_data.sql_tools import parse_single_select, validate_sql


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema", default="data/schema_catalog.json")
    parser.add_argument("--report", default="reports/augmentation_audit.json")
    parser.add_argument("--invalid-output", default="reports/augmentation_invalid.jsonl")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    catalog = json.loads(Path(args.schema).read_text(encoding="utf-8-sig"))["tables"]
    source_counts = Counter()
    mutation_counts = Counter()
    error_counts = Counter()
    queries = Counter()
    sqls = Counter()
    pairs = Counter()
    invalid = []
    for index, row in enumerate(rows):
        source_counts[str(row.get("source", "unknown"))] += 1
        mutation_counts[str(row.get("mutation", {}).get("kind", "none"))] += 1
        query = str(row.get("query", "")).strip()
        sql = str(row.get("sql", "")).strip()
        table = row.get("table")
        queries[query] += 1
        sqls[sql] += 1
        pairs[(query, sql)] += 1
        errors = []
        if len(query) < 8:
            errors.append("query_too_short")
        if not sql:
            errors.append("empty_sql")
        else:
            try:
                validate_sql(sql, catalog, table, strict_types=True)
            except Exception as exc:
                errors.append(f"sql_validation:{exc}")
        try:
            tree = parse_single_select(sql)
            projections = [item.sql(dialect="sqlite", identify=True) for item in tree.expressions]
            if len(projections) != len(set(projections)):
                errors.append("duplicate_select_expression")
        except Exception:
            pass
        if row.get("source") != "manual_original":
            validated = row.get("validated", {})
            if validated.get("teacher_semantic_review") is not True:
                errors.append("semantic_review_missing")
        if errors:
            for error in errors:
                error_counts[error.split(":", 1)[0]] += 1
            invalid.append({"index": index, "errors": errors, "sample": row})
    duplicate_queries = sum(count - 1 for count in queries.values() if count > 1)
    duplicate_sqls = sum(count - 1 for count in sqls.values() if count > 1)
    duplicate_pairs = sum(count - 1 for count in pairs.values() if count > 1)
    report = {
        "samples": len(rows),
        "valid_samples": len(rows) - len(invalid),
        "invalid_samples": len(invalid),
        "valid_rate": round((len(rows) - len(invalid)) / len(rows), 6) if rows else 0,
        "sources": dict(source_counts),
        "mutations": dict(mutation_counts),
        "errors": dict(error_counts),
        "duplicate_queries": duplicate_queries,
        "duplicate_sqls": duplicate_sqls,
        "duplicate_pairs": duplicate_pairs,
        "database_execution_checked": False,
        "semantic_equivalence_guaranteed": False,
    }
    report_path = Path(args.report)
    invalid_path = Path(args.invalid_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with invalid_path.open("w", encoding="utf-8") as handle:
        for row in invalid:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
