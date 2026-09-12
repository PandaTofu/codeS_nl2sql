from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from build_query_schema_dataset import (
    load_corpus,
    read_jsonl,
    schema_context,
    sql_structure,
    write_jsonl,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CodeS Query+Schema adapter or merged model"
    )
    parser.add_argument("--model", default="/home/ubuntu/models/CodeS-3B")
    parser.add_argument("--adapter")
    parser.add_argument("--merged-model", action="store_true")
    parser.add_argument("--validation", type=Path, required=True,
                        help="Raw query/sql JSONL or an already schema-enriched JSONL")
    parser.add_argument("--schema-corpus", type=Path,
                        help="Required when validation rows do not already contain schema")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dialect", default="mysql")
    return parser.parse_args()


def prepare_validation(args: argparse.Namespace) -> Path:
    rows = [row for _, row in read_jsonl(args.validation)]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be greater than 0")
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"validation set is empty: {args.validation}")

    missing = [index for index, row in enumerate(rows) if not str(row.get("schema") or "").strip()]
    if missing:
        if args.schema_corpus is None:
            raise SystemExit(
                "validation rows do not contain schema; provide --schema-corpus "
                "or run build_query_schema_dataset.py first"
            )
        table_records, field_records = load_corpus(args.schema_corpus)
        failures = []
        for index in missing:
            row = rows[index]
            sql = str(row.get("sql") or row.get("gold_sql") or "").strip()
            try:
                tables, fields = sql_structure(sql, args.dialect)
                row["schema"] = schema_context(
                    tables,
                    fields,
                    table_records,
                    field_records,
                    include_constraints=False,
                )
            except Exception as error:
                failures.append(f"index={index}: {type(error).__name__}: {error}")
        if failures:
            raise SystemExit("failed to attach schema:\n" + "\n".join(failures[:20]))

    args.output.mkdir(parents=True, exist_ok=True)
    prepared = args.output / "prepared_validation.jsonl"
    write_jsonl(prepared, rows)
    return prepared


def main() -> None:
    args = arguments()
    if args.merged_model and args.adapter:
        raise SystemExit("--adapter and --merged-model cannot be used together")
    if not args.merged_model and not args.adapter:
        raise SystemExit("provide --adapter, or use --merged-model")

    prepared = prepare_validation(args)
    evaluator = Path(__file__).with_name("evaluate_query_only_adapter.py")
    command = [
        sys.executable,
        str(evaluator),
        "--model", args.model,
        "--validation", str(prepared),
        "--output", str(args.output),
        "--prompt-mode", "query-schema",
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if args.merged_model:
        command.append("--merged-model")
    else:
        command.extend(["--adapter", args.adapter])
    # prepare_validation already applies the limit. Passing it again would only
    # obscure the original source sample count in the evaluator report.
    print("RUN", " ".join(command), flush=True)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
