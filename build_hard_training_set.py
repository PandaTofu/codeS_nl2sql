import argparse
import json
from collections import Counter
from pathlib import Path

from codes_data.io import read_jsonl


def key(row):
    return str(row["query"]).strip()


def is_complex(sql):
    value = f" {str(sql).upper()} "
    return any(token in value for token in (
        " JOIN ", " GROUP BY ", " HAVING ", " ORDER BY ", " LIMIT ", " OR ",
    ))


def main():
    parser = argparse.ArgumentParser(description="Build query-only hard-example continuation dataset")
    parser.add_argument("--train", default="outputs/augmentation_v2/codes_sft_train.jsonl", type=Path)
    parser.add_argument("--replay", default="reports/query_only_codes3b_v1_train_replay/predictions.jsonl", type=Path)
    parser.add_argument("--output", default="outputs/query_only_codes3b_v2/hard_train.jsonl", type=Path)
    parser.add_argument("--hard-repeat", type=int, default=4)
    parser.add_argument("--complex-repeat", type=int, default=2)
    args = parser.parse_args()
    if args.hard_repeat < 1 or args.complex_repeat < 1:
        parser.error("repeat values must be positive")

    train = read_jsonl(args.train)
    replay = read_jsonl(args.replay)
    failed_queries = {key(row) for row in replay if not row.get("rule_match", False)}
    known_queries = {key(row) for row in train}
    missing = failed_queries - known_queries
    if missing:
        parser.error(f"{len(missing)} failed replay queries are absent from training data")

    output = []
    repeat_counts = Counter()
    for row in train:
        repeat = 1
        reasons = []
        if key(row) in failed_queries:
            repeat = max(repeat, args.hard_repeat)
            reasons.append("replay_error")
        if is_complex(row["sql"]):
            repeat = max(repeat, args.complex_repeat)
            reasons.append("complex_structure")
        for copy_index in range(repeat):
            item = dict(row)
            item["hard_reasons"] = reasons
            item["hard_copy"] = copy_index
            output.append(item)
        repeat_counts[repeat] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source_samples": len(train),
        "replay_samples": len(replay),
        "replay_failed_samples": len(failed_queries),
        "output_samples": len(output),
        "repeat_distribution": dict(sorted(repeat_counts.items())),
        "source_distribution": dict(Counter(row.get("source", "manual_original") for row in output)),
        "validation_samples_used": 0,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(args.output), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
