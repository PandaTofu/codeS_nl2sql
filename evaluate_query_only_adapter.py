import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from codes_data.io import read_jsonl
from codes_data.official_like_metric import compare_sql
from train_query_only_qlora import PROMPT_TEMPLATE as QUERY_ONLY_PROMPT
from train_query_schema_qlora import PROMPT_TEMPLATE as QUERY_SCHEMA_PROMPT


def clean_sql(text):
    text = text.strip().replace("```sql", "").replace("```", "").strip()
    match = re.search(r"\b(?:SELECT|WITH)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():]
    return text.split(";", 1)[0].strip()


def normalized_text(sql):
    return " ".join(sql.strip().rstrip(";").split()).casefold()


def static_issues(sql):
    without_literals = re.sub(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*.*?\*/",
        " ", sql, flags=re.DOTALL,
    )
    issues = []
    if not re.match(r"^\s*(SELECT|WITH)\b", without_literals, re.IGNORECASE):
        issues.append("statement must start with SELECT or WITH")
    if len([part for part in without_literals.split(";") if part.strip()]) != 1:
        issues.append("multiple statements are not allowed")
    if re.search(
        r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|CALL|GRANT|REVOKE|SET|USE|LOAD|OUTFILE|INTO)\b",
        without_literals, re.IGNORECASE,
    ):
        issues.append("write or administrative keyword detected")
    if sql.count("(") != sql.count(")"):
        issues.append("unbalanced parentheses")
    return issues


def batches(items, size):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def main():
    parser = argparse.ArgumentParser(description="Evaluate query-only or Query+Schema CodeS adapter")
    parser.add_argument("--model", default="/home/ubuntu/models/CodeS-3B")
    parser.add_argument("--adapter", default="training/query_only_codes3b_v1/best_adapter")
    parser.add_argument("--validation", default="outputs/augmentation_v2/manual_validation.jsonl", type=Path)
    parser.add_argument("--output", default="reports/query_only_codes3b_v1", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompt-mode", choices=("query-only", "query-schema"),
                        default="query-only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    rows = list(read_jsonl(args.validation))
    if not rows:
        parser.error(f"validation set is empty: {args.validation}")
    if args.prompt_mode == "query-schema":
        missing_schema = [index for index, row in enumerate(rows) if not str(row.get("schema") or "").strip()]
        if missing_schema:
            parser.error(
                f"query-schema mode requires schema in every row; missing at indexes {missing_schema[:10]}"
            )
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    predictions = []
    started = time.perf_counter()
    for chunk in batches(rows, args.batch_size):
        if args.prompt_mode == "query-schema":
            prompts = [QUERY_SCHEMA_PROMPT.format(
                schema=str(row["schema"]).strip(),
                query=str(row["query"]).strip(),
            ) for row in chunk]
        else:
            prompts = [QUERY_ONLY_PROMPT.format(query=str(row["query"]).strip()) for row in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        batch_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        batch_seconds = time.perf_counter() - batch_started
        tails = generated[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(tails, skip_special_tokens=True)
        for row, raw in zip(chunk, decoded):
            predicted = clean_sql(raw)
            gold = str(row["sql"]).strip().rstrip(";")
            issues = static_issues(predicted)
            item = {
                "id": row.get("id"),
                "query": row["query"],
                "gold_sql": gold,
                "predicted_sql": predicted,
                "exact_match": normalized_text(predicted) == normalized_text(gold),
                "static_valid": not issues,
                "issues": issues,
                "latency_seconds": round(batch_seconds / len(chunk), 4),
            }
            item.update(compare_sql(gold, predicted))
            predictions.append(item)
        print(f"PROGRESS {len(predictions)}/{len(rows)}", flush=True)

    seconds = time.perf_counter() - started
    total = len(predictions)
    latencies = sorted(item["latency_seconds"] for item in predictions)
    percentile = lambda ratio: latencies[min(round((total - 1) * ratio), total - 1)]
    summary = {
        "samples": total,
        "exact_match": sum(item["exact_match"] for item in predictions) / total,
        "rule_match_rate": sum(item["rule_match"] for item in predictions) / total,
        "average_rule_score": sum(item["rule_score"] for item in predictions) / total,
        "static_valid_rate": sum(item["static_valid"] for item in predictions) / total,
        "parse_error_samples": sum(bool(item["parse_errors"]) for item in predictions),
        "inference_seconds": round(seconds, 3),
        "average_seconds_per_sample": round(seconds / total, 3),
        "average_generation_latency_seconds": sum(latencies) / total,
        "p50_generation_latency_seconds": percentile(0.5),
        "p95_generation_latency_seconds": percentile(0.95),
        "model": args.model,
        "adapter": args.adapter,
        "validation": str(args.validation),
        "prompt_mode": args.prompt_mode,
    }
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
