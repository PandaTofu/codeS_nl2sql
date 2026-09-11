import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge a CodeS QLoRA adapter into its base model")
    parser.add_argument("--base", default="/home/ubuntu/models/CodeS-3B")
    parser.add_argument("--adapter", default="training/query_only_codes3b_mixed_v1/best_adapter")
    parser.add_argument("--output", default="models/CodeS-3B-NL2SQL-Merged", type=Path)
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    adapted = PeftModel.from_pretrained(base, args.adapter)
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output)
    manifest = {
        "base_model": args.base,
        "adapter": args.adapter,
        "output": str(args.output),
        "dtype": "bfloat16",
        "safe_merge": True,
    }
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
