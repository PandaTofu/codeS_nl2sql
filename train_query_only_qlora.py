import argparse
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from codes_data.io import read_jsonl


SOURCE_RATIOS = {
    "manual_original": 0.60,
    "sql_to_question": 0.30,
    "question_to_sql": 0.10,
}

PROMPT_TEMPLATE = """将下面的中文问题转换为一条MySQL只读SQL。只输出SQL，不要解释或Markdown。
### Question
{query}
### SQL
"""


class QueryOnlyDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, validation=False):
        self.items = []
        self.skipped = []
        for index, row in enumerate(read_jsonl(path)):
            query = str(row.get("query", "")).strip()
            sql = str(row.get("sql", "")).strip().rstrip(";")
            source = "manual_original" if validation else str(row.get("source", "manual_original"))
            if not query or not sql:
                self.skipped.append({"index": index, "reason": "empty_query_or_sql"})
                continue
            prompt = PROMPT_TEMPLATE.format(query=query)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if tokenizer.bos_token_id is not None:
                prompt_ids = [tokenizer.bos_token_id] + prompt_ids
            target_ids = tokenizer(sql, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                target_ids.append(tokenizer.eos_token_id)
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_length:
                self.skipped.append({"index": index, "reason": "over_max_length", "tokens": len(input_ids)})
                continue
            self.items.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + target_ids,
                "source": source,
            })
        if not self.items:
            raise ValueError(f"no usable samples in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {key: item[key] for key in ("input_ids", "attention_mask", "labels")}

    @property
    def sources(self):
        return [item["source"] for item in self.items]


class CausalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        length = max(len(item["input_ids"]) for item in features)
        return {
            "input_ids": torch.tensor([
                item["input_ids"] + [self.pad_token_id] * (length - len(item["input_ids"]))
                for item in features
            ], dtype=torch.long),
            "attention_mask": torch.tensor([
                item["attention_mask"] + [0] * (length - len(item["attention_mask"]))
                for item in features
            ], dtype=torch.long),
            "labels": torch.tensor([
                item["labels"] + [-100] * (length - len(item["labels"]))
                for item in features
            ], dtype=torch.long),
        }


class BalancedTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        sources = dataset.sources
        counts = Counter(sources)
        unknown = sorted(set(counts) - set(SOURCE_RATIOS))
        if unknown:
            raise ValueError(f"unknown training sources: {unknown}")
        weights = [SOURCE_RATIOS[source] / counts[source] for source in sources]
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )


def training_arguments(args):
    parameters = inspect.signature(TrainingArguments).parameters
    values = {
        "output_dir": str(args.output),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": 2,
    }
    if "eval_strategy" in parameters:
        values["eval_strategy"] = "epoch"
    else:
        values["evaluation_strategy"] = "epoch"
    return TrainingArguments(**{key: value for key, value in values.items() if key in parameters})


def main():
    parser = argparse.ArgumentParser(description="Query-only CodeS-3B QLoRA baseline")
    parser.add_argument("--model", default="/home/ubuntu/models/CodeS-3B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--train", default="outputs/augmentation_v2/codes_sft_train.jsonl", type=Path)
    parser.add_argument("--validation", default="outputs/augmentation_v2/manual_validation.jsonl", type=Path)
    parser.add_argument("--output", default="training/query_only_codes3b_v1", type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        parser.error("GPU does not support BF16")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = QueryOnlyDataset(args.train, tokenizer, args.max_length)
    validation_dataset = QueryOnlyDataset(args.validation, tokenizer, args.max_length, validation=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    architecture = str(getattr(model.config, "model_type", ""))
    if architecture != "gpt_bigcode":
        parser.error(f"expected GPTBigCode/CodeS model, got model_type={architecture}")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"],
            fan_in_fan_out=True,
        ))
    model.config.use_cache = False
    model.print_trainable_parameters()
    source_counts = Counter(train_dataset.sources)
    run_config = {
        "mode": "query_only",
        "schema_in_prompt": False,
        "model": args.model,
        "initial_adapter": str(args.adapter) if args.adapter else None,
        "model_type": architecture,
        "train_file": str(args.train),
        "validation_file": str(args.validation),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_source_counts": dict(source_counts),
        "sampling_ratios": SOURCE_RATIOS,
        "train_skipped": train_dataset.skipped,
        "validation_skipped": validation_dataset.skipped,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05,
                 "target_modules": ["c_attn", "c_proj", "c_fc"]},
    }
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in run_config.items() if "skipped" not in key}, ensure_ascii=False, indent=2))
    trainer = BalancedTrainer(
        model=model,
        args=training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    adapter = args.output / "best_adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    summary = {
        **run_config,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "global_steps": trainer.state.global_step,
        "optimizer_steps_per_epoch": math.ceil(len(train_dataset) / args.batch_size / args.gradient_accumulation),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "best_adapter": str(adapter),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
