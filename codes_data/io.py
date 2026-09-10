import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def append_jsonl(path, row):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_name(sample_id, validation_ratio):
    value = int(hashlib.sha256(str(sample_id).encode()).hexdigest()[:12], 16) / 0xFFFFFFFFFFFF
    return "validation" if value < validation_ratio else "train"


def stable_key(*parts):
    payload = "\u241f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

