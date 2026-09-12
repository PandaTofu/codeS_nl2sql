from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


TABLE_DOMAINS = {
    "air_conditioner": "空调",
    "computer_join_config": "电脑SKU配置",
    "computer_join_main": "电脑产品与型号",
    "computer_join_price": "电脑价格记录",
    "desktop_computer": "台式机",
    "digital_camera": "数码相机",
    "electric_vehicle": "电动车",
    "headphones": "耳机",
    "laptop": "笔记本电脑",
    "microwave_oven": "微波炉",
    "printer": "打印机",
    "smartphone": "智能手机",
    "television": "电视",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将Schema目录转换为CodeS领域CLM语料")
    parser.add_argument("--schema", type=Path, default=Path("data/schema_catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/schema_domain_v1"))
    parser.add_argument("--chunk-size", type=int, default=10)
    return parser.parse_args()


def record(kind: str, table: str, text: str, **metadata) -> dict:
    fingerprint = json.dumps([kind, table, text, metadata], ensure_ascii=False, sort_keys=True)
    return {
        "id": hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16],
        "kind": kind,
        "table": table,
        "text": text,
        **metadata,
    }


def field_fact(table: str, field: str, definition: dict) -> tuple[str, str]:
    field_type = definition.get("type") or definition.get("sql_type") or "未知"
    description = str(definition.get("description") or "").strip()
    unit = definition.get("unit")
    values = list(definition.get("values") or [])[:30]
    examples = list(definition.get("example_values") or [])[:10]
    value_text = ""
    if values:
        value_text = "，允许的枚举值包括：" + "、".join(map(str, values))
    elif examples:
        value_text = "，示例值包括：" + "、".join(map(str, examples))
    unit_text = f"，单位是{unit}" if unit else ""
    detail_text = f"，含义或约束为：{description}" if description else ""
    train = (
        f"在表{table}中，字段{field}的数据类型是{field_type}{unit_text}"
        f"{detail_text}{value_text}。生成SQL时必须准确使用字段名{field}。"
    )
    validation = (
        f"Schema知识：{field}属于表{table}，类型为{field_type}{unit_text}"
        f"{detail_text}{value_text}。"
    )
    return train, validation


def build(catalog: dict, chunk_size: int) -> tuple[list[dict], list[dict]]:
    train = []
    validation = []
    tables = catalog["tables"]
    for table, fields in tables.items():
        domain = TABLE_DOMAINS.get(table, table)
        train.append(record(
            "table_overview",
            table,
            f"表{table}保存{domain}数据，共有{len(fields)}个字段。用户查询涉及{domain}时，应优先考虑表{table}。",
        ))
        validation.append(record(
            "table_overview_validation",
            table,
            f"Schema知识：{table}是{domain}领域的数据表，包含{len(fields)}个字段。",
        ))
        items = list(fields.items())
        for start in range(0, len(items), chunk_size):
            group = items[start : start + chunk_size]
            compact = "、".join(
                f"{field}({definition.get('sql_type') or definition.get('type') or '未知'})"
                for field, definition in group
            )
            names = [field for field, _ in group]
            train.append(record(
                "schema_chunk",
                table,
                f"表{table}包含字段：{compact}。SQL只能使用Schema中存在的准确字段名。",
                fields=names,
            ))
            validation.append(record(
                "schema_chunk_validation",
                table,
                f"Schema知识：数据表{table}的这一组字段为{compact}。",
                fields=names,
            ))
        for field, definition in items:
            train_text, validation_text = field_fact(table, field, definition)
            train.append(record(
                "field_definition",
                table,
                train_text,
                field=field,
                sql_type=definition.get("sql_type"),
            ))
            validation.append(record(
                "field_definition_validation",
                table,
                validation_text,
                field=field,
                sql_type=definition.get("sql_type"),
            ))

    join_tables = [name for name in tables if name.startswith("computer_join_")]
    for left_index, left in enumerate(join_tables):
        for right in join_tables[left_index + 1 :]:
            shared = sorted(set(tables[left]) & set(tables[right]))
            if not shared:
                continue
            fields = "、".join(shared)
            train.append(record(
                "join_relation",
                "computer_join_*",
                f"表{left}与表{right}共有字段：{fields}。多表查询应按照问题的数据粒度选择合适的共同标识字段连接，避免笛卡尔积。",
                left=left,
                right=right,
                shared_fields=shared,
            ))
            validation.append(record(
                "join_relation_validation",
                "computer_join_*",
                f"Schema知识：{left}和{right}可通过共同字段连接，共同字段包括{fields}。",
                left=left,
                right=right,
                shared_fields=shared,
            ))
    return train, validation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = arguments()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size必须大于0")
    catalog = json.loads(args.schema.read_text(encoding="utf-8-sig"))
    train, validation = build(catalog, args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "schema_train.jsonl", train)
    write_jsonl(args.output_dir / "schema_validation.jsonl", validation)
    summary = {
        "schema": str(args.schema),
        "tables": len(catalog["tables"]),
        "fields": sum(len(fields) for fields in catalog["tables"].values()),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_kinds": dict(sorted(Counter(item["kind"] for item in train).items())),
        "chunk_size": args.chunk_size,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
