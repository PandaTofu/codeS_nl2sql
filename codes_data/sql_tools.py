import random

import sqlglot
from sqlglot import exp


NUMERIC_TYPES = {"INTEGER", "INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"}


def parse_single_select(sql):
    statements = sqlglot.parse(sql, read="sqlite")
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("only one SELECT statement is supported")
    if any(statements[0].find(kind) for kind in (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop)):
        raise ValueError("unsafe SQL")
    return statements[0]


def referenced_tables(tree):
    return {node.name for node in tree.find_all(exp.Table)}


def validate_sql(sql, catalog, expected_table=None):
    tree = parse_single_select(sql)
    tables = referenced_tables(tree)
    if expected_table and tables != {expected_table}:
        raise ValueError(f"expected only {expected_table}, got {sorted(tables)}")
    if not tables or any(table not in catalog for table in tables):
        raise ValueError(f"unknown table: {sorted(tables)}")
    available = set().union(*(catalog[table] for table in tables))
    unknown = sorted({column.name for column in tree.find_all(exp.Column) if column.name not in available})
    if unknown:
        raise ValueError(f"unknown columns: {unknown}")
    return tree.sql(dialect="sqlite", identify=False)


def _same_type_fields(fields, current):
    current_type = str(fields[current].get("sql_type", "TEXT")).upper()
    numeric = current_type in NUMERIC_TYPES
    return [
        name for name, info in fields.items()
        if name != current and (str(info.get("sql_type", "TEXT")).upper() in NUMERIC_TYPES) == numeric
    ]


def mutate_sql(sql, table, fields, rng):
    tree = parse_single_select(sql).copy()
    candidates = []
    literals = list(tree.find_all(exp.Literal))
    for literal in literals:
        if literal.is_number:
            candidates.append(("numeric_value", literal))
        elif literal.is_string:
            parent = literal.parent
            column = parent.left if isinstance(parent, exp.Binary) else None
            if isinstance(column, exp.Column):
                values = [str(v) for v in fields.get(column.name, {}).get("values", [])]
                alternatives = [v for v in values if v != literal.this and len(v) > 1]
                if alternatives:
                    candidates.append(("enum_value", (literal, alternatives)))
    for column in tree.find_all(exp.Column):
        if column.name in fields and _same_type_fields(fields, column.name):
            candidates.append(("field_substitution", column))
    if not candidates:
        raise ValueError("no safe mutation candidate")
    kind, target = rng.choice(candidates)
    detail = {}
    if kind == "numeric_value":
        value = float(target.this)
        delta = max(1.0, abs(value) * rng.choice((0.1, 0.2)))
        new_value = max(0.0, value + rng.choice((-delta, delta)))
        rendered = str(int(round(new_value))) if target.this.isdigit() else f"{new_value:.2f}".rstrip("0").rstrip(".")
        detail = {"from": target.this, "to": rendered}
        target.replace(exp.Literal.number(rendered))
    elif kind == "enum_value":
        literal, alternatives = target
        value = rng.choice(alternatives)
        detail = {"from": literal.this, "to": value}
        literal.replace(exp.Literal.string(value))
    else:
        alternatives = _same_type_fields(fields, target.name)
        value = rng.choice(alternatives)
        detail = {"from": target.name, "to": value}
        target.set("this", exp.Identifier(this=value, quoted=False))
    mutated = tree.sql(dialect="sqlite", identify=False)
    if sqlglot.parse_one(mutated, read="sqlite") == parse_single_select(sql):
        raise ValueError("mutation did not change SQL")
    return mutated, {"kind": kind, **detail}


def schema_text(table, fields, sql=None):
    referenced = set()
    if sql:
        referenced = {column.name for column in parse_single_select(sql).find_all(exp.Column)}
    lines = []
    for name, info in fields.items():
        if referenced and name not in referenced:
            continue
        sql_type = str(info.get("sql_type", "TEXT")).upper()
        values = [str(value) for value in info.get("values", [])]
        suffix = f"; 可选值={values[:20]}" if values and len(values) <= 20 else ""
        lines.append(f"- {name}: {sql_type}{suffix}")
    return f"表名：{table}\n字段：\n" + "\n".join(lines)
