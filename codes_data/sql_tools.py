import random

import sqlglot
from sqlglot import exp


NUMERIC_TYPES = {"INTEGER", "INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"}
COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def parse_single_select(sql):
    statements = sqlglot.parse(sql, read="sqlite")
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("only one SELECT statement is supported")
    if any(statements[0].find(kind) for kind in (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop)):
        raise ValueError("unsafe SQL")
    return statements[0]


def referenced_tables(tree):
    return {node.name for node in tree.find_all(exp.Table)}


def _field_kind(info):
    raw = str(info.get("type", ""))
    sql_type = str(info.get("sql_type", "TEXT")).upper()
    values = {str(value) for value in info.get("values", [])}
    if "布尔" in raw or values == {"是", "否"}:
        return "boolean"
    if "日期" in raw or "时间" in raw:
        return "date"
    if sql_type in NUMERIC_TYPES:
        return "integer" if sql_type in {"INTEGER", "INT"} else "number"
    if values:
        return "enum"
    return "text"


def _predicate_parts(node):
    if isinstance(node, COMPARISONS) and isinstance(node.left, exp.Column) and isinstance(node.right, exp.Literal):
        yield node.left, node.right, node
    if isinstance(node, exp.In) and isinstance(node.this, exp.Column):
        for value in node.expressions:
            if isinstance(value, exp.Literal):
                yield node.this, value, node


def validate_sql(sql, catalog, expected_table=None, strict_types=False):
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
    rendered_projections = [item.sql(dialect="sqlite", identify=True) for item in tree.expressions]
    if len(rendered_projections) != len(set(rendered_projections)):
        raise ValueError("duplicate SELECT expressions")
    for item in tree.expressions:
        value = item.this if isinstance(item, exp.Alias) else item
        if isinstance(value, exp.Literal):
            raise ValueError("literal SELECT expression is not allowed")
    if strict_types:
        for node in tree.walk():
            for column, literal, predicate in _predicate_parts(node):
                info = next((catalog[table][column.name] for table in tables if column.name in catalog[table]), {})
                kind = _field_kind(info)
                value = str(literal.this)
                if kind in {"integer", "number"} and literal.is_string:
                    raise ValueError(f"numeric field uses quoted value: {column.name}")
                if kind == "boolean" and value not in {"是", "否"}:
                    raise ValueError(f"invalid boolean value: {column.name}={value}")
                if kind == "boolean" and not isinstance(predicate, (exp.EQ, exp.NEQ, exp.In)):
                    raise ValueError(f"invalid boolean operator: {column.name}")
                values = {str(item) for item in info.get("values", [])}
                if kind == "enum" and isinstance(predicate, (exp.EQ, exp.In)) and values and value not in values:
                    raise ValueError(f"unknown enum value: {column.name}={value}")
    return tree.sql(dialect="sqlite", identify=True)


def _same_type_fields(fields, current):
    current_type = _field_kind(fields[current])
    return [
        name for name, info in fields.items()
        if name != current and _field_kind(info) == current_type
    ]


def mutate_sql(sql, table, fields, rng):
    tree = parse_single_select(sql).copy()
    candidates = []
    for node in tree.walk():
        for column, literal, predicate in _predicate_parts(node):
            kind = _field_kind(fields.get(column.name, {}))
            if literal.is_number and kind in {"integer", "number"}:
                candidates.append(("numeric_value", (literal, kind)))
            elif literal.is_string and kind in {"enum", "boolean"}:
                values = [str(v) for v in fields.get(column.name, {}).get("values", [])]
                alternatives = [v for v in values if v != literal.this]
                if alternatives:
                    candidates.append(("enum_value", (literal, alternatives)))
            if isinstance(predicate, (exp.GT, exp.GTE, exp.LT, exp.LTE)) and kind in {"integer", "number", "date"}:
                candidates.append(("comparison_operator", predicate))
    used_columns = {column.name for column in tree.find_all(exp.Column)}
    for projection in tree.expressions:
        if isinstance(projection, exp.Column) and projection.name in fields:
            alternatives = [name for name in _same_type_fields(fields, projection.name) if name not in used_columns]
            if alternatives:
                candidates.append(("select_field_substitution", (projection, alternatives)))
    for ordered in tree.find_all(exp.Ordered):
        candidates.append(("order_direction", ordered))
    limit = tree.args.get("limit")
    if limit and isinstance(limit.expression, exp.Literal) and limit.expression.is_number:
        candidates.append(("limit_value", limit.expression))
    if not candidates:
        raise ValueError("no safe mutation candidate")
    kind, target = rng.choice(candidates)
    detail = {}
    if kind == "numeric_value":
        target, field_kind = target
        value = float(target.this)
        delta = max(1.0, abs(value) * rng.choice((0.1, 0.2)))
        new_value = max(0.0, value + rng.choice((-delta, delta)))
        rendered = str(int(round(new_value))) if field_kind == "integer" else f"{new_value:.2f}".rstrip("0").rstrip(".")
        detail = {"from": target.this, "to": rendered}
        target.replace(exp.Literal.number(rendered))
    elif kind == "enum_value":
        literal, alternatives = target
        value = rng.choice(alternatives)
        detail = {"from": literal.this, "to": value}
        literal.replace(exp.Literal.string(value))
    elif kind == "select_field_substitution":
        target, alternatives = target
        value = rng.choice(alternatives)
        detail = {"from": target.name, "to": value}
        target.set("this", exp.Identifier(this=value, quoted=False))
    elif kind == "comparison_operator":
        replacements = {
            exp.GT: exp.GTE, exp.GTE: exp.GT,
            exp.LT: exp.LTE, exp.LTE: exp.LT,
        }
        replacement = replacements[type(target)](this=target.this.copy(), expression=target.expression.copy())
        detail = {"from": target.key.upper(), "to": replacement.key.upper()}
        target.replace(replacement)
    elif kind == "order_direction":
        old = "DESC" if target.args.get("desc") else "ASC"
        target.set("desc", not bool(target.args.get("desc")))
        detail = {"from": old, "to": "ASC" if old == "DESC" else "DESC"}
    else:
        value = int(target.this)
        new_value = value + rng.choice((-2, -1, 1, 2))
        if new_value <= 0:
            new_value = value + 1
        detail = {"from": value, "to": new_value}
        target.replace(exp.Literal.number(new_value))
    mutated = tree.sql(dialect="sqlite", identify=True)
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
