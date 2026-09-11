from __future__ import annotations

from decimal import Decimal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


INTEGER_COUNT_COLUMNS = frozenset(name.lower() for name in (
    "雷电接口数量", "USB接口数量", "USB_A接口数量", "USB_C接口数量",
    "USB_Type_C接口数量", "HDMI接口数量", "DisplayPort接口数量",
    "内存插槽数", "内存插槽数量", "网卡数量", "风扇数量", "扬声器数量",
    "麦克风数量", "摄像头数量", "墨盒数量", "随机耗材数量",
    "实体按键数量", "自动菜单数量", "库存数量", "支持多点连接数量",
))


def normalize_output_aliases(tree):
    if not isinstance(tree, exp.Select):
        return
    projections = [item for item in tree.expressions if isinstance(item, exp.Alias)]
    aliases = {item.alias: item.this for item in projections}
    if len(aliases) != len(projections):
        return
    source_columns = {column.name for item in tree.expressions for column in item.find_all(exp.Column)}
    replacements = []
    for key in ("order", "group", "having"):
        clause = tree.args.get(key)
        if clause is None:
            continue
        for node in clause.walk(prune=lambda n: isinstance(n, exp.Query)):
            if isinstance(node, exp.Column) and not node.table and node.name in aliases:
                expression = aliases[node.name]
                if node.name in source_columns and not (
                    isinstance(expression, exp.Column) and expression.name == node.name
                ):
                    return
                replacements.append((node, expression.copy()))
    for node, expression in replacements:
        node.replace(expression)
    # 子查询的输出别名可能被外层引用，保留其接口。
    tree.set("expressions", [item.this if isinstance(item, exp.Alias) else item for item in tree.expressions])


def parse_sql(sql):
    statements = sqlglot.parse(sql, read="mysql")
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("需要一条 SELECT 查询")
    tree = statements[0]
    if any(isinstance(node, (exp.DDL, exp.DML, exp.Into)) for node in tree.walk()):
        raise ValueError("不允许写入语句")
    for node in tree.find_all(exp.Identifier):
        node.set("this", node.this.lower())
        node.set("quoted", False)
    normalize_output_aliases(tree)
    for select in tree.find_all(exp.Select):
        source = select.args.get("from_")
        sources = ([source.this] if source else []) + [j.this for j in select.args.get("joins", [])]
        tables = [s for s in sources if isinstance(s, exp.Table)]
        names = [".".join(part.name for part in t.parts) for t in tables]
        aliases = {t.alias_or_name: name for t, name in zip(tables, names) if names.count(name) == 1}
        nodes = list(select.walk(prune=lambda n: n is not select and isinstance(n, exp.Query)))
        output_aliases = {e.alias for e in select.expressions if e.alias}
        for node in nodes:
            if isinstance(node, exp.Column):
                if node.table in aliases:
                    node.set("table", exp.to_identifier(aliases[node.table]))
                elif not node.table and len(sources) == len(tables) == 1 and node.name not in output_aliases:
                    node.set("table", exp.to_identifier(names[0]))
        for table in tables:
            if table.alias_or_name in aliases:
                table.set("alias", None)
        joins = select.args.get("joins", [])
        for join in joins:
            if str(join.args.get("kind", "")).upper() == "INNER":
                join.set("kind", None)
        if source and joins and all(isinstance(s, exp.Table) for s in sources) and all(
            not any(v for k, v in j.args.items() if k != "this") for j in joins
        ):
            sources.sort(key=lambda s: s.sql())
            source.set("this", sources[0])
            select.set("joins", [exp.Join(this=s) for s in sources[1:]])
    return tree


def unwrap(node):
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def number(node):
    node = unwrap(node)
    if isinstance(node, exp.Neg):
        value = number(node.this)
        return -value if value is not None else None
    if isinstance(node, exp.Literal) and not node.is_string:
        return Decimal(node.this)
    return None


def combine(results):
    results = list(results)
    return (
        min((r[0] for r in results), default=1.0),
        sorted({rule for r in results for rule in r[1]}),
        [reason for r in results for reason in r[2]],
    )


def unordered(gold, pred, path, relaxed=False):
    if len(gold) != len(pred):
        return 0.0, [], [f"{path}: 项目数量不同 ({len(gold)} / {len(pred)})"]
    matrix = [[match(g, p, path, relaxed) for p in pred] for g in gold]
    for threshold in (1.0, 0.9, 0.8):
        assigned = {}

        def assign(i, seen):
            for j, result in enumerate(matrix[i]):
                if j not in seen and result[0] >= threshold:
                    seen.add(j)
                    if j not in assigned or assign(assigned[j], seen):
                        assigned[j] = i
                        return True
            return False

        if all(assign(i, set()) for i in range(len(gold))):
            return combine(matrix[i][j] for j, i in assigned.items())
    return 0.0, [], [f"{path}: 存在无法匹配的项目"]


def conjunction(node):
    node = unwrap(node)
    if isinstance(node, exp.And):
        return conjunction(node.this) + conjunction(node.expression)
    return [node]


def model_filter(gold, pred):
    def models(items):
        return {(item.this.table, item.expression.this) for item in items
                if isinstance(item, exp.EQ) and isinstance(item.this, exp.Column)
                and item.this.name == "型号" and isinstance(item.expression, exp.Literal)
                and item.expression.is_string}

    common = models(gold) & models(pred)
    tables = {table for table, _ in common if table}

    def keep(item):
        return not (isinstance(item, exp.EQ) and isinstance(item.this, exp.Column)
                    and item.this.table in tables and item.this.name in {"品牌", "系列"}
                    and isinstance(item.expression, exp.Literal) and item.expression.is_string)

    return [g for g in gold if keep(g)], [p for p in pred if keep(p)]


def match(gold, pred, path="sql", relaxed=False):
    gold, pred = unwrap(gold), unwrap(pred)
    if isinstance(gold, exp.Expression) and isinstance(pred, exp.Expression):
        gnum, pnum = number(gold), number(pred)
        if gnum is not None and pnum is not None and gnum == pnum:
            return 1.0, ["numeric_format"] if gold != pred else [], []
        if isinstance(gold, exp.And) and isinstance(pred, exp.And):
            gold_items, pred_items = conjunction(gold), conjunction(pred)
            if relaxed:
                filtered_gold, filtered_pred = model_filter(gold_items, pred_items)
                result = unordered(filtered_gold, filtered_pred, path + ".and", relaxed)
                if len(gold_items) != len(filtered_gold) or len(pred_items) != len(filtered_pred):
                    return combine((result, (1.0, ["model_brand_series"], [])))
                return result
            return unordered(gold_items, pred_items, path + ".and", relaxed)
        if isinstance(gold, exp.EQ) and isinstance(pred, exp.EQ):
            direct = combine((
                match(gold.this, pred.this, path + ".eq.left", relaxed),
                match(gold.expression, pred.expression, path + ".eq.right", relaxed),
            ))
            reversed_result = combine((
                match(gold.this, pred.expression, path + ".eq.left", relaxed),
                match(gold.expression, pred.this, path + ".eq.right", relaxed),
            ))
            if reversed_result[0] > direct[0]:
                return combine((reversed_result, (1.0, ["commutative_equality"], [])))
            return direct
        if relaxed:
            if isinstance(gold, exp.And) or isinstance(pred, exp.And):
                gs, ps = conjunction(gold), conjunction(pred)
                filtered_g, filtered_p = model_filter(gs, ps)
                result = unordered(filtered_g, filtered_p, path + ".and", True)
                if len(gs) != len(filtered_g) or len(ps) != len(filtered_p):
                    result = combine([result, (1.0, ["model_brand_series"], [])])
                return result
            if isinstance(gold, (exp.GT, exp.GTE, exp.LT, exp.LTE)) and isinstance(pred, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                same_column = gold.this == pred.this
                gv, pv = number(gold.expression), number(pred.expression)
                if same_column and gv is not None and pv is not None:
                    if gv == pv and type(gold) != type(pred) and (
                        isinstance(gold, (exp.GT, exp.GTE)) == isinstance(pred, (exp.GT, exp.GTE))
                    ):
                        return 1.0, ["open_closed_boundary"], []
                    pairs = {(type(gold), gv), (type(pred), pv)}
                    if isinstance(gold.this, exp.Column) and gold.this.name in INTEGER_COUNT_COLUMNS and pairs == {(exp.GT, Decimal(0)), (exp.GTE, Decimal(1))}:
                        return 1.0, ["integer_count"], []
            if {type(gold), type(pred)} == {exp.EQ, exp.Like} and gold.this == pred.this and isinstance(gold.this, exp.Column):
                like, equal = (gold, pred) if isinstance(gold, exp.Like) else (pred, gold)
                lv, ev = like.expression, equal.expression
                if isinstance(lv, exp.Literal) and isinstance(ev, exp.Literal) and lv.is_string and ev.is_string:
                    if lv.this == "%" + ev.this + "%" and not any(c in ev.this for c in "%_\\"):
                        return (0.8 if like is gold else 0.9), ["like_equal"], [f"{path}: LIKE 与等值条件部分匹配"]
        if type(gold) is not type(pred):
            return 0.0, [], [f"{path}: {gold.key} / {pred.key} 不同"]
        results = []
        for key in sorted(gold.args.keys() | pred.args.keys()):
            g, p = gold.args.get(key), pred.args.get(key)
            child_path = f"{path}.{gold.key}.{key}"
            child_relaxed = isinstance(gold, exp.Where) or (relaxed and isinstance(gold, exp.And))
            positional = isinstance(gold, exp.Select) and any(
                isinstance(n, exp.Literal) and not n.is_string
                for clause in (gold.args.get("order"), gold.args.get("group"), pred.args.get("order"), pred.args.get("group"))
                if clause for n in clause.walk()
            )
            if key == "expressions" and isinstance(gold, (exp.Select, exp.Group, exp.In)) and not positional:
                results.append(unordered(g or [], p or [], child_path))
            else:
                results.append(match(g, p, child_path, child_relaxed))
        return combine(results)
    if isinstance(gold, list) and isinstance(pred, list):
        if len(gold) == len(pred):
            return combine(match(g, p, f"{path}[{i}]", relaxed) for i, (g, p) in enumerate(zip(gold, pred)))
    elif gold == pred:
        return 1.0, [], []
    return 0.0, [], [f"{path}: {gold!s} / {pred!s} 不同"]


def compare_sql(gold_sql: str, predicted_sql: str) -> dict:
    trees, errors = [], {}
    for label, sql in (("gold", gold_sql), ("predicted", predicted_sql)):
        try:
            trees.append(parse_sql(sql))
        except (SqlglotError, ValueError) as error:
            errors[label] = str(error)
    if errors:
        return {"rule_score": 0.0, "rule_match": False, "matched_rules": [],
                "differences": ["SQL 解析失败或不是单条只读查询"], "parse_errors": errors}
    score, rules, differences = match(*trees)
    return {"rule_score": score, "rule_match": score == 1.0,
            "matched_rules": rules if score else [], "differences": differences,
            "parse_errors": {}}

