# CodeS NL2SQL 数据增强

本工程从 `data/manual_raw_train.jsonl` 构建 CodeS 领域微调数据。原始数据固定划分后，只使用训练区生成增强样本，验证区保持隔离。

增强包含两条路径：

- `sql_to_question`：使用 `sqlglot` 对 SQL 做受控变异，再由 Qwen3-14B 写成自然中文问题。
- `question_to_sql`：由 Qwen3-14B 根据原始样本和 Schema 生成新的问题与 SQL，再进行 SQL 解析及 Schema 校验。

当前数据不含真实数据库，因此只能验证 SQL 语法、表名和字段名，不能验证执行结果与问题语义。生成结果中的 `database_execution` 固定为 `false`。

## 输出文件

- `augmentation.jsonl`：增强样本及来源追踪信息。
- `manual_train.jsonl`：固定划分后的人工原始训练区。
- `manual_validation.jsonl`：严格隔离的人工验证区。
- `codes_sft_train.jsonl`：人工训练区与增强数据的合并训练集。
- `failures.jsonl`：生成或校验失败记录。
- `summary.json`：本次生成汇总。

