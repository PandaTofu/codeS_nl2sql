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

## 训练前审计

使用 `audit_dataset.py` 检查重复样本、SQL结构、Schema、字段类型和值域。语义审查由生成阶段的第二次教师模型请求完成，但它仍不等于人工标注或数据库执行验证。

## Query-only CodeS-3B 基线

`train_query_only_qlora.py` 使用固定提示词和 Query 直接监督生成 SQL，不把 Schema 放入训练输入。训练只计算 SQL Token 的损失，并以人工原始样本60%、SQL驱动增强30%、问题驱动增强10%的比例加权采样。每个 Epoch 在隔离的人工验证集上评估，最终保存验证损失最低的 Adapter。

`evaluate_query_only_adapter.py` 在隔离验证集上执行确定性推理，使用已验证的 MySQL SQL 分层规则评分，输出严格 EM、规则满分率、平均规则分、静态有效率和推理耗时。

`build_hard_training_set.py` 根据训练集回放报告重复错误样本和复杂 SQL 结构，不读取隔离验证集。`train_query_only_qlora.py --adapter` 从现有 Adapter 继续训练。

`augment_validation_queries.py` 为160条验证样本各生成2条经教师审查的等价Query，单独保存320条增强数据，并与第二轮困难训练集合并。该流程会产生验证集泄漏，后续在原160条上的评估不再代表独立泛化能力。

`rewrite_manual_queries.py` 对1000条人工原始样本做经教师审查的轻量Query同义改写，保持ID和SQL不变；随后将第一轮合并训练集中的840条 `manual_original` 替换为对应改写版。旧训练集缺少ID时，使用原始Query和SQL唯一反查ID。

## 表路由与 Schema Linker 监督数据

`build_schema_supervision.py` 使用 SQLGlot 从带标准 SQL 的样本中提取物理表、字段及其 SELECT/FILTER/GROUP/HAVING/ORDER/JOIN 角色。它输出表路由样本、字段角色样本，以及适合训练字段相关性二分类器的正负候选对。

推荐保留 `round1_val.jsonl` 为显式验证来源：

```bash
python build_schema_supervision.py \
  --input data/manual_raw_train.jsonl \
  --input data/round1_test.jsonl \
  --validation-input data/round1_val.jsonl \
  --schema data/schema_catalog.json \
  --output-dir training/schema_supervision_v1 \
  --hard-negatives 2 \
  --random-negatives 2 \
  --seed 42
```

若三个文件都作为同一数据池，则省略 `--validation-input`，将它也作为 `--input`；脚本会按规范化 Query 的稳定哈希划分15%验证集，确保完全相同的 Query 不会跨集合。

## 固定 Schema 领域训练

`build_schema_domain_corpus.py` 把13张表的表用途、字段分组、字段类型、描述、枚举值和关联表共同字段转换为短篇领域语料。`train_schema_qlora.py` 在原始 CodeS-3B 上执行领域 CLM QLoRA；该阶段不训练 Query→SQL，也不应从现有 NL2SQL Adapter 继续训练。

```bash
python build_schema_domain_corpus.py \
  --schema data/schema_catalog.json \
  --output-dir training/schema_domain_v1 \
  --chunk-size 10

python train_schema_qlora.py \
  --model /home/ubuntu/models/CodeS-3B \
  --train training/schema_domain_v1/schema_train.jsonl \
  --validation training/schema_domain_v1/schema_validation.jsonl \
  --output training/codes3b_schema_qlora_v1 \
  --epochs 2 \
  --max-length 384 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation 8 \
  --learning-rate 5e-5
```

训练结束后把领域 Adapter 合并为中间模型：

```bash
python docker_build/merge_adapter.py \
  --base /home/ubuntu/models/CodeS-3B \
  --adapter training/codes3b_schema_qlora_v1/best_adapter \
  --output models/CodeS-3B-Schema-v1
```

下一阶段以 `models/CodeS-3B-Schema-v1` 为基础模型，使用 Query→SQL 数据重新训练新的 NL2SQL Adapter；不要覆盖现有 Query-only 基线。
