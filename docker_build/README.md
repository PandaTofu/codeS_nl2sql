# 复赛 Docker API

该目录提供 `GET /health`、`POST /predict` 和 `POST /free_task`。

`/predict` 使用与 Query-only CodeS 训练一致的固定 Prompt。推理超时时调用 vLLM `abort` 终止请求，并返回HTTP 200、非空SQL和非空自然语言回复。

合并基础模型与Adapter：

```bash
python docker_build/merge_adapter.py \
  --base /home/ubuntu/models/CodeS-3B \
  --adapter training/query_only_codes3b_mixed_v1/best_adapter \
  --output models/CodeS-3B-NL2SQL-Merged
```

从仓库根目录构建并运行：

```bash
docker build --platform linux/amd64 \
  -f docker_build/Dockerfile \
  -t codes-nl2sql-semifinal:latest .

docker run --rm --gpus all -p 8000:8000 --name codes-nl2sql-api \
  codes-nl2sql-semifinal:latest
```

另开终端验证：

```bash
python3 docker_build/test_api.py
```
