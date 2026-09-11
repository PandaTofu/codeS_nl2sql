FROM vllm/vllm-openai:v0.10.2

WORKDIR /app

ARG MODEL_DIR=models/CodeS-3B-NL2SQL-Merged

COPY docker_build/app.py docker_build/config.json docker_build/entrypoint.sh /app/
COPY ${MODEL_DIR}/ /app/model/

RUN chmod +x /app/entrypoint.sh \
    && python3 -c "import fastapi, uvicorn, vllm"

ENV NL2SQL_CONFIG=/app/config.json
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
