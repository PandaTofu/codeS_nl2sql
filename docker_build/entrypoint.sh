#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${NL2SQL_CONFIG:-/app/config.json}"
SERVICE_HOST="$(python3 -c "import json; print(json.load(open('${CONFIG_PATH}', encoding='utf-8'))['service']['host'])")"
SERVICE_PORT="$(python3 -c "import json; print(json.load(open('${CONFIG_PATH}', encoding='utf-8'))['service']['port'])")"

exec python3 -m uvicorn app:app --host "${SERVICE_HOST}" --port "${SERVICE_PORT}"
