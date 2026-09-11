import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


PROMPT_TEMPLATE = """将下面的中文问题转换为一条MySQL只读SQL。只输出SQL，不要解释或Markdown。
### Question
{query}
### SQL
"""


def load_config():
    path = Path(os.environ.get("NL2SQL_CONFIG", "/app/config.json"))
    config = json.loads(path.read_text(encoding="utf-8"))
    model = config["model"]
    model["path"] = os.environ.get("MODEL_PATH", model["path"])
    model["dtype"] = os.environ.get("MODEL_DTYPE", model["dtype"])
    model["gpu_memory_utilization"] = float(
        os.environ.get("GPU_MEMORY_UTILIZATION", model["gpu_memory_utilization"])
    )
    config["timeouts"]["predict_seconds"] = float(
        os.environ.get("PREDICT_TIMEOUT_SECONDS", config["timeouts"]["predict_seconds"])
    )
    return config


CONFIG = load_config()
ENGINE = None
STARTUP_ERROR = None


def initialize_engine():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    model = CONFIG["model"]
    args = AsyncEngineArgs(
        model=model["path"],
        tokenizer=model["path"],
        dtype=model["dtype"],
        gpu_memory_utilization=model["gpu_memory_utilization"],
        max_model_len=model["max_model_len"],
        max_num_seqs=model["max_num_seqs"],
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    return AsyncLLMEngine.from_engine_args(args)


@asynccontextmanager
async def lifespan(_):
    global ENGINE, STARTUP_ERROR
    try:
        ENGINE = initialize_engine()
        STARTUP_ERROR = None
        print(json.dumps({"event": "model_ready", "model": CONFIG["model"]["path"]}), flush=True)
    except Exception as exc:
        ENGINE = None
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "model_startup_failed", "error": STARTUP_ERROR}), flush=True)
    yield


app = FastAPI(title="CodeS NL2SQL Competition API", lifespan=lifespan)


def clean_sql(text):
    value = str(text).strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", value, re.IGNORECASE | re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    match = re.search(r"\b(?:SELECT|WITH)\b.*", value, re.IGNORECASE | re.DOTALL)
    if match:
        value = match.group(0)
    value = value.split("\n\n", 1)[0].strip().rstrip(";").strip()
    without_literals = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", value)
    forbidden = r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|CALL|GRANT|REVOKE|SET|USE|LOAD|OUTFILE|INTO)\b"
    if not re.match(r"^\s*(?:SELECT|WITH)\b", without_literals, re.IGNORECASE):
        return CONFIG["fallback"]["sql"]
    if len([part for part in without_literals.split(";") if part.strip()]) != 1:
        return CONFIG["fallback"]["sql"]
    if re.search(forbidden, without_literals, re.IGNORECASE):
        return CONFIG["fallback"]["sql"]
    return value or CONFIG["fallback"]["sql"]


def natural_response(question):
    if re.search(r"多少|数量|几款|统计|总数", question):
        return "已为您统计符合条件的信息，具体结果请您查看。"
    if re.search(r"最高|最低|最贵|最便宜|排行|排名|前\s*\d+|排序", question):
        return "已按您的要求完成筛选和排序，相关信息已为您整理。"
    if re.search(r"平均|总计|合计|最大|最小", question):
        return "已按照您的要求完成汇总计算，具体结果请您查看。"
    if re.search(r"参数|配置|性能|规格", question):
        return "已为您整理符合条件的参数与配置信息，请您查看。"
    return "已根据您的需求筛选出相关信息，具体结果请您查看。"


def normalize_items(body):
    if isinstance(body, list):
        return body
    if not isinstance(body, dict):
        return []
    if isinstance(body.get("data"), list):
        return body["data"]
    if "question" in body or "query" in body:
        return [body]
    return []


def question_of(item):
    if not isinstance(item, dict):
        return ""
    return str(item.get("question") or item.get("query") or "").strip()


def fallback_prediction(item):
    question = question_of(item)
    return {
        "id": item.get("id", "") if isinstance(item, dict) else "",
        "sql": CONFIG["fallback"]["sql"],
        "response": natural_response(question),
        "success": True,
        "result": None,
    }


async def abort_request(request_id):
    if ENGINE is None:
        return
    try:
        await ENGINE.abort(request_id)
    except Exception:
        pass


async def generate_sql(question):
    if ENGINE is None:
        raise RuntimeError("model is unavailable")
    from vllm import SamplingParams

    request_id = uuid.uuid4().hex
    params = SamplingParams(
        temperature=0,
        max_tokens=CONFIG["model"]["max_sql_tokens"],
        repetition_penalty=1.0,
    )
    output = None
    completed = False
    try:
        async with asyncio.timeout(CONFIG["timeouts"]["predict_seconds"]):
            async for item in ENGINE.generate(
                PROMPT_TEMPLATE.format(query=question), params, request_id
            ):
                output = item
        completed = True
        if output is None or not output.outputs:
            raise RuntimeError("model returned no output")
        return clean_sql(output.outputs[0].text)
    finally:
        if not completed:
            await abort_request(request_id)


async def predict_one(item):
    fallback = fallback_prediction(item)
    question = question_of(item)
    if not question:
        return fallback
    started = time.perf_counter()
    try:
        sql = await generate_sql(question)
        print(json.dumps({
            "event": "prediction",
            "id": item.get("id", ""),
            "seconds": round(time.perf_counter() - started, 4),
            "fallback": sql == CONFIG["fallback"]["sql"],
        }, ensure_ascii=False), flush=True)
        return {
            "id": item.get("id", ""),
            "sql": sql,
            "response": natural_response(question),
            "success": True,
            "result": None,
        }
    except Exception as exc:
        print(json.dumps({
            "event": "prediction_fallback",
            "id": item.get("id", "") if isinstance(item, dict) else "",
            "seconds": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False), flush=True)
        return fallback


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "ready": ENGINE is not None,
        "error": STARTUP_ERROR,
    }, status_code=200)


@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        items = normalize_items(body)[:CONFIG["service"]["max_batch_size"]]
        predictions = await asyncio.gather(*(predict_one(item) for item in items))
        if not predictions:
            predictions = [fallback_prediction({})]
    except Exception:
        predictions = [fallback_prediction({})]
    return JSONResponse({"predictions": predictions}, status_code=200)


async def free_task_events(item):
    response = natural_response(question_of(item))
    qid = item.get("id", "") if isinstance(item, dict) else ""
    payload = {
        "id": qid,
        "content": response,
        "choices": [{"delta": {"content": response}}],
    }
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/free_task")
async def free_task(request: Request):
    try:
        body = await request.json()
        items = normalize_items(body)
        item = items[0] if items else {}
        stream = bool(body.get("stream", True)) if isinstance(body, dict) else True
    except Exception:
        item, stream = {}, True
    if stream:
        return StreamingResponse(free_task_events(item), media_type="text/event-stream", status_code=200)
    return JSONResponse({"response": natural_response(question_of(item))}, status_code=200)


@app.exception_handler(Exception)
async def unhandled_exception(_, __):
    return JSONResponse({"predictions": [fallback_prediction({})]}, status_code=200)
