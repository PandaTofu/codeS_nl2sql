import json
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def call(path, body=None):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read().decode("utf-8")


print("health", call("/health"))
print("predict_batch", call("/predict", {
    "data": [{"id": "q001", "question": "查询处理器品牌为Intel的台式机"}]
}))
print("predict_flat", call("/predict", {
    "id": "q002", "question": "查询三星手机的型号"
}))
print("predict_invalid", call("/predict", {}))


stream_request = urllib.request.Request(
    BASE_URL + "/free_task",
    data=json.dumps({"id": "q003", "question": "测试首字", "stream": True}).encode("utf-8"),
    headers={"Content-Type": "application/json; charset=utf-8"},
)
with urllib.request.urlopen(stream_request, timeout=30) as response:
    print("free_task", response.status)
    for line in response:
        print(line.decode("utf-8").rstrip())
