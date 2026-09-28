"""Local check of the logging proxy against a fake vLLM upstream: python test_proxy.py"""
import asyncio
import json
import tempfile

from aiohttp import ClientSession, web
from yarl import URL

import zerohost_vllm as zh

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


async def fake_vllm(request):
    body = await request.json()
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for piece in ["Hel", "lo"]:
            await resp.write(f'data: {json.dumps({"choices": [{"delta": {"content": piece}, "finish_reason": None}]})}\n\n'.encode())
        await resp.write(f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\ndata: [DONE]\n\n'.encode())
        return resp
    return web.json_response({"choices": [{"message": {"content": "Hi there"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 2}})


async def main():
    zh.DATA_DIR = tempfile.mkdtemp()
    zh.VLLM_PORT, zh.PROXY_PORT = 18001, 18000
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", fake_vllm)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", zh.VLLM_PORT).start()
    asyncio.create_task(zh.serve_proxy(KEY))
    await asyncio.sleep(0.5)

    url = f"http://127.0.0.1:{zh.PROXY_PORT}"
    req = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    async with ClientSession() as s:
        async with s.post(f"{url}/v1/chat/completions", json=req) as r:
            assert r.status == 401, r.status
        async with s.post(f"{url}/v1/chat/completions", json=req, headers=AUTH) as r:
            assert (await r.json())["choices"][0]["message"]["content"] == "Hi there"
        async with s.post(f"{url}/v1/chat/completions", json={**req, "stream": True}, headers=AUTH) as r:
            assert "Hel" in await r.text()
        async with s.get(f"{url}/admin/files/requests.jsonl", headers=AUTH) as r:
            traces = [json.loads(l) for l in (await r.text()).splitlines()]
        async with s.get(URL(f"{url}/admin/files/..%2F..%2Fsecrets.json", encoded=True), headers=AUTH) as r:
            assert r.status == 404, r.status

    assert len(traces) == 2, traces  # rejected (401) calls are not traced
    plain, stream = traces
    assert plain["output"] == "Hi there" and plain["usage"]["completion_tokens"] == 2 and plain["tokens_per_s"] > 0
    assert plain["request"]["messages"][0]["content"] == "hello"
    assert stream["output"] == "Hello" and stream["finish_reason"] == "stop" and stream["ttfb_s"] >= 0
    print("all proxy checks passed")
    print(json.dumps(stream, indent=1))


asyncio.run(main())
