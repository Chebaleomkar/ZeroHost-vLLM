"""ZeroHost vLLM: serve Qwen2.5-7B-Instruct-AWQ on one Kaggle T4 behind an ngrok static domain.

Secrets (NGROK_AUTHTOKEN, NGROK_DOMAIN, VLLM_API_KEY) are read from secrets.json in your
private <kaggle-username>/zerohost-secrets dataset (created by setup.py).

Traffic: ngrok -> logging proxy (PROXY_PORT) -> vLLM (VLLM_PORT). Everything observable is written
to DATA_DIR: requests.jsonl (one trace per request), metrics.jsonl (15s snapshots),
bench_c*.json (startup benchmarks) and vllm.log. Download live via GET /admin/files/<name>.
"""
import asyncio
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
SERVED_NAME = "qwen2.5-7b-instruct"
VLLM_SPEC = "vllm==0.30.0"  # verified on Kaggle T4, driver 580 / CUDA 13.0
PROXY_PORT = 8000
VLLM_PORT = 8001
DATA_DIR = "/kaggle/working/zerohost"
LOG_PATH = f"{DATA_DIR}/vllm.log"
STARTUP_TIMEOUT_S = 20 * 60
CHECK_INTERVAL_S = 30
METRICS_INTERVAL_S = 15
RUN_HOURS = 11.75  # exit before Kaggle's 12h kill so /kaggle/working is saved as output
BENCH_CONCURRENCY = [1, 4, 8, 16]
BENCH_TIMEOUT_S = 10 * 60  # per concurrency level; a hung benchmark must not block the tunnel


def log(msg):
    print(f"[zerohost {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_jsonl(name, record, lock=threading.Lock()):
    with lock, open(f"{DATA_DIR}/{name}", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def install():
    log(f"installing {VLLM_SPEC} and pyngrok")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", VLLM_SPEC, "pyngrok"], check=True)


def load_secrets():
    # CLI pushes drop attached Kaggle Secrets, so they come from the private zerohost-secrets dataset
    paths = glob.glob("/kaggle/input/**/secrets.json", recursive=True)
    if not paths:
        raise FileNotFoundError("secrets.json not found; attach the private zerohost-secrets dataset")
    with open(paths[0]) as f:
        return json.load(f)


def start_vllm(api_key):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--dtype", "half",  # T4 (sm75) has no bf16
        "--quantization", "awq",
        "--max-model-len", "32768",  # Qwen2.5 native context; agent clients need the room
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "16",
        "--served-model-name", SERVED_NAME,
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",  # OpenAI tool calls for Qwen2.5
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_API_KEY": api_key}
    logfile = open(LOG_PATH, "a")
    log(f"starting vLLM, logging to {LOG_PATH}")
    return subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def tail_log(n=60):
    with open(LOG_PATH) as f:
        return "".join(f.readlines()[-n:])


def wait_until_healthy(proc):
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}:\n{tail_log()}")
        if healthy():
            log("vLLM is healthy")
            return
        time.sleep(10)
    raise TimeoutError(f"vLLM not healthy after {STARTUP_TIMEOUT_S}s:\n{tail_log()}")


def run_benchmarks(api_key):
    """Load-test vLLM directly (bypassing the proxy, so these don't appear in requests.jsonl)."""
    env = {**os.environ, "OPENAI_API_KEY": api_key}
    for c in BENCH_CONCURRENCY:
        log(f"benchmark: concurrency {c}")
        try:
            result = subprocess.run([
            sys.executable, "-c", "from vllm.entrypoints.cli.main import main; main()", "bench", "serve",
                "--base-url", f"http://127.0.0.1:{VLLM_PORT}",
                "--model", SERVED_NAME, "--tokenizer", MODEL,
                "--dataset-name", "random", "--random-input-len", "512", "--random-output-len", "128",
                "--num-prompts", str(max(16, c * 4)), "--max-concurrency", str(c),
                "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
                "--save-result", "--result-dir", DATA_DIR, "--result-filename", f"bench_c{c}.json",
            ], env=env, capture_output=True, text=True, timeout=BENCH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log(f"benchmark c={c} timed out after {BENCH_TIMEOUT_S}s, skipping the rest")
            return
        if result.returncode != 0:
            log(f"benchmark c={c} failed, skipping the rest:\n{(result.stdout + result.stderr)[-2000:]}")
            return
        summary = [l for l in result.stdout.splitlines() if "throughput" in l.lower() or "Median TTFT" in l]
        log("\n".join(summary))


def sample_metrics():
    """Snapshot vLLM's Prometheus counters and GPU 0 usage every METRICS_INTERVAL_S."""
    while True:
        record = {"ts": time.time()}
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/metrics", timeout=5) as r:
                for line in r.read().decode().splitlines():
                    if line.startswith("vllm:") and "_bucket{" not in line and "_created{" not in line:
                        name, value = line.rsplit(" ", 1)
                        record[name] = float(value)
            gpu = subprocess.run(
                ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            record["gpu_util_pct"], record["gpu_mem_mib"], record["gpu_temp_c"], record["gpu_power_w"] = (
                float(x) for x in gpu.split(","))
        except Exception as e:
            record["error"] = str(e)
        append_jsonl("metrics.jsonl", record)
        time.sleep(METRICS_INTERVAL_S)


def summarize_response(raw, content_type):
    """Pull output text, usage and finish_reason out of a JSON or SSE completion response."""
    if "text/event-stream" in content_type:
        text, usage, finish = [], None, None
        for line in raw.decode(errors="replace").splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                text.append((choice.get("delta") or {}).get("content") or choice.get("text") or "")
                finish = choice.get("finish_reason") or finish
        return {"output": "".join(text), "usage": usage, "finish_reason": finish}
    body = json.loads(raw)
    choice = (body.get("choices") or [{}])[0]
    output = (choice.get("message") or {}).get("content") or choice.get("text")
    return {"output": output, "usage": body.get("usage"), "finish_reason": choice.get("finish_reason")}


async def serve_proxy(api_key):
    import aiohttp
    from aiohttp import web

    upstream = f"http://127.0.0.1:{VLLM_PORT}"
    hop_headers = {"host", "content-length", "transfer-encoding", "connection"}
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def handle(request):
        if request.path != "/health" and request.headers.get("Authorization") != f"Bearer {api_key}":
            return web.json_response({"error": "unauthorized"}, status=401)
        if request.path == "/admin/files":
            return web.json_response(sorted(os.listdir(DATA_DIR)))
        if request.path.startswith("/admin/files/"):
            path = os.path.join(DATA_DIR, os.path.basename(request.path))
            return web.FileResponse(path) if os.path.isfile(path) else web.Response(status=404)

        body = await request.read()
        trace = {"id": str(uuid.uuid4()), "ts": time.time(), "method": request.method, "path": request.path_qs,
                 "client_ip": request.headers.get("X-Forwarded-For", request.remote)}
        try:
            trace["request"] = json.loads(body) if body else None
        except ValueError:
            trace["request"] = body.decode(errors="replace")
        t0 = time.perf_counter()
        chunks = []
        try:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in hop_headers}
            async with session.request(request.method, upstream + request.path_qs, headers=headers, data=body) as up:
                trace["status"] = up.status
                resp = web.StreamResponse(status=up.status, headers={
                    k: v for k, v in up.headers.items() if k.lower() not in hop_headers})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    if not chunks:
                        trace["ttfb_s"] = round(time.perf_counter() - t0, 4)
                    chunks.append(chunk)
                    await resp.write(chunk)
                await resp.write_eof()
                content_type = up.headers.get("Content-Type", "")
            return resp
        except Exception as e:
            trace["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            trace["latency_s"] = round(time.perf_counter() - t0, 4)
            if chunks and request.path.startswith("/v1/") and "error" not in trace:
                try:
                    trace.update(summarize_response(b"".join(chunks), content_type))
                    tokens = (trace.get("usage") or {}).get("completion_tokens")
                    if tokens:
                        trace["tokens_per_s"] = round(tokens / trace["latency_s"], 2)
                except Exception as e:
                    trace["parse_error"] = str(e)
            append_jsonl("requests.jsonl", trace)

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PROXY_PORT).start()
    log(f"logging proxy on :{PROXY_PORT} -> vLLM :{VLLM_PORT}")
    await asyncio.Event().wait()


def start_daemon(target, *args):
    threading.Thread(target=target, args=args, daemon=True).start()


def open_tunnel(secrets, attempts=10):
    from pyngrok import conf, ngrok

    conf.get_default().auth_token = secrets["NGROK_AUTHTOKEN"]
    # retries cover a previous session still holding the static domain during a redeploy
    for attempt in range(1, attempts + 1):
        try:
            tunnel = ngrok.connect(PROXY_PORT, domain=secrets["NGROK_DOMAIN"])
            log(f"PUBLIC_URL {tunnel.public_url}/v1")
            return tunnel
        except Exception as e:
            log(f"tunnel attempt {attempt}/{attempts} failed: {e}")
            if attempt == attempts:
                raise
            time.sleep(30)


def supervise(proc, secrets, deadline):
    from pyngrok import ngrok

    while time.time() < deadline:
        time.sleep(CHECK_INTERVAL_S)
        if proc.poll() is not None:
            log(f"vLLM died (code {proc.returncode}), restarting:\n{tail_log(20)}")
            proc = start_vllm(secrets["VLLM_API_KEY"])
            wait_until_healthy(proc)
        try:
            if not ngrok.get_tunnels():
                log("tunnel missing, reconnecting")
                open_tunnel(secrets)
        except Exception as e:
            log(f"tunnel check failed: {e}")
    log(f"RUN_HOURS={RUN_HOURS} reached, shutting down so outputs are saved")
    ngrok.kill()
    proc.terminate()


def main():
    deadline = time.time() + RUN_HOURS * 3600
    secrets = load_secrets()  # fail fast before the long install if secrets aren't attached
    os.makedirs(DATA_DIR, exist_ok=True)
    subprocess.run(["nvidia-smi"])
    install()
    proc = start_vllm(secrets["VLLM_API_KEY"])
    wait_until_healthy(proc)
    start_daemon(sample_metrics)
    run_benchmarks(secrets["VLLM_API_KEY"])
    start_daemon(lambda: asyncio.run(serve_proxy(secrets["VLLM_API_KEY"])))
    open_tunnel(secrets)
    supervise(proc, secrets, deadline)  # blocks so the Kaggle session stays alive


if __name__ == "__main__":
    main()
