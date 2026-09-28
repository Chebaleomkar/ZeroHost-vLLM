# Data from our run

Collected on 2026-09-28 from one Kaggle T4 session (vLLM 0.30.0, Qwen2.5-7B-Instruct-AWQ, max-model-len 32768, max-num-seqs 16). Almost all traffic is load testing (GuideLLM) plus manual testing with Postman and a coding agent, not external users. The ngrok domain is replaced with `YOUR-DOMAIN`.

| File | Contents |
|---|---|
| `traces_metadata.jsonl` | 3,501 request traces, **sanitized**: timings, token counts, status, finish reason, stream/tools flags. Prompt and output text and client IPs were removed. `error_type` is mostly `ClientConnectionResetError`, from GuideLLM cancelling in-flight requests when each benchmark window ended. `ttfb_s` is time to first byte, not true TTFT. |
| `metrics.jsonl` | 347 snapshots, 15 s apart: vLLM Prometheus counters and gauges, plus GPU 0 utilization, memory, temperature and power. |
| `startup_bench/bench_c{1,4,8,16}.json` | `vllm bench serve` results at each concurrency, run inside Kaggle directly against vLLM. |
| `vllm.log` | vLLM server log for the session. |
| `guidellm/` | GuideLLM reports (JSON, CSV, HTML; open the HTML in a browser). |

## Which GuideLLM runs to trust

| Run | Status |
|---|---|
| `20260928_204307_p512_o128_sweep` | **Clean.** Use this one. Levels 1-2 are distorted by the sweep's own throughput phase (~512 requests fired at once); levels 3-9 are clean. |
| `20260928_202714_p256_o64_synchronous` | Clean single-user smoke test. |
| `20260928_202805_p512_o128_sweep`, `20260928_202832_p512_o128_sweep` | **Contaminated.** Two sweeps (plus other tests) ran at the same time. Kept as a lesson in why benchmarks must run in isolation. |
| `20260928_202905_p256_o64_synchronous` | **Contaminated.** 0 requests completed; stuck behind the other runs' queues. |
