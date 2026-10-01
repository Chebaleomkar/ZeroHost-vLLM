# ZeroHost vLLM

**A production-style LLM inference server on free compute.** Qwen2.5-7B-Instruct (4-bit AWQ) served with vLLM on a free Kaggle T4 GPU, exposed as an authenticated, OpenAI-compatible HTTPS API through an ngrok static domain. It supports tool calling, logs a trace for every request, and was benchmarked with 3,500+ requests.

Point any OpenAI client at it: Postman, your backend, the OpenAI SDK, or a coding agent.

```
 your app / Postman / coding agent
            │  HTTPS + Bearer key
            ▼
   ngrok static domain  (https://<you>.ngrok-free.app/v1)
            │
 ┌──────────┴──────────────── Kaggle notebook, T4 GPU ──────────────┐
 │  logging proxy :8000  ── auth, full per-request traces (JSONL)   │
 │            │                                                     │
 │  vLLM :8001  ── Qwen2.5-7B-Instruct-AWQ, fp16, 32K ctx, tools    │
 │                                                                  │
 │  metrics sampler ── vLLM Prometheus + GPU stats every 15s        │
 │  supervisor      ── restarts vLLM/tunnel, clean exit at 11h45m   │
 └──────────────────────────────────────────────────────────────────┘
```

---

## Quick start (about 25 minutes, mostly waiting)

### What you need
- A **Kaggle account** with phone verification (required for GPU and internet access). Free tier: 30 GPU hours/week, 12 hours per session.
- A free **ngrok account**:
  - Your **authtoken**: [dashboard.ngrok.com](https://dashboard.ngrok.com) > Your Authtoken
  - Your free **static domain**: dashboard > Domains (every account gets one, e.g. `something.ngrok-free.app`)
- **Python 3.10+** on your machine.

### 1. Clone and log in to Kaggle
```bash
git clone https://github.com/Chebaleomkar/ZeroHost-vLLM.git
cd ZeroHost-vLLM
pip install kaggle
python -m kaggle auth login        # opens the browser; or put kaggle.json in ~/.kaggle/
```

### 2. Run the setup
```bash
python setup.py
```
It asks for your ngrok authtoken (input is hidden) and domain, and generates an API key for you. It then:
- uploads them as a **private** Kaggle dataset `<you>/zerohost-secrets`
- writes `kernel-metadata.json` for your account
- optionally pushes the notebook, which starts the server

Your secrets stay in `zerohost-secrets/`, which is git-ignored. Nothing secret lives in the code.

### 3. Wait for it to come up (~15-20 min)
```bash
python -m kaggle kernels logs -f <your-kaggle-username>/zerohost-vllm
```
Startup: install vLLM (~5 min), load the model (~3-5 min), run a quick startup benchmark (~5 min), then open the tunnel. It's ready when the log prints:
```
[zerohost 20:07:39] PUBLIC_URL https://<your-domain>/v1
```

### 4. Call it
```bash
curl https://<your-domain>/v1/chat/completions \
  -H "Authorization: Bearer <your API key>" \
  -H "ngrok-skip-browser-warning: 1" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-domain>/v1",
    api_key="<your API key>",
    default_headers={"ngrok-skip-browser-warning": "1"},
)
reply = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Explain KV cache in two sentences."}],
)
print(reply.choices[0].message.content)
```

Tool calling works the standard OpenAI way (`tools=[...]`, `tool_choice="auto"`).

### 5. Stop it
In Kaggle: **Active Events** (bottom-left of the sidebar), then **Stop session**. Otherwise it runs until the 12h limit and uses your weekly GPU quota. The Kaggle CLI can't stop sessions.

To restart later: `python -m kaggle kernels push -p .` (same URL, ready in ~15-20 min).
To rotate keys: run `python setup.py` again, then push.

---

## Using it from a coding agent (Pi example)

Add a provider to `~/.pi/agent/models.json`:
```json
"zerohost": {
  "baseUrl": "https://<your-domain>/v1",
  "api": "openai-completions",
  "headers": { "ngrok-skip-browser-warning": "1" },
  "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false,
              "supportsStore": false, "maxTokensField": "max_tokens" },
  "models": [{ "id": "qwen2.5-7b-instruct", "contextWindow": 32768, "maxTokens": 4096 }]
}
```
Put the key in `~/.pi/agent/auth.json` as `"zerohost": { "type": "api_key", "key": "<your API key>" }`, then run `pi --provider zerohost --model qwen2.5-7b-instruct`. The same pattern works for any OpenAI-compatible tool.

---

## Observability: every request is traced

The proxy writes everything to `/kaggle/working/zerohost/`. You can download files while the server runs:
```bash
curl -H "Authorization: Bearer <key>" -H "ngrok-skip-browser-warning: 1" https://<your-domain>/admin/files               # list files
curl -H "Authorization: Bearer <key>" -H "ngrok-skip-browser-warning: 1" -O https://<your-domain>/admin/files/requests.jsonl
```
| File | What's in it |
|---|---|
| `requests.jsonl` | One trace per request: full prompt and output, status, finish reason, token usage, time to first byte, latency, tokens/s |
| `metrics.jsonl` | Every 15s: vLLM Prometheus metrics (TTFT, ITL, queue depth, KV-cache usage) and GPU util, memory, temperature, power |
| `bench_c*.json` | Startup benchmark at concurrency 1/4/8/16 |
| `vllm.log` | Full vLLM server log |

Files are also saved as Kaggle notebook output when the session ends cleanly at 11h45m.

---

## Benchmarks

Measured with [GuideLLM](https://github.com/vllm-project/guidellm) through the public URL (so network time is included), using 512-token prompts and 128-token replies. Raw data is in [`data/`](data/).

**What one user gets**

| Metric | Median | p95 |
|---|---|---|
| Time to first token (TTFT) | 768 ms | 913 ms |
| Inter-token latency (ITL) | 26.6 ms | 27.4 ms |
| Time per output token (TPOT) | 32.6 ms | 34.0 ms |
| Speed | ~30-38 tokens/s | |
| Full 128-token reply | 4.2 s | 4.2 s |

**How it behaves as load grows (clean constant-rate run)**

| Load (req/s) | In flight | TTFT p50 / p95 | ITL p50 | Speed per user | Total output tok/s |
|---|---|---|---|---|---|
| 0.35 | 1.7 | 1,168 / 1,221 ms | 30.3 ms | ~33 tok/s | 42.7 |
| 0.46 | 2.3 | 1,096 / 1,202 ms | 35.0 ms | ~29 tok/s | 53.3 |
| 0.57 | 3.0 | 1,088 / 1,174 ms | 35.9 ms | ~28 tok/s | 68.3 |
| 0.68 | 3.8 | 1,093 / 1,148 ms | 40.6 ms | ~25 tok/s | 78.9 |

**Startup benchmark (`vllm bench serve`, direct to vLLM, no network)**

| Concurrency | Output tok/s | Median TTFT |
|---|---|---|
| 1 | 33.6 | 436 ms |
| 4 | 106.2 | 112 ms |
| 8 | 145.4 | 2,631 ms (outlier) |
| 16 | 214.2 | 489 ms |

**In plain words**
- **1 user:** the first word appears in under a second, then ~38 tokens/s. That's faster than you can read.
- **~4 users generating at once:** ~1.1 s to the first word, ~25 tokens/s each. This is the comfortable limit: about 0.7 requests/s, or ~2,500 replies/hour.
- **Past ~12-16 at once:** requests queue and the wait for the first word grows to 10-20 s.
- Real chat users mostly read and type, so ~4 at once is roughly **40 casual users**, or **2-3 coding-agent users**.

**GPU utilization finding:** under load, GPU 0 ran at 100% compute, but the KV cache used only 2.5% on average (18% peak) of its reserved memory. Kaggle's T4 machine also has a **second T4 that sat idle** (this build pins one GPU). Running a second vLLM on GPU 1 behind the proxy should roughly double capacity. See the roadmap.

---

## The journey (and what broke along the way)

1. **Pick a model that fits a T4.** 16 GB of VRAM, no bf16, compute capability 7.5. Qwen2.5-7B-Instruct-AWQ is ~5.5 GB in 4-bit, which leaves plenty of room for KV cache. The first surprise: Qwen defaults to bf16, which the T4 doesn't support, so `--dtype half` is mandatory.
2. **Run it in the background on Kaggle.** "Background" processes die when the notebook finishes, and a pushed notebook finishes immediately. The script therefore has to block: a supervisor loop that restarts vLLM or the tunnel if either dies, and exits cleanly before Kaggle's 12h kill so the outputs are saved.
3. **Get secrets in without hardcoding them.** Kaggle Secrets worked in the editor but **silently disappeared on every CLI push** (a known Kaggle limitation). The fix: a private Kaggle dataset holding `secrets.json`, created by `setup.py`.
4. **A stable public URL.** An ngrok static domain means the backend URL never changes between sessions. A health check gates the tunnel, so nobody ever gets a 502 during startup.
5. **Make it work for agents.** The first coding-agent request failed with `"auto" tool choice requires --enable-auto-tool-choice`. Adding the Hermes tool parser and a 32K context (vLLM measured 6.58 GiB of KV cache: 123,184 tokens) made tool calls work.
6. **Observe everything.** A small aiohttp proxy in front of vLLM adds auth to every path (vLLM's `/metrics` was publicly readable before), full-text traces per request, and a file endpoint to download them while the server runs.
7. **Benchmark honestly.** The first sweep looked awful: TTFT p95 of 13 s. It turned out three benchmarks were running at once, and GuideLLM's throughput phase fires ~512 requests together. Run in isolation, the same load gave a 1.1 s p95. **Never benchmark two things at once.**
8. **Question "100% GPU".** The busy GPU hid an idle second GPU and a mostly empty KV cache. Measuring utilization is what showed the real next step.

Other gotchas: pushing a new Kaggle version doesn't stop the old one (2 GPU sessions max, and the old one keeps the ngrok domain); stopping sessions is UI-only; GuideLLM doesn't support Windows (use WSL).

---

## Run your own benchmarks

On Linux, macOS or WSL:
```bash
uv tool install "guidellm[recommended]"      # or: pip install "guidellm[recommended]"
bash bench.sh 256 64 synchronous 30          # single user
bash bench.sh 512 128 sweep 60               # automatic load sweep
bash bench.sh 512 128 constant 60 --override profile.rate 0.8,1.0,1.2,1.4
```
Reports (JSON, CSV, HTML) are written to `results/`. `bench.sh` reads your domain and key from `zerohost-secrets/secrets.json`.

---

## Repo layout

| File | Purpose |
|---|---|
| `zerohost_vllm.py` | The Kaggle script: install, vLLM, metrics sampler, startup benchmark, logging proxy, tunnel, supervisor |
| `setup.py` | One-time setup for your account: secrets dataset, kernel metadata, push |
| `kernel-metadata.template.json` | Kaggle kernel config (single T4, GPU + internet on) |
| `bench.sh` | GuideLLM benchmark wrapper |
| `test_proxy.py` | Local test of the logging proxy against a fake vLLM (`python test_proxy.py`) |
| `data/` | Sanitized data from our run: request metadata (no prompt text), metrics, benchmarks, vLLM log |

## Limitations
- Free-tier limits: Kaggle gives 12 h per session and 30 GPU h/week, and ngrok's free tier caps monthly requests and bandwidth. This is for demos, small teams and learning, not public traffic.
- Every start is a fresh machine: ~15-20 min to ready.
- A 7B model handles chat and simple agent tasks well but makes mistakes on complex multi-step coding.
- Traces store full prompts and outputs. Tell your users, and treat downloaded traces as sensitive.

## Roadmap
- Use both T4s: one vLLM per GPU, with the proxy load-balancing between them (~2x capacity).
- Capture tool calls and true TTFT/TPOT in traces; tag requests to separate benchmark and real traffic.
- Cache the pip install and model weights in Kaggle datasets to cut boot time to ~5 min.
- Make the startup benchmark optional.
