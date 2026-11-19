# OLMo-2-0425-1B-Instruct-FP8 Inference System Implementation Plan

## 1. Goal
Build a production-grade remote inference system for `OLMo-2-0425-1B-Instruct-FP8` running on a Quadro P4000, with remote access from other machines, safe context management, telemetry, and quality-aware configuration.

The selected implementation path is a custom inference service built around the existing `src/inference.py` engine, wrapped by a lightweight FastAPI server.

---

## 2. Architecture Overview

### 2.1 Core components
- `Model Host`: Single GPU process that loads and serves the model.
- `API Server`: Exposes inference endpoints over HTTP or gRPC.
- `Request Queue`: Controls concurrency and serializes requests on the limited GPU.
- `Telemetry & Logging`: Tracks latency, GPU memory, token usage, and quality indicators.
- `Client(s)`: Remote machines that send inference requests to the API.

### 2.2 High-level flow
1. System boots and loads the FP8 model into GPU memory.
2. API server starts and exposes `/health`, `/generate`, `/metrics`, `/reload`.
3. Clients send prompts and generation parameters.
4. Server validates request length, enqueues it, and processes sequentially.
5. Outputs are returned, logs persisted, and telemetry updated.

---

## 3. Platform and Hardware Constraints

### 3.1 Quadro P4000 constraints
- Total VRAM: 8 GB.
- Working set in experiments: ~2.8–3.4 GB for generation.
- KV cache scales with output size and context size.
- Very long outputs (75–100% context) slow to 6–13 tok/s.
- Concurrency: likely 1 request at a time for safe operation.

### 3.2 Practical limits
- Prefer `max_new_tokens` well below full context for remote use.
- Use 10%/25%/50% context as safer presets.
- Reserve GPU headroom for OS/process overhead.

---

## 4. Model and Inference Configuration

### 4.1 Base model
- `artifacts/OLMo-2-0425-1B-Instruct-FP8`
- loaded with `torch_dtype=torch.float16` on CUDA
- `disable_eos=True` when forcing exact max token counts

### 4.2 Recommended generation settings
- `temperature`: 0.7
- `top_p`: 0.9
- `top_k`: 50
- `repetition_penalty`: 1.1
- `do_sample`: true
- `pad_token_id`: from tokenizer

### 4.3 Safe token presets
- 10% full context: fast, high quality
- 25% full context: still good
- 50% full context: marginal quality drop
- 75% full context: likely repetition/drift
- 100% full context: not recommended for production

---

## 5. API Design

### 5.1 Endpoints
- `GET /health`
  - returns model loaded state, device status, and uptime.
- `GET /metrics`
  - returns GPU usage, request count, average latency.
- `POST /generate`
  - body: `prompt`, `max_new_tokens`, `temperature`, `top_p`, `disable_eos`, `model_path`
  - response: `text`, `prompt_tokens`, `generated_tokens`, `latency`, `cache_mb`, `warnings`
- `POST /validate`
  - optional: compute prompt length and recommend safe max_new_tokens.
- `POST /reload`
  - optional: reload model from disk when switching artifacts.

### 5.2 Request validation
- Tokenize prompt and measure `prompt_length`.
- Verify `prompt_length + max_new_tokens <= max_position_embeddings`.
- Reject or truncate if the request would overflow.
- Return a structured warning if request is near the limit.

---

## 6. Implementation Plan

### 6.1 Step 1: Dependency management
- Add or update `requirements.txt` with:
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `transformers`
  - `torch`
  - `safetensors`
  - `psutil`
  - `python-multipart` (if needed)

### 6.2 Step 2: Core API server
- Create `app/server.py` or `app/main.py` as a thin wrapper around the existing `src/inference.py` engine.
- Use `FastAPI` with a single global model instance.
- Keep the server simple: validate requests, serialize GPU access, call `OLMoInferenceEngine.generate()`, and return structured output.
- Add middleware for request IDs and CORS if needed.
- Build a model loader utility in `src/inference.py` and expose a loader function.

### 6.3 Step 3: Request queue / locking
- Implement an in-process lock or queue using `asyncio.Lock` or `threading.Lock`.
- Ensure `/generate` waits for the GPU to be available.
- Optionally use a bounded async queue with `Queue(maxsize=1)`.

### 6.4 Step 4: Telemetry and logging
- Add structured JSON logging for each request.
- Log:
  - request ID
  - prompt length
  - requested and returned tokens
  - latency
  - model path
  - GPU memory
  - warnings and errors
- Expose `/metrics` with CPU/GPU telemetry and request counters.

### 6.5 Step 5: Context management
- Add prompt length tracking in the request validator.
- Return `recommended_max_new_tokens` if overflow is detected.
- Optionally support a default generation cap such as `max_new_tokens = min(requested, safe_max)`.

### 6.6 Step 6: Remote access and security
- Bind only to the internal network or `127.0.0.1`.
- Use SSH tunnel or VPN for remote access if needed.
- Add API key header authentication if the service is exposed beyond localhost.
- Keep the service behind a firewall or reverse proxy when possible.

---

## 7. Production considerations

### 7.1 Stability
- Use a process manager like `systemd`, `pm2`, or `supervisord`.
- Keep the model loaded persistently.
- Restart on failure.

### 7.2 Monitoring
- Collect:
  - GPU memory usage
  - request latency
  - error counts
  - average token throughput
- Use logs or a simple metrics endpoint.

### 7.3 Scalability
- Quadro P4000 is not horizontally scalable from one machine.
- For more throughput, use a stronger GPU or multiple GPU hosts behind a load balancer.
- For now, design as a single-GPU, low-concurrency service.

### 7.4 Cost of long outputs
- Very large `max_new_tokens` causes much slower throughput.
- Prefer 128 / 256 / 512 as default remote options.
- Use full-context only for offline/experimental workloads.

---

## 8. Example Implementation Structure

```
app/
  server.py
  schemas.py
  config.py
src/
  inference.py
  utils.py
  evaluate.py
experiments/
  fp8_context_experiment.py
configs/
  inference_fp16.yaml
  inference_fp8.yaml
requirements.txt
```

### app/server.py responsibilities
- load model once
- validate requests
- queue inference calls
- return structured response
- log telemetry

### src/inference.py responsibilities
- tokenizer/model loading
- generate() abstraction
- prompt token counting and warnings
- GPU memory helpers

---

## 9. Suggested roadmap

### Phase 1: Prototype
- Build a FastAPI `generate` endpoint.
- Add token-length validation.
- Deploy on the P4000 machine.
- Test from another computer using curl.

### Phase 2: Harden
- Add queueing / request lock.
- Add `/health` and `/metrics`.
- Add persistent logs.
- Add remote access controls.

### Phase 3: Quality & tuning
- Add preset `max_new_tokens` values: 10%, 25%, 50%, 75%.
- Add a client-side slider if you want a UI.
- Monitor quality drift and restrict 100% context to non-production use.

---

## 10. Recommended first commands

```bash
cd "d:/Waris/Projects/Olmo Experiments"
.venv\Scripts\python.exe -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Then test from another machine:

```bash
curl -X POST http://<gpu-host>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short greeting to a new user.","max_new_tokens":128}'
```

---

## 11. Risks and mitigation

| Risk | Why it matters | Mitigation |
|---|---|---|
| GPU OOM | P4000 has only 8 GB | cap tokens, queue requests, reserve headroom |
| Long latency | remote clients time out | enforce shorter max responses, async service |
| Quality drift | 100% context is noisy | use 10/25/50% presets only |
| Dependency failures | SSL/install issues | use locked environment and offline installs |
| Security exposure | open API accessible | use internal network and auth |

---

## 12. Final recommendation
For this hardware, build it as a **single-host, low-concurrency inference service** using the existing custom `src/inference.py` engine wrapped by a lightweight FastAPI server. Treat the full 4096-context mode as an experimental option, not the default remote mode. Start with 128–512 output tokens and only allow higher generation lengths after you confirm infrastructure stability.

---

## 13. Next implementation step I can do now
- build the actual `FastAPI` service in `app/server.py`
- add a `configs/inference_fp8.yaml`
- add request validation and telemetry
- provide a sample `curl` client workflow
