# Olmo Experiments

Auditable, production-grade repository for downloading, converting, serving, and evaluating the **OLMo-2 1B** language model.

---

## 📂 Repository Structure

```text
Olmo Experiments
├── .gitignore
├── README.md               # Master technical documentation, assumptions & hardware log
├── requirements.txt         # Pinned software versions
├── configs/                # Hyperparameter and configuration tracking
│   ├── inference_fp16.yaml
│   └── lora_finetune.yaml
├── data/                   # Data registries (strictly git-ignored for raw items)
│   ├── processed/
│   └── raw/
├── notebooks/              # For dirty exploration / scratchpad analysis only
│   └── 01_initial_exploration.ipynb
├── src/                    # Auditable production-grade source code
│   ├── __init__.py
│   ├── download.py         # Handles explicit repository artifact pulling
│   ├── convert.py          # Precision transformations and weight exports
│   ├── inference.py        # Optimized serving engine interface
│   ├── evaluate.py         # Standardized benchmark runner
│   └── utils.py            # Logging and telemetry configurations
└── artifacts/              # Serialization path for local weights (git-ignored)
    ├── OLMo-2-0425-1B/
    ├── OLMo-2-0425-1B-FP16/
    └── adapters/
```

---

## 🛠️ Environment Setup

1. **Create Python Virtual Environment:**
   ```bash
   uv venv .venv
   .venv\Scripts\activate
   ```

2. **Install Pinned Dependencies:**
   ```bash
   uv pip install -r requirements.txt --native-tls
   ```

---

## 💻 Hardware Log & Resource Assumptions

The table below outlines expected resource utilization metrics for working with the **OLMo-2 1B** model:

| Task / Setup | Model Precision | Target Hardware | Expected VRAM | Max Context Length | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inference** | `float32` | NVIDIA CUDA (8GB+) | ~4.5 GB | 2048 | Baseline benchmark, memory heavy |
| **Inference** | `float16` / `bfloat16` | NVIDIA CUDA / Apple MPS | ~2.5 GB | 2048 | Recommended for production |
| **Inference** | `GGUF Q4_K_M` | CPU / CUDA (4GB+) | ~1.5 GB | 4096 | Optimized for low-resource HW |
| **LoRA Fine-tune** | `bfloat16` (Mixed) | NVIDIA CUDA (16GB+) | ~12.0 GB | 1024 | Rank=8, batch=2, gradient accumulation=4 |

### Hardware Assertions & Assumptions
1. **PyTorch Version**: Compiled against PyTorch 2.3.1 with CUDA or MPS execution support.
2. **Flash Attention / SDPA**: Defaulting to PyTorch Native Scaled Dot Product Attention (`sdpa`) for optimized latency.
3. **Storage Overhead**:
   - `artifacts/OLMo-2-0425-1B/`: ~4 GB (raw weights)
   - `artifacts/OLMo-2-0425-1B-FP16/`: ~2 GB (exported FP16 weights)

---

## 🚀 Execution Guide

### 1. Download Model
Pull explicit weights from Hugging Face Hub (defaulting to `allenai/OLMo-2-0425-1B` or specified checkpoint):
```bash
python -m src.download --repo-id allenai/OLMo-2-0425-1B --output-dir artifacts/OLMo-2-0425-1B
```

### 2. Precision Conversion
Convert downloaded FP32/BF16 weights to explicit FP16 format and save in the target artifacts directory:
```bash
python -m src.convert --input-dir artifacts/OLMo-2-0425-1B --output-dir artifacts/OLMo-2-0425-1B-FP16
```

### 3. serving Engine / Inference
Run generation benchmarks using configured YAML inputs:
```bash
python -m src.inference --config configs/inference_fp16.yaml --prompt "Explain quantum computing in one sentence:"
```

### 4. Chat-style inference
Use a system/user prompt without writing a custom script.

Single system/user example:
```bash
python -m src.inference --config configs/inference_fp16.yaml \
  --system "You are a helpful assistant." \
  --user "Summarize the difference between FP16 and BF16."
```

### 5. Generate a very large response
To experiment with a short prompt and a huge output, override `--max-new-tokens`.

For example, if your prompt is 10 tokens and you want roughly 4086 new tokens:
```bash
python -m src.inference --config configs/inference_fp16.yaml \
  --prompt "A quick test for the model." \
  --max-new-tokens 4086 \
  --log-file data/processed/inference_large_response.jsonl
```

If you want to disable EOS stopping and force generation until the token limit, use:
```bash
python -m src.inference --config configs/inference_fp16.yaml \
  --prompt "A quick test for the model." \
  --max-new-tokens 4086 \
  --disable-eos \
  --log-file data/processed/inference_large_response.jsonl
```

> Note: the model’s hard limit is 4096 total tokens, so keep `prompt_tokens + max_new_tokens <= 4096`.

Multiple messages example:
```bash
python -m src.inference --config configs/inference_fp16.yaml \
  --message "system:You are a helpful assistant." \
  --message "user:Please list three advantages of half-precision inference." \
  --message "assistant:Sure, here are the benefits so far." \
  --message "user:Now compare FP16 and BF16 performance."
```

JSON messages file example (`messages.json`):
```json
[
  {"role": "system", "content": "You are a helpful AI assistant."},
  {"role": "user", "content": "Explain how to run this model in FP16."}
]
```

Then run:
```bash
python -m src.inference --config configs/inference_fp16.yaml --messages-json messages.json
```

### 5. Dynamic Configuration Overrides (OmegaConf)
You can dynamically override nested parameters inside the YAML configuration file from the CLI using dot-notation without editing the physical file. Any trailing CLI arguments that are not registered flags will be merged as config overrides.

Example:
```bash
python -m src.inference --config configs/inference_fp16.yaml \
  --prompt "A test prompt" \
  generation.temperature=0.99 \
  generation.max_new_tokens=64 \
  model.attn_implementation=eager
```

### Optional inference logging
Add `--log-file` to persist each query, response, and metrics.

```bash
python -m src.inference --config configs/inference_fp16.yaml \
  --system "You are a helpful assistant." \
  --user "Summarize FP16 vs BF16." \
  --log-file data/processed/inference_logs.jsonl
```

### 4. Evaluate Performance
Compute Perplexity, Time to First Token (TTFT), tokens per second throughput, and peak memory usage.

Run evaluation for a single model path:
```bash
python -m src.evaluate --model-path artifacts/OLMo-2-0425-1B-FP16
```

Run evaluation across FP16 and BF16 only:
```bash
python -m src.evaluate --evaluate-all --precision-mode half --output-json data/processed/eval_half_precisions.json
```

If you want to include FP32 despite VRAM pressure, use:
```bash
python -m src.evaluate --evaluate-all --precision-mode all --output-json data/processed/eval_all_precisions.json
```

You can also benchmark a custom set of inference configs:
```bash
python -m src.evaluate --config-list configs/inference_fp16.yaml configs/inference_bf16.yaml
```

---

## 📈 Scaling, Portability, and Hardware Fallbacks

The repository is built to scale gracefully between low-resource development GPUs (like the NVIDIA Quadro P4000) and multi-GPU cluster workstations (like the 4x RTX Pro 6000 Blackwell).

### 1. Dynamic Hardware Fallbacks
The inference pipeline automatically checks CUDA compute capability when `device_map` is set to `cuda`.
- **Pascal Fallback**: If the compute capability is less than Volta (`7.0`) (e.g., Quadro P4000), it automatically overrides precision to `float32` and falls back to standard eager attention to prevent severe emulation performance degradation.
- **Modern GPU Support**: On newer architectures (Turing, Ampere, Ada, Blackwell), it dynamically runs fast `float16`, `bfloat16`, or `float8` / `fp8` execution.

### 2. Environment Variable Configuration
You can override the model loading path globally without modifying the YAML files:
```powershell
# Set path override in Windows Powershell
$env:OLMO_MODEL_PATH="artifacts/OLMo-2-0425-1B"
python -m src.inference --prompt "Explain scale..."
```

### 3. Unified Inference Client (Local vs. API Serving)
The codebase includes a unified wrapper, `OLMoClient`, that can either load weights and run model generation locally, or route requests over the network to a high-performance server (like vLLM) running on your GPU cluster.

To query a remote vLLM/OpenAI-compatible server directly via the CLI, pass the `--server-url` argument (or define the `OLMO_SERVER_URL` environment variable):
```bash
# Query a remote vLLM server
python -m src.inference --server-url http://<server-ip>:8000/v1/completions --prompt "Hello from API mode"
```
