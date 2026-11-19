# Implementation Plan: Dual-Backend Inference Pipeline (Safetensors FP8 + GGUF Q4_K_M)

This plan details the implementation of a dual-backend inference architecture. You will be able to switch seamlessly between the **PyTorch Safetensors backend** (FP8/FP16/FP32) and the **`llama.cpp` GGUF backend** (`Q4_K_M` 4-bit quantization) simply by choosing a configuration file or passing a CLI parameter.

---

## Dual Pipeline Architecture

```text
                                 ┌──> PyTorch Engine (Safetensors FP8/FP16/FP32)
OLMoInferenceEngine (Unified) ───┼
                                 └──> llama.cpp Engine (GGUF Q4_K_M 4-bit CUDA)
```

Both backends share the **exact same Python API, telemetry format, FastAPI server endpoints (`/generate`, `/health`, `/metrics`), and client**.

---

## User Review Required

> [!IMPORTANT]
> - **Dual Support**: No existing code or PyTorch workflow will be broken. PyTorch Safetensors (FP8/FP16/FP32) remains fully functional alongside GGUF `Q4_K_M`.
> - **GGUF Conversion Tool**: We will add `src/convert_gguf.py` to convert PyTorch Safetensors checkpoints directly into `.gguf` format (`Q4_K_M`), or allow downloading a pre-converted GGUF file.
> - **Dependencies**: `llama-cpp-python` will be added to `requirements.txt`.

---

## Proposed Changes

### Dependencies

#### [MODIFY] [requirements.txt](file:///d:/Waris/Projects/Olmo%20Experiments/requirements.txt)
- Add `llama-cpp-python` to requirements.

---

### Configurations

#### [NEW] [configs/inference_gguf.yaml](file:///d:/Waris/Projects/Olmo%20Experiments/configs/inference_gguf.yaml)
- Configuration file for the GGUF `Q4_K_M` 4-bit pipeline:
  ```yaml
  model:
    backend: "llamacpp"
    path: "artifacts/OLMo-2-0425-1B-Instruct-Q4_K_M.gguf"
    n_gpu_layers: -1     # Offload 100% of layers to Quadro P4000 GPU
    n_ctx: 4096

  generation:
    max_new_tokens: 128
    temperature: 0.7
    top_p: 0.9
  ```

#### [MODIFY] [configs/inference_fp8.yaml](file:///d:/Waris/Projects/Olmo%20Experiments/configs/inference_fp8.yaml)
- Ensure the existing FP8 Safetensors config clearly specifies `backend: "pytorch"`.

---

### Conversion Utility

#### [NEW] [src/convert_gguf.py](file:///d:/Waris/Projects/Olmo%20Experiments/src/convert_gguf.py)
- Script to convert Hugging Face Safetensors weights into a quantized `.gguf` artifact (`Q4_K_M` or `Q8_0`).

---

### Core Inference Engine

#### [MODIFY] [src/inference.py](file:///d:/Waris/Projects/Olmo%20Experiments/src/inference.py)
- Update `OLMoInferenceEngine`:
  - Detect `backend` key (`llamacpp` vs `pytorch`) or file extension (`.gguf`).
  - If `llamacpp`: Initialize `llama_cpp.Llama` with `n_gpu_layers=-1` and CUDA acceleration.
  - Implement `generate()` for `llama.cpp`, mapping outputs and execution metrics into the standard telemetry dictionary (`throughput_tokens_per_sec`, `tokenization_seconds`, `generation_seconds`, `memory_allocated_mb`).
  - If `pytorch`: Execute standard PyTorch / Hugging Face pipeline as before.

---

### Documentation & Commands

#### [MODIFY] [Commands.txt](file:///d:/Waris/Projects/Olmo%20Experiments/Commands.txt)
- Add command examples for running FP8 PyTorch inference vs. GGUF `Q4_K_M` inference.

#### [MODIFY] [README.md](file:///d:/Waris/Projects/Olmo%20Experiments/README.md)
- Document the dual-backend architecture and performance comparison metrics.

---

## Verification Plan

### Automated Tests
1. **Install `llama-cpp-python`:**
   ```powershell
   uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/wheels/cxx11/cu121 --native-tls
   ```
2. **Verify Safetensors FP8 Pipeline:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.inference --config configs/inference_fp8.yaml --prompt "Testing FP8 Safetensors pipeline:"
   ```
3. **Verify GGUF Q4_K_M Pipeline:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.inference --config configs/inference_gguf.yaml --prompt "Testing GGUF Q4_K_M pipeline:"
   ```
4. **Compare Throughput Metrics:**
   Verify tokens/sec metrics emitted by both engines and confirm server compatibility with `app/server.py`.
