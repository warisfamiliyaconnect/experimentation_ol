"""FastAPI server for the OLMo-2 inference engine.

This server wraps the OLMoInferenceEngine with a production-grade HTTP API.
It is hardware-portable: the engine auto-detects the GPU compute capability
and falls back to FP32/eager on Pascal hardware (Quadro P4000) while
utilizing BF16/FP16 on modern Blackwell/Ada/Ampere GPUs.

Environment Variables:
    OLMO_MODEL_PATH   - Override model artifact directory (e.g. artifacts/OLMo-2-0425-1B-FP16)
    OLMO_CONFIG_PATH  - Override YAML config file path (e.g. configs/inference_fp16.yaml)
    OLMO_DEVICE       - Override device (cuda | cpu | mps). Defaults to auto-detect.

Endpoints:
    GET  /health     - Server liveness, model status, uptime
    GET  /metrics    - GPU memory, request counts, average latency
    POST /generate   - Text generation
    POST /validate   - Pre-flight check: fits in context window?
    POST /reload     - Hot-reload model from a new path
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import OLMoInferenceEngine
from src.utils import get_hardware_telemetry, setup_logger
from src.config import settings
from src.telemetry import TelemetryLogger, TelemetryEvent

logger = setup_logger("server")
telemetry_logger = TelemetryLogger()

# ---------------------------------------------------------------------------
# Server-wide state
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "engine": None,
    "startup_time": None,
    "request_count": 0,
    "total_latency_seconds": 0.0,
    "active_requests": 0,
}

_engine_lock = asyncio.Lock()


def _resolve_startup_config() -> tuple[str, str, Optional[str]]:
    """Read startup configuration from pydantic-settings (loaded from .env)."""
    model_path = settings.olmo_model_path
    config_path = settings.olmo_config_path
    device = settings.olmo_device  # None → auto-detect inside engine
    return model_path, config_path, device


# ---------------------------------------------------------------------------
# Lifespan: replaces deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the inference engine once at startup; shut down cleanly at exit."""
    model_path, config_path, device = _resolve_startup_config()
    logger.info(f"Starting server — model_path={model_path}, config_path={config_path}, device={device or 'auto'}")

    _state["engine"] = OLMoInferenceEngine(
        config_path=config_path,
        model_path=model_path,
        device=device,
    )
    _state["startup_time"] = time.time()
    logger.info("Inference engine loaded successfully. Server ready.")

    yield  # Server is live here

    logger.info("Server shutting down — releasing model from GPU memory.")
    _state["engine"] = None


app = FastAPI(
    title="OLMo-2 Inference Server",
    description=(
        "Production-grade HTTP API for OLMo-2 1B inference. "
        "Hardware-portable: auto-detects precision (FP32 on Pascal, FP16/BF16 on modern GPUs)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt to generate from.")
    max_new_tokens: int = Field(128, ge=1, le=4096, description="Maximum tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling top-p.")
    disable_eos: bool = Field(False, description="Force generation until max_new_tokens (no EOS stop).")
    messages: Optional[List[Dict[str, str]]] = Field(
        None, description="Chat-style message list. If set, overrides `prompt`."
    )


class GenerateResponse(BaseModel):
    generated_text: str
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    total_latency_seconds: float
    tokens_per_sec: Optional[float] = None


class ValidateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to pre-flight check.")
    requested_max_new_tokens: int = Field(128, ge=1, le=4096)


class ValidateResponse(BaseModel):
    prompt_token_count: int
    requested_max_new_tokens: int
    context_limit: int
    headroom_tokens: int
    fits: bool
    recommended_max_new_tokens: int
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_engine() -> OLMoInferenceEngine:
    engine = _state["engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    return engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Server and model health check")
async def health() -> Dict[str, Any]:
    """Return uptime, model info, and hardware telemetry."""
    engine = _require_engine()
    telemetry = get_hardware_telemetry()

    avg_latency = (
        _state["total_latency_seconds"] / _state["request_count"]
        if _state["request_count"] > 0
        else None
    )

    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _state["startup_time"], 2),
        "model_path": engine.model_path,
        "device": engine.device,
        "torch_dtype": str(engine.torch_dtype),
        "attn_implementation": engine.attn_implementation,
        "request_count": _state["request_count"],
        "avg_latency_seconds": round(avg_latency, 4) if avg_latency else None,
        "telemetry": telemetry,
    }


@app.get("/metrics", summary="Runtime GPU and performance metrics")
async def metrics() -> Dict[str, Any]:
    """Return GPU memory usage, request throughput, and average latency."""
    engine = _require_engine()
    telemetry = get_hardware_telemetry()

    avg_latency = (
        _state["total_latency_seconds"] / _state["request_count"]
        if _state["request_count"] > 0
        else None
    )

    return {
        "model_path": engine.model_path,
        "device": engine.device,
        "torch_dtype": str(engine.torch_dtype),
        "request_count": _state["request_count"],
        "total_latency_seconds": round(_state["total_latency_seconds"], 4),
        "avg_latency_seconds": round(avg_latency, 4) if avg_latency else None,
        "hardware": {
            "cuda_available": telemetry["cuda_available"],
            "cuda_device_name": telemetry.get("cuda_device_name"),
            "cuda_allocated_mb": telemetry.get("cuda_allocated_mb"),
            "cuda_reserved_mb": telemetry.get("cuda_reserved_mb"),
        },
    }


@app.post("/validate", response_model=ValidateResponse, summary="Pre-flight context window check")
async def validate(request: ValidateRequest) -> ValidateResponse:
    """Check whether a prompt + token count fits within the model's context window.

    Clients can call this before /generate to get a recommendation on safe
    max_new_tokens values and avoid rejected requests.
    """
    engine = _require_engine()
    context_limit = getattr(engine.model.config, "max_position_embeddings", 4096)
    prompt_tokens = engine.tokenizer(request.prompt, return_tensors="pt").input_ids.shape[1]
    headroom = context_limit - prompt_tokens
    fits = (prompt_tokens + request.requested_max_new_tokens) <= context_limit
    recommended = min(request.requested_max_new_tokens, max(headroom, 0))

    warning = None
    if not fits:
        warning = (
            f"Prompt ({prompt_tokens} tokens) + requested tokens ({request.requested_max_new_tokens}) "
            f"exceeds context limit ({context_limit}). Reduce max_new_tokens to {headroom}."
        )
    elif recommended > context_limit * 0.75:
        warning = "Requesting > 75% of context window may cause quality degradation or repetition."

    return ValidateResponse(
        prompt_token_count=prompt_tokens,
        requested_max_new_tokens=request.requested_max_new_tokens,
        context_limit=context_limit,
        headroom_tokens=headroom,
        fits=fits,
        recommended_max_new_tokens=recommended,
        warning=warning,
    )


@app.post("/generate", response_model=GenerateResponse, summary="Text generation")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text from a prompt or chat-style messages list.

    Requests are serialized through an async lock to prevent concurrent
    GPU access on single-GPU deployments (Quadro P4000).
    """
    engine = _require_engine()
    context_limit = getattr(engine.model.config, "max_position_embeddings", 4096)

    # Pre-flight context check
    check_prompt = request.prompt
    if request.messages:
        check_prompt = " ".join(m.get("content", "") for m in request.messages)
    prompt_tokens = engine.tokenizer(check_prompt, return_tensors="pt").input_ids.shape[1]
    if prompt_tokens + request.max_new_tokens > context_limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Request exceeds model context limit. "
                f"prompt_tokens={prompt_tokens}, max_new_tokens={request.max_new_tokens}, "
                f"limit={context_limit}. Reduce max_new_tokens to ≤ {context_limit - prompt_tokens}."
            ),
        )

    request_received_at = time.time()
    trace_id = telemetry_logger.generate_trace_id()
    
    _state["active_requests"] += 1
    queue_depth = max(0, _state["active_requests"] - 1)
    
    tel_event = TelemetryEvent(
        trace_id=trace_id,
        timestamp="",
        model_version=engine.model_path,
        active_requests=_state["active_requests"],
        queue_depth=queue_depth,
        context_length=context_limit
    )

    try:
        async with _engine_lock:
            queueing_latency = time.time() - request_received_at
            result = engine.generate(
                prompt=request.prompt if not request.messages else None,
                messages=request.messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                disable_eos=request.disable_eos,
            )
            end = time.time()

        total_latency = end - request_received_at
        _state["request_count"] += 1
        _state["total_latency_seconds"] += total_latency

        perf = result.get("performance_metrics", {}) or {}
        perf["queueing_latency_seconds"] = round(queueing_latency, 4)
        perf["end_to_end_latency_seconds"] = round(total_latency, 4)

        # Populate Telemetry success path
        tel_event.prompt = result.get("prompt")
        tel_event.response = result.get("generated_text")
        tel_event.input_tokens = result.get("prompt_token_count")
        tel_event.output_tokens = result.get("response_token_count")
        tel_event.total_tokens = result.get("full_sequence_token_count")
        
        tel_event.queue_latency = queueing_latency
        tel_event.prefill_latency = perf.get("prefill_latency_seconds")
        tel_event.itl = perf.get("inter_token_latency_seconds")
        tel_event.e2e_latency = total_latency
        
        tel_event.tokens_per_second = result.get("tokens_per_sec")
        tel_event.gpu_utilization = perf.get("gpu_utilization_percent")
        tel_event.gpu_memory_allocated_mb = result.get("memory_allocated_mb")
        tel_event.kv_cache_utilization = perf.get("kv_cache_utilization")
        
        if result.get("response_token_count", 0) >= request.max_new_tokens:
            tel_event.finish_reason = "length"
        else:
            tel_event.finish_reason = "eos" if not request.disable_eos else "stop"
            
        telemetry_logger.log_event(tel_event)

    except asyncio.CancelledError:
        tel_event.status = "error"
        tel_event.error = "Request cancelled by client"
        tel_event.is_cancellation = True
        telemetry_logger.log_event(tel_event)
        raise
    except Exception as e:
        tel_event.status = "error"
        tel_event.error = str(e)
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            tel_event.is_oom = True
        telemetry_logger.log_event(tel_event)
        raise e
    finally:
        _state["active_requests"] -= 1

    # Determine finish reason
    if result.get("response_token_count", 0) >= request.max_new_tokens:
        finish_reason = "length"
    else:
        finish_reason = "eos" if not request.disable_eos else "stop"

    return GenerateResponse(
        generated_text=result["generated_text"],
        prompt_token_count=result.get("prompt_token_count"),
        response_token_count=result.get("response_token_count"),
        finish_reason=finish_reason,
        total_latency_seconds=total_latency,
        tokens_per_sec=result.get("tokens_per_sec"),
    )


@app.post("/reload", summary="Hot-reload model")
async def reload_model(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Reload the inference engine with a different model or config without restarting the server."""
    global _state
    engine = _require_engine()

    new_model_path = model_path or engine.model_path
    new_config_path = config_path or settings.olmo_config_path
    new_device = device or settings.olmo_device

    async with _engine_lock:
        logger.info(f"Reloading engine — model_path={new_model_path}, config_path={new_config_path}")
        _state["engine"] = OLMoInferenceEngine(
            config_path=new_config_path,
            model_path=new_model_path,
            device=new_device,
        )
        _state["request_count"] = 0
        _state["total_latency_seconds"] = 0.0

    return {
        "status": "reloaded",
        "model_path": _state["engine"].model_path,
        "device": _state["engine"].device,
        "config_path": new_config_path,
    }
