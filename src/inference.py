# importing the necessary libraries. 
"""Inference engine for OLMo-2 model serving.

This module provides a lightweight wrapper around the Hugging Face
transformers API to load the OLMo-2 model, run generation, and emit
structured telemetry for later analysis.
"""

import argparse
import json
import os
import subprocess
import time
import yaml
import torch
from omegaconf import OmegaConf
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import setup_logger, get_hardware_telemetry
from src.config import settings

logger = setup_logger("inference")


def _timestamped_path(path: str) -> str:
    """Return a filesystem path with a timestamp inserted before the extension.

    The helper is used when writing logs or output artifacts so each run gets
    a unique file name. If the base name contains the placeholder ``{timestamp}``,
    it is replaced with the current timestamp; otherwise a suffix of
    ``_YYYYMMDD_HHMMSS`` is appended before the extension.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(path)
    dirn = os.path.dirname(path) or "."
    if "{timestamp}" in base:
        base = base.replace("{timestamp}", ts)
        return os.path.join(dirn, base)
    name, ext = os.path.splitext(base)
    return os.path.join(dirn, f"{name}_{ts}{ext}")

class OLMoInferenceEngine:
    """Inference wrapper for OLMo-2 model loading and generation.

    This engine reads YAML configuration, loads the tokenizer and model,
    supports FP16/BF16/FP8 variants, and exposes a single `generate()` method
    for text generation.
    """
    def __init__(self, config_path: str = None, model_path: str = None, device: str = None, config_overrides: Optional[List[str]] = None):
        """Initialize the inference engine and load the tokenizer and model.

        The constructor reads any YAML configuration, selects the execution
        device, loads the tokenizer and model weights, and prepares the engine
        for later generation calls.
        """
        self.config = {}

        # Load configuration using OmegaConf if the file exists.
        if config_path and os.path.exists(config_path):
            logger.info(f"Loading configurations using OmegaConf from {config_path}")
            try:
                loaded_cfg = OmegaConf.load(config_path)
                if config_overrides:
                    logger.info(f"Applying dynamic overrides: {config_overrides}")
                    cli_cfg = OmegaConf.from_dotlist(config_overrides)
                    loaded_cfg = OmegaConf.merge(loaded_cfg, cli_cfg)
                OmegaConf.resolve(loaded_cfg)
                self.config = OmegaConf.to_container(loaded_cfg, resolve=True)
            except Exception as e:
                logger.error(f"Failed to load configuration via OmegaConf: {e}")
                raise e

        # Override config settings with settings object or explicit parameters when provided.
        self.model_path = model_path or self.config.get("model", {}).get("path") or settings.olmo_model_path

        # Detect backend: prefer explicit config, otherwise infer from extension.
        self.backend = self.config.get("model", {}).get("backend", "pytorch")
        if self.model_path.endswith(".gguf"):
            self.backend = "llamacpp"

        logger.info(f"Using inference backend: {self.backend}")

        if self.backend == "llamacpp":
            self._init_llamacpp()
        else:
            self._init_pytorch(device)

    def _init_llamacpp(self):
        """Initialize the llama.cpp backend using llama-cpp-python."""
        from llama_cpp import Llama

        logger.info(f"Initializing llama.cpp backend with model: {self.model_path}")

        n_gpu_layers = self.config.get("model", {}).get("n_gpu_layers", -1)
        n_ctx = self.config.get("model", {}).get("n_ctx", 4096)

        start_time = time.time()
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False
        )
        self.model_loading_latency_seconds = time.time() - start_time

        # llama.cpp manages its own tokenizer; we wrap it for API compatibility.
        self.tokenizer = self.model.tokenizer
        self.device = "cuda" if n_gpu_layers > 0 else "cpu"
        self.torch_dtype = "gguf-quantized"
        self.attn_implementation = "llama-cpp"

        logger.info(f"llama.cpp model load completed in {self.model_loading_latency_seconds:.2f} seconds.")

    def _init_pytorch(self, device: str = None):
        """Initialize the PyTorch backend."""
        telemetry = get_hardware_telemetry()
        self.device = device or self.config.get("model", {}).get("device_map", None)
        if not self.device or self.device == "auto":
            self.device = telemetry["device"]

        logger.info(f"Using serving hardware backend: {self.device}")

        # Perform hardware capability checks for CUDA devices to determine dtypes and attention mechanisms dynamically
        if self.device == "cuda" and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            logger.info(f"Detected CUDA device compute capability: {major}.{minor}")

            dtype_str = self.config.get("model", {}).get("torch_dtype", "float16")
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "float8": torch.float8_e4m3fn,
                "fp8": torch.float8_e4m3fn,
            }
            self.torch_dtype = dtype_map.get(dtype_str, torch.float16)

            # Fallback to float16 if bfloat16 is requested but not supported by the hardware
            if self.torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning("bfloat16 requested but not supported by the local GPU. Falling back to float16.")
                self.torch_dtype = torch.float16

            self.attn_implementation = self.config.get("model", {}).get("attn_implementation", "sdpa")
            
            # Disable SDPA for older Pascal architectures (major < 7) as it is not supported or performs poorly
            if major < 7:
                logger.warning(
                    f"GPU compute capability {major}.{minor} is older than Volta (7.0). "
                    "Forcing 'eager' attention implementation to prevent compatibility issues."
                )
                self.attn_implementation = "eager"
        else:
            # Non-CUDA devices (CPU, MPS)
            dtype_str = self.config.get("model", {}).get("torch_dtype", "float32" if self.device == "cpu" else "float16")
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.torch_dtype = dtype_map.get(dtype_str, torch.float32 if self.device == "cpu" else torch.float16)
            self.attn_implementation = self.config.get("model", {}).get("attn_implementation", "sdpa")

        # Determine load-time quantization options if requested in config.
        self.load_in_8bit = self.config.get("model", {}).get("load_in_8bit", False)
        quantization_config = None
        if self.load_in_8bit and self.device == "cuda":
            logger.info("Configuring model loading in 8-bit mode (LLM.int8())")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer from model path and ensure there is a valid pad token.
        logger.info(f"Loading tokenizer from '{self.model_path}'")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model weights, using float16 for FP8 initial loading if needed.
        logger.info(f"Loading model weights from '{self.model_path}' (dtype={self.torch_dtype}, quant_8bit={self.load_in_8bit})")
        start_time = time.time()
        load_dtype = self.torch_dtype if self.torch_dtype != torch.float8_e4m3fn else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=load_dtype,
            device_map=self.device if self.device != "cpu" else None,
            low_cpu_mem_usage=True,
            attn_implementation=self.attn_implementation,
            quantization_config=quantization_config
        )
        if self.torch_dtype == torch.float8_e4m3fn:
            self.model = self.model.to(self.torch_dtype)
            self.model.config.torch_dtype = self.torch_dtype
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model_loading_latency_seconds = time.time() - start_time
        logger.info(f"Model load completed in {self.model_loading_latency_seconds:.2f} seconds.")


    def build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format a chat message list into a single prompt string.

        The output is serialized with explicit role markers so the model can
        distinguish system, user, and assistant statements.
        """
        prompt_blocks = []
        for message in messages:
            role = message.get("role", "user").strip().lower()
            content = message.get("content", "").strip()
            if role == "system":
                prompt_blocks.append(f"[SYSTEM]\n{content}")
            elif role == "assistant":
                prompt_blocks.append(f"[ASSISTANT]\n{content}")
            else:
                prompt_blocks.append(f"[USER]\n{content}")

        prompt_blocks.append("[ASSISTANT]\n")
        return "\n\n".join(prompt_blocks)

    def _get_model_dtype(self) -> torch.dtype:
        """Return the model parameter datatype for memory estimation."""
        if hasattr(self.model, "dtype"):
            return self.model.dtype
        for param in self.model.parameters():
            return param.dtype
        return torch.float16

    def _get_gpu_utilization_percent(self) -> Optional[float]:
        """Try to read GPU utilization from nvidia-smi when the tool is available."""
        if self.device != "cuda":
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            value = result.stdout.strip().splitlines()[0].strip()
            return float(value)
        except (FileNotFoundError, subprocess.SubprocessError, ValueError, TimeoutError):
            return None

    def _get_memory_bandwidth_utilization_percent(self) -> Optional[float]:
        """Estimate memory pressure as a proxy for memory-bandwidth utilization."""
        if self.device != "cuda":
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            used_str, total_str = result.stdout.strip().splitlines()[0].split(",")
            used = float(used_str.strip())
            total = float(total_str.strip())
            if total <= 0:
                return None
            return round((used / total) * 100.0, 2)
        except (FileNotFoundError, subprocess.SubprocessError, ValueError, TimeoutError):
            return None

    def _build_performance_metrics(
        self,
        tokenization_seconds: float,
        generation_seconds: float,
        response_token_count: int,
        prompt_token_count: int,
        max_position_embeddings: Optional[int],
        queueing_latency_seconds: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """Build a structured performance summary for latency and utilization metrics."""
        prefill_latency_seconds = tokenization_seconds
        decode_latency_seconds = generation_seconds
        inter_token_latency_seconds = decode_latency_seconds / response_token_count if response_token_count > 0 else None
        throughput_tokens_per_second = response_token_count / generation_seconds if generation_seconds > 0 else None
        kv_cache_utilization = None
        if max_position_embeddings and max_position_embeddings > 0:
            kv_cache_utilization = round((prompt_token_count + response_token_count) / max_position_embeddings, 4)

        return {
            "request_latency_seconds": tokenization_seconds + generation_seconds,
            "queueing_latency_seconds": queueing_latency_seconds,
            "tokenization_latency_seconds": tokenization_seconds,
            "prefill_latency_seconds": prefill_latency_seconds,
            "ttft_seconds": prefill_latency_seconds,
            "decode_latency_seconds": decode_latency_seconds,
            "inter_token_latency_seconds": inter_token_latency_seconds,
            "throughput_tokens_per_second": throughput_tokens_per_second,
            "gpu_utilization_percent": self._get_gpu_utilization_percent(),
            "memory_bandwidth_utilization_percent": self._get_memory_bandwidth_utilization_percent(),
            "kv_cache_utilization": kv_cache_utilization,
            "attention_delay_seconds": None,
            "network_latency_seconds": None,
            "model_loading_latency_seconds": self.model_loading_latency_seconds,
            "scheduler_latency_seconds": queueing_latency_seconds,
        }

    def estimate_kv_cache_size_mb(self, num_tokens: int) -> float:
        """Estimate the KV cache memory usage for a token count in MB."""
        if self.backend == "llamacpp":
            # Rough estimate for Q4_K_M / FP16 KV cache in llama.cpp
            # Usually 2 bytes per element for KV cache (FP16)
            # hidden_size * num_layers * 2 * num_tokens
            # For OLMo-2 1B: hidden_size=2048, num_layers=16
            hidden_size = 2048
            num_layers = 16
            bytes_per_element = 2
            total_bytes = 2 * hidden_size * num_layers * num_tokens * bytes_per_element
            return total_bytes / (1024 ** 2)

        cfg = self.model.config
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
        if hidden_size is None or num_layers is None:
            return 0.0

        bytes_per_element = torch.tensor([], dtype=self._get_model_dtype()).element_size()
        total_bytes = 2 * hidden_size * num_layers * num_tokens * bytes_per_element
        return total_bytes / (1024 ** 2)

    def estimate_runtime_memory_breakdown_mb(self, prompt_token_count: int, response_token_count: int) -> Dict[str, float]:
        """Estimate memory usage for weights, KV cache, activations, and scratch buffers in MB."""
        model_size_mb = 0.0
        try:
            if self.backend == "llamacpp":
                # For GGUF, model size is simply the file size
                model_size_mb = os.path.getsize(self.model_path) / (1024 ** 2)
            else:
                param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
                model_size_mb = param_bytes / (1024 ** 2)
        except Exception:
            model_size_mb = 0.0

        kv_cache_mb = self.estimate_kv_cache_size_mb(prompt_token_count + response_token_count)
        activations_mb = max(kv_cache_mb * 0.5, 16.0)
        temporary_buffers_mb = max(kv_cache_mb * 0.25, 8.0)
        total_estimated_mb = model_size_mb + kv_cache_mb + activations_mb + temporary_buffers_mb

        return {
            "model_weights_mb": round(model_size_mb, 3),
            "kv_cache_mb": round(kv_cache_mb, 3),
            "activations_mb": round(activations_mb, 3),
            "temporary_buffers_mb": round(temporary_buffers_mb, 3),
            "total_estimated_runtime_mb": round(total_estimated_mb, 3),
        }

    def _build_log_entry(
        self,
        prompt_text: str,
        generated_text: str,
        metrics: Dict[str, Any],
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Build a structured JSON log entry for each generation call."""
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "torch_dtype": str(self.torch_dtype),
            "query": prompt_text,
            "messages": messages,
            "response": generated_text,
            "metrics": metrics
        }

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        disable_eos: bool = False
    ) -> Dict[str, Any]:
        """Run text generation and return both output and telemetry."""
        gen_config = self.config.get("generation", {})

        # Resolve generation parameters, using configuration defaults if not provided.
        m_tokens = max_new_tokens if max_new_tokens is not None else gen_config.get("max_new_tokens", 128)
        temp = temperature if temperature is not None else gen_config.get("temperature", 0.7)
        tp = top_p if top_p is not None else gen_config.get("top_p", 0.9)
        top_k = gen_config.get("top_k", None)
        repetition_penalty = gen_config.get("repetition_penalty", 1.0)
        do_sample = gen_config.get("do_sample", True)
        pad_token_id = gen_config.get("pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else None

        # Build the prompt text from either raw prompt or serialized chat messages.
        if messages:
            prompt_text = self.build_prompt_from_messages(messages)
        elif prompt is not None:
            prompt_text = prompt
        else:
            raise ValueError("Either prompt or messages must be provided for generation.")

        if self.backend == "llamacpp":
            # LLAMA.CPP IMPLEMENTATION
            tokenization_start = time.time()
            input_tokens = self.model.tokenize(prompt_text.encode("utf-8"))
            tokenization_end = time.time()
            input_len = len(input_tokens)

            generation_start = time.time()
            # llama-cpp-python generate uses a different API
            output = self.model(
                prompt=prompt_text,
                max_tokens=m_tokens,
                temperature=temp,
                top_p=tp,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stop=["<|endoftext|>"] if not disable_eos else None
            )
            generation_end = time.time()

            gen_text = output["choices"][0]["text"]
            full_text = prompt_text + gen_text

            # Estimate total tokens using tokenizer
            full_tokens = self.model.tokenize(full_text.encode("utf-8"))
            full_sequence_len = len(full_tokens)
            response_token_count = full_sequence_len - input_len

            generation_seconds = generation_end - generation_start
            tokenization_seconds = tokenization_end - tokenization_start
            total_latency_seconds = generation_end - tokenization_start
            tokens_per_sec = response_token_count / generation_seconds if generation_seconds > 0 else 0.0

            # Memory estimation for llama.cpp
            memory_allocated_mb = 0.0
            memory_reserved_mb = 0.0
            peak_memory_mb = 0.0
            # llama-cpp doesn't expose easy torch-like memory stats via python API
            # we can try to use nvidia-smi as a proxy
            gpu_util = self._get_gpu_utilization_percent()

            kv_cache_mb = self.estimate_kv_cache_size_mb(full_sequence_len)
            memory_breakdown = self.estimate_runtime_memory_breakdown_mb(
                prompt_token_count=input_len,
                response_token_count=response_token_count,
            )

            # Llama.cpp provides some internal metrics in some versions, but here we use estimated
            max_position_embeddings = self.config.get("model", {}).get("n_ctx", 4096)
            performance_metrics = self._build_performance_metrics(
                tokenization_seconds=tokenization_seconds,
                generation_seconds=generation_seconds,
                response_token_count=response_token_count,
                prompt_token_count=input_len,
                max_position_embeddings=max_position_embeddings,
            )

            result = {
                "prompt": prompt_text,
                "prompt_token_count": input_len,
                "response_token_count": response_token_count,
                "full_sequence_token_count": full_sequence_len,
                "generated_text": gen_text.strip(),
                "full_text": full_text.strip(),
                "tokenization_seconds": tokenization_seconds,
                "generation_seconds": generation_seconds,
                "total_latency_seconds": total_latency_seconds,
                "new_tokens_count": response_token_count,
                "tokens_per_sec": tokens_per_sec,
                "kv_cache_mb": kv_cache_mb,
                "memory_breakdown": memory_breakdown,
                "memory_allocated_mb": memory_allocated_mb,
                "memory_reserved_mb": memory_reserved_mb,
                "peak_memory_mb": peak_memory_mb,
                "performance_metrics": performance_metrics,
            }
        else:
            # PYTORCH IMPLEMENTATION
            tokenization_start = time.time()
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            tokenization_end = time.time()
            input_len = inputs.input_ids.shape[1]

            # Run generation while capturing GPU memory usage when possible.
            generation_start = time.time()
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            eos_token_id = None if disable_eos else self.tokenizer.eos_token_id
            if disable_eos:
                logger.warning("EOS stopping disabled for this generation run.")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=m_tokens,
                    temperature=temp,
                    top_p=tp,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=self.config.get("model", {}).get("use_cache", True)
                )

            if self.device == "cuda":
                torch.cuda.synchronize()

            generation_end = time.time()

            output_tokens = outputs[0]
            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

            full_sequence_len = len(output_tokens)
            response_token_count = full_sequence_len - input_len
            generation_seconds = generation_end - generation_start
            tokenization_seconds = tokenization_end - tokenization_start
            total_latency_seconds = generation_end - tokenization_start
            tokens_per_sec = response_token_count / generation_seconds if generation_seconds > 0 else 0.0

            # Strip the prompt prefix from the decoded output for a clean response.
            gen_text = output_text[len(prompt_text):] if output_text.startswith(prompt_text) else output_text

            memory_allocated_mb = 0.0
            memory_reserved_mb = 0.0
            peak_memory_mb = 0.0
            if self.device == "cuda":
                memory_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                memory_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            kv_cache_mb = self.estimate_kv_cache_size_mb(full_sequence_len)
            memory_breakdown = self.estimate_runtime_memory_breakdown_mb(
                prompt_token_count=input_len,
                response_token_count=response_token_count,
            )
            max_position_embeddings = getattr(self.model.config, "max_position_embeddings", None)
            performance_metrics = self._build_performance_metrics(
                tokenization_seconds=tokenization_seconds,
                generation_seconds=generation_seconds,
                response_token_count=response_token_count,
                prompt_token_count=input_len,
                max_position_embeddings=max_position_embeddings,
            )

            result = {
                "prompt": prompt_text,
                "prompt_token_count": input_len,
                "response_token_count": response_token_count,
                "full_sequence_token_count": full_sequence_len,
                "generated_text": gen_text.strip(),
                "full_text": output_text.strip(),
                "tokenization_seconds": tokenization_seconds,
                "generation_seconds": generation_seconds,
                "total_latency_seconds": total_latency_seconds,
                "new_tokens_count": response_token_count,
                "tokens_per_sec": tokens_per_sec,
                "kv_cache_mb": kv_cache_mb,
                "memory_breakdown": memory_breakdown,
                "memory_allocated_mb": memory_allocated_mb,
                "memory_reserved_mb": memory_reserved_mb,
                "peak_memory_mb": peak_memory_mb,
                "performance_metrics": performance_metrics,
            }

        log_entry = self._build_log_entry(
            prompt_text=prompt_text,
            generated_text=result["generated_text"],
            metrics={
                "prompt_token_count": result["prompt_token_count"],
                "response_token_count": result["response_token_count"],
                "full_sequence_token_count": result["full_sequence_token_count"],
                "tokenization_seconds": result["tokenization_seconds"],
                "generation_seconds": result["generation_seconds"],
                "total_latency_seconds": result["total_latency_seconds"],
                "tokens_per_sec": result["tokens_per_sec"],
                "kv_cache_mb": result["kv_cache_mb"],
                "memory_breakdown": result["memory_breakdown"],
                "memory_allocated_mb": result["memory_allocated_mb"],
                "memory_reserved_mb": result["memory_reserved_mb"],
                "peak_memory_mb": result["peak_memory_mb"]
            },
            messages=messages
        )
        logger.info("Inference log: %s", json.dumps(log_entry, ensure_ascii=False))

        return result

class OLMoClient:
    """Unified client/router that wraps both local inference and remote server API.

    This abstraction allows downstream code to remain unchanged when switching
    from local GPU development (e.g. on low-end hardware) to querying a high-throughput
    serving instance (like vLLM) in a cluster/production environment.
    """
    def __init__(self, mode: str = "auto", server_url: str = None, config_path: str = None, model_path: str = None, device: str = None, config_overrides: Optional[List[str]] = None):
        self.mode = mode.lower()
        self.server_url = server_url or settings.olmo_server_url
        
        if self.server_url:
            self.mode = "api"
            logger.info(f"Client configured in API-client mode pointing to: {self.server_url}")
        elif self.mode == "auto":
            self.mode = "local"
            
        if self.mode == "local":
            logger.info("Client configured in local inference mode. Loading model...")
            self.engine = OLMoInferenceEngine(config_path=config_path, model_path=model_path, device=device, config_overrides=config_overrides)
        else:
            self.engine = None

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        disable_eos: bool = False
    ) -> Dict[str, Any]:
        """Run text generation and return both output and telemetry, routing to local or API server."""
        if self.mode == "api":
            import urllib.request
            import urllib.error
            
            # Use /v1/completions endpoint for standard prompts or /v1/chat/completions for messages
            url = self.server_url
            
            if messages:
                if not url.endswith("/v1/chat/completions"):
                    url = url.rstrip("/")
                    if url.endswith("/v1"):
                        url += "/chat/completions"
                    else:
                        url += "/v1/chat/completions"
                payload = {
                    "messages": messages,
                    "max_tokens": max_new_tokens or 128,
                    "temperature": temperature if temperature is not None else 0.7,
                }
                if top_p is not None:
                    payload["top_p"] = top_p
            else:
                if not url.endswith("/v1/completions") and not url.endswith("/v1/chat/completions"):
                    url = url.rstrip("/")
                    if url.endswith("/v1"):
                        url += "/completions"
                    else:
                        url += "/v1/completions"
                payload = {
                    "prompt": prompt,
                    "max_tokens": max_new_tokens or 128,
                    "temperature": temperature if temperature is not None else 0.7,
                    "stop": None if disable_eos else ["<|endoftext|>"]
                }
                if top_p is not None:
                    payload["top_p"] = top_p
            
            start_time = time.time()
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    res = json.loads(response.read().decode("utf-8"))
                latency = time.time() - start_time
                
                if messages:
                    generated_text = res["choices"][0]["message"]["content"]
                    resolved_prompt = json.dumps(messages, ensure_ascii=False)
                else:
                    generated_text = res["choices"][0]["text"]
                    resolved_prompt = prompt
                
                # Mock a return payload structure compatible with local engine's return structure
                return {
                    "prompt": resolved_prompt,
                    "generated_text": generated_text,
                    "total_latency_seconds": latency,
                    "tokens_per_sec": len(generated_text.split()) / latency if latency > 0 else 0.0,
                    "prompt_token_count": res.get("usage", {}).get("prompt_tokens"),
                    "response_token_count": res.get("usage", {}).get("completion_tokens"),
                    "full_sequence_token_count": res.get("usage", {}).get("total_tokens"),
                }
            except Exception as e:
                logger.error(f"API generation request failed: {e}")
                raise e
        else:
            return self.engine.generate(
                prompt=prompt,
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                disable_eos=disable_eos
            )


if __name__ == "__main__":
    # CLI entrypoint for one-shot inference runs.
    # This allows the engine to be executed directly from the command line
    # and also provides an easy way to validate the model loading path.
    parser = argparse.ArgumentParser(description="OLMo Optimized Inference serving CLI")
    parser.add_argument("--config", type=str, default="configs/inference_fp16.yaml", help="Path to YAML configuration")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit path to model directory (overrides config)")
    parser.add_argument("--prompt", type=str, default="Developing an agentic coding assistant requires", help="Input text prompt")
    parser.add_argument("--system", type=str, default=None, help="System message content for chat-style prompts")
    parser.add_argument("--user", type=str, default=None, help="User message content for chat-style prompts")
    parser.add_argument("--message", type=str, action="append", default=None, help="Additional chat message in role:content format, e.g. user:What is AI?")
    parser.add_argument("--messages-json", type=str, default=None, help="Path to a JSON file containing a list of messages")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum number of new tokens to generate")
    parser.add_argument("--disable-eos", action="store_true", help="Disable EOS stopping so generation runs until max_new_tokens")
    parser.add_argument("--device", type=str, default=None, help="Force execution device (cuda, cpu, mps)")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path to write inference logs")
    parser.add_argument("--log-format", type=str, choices=["jsonl", "json"], default="jsonl", help="Log file format")
    parser.add_argument("--server-url", type=str, default=None, help="Optional completions API endpoint to run in API mode instead of loading local weights")

    args, overrides = parser.parse_known_args()

    try:
        client = OLMoClient(
            server_url=args.server_url,
            config_path=args.config,
            model_path=args.model_path,
            device=args.device,
            config_overrides=overrides
        )

        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        if args.user:
            messages.append({"role": "user", "content": args.user})
        if args.message:
            for item in args.message:
                if ":" not in item:
                    raise ValueError("Messages must be passed as role:content")
                role, content = item.split(":", 1)
                messages.append({"role": role.strip(), "content": content.strip()})
        if args.messages_json:
            with open(args.messages_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("messages-json file must contain a JSON list of message objects")
            messages.extend(loaded)

        if messages:
            result = client.generate(messages=messages, max_new_tokens=args.max_new_tokens, disable_eos=args.disable_eos)
        else:
            result = client.generate(prompt=args.prompt, max_new_tokens=args.max_new_tokens, disable_eos=args.disable_eos)

        # Build the log entry (used for all outputs)
        from src.telemetry import TelemetryLogger, TelemetryEvent
        telemetry_logger = TelemetryLogger()
        tel_event = TelemetryEvent(
            trace_id=telemetry_logger.generate_trace_id(),
            timestamp="",
            model_version=client.engine.model_path if client.engine else (args.model_path or "api-server"),
            prompt=result.get("prompt"),
            response=result.get("generated_text"),
            input_tokens=result.get("prompt_token_count"),
            output_tokens=result.get("response_token_count"),
            total_tokens=result.get("full_sequence_token_count"),
            queue_latency=result.get("performance_metrics", {}).get("queueing_latency_seconds"),
            prefill_latency=result.get("performance_metrics", {}).get("prefill_latency_seconds"),
            itl=result.get("performance_metrics", {}).get("inter_token_latency_seconds"),
            e2e_latency=result.get("total_latency_seconds"),
            tokens_per_second=result.get("tokens_per_sec"),
            gpu_utilization=result.get("performance_metrics", {}).get("gpu_utilization_percent"),
            gpu_memory_allocated_mb=result.get("memory_allocated_mb"),
            kv_cache_utilization=result.get("performance_metrics", {}).get("kv_cache_utilization"),
            finish_reason="length" if result.get("response_token_count", 0) >= (args.max_new_tokens or 128) else "eos"
        )
        telemetry_logger.log_event(tel_event)
        
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": args.config,
            "model_path": client.engine.model_path if client.engine else (args.model_path or "api-server"),
            "device": client.engine.device if client.engine else "api",
            "query": result["prompt"],
            "prompt_token_count": result.get("prompt_token_count"),
            "response_token_count": result.get("response_token_count"),
            "full_sequence_token_count": result.get("full_sequence_token_count"),
            "response": result["generated_text"],
            "metrics": {
                "tokenization_seconds": result.get("tokenization_seconds"),
                "generation_latency_seconds": result.get("generation_seconds"),
                "total_latency_seconds": result.get("total_latency_seconds"),
                "throughput_tokens_per_sec": result.get("tokens_per_sec"),
                "kv_cache_mb": result.get("kv_cache_mb"),
                "memory_allocated_mb": result.get("memory_allocated_mb"),
                "memory_reserved_mb": result.get("memory_reserved_mb"),
                "peak_memory_mb": result.get("peak_memory_mb")
            }
        }

        # Optional: write to user-specified log file (timestamped)
        if args.log_file:
            log_path = _timestamped_path(args.log_file)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Inference log appended to {log_path}")

        # Always update the central inference JSON file as a proper JSON array.
        # Path is driven by OLMO_LOG_FILE in .env via pydantic-settings.
        central_path = settings.olmo_log_file
        os.makedirs(os.path.dirname(central_path), exist_ok=True)
        try:
            # Read existing content, accept empty, array, single object, or newline-delimited JSON.
            # This preserves prior logs while keeping the central file in an easy-to-consume array.
            if not os.path.exists(central_path) or os.path.getsize(central_path) == 0:
                data = []
            else:
                with open(central_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    data = []
                else:
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            data = parsed
                        else:
                            data = [parsed]
                    except json.JSONDecodeError:
                        # Fallback: try to parse as JSONL (one JSON object per line)
                        data = []
                        for line in content.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data.append(json.loads(line))
                            except Exception:
                                # skip malformed lines
                                continue

            data.append(entry)
            with open(central_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Central inference JSON updated at {central_path} (array)")
        except Exception as e:
            logger.error(f"Failed to update central inference JSON at {central_path}: {e}")

        print("\n" + "="*50)
        if messages:
            print("Messages:")
            for msg in messages:
                print(f"  {msg['role'].upper()}: {msg['content']}")
            print("="*50)
        else:
            print(f"Prompt: {result['prompt']}")
            print("="*50)

        print(f"Generated Output:\n{result['generated_text']}")
        print("="*50)
        print(f"Generation Latency: {result.get('total_latency_seconds', 0.0):.2f} seconds")
        print(f"Throughput Speed:   {result['tokens_per_sec']:.2f} tokens/second")
        memory_breakdown = result.get("memory_breakdown", {})
        if memory_breakdown:
            print("Memory Breakdown:")
            for key, value in memory_breakdown.items():
                print(f"  {key}: {value:.3f} MB")
        print("="*50 + "\n")
    except Exception as e:
        logger.error(f"Inference error occurred: {e}")
