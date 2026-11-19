import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

logger = logging.getLogger("telemetry")

@dataclass
class TelemetryEvent:
    # 1. REQUEST / RESPONSE
    trace_id: str
    timestamp: str
    model_version: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    status: str = "success"
    error: Optional[str] = None
    finish_reason: Optional[str] = None

    # 2. LATENCY (in seconds)
    queue_latency: Optional[float] = None
    prefill_latency: Optional[float] = None # TTFT
    itl: Optional[float] = None # Inter-token latency / TPOT
    e2e_latency: Optional[float] = None

    # 3. THROUGHPUT / TOKEN
    tokens_per_second: Optional[float] = None
    cached_tokens: Optional[int] = None
    cache_hit_ratio: Optional[float] = None
    context_length: Optional[int] = None

    # 4. SYSTEM
    gpu_utilization: Optional[float] = None
    gpu_memory_allocated_mb: Optional[float] = None
    kv_cache_utilization: Optional[float] = None
    active_requests: Optional[int] = None
    queue_depth: Optional[int] = None
    batch_size: Optional[int] = 1

    # 5. RELIABILITY
    is_timeout: bool = False
    is_retry: bool = False
    is_cancellation: bool = False
    is_oom: bool = False
    
    # 6. QUALITY / OUTCOME (Usually async/post-inference)
    user_feedback: Optional[str] = None # e.g., 'thumbs_up', 'thumbs_down'
    task_success: Optional[bool] = None
    tool_success: Optional[bool] = None
    safety_flags: Optional[List[str]] = None
    human_escalation: bool = False

class TelemetryLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "telemetry.jsonl")
        self._lock = threading.Lock()
        
    def generate_trace_id(self) -> str:
        return str(uuid.uuid4())
        
    def log_event(self, event: TelemetryEvent):
        if not event.timestamp:
            event.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            
        # Truncate prompt and response if they exceed 1000 characters
        if event.prompt and len(event.prompt) > 1000:
            event.prompt = event.prompt[:1000] + "... (truncated)"
        if event.response and len(event.response) > 1000:
            event.response = event.response[:1000] + "... (truncated)"
            
        event_dict = asdict(event)
        
        # Filter out None values to keep the JSONL clean, if desired.
        # But we will keep them for a strict schema.
        
        try:
            with self._lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event_dict) + "\n")
        except Exception as e:
            logger.error(f"Failed to write telemetry: {e}")
