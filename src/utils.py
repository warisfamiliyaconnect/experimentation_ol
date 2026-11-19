import logging
import os
import sys
import torch
from typing import Dict, Any

def setup_logger(name: str = "olmo_experiments") -> logging.Logger:
    """
    Sets up a standardized stream logger with clean formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def get_hardware_telemetry() -> Dict[str, Any]:
    """
    Retrieves system and hardware telemetry details such as active device,
    available memory, and execution backends.
    """
    telemetry = {
        "device": "cpu",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_allocated_mb": 0.0,
        "cuda_reserved_mb": 0.0,
        "mps_available": False,
    }

    if torch.cuda.is_available():
        telemetry["device"] = "cuda"
        telemetry["cuda_available"] = True
        telemetry["cuda_device_count"] = torch.cuda.device_count()
        telemetry["cuda_device_name"] = torch.cuda.get_device_name(0)
        telemetry["cuda_allocated_mb"] = torch.cuda.memory_allocated(0) / (1024 ** 2)
        telemetry["cuda_reserved_mb"] = torch.cuda.memory_reserved(0) / (1024 ** 2)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        telemetry["device"] = "mps"
        telemetry["mps_available"] = True

    return telemetry
