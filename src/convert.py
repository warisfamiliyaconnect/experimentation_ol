import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import setup_logger, get_hardware_telemetry

logger = setup_logger("convert")


def convert_precision(input_dir: str, output_dir: str, target_dtype: str = "float16") -> None:
    """
    Loads model weights from input_dir, downcasts or changes dtype, and exports them to output_dir.
    """
    logger.info(f"Checking hardware telemetry before precision conversion...")
    telemetry = get_hardware_telemetry()
    logger.info(f"Telemetry: {telemetry}")
    
    # Map string dtype to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8": torch.float8_e4m3fn,
        "fp8": torch.float8_e4m3fn,
    }
    
    if target_dtype not in dtype_map:
        raise ValueError(f"Unsupported target dtype: {target_dtype}. Choose from {list(dtype_map.keys())}")
        
    torch_dtype = dtype_map[target_dtype]
    
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        raise FileNotFoundError(f"Input weight path '{input_dir}' is empty or does not exist.")
        
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading tokenizer from '{input_dir}'")
    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Loading model from '{input_dir}' (this may take several minutes)...")
    # Load model on CPU to avoid running out of VRAM, then convert precision
    model = AutoModelForCausalLM.from_pretrained(
        input_dir,
        torch_dtype=torch.float32,  # load in float32 first to ensure baseline conversion integrity
        low_cpu_mem_usage=True
    )
    
    logger.info(f"Converting model weights to {target_dtype} precision...")
    if target_dtype in {"float8", "fp8"}:
        model = model.to(torch_dtype)
        model.config.torch_dtype = torch_dtype
    else:
        model = model.to(torch_dtype)
    
    logger.info(f"Exporting converted weights and config files to '{output_dir}'")
    model.save_pretrained(output_dir, safe_serialization=True)
    logger.info(f"Precision conversion successfully complete. Checkpoints saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLMo Model Weight Precision Converter")
    parser.add_argument("--input-dir", type=str, default="artifacts/OLMo-2-0425-1B", help="Path to raw weights directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/OLMo-2-0425-1B-FP16", help="Path to save converted weights")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32", "float8", "fp8"], help="Target model precision")
    
    args = parser.parse_args()
    convert_precision(input_dir=args.input_dir, output_dir=args.output_dir, target_dtype=args.dtype)
