import argparse
import os
import subprocess
import urllib.request
import zipfile
from src.utils import setup_logger

logger = setup_logger("convert_gguf")

LLAMA_CPP_URL = "https://github.com/ggerganov/llama.cpp/releases/download/b3438/llama-b3438-bin-win-cuda-cu12.2-x64.zip"

def download_and_extract_llama_cpp(tools_dir="tools/llama_bin"):
    os.makedirs(tools_dir, exist_ok=True)
    quantize_exe = os.path.join(tools_dir, "llama-quantize.exe")
    if os.path.exists(quantize_exe):
        return quantize_exe
        
    zip_path = os.path.join(tools_dir, "llama_bin.zip")
    logger.info(f"Downloading pre-built llama.cpp binaries from {LLAMA_CPP_URL}")
    urllib.request.urlretrieve(LLAMA_CPP_URL, zip_path)
    
    logger.info(f"Extracting to {tools_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tools_dir)
        
    if not os.path.exists(quantize_exe):
        raise FileNotFoundError(f"Failed to find {quantize_exe} after extraction.")
    return quantize_exe

def main():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF and quantize")
    parser.add_argument("--input-dir", required=True, help="Path to HF model directory")
    parser.add_argument("--output-gguf", required=True, help="Path for the final quantized GGUF")
    parser.add_argument("--quant-type", default="Q4_K_M", help="Quantization type (e.g. Q4_K_M, Q8_0)")
    args = parser.parse_args()

    input_dir = args.input_dir
    final_gguf = args.output_gguf
    quant_type = args.quant_type

    # 1. Convert to F16 GGUF
    f16_gguf = final_gguf.replace(".gguf", "-F16.gguf")
    convert_script = os.path.join("tools", "llama.cpp", "convert_hf_to_gguf.py")
    
    if not os.path.exists(convert_script):
        logger.error(f"Cannot find {convert_script}. Please clone llama.cpp into tools/llama.cpp first.")
        return

    import sys
    logger.info(f"Converting HF model {input_dir} to F16 GGUF {f16_gguf}...")
    convert_cmd = [sys.executable, convert_script, input_dir, "--outfile", f16_gguf, "--outtype", "f16"]
    subprocess.run(convert_cmd, check=True)

    # 2. Download quantize tool if needed
    quantize_exe = download_and_extract_llama_cpp()

    # 3. Quantize to target
    logger.info(f"Quantizing {f16_gguf} to {quant_type} -> {final_gguf}...")
    quantize_cmd = [quantize_exe, f16_gguf, final_gguf, quant_type]
    subprocess.run(quantize_cmd, check=True)
    
    logger.info(f"Quantization complete: {final_gguf}")

if __name__ == "__main__":
    main()
