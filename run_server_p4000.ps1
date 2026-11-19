# run_server_p4000.ps1
# ============================================================
# Stage 2 & 3: Launch the OLMo-2 API server on the Quadro P4000.
#
# The server:
#   - Auto-detects the Pascal GPU (compute capability 6.1)
#   - Forces float32 + eager attention (avoids 64x FP16 emulation slowdown)
#   - Listens on 0.0.0.0:8000, accessible from other LAN devices
#
# Usage:
#   .\run_server_p4000.ps1
#
# From another machine on your LAN:
#   python -m app.client --server http://<this-machine-ip>:8000
# ============================================================

# --- Configuration ---
$env:OLMO_MODEL_PATH = "artifacts/OLMo-2-0425-1B"   # FP32 base weights for P4000 Pascal
$env:OLMO_CONFIG_PATH = "configs/inference_fp32.yaml" # FP32 config (engine will auto-override precision anyway)
# OLMO_DEVICE not set → auto-detect (will pick 'cuda' and then fallback to float32 on Pascal 6.1)

$Host_Port = 8000
$Workers   = 1   # P4000 is single GPU — 1 worker only

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  OLMo-2 Inference Server — Quadro P4000 (Development)"     -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Model Path  : $env:OLMO_MODEL_PATH"
Write-Host "  Config Path : $env:OLMO_CONFIG_PATH"
Write-Host "  Port        : $Host_Port"
Write-Host "  Workers     : $Workers (single GPU, serialized requests)"
Write-Host ""
Write-Host "  Listening on: http://0.0.0.0:$Host_Port"
Write-Host "  LAN access  : http://<this-machine-IP>:$Host_Port"
Write-Host ""
Write-Host "  Test from another machine:"
Write-Host "    python -m app.client --server http://<IP>:$Host_Port"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if present
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

# Launch the server
.\.venv\Scripts\uvicorn `
    app.server:app `
    --host 0.0.0.0 `
    --port $Host_Port `
    --workers $Workers `
    --log-level info
