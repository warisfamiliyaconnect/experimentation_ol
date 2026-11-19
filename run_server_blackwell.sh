#!/usr/bin/env bash
# run_server_blackwell.sh
# ============================================================
# Stage 3 (Production): Launch 4 independent OLMo-2 API server
# workers on the 4x RTX PRO 6000 Blackwell system.
#
# Architecture:
#   Worker 0  GPU 0  → Port 8000
#   Worker 1  GPU 1  → Port 8001
#   Worker 2  GPU 2  → Port 8002
#   Worker 3  GPU 3  → Port 8003
#
#   Nginx (port 80) → round-robin load balancer across 8000–8003
#
# Usage:
#   chmod +x run_server_blackwell.sh
#   ./run_server_blackwell.sh
#
#   Stop all workers:
#   ./run_server_blackwell.sh stop
# ============================================================

set -euo pipefail

# --- Configuration ---
MODEL_PATH="${OLMO_MODEL_PATH:-artifacts/OLMo-2-0425-1B-FP16}"
CONFIG_PATH="${OLMO_CONFIG_PATH:-configs/inference_fp16.yaml}"
LOG_DIR="logs/server"
PID_DIR="logs/pids"
BASE_PORT=8000
NUM_GPUS=4

mkdir -p "$LOG_DIR" "$PID_DIR"

# -------------------------------------------------------------------
# STOP: Kill all running worker processes
# -------------------------------------------------------------------
stop_workers() {
    echo ""
    echo "Stopping all OLMo-2 server workers..."
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        PID_FILE="$PID_DIR/worker_gpu${i}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "  Stopping Worker $i (PID $PID) on GPU $i..."
                kill "$PID"
            fi
            rm -f "$PID_FILE"
        fi
    done
    echo "All workers stopped."
    echo ""
}

if [ "${1:-}" = "stop" ]; then
    stop_workers
    exit 0
fi

# -------------------------------------------------------------------
# START: Launch one uvicorn process per GPU
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  OLMo-2 Inference Server — 4x RTX PRO 6000 Blackwell"
echo "============================================================"
echo "  Model Path  : $MODEL_PATH"
echo "  Config Path : $CONFIG_PATH"
echo "  GPUs        : $NUM_GPUS"
echo "  Ports       : $BASE_PORT – $((BASE_PORT + NUM_GPUS - 1))"
echo "  Logs        : $LOG_DIR/"
echo ""
echo "  To load balance across workers, point Nginx to:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "    Worker $i → GPU $i → Port $((BASE_PORT + i))"
done
echo ""
echo "  Stop all workers: ./run_server_blackwell.sh stop"
echo "============================================================"
echo ""

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="$LOG_DIR/worker_gpu${i}.log"
    PID_FILE="$PID_DIR/worker_gpu${i}.pid"

    echo "Starting Worker $i on GPU $i, port $PORT → log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$i \
    OLMO_MODEL_PATH="$MODEL_PATH" \
    OLMO_CONFIG_PATH="$CONFIG_PATH" \
    uvicorn app.server:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --workers 1 \
        --log-level info \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "  Worker $i started (PID $(cat "$PID_FILE"))"
done

echo ""
echo "All $NUM_GPUS workers started."
echo ""
echo "Check worker status:"
echo "  nvidia-smi"
echo ""
echo "Tail a worker log:"
echo "  tail -f $LOG_DIR/worker_gpu0.log"
echo ""
echo "Test a worker directly:"
echo "  python -m app.client --server http://localhost:$BASE_PORT"
echo ""
echo "Test via Nginx load balancer (after nginx -c nginx_blackwell.conf):"
echo "  python -m app.client --server http://localhost"
echo ""
