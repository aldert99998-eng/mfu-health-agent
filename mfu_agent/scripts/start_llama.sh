#!/usr/bin/env bash
# Idempotent starter for llama.cpp's llama-server on port 8000.
# Skips startup if :8000 already answers /v1/models.
#
# Overrides:
#   MFU_LOCAL_MODEL     — .gguf file name or absolute path
#                          (default: first file in /home/albert/models/*.gguf)
#   MFU_LLAMA_EXEC      — path to llama-server binary
#                          (default: /home/albert/llama.cpp/build/bin/llama-server)
#   MFU_LLAMA_LOG       — log file (default: /tmp/llama-server.log)
#   MFU_LOCAL_MODELS_DIR — models root (default: /home/albert/models)

set -u

PORT=8000
LLAMA_EXEC=${MFU_LLAMA_EXEC:-/home/albert/llama.cpp/build/bin/llama-server}
MODELS_DIR=${MFU_LOCAL_MODELS_DIR:-/home/albert/models}
LOG=${MFU_LLAMA_LOG:-/tmp/llama-server.log}

# Already running and healthy?
if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[start_llama] llama-server already up on :${PORT}"
    exit 0
fi

if [ ! -x "$LLAMA_EXEC" ]; then
    echo "[start_llama] ERROR: llama-server binary not found at $LLAMA_EXEC" >&2
    echo "[start_llama] set MFU_LLAMA_EXEC to override" >&2
    exit 1
fi

# Resolve model path.
MODEL_ARG=${MFU_LOCAL_MODEL:-}
if [ -z "$MODEL_ARG" ]; then
    MODEL_PATH=$(find "$MODELS_DIR" -maxdepth 1 -name '*.gguf' -type f | sort | head -n 1)
elif [ -f "$MODEL_ARG" ]; then
    MODEL_PATH="$MODEL_ARG"
else
    MODEL_PATH="$MODELS_DIR/$MODEL_ARG"
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "[start_llama] ERROR: no .gguf model found (MODELS_DIR=$MODELS_DIR)" >&2
    exit 1
fi

echo "[start_llama] starting $MODEL_PATH on :${PORT} (log: $LOG)"
mkdir -p "$(dirname "$LOG")"
nohup "$LLAMA_EXEC" \
    -m "$MODEL_PATH" \
    --port "$PORT" --host 0.0.0.0 \
    -ngl 99 -c 32768 --parallel 2 \
    --reasoning-format none --chat-template chatml \
    >> "$LOG" 2>&1 &

# Wait up to 90s for /v1/models to answer.
for i in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[start_llama] ready in ${i}s"
        exit 0
    fi
    sleep 1
done

echo "[start_llama] ERROR: llama-server did not answer /v1/models in 90s. See $LOG" >&2
exit 1
