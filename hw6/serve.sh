#!/usr/bin/env bash
set -euo pipefail

# Включаем API очистки кеша
export VLLM_SERVER_DEV_MODE=1

exec vllm serve Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port "${PORT:-8616}" "$@"
