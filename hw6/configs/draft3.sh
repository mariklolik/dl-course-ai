set -euo pipefail
cd "$(dirname "$0")/.."
exec bash serve.sh --max-num-seqs 32 --max-num-batched-tokens 2048 --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}' "$@"
