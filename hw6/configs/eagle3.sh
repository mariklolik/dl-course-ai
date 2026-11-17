set -euo pipefail
cd "$(dirname "$0")/.."
exec bash serve.sh --max-num-seqs 32 --max-num-batched-tokens 2048 --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}' "$@"
