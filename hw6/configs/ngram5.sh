set -euo pipefail
cd "$(dirname "$0")/.."
exec bash serve.sh --max-num-seqs 32 --max-num-batched-tokens 2048 --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":2,"prompt_lookup_max":5}' "$@"
