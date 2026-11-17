set -euo pipefail
cd "$(dirname "$0")/.."
exec bash serve.sh --max-num-seqs 64 --max-num-batched-tokens 4096 "$@"
