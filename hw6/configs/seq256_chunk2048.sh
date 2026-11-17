set -euo pipefail
cd "$(dirname "$0")/.."
exec bash serve.sh --max-num-seqs 256 --max-num-batched-tokens 2048 "$@"
