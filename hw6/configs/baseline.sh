set -euo pipefail
cd "$(dirname "$0")/.."
exec bash serve.sh  "$@"
