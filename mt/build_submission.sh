#!/usr/bin/env bash
# Build an upload-ready contest zip on any machine with internet (no GPU needed).
#
#   ./build_submission.sh nllb     # -> submission_nllb.zip
#   ./build_submission.sh gemma    # -> submission_gemma.zip   (gemma is gated:
#                                  #    run `huggingface-cli login` & accept license first)
#
# The resulting zip is "flat" (Dockerfile, solution.py, weights/ at the root),
# exactly as the contest expects. Upload it at contest.yandex.ru.
set -euo pipefail

MODEL="${1:-nllb}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/build_${MODEL}"
ZIP="$HERE/submission_${MODEL}.zip"

case "$MODEL" in
  nllb)  DOWNLOAD="download_weights_nllb.py";  DOCKER="Dockerfile.nllb"  ;;
  gemma) DOWNLOAD="download_weights_gemma.py"; DOCKER="Dockerfile.gemma" ;;
  *) echo "usage: $0 [nllb|gemma]"; exit 1 ;;
esac

echo ">> preparing $BUILD"
rm -rf "$BUILD"; mkdir -p "$BUILD"
cp "$HERE/solution.py"        "$BUILD/solution.py"
cp "$HERE/$DOWNLOAD"          "$BUILD/download_weights.py"
cp "$HERE/$DOCKER"            "$BUILD/Dockerfile"

echo ">> downloading weights (this is the only step that needs internet)"
( cd "$BUILD" && python3 download_weights.py )

echo ">> zipping -> $ZIP"
rm -f "$ZIP"
( cd "$BUILD" && zip -r -q "$ZIP" Dockerfile solution.py download_weights.py weights )
echo ">> done: $ZIP ($(du -h "$ZIP" | cut -f1))"
echo ">> upload this file at contest.yandex.ru"
