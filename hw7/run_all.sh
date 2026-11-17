set -euo pipefail
cd "$(dirname "$0")"
python data.py
test -f artifacts_compression_hw/evaluations/baseline.json || python evaluate.py
test -f artifacts_compression_hw/training/smoke/summary.json || python distill.py smoke
test -f artifacts_compression_hw/teacher_answers/manifest.json || python distill.py teacher
for name in awq_massive awq_wikitext gptq_g128 gptq_n64 gptq_g64 sq_a05 sq_a08 gptq_lastbf16; do
    test -f "artifacts_compression_hw/quantization/$name.json" || python quantize.py --config "$name"
done
for name in s128 s1024 s1024_e3; do
    test -f "artifacts_compression_hw/evaluations/$name.json" || python distill.py train --config "$name"
done
test -f artifacts_compression_hw/student_selection.json || python distill.py select
