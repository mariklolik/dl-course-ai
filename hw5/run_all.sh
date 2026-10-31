set -eu
python prepare_data.py --train-size 2048 --validation-size 128 --max-prompt-tokens 3072 --seed 42
python evaluate.py Qwen/Qwen3.5-2B --output outputs/base
CUDA_VISIBLE_DEVICES= python verify_references.py
for name in rl1 rl2 rl3; do
    python train.py "$name"
    python export.py "$name"
    python evaluate.py "outputs/$name/merged" --output "outputs/$name"
done
for name in rl1 rl2 rl3; do
    other="$(python -c 'import json,sys; print(240-json.load(open(sys.argv[1]))["selected_step"])' "outputs/$name/training_summary.json")"
    CUDA_VISIBLE_DEVICES= python export.py "$name" --step "$other"
    python evaluate.py "outputs/$name/merged-step$other" --output "outputs/$name/step-$other"
done
python train.py rl3 --steps 320 --output outputs/rl4
CUDA_VISIBLE_DEVICES= python export.py rl3 --root outputs/rl4
python evaluate.py outputs/rl4/merged --output outputs/rl4
python analyze.py
