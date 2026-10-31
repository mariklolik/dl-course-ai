set -eu
name="$1"
python train.py "$name"
python export.py "$name"
python evaluate.py "outputs/$name/merged" --output "outputs/$name"
