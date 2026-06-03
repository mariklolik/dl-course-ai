"""Download NLLB-200 weights into ./weights (run ONCE locally before docker build).

    pip install -U "transformers==5.5.0" huggingface_hub
    python3 download_weights.py

The grading server has no internet (--network none), so the weights must be
baked into the image via `COPY . .`. To save image space we re-save the model
in float16 (the L4 GPU runs it in fp16 anyway) which roughly halves the size.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ID = "facebook/nllb-200-distilled-1.3B"  # swap to nllb-200-3.3B for max BLEU
LOCAL_DIR = "weights"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(REPO_ID, dtype=torch.float16)
    tok.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR, safe_serialization=True)
    print(f"weights saved to: {LOCAL_DIR}")


if __name__ == "__main__":
    main()
