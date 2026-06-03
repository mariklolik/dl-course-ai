"""Download gemma-4-E2B-it weights into ./weights (run ONCE locally before build).

    pip install -U "transformers==5.5.0" huggingface_hub
    huggingface-cli login          # gemma is gated: accept the license first
    python3 download_weights.py

The grading server has no internet, so weights are baked into the image via
`COPY . .` and end up at /workspace/weights inside the container.
"""
from huggingface_hub import snapshot_download

REPO_ID = "google/gemma-4-E2B-it"
LOCAL_DIR = "weights"


def main() -> None:
    path = snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR)
    print(f"weights downloaded to: {path}")


if __name__ == "__main__":
    main()
