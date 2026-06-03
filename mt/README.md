# English → Russian paragraph translator (Yandex contest)

Two interchangeable submissions share **one** `solution.py`. It auto-detects the
model family from `weights/config.json`, so the same code runs an NLLB NMT model
or a Gemma instruction model — whichever weights you package.

| Submission | Model | Why |
|---|---|---|
| **NLLB** (recommended first try) | `facebook/nllb-200-distilled-1.3B` | Dedicated en→ru NMT; BLEU rewards literal n-gram overlap, where NMT usually beats a small LLM. ~2.6 GB in fp16. Swap `REPO_ID` to `nllb-200-3.3B` for a bit more BLEU. |
| **Gemma** | `google/gemma-4-E2B-it` | The contest's recommended model; best paragraph coherence; newest. Improved prompt + output cleanup over the baseline. Gated — accept the HF license first. |

## What's here

```
solution.py                 # shared entry point (NLLB + Gemma paths, auto-detected)
download_weights_nllb.py     download_weights_gemma.py
Dockerfile.nllb              Dockerfile.gemma
build_submission.sh         # one command -> upload-ready zip
eval_bleu.py                # reproduces the contest metric (mean per-example sacrebleu)
sample_en_ru.json           # tiny parallel set to smoke-test scoring
```

## Build a submission (any machine with internet, **no GPU needed**)

```bash
pip install -U "transformers==5.5.0" torch huggingface_hub sentencepiece sacremoses
./build_submission.sh nllb      # -> submission_nllb.zip   (flat, ready to upload)
# gemma is gated:
huggingface-cli login           # accept the gemma-4 license on HF first
./build_submission.sh gemma     # -> submission_gemma.zip
```

Upload the resulting zip at `contest.yandex.ru`. The container reads
`/workspace/input.pickle` and writes `/workspace/output.json`
(`[{"rid": int, "translation": str}, ...]`).

## Test locally with Docker (optional)

```bash
docker build -t my-solution -f Dockerfile.nllb .   # after weights/ is populated
mkdir -p out
docker run --rm --gpus all \
  -v "$PWD/example-input.pickle":/workspace/input.pickle \
  my-solution
# drop --gpus all if you have no GPU
```

## A/B which model scores higher (offline, before spending a contest attempt)

```bash
python eval_bleu.py make-input --parallel sample_en_ru.json \
    --out-pickle input.pickle --out-refs refs.json
WEIGHTS_DIR=weights python solution.py        # run inside a populated build dir
python eval_bleu.py score --refs refs.json --hyps output.json
```

Swap `sample_en_ru.json` for FLORES-200 devtest (eng_Latn / rus_Cyrl) for a
realistic estimate. Submit whichever model scores higher; you have 4 attempts / 24h.

## Notes
- Runtime container has **no internet** — weights are baked into the image via `COPY . .`.
- Deterministic decoding (beam search / greedy, no sampling) for reproducible scores.
- Limits honoured: ≤45 min for 69 examples on one L4, image < 20 GB.
- `Dockerfile.*` keep the Yandex base image (fast, guaranteed). Replace the `FROM`
  line with a public `pytorch/pytorch` CUDA image to build outside the contest.
