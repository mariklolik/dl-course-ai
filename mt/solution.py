"""English -> Russian paragraph translator (Yandex contest entry point).

Input:   /workspace/input.pickle   -> [{"rid": int, "src": str}, ...]
Output:  /workspace/output.json    -> [{"rid": int, "translation": str}, ...]
Weights: /workspace/weights        (downloaded beforehand by download_weights.py)

A single file drives BOTH submissions. The model family is auto-detected from
weights/config.json so the identical script works whether weights/ holds an
NLLB seq2seq model or a Gemma instruction model:

  * model_type m2m_100 / fsmt  -> dedicated NMT path (sentence-split + beam search)
  * everything else (gemma...)  -> instruction-LLM path (chat prompt + cleanup)

Override detection with the env var MODEL_KIND=nllb|gemma if ever needed.
"""
import json
import os
import pickle
import re

import torch


WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "./weights")
# 13a-style BLEU rewards literal n-gram overlap, so we decode deterministically.
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "5"))
# Keep each NMT chunk comfortably under the model's positional limit (1024).
MAX_SRC_TOKENS = int(os.environ.get("MAX_SRC_TOKENS", "400"))
TRANSLATE_BATCH = int(os.environ.get("TRANSLATE_BATCH", "16"))
GEMMA_MAX_NEW_TOKENS = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "1280"))


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def find_input_path() -> str:
    for p in ("input.pickle", "/workspace/input.pickle"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("input.pickle not found in cwd or /workspace")


def write_output(results: list) -> None:
    # The official baseline writes ./output.json from WORKDIR=/workspace.
    # We mirror that and additionally honour an out/ mount if one is present.
    targets = ["output.json"]
    for d in ("out", "/workspace/out"):
        if os.path.isdir(d):
            targets.append(os.path.join(d, "output.json"))
    for t in targets:
        with open(t, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)


def detect_model_kind() -> str:
    forced = os.environ.get("MODEL_KIND")
    if forced:
        return forced.lower()
    cfg_path = os.path.join(WEIGHTS_DIR, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return "gemma"
    mt = str(cfg.get("model_type", "")).lower()
    archs = " ".join(cfg.get("architectures", []) or []).lower()
    if "m2m_100" in mt or "fsmt" in mt or "nllb" in mt or "marian" in mt or "m2m" in archs:
        return "nllb"
    return "gemma"


# --------------------------------------------------------------------------- #
# Sentence splitting (pure python, no runtime downloads)
# --------------------------------------------------------------------------- #
# Abbreviations that should NOT terminate a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc",
    "ltd", "co", "corp", "no", "vol", "fig", "al", "e.g", "i.e", "u.s",
    "u.k", "u.n", "ph.d", "a.m", "p.m", "approx", "dept", "gov", "min", "max",
}
_SENT_BOUNDARY = re.compile(r"([.!?]+[\"')\]]?)\s+(?=[\"'(\[]?[A-ZА-ЯЁ0-9])")


def split_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    # Insert split markers after sentence-final punctuation, then post-filter
    # out breaks that immediately follow a known abbreviation.
    pieces = _SENT_BOUNDARY.split(text)
    # _SENT_BOUNDARY.split yields [chunk, punct, chunk, punct, ...]; restitch.
    sentences = []
    buf = ""
    i = 0
    while i < len(pieces):
        buf += pieces[i]
        if i + 1 < len(pieces):
            punct = pieces[i + 1]
            buf += punct
            last_word = re.split(r"[\s]", buf.strip())[-1].lower().rstrip(".!?\"')]")
            if last_word in _ABBREV:
                buf += " "  # not a real boundary, keep accumulating
            else:
                sentences.append(buf.strip())
                buf = ""
            i += 2
        else:
            i += 1
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]


def pack_chunks(sentences: list, tokenizer) -> list:
    """Greedily merge consecutive sentences into chunks <= MAX_SRC_TOKENS."""
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        n = len(tokenizer.tokenize(s))
        if n > MAX_SRC_TOKENS:
            # Hard-split an over-long single sentence on token windows.
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            ids = tokenizer.encode(s, add_special_tokens=False)
            for j in range(0, len(ids), MAX_SRC_TOKENS):
                chunks.append(tokenizer.decode(ids[j:j + MAX_SRC_TOKENS]))
            continue
        if cur_len + n > MAX_SRC_TOKENS and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# --------------------------------------------------------------------------- #
# NLLB / seq2seq NMT path
# --------------------------------------------------------------------------- #
def run_nllb(rows: list) -> list:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(WEIGHTS_DIR, dtype=dtype)
    model.to(device).eval()

    # NLLB needs a source language and a forced target BOS; FSMT/Marian do not.
    is_nllb = hasattr(tokenizer, "lang_code_to_id") or "nllb" in WEIGHTS_DIR.lower() \
        or getattr(model.config, "model_type", "") == "m2m_100"
    forced_bos = None
    if is_nllb:
        tokenizer.src_lang = "eng_Latn"
        forced_bos = tokenizer.convert_tokens_to_ids("rus_Cyrl")

    # Flatten every document into chunks, remember which doc each chunk belongs to.
    chunk_texts, chunk_owner = [], []
    per_doc_counts = []
    for idx, row in enumerate(rows):
        sents = split_sentences(row["src"])
        chunks = pack_chunks(sents, tokenizer) if sents else [row["src"]]
        per_doc_counts.append(len(chunks))
        for c in chunks:
            chunk_texts.append(c)
            chunk_owner.append(idx)

    translations = [""] * len(chunk_texts)
    gen_kwargs = dict(num_beams=NUM_BEAMS, max_length=512, no_repeat_ngram_size=3)
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos

    for b in range(0, len(chunk_texts), TRANSLATE_BATCH):
        batch = chunk_texts[b:b + TRANSLATE_BATCH]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SRC_TOKENS + 16).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        dec = tokenizer.batch_decode(out, skip_special_tokens=True)
        for k, t in enumerate(dec):
            translations[b + k] = t.strip()

    # Reassemble per-document translations in original order.
    results, pos = [], 0
    for idx, row in enumerate(rows):
        n = per_doc_counts[idx]
        joined = " ".join(translations[pos:pos + n]).strip()
        pos += n
        results.append({"rid": row["rid"], "translation": joined or row["src"]})
    return results


# --------------------------------------------------------------------------- #
# Gemma instruction-LLM path
# --------------------------------------------------------------------------- #
GEMMA_PROMPT = (
    "Ты профессиональный переводчик. Переведи приведённый ниже текст с "
    "английского языка на русский язык.\n"
    "Требования:\n"
    "- В ответе должен быть ТОЛЬКО перевод и ничего больше.\n"
    "- Не добавляй пояснений, комментариев, кавычек или вводных фраз.\n"
    "- Сохраняй исходное деление на предложения, числа, имена и термины.\n\n"
    "Текст для перевода:\n"
)

_LEAD_JUNK = re.compile(
    r"^\s*(вот\s+перевод[^:]*:|перевод[^:]*:|translation:|здесь\s+перевод[^:]*:)\s*",
    re.IGNORECASE,
)


def clean_gemma(text: str, src: str) -> str:
    t = text.strip()
    t = _LEAD_JUNK.sub("", t).strip()
    # Strip a single layer of wrapping quotes the model sometimes adds.
    if len(t) >= 2 and t[0] in "\"'«“" and t[-1] in "\"'»”":
        t = t[1:-1].strip()
    return t or src


def run_gemma(rows: list) -> list:
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR, dtype="auto", device_map="auto"
    )

    results = []
    for row in rows:
        text = processor.apply_chat_template(
            [{"role": "user", "content": GEMMA_PROMPT + row["src"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=GEMMA_MAX_NEW_TOKENS, do_sample=False
            )
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        try:
            content = processor.parse_response(response)["content"]
        except Exception:
            content = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        results.append({
            "rid": row["rid"],
            "translation": clean_gemma(content, row["src"]),
        })
        inputs.to("cpu")
    return results


# --------------------------------------------------------------------------- #
def main() -> None:
    with open(find_input_path(), "rb") as f:
        rows = pickle.load(f)

    kind = detect_model_kind()
    print(f"[solution] model kind: {kind} | {len(rows)} rows | cuda={torch.cuda.is_available()}")

    if kind == "nllb":
        results = run_nllb(rows)
    else:
        results = run_gemma(rows)

    # Safety net: guarantee one entry per input rid, in order.
    by_rid = {r["rid"]: r for r in results}
    final = [by_rid.get(row["rid"], {"rid": row["rid"], "translation": row["src"]})
             for row in rows]
    write_output(final)
    print(f"[solution] wrote {len(final)} translations")


if __name__ == "__main__":
    main()
