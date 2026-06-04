"""Head-to-head: baseline-Gemma vs improved-Gemma vs NLLB on FLORES-200 en->ru.

Reproduces the contest metric (mean per-example sacrebleu BLEU) on a paragraph
benchmark built from FLORES dev. Reuses the SHIPPED solution.py code paths so
the numbers reflect what actually gets submitted.

    NLLB_DIR=/tmp/nllb_src GEMMA_DIR=/tmp/gemma_src python3 compare.py
"""
import json, os, time
import torch
import sacrebleu
import solution as S

BENCH = json.load(open("bench_flores.json", encoding="utf-8"))
REFS = [b["ref"] for b in BENCH]


def mean_bleu(hyps):
    scores = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, REFS)]
    return sum(scores) / len(scores), scores


# Exact prompt from the official baseline solution.py.
BASELINE_PROMPT = (
    "\nТы профессиональный переводчик. Твоя задача перевести текст с английского "
    "на русский.\nВАЖНО чтобы в ответе был только перевод без каких-либо "
    "комментариев и уточнений.\n\nТекст для перевода:\n"
)

results = {}


def run_nllb():
    S.WEIGHTS_DIR = os.environ["NLLB_DIR"]
    rows = [{"rid": i, "src": b["src"]} for i, b in enumerate(BENCH)]
    t = time.time()
    out = S.run_nllb(rows)
    dt = time.time() - t
    hyps = [o["translation"] for o in sorted(out, key=lambda x: x["rid"])]
    results["NLLB-1.3B (mine)"] = (mean_bleu(hyps)[0], hyps, dt)


def run_gemma_both():
    from transformers import AutoModelForCausalLM, AutoProcessor
    W = os.environ["GEMMA_DIR"]
    proc = AutoProcessor.from_pretrained(W)
    model = AutoModelForCausalLM.from_pretrained(W, dtype="auto")

    def generate(prompt, max_new, greedy):
        text = proc.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = proc(text=text, return_tensors="pt").to(model.device)
        il = inp["input_ids"].shape[-1]
        kw = {"max_new_tokens": max_new}
        if greedy:
            kw["do_sample"] = False
        with torch.no_grad():
            o = model.generate(**inp, **kw)
        resp = proc.decode(o[0][il:], skip_special_tokens=False)
        try:
            return proc.parse_response(resp)["content"]
        except Exception:
            return proc.decode(o[0][il:], skip_special_tokens=True)

    # Baseline: baseline prompt, default generation (as the baseline ships), no cleanup.
    torch.manual_seed(0)
    t = time.time()
    base = [generate(BASELINE_PROMPT + b["src"], 1024, greedy=False) for b in BENCH]
    results["Gemma baseline"] = (mean_bleu(base)[0], base, time.time() - t)

    # Improved: my prompt + greedy + clean_gemma (the shipped behaviour).
    t = time.time()
    imp = []
    for b in BENCH:
        raw = generate(S.GEMMA_PROMPT + b["src"], 512, greedy=True)
        imp.append(S.clean_gemma(raw, b["src"]))
    results["Gemma improved (mine)"] = (mean_bleu(imp)[0], imp, time.time() - t)


if __name__ == "__main__":
    run_nllb()
    print("NLLB done")
    run_gemma_both()
    print("\n================ RESULTS (mean per-example BLEU) ================")
    for name, (score, _, dt) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:28s} BLEU={score:6.2f}   ({dt:.0f}s)")
    json.dump({k: {"bleu": v[0], "hyps": v[1], "secs": v[2]} for k, v in results.items()},
              open("compare_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote compare_results.json")
