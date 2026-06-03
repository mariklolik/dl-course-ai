"""Offline BLEU harness — reproduces the contest metric so you can A/B models.

The contest score is the arithmetic mean over examples of per-example sacrebleu
BLEU (sentence-level, default 13a tokenizer). This script computes exactly that.

Typical flow (no GPU needed for tiny samples):

    # 1. Turn a parallel set into a contest-style input.pickle + reference file
    python eval_bleu.py make-input  --parallel sample_en_ru.json \
                                    --out-pickle input.pickle --out-refs refs.json
    # 2. Run the actual solution on it (inside the submission folder)
    WEIGHTS_DIR=weights python solution.py
    # 3. Score the produced output.json against the references
    python eval_bleu.py score --refs refs.json --hyps output.json

parallel JSON format: [{"src": "...english...", "ref": "...russian..."}, ...]

For a real estimate, plug in FLORES-200 devtest (eng_Latn / rus_Cyrl) or any
held-out en->ru parallel corpus as sample_en_ru.json.
"""
import argparse
import json
import pickle

import sacrebleu


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_make_input(args):
    pairs = load_json(args.parallel)
    rows, refs = [], []
    for i, p in enumerate(pairs):
        rows.append({"rid": i, "src": p["src"]})
        refs.append({"rid": i, "ref": p["ref"]})
    with open(args.out_pickle, "wb") as f:
        pickle.dump(rows, f)
    with open(args.out_refs, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False)
    print(f"wrote {len(rows)} rows -> {args.out_pickle}, refs -> {args.out_refs}")


def cmd_score(args):
    refs = {r["rid"]: r["ref"] for r in load_json(args.refs)}
    hyps = {h["rid"]: h["translation"] for h in load_json(args.hyps)}
    missing = set(refs) - set(hyps)
    if missing:
        print(f"WARNING: {len(missing)} rids missing from hyps (scored as empty)")

    scores = []
    for rid, ref in refs.items():
        hyp = hyps.get(rid, "")
        # effective_order=True is required for stable sentence-level BLEU.
        bleu = sacrebleu.sentence_bleu(hyp, [ref])
        scores.append(bleu.score)
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"examples: {len(scores)}")
    print(f"mean per-example BLEU: {mean:.2f}")
    return mean


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-input")
    m.add_argument("--parallel", required=True)
    m.add_argument("--out-pickle", default="input.pickle")
    m.add_argument("--out-refs", default="refs.json")
    m.set_defaults(func=cmd_make_input)

    s = sub.add_parser("score")
    s.add_argument("--refs", required=True)
    s.add_argument("--hyps", default="output.json")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
