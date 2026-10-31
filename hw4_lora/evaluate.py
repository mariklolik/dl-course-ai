import json
import math
from pathlib import Path
from typing import Any

from data import LABELS, encode_example, prompt_ids, write_json


def parse_label(text: str) -> int:
    stripped = text.strip()
    return LABELS.index(stripped) if stripped in LABELS else -1


def score(rows: list[dict]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    gold = [row["label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    return {
        "n": len(rows),
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, labels=list(range(6)), average="macro", zero_division=0),
        "invalid_outputs": sum(value == -1 for value in predicted),
        "per_class": classification_report(gold, predicted, labels=list(range(6)), target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_labels": ["invalid"] + LABELS,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=[-1] + list(range(6))).tolist(),
    }


def evaluate_model(model: Any, tokenizer: Any, rows: list[dict], max_length: int, batch_size: int, destination: Path) -> dict:
    import torch
    from transformers import DataCollatorForSeq2Seq

    predictions = []
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            current = rows[start:start + batch_size]
            inputs = tokenizer.pad({"input_ids": [prompt_ids(tokenizer, row["text"], max_length)[0] for row in current]}, padding=True, return_tensors="pt").to(model.device)
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=8, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, output in zip(current, decoded):
                predictions.append({**row, "reference": LABELS[row["label"]], "output": output, "predicted_label": parse_label(output)})
            if start % (batch_size * 10) == 0:
                print(f"evaluation {destination.name}: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
        tokenizer.padding_side = "right"
        collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
        nll_sum, target_count = 0.0, 0
        for start in range(0, len(rows), batch_size):
            encoded = [encode_example(tokenizer, row, max_length) for row in rows[start:start + batch_size]]
            batch = collator([{key: value for key, value in item.items() if key != "truncated"} for item in encoded])
            batch = {key: value.to(model.device) for key, value in batch.items()}
            count = int(batch["labels"][:, 1:].ne(-100).sum())
            loss = float(model(**batch).loss)
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite evaluation loss")
            nll_sum += loss * count
            target_count += count
    metrics = score(predictions)
    metrics.update({"completion_nll": nll_sum / target_count, "completion_tokens": target_count})
    write_json(destination, predictions)
    return metrics
