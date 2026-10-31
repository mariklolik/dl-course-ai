import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SYSTEM = "Classify the emotion. Reply with exactly one label: sadness, joy, love, anger, fear, surprise."


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def clean_splits(source: dict[str, Any]) -> tuple[dict, dict]:
    seen: dict[str, int] = {}
    result: dict[str, list] = {}
    audit = {"original": {}, "removed": {}, "conflicting_duplicates": 0}
    for split in ("test", "validation", "train"):
        result[split] = []
        audit["original"][split] = len(source[split])
        for index, row in enumerate(source[split]):
            key = normalize(row["text"])
            label = int(row["label"])
            if key in seen:
                audit["conflicting_duplicates"] += int(seen[key] != label)
                continue
            seen[key] = label
            result[split].append({"id": f"{split}:{index}", "text": row["text"], "label": label})
        audit["removed"][split] = len(source[split]) - len(result[split])
    return result, audit


def prompt_ids(tokenizer: Any, text: str, max_length: int) -> tuple[list[int], bool]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    budget = max_length - 8
    if budget < 1:
        raise ValueError("max_length must exceed eight reserved generation tokens")
    return tokens[-budget:], len(tokens) > budget


def encode_example(tokenizer: Any, row: dict, max_length: int) -> dict:
    prompt, truncated = prompt_ids(tokenizer, row["text"], max_length)
    target = tokenizer.encode(LABELS[row["label"]], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(target) > 8:
        raise ValueError("Target exceeds reserved completion length")
    tokens = prompt + target
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": [-100] * len(prompt) + target, "truncated": truncated}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(destination: Path, config: dict) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    source = load_dataset(config["dataset"], "split", revision=config["dataset_revision"])
    splits, audit = clean_splits(source)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    audit.update({"dataset": config["dataset"], "revision": config["dataset_revision"], "counts": {}, "sha256": {}, "class_counts": {}, "truncated_prompts": {}, "target_tokens": {}, "max_encoded_length": {}})
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = destination / f"{split}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        path.write_text(content)
        encoded = [encode_example(tokenizer, row, config["training"]["max_length"]) for row in rows]
        audit["counts"][split] = len(rows)
        audit["sha256"][split] = hashlib.sha256(content.encode()).hexdigest()
        audit["class_counts"][split] = {name: sum(row["label"] == index for row in rows) for index, name in enumerate(LABELS)}
        audit["truncated_prompts"][split] = sum(item["truncated"] for item in encoded)
        audit["target_tokens"][split] = sum(sum(token != -100 for token in item["labels"]) for item in encoded)
        audit["max_encoded_length"][split] = max(len(item["input_ids"]) for item in encoded)
    write_json(destination / "manifest.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiments.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, json.loads(args.config.read_text())), indent=2))
