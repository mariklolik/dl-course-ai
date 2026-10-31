import json
import unittest
from pathlib import Path


class PipelineTests(unittest.TestCase):
    def test_dedup_preserves_heldout_priority(self):
        from data import clean_splits

        source = {
            "train": [{"text": " A  text ", "label": 0}, {"text": "only train", "label": 1}],
            "validation": [{"text": "a text", "label": 2}, {"text": "valid", "label": 3}],
            "test": [{"text": "A TEXT", "label": 4}, {"text": "test", "label": 5}, {"text": "test ", "label": 5}],
        }
        splits, audit = clean_splits(source)
        self.assertEqual([row["text"] for row in splits["train"]], ["only train"])
        self.assertEqual([row["label"] for row in splits["test"]], [4, 5])
        self.assertEqual(audit["removed"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(audit["conflicting_duplicates"], 2)

    def test_encoding_masks_prompt_and_retains_entire_target(self):
        from data import encode_example

        class Tokenizer:
            eos_token_id = 9

            def apply_chat_template(self, messages, **kwargs):
                return list(range(20))

            def encode(self, text, **kwargs):
                return [71, 72]

        example = encode_example(Tokenizer(), {"text": "sample", "label": 0}, max_length=16)
        self.assertEqual(example["input_ids"][-3:], [71, 72, 9])
        self.assertEqual(example["labels"], [-100] * 8 + [71, 72, 9])
        self.assertEqual(example["attention_mask"], [1] * 11)
        self.assertTrue(example["truncated"])

    def test_strict_parser_rejects_explanations_and_unknown_labels(self):
        from evaluate import parse_label

        self.assertEqual(parse_label(" sadness\n"), 0)
        for text in ["Sadness", "sadness because", "joy.", "", "neutral"]:
            self.assertEqual(parse_label(text), -1)

    def test_lora_ablation_changes_exactly_one_parameter(self):
        config = json.loads(Path(__file__).with_name("experiments.json").read_text())
        reference = config["experiments"]["lora"]
        self.assertEqual(set(config["experiments"]), {"base", "full", "lora", "r4", "r64", "alpha8", "alpha128", "dropout0", "dropout015", "attention", "all_linear"})
        for name, case in config["experiments"].items():
            if name in {"base", "full", "lora"}:
                continue
            self.assertEqual(sum(case[key] != reference[key] for key in reference), 1, name)


if __name__ == "__main__":
    unittest.main()
