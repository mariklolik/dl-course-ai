import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Trainer
from transformers import TrainingArguments, TrainerCallback, get_scheduler, set_seed

MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000
TRAINING_CONFIG = {
    "optim": "adamw_torch_fused",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 20,
    "logging_steps": 5,
    "eval_strategy": "no",
    "save_strategy": "no",
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": False,
    "dataloader_num_workers": 4,
    "torch_compile": False,
    "report_to": "none",
    "seed": 42,
    "data_seed": 42,
    "prediction_loss_only": True,
    "include_num_input_tokens_seen": True,
}


class TimeoutCallback(TrainerCallback):
    def __init__(self, timeout_seconds, output_dir):
        self.timeout_seconds = timeout_seconds
        self.output_dir = Path(output_dir)
        self.start_time = None
        self.elapsed = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        self.elapsed = time.monotonic() - self.start_time
        if self.elapsed > self.timeout_seconds:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs, **kwargs):
        record = dict(logs, step=state.global_step)
        record["elapsed_seconds"] = time.monotonic() - self.start_time if self.start_time else 0.0
        with (self.output_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    encoded = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    encoded["labels"] = [
        [token if active else -100 for token, active in zip(ids, mask)]
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
    ]
    return encoded


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for index in range(num_shards):
        ds.shard(num_shards, index, contiguous=True).to_parquet(str(Path(output_dir) / f"{index:05d}.parquet"))


def prepare_dataset(output_dir=OUTPUT_DIR):
    tokenizer = prepare_tokenizer()
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    manifest = {"dataset": "wikimedia/wikipedia", "config": "20231101.ru", "rows": len(dataset),
                "fingerprint": dataset._fingerprint, "validation_size": VALIDATION_SIZE,
                "validation_ids_sha256": hashlib.sha256(json.dumps(dataset[:VALIDATION_SIZE]["id"]).encode()).hexdigest(),
                "first_train_id": dataset[VALIDATION_SIZE]["id"], "first_eval_id": dataset[0]["id"],
                "max_length": MAX_LENGTH, "tokenizer": TOKENIZER_NAME}
    tokenized = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer},
                            remove_columns=dataset.column_names, num_proc=8)
    save_as_parquets(tokenized, output_dir)
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted(str(path) for path in Path(data_dir).glob("*.parquet"))
    return load_dataset("parquet", data_files=files, split="train")


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    return dataset.select(range(validation_size, len(dataset))), dataset.select(range(validation_size))


def create_model(tokenizer):
    model_config = {
        "hidden_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "head_dim": 128,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
    }
    config = Qwen3Config(vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, **model_config)
    return Qwen3ForCausalLM._from_config(config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)


def generate_examples(model, tokenizer):
    model.eval()
    examples = []
    for prompt in ["Москва", "В начале XX века", "Солнечная система", "Искусственный интеллект"]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        for sampling in [False, True]:
            set_seed(123)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=64, do_sample=sampling,
                                        **({"temperature": 0.8, "top_p": 0.9} if sampling else {}),
                                        pad_token_id=tokenizer.pad_token_id)
            examples.append({"prompt": prompt, "sampling": sampling,
                             "text": tokenizer.decode(output[0], skip_special_tokens=True),
                             "token_ids": output[0].tolist()})
    model.train()
    return examples


def train_model(data_dir, output_dir, settings):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires exactly one visible GPU")
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    set_seed(42)
    tokenizer = prepare_tokenizer()
    train, evaluation = split_dataset(load_tokenized_dataset(data_dir))
    model = create_model(tokenizer)
    config = TRAINING_CONFIG | settings
    scheduler_steps = config.pop("scheduler_steps", None)
    scheduler_type = config.pop("lr_scheduler_type", "constant_with_warmup")
    arguments = TrainingArguments(output_dir=output_dir, lr_scheduler_type=scheduler_type, **config)
    callback = TimeoutCallback(MAX_TRAINING_TIME_SECONDS, output_dir)
    trainer = Trainer(model=model, args=arguments, train_dataset=train, eval_dataset=evaluation,
                      processing_class=tokenizer, callbacks=[callback])
    if scheduler_steps:
        trainer.create_optimizer()
        trainer.lr_scheduler = get_scheduler(scheduler_type, trainer.optimizer,
                                             num_warmup_steps=config["warmup_steps"], num_training_steps=scheduler_steps)
    initial = trainer.evaluate()
    before = generate_examples(model, tokenizer)
    torch.cuda.reset_peak_memory_stats()
    set_seed(42)
    trainer.train()
    final = trainer.evaluate()
    trainer.save_state()
    trainer.save_model(str(Path(output_dir) / "final"))
    after = generate_examples(model, tokenizer)
    metrics = {"initial": initial, "final": final, "perplexity": math.exp(final["eval_loss"]),
               "training_seconds": callback.elapsed, "steps": trainer.state.global_step,
               "token_slots": trainer.state.num_input_tokens_seen, "gpu": torch.cuda.get_device_name(),
               "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
               "parameters": sum(p.numel() for p in model.parameters()), "settings": settings,
               "training_arguments": arguments.to_dict(), "model_config": model.config.to_dict(),
               "train_rows": len(train), "eval_rows": len(evaluation),
               "optimizer_groups": [{"weight_decay": group["weight_decay"],
                                     "parameters": sum(p.numel() for p in group["params"])}
                                    for group in trainer.optimizer.param_groups]}
    (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (Path(output_dir) / "generations.json").write_text(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))
    print(json.dumps({"final_eval_loss": final["eval_loss"], "training_seconds": callback.elapsed}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "train"])
    parser.add_argument("--data-dir", default=OUTPUT_DIR)
    parser.add_argument("--output-dir", default="runs/main")
    parser.add_argument("--settings")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_dataset(args.data_dir)
    else:
        settings = json.loads(Path(args.settings).read_text()) if args.settings else {}
        train_model(args.data_dir, args.output_dir, settings)
