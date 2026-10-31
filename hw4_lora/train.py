import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import time
from pathlib import Path

from data import encode_example, prepare, read_rows, write_json


def main() -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--name")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiments.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    case = config["experiments"][args.case]
    settings = config["training"]
    name = args.name or args.case
    if (args.max_steps > 0 or args.train_limit or args.eval_limit) and not name.startswith("smoke_"):
        raise ValueError("Limited runs require --name smoke_<name>")
    destination = args.output / name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "summary.json").exists():
        previous = json.loads((destination / "summary.json").read_text())
        if previous["config"] != config or previous["case"] != case:
            raise ValueError("Existing results use a different configuration")
        print(f"Already complete: {destination}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for these experiments")
    set_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not (args.data / "manifest.json").exists():
        prepare(args.data, config)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["dataset"] != config["dataset"] or manifest["revision"] != config["dataset_revision"]:
        raise ValueError("Prepared dataset uses a different source")
    splits = {}
    for split in ("train", "validation", "test"):
        path = args.data / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][split]:
            raise ValueError(f"Dataset hash mismatch: {split}")
        rows = read_rows(path)
        limit = args.train_limit if split == "train" else args.eval_limit
        splits[split] = rows[:limit] if limit else rows
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    checkpoint = destination / "checkpoint"
    model_source = checkpoint if args.evaluate_only and case["mode"] == "full" else config["model"]
    load_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if model_source == config["model"]:
        load_kwargs["revision"] = config["model_revision"]
    model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs).to("cuda")
    if case["mode"] == "lora":
        if args.evaluate_only:
            model = PeftModel.from_pretrained(model, checkpoint)
        else:
            model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=case["r"], lora_alpha=case["alpha"], lora_dropout=case["dropout"], target_modules=case["target_modules"], bias="none"))
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) if case["mode"] != "base" else 0
    if case["mode"] == "full" and not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full FT must train every model parameter")
    model.config.use_cache = False
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    encoded = [encode_example(tokenizer, row, settings["max_length"]) for row in splits["train"]]
    train_data = Dataset.from_list([{key: value for key, value in row.items() if key != "truncated"} for row in encoded])
    smoke_batch = {key: value.to(model.device) for key, value in collator([train_data[0], train_data[1]]).items()}
    model.eval()
    with torch.no_grad():
        initial_loss = float(model(**smoke_batch).loss)
    del smoke_batch
    if not math.isfinite(initial_loss):
        raise FloatingPointError("Initial forward pass has nonfinite loss")
    print(f"case={args.case} total={total} trainable={trainable} smoke_loss={initial_loss:.6f}", flush=True)
    versions = {package: importlib.metadata.version(package) for package in ("torch", "transformers", "peft", "datasets", "accelerate", "scikit-learn")}
    training = {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_seconds": 0.0,
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
        "optimizer_steps": 0,
        "initial_forward_loss": initial_loss,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "versions": versions,
        "seed": settings["seed"],
        "train_examples": len(train_data),
        "planned_target_tokens": sum(sum(value != -100 for value in row["labels"]) for row in encoded),
    }
    if args.evaluate_only and case["mode"] != "base":
        training = json.loads((destination / "training.json").read_text())
    elif case["mode"] != "base":
        class FiniteLoss(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key in ("loss", "grad_norm", "train_loss"):
                    if key in (logs or {}) and not math.isfinite(logs[key]):
                        raise FloatingPointError(f"Nonfinite {key} at step {state.global_step}")

        arguments = TrainingArguments(
            output_dir=str(destination / "trainer"),
            num_train_epochs=settings["num_train_epochs"],
            max_steps=args.max_steps,
            per_device_train_batch_size=settings["per_device_train_batch_size"],
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            gradient_checkpointing=settings["gradient_checkpointing"],
            learning_rate=case["learning_rate"],
            lr_scheduler_type=settings["lr_scheduler_type"],
            warmup_ratio=settings["warmup_ratio"],
            weight_decay=settings["weight_decay"],
            optim="adamw_torch_fused",
            bf16=True,
            tf32=True,
            max_grad_norm=1.0,
            logging_steps=10,
            logging_nan_inf_filter=False,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=settings["seed"],
            data_seed=settings["seed"],
            dataloader_num_workers=0,
            dataloader_drop_last=False,
            disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=train_data, data_collator=collator, processing_class=tokenizer, callbacks=[FiniteLoss()])
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        training.update({
            "train_seconds": time.perf_counter() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "optimizer_steps": trainer.state.global_step,
            "train_metrics": result.metrics,
            "actual_epochs": trainer.state.epoch,
            "target_tokens_one_epoch": training["planned_target_tokens"],
            "consumed_examples_inferred_from_epoch": round(trainer.state.epoch * len(train_data)),
            "consumed_target_tokens_if_complete_epoch": training["planned_target_tokens"] if trainer.state.epoch == 1.0 else None,
            "limited_smoke_run": name.startswith("smoke_"),
        })
        write_json(destination / "training_history.json", trainer.state.log_history)
        write_json(destination / "training.json", training)
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    if case["mode"] != "base":
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if case["mode"] == "full":
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["model_revision"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
            model = PeftModel.from_pretrained(model, checkpoint)
        training["checkpoint_reloaded_for_evaluation"] = True
    write_json(destination / "training.json", training)
    from evaluate import evaluate_model

    metrics = {}
    for split in ("validation", "test"):
        metrics[split] = evaluate_model(model, tokenizer, splits[split], settings["max_length"], settings["eval_batch_size"], destination / f"{split}_predictions.json")
        print(json.dumps({"case": name, "split": split, **metrics[split]}), flush=True)
    write_json(destination / "summary.json", {"name": name, "case_name": args.case, "case": case, "config": config, "data_manifest": manifest, "training": training, "metrics": metrics, "smoke": name.startswith("smoke_")})
    print(f"COMPLETE {name}", flush=True)


if __name__ == "__main__":
    main()
