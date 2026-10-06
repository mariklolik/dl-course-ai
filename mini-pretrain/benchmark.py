import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed
from your_solution import create_model, load_tokenized_dataset, prepare_tokenizer, split_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--settings", required=True)
args = parser.parse_args()
settings = json.loads(Path(args.settings).read_text())
set_seed(42)
tokenizer = prepare_tokenizer()
train, evaluation = split_dataset(load_tokenized_dataset(args.data_dir))
train = train.with_format("torch")
evaluation = evaluation.with_format("torch")
model = create_model(tokenizer).cuda()
if settings.get("fp32_weights", False):
    model = model.float()
torch.backends.cuda.matmul.allow_tf32 = settings.get("tf32", True)
torch.backends.cudnn.allow_tf32 = settings.get("tf32", True)
arguments = TrainingArguments(output_dir=str(Path(args.output).parent), report_to="none",
                              learning_rate=settings.get("learning_rate", 3e-4), weight_decay=0.01,
                              optim="adamw_torch_fused" if settings.get("fused", True) else "adamw_torch")
trainer = Trainer(model=model, args=arguments)
trainer.create_optimizer()
optimizer = trainer.optimizer
compiled = torch.compile(model) if settings.get("compile", False) else model
batch_size = settings.get("batch_size", 16)
accumulation = settings.get("accumulation", 1)
dtype = torch.float16 if settings.get("dtype") == "fp16" else torch.bfloat16
scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
records = []
torch.cuda.reset_peak_memory_stats()
try:
    for step in range(settings.get("steps", 12)):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        valid_tokens = 0
        micro_losses = []
        for micro in range(accumulation):
            offset = (step * accumulation + micro) * batch_size
            batch = {key: value.cuda() for key, value in train[offset:offset + batch_size].items()}
            valid_tokens += int(batch["attention_mask"][:, 1:].sum())
            with torch.autocast("cuda", dtype=dtype):
                loss = compiled(**batch).loss / accumulation
            micro_losses.append(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        records.append({"step": step + 1, "seconds": time.monotonic() - started,
                        "loss": float(torch.stack(micro_losses).sum()), "valid_tokens": valid_tokens,
                        "token_slots": batch_size * accumulation * 512})
        print(json.dumps(records[-1]), flush=True)
    model.eval()
    losses = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for offset in range(0, 128, 16):
            batch = {key: value.cuda() for key, value in evaluation[offset:offset + 16].items()}
            losses.append(float(model(**batch).loss))
    result = {"status": "ok", "settings": settings, "records": records,
              "probe_eval_loss_128": sum(losses) / len(losses),
              "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
              "gpu": torch.cuda.get_device_name(), "parameters": sum(p.numel() for p in model.parameters()),
              "optimizer_groups": [{"weight_decay": group["weight_decay"],
                                    "parameters": sum(p.numel() for p in group["params"])}
                                   for group in optimizer.param_groups]}
except torch.cuda.OutOfMemoryError as error:
    result = {"status": "out_of_memory", "settings": settings, "records": records,
              "error": str(error).split("\n")[0]}
Path(args.output).write_text(json.dumps(result, indent=2))
