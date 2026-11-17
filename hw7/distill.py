import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from data import (ARTIFACTS, BASE_MODEL_ID, BASE_REVISION, GENERATE_BATCH_SIZE, SEED,
                  TEACHER_MODEL_ID, TEACHER_REVISION, encode_prompt, load_data, system_prompt)
from evaluate import clean_memory, evaluate_checkpoint, generate_answer_ids, load_model, parse_answer


class TeacherAnswerDataset:
    def __init__(self, path: str | Path, samples: int | None = None) -> None:
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        if samples is not None:
            self.rows = self.rows[:samples]
        if not self.rows or any(not row['prompt_ids'] or not row['answer_ids'] for row in self.rows):
            raise ValueError('Пустые teacher tokens')

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {'input_ids': row['prompt_ids'] + row['answer_ids'],
                'attention_mask': [1] * (len(row['prompt_ids']) + len(row['answer_ids'])),
                'labels': [-100] * len(row['prompt_ids']) + row['answer_ids']}


def collate_teacher_answers(features: list[dict], pad_id: int):
    import torch

    width = max(len(feature['input_ids']) for feature in features)
    batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for feature in features:
        pad = width - len(feature['input_ids'])
        batch['input_ids'].append(feature['input_ids'] + [pad_id] * pad)
        batch['attention_mask'].append(feature['attention_mask'] + [0] * pad)
        batch['labels'].append(feature['labels'] + [-100] * pad)
    return {key: torch.tensor(value) for key, value in batch.items()}


def collect_teacher_answers() -> None:
    from transformers import AutoConfig, AutoTokenizer

    folder = ARTIFACTS / 'teacher_answers'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'answers.jsonl'
    if output.exists():
        raise FileExistsError(output)
    train, _, intents, slots = load_data()
    rows = train.shuffle(seed=SEED).select(range(1024))
    model, tokenizer = load_model(TEACHER_MODEL_ID, TEACHER_REVISION)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    assert tokenizer.get_vocab() == base_tokenizer.get_vocab()
    assert tokenizer.chat_template == base_tokenizer.chat_template
    assert tokenizer.eos_token_id == base_tokenizer.eos_token_id
    assert tokenizer.pad_token_id == base_tokenizer.pad_token_id
    assert model.config.vocab_size == AutoConfig.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION).vocab_size
    start = time.monotonic()
    prompt = system_prompt(intents, slots)
    records = []
    try:
        with output.with_suffix('.tmp').open('w') as handle:
            for offset in range(0, len(rows), GENERATE_BATCH_SIZE):
                batch = list(rows.select(range(offset, min(offset + GENERATE_BATCH_SIZE, len(rows)))))
                prompts = [encode_prompt(tokenizer, row['text'], prompt) for row in batch]
                answers = generate_answer_ids(model, tokenizer, prompts)
                for row, prompt_ids, answer_ids in zip(batch, prompts, answers, strict=True):
                    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
                    ended = answer_ids[-1] == tokenizer.eos_token_id
                    record = dict(row, teacher_answer=answer, prompt_ids=prompt_ids, answer_ids=answer_ids,
                                  ended_eos=ended, truncated=not ended, valid=parse_answer(answer, intents, slots) is not None)
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                print(json.dumps({'teacher_answers': len(records), 'total': len(rows)}), flush=True)
        output.with_suffix('.tmp').replace(output)
        summary = {'teacher': TEACHER_MODEL_ID, 'revision': TEACHER_REVISION, 'n': len(records),
                   'seed': SEED, 'ids': [r['id'] for r in records], 'seconds': time.monotonic() - start,
                   'ended_eos': sum(r['ended_eos'] for r in records), 'valid': sum(r['valid'] for r in records),
                   'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'filtered': 0,
                   'prompt_tokens': sum(len(r['prompt_ids']) for r in records),
                   'answer_tokens': sum(len(r['answer_ids']) for r in records)}
        (folder / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        del model
        clean_memory()


def select_student() -> None:
    names = json.loads(Path('configs.json').read_text())['distillation']
    metrics = [json.loads((ARTIFACTS / 'evaluations' / f'{name}.json').read_text())
               for name in names if (ARTIFACTS / 'evaluations' / f'{name}.json').exists()]
    assert {'s128', 's1024', 's1024_e3'} <= {row['run'] for row in metrics}
    best = max(metrics, key=lambda row: row['quality'])
    source = ARTIFACTS / 'students' / best['run']
    target = ARTIFACTS / 'student_sft'
    shutil.copytree(source, target, copy_function=os.link)
    files = {}
    for path in sorted(target.iterdir()):
        if path.is_file():
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            with (source / path.name).open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest}
    teacher = json.loads((ARTIFACTS / 'evaluations/teacher.json').read_text())
    baseline = json.loads((ARTIFACTS / 'evaluations/baseline.json').read_text())
    selection = {'selected_run': best['run'], 'metric': 'dev Quality', 'quality': best['quality'],
                 'teacher_quality': teacher['quality'], 'baseline_quality': baseline['quality'],
                 'relative_quality': best['quality'] / teacher['quality'],
                 'target_reached': best['quality'] >= 0.9 * teacher['quality'],
                 'source_checkpoint': str(source), 'canonical_checkpoint': str(target), 'files': files}
    (ARTIFACTS / 'student_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({key: value for key, value in selection.items() if key != 'files'}), flush=True)


def train_student(name: str, smoke: bool = False) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    configs = json.loads(Path('configs.json').read_text())['distillation']
    config = configs[name]
    output = ARTIFACTS / 'training' / ('smoke' if smoke else name)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / 'summary.json'
    if history_path.exists():
        raise FileExistsError(history_path)
    set_seed(SEED)
    torch.set_float32_matmul_precision('highest')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, revision=BASE_REVISION, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
    model.config.use_cache = False
    answer_path = ARTIFACTS / 'teacher_answers/answers.jsonl'
    if smoke:
        train, _, intents, slots = load_data()
        prompt = system_prompt(intents, slots)
        rows = [dict(row, prompt_ids=encode_prompt(tokenizer, row['text'], prompt),
                     answer_ids=tokenizer.encode(row['gold'], add_special_tokens=False) + [tokenizer.eos_token_id])
                for row in train.select(range(128))]
        rows = sorted(rows, key=lambda row: len(row['prompt_ids']) + len(row['answer_ids']), reverse=True)[:16]
        answer_path = output / 'hardware_smoke_answers.jsonl'
        answer_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    dataset = TeacherAnswerDataset(answer_path, 16 if smoke else config['samples'])
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_parameters == total_parameters
    tracked_weight = model.model.layers[0].self_attn.q_proj.weight
    initial_weight = tracked_weight[0].detach().float().cpu().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], foreach=False)
    smoke_state = {}

    class CheckTraining(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if not smoke_state:
                smoke_state['gradient_dtypes'] = dict(Counter(str(p.grad.dtype) for p in model.parameters() if p.grad is not None))

        def on_optimizer_step(self, args, state, control, **kwargs):
            if 'optimizer_state_dtypes' not in smoke_state:
                smoke_state['optimizer_state_dtypes'] = dict(Counter(str(v.dtype) for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)))
                smoke_state['foreach'] = optimizer.defaults['foreach']
                smoke_state['parameter_dtypes'] = dict(Counter(str(p.dtype) for p in model.parameters()))
                smoke_state['optimizer_states'] = len(optimizer.state)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in (logs or {}).values()):
                raise ValueError('Nonfinite training metric')

    args = TrainingArguments(
        output_dir=str(output / 'trainer'), seed=SEED, data_seed=SEED, bf16=True,
        num_train_epochs=config['epochs'], max_steps=1 if smoke else -1,
        learning_rate=config['learning_rate'], per_device_train_batch_size=1,
        gradient_accumulation_steps=16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        lr_scheduler_type='cosine', warmup_ratio=0.0 if smoke else 0.05, max_grad_norm=1.0,
        logging_steps=1, save_strategy='no', report_to='none',
        remove_unused_columns=False, dataloader_num_workers=0,
    )
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=functools.partial(collate_teacher_answers, pad_id=pad),
                      optimizers=(optimizer, None), callbacks=[CheckTraining()])
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    result = trainer.train()
    assert smoke_state['optimizer_states'] > 0 and smoke_state['foreach'] is False, smoke_state
    delta = float((tracked_weight[0].detach().float().cpu() - initial_weight).abs().max())
    assert delta > 0, 'Веса не изменились'
    assert trainer.state.global_step == (1 if smoke else math.ceil(len(dataset) / 16) * config['epochs'])
    summary = dict(config, run=name, smoke=smoke, trainable_parameters=trainable_parameters,
                   total_parameters=total_parameters, seconds=time.monotonic() - start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                   planned_tokens=sum(len(r['prompt_ids']) + len(r['answer_ids']) for r in dataset.rows) * (1 if smoke else config['epochs']),
                   target_tokens=sum(len(r['answer_ids']) for r in dataset.rows) * (1 if smoke else config['epochs']),
                   optimizer_defaults=optimizer.defaults,
                   batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
                   gradient_checkpointing=args.gradient_checkpointing, warmup_ratio=args.warmup_ratio,
                   scheduler=args.lr_scheduler_type.value, max_grad_norm=args.max_grad_norm,
                   examples=len(dataset), filtered=0, steps=trainer.state.global_step,
                   tracked_weight_max_abs_delta=delta,
                   label_source='MASSIVE train gold, hardware smoke only' if smoke else 'Qwen3-32B greedy answers',
                   answers_sha256=hashlib.sha256(answer_path.read_bytes()).hexdigest(),
                   selected_ids_sha256=hashlib.sha256(json.dumps([row['id'] for row in dataset.rows]).encode()).hexdigest(),
                   sequence_tokens_min=min(len(row['prompt_ids']) + len(row['answer_ids']) for row in dataset.rows),
                   sequence_tokens_max=max(len(row['prompt_ids']) + len(row['answer_ids']) for row in dataset.rows),
                   prompt_tokens_max=max(len(row['prompt_ids']) for row in dataset.rows),
                   answer_tokens_max=max(len(row['answer_ids']) for row in dataset.rows),
                   train_metrics=result.metrics, optimizer=smoke_state, base_revision=BASE_REVISION)
    (output / 'history.json').write_text(json.dumps(trainer.state.log_history, indent=2) + '\n')
    if not smoke:
        checkpoint = ARTIFACTS / 'students' / name
        if checkpoint.exists() and any(checkpoint.iterdir()):
            raise FileExistsError(checkpoint)
        model.config.use_cache = True
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size='1GB')
        tokenizer.save_pretrained(checkpoint)
        summary['checkpoint'] = str(checkpoint)
    history_path.write_text(json.dumps(summary, indent=2) + '\n')
    del trainer, model, optimizer, tracked_weight
    clean_memory()
    if not smoke:
        evaluate_checkpoint(checkpoint, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['teacher', 'train', 'smoke', 'select'])
    parser.add_argument('--config', default='s128')
    args = parser.parse_args()
    if args.stage == 'teacher':
        evaluate_checkpoint(TEACHER_MODEL_ID, 'teacher', TEACHER_REVISION)
        collect_teacher_answers()
    elif args.stage == 'select':
        select_student()
    else:
        train_student(args.config, smoke=args.stage == 'smoke')
