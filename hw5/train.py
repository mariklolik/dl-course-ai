import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path


def training_config(configs: dict, name: str) -> dict:
    return configs['common'] | configs[name]


def main() -> None:
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from vllm import SamplingParams

    from rewards import ToolReward

    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['rl1', 'rl2', 'rl3'])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = training_config(json.loads(Path('configs.json').read_text()), args.name)
    if args.steps:
        config['max_steps'] = args.steps
        config['save_steps'] = args.steps
    output = args.output or Path('outputs') / args.name
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    set_seed(config['seed'])
    base_path = snapshot_download(config['model'], revision=config['revision'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'])
    dataset = load_dataset('json', data_files={'train': 'data/train.jsonl', 'validation': 'data/validation.jsonl'})
    model_config = AutoConfig.from_pretrained(config['model'], revision=config['revision'])
    eos_ids = model_config.get_text_config().eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    reward = ToolReward(tokenizer, config['reward'], eos_token_ids=eos_ids)
    exact_reward = ToolReward(tokenizer, 'exact', eos_token_ids=eos_ids)
    validation = dataset['validation']
    selections = []

    class Validate(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in (logs or {}).values()):
                raise RuntimeError('Nonfinite training metric')

        def on_save(self, training_args, state, control, **kwargs):
            trainer.vllm_generation.sync_weights()
            responses = trainer.vllm_generation.llm.generate(
                validation['prompt'],
                SamplingParams(temperature=0, max_tokens=config['max_completion_length'], skip_special_tokens=False),
                use_tqdm=False,
            )
            texts = [r.outputs[0].text for r in responses]
            scores = exact_reward(
                completions=texts, expected=validation['expected'], tools_json=validation['tools_json'],
                finish_reason=[r.outputs[0].finish_reason for r in responses],
            )
            records = [dict(id=row['id'], response=text, exact=score, finish_reason=response.outputs[0].finish_reason)
                       for row, text, score, response in zip(validation, texts, scores, responses, strict=True)]
            (output / f'validation-{state.global_step}.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records)
            )
            accuracy = sum(scores) / len(scores)
            selections.append({'step': state.global_step, 'accuracy': accuracy})
            (output / 'selection.json').write_text(json.dumps(selections, indent=2) + '\n')
            print(json.dumps({'validation_step': state.global_step, 'validation_exact': accuracy}), flush=True)

    training_args = GRPOConfig(
        output_dir=str(output / 'adapters'), seed=config['seed'], data_seed=config['seed'],
        max_steps=config['max_steps'], learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        num_generations=config['num_generations'], max_completion_length=config['max_completion_length'],
        temperature=config['temperature'], beta=config['beta'], epsilon=config['epsilon'],
        loss_type=config['loss_type'], scale_rewards=config['scale_rewards'],
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type='constant_with_warmup',
        logging_steps=1, save_steps=config['save_steps'], save_total_limit=2,
        report_to='none', remove_unused_columns=False,
        use_vllm=True, vllm_mode='colocate', vllm_gpu_memory_utilization=0.35,
        vllm_max_model_length=4096,
        model_init_kwargs={'revision': config['revision'], 'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'},
        mask_truncated_completions=True,
    )
    trainer = GRPOTrainer(
        model=base_path, args=training_args, processing_class=tokenizer,
        train_dataset=dataset['train'], reward_funcs=reward, callbacks=[Validate()],
        peft_config=LoraConfig(
            r=config['lora_rank'], lora_alpha=32, lora_dropout=0,
            target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)',
            task_type='CAUSAL_LM', bias='none',
        ),
    )
    start = time.monotonic()
    trainable, total = trainer.model.get_nb_trainable_parameters()
    result = trainer.train()
    trainer.save_state()
    best = max(selections, key=lambda row: (row['accuracy'], -row['step']))
    summary = {
        'selected_step': best['step'], 'validation_exact': best['accuracy'],
        'seconds': time.monotonic() - start, 'trainable_parameters': trainable,
        'total_parameters': total, 'metrics': result.metrics,
        'gpu': torch.cuda.get_device_name(),
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'versions': {p: version(p) for p in ('torch', 'trl', 'transformers', 'peft', 'vllm', 'datasets')},
    }
    (output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'training_history.json').write_text(json.dumps(trainer.state.log_history, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
