import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path


WIKITEXT_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def recipe_spec(config: dict, last_block: int) -> dict:
    weights = {'num_bits': 4, 'type': 'int', 'symmetric': True, 'strategy': 'group', 'group_size': config.get('group_size', 128), 'dynamic': False}
    scheme = {'targets': ['Linear'], 'weights': weights}
    ignore = ['lm_head']
    if config.get('last_block_bf16'):
        ignore.append(rf're:model\.layers\.{last_block}\..*')
    modifiers = {}
    if config['method'] == 'smoothquant':
        modifiers['SmoothQuantModifier'] = {'smoothing_strength': config['alpha'], 'mappings': [
            [['re:.*q_proj', 're:.*k_proj', 're:.*v_proj'], 're:.*input_layernorm'],
            [['re:.*gate_proj', 're:.*up_proj'], 're:.*post_attention_layernorm'],
        ]}
        scheme['weights'] = {'num_bits': 8, 'type': 'int', 'symmetric': True, 'strategy': 'channel', 'dynamic': False}
        scheme['input_activations'] = {'num_bits': 8, 'type': 'int', 'symmetric': True, 'strategy': 'token', 'dynamic': True, 'observer': None}
    name = 'AWQModifier' if config['method'] == 'awq' else 'GPTQModifier'
    modifiers[name] = {'config_groups': {'group_0': scheme}, 'ignore': ignore}
    return {'quant_stage': {'quant_modifiers': modifiers}}


def calibration_set(config: dict, tokenizer, train, prompt: str):
    from datasets import Dataset, load_dataset
    from data import MAX_PROMPT_TOKENS, SEED, encode_prompt

    source = 'massive'
    if config['calibration'] == 'wikitext':
        train = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', revision=WIKITEXT_REVISION)
        train = train.add_column('source_id', list(range(len(train))))
        source = 'Salesforce/wikitext:wikitext-2-raw-v1:train'
    ordered = train.shuffle(seed=SEED)
    samples, manifest, skipped = [], [], 0
    for index, row in enumerate(ordered):
        text = row['text']
        if not text.strip():
            skipped += 1
            continue
        try:
            ids = encode_prompt(tokenizer, text, prompt)
        except ValueError:
            if config['calibration'] != 'wikitext':
                raise
            skipped += 1
            continue
        assert 0 < len(ids) <= MAX_PROMPT_TOKENS
        samples.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
        manifest.append({'id': str(row.get('source_id', row.get('id', index))), 'tokens': len(ids), 'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        if len(samples) == config['samples']:
            break
    if len(samples) != config['samples']:
        raise ValueError('Недостаточно калибровочных примеров с полным prompt <=2048')
    lengths = [r['tokens'] for r in manifest]
    metadata = {'source': source, 'revision': WIKITEXT_REVISION if config['calibration'] == 'wikitext' else None, 'seed': SEED, 'samples': manifest, 'skipped': skipped, 'total_tokens': sum(lengths), 'min_tokens': min(lengths), 'max_tokens': max(lengths), 'mean_tokens': sum(lengths) / len(lengths), 'sha256': hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()}
    return Dataset.from_list(samples), metadata



def recovery_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    for key in ('quantization_seconds', 'save_seconds', 'peak_quantization_allocated_bytes', 'peak_quantization_reserved_bytes'):
        metadata.setdefault(key, None)
    metadata['recovery_note'] = 'Evaluation resumed from saved checkpoint; calibration/save were not repeated. Missing pre-failure timings and CUDA peaks are unknown.'
    return metadata


def finalize_checkpoint(output: Path, run_name: str, metadata: dict) -> None:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from data import ARTIFACTS
    from evaluate import evaluate_checkpoint

    config = metadata['config']
    saved = json.loads((output / 'config.json').read_text())
    last_block = saved['num_hidden_layers'] - 1
    quantization = saved['quantization_config']
    metadata['saved_quantization_config'] = quantization
    checkpoint_files, tensor_dtypes = {}, {}
    for path in sorted(output.glob('*.safetensors')):
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        checkpoint_files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest}
        with safe_open(path, framework='pt', device='cpu') as handle:
            tensor_dtypes.update({key: handle.get_slice(key).get_dtype() for key in handle.keys()})
    assert checkpoint_files
    assert tensor_dtypes['lm_head.weight'] == 'BF16'
    assert tensor_dtypes['model.embed_tokens.weight'] == 'BF16'
    assert all(dtype == 'BF16' for key, dtype in tensor_dtypes.items() if key.endswith('norm.weight'))
    if config.get('last_block_bf16'):
        keys = [key for key in tensor_dtypes if key.startswith(f'model.layers.{last_block}.')]
        assert keys and all(tensor_dtypes[key] == 'BF16' for key in keys)
        assert not any(key.endswith('weight_packed') for key in keys)
    metadata['checkpoint_files'] = checkpoint_files
    metadata['checkpoint_bytes'] = sum(item['bytes'] for item in checkpoint_files.values())
    metadata['tensor_dtypes'] = dict(Counter(tensor_dtypes.values()))
    snapshot = Path(snapshot_download(metadata['base_model'], revision=metadata['base_revision'], local_files_only=True))
    metadata['dense_checkpoint_bytes'] = sum(p.stat().st_size for p in snapshot.glob('*.safetensors'))
    assert metadata['dense_checkpoint_bytes'] > 0
    metadata['disk_ratio'] = metadata['dense_checkpoint_bytes'] / metadata['checkpoint_bytes']
    metrics, _ = evaluate_checkpoint(output, run_name)
    metadata['metrics'] = metrics
    metadata['status'] = 'complete'
    (ARTIFACTS / 'quantization' / f'{run_name}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({'run': run_name, 'status': 'complete', 'checkpoint_bytes': metadata['checkpoint_bytes'], 'disk_ratio': metadata['disk_ratio'], 'metrics': metrics}), flush=True)


def main() -> None:
    configs = json.loads(Path(__file__).with_name('configs.json').read_text())['quantization']
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=list(configs))
    parser.add_argument('--resume-evaluation', action='store_true')
    args = parser.parse_args()
    config = configs[args.config]

    import torch
    from huggingface_hub import snapshot_download
    from llmcompressor import oneshot
    from transformers import set_seed
    from data import ARTIFACTS, BASE_MODEL_ID, BASE_REVISION, DATASET_REVISION, MAX_PROMPT_TOKENS, SEED, load_data, system_prompt
    from evaluate import clean_memory, load_model

    output = ARTIFACTS / 'quantized' / args.config
    metadata_dir = ARTIFACTS / 'quantization'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_evaluation:
        if not output.is_dir():
            raise FileNotFoundError(output)
        if (metadata_dir / f'{args.config}.json').exists():
            raise FileExistsError('Checkpoint уже оценён')
        metadata = recovery_metadata(metadata_dir / f'{args.config}.calibration.json')
        assert metadata['config'] == config
        finalize_checkpoint(output, args.config, metadata)
        return
    output.mkdir(parents=True, exist_ok=False)
    set_seed(SEED)
    train, _, intents, slots = load_data()
    snapshot = Path(snapshot_download(BASE_MODEL_ID, revision=BASE_REVISION, local_files_only=True))
    model, tokenizer = load_model(snapshot)
    last_block = model.config.num_hidden_layers - 1
    calibration, calibration_metadata = calibration_set(config, tokenizer, train, system_prompt(intents, slots))
    calibration_metadata['massive_revision'] = DATASET_REVISION
    recipe = recipe_spec(config, last_block)
    metadata = {'config': config, 'base_model': BASE_MODEL_ID, 'base_revision': BASE_REVISION, 'calibration': calibration_metadata, 'recipe': recipe, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu': torch.cuda.get_device_name(0), 'parameter_count': model.num_parameters()}
    (metadata_dir / f'{args.config}.calibration.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    oneshot(model=model, tokenizer=tokenizer, dataset=calibration, recipe=json.dumps(recipe), max_seq_length=MAX_PROMPT_TOKENS, num_calibration_samples=len(calibration), shuffle_calibration_samples=False, pad_to_max_length=False, log_dir=str(ARTIFACTS / 'logs' / args.config))
    torch.cuda.synchronize()
    metadata['quantization_seconds'] = time.perf_counter() - started
    metadata['peak_quantization_allocated_bytes'] = torch.cuda.max_memory_allocated()
    metadata['peak_quantization_reserved_bytes'] = torch.cuda.max_memory_reserved()
    started = time.perf_counter()
    model.save_pretrained(output, save_compressed=True, safe_serialization=True, max_shard_size='1GB')
    tokenizer.save_pretrained(output)
    metadata['save_seconds'] = time.perf_counter() - started
    del model, tokenizer, calibration
    clean_memory()
    finalize_checkpoint(output, args.config, metadata)


if __name__ == '__main__':
    main()
