import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from data import (ARTIFACTS, BASE_MODEL_ID, BASE_REVISION, GENERATE_BATCH_SIZE,
                  MAX_NEW_TOKENS, SEED, encode_prompt, load_data, system_prompt)


def clean_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def trim_answer(token_ids: list[int], eos_token_id: int, pad_token_id: int) -> list[int]:
    if eos_token_id in token_ids:
        return token_ids[:token_ids.index(eos_token_id) + 1]
    while token_ids and token_ids[-1] == pad_token_id:
        token_ids = token_ids[:-1]
    return token_ids


def generate_answer_ids(model, tokenizer, prompts: list[list[int]]) -> list[list[int]]:
    import torch
    from transformers import GenerationConfig

    if not prompts or any(not ids for ids in prompts):
        raise ValueError('Нужен непустой батч непустых промптов')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(map(len, prompts))
    input_ids = torch.tensor([[pad] * (width - len(ids)) + ids for ids in prompts], device=model.device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts], device=model.device)
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=pad, use_cache=True)
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=mask, generation_config=config,
                                use_model_defaults=False, do_sample=False)
    return [trim_answer(ids, config.eos_token_id, pad) for ids in output[:, width:].tolist()]


def normalize_value(value: str) -> str:
    return ' '.join(value.casefold().split())


def parse_answer(text: str, intents, slot_types) -> dict | None:
    try:
        obj = json.loads(text.strip())
        if not isinstance(obj, dict) or set(obj) != {'intent', 'slots'}:
            return None
        if not isinstance(obj['intent'], str) or obj['intent'] not in intents:
            return None
        if not isinstance(obj['slots'], list):
            return None
        for slot in obj['slots']:
            if not isinstance(slot, list) or len(slot) != 2:
                return None
            kind, value = slot
            if (not isinstance(kind, str) or kind not in slot_types
                    or not isinstance(value, str) or not value.strip()):
                return None
        return obj
    except (ValueError, TypeError):
        return None


def slot_counter(obj: dict) -> Counter:
    return Counter((kind, normalize_value(value)) for kind, value in obj['slots'])


def compute_metrics(predictions, golds, intents, slot_types) -> dict:
    if len(predictions) != len(golds) or not golds:
        raise ValueError('Нужны непустые списки одинаковой длины')
    correct_intent = exact = valid = tp = fp = fn = 0
    for text, gold_text in zip(predictions, golds):
        gold = parse_answer(gold_text, intents, slot_types)
        if gold is None:
            raise ValueError('Невалидная gold-разметка')
        pred = parse_answer(text, intents, slot_types)
        gold_slots = slot_counter(gold)
        pred_slots = Counter() if pred is None else slot_counter(pred)
        tp += sum((pred_slots & gold_slots).values())
        fp += sum((pred_slots - gold_slots).values())
        fn += sum((gold_slots - pred_slots).values())
        if pred is not None:
            valid += 1
            intent_ok = pred['intent'] == gold['intent']
            correct_intent += int(intent_ok)
            exact += int(intent_ok and pred_slots == gold_slots)
    n = len(golds)
    denominator = 2 * tp + fp + fn
    slot_f1 = 2 * tp / denominator if denominator else float(valid == n)
    intent_acc = correct_intent / n
    return {'quality': 0.5 * (intent_acc + slot_f1), 'intent_accuracy': intent_acc,
            'slot_micro_f1': slot_f1, 'frame_exact_match': exact / n,
            'valid_json_rate': valid / n, 'n': n}


def evaluate(model, tokenizer, ds, intents, slot_types) -> tuple[dict, list[str]]:
    model.eval()
    prompt = system_prompt(intents, slot_types)
    predictions = []
    for start in range(0, len(ds), GENERATE_BATCH_SIZE):
        batch = [encode_prompt(tokenizer, row['text'], prompt)
                 for row in ds.select(range(start, min(start + GENERATE_BATCH_SIZE, len(ds))))]
        answers = generate_answer_ids(model, tokenizer, batch)
        predictions.extend(tokenizer.decode(ids, skip_special_tokens=True) for ids in answers)
        print(json.dumps({'evaluated': len(predictions), 'total': len(ds)}), flush=True)
    return compute_metrics(predictions, ds['gold'], intents, slot_types), predictions


def load_model(model_path, revision=None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig, set_seed

    set_seed(SEED)
    torch.set_float32_matmul_precision('highest')
    config = AutoConfig.from_pretrained(str(model_path), revision=revision)
    quantization = getattr(config, 'quantization_config', None)
    load_int8 = bool(quantization and quantization.get('format') == 'int-quantized')
    kwargs = {'quantization_config': CompressedTensorsConfig(run_compressed=load_int8)} if quantization else {}
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), revision=revision, torch_dtype=torch.bfloat16,
        device_map='cuda:0', attn_implementation='sdpa', **kwargs,
    ).eval()
    if load_int8:
        from compressed_tensors.compressors import ModelCompressor

        ModelCompressor.from_pretrained(str(model_path)).decompress_model(model)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), revision=revision)
    return model, tokenizer


def quantization_state(model) -> dict:
    state = {'weight_modules': 0, 'weight_scale_modules': 0, 'input_modules': 0, 'wrapped_input_modules': 0,
             'enabled_input_modules': 0, 'input_bits': {}, 'input_dynamic': {}, 'input_status': {},
             'static_input_modules': 0, 'static_input_scale_modules': 0, 'input_scale_modules': 0,
             'weight_dtypes': {}}
    for module in model.modules():
        scheme = getattr(module, 'quantization_scheme', None)
        state['weight_modules'] += int(scheme is not None and scheme.weights is not None)
        state['weight_scale_modules'] += int(scheme is not None and scheme.weights is not None and hasattr(module, 'weight_scale'))
        if scheme is not None and scheme.weights is not None:
            dtype = str(module.weight.dtype)
            state['weight_dtypes'][dtype] = state['weight_dtypes'].get(dtype, 0) + 1
        inputs = None if scheme is None else scheme.input_activations
        if inputs is not None:
            state['input_modules'] += 1
            forward = getattr(module.forward, '__func__', module.forward)
            code = getattr(forward, '__code__', None)
            state['wrapped_input_modules'] += int(code is not None and code.co_name == 'wrapped_forward')
            state['enabled_input_modules'] += int(getattr(module, 'quantization_enabled', True))
            state['input_scale_modules'] += int(hasattr(module, 'input_scale'))
            state['static_input_modules'] += int(not inputs.dynamic)
            state['static_input_scale_modules'] += int(not inputs.dynamic and hasattr(module, 'input_scale'))
            status = getattr(module, 'quantization_status', None)
            for name, value in [('input_bits', str(inputs.num_bits)), ('input_dynamic', str(inputs.dynamic)),
                                ('input_status', str(getattr(status, 'value', status)))]:
                state[name][value] = state[name].get(value, 0) + 1
    return state


def evaluate_checkpoint(model_path, run_name: str, revision=None) -> tuple[dict, list[str]]:
    import torch

    _, dev, intents, slots = load_data()
    start = time.monotonic()
    model, tokenizer = load_model(model_path, revision)
    try:
        quant = quantization_state(model)
        initial_compressed = getattr(getattr(model, 'hf_quantizer', None), 'run_compressed', None)
        if run_name.startswith('sq_'):
            assert quant['input_modules'] > 0 and quant['input_bits'] == {'8': quant['input_modules']}, quant
            assert quant['enabled_input_modules'] == quant['wrapped_input_modules'] == quant['input_modules'], quant
            assert quant['input_status'] == {'frozen': quant['input_modules']}, quant
            assert quant['static_input_scale_modules'] == quant['static_input_modules'], quant
            assert quant['weight_dtypes'] == {'torch.bfloat16': quant['weight_modules']}, quant
        metrics, predictions = evaluate(model, tokenizer, dev, intents, slots)
        records = [dict(row, prediction=prediction) for row, prediction in zip(dev, predictions, strict=True)]
        folder = ARTIFACTS / 'evaluations'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{run_name}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
        metrics |= {'run': run_name, 'model': str(model_path), 'revision': revision,
                    'seconds': time.monotonic() - start, 'seed': SEED, 'batch_size': GENERATE_BATCH_SIZE,
                    'max_new_tokens': MAX_NEW_TOKENS, 'thinking': False, 'decoding': 'greedy',
                    'use_model_defaults': False, 'do_sample': False,
                    'predictions_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'dev_sha256': json.loads((ARTIFACTS / 'data/manifest.json').read_text())['sha256']['dev'],
                    'quantization_state': quant, 'run_compressed': False,
                    'initial_load_run_compressed': initial_compressed,
                    'explicit_decompress_model': initial_compressed is True,
                    'effective_eval_layout': 'bf16_weights_with_fake_quant' if quant['weight_modules'] else 'bf16_dense',
                    'gpu': torch.cuda.get_device_name(), 'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
                    'versions': {name: version(name) for name in ('torch', 'transformers', 'compressed-tensors')}}
        (folder / f'{run_name}.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        return metrics, predictions
    finally:
        del model
        clean_memory()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=BASE_MODEL_ID)
    parser.add_argument('--revision', default=BASE_REVISION)
    parser.add_argument('--run', default='baseline')
    args = parser.parse_args()
    evaluate_checkpoint(args.model, args.run, args.revision)
