import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from evaluate import compute_metrics, parse_answer, slot_counter


METRICS = ('quality', 'intent_accuracy', 'slot_micro_f1', 'frame_exact_match', 'valid_json_rate', 'n')


def read_runs(artifacts: Path) -> dict:
    configs = json.loads(Path(__file__).with_name('configs.json').read_text())
    manifest = json.loads((artifacts / 'data/manifest.json').read_text())
    dev_path = artifacts / 'data/dev.jsonl'
    digest = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    assert digest == manifest['sha256']['dev'], 'Changed dev data'
    dev = [json.loads(line) for line in dev_path.read_text().splitlines()]
    assert len({row['id'] for row in dev}) == len(dev), 'Duplicate dev IDs'
    runs = {}
    for path in sorted((artifacts / 'evaluations').glob('*.json')):
        saved = json.loads(path.read_text())
        assert saved['run'] == path.stem, path
        predictions_path = path.with_suffix('.jsonl')
        assert hashlib.sha256(predictions_path.read_bytes()).hexdigest() == saved['predictions_sha256'], path
        predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
        assert [[row[key] for key in ('id', 'text', 'gold')] for row in predictions] == [[row[key] for key in ('id', 'text', 'gold')] for row in dev], path
        assert saved['dev_sha256'] == digest, path
        measured = compute_metrics([row['prediction'] for row in predictions], [row['gold'] for row in dev], manifest['intents'], manifest['slot_types'])
        assert all(math.isclose(measured[key], saved[key], abs_tol=1e-12) for key in METRICS), path
        row = {'run': path.stem, 'method': 'dense', **measured}
        quant_path = artifacts / 'quantization' / path.name
        training_path = artifacts / 'training' / path.stem / 'summary.json'
        if path.stem in configs['quantization']:
            assert quant_path.exists(), f'Missing quantization metadata: {path.stem}'
        if path.stem in configs['distillation']:
            assert training_path.exists(), f'Missing training metadata: {path.stem}'
        if quant_path.exists():
            quant = json.loads(quant_path.read_text())
            assert quant['status'] == 'complete', quant_path
            assert quant['config'] == configs['quantization'][path.stem], quant_path
            assert all(measured[key] == quant['metrics'][key] for key in METRICS), quant_path
            if quant['config']['method'] == 'smoothquant':
                state = saved['quantization_state']
                count = state['input_modules']
                assert count > 0 and state['input_bits'] == {'8': count} and state['input_dynamic'] == {'True': count}, quant_path
                assert state['input_status'] == {'frozen': count} and state['wrapped_input_modules'] == state['enabled_input_modules'] == count, quant_path
                assert state['weight_dtypes'] == {'torch.bfloat16': state['weight_modules']}, quant_path
                assert saved['initial_load_run_compressed'] is True and saved['explicit_decompress_model'] is True, quant_path
            row |= {'method': quant['config']['method'], 'calibration': quant['config']['calibration'],
                    'checkpoint_bytes': quant['checkpoint_bytes'], 'dense_checkpoint_bytes': quant['dense_checkpoint_bytes'],
                    'disk_ratio': quant['disk_ratio'], 'samples': quant['config']['samples'],
                    'group_size': quant['config'].get('group_size'), 'alpha': quant['config'].get('alpha'),
                    'last_block_bf16': quant['config'].get('last_block_bf16', False),
                    'calibration_tokens': quant['calibration']['total_tokens'],
                    'quantization_seconds': quant['quantization_seconds']}
            assert row['checkpoint_bytes'] == sum(v['bytes'] for v in quant['checkpoint_files'].values()), quant_path
            assert math.isclose(row['disk_ratio'], row['dense_checkpoint_bytes'] / row['checkpoint_bytes']), quant_path
        if training_path.exists():
            training = json.loads(training_path.read_text())
            assert not training['smoke'] and training['trainable_parameters'] == training['total_parameters'], training_path
            assert training['optimizer']['foreach'] is False and training['optimizer']['optimizer_states'] > 0, training_path
            teacher_path = artifacts / 'teacher_answers/answers.jsonl'
            assert hashlib.sha256(teacher_path.read_bytes()).hexdigest() == training['answers_sha256'], training_path
            selected = [json.loads(line) for line in teacher_path.read_text().splitlines()][:training['samples']]
            assert hashlib.sha256(json.dumps([item['id'] for item in selected]).encode()).hexdigest() == training['selected_ids_sha256'], training_path
            assert len(selected) == training['examples'] == training['samples'] and training['filtered'] == 0, training_path
            assert training['planned_tokens'] == sum(len(item['prompt_ids']) + len(item['answer_ids']) for item in selected) * training['epochs'], training_path
            assert training['target_tokens'] == sum(len(item['answer_ids']) for item in selected) * training['epochs'], training_path
            assert training['tracked_weight_max_abs_delta'] > 0, training_path
            row |= {'method': 'full_sft', 'samples': training['samples'], 'epochs': training['epochs'],
                    'learning_rate': training['learning_rate'], 'train_seconds': training['seconds'],
                    'train_loss': training['train_metrics']['train_loss'], 'optimizer_steps': training['steps']}
        runs[path.stem] = row
    if not runs:
        raise ValueError('No completed evaluations')
    return runs


def plots(output: Path, artifacts: Path, runs: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    configs = json.loads(Path(__file__).with_name('configs.json').read_text())
    quant = [runs[name] for name in configs['quantization']]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    positions = list(range(len(quant)))
    axes[0].barh(positions, [100 * row['quality'] for row in quant], color='#1e5a78')
    axes[0].set(yticks=positions, yticklabels=[row['run'] for row in quant], xlabel='Quality, %', title='Квантизация: качество на dev')
    axes[0].axvline(100 * runs['baseline']['quality'], color='#d77a3f', linestyle='--', label='BF16 baseline')
    axes[0].legend(frameon=False)
    axes[0].invert_yaxis()
    axes[1].barh(positions, [row['checkpoint_bytes'] / 1e9 for row in quant], color='#8eabb8')
    axes[1].set(yticks=positions, yticklabels=[row['run'] for row in quant], xlabel='Сохранённые safetensors, GB', title='Размер сжатых весов на диске')
    axes[1].axvline(quant[0]['dense_checkpoint_bytes'] / 1e9, color='#d77a3f', linestyle='--', label='Исходные BF16 веса')
    axes[1].legend(frameon=False)
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=0.2)
    fig.savefig(output / 'quantization.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    students = sorted([row for row in runs.values() if row['method'] == 'full_sft'], key=lambda row: (row['samples'], row['epochs']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    comparison = [runs['baseline'], runs['teacher'], *students]
    axes[0].bar([row['run'] for row in comparison], [100 * row['quality'] for row in comparison], color=['#8eabb8', '#d77a3f', *['#1e5a78'] * len(students)])
    axes[0].axhline(90 * runs['teacher']['quality'], color='#687780', linestyle='--', label='90% Quality учителя')
    axes[0].set(ylabel='Quality, %', title='Hard-label дистилляция')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].legend(frameon=False)
    for row in students:
        history = json.loads((artifacts / 'training' / row['run'] / 'history.json').read_text())
        points = [(item['step'], item['loss']) for item in history if 'loss' in item]
        assert points and all(math.isfinite(y) for _, y in points)
        axes[1].plot(*zip(*points), label=row['run'])
    axes[1].set(xlabel='Optimizer step', ylabel='Train loss', title='Обучение на ответах учителя')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis='y', alpha=0.2)
    fig.savefig(output / 'distillation.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def error_analysis(artifacts: Path, runs: dict, best: str) -> dict:
    info = json.loads((artifacts / 'data/manifest.json').read_text())
    records = {name: list(map(json.loads, (artifacts / 'evaluations' / f'{name}.jsonl').read_text().splitlines())) for name in runs}
    counts, parsed = {}, {}
    for name, rows in records.items():
        kinds, parsed[name] = [], []
        for row in rows:
            pred = parse_answer(row['prediction'], info['intents'], info['slot_types'])
            gold = parse_answer(row['gold'], info['intents'], info['slot_types'])
            parsed[name].append(pred)
            if pred is None:
                try:
                    json.loads(row['prediction'])
                    kind = 'invalid_schema'
                except json.JSONDecodeError as error:
                    extra_brace = error.msg == 'Extra data' and row['prediction'].rstrip().endswith('}}') and error.pos == len(row['prediction'].rstrip()) - 1
                    kind = 'extra_closing_brace' if extra_brace else 'invalid_json'
            elif pred['intent'] != gold['intent']:
                kind = 'wrong_intent'
            else:
                kind = 'exact' if slot_counter(pred) == slot_counter(gold) else 'wrong_slots'
            kinds.append(kind)
        counts[name] = dict(Counter(kinds))
        assert sum(counts[name].values()) == runs[name]['n']
    examples = []
    for kind in ('format_fixed', 'slot_error'):
        for i, row in enumerate(records[best]):
            gold = parse_answer(row['gold'], info['intents'], info['slot_types'])
            pred = parsed[best][i]
            exact = pred is not None and pred['intent'] == gold['intent'] and slot_counter(pred) == slot_counter(gold)
            eligible = (parsed['baseline'][i] is None and exact) if kind == 'format_fixed' else (pred is not None and pred['intent'] == gold['intent'] and not exact)
            if eligible:
                examples.append({key: row[key] for key in ('id', 'text', 'gold')} | {'kind': kind, 'predictions': {name: records[name][i]['prediction'] for name in ('baseline', 'teacher', best)}})
                break
    return {'counts': counts, 'examples': examples}


def main() -> None:
    artifacts = Path('artifacts_compression_hw')
    runs = read_runs(artifacts)
    configs = json.loads(Path('configs.json').read_text())
    required = {'baseline', 'teacher', 's128', 's1024', 's1024_e3', *configs['quantization']}
    assert required <= runs.keys(), f'Missing evaluations: {required - runs.keys()}'
    data = json.loads((artifacts / 'data/manifest.json').read_text())
    train_path = artifacts / 'data/train.jsonl'
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == data['sha256']['train']
    train = {item['id']: item for item in map(json.loads, train_path.read_text().splitlines())}
    assert len(train) == data['train_size']
    dev_ids = {item['id'] for item in map(json.loads, (artifacts / 'data/dev.jsonl').read_text().splitlines())}
    teacher_path = artifacts / 'teacher_answers/answers.jsonl'
    teacher_info = json.loads(teacher_path.with_name('manifest.json').read_text())
    assert hashlib.sha256(teacher_path.read_bytes()).hexdigest() == teacher_info['sha256']
    answers = list(map(json.loads, teacher_path.read_text().splitlines()))
    assert len(answers) == teacher_info['n'] == 1024 and len({item['id'] for item in answers}) == 1024
    assert not dev_ids.intersection(train)
    assert all(all(item[key] == train[item['id']][key] for key in ('text', 'gold')) for item in answers)
    calibration = {name: json.loads((artifacts / 'quantization' / f'{name}.json').read_text())['calibration'] for name in configs['quantization']}
    reference = calibration['gptq_g128']['samples']
    assert calibration['gptq_n64']['samples'] == reference[:64]
    assert all(value['samples'] == reference for name, value in calibration.items() if name not in ('awq_wikitext', 'gptq_n64'))
    output = Path('outputs')
    output.mkdir(exist_ok=True)
    rows = list(runs.values())
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    students = [row for row in rows if row['method'] == 'full_sft']
    best = max(students, key=lambda row: row['quality'])
    verification = {'runs': len(rows), 'verified_dev_predictions': sum(row['n'] for row in rows),
                    'best_student': best['run'], 'student_teacher_ratio': best['quality'] / runs['teacher']['quality'],
                    'student_above_baseline': best['quality'] > runs['baseline']['quality'],
                    'target_reached': best['quality'] >= 0.9 * runs['teacher']['quality']}
    (output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    (output / 'error_analysis.json').write_text(json.dumps(error_analysis(artifacts, runs, best['run']), ensure_ascii=False, indent=2) + '\n')
    plots(output, artifacts, runs)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
