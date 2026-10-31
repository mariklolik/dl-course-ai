import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

NAMES = ('base', 'rl1', 'rl2', 'rl3', 'rl4')
GROUPS = {'simple': ['simple'], 'multiple': ['multiple', 'live_multiple'], 'parallel': ['parallel'], 'parallel_multiple': ['parallel_multiple'], 'irrelevance': ['irrelevance', 'live_irrelevance']}
BACKGROUND, INK, TEAL, MUTED = '#f7f5f0', '#193139', '#007c78', '#687a7c'
COLORS = {'base': '#a5bfba', 'rl1': '#193139', 'rl2': '#007c78', 'rl3': '#a8553a', 'rl4': '#6a5e91'}


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate(predictions):
    categories = defaultdict(list)
    for row in predictions:
        categories[row['category']].append(bool(row['check']['valid']))
    details = {key: {'correct': sum(values), 'total': len(values), 'accuracy': mean(values)} for key, values in categories.items()}
    groups = {key: mean(details[category]['accuracy'] for category in members) for key, members in GROUPS.items()}
    return {'overall': mean(groups.values()), 'groups': groups, 'categories': details, 'pooled': mean(bool(row['check']['valid']) for row in predictions)}


def error_type(row, source):
    if row['check']['valid']:
        return 'correct'
    if 'irrelevance' in row['category'] and row['calls']:
        return 'false_call_irrelevance'
    if row['finish_reason'] == 'length':
        return 'truncated'
    if not row['response'].strip():
        return 'empty'
    if 'irrelevance' not in row['category'] and not row['calls']:
        return 'missed_required_call'
    definitions = {tool['name']: tool for tool in source['function']}
    expected_names = {name for call in source['ground_truth'] for name in call}
    for call in row['calls']:
        for name, arguments in call.items():
            if name not in expected_names:
                return 'wrong_function'
            required = definitions.get(name, {}).get('parameters', {}).get('required', [])
            if isinstance(arguments, dict) and set(required) - set(arguments):
                return 'missed_required_argument'
    if len(row['calls']) != len(source['ground_truth']):
        return 'wrong_call_count'
    return 'argument_or_format'


def read_runs(output, benchmark):
    source = {row['id']: row for row in lines(benchmark)}
    assert len(source) == 350, 'Expected original 350 BFCL rows'
    digest = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    runs, errors, csv_rows = {}, {}, []
    for name in NAMES:
        root = output / name
        predictions, metrics = lines(root / 'predictions.jsonl'), load(root / 'metrics.json')
        assert len(predictions) == 350 and {row['id'] for row in predictions} == set(source)
        assert metrics['benchmark_sha256'] == digest, f'{name}: benchmark hash'
        assert all(row['category'] == source[row['id']]['category'] for row in predictions)
        measured = aggregate(predictions)
        assert math.isclose(measured['overall'], metrics['overall'], abs_tol=1e-12)
        for category, values in measured['categories'].items():
            assert all(math.isclose(values[key], metrics['categories'][category][key], abs_tol=1e-12) for key in values)
            csv_rows.append({'model': name, 'metric': category, **values, 'aggregation': 'category'})
        for group, value in measured['groups'].items():
            assert math.isclose(value, metrics['groups'][group], abs_tol=1e-12)
            csv_rows.append({'model': name, 'metric': group, 'accuracy': value, 'aggregation': 'group_macro'})
        csv_rows.append({'model': name, 'metric': 'overall', 'accuracy': measured['overall'], 'aggregation': 'five_group_macro'})
        counts, matrix, records = Counter(), Counter(), []
        for row in predictions:
            original = source[row['id']]
            kind = error_type(row, original)
            counts[kind] += 1
            matrix[f"{len(original['ground_truth'])}->{len(row['calls'])}"] += 1
            if kind != 'correct':
                records.append(row | {'error_type': kind, 'question': original['question'], 'ground_truth': original['ground_truth'], 'function': original['function']})
        errors[name] = {'counts': dict(counts), 'expected_to_actual_call_counts': dict(matrix), 'errors': records}
        run = {'metrics': measured, 'receipt': metrics, 'predictions': predictions}
        for filename in ('config', 'training_summary', 'selection', 'training_history', 'export_verification'):
            run[filename] = load(root / f'{filename}.json') if (root / f'{filename}.json').exists() else None
        if name != 'base':
            assert all(run[key] is not None for key in ('config', 'training_summary', 'selection', 'training_history', 'export_verification'))
            selected = max(run['selection'], key=lambda row: (row['accuracy'], -row['step']))
            assert run['training_summary']['selected_step'] == selected['step'] == run['export_verification']['selected_step']
            assert math.isclose(run['training_summary']['validation_exact'], selected['accuracy'], abs_tol=1e-12)
            assert run['export_verification']['merged'] is True
            assert any(norm > 0 for norm in run['export_verification']['lora_b_norms'].values())
            run['bfcl_steps'] = []
            for checkpoint in run['selection']:
                step = checkpoint['step']
                step_root = root if step == selected['step'] else root / f'step-{step}'
                step_predictions, step_receipt = lines(step_root / 'predictions.jsonl'), load(step_root / 'metrics.json')
                assert len(step_predictions) == 350 and {r['id'] for r in step_predictions} == set(source)
                assert step_receipt['benchmark_sha256'] == digest
                assert all(r['category'] == source[r['id']]['category'] for r in step_predictions)
                step_export = load(root / ('export_verification.json' if step == selected['step'] else f'export_verification-step{step}.json'))
                assert step_export['exported_step'] == step
                scores = aggregate(step_predictions)
                assert all(scores[key] == step_receipt[key] for key in ('overall', 'groups', 'categories'))
                run['bfcl_steps'].append({'step': step, 'overall': scores['overall'], 'irrelevance': scores['groups']['irrelevance'], 'validation_exact': checkpoint['accuracy']})
        runs[name] = run
    controls = [{key: value for key, value in runs[name]['config'].items() if key not in ('reward', 'loss_type', 'scale_rewards')} for name in NAMES[1:4]]
    assert controls[0] == controls[1] == controls[2], 'Uncontrolled configuration difference'
    with (output / 'results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 'metric', 'accuracy', 'correct', 'total', 'aggregation'])
        writer.writeheader()
        writer.writerows(csv_rows)
    (output / 'error_analysis.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2) + '\n')
    (output / 'bfcl_checkpoints.json').write_text(json.dumps({name: runs[name]['bfcl_steps'] for name in NAMES[1:]}, indent=2) + '\n')
    return runs, errors


def plots(output, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'text.color': INK, 'axes.labelcolor': INK, 'figure.facecolor': BACKGROUND, 'axes.facecolor': BACKGROUND, 'savefig.facecolor': BACKGROUND, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout='constrained')
    for name in NAMES[1:]:
        history = runs[name]['training_history']
        for axis, keys in zip(axes[:2], [('loss',), ('reward', 'rewards/tool_shaped/mean', 'rewards/tool_exact/mean')]):
            key = next((key for key in keys if any(key in row for row in history)), None)
            points = [(row['step'], row[key]) for row in history if key and key in row and 'step' in row]
            if points:
                assert all(math.isfinite(y) for _, y in points)
                axis.plot(*zip(*points), color=COLORS[name], linewidth=1.2, label=name)
        points = [(row['step'], row['accuracy']) for row in runs[name]['selection']]
        axes[2].plot(*zip(*points), marker='o', color=COLORS[name], label=name)
    for axis, title in zip(axes, ['Train loss', 'Train reward: разные функции', 'ToolACE validation exact']):
        axis.set(title=title, xlabel='Optimizer step')
        axis.grid(axis='y', alpha=.2)
        if axis.lines:
            axis.legend(frameon=False)
        else:
            axis.text(.5, .5, 'Нет сохранённой метрики', transform=axis.transAxes, ha='center')
    figure.savefig(output / 'loss_reward.png', dpi=220, bbox_inches='tight')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 4), layout='constrained')
    metrics = ['overall', *GROUPS]
    for index, name in enumerate(NAMES):
        values = [100 * (runs[name]['metrics']['overall'] if key == 'overall' else runs[name]['metrics']['groups'][key]) for key in metrics]
        axis.bar([i + (index - (len(NAMES) - 1) / 2) * .19 for i in range(len(metrics))], values, width=.18, color=COLORS[name], label=name)
    axis.set(xticks=range(len(metrics)), xticklabels=metrics, ylabel='Точность, %', ylim=(0, 105), title='Оригинальный BFCL-срез: 350 сохранённых предсказаний на модель')
    axis.tick_params(axis='x', labelsize=12)
    axis.grid(axis='y', alpha=.2)
    axis.legend(frameon=False, ncol=len(NAMES))
    figure.savefig(output / 'benchmark.png', dpi=220, bbox_inches='tight')
    plt.close(figure)


def main():
    from report import report

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--benchmark', type=Path, default=Path('bfcl.jsonl'))
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    runs, errors = read_runs(args.outputs, args.benchmark)
    plots(args.outputs, runs)
    report(args.outputs, runs, errors, load(args.outputs / 'eda.json'), load(args.receipt) if args.receipt else None, (BACKGROUND, INK, TEAL, MUTED))
    print(json.dumps({'verified_predictions': 350 * (len(runs) + 3), 'outputs': str(args.outputs), 'overall': {name: run['metrics']['overall'] for name, run in runs.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
