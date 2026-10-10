import argparse
import ast
import json
import math
from pathlib import Path
from zipfile import ZipFile


LABELS = {
    'baseline': 'SDPA · baseline', 'batch16': 'Batch 16 × accumulation 4',
    'checkpoint': 'SDPA + checkpointing', 'fa2': 'FlashAttention 2',
    'padding_free': 'FA2 + padding_free', 'packing': 'FA2 + packing',
    'padding_compile': 'FA2 + padding_free + compile',
    'padding_liger': 'FA2 + padding_free + Liger',
    'offloading': 'FA2 + activation offloading',
    'padding_liger64': 'FA2 + padding_free + Liger · batch 64',
    'padding_liger128': 'FA2 + padding_free + Liger · batch 128',
    'packing_liger64': 'FA2 + packing + Liger · batch 64',
    'packing_liger128': 'FA2 + packing + Liger · batch 128',
    'final': 'Выбранная комбинация · final',
}
BACKGROUND, INK, MUTED = '#f7f5f0', '#193139', '#687a7c'
TEAL, PALE, ERROR = '#007c78', '#a5bfba', '#a8553a'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def loss_curve(console):
    elapsed, points = None, []
    for line in console.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
        if not isinstance(record, dict):
            continue
        if 'step' in record and finite(record.get('elapsed_seconds')):
            elapsed = float(record['elapsed_seconds'])
        if elapsed is not None and finite(record.get('loss')):
            points.append((elapsed, float(record['loss'])))
    return points


def load_runs(root):
    design = json.loads((root / 'experiments.json').read_text())
    runs = []
    with ZipFile(root / 'logs.zip') as archive:
        members = set(archive.namelist())
        for case in design['cases']:
            name = case['name']
            status_path, log_path = f'logs/{name}.status.json', f'logs/{name}.log'
            if status_path not in members or log_path not in members:
                raise ValueError(f'Incomplete archive: {name} lacks terminal status or console log')
            status = json.loads(archive.read(status_path))
            if status['status'] not in ('ok', 'failed'):
                raise ValueError(f'Nonterminal status: {name}')
            console = archive.read(log_path).decode(errors='replace')
            summary, settings = None, None
            settings_path = f'settings/{name}.json'
            if settings_path in members:
                settings = json.loads(archive.read(settings_path))
            if status['status'] == 'ok':
                summary_path = f'results/{name}/summary.json'
                if summary_path not in members or settings is None:
                    raise ValueError(f'Incomplete successful run: {name}')
                if status.get('returncode') != 0:
                    raise ValueError(f'Contradictory process status: {name}')
                summary = json.loads(archive.read(summary_path))
            errors = [line.strip() for line in console.splitlines() if 'Error:' in line]
            runs.append({**case, 'status': status['status'], 'summary': summary,
                         'settings': settings, 'curve': loss_curve(console),
                         'reason': errors[-1] if errors else f"exit {status.get('returncode', '?')}"})
    for name in ('baseline', 'final'):
        if not any(run['name'] == name and run['status'] == 'ok' for run in runs):
            raise ValueError(f'Missing successful {name} run')
    return runs


def save(figure, figures, name):
    import matplotlib.pyplot as plt
    for extension in ('png', 'svg'):
        figure.savefig(figures / f'{name}.{extension}', dpi=220, bbox_inches='tight', pad_inches=0.3)
    plt.close(figure)


def render(root, figures):
    runs = load_runs(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
                         'figure.facecolor': BACKGROUND, 'axes.facecolor': BACKGROUND,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': BACKGROUND})
    with ZipFile(root / 'logs.zip') as archive:
        gpu = json.loads(archive.read('data_audit.json'))['gpu']
    figures.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, max(8, len(runs) * 0.55 + 2)),
                                gridspec_kw={'width_ratios': [1.25, 1, 1]}, layout='constrained')
    figure.suptitle('Скорость, память и loss', fontsize=27, fontweight='bold', x=0.025, ha='left')
    titles = ('Полезные токены / с ↑', 'Пиковая память · GiB ↓', 'Validation loss ↓')
    metrics = ('tokens_per_second', 'peak_allocated_gib', 'validation_loss')
    for axis, title in zip(axes, titles):
        axis.set_title(title, loc='left', fontweight='bold', pad=20)
        axis.set_ylim(len(runs) - 0.4, -0.7)
        axis.set_axisbelow(True)
        axis.grid(axis='x', color='#dce2df', linewidth=0.7)
        axis.tick_params(axis='y', length=0)
    for index, run in enumerate(runs):
        color = TEAL if run['name'] == 'final' else INK if run['name'] == 'baseline' else PALE
        summary = run['summary'] or {}
        for axis, metric in zip(axes, metrics):
            value = summary.get(metric)
            if not finite(value):
                label = 'ошибка' if run['status'] == 'failed' else 'нет замера'
                axis.text(0.02, index, label, transform=axis.get_yaxis_transform(),
                          color=ERROR if run['status'] == 'failed' else MUTED, va='center', fontsize=10)
                continue
            if metric == 'validation_loss':
                axis.scatter(value, index, s=52 if run['name'] == 'final' else 36,
                             color=color, marker='D' if run['name'] == 'final' else 'o', zorder=3)
            else:
                axis.barh(index, value, height=0.52, color=color, zorder=3)
            label = f'{value:,.0f}'.replace(',', ' ') if metric == 'tokens_per_second' else f'{value:.2f}'
            axis.annotate(label, (value, index), xytext=(7, 0), textcoords='offset points',
                          va='center', color=INK, fontsize=10, fontweight='bold' if run['name'] == 'final' else 'normal')
        reserved = summary.get('peak_reserved_gib')
        if finite(reserved):
            axes[1].barh(index, reserved, height=0.65, fill=False, edgecolor=color, linewidth=1.3, zorder=2)
    labels = [f"{LABELS.get(run['name'], run['name'])}   ·   {run['seconds']} с" for run in runs]
    axes[0].set_yticks(range(len(runs)), labels, fontsize=10)
    for axis in axes[1:]:
        axis.set_yticks(range(len(runs)), [''] * len(runs))
    for axis, metric in zip(axes[:2], metrics[:2]):
        values = [(run['summary'] or {}).get(metric) for run in runs]
        if metric == 'peak_allocated_gib':
            values += [(run['summary'] or {}).get('peak_reserved_gib') for run in runs]
        maximum = max((value for value in values if finite(value)), default=1)
        axis.set_xlim(0, maximum * 1.25)
    losses = [(run['summary'] or {}).get('validation_loss') for run in runs]
    losses = [value for value in losses if finite(value)]
    if losses:
        span = max(max(losses) - min(losses), 0.2)
        axes[2].set_xlim(min(losses) - span * 0.15, max(losses) + span * 0.42)
    axes[1].set_xlabel('Заливка: allocated · контур: reserved', fontsize=10)
    axes[0].set_xlabel('Первые 10 шагов исключены', fontsize=10)
    axes[2].set_xlabel('512 абзацев · вес по токенам', fontsize=10)
    figure.text(0.025, -0.025, f'{gpu}; Qwen3 с нуля; один запуск на конфигурацию; seed 42; без интервалов неопределённости; '
                 '90 с: поиск; 300 с: baseline и повтор выбранной комбинации', fontsize=10, color=MUTED)
    save(figure, figures, 'comparison')
    selected = [run for run in runs if run['name'] in ('baseline', 'final') and run['curve']]
    if len(selected) == 2:
        figure, axis = plt.subplots(figsize=(11.8, 5.8), layout='constrained')
        figure.suptitle('Обучение за 300 секунд', fontsize=25, fontweight='bold', x=0.025, ha='left')
        for run in selected:
            times, losses = zip(*run['curve'])
            axis.plot(times, losses, color=TEAL if run['name'] == 'final' else INK,
                      linewidth=1.8, linestyle='-' if run['name'] == 'final' else '--',
                      marker='o', markersize=3, label=LABELS[run['name']])
        axis.set(xlabel='Секунды обучения, включая первые 10 шагов', ylabel='Train loss · лог каждые 10 шагов')
        axis.grid(axis='y', color='#dce2df', linewidth=0.7)
        axis.legend(frameon=False, fontsize=11)
        figure.text(0.025, -0.025, 'Исходные записи loss; без сглаживания; seed 42; '
                     'разные скорости: разное число шагов и положение внутри LR warmup', fontsize=10, color=MUTED)
        save(figure, figures, 'training')
    return runs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    render(args.root, args.figures or args.root / 'figures')
