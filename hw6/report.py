import csv
import re
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


BLUE = '#1e5a78'
ORANGE = '#d77a3f'


def plot_fixed(rows: list[dict]) -> None:
    fixed = sorted([r for r in rows if r['group'] == 'fixed' and r['mean_s'] is not None and r['complete'] == 'True' and int(r['failed']) == 0], key=lambda r: r['mean_s'])
    baseline = next(r['mean_s'] for r in fixed if r['config'] == 'baseline')
    fig, ax = plt.subplots(figsize=(8, max(3.1, 0.39 * len(fixed) + 1.5)))
    colors = [BLUE if r == fixed[0] else '#8eabb8' for r in fixed]
    ax.barh([r['config'] for r in fixed], [r['mean_s'] for r in fixed], color=colors, label='Mean')
    ax.scatter([r['p90_s'] for r in fixed], range(len(fixed)), color=ORANGE, label='P90', zorder=3)
    for i, row in enumerate(fixed):
        ax.text(max(row['mean_s'], row['p90_s']) + 0.08, i, f"{row['mean_s']:.2f} с; S={baseline / row['mean_s']:.2f}", va='center', fontsize=8)
    ax.set(xlabel='E2E latency, с', xlim=(0, max(r['p90_s'] for r in fixed) * 1.3))
    ax.invert_yaxis()
    ax.set_title('120 RPM: средняя задержка и P90', pad=30)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig('outputs/configurations.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_speculative(rows: list[dict], group: str, cap: int) -> None:
    sweep = [r for r in rows if r['group'] == group]
    if not sweep:
        return
    if group == 'large5':
        rates = {r['rpm'] for r in sweep}
        sweep.extend(r for r in rows if r['group'] == 'large' and r['config'] == 'seq256_chunk2048' and r['rpm'] in rates)
    names = sorted({r['config'] for r in sweep})
    plain = next(name for name in names if not name.startswith(('eagle', 'ngram', 'draft')))
    spec = next(name for name in names if name != plain)
    for name in names:
        if group == 'sweep' and not any(r['config'] == name and r['rpm'] == 120 for r in sweep):
            sweep.append(next(r for r in rows if r['group'] == 'fixed' and r['config'] == name and r['rpm'] == 120))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, color, label in [(plain, BLUE, 'Без спекдека'), (spec, ORANGE, 'Со спекдеком')]:
        series = sorted([r for r in sweep if r['config'] == name], key=lambda r: r['rpm'])
        axes[0].plot([r['rpm'] for r in series], [r['p90_s'] for r in series], 'o-', color=color, label=label)
    paired = {name: {r['rpm']: r['p90_s'] for r in sweep if r['config'] == name and r['p90_s'] is not None and r['complete'] == 'True' and int(r['failed']) == 0} for name in names}
    rates = sorted(paired[plain].keys() & paired[spec].keys())
    speedups = [paired[plain][rpm] / paired[spec][rpm] for rpm in rates]
    axes[1].plot(rates, speedups, 'o-', color=ORANGE)
    axes[1].axhline(1, color='#687780', linestyle='--', linewidth=1)
    worse = [rpm for rpm, speedup in zip(rates, speedups) if speedup < 1]
    if worse:
        axes[1].axvline(worse[0], color='#687780', linestyle=':', linewidth=1)
        axes[1].text(worse[0], max(speedups), f' {worse[0]} RPM', fontsize=8, va='top', ha='right')
    axes[0].set(xlabel='Нагрузка, RPM', ylabel='P90 E2E latency, с', title=f'Задержка, cap={cap}, k={5 if group == "large5" else 3}')
    axes[1].set(xlabel='Нагрузка, RPM', ylabel='P90 без спекдека / P90 со спекдеком', title='Ускорение по P90')
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    filename = {'sweep': 'speculative', 'large': 'speculative_cap256', 'large5': 'speculative_cap256_k5'}[group]
    fig.savefig(f'outputs/{filename}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_load(rows: list[dict]) -> None:
    load = sorted([r for r in rows if r['group'] == 'load' and int(r['failed']) == 0], key=lambda r: r['rpm'])
    if not load:
        return
    available = [r for r in load if r['p90_s'] is not None]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r['rpm'] for r in available], [r['p90_s'] for r in available], color=BLUE, linewidth=1)
    for row in load:
        if row['p90_s'] is None:
            ax.text(row['rpm'], 0.97, 'Нет успешных ответов', transform=ax.get_xaxis_transform(), rotation=90, va='top', fontsize=8)
        else:
            ax.scatter(row['rpm'], row['p90_s'], color=BLUE if row['sla_pass'] == 'True' else ORANGE, s=45, zorder=3)
    passing = [r for r in load if r['sla_pass'] == 'True']
    if passing:
        best = max(passing, key=lambda r: r['rpm'])
        ax.scatter(best['rpm'], best['p90_s'], marker='*', color=BLUE, edgecolor='white', s=240, zorder=4)
        ax.annotate(f"Максимум на сетке: {best['rpm']} RPM", (best['rpm'], best['p90_s']), xytext=(-15, 45), ha='right', textcoords='offset points', arrowprops={'arrowstyle': '->', 'color': BLUE}, fontsize=9)
    ax.axhline(15, color='#687780', linestyle='--', linewidth=1, label='SLA: P90 < 15 с')
    ax.set(xlabel='Нагрузка, RPM', ylabel='P90 E2E latency, с', title='Нагрузочное тестирование: 300 секунд на точку\nТочки с ошибками приведены отдельно в таблице')
    ax.set_ylim(0, max(18, max(r['p90_s'] for r in available) * 1.3))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('outputs/load_test.png', dpi=200, bbox_inches='tight')
    plt.close(fig)



def write_pdf(rows: list[dict]) -> None:
    from matplotlib.font_manager import FontProperties, findfont
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(TTFont('Body', findfont(FontProperties(family='DejaVu Sans'))))
    styles = {name: ParagraphStyle(name, fontName='Body', fontSize=size, leading=size * 1.4, textColor=colors.HexColor('#17313f'), spaceAfter=8) for name, size in [('body', 9), ('small', 7.6), ('title', 23), ('heading', 14)]}
    story = []

    def text(value, style='body'):
        value = escape(value).replace('`', '').replace('**', '')
        value = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#1e5a78">\1</link>', value)
        value = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', value)
        story.append(Paragraph(value, styles[style]))

    def table(data, widths=None):
        if widths is None:
            weights = [2] + [1] * (len(data[0]) - 1)
            widths = [505 * weight / sum(weights) for weight in weights]
        cells = [[Paragraph(escape(str(cell)).replace('`', ''), styles['small']) for cell in row] for row in data]
        grid = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        grid.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce8ee')), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor(BLUE)), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6f7')]), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
        story.extend([grid, Spacer(1, 10)])

    def figure(path, width=490):
        with PILImage.open(path) as image:
            w, h = image.size
        story.append(Image(path, width=width, height=width * h / w))
        story.append(Spacer(1, 8))

    fixed = [r for r in rows if r['group'] == 'fixed' and r['complete'] == 'True' and int(r['failed']) == 0]
    baseline = next(r for r in fixed if r['config'] == 'baseline')
    best = min(fixed, key=lambda r: r['mean_s'])
    text('ДЗ6: ускорение инференса', 'title')
    text('Одна H100 80GB, Qwen/Qwen3-8B в BF16. Исходные 640 запросов, temperature=0, seed=42, max_tokens=512, thinking выключен. Замер через AIPerf 0.13.0 при 120 RPM.')
    text('vLLM 0.22.0 запущен нативно: Python 3.12.14, Torch 2.11.0 / CUDA 13.0. Это отличается от исходного cu129-образа: Docker socket недоступен пользователю. Все сравнения выполнены в одном окружении на одной GPU.')
    table([['Конфигурация', 'Mean, с', 'P90, с', 'S', 'Токенов', 'Acceptance, %']] + [[r['config'], f"{r['mean_s']:.3f}", f"{r['p90_s']:.3f}", f"{baseline['mean_s'] / r['mean_s']:.3f}", f"{float(r['mean_output_tokens']):.2f}", f"{100 * float(r['spec_acceptance_rate']):.1f}" if r['spec_acceptance_rate'] else '-'] for r in fixed], [148, 56, 56, 50, 75, 120])
    figure('outputs/configurations.png', 460)
    text(f"Лучший из проверенных: {best['config']}, Mean {best['mean_s']:.3f} с, ускорение {baseline['mean_s'] / best['mean_s']:.3f}. Во всех строках 640 успешных запросов без ошибок. В таблице опорные прогоны; контрольные повторы приведены отдельно.")
    source = Path('README.md').read_text().split('![Сравнение конфигураций](outputs/configurations.png)', 1)[1]
    lines = ['## Что дали настройки', *source.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('## '):
            story.append(PageBreak())
            text(line[3:], 'heading')
        elif line.startswith('### '):
            text(line[4:], 'heading')
        elif line.startswith('!['):
            figure(re.fullmatch(r'!\[[^]]*\]\(([^)]+)\)', line)[1])
        elif line.startswith('|'):
            data = [line.strip('|').split('|')]
            while i < len(lines) and lines[i].startswith('|'):
                if not re.fullmatch(r'[| :\-]+', lines[i]):
                    data.append(lines[i].strip('|').split('|'))
                i += 1
            table([[re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', c.strip()) for c in row] for row in data])
        elif line.startswith('```'):
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            story.append(Preformatted('\n'.join(code), styles['small'], maxLineLength=100))
            i += 1
        else:
            text(line)

    def decorate(canvas, document):
        canvas.setFont('Body', 7.5)
        canvas.setFillColor(colors.HexColor('#6c7b83'))
        canvas.drawString(45, 22, 'ДЗ6 · Qwen3-8B · H100 80GB')
        canvas.drawRightString(550, 22, str(document.page))

    SimpleDocTemplate('outputs/report.pdf', pagesize=(595.28, 841.89), rightMargin=45, leftMargin=45, topMargin=38, bottomMargin=38).build(story, onFirstPage=decorate, onLaterPages=decorate)


def main() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    with Path('outputs/results.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['rpm'] = int(row['rpm'])
        for key in ('mean_s', 'p90_s'):
            row[key] = float(row[key]) if row[key] else None
    plot_fixed(rows)
    plot_speculative(rows, 'sweep', 32)
    plot_speculative(rows, 'large', 256)
    plot_speculative(rows, 'large5', 256)
    plot_load(rows)
    write_pdf(rows)


if __name__ == '__main__':
    main()
