import json
from html import escape
from pathlib import Path
from statistics import mean


def reward_examples():
    from rewards import score_calls
    target = [{'name': 'weather', 'arguments': {'city': 'Paris'}}]
    wrong_city = [{'name': 'weather', 'arguments': {'city': 'London'}}]
    wrong_name = [{'name': 'other', 'arguments': {'city': 'Paris'}}]
    cases = [('Точный вызов', target, target, None), ('Верное имя, неверный city', wrong_city, target, None), ('Неверное имя', wrong_name, target, None), ('Пропущен обязательный вызов', [], target, None), ('Непустой ответ при no_call', [], [], None), ('Повреждённая разметка', None, target, '<tool_call>')]
    rows = [['Расчётный пример', 'Exact', 'Shaped']]
    for label, calls, expected, response in cases:
        if response is None:
            response = ''.join('<tool_call><function=' + call['name'] + '><parameter=city>' + call['arguments']['city'] + '</parameter></function></tool_call>' for call in calls) if calls else 'Ответ без вызова'
        rows.append([label, *[f'{score_calls(response, calls, expected, mode):.2f}' for mode in ('exact', 'shaped')]])
    return rows


def report(output, runs, errors, eda, receipt, palette):
    NAMES, GROUPS = tuple(runs), runs["base"]["metrics"]["groups"]
    BACKGROUND, INK, TEAL, MUTED = palette
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    fonts = [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path('/System/Library/Fonts/Supplemental/Arial.ttf'), Path(get_data_path()) / 'fonts/ttf/DejaVuSans.ttf']
    font = next((path for path in fonts if path.exists()), None)
    if font is None:
        raise FileNotFoundError('Install DejaVu Sans with Cyrillic support')
    pdfmetrics.registerFont(TTFont('Body', str(font)))
    styles = {name: ParagraphStyle(name, fontName='Body', fontSize=size, leading=size * 1.4, textColor=colors.HexColor(INK), spaceAfter=8) for name, size in [('body', 9), ('small', 7.6), ('title', 23), ('heading', 14)]}
    story = []
    def text(value, style='body'):
        story.append(Paragraph(escape(str(value)).replace('\n', '<br/>'), styles[style]))
    def table(rows):
        cells = [[Paragraph(escape(str(cell)), styles['small']) for cell in row] for row in rows]
        grid = Table(cells, hAlign='LEFT', repeatRows=1)
        grid.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce6e0')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor(TEAL)), ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
        story.extend([grid, Spacer(1, 8)])
    def page(title):
        if story:
            story.append(PageBreak())
        text(title, 'heading')
    selected = max(NAMES[1:], key=lambda name: runs[name]['metrics']['overall'])
    validation_choice = max(NAMES[1:], key=lambda name: runs[name]['training_summary']['validation_exact'])
    text('RL для вызова инструментов', 'title')
    text('Qwen3.5-2B · ToolACE · GRPO · домашнее задание 5')
    text(f'Лучший наблюдаемый BFCL среди четырёх конфигураций: {selected}. BFCL macro score: {runs[selected]["metrics"]["overall"]:.2%}; исходная модель: {runs["base"]["metrics"]["overall"]:.2%}. Максимум по ToolACE validation среди конфигураций: {validation_choice}. Шаг каждой конфигурации выбран по validation, до дополнительных BFCL-замеров.')
    story.append(Image(str(output / 'benchmark.png'), width=490, height=163))
    table([['Модель', 'Overall', 'Simple', 'Multiple', 'Parallel', 'Parallel + multiple', 'Irrelevance']] + [[name, f'{runs[name]["metrics"]["overall"]:.2%}', *[f'{runs[name]["metrics"]["groups"][group]:.2%}' for group in GROUPS]] for name in NAMES])
    text('Pooled accuracy по 350 строкам: ' + '; '.join(f'{name}={runs[name]["metrics"]["pooled"]:.2%}' for name in NAMES), 'small')
    delta_base = 100 * (runs[selected]['metrics']['overall'] - runs['base']['metrics']['overall'])
    text(f'Выбранная RL-модель не улучшила baseline: изменение {delta_base:+.2f} п.п.' if delta_base <= 0 else f'Выбранная RL-модель выше baseline на {delta_base:.2f} п.п. в этом запуске.')
    text('Основной score: среднее пяти групп; внутри multiple и irrelevance сначала усредняются исходные категории. Pooled accuracy по 350 строкам приведена отдельно. Все метрики сверены с metrics.json и SHA256 исходного bfcl.jsonl. BFCL не использовался для обучения и выбора checkpoint.', 'small')
    page('Данные и воспроизводимость')
    text('Обучение только RL на Team-ACE/ToolACE; дополнительного SFT нет. Reference calls передаются reward-функции и не добавляются к целевому prompt. История содержит только предыдущие сообщения. Native tokenizer формирует XML-вызовы; enable_thinking=False.')
    table([['Данные', 'Измерение'], ['Исходных диалогов', eda['raw_dialogues']], ['Точных дубликатов', eda['duplicate_dialogues']], ['Решений с вызовом / без вызова', f"{eda['eligible_categories']['call']} / {eda['eligible_categories']['no_call']}"], ['Исключения', f"Альтернативные схемы: {eda['excluded']['unsupported_tool_schema']}; некорректные цепочки вызовов: {eda['excluded']['invalid_reference_call']}; ответы после tool-result: {eda['excluded']['post_tool_assistant']}"], ['Пересечение диалогов split', eda['split_dialogue_overlap']]])
    split_rows = [['Split', 'Диалогов', 'Примеров', 'Call / no-call', 'Prompt p50 / p95 / max']]
    for split, values in eda['splits'].items():
        lengths, counts = values['selected_prompt_tokens'], values['selected_categories']
        split_rows.append([split, values['dialogues'], values['selected'], f"{counts['call']} / {counts['no_call']}", f"{lengths['p50']} / {lengths['p95']} / {lengths['p100']}"])
    table(split_rows)
    text('Детерминированный split по хешу целого исходного диалога, seed 42; одинаковые строки дедуплицируются. Лимит prompt применяется после split; затем выборка с целевой долей no_call 25%. Альтернативные схемы и непарсируемые reference-вызовы исключены с явным учётом. Имена функций нормализованы с разрешением коллизий; AST literal_eval не выполняет код.')
    text(f'Dataset revision: {eda["dataset_revision"]}\nBase revision: {eda["model_revision"]}', 'small')
    text('Источник: https://huggingface.co/datasets/Team-ACE/ToolACE\nМодель: https://huggingface.co/Qwen/Qwen3.5-2B\nМетод: https://huggingface.co/docs/trl/grpo_trainer', 'small')
    page('Три контролируемых запуска и RL4')
    story.append(Image(str(output / 'loss_reward.png'), width=490, height=147))
    table([['Run', 'Reward', 'Loss / scale', 'Шаг', 'Train, с', 'Eval, с']] + [[name, runs[name]['config']['reward'], f'{runs[name]["config"]["loss_type"]} / {runs[name]["config"]["scale_rewards"]}', runs[name]['training_summary']['selected_step'], f'{runs[name]["training_summary"]["seconds"]:.1f}', runs[name]['receipt']['elapsed_seconds']] for name in NAMES[1:]])
    text('Гипотеза rl1: строгая exact-награда повысит долю полностью правильных вызовов. Гипотеза rl2: частичный credit за имена и аргументы даст более плотный сигнал и улучшит exact accuracy. Гипотеза rl3: устранение нормализации по длине ответа и std reward уменьшит соответствующие смещения; проверяем совместное изменение, а не каждый фактор отдельно.', 'small')
    text('GRPO усредняет token-loss каждого ответа по его длине; Dr. GRPO использует фиксированный T_max из max_completion_length и здесь одновременно scale_rewards=none. Численное значение T_max берётся из фактического config ниже.', 'small')
    common = runs['rl1']['config']
    keys = ['seed', 'max_steps', 'learning_rate', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'num_generations', 'max_completion_length', 'temperature', 'beta', 'epsilon', 'lora_rank']
    text('Фактическая конфигурация rl1: ' + '; '.join(f'{key}={common[key]}' for key in keys), 'small')
    for name in NAMES[1:]:
        checks = '; '.join(f"шаг {row['step']}: {row['accuracy']:.2%}" for row in runs[name]['selection'])
        text(f'{name}: ToolACE validation - {checks}; полные merged-веса экспортированы.', 'small')
    versions = (receipt or {}).get('versions', runs['rl1']['training_summary']['versions'])
    hardware = ', '.join(sorted({runs[name]['training_summary']['gpu'] for name in NAMES[1:]}))
    runtime = '; '.join(f'{name} {versions[name]}' for name in ('torch', 'trl', 'transformers', 'peft', 'vllm'))
    text(f'Обучение и оценка: avi-gn-fsk34, {hardware}. {runtime}. Native Torch CUDA {(receipt or {}).get("torch_cuda_build", "не записана")}; driver {(receipt or {}).get("driver", "не записан")}. Полный манифест: environment.json и environment.freeze.txt.', 'small')
    text('Время train включает validation и обучение, но не загрузку модели. JIT/кэши различались между запусками, поэтому это не сравнение чистой скорости алгоритмов. Eval измерен отдельным evaluator.', 'small')
    text('Гипотеза RL4: дополнительный бюджет обновлений улучшит перенос. Это длительный запуск того же Dr. GRPO из исходных весов: 320 шагов, один финальный validation/checkpoint, warmup 16 вместо 8. Это разведочный эксперимент; длительность и warmup здесь не разделены.')
    text('Кривые показывают только реальные записи без сглаживания. Значения exact и shaped reward, а также loss с разными нормировками не сравниваются как единая шкала качества. Один seed и один запуск на конфигурацию; доверительные интервалы не оценивались.', 'small')
    page('Ответы: reward, абляции и преимущество')
    text('1. Почему такая награда?', 'heading')
    text('Exact: 1 только при совпадении мультимножества (имя, JSON аргументов). Shaped: R = 0.15 + 0.25 F1_имён + 0.30 F1_листьев_аргументов + 0.30 I_exact. Формула применяется только при корректном завершённом вызове и хотя бы одном совпавшем имени. Иначе 0. Для no_call награда 1 за непустой корректный ответ без вызова, иначе 0. Награда проверяет решение о вызове, имена и значения; мультимножества сохраняют повторные вызовы и не зависят от порядка. Частичная F1 листьев не закрепляет аргументы за отдельным повторным вызовом; корректность такого соответствия гарантирует только exact-компонента.')
    table(reward_examples())
    text('Это расчётные примеры score_calls, не предсказания модели. В примере wrong city есть один лист аргумента: F1 имени=1, F1 листьев=0, exact=0, поэтому shaped=0.40.', 'small')
    text('2. Что дали абляции?', 'heading')
    comparison = {name: next(p['overall'] for p in runs[name]['bfcl_steps'] if p['step'] == (320 if name == 'rl4' else 160)) for name in NAMES[1:]}
    text('Для RL1-RL3 ниже сравнивается одинаковый шаг 160, чтобы не смешивать эффект настройки с выбором checkpoint. RL4 сравнивается на 320 шагах.')
    for left, right, reason in [('rl1', 'rl2', 'замена exact на shaped'), ('rl2', 'rl3', 'совместная замена loss на dr_grpo и отключение scale'), ('rl3', 'rl4', 'бюджет 320 и warmup 16 вместо 160 и 8')]:
        delta = 100 * (comparison[right] - comparison[left])
        text(f'{left} → {right}: {reason}; изменение BFCL macro {delta:+.2f} п.п. Это результат конкретных запусков; пары RL2/RL3 и RL3/RL4 меняют связанные параметры, поэтому их отдельные эффекты не идентифицируются.')
    changes = [('форма награды', comparison['rl2'] - comparison['rl1']), ('две нормировки Dr. GRPO', comparison['rl3'] - comparison['rl2']), ('длительность и warmup', comparison['rl4'] - comparison['rl3'])]
    strongest = max(changes, key=lambda pair: abs(pair[1]))
    text(f'Наибольшее по модулю изменение: {strongest[0]}, {100 * strongest[1]:+.2f} п.п. Это сравнение при одном seed, а не оценка устойчивого среднего эффекта.')
    page('3. Совпадает ли рост reward с ростом BFCL?')
    reward_rows = [['Run', 'Шаг', 'ToolACE val', 'BFCL overall', 'Irrelevance', 'Reward, 20 шагов']]
    directions = []
    for name in NAMES[1:]:
        values = [row['reward'] for row in runs[name]['training_history'] if 'reward' in row]
        points = runs[name]['bfcl_steps']
        windows = {point['step']: mean(values[max(0, point['step'] - 20):point['step']]) for point in points}
        for point in points:
            reward_rows.append([name, point['step'], f"{point['validation_exact']:.2%}", f"{point['overall']:.2%}", f"{point['irrelevance']:.2%}", f"{windows[point['step']]:.4f}"])
        if len(points) == 2:
            reward_delta = windows[points[1]['step']] - windows[points[0]['step']]
            bfcl_delta = points[1]['overall'] - points[0]['overall']
            direction = 'Направления разошлись' if reward_delta * bfcl_delta < 0 else 'Направления совпали либо одно изменение нулевое'
            directions.append(f'{name}: Δreward={reward_delta:+.4f}; ΔBFCL={100 * bfcl_delta:+.2f} п.п. {direction}.')
    table(reward_rows)
    for description in directions:
        text(description)
    text('Для RL1-RL3 BFCL дополнительно измерен у обоих checkpoints после выбора по validation; у RL4 измерен только финальный шаг 320. Две точки показывают направление изменения, но не устойчивую корреляцию или причинность. Reward-окна содержат разные on-policy промпты и ответы; это описательное сравнение, не контроль на одном наборе. Все дополнительные predictions сохранены в step-N; BFCL не использовался для выбора шага.')
    text('4. Если в группе все reward одинаковы?', 'heading')
    text('A_i = (r_i - mean(r)) / (std(r) + epsilon) = 0; при scale_rewards=none также r_i - mean(r) = 0. Policy-gradient член от reward обнуляется. Полный градиент не обязан быть нулём: beta > 0 сохраняет KL к reference; возможен вклад KL и состояния оптимизатора. В самом начале, если policy совпадает с reference и накопленных состояний нет, этот вклад также может быть нулевым.')
    samples = []
    for kind in ['false_call_irrelevance', 'missed_required_call', 'missed_required_argument']:
        candidates = [(name, row) for name in [selected, *[n for n in NAMES if n != selected]] for row in errors[name]['errors'] if row['error_type'] == kind]
        if candidates:
            samples.append(candidates[0])
    for name in [selected, 'base']:
        for row in errors[name]['errors']:
            if row['category'] not in {item['category'] for _, item in samples} and len(samples) < 6:
                samples.append((name, row))
    page('5. Ошибки на исходных 350 примерах')
    table([['Run', 'Ошибок', 'False call', 'Missed call', 'Missing arg', 'Length']] + [[name, 350 - errors[name]['counts'].get('correct', 0), *[errors[name]['counts'].get(key, 0) for key in ['false_call_irrelevance', 'missed_required_call', 'missed_required_argument', 'truncated']]] for name in NAMES])
    text('Таксономия ошибок основана на сохранённом check.valid и структурных признаках; argument_or_format не является отдельной семантической переоценкой. Ноль в таблице означает ноль наблюдавшихся случаев. Полные ответы, tool schema, ground truth и матрицы числа вызовов: error_analysis.json; все 350 строк: predictions.jsonl каждого run.', 'small')
    explanations = {
        'false_call_irrelevance': 'Эталон не требует вызова. Модель всё же использовала доступный инструмент вместо ответа без вызова.',
        'missed_required_call': 'Модель ответила обычным текстом вместо вызова доступной функции. Даже содержательный текст не заменяет требуемый вызов.',
        'missed_required_argument': 'Выбрана нужная функция, но пропущен обязательный аргумент. Отсутствующее поле указано checker.',
        'argument_or_format': 'Вызовы распознаны, но их аргументы или соответствие полного набора не проходят AST-проверку. Ниже видны полученные значения и ошибка checker.',
    }
    for index, (name, row) in enumerate(samples):
        if index == 2:
            page('Примеры ошибок и проверка артефактов')
        beginning = len(story)
        text(f'{name} / {row["category"]} / {row["id"]}: {row["error_type"]}', 'small')
        text(explanations.get(row['error_type'], 'Сохранённый ответ не прошёл официальный checker.'), 'small')
        question = '\n'.join(message.get('content', '') for message in row['question'])
        response = row['calls'] if row['calls'] else row['response']
        for label, value, limit in [('Запрос', question, 500), ('Ожидание', row['ground_truth'], 500), ('Ответ / распознанные вызовы', response, 650), ('Checker', row['check'].get('error'), 400)]:
            raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            text(label + ': ' + raw[:limit] + (' ... [обрезано; полный текст в JSON]' if len(raw) > limit else ''), 'small')
        story[beginning:] = [KeepTogether(story[beginning:])]
    text('Полные merged-веса подтверждаются export_verification.json и хешами файлов; ненулевые LoRA B norms доказывают обновление adapter. Эти проверки не заменяют внешнюю оценку качества. Negative results и ошибки сохранены; SFT и обучение на BFCL не выполнялись.', 'small')
    def decorate(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(BACKGROUND))
        canvas.rect(0, 0, 595.28, 841.89, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.setFont('Body', 8)
        canvas.drawString(45, 22, 'HW5 / Qwen3.5-2B / ToolACE RL')
        canvas.drawRightString(550, 22, str(document.page))
        canvas.restoreState()
    SimpleDocTemplate(str(output / 'report.pdf'), pagesize=(595.28, 841.89), rightMargin=45, leftMargin=45, topMargin=38, bottomMargin=38).build(story, onFirstPage=decorate, onLaterPages=decorate)
