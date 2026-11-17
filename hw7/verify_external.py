import json
import math
from pathlib import Path

from analyze import METRICS
from data import ARTIFACTS, load_data
from evaluate import clean_memory, evaluate, load_model
from restore_artifacts import restore


def main() -> None:
    manifest = json.loads((ARTIFACTS / 'manifest.json').read_text())
    selection = json.loads((ARTIFACTS / 'student_selection.json').read_text())
    restored = Path('.work/external_restore')
    restored.mkdir(parents=True, exist_ok=True)
    subset = {**manifest, 'files': {name: value for name, value in manifest['files'].items() if name.startswith('student_sft/')}}
    assert subset['files']
    subset['total_bytes'] = sum(value['size'] for value in subset['files'].values())
    (restored / 'manifest.json').write_text(json.dumps(subset, indent=2))
    restore(restored)
    _, dev, intents, slots = load_data()
    model, tokenizer = load_model(restored / 'student_sft')
    try:
        metrics, predictions = evaluate(model, tokenizer, dev, intents, slots)
    finally:
        del model
        clean_memory()
    source = ARTIFACTS / 'evaluations' / selection['selected_run']
    expected = json.loads(source.with_suffix('.json').read_text())
    original = list(map(json.loads, source.with_suffix('.jsonl').read_text().splitlines()))
    equal_metrics = all(math.isclose(metrics[key], expected[key], abs_tol=1e-12) for key in METRICS)
    equal_predictions = predictions == [row['prediction'] for row in original]
    output = ARTIFACTS / 'verification/external_student'
    output.mkdir(parents=True, exist_ok=True)
    records = [dict(row, prediction=prediction) for row, prediction in zip(dev, predictions, strict=True)]
    (output / 'predictions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    receipt = {'repo_id': manifest['repo_id'], 'revision': manifest['revision'],
               'selected_run': selection['selected_run'], 'files': len(subset['files']),
               'metrics': metrics, 'equal_metrics': equal_metrics, 'equal_predictions': equal_predictions}
    (output / 'metrics.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert equal_metrics and equal_predictions, receipt
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
