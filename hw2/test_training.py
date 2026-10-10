import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


ROOT = Path(__file__).parent


def original():
    with ZipFile(ROOT / 'logs.zip') as archive:
        source = archive.read('train_template.py').decode()
    module = ModuleType('original')
    exec(compile(source, 'train_template.py', 'exec'), module.__dict__)
    return source, module


def test_fixed_training_contract_and_core():
    source, template = original()
    actual = ast.parse((ROOT / 'train.py').read_text())
    expected = ast.parse(source)
    for name in ('prepare_data', 'CausalCollator', 'BenchmarkTrainer', 'Benchmark', 'evaluate', 'main'):
        a = next(n for n in actual.body if getattr(n, 'name', None) == name)
        b = next(n for n in expected.body if getattr(n, 'name', None) == name)
        assert ast.dump(a) == ast.dump(b), name
    design = json.loads((ROOT / 'experiments.json').read_text())
    assert design['base'] == template.TRAINING_CONFIG
    assert len(design['cases']) >= 6
    for case in design['cases']:
        assert case['attention'] in ('sdpa', 'flash_attention_2')
        if case['name'] in ('baseline', 'final'):
            assert case['seconds'] == 300


def test_runner_rejects_fixed_training_changes():
    _, template = original()
    spec = importlib.util.spec_from_file_location('runner', ROOT / 'run_experiments.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for key, value in (('learning_rate', 1e-3), ('max_length', 256), ('seed', 43)):
        with pytest.raises(ValueError, match=key):
            runner.training_config(template.TRAINING_CONFIG, {key: value}, 'invalid')


def test_padding_free_preserves_targets_and_boundaries():
    _, template = original()
    samples = [{'input_ids': [1, 2, 3, 0]}, {'input_ids': [4, 5, 0]}]
    padded = template.CausalCollator(0).torch_call(samples)
    flat = DataCollatorForLanguageModeling(0, padding_free=True).torch_call(samples)
    assert padded['labels'].ne(-100).sum().item() == flat['labels'].ne(-100).sum().item() == 5
    assert flat['position_ids'].tolist() == [[0, 1, 2, 3, 0, 1, 2]]
    assert flat['labels'][flat['position_ids'].eq(0)].eq(-100).all()
    assert flat['input_ids'].tolist() == [[1, 2, 3, 0, 4, 5, 0]]


def test_summary_excludes_first_ten_steps():
    _, template = original()
    benchmark = template.Benchmark(None, 300)
    benchmark.start, benchmark.previous_end, benchmark.previous_tokens = 0, 12, 1200
    benchmark.steps = [dict(step_seconds=1, target_tokens=100) for _ in range(12)]
    summary = benchmark.summary()
    assert summary['tokens_per_second'] == 100
    assert summary['measured_steps'] == 2
    assert summary['optimizer_steps'] == 12


def test_bfd_keeps_document_boundaries_and_useful_tokens():
    from datasets import Dataset
    from trl.data_utils import pack_dataset

    samples = [{'input_ids': [1, 2, 3, 0]}, {'input_ids': [4, 5, 0]},
               {'input_ids': [6, 7, 8, 9, 0]}]
    packed = pack_dataset(Dataset.from_list(samples), seq_length=8, strategy='bfd')
    batch = DataCollatorForLanguageModeling(0, padding_free=True).torch_call(list(packed))
    assert all(len(row['input_ids']) <= 8 for row in packed)
    assert batch['labels'].ne(-100).sum().item() == 9
    assert batch['position_ids'].eq(0).sum().item() == 3
    assert batch['labels'][batch['position_ids'].eq(0)].eq(-100).all()
    assert sorted(batch['input_ids'][0].tolist()) == sorted(sum((s['input_ids'] for s in samples), []))
