import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ALLOWED = {
    'per_device_train_batch_size', 'gradient_accumulation_steps',
    'gradient_checkpointing', 'torch_compile', 'use_liger_kernel',
    'activation_offloading', 'packing', 'padding_free',
}


def training_config(base, changes, name):
    invalid = changes.keys() - ALLOWED
    if invalid:
        raise ValueError(', '.join(sorted(invalid)))
    return {**base, **changes, 'output_dir': f'results/{name}'}


def run_case(name):
    design = json.loads(Path('experiments.json').read_text())
    case = next(c for c in design['cases'] if c['name'] == name)
    spec = importlib.util.spec_from_file_location('training', 'train.py')
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    training.TRAINING_CONFIG = training_config(design['base'], case['changes'], name)
    training.ATTENTION_IMPLEMENTATION = case['attention']
    training.TRAIN_SECONDS = case['seconds']
    settings = {**case, 'training_config': training.TRAINING_CONFIG,
                'model_config': training.MODEL_CONFIG}
    Path('settings').mkdir(exist_ok=True)
    Path(f'settings/{name}.json').write_text(json.dumps(settings, indent=2) + '\n')
    training.main()


def run_sequence(names):
    Path('logs').mkdir(exist_ok=True)
    for name in names:
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'TORCHINDUCTOR_CACHE_DIR': str(Path(f'compile-cache/{name}').resolve())}
        with Path(f'logs/{name}.log').open('w') as console:
            result = subprocess.run([sys.executable, '-u', __file__, '--case', name],
                                    env=env, stdout=console, stderr=subprocess.STDOUT)
        summary = Path(f'results/{name}/summary.json')
        record = {'name': name, 'returncode': result.returncode,
                  'status': 'ok' if result.returncode == 0 and summary.exists() else 'failed'}
        Path(f'logs/{name}.status.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case')
    parser.add_argument('--sequence', nargs='+')
    args = parser.parse_args()
    if args.case:
        run_case(args.case)
    elif args.sequence:
        run_sequence(args.sequence)
    else:
        parser.error('Choose --case or --sequence')
