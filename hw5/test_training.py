import json
import unittest
from pathlib import Path

from train import training_config


class TrainingTests(unittest.TestCase):
    def test_controlled_ablation_and_pinned_original_model(self):
        configs = json.loads(Path(__file__).with_name('configs.json').read_text())
        first, second, third = [training_config(configs, name) for name in ('rl1', 'rl2', 'rl3')]
        assert first['model'] == 'Qwen/Qwen3.5-2B'
        assert first['revision'] == '15852e8c16360a2fea060d615a32b45270f8a8fc'
        assert {k for k in first if first[k] != second[k]} == {'reward'}
        assert {k for k in second if second[k] != third[k]} == {'loss_type', 'scale_rewards'}
        for config in (first, second, third):
            assert config['num_generations'] >= 2
            assert config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] % config['num_generations'] == 0
            assert config['max_steps'] >= 100
            assert config['seed'] == 42
            assert config['beta'] > 0
