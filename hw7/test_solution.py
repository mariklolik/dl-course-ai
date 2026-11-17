import json
import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate import compute_metrics, parse_answer, quantization_state, trim_answer
from distill import TeacherAnswerDataset, collate_teacher_answers, select_student


class SolutionTest(unittest.TestCase):
    def test_saved_int8_weights_reload_with_activation_quantization(self):
        try:
            import torch
            from datasets import Dataset
            from compressed_tensors.compressors import ModelCompressor
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import GPTQModifier
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from transformers import AutoModelForCausalLM, CompressedTensorsConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest('Нужно окружение из requirements.txt')
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                            head_dim=8)).to(torch.bfloat16).eval()
        samples = Dataset.from_list([{'input_ids': [1, 2, 3, 4], 'attention_mask': [1, 1, 1, 1]}] * 2)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, '[PAD]': 5}, unk_token='[UNK]')),
                                            unk_token='[UNK]', pad_token='[PAD]')
        with tempfile.TemporaryDirectory() as folder:
            model.config._name_or_path = folder
            oneshot(model=model, tokenizer=tokenizer, dataset=samples, recipe=GPTQModifier(scheme='W8A8', targets=['Linear'], ignore=['lm_head']),
                    max_seq_length=4, num_calibration_samples=2, shuffle_calibration_samples=False)
            model.save_pretrained(folder, save_compressed=True)
            kwargs = {'torch_dtype': torch.bfloat16, 'device_map': 'cpu',
                      'quantization_config': CompressedTensorsConfig(run_compressed=False)}
            with self.assertRaisesRegex(RuntimeError, 'Only Tensors of floating point'):
                AutoModelForCausalLM.from_pretrained(folder, **kwargs)
            kwargs['quantization_config'] = CompressedTensorsConfig(run_compressed=True)
            loaded = AutoModelForCausalLM.from_pretrained(folder, **kwargs)
            ModelCompressor.from_pretrained(folder).decompress_model(loaded)
            state = quantization_state(loaded)
            self.assertEqual(state['input_modules'], 7)
            self.assertEqual(state['input_bits'], {'8': 7})
            self.assertEqual(state['input_status'], {'frozen': 7})
            self.assertEqual(state['input_dynamic'], {'True': 7})
            self.assertEqual(state['wrapped_input_modules'], 7)
            self.assertEqual(loaded.model.layers[0].self_attn.q_proj.weight.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(loaded(input_ids=torch.tensor([[1, 2, 3, 4]])).logits).all())

    def test_selection_preserves_the_evaluated_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'evaluations').mkdir()
            for name, quality in [('baseline', 0.3), ('teacher', 0.6), ('s128', 0.4),
                                  ('s1024', 0.58), ('s1024_e3', 0.57)]:
                (root / 'evaluations' / f'{name}.json').write_text(json.dumps({'run': name, 'quality': quality}))
                if name.startswith('s'):
                    checkpoint = root / 'students' / name
                    checkpoint.mkdir(parents=True)
                    (checkpoint / 'model.safetensors').write_bytes(name.encode())
            with patch('distill.ARTIFACTS', root), contextlib.redirect_stdout(io.StringIO()):
                select_student()
                with self.assertRaises(FileExistsError):
                    select_student()
            metadata = json.loads((root / 'student_selection.json').read_text())
            self.assertEqual(metadata['selected_run'], 's1024')
            self.assertTrue(metadata['target_reached'])
            self.assertEqual((root / 'student_sft/model.safetensors').read_bytes(), b's1024')
            self.assertEqual(metadata['files']['model.safetensors']['sha256'], hashlib.sha256(b's1024').hexdigest())

    def test_right_padding_masks_only_padding_when_pad_equals_eos(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Нужно окружение из requirements.txt')
        features = [dict(input_ids=[7, 9, 2], attention_mask=[1, 1, 1], labels=[-100, 9, 2]),
                    dict(input_ids=[8, 2], attention_mask=[1, 1], labels=[-100, 2])]
        batch = collate_teacher_answers(features, pad_id=2)
        self.assertEqual(batch['input_ids'].tolist(), [[7, 9, 2], [8, 2, 2]])
        self.assertEqual(batch['labels'].tolist(), [[-100, 9, 2], [-100, 2, -100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_generation_is_greedy_even_when_model_defaults_sample(self):
        try:
            import torch
            from transformers import GenerationConfig
            from transformers.generation.utils import GenerationMixin
        except ImportError:
            self.skipTest('Нужно окружение из requirements.txt')
        test = self

        class Model:
            device = torch.device('cpu')
            generation_config = GenerationConfig(do_sample=True, temperature=0.6, transformers_version='4.55.2')

            def generate(self, **kwargs):
                config, kwargs = GenerationMixin._prepare_generation_config(self, **kwargs)
                test.assertFalse(config.do_sample)
                test.assertEqual(kwargs['input_ids'].tolist(), [[0, 5], [6, 7]])
                test.assertEqual(kwargs['attention_mask'].tolist(), [[0, 1], [1, 1]])
                return torch.tensor([[0, 5, 9, 2], [6, 7, 8, 2]])

        from evaluate import generate_answer_ids

        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        self.assertEqual(generate_answer_ids(Model(), tokenizer, [[5], [6, 7]]), [[9, 2], [8, 2]])

    def test_reloaded_activation_quantization_reports_status(self):
        def wrapped_forward():
            return None

        scheme = SimpleNamespace(weights=object(), input_activations=SimpleNamespace(num_bits=8, dynamic=True))
        module = SimpleNamespace(quantization_scheme=scheme, forward=wrapped_forward,
                                 quantization_enabled=True, quantization_status='frozen',
                                 weight=SimpleNamespace(dtype='torch.bfloat16'))
        state = quantization_state(SimpleNamespace(modules=lambda: [module]))
        self.assertEqual(state['input_status'], {'frozen': 1})
        self.assertEqual(state['input_dynamic'], {'True': 1})
        self.assertEqual(state['wrapped_input_modules'], 1)

    def test_eos_survives_when_it_is_also_padding(self):
        self.assertEqual(trim_answer([10, 2, 2, 2], 2, 2), [10, 2])
        self.assertEqual(trim_answer([10, 11, 0, 0], 2, 0), [10, 11])
        self.assertEqual(trim_answer([10, 11], 2, 0), [10, 11])

    def test_json_schema_is_strict(self):
        for text in ['```json\n{"intent":"a","slots":[]}\n```',
                     '{"intent":"a","slots":[],"extra":1}',
                     '{"intent":"a","slots":[["t",""]]}']:
            self.assertIsNone(parse_answer(text, ['a'], ['t']))

    def test_duplicate_slots_count_as_false_positives(self):
        gold = '{"intent":"a","slots":[["t"," Hello  WORLD "]]}'
        pred = '{"intent":"a","slots":[["t","hello world"],["t","hello world"]]}'
        metrics = compute_metrics([pred], [gold], ['a'], ['t'])
        self.assertAlmostEqual(metrics['slot_micro_f1'], 2 / 3)
        self.assertEqual(metrics['frame_exact_match'], 0)

    def test_invalid_empty_slots_do_not_get_full_credit(self):
        gold = '{"intent":"a","slots":[]}'
        metrics = compute_metrics(['not json'], [gold], ['a'], ['t'])
        self.assertEqual(metrics['quality'], 0)
        self.assertEqual(metrics['valid_json_rate'], 0)

    def test_wrong_intent_keeps_independent_slot_metric(self):
        pred = '{"intent":"b","slots":[["t","x"]]}'
        gold = '{"intent":"a","slots":[["t","x"]]}'
        metrics = compute_metrics([pred], [gold], ['a', 'b'], ['t'])
        self.assertEqual(metrics['quality'], 0.5)
        self.assertEqual(metrics['frame_exact_match'], 0)

    def test_training_masks_prompt_without_dropping_eos(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'answers.jsonl'
            rows = [dict(prompt_ids=[7, 8], answer_ids=[9, 2], ended_eos=True),
                    dict(prompt_ids=[7], answer_ids=[10, 11], ended_eos=False)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            ds = TeacherAnswerDataset(path, 2)
            self.assertEqual(ds[0]['input_ids'], [7, 8, 9, 2])
            self.assertEqual(ds[0]['labels'], [-100, -100, 9, 2])
            self.assertEqual(ds[1]['labels'], [-100, 10, 11])
            self.assertEqual(len(ds), 2)


if __name__ == '__main__':
    unittest.main()
