import json
import tempfile
import unittest
from pathlib import Path

from quantize import recipe_spec, recovery_metadata


class QuantizationRecipeTest(unittest.TestCase):
    def test_w4_group_size_and_last_block_exclusion(self):
        recipe = recipe_spec({'method': 'gptq', 'group_size': 64, 'last_block_bf16': True}, 31)
        modifier = recipe['quant_stage']['quant_modifiers']['GPTQModifier']
        weights = modifier['config_groups']['group_0']['weights']
        self.assertEqual(weights['num_bits'], 4)
        self.assertEqual(weights['group_size'], 64)
        self.assertTrue(weights['symmetric'])
        self.assertEqual(modifier['ignore'], ['lm_head', r're:model\.layers\.31\..*'])

    def test_smoothquant_retains_dynamic_int8_activations(self):
        recipe = recipe_spec({'method': 'smoothquant', 'alpha': 0.8}, 31)
        modifiers = recipe['quant_stage']['quant_modifiers']
        self.assertEqual(modifiers['SmoothQuantModifier']['smoothing_strength'], 0.8)
        self.assertEqual(len(modifiers['SmoothQuantModifier']['mappings']), 2)
        scheme = modifiers['GPTQModifier']['config_groups']['group_0']
        self.assertEqual(scheme['weights']['num_bits'], 8)
        self.assertEqual(scheme['input_activations']['num_bits'], 8)
        self.assertEqual(scheme['input_activations']['strategy'], 'token')
        self.assertTrue(scheme['input_activations']['dynamic'])
        self.assertTrue(scheme['input_activations']['symmetric'])

    def test_recovery_preserves_calibration_without_invented_timings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.json'
            original = {'config': {'method': 'smoothquant'}, 'calibration': {'sha256': 'abc'}, 'recipe': {'alpha': 0.5}}
            path.write_text(json.dumps(original))
            recovered = recovery_metadata(path)
            self.assertEqual(recovered['calibration'], original['calibration'])
            self.assertEqual(recovered['recipe'], original['recipe'])
            self.assertIsNone(recovered['quantization_seconds'])
            self.assertIsNone(recovered['save_seconds'])
            self.assertIsNone(recovered['peak_quantization_allocated_bytes'])
            self.assertIn('recovery_note', recovered)
            self.assertEqual(json.loads(path.read_text()), original)

    def test_awq_is_weights_only(self):
        recipe = recipe_spec({'method': 'awq', 'group_size': 128}, 31)
        modifier = recipe['quant_stage']['quant_modifiers']['AWQModifier']
        scheme = modifier['config_groups']['group_0']
        self.assertEqual(scheme['weights']['group_size'], 128)
        self.assertNotIn('input_activations', scheme)
        self.assertEqual(modifier['ignore'], ['lm_head'])


if __name__ == '__main__':
    unittest.main()
