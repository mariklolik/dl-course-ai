import unittest
from pathlib import Path

from export import export_targets


class ExportTests(unittest.TestCase):
    def test_auxiliary_step_preserves_selected_model_and_receipt(self):
        root = Path('outputs/rl1')
        selected = export_targets(root, 80)
        auxiliary = export_targets(root, 80, 160)
        self.assertEqual(selected, (80, root / 'merged', root / 'export_verification.json'))
        self.assertEqual(auxiliary, (160, root / 'merged-step160', root / 'export_verification-step160.json'))
        self.assertNotEqual(selected[1], auxiliary[1])
        self.assertNotEqual(selected[2], auxiliary[2])
