import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analyze import read_runs
from evaluate import compute_metrics


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        (self.root / 'data').mkdir()
        (self.root / 'evaluations').mkdir()
        self.rows = [{'id': '1', 'text': 'hello', 'gold': '{"intent":"a","slots":[]}'}]
        self.dev = self.root / 'data/dev.jsonl'
        self.dev.write_text(json.dumps(self.rows[0]) + '\n')
        (self.root / 'data/manifest.json').write_text(json.dumps({
            'sha256': {'dev': hashlib.sha256(self.dev.read_bytes()).hexdigest()},
            'intents': ['a'], 'slot_types': ['t'],
        }))
        self.predictions = self.root / 'evaluations/baseline.jsonl'
        self.predictions.write_text(json.dumps(self.rows[0] | {'prediction': self.rows[0]['gold']}) + '\n')
        self.metrics = compute_metrics([self.rows[0]['gold']], [self.rows[0]['gold']], ['a'], ['t'])
        self.metrics |= {'run': 'baseline', 'predictions_sha256': hashlib.sha256(self.predictions.read_bytes()).hexdigest(),
                         'dev_sha256': hashlib.sha256(self.dev.read_bytes()).hexdigest()}
        self.metric_path = self.root / 'evaluations/baseline.json'
        self.metric_path.write_text(json.dumps(self.metrics))

    def tearDown(self):
        self.folder.cleanup()

    def test_saved_metrics_are_recomputed(self):
        runs = read_runs(self.root)
        self.assertEqual(runs['baseline']['quality'], 1)
        self.metrics['quality'] = 0.9
        self.metric_path.write_text(json.dumps(self.metrics))
        with self.assertRaises(AssertionError):
            read_runs(self.root)

    def test_changed_prediction_file_is_rejected(self):
        self.predictions.write_text('changed')
        with self.assertRaises(AssertionError):
            read_runs(self.root)

    def test_wrong_ids_are_rejected_even_with_updated_file_hash(self):
        self.predictions.write_text(json.dumps(self.rows[0] | {'id': 'other', 'prediction': self.rows[0]['gold']}) + '\n')
        self.metrics['predictions_sha256'] = hashlib.sha256(self.predictions.read_bytes()).hexdigest()
        self.metric_path.write_text(json.dumps(self.metrics))
        with self.assertRaises(AssertionError):
            read_runs(self.root)


if __name__ == '__main__':
    unittest.main()
