import json
import tempfile
import unittest
from pathlib import Path

from analyze import audit_records, compare_outputs, minute_rate_verified, read_summary


class ResultsTest(unittest.TestCase):
    def test_latency_units_and_failed_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile_export_aiperf.json'
            path.write_text(json.dumps({
                'is_complete': True, 'was_cancelled': True,
                'request_latency': {'unit': 'ms', 'avg': 2000.0, 'p90': 3500.0},
                'request_count': {'unit': 'requests', 'avg': 640},
                'error_request_count': {'unit': 'requests', 'avg': 2},
                'output_sequence_length': {'unit': 'tokens', 'avg': 200.0},
                'request_throughput': {'unit': 'requests/sec', 'avg': 1.9},
            }))
            row = read_summary(path)
            self.assertEqual(row['mean_s'], 2.0)
            self.assertEqual(row['p90_s'], 3.5)
            self.assertEqual(row['completed'], 640)
            self.assertEqual(row['failed'], 2)
            self.assertFalse(row['sla_pass'])
            self.assertFalse(row['complete'])

    def test_warmup_and_error_are_excluded_from_latency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile_export.jsonl'
            records = [{
                'metadata': {'benchmark_phase': phase, 'credit_issued_ns': start,
                             'request_start_ns': start + 100000000,
                             'request_end_ns': start + 100000000 + int(latency * 1000000),
                             'session_num': str(start), 'x_request_id': f'{phase}-{start}'},
                'metrics': {'request_latency': {'unit': 'ms', 'value': latency}},
                'error': error,
            } for phase, start, latency, error in [
                ('warmup', 0, 500000, None),
                ('profiling', 1000000000, 1000, None),
                ('profiling', 1500000000, 2000, None),
                ('profiling', 2000000000, 3000, None),
                ('profiling', 2500000000, 120000, {'message': 'timeout'}),
            ]]
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            result = audit_records(path)
            self.assertEqual(result['records_completed'], 3)
            self.assertEqual(result['records_failed'], 1)
            self.assertEqual(result['exact_mean_s'], 2.0)
            self.assertAlmostEqual(result['exact_p90_s'], 2.8)
            self.assertEqual(result['issue_span_s'], 1.5)
            self.assertEqual(result['offered_rpm'], 120)
            self.assertEqual(result['actual_start_rpm'], 120)
            self.assertAlmostEqual(result['effective_p90_s'], 2.9)
            self.assertEqual(result['client_queue_p90_s'], 0.1)
            self.assertTrue(result['unique_request_ids'])
            self.assertEqual(result['start_counts_60s'], [4, 0, 0, 0, 0])

    def test_output_comparison_uses_session_order(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline = Path(folder) / 'baseline.json'
            candidate = Path(folder) / 'candidate.json'
            baseline.write_text(json.dumps({'data': [
                {'session_num': 0, 'response_text': 'abc', 'metrics': {'output_sequence_length': 1, 'input_sequence_length': 10}},
                {'session_num': 1, 'response_text': 'def', 'metrics': {'output_sequence_length': 1, 'input_sequence_length': 10}},
            ]}))
            candidate.write_text(json.dumps({'data': [
                {'session_num': 1, 'response_text': 'defg', 'metrics': {'output_sequence_length': 2, 'input_sequence_length': 10}},
                {'session_num': 0, 'response_text': 'abc', 'metrics': {'output_sequence_length': 1, 'input_sequence_length': 10}},
            ]}))
            result = compare_outputs(baseline, candidate)
            self.assertEqual(result['compared_outputs'], 2)
            self.assertEqual(result['changed_texts'], 1)
            self.assertEqual(result['changed_token_lengths'], 1)
            records = json.loads(candidate.read_text())
            records['data'][0]['metrics']['input_sequence_length'] = 99
            candidate.write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                compare_outputs(baseline, candidate)

    def test_all_failed_run_keeps_the_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = Path(folder) / 'summary.json'
            records = Path(folder) / 'records.jsonl'
            summary.write_text(json.dumps({
                'is_complete': False,
                'error_request_count': {'avg': 1},
            }))
            records.write_text(json.dumps({
                'metadata': {'benchmark_phase': 'profiling', 'credit_issued_ns': 1000000000},
                'error': {'message': 'timeout'},
                'metrics': {},
            }) + '\n')
            row = read_summary(summary)
            audit = audit_records(records)
            self.assertIsNone(row['mean_s'])
            self.assertIsNone(row['p90_s'])
            self.assertEqual(row['failed'], 1)
            self.assertFalse(row['sla_pass'])
            self.assertIsNone(audit['exact_mean_s'])
            self.assertIsNone(audit['exact_p90_s'])
            self.assertEqual(audit['records_failed'], 1)

    def test_client_queue_does_not_hide_behind_issued_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile_export.jsonl'
            records = [{
                'metadata': {
                    'benchmark_phase': 'profiling', 'session_num': i, 'x_request_id': str(i),
                    'credit_issued_ns': i * 500000000, 'request_start_ns': i * 2000000000,
                    'request_end_ns': i * 2000000000 + 1000000000,
                },
                'metrics': {'request_latency': {'unit': 'ms', 'value': 1000}},
            } for i in range(3)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            audit = audit_records(path)
            self.assertEqual(audit['offered_rpm'], 120)
            self.assertEqual(audit['actual_start_rpm'], 30)
            self.assertEqual(audit['exact_p90_s'], 1)
            self.assertAlmostEqual(audit['effective_p90_s'], 3.7)

    def test_minute_boundaries_allow_one_credit_of_cumulative_jitter(self):
        self.assertTrue(minute_rate_verified([480, 480, 481, 480, 478], 480))
        self.assertFalse(minute_rate_verified([480, 480, 482, 479, 478], 480))
        self.assertFalse(minute_rate_verified([480, 480, 481, 480, 470], 480))
        self.assertFalse(minute_rate_verified([30] * 5, 120))

    def test_speculative_acceptance_comes_from_server_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile_export_aiperf.json'
            path.write_text(json.dumps({
                'is_complete': True,
                'request_latency': {'unit': 'ms', 'avg': 1000, 'p90': 1000},
                'request_count': {'avg': 1}, 'request_throughput': {'avg': 2},
                'output_sequence_length': {'avg': 100},
            }))
            path.with_name('server_metrics_export.json').write_text(json.dumps({'metrics': {
                'vllm:spec_decode_num_draft_tokens': {'series': [{'stats': {'total': 100}}]},
                'vllm:spec_decode_num_accepted_tokens': {'series': [{'stats': {'total': 30}}]},
                'vllm:spec_decode_num_drafts': {'series': [{'stats': {'total': 20}}]},
                'vllm:prefix_cache_queries': {'series': [{'stats': {'total': 400}}]},
                'vllm:prefix_cache_hits': {'series': [{'stats': {'total': 100}}]},
                'vllm:num_requests_running': {'series': [{'stats': {'avg': 8, 'max': 12}}]},
                'vllm:num_requests_waiting': {'series': [{'stats': {'avg': 3}}]},
            }}))
            row = read_summary(path)
            self.assertEqual(row['spec_acceptance_rate'], 0.3)
            self.assertEqual(row['spec_draft_steps'], 20)
            self.assertEqual(row['prefix_cache_token_hit_rate'], 0.25)
            self.assertEqual(row['server_waiting_avg'], 3)
            self.assertEqual(row['server_running_max'], 12)


if __name__ == '__main__':
    unittest.main()
