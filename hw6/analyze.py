import csv
import hashlib
import json
import statistics
from itertools import accumulate
from pathlib import Path


def read_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    latency = data.get('request_latency')
    scale = {'ms': 0.001, 's': 1.0}[latency['unit']] if latency else 1.0
    failed = int(data.get('error_request_count', {}).get('avg', 0))
    complete = bool(data['is_complete']) and not data.get('was_cancelled') and not data.get('incomplete_reason')
    p90 = latency['p90'] * scale if latency else None
    counters = {}
    server = path.with_name('server_metrics_export.json')
    if server.exists():
        counters = json.loads(server.read_text())['metrics']
    spec = {}
    for key in ('draft_tokens', 'accepted_tokens', 'drafts'):
        series = counters.get('vllm:spec_decode_num_' + key, {}).get('series', [])
        spec[key] = sum(item['stats']['total'] for item in series)
    prefix = {}
    for key in ('queries', 'hits'):
        series = counters.get('vllm:prefix_cache_' + key, {}).get('series', [])
        prefix[key] = sum(item['stats']['total'] for item in series)
    gauge = {}
    for key in ('running', 'waiting'):
        series = counters.get('vllm:num_requests_' + key, {}).get('series', [])
        gauge[key] = sum(item['stats']['avg'] for item in series) if series else None
        if key == 'running':
            gauge['running_max'] = sum(item['stats']['max'] for item in series) if series else None
    return {
        'mean_s': latency['avg'] * scale if latency else None,
        'p90_s': p90,
        'completed': int(data.get('request_count', {}).get('avg', 0)),
        'failed': failed,
        'mean_output_tokens': data.get('output_sequence_length', {}).get('avg'),
        'throughput_rps': data.get('request_throughput', {}).get('avg', 0),
        'complete': complete,
        'server_running_avg': gauge['running'],
        'server_running_max': gauge['running_max'],
        'server_waiting_avg': gauge['waiting'],
        'spec_draft_tokens': spec['draft_tokens'],
        'spec_accepted_tokens': spec['accepted_tokens'],
        'spec_draft_steps': spec['drafts'],
        'prefix_cache_token_hit_rate': prefix['hits'] / prefix['queries'] if prefix['queries'] else None,
        'spec_acceptance_rate': spec['accepted_tokens'] / spec['draft_tokens'] if spec['draft_tokens'] else None,
        'sla_pass': complete and failed == 0 and p90 is not None and p90 < 15,
    }


def percentile_90(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.quantiles(values, n=10, method='inclusive')[8] if len(values) > 1 else values[0]


def audit_records(path: Path) -> dict:
    with path.open() as handle:
        records = [json.loads(line) for line in handle]
    records = [r for r in records if r['metadata']['benchmark_phase'] == 'profiling']
    success = [r for r in records if not r.get('error')]
    latencies = [r['metrics']['request_latency']['value'] / 1000 for r in success]
    issued = [r['metadata']['credit_issued_ns'] for r in records]
    mean = statistics.mean(latencies) if latencies else None
    span = (max(issued) - min(issued)) / 1e9 if issued else 0.0
    started = [r for r in records if r['metadata'].get('request_start_ns') is not None]
    starts = [r['metadata']['request_start_ns'] for r in started]
    start_span = (max(starts) - min(starts)) / 1e9 if starts else 0.0
    queue = [(r['metadata']['request_start_ns'] - r['metadata']['credit_issued_ns']) / 1e9 for r in started]
    effective = [(r['metadata']['request_end_ns'] - r['metadata']['credit_issued_ns']) / 1e9 for r in success]
    origin = min(issued) if issued else 0
    windows = [(origin + i * 60_000_000_000, origin + (i + 1) * 60_000_000_000) for i in range(5)]
    credit_counts = [sum(lo <= t < hi for t in issued) for lo, hi in windows]
    start_counts = [sum(lo <= t < hi for t in starts) for lo, hi in windows]
    unique = {}
    for field in ('session_num', 'x_request_id'):
        values = [r['metadata'].get(field) for r in records]
        unique[field] = None not in values and len(set(values)) == len(values)
    return {
        'records_completed': len(success),
        'records_failed': len(records) - len(success),
        'unique_request_ids': unique['x_request_id'],
        'unique_session_ids': unique['session_num'],
        'credit_counts_60s': credit_counts,
        'start_counts_60s': start_counts,
        'exact_mean_s': mean,
        'exact_p90_s': percentile_90(latencies),
        'effective_p90_s': percentile_90(effective),
        'client_queue_p90_s': percentile_90(queue),
        'client_queue_max_s': max(queue) if queue else None,
        'actual_started_count': len(starts),
        'actual_start_span_s': start_span,
        'actual_start_rpm': (len(starts) - 1) * 60 / start_span if start_span else 0.0,
        'issue_span_s': span,
        'offered_rpm': (len(records) - 1) * 60 / span if span else 0.0,
    }


def compare_outputs(baseline: Path, candidate: Path) -> dict:
    reference = {r['session_num']: r for r in json.loads(baseline.read_text())['data']}
    outputs = {r['session_num']: r for r in json.loads(candidate.read_text())['data']}
    if reference.keys() != outputs.keys():
        raise ValueError(f'Разный набор запросов: {candidate}')
    changed_inputs = sum(reference[k]['metrics']['input_sequence_length'] != outputs[k]['metrics']['input_sequence_length'] for k in reference)
    if changed_inputs:
        raise ValueError(f'Разные входы в сравнении: {candidate}')
    texts = [outputs[key]['response_text'] for key in sorted(outputs)]
    return {
        'compared_outputs': len(outputs),
        'input_length_mismatches': changed_inputs,
        'changed_texts': sum(reference[k]['response_text'] != outputs[k]['response_text'] for k in reference),
        'changed_token_lengths': sum(
            reference[k]['metrics']['output_sequence_length'] != outputs[k]['metrics']['output_sequence_length']
            for k in reference
        ),
        'output_text_sha256': hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
    }


def minute_rate_verified(counts: list[int], rpm: int) -> bool:
    return len(counts) == 5 and all(abs(n - i * rpm) <= 1 for i, n in enumerate(accumulate(counts), 1))


def main() -> None:
    rows = []
    for path in sorted(Path('outputs').glob('*/profile/profile_export_aiperf.json')):
        group, name, rpm = path.parents[1].name.rsplit('-', 2)
        row = {'group': group, 'config': name, 'rpm': int(rpm), **read_summary(path)}
        audit = audit_records(path.with_name('profile_export.jsonl'))
        assert audit['records_completed'] == row['completed'], path
        assert audit['records_failed'] == row['failed'], path
        if row['completed']:
            assert abs(audit['exact_mean_s'] - row['mean_s']) < 0.0001, path
            assert abs(audit['exact_p90_s'] - row['p90_s']) < 0.001, path
        if group in ('fixed', 'repeat', 'sweep', 'large', 'large5') and row['completed'] == 640 and row['failed'] == 0:
            audit.update(compare_outputs(
                Path('outputs/fixed-baseline-120/profile/outputs.json'),
                path.with_name('outputs.json'),
            ))
        if group == 'load':
            audit['duration_verified'] = audit['issue_span_s'] >= 299
            audit['rate_verified'] = abs(audit['actual_start_rpm'] - int(rpm)) < 0.01 * int(rpm)
            audit['count_verified'] = abs(audit['actual_started_count'] - 5 * int(rpm)) <= 1
            audit['bins_verified'] = all(minute_rate_verified(counts, int(rpm)) for counts in (audit['credit_counts_60s'], audit['start_counts_60s']))
            row['sla_pass'] = (row['sla_pass'] and audit['duration_verified'] and audit['rate_verified']
                               and audit['count_verified'] and audit['bins_verified']
                               and audit['unique_request_ids'] and audit['unique_session_ids']
                               and audit['effective_p90_s'] is not None
                               and audit['effective_p90_s'] < 15)
        rows.append({**row, **audit})
    if not rows:
        raise ValueError('Нет завершённых экспортов AIPerf')
    with Path('outputs/results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
