import argparse
import ast
import hashlib
import io
import json
import random
import re
import tokenize
from collections import Counter
from pathlib import Path

DATASET = 'Team-ACE/ToolACE'
DATASET_REVISION = '6bda777c88d21e5a204703c1ee45597a8fa4f734'
MODEL = 'Qwen/Qwen3.5-2B'
MODEL_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
MARKER = 'Here is a list of functions in JSON format that you can invoke:'
SYSTEM = 'Use the supplied tools when appropriate. Ask for missing required information. Otherwise answer normally. Do not invent arguments.'


def normalize_schema(schema):
    result = dict(schema)
    aliases = {'dict': 'object', 'int': 'integer', 'float': 'number', 'bool': 'boolean', 'list': 'array'}
    kind = result.get('type')
    if isinstance(kind, str):
        result['type'] = aliases.get(kind, kind)
    if 'properties' in result:
        result['properties'] = {k: normalize_schema(v) for k, v in result['properties'].items()}
    if isinstance(result.get('items'), dict):
        result['items'] = normalize_schema(result['items'])
    return result


def extract_tools(system):
    if MARKER not in system:
        raise ValueError('unsupported_tool_schema')
    raw, _ = json.JSONDecoder().raw_decode(system.split(MARKER, 1)[1].lstrip())
    names, tools = {}, []
    for index, tool in enumerate(raw):
        original = tool['name']
        if original in names:
            raise ValueError('duplicate_tool_name')
        name = re.sub(r'\W+', '_', original, flags=re.ASCII).strip('_') or 'tool'
        if name[0].isdigit():
            name = 'tool_' + name
        base, suffix = name, index
        while name in names.values():
            name = base + '_' + str(suffix)
            suffix += 1
        names[original] = name
        tools.append({'type': 'function', 'function': {'name': name, 'description': tool['description'], 'parameters': normalize_schema(tool['parameters'])}})
    return tools, names


def parse_calls(text, names):
    text = text.strip()
    if not (text.startswith('[') and text.endswith(']')):
        raise ValueError('invalid_call_list')
    remaining, calls = text[1:-1].strip(), []
    while remaining:
        original = next((name for name in sorted(names, key=len, reverse=True) if remaining.startswith(name + '(')), None)
        if original is None:
            raise ValueError('unknown_function')
        expression = 'f' + remaining[len(original):]
        offsets, position = [], 0
        for line in expression.splitlines(keepends=True):
            offsets.append(position)
            position += len(line)
        depth, end = 0, None
        try:
            for token in tokenize.generate_tokens(io.StringIO(expression).readline):
                if token.type == tokenize.OP:
                    depth += (token.string == '(') - (token.string == ')')
                    if token.string == ')' and depth == 0:
                        end = offsets[token.end[0] - 1] + token.end[1]
                        break
            if end is None:
                raise ValueError('unclosed_call')
            node = ast.parse(expression[:end], mode='eval').body
            if not isinstance(node, ast.Call) or node.args or any(k.arg is None for k in node.keywords):
                raise ValueError('unsupported_arguments')
            arguments = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            if len(arguments) != len(node.keywords):
                raise ValueError('duplicate_arguments')
            json.dumps(arguments, allow_nan=False)
        except (SyntaxError, TypeError, tokenize.TokenError, OverflowError) as error:
            raise ValueError('invalid_call_syntax') from error
        calls.append({'name': names[original], 'arguments': arguments})
        remaining = remaining[len(original) + end - 1:].strip()
        if remaining and not remaining.startswith(','):
            raise ValueError('invalid_call_separator')
        if remaining:
            remaining = remaining[1:].strip()
            if not remaining:
                raise ValueError('trailing_call_separator')
    return calls


def extract_examples(rows):
    examples, excluded, seen, mappings = [], Counter(), set(), {}
    schema_types, reference_errors = Counter(), Counter()
    audit = {'raw_dialogues': len(rows), 'duplicate_dialogues': 0, 'excluded': excluded}
    for row in rows:
        dialogue_id = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if dialogue_id in seen:
            audit['duplicate_dialogues'] += 1
            continue
        seen.add(dialogue_id)
        try:
            tools, names = extract_tools(row['system'])
        except (ValueError, KeyError, TypeError) as error:
            excluded[str(error)] += 1
            continue
        mappings.update({raw: new for raw, new in names.items() if raw != new})
        pending = [tool['function']['parameters'] for tool in tools]
        while pending:
            schema = pending.pop()
            schema_types[json.dumps(schema.get('type'))] += 1
            pending.extend(schema.get('properties', {}).values())
            if isinstance(schema.get('items'), dict):
                pending.append(schema['items'])
        prefix = row['system'].split(MARKER, 1)[0]
        context = '\n'.join(re.findall(r'(?:Today is|The current time is)[^\n]+', prefix))
        messages = [{'role': 'system', 'content': SYSTEM + ('\n' + context if context else '')}]
        for index, turn in enumerate(row['conversations']):
            role, text = turn['from'], turn['value']
            if role != 'assistant':
                messages.append({'role': role, 'content': text})
                continue
            try:
                calls = parse_calls(text, names) if text.strip().startswith('[') else []
            except ValueError as error:
                reference_errors[type(error).__name__ + ':' + str(error).split(':', 1)[0]] += 1
                excluded['invalid_reference_call'] += 1
                break
            if index and row['conversations'][index - 1]['from'] == 'user':
                examples.append({'id': f'{dialogue_id}:{index}', 'dialogue_id': dialogue_id, 'messages': list(messages), 'expected': json.dumps(calls, ensure_ascii=False), 'tools_json': json.dumps(tools, ensure_ascii=False), 'category': 'call' if calls else 'no_call'})
            else:
                excluded['post_tool_assistant'] += 1
            message = {'role': 'assistant', 'content': '' if calls else text}
            if calls:
                message['tool_calls'] = [{'type': 'function', 'function': call} for call in calls]
            messages.append(message)
    audit['excluded'] = dict(excluded)
    audit['name_mapping'] = mappings
    audit['normalized_schema_types'] = dict(schema_types)
    audit['reference_errors'] = dict(reference_errors)
    audit['system_conversion'] = 'Native tool instructions replace legacy call syntax; source date/time context is preserved.'
    audit['eligible_categories'] = dict(Counter(row['category'] for row in examples))
    audit['reference_call_counts'] = dict(Counter(len(json.loads(row['expected'])) for row in examples))
    return examples, audit


def split_dialogues(examples, seed=42, validation_fraction=0.15):
    train, validation = [], []
    for row in examples:
        digest = hashlib.sha256(f'{seed}:{row["dialogue_id"]}'.encode()).hexdigest()
        (validation if int(digest[:16], 16) / 16**16 < validation_fraction else train).append(row)
    return train, validation


def select_balanced(pool, size, no_call_fraction=0.25, seed=42):
    rng = random.Random(seed)
    groups = [[row for row in pool if row['category'] == category] for category in ('call', 'no_call')]
    for group in groups:
        rng.shuffle(group)
    n_no_call = min(round(size * no_call_fraction), len(groups[1]))
    n_call = min(size - n_no_call, len(groups[0]))
    n_no_call = min(size - n_call, len(groups[1]))
    selected = groups[0][:n_call] + groups[1][:n_no_call]
    rng.shuffle(selected)
    return selected


def distribution(values):
    values = sorted(values)
    if not values:
        return {}
    return {'count': len(values), **{f'p{q}': values[round((len(values) - 1) * q / 100)] for q in (0, 25, 50, 75, 90, 95, 99, 100)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data')
    parser.add_argument('--eda', default='outputs/eda.json')
    parser.add_argument('--train-size', type=int, default=2048)
    parser.add_argument('--validation-size', type=int, default=128)
    parser.add_argument('--max-prompt-tokens', type=int, default=3072)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-revision', default=DATASET_REVISION)
    parser.add_argument('--model-revision', default=MODEL_REVISION)
    parser.add_argument('--source-file')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    source = Path(args.source_file or hf_hub_download(DATASET, 'data.json', repo_type='dataset', revision=args.dataset_revision))
    raw = source.read_bytes()
    rows = json.loads(raw)
    examples, audit = extract_examples(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.model_revision)
    train_pool, validation_pool = split_dialogues(examples, args.seed)
    audit.update({'dataset': DATASET, 'dataset_revision': args.dataset_revision, 'source_sha256': hashlib.sha256(raw).hexdigest(), 'model': MODEL, 'model_revision': args.model_revision, 'seed': args.seed, 'max_prompt_tokens': args.max_prompt_tokens})
    audit['raw_roles'] = dict(Counter(turn['from'] for row in rows for turn in row['conversations']))
    audit['raw_dialogue_turns'] = distribution([len(row['conversations']) for row in rows])
    audit['raw_system_characters'] = distribution([len(row['system']) for row in rows])
    audit['raw_assistant_characters'] = distribution([len(t['value']) for row in rows for t in row['conversations'] if t['from'] == 'assistant'])
    audit['splits'] = {}
    retained_ids = {}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split, pool, size in [('train', train_pool, args.train_size), ('validation', validation_pool, args.validation_size)]:
        retained, lengths = [], []
        for row in pool:
            messages = row['messages']
            prompt = tokenizer.apply_chat_template(messages, tools=json.loads(row['tools_json']), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            length = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
            lengths.append(length)
            if length <= args.max_prompt_tokens:
                retained.append({k: v for k, v in row.items() if k != 'messages'} | {'prompt': prompt, 'prompt_tokens': length})
        selected = select_balanced(retained, size, seed=args.seed)
        if len(selected) != size:
            raise ValueError(f'{split}: requested {size}, available {len(selected)}')
        retained_ids[split] = {row['dialogue_id'] for row in selected}
        with (output / f'{split}.jsonl').open('w') as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        audit['splits'][split] = {'eligible': len(pool), 'eligible_categories': dict(Counter(row['category'] for row in pool)), 'prompt_tokens_before_filter': distribution(lengths), 'excluded_over_length': len(pool) - len(retained), 'retained': len(retained), 'retained_categories': dict(Counter(row['category'] for row in retained)), 'selected': len(selected), 'selected_categories': dict(Counter(row['category'] for row in selected)), 'selected_prompt_tokens': distribution([row['prompt_tokens'] for row in selected]), 'dialogues': len(retained_ids[split])}
    audit['split_dialogue_overlap'] = len(retained_ids['train'] & retained_ids['validation'])
    if audit['split_dialogue_overlap']:
        raise ValueError('dialogue_split_leakage')
    eda = Path(args.eda)
    eda.parent.mkdir(parents=True, exist_ok=True)
    eda.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('name_mapping',)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
