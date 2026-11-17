import hashlib
import json
import re
from pathlib import Path


SEED = 42
BASE_MODEL_ID = 'Qwen/Qwen3-8B'
BASE_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
TEACHER_MODEL_ID = 'Qwen/Qwen3-32B'
TEACHER_REVISION = '9216db5781bf21249d130ec9da846c4624c16137'
DATASET_REVISION = 'ed58ac423a2f4121720918bf5301577edce4ffd3'
ARTIFACTS = Path('artifacts_compression_hw')
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 128
GENERATE_BATCH_SIZE = 4
SLOT_PATTERN = re.compile(r'\[([^:\[\]]+?)\s*:\s*([^\[\]]*?)\]')


def extract_slots(annot_utt: str) -> list[list[str]]:
    return [[m.group(1).strip(), m.group(2).strip()]
            for m in SLOT_PATTERN.finditer(annot_utt)]


def target_json(intent: str, slots: list[list[str]]) -> str:
    return json.dumps({'intent': intent, 'slots': slots}, ensure_ascii=False, separators=(',', ':'))


def system_prompt(intents: list[str], slot_types: list[str]) -> str:
    return (
        'Определи intent и извлеки slots. Верни только JSON вида '
        '{"intent":"...","slots":[["тип","значение"],...]}. '
        'Значения slots копируй из запроса. Если slots нет, верни пустой список. '
        'Не добавляй объяснения.\nДопустимые intents: ' + ', '.join(intents)
        + '\nДопустимые типы slots: ' + ', '.join(slot_types)
    )


def encode_prompt(tokenizer, text: str, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': text}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f'Prompt имеет {len(ids)} токенов, лимит {MAX_PROMPT_TOKENS}')
    return ids


def load_data():
    from datasets import Dataset, load_dataset

    folder = ARTIFACTS / 'data'
    manifest = folder / 'manifest.json'
    if manifest.exists():
        info = json.loads(manifest.read_text())
        assert info['revision'] == DATASET_REVISION and info['seed'] == SEED
        rows = {}
        for split, digest in info['sha256'].items():
            source = folder / f'{split}.jsonl'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, source
            rows[split] = Dataset.from_list([json.loads(line) for line in source.read_text().splitlines()])
        return rows['train'], rows['dev'], info['intents'], info['slot_types']
    url = f'hf://datasets/AmazonScience/massive@{DATASET_REVISION}/ru-RU'
    raw = load_dataset('parquet', data_files={
        split: f'{url}/{split}/0000.parquet' for split in ('train', 'validation')
    })
    intents = list(raw['train'].features['intent'].names)
    slot_types = sorted({kind for row in raw['train'] for kind, value in extract_slots(row['annot_utt'])})
    converted = {}
    for split in ('train', 'validation'):
        converted[split] = Dataset.from_list([
            {'id': str(row['id']), 'text': row['utt'],
             'gold': target_json(intents[row['intent']], extract_slots(row['annot_utt']))}
            for row in raw[split]
        ])
    rows = {'train': converted['train'], 'dev': converted['validation'].shuffle(seed=SEED).select(range(256))}
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, dataset in rows.items():
        source = folder / f'{split}.jsonl'
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in dataset)
        source.with_suffix('.tmp').write_text(content)
        source.with_suffix('.tmp').replace(source)
        hashes[split] = hashlib.sha256(source.read_bytes()).hexdigest()
    info = {'dataset': 'AmazonScience/massive', 'locale': 'ru-RU', 'revision': DATASET_REVISION,
            'seed': SEED, 'sha256': hashes, 'intents': intents, 'slot_types': slot_types,
            'train_size': len(rows['train']), 'dev_size': len(rows['dev']), 'test_used': False}
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    return rows['train'], rows['dev'], intents, slot_types


if __name__ == '__main__':
    train, dev, intents, slots = load_data()
    print(json.dumps({'train': len(train), 'dev': len(dev), 'intents': len(intents), 'slot_types': len(slots)}))
