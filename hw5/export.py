import argparse
import hashlib
import json
from pathlib import Path


def export_targets(root: Path, selected_step: int, step: int | None = None) -> tuple:
    suffix = f'-step{step}' if step is not None else ''
    return selected_step if step is None else step, root / f'merged{suffix}', root / f'export_verification{suffix}.json'


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['rl1', 'rl2', 'rl3'])
    parser.add_argument('--step', type=int)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    root = args.root or Path('outputs') / args.name
    summary = json.loads((root / 'training_summary.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    step, target, receipt = export_targets(root, summary['selected_step'], args.step)
    adapter = root / 'adapters' / f'checkpoint-{step}'
    base = AutoModelForImageTextToText.from_pretrained(
        config['model'], revision=config['revision'], dtype=torch.bfloat16, device_map='cpu',
    )
    adapted = PeftModel.from_pretrained(base, adapter)
    norms = {name: float(p.float().norm()) for name, p in adapted.named_parameters() if 'lora_B' in name}
    if not norms or not any(norm > 0 for norm in norms.values()):
        raise RuntimeError('The selected adapter has no policy update')
    merged = adapted.merge_and_unload()
    merged.save_pretrained(target, safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(config['model'], revision=config['revision']).save_pretrained(target)
    hashes = {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
              for p in target.iterdir() if p.is_file()}
    receipt.write_text(json.dumps({
        'selected_step': summary['selected_step'], 'exported_step': step, 'base_revision': config['revision'],
        'merged': True, 'lora_b_norms': norms, 'files_sha256': hashes,
    }, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
