import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download


ARTIFACTS = Path(__file__).resolve().parent / 'artifacts_compression_hw'


def restore(artifacts: Path) -> None:
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    if not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
        raise ValueError('An immutable Hub revision is required')
    files = manifest['files']
    if not files:
        raise ValueError('The checkpoint manifest is empty')
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError(f'Unsafe artifact path: {name}')
    snapshot_download(
        repo_id=manifest['repo_id'], revision=manifest['revision'], token=False,
        allow_patterns=list(files), local_dir=artifacts,
    )
    for name, expected in files.items():
        path = artifacts / name
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size != expected['size'] or digest != expected['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    print(f"Verified {len(files)} files at {manifest['revision']}")


def main() -> None:
    restore(ARTIFACTS)


if __name__ == '__main__':
    main()
