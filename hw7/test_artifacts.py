import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import restore_artifacts


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.file = self.root / 'quantized/example/config.json'
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b'checkpoint')
        self.manifest = {
            'repo_id': 'mariklolik228/dl-course-hw7-compression',
            'revision': 'e' * 40,
            'files': {'quantized/example/config.json': {
                'size': 10, 'sha256': hashlib.sha256(b'checkpoint').hexdigest(),
            }},
        }

    def tearDown(self):
        self.folder.cleanup()

    def run_restore(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        with patch.object(restore_artifacts, 'ARTIFACTS', self.root):
            restore_artifacts.main()

    @patch('restore_artifacts.snapshot_download')
    def test_valid_checkpoint_is_verified(self, download):
        self.run_restore()
        self.assertEqual(download.call_args.kwargs['revision'], 'e' * 40)
        self.assertEqual(download.call_args.kwargs['allow_patterns'], list(self.manifest['files']))
        self.assertIs(download.call_args.kwargs['token'], False)

    @patch('restore_artifacts.snapshot_download')
    def test_corrupted_checkpoint_fails(self, download):
        self.file.write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            self.run_restore()

    @patch('restore_artifacts.snapshot_download')
    def test_path_escape_and_unpinned_revision_fail_before_download(self, download):
        original = self.manifest['files']
        self.manifest['files'] = {'../outside': next(iter(original.values()))}
        with self.assertRaises(ValueError):
            self.run_restore()
        self.manifest['files'] = original
        self.manifest['revision'] = 'main'
        with self.assertRaises(ValueError):
            self.run_restore()
        download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
