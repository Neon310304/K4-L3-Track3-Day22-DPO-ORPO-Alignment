import hashlib
import json
import zipfile

import pytest

from export_training_checkpoint import export_checkpoint


def trained(root):
    for adapter in ("sft-mini", "dpo"):
        folder = root / "adapters" / adapter
        folder.mkdir(parents=True)
        (folder / "adapter_config.json").write_text('{}')
        (folder / "adapter_model.safetensors").write_bytes(b'actual-adapter-' + adapter.encode())
    merged = root / "models/sft-merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{}')
    (merged / "model.safetensors").write_bytes(b'large-policy-excluded')
    (root / '.env').write_text('PRIVATE=excluded')
    (root / 'adapters/dpo/.env').write_text('PRIVATE=excluded')
    cache = root / '.cache'
    cache.mkdir()
    (cache / 'download.json').write_text('{}')


def test_backup_recovers_exact_adapter_bytes_without_cache_secrets_or_merged_weights(tmp_path):
    trained(tmp_path)
    output = tmp_path / 'backup.zip'
    manifest = export_checkpoint(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read('training-backup-manifest.json')) == manifest
        assert set(archive.namelist()) == set(manifest['files']) | {'training-backup-manifest.json'}
        assert 'models/sft-merged/config.json' in archive.namelist()
        assert not any('.env' in name or '.cache' in name or name == 'models/sft-merged/model.safetensors' for name in archive.namelist())
        for name, record in manifest['files'].items():
            payload = archive.read(name)
            assert payload == (tmp_path / name).read_bytes()
            assert record == {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    assert manifest['complete_submission'] is False
    assert manifest['contains_lora_weights'] is True


def test_existing_backup_is_preserved(tmp_path):
    trained(tmp_path)
    output = tmp_path / 'backup.zip'
    output.write_bytes(b'prior')
    with pytest.raises(FileExistsError):
        export_checkpoint(tmp_path, output)
    assert output.read_bytes() == b'prior'


def test_missing_training_weights_does_not_create_a_false_backup(tmp_path):
    trained(tmp_path)
    (tmp_path / 'adapters/dpo/adapter_model.safetensors').unlink()
    with pytest.raises(ValueError, match='dpo'):
        export_checkpoint(tmp_path, tmp_path / 'backup.zip')
    assert not (tmp_path / 'backup.zip').exists()
