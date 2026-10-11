"""Unpack bundled neural libraries once, before the dynamic loader starts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


def cache_id(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def prepare(root):
    import fcntl
    root = Path(root)
    manifest = json.loads((root / 'runtime-cache.json').read_text())
    files = manifest['files']
    key = cache_id(files)
    if manifest['id'] != key:
        raise ValueError('Invalid bundled runtime manifest.')
    for item in files:
        path = Path(item['path'])
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('Invalid bundled runtime path.')
    base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'fish-analysis/runtimes'
    base.mkdir(parents=True, exist_ok=True)
    destination = base / key
    with (base / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = destination / '.ready'
        if marker.is_file() and marker.read_text() == key and all(
            (destination / item['path']).is_file() and
            (destination / item['path']).stat().st_size == item['size'] for item in files
        ):
            return destination
        with tempfile.TemporaryDirectory(prefix=key + '.', dir=base) as staging:
            staging = Path(staging)
            expected = {item['path']: item for item in files}
            with tarfile.open(root / 'runtime-libraries.tar.xz', 'r:xz') as archive:
                for member in archive:
                    item = expected.pop(member.name, None)
                    if item is None or not member.isfile() or member.size != item['size']:
                        raise ValueError('Unexpected file in bundled runtime archive.')
                    target = staging / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as source, target.open('wb') as output:
                        while block := source.read(1024 * 1024):
                            digest.update(block)
                            output.write(block)
                    if digest.hexdigest() != item['sha256']:
                        raise ValueError('Bundled runtime library checksum mismatch.')
                    target.chmod(0o755)
            if expected:
                raise ValueError('Bundled runtime archive is incomplete.')
            (staging / '.ready').write_text(key)
            if destination.exists():
                shutil.rmtree(destination)
            staging.rename(destination)
    return destination
