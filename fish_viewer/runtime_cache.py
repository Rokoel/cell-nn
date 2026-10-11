"""Download and verify neural libraries once, before the dynamic loader starts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import ssl
import urllib.request
from contextlib import ExitStack


class Cancelled(RuntimeError):
    pass


class PartsReader:
    def __init__(self, streams):
        self.streams = iter(streams)
        self.current = next(self.streams, None)

    def read(self, size):
        blocks = []
        while self.current is not None and size > 0:
            block = self.current.read(size)
            if block:
                blocks.append(block)
                size -= len(block)
            else:
                self.current = next(self.streams, None)
        return b"".join(blocks)


def download_parts(manifest, base, progress, cancelled):
    import certifi
    paths = []
    total = sum(part["size"] for part in manifest["parts"])
    completed = 0
    for part in manifest["parts"]:
        name = part["name"]
        if Path(name).name != name:
            raise ValueError("Invalid runtime asset name.")
        target = base / (part["sha256"] + ".download")
        def valid():
            if not target.is_file() or target.stat().st_size != part["size"]:
                return False
            with target.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest() == part["sha256"]
        if not valid():
            seed = os.environ.get("FISH_RUNTIME_DOWNLOAD_DIR")
            source = (Path(seed) / name).open("rb") if seed else urllib.request.urlopen(
                part["url"], timeout=60, context=ssl.create_default_context(cafile=certifi.where()))
            temporary = target.with_suffix(".partial")
            digest = hashlib.sha256()
            size = 0
            try:
                with source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        if cancelled():
                            raise Cancelled("Runtime setup cancelled. Restart the app to retry.")
                        size += len(block)
                        if size > part["size"]:
                            raise ValueError("Runtime download exceeds expected size.")
                        digest.update(block)
                        output.write(block)
                        progress("Downloading neural runtime", completed + size, total)
                if size != part["size"] or digest.hexdigest() != part["sha256"]:
                    raise ValueError("Runtime download checksum mismatch.")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        completed += part["size"]
        progress("Downloading neural runtime", completed, total)
        paths.append(target)
    return paths


def cache_id(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def prepare(root, progress=lambda *args: None, cancelled=lambda: False):
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
            parts = download_parts(manifest, base, progress, cancelled) if manifest.get('parts') else [root / 'runtime-libraries.tar.xz']
            total = sum(item['size'] for item in files)
            completed = 0
            with ExitStack() as stack:
                streams = [stack.enter_context(path.open('rb')) for path in parts]
                archive = stack.enter_context(tarfile.open(fileobj=PartsReader(streams), mode='r|xz'))
                for member in archive:
                    item = expected.pop(member.name, None)
                    if item is None or not member.isfile() or member.size != item['size']:
                        raise ValueError('Unexpected file in bundled runtime archive.')
                    target = staging / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as source, target.open('wb') as output:
                        while block := source.read(1024 * 1024):
                            if cancelled():
                                raise Cancelled('Runtime setup cancelled. Restart the app to retry.')
                            digest.update(block)
                            output.write(block)
                            completed += len(block)
                            progress('Installing neural runtime', completed, total)
                    if digest.hexdigest() != item['sha256']:
                        raise ValueError('Bundled runtime library checksum mismatch.')
                    target.chmod(0o755)
            if expected:
                raise ValueError('Bundled runtime archive is incomplete.')
            (staging / '.ready').write_text(key)
            if destination.exists():
                shutil.rmtree(destination)
            staging.rename(destination)
            if manifest.get("parts"):
                for path in parts:
                    path.unlink(missing_ok=True)
    return destination
