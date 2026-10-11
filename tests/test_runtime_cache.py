import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

from fish_viewer.runtime_cache import cache_id, prepare

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux bundled runtime cache')


def bundle(root, content=b'library contents', path='nvidia/cudnn/lib/library.so'):
    files = [dict(path=path, size=len(content), sha256=hashlib.sha256(content).hexdigest())]
    (root / 'runtime-cache.json').write_text(json.dumps(dict(id=cache_id(files), files=files)))
    with tarfile.open(root / 'runtime-libraries.tar.xz', 'w:xz') as archive:
        info = tarfile.TarInfo(path)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return files[0]


def test_unpack_reuse_and_repair_incomplete_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    item = bundle(tmp_path)
    destination = prepare(tmp_path)
    library = destination / item['path']
    assert library.read_bytes() == b'library contents'
    assert prepare(tmp_path) == destination
    library.write_bytes(b'partial')
    assert prepare(tmp_path) == destination
    assert library.read_bytes() == b'library contents'
    assert not list(destination.parent.glob(destination.name + '.*' + '/'))


def test_corrupt_archive_never_becomes_ready(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    item = bundle(tmp_path)
    item['sha256'] = '0' * 64
    (tmp_path / 'runtime-cache.json').write_text(json.dumps(dict(id=cache_id([item]), files=[item])))
    with pytest.raises(ValueError, match='checksum'):
        prepare(tmp_path)
    assert not list((tmp_path / 'cache').rglob('.ready'))


def test_archive_cannot_escape_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    bundle(tmp_path, path='../outside.so')
    with pytest.raises(ValueError, match='path'):
        prepare(tmp_path)
    assert not (tmp_path / 'outside.so').exists()
