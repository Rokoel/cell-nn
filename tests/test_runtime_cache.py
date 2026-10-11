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


def external_bundle(root):
    item = bundle(root)
    archive = (root / 'runtime-libraries.tar.xz').read_bytes()
    parts = []
    for index, data in enumerate((archive[:len(archive)//2], archive[len(archive)//2:])):
        name = f'runtime.part{index}'
        (root / name).write_bytes(data)
        parts.append(dict(name=name, size=len(data), sha256=hashlib.sha256(data).hexdigest(), url=''))
    (root / 'runtime-cache.json').write_text(json.dumps(dict(id=cache_id([item]), files=[item], parts=parts)))
    (root / 'runtime-libraries.tar.xz').unlink()
    return item, parts


def test_http_download_multiple_parts_and_offline_reuse(tmp_path, monkeypatch):
    import functools
    import http.server
    import threading
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    item, parts = external_bundle(tmp_path)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest_path = tmp_path / 'runtime-cache.json'
    manifest = json.loads(manifest_path.read_text())
    for part in manifest['parts']:
        part['url'] = f'http://127.0.0.1:{server.server_port}/{part["name"]}'
    manifest_path.write_text(json.dumps(manifest))
    progress = []
    try:
        destination = prepare(tmp_path, lambda *values: progress.append(values))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert (destination / item['path']).read_bytes() == b'library contents'
    assert {p[0] for p in progress} == {'Downloading neural runtime', 'Installing neural runtime'}
    assert not list(destination.parent.glob('*.download'))
    assert prepare(tmp_path) == destination


def test_download_hash_failure_and_cancel_never_publish_cache(tmp_path, monkeypatch):
    from fish_viewer.runtime_cache import Cancelled
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('FISH_RUNTIME_DOWNLOAD_DIR', str(tmp_path))
    item, parts = external_bundle(tmp_path)
    with pytest.raises(Cancelled):
        prepare(tmp_path, cancelled=lambda: True)
    assert not list((tmp_path / 'cache').rglob('.ready'))
    (tmp_path / parts[0]['name']).write_bytes(b'X' * parts[0]['size'])
    with pytest.raises(ValueError, match='checksum'):
        prepare(tmp_path)
    assert not list((tmp_path / 'cache').rglob('.ready'))
    assert not list((tmp_path / 'cache').rglob('*.partial'))


def test_setup_worker_progress_and_result(tmp_path, monkeypatch):
    from fish_viewer.runtime_setup import SetupWorker
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    item = bundle(tmp_path)
    worker = SetupWorker(tmp_path)
    worker.run()
    assert worker.error is None
    assert (worker.result / item['path']).read_bytes() == b'library contents'
