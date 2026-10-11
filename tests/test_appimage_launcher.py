"""Offline runtime discovery in the actual AppImage shell launcher."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux AppImage launcher')


@pytest.mark.parametrize('adjacent,override', [(True, None), (False, None), (True, 'explicit assets')])
def test_launcher_discovers_adjacent_parts_and_preserves_override(tmp_path, adjacent, override):
    appdir = tmp_path / 'extracted app'
    internal = appdir / 'usr/FISH-Analysis/_internal'
    internal.mkdir(parents=True)
    (internal / 'runtime-cache.json').write_text('{}')
    launcher = appdir / 'AppRun'
    shutil.copy2(Path(__file__).parents[1] / 'packaging/AppRun', launcher)
    launcher.chmod(0o755)
    binary = internal.parent / 'FISH-Analysis'
    binary.write_text('''#!/bin/sh
set -eu
if [ "$1" = "--prepare-runtime" ]; then
    printf '%s' "${FISH_RUNTIME_DOWNLOAD_DIR-}" > "$FISH_TEST_CAPTURE"
    printf '%s\\n' "$FISH_TEST_CACHE"
else
    printf '%s\\n' "$@" > "$FISH_TEST_ARGUMENTS"
fi
''')
    binary.chmod(0o755)
    cache = tmp_path / 'cache'
    cache.mkdir()
    release = tmp_path / 'release assets'
    release.mkdir()
    image = release / 'FISH-Analysis-test-linux-x64.AppImage'
    image.touch()
    if adjacent:
        image.with_suffix('.runtime.part01').touch()
    captured = tmp_path / 'seed.txt'
    arguments = tmp_path / 'arguments.txt'
    env = dict(os.environ, APPIMAGE=str(image), FISH_TEST_CAPTURE=str(captured), FISH_TEST_CACHE=str(cache), FISH_TEST_ARGUMENTS=str(arguments))
    env.pop('FISH_RUNTIME_DOWNLOAD_DIR', None)
    if override:
        env['FISH_RUNTIME_DOWNLOAD_DIR'] = str(tmp_path / override)
    subprocess.run([str(launcher), '--check-runtime', '--device', 'cpu'], cwd=tmp_path, env=env, check=True)
    expected = str(tmp_path / override) if override else str(release) if adjacent else ''
    assert captured.read_text() == expected
    assert arguments.read_text().splitlines() == ['--check-runtime', '--device', 'cpu']
