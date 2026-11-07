import os

import pytest

from mdvoro.utils import read_config as rc


@pytest.fixture
def files(tmp_path, monkeypatch):
    """config.yaml, user and local config in tmp_path; environment cleared"""
    base, user, local = tmp_path / 'config.yaml', tmp_path / 'user.yaml', tmp_path / 'config.local.yaml'
    monkeypatch.setattr(rc, 'config_files', lambda: [str(base), str(user), str(local)])
    for var in rc.ENV.values():
        monkeypatch.delenv(var, raising=False)
    base.write_text('tetgen: tetgen\nvoro++: voro++\n')
    return base, user, local


def test_defaults(files):
    c = rc.read_config()
    assert (c.alias_tetgen, c.alias_voropp) == ('tetgen', 'voro++')


def test_override_order(files, monkeypatch):
    base, user, local = files
    user.write_text('tetgen: /user/tetgen\nvoro++: /user/voro++\n')
    local.write_text('tetgen: /local/tetgen\n')        # one key only
    c = rc.read_config()
    assert (c.alias_tetgen, c.alias_voropp) == ('/local/tetgen', '/user/voro++')
    monkeypatch.setenv('MDVORO_VOROPP', '/env/voro++')
    assert rc.read_config().alias_voropp == '/env/voro++'


def test_expands_home_and_vars(files, monkeypatch):
    _, _, local = files
    monkeypatch.setenv('TG_DIR', '/opt/tg')
    local.write_text('tetgen: $TG_DIR/tetgen\nvoro++: ~/bin/voro++\n')
    c = rc.read_config()
    assert c.alias_tetgen == '/opt/tg/tetgen'
    assert c.alias_voropp == os.path.expanduser('~/bin/voro++')


def test_empty_and_missing_files(files):
    base, user, _ = files
    user.write_text('')                                 # empty file: ignored
    base.unlink()                                       # no config at all: attributes stay None
    c = rc.read_config()
    assert (c.alias_tetgen, c.alias_voropp) == (None, None)


def test_tracked_config_has_no_local_paths():
    with open(os.path.join(rc.ROOT_DIR, 'config.yaml')) as f:
        text = f.read()
    assert '/home/' not in text and 'tetgen: tetgen' in text
