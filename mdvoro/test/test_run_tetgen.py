import os
import stat

import pytest

from mdvoro.mesh import run_tetgen as rt


class _Config:
    def __init__(self, exe):
        self.alias_tetgen = exe


def _fake_tetgen(tmp_path, body):
    exe = tmp_path / 'tetgen'
    exe.write_text('#!/bin/sh\n' + body + '\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_timeout_scales_with_input_size(tmp_path, monkeypatch):
    monkeypatch.delenv('MDVORO_TETGEN_TIMEOUT', raising=False)
    small = tmp_path / 'small.poly'
    small.write_bytes(b'x')
    big = tmp_path / 'big.poly'
    big.write_bytes(b'x' * 40 * 2**20)
    assert rt.tetgen_timeout(small) == 600.0
    assert rt.tetgen_timeout(big) == pytest.approx(1200.0)
    monkeypatch.setenv('MDVORO_TETGEN_TIMEOUT', '0')
    assert rt.tetgen_timeout(small) is None


def test_hung_tetgen_raises_instead_of_blocking(tmp_path, monkeypatch):
    poly = tmp_path / 'temp.poly'
    poly.write_text('')
    monkeypatch.setattr(rt, 'read_config', lambda: _Config(_fake_tetgen(tmp_path, 'sleep 30')))
    monkeypatch.setenv('MDVORO_TETGEN_TIMEOUT', '0.5')
    with pytest.raises(RuntimeError, match='did not finish'):
        rt.plc_self_intersects(str(poly))
    with pytest.raises(RuntimeError, match='did not finish'):
        rt.run_tetgen(str(poly), '-pA')


def test_clean_plc_reports_nothing(tmp_path, monkeypatch):
    poly = tmp_path / 'temp.poly'
    poly.write_text('')
    monkeypatch.setattr(rt, 'read_config', lambda: _Config(_fake_tetgen(tmp_path, 'exit 0')))
    assert rt.plc_self_intersects(str(poly)).shape == (0, 3)
