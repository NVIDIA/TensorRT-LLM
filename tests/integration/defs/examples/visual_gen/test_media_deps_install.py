# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast CPU unit tests for `_visual_gen_deps` wheel setup, teardown, and timeout fallback."""

import importlib.util
import os
import shutil
import subprocess

import defs.examples.visual_gen.conftest as vg_conftest
import pytest
from defs.examples.visual_gen.conftest import _check_output_new_session, _visual_gen_deps


def _which_on_injected_path(cmd: str) -> str | None:
    """Resolve `cmd` only from non-system entries prepended to `PATH`."""
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or entry.startswith(("/usr", "/bin", "/sbin")):
            continue
        cand = os.path.join(entry, cmd)
        if os.path.isfile(cand):
            return cand
    return None


def test_visual_gen_deps_wheel_setup_restores_path_and_cleans_tempdir(monkeypatch, tmp_path):
    """Wheel setup exposes `ffmpeg` on `PATH` and restores `PATH` / removes tempdir on teardown."""
    fake_exe = tmp_path / "ffmpeg-wheel-bin"
    fake_exe.write_text("#!/bin/sh\nexit 0\n")
    fake_exe.chmod(0o755)
    orig_path = "/custom/test/bin:/usr/bin"
    monkeypatch.setenv("PATH", orig_path)
    monkeypatch.setenv("TRTLLM_AUTO_INSTALL_MEDIA_DEPS", "1")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        shutil, "which", lambda cmd: _which_on_injected_path(cmd) if cmd == "ffmpeg" else None
    )
    apt_calls = []
    monkeypatch.setattr(vg_conftest, "install_ffmpeg_via_apt", lambda: apt_calls.append(1))

    class _StubVenv:
        def __init__(self):
            self.calls = []

        def run_cmd(self, args, **kwargs):
            self.calls.append((args, kwargs))
            if args[:1] == ["-c"]:
                return f"{fake_exe}\n"
            return 0

    venv = _StubVenv()
    gen = _visual_gen_deps.__wrapped__(venv, None)
    next(gen)
    resolved = shutil.which("ffmpeg")
    assert resolved is not None and os.path.islink(resolved)
    assert os.readlink(resolved) == str(fake_exe)
    bin_dir = os.path.dirname(resolved)
    assert len(venv.calls) == 2
    assert venv.calls[0] == (["-m", "pip", "install", "imageio-ffmpeg==0.8.1"], {"timeout": 240})
    assert venv.calls[1][1] == {
        "caller": _check_output_new_session,
        "print_script": False,
        "timeout": 60,
    }
    assert apt_calls == []

    with pytest.raises(StopIteration):
        next(gen)
    assert os.environ["PATH"] == orig_path
    assert not os.path.exists(bin_dir)


def test_visual_gen_deps_timeout_falls_back_to_apt_without_probing_wheel(monkeypatch):
    """A timed-out wheel install falls back to `install_ffmpeg_via_apt` without probing the wheel."""
    orig_path = "/custom/test/bin:/usr/bin"
    monkeypatch.setenv("PATH", orig_path)
    monkeypatch.setenv("TRTLLM_AUTO_INSTALL_MEDIA_DEPS", "1")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    apt_calls = []
    monkeypatch.setattr(vg_conftest, "install_ffmpeg_via_apt", lambda: apt_calls.append(1))

    class _TimedOutInstallVenv:
        def __init__(self):
            self.calls = []

        def run_cmd(self, args, **kwargs):
            self.calls.append((args, kwargs))
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))

    venv = _TimedOutInstallVenv()
    gen = _visual_gen_deps.__wrapped__(venv, None)
    next(gen)
    assert len(venv.calls) == 1
    assert apt_calls == [1]
    with pytest.raises(StopIteration):
        next(gen)
    assert os.environ["PATH"] == orig_path

    class _TimedOutLookupVenv:
        def __init__(self):
            self.calls = []

        def run_cmd(self, args, **kwargs):
            self.calls.append((args, kwargs))
            if args[:1] == ["-c"]:
                raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))
            return 0

    lookup_venv = _TimedOutLookupVenv()
    gen2 = _visual_gen_deps.__wrapped__(lookup_venv, None)
    next(gen2)
    assert len(lookup_venv.calls) == 2
    assert apt_calls == [1, 1]
    with pytest.raises(StopIteration):
        next(gen2)
    assert os.environ["PATH"] == orig_path
