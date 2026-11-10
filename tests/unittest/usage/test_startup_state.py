# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local consecutive startup failure count."""

import errno
import json
import os
import stat
import time
from pathlib import Path

import pytest

from tensorrt_llm.usage import startup_state

pytestmark = pytest.mark.cpu_only

_DAY = 24 * 3600
_KEY = "0123456789abcdef"


def _state_file(state_dir: Path) -> Path:
    return state_dir / f"startup-{_KEY}.json"


def _load(state_dir: Path) -> dict:
    return json.loads(_state_file(state_dir).read_text())


@pytest.mark.parametrize(
    "streak,expected",
    [
        (-1, 0),
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 3),
        (9, 3),
        (10, 10),
        (29, 10),
        (30, 30),
        (99, 30),
        (100, 100),
        (299, 100),
        (300, 300),
        (999, 300),
        (1000, 1000),
        (10**6, 1000),
    ],
)
def test_bucket_lower_bounds(streak, expected):
    assert startup_state.bucket(streak) == expected


class TestLaunchKey:
    _ENV = {"CUDA_VISIBLE_DEVICES": "0,1"}

    def _key(self, argv=("trtllm-serve", "model"), environ=None, version="1.4.0"):
        return startup_state.launch_key(
            version, argv=list(argv), environ=self._ENV if environ is None else environ
        )

    def test_is_deterministic_and_reveals_no_value(self):
        key = self._key()
        assert key == self._key()
        assert len(key) == 16 and int(key, 16) >= 0
        for value in ("trtllm-serve", "model", "0,1", "1.4.0"):
            assert value not in key

    @pytest.mark.parametrize(
        "name",
        [
            "SLURM_PROCID",
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "PMIX_RANK",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "TLLM_DISAGG_INSTANCE_IDX",
            "TRTLLM_DISAGG_WORKER_PROCESS_ID",
            "TLLM_EXECUTOR_FRONTEND_ID",
        ],
    )
    def test_separates_launch_variables(self, name):
        assert self._key(environ={**self._ENV, name: "7"}) != self._key()

    def test_separates_versions_and_command_lines(self):
        assert self._key(version="1.4.1") != self._key()
        assert self._key(argv=("trtllm-serve", "model", "--port", "8001")) != self._key()

    def test_ignores_per_start_and_unrelated_variables(self):
        environ = {
            **self._ENV,
            "SLURM_JOB_ID": "123",
            "TRTLLM_DISAGG_DEPLOYMENT_ID": "random",
            "HOME": "/x",
        }
        assert self._key(environ=environ) == self._key()


class TestStateDir:
    def test_precedence(self, monkeypatch, tmp_path):
        environ = {"TRTLLM_USAGE_STATE_DIR": str(tmp_path / "a"), "XDG_CACHE_HOME": "/cache"}
        assert startup_state.state_dir(environ) == tmp_path / "a"
        assert startup_state.state_dir({"XDG_CACHE_HOME": "/cache"}) == Path(
            "/cache/tensorrt_llm/usage"
        )
        monkeypatch.setattr(Path, "home", lambda: Path("/home/u"))
        assert startup_state.state_dir({"XDG_CACHE_HOME": "relative"}) == Path(
            "/home/u/.cache/tensorrt_llm/usage"
        )

    def test_missing_home_is_unavailable(self, monkeypatch):
        def no_home():
            raise RuntimeError("no home")

        monkeypatch.delenv("TRTLLM_USAGE_STATE_DIR")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", no_home)
        assert startup_state.state_dir() is None
        assert startup_state.consult(_KEY, "increment") == (0, "unavailable")


class TestConsult:
    def test_increment_creates_private_state(self, isolated_startup_state_dir):
        state_dir = isolated_startup_state_dir
        assert startup_state.consult(_KEY, "increment", now=1000.0) == (0, "new")
        assert _load(state_dir) == {"v": 1, "streak": 1, "last_failure": 1000}
        assert stat.S_IMODE(os.stat(state_dir).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(_state_file(state_dir)).st_mode) == 0o600

        assert startup_state.consult(_KEY, "increment", now=1010.0) == (1, "persisted")
        assert startup_state.consult(_KEY, "increment", now=1020.0) == (2, "persisted")
        assert _load(state_dir)["streak"] == 3
        assert sorted(os.listdir(state_dir)) == [_state_file(state_dir).name]

    def test_reset_returns_prior_and_writes_zero(self, isolated_startup_state_dir):
        state_dir = isolated_startup_state_dir
        for _ in range(4):
            startup_state.consult(_KEY, "increment", now=1000.0)
        assert startup_state.consult(_KEY, "reset", now=1001.0) == (4, "persisted")
        assert _load(state_dir) == {"v": 1, "streak": 0, "last_failure": 0}
        assert startup_state.consult(_KEY, "increment", now=1002.0) == (0, "persisted")

    def test_reset_of_a_new_launch_writes_the_file(self, isolated_startup_state_dir):
        assert startup_state.consult(_KEY, "reset") == (0, "new")
        assert _load(isolated_startup_state_dir)["streak"] == 0
        assert startup_state.consult(_KEY, "read") == (0, "persisted")

    def test_read_never_writes(self, isolated_startup_state_dir):
        state_dir = isolated_startup_state_dir
        assert startup_state.consult(_KEY, "read") == (0, "new")
        assert not state_dir.exists()

        startup_state.consult(_KEY, "increment", now=1000.0)
        before = _state_file(state_dir).stat()
        assert startup_state.consult(_KEY, "read", now=1001.0) == (1, "persisted")
        after = _state_file(state_dir).stat()
        assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)

    def test_old_streak_expires(self, isolated_startup_state_dir):
        now = time.time()
        startup_state.consult(_KEY, "increment", now=now - 2 * _DAY)
        assert startup_state.consult(_KEY, "read", now=now) == (0, "persisted")
        assert startup_state.consult(_KEY, "increment", now=now) == (0, "persisted")
        assert _load(isolated_startup_state_dir)["streak"] == 1

    @pytest.mark.parametrize(
        "content",
        [
            b"not json",
            b"[]",
            b'{"v": 2, "streak": 3, "last_failure": 0}',
            b'{"v": 1, "streak": -1, "last_failure": 0}',
            b'{"v": 1, "streak": "3", "last_failure": 0}',
            b'{"v": 1, "streak": true, "last_failure": 0}',
            b'{"v": 1, "streak": 3}',
            b'{"v": 1, "streak": 3, "last_failure": 0, "pad": "' + b"x" * 5000 + b'"}',
        ],
    )
    def test_corrupt_state_restarts_from_zero(self, isolated_startup_state_dir, content):
        state_dir = isolated_startup_state_dir
        state_dir.mkdir()
        _state_file(state_dir).write_bytes(content)
        assert startup_state.consult(_KEY, "increment") == (0, "new")
        assert _load(state_dir)["streak"] == 1

    def test_unusable_directory_is_unavailable(self, monkeypatch, tmp_path):
        # A directory below a regular file fails with ENOTDIR, also for root.
        blocker = tmp_path / "file"
        blocker.write_text("")
        monkeypatch.setenv("TRTLLM_USAGE_STATE_DIR", str(blocker / "state"))
        for action in ("read", "increment", "reset"):
            assert startup_state.consult(_KEY, action) == (0, "unavailable")

    def test_failed_write_keeps_prior_and_leaves_no_temporary_file(
        self, monkeypatch, isolated_startup_state_dir
    ):
        state_dir = isolated_startup_state_dir
        startup_state.consult(_KEY, "increment", now=1000.0)

        def read_only(*args, **kwargs):
            raise OSError(errno.EROFS, "Read-only file system")

        monkeypatch.setattr(startup_state.os, "replace", read_only)
        assert startup_state.consult(_KEY, "increment", now=1001.0) == (1, "unavailable")
        assert sorted(os.listdir(state_dir)) == [_state_file(state_dir).name]
        assert _load(state_dir)["streak"] == 1

    def test_unexpected_error_is_swallowed(self, monkeypatch):
        def broken(*args, **kwargs):
            raise ValueError("unexpected")

        monkeypatch.setattr(startup_state, "_read", broken)
        assert startup_state.consult(_KEY, "increment") == (0, "unavailable")

    def test_empty_key_is_unavailable(self, isolated_startup_state_dir):
        assert startup_state.consult("", "increment") == (0, "unavailable")
        assert not isolated_startup_state_dir.exists()

    def test_new_file_prunes_only_old_state_files(self, isolated_startup_state_dir):
        state_dir = isolated_startup_state_dir
        state_dir.mkdir()
        old = time.time() - 31 * _DAY
        names = {
            "startup-aaaaaaaaaaaaaaaa.json": old,
            ".startup-left.tmp": old,
            "startup-bbbbbbbbbbbbbbbb.json": time.time(),
            "user-notes.json": old,
        }
        for name, mtime in names.items():
            (state_dir / name).write_text("{}")
            os.utime(state_dir / name, (mtime, mtime))

        startup_state.consult(_KEY, "increment")
        assert sorted(os.listdir(state_dir)) == sorted(
            [_state_file(state_dir).name, "startup-bbbbbbbbbbbbbbbb.json", "user-notes.json"]
        )
