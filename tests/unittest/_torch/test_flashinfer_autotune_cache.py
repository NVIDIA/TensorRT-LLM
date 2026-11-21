# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-rank FlashInfer autotune cache file: what each ``autotune`` call hands FlashInfer."""

import os
import sys
import types

import pytest

from tensorrt_llm._torch import flashinfer_autotune_cache as fac

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def calls(monkeypatch):
    seen = []
    fake = types.ModuleType("flashinfer")
    fake.autotune = lambda tune_mode=True, cache=None, *a, **k: seen.append((tune_mode, cache, k))
    monkeypatch.setitem(sys.modules, "flashinfer", fake)
    monkeypatch.setattr(fac, "mpi_rank", lambda: 3)
    monkeypatch.setattr(fac, "_STATE", {"loaded": False, "logged": False, "bundled": None})
    for env in (fac.CACHE_DIR_ENV, fac.FROZEN_ENV):
        monkeypatch.delenv(env, raising=False)
    return seen


def test_unset_is_flashinfers_own_call(calls):
    fac.autotune(tune_mode=True)
    fac.autotune(tune_mode=False, skip_ops="mxfp8_gemm")
    assert calls == [(True, None, {}), (False, None, {"skip_ops": "mxfp8_gemm"})]


def test_learning_mode_tunes_into_this_ranks_file(calls, monkeypatch, tmp_path):
    monkeypatch.setenv(fac.CACHE_DIR_ENV, str(tmp_path))
    fac.autotune(tune_mode=True)
    fac.autotune(tune_mode=True)
    path = str(tmp_path / "rank3.json")
    assert calls == [(True, path, {}), (True, path, {})]


def test_learning_mode_reloads_the_file_only_when_it_changes(calls, monkeypatch, tmp_path):
    monkeypatch.setenv(fac.CACHE_DIR_ENV, str(tmp_path))
    path = tmp_path / "rank3.json"
    path.write_text("{}")
    fac.autotune(tune_mode=False)
    fac.autotune(tune_mode=False)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    fac.autotune(tune_mode=False)
    fac.autotune(tune_mode=False)
    monkeypatch.setenv(fac.CACHE_DIR_ENV, str(tmp_path / "other"))
    fac.autotune(tune_mode=False)
    other = str(tmp_path / "other" / "rank3.json")
    assert [c[1] for c in calls] == [str(path), None, str(path), None, other]


def test_frozen_mode_loads_the_file_once_and_never_tunes(calls, monkeypatch, tmp_path):
    monkeypatch.setenv(fac.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(fac.FROZEN_ENV, "1")
    fac.autotune(tune_mode=True)
    fac.autotune(tune_mode=True)
    fac.autotune(tune_mode=False, skip_ops="mxfp8_gemm")
    path = str(tmp_path / "rank3.json")
    assert calls == [
        (False, path, {}),
        (False, None, {}),
        (False, None, {"skip_ops": "mxfp8_gemm"}),
    ]


def test_an_explicit_cache_is_left_alone(calls, monkeypatch, tmp_path):
    monkeypatch.setenv(fac.CACHE_DIR_ENV, str(tmp_path))
    fac.autotune(True, "/elsewhere.json")
    assert calls == [(True, "/elsewhere.json", {})]


def test_an_explicit_cache_makes_the_next_call_reload_this_ranks_file(calls, monkeypatch, tmp_path):
    monkeypatch.setenv(fac.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(fac.FROZEN_ENV, "1")
    fac.autotune(tune_mode=True)
    fac.autotune(tune_mode=True, cache="B")
    fac.autotune(tune_mode=True)
    path = str(tmp_path / "rank3.json")
    assert [c[1] for c in calls] == [path, "B", path]


def test_bundled_file_is_loaded_once_without_a_cache_dir(calls, tmp_path):
    bundle = str(tmp_path / "bundle.json")
    fac.use_bundled_file(bundle)
    fac.autotune(tune_mode=True)
    fac.autotune(tune_mode=True)
    assert calls == [(False, bundle, {}), (False, None, {})]
