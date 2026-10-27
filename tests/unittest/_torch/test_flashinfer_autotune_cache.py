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
"""The per-rank FlashInfer autotune cache file: what each ``autotune`` call hands FlashInfer."""

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
    fac.autotune(tune_mode=False)
    path = str(tmp_path / "rank3.json")
    assert calls == [(True, path, {}), (False, path, {})]


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
