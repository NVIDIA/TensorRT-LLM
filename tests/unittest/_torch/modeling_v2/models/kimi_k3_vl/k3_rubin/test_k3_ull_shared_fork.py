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
"""Where the Kimi-K3 ULL MoE front forks the shared expert's GEMMs, and in what order it issues the fork.

Under CUDA graph capture the order matters: the kernel captured first among an early-releasing kernel's
dependents keeps that kernel's stream and its release. A_mid's must be the front's head, the head's the
route kernel; the shared GEMMs depend on A_mid above ``_K3_ULL_FRONT_FORK_AT_HEAD_MAX_M`` rows and on the
head up to it, and are issued after the front either way. These tests drive
``KimiK3MoERuntime._ull_fork_shared`` with a stand-in object that logs the event records and the shared
launch, and run the returned callbacks where the front would.

CPU only: no kernels, no CUDA graphs.
"""

import importlib
from types import SimpleNamespace

import pytest

TEXT_MODEL = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.text_model"
)


@pytest.fixture
def text_model():
    return importlib.import_module(TEXT_MODEL)


@pytest.mark.parametrize(
    "early, lead, rows, expected",
    [
        (False, False, 8, None),
        (True, False, 32, None),
        (False, True, 8, "start"),
        (False, True, 32, "start"),
        (True, True, 8, "head"),
        (True, True, 16, "head"),
        (True, True, 24, "after_head"),
        (True, True, 32, "after_head"),
    ],
)
def test_fork_placement(text_model, monkeypatch, early, lead, rows, expected):
    monkeypatch.setattr(text_model, "_K3_ULL_FRONT_EARLY", early)
    monkeypatch.setattr(text_model, "_K3_ULL_FRONT_FORK_AT_HEAD_MAX_M", 16)
    assert text_model._ull_shared_fork(lead, rows) == expected


def _fork_log(text_model, monkeypatch, early, lead, rows):
    """Issue order of a front on ``rows`` rows: event records, the shared launch and the head."""
    monkeypatch.setattr(text_model, "_K3_ULL_FRONT_EARLY", early)
    monkeypatch.setattr(text_model, "_K3_ULL_FRONT_FORK_AT_HEAD_MAX_M", 16)
    log = []
    partial = object()

    def shared_first(identity, persist):
        log.append(("shared", identity, persist))
        return partial

    moe = SimpleNamespace(
        moe_main_event=SimpleNamespace(record=lambda: log.append("record")),
        _ull_shared_first=shared_first,
        _shared_gemms_can_lead=lambda: lead,
    )
    after_head, after_front, forked = text_model.KimiK3MoERuntime._ull_fork_shared(
        moe, "x", False, rows
    )
    log.append("head")
    if after_head is not None:
        after_head()
    log.append("route")
    if after_front is not None:
        after_front()
    return log, forked, partial


@pytest.mark.parametrize(
    "early, rows, expected",
    [
        # Without the early release: the event and the shared GEMMs before the front.
        (False, 32, ["record", "shared", "head", "route"]),
        # Few rows: the event after the head (the shared GEMMs depend on it), the shared GEMMs after
        # the route kernel, so the head's first dependent is the route kernel.
        (True, 16, ["head", "record", "route", "shared"]),
        # More rows: the event before the head (the shared GEMMs depend on A_mid only, so the head is
        # A_mid's first dependent), the shared GEMMs after the front, and no second record.
        (True, 32, ["record", "head", "route", "shared"]),
    ],
)
def test_fork_order(text_model, monkeypatch, early, rows, expected):
    log, forked, partial = _fork_log(text_model, monkeypatch, early, True, rows)
    assert [e if isinstance(e, str) else e[0] for e in log] == expected
    assert ("shared", "x", False) in log
    assert forked == [partial]


def test_no_fork_without_lead(text_model, monkeypatch):
    log, forked, _ = _fork_log(text_model, monkeypatch, True, False, 32)
    assert log == ["head", "route"]
    assert forked == []
