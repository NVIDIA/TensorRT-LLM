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
"""Fallback contracts for the finalize+all-reduce fusion setup.

Everything here is host-side plumbing: no GPU, no IPC. The real workspace
constructor is replaced so the tests pin the *selection* behavior -- when the
model keeps the fusion, and when it must degrade to the unfused finalize +
AllReduce path without failing model init.
"""

import pytest

from tensorrt_llm._torch.models import musespark_open_finalize_ar as finalize_ar
from tensorrt_llm._torch.models.modeling_musespark_open import _build_finalize_ar_workspace

MAPPING = object()


@pytest.fixture
def warnings_seen(monkeypatch):
    from tensorrt_llm._torch.models import modeling_musespark_open as modeling

    seen = []
    monkeypatch.setattr(modeling.logger, "warning", seen.append)
    return seen


@pytest.mark.parametrize("error", [ValueError("tp mismatch"), RuntimeError("peer not accessible")])
def test_ipc_probe_failure_falls_back_with_warning(monkeypatch, warnings_seen, error):
    """An IPC-incapable topology must not crash model init."""

    def raise_probe_failure(mapping):
        assert mapping is MAPPING
        raise error

    monkeypatch.setattr(finalize_ar, "HAS_CUTEDSL", True)
    monkeypatch.setattr(finalize_ar, "FinalizeIpcWorkspace", raise_probe_failure)
    assert _build_finalize_ar_workspace(MAPPING) is None
    assert len(warnings_seen) == 1
    assert str(error) in warnings_seen[0]


def test_workspace_built_when_probe_succeeds(monkeypatch, warnings_seen):
    workspace = object()
    monkeypatch.setattr(finalize_ar, "HAS_CUTEDSL", True)
    monkeypatch.setattr(finalize_ar, "FinalizeIpcWorkspace", lambda mapping: workspace)
    assert _build_finalize_ar_workspace(MAPPING) is workspace
    assert warnings_seen == []


def test_no_cutedsl_returns_none_without_probe(monkeypatch, warnings_seen):
    def must_not_construct(mapping):
        raise AssertionError("FinalizeIpcWorkspace must not be constructed without CuTe DSL")

    monkeypatch.setattr(finalize_ar, "HAS_CUTEDSL", False)
    monkeypatch.setattr(finalize_ar, "FinalizeIpcWorkspace", must_not_construct)
    assert _build_finalize_ar_workspace(MAPPING) is None
    assert warnings_seen == []
