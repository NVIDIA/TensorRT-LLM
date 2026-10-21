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
"""The DeepGEMM helpers started while the weights load, then refreshed at
warmup with the shapes weight loading adds (planning only, no helper
processes)."""

import json
import threading

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch import jit_prefetch_deep_gemm as jdg


def _provider(shapes, max_num_tokens=64, priority_ms=(), first_ms=()):
    p = jdg.FP8LinearDeepGemmProvider(
        torch.nn.Module(),
        max_num_tokens=max_num_tokens,
        priority_ms=priority_ms,
        first_ms=first_ms,
    )
    p.shapes = sorted(shapes)
    return p


class _Prefetcher(jp.JitPrefetcher):
    """A JitPrefetcher whose DeepGEMM requests are recorded, not sent."""

    def __init__(self):
        self.rank = 0
        self._lock = threading.Lock()
        self.submitted = []
        self.events = []
        self._dg = None
        self._dg_moe = None

    def _submit(self, key, spec, label, priority, kind="triton"):
        self.submitted.append((json.loads(spec), priority))
        return True

    def _event(self, msg):
        self.events.append(msg)


def _wait_enum(pf):
    for t in threading.enumerate():
        if t.name == "jit_prefetch_dg_enum2":
            t.join(10)


@pytest.mark.cpu_only
def test_refresh_queues_only_the_shapes_weight_loading_added():
    pf = _Prefetcher()
    pf._dg = _provider([(512, 256)], priority_ms=[7, 14], first_ms=[7, 1])
    pf._dg.plan_tokens(7)
    loaded = _provider([(512, 256), (1024, 256)], priority_ms=[7, 14], first_ms=[7, 1])
    assert pf.refresh_deep_gemm(loaded) == 1
    _wait_enum(pf)
    assert pf._dg is loaded
    # The planned M carry over, so a batch of 7 is not planned again.
    assert loaded.plan_tokens(7) == []
    shapes = {(s["n"], s["k"]) for s, _ in pf.submitted}
    assert shapes == {(1024, 256)}
    # The first warmup forwards' layouts of the new shape (their token
    # counts, then the CUDA graph token counts, then a full batch) are queued
    # ahead of the background enumeration; then the new shape gets the same
    # coverage as the others.
    first = [(s["m"], p) for s, p in pf.submitted[:4]]
    assert first == [(1, jp._PRIO_BATCH), (7, jp._PRIO_BATCH), (14, jp._PRIO_BATCH),
                     (64, jp._PRIO_BATCH)]  # fmt: skip
    rest = [s["m"] for s, p in pf.submitted[4:]]
    assert all(p == jp._PRIO_ENUM for _, p in pf.submitted[4:])
    assert 32 in rest and rest[:3] == [1, 7, 14]


@pytest.mark.cpu_only
def test_refresh_without_new_shapes_queues_nothing():
    pf = _Prefetcher()
    pf._dg = _provider([(512, 256)])
    assert pf.refresh_deep_gemm(_provider([(512, 256)])) == 0
    assert pf.submitted == []


@pytest.mark.cpu_only
def test_refresh_before_any_start_is_a_no_op():
    pf = _Prefetcher()
    assert pf.refresh_deep_gemm(_provider([(512, 256)])) == 0
    assert pf._dg is None and pf.submitted == []


@pytest.mark.cpu_only
def test_provider_plans_the_shapes_a_model_creates_while_loading():
    """A module that names the FP8 GEMMs its weight loading will create has
    them planned by a provider built before loading."""

    class _Loads(torch.nn.Module):
        def jit_prefetch_fp8_weight_read_shapes(self):
            return {(1024, 256), (512, 128)}

    model = torch.nn.ModuleDict({"text": _Loads()})
    p = jdg.FP8LinearDeepGemmProvider(model, max_num_tokens=64)
    assert p.shapes == [(512, 128), (1024, 256)]

    class _Fails(torch.nn.Module):
        def jit_prefetch_fp8_weight_read_shapes(self):
            raise RuntimeError("no prediction")

    # A failed prediction leaves the shapes to the loaded model.
    assert jdg.FP8LinearDeepGemmProvider(_Fails(), max_num_tokens=64).shapes == []


@pytest.mark.cpu_only
def test_start_deep_gemm_builds_is_off_without_prefetch(monkeypatch):
    monkeypatch.delenv("TLLM_JIT_PREFETCH", raising=False)
    assert not jp.start_deep_gemm_builds(torch.nn.Module(), 0, max_num_tokens=64)
    assert not jp.start_weight_load_builds(
        torch.nn.Module(), 0, {"max_num_tokens": 64, "priority_ms": [7]}
    )
