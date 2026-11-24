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
"""CPU tests for TRTLLM_CUTEDSL_DENSE_GEMM_MAX_SPLIT_K.

The env var caps the split-K factors the SM107 dense block-scaled GEMM
autotuner profiles. Unset, the tactic list and the autotuner cache key are
unchanged; set, the list keeps only split-K factors up to the cap and the cache
key records the narrowed candidates so capped and uncapped runs never share
cache entries.
"""

import pytest
import torch

from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

pytestmark = pytest.mark.cpu_only

ENV = "TRTLLM_CUTEDSL_DENSE_GEMM_MAX_SPLIT_K"
DEFAULT = (1, 2, 4, 8)

skip_no_sm107_cute_dsl = pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE, reason="CuTe DSL package without SM107 support"
)


def test_env_name():
    assert cute_dsl_custom_ops._DENSE_GEMM_MAX_SPLIT_K_ENV == ENV


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_unset_keeps_default(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, raw)
    assert cute_dsl_custom_ops._dense_gemm_max_split_k() is None
    assert cute_dsl_custom_ops._dense_gemm_split_k_candidates(DEFAULT) is DEFAULT


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1", (1,)),
        ("2", (1, 2)),
        ("3", (1, 2)),
        (" 4 ", (1, 2, 4)),
        ("8", DEFAULT),
        ("16", DEFAULT),
    ],
)
def test_cap_filters_candidates(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, raw)
    assert cute_dsl_custom_ops._dense_gemm_max_split_k() == int(raw)
    assert cute_dsl_custom_ops._dense_gemm_split_k_candidates(DEFAULT) == expected


@pytest.mark.parametrize("raw", ["0", "-2", "two", "1.5"])
def test_invalid_cap_raises(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match=ENV):
        cute_dsl_custom_ops._dense_gemm_split_k_candidates(DEFAULT)


class _RecordingKernel:
    """Stand-in for the SM107 dense kernel: accepts every configuration."""

    split_ks = []

    @classmethod
    def can_implement(cls, *args):
        cls.split_ks.append(args[-1])
        return True


class _NoMixedClustersKernel:
    @staticmethod
    def can_implement(*args):
        return False


def _nvfp4_tactics(monkeypatch, m=512, n=4096, k=4096):
    runner_cls = cute_dsl_custom_ops.CuteDSLNVFP4RubinLinear
    _RecordingKernel.split_ks = []
    monkeypatch.setattr(cute_dsl_custom_ops, "get_sm_version", lambda: 107)
    monkeypatch.setattr(runner_cls, "kernel_class", _RecordingKernel)
    monkeypatch.setattr(
        cute_dsl_custom_ops,
        "Sm107BlockScaledPersistentDenseGemmMixedClustersKernel",
        _NoMixedClustersKernel,
    )
    runner = runner_cls(torch.bfloat16)
    packed_k = k // runner_cls.packed_k_multiplier
    inputs = [
        torch.empty((m, packed_k), dtype=torch.uint8),
        torch.empty((n, packed_k), dtype=torch.uint8),
    ]
    return runner.get_valid_tactics(inputs, None), set(_RecordingKernel.split_ks)


@skip_no_sm107_cute_dsl
def test_runner_split_k_candidates_match_default():
    assert cute_dsl_custom_ops.CuteDSLBlockScaledRubinLinear.split_k_candidates == DEFAULT


@skip_no_sm107_cute_dsl
@pytest.mark.parametrize("cap", [1, 2, 4])
def test_runner_tactics_drop_only_larger_split_k(monkeypatch, cap):
    monkeypatch.delenv(ENV, raising=False)
    uncapped, probed = _nvfp4_tactics(monkeypatch)
    assert probed == set(DEFAULT)
    assert {t[8] for t in uncapped if t[0] == "base"} == set(DEFAULT)

    monkeypatch.setenv(ENV, str(cap))
    capped, probed = _nvfp4_tactics(monkeypatch)
    allowed = {s for s in DEFAULT if s <= cap}
    # Larger factors are not even offered to can_implement, so they are never
    # compiled or profiled.
    assert probed == allowed
    assert capped == [t for t in uncapped if t[0] != "base" or t[8] in allowed]


@skip_no_sm107_cute_dsl
def test_runner_cache_key(monkeypatch):
    runner = cute_dsl_custom_ops.CuteDSLNVFP4RubinLinear(torch.bfloat16)
    monkeypatch.delenv(cute_dsl_custom_ops._DENSE_GEMM_SCHEDULER_ENV, raising=False)

    monkeypatch.delenv(ENV, raising=False)
    unset_key = runner.unique_id()
    assert unset_key == (torch.bfloat16, False, True)

    # A cap that keeps every candidate leaves the tactic space, and so the
    # key, unchanged: existing cache files stay valid.
    for raw in ("8", "16"):
        monkeypatch.setenv(ENV, raw)
        assert runner.unique_id() == unset_key

    keys = {}
    for cap in (1, 2, 4):
        monkeypatch.setenv(ENV, str(cap))
        keys[cap] = runner.unique_id()
        assert keys[cap] != unset_key
        assert keys[cap][-1] == ("split_k", tuple(s for s in DEFAULT if s <= cap))
    assert len(set(keys.values())) == 3
    # "3" offers the same candidates as "2", so it shares that cache entry.
    monkeypatch.setenv(ENV, "3")
    assert runner.unique_id() == keys[2]

    # The cap composes with the scheduler pin.
    monkeypatch.setenv(cute_dsl_custom_ops._DENSE_GEMM_SCHEDULER_ENV, "static")
    monkeypatch.setenv(ENV, "1")
    assert runner.unique_id() == unset_key + ("static", ("split_k", (1,)))
