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
"""Multi-GPU Kimi K3 MLA head layouts, KV ownership and prefill/decode handoff."""

import pytest
import torch
from _torch.attention.helix_context_utils import _check_context_heads, _Scenario
from _torch.attention.multi_gpu.helix_test_utils import run_helix_test
from utils.util import skip_pre_blackwell


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize("fp8_cache", [False, True], ids=["bf16-kv", "fp8-kv"])
def test_k3_mla_helix_compact_cache(fp8_cache):
    run_helix_test(
        _check_context_heads,
        _Scenario(
            cp_size=4, fp8=True, fp8_cache=fp8_cache, check_kv_ownership=True, compact_cache=True
        ),
        "nccl",
        world_size=4,
        max_mismatch_ratio=0.0,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize("fp8_cache", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-weights", "fp8-weights"])
@pytest.mark.parametrize(
    "tp_size,cp_size,num_heads",
    [
        (1, 1, 32),
        (1, 2, 32),
        pytest.param(
            2,
            2,
            32,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        pytest.param(
            1,
            4,
            32,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        (1, 2, 6),
    ],
    ids=["tp1-cp1", "tp1-cp2", "tp2-cp2", "tp1-cp4", "tp1-cp2-unaligned"],
)
def test_k3_mla_helix_chunked_history(fp8, fp8_cache, tp_size, cp_size, num_heads):
    run_helix_test(
        _check_context_heads,
        _Scenario(
            tp_size=tp_size,
            cp_size=cp_size,
            num_heads=num_heads,
            gate=num_heads != 6,
            fp8=fp8,
            fp8_cache=fp8_cache,
            check_kv_ownership=True,
            chunked_history=True,
        ),
        "nccl",
        world_size=tp_size * cp_size,
        max_mismatch_ratio=0.0,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize("fp8_cache", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16-weights", "fp8-weights"])
@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
@pytest.mark.parametrize(
    "tp_size,cp_size,num_heads",
    [
        (1, 2, 32),
        pytest.param(
            2,
            2,
            32,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        pytest.param(
            1,
            4,
            32,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        (1, 2, 6),
    ],
    ids=["tp1-cp2", "tp2-cp2", "tp1-cp4", "tp1-cp2-unaligned"],
)
def test_k3_mla_helix_owned_prefill(fp8, fp8_cache, gate, tp_size, cp_size, num_heads):
    run_helix_test(
        _check_context_heads,
        _Scenario(
            tp_size=tp_size,
            cp_size=cp_size,
            num_heads=num_heads,
            fp8=fp8,
            gate=gate,
            check_kv_ownership=True,
            fp8_cache=fp8_cache,
        ),
        "nccl",
        world_size=tp_size * cp_size,
        max_mismatch_ratio=0.0,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@skip_pre_blackwell
@pytest.mark.parametrize(
    "fp8,gate,tp_size,cp_size,num_heads",
    [
        pytest.param(False, True, 1, 1, 32, id="bf16-control-tp1-cp1"),
        pytest.param(False, False, 1, 2, 32, id="bf16-no-gate-tp1-cp2"),
        pytest.param(True, True, 1, 2, 32, id="fp8-gate-tp1-cp2"),
        pytest.param(
            True,
            True,
            2,
            2,
            32,
            id="fp8-gate-tp2-cp2",
            marks=pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
        ),
        pytest.param(True, False, 1, 2, 6, id="fp8-no-gate-tp1-cp2-unaligned-q-scale-block"),
    ],
)
def test_k3_mla_helix_context_heads(fp8, gate, tp_size, cp_size, num_heads):
    scenario = _Scenario(tp_size=tp_size, cp_size=cp_size, num_heads=num_heads, fp8=fp8, gate=gate)
    run_helix_test(
        _check_context_heads,
        scenario,
        "nccl",
        world_size=scenario.tp_size * scenario.cp_size,
        max_mismatch_ratio=0.0,
    )
