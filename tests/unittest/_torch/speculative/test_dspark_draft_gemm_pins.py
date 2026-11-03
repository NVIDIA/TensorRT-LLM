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
"""TLLM_DSPARK_DRAFT_GEMM shape pins on GB300 (SM103): the TP32 drafter's LM head shard takes its (N, K) pin, and
every output element is bitwise the role pin's (same split_k, other tile / stages); other shapes keep the role pins."""

import pytest
import torch

from tensorrt_llm._torch.speculative import dspark_draft_gemm as dg

LM_HEAD_TP32 = (5120, 7168)


def _sm103() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)


pytestmark = pytest.mark.skipif(not _sm103(), reason="the SM103 shape pins need a GB300")


def _role_pin(role, m):
    return next(p for p in dg._PINS[role] if m <= p[0])


def _run(x, weight, tactic):
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic, run_splitk_dense

    from tensorrt_llm._torch.kernels.blackwell.dspark_draft_gemm_push import ensure_cute_runtime

    ensure_cute_runtime()
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    run_splitk_dense(x, weight.t(), None, out, False, SplitKTactic(*tactic))
    return out


def test_shape_pins_take_precedence_only_for_their_shapes():
    dev = torch.device("cuda")
    assert dg._pins("lm_head", *LM_HEAD_TP32, dev) is dg._PINS_SM103[LM_HEAD_TP32]
    # The TP16 LM head shard and the other roles keep the role pins.
    assert dg._pins("lm_head", 10240, 7168, dev) is dg._PINS["lm_head"]
    assert dg._pins("gate_up", 1792, 7168, dev) is dg._PINS["gate_up"]
    for rows, kind, tactic in dg._PINS_SM103[LM_HEAD_TP32]:
        role_tactic = _role_pin("lm_head", rows)[2]
        assert kind == "splitk" and tactic[2] == role_tactic[2], (
            "a shape pin must keep the role pin's split_k"
        )


@pytest.mark.parametrize("m", [1, 3, 8, 9, 13, 16, 17, 24, 32])
@pytest.mark.parametrize("seed", [0, 1])
def test_lm_head_tp32_pin_is_bitwise_the_role_pin(m, seed, monkeypatch):
    monkeypatch.setenv(dg.DRAFT_GEMM_ENV_VAR, "1")
    n, k = LM_HEAD_TP32
    gen = torch.Generator(device="cuda").manual_seed(seed * 100 + m)
    weight = (torch.randn(n, k, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
    # Wide dynamic range, exact zeros and negative zeros in the activations.
    x[:, ::7] *= 64.0
    x[:, 5::11] = 0.0
    x[:, 3::13] = -0.0
    ref = _run(x, weight, _role_pin("lm_head", m)[2])
    got = dg.draft_gemm(x, weight, "lm_head")
    torch.cuda.synchronize()
    assert got is not None
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
