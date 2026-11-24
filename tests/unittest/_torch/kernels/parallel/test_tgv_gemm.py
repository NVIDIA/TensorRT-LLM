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
"""The TGV GEMM with the pair-synchronized 2-CTA teardown (``kernels/blackwell/tgv_gemm``).

The fork changes when the MMA warp releases tensor memory and how the two CTAs of a pair exit, not the GEMM:
through the measured low-M plan its output is FlashInfer's stock TGV kernel's bit for bit, 1-CTA and 2-CTA alike,
and launches queued back to back all complete with that same output. The fault the teardown removes needs one
CTA of a pair to be delayed at one point, which a test cannot force.
"""

import pytest
import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.low_m_gemm import planned_low_m_linear
from tensorrt_llm._torch.modules.low_m_gemm_plans import KIMI_K31_DECODE_PLAN
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="the TGV CuTe DSL kernel needs an SM100 / SM103 GPU",
)

# (out features, in features, rows, TGV tactic, 2-CTA): rows of KIMI_K31_DECODE_PLAN.
_TP16_O_PROJ = (7168, 768, 8, 11, False)
_ATTENTION_DP_O_PROJ = (7168, 12288, 16, 18, True)


def _operands(n: int, k: int, m: int):
    gen = torch.Generator(device="cuda").manual_seed(m * 7919 + n + k)
    a = torch.randn(m, k, generator=gen, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, generator=gen, device="cuda") / k**0.5).to(torch.bfloat16)
    return a, w


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


@pytest.mark.parametrize(
    "n,k,m,tactic,two_cta", [_TP16_O_PROJ, _ATTENTION_DP_O_PROJ], ids=["1cta", "2cta"]
)
def test_planned_tgv_row_is_bitwise_stock_tgv(n, k, m, tactic, two_cta) -> None:
    from flashinfer.gemm.kernels.tgv_gemm_cute_ext import run_tgv_cute_ext

    from tensorrt_llm._torch.kernels.blackwell import tgv_gemm

    assert (m, ("tgv", tactic)) in KIMI_K31_DECODE_PLAN[(n, k)]
    assert tgv_gemm.tactic_config(tactic)[3] is two_cta
    a, w = _operands(n, k, m)
    out = planned_low_m_linear(KIMI_K31_DECODE_PLAN, a, w)
    ref = torch.empty_like(out)
    run_tgv_cute_ext(a, w.t(), None, ref, get_env_enable_pdl(), tactic)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))
    torch.testing.assert_close(out.float(), a.float() @ w.float().t(), rtol=2e-2, atol=2e-2)


def test_back_to_back_2cta_launches_complete_with_the_same_bits() -> None:
    """With PDL a launch can start while the previous one may still be tearing its CTA pairs down."""
    from tensorrt_llm._torch.kernels.blackwell import tgv_gemm

    n, k, m, tactic, _ = _ATTENTION_DP_O_PROJ
    cfg = tgv_gemm.tactic_config(tactic)
    a, w = _operands(n, k, m)
    first = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    tgv_gemm.run_tgv(a, w.t(), first, True, cfg)
    torch.cuda.synchronize()
    # Several buffers, so the last launches are all checked, not only the final one.
    outs = [torch.empty_like(first) for _ in range(8)]
    for i in range(512):
        tgv_gemm.run_tgv(a, w.t(), outs[i % len(outs)], True, cfg)
    torch.cuda.synchronize()
    for out in outs:
        assert torch.equal(_bits(out), _bits(first))


def test_config_validation() -> None:
    from tensorrt_llm._torch.kernels.blackwell import tgv_gemm

    tgv_gemm.validate_config(64, 16, 12, True)
    with pytest.raises(ValueError):
        tgv_gemm.validate_config(64, 8, 12, True)  # 2-CTA needs cta_n in steps of 16
    with pytest.raises(ValueError):
        tgv_gemm.validate_config(96, 16, 12, False)
