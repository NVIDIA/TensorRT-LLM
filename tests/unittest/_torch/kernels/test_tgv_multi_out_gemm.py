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
"""tgv_multi_out_gemm: each row block of the packed weight lands in its own output, bitwise TGV's GEMM of the
packed weight (BF16 blocks) or its FP32 accumulator (FP32 blocks)."""

import pytest
import torch

from tensorrt_llm._torch.kernels.blackwell import tgv_multi_out_gemm as mo
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103) or not mo.IS_AVAILABLE,
    reason="needs an SM100 / SM103 GPU with the TGV CuTe DSL kernel",
)


@pytest.mark.parametrize("m", [1, 96, 128])
@pytest.mark.parametrize("tactic", [24])
@pytest.mark.parametrize("rows", [(896, 3584, 768), (896, 3584, 1536)])
def test_blocks_match_tgv_on_the_packed_weight(m, tactic, rows):
    from flashinfer.gemm.kernels.tgv_gemm_cute_ext import run_tgv_cute_ext

    torch.manual_seed(m + tactic + sum(rows))
    k = 7168
    weight = (torch.randn(sum(rows), k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    dtypes = (torch.float32,) + (torch.bfloat16,) * (len(rows) - 1)
    outs = mo.multi_out_linear(x, weight, rows, dtypes, tactic, use_pdl=True)
    assert outs is not None
    single = torch.empty(m, sum(rows), device="cuda", dtype=torch.bfloat16)
    run_tgv_cute_ext(x, weight.t(), None, single, True, tactic)
    ref = x.float() @ weight.float().t()
    off = 0
    for out, r, dt in zip(outs, rows, dtypes):
        assert out.shape == (m, r) and out.dtype == dt and out.is_contiguous()
        assert torch.equal(out.to(torch.bfloat16), single[:, off : off + r])
        torch.testing.assert_close(out.float(), ref[:, off : off + r], rtol=2e-2, atol=2e-2)
        off += r


def test_unserved_operands():
    k = 7168
    w = torch.zeros(896 + 3584, k, device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(8, k, device="cuda", dtype=torch.bfloat16)
    assert (
        mo.multi_out_linear(x, w, (896, 3584), (torch.float32, torch.bfloat16), 21, True)
        is not None
    )
    # A block that is not a multiple of the tactic's weight-row tile.
    assert mo.multi_out_linear(x, w, (900, 3580), (torch.float32, torch.bfloat16), 21, True) is None
    # Blocks that do not cover the weight, too many blocks, a non-BF16 input.
    assert mo.multi_out_linear(x, w, (896, 3520), (torch.float32, torch.bfloat16), 21, True) is None
    assert mo.multi_out_linear(x, w, (896,) * 5, (torch.bfloat16,) * 5, 21, True) is None
    assert (
        mo.multi_out_linear(x.float(), w, (896, 3584), (torch.float32, torch.bfloat16), 21, True)
        is None
    )
