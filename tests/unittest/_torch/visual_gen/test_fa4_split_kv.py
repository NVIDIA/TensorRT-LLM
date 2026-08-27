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
"""Numerical parity for FlashAttn4's sm100 split-KV kernel.

FlashAttn4Attention passes ``num_splits=0``, so FA4's heuristic picks the split
count and any count above 1 selects a separate split-KV kernel. That kernel is
compiled by the CuTe DSL at first use, so a DSL that rejects it takes down every
FA4 config at pipeline load rather than at import. The other FA4 tests use K/V
short enough that the heuristic returns 1 (it short-circuits at
``num_n_blocks <= 4``), leaving the split-KV kernel uncovered; the K/V length
here is what reaches it.

Requires CUDA (FA4 is GPU-only).
"""

import pytest
import torch

try:
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import _flash_attn_fwd

    FA4_AVAILABLE = _flash_attn_fwd is not None
except ImportError:
    FA4_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 requires CUDA"),
    pytest.mark.skipif(not FA4_AVAILABLE, reason="FA4 kernel not available"),
]

# S_kv must stay well above FA4's `num_n_blocks <= 4` short-circuit for the
# `auto` case to reach the split-KV kernel; the short S_q keeps occupancy low
# enough that the heuristic prefers splitting, which is the LTX-2 cross-attn shape.
B, S_Q, S_KV, H, D = 1, 64, 4096, 8, 128


@pytest.mark.parametrize("num_splits", [0, 8], ids=["auto", "forced"])
def test_split_kv_matches_sdpa(num_splits):
    """FA4 split-KV output matches SDPA. Fails to compile on CUTLASS DSL < 4.6.2."""
    torch.manual_seed(0)
    device = "cuda"
    q, k, v = (
        torch.randn(B, s, H, D, dtype=torch.bfloat16, device=device) for s in (S_Q, S_KV, S_KV)
    )

    out, _, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        seqused_k=None,
        softmax_scale=D**-0.5,
        causal=False,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        softcap=0.0,
        pack_gqa=None,
        mask_mod=None,
        block_sparse_tensors=None,
        return_lse=True,
        num_splits=num_splits,
    )

    # SDPA wants [B, H, S, D]; FA4 uses [B, S, H, D].
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    torch.testing.assert_close(
        out,
        ref,
        rtol=2e-2,
        atol=2e-2,
        msg=f"FA4 num_splits={num_splits} diverges from SDPA",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
