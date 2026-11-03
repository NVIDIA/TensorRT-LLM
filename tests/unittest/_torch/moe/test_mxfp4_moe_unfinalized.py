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
"""The TRTLLM-Gen MXFP4 x MXFP8 runner's unfinalized output.

``trtllm::mxe4m3_mxe2m1_block_scale_moe_runner_unfinalized`` must return the
FC2 rows and the (token, slot) -> row map from which the finalize (FP32 in
slot order, one BF16 rounding, with the caller's routing weights) reproduces
the finalized op's output bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._utils import is_sm_100f

_NUM_EXPERTS = 896
_TOP_K = 16
_HIDDEN = 3584
_INTER = 768


def _reference_finalize(rows, weights, perm, num_tokens):
    out = torch.zeros(num_tokens, rows.shape[1], dtype=torch.float32, device=rows.device)
    perm = perm.view(num_tokens, _TOP_K)
    weights = weights.view(num_tokens, _TOP_K).float()
    for slot in range(_TOP_K):
        idx = perm[:, slot]
        valid = idx >= 0
        x = rows[idx.clamp_min(0).long()].float()
        out = torch.where(valid[:, None], torch.addcmul(out, weights[:, slot : slot + 1], x), out)
    return out.to(torch.bfloat16)


def _runner_has_unfinalized() -> bool:
    runner = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(0, True)
    return hasattr(runner, "run_moe_unfinalized")


@pytest.mark.skipif(not torch.cuda.is_available() or not is_sm_100f(), reason="needs SM100 family")
@pytest.mark.parametrize("num_tokens", [7, 28, 112])
@pytest.mark.parametrize("local_offset", [0, 448])
def test_unfinalized_matches_finalized(num_tokens, local_offset):
    if not _runner_has_unfinalized():
        pytest.skip("this build's MXFP4 runner has no run_moe_unfinalized")
    torch.manual_seed(num_tokens + local_offset)
    local = 448
    dev = "cuda"
    # Random FP4 codes; E8M0 scales near 2^-7 keep FC1 / FC2 finite.
    w13 = torch.randint(0, 256, (local, 2 * _INTER, _HIDDEN // 2), dtype=torch.uint8, device=dev)
    s13 = torch.randint(118, 122, (local, 2 * _INTER, _HIDDEN // 32), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (local, _HIDDEN, _INTER // 2), dtype=torch.uint8, device=dev)
    s2 = torch.randint(118, 122, (local, _HIDDEN, _INTER // 32), dtype=torch.uint8, device=dev)
    x = torch.randint(0, 256, (num_tokens, _HIDDEN), dtype=torch.uint8, device=dev)
    x = x.bitwise_and(0x77).view(torch.float8_e4m3fn)  # finite e4m3 codes
    x_sf = torch.full((num_tokens * _HIDDEN // 32,), 127, dtype=torch.uint8, device=dev)
    ids = torch.stack(
        [torch.randperm(_NUM_EXPERTS, device=dev)[:_TOP_K] for _ in range(num_tokens)]
    )
    ids = ids.to(torch.int32)
    weights = (torch.rand(num_tokens, _TOP_K, device=dev) + 0.5).to(torch.bfloat16)
    alpha = torch.full((local,), 4.0 * 25.0, dtype=torch.float32, device=dev)
    beta = torch.full((local,), 0.25, dtype=torch.float32, device=dev)
    args = (
        None, None, x, x_sf, w13, s13, None, alpha, beta, None, w2, s2, None,
        _NUM_EXPERTS, _TOP_K, None, None, _INTER, _HIDDEN, None, local_offset, local, 1.0, 2, 3,
    )  # fmt: skip
    with autotune(False):
        finalized = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
            *args, topk_weights=weights, topk_ids=ids
        )
        rows, row_weights, perm = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner_unfinalized(
            *args, topk_weights=weights, topk_ids=ids
        )
    # Precomputed routing: the op's weight output is unwritten, the caller's weights apply.
    assert row_weights.shape == (num_tokens, _TOP_K)
    assert row_weights.data_ptr() != weights.data_ptr()
    assert perm.numel() == num_tokens * _TOP_K
    local_ids = (ids >= local_offset) & (ids < local_offset + local)
    assert torch.equal(perm.view(num_tokens, _TOP_K) >= 0, local_ids)
    reference = _reference_finalize(rows, weights, perm, num_tokens)
    assert torch.isfinite(finalized.float()).all()
    assert torch.equal(finalized.view(torch.int16), reference.view(torch.int16))
