# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash MTP layer's fused kernels against the modules they replace."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.kernels import (  # noqa: E501
    drafting_rows,
    embed_rms_norm_columns,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton kernel)")


@pytest.mark.parametrize("hidden", [4096, 96])
@pytest.mark.parametrize("cols", [None, (0, 0.5), (0.5, 1.0)])
@pytest.mark.parametrize("rows", [0, 1, 7, 257])
def test_embed_rms_norm_columns_matches_modules(hidden, cols, rows):
    torch.manual_seed(0)
    vocab, eps = 1000, 1e-5
    embed = Embedding(vocab, hidden, dtype=torch.bfloat16).cuda()
    norm = RMSNorm(hidden_size=hidden, eps=eps, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        embed.weight.normal_()
        norm.weight.uniform_(0.5, 1.5)
    ids = torch.randint(0, vocab, (rows,), dtype=torch.int32, device="cuda")
    if rows > 1:
        ids[0] = -1  # rejected draft slot padding
        ids[-1] = vocab + 3
    lo, hi = (0, hidden) if cols is None else (int(cols[0] * hidden), int(cols[1] * hidden))

    got = embed_rms_norm_columns(ids, embed.weight, norm.weight, eps, lo, hi)
    if rows == 0:  # (the module path cannot launch its norm on zero rows)
        assert got.shape == (0, hi - lo) and got.dtype == embed.weight.dtype
        return
    want = norm(embed(ids))[:, lo:hi]

    assert got.shape == want.shape and got.dtype == want.dtype
    torch.testing.assert_close(got, want, atol=0, rtol=1.6e-2)
    if rows > 1:
        assert not got[0].any() and not got[-1].any()


@pytest.mark.parametrize("n", [0, 1, 5, 64, 300])
def test_drafting_rows_matches_torch(n):
    width = 4
    batch_indices = torch.arange(n, dtype=torch.int32, device="cuda")
    num_accepted = torch.randint(1, width + 1, (n,), dtype=torch.int32, device="cuda")
    got = drafting_rows(batch_indices, num_accepted, width)
    want = batch_indices.long() * width + num_accepted - 1
    assert got.dtype == torch.int64
    torch.testing.assert_close(got, want, atol=0, rtol=0)
