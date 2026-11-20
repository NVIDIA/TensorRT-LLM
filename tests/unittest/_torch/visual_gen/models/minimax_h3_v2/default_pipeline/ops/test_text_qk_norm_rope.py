# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.default_pipeline.ops.text_qk_norm_rope import (
    text_qk_norm_rope,
)


@pytest.mark.parametrize("cell", text_qk_norm_rope.CELLS, ids=lambda cell: cell.why)
def test_packed_text_qk_norm_rope(cell):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    params = dict(cell.spec)
    tokens = params.pop("tokens")
    q_heads, kv_heads, dim = params["num_heads_q"], params["num_heads_kv"], params["head_dim"]
    rng = torch.Generator(device="cuda").manual_seed(1892)

    def randn(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=rng)

    qkv = randn((tokens, (q_heads + 2 * kv_heads) * dim))
    q_weight, k_weight = randn((dim,)), randn((dim,))
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    if params.get("factor", 1.0) != 1.0:
        positions += 2048
    args = (qkv, q_weight, k_weight, positions)
    qk_width = (q_heads + kv_heads) * dim
    value = qkv[:, qk_width:].clone()
    text_qk_norm_rope.is_valid(*args, **params)
    expected = text_qk_norm_rope.reference(qkv.clone(), q_weight, k_weight, positions, **params)
    actual = text_qk_norm_rope(*args, **params)
    assert actual.data_ptr() == qkv.data_ptr()
    torch.testing.assert_close(
        actual[:, :qk_width], expected[:, :qk_width].to(torch.bfloat16), rtol=8e-3, atol=1e-3
    )
    torch.testing.assert_close(actual[:, qk_width:], value, rtol=0, atol=0)
