# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.rope import (
    apply_minimax_h3_qk_norm_rope_bf16 as op,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H3 fused RoPE requires CUDA")
@pytest.mark.parametrize("cell", op.CELLS, ids=lambda cell: cell.why)
def test_h3_fused_rope_matches_reference(cell):
    spec = cell.spec
    batch, seq, heads, dim = (spec[key] for key in ("batch", "seq", "heads", "dim"))
    generator = torch.Generator(device="cuda").manual_seed(137)
    qkv = torch.randn(
        batch, seq, 3 * heads * dim, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weights = torch.randn(2, dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    angles = torch.randn(seq, spec["rotary_dim"], device="cuda", generator=generator)
    args = (qkv, weights[0], weights[1], angles.cos(), angles.sin(), 1e-5, heads, dim)
    torch.testing.assert_close(op(*args), op.reference(*args), rtol=1e-2, atol=1e-2)
