# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.default_pipeline.ops.text_attention import (
    text_causal_attention,
)


@pytest.mark.parametrize("cell", text_causal_attention.CELLS, ids=lambda cell: cell.why)
def test_text_attention_matches_causal_reference(cell):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    spec = cell.spec
    tokens, dim = spec["tokens"], spec["dim"]
    rng = torch.Generator(device="cuda").manual_seed(13291)

    def inputs(heads):
        return torch.randn((tokens, heads, dim), dtype=torch.bfloat16, generator=rng, device="cuda")

    args = (
        inputs(spec["heads"]),
        inputs(spec["kv_heads"]),
        inputs(spec["kv_heads"]),
        torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
    )
    text_causal_attention.is_valid(*args)
    expected = text_causal_attention.reference(*args)
    actual = text_causal_attention(*args)
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), atol=0.01, rtol=0.01)
