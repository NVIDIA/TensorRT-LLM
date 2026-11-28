# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.deco_attention import (
    DECO_MIN_SEQ_LEN,
    deco_attention,
)


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref.float()).norm() / ref.float().norm())


@pytest.mark.parametrize("cell", deco_attention.CELLS, ids=lambda cell: cell.why)
def test_deco_attention(cell, require_deco_kernels):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("DeCo-Attention cells are certified on SM100")
    v_dtype, tokens, padding = cell.spec["v_dtype"], cell.spec["tokens"], cell.spec["padding"]
    require_deco_kernels(v_dtype)
    paddings = padding if isinstance(padding, tuple) else (padding,)
    shape = (len(paddings), tokens, 2, 128)
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Scores with standard deviation 4 let a few keys dominate each row.
    q = (torch.randn(shape, device="cuda", generator=generator) * 4).bfloat16()
    k = torch.randn(shape, device="cuda", generator=generator).bfloat16()
    v = torch.randn(shape, device="cuda", generator=generator)
    v = (v + torch.randn((1, 1, 2, 128), device="cuda", generator=generator)).bfloat16()
    valid = torch.tensor([tokens - p for p in paddings], device="cuda")
    tokens_mask = torch.arange(tokens, device="cuda")[None] < valid[:, None]
    kwargs = dict(v_dtype=v_dtype)
    if any(paddings):
        kwargs["key_padding_mask"] = tokens_mask
        tags = torch.zeros(tokens, dtype=torch.long, device="cuda")
        tags[:64], tags[64:96], tags[int(valid.max()) :] = 1, 2, -1
        kwargs["token_tags"] = tags
        k[~tokens_mask] = float("nan")
        v[~tokens_mask] = float("nan")
    deco_attention.is_valid(q, k, v, **kwargs)
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention as provider

    with patch.object(provider, "prepare", wraps=provider.prepare) as prepare:
        actual = deco_attention(q, k, v, **kwargs)
    quantized = tokens >= DECO_MIN_SEQ_LEN
    assert prepare.call_count == int(quantized)
    if any(paddings) and quantized:
        assert prepare.call_args.kwargs["valid_seq_lens"].tolist() == valid.tolist()
        assert prepare.call_args.kwargs["rows"] is not None
    assert torch.equal(actual[~tokens_mask], torch.zeros_like(actual[~tokens_mask]))
    expected = deco_attention.reference(q, k, v, **kwargs)
    out, ref = actual[tokens_mask], expected[tokens_mask]
    assert torch.isfinite(out).all()
    if not quantized:
        torch.testing.assert_close(out.float(), ref.float(), rtol=0.02, atol=0.02)
        return
    # Uniform weights ignore Q and K; the quantized scores must follow them more closely.
    uniform = v[tokens_mask].float().mean(dim=0, keepdim=True).expand_as(ref)
    assert _relative_error(out, ref) < max(0.5 * _relative_error(uniform, ref), 0.02)
