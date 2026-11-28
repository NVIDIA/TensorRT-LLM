# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.deco_attention import (
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
    shape = (1, tokens, 2, 128)
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Scores with standard deviation 4 let a few keys dominate each row.
    q = (torch.randn(shape, device="cuda", generator=generator) * 4).bfloat16()
    k = torch.randn(shape, device="cuda", generator=generator).bfloat16()
    v = torch.randn(shape, device="cuda", generator=generator)
    v = (v + torch.randn((1, 1, 2, 128), device="cuda", generator=generator)).bfloat16()
    valid = tokens - padding
    kwargs = dict(v_dtype=v_dtype)
    if padding:
        kwargs["key_padding_mask"] = (torch.arange(tokens, device="cuda") < valid)[None]
        tags = torch.zeros(tokens, dtype=torch.long, device="cuda")
        tags[:64], tags[64:96], tags[valid:] = 1, 2, -1
        kwargs["token_tags"] = tags
        k[:, valid:] = float("nan")
        v[:, valid:] = float("nan")
    deco_attention.is_valid(q, k, v, **kwargs)
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention as provider

    with patch.object(provider, "prepare", wraps=provider.prepare) as prepare:
        actual = deco_attention(q, k, v, **kwargs)
    prepare.assert_called_once()
    if padding:
        assert prepare.call_args.kwargs["valid_seq_lens"].tolist() == [valid]
        assert prepare.call_args.kwargs["rows"] is not None
        assert torch.equal(actual[:, valid:], torch.zeros_like(actual[:, valid:]))
    expected = deco_attention.reference(q, k, v, **kwargs)
    out, ref = actual[:, :valid], expected[:, :valid]
    assert torch.isfinite(out).all()
    # Uniform weights ignore Q and K; the quantized scores must follow them more closely.
    uniform = v[:, :valid].float().mean(dim=1, keepdim=True).expand_as(ref)
    assert _relative_error(out, ref) < max(0.5 * _relative_error(uniform, ref), 0.02)
