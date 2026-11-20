# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention import flash_attention_4 as fa4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 needs CUDA")
@pytest.mark.parametrize("cell", fa4.flash_attention_4.CELLS, ids=lambda cell: cell.why)
def test_flash_attention_4(cell):
    s = cell.spec
    generator = torch.Generator(device="cuda").manual_seed(37)

    def tensor(seq, heads):
        return torch.randn(
            2, seq, heads, s["dim"], dtype=s["dtype"], device="cuda", generator=generator
        )

    q, k, v = tensor(s["sq"], s["hq"]), tensor(s["sk"], s["hk"]), tensor(s["sk"], s["hk"])
    lengths = (
        None
        if s["lengths"] is None
        else torch.tensor(s["lengths"], dtype=torch.int32, device="cuda")
    )
    kwargs = dict(softmax_scale=0.09375, causal=s["causal"], seqused_k=lengths)
    op = fa4.flash_attention_4
    op.is_valid(q, k, v, **kwargs)
    out, lse = op(q, k, v, **kwargs)
    reference, reference_lse = op.reference(q, k, v, **kwargs)
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(lse, reference_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.cpu_only
def test_missing_fa4_preserves_import_error(monkeypatch):
    cause = ImportError("missing FA4 dependency")
    monkeypatch.setattr(fa4, "_flash_attn_fwd", None)
    monkeypatch.setattr(fa4, "_flash_attn_fwd_import_error", cause)
    q = torch.zeros(1, 4, 1, 64, dtype=torch.bfloat16)
    with pytest.raises(ImportError, match="FlashAttention 4 is not available") as error:
        fa4.flash_attention_4(q, q, q, softmax_scale=0.125)
    assert error.value.__cause__ is cause


@pytest.mark.cpu_only
def test_valid_key_length_fits_kv():
    q = torch.zeros(1, 4, 1, 64, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="valid key lengths"):
        fa4.flash_attention_4.is_valid(
            q, q, q, softmax_scale=0.125, seqused_k=torch.tensor([5], dtype=torch.int32)
        )
