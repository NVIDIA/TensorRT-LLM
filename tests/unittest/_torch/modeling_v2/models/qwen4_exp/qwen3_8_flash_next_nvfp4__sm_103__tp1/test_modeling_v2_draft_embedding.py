# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MTP draft layer's front end: the token embedding under rejection sampling, and the
dense projections' dispatch between `fp8_gemv` and the bf16 projection."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as T,
)
from tensorrt_llm._torch.speculative.interface import rejection_sampling_one_model

assert torch.cuda.is_available(), "the draft embedding test requires a CUDA device"


def test_draft_embedding_reads_row_0_for_rejected_slots() -> None:
    """Rejection sampling pads the slots past a request's accepted count with -1 (here every
    draft is rejected at slot 0); the front end embeds those as row 0 and every valid id as
    the plain lookup does."""
    batch, draft, vocab, hidden = 8, 3, 4096, 256
    draft_tokens = torch.arange(batch * draft, dtype=torch.int32, device="cuda").reshape(
        batch, draft
    )
    draft_probs = torch.zeros(batch, draft, vocab, device="cuda")
    draft_probs.scatter_(2, draft_tokens.long().unsqueeze(-1), 1.0)
    target_probs = torch.zeros(batch, draft + 1, vocab, device="cuda")
    target_probs[..., vocab - 1] = 1.0
    seed = torch.zeros(batch, dtype=torch.int64, device="cuda")
    offset = torch.arange(batch, dtype=torch.int64, device="cuda")
    accepted, _ = rejection_sampling_one_model(
        draft_probs, draft_tokens, target_probs, seed, offset
    )
    assert bool((accepted[:, 1:] == -1).all()) and bool((accepted[:, 0] == vocab - 1).all())

    g = torch.Generator(device="cuda").manual_seed(0)
    table = torch.randn(vocab, hidden, generator=g, device="cuda").to(torch.bfloat16)
    valid_ids = torch.randint(0, vocab, (257,), generator=g, device="cuda", dtype=torch.int32)
    for ids in (accepted.flatten(), valid_ids, valid_ids.long()):
        e = T._draft_embedding(ids, table)
        valid = ids >= 0
        assert torch.equal(e[valid], table[ids[valid].long()])
        assert torch.equal(e[~valid], table[0].expand(int((~valid).sum()), -1))


@pytest.mark.parametrize("rows", [1, 16, 17])
def test_draft_projection_dispatch(rows: int) -> None:
    """`_draft_mm`: `fp8_gemv` over the packed copy up to `_MTP_DENSE_FP8_MAX_ROWS` rows, the
    bf16 projection past them and without a copy."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    w = (torch.randn(2560, 2560, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(rows, 2560, generator=g, device="cuda").to(torch.bfloat16)
    packed = T.pack_from_bf16(w)
    got = T._draft_mm(x, w.t(), packed, T.cublas_mm)
    if rows <= T._MTP_DENSE_FP8_MAX_ROWS:
        assert torch.equal(got, T.fp8_gemv_op(x, packed))
        ref = x.float() @ w.float().t()
        assert ((got.float() - ref).norm() / ref.norm()).item() < 0.05
    else:
        assert torch.equal(got, T.cublas_mm(x, w.t()))
    assert torch.equal(T._draft_mm(x, w.t(), None, T.cublas_mm), T.cublas_mm(x, w.t()))
