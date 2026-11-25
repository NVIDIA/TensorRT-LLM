# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The targets' token embedding entry (the engine's replicated `Embedding` module): rows of the table, bit for bit, with
the module's torch ops and with stock's masked-embedding kernel (`masked_kernel`), whose rejected-draft ids (-1) read
zeros as the torch ops' masking does."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.embedding.token_embedding import (  # noqa: E501
    token_embedding,
    token_embedding_create,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


@pytest.mark.parametrize(
    "cell", token_embedding.CELLS, ids=[c.why[:44] for c in token_embedding.CELLS]
)
def test_cells(cell, certifying):  # noqa: F811
    certifying(token_embedding)
    s, gen = cell.spec, gen_for(cell)
    masked = s.get("masked_kernel", False)
    emb = token_embedding_create(s["vocab"], s["hidden"], torch.bfloat16, masked_kernel=masked).to(
        DEV
    )
    assert emb.use_masked_embedding_kernel is masked
    emb.weight.data.copy_(torch.randn(s["vocab"], s["hidden"], generator=gen, device=DEV))
    ids = torch.randint(0, s["vocab"], (s["n"],), generator=gen, device=DEV, dtype=torch.int32)
    ids[: s.get("rejected", 0)] = -1
    out = token_embedding(emb, ids)
    assert torch.equal(out, token_embedding.reference(emb, ids))
    if masked:  # the same bits as the module's torch ops
        assert torch.equal(out, _torch_ops(emb, ids))


def _torch_ops(emb, ids):
    emb.use_masked_embedding_kernel = False
    try:
        return emb(ids)
    finally:
        emb.use_masked_embedding_kernel = True


def test_masked_kernel_launches_stocks_kernel(monkeypatch):
    """`masked_kernel=True` runs stock's masked-embedding kernel (one launch), the default the module's torch ops."""
    from tensorrt_llm._torch.kernels.blackwell import masked_embedding as mk

    calls = []
    real = mk.masked_embedding
    monkeypatch.setattr(mk, "masked_embedding", lambda *a: calls.append(1) or real(*a))
    ids = torch.tensor([3, -1, 7], device=DEV, dtype=torch.int32)
    token_embedding(token_embedding_create(16, 64, masked_kernel=True).to(DEV), ids)
    assert calls == [1]
    token_embedding(token_embedding_create(16, 64).to(DEV), ids)
    assert calls == [1]


def test_rejections():
    emb = token_embedding_create(16, 8).to(DEV)
    token_embedding.is_valid(emb, torch.zeros(2, dtype=torch.int64, device=DEV))
    with pytest.raises(AssertionError):
        token_embedding.is_valid(emb, torch.zeros(2, device=DEV))
