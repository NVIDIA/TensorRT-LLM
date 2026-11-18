# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding lookups: the one-launch gather of a full table equals the masked lookup bitwise, and a
tensor-parallel rank embeds only its vocabulary shard."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import TensorParallelMode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("hidden", [64, 12288])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_kernel_matches_masked_lookup(hidden, dtype):
    vocab = 1000
    g = torch.Generator(device="cuda").manual_seed(hidden)
    stock = Embedding(num_embeddings=vocab, embedding_dim=hidden, dtype=dtype).cuda()
    gather = Embedding(
        num_embeddings=vocab, embedding_dim=hidden, dtype=dtype, use_gather_kernel=True
    ).cuda()
    with torch.no_grad():
        stock.weight.copy_(torch.randn((vocab, hidden), generator=g, device="cuda").to(dtype))
        gather.weight.copy_(stock.weight)
    # In range, the table edges, and the rejection sampler's -1 padding / out-of-range ids.
    ids = torch.tensor([[0, 1, vocab - 1, 17], [-1, vocab, 5, -1]], device="cuda")
    in_range = (ids >= 0) & (ids < vocab)

    out = gather(ids)
    ref = F.embedding(ids.clamp(0, vocab - 1), stock.weight).masked_fill(~in_range.unsqueeze(-1), 0)

    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
    assert gather(ids[:, :0]).shape == (2, 0, hidden)


@pytest.mark.cpu_only
def test_tensor_parallel_rank_embeds_its_shard():
    emb = Embedding(num_embeddings=8, embedding_dim=4, dtype=torch.float32)
    # Rank 1 of a 2-way column-parallel vocabulary of 16, without a multi-rank Mapping.
    emb.tp_size, emb.tp_rank, emb.tp_mode = 2, 1, TensorParallelMode.COLUMN
    emb.vocab_start_index, emb.vocab_end_index = 8, 16
    with torch.no_grad():
        emb.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    # Single-process stand-in for the TP reduction (a registered submodule otherwise).
    emb._modules.pop("all_reduce", None)
    object.__setattr__(emb, "all_reduce", lambda output, all_reduce_params=None: output)

    out = emb(torch.tensor([0, 7, 8, 15]))

    # Rank 1 owns tokens 8..15 (its rows 0..7); tokens 0 and 7 embed to zeros.
    assert torch.count_nonzero(out[:2]) == 0
    torch.testing.assert_close(out[2:], emb.weight[[0, 7]])
