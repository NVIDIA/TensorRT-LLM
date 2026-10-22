# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tensor-parallel Embedding run outside torch.compile tracing builds its pre-communication
wrapper once per module, and its output is unchanged."""

import pytest
import torch

from tensorrt_llm._torch.modules import embedding as embedding_module
from tensorrt_llm._torch.modules.embedding import Embedding, pre_comm_embedding_ops
from tensorrt_llm._torch.modules.linear import TensorParallelMode

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("compile_embedding", [False, True])
def test_pre_comm_wrapper_built_once(monkeypatch, compile_embedding):
    compiled = []

    def fake_compile(fn, **kwargs):
        compiled.append(kwargs)
        return fn

    monkeypatch.setattr(embedding_module.torch, "compile", fake_compile)
    emb = Embedding(
        num_embeddings=8,
        embedding_dim=4,
        dtype=torch.float32,
        enable_torch_compile_for_embedding=compile_embedding,
    )
    # Rank 1 of a 2-way column-parallel vocabulary of 16, without a multi-rank Mapping.
    emb.tp_size, emb.tp_rank, emb.tp_mode = 2, 1, TensorParallelMode.COLUMN
    emb.vocab_start_index, emb.vocab_end_index = 8, 16
    with torch.no_grad():
        emb.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    # Single-process stand-in for the TP reduction (a registered submodule otherwise).
    emb._modules.pop("all_reduce", None)
    object.__setattr__(emb, "all_reduce", lambda output: output)
    tokens = torch.tensor([0, 7, 8, 15])

    outputs = [emb(tokens) for _ in range(3)]

    assert len(compiled) == (1 if compile_embedding else 0)
    expected = pre_comm_embedding_ops(
        tokens,
        emb.weight,
        emb.tp_size,
        emb.tp_rank,
        emb.tp_mode,
        emb.vocab_start_index,
        emb.vocab_end_index,
        emb.gather_output,
        emb.padding_size,
    )
    for output in outputs:
        torch.testing.assert_close(output, expected)
    # Rank 1 owns tokens 8..15: tokens 0 and 7 embed to zeros.
    assert torch.count_nonzero(outputs[0][:2]) == 0
