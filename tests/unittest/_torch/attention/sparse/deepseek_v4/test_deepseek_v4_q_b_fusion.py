# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.module import (
    _dsv4_q_b_gemm_cu_q_seqlens,
)


def test_q_b_gemm_cu_seqlens_per_phase() -> None:
    ctx_cu = torch.tensor([0, 8, 20], dtype=torch.int32)
    gen_cu = torch.arange(9, dtype=torch.int32)
    owner = SimpleNamespace()

    context_only = SimpleNamespace(
        cu_q_seqlens=None,
        num_contexts=2,
        num_generations=0,
        mla_prepare_ctx_cu_seqlens=lambda: ctx_cu,
        cu_seq_lens_cuda=gen_cu,
    )
    assert torch.equal(_dsv4_q_b_gemm_cu_q_seqlens(owner, context_only), ctx_cu)

    generation_only = SimpleNamespace(
        cu_q_seqlens=None,
        num_contexts=0,
        num_generations=4,
        cu_seq_lens_cuda=gen_cu,
    )
    assert torch.equal(_dsv4_q_b_gemm_cu_q_seqlens(owner, generation_only), gen_cu[:5])


def test_q_b_gemm_cu_seqlens_absent_without_dsa_metadata() -> None:
    metadata = SimpleNamespace(cu_q_seqlens=None, num_contexts=0, num_generations=4)
    assert _dsv4_q_b_gemm_cu_q_seqlens(SimpleNamespace(), metadata) is None
