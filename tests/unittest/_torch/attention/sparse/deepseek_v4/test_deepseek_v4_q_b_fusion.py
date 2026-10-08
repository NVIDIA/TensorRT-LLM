# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the DeepSeek-V4 q_b GEMM fusion metadata helpers."""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
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
