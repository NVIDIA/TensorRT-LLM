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
"""The Cosmos3 rotary table must stay fp32-accurate when fp32 GEMMs run in TF32.

NGC PyTorch images set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1. Positions above 2048
do not fit TF32's 10-bit mantissa, so a rotary table built as a K=1 matmul is
off by radians for late tokens on GPUs where cuBLAS picks a TF32 kernel for
that shape (Blackwell). The table is an outer product and must not go through
a GEMM.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
    Qwen3VLTextRotaryEmbedding,
)

pytestmark = pytest.mark.cosmos3

HEAD_DIM = 128
ROPE_AXES = [24, 20, 20]  # sums to HEAD_DIM // 2, the Cosmos3-Nano layout
# Text tokens (up to 4096) + margin + fps-scaled vision positions land well
# above 2048, the largest integer TF32 represents exactly. cuBLAS picks a TF32
# kernel for the K=1 GEMM only at some problem sizes, so sweep the sizes a
# Cosmos3 request actually produces.
SEQUENCE_LENGTHS = [4096, 6240, 8192, 10336, 16384]


def _rotary() -> Qwen3VLTextRotaryEmbedding:
    pretrained = SimpleNamespace(
        rope_theta=1_000_000.0,
        head_dim=HEAD_DIM,
        max_position_embeddings=262_144,
        rope_axes_dim=ROPE_AXES,
        rope_scaling=None,
    )
    return Qwen3VLTextRotaryEmbedding(SimpleNamespace(pretrained_config=pretrained))


def _reference_cos_sin(rotary: Qwen3VLTextRotaryEmbedding, position_ids: torch.Tensor):
    """Same math as the module, in float64 and without any matmul."""
    inv = rotary.inv_freq.double()[None, None, :, None]  # [1, 1, D/2, 1]
    pos = position_ids.double()[:, :, None, :]  # [3, B, 1, N]
    freqs = (inv * pos).transpose(2, 3)  # [3, B, N, D/2]
    freqs = rotary.apply_interleaved_mrope(freqs.clone(), rotary.mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
def test_rotary_table_matches_fp64_under_tf32(allow_tf32: bool, seq_len: int):
    device = torch.device("cuda")
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        rotary = _rotary().to(device)
        base = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 3D mRoPE ids: temporal axis fps-scaled (fractional), spatial axes integer
        position_ids = torch.stack([base * 24.0 / 10.0, base, base * 0.5], dim=0)[:, None, :]
        probe = torch.empty(0, dtype=torch.float32, device=device)

        cos, sin = rotary(probe, position_ids)
        ref_cos, ref_sin = _reference_cos_sin(rotary, position_ids)

        # fp32 evaluates angles of ~4e4 rad with ~6e-8 relative precision, so a
        # few 1e-3 absolute is the honest fp32 floor; TF32 breakage is O(1).
        tol = 5e-3
        assert (cos.double() - ref_cos).abs().max().item() < tol
        assert (sin.double() - ref_sin).abs().max().item() < tol
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
