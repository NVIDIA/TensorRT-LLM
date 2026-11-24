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
"""The K3 MLA drafter's RMSNorm + RoPE launchers (``k3_rubin/spec/dspark_mla_rope.py``) write
bitwise what the launches they replace write: the ``TLLM_DSPARK_TAIL_V2`` kernels against the
per-layer / separate launches, and the row-strided ``k3r_dspark_rmsnorm_rope`` against main's
``cute_dsl_dspark_rmsnorm_rope`` on a contiguous copy."""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="needs an SM100-family GPU"
)

_SPEC = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.spec"

KV_LORA = 512
ROPE = 64
NOPE = 128
HEAD_DIM = KV_LORA + ROPE
EPS = 1e-5


def _mla_rope():
    import importlib

    return importlib.import_module(f"{_SPEC}.dspark_mla_rope")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _freqs_table(num_positions: int, gen) -> torch.Tensor:
    """A ``[P, ROPE // 2, 2]`` FP32 (cos, sin) table with YaRN-like magnitudes."""
    angle = torch.rand(num_positions, ROPE // 2, device="cuda", generator=gen) * 6.3
    scale = 1.0 + 0.3 * torch.rand(num_positions, 1, device="cuda", generator=gen)
    return torch.stack([angle.cos() * scale, angle.sin() * scale], dim=-1).contiguous()


def _rows(n, width, gen, row_stride=None):
    """BF16 rows with a spread of magnitudes; optionally a row-strided view."""
    row_stride = row_stride or width
    base = torch.randn(n, row_stride, device="cuda", generator=gen)
    base = base * torch.logspace(-2, 2, n, device="cuda").unsqueeze(1)
    return base.to(torch.bfloat16)[:, :width]


@pytest.mark.parametrize("num_tokens", [1, 7, 14, 28, 33])
def test_ctx_norm_rope_matches_per_layer_kernel(num_tokens):
    mla_rope = _mla_rope()

    layers = 5
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    latent = _rows(num_tokens, layers * HEAD_DIM, gen).view(num_tokens, layers, HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(layers, KV_LORA, device="cuda", generator=gen)).to(
        torch.bfloat16
    )
    table = _freqs_table(4096, gen)
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda", generator=gen)

    out = mla_rope.dspark_mla_ctx_norm_rope(
        latent, torch.nn.Parameter(weight), positions, table, EPS
    )

    compiled = mla_rope._compile_rmsnorm_rope_strided(
        HEAD_DIM, ROPE, 1, EPS, True, True, False, KV_LORA
    )
    gathered = table[positions].contiguous()
    ref = torch.empty_like(out)
    for layer in range(layers):
        compiled(latent[:, layer], weight[layer], gathered, ref[layer])
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("num_tokens", [7, 14, 28, 32])
@pytest.mark.parametrize("page_size", [32, 64])
def test_block_rope_matches_separate_launches(num_tokens, page_size):
    import importlib

    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rope_into,
    )

    mla_rope = _mla_rope()
    scatter_rows_to_pool = importlib.import_module(
        f"{_SPEC}.spec_tail_kernels"
    ).scatter_rows_to_pool

    heads = 4
    gen = torch.Generator(device="cuda").manual_seed(100 * num_tokens + page_size)
    q = _rows(num_tokens * heads, NOPE + ROPE, gen).view(num_tokens, heads, NOPE + ROPE)
    # The block latents are the kv half of the fused [q_a | kv_a] GEMM output.
    kv = _rows(num_tokens, HEAD_DIM, gen, row_stride=1536 + HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(KV_LORA, device="cuda", generator=gen)).to(torch.bfloat16)
    table = _freqs_table(8192, gen)
    positions = torch.randint(0, 8192, (num_tokens,), device="cuda", generator=gen)
    # An interleaved multi-layer page space, like the draft KV manager's view.
    num_pages = 40
    space = torch.randn(num_pages, 3, page_size, HEAD_DIM, device="cuda", generator=gen)
    pool = space.to(torch.bfloat16)[:, 1]
    pages = torch.randperm(num_pages, device="cuda", generator=gen)[:num_tokens].long()
    offsets = torch.randint(0, page_size, (num_tokens,), device="cuda", generator=gen)

    query = torch.full((num_tokens, heads, HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda")
    pool_new = pool.clone()
    mla_rope.dspark_mla_block_rope(
        q, query, kv, torch.nn.Parameter(weight), positions, table, pool_new, pages, offsets, EPS
    )

    gathered = table[positions].contiguous()
    query_ref = torch.full_like(query, 7.0)
    cute_dsl_dspark_rope_into(q, gathered, query_ref, heads, ROPE, KV_LORA)
    blk_kv = mla_rope.k3r_dspark_rmsnorm_rope(
        kv, weight, gathered, 1, ROPE, EPS, True, True, False, KV_LORA
    )
    pool_ref = pool.clone()
    scatter_rows_to_pool(blk_kv, pool_ref, pages, offsets)
    torch.cuda.synchronize()
    assert torch.equal(_bits(query), _bits(query_ref))
    assert torch.equal(_bits(pool_new), _bits(pool_ref))


@pytest.mark.parametrize("num_rows", [1, 7, 33])
@pytest.mark.parametrize("row_stride", [None, 1536 + HEAD_DIM])
@pytest.mark.parametrize(
    "norm_dim,apply_weight", [(KV_LORA, True), (None, True), (None, False)], ids=str
)
def test_strided_rmsnorm_rope_matches_main_op(num_rows, row_stride, norm_dim, apply_weight):
    """``k3r_dspark_rmsnorm_rope`` reads a row-strided view in place; main's op, compiled for
    compact rows, gets a contiguous copy of the same rows. Same bits."""
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
    )

    mla_rope = _mla_rope()
    gen = torch.Generator(device="cuda").manual_seed(num_rows * 11 + (row_stride or 0))
    x = _rows(num_rows, HEAD_DIM, gen, row_stride=row_stride)
    weight_len = HEAD_DIM if norm_dim is None else norm_dim
    weight = (1.0 + 0.5 * torch.randn(weight_len, device="cuda", generator=gen)).to(torch.bfloat16)
    table = _freqs_table(num_rows, gen)
    args = (ROPE, EPS, apply_weight, True, False, norm_dim)
    out = mla_rope.k3r_dspark_rmsnorm_rope(x, weight, table, 1, *args)
    ref = cute_dsl_dspark_rmsnorm_rope(x.contiguous(), weight, table, 1, *args)
    torch.cuda.synchronize()
    assert out.is_contiguous() and out.shape == x.shape
    assert torch.equal(_bits(out), _bits(ref))
