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
"""The K3 MLA drafter's RMSNorm + RoPE entries (``ops/spec/k3_dspark_*``) write bitwise what
the launches they replace write: the ``TLLM_DSPARK_TAIL_V2`` kernels against the
per-layer / separate launches, and the row-strided ``k3_dspark_rmsnorm_rope`` against main's
``cute_dsl_dspark_rmsnorm_rope`` on a contiguous copy."""

import zlib

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="needs an SM100-family GPU"
)

_OPS = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec"

KV_LORA = 512
ROPE = 64
NOPE = 128
HEAD_DIM = KV_LORA + ROPE
EPS = 1e-5


def _entry(name):
    import importlib

    return getattr(importlib.import_module(f"{_OPS}.{name}"), name)


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
    layers = 5
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    latent = _rows(num_tokens, layers * HEAD_DIM, gen).view(num_tokens, layers, HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(layers, KV_LORA, device="cuda", generator=gen)).to(
        torch.bfloat16
    )
    table = _freqs_table(4096, gen)
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda", generator=gen)

    out = _entry("k3_dspark_mla_ctx_norm_rope")(
        latent, torch.nn.Parameter(weight), positions, table, EPS
    )

    import importlib

    into = importlib.import_module(f"{_OPS}.k3_dspark_rmsnorm_rope_into")
    compiled = into.build_rmsnorm_rope_into(HEAD_DIM, ROPE, 1, EPS, True, True, False, KV_LORA)
    gathered = table[positions].contiguous()
    ref = torch.empty_like(out)
    for layer in range(layers):
        into.k3_dspark_rmsnorm_rope_into(
            latent[:, layer], weight[layer], gathered, ref[layer], compiled
        )
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("num_tokens", [7, 14, 28, 32])
@pytest.mark.parametrize("page_size", [32, 64])
def test_block_rope_matches_separate_launches(num_tokens, page_size):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rope_into,
    )

    scatter_rows_to_pool = _entry("k3_scatter_rows_to_pool")

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
    _entry("k3_dspark_mla_block_rope")(
        q, query, kv, torch.nn.Parameter(weight), positions, table, pool_new, pages, offsets, EPS
    )

    gathered = table[positions].contiguous()
    query_ref = torch.full_like(query, 7.0)
    cute_dsl_dspark_rope_into(q, gathered, query_ref, heads, ROPE, KV_LORA)
    blk_kv = _entry("k3_dspark_rmsnorm_rope")(
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
    """``k3_dspark_rmsnorm_rope`` reads a row-strided view in place; main's op, compiled for
    compact rows, gets a contiguous copy of the same rows. Same bits."""
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
    )

    gen = torch.Generator(device="cuda").manual_seed(num_rows * 11 + (row_stride or 0))
    x = _rows(num_rows, HEAD_DIM, gen, row_stride=row_stride)
    weight_len = HEAD_DIM if norm_dim is None else norm_dim
    weight = (1.0 + 0.5 * torch.randn(weight_len, device="cuda", generator=gen)).to(torch.bfloat16)
    table = _freqs_table(num_rows, gen)
    args = (ROPE, EPS, apply_weight, True, False, norm_dim)
    out = _entry("k3_dspark_rmsnorm_rope")(x, weight, table, 1, *args)
    ref = cute_dsl_dspark_rmsnorm_rope(x.contiguous(), weight, table, 1, *args)
    torch.cuda.synchronize()
    assert out.is_contiguous() and out.shape == x.shape
    assert torch.equal(_bits(out), _bits(ref))


# The certified cells of the four entries, each against the entry's own FP64 reference.

pytest.importorskip("cutlass")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec import (  # noqa: E402, E501
    k3_dspark_mla_block_rope as _block_rope_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec import (  # noqa: E402, E501
    k3_dspark_mla_ctx_norm_rope as _ctx_norm_rope_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec import (  # noqa: E402, E501
    k3_dspark_rmsnorm_rope as _rmsnorm_rope_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec import (  # noqa: E402, E501
    k3_dspark_rmsnorm_rope_into as _rmsnorm_rope_into_entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402, F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

rmsnorm_rope_op = _rmsnorm_rope_entry.k3_dspark_rmsnorm_rope
rmsnorm_rope_into_op = _rmsnorm_rope_into_entry.k3_dspark_rmsnorm_rope_into
ctx_norm_rope_op = _ctx_norm_rope_entry.k3_dspark_mla_ctx_norm_rope
block_rope_op = _block_rope_entry.k3_dspark_mla_block_rope


def _on_archs(op) -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in op.ARCHS}


def _cells(op):
    return pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])


def _cell_gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2 ulp per element against the row scale, 0.5 ulp relative RMS.

    One BF16 rounding on each side of an FP32 (kernel) against an FP64 (reference) RMSNorm and
    rotation. The FP32 sum of squares over 512 columns and the approximate reciprocal square root
    move a normalized value by a few FP32 ulps, the rotation's products by one; so the two differ
    only where that value sits that close to a BF16 rounding midpoint, by one BF16 step of that
    element: at most 2 units of ``ULP[bf16]`` against the row scale. Those flips are rare enough
    that the relative RMS stays far below half an ulp. A wrong pair order, sign or position misses
    by O(1).
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


def _norm_weight(n, gen, layers=None):
    shape = (n,) if layers is None else (layers, n)
    return (1.0 + 0.5 * torch.randn(*shape, device="cuda", generator=gen)).to(torch.bfloat16)


@pytest.mark.skipif(not _on_archs(rmsnorm_rope_op), reason="not an ARCHS architecture")
@_cells(rmsnorm_rope_op)
def test_rmsnorm_rope_certified_cells(cell, certifying):  # noqa: F811 — fixture by name
    certifying(rmsnorm_rope_op)
    gen = _cell_gen(cell)
    rows = cell.spec["rows"]
    x = _rows(rows, HEAD_DIM, gen, row_stride=cell.spec["row_stride"])
    weight = _norm_weight(KV_LORA, gen)
    table = _freqs_table(rows, gen)
    before = x.clone()
    args = (weight, table, 1, ROPE, EPS, True, True, False, KV_LORA)
    out = rmsnorm_rope_op(x, *args)
    assert out.is_contiguous() and out.shape == x.shape and out.dtype == torch.bfloat16
    assert torch.equal(_bits(x), _bits(before)), "the op must not write its input"
    _compare(out, rmsnorm_rope_op.reference(x, *args))


@pytest.mark.skipif(not _on_archs(rmsnorm_rope_into_op), reason="not an ARCHS architecture")
@_cells(rmsnorm_rope_into_op)
def test_rmsnorm_rope_into_certified_cells(cell, certifying):  # noqa: F811 — fixture by name
    certifying(rmsnorm_rope_into_op)
    gen = _cell_gen(cell)
    rows, layers = cell.spec["rows"], cell.spec["layers"]
    latent = _rows(rows, layers * HEAD_DIM, gen).view(rows, layers, HEAD_DIM)
    weight = _norm_weight(KV_LORA, gen, layers)
    table = _freqs_table(rows, gen)
    compiled = _rmsnorm_rope_into_entry.build_rmsnorm_rope_into(
        HEAD_DIM, ROPE, 1, EPS, True, True, False, KV_LORA
    )
    before = latent.clone()
    out = torch.full((layers, rows, HEAD_DIM), 3.0, dtype=torch.bfloat16, device="cuda")
    for layer in range(layers):
        # Each layer's slice of the GEMM output is read in place, as the drafter does.
        assert (
            rmsnorm_rope_into_op(
                latent[:, layer], weight[layer], table, out[layer], compiled
            ).data_ptr()
            == out[layer].data_ptr()
        )
    assert torch.equal(_bits(latent), _bits(before)), "the op must not write its input"
    for layer in range(layers):
        _compare(
            out[layer],
            rmsnorm_rope_into_op.reference(
                latent[:, layer], weight[layer], table, out[layer], compiled
            ),
        )


@pytest.mark.skipif(not _on_archs(ctx_norm_rope_op), reason="not an ARCHS architecture")
@_cells(ctx_norm_rope_op)
def test_ctx_norm_rope_certified_cells(cell, certifying):  # noqa: F811 — fixture by name
    certifying(ctx_norm_rope_op)
    gen = _cell_gen(cell)
    rows, layers = cell.spec["rows"], cell.spec["layers"]
    latent = _rows(rows, layers * HEAD_DIM, gen).view(rows, layers, HEAD_DIM)
    weight = torch.nn.Parameter(_norm_weight(KV_LORA, gen, layers), requires_grad=False)
    table = _freqs_table(16384, gen)
    positions = torch.randint(0, 16384, (rows,), device="cuda", generator=gen)
    before = latent.clone()
    out = ctx_norm_rope_op(latent, weight, positions, table, EPS)
    assert out.shape == (layers, rows, HEAD_DIM) and out.dtype == torch.bfloat16
    assert torch.equal(_bits(latent), _bits(before)), "the op must not write its input"
    _compare(out, ctx_norm_rope_op.reference(latent, weight, positions, table, EPS))


@pytest.mark.skipif(not _on_archs(block_rope_op), reason="not an ARCHS architecture")
@_cells(block_rope_op)
def test_block_rope_certified_cells(cell, certifying):  # noqa: F811 — fixture by name
    certifying(block_rope_op)
    gen = _cell_gen(cell)
    tokens, heads, page_size, num_pages = cell.spec["tokens"], 64, 128, 24
    q = _rows(tokens * heads, NOPE + ROPE, gen).view(tokens, heads, NOPE + ROPE)
    kv_row_stride = cell.spec["kv_row_stride"]
    kv = _rows(
        tokens, HEAD_DIM, gen, row_stride=kv_row_stride if kv_row_stride != HEAD_DIM else None
    )
    weight = torch.nn.Parameter(_norm_weight(KV_LORA, gen), requires_grad=False)
    table = _freqs_table(16384, gen)
    positions = torch.randint(0, 16384, (tokens,), device="cuda", generator=gen)
    # One layer's view of an interleaved multi-layer page space; distinct (page, offset) slots.
    space = torch.randn(num_pages, 3, page_size, HEAD_DIM, device="cuda", generator=gen)
    pool = space.to(torch.bfloat16)[:, 1]
    slots = torch.randperm(num_pages * page_size, device="cuda", generator=gen)[:tokens]
    pages, offsets = slots // page_size, slots % page_size
    query = torch.full((tokens, heads, HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda")
    ref_query, ref_pool = block_rope_op.reference(
        q, query, kv, weight, positions, table, pool, pages, offsets, EPS
    )
    before_q, before_kv = q.clone(), kv.clone()
    block_rope_op(q, query, kv, weight, positions, table, pool, pages, offsets, EPS)
    assert torch.equal(_bits(q), _bits(before_q)) and torch.equal(_bits(kv), _bits(before_kv))
    # Columns and slots the op does not write keep their bits.
    assert torch.equal(_bits(query[..., :KV_LORA]), _bits(ref_query[..., :KV_LORA]))
    written = torch.zeros(num_pages, page_size, dtype=torch.bool, device="cuda")
    written[pages, offsets] = True
    assert torch.equal(_bits(pool[~written]), _bits(ref_pool[~written]))
    _compare(query[..., KV_LORA:], ref_query[..., KV_LORA:])
    _compare(pool[pages, offsets], ref_pool[pages, offsets])
