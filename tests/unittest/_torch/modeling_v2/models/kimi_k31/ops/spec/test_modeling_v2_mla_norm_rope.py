# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 MLA latent RMSNorm + RoPE entries (CuTe DSL).

A sample of each entry's certified cells is gated bit for bit against the entry's reference (the drafter's torch
math): the context latents, the query's rope half, and the BF16 pool rows. The rest of the
query and the pool must not change.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    build_mla_yarn_freqs_cis,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    dflash2_mla_block_rope as block_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    dflash2_mla_ctx_norm_rope as ctx_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    dflash2_mla_ctx_norm_rope_store as store_op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

pytestmark = on_sm(100, 103, 107, reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x")

#: Positions of the drafter's YaRN table in these tests (past the 32k original context).
_MAX_POSITIONS = 65536


#: The K3.1 DFlash2 drafter's YaRN table: theta 50000, factor 32 over 32768, beta 32/1.
_YARN = dict(
    dim=64,
    base=50000.0,
    scaling_factor=32.0,
    original_max_position_embeddings=32768,
    max_position_embeddings=_MAX_POSITIONS,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=1.0,
)


@pytest.fixture(scope="module")
def freqs_cis() -> torch.Tensor:
    return build_mla_yarn_freqs_cis(device="cuda", **_YARN)


def _assert_bitwise(got, want, bits_dtype, what: str) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype, (what, got.shape, want.shape)
    mismatched = int(
        (got.contiguous().view(bits_dtype) != want.contiguous().view(bits_dtype)).sum()
    )
    assert mismatched == 0, f"{what}: {mismatched}/{got.numel()} values differ from the reference"


def _build_ctx(spec, freqs, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, layers, rank, rope = spec["tokens"], spec["layers"], spec["rank"], spec["rope"]
    latent = (2.0 * torch.randn(n, layers, rank + rope, generator=g, device="cuda")).bfloat16()
    weight = (1.0 + 0.2 * torch.randn(layers, rank, generator=g, device="cuda")).bfloat16()
    if spec["positions"] == "far":
        positions = torch.randint(
            32768, _MAX_POSITIONS, (n,), generator=g, device="cuda", dtype=torch.int32
        )
    else:
        positions = torch.randint(0, 40000, (n,), generator=g, device="cuda")
    return latent, weight, spec["eps"], freqs, positions


def _build_block(spec, freqs, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = spec["batch"] * spec["width"]
    heads, nope, rank, rope = spec["heads"], spec["nope"], spec["rank"], spec["rope"]
    # q is a view of the q_b projection output, query a fresh buffer with a known pattern.
    q = (2.0 * torch.randn(tokens, heads * (nope + rope), generator=g, device="cuda")).bfloat16()
    q = q.view(tokens, heads, nope + rope)
    query = torch.randn(tokens, heads, rank + rope, generator=g, device="cuda").bfloat16()
    kv = (2.0 * torch.randn(tokens, rank + rope, generator=g, device="cuda")).bfloat16()
    weight = (1.0 + 0.2 * torch.randn(rank, generator=g, device="cuda")).bfloat16()
    ctx = torch.randint(0, 30000, (spec["batch"], 1), generator=g, device="cuda")
    positions = (ctx + torch.arange(spec["width"], device="cuda")).view(-1)
    page_size = spec["page_size"]
    num_pages = tokens // page_size + 16
    pool = torch.randn(num_pages, page_size, rank + rope, generator=g, device="cuda").bfloat16()
    slots = torch.randperm(num_pages * page_size, generator=g, device="cuda")[:tokens]
    index_dtype = spec["index_dtype"]
    return (
        q,
        query,
        kv,
        weight,
        spec["eps"],
        freqs,
        positions,
        pool,
        (slots // page_size).to(index_dtype),
        (slots % page_size).to(index_dtype),
    )


CTX_CELLS = sample_cells(ctx_op.CELLS, size_key="tokens")[:2]


@pytest.mark.parametrize("cell", CTX_CELLS, ids=[c.why[:48] for c in CTX_CELLS])
def test_ctx_certified_cells(cell, freqs_cis, certifying) -> None:  # noqa: F811
    certifying(ctx_op)
    args = _build_ctx(cell.spec, freqs_cis, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = ctx_op(*args)
    assert out.is_contiguous()
    _assert_bitwise(out, ctx_op.reference(*args), torch.int16, "context latents")


BLOCK_CELLS = sample_cells(block_op.CELLS, size_key="batch")[:2]


@pytest.mark.parametrize("cell", BLOCK_CELLS, ids=[c.why[:48] for c in BLOCK_CELLS])
def test_block_certified_cells(cell, freqs_cis, certifying) -> None:  # noqa: F811
    certifying(block_op)
    args = _build_block(cell.spec, freqs_cis, seed=zlib.crc32(cell.why.encode()) % 2**31)
    query_in, pool_in = args[1].clone(), args[7].clone()
    want_query, want_pool = block_op.reference(*args)
    query, pool = block_op(*args)
    rank = cell.spec["rank"]
    # The latent half of the query belongs to the absorption GEMM: untouched.
    assert torch.equal(query[..., :rank].view(torch.int16), query_in[..., :rank].view(torch.int16))
    _assert_bitwise(query[..., rank:], want_query[..., rank:], torch.int16, "query rope")
    bits = torch.int16
    pool, pool_in, want_pool = (t.view(bits) for t in (pool, pool_in, want_pool))
    written = torch.zeros(pool.shape[:2], dtype=torch.bool, device=pool.device)
    written[args[8].long(), args[9].long()] = True
    assert torch.equal(pool[~written], pool_in[~written]), "stray pool write"
    _assert_bitwise(pool[written], want_pool[written], bits, "pool rows")


def _build_ctx_store(spec, freqs, seed):
    latent, weight, eps, freqs, positions = _build_ctx(spec, freqs, seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    n, layers, head_dim = latent.shape
    pool_rows = 33 * 32
    pool = torch.randn(layers, pool_rows, head_dim, generator=g, device="cuda").bfloat16()
    rows = torch.randperm(pool_rows, generator=g, device="cuda")[:n].long()
    keep = torch.zeros(n, dtype=torch.uint8, device="cuda")
    keep[torch.randperm(n, generator=g, device="cuda")[: spec["keep"]]] = 1
    return latent, weight, eps, freqs, positions, pool, rows, keep


STORE_CELLS = sample_cells(store_op.CELLS, size_key="tokens")[:2]


@pytest.mark.parametrize("cell", STORE_CELLS, ids=[c.why[:48] for c in STORE_CELLS])
def test_ctx_store_certified_cells(cell, freqs_cis, certifying) -> None:  # noqa: F811
    args = list(_build_ctx_store(cell.spec, freqs_cis, zlib.crc32(cell.why.encode()) % 2**31))
    # Negative and non-finite latents: the masked rows keep the sign of zero and the NaN.
    args[0][0, 0, 3] = float("-inf")
    args[0][-1, 1, 7] = float("nan")
    ref_args = list(args)
    ref_args[5] = args[5].clone()
    certifying(store_op)
    store_op(*args)
    store_op.reference(*ref_args)
    _assert_bitwise(args[5], ref_args[5], torch.int16, "pool")
