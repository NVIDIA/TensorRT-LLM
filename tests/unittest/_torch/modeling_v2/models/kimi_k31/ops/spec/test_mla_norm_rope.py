# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 MLA latent RMSNorm + RoPE entries (CuTe DSL).

Every certified cell is gated bit for bit against the entry's reference (the drafter's torch
math): the context latents, the query's rope half, and the pool rows (BF16 or FP8 bytes). The
rest of the query and the pool must not change.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    build_mla_yarn_freqs_cis,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    dflash2_mla_block_rope as block_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    dflash2_mla_ctx_norm_rope as ctx_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (  # noqa: E501
    dflash2_mla_ctx_norm_rope_store as store_op,
)
from tensorrt_llm._torch.models.modeling_dspark import build_dspark_mla_yarn_freqs_cis
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)

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


def test_yarn_table_is_the_engine_table(freqs_cis):
    """The family YaRN table is bitwise the engine DSpark drafter's (the table every measured drafter run used)."""
    engine = build_dspark_mla_yarn_freqs_cis(device="cuda", **_YARN)
    assert torch.equal(
        torch.view_as_real(freqs_cis).view(torch.int32),
        torch.view_as_real(engine).view(torch.int32),
    )


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
    pool = torch.randn(num_pages, page_size, rank + rope, generator=g, device="cuda")
    if spec["pool"] == "fp8":
        pool = pool.clamp(-448, 448).to(torch.float8_e4m3fn)
    else:
        pool = pool.bfloat16()
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


@pytest.mark.parametrize("cell", ctx_op.CELLS, ids=[c.why[:48] for c in ctx_op.CELLS])
def test_ctx_certified_cells(cell, freqs_cis, certifying) -> None:  # noqa: F811
    certifying(ctx_op)
    args = _build_ctx(cell.spec, freqs_cis, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = ctx_op(*args)
    assert out.is_contiguous()
    _assert_bitwise(out, ctx_op.reference(*args), torch.int16, "context latents")


@pytest.mark.parametrize("cell", block_op.CELLS, ids=[c.why[:48] for c in block_op.CELLS])
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
    bits = torch.int16 if pool.dtype == torch.bfloat16 else torch.uint8
    pool, pool_in, want_pool = (t.view(bits) for t in (pool, pool_in, want_pool))
    written = torch.zeros(pool.shape[:2], dtype=torch.bool, device=pool.device)
    written[args[8].long(), args[9].long()] = True
    assert torch.equal(pool[~written], pool_in[~written]), "stray pool write"
    _assert_bitwise(pool[written], want_pool[written], bits, f"{cell.spec['pool']} pool rows")


def test_one_launch_per_call(freqs_cis) -> None:
    """Each op is one kernel; the references are the launch counts they replace (printed)."""
    cases = (
        ("ctx", ctx_op, _build_ctx(ctx_op.CELLS[1].spec, freqs_cis, seed=1)),
        ("block", block_op, _build_block(block_op.CELLS[0].spec, freqs_cis, seed=1)),
    )
    for name, op, args in cases:
        op(*args), op.reference(*args)
        torch.cuda.synchronize()
        counts = {}
        for which, fn in (("op", op), ("reference", op.reference)):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                fn(*args)
                torch.cuda.synchronize()
            counts[which] = sum(
                1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
            )
        print(f"{name}: device launches per call: {counts}")
        assert counts["op"] == 1, (name, counts)


def test_block_cuda_graph_replay(freqs_cis) -> None:
    """Captured after one eager call, replayed on new rows: the same values as eager."""
    spec = block_op.CELLS[2].spec
    args = _build_block(spec, freqs_cis, seed=2)
    block_op(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        block_op(*args)
    for seed in (3, 4):
        new = _build_block(spec, freqs_cis, seed=seed)
        for i in (0, 1, 2, 6):  # q, query, kv, positions: the per-step inputs
            args[i].copy_(new[i])
        eager = [t.clone() for t in (args[1], args[7])]
        block_op(args[0], eager[0], *args[2:7], eager[1], *args[8:])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(args[1].view(torch.int16), eager[0].view(torch.int16))
        assert torch.equal(args[7].view(torch.int16), eager[1].view(torch.int16))  # BF16 pool


def test_guard_refuses_non_bf16_latents(freqs_cis) -> None:
    latent, *rest = _build_ctx(ctx_op.CELLS[0].spec, freqs_cis, seed=5)
    with validating(ctx_op), pytest.raises(AssertionError, match="BF16"):
        ctx_op(latent.float(), *rest)


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


@pytest.mark.parametrize("cell", store_op.CELLS, ids=[c.why[:48] for c in store_op.CELLS])
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
