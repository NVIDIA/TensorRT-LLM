# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the DFlash2 drafter's query fold (q_a norm + absorbed query + RoPE + block store, one launch).

Every certified cell holds the block latents bit for bit against ``dflash2_mla_block_rope`` (the launch whose store it
replaces) and the query against the FP64 path of the original weights (q_a RMSNorm, q_b, W_UK, RoPE), within the
envelope of the unfused absorbed path (q_a norm -> absorbed warp split-K GEMM -> ``dflash2_mla_block_rope``) it
replaces; and against the op's own FP64 reference. The fold is driven between a producer and a reader with PDL on,
captured in a CUDA graph, and its replay must equal the eager call.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.rubin_ull_dflash2_query_fold import (  # noqa: E501
    PreparedQueryFold,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_main import (
    module as vllm_main,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_conv_add_rmsnorm import (  # noqa: E501
    dflash2_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (
    dflash2_mla_block_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_query_fold import (  # noqa: E501
    dflash2_query_fold as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_query_fold import (  # noqa: E501
    fold_query_weight,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)

HEADS, NOPE, ROPE, KV_LORA, Q_LORA = 6, 128, 64, 512, 1536
HEAD = KV_LORA + ROPE
EPS = 1e-6
POSITIONS = 4096
PAGES = 64
# The helix16 Rubin rows' warp split-K tactic of the absorbed query [3456, 1536].
TACTIC = (16, 8, 256, 6, 2)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.fixture(scope="module")
def layer():
    g = torch.Generator(device="cuda").manual_seed(13)
    q_b = (
        torch.randn(HEADS * (NOPE + ROPE), Q_LORA, generator=g, device="cuda") * Q_LORA**-0.5
    ).bfloat16()
    k_b = (torch.randn(HEADS, NOPE, KV_LORA, generator=g, device="cuda") * NOPE**-0.5).bfloat16()
    q_norm = (1 + 0.1 * torch.randn(Q_LORA, generator=g, device="cuda")).bfloat16()
    kv_norm = (1 + 0.1 * torch.randn(KV_LORA, generator=g, device="cuda")).bfloat16()
    angles = torch.rand(POSITIONS, ROPE // 2, generator=g, device="cuda") * 6.283
    freqs = torch.polar(torch.full_like(angles, 1.0), angles).contiguous()
    w = vllm_main("cutedsl.gemm.warp_splitk")
    tactic = w.WarpSplitKTactic(*TACTIC)
    fold = PreparedQueryFold(fold_query_weight(q_b, k_b, q_norm), kv_norm, EPS, tactic)
    q = q_b.view(HEADS, -1, Q_LORA).double()
    absorbed = (
        torch.cat((torch.matmul(k_b.double().transpose(1, 2), q[:, :NOPE]), q[:, NOPE:]), dim=1)
        .reshape(-1, Q_LORA)
        .bfloat16()
        .contiguous()
    )
    unfused = w.PreparedWarpSplitK(absorbed, {m: tactic for m in range(1, 9)})
    return dict(
        q_b=q_b, k_b=k_b, q_norm=q_norm, kv_norm=kv_norm, freqs=freqs, fold=fold, unfused=unfused
    )


def _inputs(rows: int, seed: int, positions_dtype, index_dtype, page_size: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_kv = (3 * torch.randn(rows, Q_LORA + HEAD, generator=g, device="cuda")).bfloat16()
    positions = torch.randint(0, POSITIONS, (rows,), generator=g, device="cuda").to(positions_dtype)
    slots = torch.randperm(PAGES * page_size, generator=g, device="cuda")[:rows]
    pages, offsets = (slots // page_size).to(index_dtype), (slots % page_size).to(index_dtype)
    pool = (torch.randn(PAGES, page_size, HEAD, generator=g, device="cuda")).to(torch.float8_e4m3fn)
    return q_kv, positions, pool, pages, offsets


def _fp64_query(layer, q_kv, positions):
    """The original weights' path in FP64: q_a RMSNorm, q_b, W_UK absorption, RoPE (adjacent pairs)."""
    x = q_kv[:, :Q_LORA].double()
    qa = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * layer["q_norm"].double()
    q = (qa @ layer["q_b"].double().t()).view(-1, HEADS, NOPE + ROPE)
    latent = torch.einsum("rak,akl->ral", q[..., :NOPE], layer["k_b"].double())
    pe = torch.view_as_complex(q[..., NOPE:].reshape(*q.shape[:2], -1, 2).contiguous())
    rot = layer["freqs"][positions.long()].unsqueeze(1).to(torch.complex128)
    return torch.cat((latent, torch.view_as_real(pe * rot).flatten(-2)), dim=-1)


def _unfused(layer, q_kv, positions, pool, pages, offsets):
    """q_a norm -> absorbed query GEMM -> dflash2_mla_block_rope: the path the fold replaces."""
    rows = q_kv.shape[0]
    qa = dflash2_rmsnorm(q_kv[:, :Q_LORA], layer["q_norm"], EPS, True)
    query = layer["unfused"].run(
        qa, torch.empty(rows, HEADS * HEAD, dtype=qa.dtype, device=qa.device)
    )
    query = query.view(rows, HEADS, HEAD)
    pool = pool.clone()
    dflash2_mla_block_rope(
        query[..., KV_LORA - NOPE :], query, q_kv[:, Q_LORA:], layer["kv_norm"], EPS, layer["freqs"], positions,
        pool, pages, offsets,
    )  # fmt: skip
    return query, pool


def _rel_l2(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, layer) -> None:
    s = cell.spec
    q_kv, positions, pool, pages, offsets = _inputs(
        s["rows"],
        zlib.crc32(cell.why.encode()) % 2**31,
        s["positions"],
        s["index_dtype"],
        s["page_size"],
    )
    pool_in = pool.clone()
    with validating(op):
        query, pool_out = op(layer["fold"], q_kv, layer["freqs"], positions, pool, pages, offsets)
    ref_query, ref_pool = _unfused(layer, q_kv, positions, pool_in, pages, offsets)
    torch.cuda.synchronize()
    assert torch.equal(pool_out.view(torch.uint8), ref_pool.view(torch.uint8)), (
        "the block latents are not dflash2_mla_block_rope's bytes"
    )
    f64 = _fp64_query(layer, q_kv, positions)
    err, base = _rel_l2(query, f64), _rel_l2(ref_query, f64)
    assert err <= 1.25 * base + 1e-4, f"query rel-L2 {err:.3e} against FP64, unfused {base:.3e}"
    err_max = ((query.double() - f64).abs().max() / f64.abs().max()).item()
    assert err_max < 2e-2, f"query max-rel error {err_max:.3e} against FP64"
    own, _ = op.reference(layer["fold"], q_kv, layer["freqs"], positions, pool_in, pages, offsets)
    assert _rel_l2(query, own.double()) < 4e-3, "query far from the op's FP64 reference"


def test_graph_replay_equals_eager_between_pdl_neighbours(layer) -> None:
    """producer -> fold -> reader captured in a graph: the replay's bits are the eager ones."""
    q_kv0, positions, pool, pages, offsets = _inputs(8, 5, torch.int64, torch.int64, 64)

    def chain():
        q_kv = q_kv0 * 1.0  # the producer kernel the fold waits on
        query, _ = op(layer["fold"], q_kv, layer["freqs"], positions, pool, pages, offsets)
        return query * 1.0, pool.view(torch.uint8).clone()

    eager = [t.clone() for t in chain()]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = chain()
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(outs, eager):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_is_valid_refuses_more_rows_than_a_token_tile(layer) -> None:
    q_kv, positions, pool, pages, offsets = _inputs(8, 3, torch.int64, torch.int64, 64)
    q_kv = torch.cat((q_kv, q_kv[:1]))
    with pytest.raises(AssertionError, match="rows"):
        op.is_valid(layer["fold"], q_kv, layer["freqs"], positions, pool, pages, offsets)
