# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (SM 10.7) for the attention/m3_fullctx_decode entry: the dense layers' full-context FP8 decode on PrimTS
split-KV decode at the recipe the target pins per (query rows per request, requests).

Every cell runs against the fp32 reference (q rounded to E4M3 first, as the kernel runs it), eager and replayed from a
CUDA graph (the same bits), and twice (the same bits). The K/V pages are read through the strided views of a flat
sub-page pool, as the KV cache manager hands them out.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.attention.m3_fullctx_decode import (  # noqa: E501
    HEAD_DIM,
    HEADS,
    KV_HEADS,
    MAX_KV_LEN,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.attention.m3_fullctx_decode import (  # noqa: E501
    m3_fullctx_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_fullctx_decode requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the tp4 target is certified on SM 10.7 only", allow_module_level=True)

PAGE = 128
SM_SCALE = HEAD_DIM**-0.5
# The FP8-KV band of the in-tree dense decode test (test_minimax_m3_dense_decode.test_matches_reference).
FP8_TOL = 6e-2


class _Pool:
    """The one method the entry asks of a KV cache manager."""

    def __init__(self, pool: torch.Tensor, subpages: int):
        self.pool = pool
        self.subpages = subpages

    def get_kv_subpage_pool(self, layer_idx: int, kv_layout: str = "HND"):
        assert kv_layout == "HND"
        return self.pool, self.subpages


def _inputs(kv_lens, qlen: int, seed: int, subpages: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(kv_lens)
    pages = max((n + PAGE - 1) // PAGE for n in kv_lens)
    slots = batch * pages + 3
    pool = torch.randn(
        (slots * subpages, KV_HEADS, PAGE, HEAD_DIM), device="cuda", generator=g
    ).clamp(-4, 4)
    perm = torch.randperm(slots, generator=g, device="cuda")[: batch * pages]
    block_table = perm.reshape(batch, pages).to(torch.int32)
    # The last third of every request's values is shifted by +1, so a split that missed the tail fails the band.
    for b, n in enumerate(kv_lens):
        start = 2 * n // 3
        for page in range(start // PAGE, (n + PAGE - 1) // PAGE):
            v = int(block_table[b, page]) * subpages + 1
            pool[v, :, (start % PAGE if page == start // PAGE else 0) :] += 1.0
    # q as the layer hands it over on decode steps: the FP8 query columns of the fused norm/RoPE kernel's
    # [q | k | v] output (row stride wider than q), so no cast makes it compact.
    qkv = torch.randn(
        (batch * qlen, (HEADS + 2 * KV_HEADS) * HEAD_DIM), device="cuda", generator=g
    ).to(torch.float8_e4m3fn)
    q = qkv[:, : HEADS * HEAD_DIM].view(batch * qlen, HEADS, HEAD_DIM)
    seq_lens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    return q, _Pool(pool.to(torch.float8_e4m3fn), subpages), block_table, seq_lens


@pytest.mark.parametrize("subpages", [2, 3], ids=["kv", "kv+index"])
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, subpages, certifying) -> None:  # noqa: F811
    spec = cell.spec
    qlen, nreq = spec["qlen"], len(spec["kv_lens"])
    recipe = dispatch.FULLCTX_DECODE_RECIPES[(qlen, nreq)]
    q, manager, block_table, seq_lens = _inputs(
        spec["kv_lens"], qlen, zlib.crc32(cell.why.encode()) % 2**31, subpages
    )
    out = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
    args = (q, manager, 0, block_table, seq_lens, SM_SCALE, out, qlen, recipe)
    certifying(op)
    ref = op.reference(*args)
    eager = op(*args).clone()
    torch.testing.assert_close(eager.float(), ref, rtol=FP8_TOL, atol=FP8_TOL)
    again = op(*args).clone()
    assert torch.equal(again, eager), "not deterministic"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "the CUDA-graph replay differs from the eager call"


# Plans compiled at other K/V stage depths stay alive for the whole session: a compiled decode is cached per config
# and is only ever run with the buffers of a live plan.
_OTHER_DEPTH_PLANS = []


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_the_kv_stage_depth_keeps_the_bits(cell) -> None:
    """kv_stages only sets how many K/V tiles the load pipeline keeps in flight: the recipe's plan and plans compiled
    at 12 and at 4 stages give the same bits."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
        m3_fullctx_decode as kernel,
    )
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    if not kernel.has_own_reducer():
        pytest.skip("this build does not carry minimax_m3_sm107tp4_fullctx_reduce")
    spec = cell.spec
    qlen, nreq = spec["qlen"], len(spec["kv_lens"])
    recipe = dispatch.FULLCTX_DECODE_RECIPES[(qlen, nreq)]
    q, manager, block_table, seq_lens = _inputs(spec["kv_lens"], qlen, 13, 2)
    k_pages, v_pages = kernel.kv_page_views(manager.pool, manager.subpages)
    kernel.prebuild(q.device, {(qlen, nreq): recipe}, MAX_KV_LEN, True, get_env_enable_pdl())
    plans = [kernel.plan_for(q.device, qlen, nreq, True)]
    for stages in (12, 4):
        config = {**recipe["config"], "kv_stages": stages}
        plans.append(
            kernel.FullCtxPlan(
                q.device,
                qlen,
                nreq,
                MAX_KV_LEN,
                {**recipe, "config": config},
                True,
                get_env_enable_pdl(),
            )
        )
        _OTHER_DEPTH_PLANS.append(plans[-1])
    outs = []
    for plan in plans:
        out = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
        plan.run(q, k_pages, v_pages, block_table, seq_lens, SM_SCALE, out)
        outs.append(out)
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


def test_the_cells_cover_the_dispatch_rows() -> None:
    certified = {(c.spec["qlen"], len(c.spec["kv_lens"])) for c in op.CELLS}
    assert set(dispatch.FULLCTX_DECODE.keys) <= certified


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_the_bf16_merge_is_the_primts_reducer_to_one_ulp(cell) -> None:
    """The BF16 split-KV merge op against PrimTS's own FP16 reducer and the cast: the same partials, merged once
    in fp32 and rounded once, so within one BF16 spacing of the FP16-then-BF16 result."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
        m3_fullctx_decode as kernel,
    )

    if not kernel.has_own_reducer():
        pytest.skip("this build does not carry minimax_m3_sm107tp4_fullctx_reduce")
    spec = cell.spec
    qlen, nreq = spec["qlen"], len(spec["kv_lens"])
    recipe = dispatch.FULLCTX_DECODE_RECIPES[(qlen, nreq)]
    q, manager, block_table, seq_lens = _inputs(spec["kv_lens"], qlen, 11, 2)
    outs = []
    for own in (True, False):
        out = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
        op(q, manager, 0, block_table, seq_lens, SM_SCALE, out, qlen, recipe, own_reducer=own)
        outs.append(out.float())
    ours, primts = outs
    _, exp = torch.frexp(primts.abs())
    spacing = torch.where(
        primts == 0,
        torch.full_like(primts, 2.0**-133),
        torch.ldexp(torch.ones_like(primts), exp - 8),
    )
    assert bool(((ours - primts).abs() <= spacing).all())


def _tile_edge_kv_lens(plan, qlen: int):
    """KV lengths around every kind of boundary the split ranges have, so each query row of the Q CTA in turn ends
    its keys on a KV tile edge (k * tile + d, d in -1 .. qlen - 1): the first tiles, the reducer's 8-group stride, the
    lengths where a split's range grows (splits * insts tiles) and a split edge past it."""
    tile, insts, splits = plan.tile_kv, plan.insts_kv, int(plan.recipe["splits"])
    grow = splits * insts
    tiles = sorted({1, 2, 8, 9, grow - 1, grow, grow + 1, grow + 2 * insts, 2 * grow})
    return [k * tile + d for k in tiles for d in range(-1, qlen)]


@pytest.mark.parametrize("qlen,nreq", sorted(dispatch.FULLCTX_DECODE_RECIPES))
def test_rows_with_no_key_in_a_trailing_split(qlen, nreq) -> None:
    """A row whose keys end before its Q CTA's causal union does has trailing splits holding none of its keys
    (log2-LSE -inf); both reducers must leave them out. [1300, 1281] at 4 query rows is the batch a GSM8K decode
    step first hit (NaN rows from exp2(-inf - -inf))."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
        m3_fullctx_decode as kernel,
    )
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    recipe = dispatch.FULLCTX_DECODE_RECIPES[(qlen, nreq)]
    reducers = (True, False) if kernel.has_own_reducer() else (False,)
    kernel.prebuild(
        torch.device("cuda"), {(qlen, nreq): recipe}, MAX_KV_LEN, False, get_env_enable_pdl()
    )
    lens = _tile_edge_kv_lens(kernel.plan_for(torch.device("cuda"), qlen, nreq, False), qlen)
    batches = [[n] for n in lens] if nreq == 1 else [[a, b] for a, b in zip(lens, lens[::-1])]
    batches.append([1300, 1281][:nreq])
    failures = []
    for kv_lens in batches:
        q, manager, block_table, seq_lens = _inputs(kv_lens, qlen, sum(kv_lens), 2)
        out = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
        args = (q, manager, 0, block_table, seq_lens, SM_SCALE, out, qlen, recipe)
        ref = op.reference(*args)
        for own in reducers:
            out.zero_()
            op(*args, own_reducer=own)
            got = out.float()
            if not bool(torch.isfinite(got).all()) or not torch.allclose(
                got, ref, rtol=FP8_TOL, atol=FP8_TOL
            ):
                failures.append((kv_lens, "bf16-merge" if own else "primts-reducer"))
    assert not failures, failures


@pytest.mark.parametrize("qlen,nreq", sorted(dispatch.FULLCTX_DECODE_RECIPES))
def test_kv_lengths_past_the_policy_length(qlen, nreq) -> None:
    """The plans are compiled for MAX_KV_LEN of K/V with PrimTS's policy sized at 1 << 20 (POLICY_KV_LEN): caches just
    past the policy length (the AgentX bound is 1 << 20 plus the speculative rows) and well past it stay exact."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
        m3_fullctx_decode as kernel,
    )
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    recipe = dispatch.FULLCTX_DECODE_RECIPES[(qlen, nreq)]
    reducers = (True, False) if kernel.has_own_reducer() else (False,)
    lengths = [(1 << 20) + 1, (1 << 20) + 8] + ([1_500_001] if nreq == 1 else [])
    batches = [[n] * nreq if nreq == 1 else [n, (1 << 20) - 3] for n in lengths]
    failures = []
    for kv_lens in batches:
        q, manager, block_table, seq_lens = _inputs(kv_lens, qlen, 13, 2)
        out = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
        args = (q, manager, 0, block_table, seq_lens, SM_SCALE, out, qlen, recipe)
        ref = op.reference(*args)
        for own in reducers:
            kernel.prebuild(
                torch.device("cuda"), {(qlen, nreq): recipe}, MAX_KV_LEN, own, get_env_enable_pdl()
            )
            out.zero_()
            op(*args, own_reducer=own)
            got = out.float()
            if not bool(torch.isfinite(got).all()) or not torch.allclose(
                got, ref, rtol=FP8_TOL, atol=FP8_TOL
            ):
                failures.append((kv_lens, "bf16-merge" if own else "primts-reducer"))
    assert not failures, failures


def _swizzled_scale_offsets(rows: int, width: int) -> torch.Tensor:
    """Offsets of rows [0, rows)'s UE8M0 scales in the 128x4-swizzled layout of mxfp8_quantize(x, True)."""
    k_sf = width // 32
    k_tiles = (k_sf + 3) // 4
    r = torch.arange(rows, device="cuda").view(-1, 1)
    k = torch.arange(k_sf, device="cuda").view(1, -1)
    offsets = (
        (r // 128) * k_tiles * 512 + (k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4
    )
    return offsets.reshape(-1)


@pytest.mark.parametrize("splits,rows,seq_len_q", [(108, 8, 1), (64, 8, 1), (96, 32, 4)])
def test_the_merge_mxfp8_copy_is_stock_mxfp8_quantize(splits, rows, seq_len_q) -> None:
    """The TP4 merge's MXFP8 copy (the dense o_proj hand-off) equals mxfp8_quantize(out, True) bit for bit: E4M3 rows
    and the real rows' swizzled UE8M0 scales, with an all-zero head row (amax 0) and saturating outliers; the BF16
    output is the merge without the copy."""
    ops = torch.ops.trtllm
    if not hasattr(ops, "minimax_m3_sm107tp4_fullctx_reduce") or "q8" not in str(
        ops.minimax_m3_sm107tp4_fullctx_reduce.default._schema
    ):
        pytest.skip("needs the TP4 merge op with the MXFP8 copy")
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(splits * 7 + rows)
    nreq, kvh = 2, 1
    tokens = nreq * seq_len_q
    width = kvh * rows // seq_len_q * 128
    for dtype in (torch.float16, torch.bfloat16):
        po = torch.randn(nreq, kvh, splits, rows, 128, device=dev, generator=g) * 3
        po[0, 0, :, 1] = 0.0
        po[1, 0, :, 0, 5] = 6.0e4
        po = po.to(dtype).contiguous()
        ps = (torch.randn(nreq, kvh, splits, rows, device=dev, generator=g) * 4).contiguous()
        seq = torch.tensor([165521, 120011], dtype=torch.int32, device=dev)
        args = (po, ps, seq)
        tail = (seq_len_q, seq_len_q, 128, 1, seq_len_q > 1, False)
        plain = torch.zeros(tokens, width, dtype=torch.bfloat16, device=dev)
        ops.minimax_m3_sm107tp4_fullctx_reduce(*args, plain, *tail)
        out = torch.zeros(tokens, width, dtype=torch.bfloat16, device=dev)
        q8 = torch.empty(128, width, dtype=torch.float8_e4m3fn, device=dev)
        sf = torch.zeros(128 * (width // 32), dtype=torch.uint8, device=dev)
        ops.minimax_m3_sm107tp4_fullctx_reduce(*args, out, *tail, q8, sf)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), plain.view(torch.int16))
        stock_q, stock_sf = ops.mxfp8_quantize(out, True)
        assert torch.equal(
            q8[:tokens].view(torch.uint8), stock_q.view(torch.uint8).reshape(tokens, -1)
        ), dtype
        at = _swizzled_scale_offsets(tokens, width)
        assert torch.equal(sf[at], stock_sf.view(torch.uint8).reshape(-1)[at]), dtype
