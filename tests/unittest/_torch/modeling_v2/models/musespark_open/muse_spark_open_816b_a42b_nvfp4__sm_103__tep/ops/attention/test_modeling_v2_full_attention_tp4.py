# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the full-attention decode entries at a tep4 rank's shape: 32 query / 4 KV heads.

``full_attention_split_kv`` and ``full_attention_prep_fold`` serve a tep4 rank with their two-pair
build (``kv_heads=4``): a grid of ``splits x 2`` CTAs, each running the tep8 build's computation
on one pair of KV heads (2 KV heads, their 16 query heads). ``combine_norm_gate`` merges the
partials of 32 heads (``heads=32``). What is gated here:

* each pair's partials are bit for bit the tep8 build's on that pair's slice of the operand and
  the cache (same splits), at short, long and 1M-key contexts and at the table width a 1M-token
  ``max_seq_len`` gives (512 page ids per CTA at ``SMs / 2`` splits): the tep4 build inherits
  every claim the tep8 op test certifies;
* merged by ``combine_only``, the attention output is no farther from an fp64 reference than the
  built-in trtllm-gen path at the same shape (max and mean-abs error both <= 1.0x the
  built-in's; the built-in rounds Q to one E4M3 term, the op keeps Q at BF16 precision);
* a PDL launch writes the same bits; repeated calls are bitwise equal;
* ``combine_norm_gate`` at 32 heads is bit for bit ``attention_norm_gate`` applied to
  ``combine_only``, and within its fp64 reference band;
* ``full_attention_prep_fold`` at tep4 writes the partials and the cache bytes of
  ``preattn_prep_append`` (32 / 4 heads) followed by ``full_attention_split_kv``, bit for bit,
  in checkpoint order and permuted, and its gate view is the projection's gate;
* the ops refuse a pool of another head count and a fused operand of the other rank's width.

``MODELING_V2_TIMING=1`` also runs the microbenchmark (printed, never gated).
"""

import math
import os
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    combine_norm_gate as cng,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    full_attention_prep_fold as fold_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    full_attention_split_kv as fa4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    norm_gate as ng,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    CKPT_EPS,
    CKPT_Q_SCALE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep_append import (  # noqa: E501
    preattn_prep_append,
)
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="full_attention_split_kv is built for sm_103",
)

op = fa4.full_attention_split_kv
fold_op = fold_mod.full_attention_prep_fold
DEV = "cuda"
FP8 = torch.float8_e4m3fn
HEAD_DIM, PAGE = fa4.HEAD_DIM, fa4.PAGE
#: One tep4 rank: 32 query heads, 4 KV heads; one pair (the tep8 build's shape): 16 / 2.
H, G = 32, 4
PAIR_H, PAIR_G = 16, 2
Q_SCALE = 5.473
EPS = 1e-5
#: The softmax scale ``run_generation`` passes to the prep (its FMHA scale words).
BMM1_HOST = 1.0 / math.sqrt(HEAD_DIM)


def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _splits() -> int:
    """The tep4 install's splits per pair: half the SMs."""
    return fa4.default_splits(G, _sm_count())


class _Step:
    """One decode step at ``heads`` / ``kv_heads``: a fused-QKV operand ``[q | k | v]`` and an
    FP8 pool in the TRT-LLM paged layout (logical page ``i`` is K block ``2 perm[i]`` and V
    block ``2 perm[i] + 1``)."""

    def __init__(self, rows, ctx, *, seed=0, heads=H, kv_heads=G, width=None, splits=None):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx, self.heads, self.kv_heads = rows, ctx, heads, kv_heads
        self.splits = _splits() if splits is None else splits
        pages = -(-ctx // PAGE)
        width = pages if width is None else width
        spare = width + 4
        perm = torch.randperm(spare, generator=g, device=DEV)[:width].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = torch.randn(2 * spare, kv_heads, PAGE, HEAD_DIM, generator=g, device=DEV).to(
            FP8
        )
        q = torch.randn(rows, heads * HEAD_DIM, generator=g, device=DEV) * Q_SCALE
        kv = torch.randn(rows, 2 * kv_heads * HEAD_DIM, generator=g, device=DEV)
        self.qkv = torch.cat((q, kv), dim=1).to(torch.bfloat16)
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.scale = torch.tensor([1.0], dtype=torch.float32, device=DEV)
        self.inv_scale = 1.0 / self.scale

    def args(self, pool, splits=None, pdl=False):
        return (
            self.qkv,
            pool,
            self.table,
            self.seq_lens,
            self.scale,
            self.splits if splits is None else splits,
            pdl,
        )

    def pair_slice(self, pair: int):
        """The tep8-shaped step of one pair: its 16 query heads, its 2 K and 2 V heads, and a
        contiguous pool of its 2 heads (same blocks, same table)."""
        q = self.qkv[:, pair * PAIR_H * HEAD_DIM : (pair + 1) * PAIR_H * HEAD_DIM]
        k0 = self.heads * HEAD_DIM
        v0 = k0 + self.kv_heads * HEAD_DIM
        k = self.qkv[:, k0 + pair * PAIR_G * HEAD_DIM : k0 + (pair + 1) * PAIR_G * HEAD_DIM]
        v = self.qkv[:, v0 + pair * PAIR_G * HEAD_DIM : v0 + (pair + 1) * PAIR_G * HEAD_DIM]
        qkv = torch.cat((q, k, v), dim=1).contiguous()
        return qkv


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


_BUILTIN_BUFFERS = {}


def _builtin_buffers():
    if not _BUILTIN_BUFFERS:
        _BUILTIN_BUFFERS["workspace"] = torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)
        _BUILTIN_BUFFERS["counter"] = torch.zeros(4096, dtype=torch.int32, device=DEV)
    return _BUILTIN_BUFFERS["workspace"], _BUILTIN_BUFFERS["counter"]


def _builtin_step(step: _Step, pool: torch.Tensor, fmha: bool = True, window: int = 0):
    """The built-in decode step on ``pool`` at the step's head counts: thop's generation
    preprocess (the K/V append and Q to E4M3), then flashinfer's trtllm-gen decode."""
    import flashinfer.decode as fdec

    rows, ctx, heads, kv_heads = step.rows, step.ctx, step.heads, step.kv_heads
    win = window if window else ctx
    workspace, counter = _builtin_buffers()
    (q_processed, kv_pool, block_tables, _, bmm1_scale, bmm2_scale, fmha_workspace, _, _,
     max_kv_len, window_left, _) = thop.trtllm_gen_generation_preprocess(
        step.qkv, workspace, step.seq_lens, None, None, step.table.view(1, 1, 2, -1),
        torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64),
        torch.tensor([[0, 0]], dtype=torch.int32), step.scale, step.inv_scale, None, None, None,
        None, 0, 0, heads, kv_heads, HEAD_DIM, PAGE, int(QuantMode.FP8_KV_CACHE), win, win, rows,
        1, rows, ctx, 0, 10000.0, 0, 1.0, 1024, 0, 1.0 / math.sqrt(HEAD_DIM), 1.0, True, rows, 0,
        _sm_count(), pool.shape[0], 2, True, False)  # fmt: skip
    if not fmha:
        return None
    width = heads * HEAD_DIM
    q8 = q_processed.view(torch.uint8).flatten()[: rows * width].view(FP8)
    out = torch.empty(rows, width, dtype=torch.bfloat16, device=DEV)
    fdec.trtllm_batch_decode_with_kv_cache(
        query=q8.view(rows, heads, HEAD_DIM),
        kv_cache=(kv_pool, kv_pool),
        workspace_buffer=fmha_workspace,
        block_tables=block_tables,
        seq_lens=step.seq_lens,
        max_seq_len=max_kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=window_left,
        out=out.view(rows, heads, HEAD_DIM),
        kv_layout="HND",
        enable_pdl=False,
        backend="trtllm-gen",
        q_len_per_req=rows,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=bmm1_scale.narrow(0, 1, 1),
        multi_ctas_kv_counter_buffer=counter,
    )
    return out


def _appended(step: _Step) -> torch.Tensor:
    pool = step.pool.clone()
    _builtin_step(step, pool, fmha=False)
    return pool


def _fp64_reference(step: _Step, pool: torch.Tensor) -> torch.Tensor:
    """fp64 attention of BF16 Q (never quantized) over the FP8 K/V the cache holds."""
    rows, ctx = step.rows, step.ctx
    keys = torch.arange(ctx, device=DEV)
    raw = _bytes(pool)

    def gather(kv):
        return raw[step.table[kv, keys // PAGE].long(), :, keys % PAGE].view(FP8).double()

    k, v = gather(0), gather(1)
    q = step.qkv[:, : step.heads * HEAD_DIM].double().view(rows, step.heads, HEAD_DIM)
    out = torch.empty(rows, step.heads, HEAD_DIM, dtype=torch.float64, device=DEV)
    group = step.heads // step.kv_heads
    for t in range(rows):
        end = ctx - rows + t + 1
        for h in range(step.kv_heads):
            s = q[t, h * group : (h + 1) * group] @ k[:end, h].T / math.sqrt(HEAD_DIM)
            out[t, h * group : (h + 1) * group] = torch.softmax(s, -1) @ v[:end, h]
    return out.reshape(rows, -1)


def _errors(out, ref):
    d = (out.double() - ref).abs()
    return float(d.max()), float(d.mean())


def _written(po: torch.Tensor, pl: torch.Tensor, rows: int, heads: int):
    """The partials the op writes: every row's log2-sum-exp, and O of the splits with keys."""
    o, lse = po[: rows * heads].clone(), pl[: rows * heads].clone()
    o[~torch.isfinite(lse)] = 0.0
    return o, lse


def _pair_rows(rows: int, pair: int) -> torch.Tensor:
    """The tep4 partial rows (token * 32 + query head) of one pair's 16 query heads."""
    tokens = torch.arange(rows, device=DEV)[:, None]
    heads = pair * PAIR_H + torch.arange(PAIR_H, device=DEV)[None, :]
    return (tokens * H + heads).flatten()


# --------------------------------------------------------------------------
# Each pair is the tep8 build on its slice
# --------------------------------------------------------------------------
_PAIR_CASES = [
    (8, 100000),
    (8, 455031),
    (8, 100003),
    (3, 100003),
    (1, 100000),
    (8, 19457),
    (8, 8192),
    (8, 1027),
    (8, 161),
    (8, 40),
    (8, 8),
    (1, 1),
]


def _assert_pairs_are_tep8(step: _Step, pool: torch.Tensor, pdl: bool = False) -> None:
    po, pl = op(*step.args(pool, pdl=pdl))
    got_o, got_lse = _written(po, pl, step.rows, H)
    # The tep8 build stages at most 256 page ids per split: it reads the table cut to the
    # context's pages (the pages read are the same; only the staging bound depends on the width).
    table = step.table[:, : -(-step.ctx // PAGE)].contiguous()
    for pair in range(G // PAIR_G):
        sliced = step.pair_slice(pair)
        pair_pool = pool[:, pair * PAIR_G : (pair + 1) * PAIR_G].contiguous()
        ref_o, ref_lse = op(sliced, pair_pool, table, step.seq_lens, step.scale, step.splits, pdl)
        ref_o, ref_lse = _written(ref_o, ref_lse, step.rows, PAIR_H)
        rows = _pair_rows(step.rows, pair)
        assert torch.equal(got_lse[rows], ref_lse), (step.rows, step.ctx, pair, "lse")
        assert torch.equal(got_o[rows], ref_o), (step.rows, step.ctx, pair, "O")


@pytest.mark.parametrize("rows, ctx", _PAIR_CASES)
def test_each_pair_is_the_tep8_build_on_its_slice(rows, ctx) -> None:
    step = _Step(rows, ctx, seed=ctx + rows)
    _assert_pairs_are_tep8(step, _appended(step))


def test_each_pair_is_the_tep8_build_at_the_served_table_width() -> None:
    """A 1M-token ``max_seq_len`` gives a 32768-page table: at ``SMs / 2`` splits per pair the
    tep4 build stages up to 512 page ids per CTA (two passes). Contexts whose pages fit the tep8
    build's 256 per split are compared pair by pair; at 1M keys (over 256 pages per CTA, so the
    second pass is read) the result is the one on a table cut to the context, and
    ``test_no_farther_from_fp64_than_the_builtin`` gates its accuracy."""
    top = 1048581
    step = _Step(8, top, seed=11, width=32768 + 2)
    assert -(-step.table.shape[1] // step.splits) > fa4.MAX_CTA_PAGES
    pool = _appended(step)
    wide = _written(*op(*step.args(pool)), 8, H)
    cut = step.table[:, : -(-top // PAGE)].contiguous()
    assert -(-cut.shape[1] // step.splits) > fa4.MAX_CTA_PAGES
    narrow = _written(*op(step.qkv, pool, cut, step.seq_lens, step.scale, step.splits), 8, H)
    assert torch.equal(wide[0], narrow[0]) and torch.equal(wide[1], narrow[1])
    for ctx in (455031, 100003):
        step.seq_lens.fill_(ctx)
        step.ctx = ctx
        _assert_pairs_are_tep8(step, pool)


def test_a_pdl_launch_writes_the_same_bits_and_calls_are_deterministic() -> None:
    step = _Step(8, 455031, seed=5)
    pool = _appended(step)
    first = _written(*op(*step.args(pool)), 8, H)
    pdl = _written(*op(*step.args(pool, pdl=True)), 8, H)
    assert torch.equal(first[0], pdl[0]) and torch.equal(first[1], pdl[1])
    for _ in range(5):
        again = _written(*op(*step.args(pool)), 8, H)
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    _assert_pairs_are_tep8(step, pool, pdl=True)


# --------------------------------------------------------------------------
# Against the built-in path at the tep4 shape, both measured from fp64
# --------------------------------------------------------------------------
_BUILTIN_CASES = [
    (8, 100000, 0),
    (8, 455031, 0),
    (8, 1048581, 0),
    (3, 100003, 0),
    (1, 100000, 0),
    (8, 2056, 0),
    (8, 300, 0),
    (4, 40, 0),
    (8, 100000, 1),
    (8, 100000, 2),
]


@pytest.mark.parametrize("rows, ctx, seed", _BUILTIN_CASES)
def test_no_farther_from_fp64_than_the_builtin(rows, ctx, seed) -> None:
    """Tolerance: the op's max-abs and mean-abs errors against fp64 are each at most the
    built-in trtllm-gen path's on the same inputs (the tep8 op test's policy bound)."""
    step = _Step(rows, ctx, seed=7919 * seed + ctx + rows)
    pool = step.pool.clone()
    builtin = _builtin_step(step, pool)
    ref = _fp64_reference(step, pool)
    po, pl = op(*step.args(pool))
    out = cng.combine_only(po, pl, rows, heads=H)
    o_max, o_mean = _errors(out, ref)
    b_max, b_mean = _errors(builtin, ref)
    print(
        f"\ntep4 rows {rows} ctx {ctx} seed {seed}: op max {o_max:.4g} mean {o_mean:.3g}; "
        f"builtin max {b_max:.4g} mean {b_mean:.3g}; ratios {o_max / max(b_max, 1e-30):.3f} / "
        f"{o_mean / max(b_mean, 1e-30):.3f}"
    )
    assert o_max <= b_max and o_mean <= b_mean, (o_max, b_max, o_mean, b_mean)


# --------------------------------------------------------------------------
# combine_norm_gate at 32 heads
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gate_ndim", (2, 3))
@pytest.mark.parametrize("rows, ctx", [(8, 100000), (8, 8192), (1, 40)])
def test_combine_norm_gate_is_the_norm_gate_of_the_merge(rows, ctx, gate_ndim) -> None:
    step = _Step(rows, ctx, seed=ctx)
    pool = _appended(step)
    po, pl = op(*step.args(pool))
    g = torch.Generator(device=DEV).manual_seed(ctx)
    gate = torch.randn(rows, H * HEAD_DIM, generator=g, device=DEV).to(torch.bfloat16)
    if gate_ndim == 3:
        gate = gate.view(rows, H, HEAD_DIM)
    merged = cng.combine_only(po, pl, rows, heads=H)
    want = ng.attention_norm_gate(merged, gate, EPS)
    for seq_lens in (None, step.seq_lens):
        got = cng.combine_norm_gate(po, pl, rows, gate, EPS, seq_lens=seq_lens, heads=H)
        assert torch.equal(got, want), (rows, ctx, gate_ndim, seq_lens is None)
    ref = cng.combine_norm_gate.reference(po, pl, rows, gate, EPS, seq_lens=step.seq_lens, heads=H)
    diff = (got.float() - ref.float()).abs()
    ulp = torch.maximum(got.float().abs(), ref.float().abs()) * 2.0**-7
    assert bool((diff <= ulp + 1e-6).all()), float((diff - ulp).max())


def test_combine_norm_gate_refuses_other_head_counts() -> None:
    po, pl = fa4.partial_buffers(4, DEV, G)
    gate = torch.zeros(8, H * HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError):
        cng.combine_norm_gate(po, pl, 8, gate, EPS, heads=24)
    with pytest.raises(ValueError):
        cng.combine_norm_gate(po, pl, 8, gate[:, : 16 * HEAD_DIM], EPS, heads=H)


# --------------------------------------------------------------------------
# full_attention_prep_fold at tep4
# --------------------------------------------------------------------------
def _raw_projection(rows: int, seed: int, permuted: bool) -> torch.Tensor:
    """A raw QKV projection row of the tep4 rank: 32 query heads and their gates, 4 K and
    4 V heads (checkpoint order: q and gate interleaved per head, then K, V; permuted:
    q, K, V, gate), each head at its own random magnitude. Both layouts are random data here:
    the comparison reads the same layout on both sides."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    width = fold_mod.raw_width(G)
    assert width == 4608
    heads = width // HEAD_DIM
    x = torch.randn(rows, heads, HEAD_DIM, generator=g, device=DEV)
    x = x * torch.exp(torch.randn(rows, heads, 1, generator=g, device=DEV))
    return x.reshape(rows, width).to(torch.bfloat16)


def _poison_step_slots(step: _Step, pool: torch.Tensor) -> None:
    """E4M3 NaN in the step's own K and V slots (every head), which the step writes."""
    positions = torch.arange(step.ctx - step.rows, step.ctx, device=DEV)
    for kv in range(2):
        _bytes(pool)[step.table[kv, positions // PAGE].long(), :, positions % PAGE] = 0x7F


_FOLD_CASES = [
    (8, 100000, False),
    (8, 455031, False),
    (8, 8200, False),
    (8, 8217, False),
    (8, 8231, False),
    (3, 100003, False),
    (1, 1, False),
    (8, 40, False),
    (8, 100000, True),
    (8, 8217, True),
]


@pytest.mark.parametrize("rows, ctx, permuted", _FOLD_CASES)
def test_the_fold_is_the_prep_append_then_split_kv(rows, ctx, permuted) -> None:
    """Bit for bit: the fold's partials and cache bytes are those of ``preattn_prep_append``
    at 32 / 4 heads followed by the tep4 ``full_attention_split_kv``."""
    step = _Step(rows, ctx, seed=zlib.crc32(f"{rows} {ctx} {permuted}".encode()))
    qkv = _raw_projection(rows, ctx + rows, permuted)
    splits = step.splits
    _poison_step_slots(step, step.pool)
    # The two-launch chain.
    chain_pool = step.pool.clone()
    q8 = torch.empty(rows * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    scales = torch.empty(3, dtype=torch.float32, device=DEV)
    fused, _ = preattn_prep_append(
        qkv, H, G, CKPT_EPS, CKPT_Q_SCALE, permuted, chain_pool, step.table, step.seq_lens,
        step.scale, step.scale, q8, scales[:2], scales[2:], BMM1_HOST,
    )  # fmt: skip
    chain = _written(*op(fused, chain_pool, step.table, step.seq_lens, step.scale, splits), rows, H)
    # The fold.
    fold_pool = step.pool.clone()
    folded = _written(
        *fold_op(qkv, CKPT_EPS, CKPT_Q_SCALE, permuted, fold_pool, step.table, step.seq_lens,
                 step.scale, splits),
        rows,
        H,
    )  # fmt: skip
    assert torch.equal(_bytes(fold_pool), _bytes(chain_pool)), "cache bytes"
    assert torch.equal(folded[1], chain[1]), "log2-sum-exp"
    assert torch.equal(folded[0], chain[0]), "O"
    # The gate view is the projection's gate.
    gate = fold_mod.gate_view(qkv, permuted)
    if permuted:
        want = qkv[:, (H + 2 * G) * HEAD_DIM :]
    else:
        want = qkv[:, : 2 * H * HEAD_DIM].view(rows, H, 2, HEAD_DIM)[:, :, 1, :]
    assert torch.equal(gate.reshape(rows, -1), want.reshape(rows, -1))


def test_the_fold_with_pdl_writes_the_same_bits() -> None:
    step = _Step(8, 455031, seed=3)
    qkv = _raw_projection(8, 3, False)
    outs = []
    for pdl in (False, True):
        pool = step.pool.clone()
        po, pl = fold_op(
            qkv, CKPT_EPS, CKPT_Q_SCALE, False, pool, step.table, step.seq_lens, step.scale,
            step.splits, pdl,
        )  # fmt: skip
        outs.append((_written(po, pl, 8, H), _bytes(pool).clone()))
    assert torch.equal(outs[0][1], outs[1][1])
    assert torch.equal(outs[0][0][0], outs[1][0][0]) and torch.equal(outs[0][0][1], outs[1][0][1])


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
def test_the_ops_refuse_other_rank_shapes() -> None:
    step = _Step(8, 1000, seed=6)
    pool = _appended(step)
    three = torch.zeros(pool.shape[0], 3, PAGE, HEAD_DIM, dtype=FP8, device=DEV)
    with pytest.raises(ValueError):
        op(*step.args(three))
    tep8 = _Step(8, 1000, seed=6, heads=PAIR_H, kv_heads=PAIR_G)
    with pytest.raises(ValueError):  # a tep8 operand on a tep4 pool
        op(tep8.qkv, pool, step.table, step.seq_lens, step.scale, step.splits)
    wide = _Step(8, 1000, seed=6, width=2 * fa4.cta_pages(G) + 1)
    with pytest.raises(ValueError):  # more than 512 pages per split
        op(*wide.args(wide.pool, splits=2))
    with pytest.raises(ValueError):
        fold_op(step.qkv, CKPT_EPS, 1.0, False, pool, step.table, step.seq_lens, step.scale, 4)
    assert fa4.cta_pages(2) == 256 and fa4.cta_pages(4) == 512
    with pytest.raises(ValueError):
        fa4.cta_pages(8)
    assert fa4.default_splits(2, 148) == 148 and fa4.default_splits(4, 148) == 74


# --------------------------------------------------------------------------
# Microbenchmark (MODELING_V2_TIMING=1): one full-attention layer's decode attention at tep4
# --------------------------------------------------------------------------
def _graph_us(fn, iters: int = 20, reps: int = 5) -> float:
    """Median microseconds per call of ``fn`` replayed ``iters`` times inside one CUDA graph."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    return sorted(times)[len(times) // 2]


@pytest.mark.skipif(not os.environ.get("MODELING_V2_TIMING"), reason="MODELING_V2_TIMING unset")
@pytest.mark.parametrize("ctx", (8192, 100000, 455031))
def test_microbenchmark_full_attention_tep4(ctx) -> None:
    """Per layer, B=1, 8 rows: the built-in chain (generation preprocess + trtllm-gen FMHA +
    norm-gate) vs the fast paths (``full_attention_prep_append``'s prep + split-KV +
    combine_norm_gate; the fold + combine_norm_gate). Printed, never gated."""
    rows = 8
    step = _Step(rows, ctx, seed=1)
    qkv_raw = _raw_projection(rows, 1, False)
    gate = qkv_raw[:, : 2 * H * HEAD_DIM].view(rows, H, 2, HEAD_DIM)[:, :, 1, :]
    pool = step.pool.clone()
    q8 = torch.empty(rows * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    scales = torch.empty(3, dtype=torch.float32, device=DEV)
    part_o, part_lse = fa4.partial_buffers(step.splits, DEV, G)

    def builtin():
        out = _builtin_step(step, pool)
        ng.attention_norm_gate(out, gate, EPS)

    def fmha_only():
        _builtin_step(step, pool)

    def split_kv():
        fused, _ = preattn_prep_append(
            qkv_raw, H, G, CKPT_EPS, CKPT_Q_SCALE, False, pool, step.table, step.seq_lens,
            step.scale, step.scale, q8, scales[:2], scales[2:], BMM1_HOST, enable_pdl=True,
        )  # fmt: skip
        op(fused, pool, step.table, step.seq_lens, step.scale, step.splits, True,
           part_o=part_o, part_lse=part_lse, rescale_cols=fa4.SPILL_FREE_RESCALE_COLS)  # fmt: skip
        cng.combine_norm_gate(part_o, part_lse, rows, gate, EPS, seq_lens=step.seq_lens, heads=H)

    def prep_fold():
        fold_op(qkv_raw, CKPT_EPS, CKPT_Q_SCALE, False, pool, step.table, step.seq_lens, step.scale,
                step.splits, True, part_o=part_o, part_lse=part_lse,
                rescale_cols=fa4.SPILL_FREE_RESCALE_COLS)  # fmt: skip
        cng.combine_norm_gate(part_o, part_lse, rows, gate, EPS, seq_lens=step.seq_lens, heads=H)

    results = {name: _graph_us(fn) for name, fn in (
        ("builtin preprocess+fmha+norm_gate", builtin),
        ("builtin preprocess+fmha", fmha_only),
        ("prep_append+split_kv+combine_norm_gate", split_kv),
        ("prep_fold+combine_norm_gate", prep_fold),
    )}  # fmt: skip
    line = ", ".join(f"{name} {us:.1f} us" for name, us in results.items())
    print(f"\nTIMING tep4 full-attention layer ctx {ctx} rows {rows} splits {step.splits}: {line}")
