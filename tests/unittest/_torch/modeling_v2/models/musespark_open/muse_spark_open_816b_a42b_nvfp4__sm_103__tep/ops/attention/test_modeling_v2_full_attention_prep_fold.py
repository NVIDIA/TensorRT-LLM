# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open full_attention_prep_fold entry (not routed).

What is gated, against ``preattn_prep_append`` followed by ``full_attention_split_kv`` on the same raw
projection and the same cache (the folded-prep + split-KV path the entry replaces):

* the partials bit for bit (every row's log2-sum-exp, the O of every split that holds keys)
  and every byte of the cache after the step, with the step's own slots poisoned (E4M3 NaN)
  beforehand -- on every certified cell, at every offset of the step's first row in its page
  (``past % 32`` in 0, 24, 25, 31; 1, 3 and 8 rows; 8k, 100k and 455k keys; checkpoint and
  permuted projection layouts), at every context up to 300 keys, and, in test builds of one
  page per split, with a split boundary between the two pages a straddling step writes, so
  that two CTAs append (the op's split rule always gives both pages to the last CTA, which is
  asserted over every context to 1.1M keys);
* the output gate: the entry's gate view of the projection is the prep's gate copy bit for bit,
  and ``combine_norm_gate`` gives the same bits on either;
* the merged partials within ``full_attention_split_kv``'s E4M3-P bound of its reference on the prep's
  operand, and the entry's own reference (the prep's reference arithmetic and append, then
  ``full_attention_split_kv``'s reference) equal to that reference wherever its operand and cache bytes
  are the prep's;
* a captured graph replays the eager bits at other contexts (PDL on and off; the op's build
  and the spill-free build);
* PDL memory order: a predecessor that releases the entry at its entry and writes the
  projection 200 us later is observed (the entry's Q, K and V reads follow its wait), where the
  same writer with no dependency is not; and the build whose loader does not wait for the
  append (``app_wait=False``) is caught streaming the step's poisoned page;
* repeated calls are bitwise equal; malformed inputs are refused;
* the spill-free build (the O rescale in 16-column chunks) gives the fold build's partials
  and cache bytes bit for bit: at the op's rescale threshold on random data and on "spike" data
  (one key per 128-key tile raised to an increasing constant, so most tiles rescale), and in
  test builds that rescale on every rise (``rescale_log2=0``), at 1, 3 and 8 rows and at
  contexts where a warpgroup has many tiles (100k and 455k keys; 1,027, 8,192 and 19,457 keys
  over 1, 8 and 19 splits).
"""

import contextlib
import math
import zlib

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    combine_norm_gate as cng,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    full_attention_prep_fold as faf,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    full_attention_split_kv as fa4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    preattn_prep_append as ppa,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    CKPT_EPS,
    CKPT_Q_SCALE,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="full_attention_prep_fold is built for sm_103 and sm_107",
)

op = faf.full_attention_prep_fold
DEV = "cuda"
FP8 = torch.float8_e4m3fn
HEAD_DIM, PAGE = fa4.HEAD_DIM, fa4.PAGE
H, G = 16, 2
Q_WIDTH, KV_WIDTH = H * HEAD_DIM, G * HEAD_DIM
RAW = faf.RAW_WIDTH
#: The softmax scale ``run_generation`` passes.
BMM1_HOST = 1.0 / math.sqrt(HEAD_DIM)
#: The norm-gate's epsilon (``o_head_norm``).
GATE_EPS = 1e-5
#: Keys per tile of the kernel's key loop.
_TILE = 4 * PAGE
#: E4M3-exact increasing key values of the spike data.
_SPIKES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0,
           32.0, 40.0, 48.0, 64.0, 80.0, 96.0, 128.0, 160.0, 192.0, 256.0, 320.0, 384.0, 448.0)  # fmt: skip


def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _permutation() -> torch.Tensor:
    """Checkpoint column order -> the low-latency layout ``[q | k | v | gate]``."""
    interleaved = torch.arange(2 * Q_WIDTH, device=DEV).reshape(H, 2 * HEAD_DIM)
    return torch.cat(
        (
            interleaved[:, :HEAD_DIM].reshape(-1),
            torch.arange(2 * Q_WIDTH, RAW, device=DEV),
            interleaved[:, HEAD_DIM:].reshape(-1),
        )
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


class _Step:
    """One decode step of a full-attention layer: the raw projection and the paged FP8 cache.

    Logical page ``i`` is K block ``2 perm[i]`` and V block ``2 perm[i] + 1`` of the pool,
    ``perm`` a random permutation; the pool holds random FP8 everywhere (a stray write shows
    up in the byte comparison) and E4M3 NaN in the step's own slots, which the step writes.
    """

    def __init__(
        self,
        rows,
        ctx,
        *,
        seed=0,
        splits=None,
        scale=1.0,
        permuted=False,
        q_scale=CKPT_Q_SCALE,
        width=None,
        spikes=False,
    ):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx = rows, ctx
        self.splits = _sm_count() if splits is None else splits
        self.permuted, self.q_scale = permuted, q_scale
        pages = -(-ctx // PAGE)
        width = pages if width is None else width
        spare = width + 4
        perm = torch.randperm(spare, generator=g, device=DEV)[:width].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = torch.randn(2 * spare, G, PAGE, HEAD_DIM, generator=g, device=DEV).to(FP8)
        heads = RAW // HEAD_DIM
        x = torch.randn(rows, heads, HEAD_DIM, generator=g, device=DEV)
        x = x * torch.exp(torch.randn(rows, heads, 1, generator=g, device=DEV))
        if spikes:
            # The query heads (head h at 2h in the checkpoint layout) all positive.
            x[:, 0 : 2 * H : 2] = x[:, 0 : 2 * H : 2].abs()
        projection = x.reshape(rows, RAW).to(torch.bfloat16)
        self.qkv = projection[:, _permutation()].contiguous() if permuted else projection
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.scale = torch.tensor([scale], dtype=torch.float32, device=DEV)
        self.inv_scale = 1.0 / self.scale
        if spikes:
            self.spike(self.pool, shift=seed)
        self.poison(self.pool)

    def spike(self, pool: torch.Tensor, shift: int = 0) -> None:
        """One key per 128-key tile before the step's rows set to an increasing E4M3 constant
        (every K dim, both KV heads): with all-positive queries each such tile raises every
        row's running maximum by far more than 2^4, so the served threshold rescales O there."""
        visible = self.ctx - self.rows
        for tile in range(-(-visible // _TILE)):
            position = tile * _TILE + (tile * 37 + shift) % _TILE
            if position < visible:
                block = int(self.table[0, position // PAGE])
                pool[block, :, position % PAGE, :] = _SPIKES[tile % len(_SPIKES)]

    def slots(self):
        """(block, slot) of the step's K and V rows: positions ctx - rows .. ctx - 1."""
        positions = torch.arange(self.ctx - self.rows, self.ctx, device=DEV)
        k = self.table[0, positions // PAGE].long()
        v = self.table[1, positions // PAGE].long()
        return k, v, positions % PAGE

    def poison(self, pool: torch.Tensor) -> None:
        k, v, slot = self.slots()
        _bytes(pool)[k, :, slot] = 0x7F
        _bytes(pool)[v, :, slot] = 0x7F

    def fold_args(self, pool, splits=None, pdl=False, qkv=None):
        return (
            self.qkv if qkv is None else qkv,
            CKPT_EPS,
            self.q_scale,
            self.permuted,
            pool,
            self.table,
            self.seq_lens,
            self.scale,
            self.splits if splits is None else splits,
            pdl,
        )


def _prep_then_v4(step: _Step, pool: torch.Tensor, pdl: bool = False, qkv=None):
    """The path the entry replaces: ``preattn_prep_append`` (writes the cache), then
    ``full_attention_split_kv`` on its fused operand. Returns (part_o, part_lse, fused operand, gate)."""
    rows = step.rows
    q8 = torch.empty(rows * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    bmm1 = torch.empty(2, dtype=torch.float32, device=DEV)
    bmm2 = torch.empty(1, dtype=torch.float32, device=DEV)
    fused, gate = ppa.preattn_prep_append(
        step.qkv if qkv is None else qkv, H, G, CKPT_EPS, step.q_scale, step.permuted, pool,
        step.table, step.seq_lens, step.scale, step.inv_scale, q8, bmm1, bmm2, BMM1_HOST, 1.0,
        pdl,
    )  # fmt: skip
    po, pl = fa4.full_attention_split_kv(
        fused, pool, step.table, step.seq_lens, step.scale, step.splits, pdl, False
    )
    return po, pl, fused, gate


def _written(po: torch.Tensor, pl: torch.Tensor, rows: int):
    """Every row's log2-sum-exp, and O of the splits with keys (a key-less split writes
    ``-inf`` and leaves its O as it was)."""
    o, lse = po[: rows * H].clone(), pl[: rows * H].clone()
    o[~torch.isfinite(lse)] = 0.0
    return o, lse


def _assert_same(step: _Step, pool_a, pool_b, a, b, what: str) -> None:
    oa, la = _written(a[0], a[1], step.rows)
    ob, lb = _written(b[0], b[1], step.rows)
    assert torch.equal(la, lb), f"{what}: log2-sum-exp differs"
    assert torch.equal(_bytes(oa), _bytes(ob)), f"{what}: partial O differs"
    differ = int((_bytes(pool_a) != _bytes(pool_b)).sum())
    assert differ == 0, f"{what}: {differ} cache bytes differ"
    k, v, slot = step.slots()
    for blocks in (k, v):
        assert not (_bytes(pool_b)[blocks, :, slot] == 0x7F).all(-1).any(), (
            f"{what}: a step slot kept its poison"
        )


def _compare(step: _Step, pdl: bool = False, splits=None):
    """Run both paths on copies of the step's cache and gate their bits; returns the prep
    path's (fused operand, gate) and the entry's (partials, cache)."""
    pool_a, pool_b = step.pool.clone(), step.pool.clone()
    if splits is not None:
        step.splits = splits
    po_a, pl_a, fused, gate = _prep_then_v4(step, pool_a, pdl)
    po_b, pl_b = op(*step.fold_args(pool_b, pdl=pdl))
    torch.cuda.synchronize()
    _assert_same(step, pool_a, pool_b, (po_a, pl_a), (po_b, pl_b), f"{step.rows}x{step.ctx}")
    return fused, gate, (po_b, pl_b), pool_b


# --------------------------------------------------------------------------
# full_attention_split_kv's bound, on the prep's operand
# --------------------------------------------------------------------------
def _gather_kv(pool, table, ctx, scale):
    keys = torch.arange(ctx, device=DEV)
    raw = _bytes(pool)

    def gather(kv):
        return raw[table[kv, keys // PAGE].long(), :, keys % PAGE].view(FP8).double() / scale

    return gather(0), gather(1)


def _op_q(fused: torch.Tensor) -> torch.Tensor:
    q = fused[:, :Q_WIDTH].float()
    c = q * 1048577.0
    hi = c - (c - q)

    def e4m3(x):
        return x.clamp(-448.0, 448.0).to(FP8).float()

    return e4m3(hi) + e4m3(q - hi)


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    _, exponent = torch.frexp(x.double().abs())
    return torch.ldexp(torch.ones_like(x, dtype=torch.float64), exponent - 8)


def _e4m3_p_bound(fused, pool, table, ctx, scale=1.0):
    """``full_attention_split_kv``'s E4M3-P bound (its op test's derivation) on the operand ``fused``."""
    rows = fused.shape[0]
    k, v = _gather_kv(pool, table, ctx, scale)
    q = _op_q(fused).double().view(rows, H, HEAD_DIM)
    bound = torch.empty(rows, H, HEAD_DIM, dtype=torch.float64, device=DEV)
    group = H // G
    for t in range(rows):
        end = ctx - rows + t + 1
        for h in range(G):
            s = q[t, h * group : (h + 1) * group] @ k[:end, h].T / math.sqrt(HEAD_DIM)
            p = torch.exp(s - s.amax(-1, keepdim=True))
            dp = torch.where(p >= 2.0**-10, 2.0**-4 * p, p.clamp(max=2.0**-14)) + 2.0**-10 * p
            vv = v[:end, h].abs()
            den = p.sum(-1, keepdim=True)
            bound[t, h * group : (h + 1) * group] = (dp @ vv + 2.0**-10 * (p @ vv)) / den
    return bound.reshape(rows, Q_WIDTH)


# --------------------------------------------------------------------------
# Certified cells
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    """Bit for bit the prep then ``full_attention_split_kv`` (partials, cache, gate, norm-gate); within
    ``full_attention_split_kv``'s bound of its reference on the prep's operand; the entry's reference
    equals that reference wherever its operand and cache are the prep's."""
    certifying(op)
    spec = cell.spec
    step = _Step(
        spec["rows"],
        spec["ctx"],
        seed=zlib.crc32(cell.why.encode()),
        splits=spec.get("splits"),
        scale=spec.get("scale", 1.0),
        permuted=spec.get("permuted", False),
        q_scale=spec.get("q_scale", CKPT_Q_SCALE),
    )
    rows, ctx = step.rows, step.ctx
    before_q = step.qkv.clone()
    fused, gate, (po, pl), pool = _compare(step, pdl=spec.get("pdl", False))
    assert torch.equal(step.qkv, before_q)

    # The output gate: a view of the projection, the prep's copy bit for bit.
    view = faf.gate_view(step.qkv, step.permuted)
    assert torch.equal(view.reshape(rows, -1), gate.reshape(rows, -1))
    a = cng.combine_norm_gate(po, pl, rows, view, GATE_EPS, "exit", seq_lens=step.seq_lens)
    b = cng.combine_norm_gate(po, pl, rows, gate, GATE_EPS, "exit", seq_lens=step.seq_lens)
    assert torch.equal(a, b)

    # full_attention_split_kv's bound on the prep's operand.
    merged = fa4.merge_partials(po, pl, rows)
    assert not torch.isnan(merged).any()
    v4_ref = fa4.full_attention_split_kv.reference(
        fused, pool, step.table, step.seq_lens, step.scale, 1
    )
    bound = _e4m3_p_bound(fused, pool, step.table, ctx, float(step.scale[0]))
    limit = bound + _bf16_ulp(torch.maximum(merged.double().abs(), v4_ref.double().abs()))
    used = float(((merged.double() - v4_ref.double()).abs() / limit).max())
    assert used <= 1.0, (cell.why, used)

    # The entry's reference: the prep's reference arithmetic and append, then v4's reference.
    ref_pool = step.pool.clone()
    ref = op.reference(*step.fold_args(ref_pool))
    q8 = torch.empty(rows * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    scales = torch.empty(3, dtype=torch.float32, device=DEV)
    ref_fused, _ = ppa.preattn_prep_append.reference(
        step.qkv, H, G, CKPT_EPS, step.q_scale, step.permuted, step.pool.clone(), step.table,
        step.seq_lens, step.scale, step.scale, q8, scales[:2], scales[2:], 1.0,
    )  # fmt: skip
    k, v, slot = step.slots()
    same_kv = torch.stack(
        [
            (_bytes(ref_pool)[k, g, slot] == _bytes(pool)[k, g, slot]).all()
            & (_bytes(ref_pool)[v, g, slot] == _bytes(pool)[v, g, slot]).all()
            for g in range(G)
        ]
    )  # per KV head
    same_q = (ref_fused[:, :Q_WIDTH] == fused[:, :Q_WIDTH]).view(rows, H, HEAD_DIM).all(-1)
    usable = same_q & same_kv.repeat_interleave(H // G)[None, :]
    assert float(usable.float().mean()) >= 0.75, (cell.why, usable)
    want = v4_ref.view(rows, H, HEAD_DIM)[usable]
    assert torch.equal(ref.view(rows, H, HEAD_DIM)[usable], want), cell.why


# --------------------------------------------------------------------------
# Where the step's rows fall
# --------------------------------------------------------------------------
_OFFSETS = (0, 24, 25, 31)
_BASES = (8192, 100000, 455000)


def _ctx(base: int, rows: int, offset: int) -> int:
    """A context near ``base`` whose step's first row sits at ``offset`` in its page."""
    past = (base // PAGE) * PAGE + offset
    return past + rows


@pytest.mark.parametrize("permuted", (False, True), ids=("ckpt", "permuted"))
@pytest.mark.parametrize("base", _BASES)
@pytest.mark.parametrize("rows", (1, 3, 8))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_every_page_offset_of_the_step(offset, rows, base, permuted) -> None:
    """``past % 32`` in 0, 24, 25, 31: the step opens a page, fills one to its end, or (25 and
    31 at 8 rows, 31 at 3 rows) straddles two pages."""
    ctx = _ctx(base, rows, offset)
    step = _Step(rows, ctx, seed=ctx * 4 + rows + 100 * permuted, permuted=permuted)
    _compare(step)


@pytest.mark.parametrize("splits", (None, 96, 512, 1))
def test_the_ops_split_rule_gives_both_pages_to_the_last_cta(splits) -> None:
    """With at least ``MIN_PAGES`` pages per split the last keyed split holds at least three
    pages, so the (at most two) pages the step writes are both its own, at every context up to
    1.1M keys and every row count."""
    splits = _sm_count() if splits is None else splits
    ctx = torch.arange(1, 1_100_001, dtype=torch.int64)
    pages = (ctx + PAGE - 1) // PAGE
    eff = torch.clamp(-(-pages // fa4.MIN_PAGES), max=splits)
    last_start = (eff - 1) * pages // eff
    for rows in range(1, 9):
        keep = ctx >= rows
        first = (ctx - rows) // PAGE
        assert bool((last_start[keep] <= first[keep]).all()), rows


def _one_page_per_split_builds(step: _Step):
    """``full_attention_split_kv`` and the fold kernel, both built with one page per split."""
    dev = torch.cuda.current_device()
    key = faf.fold_key(CKPT_EPS, step.q_scale, step.permuted)
    base = fa4._compiled_kernel(dev, False, False, True, min_pages=1)
    fold = fa4._compiled_kernel(dev, False, False, True, fold=key, min_pages=1)
    return base, fold


@pytest.mark.parametrize("permuted", (False, True), ids=("ckpt", "permuted"))
@pytest.mark.parametrize("rows, offset", [(8, 25), (8, 31), (3, 31), (2, 31)])
@pytest.mark.parametrize("base", (2048, 8192))
def test_a_split_boundary_between_the_steps_two_pages(base, rows, offset, permuted) -> None:
    """Two CTAs append: with one page per split (a test build of both kernels; the op's rule
    never splits there) each CTA writes the rows on its own page and releases its own loader.
    Partials and cache bit for bit the prep then ``full_attention_split_kv``'s same build."""
    ctx = _ctx(base, rows, offset)
    pages = -(-ctx // PAGE)
    assert pages <= fa4.MAX_SPLITS and (ctx - 1) // PAGE == (ctx - rows) // PAGE + 1
    step = _Step(rows, ctx, seed=ctx + 7 * rows, splits=pages, permuted=permuted)
    base_build, fold_build = _one_page_per_split_builds(step)
    pool_a, pool_b = step.pool.clone(), step.pool.clone()
    q8 = torch.empty(rows * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    bmm = torch.empty(3, dtype=torch.float32, device=DEV)
    fused, _ = ppa.preattn_prep_append(
        step.qkv, H, G, CKPT_EPS, step.q_scale, step.permuted, pool_a, step.table, step.seq_lens,
        step.scale, step.inv_scale, q8, bmm[:2], bmm[2:], BMM1_HOST, 1.0, False,
    )  # fmt: skip
    want = _build_partials(base_build, step, fused, pool_a)
    got = _build_partials(fold_build, step, step.qkv, pool_b)
    torch.cuda.synchronize()
    _assert_same(step, pool_a, pool_b, want, got, f"{rows}x{ctx} one page per split")


@pytest.mark.parametrize("rows", (1, 3, 8))
def test_every_short_context(rows) -> None:
    """Every context up to 300 keys: one to three splits, a last tile padded past the CTA's
    range (at 129 .. 135 keys and 8 rows the first CTA's padding reaches the step's first
    page), the first steps of a sequence."""
    for ctx in range(rows, 301):
        _compare(_Step(rows, ctx, seed=ctx * 8 + rows))


@pytest.mark.parametrize("rows", range(1, 9))
def test_every_row_count(rows) -> None:
    for ctx in (8203, 100003):
        _compare(_Step(rows, ctx, seed=ctx + rows))


@pytest.mark.parametrize("q_scale", (CKPT_Q_SCALE, 1.0, 0.7071))
def test_the_query_pre_scale(q_scale) -> None:
    _compare(_Step(8, 100003, seed=5, q_scale=q_scale))


# --------------------------------------------------------------------------
# Launch properties
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rescale_cols", (fa4.HEAD_DIM, fa4.SPILL_FREE_RESCALE_COLS), ids=("whole", "spill_free")
)
@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
def test_a_captured_graph_replays_the_eager_bits_at_other_contexts(pdl, rescale_cols) -> None:
    """The context and the step's slots are read on the device: one capture serves every
    step; each replay gives the prep-then-v4 bits on the same cache (the op's build and
    the spill-free build)."""
    top = 455031
    captured = _Step(8, 262145, seed=1, width=-(-top // PAGE))
    static_qkv, static_seq = captured.qkv.clone(), captured.seq_lens.clone()
    pool = captured.pool
    part_o, part_lse = fa4.partial_buffers(captured.splits, DEV)

    def fold():
        op(static_qkv, CKPT_EPS, CKPT_Q_SCALE, False, pool, captured.table, static_seq,
           captured.scale, captured.splits, pdl, True, part_o, part_lse, rescale_cols)  # fmt: skip

    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fold()  # compiles outside the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fold()
    g = torch.Generator(device=DEV).manual_seed(3)
    for ctx in (8, 40, 1027, 8192, 8217, 8231, 19457, 100003, 262145, top):
        static_seq.fill_(ctx)
        static_qkv.copy_(torch.randn(static_qkv.shape, generator=g, device=DEV).to(torch.bfloat16))
        step = _Step(8, ctx, seed=ctx, width=captured.table.shape[1])
        step.table, step.qkv, step.seq_lens = captured.table, static_qkv, static_seq
        step.poison(pool)
        before = pool.clone()
        graph.replay()
        torch.cuda.synchronize()
        want_pool = before.clone()
        po, pl, _, _ = _prep_then_v4(step, want_pool, pdl)
        torch.cuda.synchronize()
        _assert_same(step, want_pool, pool, (po, pl), (part_o, part_lse), f"graph at {ctx}")


@triton.jit
def _globaltimer(anchor):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [anchor], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _late_projection_kernel(qkv, src_qkv, qkv_bytes, delay_ns, EARLY_TRIGGER: tl.constexpr,
                            CHUNK: tl.constexpr):  # fmt: skip
    """Test-only PDL primary: release the dependent at entry (optionally), spin, then write the
    projection."""
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offs = tl.arange(0, CHUNK)
    for c in range(0, qkv_bytes, CHUNK):
        o = c + offs
        m = o < qkv_bytes
        tl.store(qkv + o, tl.load(src_qkv + o, mask=m), mask=m)


def _late_projection(qkv, want, delay_ns, early, stream=None):
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        _late_projection_kernel[(1,)](
            _bytes(qkv), _bytes(want), qkv.numel() * 2, delay_ns, EARLY_TRIGGER=early,
            CHUNK=1024, num_warps=1, launch_pdl=True,
        )  # fmt: skip


def _build_partials(build, step, qkv, pool):
    """One compiled build of ``full_attention_split_kv``'s or the fold kernel, launched with the op's
    arguments (``qkv``: the fused operand or the raw projection)."""
    import cuda.bindings.driver as cuda

    po, pl = fa4.partial_buffers(step.splits, DEV)
    build(
        qkv.data_ptr(), pool.data_ptr(), step.table.data_ptr(), step.seq_lens.data_ptr(),
        step.scale.data_ptr(), po.data_ptr(), pl.data_ptr(), step.rows, qkv.stride(0),
        pool.shape[0], step.table.shape[1], step.table.stride(0), step.splits,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )  # fmt: skip
    return po, pl


def _under_late_projection(step, launch, tries=1):
    """``launch(qkv, pool)`` as the PDL dependent of a primary that releases it at entry, spins
    200 us and then writes the projection (all ones until then). Returns, per try, whether the
    partials and the cache equal the prep-then-v4 path's on the written projection."""
    want_pool = step.pool.clone()
    want = _prep_then_v4(step, want_pool)
    equal = []
    for _ in range(tries):
        pool = step.pool.clone()
        qkv = torch.ones_like(step.qkv)
        torch.cuda.synchronize()
        _late_projection(qkv, step.qkv.clone(), 200_000, True)
        got = launch(qkv, pool)
        torch.cuda.synchronize()
        oa, la = _written(want[0], want[1], step.rows)
        ob, lb = _written(got[0], got[1], step.rows)
        equal.append(
            bool(torch.equal(la, lb))
            and bool(torch.equal(_bytes(oa), _bytes(ob)))
            and bool(torch.equal(_bytes(pool), _bytes(want_pool)))
        )
    return equal


@pytest.mark.parametrize("ctx, splits", [(100003, None), (8195, None), (40003, 96)])
def test_a_late_writer_of_the_projection_is_observed(ctx, splits) -> None:
    """PDL memory order: the entry's Q, K and V reads (and so its cache stores) follow its
    wait. Control: the same writer on another stream with no dependency is not observed."""
    step = _Step(8, ctx, seed=21, splits=splits)

    def shipped(qkv, pool):
        return op(*step.fold_args(pool, pdl=True, qkv=qkv))

    assert _under_late_projection(step, shipped) == [True]
    raced = []
    side = torch.cuda.Stream()
    want_pool = step.pool.clone()
    want = _prep_then_v4(step, want_pool)
    for _ in range(3):
        pool, qkv = step.pool.clone(), torch.ones_like(step.qkv)
        torch.cuda.synchronize()
        _late_projection(qkv, step.qkv.clone(), 20_000_000, False, stream=side)
        po, pl = op(*step.fold_args(pool, pdl=True, qkv=qkv))
        torch.cuda.synchronize()
        raced.append(bool(torch.equal(_written(po, pl, 8)[1], _written(*want[:2], 8)[1])))
        if not raced[-1]:
            break
    assert not all(raced), "the control cannot tell a missing dependency from a present one"


@pytest.mark.parametrize("ctx, splits", [(8195, None), (40003, 96)])
def test_a_loader_that_does_not_wait_for_the_append_is_caught(ctx, splits) -> None:
    """Negative control of the append's release: the build whose loader does not wait on
    ``app`` streams the step's page (poisoned) before the side warps wrote it."""
    step = _Step(8, ctx, seed=21, splits=splits)
    key = faf.fold_key(CKPT_EPS, step.q_scale, step.permuted)
    no_wait = fa4._compiled_kernel(torch.cuda.current_device(), True, False, True, fold=key,
                                   app_wait=False)  # fmt: skip
    equal = _under_late_projection(step, lambda q, p: _build_partials(no_wait, step, q, p), tries=3)
    assert not all(equal), "a loader without the append's release went unnoticed"


@pytest.mark.parametrize(
    "rescale_cols", (fa4.HEAD_DIM, fa4.SPILL_FREE_RESCALE_COLS), ids=("whole", "spill_free")
)
@pytest.mark.parametrize("rows, ctx", [(8, 455031), (3, 100003), (8, 1027)])
def test_the_op_is_deterministic(rows, ctx, rescale_cols) -> None:
    step = _Step(rows, ctx, seed=9)
    first_pool = step.pool.clone()
    first = _written(*op(*step.fold_args(first_pool), rescale_cols=rescale_cols), rows)
    for _ in range(5):
        pool = step.pool.clone()
        got = _written(*op(*step.fold_args(pool), rescale_cols=rescale_cols), rows)
        assert torch.equal(got[1], first[1]) and torch.equal(_bytes(got[0]), _bytes(first[0]))
        assert torch.equal(_bytes(pool), _bytes(first_pool))


def _can_rescale(ctx: int, splits) -> bool:
    """Whether some warpgroup of a launch over ``splits`` CTAs (None: the SM count) has a second
    tile, the first one it can rescale on: the two warpgroups take every other tile of their
    CTA, so a CTA needs three tiles or more."""
    pages = -(-ctx // PAGE)
    keyed = min(_sm_count() if splits is None else splits, -(-pages // fa4.MIN_PAGES))
    return -(-(-(-pages // keyed)) // 4) >= 3


def _fold_build(step: _Step, pdl: bool, rescale_cols: int, rescale_log2: float):
    """A test build of the fold kernel with this rescale threshold (the op's is 4)."""
    key = faf.fold_key(CKPT_EPS, step.q_scale, step.permuted)
    return fa4._compiled_kernel(torch.cuda.current_device(), pdl, False, True, fold=key,
                                rescale_cols=rescale_cols, rescale_log2=rescale_log2)  # fmt: skip


def _partials_differ(step: _Step, a, b) -> bool:
    oa, la = _written(a[0], a[1], step.rows)
    ob, lb = _written(b[0], b[1], step.rows)
    return not (torch.equal(la, lb) and torch.equal(_bytes(oa), _bytes(ob)))


#: Cells of the spill-free fold build at the op's threshold: (rows, ctx, splits; None = the SM
#: count). 100k and 455k keys over the SM count and 1,027 / 8,192 / 19,457 keys over 1 / 8 / 19
#: splits give a warpgroup three tiles or more; 8,192 keys over the SM count (one tile per CTA),
#: 40 and 131 keys cannot rescale.
_SPILL_FREE_CELLS = [
    (8, 455031, None),
    (3, 455031, None),
    (1, 455031, None),
    (8, 100003, None),
    (3, 100003, None),
    (1, 100003, None),
    (3, 19457, 19),
    (8, 8192, 8),
    (1, 8192, 8),
    (1, 1027, 1),
    (8, 8192, None),
    (8, 40, None),
    (8, 131, None),
]


@pytest.mark.parametrize("data", ("random", "spikes"))
@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
@pytest.mark.parametrize("permuted", (False, True), ids=("ckpt", "permuted"))
@pytest.mark.parametrize("rows, ctx, splits", _SPILL_FREE_CELLS)
def test_the_spill_free_fold_build_gives_the_fold_builds_bits(
    rows, ctx, splits, permuted, pdl, data
) -> None:
    """The spill-free build on the fold build (``rescale_cols=16``): the same partials and the same
    cache bytes as the fold build with the whole-head rescale (and so as prep -> v4). On spike
    data, where a warpgroup can rescale, the op's build did rescale: a build that never
    rescales (``rescale_log2=1e9``) gives other partials."""
    step = _Step(rows, ctx, seed=ctx + rows, splits=splits, permuted=permuted,
                 spikes=data == "spikes")  # fmt: skip
    pool_a, pool_b = step.pool.clone(), step.pool.clone()
    a = op(*step.fold_args(pool_a, pdl=pdl))
    b = op(*step.fold_args(pool_b, pdl=pdl), rescale_cols=fa4.SPILL_FREE_RESCALE_COLS)
    torch.cuda.synchronize()
    _assert_same(step, pool_a, pool_b, a, b, f"spill-free fold {rows}x{ctx}")
    if data == "spikes" and _can_rescale(ctx, splits):
        never = _fold_build(step, pdl, HEAD_DIM, 1.0e9)
        c = _build_partials(never, step, step.qkv, step.pool.clone())
        torch.cuda.synchronize()
        assert _partials_differ(step, a, c), "the spikes did not make the op's fold build rescale"


#: Cells of the builds that rescale on every rise: (ctx, splits), each with three tiles or more
#: in a CTA.
_RESCALING_CELLS = [(455031, None), (100003, None), (19457, 19), (8192, 8), (1027, 1)]


@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
@pytest.mark.parametrize("permuted", (False, True), ids=("ckpt", "permuted"))
@pytest.mark.parametrize("rows", (1, 3, 8))
@pytest.mark.parametrize("ctx, splits", _RESCALING_CELLS)
def test_the_chunked_fold_rescale_gives_the_whole_head_rescales_bits(
    ctx, splits, rows, permuted, pdl
) -> None:
    """With ``rescale_log2=0`` (test builds only) a tile rescales a row's O whenever it raises
    the row's running maximum, so the fold build's rescale runs on many tiles instead of
    rarely. The 16-column build gives the whole-head build's partials and cache bytes. The
    rescale did run: those builds' partials differ from the op's (``rescale_log2=4``)."""
    assert _can_rescale(ctx, splits), (ctx, splits)
    step = _Step(rows, ctx, seed=ctx + 7 * rows, splits=splits, permuted=permuted)
    pool_op, pool_a, pool_b = step.pool.clone(), step.pool.clone(), step.pool.clone()
    shipped = op(*step.fold_args(pool_op, pdl=pdl))
    want = _build_partials(_fold_build(step, pdl, HEAD_DIM, 0.0), step, step.qkv, pool_a)
    got = _build_partials(
        _fold_build(step, pdl, fa4.SPILL_FREE_RESCALE_COLS, 0.0), step, step.qkv, pool_b
    )
    torch.cuda.synchronize()
    assert _partials_differ(step, want, shipped), "no row rescaled: the case does not exercise it"
    _assert_same(step, pool_a, pool_b, want, got, f"forced-rescale fold {rows}x{ctx}")


def test_rejected_inputs() -> None:
    step = _Step(8, 1000, seed=6)
    pool = step.pool.clone()
    with pytest.raises(ValueError):
        op(*step.fold_args(pool, qkv=step.qkv[:, :1280].contiguous()))  # the fused operand
    nine = _Step(9, 1000, seed=6)
    with pytest.raises(ValueError):
        op(*nine.fold_args(nine.pool))
    with pytest.raises(ValueError):
        op(*step.fold_args(pool.view(torch.uint8)))
    with pytest.raises(ValueError):
        op(*step.fold_args(pool, splits=0))
    with validating(op):
        short = _Step(8, 1000, seed=6)
        short.seq_lens.fill_(7)
        with pytest.raises(ValueError):
            op(*short.fold_args(pool))
    # A build not compiled yet is not compiled inside a capture.
    fresh = _Step(8, 1000, seed=6, q_scale=3.25)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before capturing"):
        with torch.cuda.graph(graph):
            op(*fresh.fold_args(fresh.pool.clone()))
