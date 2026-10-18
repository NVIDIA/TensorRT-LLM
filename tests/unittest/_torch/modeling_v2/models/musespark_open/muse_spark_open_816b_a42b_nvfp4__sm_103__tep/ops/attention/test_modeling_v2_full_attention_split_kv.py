# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open full_attention_split_kv and combine_norm_gate entries (not routed).

What is gated:

* full_attention_split_kv: the merged partials are within a bound of the entry's reference derived from
  the kernel's E4M3 rounding of P; merged by combine_only they are no farther from an fp64
  reference than the built-in trtllm-gen path (<= 1.0x, max and mean, every long and short
  case), and on the long contexts within the declared budget (max-abs <= 0.15x and mean-abs
  <= 0.035x the built-in's) except the listed seeds that exceed it; with a KV scale of 0.5 or
  2 the same (within the bound, no farther from fp64 than the built-in fed that scale); NaN
  bytes in the unwritten
  slots of the last page and in unreferenced pages change no output bit; repeated calls are
  bitwise equal; one captured graph replays at other contexts (8 to 1M keys) exactly as eager
  calls do; the kernel's layout does not depend on the first call; a PDL launch observes late
  writers of the step's own pages, of the query and of older pages, in configurations where a
  build whose loader does not wait, or one that streams older pages before the wait, is caught;
  the kernel's slower builds (``k_tile=False``, ``p_split=False``) write the op's build's
  partials bit for bit; the spill-free build (``rescale_cols`` 16 or 32)
  writes the op's build's partials bit for bit on every certified case, PDL or not, NaN bytes
  included, and also on builds that rescale O whenever a row's maximum rises
  (``rescale_log2=0``); it is deterministic, replays from one captured graph at other contexts
  and observes the late writers the op's build observes.
* combine_norm_gate: within a derived band of its fp64 reference; bit for bit
  ``attention_norm_gate`` applied to ``combine_only``; key-less splits (``-inf``) with NaN
  bytes in their O change nothing; with ``seq_lens`` the splits past the keyed count are not
  read and the result is bit for bit the unmasked one on full_attention_split_kv's partials;
  deterministic.
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
    full_attention_split_kv as fa4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    norm_gate as ng,
)
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode

__extra_import_path__ = ["../../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="full_attention_split_kv is built for sm_103 and sm_107",
)

op = fa4.full_attention_split_kv
DEV = "cuda"
FP8 = torch.float8_e4m3fn
HEAD_DIM, PAGE = fa4.HEAD_DIM, fa4.PAGE
#: One tep8 rank: 16 query heads, 2 KV heads.
H, G = 16, 2
Q_WIDTH, KV_WIDTH = H * HEAD_DIM, G * HEAD_DIM
#: The checkpoint's query pre-scale: a post-norm query element has this RMS.
Q_SCALE = 5.473
EPS = 1e-5


def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


class _Step:
    """One decode step: a fused-QKV operand and an FP8 pool in the TRT-LLM paged layout.

    Logical page ``i`` is K block ``2 perm[i]`` and V block ``2 perm[i] + 1`` of the pool,
    with ``perm`` a random permutation. ``pool`` holds the cache before the step; ``appended()``
    is the cache after the step's K/V append (the built-in append, which the folded prep
    ``preattn_prep_append`` reproduces byte for byte).
    """

    def __init__(self, rows, ctx, *, seed=0, scale=1.0, q_gain=1.0, splits=None, width=None):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx = rows, ctx
        self.splits = _sm_count() if splits is None else splits
        pages = -(-ctx // PAGE)
        width = pages if width is None else width
        spare = width + 4
        perm = torch.randperm(spare, generator=g, device=DEV)[:width].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = torch.randn(2 * spare, G, PAGE, HEAD_DIM, generator=g, device=DEV).to(FP8)
        q = torch.randn(rows, Q_WIDTH, generator=g, device=DEV) * (Q_SCALE * q_gain)
        k = torch.randn(rows, KV_WIDTH, generator=g, device=DEV)
        v = torch.randn(rows, KV_WIDTH, generator=g, device=DEV)
        self.qkv = torch.cat((q, k, v), dim=1).to(torch.bfloat16)
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.scale = torch.tensor([scale], dtype=torch.float32, device=DEV)
        self.inv_scale = 1.0 / self.scale

    def appended(self) -> torch.Tensor:
        pool = self.pool.clone()
        _builtin_step(self, pool, fmha=False)
        return pool

    def args(self, pool, table=None, splits=None, pdl=False, prefetch=False):
        return (
            self.qkv,
            pool,
            self.table if table is None else table,
            self.seq_lens,
            self.scale,
            self.splits if splits is None else splits,
            pdl,
            prefetch,
        )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


_BUILTIN_BUFFERS = {}


def _builtin_buffers():
    if not _BUILTIN_BUFFERS:
        _BUILTIN_BUFFERS["workspace"] = torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)
        _BUILTIN_BUFFERS["counter"] = torch.zeros(4096, dtype=torch.int32, device=DEV)
    return _BUILTIN_BUFFERS["workspace"], _BUILTIN_BUFFERS["counter"]


def _builtin_step(step: _Step, pool: torch.Tensor, fmha: bool = True):
    """The built-in decode step on ``pool``: thop's generation preprocess (the K/V append and Q
    to E4M3), then flashinfer's trtllm-gen decode; ``None`` with ``fmha=False``."""
    import flashinfer.decode as fdec

    rows, ctx = step.rows, step.ctx
    workspace, counter = _builtin_buffers()
    (q_processed, kv_pool, block_tables, _, bmm1_scale, bmm2_scale, fmha_workspace, _, _,
     max_kv_len, window_left, _) = thop.trtllm_gen_generation_preprocess(
        step.qkv, workspace, step.seq_lens, None, None, step.table.view(1, 1, 2, -1),
        torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64),
        torch.tensor([[0, 0]], dtype=torch.int32), step.scale, step.inv_scale, None, None, None,
        None, 0, 0, H, G, HEAD_DIM, PAGE, int(QuantMode.FP8_KV_CACHE), ctx, ctx, rows,
        1, rows, ctx, 0, 10000.0, 0, 1.0, 1024, 0, 1.0 / math.sqrt(HEAD_DIM), 1.0, True, rows, 0,
        _sm_count(), pool.shape[0], 2, True, False)  # fmt: skip
    if not fmha:
        return None
    q8 = q_processed.view(torch.uint8).flatten()[: rows * Q_WIDTH].view(FP8).view(rows, H, HEAD_DIM)
    out = torch.empty(rows, Q_WIDTH, dtype=torch.bfloat16, device=DEV)
    fdec.trtllm_batch_decode_with_kv_cache(
        query=q8,
        kv_cache=(kv_pool, kv_pool),
        workspace_buffer=fmha_workspace,
        block_tables=block_tables,
        seq_lens=step.seq_lens,
        max_seq_len=max_kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=window_left,
        out=out.view(rows, H, HEAD_DIM),
        kv_layout="HND",
        enable_pdl=False,
        backend="trtllm-gen",
        q_len_per_req=rows,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=bmm1_scale.narrow(0, 1, 1),
        multi_ctas_kv_counter_buffer=counter,
    )
    return out


def _attention(
    step: _Step, pool: torch.Tensor, rescale_cols: int = HEAD_DIM, **kwargs
) -> torch.Tensor:
    """full_attention_split_kv, then combine_only: the BF16 attention output ``[rows, 1024]``."""
    po, pl = op(*step.args(pool, **kwargs), rescale_cols=rescale_cols)
    return cng.combine_only(po, pl, step.rows)


def _gather_kv(pool, table, ctx, scale):
    keys = torch.arange(ctx, device=DEV)
    raw = _bytes(pool)

    def gather(kv):
        return raw[table[kv, keys // PAGE].long(), :, keys % PAGE].view(FP8).double() / scale

    return gather(0), gather(1)  # [ctx, G, 64] each


def _fp64_reference(qkv, pool, table, ctx, scale=1.0):
    """fp64 attention of BF16 Q (never quantized) over the FP8 K/V the cache holds."""
    rows = qkv.shape[0]
    k, v = _gather_kv(pool, table, ctx, scale)
    q = qkv[:, :Q_WIDTH].double().view(rows, H, HEAD_DIM)
    out = torch.empty(rows, H, HEAD_DIM, dtype=torch.float64, device=DEV)
    group = H // G
    for t in range(rows):
        end = ctx - rows + t + 1
        for h in range(G):
            s = q[t, h * group : (h + 1) * group] @ k[:end, h].T / math.sqrt(HEAD_DIM)
            out[t, h * group : (h + 1) * group] = torch.softmax(s, -1) @ v[:end, h]
    return out.reshape(rows, Q_WIDTH)


def _errors(out, ref):
    d = (out.double() - ref).abs()
    return float(d.max()), float(d.mean())


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """One BF16 ulp at |x| (2^-133 at zero)."""
    _, exponent = torch.frexp(x.double().abs())
    return torch.ldexp(torch.ones_like(x, dtype=torch.float64), exponent - 8)


def _op_q(qkv: torch.Tensor) -> torch.Tensor:
    """The op's Q, Q_hi + Q_lo, formed as the entry's reference forms it."""
    q = qkv[:, :Q_WIDTH].float()
    c = q * 1048577.0
    hi = c - (c - q)

    def e4m3(x):
        return x.clamp(-448.0, 448.0).to(FP8).float()

    return e4m3(hi) + e4m3(q - hi)


def _e4m3_p_bound(qkv, pool, table, ctx, scale=1.0):
    """Per element, how far the merged partials may sit from the entry's reference (fp64).

    Both see the same Q (Q_hi + Q_lo), the same E4M3 K/V and fp32-accumulated scores; the
    reference keeps P exact. The kernel rounds each p_j = exp(s_j - max s) to E4M3 after
    scaling it by 2^(4 + max s - m), m its running maximum (m <= max s, and m is at least
    max s - 4 in log2 units, so the scaled value stays below 448): a scaled value of at least
    2^-6 is normal and moves by at most 2^-4 of itself; a smaller one moves by at most 2^-10
    of the scale, i.e. at most min(p_j, 2^-14) in p units. So p_j >= 2^-10 moves by at most
    2^-4 p_j and a smaller p_j by at most min(p_j, 2^-14), plus an allowance of 2^-10 p_j for
    exp2.approx and the score's fp32 rounding. The denominator sums the unrounded P, so the
    output moves by at most sum_j dp_j |V_j| / sum_j p_j. Each split's normalized O is stored
    in FP16 (2^-11 relative) and the in-CTA merge uses approximate exp2 / reciprocal / log2:
    2^-10 sum_j p_j |V_j| / sum_j p_j covers them.
    """
    rows = qkv.shape[0]
    k, v = _gather_kv(pool, table, ctx, scale)
    q = _op_q(qkv).double().view(rows, H, HEAD_DIM)
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


def _check_bound(out, ref, bound, what):
    limit = bound + _bf16_ulp(torch.maximum(out.double().abs(), ref.double().abs()))
    diff = (out.double() - ref.double()).abs()
    used = float((diff / limit).max())
    print(f"\n{what}: max |op - reference| {float(diff.max()):.4g}, {used:.3f} of the E4M3-P bound")
    assert used <= 1.0, (what, used)


# --------------------------------------------------------------------------
# full_attention_split_kv: cells against its own reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    """The merged partials are within ``_e4m3_p_bound`` of the reference (one BF16 ulp of
    slack); the operands are read-only."""
    certifying(op)
    spec = cell.spec
    step = _Step(
        spec["rows"], spec["ctx"], seed=zlib.crc32(cell.why.encode()), splits=spec.get("splits")
    )
    pool = step.appended()
    before_q, before_pool = step.qkv.clone(), pool.clone()
    kw = dict(pdl=spec.get("pdl", False), prefetch=spec.get("prefetch", False))
    po, pl = op(*step.args(pool, **kw))
    assert torch.equal(step.qkv, before_q) and torch.equal(_bytes(pool), _bytes(before_pool))
    ref = op.reference(*step.args(pool, **kw))
    merged = fa4.merge_partials(po, pl, step.rows)
    assert not torch.isnan(merged).any()
    bound = _e4m3_p_bound(step.qkv, pool, step.table, step.ctx)
    _check_bound(merged, ref, bound, cell.why)


# --------------------------------------------------------------------------
# full_attention_split_kv + combine_only against the built-in path, both measured from fp64
# --------------------------------------------------------------------------
_LONG_CASES = [(8, 1048581), (3, 100003), (1, 100000)]
#: The two contexts the declared budget names, 8 rows, over 8 seeds (seed 0 is ``ctx + 8``).
_SEEDED_CONTEXTS = (100000, 455031)
_SEEDS = tuple(range(8))
_SHORT_CASES = [
    (rows, ctx, gain, seed)
    for ctx in (9, 40, 41, 64, 100, 300, 1000, 2056)
    for rows in (1, 4, 8)
    for gain in (1.0, 0.5, 2.0)
    for seed in (0, 1)
    if rows <= ctx
]
#: The declared budget on the long contexts (max-abs, mean-abs), as fractions of the built-in's.
LONG_BUDGET = (0.15, 0.035)
#: (ctx, seed) cases of the 8-seed sweep that exceed the declared budget (mean-abs above
#: 0.035x). They are asserted to stay outside it, so this list cannot go stale; every case
#: still meets the policy bound (no farther from fp64 than the built-in).
OVER_BUDGET = frozenset({(455031, 3), (455031, 4)})


def _ratios(step: _Step, what: str):
    builtin_pool = step.pool.clone()
    builtin = _builtin_step(step, builtin_pool)
    ref = _fp64_reference(step.qkv, builtin_pool, step.table, step.ctx)
    out = _attention(step, builtin_pool)
    o_max, o_mean = _errors(out, ref)
    b_max, b_mean = _errors(builtin, ref)
    tiny = 1e-30
    r_max, r_mean = o_max / max(b_max, tiny), o_mean / max(b_mean, tiny)
    print(
        f"\n{what}: op max {o_max:.4g} mean {o_mean:.3g}; builtin max {b_max:.4g} mean "
        f"{b_mean:.3g}; ratios {r_max:.3f} / {r_mean:.3f}"
    )
    return r_max, r_mean


def _within_budget(r_max: float, r_mean: float) -> bool:
    return r_max <= LONG_BUDGET[0] and r_mean <= LONG_BUDGET[1]


@pytest.mark.parametrize("rows, ctx", _LONG_CASES)
def test_long_contexts_meet_the_declared_budget(rows, ctx) -> None:
    r_max, r_mean = _ratios(_Step(rows, ctx, seed=ctx + rows), f"rows {rows} ctx {ctx}")
    assert _within_budget(r_max, r_mean), (r_max, r_mean)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("ctx", _SEEDED_CONTEXTS)
def test_the_declared_budget_over_seeds(ctx, seed) -> None:
    """The policy bound on every seed; the declared budget on every seed but the listed ones,
    which must still exceed it."""
    step = _Step(8, ctx, seed=7919 * seed + ctx + 8)
    r_max, r_mean = _ratios(step, f"rows 8 ctx {ctx} seed {seed}")
    assert r_max <= 1.0 and r_mean <= 1.0, (r_max, r_mean)
    if (ctx, seed) in OVER_BUDGET:
        assert not _within_budget(r_max, r_mean), (
            f"ctx {ctx} seed {seed} is listed in OVER_BUDGET but is within it: {r_max}, {r_mean}"
        )
        pytest.xfail(
            f"declared budget exceeded: max {r_max:.4f} / mean {r_mean:.4f} of the built-in's "
            f"error vs {LONG_BUDGET}"
        )
    assert _within_budget(r_max, r_mean), (r_max, r_mean)


@pytest.mark.parametrize("rows, ctx, gain, seed", _SHORT_CASES)
def test_short_contexts_are_no_farther_from_fp64_than_the_builtin(rows, ctx, gain, seed) -> None:
    step = _Step(rows, ctx, seed=1000 * seed + 10 * ctx + rows, q_gain=gain)
    r_max, r_mean = _ratios(step, f"rows {rows} ctx {ctx} q x{gain} seed {seed}")
    assert r_max <= 1.0 and r_mean <= 1.0, (r_max, r_mean)


@pytest.mark.parametrize("scale", (0.5, 2.0))
@pytest.mark.parametrize("rows, ctx", [(8, 2056), (8, 100003)])
def test_a_non_unit_kv_scale_dequantizes_as_the_builtin(rows, ctx, scale) -> None:
    """The op dequantizes with ``kv_scale_orig_quant``, where the built-in path uses the
    bmm scale words its preprocess derives from the same scale (the split-KV attention's
    stand-in passes the former and drops the latter). With a KV scale other than 1, the step's K/V appended
    with it by the built-in preprocess: the merged partials are within the E4M3-P bound of
    the op's reference, and against fp64 of the cache dequantized with that scale they are
    no farther than the built-in path fed the same scale (worst element and mean-abs)."""
    step = _Step(rows, ctx, seed=ctx + rows + int(100 * scale), scale=scale)
    pool = step.pool.clone()
    builtin = _builtin_step(step, pool)
    po, pl = op(*step.args(pool))
    merged = fa4.merge_partials(po, pl, step.rows)
    own = op.reference(*step.args(pool))
    bound = _e4m3_p_bound(step.qkv, pool, step.table, step.ctx, scale)
    _check_bound(merged, own, bound, f"rows {rows} ctx {ctx} kv scale {scale}")
    ref = _fp64_reference(step.qkv, pool, step.table, step.ctx, scale)
    o_max, o_mean = _errors(cng.combine_only(po, pl, step.rows), ref)
    b_max, b_mean = _errors(builtin, ref)
    print(
        f"\nkv scale {scale} rows {rows} ctx {ctx}: op max {o_max:.4g} mean {o_mean:.3g}; "
        f"builtin max {b_max:.4g} mean {b_mean:.3g}"
    )
    assert o_max <= b_max and o_mean <= b_mean, (o_max, b_max, o_mean, b_mean)


# --------------------------------------------------------------------------
# full_attention_split_kv: robustness and launch properties
# --------------------------------------------------------------------------
def _poison_unwritten(step: _Step, pool: torch.Tensor) -> None:
    """E4M3 NaN (0x7F) in every slot past the context of the last page (K and V blocks),
    and in every block the table does not reference."""
    raw = _bytes(pool)
    last = (step.ctx - 1) // PAGE
    tail = step.ctx % PAGE
    if tail:
        for kv in range(2):
            raw[int(step.table[kv, last]), :, tail:] = 0x7F
    used = torch.zeros(pool.shape[0], dtype=torch.bool, device=DEV)
    used[step.table[:, : last + 1].flatten().long()] = True
    raw[~used] = 0x7F


@pytest.mark.parametrize("rows, ctx", [(8, 100003), (8, 455031), (3, 1027), (8, 40), (5, 1048581)])
def test_unwritten_slots_do_not_reach_the_output(rows, ctx) -> None:
    step = _Step(rows, ctx, seed=ctx)
    pool = step.appended()
    clean = _attention(step, pool.clone())
    _poison_unwritten(step, pool)
    poisoned = _attention(step, pool)
    assert not torch.isnan(poisoned.float()).any()
    assert torch.equal(poisoned, clean)


def _written(po: torch.Tensor, pl: torch.Tensor, rows: int):
    """The partials the op writes: every row's log2-sum-exp, and O of the splits with keys (a
    key-less split writes ``-inf`` and leaves its O as it was)."""
    o, lse = po[: rows * H].clone(), pl[: rows * H]
    o[~torch.isfinite(lse)] = 0.0
    return o, lse


@pytest.mark.parametrize("rows, ctx", [(8, 455031), (3, 100003), (8, 1027)])
def test_the_op_is_deterministic(rows, ctx) -> None:
    step = _Step(rows, ctx, seed=9)
    pool = step.appended()
    first = _written(*op(*step.args(pool)), rows)
    for _ in range(10):
        got = _written(*op(*step.args(pool)), rows)
        assert torch.equal(got[0], first[0]) and torch.equal(got[1], first[1])


@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
def test_a_captured_graph_replays_at_other_contexts(pdl) -> None:
    """The context length is read on the device: one capture serves every step, from the
    single-split path to 1M keys; the combine that derives the keyed splits from the same
    ``seq_lens`` gives the bits of the one that reads every split."""
    _check_graph_replays(pdl, HEAD_DIM)


def _check_graph_replays(pdl: bool, rescale_cols: int) -> None:
    top = 1048581
    captured = _Step(8, 262145, seed=1, width=-(-top // PAGE))
    static_qkv, static_seq = captured.qkv.clone(), captured.seq_lens.clone()
    pool = captured.pool
    part_o, part_lse = fa4.partial_buffers(captured.splits, DEV)

    def step_fn():
        op(static_qkv, pool, captured.table, static_seq, captured.scale, captured.splits, pdl,
           False, True, part_o, part_lse, rescale_cols)  # fmt: skip
        keyed = cng.combine_only(part_o, part_lse, 8, "after_wait", seq_lens=static_seq)
        return torch.cat((cng.combine_only(part_o, part_lse, 8), keyed))

    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step_fn()  # compiles outside the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step_fn()
    g = torch.Generator(device=DEV).manual_seed(3)
    for ctx in (8, 40, 1027, 8192, 19457, 100003, 262145, 455031, top):
        for fresh in (False, True):
            if fresh:
                static_qkv.copy_(
                    torch.randn(static_qkv.shape, generator=g, device=DEV).to(torch.bfloat16)
                )
            static_seq.fill_(ctx)
            graph.replay()
            torch.cuda.synchronize()
            got = graph_out.clone()
            eager = step_fn()
            assert torch.equal(got, eager), (ctx, fresh)
            assert torch.equal(got[:8], got[8:]), (ctx, fresh)


def test_a_first_call_on_a_one_page_table_does_not_fix_the_layout() -> None:
    """The kernel is compiled against pointers and explicit strides: a first call on a
    [2, 1] table must not change how later tables are read."""
    fa4._compiled_kernel.cache_clear()
    fa4._compiled_keys.clear()
    first = _Step(8, 8, seed=4)
    assert tuple(first.table.shape) == (2, 1)
    op(*first.args(first.appended()))
    for rows, ctx in ((8, 33), (8, 40), (8, 97), (8, 1000), (8, 100003)):
        step = _Step(rows, ctx, seed=ctx)
        pool = step.appended()
        po, pl = op(*step.args(pool))
        ref = op.reference(*step.args(pool))
        bound = _e4m3_p_bound(step.qkv, pool, step.table, ctx)
        merged = fa4.merge_partials(po, pl, rows)
        _check_bound(merged, ref, bound, f"rows {rows} ctx {ctx} after a one-page first call")


def test_rejected_inputs() -> None:
    step = _Step(8, 1000, seed=6)
    pool = step.appended()
    for splits in (0, fa4.MAX_SPLITS + 1):
        with pytest.raises(ValueError):
            op(*step.args(pool, splits=splits))
    nine = _Step(9, 1000, seed=6)
    with pytest.raises(ValueError):
        op(*nine.args(nine.pool))
    with pytest.raises(ValueError):
        op(step.qkv[:, :-64], *step.args(pool)[1:])
    with pytest.raises(ValueError):
        op(*step.args(pool, table=step.table.t().contiguous().t()))  # column-major [2, pages]
    with pytest.raises(ValueError):
        op(*step.args(pool, table=step.table.long()))
    with pytest.raises(ValueError):
        op(*step.args(pool.view(torch.uint8)))
    wide = _Step(8, 1000, seed=6, width=fa4.MAX_CTA_PAGES * 2 + 1)
    with pytest.raises(ValueError):
        op(*wide.args(wide.pool, splits=2))  # more than 256 pages per split
    # Inputs the kernel would take and answer wrongly: the guard refuses them.
    with validating(op):
        with pytest.raises(ValueError):
            op(*step.args(pool, table=step.table[:, :10].contiguous()))
        bad = step.table.clone()
        bad[1, 3] = pool.shape[0]
        with pytest.raises(ValueError):
            op(*step.args(pool, table=bad))
        short = _Step(8, 1000, seed=6)
        short.seq_lens.fill_(7)
        with pytest.raises(ValueError):
            op(*short.args(pool))


@triton.jit
def _globaltimer(anchor):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [anchor], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _late_writer_kernel(pool, src_pool, blocks, qkv, src_qkv, qkv_bytes, delay_ns,
                        EARLY_TRIGGER: tl.constexpr, BLOCK_BYTES: tl.constexpr,
                        NUM_BLOCKS: tl.constexpr, CHUNK: tl.constexpr):  # fmt: skip
    """Test-only PDL primary: release the dependent at entry (optionally), spin, then write the
    step's pages (whole K and V blocks) and the fused operand."""
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    start = _globaltimer(pid)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(pid)
    offs = tl.arange(0, CHUNK)
    if pid < NUM_BLOCKS:
        b = tl.load(blocks + pid).to(tl.int64)
        for c in range(0, BLOCK_BYTES, CHUNK):
            o = b * BLOCK_BYTES + c + offs
            tl.store(pool + o, tl.load(src_pool + o))
    else:
        for c in range(0, qkv_bytes, CHUNK):
            o = c + offs
            m = o < qkv_bytes
            tl.store(qkv + o, tl.load(src_qkv + o, mask=m), mask=m)


def _step_pages(step):
    """The logical pages that hold a key of the step's own rows."""
    return range((step.ctx - step.rows) // PAGE, (step.ctx - 1) // PAGE + 1)


def _late_write(step, pool, want_pool, qkv, want_qkv, delay_ns, early, stream=None, pages=None):
    """Launch ``_late_writer_kernel``: after ``delay_ns`` it copies the K and V blocks of
    ``pages`` (default: the step's own) from ``want_pool`` into ``pool``, and ``want_qkv`` into
    ``qkv``."""
    pages = _step_pages(step) if pages is None else pages
    blocks = step.table[:, pages.start : pages.stop].flatten().contiguous()
    block_bytes = G * PAGE * HEAD_DIM
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        _late_writer_kernel[(blocks.numel() + 1,)](
            _bytes(pool),
            _bytes(want_pool),
            blocks,
            _bytes(qkv),
            _bytes(want_qkv),
            qkv.numel() * 2,
            delay_ns,
            EARLY_TRIGGER=early,
            BLOCK_BYTES=block_bytes,
            NUM_BLOCKS=blocks.numel(),
            CHUNK=1024,
            num_warps=1,
            launch_pdl=True,
        )


def _stale(want_pool, step, pages):
    """``want_pool`` with E4M3 1.0 in every byte of the K and V blocks of ``pages``."""
    pool = want_pool.clone()
    for b in step.table[:, pages.start : pages.stop].flatten().tolist():
        _bytes(pool)[b] = 0x38
    return pool


def _build_partials(build, step, qkv, pool):
    """One compiled build of the kernel, launched with ``full_attention_split_kv``'s arguments; returns
    its partials."""
    import cuda.bindings.driver as cuda

    po, pl = fa4.partial_buffers(step.splits, DEV)
    build(
        qkv.data_ptr(), pool.data_ptr(), step.table.data_ptr(), step.seq_lens.data_ptr(),
        step.scale.data_ptr(), po.data_ptr(), pl.data_ptr(), step.rows, qkv.stride(0),
        pool.shape[0], step.table.shape[1], step.table.stride(0), step.splits,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )  # fmt: skip
    return po, pl


def _launch_build(build, step, qkv, pool):
    """One compiled build of the kernel, launched as ``full_attention_split_kv`` launches it (PDL), then
    ``combine_only``."""
    return cng.combine_only(*_build_partials(build, step, qkv, pool), step.rows)


def _shipped(step, qkv, pool, rescale_cols=HEAD_DIM):
    """The op's build (PDL, no pre-wait page prefetch), then ``combine_only``."""
    po, pl = op(
        qkv, pool, step.table, step.seq_lens, step.scale, step.splits, True, False,
        rescale_cols=rescale_cols,
    )  # fmt: skip
    return cng.combine_only(po, pl, step.rows)


def _under_late_writer(step, want_pool, launch, *, older=False, tries=1):
    """``launch(qkv, pool)`` as the PDL dependent of a primary that releases it at entry, spins
    200 us and then writes either the step's pages and the query (both stale until then) or,
    with ``older``, the first two pages (CTA 0's first ring; the query is current). Returns,
    per try, whether the output equals the eager one on the written cache."""
    want = _attention(step, want_pool)
    pages = range(0, 2) if older else _step_pages(step)
    equal = []
    for _ in range(tries):
        pool = _stale(want_pool, step, pages)
        qkv = step.qkv.clone() if older else torch.ones_like(step.qkv)
        torch.cuda.synchronize()
        _late_write(step, pool, want_pool, qkv, step.qkv.clone(), 200_000, True, pages=pages)
        got = launch(qkv, pool)
        torch.cuda.synchronize()
        equal.append(bool(torch.equal(got, want)))
    return equal


#: (context, splits; None = the SM count) where a build whose loader does not wait is caught:
#: at 8,195 keys the step's pages sit in the first ring of a CTA that starts at once, at 40,003
#: keys and 96 splits the owning CTA is resident while the writer spins. (At 100,003 keys they
#: sit behind a full ring, and at 40,003 keys with one CTA per SM the owning CTA becomes
#: resident only after the writer exits: both pass with or without the loader's wait.)
_LOADER_WAIT_CAUGHT = [(8195, None), (40003, 96)]


@pytest.mark.parametrize("ctx, splits", [(100003, None)] + _LOADER_WAIT_CAUGHT)
def test_a_late_writer_of_the_steps_pages_and_query_is_observed(ctx, splits) -> None:
    """PDL memory order. The primary releases the kernel at its entry, then spins 200 us
    before it writes the step's own pages and the query: the PDL-launched kernel is resident
    long before, and still gives the eager bits, so it reads the step's pages and Q only after
    its ``griddepcontrol.wait``. The control can fail: the same writer on a second stream with
    no dependency leaves the kernel reading stale data."""
    step = _Step(8, ctx, seed=21, splits=splits)
    want_pool = step.appended()
    assert _under_late_writer(step, want_pool, lambda q, p: _shipped(step, q, p)) == [True]

    want = _attention(step, want_pool)
    pages = _step_pages(step)
    raced_equal = []
    side = torch.cuda.Stream()
    for _ in range(3):
        pool, qkv = _stale(want_pool, step, pages), torch.ones_like(step.qkv)
        torch.cuda.synchronize()
        _late_write(step, pool, want_pool, qkv, step.qkv.clone(), 20_000_000, False, stream=side)
        raced = _shipped(step, qkv, pool)
        torch.cuda.synchronize()
        raced_equal.append(torch.equal(raced, want))
        if not raced_equal[-1]:
            break
    assert not all(raced_equal), "the control cannot tell a missing dependency from a present one"


@pytest.mark.parametrize("ctx, splits", _LOADER_WAIT_CAUGHT)
def test_a_loader_without_the_wait_is_caught(ctx, splits) -> None:
    """Negative control of the page half: the build whose loader never waits streams the
    step's pages before the late writer wrote them, and the same configuration shows it."""
    step = _Step(8, ctx, seed=21, splits=splits)
    want_pool = step.appended()
    no_wait = fa4._compiled_kernel(torch.cuda.current_device(), True, False, True, False)
    equal = _under_late_writer(
        step, want_pool, lambda q, p: _launch_build(no_wait, step, q, p), tries=3
    )
    assert not all(equal), "a loader without the dependency wait went unnoticed"


#: The kernel's slower builds (see ``FullAttnSplitKvKernel``): one N=32 MMA set per page instead
#: of the 128-key tile, and P released as one tile instead of two halves.
_SAME_VALUE_BUILDS = {
    "k_tile_off": dict(k_tile=False),
    "p_split_off": dict(p_split=False),
}


@pytest.mark.parametrize("rows, ctx", [(8, 100003), (3, 1027), (8, 40)])
@pytest.mark.parametrize("build", sorted(_SAME_VALUE_BUILDS))
def test_the_slower_builds_give_the_ops_bits(build, rows, ctx) -> None:
    """``k_tile=False`` and ``p_split=False`` change how the MMAs are issued and P is
    released, not the arithmetic: each writes the op's build's partials bit for bit."""
    step = _Step(rows, ctx, seed=ctx + rows)
    pool = step.appended()
    want = _written(*op(*step.args(pool)), rows)
    kernel = fa4._compiled_kernel(
        torch.cuda.current_device(), False, False, True, **_SAME_VALUE_BUILDS[build]
    )
    got = _written(*_build_partials(kernel, step, step.qkv, pool), rows)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), (build, rows, ctx)


@pytest.mark.parametrize("ctx, splits", [(100003, None), (8195, None)])
def test_a_late_writer_of_older_pages_is_observed(ctx, splits) -> None:
    """The op's build reads no page before its wait: a late writer of the first two pages (well
    below the step's own) is observed. Negative control: the build that streams those pages
    before the wait (``prefetch=True``) shows the stale bytes in the same configuration."""
    step = _Step(8, ctx, seed=21, splits=splits)
    want_pool = step.appended()
    shipped = _under_late_writer(
        step, want_pool, lambda q, p: _shipped(step, q, p), older=True, tries=3
    )
    assert all(shipped), shipped

    def prefetching(q, p):
        po, pl = op(q, p, step.table, step.seq_lens, step.scale, step.splits, True, True)
        return cng.combine_only(po, pl, step.rows)

    early = _under_late_writer(step, want_pool, prefetching, older=True, tries=3)
    assert not all(early), "the older-pages writer cannot tell a pre-wait page read"


# --------------------------------------------------------------------------
# full_attention_split_kv: the spill-free build
# --------------------------------------------------------------------------
#: The op's certified cases (the 13 contexts and the PDL / grid-size cells; the pre-wait
#: prefetch cell is the op's own negative-control build): (rows, ctx, splits, pdl).
_SPILL_FREE_CASES = [
    (c.spec["rows"], c.spec["ctx"], c.spec.get("splits"), c.spec.get("pdl", False))
    for c in op.CELLS
    if not c.spec.get("prefetch", False)
]


def _same_written(got, want) -> bool:
    return torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.mark.parametrize("rescale_cols", (16, 32))
@pytest.mark.parametrize(
    "rows, ctx, splits, pdl", _SPILL_FREE_CASES, ids=[str(c) for c in _SPILL_FREE_CASES]
)
def test_the_spill_free_build_gives_the_ops_bits(rescale_cols, rows, ctx, splits, pdl) -> None:
    """``rescale_cols`` 16 (the spill-free build) or 32 rescales O in chunks with the same multiply
    per element: every written partial (O of the keyed splits, every log2-sum-exp) is the op's
    default build's, bit for bit, and so is their merge."""
    step = _Step(rows, ctx, seed=ctx * 8 + rows, splits=splits)
    pool = step.appended()
    want = _written(*op(*step.args(pool, pdl=pdl)), rows)
    got = _written(*op(*step.args(pool, pdl=pdl), rescale_cols=rescale_cols), rows)
    assert _same_written(got, want), (rescale_cols, rows, ctx, splits, pdl)
    assert not torch.isnan(got[0]).any()


#: Cases for the builds that rescale O whenever a row's running maximum rises: (rows, ctx,
#: splits; None = the SM count). A warpgroup rescales only from its second tile on, so the short
#: contexts take fewer splits (19,457 keys over 19 CTAs: 8 tiles each; 1,027 keys in one CTA: 9).
_RESCALING_CASES = [
    (8, 455031, None),
    (8, 100003, None),
    (3, 19457, 19),
    (8, 1027, 1),
    (1, 100000, None),
    (8, 1048581, None),
]


@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
@pytest.mark.parametrize("rows, ctx, splits", _RESCALING_CASES)
def test_the_chunked_rescale_gives_the_whole_head_rescales_bits(rows, ctx, splits, pdl) -> None:
    """With ``rescale_log2=0`` (test builds only) a tile rescales a row's O whenever it raises
    the row's running maximum, so the rescale runs on many tiles of every case instead of
    rarely. The chunked builds (16 and 32 columns) give the whole-head build's bits. The
    rescale did run: those builds' partials differ from the op's build (``rescale_log2=4``),
    which happens only when some row raised its maximum after its warpgroup's first tile."""
    step = _Step(rows, ctx, seed=ctx + 77 * rows, splits=splits)
    pool = step.appended()
    dev = torch.cuda.current_device()
    whole = fa4._compiled_kernel(dev, pdl, False, True, rescale_log2=0.0)
    want = _written(*_build_partials(whole, step, step.qkv, pool), rows)
    shipped = _written(*op(*step.args(pool, pdl=pdl)), rows)
    changed = (want[0] != shipped[0]).any(dim=-1)
    print(f"\n{rows} rows, {ctx} keys: {int(changed.sum())} of {changed.numel()} (row, split) "
          "partials differ from the op's build")  # fmt: skip
    assert bool(changed.any()), "no row rescaled: the case does not exercise the rescale"
    for rescale_cols in (16, 32):
        chunked = fa4._compiled_kernel(
            dev, pdl, False, True, rescale_log2=0.0, rescale_cols=rescale_cols
        )
        got = _written(*_build_partials(chunked, step, step.qkv, pool), rows)
        assert _same_written(got, want), (rescale_cols, rows, ctx, pdl)


@pytest.mark.parametrize("rows, ctx", [(8, 100003), (8, 455031), (3, 1027), (8, 40), (5, 1048581)])
def test_the_spill_free_build_is_nan_proof_and_deterministic(rows, ctx) -> None:
    """NaN bytes in the unwritten slots of the last page and in unreferenced pages change no
    output bit of the spill-free build, which equals the op's build and repeats bit for bit."""
    step = _Step(rows, ctx, seed=ctx)
    pool = step.appended()
    clean = _attention(step, pool.clone())
    first = _written(*op(*step.args(pool), rescale_cols=fa4.SPILL_FREE_RESCALE_COLS), rows)
    for _ in range(3):
        again = _written(*op(*step.args(pool), rescale_cols=fa4.SPILL_FREE_RESCALE_COLS), rows)
        assert _same_written(again, first)
    _poison_unwritten(step, pool)
    poisoned = _attention(step, pool, rescale_cols=fa4.SPILL_FREE_RESCALE_COLS)
    assert not torch.isnan(poisoned.float()).any()
    assert torch.equal(poisoned, clean)


@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
def test_a_captured_graph_of_the_spill_free_build_replays_at_other_contexts(pdl) -> None:
    _check_graph_replays(pdl, fa4.SPILL_FREE_RESCALE_COLS)


@pytest.mark.parametrize("ctx, splits", [(100003, None)] + _LOADER_WAIT_CAUGHT)
def test_the_spill_free_build_observes_late_writers(ctx, splits) -> None:
    """The PDL memory order of the spill-free build (the late writers the op's build is tested
    with; their negative controls are build-independent and run above)."""
    step = _Step(8, ctx, seed=21, splits=splits)
    want_pool = step.appended()
    rc = fa4.SPILL_FREE_RESCALE_COLS
    assert _under_late_writer(step, want_pool, lambda q, p: _shipped(step, q, p, rc)) == [True]
    older = _under_late_writer(
        step, want_pool, lambda q, p: _shipped(step, q, p, rc), older=True, tries=3
    )
    assert all(older), older


def test_rejected_rescale_cols() -> None:
    step = _Step(8, 1027, seed=1)
    pool = step.appended()
    with pytest.raises(ValueError, match="rescale_cols 8"):
        op(*step.args(pool), rescale_cols=8)


# --------------------------------------------------------------------------
# combine_norm_gate
# --------------------------------------------------------------------------
cop = cng.combine_norm_gate


def _partials(rows, splits, empty=0, seed=0):
    """Partials as full_attention_split_kv leaves them: FP16 O near unit scale, log2-sum-exp spread over
    a few units; ``empty`` key-less splits carry ``-inf`` and NaN bytes in their O."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    po = torch.randn(fa4.OUT_ROWS, splits, HEAD_DIM, generator=g, device=DEV).to(torch.float16)
    pl = torch.randn(fa4.OUT_ROWS, splits, generator=g, device=DEV) * 6.0
    if empty:
        idx = torch.randperm(splits, generator=g, device=DEV)[:empty]
        pl[:, idx] = float("-inf")
        po[:, idx] = float("nan")
    return po, pl


def _keyed_partials(rows, splits, ctx, seed=0):
    """Partials of a ``ctx``-key step: the first ``keyed_splits(ctx, splits)`` splits as
    ``_partials``; every later split holds NaN bytes in its O and a finite log2-sum-exp larger
    than any keyed one, so reading either would change the merge. Returns them and the
    int32 ``seq_lens``."""
    po, pl = _partials(rows, splits, 0, seed)
    keyed = fa4.keyed_splits(ctx, splits)
    po[:, keyed:] = float("nan")
    pl[:, keyed:] = 100.0
    return po, pl, torch.tensor([ctx], dtype=torch.int32, device=DEV)


def _gate(rows, ndim, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    gate = torch.randn(rows, H, HEAD_DIM, generator=g, device=DEV).to(torch.bfloat16)
    return gate if ndim == 3 else gate.view(rows, H * HEAD_DIM)


@pytest.mark.parametrize("cell", cop.CELLS, ids=[c.why[:48] for c in cop.CELLS])
def test_combine_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    """Within two BF16 ulps at the row's scale of the fp64 reference (``element_ulp=4`` in
    ``ULP`` units of 2^-8), and half an ulp in RMS: the reference rounds the fp64 merge to BF16
    where the op rounds its fp32 merge (one ulp where the two straddle a rounding boundary),
    and the norm-gate's BF16 rounding points can then differ by one more ulp."""
    certifying(cop)
    spec = cell.spec
    rows = spec["rows"]
    seed = zlib.crc32(cell.why.encode())
    seq_lens = None
    if "ctx" in spec:
        po, pl, seq_lens = _keyed_partials(rows, spec["splits"], spec["ctx"], seed=seed)
    else:
        po, pl = _partials(rows, spec["splits"], spec.get("empty", 0), seed=seed)
    gate = _gate(rows, spec["gate_ndim"])
    trigger = spec.get("trigger", "after_wait")
    out = cop(po, pl, rows, gate, EPS, trigger, seq_lens=seq_lens)
    ref = cop.reference(po, pl, rows, gate, EPS, trigger, seq_lens=seq_lens)
    assert not torch.isnan(out.float()).any()
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=1.0)


@pytest.mark.parametrize(
    "rows, splits, empty", [(8, 152, 0), (8, 152, 88), (3, 19, 5), (1, 304, 0)]
)
@pytest.mark.parametrize("trigger", ("after_wait", "exit", "entry"))
def test_the_fused_norm_gate_is_norm_gate_of_the_merge(rows, splits, empty, trigger) -> None:
    po, pl = _partials(rows, splits, empty, seed=rows * 1000 + splits)
    gate = _gate(rows, 2)
    fused = cop(po, pl, rows, gate, EPS, trigger)
    two = ng.attention_norm_gate(cng.combine_only(po, pl, rows), gate, EPS, trigger)
    assert torch.equal(fused, two)


def test_key_less_splits_change_nothing() -> None:
    rows, splits = 8, 152
    po, pl = _partials(rows, splits, 0, seed=5)
    gate = _gate(rows, 2)
    want = cop(po, pl, rows, gate, EPS)
    wider_o = torch.cat((po, torch.full_like(po[:, :40], float("nan"))), dim=1).contiguous()
    wider_l = torch.cat((pl, torch.full_like(pl[:, :40], float("-inf"))), dim=1).contiguous()
    assert torch.equal(cop(wider_o, wider_l, rows, gate, EPS), want)


@pytest.mark.parametrize("splits", (152, 96))
@pytest.mark.parametrize("ctx", (8, 40, 161, 1027, 8192, 19457, 100003))
def test_the_keyed_combine_reads_no_key_less_split(ctx, splits) -> None:
    """With ``seq_lens`` the splits past the keyed count are not read: garbage there (NaN O,
    a log2-sum-exp above every keyed one) gives the bits that ``-inf`` there gives without
    ``seq_lens``, for the fused norm-gate and for ``combine_only``."""
    rows = 8
    po, pl, seq_lens = _keyed_partials(rows, splits, ctx, seed=ctx + splits)
    clean_l = pl.clone()
    clean_l[:, fa4.keyed_splits(ctx, splits) :] = float("-inf")
    gate = _gate(rows, 2)
    assert torch.equal(
        cop(po, pl, rows, gate, EPS, seq_lens=seq_lens), cop(po, clean_l, rows, gate, EPS)
    )
    assert torch.equal(
        cng.combine_only(po, pl, rows, seq_lens=seq_lens), cng.combine_only(po, clean_l, rows)
    )


def test_combine_is_deterministic() -> None:
    po, pl = _partials(8, 152, 30, seed=7)
    gate = _gate(8, 2)
    first = cop(po, pl, 8, gate, EPS)
    for _ in range(10):
        assert torch.equal(cop(po, pl, 8, gate, EPS), first)


def test_combine_rejected_inputs() -> None:
    po, pl = _partials(8, 152, 0)
    gate = _gate(8, 2)
    with pytest.raises(ValueError):
        cop(po.float(), pl, 8, gate, EPS)
    with pytest.raises(ValueError):
        cop(po, pl[:, :151].contiguous(), 8, gate, EPS)
    with pytest.raises(ValueError):
        cop(po, pl, 9, gate, EPS)
    with pytest.raises(ValueError):
        cop(po, pl, 8, gate.float(), EPS)
    with pytest.raises(ValueError):
        cop(po, pl, 8, gate, EPS, trigger="late")
    with pytest.raises(ValueError):
        cop(po, pl, 8, gate, EPS, seq_lens=torch.tensor([8192], device=DEV))  # int64
    with pytest.raises(ValueError):
        cng.combine_only(po, pl, 8, seq_lens=torch.tensor([8192], dtype=torch.int32))  # host
