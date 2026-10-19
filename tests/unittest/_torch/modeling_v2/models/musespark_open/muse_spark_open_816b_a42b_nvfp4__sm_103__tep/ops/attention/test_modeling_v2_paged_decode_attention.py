# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open paged_decode_attention entry and its call site.

The entry makes two claims of different strength:

* The KV cache write is the built-in one, byte for byte: the step's K/V land
  at the same slots with the same E4M3 bits as ``applyBiasRopeUpdateKVCacheV2``
  writes them (run here through ``thop.trtllm_gen_generation_preprocess``, the
  launch the built-in decode step makes), saturating, NaN and subnormal inputs
  included.
* The attention output is close, not equal: within a stated band of the
  entry's fp32 reference (the cells), and no farther from an fp64 reference
  than the built-in trtllm-gen path is.

The call site is checked in the backend: a built-in ``MuseSparkOpenAttention``
at one tep8 rank's geometry (16 query and 2 KV heads of 64) with an FP8 KV
cache in a ``KVCacheManagerV2``, adopted as ``MuseSparkOpenTepDecodeAttention``.

``MODELING_V2_TIMING=1`` also runs the microbenchmark (printed, never gated).
"""

import contextlib
import json
import math
import os
import statistics
import tempfile
import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.paged_decode_attention import (  # noqa: E501
    HEAD_DIM,
    window_splits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.paged_decode_attention import (  # noqa: E501
    paged_decode_attention as op,
)
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "paged_decode_attention requires a CUDA device"

DEV = "cuda"
FP8 = torch.float8_e4m3fn
#: One tep8 rank: 128 / 8 query heads, 16 / 8 KV heads.
H, G = 16, 2
Q_WIDTH, KV_WIDTH = H * HEAD_DIM, G * HEAD_DIM
#: The checkpoint's query pre-scale: a post-norm query element has this RMS.
Q_SCALE = 5.473
WINDOW = 2048


class _Step:
    """One decode step: a fused-QKV operand and an FP8 pool in the TRT-LLM paged layout.

    Logical page ``i`` of the sequence is K page ``2 * perm[i]`` and V page
    ``2 * perm[i] + 1`` of the pool (a ``kv_factor`` 2 pool), with ``perm`` a
    random permutation, so no two logical pages are adjacent. Q has the
    post-norm, pre-scaled magnitude of this model's queries, K the post-norm one.
    """

    def __init__(self, rows, ctx, *, page=32, window=WINDOW, seed=0, scale=1.0, splits=None):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx, self.page, self.window = rows, ctx, page, window
        self.splits = window_splits(window) if splits is None else splits
        pages = -(-ctx // page)
        spare = pages + 4
        perm = torch.randperm(spare, generator=g, device=DEV)[:pages].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = torch.randn(2 * spare, G, page, HEAD_DIM, generator=g, device=DEV).to(FP8)
        q = torch.randn(rows, Q_WIDTH, generator=g, device=DEV) * Q_SCALE
        k = torch.randn(rows, KV_WIDTH, generator=g, device=DEV)
        v = torch.randn(rows, KV_WIDTH, generator=g, device=DEV)
        self.qkv = torch.cat((q, k, v), dim=1).to(torch.bfloat16)
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.scale = torch.tensor([scale], dtype=torch.float32, device=DEV)
        # The backend's kv_scale_quant_orig: computed once with the layer, never per step.
        self.inv_scale = 1.0 / self.scale

    def args(self, pool=None, table=None, q_fp8=False, enable_pdl=False):
        return (
            self.qkv,
            self.pool if pool is None else pool,
            self.table if table is None else table,
            self.seq_lens,
            self.scale,
            H,
            G,
            self.window,
            self.splits,
            q_fp8,
            enable_pdl,
        )


def _bytes(pool: torch.Tensor) -> torch.Tensor:
    return pool.view(torch.uint8)


def _assert_pool_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    diff = _bytes(got) != _bytes(want)
    if diff.any():
        first = diff.nonzero()[0].tolist()
        raise AssertionError(
            f"{what}: {int(diff.sum())} cache byte(s) differ, first at {first}: "
            f"{int(_bytes(got)[tuple(first)])} vs {int(_bytes(want)[tuple(first)])}"
        )


#: The built-in step's workspace and multi-CTA counters, allocated once (outside any capture).
_BUILTIN_BUFFERS = {}


def _builtin_buffers():
    if not _BUILTIN_BUFFERS:
        _BUILTIN_BUFFERS["workspace"] = torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)
        _BUILTIN_BUFFERS["counter"] = torch.zeros(4096, dtype=torch.int32, device=DEV)
    return _BUILTIN_BUFFERS["workspace"], _BUILTIN_BUFFERS["counter"]


def _builtin_step(step: _Step, pool: torch.Tensor, fmha: bool = True, pdl: bool = False):
    """The built-in decode step on ``pool``, launched as ``FlashInferTrtllmGenFmha.run_generation``
    launches it: thop's generation preprocess (``applyBiasRopeUpdateKVCacheV2``: the K/V append
    and Q to E4M3), then flashinfer's trtllm-gen decode. Returns the output, or ``None`` with
    ``fmha=False`` (the append only)."""
    import flashinfer.decode as fdec

    rows, ctx = step.rows, step.ctx
    window = step.window if step.window else ctx
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    workspace, counter = _builtin_buffers()
    (q_processed, kv_pool, block_tables, _, bmm1_scale, bmm2_scale, fmha_workspace, _, _,
     max_kv_len, window_left, _) = thop.trtllm_gen_generation_preprocess(
        step.qkv, workspace, step.seq_lens, None, None, step.table.view(1, 1, 2, -1),
        torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64),
        torch.tensor([[0, 0]], dtype=torch.int32), step.scale, step.inv_scale, None, None, None,
        None, 0, 0, H, G, HEAD_DIM, step.page, int(QuantMode.FP8_KV_CACHE), window, window, rows,
        1, rows, ctx, 0, 10000.0, 0, 1.0, 1024, 0, 1.0 / math.sqrt(HEAD_DIM), 1.0, True, rows, 0,
        sm, pool.shape[0], 2, True, False)  # fmt: skip
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
        enable_pdl=pdl,
        backend="trtllm-gen",
        q_len_per_req=rows,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=bmm1_scale.narrow(0, 1, 1),
        multi_ctas_kv_counter_buffer=counter,
    )
    return out


def _fp64_reference(qkv, pool, table, ctx, window, scale=1.0):
    """fp64 attention of BF16 Q (never quantized) over the FP8 K/V the cache holds after the step."""
    rows, page = qkv.shape[0], pool.shape[2]
    keys = torch.arange(ctx, device=DEV)
    raw = _bytes(pool)

    def gather(kv):
        got = raw[table[kv, keys // page].long(), :, keys % page]  # [ctx, G, 64]
        return got.view(FP8).double() / scale

    k, v = gather(0), gather(1)
    q = qkv[:, :Q_WIDTH].double().view(rows, H, HEAD_DIM)
    out = torch.empty(rows, H, HEAD_DIM, dtype=torch.float64, device=DEV)
    group = H // G
    for t in range(rows):
        pos = ctx - rows + t
        lo = max(0, pos - window + 1) if window else 0
        for h in range(G):
            s = q[t, h * group : (h + 1) * group] @ k[lo : pos + 1, h].T / math.sqrt(HEAD_DIM)
            out[t, h * group : (h + 1) * group] = torch.softmax(s, -1) @ v[lo : pos + 1, h]
    return out.reshape(rows, Q_WIDTH)


def _errors(out, ref):
    d = (out.double() - ref).abs()
    return float(d.max()), float(d.mean())


#: E4M3 keeps 3 mantissa bits: a normal value rounds by at most 2^-4 of itself. Below the
#: smallest normal, 2^-6, the spacing is 2^-9, so a value rounds by at most min(itself, 2^-10).
_E4M3_REL, _E4M3_MIN_NORMAL, _E4M3_SUB_HALF_STEP = 2.0**-4, 2.0**-6, 2.0**-10
#: Allowance for either kernel's exp2 and fp32 score rounding, relative to p.
_EXP2_REL = 2.0**-10


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """One BF16 ulp at |x| (x a normal BF16 magnitude; 2^-133 at zero)."""
    _, exponent = torch.frexp(x.double().abs())
    return torch.ldexp(torch.ones_like(x, dtype=torch.float64), exponent - 8)


def _q_fp8_tracking_bound(qkv, pool, table, ctx, window, scale=1.0):
    """Per element, how far the ``q_fp8`` op may sit from the built-in output (fp64).

    Both take the same E4M3 Q (the built-in's own conversion, byte-identical),
    the same E4M3 K/V and an fp32-accumulated S, so they see the same scores.
    They differ in P: the trtllm-gen kernel rounds P to E4M3 for its P.V
    product, the op rounds it to BF16. With p_j = exp(s_j - max_j s_j), the
    E4M3 rounding moves p_j by dp_j <= 2^-4 p_j (normal) or min(p_j, 2^-10)
    (subnormal), plus the exp2 allowance. A kernel that rounds P against a
    smaller maximum (its split's, or a stale running one) or scales P up first
    rounds each p_j by no more than that. If its denominator sums the rounded
    P as well, the output moves by at most
    (sum_j dp_j |V_j| + |O| sum_j dp_j) / (sum_j p_j - sum_j dp_j). The op's own
    BF16 rounding of P adds 2^-9 sum_j p_j |V_j| / sum_j p_j. The two BF16
    output roundings (one ulp together) are added by the caller.
    """
    rows, page = qkv.shape[0], pool.shape[2]
    keys = torch.arange(ctx, device=DEV)
    raw = _bytes(pool)

    def gather(kv):
        return raw[table[kv, keys // page].long(), :, keys % page].view(FP8).double() / scale

    k, v = gather(0), gather(1)
    s_bf16 = torch.tensor([scale], device=DEV).to(torch.bfloat16).float()
    q8 = (qkv[:, :Q_WIDTH].float() * s_bf16).to(torch.bfloat16).float().clamp(-448.0, 448.0)
    q = q8.to(FP8).double().view(rows, H, HEAD_DIM) / scale
    bound = torch.empty(rows, H, HEAD_DIM, dtype=torch.float64, device=DEV)
    group = H // G
    for t in range(rows):
        pos = ctx - rows + t
        lo = max(0, pos - window + 1) if window else 0
        for h in range(G):
            s = q[t, h * group : (h + 1) * group] @ k[lo : pos + 1, h].T / math.sqrt(HEAD_DIM)
            p = torch.exp(s - s.amax(-1, keepdim=True))  # [group, keys]
            dp = (
                torch.where(p >= _E4M3_MIN_NORMAL, _E4M3_REL * p, p.clamp(max=_E4M3_SUB_HALF_STEP))
                + _EXP2_REL * p
            )
            vv = v[lo : pos + 1, h]
            den = p.sum(-1, keepdim=True)
            out = (p @ vv) / den
            weighted = (p @ vv.abs()) / den
            e4m3 = (dp @ vv.abs() + out.abs() * dp.sum(-1, keepdim=True)) / (
                den - dp.sum(-1, keepdim=True)
            )
            bound[t, h * group : (h + 1) * group] = e4m3 + 2.0**-9 * weighted
    return bound.reshape(rows, Q_WIDTH)


# --------------------------------------------------------------------------
# Cells: the op against its own reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """The cache bytes equal the reference's write; the output is within the band below.

    Band: per 64-wide head row, 3 "scaled ulps" (units of 2^-8 times the row's
    largest magnitude) element by element and 0.35 ulp relative RMS, against
    the reference computed in true fp32 (TF32 off).
    Derivation: Q, K and V reach both sides as the same BF16 and E4M3 values,
    and S is an fp32-accumulated product of them in both, so two differences
    remain. (1) Each side rounds its result to BF16 once: an element can land
    one BF16 ulp from the reference's, and one ulp of the row's largest element
    is up to 2 scaled ulps. (2) The op rounds P to BF16 before P.V while its
    denominator sums the unrounded P: element d moves by at most
    2^-9 * sum_j p_j |V_jd| / sum_j p_j, half a scaled ulp wherever that
    weighted magnitude is within the row's largest output, one where it is
    twice it. Elementwise that is 3. In aggregate, (2) is a sum of independent
    roundings of the p_j (the key that sets a running maximum gets p = 1, which
    rounds exactly), which averages out over many keys and does not over a few,
    so the few-key cells set the RMS band (each cell prints its own).
    """
    certifying(op)
    spec = cell.spec
    step = _Step(
        spec["rows"],
        spec["ctx"],
        page=spec["page"],
        window=spec["window"],
        seed=zlib.crc32(cell.why.encode()),
        splits=spec.get("splits"),
    )
    q_fp8 = spec.get("q_fp8", False)
    before = step.qkv.clone()
    ref_pool = step.pool.clone()
    out = op(*step.args(q_fp8=q_fp8))
    assert torch.equal(step.qkv, before), "the QKV operand is read-only"
    ref = op.reference(*step.args(pool=ref_pool, q_fp8=q_fp8))
    _assert_pool_equal(step.pool, ref_pool, "cache after the step")
    got, want = out.reshape(-1, HEAD_DIM).float(), ref.reshape(-1, HEAD_DIM).float()
    scale = want.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    worst = float(((got - want).abs() / scale).max()) / 2.0**-8
    rms = float((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()) / 2.0**-8
    print(f"\n{cell.why}: worst {worst:.2f} scaled ulps, relative RMS {rms:.3f} ulp")
    assert_within_ulp(out.reshape(-1, HEAD_DIM), ref.reshape(-1, HEAD_DIM), 3.0, 0.35)


# --------------------------------------------------------------------------
# The cache write against the built-in append
# --------------------------------------------------------------------------
_APPEND_CASES = [
    (8, 100000, 32, 1.0),
    (8, 131072, 32, 1.0),
    (8, 2100, 32, 1.0),
    (3, 100003, 32, 1.0),
    (1, 100000, 32, 1.0),
    (1, 1, 32, 1.0),
    (8, 40, 32, 1.0),
    (8, 100000, 64, 1.0),
    (8, 100000, 32, 0.37),
]


@pytest.mark.parametrize("rows, ctx, page, scale", _APPEND_CASES)
def test_the_cache_write_is_the_builtin_append(rows, ctx, page, scale) -> None:
    step = _Step(rows, ctx, page=page, seed=rows * 7919 + ctx, scale=scale)
    builtin_pool = step.pool.clone()
    _builtin_step(step, builtin_pool, fmha=False)
    with validating(op):
        op(*step.args())
    _assert_pool_equal(step.pool, builtin_pool, f"rows {rows} ctx {ctx} page {page} scale {scale}")


_SPECIAL = (
    0.0, -0.0, 1.0, -1.0, 448.0, -448.0, 464.0, 480.0, 1000.0, -1e30, float("inf"), float("-inf"),
    float("nan"), -float("nan"), 2.0**-6, 2.0**-7, 2.0**-9, 2.0**-10, 3 * 2.0**-11, 2.0**-12, 1e-30,
    -(2.0**-9), 0.0625 + 2.0**-9, 17.0, 18.0, 19.0,
)  # fmt: skip


@pytest.mark.parametrize("scale", (1.0, 0.37, 3.0))
def test_the_cache_write_is_the_builtin_append_on_special_values(scale) -> None:
    """Saturation, NaN, infinities, E4M3 subnormals and ties, through the BF16 scale multiply."""
    rows = 8
    step = _Step(rows, 100000, seed=5, scale=scale)
    g = torch.Generator(device=DEV).manual_seed(11)
    values = torch.tensor(_SPECIAL, device=DEV)
    picks = torch.randint(0, len(_SPECIAL), (rows, 2 * KV_WIDTH), generator=g, device=DEV)
    step.qkv[:, Q_WIDTH:] = values[picks].to(torch.bfloat16)
    builtin_pool = step.pool.clone()
    _builtin_step(step, builtin_pool, fmha=False)
    op(*step.args())
    _assert_pool_equal(step.pool, builtin_pool, f"special values, scale {scale}")


# --------------------------------------------------------------------------
# The output against the built-in path, both measured from fp64
# --------------------------------------------------------------------------
#: Long contexts, the window edge, contexts shorter than the window, and short contexts
#: where a handful of keys carry the softmax (the last three are where a max-abs budget on
#: ``q_fp8`` would not hold: the max is one element of a softmax over a few keys).
_ERROR_CASES = [
    (8, 100000), (8, 131072), (8, 2100), (8, 2056), (8, 1000), (3, 100000), (1, 100000),
    (3, 3), (8, 9), (4, 40),
]  # fmt: skip


def _check_against_builtin(out, builtin, ref, bound, q_fp8, ctx, what):
    """The error budget against the built-in path, both measured from fp64.

    BF16 Q: no worse than the built-in in max and mean (the built-in quantizes Q
    to E4M3, which dominates its error). ``q_fp8`` takes that same E4M3 Q, so
    its error is the built-in's plus the difference P's rounding makes: mean
    within 1.10x of the built-in's at every context; max within 1.10x where at
    least a window of keys is attended (with fewer keys the max is one element
    of a softmax over a handful of keys, and moves either way with the P
    rounding); and element by element within ``_q_fp8_tracking_bound`` of the
    built-in output, plus one BF16 ulp for the two output roundings.
    """
    b_max, b_mean = _errors(builtin, ref)
    o_max, o_mean = _errors(out, ref)
    diff = (out.double() - builtin.double()).abs()
    limit = bound + _bf16_ulp(torch.maximum(out.double().abs(), builtin.double().abs()))
    used = float((diff / limit).max())
    tiny = 1e-30  # one attended key: both outputs are V itself, and both errors 0
    print(
        f"\n{what} q_fp8 {q_fp8}: op max {o_max:.4g} mean {o_mean:.3g}; builtin max "
        f"{b_max:.4g} mean {b_mean:.3g}; max ratio {o_max / max(b_max, tiny):.3f} mean ratio "
        f"{o_mean / max(b_mean, tiny):.3f}; op-builtin max {float(diff.max()):.4g}, "
        f"{used:.3f} of the E4M3-P bound (bound max {float(bound.max()):.3g})"
    )
    if not q_fp8:
        assert o_max <= b_max and o_mean <= b_mean, (what, o_max, o_mean)
        return
    assert o_mean <= 1.10 * b_mean, (what, o_mean, b_mean)
    if ctx >= WINDOW:
        assert o_max <= 1.10 * b_max, (what, o_max, b_max)
    assert used <= 1.0, (what, used)


@pytest.mark.parametrize("rows, ctx", _ERROR_CASES)
def test_the_output_is_no_farther_from_fp64_than_the_builtin(rows, ctx) -> None:
    step = _Step(rows, ctx, seed=ctx + rows)
    builtin_pool = step.pool.clone()
    builtin = _builtin_step(step, builtin_pool)
    ref = _fp64_reference(step.qkv, builtin_pool, step.table, ctx, WINDOW)
    bound = _q_fp8_tracking_bound(step.qkv, builtin_pool, step.table, ctx, WINDOW)
    for q_fp8 in (False, True):
        pool = step.pool.clone()
        out = op(*step.args(pool=pool, q_fp8=q_fp8))
        _assert_pool_equal(pool, builtin_pool, "cache after the step")
        _check_against_builtin(out, builtin, ref, bound, q_fp8, ctx, f"rows {rows} ctx {ctx}")


# --------------------------------------------------------------------------
# Launch properties
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
def test_a_captured_graph_replays_at_another_context(pdl) -> None:
    """The context length is read on the device: one capture serves every step."""
    captured = _Step(8, 100000, seed=1)
    later = _Step(8, 100013, seed=2)
    pool = later.pool.clone()  # the replay runs on the later step's cache
    static = captured.args(
        pool=pool, table=later.table.clone(), enable_pdl=pdl
    )  # captured at 100000 keys, with the captured step's operand
    static = (static[0].clone(), static[1], static[2], static[3].clone()) + static[4:]
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*static)  # warm-up: compiles both kernels outside the capture
    torch.cuda.current_stream().wait_stream(side)
    pool.copy_(later.pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(*static)
    static[0].copy_(later.qkv)
    static[3].copy_(later.seq_lens)
    pool.copy_(later.pool)
    graph.replay()
    torch.cuda.synchronize()
    eager_pool = later.pool.clone()
    eager = op(*later.args(pool=eager_pool))
    assert torch.equal(graph_out, eager)
    _assert_pool_equal(pool, eager_pool, "cache after the replay")


def test_pages_outside_the_window_are_never_read() -> None:
    """A sliding-window layer's table need not hold valid pages before the window.
    Point every page wholly before it at a page of NaNs: the output must not move."""
    step = _Step(8, 100000, seed=3)
    clean = op(*step.args(pool=step.pool.clone()))
    lo = step.ctx - step.rows - WINDOW + 1
    pool = step.pool.clone()
    poison = pool.shape[0] - 1
    assert poison not in step.table[:, lo // step.page :].tolist()[0]
    _bytes(pool)[poison] = 0x7F  # E4M3 NaN
    table = step.table.clone()
    table[:, : lo // step.page] = poison
    with validating(op):
        poisoned = op(*step.args(pool=pool, table=table))
    assert torch.equal(poisoned, clean)


def test_the_op_is_deterministic() -> None:
    step = _Step(8, 131072, seed=9)
    first = op(*step.args(pool=step.pool.clone()))
    second = op(*step.args(pool=step.pool.clone()))
    assert torch.equal(first, second)


# --------------------------------------------------------------------------
# The call site, in the backend
# --------------------------------------------------------------------------
#: Layer 0 is a sliding-window RoPE layer, layer 1 a full-attention NoPE layer.
SLIDING_LAYER, FULL_LAYER = 0, 1
PAGE = 32


def _attention_module(layer_idx: int):
    """The built-in attention at one tep8 rank's geometry, FP8 KV cache, adopted, with the
    decode FMHA selection the core makes for this device (``attention.select_decode_fmha``)."""
    from tensorrt_llm._torch.configs.musespark_open import MuseSparkOpenConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    config = MuseSparkOpenConfig(
        num_attention_heads=H,
        num_key_value_heads=G,
        moe={"topk": 8, "num_experts": 256, "override_routed_hidden_dim": 4096},
    )
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
    )
    with torch.device(DEV):
        attention = MuseSparkOpenAttention(model_config, layer_idx)
    assert attention.is_sliding == (layer_idx == SLIDING_LAYER)
    assert attention.attention_window_size == (WINDOW if attention.is_sliding else None)
    assert attention.attn.has_fp8_kv_cache
    adopted = target_attention.MuseSparkOpenTepDecodeAttention.adopt(attention)
    layer = types.SimpleNamespace(self_attn=adopted)
    target_attention.select_decode_fmha([layer], torch.cuda.get_device_capability())
    return adopted


@contextlib.contextmanager
def _sequence(rows: int, ctx: int, seed: int):
    """A ``KVCacheManagerV2`` (2 layers, FP8, 32-token pages) holding one sequence of
    ``ctx - rows`` cached keys of random E4M3 values, and the metadata of its decode step."""
    import tensorrt_llm
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    pages = -(-ctx // PAGE) + 1
    mgr = KVCacheManagerV2(
        KvCacheConfig(max_tokens=pages * PAGE),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=G,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=pages * PAGE,
        max_batch_size=1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.FP8,
    )
    try:
        mgr.add_dummy_requests([0], [ctx])
        md = TrtllmAttentionMetadata(
            num_contexts=0,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[ctx - rows]),
            seq_lens=torch.tensor([rows], dtype=torch.int),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=mgr,
            request_ids=[0],
            prompt_lens=[ctx],
        )
        md.prepare()
        g = torch.Generator(device=DEV).manual_seed(seed)
        for layer in (SLIDING_LAYER, FULL_LAYER):
            pool, table = _cache_view(md, layer)
            used = table.flatten().long().unique()
            noise = torch.randn(used.numel(), *pool.shape[1:], generator=g, device=DEV)
            _bytes(pool)[used] = _bytes(noise.to(FP8))
        yield md
    finally:
        mgr.shutdown()


def _cache_view(md, layer_idx: int):
    """The layer's pool and the sequence's [2, pages] block table, as the call site reads them."""
    attention_layer = md.kv_cache_manager.layer_offsets[layer_idx]
    total = md.kv_cache_manager.impl.get_page_index_upper_bound(attention_layer, _key_role())
    pool, tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
        md.host_kv_cache_pool_pointers,
        md.host_kv_cache_pool_mapping,
        md.kv_cache_block_offsets,
        attention_layer,
        G,
        PAGE,
        HEAD_DIM,
        2,
        int(total),
        int(QuantMode.FP8_KV_CACHE),
        0,
        1,
        torch.bfloat16,
    )
    pages = -(-int(md.kv_lens_cuda_runtime[0]) // PAGE)
    return pool, tables[0, :, :pages]


def _key_role():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    return Role.KEY


def _qkv(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(rows, Q_WIDTH, generator=g, device=DEV) * Q_SCALE
    kv = torch.randn(rows, 2 * KV_WIDTH, generator=g, device=DEV)
    return torch.cat((q, kv), dim=1).to(torch.bfloat16)


def _attn_impl(attention, qkv, md, plan):
    from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask

    attention._decode_attention = plan
    try:
        out, out_sf = attention._attn_impl(
            qkv,
            None,
            None,
            md,
            PredefinedAttentionMask.CAUSAL,
            None,
            None,
            attention.attention_window_size,
            None,
        )
    finally:
        attention._decode_attention = None
    assert out_sf is None
    return out


_PLANS = {
    "paged_decode": target_attention.DecodeAttentionPlan(sliding_window=True, q_fp8=False),
    "paged_decode_q8": target_attention.DecodeAttentionPlan(sliding_window=True, q_fp8=True),
}


@pytest.mark.parametrize("plan", sorted(_PLANS))
@pytest.mark.parametrize("rows, ctx", [(8, 5000), (8, 100000), (1, 100000), (8, 1000)])
def test_the_adopted_sliding_layer_runs_the_op_on_a_decode_step(plan, rows, ctx, monkeypatch):
    """Through the backend: the same cache bytes as the built-in step, the output the
    op computes from that cache, and an error vs fp64 within the error budget."""
    attention = _attention_module(SLIDING_LAYER)
    calls = []

    def counted(*args):
        calls.append(args)
        with validating(op):
            return op(*args)

    monkeypatch.setattr(target_attention, "paged_decode_attention", counted)
    qkv = _qkv(rows, seed=ctx)
    with _sequence(rows, ctx, seed=ctx) as md:
        pool, table = _cache_view(md, SLIDING_LAYER)
        before = pool.clone()
        builtin = _attn_impl(attention, qkv, md, None)
        builtin_pool = pool.clone()
        assert not calls
        pool.copy_(before)
        out = _attn_impl(attention, qkv, md, _PLANS[plan])
        assert len(calls) == 1
        _assert_pool_equal(pool, builtin_pool, "cache after the step")
        pool.copy_(before)
        direct = op(
            qkv, pool, table, md.kv_lens_cuda_runtime[:1], attention.attn.kv_scale_orig_quant,
            H, G, WINDOW, window_splits(WINDOW), plan == "paged_decode_q8", False,
        )  # fmt: skip
        assert torch.equal(out, direct), "the call site's op call is the direct one"
        ref = _fp64_reference(qkv, builtin_pool, table, ctx, WINDOW)
        bound = _q_fp8_tracking_bound(qkv, builtin_pool, table, ctx, WINDOW)
        _check_against_builtin(
            out,
            builtin,
            ref,
            bound,
            plan == "paged_decode_q8",
            ctx,
            f"site {plan} rows {rows} ctx {ctx}",
        )


def test_a_step_the_op_does_not_fit_runs_the_builtin(monkeypatch):
    """Nine rows: the built-in attention, the same bits as with no plan."""
    attention = _attention_module(SLIDING_LAYER)
    monkeypatch.setattr(target_attention, "paged_decode_attention", None)
    rows, ctx = 9, 5000
    qkv = _qkv(rows, seed=1)
    with _sequence(rows, ctx, seed=1) as md:
        pool, _ = _cache_view(md, SLIDING_LAYER)
        before = pool.clone()
        builtin = _attn_impl(attention, qkv, md, None)
        pool.copy_(before)
        declined = _attn_impl(attention, qkv, md, _PLANS["paged_decode"])
        assert torch.equal(declined, builtin)


def _trace_kernels(fn):
    """Run ``fn`` under the torch profiler: its result and the CUDA kernel events (name, grid)."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        path = handle.name
    try:
        prof.export_chrome_trace(path)
        with open(path) as handle:
            events = json.load(handle).get("traceEvents", [])
    finally:
        os.unlink(path)
    kernels = [
        (event["name"], tuple(event.get("args", {}).get("grid", ())))
        for event in events
        if event.get("cat") == "kernel"
    ]
    return result, kernels


# --------------------------------------------------------------------------
# Microbenchmark (MODELING_V2_TIMING=1; printed, never gated)
# --------------------------------------------------------------------------
class _ChainLayer:
    """One layer of the benchmark chain: a producer kernel writes the step's fused QKV
    operand (in the served graph a copy kernel does), the attention reads it, and a
    consumer kernel reads the attention output (the served graph's output gate)."""

    def __init__(self, step: _Step) -> None:
        self.step = step
        self.src = step.qkv.clone()
        self.zero = torch.zeros_like(step.qkv)
        g = torch.Generator(device=DEV).manual_seed(step.ctx + step.rows)
        self.gate = torch.randn(step.rows, Q_WIDTH, generator=g, device=DEV).to(torch.bfloat16)
        self.fixed = torch.randn(step.rows, Q_WIDTH, generator=g, device=DEV).to(torch.bfloat16)
        self.y = torch.empty(step.rows, Q_WIDTH, dtype=torch.bfloat16, device=DEV)


def _chain_graph(layers, attention, flush):
    """[flush] then, per layer, producer -> attention (or nothing) -> consumer, captured."""

    def body():
        if flush is not None:
            flush.zero_()
        for layer in layers:
            torch.add(layer.src, layer.zero, out=layer.step.qkv)
            out = layer.fixed if attention is None else attention(layer.step)
            torch.mul(out, layer.gate, out=layer.y)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()  # compiles and allocates outside the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def _elapsed_us(graph, replays=30):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _graph_kernels(graph):
    """Kernel names of one replay (memset and memcpy nodes are not kernels)."""
    _, kernels = _trace_kernels(graph.replay)
    return [name for name, _ in kernels]


def _bench_in_chain(steps, arms, cold, rounds=6):
    """Marginal us per layer of each arm = (graph time - chain-only graph time) / layers,
    with the arm order alternating between rounds; and each arm's kernels per call,
    counted as the kernels its graph launches beyond the chain-only graph's (no name
    filter)."""
    layers = [_ChainLayer(step) for step in steps]
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=DEV) if cold else None
    graphs = {name: _chain_graph(layers, fn, flush) for name, fn in arms.items()}
    graphs["chain"] = _chain_graph(layers, None, flush)
    times = {name: [] for name in graphs}
    names = list(graphs)
    for r in range(rounds):
        for name in names if r % 2 == 0 else names[::-1]:
            times[name].append(_elapsed_us(graphs[name]))

    def marginal(name, picked):
        return (
            statistics.median(picked(times[name])) - statistics.median(picked(times["chain"]))
        ) / len(steps)

    chain_kernels = _graph_kernels(graphs["chain"])
    result = {}
    for name in arms:
        extra = _graph_kernels(graphs[name])
        for kernel in chain_kernels:
            extra.remove(kernel)
        per_call = {}
        for kernel in extra:
            per_call[kernel[:48]] = per_call.get(kernel[:48], 0) + 1 / len(steps)
        result[name] = (
            marginal(name, lambda v: v),
            marginal(name, lambda v: v[0::2]),
            marginal(name, lambda v: v[1::2]),
            len(extra) / len(steps),
            per_call,
        )
    return result


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@pytest.mark.parametrize("ctx", (32768, 100000, 131072))
@pytest.mark.parametrize("rows", (8, 1))
def test_microbenchmark(rows, ctx) -> None:
    """Per layer, in a producer -> attention -> consumer chain, PDL on (as with
    TRTLLM_ENABLE_PDL=1), cold (a 1 GiB memset heads the graph and every layer reads its own
    pool) and warm: the 46 sliding-window layers built-in vs the paged decode op with BF16 and
    with E4M3 Q, the 16 full-attention layers built-in."""
    swa = [_Step(rows, ctx, seed=100 * i + rows) for i in range(46)]
    swa_arms = {
        "builtin": lambda s: _builtin_step(s, s.pool, pdl=True),
        "paged_decode": lambda s: op(*s.args(enable_pdl=True)),
        "paged_decode_q8": lambda s: op(*s.args(q_fp8=True, enable_pdl=True)),
    }
    full = [_Step(rows, ctx, window=0, seed=100 * i + rows + 1) for i in range(16)]
    full_arms = {
        "builtin": lambda s: _builtin_step(s, s.pool, pdl=True),
    }
    for kind, steps, arms in (("swa", swa, swa_arms), ("full", full, full_arms)):
        for cold in (True, False):
            result = _bench_in_chain(steps, arms, cold)
            for name, (us, fwd, rev, launches, kernels) in result.items():
                print(
                    f"\nBENCH {kind} rows {rows} ctx {ctx} {'cold' if cold else 'warm'} {name}: "
                    f"{us:.2f} us/layer (orders {fwd:.2f} / {rev:.2f}), {launches:.0f} kernels "
                    f"per call {kernels}"
                )
