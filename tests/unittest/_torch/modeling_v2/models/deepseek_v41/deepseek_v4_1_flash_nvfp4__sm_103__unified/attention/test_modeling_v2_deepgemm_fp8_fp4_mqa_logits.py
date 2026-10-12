# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the deepgemm_fp8_fp4_mqa_logits catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate.
This file turns a cell's spec into tensors and keeps the checks a cell cannot carry: the exact
regime's bit-identity premise, the output stride and allocation, rows past 2^31 bytes, determinism,
empty ranges and extents, CUDA-graph replay, the loud rejections, the wrong-variant controls, the
row and key axes as ranges, the engine envelope's partitions and the row-tiled long-context form.

Two input regimes, one reference:

* **exact** -- random packed bytes (E2M1 codes, UE8M0 scale bytes near 127, weights ``n * 2^-10``)
  chosen so every head term ``relu(q.k) * w`` and every partial sum of them is exactly
  representable in fp32. The op must then be bit-identical to the reference in any accumulation
  order; each case asserts that premise before it asserts equality.
* **recipe** -- bf16 values packed with the kernel's own recipe (DeepGEMM ``per_token_cast_to_fp4``,
  gran_k 32, packed UE8M0) and fp32 ``randn`` weights; compared at default fp32 tolerance.

The reference decodes the packed bytes with torch ops (even element in the low nibble, scale byte
``b`` of the int32 word = block ``b``) and computes ``sum_h relu(q_h . k) * w_h`` in float64. The
entry's ``reference`` is that computation, and the cells, ``_check_exact``, the recipe sweeps and
the controls gate through ``op.reference`` / ``op.compare``. ``_reference`` below is its float64
twin on already-decoded operands: it also returns the visibility mask and the exact-regime premise
bound, takes the wrong-variant flags the controls need, and serves the row-sliced checks (rows
past 2^31 bytes, the engine-envelope partitions); ``_long_context_reference`` decodes keys in chunks for the
2^25-key long-context form.
"""

import collections
import math
import random
import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.deepgemm_fp8_fp4_mqa_logits import (  # noqa: E501
    deepgemm_fp8_fp4_mqa_logits as op,
)
from tensorrt_llm.deep_gemm.utils import per_token_cast_to_fp4

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "deepgemm_fp8_fp4_mqa_logits requires a CUDA device"

DEV = "cuda"
NEG_INF = float("-inf")
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# V4.1 weights fold: weights_proj(x) * index_head_dim^-0.5 * index_n_heads^-0.5, with the GLOBAL
# head count (32) even when a rank holds fewer heads.
GLOBAL_FOLD = 128**-0.5 * 32**-0.5  # == 2^-6 exactly
LOCAL_FOLD_TP4 = 128**-0.5 * 8**-0.5


# ----------------------------------------------------------------------------- reference helpers


def _decode(packed, sf, swap_nibbles=False, reverse_sf_bytes=False):
    """int8 [..., 64] + int32 [...] -> float64 [..., 128]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    p = packed.view(torch.uint8).long()
    lo, hi = lut[p & 0xF], lut[p >> 4]
    if swap_nibbles:
        lo, hi = hi, lo
    v = torch.stack([lo, hi], dim=-1).flatten(-2)
    e = sf.contiguous().view(torch.uint8).view(*sf.shape, 4).long() - 127
    if reverse_sf_bytes:
        e = e.flip(-1)
    scale = ((e + 1023) << 52).view(torch.float64)  # 2^e built from the exponent bits: exact
    return (v.unflatten(-1, (4, 32)) * scale.unsqueeze(-1)).flatten(-2)


def _reference(qd, kd, w, ks, ke, relu_after_sum=False):
    """float64 logits [T, S] (-inf outside [ks, ke)), the visibility mask, and max_t,s sum_h |term|."""
    T, S = qd.shape[0], kd.shape[0]
    out = torch.empty(T, S, dtype=torch.float64, device=DEV)
    abs_sum_max = 0.0
    wd = w.double()
    for t0 in range(0, T, 512):
        dots = torch.einsum("thd,kd->thk", qd[t0 : t0 + 512], kd)
        wt = wd[t0 : t0 + 512].unsqueeze(-1)
        if relu_after_sum:
            out[t0 : t0 + 512] = (dots * wt).sum(1).clamp_min(0)
        else:
            terms = dots.clamp_min(0) * wt
            out[t0 : t0 + 512] = terms.sum(1)
            if S > 0:
                abs_sum_max = max(abs_sum_max, float(terms.abs().sum(1).max()))
    col = torch.arange(S, device=DEV)
    vis = (col[None, :] >= ks[:, None].long()) & (col[None, :] < ke[:, None].long())
    return out.masked_fill(~vis, NEG_INF), vis, abs_sum_max


# ----------------------------------------------------------------------------- input builders


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _rand_packed(rows, g, exp_lo=126, exp_hi=128):
    """Random E2M1 byte pairs (every code, both signs) + random UE8M0 bytes in [exp_lo, exp_hi]."""
    codes = torch.randint(0, 256, (rows, 64), generator=g, device=DEV, dtype=torch.int32)
    sfb = torch.randint(exp_lo, exp_hi + 1, (rows, 4), generator=g, device=DEV, dtype=torch.int32)
    return codes.to(torch.uint8).view(torch.int8), sfb.to(torch.uint8).view(torch.int32).view(rows)


def _exact_inputs(T, S, H, seed):
    """Exact regime: |dot| < 2^15 in units of 2^-4, w = n * 2^-10 with |n| <= 8 (<= 3 significant bits)."""
    g = _gen(seed)
    q, q_sf = _rand_packed(T * H, g)
    kv, kv_sf = _rand_packed(S, g)
    w_raw = torch.randint(-8, 9, (T, H), generator=g, device=DEV).float() * 2.0**-4
    return q.view(T, H, 64), q_sf.view(T, H), kv, kv_sf, w_raw


def _assert_exact_premise(abs_sum_max):
    # Terms are multiples of 2^-14 (dot unit 2^-4 times weight unit 2^-10), each <= 22 significant
    # bits. If sum_h |term| < 2^10 every partial sum in any order is an fp32-exact multiple of 2^-14.
    assert abs_sum_max < 2.0**10, f"exact-regime premise broken: max sum|term| = {abs_sum_max}"


def _causal(T, r, past=0, k_off=0):
    pos = torch.arange(past, past + T, device=DEV)
    ks = torch.full((T,), k_off, dtype=torch.int32, device=DEV)
    ke = (k_off + (pos + 1) // r).to(torch.int32)
    return ks, ke


def _call(q, q_sf, kv, kv_sf, w, ks, ke, clean=True, max_seqlen_k=0, logits_dtype=torch.float32):
    return op(q, q_sf, kv, kv_sf, w, ks, ke, clean, max_seqlen_k, logits_dtype)


def _row_stride(S, logits_dtype=torch.float32, block_kv=256):
    align = 1024 // torch.empty((), dtype=logits_dtype).element_size()
    return -(-(S + block_kv) // align) * align


def _check_exact(out, q, q_sf, kv, kv_sf, w, ks, ke):
    """The entry's gate, then bit-identity: the exact regime's premise makes the op exact."""
    _, vis, bound = _reference(_decode(q, q_sf), _decode(kv, kv_sf), w, ks, ke)
    _assert_exact_premise(bound)
    ref = op.reference(q, q_sf, kv, kv_sf, w, ks, ke, True, 0, torch.float32)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    op.compare(out, ref)
    assert torch.equal(out, ref), f"{int((out != ref).sum())} of {out.numel()} elements differ"
    return ref, vis


# ----------------------------------------------------------------------------- exact regime


def _packed(reqs, r):
    """(past, new) per request -> per-query ks/ke with each request's keys at its own offset, and S."""
    ks_l, ke_l, S = [], [], 0
    for past, n in reqs:
        ks_r, ke_r = _causal(n, r, past=past, k_off=S)
        ks_l.append(ks_r)
        ke_l.append(ke_r)
        S += (past + n) // r
    return torch.cat(ks_l), torch.cat(ke_l), S


def _logits_bytes(T, S, H):
    """The call's one allocation: rows rounded up to the 128 / H query block, row stride align(S+256, 256)."""
    return -(-T // (128 // H)) * (128 // H) * _row_stride(S) * 4


def test_exact_output_past_2gib():
    """T 16384 x S 32768: rows from 16257 on start past 2^31 bytes of the logits buffer (2.16 GB).
    The first and last 512 rows are checked; the rest is the same kernel at smaller offsets."""
    H, T, r = 32, 16384, 1
    ks, ke, S = _packed([(16384, T)], r)
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=99)
    w = w_raw * GLOBAL_FOLD
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    assert out.untyped_storage().nbytes() == _logits_bytes(T, S, H) > 2**31
    kd = _decode(kv, kv_sf)
    for rows in (slice(0, 512), slice(T - 512, T)):
        ref, _, bound = _reference(_decode(q[rows], q_sf[rows]), kd, w[rows], ks[rows], ke[rows])
        _assert_exact_premise(bound)
        assert torch.equal(out[rows], ref.float())


# ----------------------------------------------------------------------------- recipe regime


def _recipe_inputs(T, S, H, seed):
    g = _gen(seed)
    q = torch.randn(T * H, 128, generator=g, device=DEV).bfloat16()
    k = torch.randn(S, 128, generator=g, device=DEV).bfloat16()
    q_p, q_s = per_token_cast_to_fp4(q, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    k_p, k_s = per_token_cast_to_fp4(k, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    w = torch.randn(T, H, generator=g, device=DEV) * GLOBAL_FOLD
    return q_p.view(T, H, 64), q_s.view(T, H), k_p, k_s.view(S), w


def _wide_scale_inputs(T, S, H, seed):
    """UE8M0 bytes across [90, 160]; per-token / per-key exponents so fp32 stays in range."""
    g = _gen(seed)
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=seed)
    tq = torch.randint(90, 159, (T, 1, 1), generator=g, device=DEV, dtype=torch.int32)
    tk = torch.randint(90, 159, (S, 1), generator=g, device=DEV, dtype=torch.int32)
    jq = torch.randint(0, 3, (T, H, 4), generator=g, device=DEV, dtype=torch.int32)
    jk = torch.randint(0, 3, (S, 4), generator=g, device=DEV, dtype=torch.int32)
    q_sf = (tq + jq).to(torch.uint8).view(torch.int32).view(T, H)
    kv_sf = (tk + jk).to(torch.uint8).view(torch.int32).view(S)
    return q, q_sf, kv, kv_sf, w_raw * GLOBAL_FOLD


def _cell_inputs(spec, seed):
    """A cell's tensors and output form -> (q, q_sf, kv, kv_sf, w, ks, ke, clean, max_seqlen_k)."""
    H = spec["H"]
    clean, msk = True, 0
    if "reqs" in spec:
        # per-request (cached, new), each request's keys at its own offset, causal inside
        ks, ke, S = _packed(spec["reqs"], spec["r"])
        T = ks.numel()
    else:
        T, S = spec["T"], spec["S"]
        t = torch.arange(T, device=DEV)
        if spec["ranges"] == "shifted":  # clean_logits=False over shifted per-row ranges
            ks = (t % 17).to(torch.int32)
            ke = (ks + 1 + t * 2 % (S - 17)).to(torch.int32)
            clean = False
        else:  # "compressed": shifted ranges including empty rows
            ks = (t * 3).to(torch.int32)
            ke = (ks + (t % 29) + (t % 2) * 300).clamp(max=S).to(torch.int32)
            clean, msk = False, int((ke - ks).max())
    if spec["regime"] == "recipe":
        q, q_sf, kv, kv_sf, w = _recipe_inputs(T, S, H, seed)
    elif spec["regime"] == "wide_scale":
        q, q_sf, kv, kv_sf, w = _wide_scale_inputs(T, S, H, seed)
    else:
        q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed)
        w = w_raw * GLOBAL_FOLD
    return q, q_sf, kv, kv_sf, w, ks, ke, clean, msk


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    H = spec["H"]
    q, q_sf, kv, kv_sf, w, ks, ke, clean, msk = _cell_inputs(spec, zlib.crc32(cell.why.encode()))
    T, S = q.shape[0], kv.shape[0]
    if spec.get("alloc_check"):
        # The first call in a process also allocates a one-time 32 MiB buffer that stays allocated
        # (measured); a warm-up call keeps the measurement below independent of test order.
        op(q[:1], q_sf[:1], kv, kv_sf, w[:1], ks[:1], ke[:1], clean, msk, torch.float32)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    out = op(q, q_sf, kv, kv_sf, w, ks, ke, clean, msk, torch.float32)
    torch.cuda.synchronize()
    if spec.get("alloc_check"):
        # the call allocates exactly the returned logits buffer, no other workspace
        assert out.untyped_storage().nbytes() == _logits_bytes(T, S, H)
        assert torch.cuda.max_memory_allocated() - before == _logits_bytes(T, S, H)
    assert out.dtype == torch.float32 and out.shape == (T, msk or S)
    assert out.stride() == ((-(-msk // 256) * 256 if msk else _row_stride(S)), 1)
    ref = op.reference(q, q_sf, kv, kv_sf, w, ks, ke, clean, msk, torch.float32)
    op.compare(out, ref)
    if spec["regime"] == "exact":
        # Every head term and partial sum is an fp32-exact multiple of 2^-14: the op is
        # bit-identical to the reference on every element it specifies.
        _assert_exact_premise(_reference(_decode(q, q_sf), _decode(kv, kv_sf), w, ks, ke)[2])
        specified = ~torch.isnan(ref)
        assert torch.equal(out[specified], ref[specified])


def test_deterministic():
    q, q_sf, kv, kv_sf, w = _recipe_inputs(1024, 2048, 32, seed=3)
    ks, ke = _causal(1024, 2, past=3072)
    first = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    for _ in range(10):
        assert torch.equal(_call(q, q_sf, kv, kv_sf, w, ks, ke), first)


# ----------------------------------------------------------------------------- output forms


def test_compressed_form_rejects_clean_logits():
    T, S, H = 200, 1000, 32
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=6)
    w = w_raw * GLOBAL_FOLD
    t = torch.arange(T, device=DEV)
    ks = (t * 3).to(torch.int32)
    ke = (ks + (t % 29) + (t % 2) * 300).clamp(max=S).to(torch.int32)
    with pytest.raises(RuntimeError, match="not clean_logits"):
        _call(q, q_sf, kv, kv_sf, w, ks, ke, clean=True, max_seqlen_k=int((ke - ks).max()))


def test_empty_rows_and_empty_extents():
    T, S, H = 16, 16, 32
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=7)
    w = w_raw * GLOBAL_FOLD
    ks = torch.full((T,), 4, dtype=torch.int32, device=DEV)
    ke = torch.where(torch.arange(T, device=DEV) % 2 == 0, 4, 9).to(torch.int32)
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    _check_exact(out, q, q_sf, kv, kv_sf, w, ks, ke)
    assert bool(torch.isneginf(out[0::2]).all())
    # T == 0 or S == 0 cannot build the kernel's TMA descriptors; the error is not sticky.
    with pytest.raises(RuntimeError, match="CUDA_ERROR_INVALID_VALUE"):
        _call(q[:0], q_sf[:0], kv, kv_sf, w[:0], ks[:0], ke[:0])
    with pytest.raises(RuntimeError, match="CUDA_ERROR_INVALID_VALUE"):
        _call(q, q_sf, kv[:0], kv_sf[:0], w, torch.zeros_like(ks), torch.zeros_like(ke))
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    torch.cuda.synchronize()
    _check_exact(out, q, q_sf, kv, kv_sf, w, ks, ke)


def test_cuda_graph_replay():
    T, S, H = 256, 512, 32
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=8)
    w = w_raw * GLOBAL_FOLD
    ks, ke = _causal(T, 2, past=S * 2 - T)
    _call(q, q_sf, kv, kv_sf, w, ks, ke)  # JIT-compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    for seed in (9, 10):
        q2, q_sf2, kv2, kv_sf2, w_raw2 = _exact_inputs(T, S, H, seed=seed)
        ke2 = (ke - seed).clamp(min=0)
        for dst, src in (
            (q, q2),
            (q_sf, q_sf2),
            (kv, kv2),
            (kv_sf, kv_sf2),
            (w, w_raw2 * GLOBAL_FOLD),
            (ke, ke2),
        ):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        _check_exact(out, q2, q_sf2, kv2, kv_sf2, w_raw2 * GLOBAL_FOLD, ks, ke2)


# ----------------------------------------------------------------------------- loud rejections


def test_rejections():
    T, S, H = 8, 16, 32
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=12)
    w = w_raw * GLOBAL_FOLD
    ks, ke = _causal(T, 1, past=S - T)
    bad = {
        "q_fp.scalar_type()": dict(q=q.view(torch.uint8)),
        "head_dim == _head_dim": dict(kv=kv.view(torch.uint8)),
        "t.dim() == N": dict(kv_sf=kv_sf.view(S, 1)),
        "seq_len == _seq_len and num_heads == _num_heads": dict(q_sf=q_sf.view(T * H, 1)),
        "q_fp.is_contiguous()": dict(q=torch.cat([q, q], 2)[:, :, :64]),
        "kv_fp.is_contiguous()": dict(kv=torch.cat([kv, kv], 1)[:, :64]),
        "cu_seq_len_k_start.scalar_type()": dict(ks=ks.long()),
        "weights.scalar_type() != torch::kBFloat16 or logits_dtype == torch::kBFloat16": dict(
            w=w.bfloat16()
        ),
    }
    args = dict(q=q, q_sf=q_sf, kv=kv, kv_sf=kv_sf, w=w, ks=ks, ke=ke)
    for msg, override in bad.items():
        a = {**args, **override}
        with pytest.raises(RuntimeError, match=msg.replace("(", r"\(").replace(")", r"\)")):
            _call(a["q"], a["q_sf"], a["kv"], a["kv_sf"], a["w"], a["ks"], a["ke"])
    # Head counts: any multiple of 4 up to 128 is taken (24 and 128 are measured against the
    # reference in test_head_counts_off_the_cells_match_the_reference); 6 is refused.
    q2, q_sf2, _, _, w2 = _exact_inputs(T, S, 6, seed=13)
    with pytest.raises(RuntimeError, match=r"num_heads % 4 == 0"):
        _call(q2, q_sf2, kv, kv_sf, w2, ks, ke)


@pytest.mark.parametrize("heads", [24, 128])
def test_head_counts_off_the_cells_match_the_reference(heads):
    """The kernel accepts any head count that is a multiple of 4 up to 128 (128 / H query rows per
    block). Two such counts outside the cells, one that does not divide 128 and the largest, are
    held to the entry's gate on a causal ratio-1 step. Not a certification of these counts: a
    measurement that the acceptance is not a silent wrong answer."""
    T, S = 64, 256
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, heads, seed=15 + heads)
    w = w_raw * GLOBAL_FOLD
    ks, ke = _causal(T, 1, past=S - T)
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    op.compare(out, op.reference(q, q_sf, kv, kv_sf, w, ks, ke, True, 0, torch.float32))


# ----------------------------------------------------------------------------- wrong-variant controls


def test_wrong_variants_fail_the_gate():
    """The entry's gate catches each of these mistakes (counts printed for the receipt notes)."""
    T, S, H, r = 512, 256, 32, 2
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=14)
    w = w_raw * GLOBAL_FOLD
    ks, ke = _causal(T, r)
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    qd, kd = _decode(q, q_sf), _decode(kv, kv_sf)
    _, vis, _ = _reference(qd, kd, w, ks, ke)
    good = op.reference(q, q_sf, kv, kv_sf, w, ks, ke, True, 0, torch.float32)
    op.compare(out, good)
    assert torch.equal(out, good)
    controls = {
        "relu after the head sum": _reference(qd, kd, w, ks, ke, relu_after_sum=True)[0],
        "local-head weight fold (8 heads)": _reference(qd, kd, w_raw * LOCAL_FOLD_TP4, ks, ke)[0],
        # Swapping the nibble order of BOTH operands permutes both vectors alike and leaves every dot
        # product unchanged (measured: 0 differences), so the control swaps k only.
        "k nibbles swapped": _reference(qd, _decode(kv, kv_sf, swap_nibbles=True), w, ks, ke)[0],
        "scale bytes reversed": _reference(
            _decode(q, q_sf, reverse_sf_bytes=True),
            _decode(kv, kv_sf, reverse_sf_bytes=True),
            w,
            ks,
            ke,
        )[0],
        "causal end + 1": _reference(qd, kd, w, ks, ke + 1)[0],
    }
    for name, wrong in controls.items():
        wrong32 = wrong.float()
        n_diff = int((out != wrong32).sum())
        n_tol = int(
            (
                ~torch.isclose(out, wrong32, rtol=1.3e-6, atol=1e-5, equal_nan=False)
                & ~(torch.isneginf(out) & torch.isneginf(wrong32))
            ).sum()
        )
        print(
            f"control [{name}]: {n_diff} of {out.numel()} elements differ bitwise, "
            f"{n_tol} outside default fp32 tolerance ({int(vis.sum())} visible)"
        )
        assert n_tol > 0
        with pytest.raises(AssertionError):
            op.compare(out, wrong32)


def test_wrong_variants_fail_the_gate_long_context():
    """At S 32576 (packed, r = 1) the gate sees the mistakes a long key range can hide: a kernel that
    stops at the previously certified S 16384, one whose key row index wraps at 2^14, and one that
    ignores the per-request start. Checked on the last 1024 query rows (all in the third request,
    whose keys lie at rows 23480 .. 32575)."""
    H, r = 32, 1
    ks, ke, S = _packed([(16384, 4096), (0, 3000), (8000, 1096)], r)
    T = ks.numel()
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, H, seed=15)
    w = w_raw * GLOBAL_FOLD
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    rows = slice(T - 1024, T)
    qd, kd = _decode(q[rows], q_sf[rows]), _decode(kv, kv_sf)
    wr, ksr, ker = w[rows], ks[rows], ke[rows]
    _, vis, bound = _reference(qd, kd, wr, ksr, ker)
    _assert_exact_premise(bound)
    got = out[rows]
    good = op.reference(q[rows], q_sf[rows], kv, kv_sf, wr, ksr, ker, True, 0, torch.float32)
    op.compare(got, good)
    assert torch.equal(got, good)
    wrapped = kd[torch.arange(S, device=DEV) % 2**14]
    controls = {
        "keys past 16384 dropped": _reference(qd, kd, wr, ksr, ker.clamp(max=16384))[0],
        "key row index wrapped at 2^14": _reference(qd, wrapped, wr, ksr, ker)[0],
        "per-request start ignored (ks = 0)": _reference(qd, kd, wr, torch.zeros_like(ksr), ker)[0],
    }
    for name, wrong in controls.items():
        wrong32 = wrong.float()
        n_tol = int(
            (
                ~torch.isclose(got, wrong32, rtol=1.3e-6, atol=1e-5)
                & ~(torch.isneginf(got) & torch.isneginf(wrong32))
            ).sum()
        )
        print(
            f"control [{name}]: {n_tol} of {got.numel()} elements outside default fp32 tolerance "
            f"({int(vis.sum())} visible)"
        )
        assert n_tol > 0
        with pytest.raises(AssertionError):
            op.compare(got, wrong32)


# ----------------------------------------------------------------------------- row and key axes
#
# The target passes any context chunk T in 1..8192 (max_num_tokens) over any compressed key count S in
# 1..32768. These tests certify both axes as ranges, not as sampled points: a maximal run against the
# float64 reference, then every smaller extent bitwise equal to the matching slice of that run, plus
# one compiled kernel pair serving every extent (profiler kernel names).
#
# Harness sighting. Each sweep alternates two input sets with the same ranges (A for even extents, B for
# odd), walks extents downwards so each result fits in the block the previous one freed, and fills every
# returned buffer with NaN before freeing it. An element the op left unwritten therefore reads NaN or the
# other set's value, never a stale copy of the right one; each sweep asserts that most results landed on
# a NaN-filled block, and runs a one-row (or one-key) shift control that must fail.

_HEADS = 32
_S_MAX = 32768
# The one kernel a full-width (clean_logits=True) call launches at every extent: the logits kernel,
# which also writes -inf to the columns outside each row's range (no separate clean pass).
_LOGITS_KERNEL = "void deep_gemm::sm100_mqa_logits<32u, 128u, true, false, true, false, 4u, 256u,"
# Every S = 0, 1, 2, 3 or -1 mod the kernel's 256-key tile up to 32768. That includes S = 0, 1, -1 mod
# 8192 columns and every residue mod 4 (the key-scale TMA extent is rounded to 4).
_S_EDGES = sorted(
    {k * 256 + d for k in range(129) for d in (-1, 0, 1, 2, 3) if 1 <= k * 256 + d <= _S_MAX}
)


def _chunk(T, S, r):
    """One request's context chunk whose last row sees key S - 1: ks = 0, ke = (p + 1) // r for the
    absolute positions p = S*r - T .. S*r - 1, clamped to [0, S] (a row before the first compressed key
    sees none). This is the target's single-request call form."""
    pos = torch.arange(S * r - T, S * r, device=DEV)
    ke = torch.div(pos + 1, r, rounding_mode="floor").clamp(0, S).to(torch.int32)
    return torch.zeros(T, dtype=torch.int32, device=DEV), ke


def _packed3(T, S):
    """Three requests packed as the target packs a context step: each request's keys at its own offset
    (the second request starts at an odd key), causal inside each request, rows before a request's first
    key see none."""
    rows = (T // 2, T // 4 + 1, T - T // 2 - T // 4 - 1)
    s0 = min(S, (S * 5 // 8) | 1)
    s1 = (S - s0) // 3
    ks_l, ke_l, off = [], [], 0
    for n, s in zip(rows, (s0, s1, S - s0 - s1)):
        t = torch.arange(n, device=DEV)
        ks_l.append(torch.full((n,), off, dtype=torch.int32, device=DEV))
        ke_l.append((off + (s - n + t + 1).clamp(0, s)).to(torch.int32))
        off += s
    return torch.cat(ks_l), torch.cat(ke_l)


def _axis_ranges(form, T, S):
    return _packed3(T, S) if form == "packed" else _chunk(T, S, int(form[1]))


def _axis_full(regime, T, S, ks, ke, seed):
    """Inputs of one set plus the maximal run, checked against the float64 reference."""
    if regime == "exact":
        q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, _HEADS, seed)
        w = w_raw * GLOBAL_FOLD
    else:
        q, q_sf, kv, kv_sf, w = _recipe_inputs(T, S, _HEADS, seed)
    full = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    if regime == "exact":
        _check_exact(full, q, q_sf, kv, kv_sf, w, ks, ke)
    else:
        op.compare(full, op.reference(q, q_sf, kv, kv_sf, w, ks, ke, True, 0, torch.float32))
    return q, q_sf, kv, kv_sf, w, full


def _poison(out):
    """NaN over the whole storage of a returned logits buffer (the padded rows and columns included)."""
    out.as_strided((out.untyped_storage().nbytes() // 4,), (1,), 0).fill_(float("nan"))


@pytest.mark.parametrize(
    "regime,form",
    [("exact", "r1"), ("exact", "r2"), ("exact", "packed"), ("recipe", "packed")],
)
@pytest.mark.parametrize("T_max,S", [(8192, 16384), (4096, 32768)])
def test_row_axis_every_T(T_max, S, regime, form):
    """Every T in 1..T_max: the first T rows of q, q_sf, weights, ks and ke over the same keys give
    exactly the first T rows of the T_max call, which itself matches float64 (exact regime: bit-identical;
    recipe regime: default fp32 tolerance, and the prefix rows are still bitwise equal)."""
    ks, ke = _axis_ranges(form, T_max, S)
    sets = [
        _axis_full(regime, T_max, S, ks, ke, seed=T_max + S + 7 * k + len(form)) for k in (0, 1)
    ]
    torch.cuda.synchronize()
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    shift = {}
    poisoned, reused = set(), 0
    for T in range(T_max, 0, -1):
        q, q_sf, kv, kv_sf, w, full = sets[T % 2]
        out = _call(q[:T], q_sf[:T], kv, kv_sf, w[:T], ks[:T], ke[:T])
        assert out.shape == (T, S) and out.stride() == (_row_stride(S), 1)
        reused += out.data_ptr() in poisoned
        bad += (out != full[:T]).sum()
        if T in (1, 5, 128, 129, T_max // 2 + 1, T_max - 1):
            shift[T] = (out != full[1 : T + 1]).sum()  # control: the gate must see a one-row shift
        _poison(out)
        poisoned.add(out.data_ptr())
        del out
    shift = {T: int(n) for T, n in shift.items()}
    print(
        f"row axis {regime}/{form} T 1..{T_max} x S {S}: {int(bad)} mismatching elements; "
        f"{reused}/{T_max} results on a NaN-filled block; one-row-shift control mismatches {shift}"
    )
    assert int(bad) == 0
    assert reused >= T_max // 2
    assert all(n > 0 for n in shift.values())


@pytest.mark.parametrize(
    "regime,form,T,every_S",
    [
        ("exact", "r1", 8192, False),
        ("exact", "r2", 8192, False),
        ("exact", "packed", 8192, False),
        ("recipe", "packed", 8192, False),
        ("exact", "r1", 67, True),
        ("exact", "r2", 67, True),
        ("exact", "packed", 67, True),
    ],
)
def test_key_axis(regime, form, T, every_S):
    """At fixed T, S in _S_EDGES (T 8192) or every S in 1..32768 (T 67, which is not a multiple of the
    4-row query block): the first S key rows with the ranges of this form at S give, on every visible
    element, exactly the T x 32768 call in which every row sees every key, and -inf everywhere else."""
    ks0 = torch.zeros(T, dtype=torch.int32, device=DEV)
    ke0 = torch.full((T,), _S_MAX, dtype=torch.int32, device=DEV)
    sets = [_axis_full(regime, T, _S_MAX, ks0, ke0, seed=T + 3 * k + len(form)) for k in (0, 1)]
    torch.cuda.synchronize()
    s_values = range(_S_MAX, 0, -1) if every_S else _S_EDGES[::-1]
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    n_vis = torch.zeros((), dtype=torch.int64, device=DEV)
    shift = {}
    poisoned, reused = set(), 0
    for S in s_values:
        q, q_sf, kv, kv_sf, w, full = sets[S % 2]
        ks, ke = _axis_ranges(form, T, S)
        out = _call(q, q_sf, kv[:S], kv_sf[:S], w, ks, ke)
        assert out.shape == (T, S) and out.stride() == (_row_stride(S), 1)
        reused += out.data_ptr() in poisoned
        col = torch.arange(S, device=DEV)
        vis = (col[None, :] >= ks[:, None]) & (col[None, :] < ke[:, None])
        bad += ((out != full[:, :S]) & vis).sum() + (~torch.isneginf(out) & ~vis).sum()
        n_vis += vis.sum()
        if S in (255, 256, 257, 8191, 8193, 32767):
            shift[S] = (
                (out != full[:, 1 : S + 1]) & vis
            ).sum()  # control: a one-key shift must fail
        _poison(out)
        poisoned.add(out.data_ptr())
        del out
    shift = {S: int(n) for S, n in shift.items()}
    print(
        f"key axis {regime}/{form} T {T}, {len(s_values)} values of S: {int(bad)} mismatching elements "
        f"({int(n_vis)} visible checked); {reused}/{len(s_values)} results on a NaN-filled block; "
        f"one-key-shift control mismatches {shift}"
    )
    assert int(bad) == 0
    assert reused >= len(s_values) // 2
    assert all(n > 0 for n in shift.values())


# Every (T, S, ratio) of the core target's runtime-value dump (boot and accuracy gates), and the 24K
# passkey prompt's three chunks (8192, 8192, 7674 tokens; the last one is in the dump) at both ratios.
_DUMPED = [
    (1, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (14, 8, 2),
    (14, 16, 1),
    (16, 8, 2),
    (16, 16, 1),
    (48, 24, 2),
    (48, 48, 1),
    (888, 1469, 2),
    (888, 2938, 1),
    (890, 1469, 2),
    (890, 2938, 1),
    (1016, 1469, 2),
    (1016, 2938, 1),
    (1934, 969, 2),
    (1934, 1938, 1),
    (2048, 1024, 2),
    (2048, 1025, 2),
    (2048, 2048, 1),
    (2048, 2050, 1),
    (8192, 8192, 1),
    (8192, 4096, 2),
    (8192, 16384, 1),
    (8192, 8192, 2),
    (7674, 24058, 1),
    (7674, 12029, 2),
]


@pytest.mark.parametrize("T,S,r", _DUMPED)
def test_dumped_values(T, S, r):
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, _HEADS, seed=7 * T + S + r)
    w = w_raw * GLOBAL_FOLD
    ks, ke = _chunk(T, S, r)
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    _check_exact(out, q, q_sf, kv, kv_sf, w, ks, ke)
    assert out.stride() == (_row_stride(S), 1)


def test_one_compiled_variant():
    """T and S are runtime arguments only: across T 1..8192 and S 1..32768, every form, one kernel runs
    per call -- the logits kernel, which also cleans the out-of-range columns -- with the same template
    arguments at every extent."""
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(8192, _S_MAX, _HEADS, seed=41)
    w = w_raw * GLOBAL_FOLD
    ts = (1, 2, 3, 4, 5, 7, 8, 9, 67, 127, 128, 129, 1283, 4095, 4096, 4097, 7674, 8191, 8192)
    ss = (1, 2, 3, 4, 5, 255, 256, 257, 969, 8191, 8192, 8193, 16384, 24058, 32767, 32768)
    cases = [(T, S, _axis_ranges(f, T, S)) for T in ts for S in ss for f in ("r1", "r2", "packed")]
    _call(
        q[:4], q_sf[:4], kv, kv_sf, w[:4], *_chunk(4, _S_MAX, 1)
    )  # JIT, if any, outside the profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for T, S, (ks, ke) in cases:
            _call(q[:T], q_sf[:T], kv[:S], kv_sf[:S], w[:T], ks, ke)
        torch.cuda.synchronize()
    names = collections.Counter(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )
    for name, n in names.items():
        print(f"kernel x{n}: {name}")
    assert set(names.values()) == {len(cases)}, names
    assert len(names) == 1, names
    (name,) = names
    assert name.startswith(_LOGITS_KERNEL), name


# ----------------------------------------------------------------------------- multi-request steps: the engine envelope
#
# One engine step passes T <= 8192 context rows from k <= 32 context requests, each with cached + new <= 16384
# tokens. The metadata (``_prepare_v41_ctx_index_k``) gathers every request's compressed keys, cached ones
# included, request after request, so S = sum_b kv_len_b // r reaches 524288 (r 1) / 262144 (r 2), and
# gives request b's token at position p the range [cu_b, cu_b + (p + 1) // r) with cu_b = sum_{j<b}
# kv_len_j // r. ``_packed`` builds exactly that from (cached, new) per request.

_ENV_T, _ENV_K, _ENV_LEN = 8192, 32, 16384


def _bounds(reqs):
    """First row and row count of each request."""
    out, row = [], 0
    for _, n in reqs:
        out.append((row, n))
        row += n
    return out


def _check_partition(out, q, q_sf, kv, kv_sf, w, ks, ke, bounds):
    """Every element of every row. Per request, its rows over its own key block [ks of its first row,
    ke of its last row) equal the float64 reference bitwise (-inf outside each row's range); and every row's
    count of non-(-inf) elements over all S columns equals ke - ks, so everything outside the request's
    block is -inf (a NaN would count). Returns the visible elements checked; AssertionError on a mismatch."""
    T, S = out.shape
    ksl, kel = ks.tolist(), ke.tolist()
    n_vis = 0
    for row0, n in bounds:
        off, end = ksl[row0], kel[row0 + n - 1]
        rows = slice(row0, row0 + n)
        ref, vis, bound = _reference(
            _decode(q[rows], q_sf[rows]),
            _decode(kv[off:end], kv_sf[off:end]),
            w[rows],
            ks[rows] - off,
            ke[rows] - off,
        )
        _assert_exact_premise(bound)
        assert torch.equal(out[rows, off:end], ref.float()), f"request rows {row0}..{row0 + n - 1}"
        n_vis += int(vis.sum())
        del ref, vis
    for t0 in range(0, T, 1024):
        fin = (~torch.isneginf(out[t0 : t0 + 1024])).sum(1)
        assert torch.equal(fin, (ke[t0 : t0 + 1024] - ks[t0 : t0 + 1024]).long()), (
            f"rows {t0}.. outside range"
        )
    return n_vis


def _run_partition(reqs, r, seed):
    """The target's call for one partition, checked in full; None when S == 0 (the target skips the call)."""
    ks, ke, S = _packed(reqs, r)
    if S == 0:
        return None
    T = ks.numel()
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, _HEADS, seed)
    w = w_raw * GLOBAL_FOLD
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    assert out.shape == (T, S) and out.stride() == (_row_stride(S), 1)
    return T, S, _check_partition(out, q, q_sf, kv, kv_sf, w, ks, ke, _bounds(reqs))


def _random_partition(rng):
    """T in 1..8192 (half uniform, half log-uniform) split at random cuts into k in 1..min(32, T) requests
    in random order; each request's cached prefix is 0 (a fresh prompt) a third of the time, else uniform
    in 0..16384 - new."""
    T = rng.randint(1, _ENV_T) if rng.random() < 0.5 else min(_ENV_T, int(2 ** rng.uniform(0, 13)))
    k = rng.randint(1, min(_ENV_K, T))
    cuts = sorted(rng.sample(range(1, T), k - 1))
    new = [b - a for a, b in zip([0] + cuts, cuts + [T])]
    rng.shuffle(new)
    return [(0 if rng.random() < 1 / 3 else rng.randint(0, _ENV_LEN - n), n) for n in new]


_N_RANDOM, _RANDOM_BLOCKS = 200, 4


@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.parametrize("block", range(_RANDOM_BLOCKS))
def test_multi_request_random_partitions(block, r):
    """200 seeded random partitions (50 per block), each run at ratio 1 and at ratio 2, every element
    checked against float64."""
    per = _N_RANDOM // _RANDOM_BLOCKS
    stats = collections.Counter()
    for i in range(block * per, (block + 1) * per):
        reqs = _random_partition(random.Random(1000 + i))
        assert 1 <= len(reqs) <= _ENV_K and 1 <= sum(n for _, n in reqs) <= _ENV_T
        assert all(n >= 1 and c + n <= _ENV_LEN for c, n in reqs)
        res = _run_partition(reqs, r, seed=5000 + 7 * i + r)
        if res is None:
            stats["skipped (S 0)"] += 1
            continue
        T, S, n_vis = res
        stats["partitions"] += 1
        stats["rows"] += T
        stats["visible checked"] += n_vis
        stats["max S"] = max(stats["max S"], S)
        stats["max k"] = max(stats["max k"], len(reqs))
    print(f"random partitions block {block} r {r}: {dict(stats)}")
    assert stats["partitions"] + stats["skipped (S 0)"] == per


def test_multi_request_wrong_ranges_are_seen():
    """The partition check must fail when one request's range is shifted by one key, or when two requests'
    key blocks are swapped (the ranges built as if request 2's keys preceded request 1's)."""
    reqs = [(3000, 700), (0, 257), (9000, 1043)]
    r = 1
    ks, ke, S = _packed(reqs, r)
    T = ks.numel()
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, _HEADS, seed=77)
    w = w_raw * GLOBAL_FOLD
    out = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    bounds = _bounds(reqs)
    _check_partition(out, q, q_sf, kv, kv_sf, w, ks, ke, bounds)
    shift = torch.zeros_like(ks)
    shift[:700] = 1
    # Swapped: the same rows, but the ranges built as if request 2's keys preceded request 1's.
    spans = [(c + n) // r for c, n in reqs]
    offs = {0: 0, 2: spans[0], 1: spans[0] + spans[2]}
    sw = [_causal(n, r, past=c, k_off=offs[b]) for b, (c, n) in enumerate(reqs)]
    swapped_ks, swapped_ke = torch.cat([a for a, _ in sw]), torch.cat([e for _, e in sw])
    wrong = {
        "request 0 range shifted by one key": (ks + shift, ke + shift),
        "requests 1 and 2 key blocks swapped": (swapped_ks, swapped_ke),
    }
    for name, (wks, wke) in wrong.items():
        with pytest.raises(AssertionError):
            _check_partition(out, q, q_sf, kv, kv_sf, w, wks, wke, bounds)
        print(f"control [{name}]: the partition check fails")


def _maximal(T, k, r):
    """k requests sharing T new rows as evenly as possible, each cached up to 16384 total."""
    new = [T // k + (i < T % k) for i in range(k)]
    return [(_ENV_LEN - n, n) for n in new]


@pytest.mark.parametrize("r", [1, 2])
def test_multi_request_maximal_cell(r):
    """T 8192 in 32 requests of 16384 tokens each: S 524288 at r 1 (a 17.2 GB logits buffer whose last rows
    start past element 2^32) and 262144 at r 2. Every request block checked in full."""
    T, S, n_vis = _run_partition(_maximal(_ENV_T, _ENV_K, r), r, seed=31 + r)
    assert (T, S) == (_ENV_T, _ENV_K * _ENV_LEN // r)
    assert _row_stride(S) * (T - 1) > 2**32 or r == 2
    print(f"maximal cell r {r}: T {T} x S {S}, {n_vis} visible elements checked")


def test_multi_request_kernel_variant_map():
    """Kernel names over the envelope: T in 1..8192, k in 1..32 requests, cached prefix 0 or up to 16384,
    ratios 1 and 2, S up to 524288. Every call launches the same one kernel as the row and key axes'
    variant map (``test_one_compiled_variant``)."""
    ts, ks_ = (1, 5, 129, 4097, 8192), (1, 2, 17, 32)
    cells = []
    for T in ts:
        for k in ks_:
            if k > T:
                continue
            new = [T // k + (i < T % k) for i in range(k)]
            for cached in ("none", "full"):
                reqs = [((_ENV_LEN - n) if cached == "full" else 0, n) for n in new]
                for r in (1, 2):
                    ks, ke, S = _packed(reqs, r)
                    if S:
                        cells.append((T, S, ks, ke))
    S_max = max(S for _, S, _, _ in cells)
    assert S_max == _ENV_K * _ENV_LEN
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(_ENV_T, S_max, _HEADS, seed=43)
    w = w_raw * GLOBAL_FOLD
    _call(
        q[:4], q_sf[:4], kv, kv_sf, w[:4], *_chunk(4, S_max, 1)
    )  # JIT, if any, outside the profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for T, S, ks, ke in cells:
            out = _call(q[:T], q_sf[:T], kv[:S], kv_sf[:S], w[:T], ks, ke)
            del out
        torch.cuda.synchronize()
    names = collections.Counter(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )
    for name, n in names.items():
        print(f"kernel x{n}: {name}")
    print(f"variant map: {len(cells)} calls, S {min(S for _, S, _, _ in cells)}..{S_max}")
    assert set(names.values()) == {len(cells)}, names
    assert len(names) == 1, names
    (name,) = names
    assert name.startswith(_LOGITS_KERNEL), name


# Every distinct (tc, S, r) of the mqa_logits records in the core target's runtime-value dumps (boot and
# accuracy gates; 1928 cells, none outside the envelope), as "tc.S.r" when one request carries all tc rows,
# else "tc.S.r.nc.max_q_len" joined from a trtllm_sparse_mla_dsv4.varlen record with the same tc.
# ``_dumped_partition`` turns each into a consistent partition.
_DUMPED_CELLS = """
1.1.1 2.1.2 2.2.1 14.8.2 14.16.1 16.8.2 16.16.1 48.24.2 48.48.1 493.342.2 493.685.1 494.336.2 494.672.1
503.379.2 503.758.1 529.356.2 529.712.1 549.274.2 549.349.2 549.549.1 549.698.1 550.436.2 550.872.1 555.379.2
555.759.1 556.368.2 556.737.1 562.281.2 562.562.1 569.422.2 569.844.1 571.407.2 571.814.1 573.383.2 573.767.1
578.289.2 578.578.1 579.357.2 579.715.1 581.290.2 581.581.1 586.293.2 586.382.2 586.586.1 586.764.1 590.295.2
590.417.2 590.590.1 590.835.1 593.362.2 593.725.1 594.377.2 594.754.1 595.430.2 595.861.1 597.298.2 597.597.1
598.406.2 598.813.1 599.299.2 599.599.1 601.368.2 601.375.2 601.736.1 601.750.1 603.369.2 603.739.1 606.303.2
606.606.1 609.304.2 609.609.1 612.306.2 612.612.1 614.374.2 614.749.1 616.308.2 616.616.1 617.376.2 617.753.1
619.309.2 619.619.1 623.311.2 623.623.1 625.312.2 625.625.1 626.313.2 626.626.1 627.313.2 627.627.1 629.314.2
629.402.2 629.629.1 629.804.1 630.315.2 630.630.1 631.315.2 631.631.1 632.316.2 632.632.1 634.317.2 634.430.2
634.634.1 634.861.1 635.317.2 635.635.1 637.318.2 637.637.1 640.320.2 640.640.1 641.386.2 641.773.1 642.387.2
642.774.1 643.321.2 643.643.1 644.322.2 644.644.1 646.323.2 646.418.2 646.646.1 646.837.1 649.324.2 649.649.1
650.325.2 650.650.1 651.325.2 651.651.1 652.326.2 652.393.2 652.652.1 652.787.1 653.326.2 653.653.1 654.327.2
654.654.1 656.328.2 656.656.1 660.330.2 660.660.1 661.330.2 661.396.2 661.661.1 661.793.1 662.331.2 662.420.2
662.662.1 662.841.1 663.331.2 663.663.1 665.332.2 665.457.2 665.665.1 665.915.1 666.333.2 666.666.1 667.333.2
667.412.2 667.667.1 667.825.1 668.334.2 668.668.1 670.335.2 670.427.2 670.670.1 670.854.1 672.336.2 672.672.1
674.337.2 674.674.1 675.337.2 675.412.2 675.675.1 675.824.1 676.338.2 676.676.1 679.339.2 679.679.1 680.340.2
680.680.1 681.340.2 681.681.1 682.435.2 682.445.2 682.871.1 682.890.1 683.341.2 683.683.1 684.342.2 684.684.1
685.342.2 685.685.1 687.343.2 687.687.1 688.344.2 688.445.2 688.688.1 688.891.1 689.344.2 689.689.1 690.345.2
690.429.2 690.690.1 690.858.1 691.345.2 691.691.1 692.346.2 692.692.1 693.414.2 693.828.1 694.347.2 694.694.1
696.348.2 696.696.1 697.348.2 697.697.1 698.349.2 698.698.1 699.349.2 699.699.1 700.350.2 700.700.1 701.350.2
701.701.1 702.351.2 702.702.1 703.351.2 703.703.1 704.352.2 704.704.1 706.353.2 706.706.1 708.452.2 708.905.1
710.355.2 710.710.1 711.355.2 711.711.1 712.356.2 712.423.2 712.712.1 712.847.1 713.356.2 713.713.1 714.357.2
714.714.1 715.357.2 715.715.1 718.359.2 718.718.1 719.359.2 719.719.1 720.360.2 720.478.2 720.720.1 720.956.1
721.360.2 721.721.1 722.361.2 722.722.1 723.361.2 723.723.1 724.362.2 724.487.2 724.724.1 724.975.1 725.362.2
725.725.1 726.363.2 726.726.1 727.363.2 727.727.1 728.364.2 728.728.1 729.364.2 729.729.1 730.365.2 730.730.1
732.366.2 732.732.1 733.366.2 733.493.2 733.733.1 733.986.1 734.367.2 734.458.2 734.734.1 734.916.1 735.367.2
735.735.1 736.368.2 736.736.1 737.368.2 737.466.2 737.737.1 737.932.1 738.369.2 738.738.1 739.369.2 739.739.1
740.370.2 740.740.1 741.370.2 741.741.1 742.371.2 742.742.1 743.371.2 743.743.1 745.372.2 745.440.2 745.745.1
745.880.1 746.373.2 746.746.1 747.373.2 747.747.1 748.374.2 748.748.1 749.374.2 749.749.1 751.375.2 751.751.1
752.376.2 752.752.1 753.376.2 753.753.1 754.377.2 754.754.1 755.377.2 755.755.1 756.378.2 756.493.2 756.756.1
756.987.1 757.378.2 757.757.1 758.379.2 758.758.1 759.379.2 759.759.1 760.380.2 760.760.1 761.380.2 761.448.2
761.761.1 761.897.1 762.381.2 762.762.1 763.381.2 763.763.1 764.382.2 764.764.1 765.382.2 765.765.1 766.383.2
766.766.1 767.383.2 767.767.1 768.384.2 768.768.1 769.384.2 769.769.1 770.385.2 770.770.1 771.385.2 771.771.1
772.386.2 772.476.2 772.772.1 772.952.1 773.386.2 773.773.1 774.387.2 774.774.1 775.387.2 775.775.1 776.388.2
776.776.1 777.388.2 777.777.1 778.389.2 778.778.1 779.389.2 779.779.1 780.390.2 780.493.2 780.780.1 780.986.1
781.390.2 781.781.1 782.391.2 782.782.1 783.391.2 783.783.1 784.392.2 784.459.2 784.784.1 784.918.1 785.392.2
785.785.1 786.393.2 786.786.1 787.393.2 787.787.1 788.394.2 788.788.1 790.395.2 790.790.1 791.395.2 791.791.1
792.396.2 792.792.1 793.396.2 793.793.1 794.397.2 794.794.1 795.397.2 795.795.1 796.398.2 796.796.1 797.398.2
797.797.1 798.399.2 798.605.2 798.798.1 798.1211.1 799.399.2 799.799.1 800.400.2 800.800.1 801.400.2 801.801.1
802.401.2 802.471.2 802.802.1 802.943.1 803.401.2 803.803.1 804.402.2 804.804.1 805.402.2 805.805.1 806.403.2
806.806.1 807.403.2 807.807.1 808.404.2 808.808.1 809.404.2 809.809.1 810.405.2 810.810.1 811.405.2 811.811.1
812.406.2 812.491.2 812.812.1 812.982.1 813.406.2 813.499.2 813.813.1 813.999.1 814.407.2 814.814.1 815.407.2
815.815.1 816.408.2 816.816.1 817.408.2 817.817.1 818.409.2 818.818.1 819.409.2 819.819.1 820.410.2 820.820.1
822.411.2 822.822.1 823.411.2 823.823.1 824.412.2 824.824.1 825.412.2 825.493.2 825.825.1 825.986.1 826.413.2
826.826.1 827.413.2 827.827.1 828.414.2 828.828.1 829.414.2 829.829.1 830.415.2 830.830.1 831.415.2 831.831.1
832.416.2 832.832.1 833.416.2 833.833.1 834.417.2 834.834.1 835.417.2 835.835.1 836.418.2 836.836.1 837.418.2
837.519.2 837.837.1 837.1039.1 838.419.2 838.838.1 839.419.2 839.839.1 840.420.2 840.840.1 841.420.2 841.841.1
843.421.2 843.843.1 844.422.2 844.844.1 845.422.2 845.845.1 846.423.2 846.846.1 847.423.2 847.847.1 848.424.2
848.848.1 849.424.2 849.849.1 850.425.2 850.850.1 851.425.2 851.851.1 852.426.2 852.852.1 853.426.2 853.567.2
853.853.1 853.1135.1 854.427.2 854.511.2 854.580.2 854.854.1 854.1023.1 854.1160.1 855.427.2 855.855.1 856.428.2
856.856.1 857.428.2 857.857.1 858.429.2 858.530.2 858.858.1 858.1060.1 859.429.2 859.859.1 860.430.2 860.860.1
861.430.2 861.861.1 862.431.2 862.522.2 862.862.1 862.1045.1 863.431.2 863.863.1 864.432.2 864.864.1 866.433.2
866.866.1 867.433.2 867.867.1 868.434.2 868.868.1 869.434.2 869.869.1 870.435.2 870.870.1 871.435.2 871.871.1
872.436.2 872.872.1 873.436.2 873.873.1 874.437.2 874.874.1 875.437.2 875.875.1 876.438.2 876.876.1 877.438.2
877.877.1 878.439.2 878.878.1 879.439.2 879.879.1 880.440.2 880.880.1 881.440.2 881.881.1 882.441.2 882.882.1
883.441.2 883.883.1 884.442.2 884.884.1 885.442.2 885.885.1 886.443.2 886.886.1 887.443.2 887.887.1 888.444.2
888.888.1 888.1469.2 888.2938.1 889.444.2 889.889.1 890.445.2 890.890.1 890.1469.2 890.2938.1 891.445.2
891.891.1 892.446.2 892.892.1 893.446.2 893.893.1 894.447.2 894.894.1 895.447.2 895.895.1 896.448.2 896.896.1
897.448.2 897.897.1 898.449.2 898.898.1 899.449.2 899.899.1 900.450.2 900.900.1 901.450.2 901.901.1 902.451.2
902.902.1 903.451.2 903.903.1 904.452.2 904.904.1 906.453.2 906.906.1 907.453.2 907.907.1 908.454.2 908.908.1
909.454.2 909.909.1 910.455.2 910.910.1 911.455.2 911.911.1 912.456.2 912.912.1 913.456.2 913.913.1 914.457.2
914.914.1 915.457.2 915.556.2 915.915.1 915.1112.1 916.458.2 916.916.1 917.458.2 917.917.1 918.459.2 918.918.1
919.459.2 919.919.1 920.460.2 920.920.1 921.460.2 921.547.2 921.921.1 921.1094.1 922.461.2 922.922.1 923.461.2
923.923.1 924.462.2 924.924.1 925.462.2 925.925.1 926.463.2 926.926.1 927.463.2 927.927.1 928.464.2 928.928.1
929.464.2 929.929.1 930.465.2 930.930.1 931.465.2 931.931.1 932.466.2 932.932.1 933.466.2 933.933.1 934.467.2
934.934.1 935.467.2 935.935.1 936.468.2 936.936.1 937.468.2 937.937.1 938.469.2 938.938.1 939.469.2 939.939.1
940.470.2 940.940.1 941.470.2 941.941.1 942.471.2 942.942.1 943.471.2 943.943.1 944.472.2 944.944.1 945.472.2
945.945.1 946.473.2 946.946.1 947.473.2 947.947.1 949.474.2 949.949.1 950.475.2 950.950.1 951.475.2 951.951.1
952.476.2 952.952.1 953.476.2 953.953.1 954.477.2 954.954.1 955.477.2 955.955.1 956.478.2 956.956.1 957.478.2
957.957.1 958.479.2 958.958.1 959.479.2 959.959.1 960.480.2 960.960.1 961.480.2 961.961.1 962.481.2 962.962.1
963.481.2 963.963.1 964.482.2 964.964.1 965.482.2 965.965.1 966.483.2 966.966.1 967.483.2 967.967.1 968.484.2
968.968.1 969.484.2 969.969.1 970.485.2 970.970.1 972.486.2 972.972.1 973.486.2 973.571.2 973.973.1 973.1142.1
974.487.2 974.974.1 975.487.2 975.975.1 976.488.2 976.976.1 977.488.2 977.977.1 979.489.2 979.979.1 980.490.2
980.980.1 982.491.2 982.982.1 983.491.2 983.983.1 984.492.2 984.984.1 986.493.2 986.986.1 987.493.2 987.987.1
988.494.2 988.988.1 989.494.2 989.989.1 990.495.2 990.990.1 991.495.2 991.991.1 992.496.2 992.575.2 992.992.1
992.1150.1 993.496.2 993.993.1 994.497.2 994.994.1 996.498.2 996.996.1 997.498.2 997.997.1 998.499.2 998.998.1
999.499.2 999.999.1 1000.500.2 1000.1000.1 1002.501.2 1002.1002.1 1003.501.2 1003.1003.1 1004.502.2 1004.1004.1
1005.502.2 1005.1005.1 1007.503.2 1007.1007.1 1008.504.2 1008.1008.1 1011.505.2 1011.1011.1 1012.506.2
1012.580.2 1012.1012.1 1012.1161.1 1013.506.2 1013.1013.1 1014.507.2 1014.1014.1 1015.507.2 1015.1015.1
1016.508.2 1016.1016.1 1016.1469.2 1016.2938.1 1018.509.2 1018.1018.1 1019.509.2 1019.1019.1 1020.510.2
1020.1020.1 1021.510.2 1021.1021.1 1022.511.2 1022.1022.1 1023.511.2 1023.1023.1 1025.512.2 1025.1025.1
1026.513.2 1026.1026.1 1028.514.2 1028.1028.1 1031.515.2 1031.1031.1 1032.516.2 1032.1032.1 1034.517.2
1034.1034.1 1036.518.2 1036.1036.1 1038.519.2 1038.1038.1 1039.519.2 1039.1039.1 1040.520.2 1040.1040.1
1042.521.2 1042.1042.1 1043.521.2 1043.1043.1 1045.522.2 1045.1045.1 1046.523.2 1046.1046.1 1048.524.2
1048.1048.1 1050.525.2 1050.1050.1 1055.527.2 1055.1055.1 1057.528.2 1057.1057.1 1058.529.2 1058.1058.1
1062.531.2 1062.1062.1 1064.532.2 1064.1064.1 1065.532.2 1065.1065.1 1066.533.2 1066.1066.1 1067.533.2
1067.1067.1 1068.534.2 1068.1068.1 1070.535.2 1070.1070.1 1072.536.2 1072.1072.1 1076.538.2 1076.1076.1
1079.539.2 1079.1079.1 1080.540.2 1080.1080.1 1081.540.2 1081.1081.1 1082.541.2 1082.1082.1 1084.542.2
1084.1084.1 1085.542.2 1085.1085.1 1086.543.2 1086.1086.1 1087.543.2 1087.1087.1 1089.544.2 1089.1089.1
1090.545.2 1090.1090.1 1091.545.2 1091.1091.1 1092.546.2 1092.1092.1 1093.546.2 1093.1093.1 1094.547.2
1094.1094.1 1095.547.2 1095.1095.1 1098.549.2 1098.1098.1 1099.549.2 1099.1099.1 1100.550.2 1100.1100.1
1101.550.2 1101.1101.1 1104.552.2 1104.1104.1 1106.553.2 1106.1106.1 1109.554.2 1109.1109.1 1111.555.2
1111.1111.1 1115.557.2 1115.1115.1 1118.559.2 1118.1118.1 1119.559.2 1119.1119.1 1120.560.2 1120.1120.1
1122.561.2 1122.1122.1 1125.562.2 1125.1125.1 1126.563.2 1126.1126.1 1128.564.2 1128.1128.1 1129.564.2
1129.1129.1 1130.565.2 1130.1130.1 1135.567.2 1135.1135.1 1136.568.2 1136.1136.1 1137.568.2 1137.1137.1
1139.569.2 1139.1139.1 1140.570.2 1140.1140.1 1141.570.2 1141.1141.1 1142.571.2 1142.1142.1 1143.571.2
1143.1143.1 1145.572.2 1145.1145.1 1146.573.2 1146.1146.1 1149.574.2 1149.1149.1 1150.575.2 1150.1150.1
1154.577.2 1154.1154.1 1157.578.2 1157.1157.1 1160.580.2 1160.1160.1 1170.585.2 1170.1170.1 1171.585.2
1171.1171.1 1172.586.2 1172.1172.1 1177.588.2 1177.1177.1 1178.589.2 1178.1178.1 1182.591.2 1182.1182.1
1185.592.2 1185.1185.1 1187.593.2 1187.1187.1 1198.599.2 1198.1198.1 1200.600.2 1200.1200.1 1201.600.2
1201.1201.1 1204.602.2 1204.1204.1 1211.605.2 1211.1211.1 1217.608.2 1217.1217.1 1226.613.2 1226.1226.1
1236.618.2 1236.1236.1 1237.618.2 1237.1237.1 1241.620.2.2.629 1241.1241.1.2.629 1261.630.2 1261.1261.1
1273.636.2 1273.1273.1 1275.637.2.2.697 1275.1275.1.2.697 1283.769.2.2.780 1283.1538.1.2.780 1298.649.2
1298.1298.1 1301.650.2 1301.1301.1 1313.656.2.2.764 1313.1313.1.2.764 1333.666.2.2.670 1333.1333.1.2.670
1344.672.2.2.698 1344.1344.1.2.698 1345.672.2.2.746 1345.1345.1.2.746 1355.830.2.2.698 1355.1661.1.2.698
1374.687.2.2.768 1374.1374.1.2.768 1388.693.2.2.757 1388.1388.1.2.757 1391.695.2.2.730 1391.1391.1.2.730
1395.850.2.2.738 1395.1701.1.2.738 1397.698.2.2.811 1397.1397.1.2.811 1403.701.2.2.740 1403.1403.1.2.740
1407.778.2.2.858 1407.1556.1.2.858 1408.704.2.2.738 1408.1408.1.2.738 1410.806.2 1410.1614.1.2.855
1419.709.2.2.723 1423.711.2.2.734 1423.1423.1.2.734 1424.712.2.2.730 1424.1424.1.2.730 1432.805.2.2.938
1432.1610.1.2.938 1433.716.2.2.787 1433.1433.1.2.787 1439.719.2.2.729 1439.1439.1.2.729 1448.724.2.2.756
1448.1448.1.2.756 1462.731.2.2.774 1462.1462.1.2.774 1464.731.2.2.753 1464.836.2.2.782 1464.1464.1.2.782
1464.1672.1.2.782 1466.798.2 1466.1598.1.2.825 1467.733.2.2.877 1467.1467.1.2.877 1472.736.2.2.742
1473.736.2.2.777 1473.1473.1.2.777 1477.738.2.2.792 1477.1477.1.2.792 1481.740.2.2.777 1481.1481.1.2.777
1482.741.2.2.788 1482.1482.1.2.788 1484.741.2.2.835 1484.1484.1.2.835 1494.746.2.2.895 1494.1494.1.2.895
1496.814.2.2.854 1496.1628.1.2.854 1497.748.2.2.818 1497.1497.1.2.818 1498.748.2.2.779 1498.1498.1.2.779
1501.750.2 1501.1501.1 1505.752.2.2.882 1505.1505.1.2.882 1507.753.2.2.839 1507.1507.1.2.839 1508.753.2.2.847
1508.1508.1.2.847 1510.755.2.2.820 1510.1510.1.2.820 1511.755.2.2.780 1511.1511.1.2.780 1512.756.2.2.784
1512.1512.1.2.784 1513.756.2.2.788 1513.1513.1.2.788 1514.843.2.2.844 1514.849.2.2.844 1514.1687.1.2.759
1514.1698.1.2.759 1518.759.2.2.776 1518.1518.1.2.776 1519.824.2.2.793 1519.1648.1.2.793 1520.760.2.2.786
1520.1520.1.2.786 1523.761.2.2.832 1523.1523.1.2.832 1524.761.2.2.825 1524.1524.1.2.825 1525.762.2.2.831
1525.859.2.2.831 1525.1525.1.2.831 1525.1719.1.2.831 1528.763.2.2.771 1528.1528.1.2.771 1533.766.2.2.780
1533.1533.1.2.780 1535.767.2.2.848 1535.1535.1.2.848 1537.768.2.2.839 1537.1537.1.2.839 1538.768.2.2.775
1538.1538.1.2.775 1539.769.2.2.859 1539.1539.1.2.859 1542.770.2.2.799 1542.1542.1.2.799 1543.771.2.2.854
1543.1543.1.2.854 1546.772.2.2.863 1546.773.2.2.906 1546.1546.1.2.863 1547.773.2.2.784 1547.881.2.2.784
1547.1547.1.2.784 1547.1762.1.2.784 1550.774.2.2.815 1550.1550.1.2.815 1551.775.2.2.809 1551.1551.1.2.809
1554.776.2.2.799 1554.777.2.2.850 1554.1554.1.2.799 1561.780.2.2.801 1561.1561.1.2.801 1562.780.2.2.841
1562.872.2.2.828 1562.1562.1.2.828 1562.1744.1.2.828 1566.783.2.2.784 1566.1566.1.2.784 1567.783.2.2.911
1567.1567.1.2.911 1568.783.2.2.815 1568.1568.1.2.815 1569.784.2.2.888 1569.1569.1.2.888 1570.784.2.2.851
1570.1570.1.2.851 1571.785.2.2.831 1571.1571.1.2.831 1574.786.2.2.811 1574.787.2.2.834 1574.1574.1.2.834
1575.787.2.2.848 1575.1575.1.2.848 1579.789.2.2.796 1579.1579.1.2.796 1580.790.2.2.854 1580.1580.1.2.854
1583.866.2.2.1034 1583.1732.1.2.1034 1584.791.2.2.799 1584.792.2.2.826 1584.1584.1.2.799 1585.792.2.2.807
1585.1585.1.2.807 1591.795.2.2.804 1591.1591.1.2.804 1593.796.2.2.862 1593.1593.1.2.862 1594.796.2.2.801
1594.797.2.2.918 1594.1594.1.2.918 1595.797.2.2.848 1595.1595.1.2.848 1596.798.2.2.860 1596.1596.1.2.860
1601.800.2.2.909 1601.1601.1.2.909 1605.802.2.2.824 1605.1605.1.2.824 1606.802.2.2.847 1606.803.2.2.914
1606.1606.1.2.847 1608.803.2.2.873 1608.1608.1.2.873 1610.804.2.2.853 1610.1610.1.2.853 1614.807.2.2.820
1614.1614.1.2.820 1615.807.2.2.858 1615.1615.1.2.858 1616.882.2 1616.1765.1.2.941 1620.810.2.2.878
1620.1620.1.2.878 1621.810.2.2.967 1621.1621.1.2.967 1624.812.2.2.866 1624.1624.1.2.866 1625.812.2.2.869
1625.1625.1.2.869 1626.813.2.2.920 1626.1626.1.2.920 1627.813.2.2.834 1627.1627.1.2.834 1629.814.2.2.825
1629.1629.1.2.825 1630.882.2 1630.1765.1.2.889 1636.817.2.2.883 1636.1636.1.2.883 1637.818.2.2.822
1637.1637.1.2.822 1639.1639.1.2.954 1640.820.2.2.914 1640.1640.1.2.914 1641.820.2.2.868 1641.1641.1.2.868
1642.820.2.2.1005 1642.1642.1.2.1005 1644.821.2.2.869 1644.1644.1.2.869 1646.822.2.2.873 1646.1646.1.2.873
1647.823.2.2.887 1647.1647.1.2.887 1649.824.2.2.871 1649.1649.1.2.871 1654.827.2.2.924 1654.1654.1.2.924
1655.827.2.2.964 1655.1655.1.2.964 1656.827.2.2.845 1656.1656.1.2.845 1661.830.2.2.1064 1661.1661.1.2.1064
1662.830.2.2.833 1662.1662.1.2.833 1663.831.2.2.866 1663.896.2.2.866 1663.1663.1.2.866 1663.1792.1.2.866
1664.832.2.2.962 1664.899.2.2.962 1664.1664.1.2.962 1664.1800.1.2.962 1665.832.2.2.871 1665.1665.1.2.871
1666.832.2.2.943 1666.1666.1.2.943 1667.833.2.2.910 1667.1667.1.2.910 1668.833.2.2.849 1668.834.2.2.1036
1668.1668.1.2.1036 1669.834.2.2.896 1669.1669.1.2.896 1670.835.2.2.900 1670.1670.1.2.900 1672.835.2.2.841
1672.836.2.2.854 1672.920.2.2.854 1672.1672.1.2.841 1672.1841.1.2.841 1675.837.2.2.856 1675.1675.1.2.856
1676.838.2.2.880 1676.1676.1.2.880 1678.839.2.2.904 1678.1678.1.2.904 1680.839.2.2.939 1680.840.2.2.908
1680.1680.1.2.939 1682.841.2.2.930 1682.1682.1.2.930 1683.841.2.2.924 1683.1683.1.2.924 1684.842.2.2.916
1684.1684.1.2.916 1685.842.2.2.916 1685.1685.1.2.916 1686.842.2.2.925 1686.1686.1.2.925 1687.843.2.2.907
1687.1687.1.2.907 1688.944.2 1688.1890.1.2.851 1689.844.2.2.965 1689.1689.1.2.965 1691.912.2.2.907
1691.1825.1.2.907 1693.846.2.2.996 1693.1693.1.2.996 1694.847.2.2.848 1694.1694.1.2.848 1695.847.2.2.876
1695.936.2.2.876 1695.1695.1.2.876 1695.1873.1.2.876 1696.848.2.2.1008 1696.1696.1.2.1008 1699.849.2.2.1019
1699.1699.1.2.1019 1700.850.2.2.864 1700.1700.1.2.864 1701.850.2.2.883 1701.1701.1.2.883 1702.850.2.2.1003
1702.1702.1.2.1003 1703.851.2.2.919 1703.1703.1.2.919 1704.932.2 1704.1865.1.2.879 1705.852.2.2.876
1705.1705.1.2.876 1706.852.2.2.959 1706.1706.1.2.959 1707.853.2.2.856 1707.1707.1.2.856 1709.854.2.2.987
1709.1709.1.2.987 1710.854.2.2.1129 1710.855.2.2.934 1710.1710.1.2.934 1712.855.2.2.877 1712.856.2.2.910
1712.1712.1.2.877 1713.856.2.2.906 1713.1713.1.2.906 1714.857.2.2.858 1714.1714.1.2.858 1716.857.2.2.877
1716.858.2.2.912 1716.1716.1.2.877 1719.859.2.2.1007 1719.1719.1.2.1007 1720.859.2.2.917 1720.967.2
1720.1720.1.2.917 1720.1935.1.2.917 1723.861.2.2.990 1723.1723.1.2.990 1724.862.2.2.900 1724.1724.1.2.900
1725.862.2.2.925 1725.1725.1.2.925 1726.863.2.2.918 1726.1726.1.2.918 1728.863.2.2.895 1728.1728.1.2.895
1731.865.2.2.877 1731.1731.1.2.877 1734.866.2.2.929 1734.867.2.2.950 1734.1734.1.2.929 1735.867.2.2.998
1735.1735.1.2.998 1737.868.2.2.930 1737.1737.1.2.930 1738.869.2.2.954 1738.1738.1.2.954 1740.967.2
1740.1935.1.2.1003 1741.870.2.2.907 1741.1741.1.2.907 1742.870.2.2.893 1742.871.2.2.950 1742.1742.1.2.950
1743.871.2.2.917 1743.1743.1.2.917 1744.872.2.2.1004 1744.1744.1.2.1004 1747.873.2.2.966 1747.1747.1.2.966
1748.873.2.2.909 1748.1748.1.2.909 1749.874.2.2.942 1749.1749.1.2.942 1750.874.2.2.921 1750.875.2.2.918
1750.1750.1.2.918 1752.875.2.2.1087 1752.876.2.2.1100 1752.1752.1.2.1100 1753.876.2.2.993 1753.1753.1.2.993
1754.876.2.2.915 1754.877.2.2.884 1754.1754.1.2.884 1755.877.2.2.888 1755.1755.1.2.888 1756.877.2.2.951
1756.878.2.2.898 1756.1756.1.2.898 1757.878.2.2.986 1757.1757.1.2.986 1758.878.2.2.923 1758.879.2.2.1098
1758.1758.1.2.923 1759.879.2.2.1023 1759.1759.1.2.1023 1760.880.2.2.988 1760.1760.1.2.988 1764.882.2.2.952
1764.1764.1.2.952 1766.882.2.2.935 1766.1766.1.2.935 1769.884.2.2.893 1769.1769.1.2.893 1770.885.2.2.960
1770.1770.1.2.960 1771.885.2.2.1091 1771.1771.1.2.1091 1772.989.2.2.992 1772.1011.2.2.992 1772.1978.1.2.992
1772.2023.1.2.992 1773.886.2.2.1005 1773.1773.1.2.1005 1775.887.2.2.926 1775.1775.1.2.926 1776.887.2.2.953
1776.1776.1.2.953 1780.889.2.2.929 1780.1780.1.2.929 1781.890.2.2.893 1781.1781.1.2.893 1782.891.2.2.916
1782.1782.1.2.916 1783.891.2.2.1137 1783.1783.1.2.1137 1785.892.2.2.924 1785.1785.1.2.924 1790.895.2.2.966
1790.1790.1.2.966 1791.895.2.2.930 1791.1791.1.2.930 1793.995.2.2.915 1793.1990.1.2.915 1794.897.2.2.1076
1794.1794.1.2.1076 1797.898.2.2.1198 1797.1797.1.2.1198 1799.899.2.2.910 1799.1799.1.2.910 1801.900.2.2.912
1801.1801.1.2.912 1802.901.2.2.934 1802.1802.1.2.934 1803.901.2.2.996 1804.902.2.2.914 1804.1804.1.2.914
1805.902.2.2.951 1805.1805.1.2.951 1808.903.2.2.1079 1808.1808.1.2.1079 1810.904.2.2.1067 1810.1810.1.2.1067
1812.906.2.2.988 1812.1812.1.2.988 1814.906.2.2.953 1814.1814.1.2.953 1817.908.2.2.910 1817.1817.1.2.910
1818.908.2.2.1043 1818.1818.1.2.1043 1819.909.2.2.977 1819.1819.1.2.977 1820.909.2.2.1003 1820.910.2.2.952
1820.1820.1.2.1003 1821.910.2.2.1048 1821.1821.1.2.1048 1823.911.2.2.912 1823.1823.1.2.912 1824.911.2.2.963
1824.1824.1.2.963 1825.912.2.2.929 1825.1825.1.2.929 1826.999.2 1826.1999.1.2.1067 1828.914.2.2.962
1828.1828.1.2.962 1832.915.2.2.1143 1832.1832.1.2.1143 1836.918.2.2.1122 1836.1836.1.2.1122 1837.918.2.2.974
1837.1837.1.2.974 1838.918.2.2.1021 1838.1838.1.2.1021 1843.921.2.2.947 1843.1843.1.2.947 1844.921.2.2.1045
1844.1844.1.2.1045 1850.924.2.2.961 1850.1850.1.2.961 1852.926.2.2.928 1852.1852.1.2.928 1855.927.2.2.975
1855.1855.1.2.975 1856.927.2.2.977 1856.928.2.2.1122 1856.1856.1.2.977 1857.928.2.2.998 1857.1857.1.2.998
1859.929.2.2.1000 1859.1859.1.2.1000 1861.930.2.2.955 1861.1861.1.2.955 1863.931.2.2.945 1863.1863.1.2.945
1864.932.2.2.1074 1864.1864.1.2.1074 1868.1020.2 1868.2041.1.2.947 1869.934.2.2.974 1869.1869.1.2.974
1871.935.2.2.1026 1871.1871.1.2.1026 1873.936.2.2.1176 1873.1873.1.2.1176 1875.937.2.2.1122 1875.1875.1.2.1122
1876.937.2.2.1201 1876.1876.1.2.1201 1878.938.2.2.957 1878.1878.1.2.957 1882.941.2.2.1170 1882.1882.1.2.1170
1885.942.2.2.966 1885.1885.1.2.966 1887.943.2.2.945 1887.1887.1.2.945 1891.945.2.2.1136 1891.1891.1.2.1136
1892.946.2.2.966 1892.1025.2.2.966 1892.1892.1.2.966 1892.2050.1.2.966 1895.947.2.2.1003 1895.1895.1.2.1003
1901.950.2.2.1050 1901.1901.1.2.1050 1902.950.2.2.1087 1902.1902.1.2.1087 1910.954.2.2.1011 1910.955.2.2.1028
1910.1910.1.2.1011 1911.955.2.2.969 1911.1911.1.2.969 1912.956.2.2.1068 1912.1912.1.2.1068 1913.956.2.2.1086
1913.1913.1.2.1086 1914.956.2.2.1025 1914.1914.1.2.1025 1920.959.2.2.1081 1920.1920.1.2.1081 1921.960.2.2.1201
1921.1921.1.2.1201 1924.962.2.2.1004 1924.1924.1.2.1004 1925.962.2.2.1154 1925.1925.1.2.1154 1926.962.2.2.977
1926.1926.1.2.977 1931.965.2.2.1043 1931.1931.1.2.1043 1934.969.2.2.1920 1934.1938.1.2.1920 1935.967.2.2.1058
1935.1935.1.2.1058 1937.968.2.2.982 1937.1937.1.2.982 1939.969.2.2.1100 1939.1939.1.2.1100 1947.973.2.2.1150
1947.1947.1.2.1150 1952.976.2.2.1008 1952.1952.1.2.1008 1953.976.2.2.1085 1953.1953.1.2.1085 1954.977.2.2.1066
1954.1954.1.2.1066 1956.977.2.2.999 1956.1956.1.2.999 1964.982.2.2.1058 1964.1964.1.2.1058 1968.983.2.2.1091
1968.984.2.2.994 1968.1968.1.2.994 1969.984.2.2.1024 1969.1969.1.2.1024 1972.986.2.2.1076 1972.1972.1.2.1076
1973.986.2.2.1012 1973.1060.2.2.1012 1973.1973.1.2.1012 1973.2122.1.2.1012 1975.987.2.2.1055 1975.1975.1.2.1055
1976.988.2.2.988 1976.1976.1.2.988 1977.988.2.2.1119 1977.1977.1.2.1119 1979.989.2.2.1053 1979.1979.1.2.1053
1980.989.2.2.1015 1980.990.2.2.1036 1980.1980.1.2.1015 1982.990.2.2.1053 1982.991.2.2.1038 1982.1982.1.2.1038
1984.992.2.2.1094 1984.1984.1.2.1094 1987.993.2.2.1045 1987.1987.1.2.1045 1988.993.2.2.1011 1988.1988.1.2.1011
1993.996.2.2.1024 1993.1993.1.2.1024 1997.998.2.2.1032 1997.1997.1.2.1032 1998.998.2.2.999 1998.1998.1.2.999
2004.1001.2.2.1049 2004.2004.1.2.1049 2007.1003.2.2.1176 2007.2007.1.2.1176 2008.1004.2.2.1080
2008.2008.1.2.1080 2009.1004.2.2.1130 2009.2009.1.2.1130 2013.1006.2.2.1097 2013.2013.1.2.1097
2016.1008.2.2.1154 2016.2016.1.2.1154 2017.1008.2.2.1038 2017.2017.1.2.1038 2019.1009.2.2.1094
2019.2019.1.2.1094 2020.1009.2.2.1319 2020.2020.1.2.1319 2026.1100.2.2.1118 2026.2026.1.2.1118
2026.2201.1.2.1118 2034.1017.2.2.1040 2034.2034.1.2.1040 2045.1022.2.2.1080 2045.2045.1.2.1080 2048.1024.2
2048.1025.2 2048.2048.1 2048.2050.1 2051.1025.2.2.1057 2051.2051.1.2.1057 2054.1026.2.2.1099 2054.1350.2.4.688
2054.2054.1.2.1099 2054.2702.1.2.1099 2072.1036.2.2.1170 2072.2072.1.2.1170 2073.1036.2.2.1080
2073.2073.1.2.1080 2077.1038.2.2.1066 2077.2077.1.2.1066 2082.1041.2.2.1112 2082.2082.1.2.1112 2085.1042.2.3.730
2085.2085.1.3.730 2092.1046.2.2.1198 2092.2092.1.2.1198 2094.1122.2 2094.2246.1.2.1167 2105.1052.2.2.1128
2107.1053.2.2.1058 2107.2107.1.2.1058 2117.1058.2.2.1109 2117.2117.1.2.1109 2121.1060.2.2.1082
2121.2121.1.2.1082 2135.1067.2.2.1501 2135.2135.1.2.1501 2140.1070.2.2.1128 2140.2140.1.2.1128
2146.1073.2.2.1236 2146.2146.1.2.1236 2155.1077.2.3.914 2155.2155.1.3.914 2160.1079.2.2.1139 2160.2160.1.2.1139
2163.1081.2.2.1091 2163.2163.1.2.1091 2164.1081.2.2.1185 2164.2164.1.2.1185 2165.1082.2.2.1273
2165.2165.1.2.1273 2179.1089.2.2.1136 2179.2179.1.2.1136 2193.1096.2.2.1177 2193.2193.1.2.1177 2213.1106.2.3.900
2213.2213.1.3.900 2228.1113.2.2.1139 2228.2228.1.2.1139 2232.1237.2 2232.2477.1.3.835 2260.1129.2.2.1145
2260.2260.1.2.1145 2313.1251.2.3.910 2313.2504.1.3.910 2333.1166.2.3.836 2333.2333.1.3.836 2349.1174.2.2.1237
2349.2349.1.2.1237 2367.1249.2.3.1018 2367.2499.1.3.1018 2369.1184.2.3.891 2369.2369.1.3.891 2401.1307.2.3.930
2401.2616.1.3.930 2411.1204.2.3.925 2411.2411.1.3.925 2413.1206.2.2.1298 2413.2413.1.2.1298 2419.1209.2.3.846
2419.2419.1.3.846 2433.1216.2.3.936 2433.2433.1.3.936 2457.1228.2.3.896 2457.2457.1.3.896 2465.1232.2.3.923
2465.2465.1.3.923 2466.1233.2.3.896 2466.2466.1.3.896 2480.1239.2.3.936 2480.2480.1.3.936 2489.1244.2.3.883
2489.2489.1.3.883 2498.1249.2.3.884 2498.2498.1.3.884 2506.1320.2 2506.2641.1.3.1023 2511.2511.1.3.919
2523.1261.2.3.977 2523.2523.1.3.977 2541.1270.2.3.936 2541.2541.1.3.936 2556.1277.2.3.872 2556.2556.1.3.872
2605.1302.2.3.938 2605.2605.1.3.938 2626.1312.2.3.1065 2626.2626.1.3.1065 2629.1313.2.3.877 2629.2629.1.3.877
2643.1320.2.3.1095 2643.2643.1.3.1095 2644.1321.2.3.1000 2644.2644.1.3.1000 2654.1326.2.3.963 2654.2654.1.3.963
2687.1343.2.3.999 2687.2687.1.3.999 2693.1346.2.3.945 2693.2693.1.3.945 2698.1348.2.3.1015 2698.2698.1.3.1015
2713.1356.2.3.999 2713.2713.1.3.999 2715.1357.2.3.942 2715.2715.1.3.942 2721.1360.2.3.1140 2721.2721.1.3.1140
2728.1431.2 2728.2863.1.3.1045 2749.1464.2.3.1177 2749.2930.1.3.1177 2769.1384.2.3.1080 2769.2769.1.3.1080
2791.1394.2.3.977 2791.2791.1.3.977 2813.1406.2.3.1170 2813.2813.1.3.1170 2819.1408.2.3.989 2819.2819.1.3.989
2851.1425.2.3.1081 2851.2851.1.3.1081 2864.1431.2.3.1045 2864.2864.1.3.1045 2866.1432.2.3.1085
2866.2866.1.3.1085 2910.1454.2.3.1049 2910.2910.1.3.1049 2968.1483.2.3.1261 2968.2968.1.3.1261 3114.1689.2.5.702
3114.3380.1.5.702 3162.1713.2 3162.3428.1.5.701 3206.1927.2.4.1114 3206.3854.1.4.1114 3250.1623.2.4.911
3250.3250.1.4.911 3278.1708.2 3278.3417.1.5.697 3289.1643.2.4.867 3289.3289.1.4.867 3336.1667.2.3.1237
3336.3336.1.3.1237 3601.1799.2.4.1057 3601.3601.1.4.1057 8071.4430.2.12.833 8071.8866.1.12.833
8095.4119.2.10.927 8095.8243.1.10.927 8102.4059.2 8102.8123.1.10.1085 8117.4065.2.9.1097 8117.8135.1.9.1097
8119.4452.2.10.942 8119.8906.1.10.942 8135.4523.2.10.950 8135.9049.1.10.950 8136.4523.2.10.944
8136.9052.1.10.944 8141.4077.2.9.1163 8141.8159.1.9.1163 8150.4073.2.8.1320 8150.8150.1.8.1320
8153.4212.2.10.1318 8153.8430.1.10.1318 8156.4087.2 8156.8179.1.11.835 8160.4087.2.11.842 8160.8182.1.11.842
8161.4219.2.11.823 8161.8445.1.11.823 8178.4086.2.8.1141 8178.4087.2.8.1141 8178.8178.1.8.1141 8192.4096.2
8192.8192.1
"""


def _dumped_partition(tc, S, r, nc, mq):
    """(cached, new) per request consistent with one dumped cell: nc requests, the first with mq new
    tokens, the others splitting tc - mq (each in 1..mq); at ratio 2 as many of them odd as S needs;
    then cached prefixes add the missing S keys, request after request, up to 16384 tokens each.
    None if no such partition exists."""
    if nc == 1:
        if mq != tc:
            return None
        new = [tc]
    else:
        m, rest = nc - 1, tc - mq
        if not m <= rest <= m * mq:
            return None
        new = [mq] + [rest // m + (i < rest % m) for i in range(m)]
        if r == 2:
            want = max(tc - 2 * S, tc % 2) - mq % 2  # odd lengths wanted among the others
            if want < 0 or want > m:
                return None
            for _ in range(2 * m):
                odd = [i for i in range(1, nc) if new[i] % 2]
                even = [i for i in range(1, nc) if new[i] % 2 == 0]
                if len(odd) == want:
                    break
                src = even if len(odd) < want else odd
                up = [i for i in src if new[i] < mq]
                down = [i for i in src if new[i] > 1 and i != (up[0] if up else -1)]
                if not up or not down:
                    return None
                new[up[0]] += 1
                new[down[0]] -= 1
            if sum(n % 2 for n in new[1:]) != want:
                return None
    extra = S - sum(n // r for n in new)
    if extra < 0:
        return None
    reqs = []
    for n in new:
        e = min(extra, (16384 - n) // r)
        extra -= e
        reqs.append((r * e, n))
    if extra:
        return None
    return reqs


def _dumped_cells():
    return [tuple(int(x) for x in tok.split(".")) for tok in _DUMPED_CELLS.split()]


@pytest.mark.parametrize("r", [1, 2])
def test_multi_request_dumped_partitions(r):
    """One consistent partition per dumped cell at this ratio, vs float64."""
    cells = [c for c in _dumped_cells() if c[2] == r]
    stats = collections.Counter()
    for c in cells:
        tc, S, _, *rest = c
        nc, mq = rest if rest else (1, tc)
        reqs = _dumped_partition(tc, S, r, nc, mq)
        assert reqs is not None and len(reqs) == nc and sum(n for _, n in reqs) == tc
        assert sum((cached + n) // r for cached, n in reqs) == S and max(n for _, n in reqs) == mq
        T, S_run, n_vis = _run_partition(reqs, r, seed=tc + 3 * S + r)
        assert (T, S_run) == (tc, S)
        stats["cells"] += 1
        stats["multi-request"] += nc > 1
        stats["rows"] += T
        stats["visible checked"] += n_vis
    print(f"dumped partitions r {r}: {dict(stats)}")
    assert stats["cells"] == len(cells) > 0


# ----------------------------------------------------------------------------- long context: row tiles, long gathers
#
# The V4.1 target's prefill index pass (``_index_prefill`` in the target's modeling.py) calls this op once per ROW
# TILE: rows [a, b) of the step's tc <= 16384 (max_num_tokens) context rows, passed as the row-offset views q[a:b],
# q_sf[a:b], weights[a:b], ks[a:b], ke[a:b], against ALL s_len keys the step gathered (full width, clean_logits=True,
# fp32 logits). ``_index_row_tiles`` sizes the tiles so rows x align(s_len + 256, 256) <= 2^28 fp32 elements: one
# tile of all rows when they fit, otherwise tiles of 2^28 // width rows, rounded down to a multiple of 64 when that
# is >= 64. The decoder-replay pass makes the same call on its tail rows (``_tail_cand_masks``: the last 128
# positions of each bounded request, ratio-1 keys). s_len is the sum over the step's context requests of
# kv_len // r, and one request reaches Lm = 1,048,590 tokens (max_seq_len 1,048,576 plus DSpark's 14). s_len is
# certified here up to 2^25: the op that gathers the keys (indexer_k_cache_gather) misaddresses rows from 2^25 on.
#
# Two references. ``_long_context_reference`` is the float64 reference at any extent (keys decoded 2^16 rows at a
# time). ``_LongContextPalette`` draws the query rows and key rows of a call from palettes, so one float64 [Q, P]
# table serves every call: a call's expected logits are that table gathered by its draws, -inf outside each row's
# range. A kernel that reads the wrong key (query) row lands on another palette entry with probability 1 - 1/P
# (1 - 1/Q) per element; ``test_long_context_wrong_answers_are_seen`` shows the check failing for each such mistake.


_LONG_CONTEXT_TC = 16384  # context rows per step (max_num_tokens)
_LONG_CONTEXT_K = 256  # context requests per step (max_batch_size)
_LONG_CONTEXT_LM = 1048590  # the longest request: max_seq_len 1,048,576 + 14 with DSpark
_LONG_CONTEXT_BUDGET = 1 << 28  # fp32 logits elements per call (the target's _INDEX_TILE_ELEMS)
_LONG_CONTEXT_S_MAX = 1 << 25  # gathered keys per step, certified up to here


def _long_context_rows_max(s_len):
    """The tallest tile at s_len: all of a step's rows go in one call while rows <= 2^28 // width."""
    return min(_LONG_CONTEXT_TC, max(1, _LONG_CONTEXT_BUDGET // _row_stride(s_len)))


def _long_context_tiles(rows, s_len):
    """The target's row tiles (``_index_row_tiles``) of a step with ``rows`` context rows over s_len keys."""
    per = max(1, _LONG_CONTEXT_BUDGET // _row_stride(s_len))
    if per >= rows:
        return [(0, rows)]
    if per >= 64:
        per -= per % 64
    return [(a, min(rows, a + per)) for a in range(0, rows, per)]


def _long_context_mask_(out, ks, ke):
    """-inf outside each row's [ks, ke), in place, 2^26 elements at a time."""
    T, S = out.shape
    col = torch.arange(S, device=DEV)
    step = max(1, (1 << 26) // S)
    for t0 in range(0, T, step):
        vis = (col[None, :] >= ks[t0 : t0 + step, None].long()) & (
            col[None, :] < ke[t0 : t0 + step, None].long()
        )
        out[t0 : t0 + step].masked_fill_(~vis, NEG_INF)
    return out


def _long_context_reference(q, q_sf, w, kv, kv_sf, ks, ke):
    """The float64 reference at any extent, rounded once to fp32: [T, S], -inf outside [ks, ke); and
    max_t,s sum_h |term| (the exact regime's premise). Keys decoded 2^16 at a time, query rows taken 64 at a time."""
    T, S = q.shape[0], kv.shape[0]
    qd, wd = _decode(q, q_sf), w.double()
    out = torch.empty(T, S, dtype=torch.float32, device=DEV)
    bound = torch.zeros((), dtype=torch.float64, device=DEV)
    for s0 in range(0, S, 1 << 16):
        kd = _decode(kv[s0 : s0 + (1 << 16)], kv_sf[s0 : s0 + (1 << 16)])
        for t0 in range(0, T, 64):
            terms = (
                torch.einsum("thd,kd->thk", qd[t0 : t0 + 64], kd)
                .clamp_min_(0)
                .mul_(wd[t0 : t0 + 64, :, None])
            )
            out[t0 : t0 + 64, s0 : s0 + kd.shape[0]] = terms.sum(1)
            bound = torch.maximum(bound, terms.abs_().sum(1).max())
            del terms
    return _long_context_mask_(out, ks, ke), float(bound)


class _LongContextPalette:
    """Q query rows (with their head weights) and P key rows in the exact regime, and the float64 logits of every
    (query row, key row) pair, rounded to fp32: ``table[i, j]``. A call's inputs are rows drawn from the palettes;
    its expected logits are ``table`` gathered by the draws, -inf outside each row's range."""

    def __init__(self, seed, n_q=2048, n_k=8192):
        self.q, self.q_sf, self.kv, self.kv_sf, w_raw = _exact_inputs(n_q, n_k, _HEADS, seed)
        self.w = w_raw * GLOBAL_FOLD
        end = torch.full((n_q,), n_k, dtype=torch.int32, device=DEV)
        table, _, bound = _reference(
            _decode(self.q, self.q_sf),
            _decode(self.kv, self.kv_sf),
            self.w,
            torch.zeros_like(end),
            end,
        )
        _assert_exact_premise(bound)
        self.table = table.float()
        self.n_q, self.n_k = n_q, n_k

    def draw(self, T, S, g):
        """Row draws, and the inputs they make: (q, q_sf, kv, kv_sf, weights) for T query rows and S keys."""
        qi = torch.randint(0, self.n_q, (T,), generator=g, device=DEV)
        ki = torch.randint(0, self.n_k, (S,), generator=g, device=DEV)
        return qi, ki, (self.q[qi], self.q_sf[qi], self.kv[ki], self.kv_sf[ki], self.w[qi])

    def expected(self, qi, ki, ks, ke):
        return _long_context_mask_(self.table.index_select(0, qi).index_select(1, ki), ks, ke)


def _long_context_check_tiles(pal, qi, ki, inputs, ks, ke, tiles):
    """Each tile called as the target calls it (row-offset views of the step's tensors, all S keys) and compared,
    every element, with the palette reference. Returns the visible elements checked."""
    q, q_sf, kv, kv_sf, w = inputs
    S = kv.shape[0]
    n_vis = 0
    for a, b in tiles:
        assert (b - a) * _row_stride(S) <= _LONG_CONTEXT_BUDGET or b - a == 1
        out = _call(q[a:b], q_sf[a:b], kv, kv_sf, w[a:b], ks[a:b], ke[a:b])
        assert out.shape == (b - a, S) and out.stride() == (_row_stride(S), 1)
        assert out.untyped_storage().nbytes() == _logits_bytes(b - a, S, _HEADS)
        exp = pal.expected(qi[a:b], ki, ks[a:b], ke[a:b])
        bad = int((out != exp).sum())
        assert bad == 0, f"tile [{a}, {b}) over S {S}: {bad} of {out.numel()} elements differ"
        n_vis += int((ke[a:b] - ks[a:b]).clamp_min(0).sum())
        _poison(out)
        del out, exp
    return n_vis


def _long_context_sample_tiles(tiles, rng, n=16):
    """All tiles when there are at most n; else the first, the last and n - 2 others drawn at random."""
    if len(tiles) <= n:
        return tiles
    mid = sorted(rng.sample(range(1, len(tiles) - 1), n - 2))
    return [tiles[0]] + [tiles[i] for i in mid] + [tiles[-1]]


# Every distinct (rows, s_len, r) of the deepgemm_fp8_fp4_mqa_logits and .tail_cand records in the long-context
# runs' runtime-value dumps (tep4, DSpark, max_num_tokens 8192, max_seq_len 131072 / 262144 / 1048576), as
# "rows.s_len.r". The four tail_cand cells (128 rows at s_len 130773, 261860, 523951, 1048219, ratio 1) are among
# them.
_LONG_CONTEXT_DUMPED = """
1.1.1 2.1.2 2.2.1 64.32768.1 64.32768.2 128.8192.1 128.65536.1 128.65536.2 128.98304.1 128.98304.2 128.130773.1
128.192512.2 128.196608.1 128.196608.2 128.221184.1 128.221184.2 128.225280.2 128.229376.1 128.229376.2
128.261860.1 128.282624.2 128.286720.1 128.286720.2 128.290816.2 128.294912.1 128.294912.2 128.299008.2
128.421888.2 128.425984.1 128.425984.2 128.430080.2 128.434176.1 128.434176.2 128.438272.2 128.442368.1
128.442368.2 128.446464.2 128.450560.1 128.450560.2 128.454656.2 128.458752.1 128.458752.2 128.462848.2
128.523951.1 128.524288.1 128.532480.1 128.540672.1 128.548864.1 128.557056.1 128.565248.1 128.573440.1
128.581632.1 128.589824.1 128.598016.1 128.606208.1 128.614400.1 128.622592.1 128.630784.1 128.638976.1
128.647168.1 128.655360.1 128.663552.1 128.671744.1 128.679936.1 128.688128.1 128.696320.1 128.1048219.1
155.1048219.1 175.261975.2 175.523951.1 192.163840.1 192.163840.2 192.704512.1 192.712704.1 192.720896.1
192.729088.1 192.737280.1 192.745472.1 192.753664.1 192.761856.1 192.770048.1 192.778240.1 192.786432.1
192.794624.1 192.802816.1 192.811008.1 192.819200.1 192.827392.1 192.835584.1 219.524109.2 256.131072.1
256.131072.2 256.843776.1 256.851968.1 256.860160.1 256.868352.1 256.876544.1 256.884736.1 256.892928.1
256.901120.1 256.909312.1 256.917504.1 256.925696.1 256.933888.1 256.942080.1 256.950272.1 256.958464.1
256.966656.1 256.974848.1 256.983040.1 256.991232.1 256.999424.1 256.1007616.1 256.1015808.1 256.1024000.1
256.1032192.1 256.1040384.1 256.1048219.1 320.704512.1 320.712704.1 320.720896.1 320.729088.1 320.737280.1
320.745472.1 320.753664.1 320.761856.1 320.770048.1 320.778240.1 320.786432.1 320.794624.1 320.802816.1
320.811008.1 320.819200.1 320.827392.1 320.835584.1 384.606208.1 384.614400.1 384.622592.1 384.630784.1
384.638976.1 384.647168.1 384.655360.1 384.663552.1 384.671744.1 384.679936.1 384.688128.1 384.696320.1
448.352256.1 448.352256.2 448.356352.2 448.360448.1 448.360448.2 448.364544.2 448.368640.1 448.368640.2
448.372736.2 448.376832.1 448.376832.2 448.380928.2 448.524109.2 448.524288.1 448.532480.1 448.540672.1
448.548864.1 448.557056.1 448.565248.1 448.573440.1 448.581632.1 448.589824.1 448.598016.1 512.69632.2
512.102400.2 512.135168.2 512.139264.1 512.139264.2 512.167936.2 512.172032.1 512.172032.2 512.200704.2
512.204800.1 512.204800.2 512.208896.2 512.262144.1 512.262144.2 512.266240.2 512.270336.1 512.270336.2
512.274432.2 512.278528.1 512.278528.2 512.323584.2 512.327680.1 512.327680.2 512.331776.2 512.335872.1
512.335872.2 512.339968.2 512.344064.1 512.344064.2 512.348160.2 512.385024.1 512.385024.2 512.389120.2
512.393216.1 512.393216.2 512.397312.2 512.401408.1 512.401408.2 512.405504.2 512.409600.1 512.409600.2
512.413696.2 512.417792.1 512.417792.2 512.466944.1 512.466944.2 512.471040.2 512.475136.1 512.475136.2
512.479232.2 512.483328.1 512.483328.2 512.487424.2 512.491520.1 512.491520.2 512.495616.2 512.499712.1
512.499712.2 512.503808.2 512.507904.1 512.507904.2 512.512000.2 512.516096.1 512.516096.2 512.520192.2
512.523951.1 576.233472.2 576.237568.1 576.237568.2 576.241664.2 576.245760.1 576.245760.2 576.421888.2
576.425984.1 576.425984.2 576.430080.2 576.434176.1 576.434176.2 576.438272.2 576.442368.1 576.442368.2
576.446464.2 576.450560.1 576.450560.2 576.454656.2 576.458752.1 576.458752.2 576.462848.2 640.385024.1
640.385024.2 640.389120.2 640.393216.1 640.393216.2 640.397312.2 640.401408.1 640.401408.2 640.405504.2
640.409600.1 640.409600.2 640.413696.2 640.417792.1 640.417792.2 704.106496.1 704.106496.2 704.303104.1
704.303104.2 704.307200.2 704.311296.1 704.311296.2 704.315392.2 704.319488.1 704.319488.2 704.352256.1
704.352256.2 704.356352.2 704.360448.1 704.360448.2 704.364544.2 704.368640.1 704.368640.2 704.372736.2
704.376832.1 704.376832.2 704.380928.2 740.261860.1 768.143360.2 768.323584.2 768.327680.1 768.327680.2
768.331776.2 768.335872.1 768.335872.2 768.339968.2 768.344064.1 768.344064.2 768.348160.2 832.176128.2
832.180224.1 832.180224.2 832.303104.1 832.303104.2 832.307200.2 832.311296.1 832.311296.2 832.315392.2
832.319488.1 832.319488.2 896.212992.1 896.212992.2 896.217088.2 896.282624.2 896.286720.1 896.286720.2
896.290816.2 896.294912.1 896.294912.2 896.299008.2 960.261975.2 960.262144.1 960.262144.2 960.266240.2
960.270336.1 960.270336.2 960.274432.2 960.278528.1 960.278528.2 1024.36864.2 1024.73728.1 1024.73728.2
1024.147456.1 1024.147456.2 1024.249856.2 1024.253952.1 1024.253952.2 1024.258048.2 1024.261860.1 1088.110592.2
1088.233472.2 1088.237568.1 1088.237568.2 1088.241664.2 1088.245760.1 1088.245760.2 1152.184320.2 1152.188416.1
1152.188416.2 1152.221184.1 1152.221184.2 1152.225280.2 1152.229376.1 1152.229376.2 1216.212992.1 1216.212992.2
1216.217088.2 1280.114688.1 1280.114688.2 1280.151552.2 1280.200704.2 1280.204800.1 1280.204800.2 1280.208896.2
1344.192512.2 1344.196608.1 1344.196608.2 1408.77824.2 1408.184320.2 1408.188416.1 1408.188416.2 1472.118784.2
1472.176128.2 1472.180224.1 1472.180224.2 1536.155648.1 1536.155648.2 1536.159744.2 1536.167936.2 1536.172032.1
1536.172032.2 1600.163840.1 1600.163840.2 1664.81920.1 1664.81920.2 1664.122880.1 1664.122880.2 1664.155648.1
1664.155648.2 1664.159744.2 1728.40960.1 1728.40960.2 1728.151552.2 1749.130773.1 1792.147456.1 1792.147456.2
1856.143360.2 1920.135168.2 1920.139264.1 1920.139264.2 1956.130930.2 1984.130930.2 1984.131072.1 1984.131072.2
2048.86016.2 2048.126976.2 2048.130773.1 2176.122880.1 2176.122880.2 2240.118784.2 2304.45056.2 2304.90112.1
2304.90112.2 2304.114688.1 2304.114688.2 2368.110592.2 2496.106496.1 2496.106496.2 2560.94208.2 2560.102400.2
2688.98304.1 2688.98304.2 2816.49152.1 2816.49152.2 2816.94208.2 2944.90112.1 2944.90112.2 3072.86016.2
3200.53248.2 3264.81920.1 3264.81920.2 3392.77824.2 3584.57344.1 3584.57344.2 3584.73728.1 3584.73728.2
3840.69632.2 3861.65386.2 3904.61440.2 4032.65386.2 4032.65536.1 4032.65536.2 4288.61440.2 4608.57344.1
4608.57344.2 4992.53248.2 5376.49152.1 5376.49152.2 5888.45056.2 6464.40960.1 6464.40960.2 7168.36864.2
8128.32768.1 8128.32768.2 8192.4096.2 8192.8192.1 8192.8192.2 8192.12288.2 8192.16384.1 8192.16384.2
8192.20480.2 8192.24576.1 8192.24576.2 8192.28672.2
"""


def _long_context_dumped_cells():
    return [tuple(int(x) for x in tok.split(".")) for tok in _LONG_CONTEXT_DUMPED.split()]


# The s_len values of the row-axis sweeps: the tile-height changes at the 16K envelope (16128 is the longest s_len
# whose 16384 rows fit one call), the long-context runs' lengths (131072 / 262144 / 1048576, and the passkey
# prompts' 261860 / 523951 / 1048219), Lm itself (tallest tile 255, tiles of 192), both sides of the 64-row rounding
# (2^28 // width = 128 | 127 at 2096896 | 2097152, 64 | 63 at 4194048 | 4194304), and on to 2^25 (tiles of 7).
_LONG_CONTEXT_S_ROWS = (
    16128,
    16384,
    32768,
    65536,
    131072,
    261860,
    262144,
    523951,
    524288,
    1048219,
    _LONG_CONTEXT_LM,
    2096896,
    2097152,
    4194048,
    4194304,
    1 << 23,
    1 << 24,
    _LONG_CONTEXT_S_MAX,
)
_LONG_CONTEXT_S_RECIPE = (16128, 131072, _LONG_CONTEXT_LM, 4194304, _LONG_CONTEXT_S_MAX)


def test_long_context_tile_heights():
    """The tiling at the lengths the long-context runs reached; every tile covers the rows once within the budget;
    every dumped (rows, s_len) is a tile the tiling makes."""
    assert _long_context_tiles(_LONG_CONTEXT_TC, 16128) == [(0, _LONG_CONTEXT_TC)]
    assert _long_context_rows_max(16384) == 16131 and _long_context_tiles(_LONG_CONTEXT_TC, 16384)[
        :2
    ] == [
        (0, 16128),
        (16128, _LONG_CONTEXT_TC),
    ]
    assert _long_context_tiles(_LONG_CONTEXT_TC, 523951)[0] == (0, 512) and _long_context_tiles(
        _LONG_CONTEXT_TC, 1048219
    )[0] == (0, 256)
    assert _long_context_rows_max(_LONG_CONTEXT_LM) == 255 and _long_context_tiles(
        _LONG_CONTEXT_TC, _LONG_CONTEXT_LM
    )[0] == (0, 192)
    assert [_long_context_rows_max(S) for S in (2096896, 2097152, 4194048, 4194304)] == [
        128,
        127,
        64,
        63,
    ]
    assert _long_context_tiles(_LONG_CONTEXT_TC, 2097152)[0] == (0, 64) and _long_context_tiles(
        _LONG_CONTEXT_TC, 4194304
    )[0] == (0, 63)
    assert (
        _long_context_rows_max(_LONG_CONTEXT_S_MAX) == 7
        and len(_long_context_tiles(_LONG_CONTEXT_TC, _LONG_CONTEXT_S_MAX)) == 2341
    )
    for S in _LONG_CONTEXT_S_ROWS + (1, 16383, 3_000_000, 12_345_677):
        for rows in (1, 7, 63, 64, 65, 255, 256, 4097, _LONG_CONTEXT_TC - 1, _LONG_CONTEXT_TC):
            tiles = _long_context_tiles(rows, S)
            assert tiles[0][0] == 0 and tiles[-1][1] == rows
            assert all(t[1] == u[0] for t, u in zip(tiles, tiles[1:]))
            assert all(
                0 < b - a <= _long_context_rows_max(S)
                and (b - a) * _row_stride(S) <= _LONG_CONTEXT_BUDGET
                for a, b in tiles
            )
    for rows, S, _ in _long_context_dumped_cells():
        assert rows <= _long_context_rows_max(S), (rows, S)


def _long_context_full(regime, T, S, ks, ke, seed):
    """Inputs of one set and the T x S call, checked against the float64 reference."""
    if regime == "exact":
        q, q_sf, kv, kv_sf, w_raw = _exact_inputs(T, S, _HEADS, seed)
        w = w_raw * GLOBAL_FOLD
    else:
        q, q_sf, kv, kv_sf, w = _recipe_inputs(T, S, _HEADS, seed)
    full = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    ref, bound = _long_context_reference(q, q_sf, w, kv, kv_sf, ks, ke)
    if regime == "exact":
        _assert_exact_premise(bound)
        assert torch.equal(full, ref), (
            f"{int((full != ref).sum())} of {full.numel()} elements differ"
        )
    else:
        assert torch.equal(torch.isneginf(full), torch.isneginf(ref))
        torch.testing.assert_close(full, ref)
    del ref
    return q, q_sf, kv, kv_sf, w, full


_LONG_CONTEXT_ROW_CASES = (
    [("exact", "r1", S) for S in _LONG_CONTEXT_S_ROWS]
    + [("exact", "packed", S) for S in _LONG_CONTEXT_S_ROWS]
    + [("exact", "r2", 16128)]
    + [("recipe", "packed", S) for S in _LONG_CONTEXT_S_RECIPE]
)


@pytest.mark.parametrize("regime,form,S", _LONG_CONTEXT_ROW_CASES)
def test_long_context_row_axis_every_tile_height(regime, form, S):
    """Every tile height the target makes at s_len S, 1 .. min(16384, 2^28 // width): the first T rows (views of the
    tallest tile's inputs) over the same S keys give exactly the first T rows of the tallest tile, which itself
    matches float64. ``r1``: one request whose last row sees key S - 1; ``packed``: three requests, keys at their own
    offsets. Harness sighting as in ``test_row_axis_every_T``."""
    T_max = _long_context_rows_max(S)
    ks, ke = _axis_ranges(form, T_max, S)
    sets = [
        _long_context_full(regime, T_max, S, ks, ke, seed=S % 100_003 + 7 * k + len(form))
        for k in (0, 1)
    ]
    torch.cuda.synchronize()
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    shift = {}
    poisoned, reused = set(), 0
    for T in range(T_max, 0, -1):
        q, q_sf, kv, kv_sf, w, full = sets[T % 2]
        out = _call(q[:T], q_sf[:T], kv, kv_sf, w[:T], ks[:T], ke[:T])
        assert out.shape == (T, S) and out.stride() == (_row_stride(S), 1)
        assert out.untyped_storage().nbytes() == _logits_bytes(T, S, _HEADS)
        reused += out.data_ptr() in poisoned
        bad += (out != full[:T]).sum()
        if (
            T < T_max
            and T in (1, 2, 5, T_max // 2 + 1, T_max - 1)
            and bool((ke[: T + 1] > ks[: T + 1]).any())
        ):
            shift[T] = (out != full[1 : T + 1]).sum()  # control: the gate must see a one-row shift
        _poison(out)
        poisoned.add(out.data_ptr())
        del out
    shift = {T: int(n) for T, n in shift.items()}
    print(
        f"tile heights {regime}/{form} S {S}: T 1..{T_max}, {int(bad)} mismatching elements; "
        f"{reused}/{T_max} results on a NaN-filled block; one-row-shift control mismatches {shift}"
    )
    assert int(bad) == 0
    assert reused >= T_max // 2
    assert shift and all(n > 0 for n in shift.values())


def _long_context_key_sample(S_max, seed):
    """S in (16384, S_max] (``test_key_axis`` covers 1..32768): S = 0, 1, 2, 3 or -1 mod 256 within 2048 of every
    power of two from 2^15, the row-axis lengths, the dumped s_len, S_max, and 200 seeded log-uniform draws."""
    s = {S_max}
    for k in range(15, S_max.bit_length()):
        for c in range((2**k - 2048) // 256, (2**k + 2048) // 256 + 1):
            s.update(c * 256 + d for d in (-1, 0, 1, 2, 3))
    s.update(_LONG_CONTEXT_S_ROWS)
    s.update(S for _, S, _ in _long_context_dumped_cells())
    rng = random.Random(seed)
    s.update(int(2 ** rng.uniform(14, math.log2(S_max))) for _ in range(200))
    return sorted(x for x in s if 16384 < x <= S_max)


@pytest.mark.parametrize("form", ["r1", "packed"])
@pytest.mark.parametrize("T,S_max", [(7, _LONG_CONTEXT_S_MAX), (255, _LONG_CONTEXT_LM)])
def test_long_context_key_axis(T, S_max, form):
    """At the tallest tile of 2^25 (7 rows) and of Lm (255 rows): the first S keys, with this form's ranges at S,
    give on every visible element exactly the T x S_max call in which every row sees every key (itself bit-identical
    to float64), and -inf everywhere else."""
    ks0 = torch.zeros(T, dtype=torch.int32, device=DEV)
    ke0 = torch.full((T,), S_max, dtype=torch.int32, device=DEV)
    sets = [
        _long_context_full("exact", T, S_max, ks0, ke0, seed=T + 3 * k + len(form)) for k in (0, 1)
    ]
    torch.cuda.synchronize()
    s_values = _long_context_key_sample(S_max, seed=T)[::-1]
    probe = {
        s_values[1],
        s_values[2],
        s_values[len(s_values) // 2],
        s_values[-1],
        1 << 24,
        (1 << 24) + 1,
    }
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    n_vis = torch.zeros((), dtype=torch.int64, device=DEV)
    shift = {}
    poisoned, reused = set(), 0
    for S in s_values:
        q, q_sf, kv, kv_sf, w, full = sets[S % 2]
        ks, ke = _axis_ranges(form, T, S)
        out = _call(q, q_sf, kv[:S], kv_sf[:S], w, ks, ke)
        assert out.shape == (T, S) and out.stride() == (_row_stride(S), 1)
        reused += out.data_ptr() in poisoned
        col = torch.arange(S, device=DEV)
        vis = (col[None, :] >= ks[:, None]) & (col[None, :] < ke[:, None])
        bad += ((out != full[:, :S]) & vis).sum() + (~torch.isneginf(out) & ~vis).sum()
        n_vis += vis.sum()
        if S in probe:
            shift[S] = (
                (out != full[:, 1 : S + 1]) & vis
            ).sum()  # control: a one-key shift must fail
        _poison(out)
        poisoned.add(out.data_ptr())
        del out, vis
    shift = {S: int(n) for S, n in shift.items()}
    print(
        f"key axis {form} T {T}, {len(s_values)} values of S in {s_values[-1]}..{s_values[0]}: {int(bad)} "
        f"mismatching elements ({int(n_vis)} visible checked); {reused}/{len(s_values)} results on a NaN-filled "
        f"block; one-key-shift control mismatches {shift}"
    )
    assert int(bad) == 0
    assert reused >= len(s_values) // 2
    assert shift and all(n > 0 for n in shift.values())


@pytest.mark.parametrize("r", [1, 2])
def test_long_context_row_logits_do_not_depend_on_the_call(r):
    """A row's logits do not depend on the other query rows or key rows of the call, bitwise, in the recipe regime
    (where fp32 rounding is not exact). Rows: the last 16384 positions of one Lm-token request at ratio r. The same
    row gives the same logits (1) in one untiled call of all 16384 rows (64 GiB of logits at r 1), (2) in its tile
    of the target's tiling, (3) called alone over just its own keys, and (4) with the request's keys shifted inside
    a larger gather of other keys: ending a 2^25-key gather (7-row tiles), and at an odd offset (1,234,567) of a
    3,000,000-key one (64-row tiles); the last 256 rows as that step's tail rows, some also alone there."""
    S, R = _LONG_CONTEXT_LM // r, _LONG_CONTEXT_TC
    q, q_sf, kv, kv_sf, w = _recipe_inputs(R, S, _HEADS, seed=60 + r)
    ks, ke = _chunk(R, S, r)
    whole = _call(q, q_sf, kv, kv_sf, w, ks, ke)
    assert whole.untyped_storage().nbytes() == _logits_bytes(R, S, _HEADS)
    last = slice(R - 64, R)
    ref, _ = _long_context_reference(q[last], q_sf[last], w[last], kv, kv_sf, ks[last], ke[last])
    assert torch.equal(torch.isneginf(whole[last]), torch.isneginf(ref))
    torch.testing.assert_close(whole[last], ref)
    del ref
    counts = collections.Counter()
    tiles = _long_context_tiles(R, S)
    for a, b in tiles:
        out = _call(q[a:b], q_sf[a:b], kv, kv_sf, w[a:b], ks[a:b], ke[a:b])
        counts["tiles"] += int((out != whole[a:b]).sum())
        del out
    rng = random.Random(70 + r)
    alone = {0, 1, 2, 3, 4, 5, R - 5, R - 4, R - 3, R - 2, R - 1}
    alone |= {t for a, _ in tiles[1:] for t in (a - 1, a)}
    alone |= set(rng.sample(range(R), 48))
    for t in sorted(alone):
        n = int(ke[t])  # ks is 0: the row's own keys are the first n
        out = _call(
            q[t : t + 1],
            q_sf[t : t + 1],
            kv[:n],
            kv_sf[:n],
            w[t : t + 1],
            ks[t : t + 1],
            ke[t : t + 1],
        )
        counts["alone"] += int((out[0] != whole[t, :n]).sum())
        del out
    tail = R - 256
    for s_big, off in ((_LONG_CONTEXT_S_MAX, _LONG_CONTEXT_S_MAX - S), (3_000_000, 1_234_567)):
        pre, pre_sf = _rand_packed(off, _gen(80 + r))
        post, post_sf = _rand_packed(s_big - off - S, _gen(90 + r))
        big, big_sf = torch.cat([pre, kv, post]), torch.cat([pre_sf, kv_sf, post_sf])
        del pre, pre_sf, post, post_sf
        ks2, ke2 = ks + off, ke + off
        for a, b in _long_context_tiles(256, s_big):
            a, b = a + tail, b + tail
            out = _call(q[a:b], q_sf[a:b], big, big_sf, w[a:b], ks2[a:b], ke2[a:b])
            counts[f"shifted {s_big}"] += int((out[:, off : off + S] != whole[a:b]).sum())
            counts[f"shifted {s_big}"] += int((~torch.isneginf(out[:, :off])).sum())
            counts[f"shifted {s_big}"] += int((~torch.isneginf(out[:, off + S :])).sum())
            del out
        for t in (tail, tail + 1, R - 2, R - 1):
            out = _call(
                q[t : t + 1],
                q_sf[t : t + 1],
                big,
                big_sf,
                w[t : t + 1],
                ks2[t : t + 1],
                ke2[t : t + 1],
            )
            counts[f"shifted {s_big}, alone"] += int((out[0, off : off + S] != whole[t]).sum())
            del out
        del big, big_sf
    print(
        f"ratio {r}: S {S}, {len(tiles)} tiles of {tiles[0][1]}, {len(alone)} rows alone; mismatching "
        f"elements vs the untiled call: {dict(counts)}"
    )
    assert set(counts.values()) == {0}, counts


_LONG_CONTEXT_N_PARTS, _LONG_CONTEXT_PART_BLOCKS = 200, 4


def _long_context_random_partition(rng):
    """A context step of the long-context form. T in 1..16384 (half uniform, half log-uniform) split at random
    cuts into k in 1..min(256, T) requests (half uniform, half log-uniform), in random order; ratio 1 or 2. A
    request is a fresh prompt a third of the time, else cached + new is log-uniform in new..Lm, or, in a quarter of
    the steps (long-context steps), uniform in Lm/2..Lm. When the step's s_len would pass 2^25, every cached prefix
    is scaled down to fit."""
    T = (
        rng.randint(1, _LONG_CONTEXT_TC)
        if rng.random() < 0.5
        else min(_LONG_CONTEXT_TC, int(2 ** rng.uniform(0, 14)))
    )
    kmax = min(_LONG_CONTEXT_K, T)
    k = (
        rng.randint(1, kmax)
        if rng.random() < 0.5
        else min(kmax, int(2 ** rng.uniform(0, math.log2(kmax + 1))))
    )
    cuts = sorted(rng.sample(range(1, T), k - 1))
    new = [b - a for a, b in zip([0] + cuts, cuts + [T])]
    rng.shuffle(new)
    r = rng.choice((1, 2))
    long_ctx = rng.random() < 1 / 4
    reqs = []
    for n in new:
        if rng.random() < 1 / 3:
            total = n
        elif long_ctx:
            total = rng.randint(max(n, _LONG_CONTEXT_LM // 2), _LONG_CONTEXT_LM)
        else:
            total = max(n, int(2 ** rng.uniform(math.log2(n), math.log2(_LONG_CONTEXT_LM))))
        reqs.append((total - n, n))
    S = sum((c + n) // r for c, n in reqs)
    if S > _LONG_CONTEXT_S_MAX:
        f = (_LONG_CONTEXT_S_MAX * r - sum(n for _, n in reqs) - r * k) / sum(c for c, _ in reqs)
        reqs = [(int(c * f), n) for c, n in reqs]
    return reqs, r


@pytest.mark.parametrize("block", range(_LONG_CONTEXT_PART_BLOCKS))
def test_long_context_random_partitions(block):
    """200 seeded random context steps (50 per block) of the long-context form, each tiled as the target tiles it;
    up to 16 tiles per step (all when there are at most 16; else the first, the last and 14 at random) called as the
    target calls them and checked, every element, against the palette reference."""
    pal = _LongContextPalette(seed=900 + block)
    per = _LONG_CONTEXT_N_PARTS // _LONG_CONTEXT_PART_BLOCKS
    stats = collections.Counter()
    for i in range(block * per, (block + 1) * per):
        rng = random.Random(7000 + i)
        reqs, r = _long_context_random_partition(rng)
        assert (
            1 <= len(reqs) <= _LONG_CONTEXT_K and 1 <= sum(n for _, n in reqs) <= _LONG_CONTEXT_TC
        )
        assert all(n >= 1 and c >= 0 and c + n <= _LONG_CONTEXT_LM for c, n in reqs)
        ks, ke, S = _packed(reqs, r)
        assert S <= _LONG_CONTEXT_S_MAX
        if S == 0:
            stats["skipped (S 0)"] += 1
            continue
        T = ks.numel()
        qi, ki, inputs = pal.draw(T, S, _gen(8000 + i))
        tiles = _long_context_tiles(T, S)
        chk = _long_context_sample_tiles(tiles, rng)
        stats["visible checked"] += _long_context_check_tiles(pal, qi, ki, inputs, ks, ke, chk)
        stats["steps"] += 1
        stats[f"ratio {r}"] += 1
        stats["rows"] += T
        stats["tiles"] += len(tiles)
        stats["tiles checked"] += len(chk)
        stats["rows checked"] += sum(b - a for a, b in chk)
        stats["steps with S > 1M"] += S > 1 << 20
        stats["steps with S > 2^24"] += S > 1 << 24
        stats["max S"] = max(stats["max S"], S)
        stats["max k"] = max(stats["max k"], len(reqs))
        stats["max tiles"] = max(stats["max tiles"], len(tiles))
        del inputs
    print(f"long-context partitions block {block}: {dict(stats)}")
    assert stats["steps"] + stats["skipped (S 0)"] == per


def _long_context_tail_step(rng):
    """A decoder-replay step's context requests in batch order, (kv_len, has tail rows): k in 1..128 bounded
    requests finishing their prompt here (kv_len = the prompt, log-uniform in 129..Lm, or in a quarter of the steps
    uniform in Lm/2..Lm), with m in 0..32 more context requests that have no tail rows here (kv_len log-uniform in
    1..Lm); scaled down to s_len <= 2^25."""
    k = rng.randint(1, 128) if rng.random() < 0.5 else min(128, int(2 ** rng.uniform(0, 7.01)))
    m = rng.randint(0, 32) if rng.random() < 0.5 else 0
    if rng.random() < 1 / 4:
        reqs = [(rng.randint(_LONG_CONTEXT_LM // 2, _LONG_CONTEXT_LM), True) for _ in range(k)]
    else:
        reqs = [
            (max(129, int(2 ** rng.uniform(math.log2(129), math.log2(_LONG_CONTEXT_LM)))), True)
            for _ in range(k)
        ]
    reqs += [(int(2 ** rng.uniform(0, math.log2(_LONG_CONTEXT_LM))), False) for _ in range(m)]
    rng.shuffle(reqs)
    total = sum(n for n, _ in reqs)
    if total > _LONG_CONTEXT_S_MAX:
        f = (_LONG_CONTEXT_S_MAX - 129 * len(reqs)) / total
        reqs = [(max(129 if tail else 1, int(n * f)), tail) for n, tail in reqs]
    return reqs


def _long_context_tail_rows(reqs):
    """The tail rows' ranges: each bounded request's last 128 positions p see [cu_b, cu_b + p + 1) of the gather."""
    ks, ke, off = [], [], 0
    for n, tail in reqs:
        if tail:
            p = torch.arange(n - 128, n, device=DEV)
            ks.append(torch.full((128,), off, dtype=torch.int32, device=DEV))
            ke.append((off + p + 1).to(torch.int32))
        off += n
    return torch.cat(ks), torch.cat(ke), off


@pytest.mark.parametrize("block", range(2))
def test_long_context_tail_cand_steps(block):
    """The decoder-replay tail form (``deepgemm_fp8_fp4_mqa_logits.tail_cand``): 100 seeded steps (50 per block), 128
    rows per bounded request against every ratio-1 key of the step, tiled as the target tiles them; up to 16 tiles
    per step checked, every element, against the palette reference."""
    pal = _LongContextPalette(seed=930 + block)
    stats = collections.Counter()
    for i in range(block * 50, (block + 1) * 50):
        rng = random.Random(9500 + i)
        reqs = _long_context_tail_step(rng)
        ks, ke, S = _long_context_tail_rows(reqs)
        assert S <= _LONG_CONTEXT_S_MAX and all(n <= _LONG_CONTEXT_LM for n, _ in reqs)
        T = ks.numel()
        qi, ki, inputs = pal.draw(T, S, _gen(9600 + i))
        tiles = _long_context_tiles(T, S)
        chk = _long_context_sample_tiles(tiles, rng)
        stats["visible checked"] += _long_context_check_tiles(pal, qi, ki, inputs, ks, ke, chk)
        stats["steps"] += 1
        stats["tail requests"] += T // 128
        stats["tiles checked"] += len(chk)
        stats["rows checked"] += sum(b - a for a, b in chk)
        stats["max rows"] = max(stats["max rows"], T)
        stats["steps with S > 2^24"] += S > 1 << 24
        stats["max S"] = max(stats["max S"], S)
        del inputs
    print(f"tail-cand steps block {block}: {dict(stats)}")
    assert stats["steps"] == 50


def test_long_context_dumped_tiles():
    """Every (rows, s_len, r) of the long-context runs' dumps as one tile: the last ``rows`` positions of a request
    whose last row sees key s_len - 1, against the palette reference."""
    pal = _LongContextPalette(seed=950)
    cells = _long_context_dumped_cells()
    n_vis = 0
    for i, (rows, S, r) in enumerate(cells):
        ks, ke = _chunk(rows, S, r)
        qi, ki, inputs = pal.draw(rows, S, _gen(9900 + i))
        n_vis += _long_context_check_tiles(pal, qi, ki, inputs, ks, ke, [(0, rows)])
        del inputs
    print(
        f"dumped tiles: {len(cells)} cells, s_len up to {max(S for _, S, _ in cells)}, {n_vis} visible checked"
    )
    assert len(cells) == 471


def test_long_context_kernel_variant_map():
    """The tiles of the long-context form launch the same one kernel, with the same template arguments, as the
    row and key axes' variant map: heights 1, 4, 7, the tallest and one less, and the last tile of a 16384-row step,
    at every row-axis s_len up to 2^25, packed ranges."""
    q, q_sf, kv, kv_sf, w_raw = _exact_inputs(
        _LONG_CONTEXT_TC, _LONG_CONTEXT_S_MAX, _HEADS, seed=44
    )
    w = w_raw * GLOBAL_FOLD
    cases = []
    for S in _LONG_CONTEXT_S_ROWS:
        a, b = _long_context_tiles(_LONG_CONTEXT_TC, S)[-1]
        for T in sorted(
            {1, 4, 7, _long_context_rows_max(S), max(1, _long_context_rows_max(S) - 1), b - a}
        ):
            cases.append((T, S, *_packed3(T, S)))
    _call(
        q[:4], q_sf[:4], kv, kv_sf, w[:4], *_chunk(4, _LONG_CONTEXT_S_MAX, 1)
    )  # JIT, if any, outside the profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for T, S, ks, ke in cases:
            out = _call(q[:T], q_sf[:T], kv[:S], kv_sf[:S], w[:T], ks, ke)
            del out
        torch.cuda.synchronize()
    names = collections.Counter(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )
    for name, n in names.items():
        print(f"kernel x{n}: {name}")
    print(
        f"long-context variant map: {len(cases)} calls, S {min(c[1] for c in cases)}..{max(c[1] for c in cases)}"
    )
    assert set(names.values()) == {len(cases)}, names
    assert len(names) == 1, names
    (name,) = names
    assert name.startswith(_LOGITS_KERNEL), name


def test_long_context_wrong_answers_are_seen():
    """The palette check fails for each mistake a long gather could hide. The step: the last 63 positions of an
    Lm-token request whose keys end a 2^25-key gather; its last 7-row tile. Wrong expectations: ranges one key late
    (only the range edges differ), keys read one row late, key index wrapped at 2^24, the request's gather offset
    ignored, the tile's row offset ignored."""
    pal = _LongContextPalette(seed=960)
    S, off = _LONG_CONTEXT_S_MAX, _LONG_CONTEXT_S_MAX - _LONG_CONTEXT_LM
    p = torch.arange(_LONG_CONTEXT_LM - 63, _LONG_CONTEXT_LM, device=DEV)
    ks = torch.full((63,), off, dtype=torch.int32, device=DEV)
    ke = (off + p + 1).to(torch.int32)
    qi, ki, inputs = pal.draw(63, S, _gen(961))
    a, b = _long_context_tiles(63, S)[-1]
    assert (a, b) == (56, 63)
    q, q_sf, kv, kv_sf, w = inputs
    out = _call(q[a:b], q_sf[a:b], kv, kv_sf, w[a:b], ks[a:b], ke[a:b])
    assert torch.equal(out, pal.expected(qi[a:b], ki, ks[a:b], ke[a:b]))
    wrapped = ki[torch.arange(S, device=DEV) % (1 << 24)]
    wrong = {
        "ranges one key late": pal.expected(qi[a:b], ki, ks[a:b] + 1, ke[a:b] + 1),
        "keys read one row late": pal.expected(qi[a:b], ki.roll(1), ks[a:b], ke[a:b]),
        "key index wrapped at 2^24": pal.expected(qi[a:b], wrapped, ks[a:b], ke[a:b]),
        "gather offset ignored": pal.expected(qi[a:b], ki, ks[a:b] - off, ke[a:b] - off),
        "tile row offset ignored": pal.expected(qi[: b - a], ki, ks[a:b], ke[a:b]),
    }
    for name, exp in wrong.items():
        n = int((out != exp).sum())
        print(
            f"control [{name}]: {n} of {out.numel()} elements differ ({int((ke[a:b] - ks[a:b]).sum())} visible)"
        )
        assert n > 0
        del exp
