# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_top_k_ragged_deterministic entry (FlashInfer ragged top-k in its
deterministic mode, ties at the k-th value to the smaller column), at the DeepSeek-V4.1 target's two
index top-k calls: the context rows (per-request column ranges ``[ks, ke)`` over the prefill logits)
and the generation rows (per-row visible lengths over the paged logits, next_n 1 and 6), over the
envelope of up to 16384 tokens and 256 requests per step, max_seq_len up to 1048576 (DSpark on and
off), and CUDA-graph batches 1..32, 40..128 step 8, 192 and 256.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate. The
reference ranks each row's own range by the op's order (ties go to the smaller column), takes its
first 512 columns sorted ascending, then -1; the entry must equal it exactly, element for element,
including the order and the -1 padding. Because that answer is a function of the scores alone, the
same test also shows the output is bitwise identical across calls, across fresh processes and
across GPUs. This file turns a cell's spec into inputs and keeps the checks a cell cannot carry:
the CUDA-graph capture set, the mixed-step views, the row-prefix sweep of the 16384-row step, the
seeded random and dumped steps, the cross-process and cross-GPU digests, the first call inside a
capture, the wrong controls, the guards and the rejections.

The cross-process checks are plain two-device checks, not a collective: fresh processes, each on one
GPU, recompute digests that this process compares. Run as a script (``python <this file> --digests
<out.json>``) it writes the digests the cross-process test compares; ``--capture-digests
<out.json>`` writes the replay digests of the CUDA-graph capture cases the cross-process capture
test compares; ``--first-call-in-capture <out.json>`` makes the process's first call inside a CUDA
graph capture.
"""

import base64
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import zlib

import pytest
import torch
from flashinfer import top_k_ragged_transform
from flashinfer.topk import TopKTieBreak
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.flashinfer_top_k_ragged_deterministic import (  # noqa: E501
    flashinfer_top_k_ragged_deterministic as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_top_k_ragged_deterministic requires a CUDA device"

DEV = "cuda"
K = 512
# The two kernels one call launches on the pinned path (deterministic, ties SMALL, dsa_graph_safe):
# FlashInfer's one-block-per-row filtered top-k with scalar loads, then the per-row ascending sort.
KERNELS = (
    "flashinfer::sampling::FilteredTopKUnifiedKernel<float, int, 1, true, "
    "(flashinfer::sampling::FilteredTopKMode)2, (flashinfer::sampling::TopKTieBreak)1,",
    "flashinfer::sampling::FinalizeTopKIndicesKernel<true, (flashinfer::sampling::FilteredTopKMode)2, 64u, 8u,",
)


# ─── reference, inputs, calls ──────────────────────────────────────────────────────────────────
# Scores the reference ranks (and the generators fill) per chunk of rows: [16384, 1048590] fp32 is
# 64 GiB, so every whole-matrix temporary is bounded by working on row chunks of at most this many
# elements. Every shape at or below the 8192-token envelope fits one chunk.
_CHUNK = 1 << 27


def _row_step(w):
    return max(1, _CHUNK // max(w, 1))


def _reference(win, lens):
    """int64 [n, K]: ``op.reference`` over the window ``win`` (each row's range from column 0, no
    offsets): row i = the K largest of win[i, :lens[i]] with ties to the smaller column, sorted
    ascending; a row with lens <= K is 0 .. lens - 1 and then -1."""
    return op.reference(win, _zeros(win.shape[0]), lens, None, K).long()


def _control_reference(win, lens, large=False):
    """A ranking by float value rather than by the op's bit-pattern order (-0.0 ties +0.0), ties to
    the smaller column, or with ``large`` toward the larger column: the source of two controls."""
    n, w = win.shape
    L = lens.long().clamp(min=0)
    pos = torch.arange(K, device=DEV)
    ref = torch.where(pos[None, :] < L[:, None], pos[None, :], -1)
    rows = (L > K).nonzero().flatten()
    col = torch.arange(w, device=DEV)[None, :]
    step = _row_step(w)
    for a in range(0, rows.numel(), step):
        rr = rows[a : a + step]
        x = torch.where(col < L[rr][:, None], win[rr].float(), float("-inf"))
        if large:
            x = x.flip(1)
        order = torch.sort(x, dim=1, descending=True, stable=True).indices[:, :K]
        if large:
            order = w - 1 - order
        ref[rr] = order.sort(1).values
        del x, order
    return ref


def _kth_ties(win, lens, ref):
    """Per row: (in-range columns equal to the K-th selected score, selected columns equal to it)."""
    L = lens.long().clamp(min=0)
    n, w = win.shape
    col = torch.arange(w, device=DEV)[None, :]
    eq_in = torch.zeros(n, dtype=torch.int64, device=DEV)
    eq_sel = torch.zeros(n, dtype=torch.int64, device=DEV)
    step = _row_step(w)
    for a in range(0, n, step):
        rs = slice(a, a + step)
        vals = torch.gather(win[rs].float(), 1, ref[rs].clamp(min=0))
        kth = torch.where(ref[rs] >= 0, vals, float("inf")).min(1).values
        eq_in[rs] = ((win[rs] == kth[:, None]) & (col < L[rs, None])).sum(1)
        eq_sel[rs] = ((vals == kth[:, None]) & (ref[rs] >= 0)).sum(1)
    return eq_in, eq_sel


def _straddling_rows(win, lens, ref):
    """Rows where the K-th selected score is tied with an in-range score that was left out."""
    sel = lens.long() > K
    if not bool(sel.any()):
        return 0
    eq_in, eq_sel = _kth_ties(win, lens, ref)
    return int(((eq_in > eq_sel) & sel).sum())


def _max_tied(win, lens, ref):
    """The largest number of in-range columns tied with a selecting row's K-th selected score."""
    eq_in, _ = _kth_ties(win, lens, ref)
    return int(torch.where(lens.long() > K, eq_in, 0).max())


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _distinct(n, w, seed):
    """fp32 [n, w], each row a random permutation of distinct values of both signs."""
    perm = torch.argsort(torch.rand(n, w, generator=_gen(seed), device=DEV), dim=1, stable=True)
    return (perm.float() - w // 2) * 0.125


def _tie_at_kth(n, w, lens, seed):
    """fp32 [n, w]. In every row with lens > K, in random in-range columns: `above` in 0..511
    distinct scores above 5, then `m` exact copies of 5.0 with above < K < above + m (so the tie at
    the K-th value straddles the cut), the rest distinct and below 5."""
    g = _gen(seed)
    L = lens.long().clamp(min=0)
    col = torch.arange(w, device=DEV)[None, :]
    u = torch.where(col < L[:, None], torch.rand(n, w, generator=g, device=DEV), 2.0)
    rank = torch.argsort(torch.argsort(u, dim=1, stable=True), dim=1, stable=True)
    above = torch.randint(0, K, (n,), generator=g, device=DEV)
    extra = torch.randint(1, 65, (n,), generator=g, device=DEV)
    m = torch.minimum(K - above + extra, L - above).clamp(min=0)
    return torch.where(
        rank < above[:, None],
        100.0 + rank.float(),
        torch.where(rank < (above + m)[:, None], 5.0, -1.0 - u),
    )


KINDS = ("distinct", "bf16", "levels", "zero_plateau", "tie_at_kth")


def _scores(kind, n, w, lens, seed):
    """fp32 [n, w] scores of one kind: distinct; bf16-rounded normals and 8 integer levels (heavy
    exact ties everywhere); an exact-zero plateau (relu(x - 3): ~0.1 % positive, so the K-th value of
    every selecting row is 0.0, tied with thousands of columns); an exact tie constructed to
    straddle the K-th value."""
    if kind == "distinct":
        return _distinct(n, w, seed)
    if kind == "tie_at_kth":
        return _tie_at_kth(n, w, lens, seed)
    g = _gen(seed)
    if kind == "levels":
        return torch.randint(0, 8, (n, w), generator=g, device=DEV).float()
    x = torch.randn(n, w, generator=g, device=DEV)
    if kind == "bf16":
        return x.to(torch.bfloat16).float()
    assert kind == "zero_plateau"
    return torch.relu(x - 3.0)


def _mixed(n, w, lens, seed):
    """Rows cycling through the five kinds."""
    out = torch.empty(n, w, device=DEV)
    for j, kind in enumerate(KINDS[:n]):
        rows = torch.arange(j, n, len(KINDS), device=DEV)
        out[rows] = _scores(kind, rows.numel(), w, lens[rows], seed * 8 + j)
    return out


def _zeros(n):
    return torch.zeros(n, dtype=torch.int32, device=DEV)


def _call(scores, lens, row_starts=None, offsets=None):
    return op(scores, _zeros(scores.shape[0]) if offsets is None else offsets, lens, row_starts, K)


def _flashinfer_kernels(prof):
    """Names of the FlashInfer kernels launched inside a profiler session (the test's own
    reference math launches only torch kernels)."""
    return [
        e.name for e in prof.events() if e.device_type.name == "CUDA" and "flashinfer" in e.name
    ]


def _profiled(fn, names):
    """fn() in its own profiler session, appending the FlashInfer kernels it launched to `names`
    (for loops whose own torch work would make one session too large to parse)."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    names.extend(_flashinfer_kernels(prof))
    return out


def _assert_variants(prof, n_calls, what):
    """`prof` is a profiler session or the kernel names `_profiled` collected."""
    names = prof if isinstance(prof, list) else _flashinfer_kernels(prof)
    kinds = sorted(set(names))
    print(f"[{what}] kernel-variant map over {n_calls} calls: {len(names)} launches of {kinds}")
    assert len(kinds) == 2 and all(any(k in name for name in kinds) for k in KERNELS), kinds
    assert len(names) == 2 * n_calls, (len(names), n_calls)


def _sha(t):
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
    ).hexdigest()


# ─── the call's semantics ──────────────────────────────────────────────────────────────────────


def _drive_semantics():
    """Short rows (length -3, 0, 1, 511, 512) are 0 .. len-1 then -1; long rows are the reference
    exactly; offsets are added to valid entries only; row_starts makes the columns window-local.
    The result is a fresh contiguous int32 [N, 512]."""
    n, w = 96, 4096
    g = torch.Generator().manual_seed(1)
    rs = torch.randint(0, 2048, (n,), generator=g)
    lens = torch.minimum(torch.randint(-3, 4097, (n,), generator=g), w - rs)
    lens[:8] = torch.tensor([-3, 0, 1, 511, 512, 513, 1000, 2048])
    rs[:8] = torch.tensor([0, 5, 7, 100, 3, 0, 9, 2048])
    rs, lens = rs.int().to(DEV), lens.int().to(DEV)
    win = _mixed(n, w, lens, seed=11)  # row i's range in window coordinates
    x = torch.stack([win[i].roll(int(rs[i])) for i in range(n)])  # x[i, rs + j] = win[i, j]
    off = torch.randint(0, 1 << 20, (n,), generator=g).int().to(DEV)
    out = _call(x, lens, row_starts=rs, offsets=off)
    assert out.dtype == torch.int32 and out.shape == (n, K) and out.is_contiguous()
    assert out.data_ptr() != x.data_ptr()
    op.compare(out, op.reference(x, off, lens, rs, K))
    # offsets are added to valid entries only: the window-local answer plus each row's offset
    ref = _reference(win, lens)
    assert torch.equal(out.long(), torch.where(ref >= 0, ref + off[:, None].long(), -1))
    out0 = _call(x, lens.clamp(max=w))  # row_starts None: ranges start at column 0
    op.compare(out0, op.reference(x, _zeros(n), lens.clamp(max=w), None, K))
    print(f"straddling tie rows among {n}: {_straddling_rows(win, lens, ref)}")


# ─── generation rows: the paged-logits call at every engine width and row count ───────────────
# The target calls the entry once per index-source layer in every step with generation rows:
# scores = the paged-logits output [rows, W] (row stride ceil256(W), unspecified past each row's
# length), or at W > 16384 on re-index layers the contiguous torch.where(candidate_mask, logits,
# -inf) copy; lengths = compressed_visible_lens_cuda[r][tc:T] ((p + 1) // r per row); row_starts
# None; offsets = zeros. W is the metadata's static max_compressed_kv_lens[r]. Rows: at most 256
# requests x next_n 6 = 1536.
FORMS = [
    (c.spec["width"], c.spec["ratio"], c.spec["layout"])
    for c in op.CELLS
    if c.spec["kind"] == "gen"
]
ENGINE_WIDTHS = list(dict.fromkeys((w, r) for w, r, _ in FORMS))
MAX_SEQ_LENS = [16384, 32768, 65536, 131072, 262144, 1048576]
MAX_ROWS = 1536
FORM_IDS = [f"w{w}_r{r}_{f}" for w, r, f in FORMS]
assert len(ENGINE_WIDTHS) == 40 and len(FORMS) == 59


def _dspark_vis(batch, ratio, width, g, padding):
    """Per-row visible lengths (p + j + 1) // ratio of a verify step (next_n 6): request 0 at the cap
    (its last row sees exactly `width`), request 1 at position 0 or 1, the rest random, and with
    `padding` a graph padding request (all 0) last."""
    cap = width * ratio - 6
    p = torch.randint(0, cap + 1, (batch,), generator=g)
    p[0] = cap
    if batch > 1:
        p[1] = int(torch.randint(0, 2, (1,), generator=g))
    r = torch.arange(batch * 6)
    vis = (p[r // 6] + r % 6 + 1) // ratio
    if padding and batch > 2:
        vis[-6:] = 0
    return vis.to(torch.int32).to(DEV)


def _decode_vis(batch, ratio, width, g, padding):
    """Per-row visible lengths (p + 1) // ratio of a next_n 1 step: request 0 at the cap, request 1
    at position 0 or 1, one straddling K, the rest random, and with `padding` a 0 row last."""
    p = torch.randint(0, width * ratio, (batch,), generator=g)
    p[0] = width * ratio - 1
    if batch > 1:
        p[1] = int(torch.randint(0, 2, (1,), generator=g))
    if batch > 2:
        p[2] = K * ratio
    vis = (p + 1) // ratio
    if padding and batch > 3:
        vis[-1] = 0
    return vis.to(torch.int32).to(DEV)


def _candidate_keep(vis, width, g):
    """A re-index layer's keep mask [rows, width]: a row with more than 2048 blocks of 8 keeps its
    newest block plus 2047 older ones, a shorter row keeps everything visible; False past the row."""
    rows = vis.numel()
    nb_all = -(-width // 8)
    L = vis.long()
    nb = (L + 7) // 8
    blk = torch.arange(nb_all, device=DEV)[None, :]
    older = blk < (nb - 1)[:, None]
    s = torch.where(older, torch.rand(rows, nb_all, generator=g, device=DEV), -1.0)
    keep = torch.zeros(rows, nb_all, dtype=torch.bool, device=DEV)
    keep.scatter_(1, s.topk(min(2047, nb_all), dim=1).indices, True)
    keep = (keep & older) | (blk == (nb - 1)[:, None])
    keep = torch.where((nb <= 2048)[:, None], blk < nb[:, None], keep)
    col = torch.arange(width, device=DEV)[None, :]
    return keep.repeat_interleave(8, 1)[:, :width] & (col < L[:, None])


def _decode_inputs(width, form, vis, seed, out=None, kind="mixed"):
    """The target's generation-row scores for lengths `vis`: the paged-logits view (NaN past every
    row's length and in the stride padding) or the candidate-masked contiguous copy (-inf outside
    the kept blocks). Returns (the entry's scores, the scores the reference ranks). With `out`
    (a (paged buffer, masked buffer) pair) it writes in place."""
    rows = vis.numel()
    base = (
        _mixed(rows, width, vis, seed) if kind == "mixed" else _scores(kind, rows, width, vis, seed)
    )
    col = torch.arange(width, device=DEV)[None, :]
    stride = -(-width // 256) * 256
    buf = torch.full((rows, stride), float("nan"), device=DEV) if out is None else out[0]
    if out is not None:
        buf.fill_(float("nan"))
    buf[:, :width] = base.masked_fill(col >= vis[:, None].long(), float("nan"))
    if form == "paged_view":
        return buf[:, :width], base
    masked = torch.where(_candidate_keep(vis, width, _gen(seed)), base, float("-inf"))
    assert masked.stride() == (width, 1)
    if out is not None:
        out[1].copy_(masked)
        masked = out[1]
    return masked, masked


def _drive_gen(width, ratio, form):
    """Every row count of the generation rows: the 1536-row call (256 DSpark verify requests, row 0 at
    the full width, mixed tie kinds) equals the reference exactly, and a call on the first r rows
    equals the first r rows of it bitwise for every r in 1..1536 (one block per row: rows only size
    the grid). A next_n 1 step (1536 one-row requests) is exact too. Every call launches the same two
    kernels. Control: lengths one past each row's own read a NaN column and are seen."""
    seed = width + 13 * ratio + len(form)
    g = torch.Generator().manual_seed(seed)
    vis = _dspark_vis(MAX_ROWS // 6, ratio, width, g, padding=False)
    vis[0] = width
    scores, ranked = _decode_inputs(width, form, vis, seed)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        full = _call(scores, vis)
        prefix = [_call(scores[:r], vis[:r]) for r in range(1, MAX_ROWS + 1)]
        torch.cuda.synchronize()
    ref = op.reference(scores, _zeros(MAX_ROWS), vis, None, K)
    op.compare(full, ref)
    ref = ref.long()
    bad = [r for r, out in enumerate(prefix, 1) if not torch.equal(out, full[:r])]
    assert not bad, f"prefixes differing from the {MAX_ROWS}-row call: {bad[:10]}"
    del prefix
    _assert_variants(prof, MAX_ROWS + 1, f"sweep w{width} r{ratio} {form}")
    vis1 = _decode_vis(MAX_ROWS, ratio, width, g, padding=True)
    s1, _ = _decode_inputs(width, form, vis1, seed + 1)
    op.compare(_call(s1, vis1), op.reference(s1, _zeros(MAX_ROWS), vis1, None, K))
    print(
        f"[sweep w{width} r{ratio} {form}] stride {scores.stride()}; straddling tie rows "
        f"{_straddling_rows(ranked, vis, ref)} of {MAX_ROWS}"
    )
    if form == "paged_view":
        over = _call(scores, (vis + 1).clamp(max=width))
        wrong = int((over.long() != ref).any(1).sum())
        print(f"[sweep w{width}] control lengths + 1: rows differing {wrong} of {MAX_ROWS}")
        assert wrong > 0


# Every CUDA-graph batch of the envelope, per rank: Gset = every batch 1..32 plus
# TRT-LLM's padded capture list at cuda_graph_config max_batch_size 256 (1, 2, 4, 8..128 step 8,
# 192, 256; the padding-off list 1..32, 64, 128, 256 is a subset), each as decode (next_n 1: G rows)
# and as DSpark verify (next_n 6: 6G rows), at every form above (all 40 static widths, both layouts).
GSET = list(range(1, 33)) + [40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256]
assert len(GSET) == 46
GRAPH_CASES = [(w, r, f, n, b) for w, r, f in FORMS for n in (1, 6) for b in GSET]
GRAPH_IDS = [f"w{w}_r{r}_{f}_nextn{n}_G{b}" for w, r, f, n, b in GRAPH_CASES]


_FP_WEIGHTS = {}


def _fingerprint(t):
    """Digest of a 2-D tensor's bit patterns, reduced on the GPU (the capture inputs reach 6.4 GB):
    per row, the sum of its 32-bit words times seeded random odd 64-bit weights, wrapping modulo
    2**64 (so the sum's order does not matter), then SHA-256 over the row values."""
    rows, cols = t.shape
    words = t.view(torch.int32)
    if cols not in _FP_WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(20260926)
        _FP_WEIGHTS[cols] = torch.randint(-(2**62), 2**62, (cols,), generator=g, device=DEV) | 1
    w = _FP_WEIGHTS[cols]
    fp = torch.empty(rows, dtype=torch.int64, device=DEV)
    step = _row_step(cols)
    for a in range(0, rows, step):
        fp[a : a + step] = (words[a : a + step].to(torch.int64) * w).sum(1)
    return _sha(fp)


def _capture_case(width, ratio, form, next_n, batch, digests=True):
    """One case of the capture set: an eager call at the shape, the call captured, then 3 replays on
    new lengths (0/1, the width, a padding request) and new mixed-tie scores written in place.
    Returns (whether the eager call equals the reference, one record per replay: with `digests`,
    digests of the inputs the op reads (fingerprint of the scores buffer, SHA-256 of the lengths)
    and of the replay's output (SHA-256); whether the replay equals the reference and an eager
    call on the same inputs bitwise, and the rows where it differs from the previous step's
    reference)."""
    rows = batch * next_n
    seed = (
        width
        + 7 * ratio
        + 1000 * batch
        + 100000 * next_n
        + (1 if form == "candidate_masked" else 0)
    )
    g = torch.Generator().manual_seed(seed)
    stride = -(-width // 256) * 256
    bufs = (torch.empty(rows, stride, device=DEV), torch.empty(rows, width, device=DEV))
    static, read = (bufs[0][:, :width], bufs[0]) if form == "paged_view" else (bufs[1], bufs[1])
    lens = torch.empty(rows, dtype=torch.int32, device=DEV)
    off = _zeros(rows)
    make_vis = _dspark_vis if next_n == 6 else _decode_vis

    def load(it):
        vis = make_vis(batch, ratio, width, g, padding=it % 2 == 1)
        _, ranked = _decode_inputs(width, form, vis, seed * 10 + it, out=bufs)
        lens.copy_(vis)
        return _reference(ranked, vis)

    prev = load(0)
    assert int(lens.max()) == width
    eager0 = bool(torch.equal(op(static, off, lens, None, K).long(), prev))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(static, off, lens, None, K)
    replays = []
    for it in range(1, 4):
        ref = load(it)
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        got = out.clone()
        replays.append(
            {
                **(
                    {
                        "in": hashlib.sha256(
                            (_fingerprint(read) + _sha(lens)).encode()
                        ).hexdigest(),
                        "out": _sha(got),
                    }
                    if digests
                    else {}
                ),
                "exact": bool(torch.equal(got.long(), ref)),
                "eager": bool(torch.equal(got, op(static, off, lens, None, K))),
                "stale": int((got.long() != prev).any(1).sum()),
            }
        )
        prev = ref
    del graph
    return eager0, replays


@pytest.mark.parametrize("width,ratio,form,next_n,batch", GRAPH_CASES, ids=GRAPH_IDS)
def test_cuda_graph_engine_capture_set(width, ratio, form, next_n, batch):
    """The capture set at every form: the call captured after one eager call at its shape, decode
    (next_n 1, G rows) and DSpark verify (next_n 6, 6G rows) at every G in Gset. Then 3 replays
    on new lengths (0/1, the width, a padding request) and new mixed-tie scores written in place.
    Each replay equals the reference exactly and an eager call on the same inputs bitwise, and
    differs from the previous step's reference (a replay re-reading the captured step is seen)."""
    rows = batch * next_n
    eager0, replays = _capture_case(width, ratio, form, next_n, batch, digests=False)
    assert eager0
    for it, rec in enumerate(replays, 1):
        print(
            f"[graph {form} w{width} r{ratio} next_n{next_n} rows{rows} replay {it}] rows differing "
            f"from the previous step's reference: {rec['stale']} of {rows}"
        )
        assert rec["exact"], f"replay {it}"
        assert rec["eager"], f"replay {it} != eager"
        assert rec["stale"] > 0


# The cross-process capture digests: every form at G 1, 7, 40 and 256 (decode and verify; up to
# 1536 rows), plus the whole 8192-token-envelope set (G 1..8, 16, 24, 32) at its 16 forms.
_OLD_WIDTHS = set(ENGINE_WIDTHS[:11])
DIGEST_GRAPH = [
    (gid, c)
    for gid, c in zip(GRAPH_IDS, GRAPH_CASES)
    if c[4] in (1, 7, 40, 256)
    or ((c[0], c[1]) in _OLD_WIDTHS and c[4] in (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32))
]


def _capture_digests():
    """{case id: [eager call exact, replay records]} over the digest capture cases (see _capture_case)."""
    res = {}
    for gid, case in DIGEST_GRAPH:
        eager0, replays = _capture_case(*case)
        res[gid] = [eager0, replays]
        torch.cuda.empty_cache()
    return res


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_capture_set_replays_identical_across_processes_and_gpus():
    """The digest capture cases (all 59 forms at G 1, 7, 40, 256 and the 8192-token-envelope set at
    its 16 forms; 760 captures) are run again here and in 3 fresh processes on each of 2 GPUs: the
    replays' output digests are identical in all 7 processes (and every replay equals the reference
    and the eager call on the same inputs), and so are the input digests, so no difference could
    hide behind a changed input."""
    torch.cuda.empty_cache()  # the fresh processes share this GPU
    local = _capture_digests()
    assert len(local) == 760
    bad = [
        k
        for k, (e0, recs) in local.items()
        if not (e0 and all(r["exact"] and r["eager"] for r in recs))
    ]
    assert not bad, bad[:10]
    devs = _visible_devices()
    assert len(devs) >= 2, f"this test needs 2 GPUs, got CUDA_VISIBLE_DEVICES={devs}"
    tmp = tempfile.mkdtemp(prefix="fitopk_")
    procs = []
    for d in devs[:2]:
        for j in range(3):
            path = os.path.join(tmp, f"capture_{d}_{j}.json")
            env = _script_env(d)
            p = subprocess.Popen(
                [sys.executable, __file__, "--capture-digests", path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            procs.append((d, j, path, p))
    uuids = {}
    for d, j, path, p in procs:
        log, _ = p.communicate()
        assert p.returncode == 0, f"capture-digest process {d}/{j} failed:\n{log[-3000:]}"
        with open(path) as f:
            got = json.load(f)
        uuids.setdefault(d, set()).add(got["uuid"])
        differing = sorted(k for k in local if got["cases"].get(k) != local[k])
        print(
            f"process {j} on GPU {d} ({got['name']} {got['uuid']}): {len(got['cases'])} capture cases, "
            f"differing from this process: {differing[:10]}"
        )
        assert not differing and set(got["cases"]) == set(local)
    assert len({u for s in uuids.values() for u in s}) == 2, uuids
    n_replays = sum(len(recs) for _, recs in local.values())
    print(
        f"{len(local)} capture cases ({n_replays} replays) bitwise identical here and in 6 fresh processes "
        f"on GPUs {uuids}"
    )


# ─── mixed steps: the target's views ──────────────────────────────────────────────────────────
MAX_TOKENS = 16384  # the envelope's max_num_tokens: the metadata's per-token buffers hold this many


@pytest.mark.parametrize("tc_at", ["1", "2", "573", "7674", "8193", "max-1", "max"])
@pytest.mark.parametrize("next_n", [1, 6])
@pytest.mark.parametrize("gen_reqs", [32, 256])
def test_mixed_step_views(tc_at, next_n, gen_reqs):
    """In a step with tc context tokens and gen = G x next_n generation rows (G 32 or 256; T = tc +
    gen <= 16384) the target passes lengths as views of the metadata's 16384-entry
    compressed_visible_lens_cuda[r] ([:tc] for the context call, [tc:T] for the generation call, whose
    base is only 4-byte aligned at odd tc) and offsets as [:rows] views of one 16384-entry zero
    buffer. Both calls equal the same calls on contiguous copies bitwise, and neither buffer is
    written."""
    gen = gen_reqs * next_n
    tc = {"max": MAX_TOKENS - gen, "max-1": MAX_TOKENS - 1 - gen}.get(tc_at) or int(tc_at)
    T = tc + gen
    width, ratio = 16398, 1
    g = torch.Generator().manual_seed(tc * 7 + next_n + gen_reqs)
    vis_all = torch.randint(0, 8192, (MAX_TOKENS,), generator=g).int().to(DEV)
    vis_all[:tc] = (
        torch.arange(tc, device=DEV) + 1
    ).int()  # one context request at positions 0 .. tc-1
    vis_all[tc:T] = (
        _dspark_vis(gen_reqs, ratio, width, g, padding=False)
        if next_n == 6
        else _decode_vis(gen_reqs, ratio, width, g, padding=False)
    )
    zeros_all = _zeros(MAX_TOKENS)
    keep_v, keep_z = vis_all.clone(), zeros_all.clone()
    s_gen, _ = _decode_inputs(width, "paged_view", vis_all[tc:T].clone(), seed=tc)
    got = op(s_gen, zeros_all[:gen], vis_all[tc:T], None, K)
    assert torch.equal(got, op(s_gen, _zeros(gen), vis_all[tc:T].clone(), None, K))
    ks = _zeros(tc)
    buf = torch.full((tc, -(-(tc + 256) // 256) * 256), float("nan"), device=DEV)
    buf[:, :tc] = torch.where(
        torch.arange(tc, device=DEV)[None, :] < vis_all[:tc, None].long(),
        _mixed(tc, tc, vis_all[:tc], seed=tc + 1),
        float("nan"),
    )
    ctx = op(buf[:, :tc], zeros_all[:tc], vis_all[:tc], ks, K)
    assert torch.equal(ctx, op(buf[:, :tc], _zeros(tc), vis_all[:tc].clone(), ks.clone(), K))
    assert torch.equal(ctx.long(), _reference(buf[:, :tc], vis_all[:tc]))
    assert torch.equal(vis_all, keep_v) and torch.equal(zeros_all, keep_z)


# ─── context rows: per-request ranges over the prefill logits ─────────────────────────────────
# The target calls the entry once per index-source layer in every step with context requests (never
# captured: a step with a context request is not CUDA-graph eligible) with scores = the prefill
# logits [tc, S] (a view with row stride align(S + 256, 256), unspecified outside each row's range),
# or at re-index layers when S > 16384 the contiguous torch.where(candidate_mask, logits, -inf)
# copy; row_starts = ks (the request's first column); lengths = compressed_visible_lens_cuda[r][:tc]
# = ke - ks = (p + 1) // r; offsets = zeros. S = sum of kv_len // r over the step's requests.


def _packed_rows(ratio, reqs):
    """The target's rows of one context step, requests as (cached, new) in batch order: request b's
    compressed entries are columns [cu_b, cu_b + kv_b // ratio) and its token at position p scores
    [ks, ke) = [cu_b, cu_b + (p + 1) // ratio). Returns (ks, ke, S)."""
    ks, ke, col0 = [], [], 0
    for cached, new in reqs:
        kv = cached + new
        p = torch.arange(cached, kv, device=DEV)
        ks.append(torch.full((new,), col0, dtype=torch.int32, device=DEV))
        ke.append((col0 + torch.div(p + 1, ratio, rounding_mode="floor")).to(torch.int32))
        col0 += kv // ratio
    return torch.cat(ks), torch.cat(ke), col0


def _packed_logits(reqs, ratio, contiguous, seed, kind="mixed"):
    """[tc, S] scores of a packed step: each row's [ks, ke) holds scores of `kind`, every other
    column NaN. contiguous=False gives the prefill-logits layout (row stride align(S + 256, 256)).
    A request whose rows x width exceed one chunk is filled a row chunk at a time."""
    ks, ke, s = _packed_rows(ratio, reqs)
    tc = ks.numel()
    stride = s if contiguous else -(-(s + 256) // 256) * 256
    buf = torch.full((tc, max(stride, 1)), float("nan"), device=DEV)
    row0 = 0
    for b, (cached, new) in enumerate(reqs):
        width = (cached + new) // ratio
        if width:
            c0 = int(ks[row0])
            col = torch.arange(width, device=DEV)[None, :]
            step = new if new * width <= _CHUNK else _row_step(width)
            for a in range(0, new, step):
                e = min(new, a + step)
                rs = slice(row0 + a, row0 + e)
                lens = ke[rs] - ks[rs]
                sd = seed * 64 + b if step == new else (seed * 64 + b) * 8192 + a // step
                vals = (
                    _mixed(e - a, width, lens, sd)
                    if kind == "mixed"
                    else _scores(kind, e - a, width, lens, sd)
                )
                buf[rs, c0 : c0 + width] = torch.where(
                    col < lens[:, None].long(), vals, float("nan")
                )
                del vals
        row0 += new
    return buf[:, :s], ks, ke


def _packed_bad_rows(logits, ks, ke, reqs, ratio, out, ref=None):
    """Rows whose output differs from the reference over each row's own [ks, ke) span (the call's
    ``op.reference``, or ``ref`` when given), and the step's straddling rows."""
    if ref is None:
        ref = op.reference(logits, _zeros(ks.numel()), ke - ks, ks, K)
    bad = int((out != ref).any(1).sum())
    straddle, row0 = 0, 0
    for cached, new in reqs:
        width = (cached + new) // ratio
        rs = slice(row0, row0 + new)
        if width > K:
            c0 = int(ks[row0])
            win = logits[rs, c0 : c0 + width]
            straddle += _straddling_rows(win, ke[rs] - ks[rs], ref[rs].long())
        row0 += new
    return bad, straddle


def _packed_check(logits, ks, ke, reqs, ratio, out):
    """The call's output gated by the entry against its reference; returns the straddling rows."""
    ref = op.reference(logits, _zeros(ks.numel()), ke - ks, ks, K)
    op.compare(out, ref)
    return _packed_bad_rows(logits, ks, ke, reqs, ratio, out, ref)[1]


def _run_packed(logits, ks, ke):
    return _call(logits, ke - ks, row_starts=ks)


# 16384 context tokens of a mixed chunked-prefill step (max_num_tokens 16384). The first 8192: a
# fresh 3000-token prompt, a chunk on an 8192-token cached prefix, 1- and 5-token contexts, contexts
# whose rows all see <= 512, and a chunk ending at position 16383. Then a fresh 4096-token prompt,
# chunks ending at positions 32767, 131071 and 65549 (long-context max_seq_len values; 65550 is the
# 64K DSpark manager length) and a 1-token context. Columns start at unaligned offsets.
FULL_CONTEXT_STEP = [
    (0, 3000),
    (8192, 2048),
    (0, 1),
    (0, 129),
    (130, 5),
    (2, 1600),
    (4000, 430),
    (15405, 979),
    (0, 4096),
    (32768 - 1000, 1000),
    (131072 - 2048, 2048),
    (3, 1),
    (65550 - 1047, 1047),
]


@pytest.mark.parametrize("ratio", [1, 2])
def test_rows_prefix_bitwise(ratio):
    """Every row count of the context rows 1..16384: the 16384-row call equals the reference exactly on
    every row, and a call on the first r rows (same scores, ks, lengths) equals the first r rows of
    that call bitwise, for every r in 1..16384. A one-row-shifted call fails the same comparison."""
    ks, ke, s = _packed_rows(ratio, FULL_CONTEXT_STEP)
    assert ks.numel() == MAX_TOKENS
    logits, ks, ke = _packed_logits(FULL_CONTEXT_STEP, ratio, contiguous=False, seed=5000 + ratio)
    lens = ke - ks
    full = _call(logits, lens, row_starts=ks)
    bad, straddle = _packed_bad_rows(logits, ks, ke, FULL_CONTEXT_STEP, ratio, full)
    assert bad == 0
    assert int(lens.min()) <= 1 and int((lens <= K).sum()) > 0 and int((lens > K).sum()) > 0
    differing = torch.zeros((), dtype=torch.int64, device=DEV)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for r in range(1, MAX_TOKENS + 1):
            differing += (_call(logits[:r], lens[:r], row_starts=ks[:r]) != full[:r]).any(1).sum()
        torch.cuda.synchronize()
    print(
        f"ratio={ratio} S={s} lengths {int(lens.min())}..{int(lens.max())}: straddling tie rows {straddle}; "
        f"prefixes r=1..{MAX_TOKENS}: rows differing from the {MAX_TOKENS}-row call {int(differing)}"
    )
    assert int(differing) == 0
    _assert_variants(prof, MAX_TOKENS, f"prefix sweep ratio {ratio}")
    for r in (1, 2, 3, 100, 513, 4096, 8191, 8192, 12000, 16383):
        shifted = _call(logits[1 : r + 1], lens[1 : r + 1], row_starts=ks[1 : r + 1])
        assert not torch.equal(shifted, full[:r]), f"shifted call at r={r} not seen"


# Every per-row length: the 16384-token chunks of one request at the 256K DSpark manager length
# (262158 tokens; the last chunk has 14), so ratio-1 rows see every length 1..262158 and ratio-2 rows
# every length 0..131079.
EVERY_LEN_LM = 262158
EVERY_LEN_CHUNKS = -(-EVERY_LEN_LM // MAX_TOKENS)


def _drive_every_len(ratio, chunk):
    """The 17 chunks of one 262158-token request: ratio-1 rows see every length 1..262158 and ratio-2
    rows every length 0..131079, exact."""
    start = chunk * MAX_TOKENS
    reqs = [(start, min(MAX_TOKENS, EVERY_LEN_LM - start))]
    logits, ks, ke = _packed_logits(reqs, ratio, contiguous=False, seed=6000 + 100 * ratio + chunk)
    out = _run_packed(logits, ks, ke)
    straddle = _packed_check(logits, ks, ke, reqs, ratio, out)
    lens = ke - ks
    print(
        f"ratio={ratio} chunk={chunk}: lengths {int(lens.min())}..{int(lens.max())}, straddling tie rows {straddle}"
    )
    assert int(lens.max()) == (start + reqs[0][1]) // ratio


def _drive_widest(ratio):
    """The widest step the gate configuration can build: 32 context requests (max_batch_size), each
    a 256-token chunk ending at max_seq_len 16384, so 8192 rows over S = 32 * 16384 // ratio columns
    in the prefill-logits layout (8192 x 524544 fp32 = 17.2 GB at ratio 1). The last row starts past
    element 2**32 at ratio 1 and past 2**31 at ratio 2. Exact."""
    reqs = [(16384 - 256, 256)] * 32
    logits, ks, ke = _packed_logits(reqs, ratio, contiguous=False, seed=7000 + ratio)
    stride = logits.stride(0)
    assert (8192 - 1) * stride > (2**32 if ratio == 1 else 2**31)
    out = _run_packed(logits, ks, ke)
    straddle = _packed_check(logits, ks, ke, reqs, ratio, out)
    print(
        f"widest step ratio={ratio}: S={logits.shape[1]} row stride={stride}, straddling tie rows {straddle}"
    )


def _ctx_candidate_keep(ks, ke, s, seed):
    """Keep mask [tc, S] of a re-index layer's context rows: a row seeing more than 16384 positions
    keeps its newest 8-wide block plus 2047 older ones, a shorter row keeps its whole range."""
    lens = (ke - ks).long()
    keep = torch.zeros(ks.numel(), s, dtype=torch.bool, device=DEV)
    col = torch.arange(s, device=DEV)[None, :]
    inside = (col >= ks[:, None].long()) & (col < ke[:, None].long())
    for c0 in ks.unique().tolist():
        rows = (ks == c0).nonzero().flatten()
        w = int(lens[rows].max())
        local = _candidate_keep(lens[rows].int(), w, _gen(seed + c0))
        keep[rows, c0 : c0 + w] = local
    return keep & inside


def _masked_copy(logits, ks, ke, seed):
    """The re-index layers' contiguous candidate-masked copy of a one-request step's logits (every
    row starts at column 0), built a row chunk at a time: a row seeing more than 16384 positions
    keeps its newest 8-wide block plus 2047 older ones, -inf elsewhere and past the row."""
    tc, s = logits.shape
    assert int(ks.max()) == 0
    lens = (ke - ks).int()
    out = torch.empty(tc, s, device=DEV)
    step = _row_step(s)
    for a in range(0, tc, step):
        e = min(tc, a + step)
        out[a:e] = torch.where(
            _candidate_keep(lens[a:e], s, _gen(seed + a)), logits[a:e], float("-inf")
        )
    return out


def _drive_long_ctx(form):
    """The 24K passkey step from the runtime dump, packed: 7674 rows x 24058 columns, a 4001-token
    chunk on a 16384-token cached prefix (every row sees more than 16384 positions), then a fresh
    3673-token prompt. In the prefill-logits layout, and as the re-index layers' contiguous
    candidate-masked copy (-inf outside the kept blocks). Exact."""
    reqs = [(16384, 4001), (0, 3673)]
    logits, ks, ke = _packed_logits(reqs, 1, contiguous=False, seed=8100)
    assert logits.shape == (7674, 24058)
    if form == "candidate_masked":
        keep = _ctx_candidate_keep(ks, ke, logits.shape[1], seed=8200)
        logits = torch.where(keep, logits, float("-inf"))
        assert logits.stride() == (24058, 1)
    out = _run_packed(logits, ks, ke)
    straddle = _packed_check(logits, ks, ke, reqs, 1, out)
    print(f"24K step {form}: straddling tie rows {straddle}")


# The max_seq_len axis's far end: the last 16384-token chunk of one request at each manager length Lm
# of the envelope (L + 14 with DSpark, L without), so its rows see up to Lm // r compressed positions,
# in the prefill-logits layout and, at ratio 1 above 16384 positions, as the re-index layers'
# candidate-masked copy.
# Context row counts in the max_num_tokens-16384 engine dumps (prefill top-k records).
DUMPED_CTX_ROWS = [
    1,
    2,
    5,
    12,
    16,
    28,
    128,
    597,
    2252,
    4553,
    8147,
    8192,
    8317,
    9055,
    9108,
    9392,
    10441,
    10444,
    10451,
    16292,
    16327,
    16328,
    16378,
    16384,
]


def _relaxed_prefix_rows(n):
    """The row counts swept where a call takes over 1 ms, in place of every row count: every r in
    1..256, every r = {0, 1, -1} mod the SM count (the kernels run one block per row, so the SM count
    sets the wave boundaries) and mod 256, up to n, and the dumped row counts."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = set(range(1, 257)) | set(DUMPED_CTX_ROWS)
    for q in (sms, 256):
        for m in range(q, n + q, q):
            rows |= {m - 1, m, m + 1}
    return sorted(r for r in rows if 1 <= r <= n)


def _drive_far_end(lm, ratio, form):
    """The last 16384-token chunk of a request at manager length Lm: 16384 rows seeing
    (Lm - 16383) // r .. Lm // r compressed positions over S = Lm // r columns (16384 x 1049088
    fp32 = 64 GiB at Lm 1048590), exact, one launch of each kernel. At the widest (Lm 1048590, ratio
    1, a call of about 0.1 s) the row-prefix set for calls over 1 ms (_relaxed_prefix_rows) is bitwise
    equal to the leading rows of the 16384-row call."""
    reqs = [(lm - MAX_TOKENS, MAX_TOKENS)]
    logits, ks, ke = _packed_logits(reqs, ratio, contiguous=False, seed=9000 + lm + ratio)
    lens = ke - ks
    assert int(lens.max()) == lm // ratio and logits.shape == (MAX_TOKENS, lm // ratio)
    if form == "candidate_masked":
        logits = _masked_copy(logits, ks, ke, seed=9100 + lm)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        full = _call(logits, lens, row_starts=ks)
        torch.cuda.synchronize()
    _assert_variants(prof, 1, f"far end Lm {lm} ratio {ratio} {form}")
    straddle = _packed_check(logits, ks, ke, reqs, ratio, full)
    print(
        f"far end Lm={lm} r={ratio} {form}: S={logits.shape[1]} row stride={logits.stride(0)} "
        f"({logits.shape[0] * logits.stride(0) * 4 / 2**30:.1f} GiB), lengths {int(lens.min())}..{int(lens.max())}, "
        f"straddling tie rows {straddle}"
    )
    if (lm, ratio, form) == (1048590, 1, "mqa_view"):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _call(logits, lens, row_starts=ks)
        end.record()
        torch.cuda.synchronize()
        rows = _relaxed_prefix_rows(MAX_TOKENS)
        differing = torch.zeros((), dtype=torch.int64, device=DEV)
        for r in rows:
            differing += (_call(logits[:r], lens[:r], row_starts=ks[:r]) != full[:r]).any(1).sum()
        print(
            f"far end Lm={lm}: the 16384-row call takes {start.elapsed_time(end):.1f} ms; {len(rows)} prefixes "
            f"(r {rows[0]}..{rows[-1]}): rows differing from the 16384-row call {int(differing)}"
        )
        assert int(differing) == 0
    del logits, full
    torch.cuda.empty_cache()


# ─── context rows in row tiles ─────────────────────────────────────────────────────────────────
# At long contexts the target runs a step's context rows in tiles (`_index_row_tiles` in the
# target's modeling.py): tile [a, b) gets its own fresh prefill-logits buffer [b - a, S] (row stride
# align(S + 256, 256)), zeros[:b - a], and the step's lengths[a:b] and ks[a:b] (views at row offset
# a), and at re-index layers the tile's candidate-masked copy. A tile holds at most 2**28 fp32
# logits: the whole step when it fits, otherwise floor(2**28 / row stride) rows, rounded down to a
# multiple of 64 when that is at least 64.
TILE_ELEMS = 1 << 28


def _row_tiles(rows, s_len):
    stride = -(-(s_len + 256) // 256) * 256
    per = max(1, TILE_ELEMS // stride)
    if per >= rows:
        return [(0, rows)]
    if per >= 64:
        per -= per % 64
    return [(a, min(rows, a + per)) for a in range(0, rows, per)]


def _tile_logits(reqs, ratio, ks, ke, a, b, seed, kind="mixed"):
    """Rows [a, b) of a packed step as a fresh prefill-logits buffer [b - a, S] (row stride
    align(S + 256, 256)): each row's [ks, ke) holds scores of `kind`, every other column NaN."""
    s = sum((c + n) // ratio for c, n in reqs)
    buf = torch.full((b - a, -(-(s + 256) // 256) * 256), float("nan"), device=DEV)
    row0 = 0
    for j, (cached, new) in enumerate(reqs):
        lo, hi, width = max(a, row0), min(b, row0 + new), (cached + new) // ratio
        if lo < hi and width:
            c0 = int(ks[lo])
            col = torch.arange(width, device=DEV)[None, :]
            step = _row_step(width)
            for x in range(lo, hi, step):
                y = min(hi, x + step)
                lens = ke[x:y] - ks[x:y]
                sd = (seed * 1024 + j) * 4096 + (x - lo) // step
                # a one-row chunk (one-row tiles) takes the kinds in turn by row, not always the first
                k = KINDS[x % len(KINDS)] if kind == "mixed" and y - x == 1 else kind
                vals = (
                    _mixed(y - x, width, lens, sd)
                    if k == "mixed"
                    else _scores(k, y - x, width, lens, sd)
                )
                buf[x - a : y - a, c0 : c0 + width] = torch.where(
                    col < lens[:, None].long(), vals, float("nan")
                )
                del vals
        row0 += new
    return buf[:, :s]


def _tile_bad_rows(logits, ks, lens, out, ref=None):
    """Rows of one call whose output differs from the reference over the row's own range (the
    call's ``op.reference``, or ``ref`` when given), and its straddling rows; rows are grouped by
    their column start (one request's rows each)."""
    n = ks.numel()
    if ref is None:
        ref = op.reference(logits, _zeros(n), lens, ks, K)
    bad = int((out != ref).any(1).sum())
    cut = ((ks[1:] != ks[:-1]).nonzero().flatten() + 1).tolist()
    straddle = 0
    for p, q in zip([0] + cut, cut + [n]):
        w, c0 = int(lens[p:q].max()), int(ks[p])
        if w > K:
            win = logits[p:q, c0 : c0 + w]
            straddle += _straddling_rows(win, lens[p:q], ref[p:q].long())
    return bad, straddle


def _run_tiles(reqs, ratio, masked, seed, only_reqs=None):
    """Every tile of a packed step (with `only_reqs`, the tiles holding those requests' rows), each
    called as the target calls it and gated by the entry; returns (tiles, rows, rows wrong,
    straddling rows, S)."""
    ks, ke, s = _packed_rows(ratio, reqs)
    lens = ke - ks
    zeros = _zeros(MAX_TOKENS)
    tiles = _row_tiles(ks.numel(), s)
    if only_reqs is not None:
        starts = [sum(n for _, n in reqs[:j]) for j in range(len(reqs))]
        keep = [(starts[j], starts[j] + reqs[j][1]) for j in only_reqs]
        tiles = [(a, b) for a, b in tiles if any(a < hi and lo < b for lo, hi in keep)]
    rows = bad = straddle = 0
    for t, (a, b) in enumerate(tiles):
        logits = _tile_logits(reqs, ratio, ks, ke, a, b, seed * 100003 + t)
        if masked:
            logits = torch.where(
                _ctx_candidate_keep(ks[a:b], ke[a:b], s, seed + a), logits, float("-inf")
            )
        out = op(logits, zeros[: b - a], lens[a:b], ks[a:b], K)
        ref = op.reference(logits, zeros[: b - a], lens[a:b], ks[a:b], K)
        op.compare(out, ref)
        bd, st = _tile_bad_rows(logits, ks[a:b], lens[a:b], out, ref)
        rows, bad, straddle = rows + b - a, bad + bd, straddle + st
        del logits, out
    return tiles, rows, bad, straddle, s


# Tiled steps (the cells' `reqs`, as (cached, new) requests): the last 16384-token chunk at the 1M
# manager lengths (192-row tiles at ratio 1, 448 at ratio 2); an 8192-token chunk ending at 1048219,
# the 1M engine dump's longest context (256-row tiles); a 16384-token prompt (S 16384: tiles of
# 16128 and 256 rows); tiles crossing request boundaries at mixed long contexts; two 8192-token
# chunks ending at 1048590 (S 2,097,180: 64-row tiles); and 256 requests of 64 tokens ending at
# 1048590 (S 268,439,040 at ratio 1 and 134,219,520 at ratio 2, both above 2**28 / 2 columns:
# one-row tiles, of which those holding the rows of requests 0, 128 and 255 run).
def _drive_tiles(name, reqs, ratio, form):
    """Each tile of a long-context step called as the target calls it (its own padded logits
    buffer, or the candidate-masked copy, zeros[:rows], lengths[a:b] and ks[a:b] views at row
    offset a): every row of every tile equals the reference exactly."""
    reqs = [tuple(r) for r in reqs]
    only = (0, 128, 255) if name == "b256_1M" else None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        tiles, rows, bad, straddle, s = _run_tiles(
            reqs, ratio, form == "candidate_masked", 9500 + ratio, only
        )
        torch.cuda.synchronize()
    sizes = sorted({b - a for a, b in tiles})
    print(
        f"[tiles {name} r{ratio} {form}] S={s}: {len(tiles)} tiles of {sizes} rows ({rows} rows), "
        f"straddling tie rows {straddle}"
    )
    _assert_variants(prof, len(tiles), f"tiles {name} r{ratio} {form}")
    assert all((b - a) * (-(-(s + 256) // 256) * 256) <= TILE_ELEMS or b - a == 1 for a, b in tiles)
    assert bad == 0


# The context top-k's (rows, S, ratio) cells in the long-context engine dumps (tep4, DSpark,
# max_num_tokens 8192, max_batch_size 256, max_seq_len 131072, 262144 and 1048576; the target's
# row-tiled prefill), the ratio from the prefill-logits record of the same (rows, S); a cell seen at
# both ratios runs at both. Each runs as one tile: `rows` consecutive positions ending a request of
# S * ratio tokens.
DUMPED_TILE_CELLS = [
    (1, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (64, 32768, 1),
    (64, 32768, 2),
    (128, 8192, 1),
    (128, 65536, 1),
    (128, 65536, 2),
    (128, 98304, 1),
    (128, 98304, 2),
    (128, 130773, 1),
    (128, 192512, 2),
    (128, 196608, 1),
    (128, 196608, 2),
    (128, 221184, 1),
    (128, 221184, 2),
    (128, 225280, 2),
    (128, 229376, 1),
    (128, 229376, 2),
    (128, 261860, 1),
    (128, 282624, 2),
    (128, 286720, 1),
    (128, 286720, 2),
    (128, 290816, 2),
    (128, 294912, 1),
    (128, 294912, 2),
    (128, 299008, 2),
    (128, 421888, 2),
    (128, 425984, 1),
    (128, 425984, 2),
    (128, 430080, 2),
    (128, 434176, 1),
    (128, 434176, 2),
    (128, 438272, 2),
    (128, 442368, 1),
    (128, 442368, 2),
    (128, 446464, 2),
    (128, 450560, 1),
    (128, 450560, 2),
    (128, 454656, 2),
    (128, 458752, 1),
    (128, 458752, 2),
    (128, 462848, 2),
    (128, 523951, 1),
    (128, 524288, 1),
    (128, 532480, 1),
    (128, 540672, 1),
    (128, 548864, 1),
    (128, 557056, 1),
    (128, 565248, 1),
    (128, 573440, 1),
    (128, 581632, 1),
    (128, 589824, 1),
    (128, 598016, 1),
    (128, 606208, 1),
    (128, 614400, 1),
    (128, 622592, 1),
    (128, 630784, 1),
    (128, 638976, 1),
    (128, 647168, 1),
    (128, 655360, 1),
    (128, 663552, 1),
    (128, 671744, 1),
    (128, 679936, 1),
    (128, 688128, 1),
    (128, 696320, 1),
    (128, 1048219, 1),
    (155, 1048219, 1),
    (175, 261975, 2),
    (175, 523951, 1),
    (192, 163840, 1),
    (192, 163840, 2),
    (192, 704512, 1),
    (192, 712704, 1),
    (192, 720896, 1),
    (192, 729088, 1),
    (192, 737280, 1),
    (192, 745472, 1),
    (192, 753664, 1),
    (192, 761856, 1),
    (192, 770048, 1),
    (192, 778240, 1),
    (192, 786432, 1),
    (192, 794624, 1),
    (192, 802816, 1),
    (192, 811008, 1),
    (192, 819200, 1),
    (192, 827392, 1),
    (192, 835584, 1),
    (219, 524109, 2),
    (256, 131072, 1),
    (256, 131072, 2),
    (256, 843776, 1),
    (256, 851968, 1),
    (256, 860160, 1),
    (256, 868352, 1),
    (256, 876544, 1),
    (256, 884736, 1),
    (256, 892928, 1),
    (256, 901120, 1),
    (256, 909312, 1),
    (256, 917504, 1),
    (256, 925696, 1),
    (256, 933888, 1),
    (256, 942080, 1),
    (256, 950272, 1),
    (256, 958464, 1),
    (256, 966656, 1),
    (256, 974848, 1),
    (256, 983040, 1),
    (256, 991232, 1),
    (256, 999424, 1),
    (256, 1007616, 1),
    (256, 1015808, 1),
    (256, 1024000, 1),
    (256, 1032192, 1),
    (256, 1040384, 1),
    (256, 1048219, 1),
    (320, 704512, 1),
    (320, 712704, 1),
    (320, 720896, 1),
    (320, 729088, 1),
    (320, 737280, 1),
    (320, 745472, 1),
    (320, 753664, 1),
    (320, 761856, 1),
    (320, 770048, 1),
    (320, 778240, 1),
    (320, 786432, 1),
    (320, 794624, 1),
    (320, 802816, 1),
    (320, 811008, 1),
    (320, 819200, 1),
    (320, 827392, 1),
    (320, 835584, 1),
    (384, 606208, 1),
    (384, 614400, 1),
    (384, 622592, 1),
    (384, 630784, 1),
    (384, 638976, 1),
    (384, 647168, 1),
    (384, 655360, 1),
    (384, 663552, 1),
    (384, 671744, 1),
    (384, 679936, 1),
    (384, 688128, 1),
    (384, 696320, 1),
    (448, 352256, 1),
    (448, 352256, 2),
    (448, 356352, 2),
    (448, 360448, 1),
    (448, 360448, 2),
    (448, 364544, 2),
    (448, 368640, 1),
    (448, 368640, 2),
    (448, 372736, 2),
    (448, 376832, 1),
    (448, 376832, 2),
    (448, 380928, 2),
    (448, 524109, 2),
    (448, 524288, 1),
    (448, 532480, 1),
    (448, 540672, 1),
    (448, 548864, 1),
    (448, 557056, 1),
    (448, 565248, 1),
    (448, 573440, 1),
    (448, 581632, 1),
    (448, 589824, 1),
    (448, 598016, 1),
    (512, 69632, 2),
    (512, 102400, 2),
    (512, 135168, 2),
    (512, 139264, 1),
    (512, 139264, 2),
    (512, 167936, 2),
    (512, 172032, 1),
    (512, 172032, 2),
    (512, 200704, 2),
    (512, 204800, 1),
    (512, 204800, 2),
    (512, 208896, 2),
    (512, 262144, 1),
    (512, 262144, 2),
    (512, 266240, 2),
    (512, 270336, 1),
    (512, 270336, 2),
    (512, 274432, 2),
    (512, 278528, 1),
    (512, 278528, 2),
    (512, 323584, 2),
    (512, 327680, 1),
    (512, 327680, 2),
    (512, 331776, 2),
    (512, 335872, 1),
    (512, 335872, 2),
    (512, 339968, 2),
    (512, 344064, 1),
    (512, 344064, 2),
    (512, 348160, 2),
    (512, 385024, 1),
    (512, 385024, 2),
    (512, 389120, 2),
    (512, 393216, 1),
    (512, 393216, 2),
    (512, 397312, 2),
    (512, 401408, 1),
    (512, 401408, 2),
    (512, 405504, 2),
    (512, 409600, 1),
    (512, 409600, 2),
    (512, 413696, 2),
    (512, 417792, 1),
    (512, 417792, 2),
    (512, 466944, 1),
    (512, 466944, 2),
    (512, 471040, 2),
    (512, 475136, 1),
    (512, 475136, 2),
    (512, 479232, 2),
    (512, 483328, 1),
    (512, 483328, 2),
    (512, 487424, 2),
    (512, 491520, 1),
    (512, 491520, 2),
    (512, 495616, 2),
    (512, 499712, 1),
    (512, 499712, 2),
    (512, 503808, 2),
    (512, 507904, 1),
    (512, 507904, 2),
    (512, 512000, 2),
    (512, 516096, 1),
    (512, 516096, 2),
    (512, 520192, 2),
    (512, 523951, 1),
    (576, 233472, 2),
    (576, 237568, 1),
    (576, 237568, 2),
    (576, 241664, 2),
    (576, 245760, 1),
    (576, 245760, 2),
    (576, 421888, 2),
    (576, 425984, 1),
    (576, 425984, 2),
    (576, 430080, 2),
    (576, 434176, 1),
    (576, 434176, 2),
    (576, 438272, 2),
    (576, 442368, 1),
    (576, 442368, 2),
    (576, 446464, 2),
    (576, 450560, 1),
    (576, 450560, 2),
    (576, 454656, 2),
    (576, 458752, 1),
    (576, 458752, 2),
    (576, 462848, 2),
    (640, 385024, 1),
    (640, 385024, 2),
    (640, 389120, 2),
    (640, 393216, 1),
    (640, 393216, 2),
    (640, 397312, 2),
    (640, 401408, 1),
    (640, 401408, 2),
    (640, 405504, 2),
    (640, 409600, 1),
    (640, 409600, 2),
    (640, 413696, 2),
    (640, 417792, 1),
    (640, 417792, 2),
    (704, 106496, 1),
    (704, 106496, 2),
    (704, 303104, 1),
    (704, 303104, 2),
    (704, 307200, 2),
    (704, 311296, 1),
    (704, 311296, 2),
    (704, 315392, 2),
    (704, 319488, 1),
    (704, 319488, 2),
    (704, 352256, 1),
    (704, 352256, 2),
    (704, 356352, 2),
    (704, 360448, 1),
    (704, 360448, 2),
    (704, 364544, 2),
    (704, 368640, 1),
    (704, 368640, 2),
    (704, 372736, 2),
    (704, 376832, 1),
    (704, 376832, 2),
    (704, 380928, 2),
    (740, 261860, 1),
    (768, 143360, 2),
    (768, 323584, 2),
    (768, 327680, 1),
    (768, 327680, 2),
    (768, 331776, 2),
    (768, 335872, 1),
    (768, 335872, 2),
    (768, 339968, 2),
    (768, 344064, 1),
    (768, 344064, 2),
    (768, 348160, 2),
    (832, 176128, 2),
    (832, 180224, 1),
    (832, 180224, 2),
    (832, 303104, 1),
    (832, 303104, 2),
    (832, 307200, 2),
    (832, 311296, 1),
    (832, 311296, 2),
    (832, 315392, 2),
    (832, 319488, 1),
    (832, 319488, 2),
    (896, 212992, 1),
    (896, 212992, 2),
    (896, 217088, 2),
    (896, 282624, 2),
    (896, 286720, 1),
    (896, 286720, 2),
    (896, 290816, 2),
    (896, 294912, 1),
    (896, 294912, 2),
    (896, 299008, 2),
    (960, 261975, 2),
    (960, 262144, 1),
    (960, 262144, 2),
    (960, 266240, 2),
    (960, 270336, 1),
    (960, 270336, 2),
    (960, 274432, 2),
    (960, 278528, 1),
    (960, 278528, 2),
    (1024, 36864, 2),
    (1024, 73728, 1),
    (1024, 73728, 2),
    (1024, 147456, 1),
    (1024, 147456, 2),
    (1024, 249856, 2),
    (1024, 253952, 1),
    (1024, 253952, 2),
    (1024, 258048, 2),
    (1024, 261860, 1),
    (1088, 110592, 2),
    (1088, 233472, 2),
    (1088, 237568, 1),
    (1088, 237568, 2),
    (1088, 241664, 2),
    (1088, 245760, 1),
    (1088, 245760, 2),
    (1152, 184320, 2),
    (1152, 188416, 1),
    (1152, 188416, 2),
    (1152, 221184, 1),
    (1152, 221184, 2),
    (1152, 225280, 2),
    (1152, 229376, 1),
    (1152, 229376, 2),
    (1216, 212992, 1),
    (1216, 212992, 2),
    (1216, 217088, 2),
    (1280, 114688, 1),
    (1280, 114688, 2),
    (1280, 151552, 2),
    (1280, 200704, 2),
    (1280, 204800, 1),
    (1280, 204800, 2),
    (1280, 208896, 2),
    (1344, 192512, 2),
    (1344, 196608, 1),
    (1344, 196608, 2),
    (1408, 77824, 2),
    (1408, 184320, 2),
    (1408, 188416, 1),
    (1408, 188416, 2),
    (1472, 118784, 2),
    (1472, 176128, 2),
    (1472, 180224, 1),
    (1472, 180224, 2),
    (1536, 155648, 1),
    (1536, 155648, 2),
    (1536, 159744, 2),
    (1536, 167936, 2),
    (1536, 172032, 1),
    (1536, 172032, 2),
    (1600, 163840, 1),
    (1600, 163840, 2),
    (1664, 81920, 1),
    (1664, 81920, 2),
    (1664, 122880, 1),
    (1664, 122880, 2),
    (1664, 155648, 1),
    (1664, 155648, 2),
    (1664, 159744, 2),
    (1728, 40960, 1),
    (1728, 40960, 2),
    (1728, 151552, 2),
    (1749, 130773, 1),
    (1792, 147456, 1),
    (1792, 147456, 2),
    (1856, 143360, 2),
    (1920, 135168, 2),
    (1920, 139264, 1),
    (1920, 139264, 2),
    (1956, 130930, 2),
    (1984, 130930, 2),
    (1984, 131072, 1),
    (1984, 131072, 2),
    (2048, 86016, 2),
    (2048, 126976, 2),
    (2048, 130773, 1),
    (2176, 122880, 1),
    (2176, 122880, 2),
    (2240, 118784, 2),
    (2304, 45056, 2),
    (2304, 90112, 1),
    (2304, 90112, 2),
    (2304, 114688, 1),
    (2304, 114688, 2),
    (2368, 110592, 2),
    (2496, 106496, 1),
    (2496, 106496, 2),
    (2560, 94208, 2),
    (2560, 102400, 2),
    (2688, 98304, 1),
    (2688, 98304, 2),
    (2816, 49152, 1),
    (2816, 49152, 2),
    (2816, 94208, 2),
    (2944, 90112, 1),
    (2944, 90112, 2),
    (3072, 86016, 2),
    (3200, 53248, 2),
    (3264, 81920, 1),
    (3264, 81920, 2),
    (3392, 77824, 2),
    (3584, 57344, 1),
    (3584, 57344, 2),
    (3584, 73728, 1),
    (3584, 73728, 2),
    (3840, 69632, 2),
    (3861, 65386, 2),
    (3904, 61440, 2),
    (4032, 65386, 2),
    (4032, 65536, 1),
    (4032, 65536, 2),
    (4288, 61440, 2),
    (4608, 57344, 1),
    (4608, 57344, 2),
    (4992, 53248, 2),
    (5376, 49152, 1),
    (5376, 49152, 2),
    (5888, 45056, 2),
    (6464, 40960, 1),
    (6464, 40960, 2),
    (7168, 36864, 2),
    (8128, 32768, 1),
    (8128, 32768, 2),
    (8192, 4096, 2),
    (8192, 8192, 1),
    (8192, 8192, 2),
    (8192, 12288, 2),
    (8192, 16384, 1),
    (8192, 16384, 2),
    (8192, 20480, 2),
    (8192, 24576, 1),
    (8192, 24576, 2),
    (8192, 28672, 2),
]


def test_dumped_tile_cells():
    """Every dumped tile cell (471), exact; every call launches the same two kernels."""
    assert len(DUMPED_TILE_CELLS) == 471
    zeros = _zeros(MAX_TOKENS)
    straddle = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, (rows, s, ratio) in enumerate(DUMPED_TILE_CELLS):
            reqs = [(s * ratio - rows, rows)]
            ks, ke, s2 = _packed_rows(ratio, reqs)
            assert s2 == s and _row_tiles(rows, s) == [(0, rows)]
            logits = _tile_logits(reqs, ratio, ks, ke, 0, rows, 60000 + i)
            out = op(logits, zeros[:rows], ke - ks, ks, K)
            bad, st = _tile_bad_rows(logits, ks, ke - ks, out)
            assert bad == 0, (rows, s, ratio)
            straddle += st
            del logits, out
        torch.cuda.synchronize()
    print(f"{len(DUMPED_TILE_CELLS)} dumped tile cells: all exact; straddling tie rows {straddle}")
    _assert_variants(prof, len(DUMPED_TILE_CELLS), "dumped tile cells")


# Random multi-request steps over the envelope: T <= 16384 context tokens in k <= 256 requests, each
# request's context up to the step's manager length Lm (a MAX_SEQ_LENS value, DSpark on or off). The
# test holds at most 2**32 fp32 logits (16 GiB) per random step: a draw whose tc x S exceeds that has
# its cached prefixes scaled down to fit, so large S comes with small tc.
MANAGER_LENS = sorted({L + d for L in MAX_SEQ_LENS for d in (0, 14)})
MULTI_REQUEST_MAX_REQS = 256
MULTI_REQUEST_CAP = 2**32


def _multi_request_random_step(rng, ratio):
    """A random context step: T in [1, 16384] rows split into k in [1, 256] requests in random order,
    each with a random new length and a random cached prefix (kv <= Lm, Lm drawn from MANAGER_LENS),
    scaled down if T x S > 2**32. Returns (requests, Lm)."""
    t = rng.randint(1, MAX_TOKENS)
    k = rng.randint(1, min(MULTI_REQUEST_MAX_REQS, t))
    lm = rng.choice(MANAGER_LENS)
    s_max = MULTI_REQUEST_CAP // t
    while True:
        cuts = sorted(rng.sample(range(1, t), k - 1))
        news = [b - a for a, b in zip([0] + cuts, cuts + [t])]
        cached = [rng.randint(0, lm - n) for n in news]
        if sum((c + n) // ratio for c, n in zip(cached, news)) > s_max:
            f = (s_max * ratio - t) / sum(cached)
            cached = [int(c * f) for c in cached]
        reqs = list(zip(cached, news))
        s = sum((c + n) // ratio for c, n in reqs)
        if s:  # the target skips the call when S == 0
            assert s <= s_max
            return reqs, lm


@pytest.mark.parametrize("ratio", [1, 2])
def test_multi_request_random_partitions(ratio):
    """Random multi-request steps over the context rows: 200 seeded random packed steps per ratio (T
    up to 16384, up to 256 requests, contexts up to the envelope's manager lengths), alternating the
    two layouts, mixed tie kinds, against the reference exactly; every call launches the same two
    kernels. Every 20th step also runs a control with one request's ks shifted by one column, which
    the same check must see."""
    rng = random.Random(23000 + ratio)
    steps = [_multi_request_random_step(rng, ratio) for _ in range(200)]
    controls, rows, straddle, reqs_seen, calls, names = [], 0, 0, [], 0, []
    max_s, max_t, max_len = 0, 0, 0
    for i, (reqs, lm) in enumerate(steps):
        logits, ks, ke = _packed_logits(reqs, ratio, contiguous=bool(i % 2), seed=ratio * 1000 + i)
        out = _profiled(lambda logits=logits: _run_packed(logits, ks, ke), names)
        calls += 1
        bad, st = _packed_bad_rows(logits, ks, ke, reqs, ratio, out)
        assert bad == 0, (i, lm, len(reqs), bad)
        rows, straddle = rows + ks.numel(), straddle + st
        max_s, max_t = max(max_s, logits.shape[1]), max(max_t, ks.numel())
        max_len = max(max_len, int((ke - ks).max()))
        reqs_seen.append(len(reqs))
        if i % 20 == 0:
            lens = (ke - ks).long()
            row0, pick = 0, None
            for cached, new in reqs:
                if int(lens[row0 : row0 + new].max()) >= 2:
                    pick = (row0, new)
                row0 += new
            if pick is not None:
                ks_bad = ks.clone()
                a, b = pick[0], pick[0] + pick[1]
                ks_bad[a:b] += (lens[a:b] >= 1).int()  # keeps ks <= ke
                wrong = _profiled(
                    lambda logits=logits: _call(logits, ke - ks_bad, row_starts=ks_bad), names
                )
                calls += 1
                controls.append(_packed_bad_rows(logits, ks, ke, reqs, ratio, wrong)[0])
        del logits, out
        torch.cuda.empty_cache()
    print(
        f"ratio={ratio}: {len(steps)} steps, {rows} rows, T up to {max_t}, request counts "
        f"{min(reqs_seen)}..{max(reqs_seen)} ({sum(k > 32 for k in reqs_seen)} steps above 32), max S {max_s}, "
        f"longest row {max_len}, straddling tie rows {straddle}; control (one request's ks + 1) failing rows "
        f"{controls}"
    )
    assert max(reqs_seen) > 200 and max_t > 16000 and max_s > 2**22
    assert len(controls) >= 10 and all(c > 0 for c in controls), controls
    _assert_variants(names, calls, f"random steps ratio {ratio}")


# Every (tc, S) cell of the context top-k in the core runtime-value dumps (boot, GSM8K, 24K passkey,
# gate and accuracy-preview logs), joined with the ratio of the prefill-logits record at the same
# (tc, S) and with a sparse-MLA varlen record (nc requests, max_q_len) at the same tc when one is
# consistent; otherwise the smallest (nc, max_q_len) that is (9 cells). As "tc,S,ratio,nc,max_q_len"
# records, space-separated, zlib-compressed, base64. 1933 cells; 3 dumped cells are garbles outside
# any consistent step and are dropped: (48, 0) and (848, 0) (the target never calls the top-k with
# S = 0) and (81, 6) (81 rows cannot see only 6 compressed columns in <= 32 requests at either ratio).
DUMPED_CELLS = (
    "eNpdnW267CgIhLcyCzg/4iey/40NVIHS95kJN/VqdyfGGNHgaX+N//3XzWKzvQ7W/2vz71BM32+beU3sTNi+nwn7v2l4ImUe"
    "F/Z/C6Hjb0x+t+1D77OYDD3/xtiRPqG39Eif/63PPi+KdNuHlsWvh+76NxY/b/vQ0vh56GmfFR6b7UOPqT/at1b01vP0+v5m"
    "HJ/tQ584Pur1jm8taFnxfdD7b+wT6RtahkS66W1lc3gNbB/at3a1/s3eI12hz5yRblqsnD5hujTo0yId2srvjEgf0LLj96Ht"
    "2E4cvxxo39rVXr75/QotbUW6aT92/Zh+GrRv7WorO43fPxt6nP6jfWtFy55P+3frYn79oGeTH+1bK/qMVbSd/47fs/rmWnqm"
    "u7b6J/l9E1pW/D70susf56cL+uw4P2j7rMb1VYH2rV1td8MX9UcP9Gkj0l1b2WuUvyq0by30/tqtP7ZPLetHy2D5Xb2+ov38"
    "NfIPaBka6a7tenwj0je0b+1qu/7fjHSF9q2ltnttxPnZPvSO+4/ayzc+b02Ia4n7jdp/P87PWhLXOxoWarHP5/cLtKwR6a79"
    "+OL8mkL71lJ3O//G+8v2oX1rV9v923qkL+gd9YPajq9F+fQN7Vu7Wkq6QPvWrrbja3H+XaHn13+0b63o882nre6NFtd7fNC+"
    "tatbSW/QvrWr7fq0KL/RoX1rV9v1ifvJ9qGzvqf2rRWd9Z96lc8v6B33H7WXT1zfIdA72j/oaefX4/fmB+1bu9rO78TxzwYt"
    "EtcP2s7vxO/PDi0Sxwtt17/H9Z8D2rd2tZ1/tK+2D71nft61Xf8e13du6JnnE9q3VvTJ84O26x/PRtuH3ln/XS8//7h+9jxx"
    "vfP+hW4lvUH71q628+9RPqtTR3ub2rdWtBwpepTPD+id9xe0l0+U75rQO9pHai+fKA97nrn2raXedn5Zn/YH7Vu7upX0Rq37"
    "R++sb6FFR9F2viOurz0vXc+sT6F3PE9Tn9mKHuXzA9q3drXV7xH1Yy/oGc/D1L61orVVbeUz4nrsDe1bu1pKukDPbI9C73he"
    "pz7ZPkEf+3zUr32gfWupxcs/6o980DOvZ2jfWtEnry90v/0z24fO/hm199/y+yb0zvsPepX0BX3PL7RvrejTZ9FeflG/ZEP7"
    "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"
)


# The context top-k's (tc, S) cells in the max_num_tokens-16384 engine dumps of this envelope (tep4
# DSpark max_batch_size 8 and dep4 DSpark max_batch_size 256, max_seq_len 32768; and the same tep4
# engine at max_num_tokens 8192), with nc and max_q_len from the sparse-MLA varlen record at the
# same tc, as (tc, S, ratio, nc, max_q_len).
DUMPED_CELLS_16K = [
    (1, 1, 1, 1, 1),
    (2, 2, 1, 1, 2),
    (2, 1, 2, 1, 2),
    (5, 5, 1, 1, 5),
    (5, 2, 2, 1, 5),
    (12, 12, 1, 1, 12),
    (12, 6, 2, 1, 12),
    (16, 16, 1, 1, 16),
    (16, 8, 2, 1, 16),
    (28, 28, 1, 2, 16),
    (28, 14, 2, 2, 16),
    (128, 8192, 1, 1, 128),
    (128, 10441, 1, 1, 128),
    (128, 10444, 1, 1, 128),
    (128, 10451, 1, 1, 128),
    (128, 16339, 1, 1, 128),
    (128, 16384, 1, 1, 128),
    (597, 610, 1, 1, 597),
    (597, 305, 2, 1, 597),
    (2252, 10444, 1, 1, 2252),
    (2252, 5222, 2, 1, 2252),
    (4553, 10441, 1, 1, 4553),
    (4553, 5220, 2, 1, 4553),
    (8147, 16339, 1, 2, 5888),
    (8147, 8169, 2, 2, 5888),
    (8192, 8192, 1, 1, 8192),
    (8192, 4096, 2, 1, 8192),
    (8317, 8759, 1, 25, 905),
    (8317, 4372, 2, 25, 905),
    (9055, 9754, 1, 27, 893),
    (9055, 4870, 2, 27, 893),
    (9108, 9959, 1, 28, 894),
    (9108, 4972, 2, 28, 894),
    (9392, 10234, 1, 28, 890),
    (9392, 5108, 2, 28, 890),
    (10441, 10441, 1, 1, 10441),
    (10441, 5220, 2, 1, 10441),
    (10444, 10444, 1, 1, 10444),
    (10444, 5222, 2, 1, 10444),
    (10451, 10451, 1, 1, 10451),
    (10451, 5225, 2, 1, 10451),
    (16292, 16625, 1, 24, 903),
    (16292, 8307, 2, 24, 903),
    (16327, 16327, 1, 23, 944),
    (16327, 8159, 2, 23, 944),
    (16328, 16328, 1, 23, 933),
    (16328, 8158, 2, 23, 933),
    (16378, 16695, 1, 24, 934),
    (16378, 8344, 2, 24, 934),
    (16384, 16384, 1, 1, 16384),
    (16384, 8192, 2, 1, 16384),
]


def _dumped_cell_reqs(tc, s, ratio, nc, max_q_len, max_len=16384):
    """One step consistent with a dumped cell: nc requests, the first with max_q_len new tokens, the
    rest splitting tc - max_q_len evenly (where even lengths alone would give more than S compressed
    entries, pairs of them become odd, keeping the total and the longest); cached prefixes (multiples
    of ratio, filled in request order up to max_len) chosen so the sum of kv // ratio is exactly S."""
    rest = tc - max_q_len
    news = [max_q_len] + [
        rest // (nc - 1) + (1 if i < rest % (nc - 1) else 0) for i in range(nc - 1)
    ]
    if s < sum(n // ratio for n in news):
        even = [i for i in range(1, nc) if news[i] % 2 == 0]
        for i, j in zip(even[0::2], even[1::2]):
            if news[i] < max_q_len:
                news[i] += 1
                news[j] -= 1
    extra = s - sum(n // ratio for n in news)
    assert extra >= 0 and min(news) >= 1 and sum(news) == tc
    reqs = []
    for n in news:
        e = min(extra, (max_len - n) // ratio)
        reqs.append((ratio * e, n))
        extra -= e
    assert extra == 0
    return reqs


def test_multi_request_dumped_partitions():
    """Every dumped (tc, S) cell as a consistent packed step, both layouts alternating, mixed tie
    kinds, against the reference exactly: the 1933 cells of the 8192-token gate dumps, and the 51
    cells of the 16384-token envelope dumps (kv up to the 32K DSpark manager length 32782); every call
    launches the same two kernels."""
    blob = zlib.decompress(base64.b64decode("".join(DUMPED_CELLS))).decode()
    cells = [tuple(map(int, c.split(","))) + (16384,) for c in blob.split()]
    assert len(cells) == 1933
    cells += [c + (32782,) for c in DUMPED_CELLS_16K]
    straddle = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, (tc, s, ratio, nc, mq, max_len) in enumerate(cells):
            reqs = _dumped_cell_reqs(tc, s, ratio, nc, mq, max_len)
            logits, ks, ke = _packed_logits(reqs, ratio, contiguous=bool(i % 2), seed=50000 + i)
            assert logits.shape == (tc, s)
            out = _run_packed(logits, ks, ke)
            bad, st = _packed_bad_rows(logits, ks, ke, reqs, ratio, out)
            assert bad == 0, (tc, s, ratio, nc, mq)
            straddle += st
        torch.cuda.synchronize()
    print(
        f"{len(cells)} dumped cells: all exact; request counts {sorted({c[3] for c in cells})}; "
        f"straddling tie rows {straddle}"
    )
    _assert_variants(prof, len(cells), "dumped cells")


# ─── ties at the K-th value ───────────────────────────────────────────────────────────────────
def _tie_case(form, width, ratio, layout, kind, seed):
    """(scores, lengths, row_starts, ranked windows as (rows, win, lens) triples) of one tie case."""
    if form == "decode":
        g = torch.Generator().manual_seed(seed)
        vis = _dspark_vis(32, ratio, width, g, padding=False)
        vis[0] = width
        scores, ranked = _decode_inputs(width, layout, vis, seed, kind=kind)
        return scores, vis, None, [(slice(0, vis.numel()), ranked, vis)]
    reqs = [(width * ratio - 2048, 2048)]
    logits, ks, ke = _packed_logits(reqs, ratio, contiguous=False, seed=seed, kind=kind)
    assert logits.shape[1] == width
    return logits, ke - ks, ks, [(slice(0, 2048), logits, ke - ks)]


def _drive_ties(form, width, ratio, layout, kind):
    """Scores whose K-th value is tied with scores left out (bf16-rounded normals, 8 integer levels, an
    exact-zero plateau, and a tie constructed at the K-th value): the selected set is the reference's
    exactly, ties going to the smaller column, and repeat calls are bitwise identical."""
    scores, lens, rs, ranked = _tie_case(
        form, width, ratio, layout, kind, seed=width + ratio + len(kind)
    )
    out = _call(scores, lens, row_starts=rs)
    ref_all = op.reference(scores, _zeros(scores.shape[0]), lens, rs, K)
    op.compare(out, ref_all)
    straddle, tied = 0, 0
    for rows, win, L in ranked:
        ref = ref_all[rows].long()
        straddle += _straddling_rows(win, L, ref)
        tied = max(tied, _max_tied(win, L, ref))
    selecting = int((lens > K).sum())
    print(
        f"[ties {form} w{width} r{ratio} {layout} {kind}] straddling rows {straddle} of {selecting} selecting; "
        f"most columns tied with a row's K-th score {tied}"
    )
    assert straddle > 0
    if (width, layout, kind) == (32768, "paged_view", "zero_plateau"):
        # more candidates at the cut than the kernel's 16K-entry candidate buffer holds
        assert tied > 16384
    if (width, layout, kind) == (1048595, "paged_view", "zero_plateau"):
        assert tied > 200_000
    if kind == "tie_at_kth" and layout != "candidate_masked":
        # (the candidate mask sets some of the constructed scores to -inf, which can move the cut
        # off the tie)
        assert straddle == selecting
    for _ in range(3):
        assert torch.equal(_call(scores, lens, row_starts=rs), out)


def _drive_signed_zero():
    """Scores are ranked by their fp32 bit pattern, so -0.0 ranks below +0.0 and the two are not a
    tie: with 300 positive scores, 1500 columns of -0.0 first and +0.0 after them, the 212 remaining
    picks are the smallest +0.0 columns, not the -0.0 columns a value-equality ranking would take."""
    n, w = 4, 4096
    x = torch.zeros(n, w, device=DEV)
    x[:, :1500] = -0.0
    x[:, 3000:3300] = torch.arange(1, 301, device=DEV).float()
    lens = torch.full((n,), w, dtype=torch.int32, device=DEV)
    out = _call(x, lens)
    op.compare(out, op.reference(x, _zeros(n), lens, None, K))
    expected = torch.cat([torch.arange(1500, 1712), torch.arange(3000, 3300)]).to(DEV)
    assert all(torch.equal(out[i].long(), expected) for i in range(n))
    # control: a value-equality ranking (-0.0 == +0.0, ties to the smaller column)
    value_ref = _control_reference(x, lens)
    assert not torch.equal(out.long(), value_ref) and int(value_ref[0, 0]) == 0


def _drive_few_finite():
    """A selecting row with fewer than K finite scores in range (the rest -inf) gets every finite
    column plus the smallest -inf columns: the reference's answer, since -inf entries tie with each
    other."""
    n, w = 16, 4096
    g = torch.Generator().manual_seed(3)
    x = torch.full((n, w), float("-inf"), device=DEV)
    finite = [0, 1, 7, 300, 511, 512, 513, 2000] * 2
    for i, f in enumerate(finite):
        cols = torch.randperm(w, generator=g)[:f].to(DEV)
        x[i, cols] = _distinct(1, max(f, 1), seed=i)[0, :f]
    lens = torch.full((n,), w, dtype=torch.int32, device=DEV)
    out = _call(x, lens)
    op.compare(out, op.reference(x, _zeros(n), lens, None, K))


# ─── determinism across calls, processes and GPUs ─────────────────────────────────────────────
DIGEST_DECODE = [
    (16384, 1, "paged_view"),
    (16403, 1, "candidate_masked"),
    (32768, 1, "paged_view"),
    (32773, 1, "candidate_masked"),
    (8201, 2, "paged_view"),
    (1048595, 1, "paged_view"),
    (1048595, 1, "candidate_masked"),
    (524297, 2, "paged_view"),
]
DIGEST_PREFILL = [(8192, 1), (4096, 2), (131072, 1)]


def _digest_cases(repeats):
    """{case: {"in": input digest (fingerprint of the scores, SHA-256 of the int32 vectors), "out":
    SHA-256 of the output, "exact": equal to the reference}} over the generation and context forms and
    every score kind (including the ties), plus a CUDA-graph replay; every case is called `repeats`
    times and must be bitwise identical each time."""
    res = {}

    def record(name, scores, lens, rs, ranked):
        out = _call(scores, lens, row_starts=rs)
        for _ in range(repeats - 1):
            assert torch.equal(_call(scores, lens, row_starts=rs), out), f"{name}: repeat differs"
        exact = all(torch.equal(out[rows].long(), _reference(win, L)) for rows, win, L in ranked)
        ins = _fingerprint(scores) + _sha(lens) + ("" if rs is None else _sha(rs))
        res[name] = {
            "in": hashlib.sha256(ins.encode()).hexdigest(),
            "out": _sha(out),
            "exact": exact,
        }
        return out

    for kind in KINDS:
        for width, ratio, layout in DIGEST_DECODE:
            record(
                f"decode.w{width}.r{ratio}.{layout}.{kind}",
                *_tie_case(
                    "decode", width, ratio, layout, kind, seed=width + 3 * ratio + len(kind)
                ),
            )
        for width, ratio in DIGEST_PREFILL:
            record(
                f"prefill.w{width}.r{ratio}.{kind}",
                *_tie_case(
                    "prefill", width, ratio, "mqa_view", kind, seed=width + 5 * ratio + len(kind)
                ),
            )
    logits, ks, ke = _packed_logits(FULL_CONTEXT_STEP, 1, contiguous=False, seed=5001)
    record(
        "prefill.full_context_step.r1.mixed",
        logits,
        ke - ks,
        ks,
        [
            (
                slice(a, a + n),
                logits[a : a + n, int(ks[a]) : int(ks[a]) + (c + n)],
                (ke - ks)[a : a + n],
            )
            for a, (c, n) in zip(
                [sum(q[1] for q in FULL_CONTEXT_STEP[:i]) for i in range(len(FULL_CONTEXT_STEP))],
                FULL_CONTEXT_STEP,
            )
        ],
    )
    # a captured call replayed on the same inputs
    scores, lens, _, ranked = _tie_case(
        "decode", 16403, 1, "candidate_masked", "tie_at_kth", seed=99
    )
    off = _zeros(scores.shape[0])
    eager = op(scores, off, lens, None, K)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(scores, off, lens, None, K)
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    res["decode.w16403.graph_replay.tie_at_kth"] = {
        "in": _fingerprint(scores),
        "out": _sha(out),
        "exact": bool(torch.equal(out, eager))
        and bool(torch.equal(out.long(), _reference(ranked[0][1], lens))),
    }
    return res


def _script_env(device=None):
    """The environment of a fresh process running this file as a script: the one device it may see,
    and the directory ``__extra_import_path__`` names (where ``_validating`` lives), which pytest
    supplies to a collected file but a script has to be given."""
    env = dict(os.environ)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    here = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.normpath(os.path.join(here, *__extra_import_path__[0].split("/")))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (tests_dir, env.get("PYTHONPATH", "")) if p)
    return env


def _visible_devices():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_bitwise_identical_across_calls_processes_and_gpus():
    """Every digest case (generation rows at eight engine widths up to 1048595 in both layouts,
    context rows at both ratios up to S 131072 and the 16384-row FULL_CONTEXT_STEP, each with distinct,
    bf16-rounded, 8-level, zero-plateau and constructed K-th-value ties, plus a graph replay; 57
    cases) is called 5 times here, and again in 3 fresh
    processes on each of 2 GPUs. Every output is bitwise identical everywhere and equals the
    reference; the inputs' digests are identical too, so no difference could hide behind one."""
    torch.cuda.empty_cache()  # the fresh processes share this GPU
    local = _digest_cases(repeats=5)
    assert all(c["exact"] for c in local.values()), [k for k, c in local.items() if not c["exact"]]
    assert len(local) == 57
    devs = _visible_devices()
    assert len(devs) >= 2, f"this test needs 2 GPUs, got CUDA_VISIBLE_DEVICES={devs}"
    tmp = tempfile.mkdtemp(prefix="fitopk_")
    procs = []
    for d in devs[:2]:
        for j in range(3):
            path = os.path.join(tmp, f"digests_{d}_{j}.json")
            env = _script_env(d)
            p = subprocess.Popen(
                [sys.executable, __file__, "--digests", path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            procs.append((d, j, path, p))
    uuids = {}
    for d, j, path, p in procs:
        log, _ = p.communicate()
        assert p.returncode == 0, f"digest process {d}/{j} failed:\n{log[-3000:]}"
        with open(path) as f:
            got = json.load(f)
        uuids.setdefault(d, set()).add(got["uuid"])
        differing = sorted(k for k in local if got["cases"].get(k) != local[k])
        print(
            f"process {j} on GPU {d} ({got['name']} {got['uuid']}): {len(got['cases'])} cases, "
            f"differing from this process: {differing}"
        )
        assert not differing and set(got["cases"]) == set(local)
    assert len({u for s in uuids.values() for u in s}) == 2, uuids
    print(
        f"{len(local)} cases bitwise identical over 5 calls here and 6 fresh processes on GPUs {uuids}"
    )


def _first_call_in_capture():
    """The process's first call of the op is inside a CUDA graph capture; the replay is exact."""
    scores, lens, _, ranked = _tie_case(
        "decode", 16403, 1, "candidate_masked", "tie_at_kth", seed=77
    )
    off = _zeros(scores.shape[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(scores, off, lens, None, K)
    graph.replay()
    torch.cuda.synchronize()
    return {
        "exact": bool(torch.equal(out.long(), _reference(ranked[0][1], lens))),
        "out": _sha(out),
    }


def test_first_call_inside_graph_capture():
    """A fresh process whose first call of the op (FlashInfer loads its top-k module then) is inside
    a CUDA graph capture: the capture succeeds and the replay equals the reference."""
    path = os.path.join(tempfile.mkdtemp(prefix="fitopk_"), "first.json")
    p = subprocess.run(
        [sys.executable, __file__, "--first-call-in-capture", path],
        env=_script_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p.returncode == 0, p.stdout[-3000:]
    with open(path) as f:
        got = json.load(f)
    print(f"first call inside capture: {got}")
    assert got["exact"]


# ─── wrong controls ───────────────────────────────────────────────────────────────────────────
def _raw(scores, lens, rs, deterministic, tie_break):
    """FlashInfer's call as the entry makes it, with the two determinism knobs free."""
    x = scores
    if not scores.is_contiguous():
        x = scores.as_strided((scores.size(0), scores.stride(0)), (scores.stride(0), 1))
    return top_k_ragged_transform(
        x,
        _zeros(scores.shape[0]),
        lens,
        K,
        deterministic=deterministic,
        tie_break=tie_break,
        dsa_graph_safe=True,
        row_starts=rs,
    )


CONTROL_FORMS = [("decode", 16384, 1, "paged_view"), ("prefill", 8192, 1, "mqa_view")]


@pytest.mark.parametrize("kind", ["bf16", "zero_plateau", "tie_at_kth"])
@pytest.mark.parametrize(
    "form,width,ratio,layout", CONTROL_FORMS, ids=[f"{f}_w{w}" for f, w, r, lay in CONTROL_FORMS]
)
def test_control_nondeterministic_mode_varies(form, width, ratio, layout, kind):
    """The same call with deterministic=False, tie_break NONE (the same kernel family otherwise) is
    seen to vary: over 6 calls on the same inputs its raw rows differ from the first call's, and on
    tied rows its selected sets differ from the entry's. The entry's 6 calls are identical."""
    scores, lens, rs, _ = _tie_case(form, width, ratio, layout, kind, seed=width + 17 + len(kind))
    entry = [_call(scores, lens, row_starts=rs) for _ in range(6)]
    assert all(torch.equal(e, entry[0]) for e in entry)
    nd = [_raw(scores, lens, rs, False, TopKTieBreak.NONE) for _ in range(6)]
    raw = sum(int((o != nd[0]).any(1).sum()) for o in nd[1:])
    sets_vs_entry = sum(int((o.sort(1).values != entry[0]).any(1).sum()) for o in nd)
    sets_between = sum(int((o.sort(1).values != nd[0].sort(1).values).any(1).sum()) for o in nd[1:])
    rows = scores.shape[0]
    print(
        f"[control nondet {form} w{width} {kind}] over 6 calls x {rows} rows: raw rows differing from call 0 "
        f"{raw}, set rows differing from call 0 {sets_between}, set rows differing from the entry {sets_vs_entry}"
    )
    assert raw > 0 and sets_vs_entry > 0


@pytest.mark.parametrize("kind", ["distinct", "bf16", "zero_plateau", "tie_at_kth"])
@pytest.mark.parametrize(
    "form,width,ratio,layout", CONTROL_FORMS, ids=[f"{f}_w{w}" for f, w, r, lay in CONTROL_FORMS]
)
def test_control_tie_break_large_differs_on_tie_rows(form, width, ratio, layout, kind):
    """The same deterministic call with ties to the LARGER column differs from the entry on exactly
    the rows whose K-th value straddles the cut (all selecting rows of the constructed tie, none on
    distinct scores), and equals the larger-column reference."""
    scores, lens, rs, ranked = _tie_case(
        form, width, ratio, layout, kind, seed=width + 29 + len(kind)
    )
    entry = _call(scores, lens, row_starts=rs)
    large = _raw(scores, lens, rs, True, TopKTieBreak.LARGE)
    rows, win, L = ranked[0]
    straddle = _straddling_rows(win, L, _reference(win, L))
    differing = int((large != entry).any(1).sum())
    print(
        f"[control large {form} w{width} {kind}] rows differing from the entry {differing}, straddling {straddle}"
    )
    assert torch.equal(large[rows].long(), _control_reference(win, L, large=True))
    assert differing == straddle
    assert (differing > 0) == (kind != "distinct")


# ─── what the op accepts silently, and what it rejects ────────────────────────────────────────
def test_is_valid_guards_what_the_op_accepts_silently():
    """The op reads offsets, lengths and row_starts through an int32 pointer and never compares the
    lengths of offsets / lengths with the row count. Each case below is accepted by the op itself
    (called outside ``validating``) without an error and gives wrong rows; the entry's guard
    rejects each."""
    n, w = 8, 4096
    x = _distinct(n, w, seed=4)
    lens = torch.full((n,), 4000, dtype=torch.int32, device=DEV)
    rs = torch.full((n,), 5, dtype=torch.int32, device=DEV)
    off = torch.full((n,), 7, dtype=torch.int32, device=DEV)
    good = _call(x, lens, row_starts=rs, offsets=off)

    def unguarded(o, L, r):
        return op(x, o, L, r, K)

    lens8 = torch.tensor([4000] * 4 + [0] * 4, dtype=torch.int32, device=DEV)
    off8 = torch.tensor([7] * 4 + [1000] * 4, dtype=torch.int32, device=DEV)
    silent = {
        "int64 lengths": unguarded(off, lens.long(), rs),
        "int64 offsets": unguarded(off.long(), lens, rs),
        "int64 row_starts": unguarded(off, lens, rs.long()),
        "short lengths (the 4 after it read)": unguarded(off, lens8[:4], rs),
        "short offsets (the 4 after it read)": unguarded(off8[:4], lens, rs),
    }
    for name, got in silent.items():
        wrong = int((got != good).any(1).sum())
        print(f"op itself, {name}: no error, rows wrong {wrong} of {n}")
        assert wrong > 0, name
    for match, args in (
        ("offsets and lengths must be int32", (x, off, lens.long(), rs, K)),
        ("offsets and lengths must be int32", (x, off.long(), lens, rs, K)),
        ("row_starts must be int32", (x, off, lens, rs.long(), K)),
        ("one entry per row", (x, off, lens8[:4], rs, K)),
        ("one entry per row", (x, off8[:4], lens, rs, K)),
    ):
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                op(*args)
    with validating(op):  # the guard admits the well-formed call
        assert torch.equal(op(x, off, lens, rs, K), good)


def _past_storage_view():
    """A [4, 1000] view with row stride 1280 whose storage ends at the last row's column 1000."""
    store = torch.randn(3 * 1280 + 1000, device=DEV)
    return store.as_strided((4, 1000), (1280, 1))


@pytest.mark.parametrize(
    "case,match",
    [
        ("transposed", "input must be contiguous"),
        ("k 4096", "operation not supported"),
        ("fp64", "Unsupported dtype torch.float64"),
        ("short row_starts", r"row_starts must have shape \(num_rows,\)"),
        ("strided lengths", "lengths must be contiguous"),
        ("padding past the storage", "out of bounds for storage"),
    ],
)
def test_rejections(case, match):
    """Layouts and arguments the call refuses, loudly (host-side checks, before any launch)."""
    lens = torch.full((2,), 4000, dtype=torch.int32, device=DEV)
    if case == "transposed":
        args = (torch.randn(4096, 2, device=DEV).t(), _zeros(2), lens, None, K)
    elif case == "k 4096":
        args = (
            torch.randn(2, 8192, device=DEV),
            _zeros(2),
            torch.full((2,), 8192, dtype=torch.int32, device=DEV),
            None,
            4096,
        )
    elif case == "fp64":
        args = (torch.randn(2, 4096, device=DEV, dtype=torch.float64), _zeros(2), lens, None, K)
    elif case == "short row_starts":
        args = (torch.randn(2, 4096, device=DEV), _zeros(2), lens, _zeros(1), K)
    elif case == "strided lengths":
        args = (
            torch.randn(2, 4096, device=DEV),
            _zeros(2),
            torch.full((4,), 4000, dtype=torch.int32, device=DEV)[::2],
            None,
            K,
        )
    else:
        args = (
            _past_storage_view(),
            _zeros(4),
            torch.full((4,), 1000, dtype=torch.int32, device=DEV),
            None,
            K,
        )
    with pytest.raises((RuntimeError, AssertionError), match=match):
        op(*args)
    torch.cuda.synchronize()


# ─── the certified cells ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Each cell drives the entry the way the target calls it, and every call it makes is gated by
    the entry's own reference and compare (the drivers above)."""
    certifying(op)
    spec = cell.spec
    kind = spec["kind"]
    if kind == "semantics":
        _drive_semantics()
    elif kind == "gen":
        _drive_gen(spec["width"], spec["ratio"], spec["layout"])
    elif kind == "every_len":
        _drive_every_len(spec["ratio"], spec["chunk"])
    elif kind == "widest":
        _drive_widest(spec["ratio"])
    elif kind == "long_ctx":
        _drive_long_ctx(spec["layout"])
    elif kind == "far_end":
        _drive_far_end(spec["lm"], spec["ratio"], spec["layout"])
    elif kind == "tiles":
        _drive_tiles(spec["name"], spec["reqs"], spec["ratio"], spec["layout"])
    elif kind == "ties":
        _drive_ties(spec["rows"], spec["width"], spec["ratio"], spec["layout"], spec["scores"])
    elif kind == "signed_zero":
        _drive_signed_zero()
    else:
        assert kind == "few_finite", kind
        _drive_few_finite()


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    props = torch.cuda.get_device_properties(0)
    if mode == "--digests":
        payload = {"uuid": str(props.uuid), "name": props.name, "cases": _digest_cases(repeats=2)}
    elif mode == "--capture-digests":
        payload = {"uuid": str(props.uuid), "name": props.name, "cases": _capture_digests()}
    else:
        assert mode == "--first-call-in-capture"
        payload = {"uuid": str(props.uuid), "name": props.name, **_first_call_in_capture()}
    with open(path, "w") as f:
        json.dump(payload, f)
