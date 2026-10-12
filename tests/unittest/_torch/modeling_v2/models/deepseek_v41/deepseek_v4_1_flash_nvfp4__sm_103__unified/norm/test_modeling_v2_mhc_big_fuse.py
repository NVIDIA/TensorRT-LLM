# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mhc_big_fuse catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate.
This file turns a cell's spec into tensors and keeps the checks a cell cannot carry: bit-identity
across launch shapes and between the full and post/comb-only modes, the `sinkhorn_repeat <= 0`
equivalence, the wrong-reference controls, M = 0, the op's own `num_splits` rejection, the
contiguity guard, CUDA-graph replay, and the DeepSeek-V4.1 call at every row count 1..32768
(prefix sweep and its control, one compiled kernel per call, the CUDA-graph capture union).
"""

import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.norm.mhc_big_fuse import (  # noqa: E501
    layer_input_error_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.norm.mhc_big_fuse import (  # noqa: E501
    mhc_big_fuse as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mhc_big_fuse requires a CUDA device"

DEV = "cuda"
HC = 4
HIDDEN = 5120
K = HC * HIDDEN
SENTINEL = 7.0
# DeepSeek-V4.1: rms_norm_eps, hc_eps (pre and Sinkhorn), post = 2 * sigmoid, 20 Sinkhorn passes.
V41 = dict(
    rms_eps=1e-20,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
)
# The fp32 gate: torch.testing.assert_close defaults for float32.
FP32_ATOL = 1e-5


def _inputs(M, seed, num_splits=1, hidden=HIDDEN, zero_rows=0):
    """Stream, split-K mixes and square sums, built the way the hc mixes GEMM produces them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = HC * hidden
    # Per-token stream magnitudes spread over four decades, as the real stream's are.
    mag = torch.exp(torch.empty(M, 1, 1, device="cuda").uniform_(-3.0, 7.0, generator=g))
    residual = (torch.randn(M, HC, hidden, device="cuda", generator=g) * mag).to(torch.bfloat16)
    if zero_rows:
        residual[M - zero_rows :] = 0  # CUDA-graph padding rows
    fn = torch.randn(24, k, device="cuda", generator=g) * 0.05
    xf = residual.reshape(M, k).float()
    step = k // num_splits
    parts = [slice(s * step, (s + 1) * step) for s in range(num_splits)]
    y_acc = torch.stack([xf[:, p] @ fn[:, p].t() for p in parts]).contiguous()
    r_acc = torch.stack([xf[:, p].square().sum(-1) for p in parts]).contiguous()
    if num_splits == 1:
        y_acc, r_acc = y_acc[0].contiguous(), r_acc[0].contiguous()
    scale = torch.tensor([0.1, 0.05, 0.25], device="cuda")
    base = torch.empty(24, device="cuda").uniform_(-5.0, 3.0, generator=g)
    # Strongly negative comb diagonal (the checkpoint's hc bases reach -43.1): near-permutation
    # rows, where Sinkhorn converges slowly enough for an iteration-count error to be visible.
    base[8::5] = -40.0
    return y_acc, r_acc, residual, scale, base


def _extreme_inputs(M, seed):
    """Logits of magnitude ~1e3-1e4 at unit RMS (mixes == y), a zero base."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    y = torch.randn(M, 24, device=DEV, generator=g) * 2e4
    r = torch.full((M,), float(K), device=DEV)
    res = torch.randn(M, HC, HIDDEN, device=DEV, generator=g).to(torch.bfloat16)
    scale = torch.tensor([0.1, 0.05, 0.25], device=DEV)
    base = torch.zeros(24, device=DEV)
    return y, r, res, scale, base


def _call_args(y_acc, r_acc, residual, scale, base, M, hidden, num_splits, block_size, pad=3, **kw):
    """The op's arguments in order, with sentinel-filled outputs `pad` rows longer than M."""
    c = dict(V41)
    c.update(kw)
    post = torch.full((M + pad, HC), SENTINEL, device=DEV)
    comb = torch.full((M + pad, HC, HC), SENTINEL, device=DEV)
    li = torch.full((M + pad, max(hidden, 1)), SENTINEL, dtype=torch.bfloat16, device=DEV)
    if hidden == 0:
        li = torch.empty(M + pad, 0, dtype=torch.bfloat16, device=DEV)
    return (
        y_acc,
        r_acc,
        residual,
        scale,
        base,
        post,
        comb,
        li,
        M,
        K,
        hidden,
        c["rms_eps"],
        c["hc_pre_eps"],
        c["hc_sinkhorn_eps"],
        c["hc_post_mult_value"],
        c["sinkhorn_repeat"],
        num_splits,
        block_size,
    )


def _assert_untouched(out, M):
    """Rows at and beyond M keep their sentinel."""
    post, comb, li = out
    assert torch.all(post[M:] == SENTINEL) and torch.all(comb[M:] == SENTINEL)
    assert torch.all(li[M:] == SENTINEL)


def _run(y_acc, r_acc, residual, scale, base, M, hidden, num_splits, block_size, pad=3, **kw):
    """Call the op on sentinel-filled outputs with `pad` extra rows; check those stay untouched."""
    args = _call_args(
        y_acc, r_acc, residual, scale, base, M, hidden, num_splits, block_size, pad, **kw
    )
    out = op(*args)
    torch.cuda.synchronize()
    _assert_untouched(out, M)
    return tuple(t[:M] for t in out)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    M, hidden = s["M"], s["hidden_size"]
    seed = zlib.crc32(cell.why.encode())
    if s["kind"] == "extreme":
        y, r, res, scale, base = _extreme_inputs(M, seed)
    else:
        y, r, res, scale, base = _inputs(
            M, seed, num_splits=s["num_splits"], zero_rows=s["zero_rows"]
        )
    if hidden == 0:
        res = torch.empty(M, HC, 0, dtype=torch.bfloat16, device=DEV)
    args = _call_args(
        y,
        r,
        res,
        scale,
        base,
        M,
        hidden,
        s["num_splits"],
        s["block_size"],
        sinkhorn_repeat=s["sinkhorn_repeat"],
        hc_post_mult_value=s["hc_post_mult_value"],
    )
    out = op(*args)
    torch.cuda.synchronize()
    assert all(o is a for o, a in zip(out, args[5:8])), (
        "the op writes in place and returns its outputs"
    )
    _assert_untouched(out, M)
    assert all(torch.isfinite(t[:M]).all() for t in out)
    op.compare(out, op.reference(*args))


@pytest.mark.parametrize("block_size", [0, 128, 256, 512])
@pytest.mark.parametrize("M", [1, 64, 65, 300])
def test_block_sizes_give_identical_bits(M, block_size):
    """Every launch shape (incl. the 512-thread bulk-copy variant at M <= 64) gives identical bits."""
    y, r, res, scale, base = _inputs(M, seed=100 + M)
    post, comb, li = _run(y, r, res, scale, base, M, HIDDEN, 1, block_size)
    p0, c0, l0 = _run(y, r, res, scale, base, M, HIDDEN, 1, 0)
    assert torch.equal(post, p0) and torch.equal(comb, c0) and torch.equal(li, l0)


@pytest.mark.parametrize("block_size", [0, 128, 256])
@pytest.mark.parametrize("M", [1, 6, 4096])
def test_post_comb_only_mode(M, block_size):
    """hidden_size=0: post/comb bit-identical to the full call; residual unread, layer_input unwritten."""
    y, r, res, scale, base = _inputs(M, seed=200 + M)
    p_full, c_full, _ = _run(y, r, res, scale, base, M, HIDDEN, 1, block_size)
    empty_res = torch.empty(M, HC, 0, dtype=torch.bfloat16, device="cuda")
    post, comb, li = _run(y, r, empty_res, scale, base, M, 0, 1, block_size)
    assert li.shape == (M, 0)
    assert torch.equal(post, p_full) and torch.equal(comb, c_full)
    # A real stream and a real layer_input buffer: still not written when hidden_size=0.
    keep = torch.full((M, HIDDEN), SENTINEL, dtype=torch.bfloat16, device="cuda")
    post2 = torch.empty(M, HC, device="cuda")
    comb2 = torch.empty(M, HC, HC, device="cuda")
    op(y, r, res, scale, base, post2, comb2, keep, M, K, 0, *V41.values(), 1, block_size)
    torch.cuda.synchronize()
    assert torch.all(keep == SENTINEL)
    assert torch.equal(post2, p_full) and torch.equal(comb2, c_full)


def test_sinkhorn_repeat_below_one_runs_one_pass():
    """sinkhorn_repeat <= 0 is accepted and computes exactly what sinkhorn_repeat=1 does."""
    M = 16
    y, r, res, scale, base = _inputs(M, seed=450)
    one = _run(y, r, res, scale, base, M, HIDDEN, 1, 0, sinkhorn_repeat=1)
    for repeat in (0, -3):
        got = _run(y, r, res, scale, base, M, HIDDEN, 1, 0, sinkhorn_repeat=repeat)
        assert all(torch.equal(a, b) for a, b in zip(got, one))


def test_controls_fail():
    """Deliberately wrong references must land far outside the gate, or the gate certifies nothing.

    (a) 19 Sinkhorn passes instead of 20; (b) comb transposed; (c) layer_input taken to be the
    stream collapsed with a LAGGED pre (the previous sublayer's mixes), which is what DeepSeek-V4.1
    actually feeds its sublayer -- the op's layer_input is the fresh-pre collapse instead.
    Each wrong reference is put through the entry's own `compare`.
    """
    M = 256
    y, r, res, scale, base = _inputs(M, seed=600)
    args = _call_args(y, r, res, scale, base, M, HIDDEN, 1, 0, pad=0)
    out = op(*args)
    _, comb, li = out
    good = op.reference(*args)
    op.compare(out, good)

    wrong19 = op.reference(*args[:15], 19, *args[16:])
    margin19 = (comb - wrong19[1]).abs().max().item() / FP32_ATOL
    with pytest.raises(AssertionError):
        op.compare(out, wrong19)

    wrong_t = (good[0], good[1].transpose(-1, -2), good[2], good[3])
    margin_t = (comb - wrong_t[1]).abs().max().item() / FP32_ATOL
    with pytest.raises(AssertionError):
        op.compare(out, wrong_t)

    # The lagged collapse: the previous sublayer's mixes, this sublayer's stream. Its post/comb are
    # handed back unchanged so that only the layer_input gate is under test.
    y_prev, r_prev, _, _, _ = _inputs(M, seed=601)
    lag = op.reference(y_prev, r_prev, *args[2:])
    with pytest.raises(AssertionError, match="layer_input outside the gate"):
        op.compare((lag[0], lag[1], li), lag)
    lag_ratio = layer_input_error_ratio(li, lag[2], lag[3])
    lag_rel = ((li.double() - lag[2]).norm() / lag[2].norm()).item()

    print(
        f"control margins: 19-pass comb max|diff| = {margin19:.1f} x atol, "
        f"transposed comb = {margin_t:.1f} x atol, lagged-pre layer_input max err/bound = "
        f"{lag_ratio:.3e} (rel L2 {lag_rel:.3e})"
    )
    assert margin19 > 10.0 and margin_t > 10.0 and lag_ratio > 10.0


def test_m_zero_is_a_no_op():
    y, r, res, scale, base = _inputs(4, seed=700)
    post, comb, li = _run(y, r, res, scale, base, 0, HIDDEN, 1, 0, pad=4)
    assert post.numel() == 0 and comb.numel() == 0 and li.numel() == 0


@pytest.mark.parametrize("num_splits", [0, 3, 32])
def test_rejects_num_splits_outside_the_set(num_splits):
    M = 8
    y, r, res, scale, base = _inputs(M, seed=800)
    with pytest.raises(RuntimeError, match=r"only num_splits .* supported"):
        _run(y, r, res, scale, base, M, HIDDEN, num_splits, 0)


@pytest.mark.parametrize(
    "which", ["y_acc", "r_acc", "residual", "post_mix", "comb_mix", "layer_input"]
)
def test_is_valid_rejects_strided_views(which):
    """The guard refuses a strided view of each of the six data tensors."""
    M = 8
    y, r, res, scale, base = _inputs(M, seed=900)
    a = dict(
        y_acc=y,
        r_acc=r,
        residual=res,
        post_mix=torch.empty(M, HC, device="cuda"),
        comb_mix=torch.empty(M, HC, HC, device="cuda"),
        layer_input=torch.empty(M, HIDDEN, dtype=torch.bfloat16, device="cuda"),
    )
    strided = dict(
        y_acc=torch.zeros(M, 32, device="cuda")[:, :24],
        r_acc=torch.zeros(M, 2, device="cuda")[:, 0],
        residual=res.transpose(0, 1).contiguous().transpose(0, 1),
        post_mix=torch.empty(HC, M, device="cuda").t(),
        comb_mix=torch.empty(M, HC, HC, device="cuda").transpose(1, 2),
        layer_input=torch.empty(M, 2 * HIDDEN, dtype=torch.bfloat16, device="cuda")[:, :HIDDEN],
    )
    a[which] = strided[which]
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(
                a["y_acc"],
                a["r_acc"],
                a["residual"],
                scale,
                base,
                a["post_mix"],
                a["comb_mix"],
                a["layer_input"],
                M,
                K,
                HIDDEN,
                *V41.values(),
                1,
                0,
            )


@pytest.mark.parametrize("hidden", [HIDDEN, 0])
def test_cuda_graph_capture(hidden):
    """Captured once, replayed on new inputs copied into the static buffers."""
    M = 6
    y, r, res, scale, base = _inputs(M, seed=1000)
    post = torch.empty(M, HC, device="cuda")
    comb = torch.empty(M, HC, HC, device="cuda")
    li = torch.empty(M, hidden, dtype=torch.bfloat16, device="cuda")
    res_arg = res if hidden else torch.empty(M, HC, 0, dtype=torch.bfloat16, device="cuda")
    args = (y, r, res_arg, scale, base, post, comb, li, M, K, hidden, *V41.values(), 1, 0)

    def call():
        op(*args)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    y2, r2, res2, _, _ = _inputs(M, seed=1001)
    y.copy_(y2)
    r.copy_(r2)
    res.copy_(res2)
    post.fill_(SENTINEL)
    comb.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    op.compare((post, comb, li), op.reference(*args))


# ─── The target's call at every row count 1..32768, and the capture union ──
#
# DeepSeek-V4.1's core target calls this op twice per layer in post/comb-only
# mode (hidden_size = 0, zero-size residual and layer_input) with
# block_size = 128, num_splits = 1 and M = the rows of the pass: the tokens the
# rank holds this step (any integer up to the engine's max_num_tokens, 16384 at
# most in the envelope) or, in layers 21-39 of a step that runs the decoder
# bounded-replay pass, that pass's rows (up to 32768). The DSpark drafter makes
# the same call at N = 5G rows. With block_size fixed by the caller, M only
# sizes the grid (one CTA per token, CTA t reads row t of y_acc / r_acc and
# writes row t of post_mix / comb_mix). The tests below check that on the GPU
# at every M from 1 to 32768.

_N_MAX = 32768
_TARGET_BS = 128
# The row counts of the target's call that the entry's cells certify (its runtime-value dumps
# plus the tops of the step and decoder-pass ranges).
_TARGET_M = tuple(sorted({c.spec["M"] for c in op.CELLS if c.why.startswith("target call")}))
# The capture union per rank. G ranges over every CUDA-graph batch size an
# envelope engine captures: every G 1..32, plus TRT-LLM's padded default list
# for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list at 256 (1..32, 64, 128, 256)
# is a subset. A captured step is generation-only, so the trunk runs at G (a
# decode step) and 6G (the DSpark target-verify step, 1 + 5 draft tokens per
# request), and the drafter at 5G.
_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_M = tuple(sorted({k * g for g in _GSET for k in (1, 5, 6)}))
assert len(_GSET) == 46 and len(_CAPTURE_M) == 114 and max(_CAPTURE_M) == 1536


def _target_args(y, r, scale, base, post, comb, m):
    """The core target's exact call: hidden_size 0, zero-size residual / layer_input, block_size 128."""
    no_res = torch.empty(m, HC, 0, dtype=torch.bfloat16, device="cuda")
    no_li = torch.empty(m, 0, dtype=torch.bfloat16, device="cuda")
    return (y, r, no_res, scale, base, post, comb, no_li, m, K, 0, *V41.values(), 1, _TARGET_BS)


def _target_call(y, r, scale, base, post, comb, m):
    return op(*_target_args(y, r, scale, base, post, comb, m))


def test_target_call_every_row_count_is_the_same_per_row_function() -> None:
    """Every prefix M = 1..32768 of the target's call returns exactly the first M
    rows of the 32768-row result, which is held to the fp32 reference.

    Outputs are NaN-filled before every call and the row after the last is a
    guard row that must stay NaN: a skipped row or a write past row M - 1 is a
    mismatch.
    """
    y, r, res, scale, base = _inputs(_N_MAX, seed=4242, zero_rows=1)
    del res
    full_post = torch.full((_N_MAX, HC), float("nan"), device="cuda")
    full_comb = torch.full((_N_MAX, HC, HC), float("nan"), device="cuda")
    full_args = _target_args(y, r, scale, base, full_post, full_comb, _N_MAX)
    op.compare(op(*full_args), op.reference(*full_args))
    fp, fc = full_post.view(torch.int32), full_comb.view(torch.int32)
    post_buf = torch.empty(_N_MAX + 1, HC, device="cuda")
    comb_buf = torch.empty(_N_MAX + 1, HC, HC, device="cuda")
    mismatch = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    guard = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    for m in range(1, _N_MAX + 1):
        post_buf[: m + 1].fill_(float("nan"))
        comb_buf[: m + 1].fill_(float("nan"))
        _target_call(y[:m], r[:m], scale, base, post_buf[:m], comb_buf[:m], m)
        mismatch[m] = (post_buf[:m].view(torch.int32) != fp[:m]).sum() + (
            comb_buf[:m].view(torch.int32) != fc[:m]
        ).sum()
        guard[m] = (~post_buf[m].isnan()).sum() + (~comb_buf[m].isnan()).sum()
    bad = mismatch.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_N_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the guard row, first {over[:10]}"


def test_target_call_row_sweep_harness_sees_a_one_row_shift() -> None:
    """Control for the sweep: its bitwise per-row check rejects a read one row off."""
    n = _N_MAX
    y, r, _, scale, base = _inputs(n + 1, seed=4243)
    full_post = torch.empty(n, HC, device="cuda")
    full_comb = torch.empty(n, HC, HC, device="cuda")
    _target_call(y[:n], r[:n], scale, base, full_post, full_comb, n)
    for m in (1, 2, 16, 17, 2048, 2049, 16384, n):
        post = torch.empty(m, HC, device="cuda")
        comb = torch.empty(m, HC, HC, device="cuda")
        _target_call(
            y[1 : m + 1].contiguous(), r[1 : m + 1].contiguous(), scale, base, post, comb, m
        )
        n_bad = int((comb.view(torch.int32) != full_comb.view(torch.int32)[:m]).sum().item())
        n_bad_p = int((post.view(torch.int32) != full_post.view(torch.int32)[:m]).sum().item())
        print(
            f"[control row shift] M={m}: {n_bad} / {m * HC * HC} comb and {n_bad_p} / {m * HC} post elements differ"
        )
        assert n_bad > m * HC * HC // 2 and n_bad_p > m * HC // 2


_PROBE_M = tuple(
    sorted(
        set(_TARGET_M)
        | set(_CAPTURE_M)
        | {
            9,
            15,
            17,
            64,
            65,
            129,
            4096,
            8191,
            8192,
            8193,
            12288,
            16383,
            16385,
            24576,
            32767,
            _N_MAX,
        }
    )
)


def test_target_call_one_compiled_kernel_at_every_row_count() -> None:
    """Probe: the target's call launches the same single kernel at every M (compiled ahead of time, no JIT)."""
    y, r, _, scale, base = _inputs(_N_MAX, seed=4244)
    post = torch.empty(_N_MAX, HC, device="cuda")
    comb = torch.empty(_N_MAX, HC, HC, device="cuda")
    names = set()
    for m in _PROBE_M:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _target_call(y[:m], r[:m], scale, base, post[:m], comb[:m], m)
            torch.cuda.synchronize()
        launched = [
            e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(launched) == 1, (m, launched)
        names.add(launched[0])
    print(f"{len(_PROBE_M)} row counts probed; kernels launched: {sorted(names)}")
    assert len(names) == 1 and "mhcBigFuseKernel<1, 128, false, false, false>" in next(
        iter(names)
    ), names


@pytest.mark.parametrize("M", _CAPTURE_M)
def test_target_call_capture_union_replay_is_bitwise_eager(M: int) -> None:
    """The target's call captured at M after an eager call at M, replayed on new
    data: bitwise equal to an eager call on that data.

    The replay on the capture-time data must equal the eager call before the
    capture and differ from the replay on new data, so a replay that reused
    stale inputs would fail. Rows past the first carry one all-zero padding row,
    as a padded CUDA-graph batch does.
    """
    y, r, _, scale, base = _inputs(M, seed=6000 + M, zero_rows=1 if M > 1 else 0)
    post = torch.full((M, HC), SENTINEL, device="cuda")
    comb = torch.full((M, HC, HC), SENTINEL, device="cuda")
    _target_call(y, r, scale, base, post, comb, M)  # the eager call at this shape
    torch.cuda.synchronize()
    post0, comb0 = post.clone(), comb.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _target_call(y, r, scale, base, post, comb, M)
    post.fill_(SENTINEL)
    comb.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(post, post0) and torch.equal(comb, comb0), M
    comb_old = comb.clone()
    y2, r2, _, _, _ = _inputs(M, seed=7000 + M, zero_rows=1 if M > 1 else 0)
    y.copy_(y2)
    r.copy_(r2)
    post.fill_(SENTINEL)
    comb.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    post_e = torch.full((M, HC), SENTINEL, device="cuda")
    comb_e = torch.full((M, HC, HC), SENTINEL, device="cuda")
    _target_call(y2, r2, scale, base, post_e, comb_e, M)
    torch.cuda.synchronize()
    assert torch.equal(post, post_e) and torch.equal(comb, comb_e), M
    assert not torch.equal(comb, comb_old), "the replay did not read the new mixes"
    args = _target_args(y2, r2, scale, base, post, comb, M)
    op.compare((post, comb, args[7]), op.reference(*args))  # the replay's output
