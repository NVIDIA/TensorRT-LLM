# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mhc_post_mapping catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate.
This file turns a cell's spec into tensors and keeps the checks a cell cannot carry: the
transposed-comb and rolled-post controls, row independence under non-finite values, T = 0,
CUDA-graph replay, the guard and the op's dtype rejections, and the DeepSeek-V4.1 aliased call
at every row count 1..32768 (prefix sweep and its control, one compiled kernel per call, the
CUDA-graph capture union).
"""

import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.norm.mhc_post_mapping import (  # noqa: E501
    mhc_post_mapping as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mhc_post_mapping requires a CUDA device"

HC = 4
# The checkpoint's width (DeepSeek-V4.1-Flash hidden_size).
H_CKPT = 5120


def _sinkhorn(t: int, gen: torch.Generator) -> torch.Tensor:
    """A doubly-stochastic [t, 4, 4] fp32 comb, built the way the checkpoint builds it."""
    c = torch.softmax(torch.randn(t, HC, HC, device="cuda", generator=gen), -1) + 1e-6
    c = c / (c.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        c = c / (c.sum(-1, keepdim=True) + 1e-6)
        c = c / (c.sum(-2, keepdim=True) + 1e-6)
    return c.contiguous()


def _inputs(t: int, h: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(t, HC, h, device="cuda", generator=gen) * 3
    if t > 1:
        # an attention-sink row: the checkpoint's BOS stream reaches |x| ~ 3.5e5
        residual[0] *= 1e5
    residual = residual.bfloat16()
    x = (torch.randn(t, h, device="cuda", generator=gen) * 2).bfloat16()
    post = 2 * torch.sigmoid(torch.randn(t, HC, device="cuda", generator=gen))
    return residual, x, post.contiguous(), _sinkhorn(t, gen)


def _check(out, residual, x, post, comb):
    """The entry's gate against its own reference."""
    op.compare(out, op.reference(residual, x, post, comb))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    t, h = s["T"], s["H"]
    residual, x, post, comb = _inputs(t, h, seed=zlib.crc32(cell.why.encode()) % 2**31)
    if s["flat"]:
        post, comb = post.view(t, HC, 1), comb.view(t, HC * HC)
    buf = None
    if s["out"] == "alias":
        residual = residual.clone()
        out = residual
    elif s["out"] == "view":
        buf = torch.full((t + 2, HC, h), 7.0, device="cuda", dtype=torch.bfloat16)
        out = buf[:t]
    else:
        out = None
    args = (residual, x, post, comb, out)
    ref = op.reference(*args)  # before the call: an aliased call overwrites residual
    ret = op(*args)
    assert ret.shape == (t, HC, h) and ret.dtype == torch.bfloat16
    if out is not None:
        assert ret.data_ptr() == out.data_ptr(), "the op writes the given out and returns it"
    if buf is not None:
        assert bool((buf[t:] == 7.0).all()), "rows past T of the parent buffer were written"
    op.compare(ret, ref)


def test_control_transposed_comb_is_detected() -> None:
    # The deliberately wrong control: comb applied untransposed (out[j] mixes
    # comb[j, k] instead of comb[k, j]). A doubly-stochastic comb makes the two
    # orientations numerically close, so this is the error a plausible-looking
    # result would hide; the gate must reject it.
    residual, x, post, comb = _inputs(64, H_CKPT, seed=7)
    out = op(residual, x, post, comb)
    _check(out, residual, x, post, comb)
    wrong = comb.transpose(1, 2).contiguous()
    with pytest.raises(AssertionError):
        op.compare(out, op.reference(residual, x, post, wrong))
    # a wrong post (x scaled by the next copy's coefficient) is caught too
    with pytest.raises(AssertionError):
        op.compare(out, op.reference(residual, x, post.roll(1, dims=1), comb))


def test_rows_are_independent() -> None:
    # CUDA-graph padding rows may carry garbage; a non-finite value must stay in its row
    t = 64
    residual, x, post, comb = _inputs(t, H_CKPT, seed=21)
    clean = op(residual, x, post, comb)
    post[10, 2] = float("nan")
    comb[20, 1, 3] = float("inf")
    x[30, 100] = float("nan")
    residual[40, 0, 7] = float("inf")
    dirty = op(residual, x, post, comb)
    keep = torch.ones(t, dtype=torch.bool, device="cuda")
    keep[[10, 20, 30, 40]] = False
    assert torch.equal(dirty[keep], clean[keep])
    assert not bool(torch.isfinite(dirty[10, 2].float()).any())
    assert not bool(torch.isfinite(dirty[20, 3].float()).any())


def test_zero_tokens() -> None:
    residual = torch.empty(0, HC, H_CKPT, device="cuda", dtype=torch.bfloat16)
    x = torch.empty(0, H_CKPT, device="cuda", dtype=torch.bfloat16)
    post = torch.empty(0, HC, device="cuda")
    comb = torch.empty(0, HC, HC, device="cuda")
    out = op(residual, x, post, comb)
    assert out.shape == (0, HC, H_CKPT)
    torch.cuda.synchronize()
    # nothing left pending: the next call runs and is exact
    r, xx, p, c = _inputs(4, H_CKPT, seed=4)
    _check(op(r, xx, p, c), r, xx, p, c)
    torch.cuda.synchronize()


def test_cuda_graph_replay() -> None:
    t = 32
    residual, x, post, comb = _inputs(t, H_CKPT, seed=31)
    out = torch.empty_like(residual)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(residual, x, post, comb, out=out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(residual, x, post, comb, out=out)
    r2, x2, p2, c2 = _inputs(t, H_CKPT, seed=32)
    residual.copy_(r2)
    x.copy_(x2)
    post.copy_(p2)
    comb.copy_(c2)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _check(out, r2, x2, p2, c2)


def test_is_valid_rejects_silently_wrong_layouts() -> None:
    # Each of these returns wrong values (or writes out of bounds) with no error
    # when handed to the raw op; the guard refuses them before launch.
    t = 8
    residual, x, post, comb = _inputs(t, H_CKPT, seed=41)
    wide = torch.zeros(t, 2 * H_CKPT, device="cuda", dtype=torch.bfloat16)
    mixes = torch.zeros(t, 24, device="cuda")
    bad = [
        dict(x=wide[:, :H_CKPT]),  # column slice of a wider buffer
        dict(post_mix=mixes[:, 4:8]),  # column slice of a [T, 24] mixes buffer
        dict(comb_mix=comb.transpose(1, 2)),  # transpose view
        dict(residual=residual[:, :2].contiguous(), x=x),  # hc = 2
        dict(x=x[: t // 2].contiguous()),  # too few rows
        dict(post_mix=post[:, :2].contiguous()),
        dict(comb_mix=comb[:, :, :2].contiguous()),
        dict(out=torch.empty(t // 2, HC, H_CKPT, device="cuda", dtype=torch.bfloat16)),
        dict(out=torch.empty(t, HC, 2 * H_CKPT, device="cuda", dtype=torch.bfloat16)[..., :H_CKPT]),
    ]
    for override in bad:
        args = dict(residual=residual, x=x, post_mix=post, comb_mix=comb, out=None)
        args.update(override)
        with pytest.raises(AssertionError, match="must"):
            with validating(op):
                op(**args)


@pytest.mark.parametrize(
    "field,dtype",
    [
        ("x", torch.float32),
        ("x", torch.float16),
        ("residual", torch.float32),
        ("post_mix", torch.bfloat16),
        ("comb_mix", torch.bfloat16),
        ("out", torch.float32),
    ],
)
def test_wrong_dtype_raises(field: str, dtype: torch.dtype) -> None:
    residual, x, post, comb = _inputs(4, H_CKPT, seed=51)
    args = dict(
        residual=residual, x=x, post_mix=post, comb_mix=comb, out=torch.empty_like(residual)
    )
    args[field] = args[field].to(dtype)
    with pytest.raises(RuntimeError, match="expected scalar type"):
        op(**args)


# ─── The target's aliased call at every row count 1..32768, and the capture union ──
#
# DeepSeek-V4.1's core target updates the stream in place twice per layer,
# `mhc_post_mapping(h, sublayer_out, post, comb, out=h)`, at T = the rows of
# the pass: the tokens the rank holds this step (any integer up to the engine's
# max_num_tokens, 16384 at most in the envelope) or, in layers 21-39 of a step
# that runs the decoder bounded-replay pass, that pass's rows (up to 32768).
# The DSpark drafter makes the same call at N = 5G rows. The launch is one CTA
# of 256 threads per token (CTA t reads row t of every input and writes row t
# of out), so T only sizes the grid. The tests below check that on the GPU at
# every T from 1 to 32768, in the aliased form.

_N_MAX = 32768
# The row counts of the target's call that the entry's cells certify (its runtime-value dumps
# plus the tops of the step and decoder-pass ranges).
_TARGET_T = tuple(sorted({c.spec["T"] for c in op.CELLS if c.why.startswith("target call")}))
# The capture union per rank. G ranges over every CUDA-graph batch size an
# envelope engine captures: every G 1..32, plus TRT-LLM's padded default list
# for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list at 256 (1..32, 64, 128, 256)
# is a subset. A captured step is generation-only, so the trunk runs at G (a
# decode step) and 6G (the DSpark target-verify step, 1 + 5 draft tokens per
# request), and the drafter at 5G.
_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_T = tuple(sorted({k * g for g in _GSET for k in (1, 5, 6)}))
assert len(_GSET) == 46 and len(_CAPTURE_T) == 114 and max(_CAPTURE_T) == 1536


def _aliased(stream, x, post, comb):
    """The core target's exact call: out is the stream itself."""
    ret = op(stream, x, post, comb, out=stream)
    assert ret.data_ptr() == stream.data_ptr()
    return ret


def test_target_call_every_row_count_is_the_same_per_row_function() -> None:
    """Every prefix T = 1..32768 of the aliased call returns exactly the first T
    rows of the 32768-row result, which matches the kernel-order reference bit
    for bit.

    Each call runs on the first T rows of a buffer whose row T is a NaN guard
    row that must stay NaN. A row the kernel skipped would keep its old stream
    value, which differs from the updated one.
    """
    residual, x, post, comb = _inputs(_N_MAX, H_CKPT, seed=4242)
    full = residual.clone()
    full_ref = op.reference(full, x, post, comb, full)
    op.compare(_aliased(full, x, post, comb), full_ref)
    fb = full.view(torch.int16)
    buf = torch.empty(_N_MAX + 1, HC, H_CKPT, dtype=torch.bfloat16, device="cuda")
    mismatch = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    guard = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    differs = torch.empty(
        _N_MAX, HC, H_CKPT, dtype=torch.bool, device="cuda"
    )  # one buffer; .any() keeps the reduction free of a widened copy
    for t in range(1, _N_MAX + 1):
        buf[:t].copy_(residual[:t])
        buf[t].fill_(float("nan"))
        _aliased(buf[:t], x[:t], post[:t], comb[:t])
        mismatch[t] = torch.ne(buf[:t].view(torch.int16), fb[:t], out=differs[:t]).any()
        guard[t] = (~buf[t].isnan()).sum()
    bad = mismatch.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_N_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the guard row, first {over[:10]}"


def test_target_call_row_sweep_harness_sees_a_one_row_shift() -> None:
    """Control for the sweep: its bitwise per-row check rejects a read one row off."""
    n = _N_MAX
    residual, x, post, comb = _inputs(n + 1, H_CKPT, seed=4243)
    full = residual[:n].clone()
    _aliased(full, x[:n], post[:n], comb[:n])
    for t in (1, 2, 7, 1024, 2048, 16384, n):
        s = residual[1 : t + 1].clone()
        _aliased(s, x[1 : t + 1], post[1 : t + 1], comb[1 : t + 1])
        n_bad = int((s.view(torch.int16) != full.view(torch.int16)[:t]).sum().item())
        print(f"[control row shift] T={t}: {n_bad} / {t * HC * H_CKPT} elements differ")
        assert n_bad > t * HC * H_CKPT // 2


_PROBE_T = tuple(
    sorted(
        set(_TARGET_T)
        | set(_CAPTURE_T)
        | {9, 64, 65, 129, 4096, 8191, 8192, 8193, 12288, 16383, 16385, 24576, 32767, _N_MAX}
    )
)


def test_target_call_one_compiled_kernel_at_every_row_count() -> None:
    """Probe: the aliased call launches the same single kernel at every T (compiled ahead of time, no JIT)."""
    residual, x, post, comb = _inputs(_N_MAX, H_CKPT, seed=4244)
    names = set()
    for t in _PROBE_T:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _aliased(residual[:t], x[:t], post[:t], comb[:t])
            torch.cuda.synchronize()
        launched = [
            e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(launched) == 1, (t, launched)
        names.add(launched[0])
    print(f"{len(_PROBE_T)} row counts probed; kernels launched: {sorted(names)}")
    assert len(names) == 1 and "mhcPostMappingKernel" in next(iter(names)), names


@pytest.mark.parametrize("num_tokens", _CAPTURE_T)
def test_target_call_capture_union_replay_is_bitwise_eager(num_tokens: int) -> None:
    """The aliased call captured at T after an eager aliased call at T, replayed
    on new data: bitwise equal to an eager call on that data.

    The replay on the capture-time data must differ from the replay on new
    data, so a replay that reused stale inputs would fail.
    """
    t = num_tokens
    residual, x, post, comb = _inputs(t, H_CKPT, seed=6000 + t)
    stream = residual.clone()
    ref = op.reference(residual, x, post, comb)
    _aliased(stream, x, post, comb)  # the eager call at this shape
    torch.cuda.synchronize()
    op.compare(stream, ref)
    stream.copy_(residual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _aliased(stream, x, post, comb)
    graph.replay()
    torch.cuda.synchronize()
    op.compare(stream, ref)
    old = stream.clone()
    r2, x2, p2, c2 = _inputs(t, H_CKPT, seed=7000 + t)
    stream.copy_(r2)
    x.copy_(x2)
    post.copy_(p2)
    comb.copy_(c2)
    graph.replay()
    torch.cuda.synchronize()
    eager = r2.clone()
    _aliased(eager, x2, p2, c2)
    torch.cuda.synchronize()
    assert torch.equal(stream, eager), t
    assert not torch.equal(stream, old), "the replay did not read the new inputs"
    op.compare(stream, op.reference(r2, x2, p2, c2))  # the replay's output
