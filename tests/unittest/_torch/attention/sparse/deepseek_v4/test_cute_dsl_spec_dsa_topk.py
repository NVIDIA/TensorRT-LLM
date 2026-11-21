# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""specDSA multi-CTA exact top-512 (``topk_multicta_op``, CuTe DSL, SM100) against ``torch.topk``.

The kernel selects the exact top-512 of every row of the fp32 indexer logits with one cluster of 4 to 8 CTAs per row
(8 / 6 / 5 / 4 CTAs for 1-15 / 16 / 17-24 / 25-26 rows). Its fast path guesses a threshold from sampled logits and
certifies it by a count; a row where the guess does not hold, and every row of at most 8192 positions (no sample), takes
an exact radix select inside the kernel. The checks of every row are the contract of the top-K stage: all 512 positions
are in [0, n) and distinct, their values are the multiset ``torch.topk`` returns (which of several equal values at the
512th rank is taken is not specified), and a row of at most 512 positions is ``0 .. n - 1`` followed by -1. The part of
a logits row beyond its length holds garbage (1e30 by default; also +-inf and NaN), so a read past the length shows as
a selected position.

- host side (no GPU): the cluster table, the batch limits, the signatures, ``is_profitable``, the refusal of every bad
  argument, the stage order of ``tail_op.run_hybrid`` with ``topk_fn=None``;
- exactness on a B200: every cluster size at the batch-size boundaries, the row kinds (gaussian, heavy tail, quantized,
  two-valued, all-equal, planted ties at the 512th value, outlier, negative-only, signed rows like the real mostly
  negative layers, +-0, near-flat, a plateau at the top, ties in the tail, +-inf, all -inf, all +inf, subnormals),
  row lengths from 1 to 2**18 (the short rows, the band without a sample, the edges of the sampled bands, lengths that
  end inside a chunk), mixed lengths in one batch, the exact path forced on every row, views of larger and smaller
  logits buffers, garbage beyond the lengths, canaries around the output;
- warp skew: the radix fallback with pseudo-random delays after every barrier and wait of the kernel source (it
  guards the barrier between the levels of the fallback);
- CUDA graphs: capture after ``precompile`` (no warm-up launch), replay with the logits and the lengths refilled in
  place, a capture that would compile is refused, one compile per cluster size;
- the hybrid: ``tail_op.run_hybrid(topk_fn=None)`` end to end against the fp64 attention reference of the tail tests.
"""

import dataclasses
import inspect
import random

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import DS_V4_FLASH as SPEC
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import (
    DS_V4_FLASH_FP4 as SPEC_FP4,
)
from tensorrt_llm._utils import get_sm_version

from . import test_cute_dsl_spec_dsa_tail as T

K_TOP = SPEC.topk
NPAD = SPEC.max_index_len  # the logits arena of the indexer kernel: 2**18 columns
# An indexer shape without kernels: top-1024 instead of top-512.
PRO = dataclasses.replace(SPEC, name="DS-V4-Pro", topk=1024)

# The batch sizes of the tail attention, and the CTAs per row of the cluster that serves each.
MAX_ROWS = 26
CLUSTER_OF = {1: 8, 2: 8, 5: 8, 8: 8, 12: 8, 13: 8, 15: 8, 16: 6, 17: 5, 24: 5, 25: 4, 26: 4}
CLUSTER_SIZES = (8, 6, 5, 4)
BOUNDARY_BATCHES = [1, 2, 8, 12, 13, 15, 16, 17, 24, 25, 26]
REFUSED = [27, 33, 47, 74, 75, 148, 1024]
# co-resident clusters of a B200 at one CTA per SM (measured): the batch sizes of a cluster size must fit one wave
CO_RESIDENT = {8: 15, 6: 22, 5: 26, 4: 33}
# lengths: the short rows, 8192 (the largest row without a sample), the band just above it, the edges of the sampled
# bands, lengths that end inside a 128-float chunk and inside a float4, the longest row
EDGES = [
    1,
    2,
    255,
    511,
    512,
    513,
    514,
    640,
    641,
    1023,
    4096,
    8191,
    8192,
    8193,
    8200,
    12000,
    16383,
    16384,
    20000,
    28671,
    28672,
    65536,
    100001,
    131071,
    262143,
    262144,
]


def _op():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import topk_multicta_op

    return topk_multicta_op


def _supported() -> bool:
    return torch.cuda.is_available() and get_sm_version() == 100 and _op().is_supported()[0]


requires_b200 = pytest.mark.skipif(
    not _supported(), reason="clusters of up to 8 CTAs on 148 SMs, qualified on B200 only"
)


@pytest.fixture(params=["fp4", "fp8"])
def spec(request):
    """The spec under test: the kernel reads the logits only, so both indexer K dtypes are served."""
    return SPEC_FP4 if request.param == "fp4" else SPEC


# --------------------------------------------------------------------------------------------------------------------
# rows


def _row(kind, n, gen):
    """float32 [n]: one logits row of the given kind."""
    if n < 4 and kind in ("ties512", "gauss_tail_ties", "plateau_top"):
        kind = "gauss"
    if kind == "gauss":
        x = torch.randn(n, generator=gen)
    elif kind == "lognorm":  # heavy tail
        x = torch.exp(torch.randn(n, generator=gen) * 1.5)
    elif kind == "quant":  # coarse grid: thousands of ties at the 512th value
        x = torch.round(torch.randn(n, generator=gen) * 4) / 4
    elif kind == "two":
        x = (torch.rand(n, generator=gen) < 0.01).float()
    elif kind == "flat":  # all equal
        x = torch.full((n,), 1.25)
    elif kind == "ties512":  # exactly 700 elements equal to a value that sits at ranks ~300 .. 1000
        x = torch.randn(n, generator=gen)
        idx = torch.randperm(n, generator=gen)[:700]
        thr = torch.topk(x, min(300, n - 1)).values[-1] if n > 300 else x.max()
        x[idx] = thr
    elif kind == "outlier":
        x = torch.randn(n, generator=gen)
        x[int(torch.randint(0, n, (1,), generator=gen))] = 1e6
    elif kind == "neg":  # negative-only
        x = -torch.exp(torch.randn(n, generator=gen))
    elif kind == "zeros":  # many exact +-0.0
        x = torch.where(
            torch.rand(n, generator=gen) < 0.9,
            torch.zeros(n),
            torch.randn(n, generator=gen) * 0.001,
        )
        x = torch.where((x == 0) & (torch.rand(n, generator=gen) < 0.5), torch.full((n,), -0.0), x)
    elif kind == "nearflat":
        x = 3.0 + 1e-6 * torch.randn(n, generator=gen)
    elif kind == "plateau_top":  # a plateau of 5000 equal values at the very top
        x = torch.randn(n, generator=gen)
        x[torch.randperm(n, generator=gen)[: min(5000, n // 2)]] = 10.0
    elif kind == "gauss_tail_ties":
        # the tail quantized to 0.02: ties spread around ranks 400 .. 700
        x = torch.randn(n, generator=gen)
        t = torch.topk(x, min(900, n - 1)).values[-1]
        x = torch.where(x > t, torch.round(x * 50) / 50, x)
    elif kind == "inf":  # a few +-inf among normal values and some ties
        x = torch.randn(n, generator=gen)
        idx = torch.randperm(n, generator=gen)
        k = min(8, n // 4)
        x[idx[:k]] = float("inf")
        x[idx[k : 2 * k]] = float("-inf")
        x[idx[2 * k : 2 * k + min(600, n // 3)]] = 3.0
    elif kind == "ninf":  # every value -inf: all tie at the smallest key
        x = torch.full((n,), float("-inf"))
    elif kind == "pinf":  # every value +inf
        x = torch.full((n,), float("inf"))
    elif kind == "denorm":  # subnormals, zeros and tiny values of both signs
        x = torch.randn(n, generator=gen) * 1e-40
        x = torch.where(torch.rand(n, generator=gen) < 0.3, torch.zeros(n), x)
    elif kind == "signed_tail":
        # like the real mostly negative layers: ~0.4 % positives, the boundary near zero
        x = -torch.exp(torch.randn(n, generator=gen) * 0.6)
        pos = torch.rand(n, generator=gen) < 0.004
        x = torch.where(pos, torch.exp(torch.randn(n, generator=gen) * 1.2) * 0.3, x)
    else:
        raise ValueError(kind)
    return x.float()


KINDS = [
    "gauss",
    "lognorm",
    "quant",
    "two",
    "flat",
    "ties512",
    "outlier",
    "neg",
    "zeros",
    "nearflat",
    "plateau_top",
    "gauss_tail_ties",
    "inf",
    "ninf",
    "pinf",
    "denorm",
    "signed_tail",
]
# kinds without equal values at the 512th rank (a unique selection: two runs give the same set)
UNIQUE_KINDS = ["gauss", "lognorm", "neg", "signed_tail"]


def _logits(rows, npad=NPAD, poison=1e30):
    """(logits [B, npad] on the device with ``poison`` beyond each row, ctx_lens int32 [B], the rows on the device)."""
    lg = torch.full((len(rows), npad), poison, dtype=torch.float32)
    for r, x in enumerate(rows):
        lg[r, : x.numel()] = x
    ctx = torch.tensor([x.numel() for x in rows], dtype=torch.int32)
    return lg.cuda(), ctx.cuda(), [x.cuda() for x in rows]


def _make(kinds, lengths, seed, batch=None):
    """``batch`` rows (default: the longer list), row i of kind ``kinds[i % len(kinds)]`` with
    ``lengths[i % len(lengths)]`` positions."""
    gen = torch.Generator().manual_seed(seed)
    b = batch or max(len(kinds), len(lengths))
    return [_row(kinds[i % len(kinds)], lengths[i % len(lengths)], gen) for i in range(b)]


def _check(rows, out):
    """Row r of ``out`` is an exact top-512 of ``rows[r]`` (device tensors): in range, distinct, the values of
    torch.topk."""
    out = out.cpu()
    for r, x in enumerate(rows):
        n = x.numel()
        sel = out[r]
        if n <= K_TOP:
            want = torch.cat(
                [
                    torch.arange(n, dtype=torch.int32),
                    torch.full((K_TOP - n,), -1, dtype=torch.int32),
                ]
            )
            assert torch.equal(sel, want), f"row {r} (n={n}): a short row is 0 .. n - 1 and -1"
            continue
        assert bool(((sel >= 0) & (sel < n)).all()), f"row {r} (n={n}): position out of [0, n)"
        assert torch.unique(sel).numel() == K_TOP, f"row {r} (n={n}): duplicate positions"
        got = torch.sort(x[sel.to(x.device).long()], descending=True).values
        assert torch.equal(got, torch.topk(x, K_TOP).values), (
            f"row {r} (n={n}): not the top-512 values"
        )


def _run(rows, forced=False, npad=NPAD):
    """Launch on the rows (eager) and return (out, logits, ctx, device rows)."""
    op = _op()
    lg, ctx, drows = _logits(rows, npad)
    out = torch.full((len(rows), K_TOP), -9, dtype=torch.int32, device="cuda")
    if forced:
        op._fn(lg.device, op.cluster_size(len(rows)), True)(lg, ctx, out)
    else:
        op.run(lg, out, ctx)
    torch.cuda.synchronize()
    return out, lg, ctx, drows


# --------------------------------------------------------------------------------------------------------------------
# host side (no GPU)


def test_cluster_size_table():
    op = _op()
    assert op.MAX_BATCH == MAX_ROWS
    assert [op.cluster_size(b) for b in range(1, 16)] == [8] * 15
    assert [op.cluster_size(b) for b in range(16, 17)] == [6]
    assert [op.cluster_size(b) for b in range(17, 25)] == [5] * 8
    assert [op.cluster_size(b) for b in range(25, 27)] == [4] * 2
    for b, g in CLUSTER_OF.items():
        assert op.cluster_size(b) == g
    for bad in (-1, 0, *REFUSED):
        assert op.cluster_size(bad) is None


def test_clusters_of_every_batch_fit_one_wave():
    """All clusters of a launch are resident at once (a second wave costs ~8 us), and the grid fits the 148 SMs."""
    op = _op()
    for b in range(1, MAX_ROWS + 1):
        g = op.cluster_size(b)
        assert g in CO_RESIDENT
        assert b <= CO_RESIDENT[g], f"{b} clusters of {g} CTAs"
        assert b * g <= 148


def test_max_batch_is_the_range_of_the_tail_attention():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    assert _op().MAX_BATCH == tail_op.TAIL_MAX_BATCH == MAX_ROWS


def test_batch_size_support():
    op = _op()
    for b in range(1, MAX_ROWS + 1):
        assert op.is_batch_size_supported(b) == (True, "")
    for bad in (-1, 0, *REFUSED):
        ok, why = op.is_batch_size_supported(bad)
        assert not ok and f"batch size {bad} not in [1, {MAX_ROWS}]" in why


def test_is_supported_has_the_signature_of_the_other_ops():
    """is_supported(device, config) like the megakernel's, the large batch's and the tail attention's."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import (
        large_batch_op,
        megakernel_op,
        tail_op,
    )

    def params(fn):
        return list(inspect.signature(fn).parameters)

    assert params(_op().is_supported) == ["device", "config"]
    for other in (megakernel_op, large_batch_op, tail_op):
        assert params(_op().is_supported) == params(other.is_supported)


def test_is_supported_host_checks():
    op = _op()
    ok, why = op.is_supported(config=PRO)
    assert not ok and "DS-V4-Pro" in why
    # a supported spec: the answer is the device check
    for config in (None, SPEC, SPEC_FP4):
        ok, why = op.is_supported(config=config)
        assert ok == (not why)


def test_default_config_of_every_entry_point_is_the_same():
    op = _op()
    for fn in (op.precompile, op.run):
        assert inspect.signature(fn).parameters["config"].default is None, fn.__name__


@requires_b200
def test_is_supported_on_b200():
    op = _op()
    for config in (None, SPEC, SPEC_FP4):
        assert op.is_supported(torch.device("cuda"), config) == (True, "")
        assert op.is_supported(config=config) == (True, "")


def test_is_profitable():
    op = _op()
    n0 = op.MIN_PROFITABLE_ROW_LEN
    assert n0 == 16384
    i32 = torch.int32
    assert op.is_profitable(torch.full((4,), n0, dtype=i32))
    assert op.is_profitable(torch.tensor([n0, 262144, 100000], dtype=i32))
    assert not op.is_profitable(
        torch.tensor([n0, n0 - 1, 262144], dtype=i32)
    )  # the slowest row decides
    assert not op.is_profitable(torch.tensor([512, 262144], dtype=i32))
    assert not op.is_profitable(torch.zeros(0, dtype=i32))


def test_cluster_sizes_of_the_op_are_the_ones_of_the_kernel():
    """The op offers the cluster sizes the kernel is built and validated for, and the kernel refuses the others."""
    pytest.importorskip("cutlass.cute")  # CuTe DSL; no GPU needed
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import topk_multicta as TM

    op = _op()
    assert {op.cluster_size(b) for b in range(1, MAX_ROWS + 1)} == set(TM.SUPPORTED_CLUSTER_SIZES)
    assert set(TM.SUPPORTED_CLUSTER_SIZES) == set(CLUSTER_SIZES)
    for g in (0, 1, 2, 3, 7, 9, 16):
        with pytest.raises(ValueError, match="cluster size"):
            TM.TopkMultiCta(g)
    for g in CLUSTER_SIZES:
        assert TM.TopkMultiCta(g).g == g
        assert TM.TopkMultiCta(g, force_fallback=True).force_fallback


def _host_args(batch, cols=NPAD):
    """Valid CPU arguments of run (the checks run before any device work)."""
    return dict(
        logits=torch.zeros((batch, cols), dtype=torch.float32),
        topk=torch.full((batch, K_TOP), -1, dtype=torch.int32),
        ctx_lens=torch.full((batch,), 1000, dtype=torch.int32),
    )


def test_rejects_batches_above_26():
    op = _op()
    for bad in (0, *REFUSED):
        # the batch size is the leading dimension of logits (narrow rows: a CPU tensor of 1024 x 2**18 floats is 1 GiB)
        with pytest.raises(ValueError, match=rf"batch size {bad} not in \[1, {MAX_ROWS}\]"):
            op.run(**_host_args(bad, cols=128))
        for config in (SPEC, SPEC_FP4):
            with pytest.raises(ValueError, match=rf"batch size {bad} not in \[1, {MAX_ROWS}\]"):
                op.precompile(torch.device("cuda"), bad, config)


def _transposed(t):
    return t.t().contiguous().t()


# (argument, a variant of it, the error). The batch size is 4.
BAD_ARGUMENTS = [
    ("logits", lambda t: t.to(torch.float16), r"logits must be a float32 \[B, npad\] tensor"),
    ("logits", lambda t: t.to(torch.bfloat16), r"logits must be a float32 \[B, npad\] tensor"),
    ("logits", lambda t: t.to(torch.int32), r"logits must be a float32 \[B, npad\] tensor"),
    ("logits", lambda t: t.to(torch.float64), r"logits must be a float32 \[B, npad\] tensor"),
    ("logits", lambda t: t[0], r"logits must be a float32 \[B, npad\] tensor"),
    ("logits", lambda t: t[None], r"logits must be a float32 \[B, npad\] tensor"),
    ("logits", _transposed, r"logits must be a float32 \[B, npad\] tensor with unit column stride"),
    (
        "logits",
        lambda t: t[:, ::2],
        r"logits must be a float32 \[B, npad\] tensor with unit column stride",
    ),
    (
        "logits",
        lambda t: torch.zeros((4, 1025))[:, :1024],
        "logits row stride must be a multiple of 128 floats",
    ),
    (
        "logits",
        lambda t: torch.zeros((4, 1000))[:, :500],
        "logits row stride must be a multiple of 128 floats",
    ),
    (
        "logits",
        lambda t: torch.zeros((4, 2 * NPAD)),
        r"logits must have 1 to 262144 columns",
    ),
    ("logits", lambda t: torch.zeros((4, 0)), r"logits must have 1 to 262144 columns"),
    (
        "logits",
        lambda t: torch.as_strided(torch.zeros(4 * 1024), (4, 1024), (512, 1)),
        "at least the row length",
    ),
    ("topk", lambda t: t.to(torch.int64), r"topk must be a contiguous int32 \[4, 512\] tensor"),
    (
        "topk",
        lambda t: t[:, :-1].contiguous(),
        r"topk must be a contiguous int32 \[4, 512\] tensor",
    ),
    ("topk", lambda t: t[:3].contiguous(), r"topk must be a contiguous int32 \[4, 512\] tensor"),
    ("topk", _transposed, r"topk must be a contiguous int32 \[4, 512\] tensor"),
    ("topk", lambda t: t.reshape(-1), r"topk must be a contiguous int32 \[4, 512\] tensor"),
    (
        "topk",
        lambda t: torch.zeros((4, 1024), dtype=torch.int32)[:, ::2],
        "topk must be a contiguous",
    ),
    ("ctx_lens", lambda t: t.to(torch.int64), r"ctx_lens must be a contiguous int32 \[4\] tensor"),
    ("ctx_lens", lambda t: t[:3].contiguous(), r"ctx_lens must be a contiguous int32 \[4\] tensor"),
    ("ctx_lens", lambda t: t.reshape(2, 2), r"ctx_lens must be a contiguous int32 \[4\] tensor"),
    (
        "ctx_lens",
        lambda t: torch.arange(8, dtype=t.dtype)[::2],
        r"ctx_lens must be a contiguous int32 \[4\] tensor",
    ),
    (
        "ctx_lens",
        lambda t: t.to(torch.float32),
        r"ctx_lens must be a contiguous int32 \[4\] tensor",
    ),
]


@pytest.mark.parametrize(
    "name, variant, match",
    [pytest.param(*c, id=f"{i}-{c[0]}") for i, c in enumerate(BAD_ARGUMENTS)],
)
def test_rejects_bad_arguments(name, variant, match):
    args = _host_args(4, cols=1024)
    args[name] = variant(args[name])
    with pytest.raises(ValueError, match=match):
        _op().run(**args)


def test_rejects_tensors_that_are_not_on_a_cuda_device():
    """Valid arguments on the CPU: the kernel reads device memory."""
    with pytest.raises(ValueError, match="must be CUDA tensors, got logits on cpu"):
        _op().run(**_host_args(4, cols=1024))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_rejects_tensors_on_other_devices():
    """Every tensor must be on the device of logits: a tensor on the CPU is refused by name."""
    op = _op()
    args = {k: v.cuda() for k, v in _host_args(4, cols=1024).items()}
    for name in ("topk", "ctx_lens"):
        bad = dict(args)
        bad[name] = args[name].cpu()
        with pytest.raises(
            ValueError, match=rf"{name} must be on the device of logits \(cuda:\d+\), got cpu"
        ):
            op.run(**bad)
    bad = dict(args)
    bad["logits"] = args["logits"].cpu()
    with pytest.raises(ValueError, match="must be CUDA tensors, got logits on cpu"):
        op.run(**bad)


def test_unsupported_spec_is_refused_before_any_device_work():
    op = _op()
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        op.precompile(torch.device("cpu"), 4, PRO)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        op.run(**_host_args(4, cols=1024), config=PRO)


def _stub_hybrid_args(batch):
    """CPU stand-ins for the arguments of run_hybrid (the stages are replaced in these tests)."""
    return T._host_args(batch)


def test_default_launch_of_the_hybrid_follows_the_batch_size():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    assert tail_op.PDL_MAX_BATCH == 24
    assert [tail_op.default_pdl(b) for b in range(1, MAX_ROWS + 1)] == [True] * 24 + [False] * 2


@pytest.mark.parametrize("batch", [3, 24, 25, 26])
def test_run_hybrid_with_the_multicta_top_k_calls_the_stages_in_order(monkeypatch, batch):
    """K1 logits, then the multi-CTA top-K on those logits and the lengths, then the tail attention with the caller's
    tensors."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    op = _op()
    calls = []
    args = _stub_hybrid_args(batch)
    logits, ctx = object(), object()

    def k1():
        calls.append("k1")
        return logits

    def fake_topk(lg, topk, ctx_lens, *, config=None, trigger=False):
        assert lg is logits and ctx_lens is ctx, (
            "the top-K stage gets the logits of the K1 stage and ctx_lens"
        )
        assert topk is args["topk"]
        calls.append(("multicta", config, trigger))

    def fake_attention(*a, **kw):
        calls.append(("attention", a, kw))

    monkeypatch.setattr(op, "run", fake_topk)
    monkeypatch.setattr(tail_op, "run_attention", fake_attention)
    names = ("topk", "q_att", "swa_pool", "bt_swa", "c_pool", "bt_c", "kv_lens", "sinks", "out")
    tail_op.run_hybrid(k1, None, *(args[n] for n in names), 0.125, config=SPEC, ctx_lens=ctx)
    # the multi-CTA top-K triggers the dependent launch and the tail attention launches programmatically by default, up
    # to PDL_MAX_BATCH rows (above it the clusters of the tail attention fill the wave: plain launches)
    pdl = batch <= tail_op.PDL_MAX_BATCH
    assert calls[:2] == ["k1", ("multicta", SPEC, pdl)] and len(calls) == 3
    _, a, kw = calls[2]
    assert len(a) == 10 and all(x is args[n] for x, n in zip(a, names)) and a[9] == 0.125
    assert kw == {"config": SPEC, "pdl": pdl}


def test_run_hybrid_without_a_top_k_function_needs_ctx_lens(monkeypatch):
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    args = _stub_hybrid_args(3)
    called = []
    monkeypatch.setattr(tail_op, "run_attention", lambda *a, **k: called.append("attention"))
    names = ("topk", "q_att", "swa_pool", "bt_swa", "c_pool", "bt_c", "kv_lens", "sinks", "out")
    with pytest.raises(ValueError, match="needs ctx_lens"):
        tail_op.run_hybrid(lambda: called.append("k1"), None, *(args[n] for n in names))
    assert called == [], "refused before any stage ran"


def test_run_hybrid_with_a_top_k_function_does_not_touch_the_multicta_kernel(monkeypatch):
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    op = _op()
    args = _stub_hybrid_args(3)
    calls = []
    monkeypatch.setattr(op, "run", lambda *a, **k: calls.append("multicta"))
    monkeypatch.setattr(tail_op, "run_attention", lambda *a, **k: calls.append("attention"))
    names = ("topk", "q_att", "swa_pool", "bt_swa", "c_pool", "bt_c", "kv_lens", "sinks", "out")
    tail_op.run_hybrid(
        lambda: calls.append("k1"),
        lambda lg: calls.append("topk"),
        *(args[n] for n in names),
        ctx_lens=torch.zeros(3, dtype=torch.int32),
    )
    assert calls == ["k1", "topk", "attention"]


# --------------------------------------------------------------------------------------------------------------------
# kernel: exactness


@requires_b200
@pytest.mark.parametrize("batch", BOUNDARY_BATCHES)
def test_exact_mixed_kinds_at_the_cluster_boundaries(batch, spec):
    """Every cluster size (8 / 6 / 5 / 4 CTAs: 1-15 / 16 / 17-24 / 25-26 rows) on rows of every kind and long
    lengths."""
    lengths = [262144, 200000, 131071, 100001, 65536, 28672, 20000, 16384]
    rows = _make(KINDS, lengths, seed=100 + batch, batch=batch)
    lg, ctx, drows = _logits(rows)
    out = torch.full((batch, K_TOP), -9, dtype=torch.int32, device="cuda")
    _op().run(lg, out, ctx, config=spec)
    torch.cuda.synchronize()
    _check(drows, out)


@requires_b200
@pytest.mark.parametrize("batch", [3, 16, MAX_ROWS])
@pytest.mark.parametrize("kind", KINDS)
def test_exact_per_row_kind(kind, batch):
    """Rows of one kind, long and mid lengths, with 8 / 6 / 4 CTAs per row; the kinds with ties and plateaus take the
    exact fallback on most rows, the others the fast path."""
    lengths = [262144, 131071, 100001, 65536, 28672, 20000, 16384, 8193]
    out, _, _, drows = _run(_make([kind], lengths, seed=7, batch=batch))
    _check(drows, out)


@requires_b200
@pytest.mark.parametrize("n", EDGES)
def test_exact_at_every_edge_length(n):
    """Thirteen rows (6 CTAs per row) of the same length n and different kinds: the short rows (identity), the rows
    without a sample (<= 8192), the first sampled lengths and the lengths that end inside a chunk."""
    kinds = ["gauss", "lognorm", "quant", "signed_tail", "neg", "ties512", "flat"]
    out, _, _, drows = _run(_make(kinds, [n], seed=11 + n, batch=13))
    _check(drows, out)


@requires_b200
@pytest.mark.parametrize("batch", [5, 13, MAX_ROWS])
def test_exact_with_a_different_length_in_every_row(batch):
    """Rows of one batch with lengths from 1 to 2**18 (a cluster of every kind, the slowest row decides the launch)."""
    lengths = random.Random(batch).sample(EDGES, batch)
    kinds = ["gauss", "signed_tail", "quant", "lognorm", "neg", "plateau_top"]
    out, _, _, drows = _run(_make(kinds, lengths, seed=300 + batch, batch=batch))
    _check(drows, out)


@requires_b200
@pytest.mark.parametrize("batch", [1, 13, 16, 17, 25, 26])
def test_exact_fallback_forced_on_every_row(batch):
    """The in-kernel radix select on every row, including the rows whose fast path would certify: the same exactness."""
    lengths = [262144, 131071, 65536, 20000, 16384, 8193, 8192, 4096, 600, 513, 512, 100, 1]
    out, _, _, drows = _run(_make(KINDS, lengths, seed=900 + batch, batch=batch), forced=True)
    _check(drows, out)


@requires_b200
def test_forced_and_default_kernels_select_the_same_set_when_it_is_unique():
    """Rows without ties at the 512th rank have one exact selection: the fast path and the radix select give it."""
    lengths = [262144, 131071, 65536, 28672, 20000, 16384, 8193, 600]
    rows = _make(UNIQUE_KINDS, lengths, seed=5, batch=16)
    out_default, _, _, drows = _run(rows)
    out_forced, _, _, _ = _run(rows, forced=True)
    _check(drows, out_default)
    _check(drows, out_forced)
    assert torch.equal(torch.sort(out_default, dim=1).values, torch.sort(out_forced, dim=1).values)


@requires_b200
def test_planted_ties_at_the_512th_value():
    """700 equal values straddling the 512th rank on every row (and a quantized tail): the value multiset is exact
    whichever of the tied positions are taken."""
    out, _, _, drows = _run(
        _make(["ties512", "gauss_tail_ties", "quant"], [262144, 100001, 28672], seed=3, batch=18)
    )
    _check(drows, out)
    # the part of the tied value that the selection could not leave out: every strictly larger value is selected
    for r, x in enumerate(drows):
        sel = out[r].long()
        kth = torch.topk(x, K_TOP).values[-1]
        above = torch.nonzero(x > kth).flatten()
        assert bool(torch.isin(above, sel).all()), f"row {r}: a value above the 512th is missing"


@requires_b200
def test_all_equal_rows_select_512_distinct_positions():
    """Every position ties: any 512 distinct positions of [0, n) are exact."""
    out, _, _, drows = _run([_row("flat", n, None) for n in (262144, 20000, 8193, 513, 1000)] * 3)
    _check(drows, out)


@requires_b200
@pytest.mark.parametrize("poison", [1e30, float("inf"), float("-inf"), float("nan"), -1e30])
def test_never_reads_beyond_the_length(poison):
    """Everything beyond ctx_lens is garbage (the logits arena keeps the values of earlier steps): the largest values,
    infinities and NaN must not be selected or disturb a row. Lengths that end inside a 128-float chunk, a float4 and
    the 8-float sample window."""
    lengths = [
        100001,
        100003,
        100005,
        100127,
        100129,
        20001,
        16385,
        8195,
        8201,
        640,
        641,
        513,
        262143,
    ]
    rows = _make(["gauss", "neg", "quant"], lengths, seed=21, batch=13)
    lg, ctx, drows = _logits(rows, poison=poison)
    out = torch.full((13, K_TOP), -9, dtype=torch.int32, device="cuda")
    _op().run(lg, out, ctx)
    torch.cuda.synchronize()
    _check(drows, out)
    for r, x in enumerate(drows):
        assert int(out[r].max()) < x.numel()


@requires_b200
def test_exact_on_views_of_larger_and_smaller_buffers():
    """A [B, 100000] view of the [B, 2**18] logits arena (row stride 2**18), a [B, 16384] buffer (stride 16384) and a
    [B, 20000] view of a stride-20096 buffer: the row stride, not the number of columns, is the row pitch."""
    op = _op()
    gen = torch.Generator().manual_seed(2)
    for cols, stride in ((100000, NPAD), (16384, 16384), (20000, 20096)):
        b = 6
        rows = [
            _row(k, n, gen)
            for k, n in zip(["gauss", "signed_tail", "quant"] * 2, [cols, cols - 1, cols - 641] * 2)
        ]
        buf = torch.full((b, stride), 1e30)
        for r, x in enumerate(rows):
            buf[r, : x.numel()] = x
        buf = buf.cuda()
        view = buf[:, :cols]
        assert view.stride() == (stride, 1)
        ctx = torch.tensor([x.numel() for x in rows], dtype=torch.int32, device="cuda")
        out = torch.full((b, K_TOP), -9, dtype=torch.int32, device="cuda")
        op.run(view, out, ctx)
        torch.cuda.synchronize()
        _check([x.cuda() for x in rows], out)


@requires_b200
def test_writes_only_the_output_rows():
    """The 512 entries of each row are the only memory the kernel writes: canaries around the output stay."""
    op = _op()
    b = 13
    rows = _make(
        ["gauss", "quant", "ties512", "neg"], [262144, 28672, 100001, 600, 8193], seed=4, batch=b
    )
    lg, ctx, drows = _logits(rows)
    buf = torch.full((b * K_TOP + 2 * 512,), -7, dtype=torch.int32, device="cuda")
    out = buf[512 : 512 + b * K_TOP].view(b, K_TOP)
    out.fill_(-9)
    op.run(lg, out, ctx)
    torch.cuda.synchronize()
    assert bool((buf[:512] == -7).all()) and bool((buf[512 + b * K_TOP :] == -7).all())
    _check(drows, out)
    # the logits and the lengths are read only
    lg2, ctx2, _ = _logits(rows)
    assert torch.equal(lg, lg2) and torch.equal(ctx, ctx2)


@requires_b200
def test_misaligned_logits_are_refused():
    """The row loads are 16-byte accesses: a base that is not a multiple of 16 bytes is a ValueError of the compiled
    function, not a CUDA fault."""
    op = _op()
    b = 4
    buf = torch.zeros(b * NPAD + 16, dtype=torch.float32, device="cuda")
    logits = buf[1 : 1 + b * NPAD].view(b, NPAD)
    assert logits.data_ptr() % 16 == 4
    out = torch.full((b, K_TOP), -9, dtype=torch.int32, device="cuda")
    ctx = torch.full((b,), 1000, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="Misaligned"):
        op.run(logits, out, ctx)
    torch.cuda.synchronize()  # nothing was launched
    assert bool((out == -9).all())


# --------------------------------------------------------------------------------------------------------------------
# kernel: compile, CUDA graphs


@requires_b200
def test_one_compile_per_cluster_size(monkeypatch):
    """The compile cache is keyed by (device, cluster size): 26 batch sizes make 4 compiles, none for a second pass."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import topk_multicta as TM

    op = _op()
    monkeypatch.setattr(op, "_compiled", {})  # as in a fresh process
    compiles = []
    real = TM.compile_topk_multicta
    monkeypatch.setattr(
        TM, "compile_topk_multicta", lambda k: compiles.append((k.g, k.force_fallback)) or real(k)
    )
    dev = torch.device("cuda")
    for b in range(1, MAX_ROWS + 1):
        op.precompile(dev, b)
    assert sorted(g for g, _ in compiles) == sorted(CLUSTER_SIZES) and not any(
        f for _, f in compiles
    )
    assert {(d, g) for (d, g, f, t) in op._compiled} == {
        (torch.cuda.current_device(), g) for g in CLUSTER_SIZES
    }
    for b in range(1, MAX_ROWS + 1):
        op.precompile(dev, b, SPEC)
        op.precompile(dev, b, SPEC_FP4)
    assert len(compiles) == len(CLUSTER_SIZES)


def _fill_graph_buffers(lg, ctx, rows):
    lg.fill_(1e30)
    for r, x in enumerate(rows):
        lg[r, : x.numel()] = x.cuda()
    ctx.copy_(torch.tensor([x.numel() for x in rows], dtype=torch.int32))


@requires_b200
@pytest.mark.parametrize("batch", [1, 8, 13, 17, MAX_ROWS])
def test_cuda_graph_replay_with_refilled_buffers(batch):
    """precompile, capture run() (no warm-up launch), replay with the logits and the lengths refilled in place: new
    data, mixed and short lengths; every replay is exact and equals the eager launch."""
    op = _op()
    dev = torch.device("cuda")
    lg = torch.zeros((batch, NPAD), device=dev)  # the static buffers of the graph
    ctx = torch.full((batch,), 1000, dtype=torch.int32, device=dev)
    out = torch.full((batch, K_TOP), -9, dtype=torch.int32, device=dev)
    op.precompile(dev, batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run(lg, out, ctx)
    steps = [
        (["gauss"], [262144]),
        (["signed_tail", "lognorm"], [131071, 65536, 28672]),
        (["gauss", "neg"], [1, 255, 512, 513, 8192, 8193, 20000, 262144]),
        (["quant", "flat", "ties512"], [262144, 100001, 16384]),
        (["gauss"], [600]),
    ]
    for i, (kinds, lengths) in enumerate(steps):
        rows = _make(kinds, lengths, seed=40 + i, batch=batch)
        _fill_graph_buffers(lg, ctx, rows)
        out.fill_(-9)
        graph.replay()
        torch.cuda.synchronize()
        _check([x.cuda() for x in rows], out)
        # a unique selection (no ties at the 512th rank): the eager launch gives the same set
        if all(k in UNIQUE_KINDS for k in kinds):
            eager = torch.full_like(out, -9)
            op.run(lg, eager, ctx)
            torch.cuda.synchronize()
            same = torch.equal(torch.sort(eager, dim=1).values, torch.sort(out, dim=1).values)
            assert same, f"step {i}"


@requires_b200
@pytest.mark.parametrize(
    "run_config", [None, SPEC, SPEC_FP4], ids=["run-default", "run-fp8", "run-fp4"]
)
@pytest.mark.parametrize(
    "pre_config", [None, SPEC, SPEC_FP4], ids=["pre-default", "pre-fp8", "pre-fp4"]
)
def test_precompile_then_captured_run(monkeypatch, pre_config, run_config):
    """precompile(device, batch) followed by a CUDA-graph capture of run compiles nothing during the capture,
    whichever spec each call names (``None``: the default of both): there is no warm-up launch in between."""
    op = _op()
    batch = 8
    monkeypatch.setattr(op, "_compiled", {})  # as in a fresh process
    lg, ctx, drows = _logits(
        _make(["gauss", "signed_tail", "lognorm"], [262144, 100001, 65536], seed=60, batch=batch)
    )
    out = torch.full((batch, K_TOP), -9, dtype=torch.int32, device="cuda")
    op.precompile(torch.device("cuda"), batch, pre_config)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run(lg, out, ctx, config=run_config)
    graph.replay()
    torch.cuda.synchronize()
    _check(drows, out)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
@requires_b200
def test_compile_during_capture_is_refused(monkeypatch):
    """A kernel that is not compiled yet cannot be compiled inside a capture: precompile first."""
    op = _op()
    # warm-up: the CUDA context and the compile
    out, lg, ctx, drows = _run(_make(["gauss"], [100001], seed=70, batch=4))
    _check(drows, out)
    monkeypatch.setattr(op, "_compiled", {})  # as in a fresh process
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="precompile"):
        with torch.cuda.graph(graph):
            op.run(lg, out, ctx)
    monkeypatch.undo()
    out.fill_(-9)
    op.run(lg, out, ctx)
    torch.cuda.synchronize()
    _check(drows, out)


# --------------------------------------------------------------------------------------------------------------------
# warp skew: the kernel's block barriers and waits order its warps, whatever the schedule


# A pseudo-random warp-uniform delay (nanosleep, up to 1 us for a quarter of the (CTA, warp, site) triples) after every
# block barrier, named barrier, cluster wait, mbarrier wait and cp.async wait of the kernel source.
_SKEW_HELPERS = """
@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    L._asm(None, [L._v(ns, loc, ip)], "nanosleep.u32 $0;", "r", loc, ip)


@cute.jit
def _skew(site: cutlass.Constexpr, bidx, warp):
    h = Uint32((SKEW_SEED * 2654435761 + site * 40503) & 0xFFFFFFFF)
    h = h ^ (Uint32(bidx) * Uint32(2246822519)) ^ (Uint32(warp) * Uint32(3266489917))
    h = h ^ (h >> Uint32(15))
    h = h * Uint32(2246822519)
    h = h ^ (h >> Uint32(13))
    d = Int32(0)
    if (h & Uint32(3)) == Uint32(0):
        d = Int32((h >> Uint32(8)) & Uint32(1023))
    nanosleep(d)


"""


def _skewed_module(tmp_path, seed):
    """The kernel module with the delays inserted, imported as a sibling of ``topk_multicta`` (same package, so its
    relative imports resolve); returns (module, number of delay sites)."""
    import importlib.util
    import re
    import sys

    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import topk_multicta as TM

    text = open(TM.__file__).read()
    pat = re.compile(
        r"^(\s*)(L\.bar\(\d, NT\)|L\.cluster_wait\(\)|L\.wait_cl\(.*\)|cpa_wait\(.*\))\s*$"
    )
    out, sites = [], 0
    for line in text.split("\n"):
        out.append(line)
        m = pat.match(line)
        if m:
            sites += 1
            out.append(f"{m.group(1)}_skew({sites}, bidx, warp)")
    text = "\n".join(out)
    anchor = "@cute.jit\ndef _umin(a, b):"
    assert text.count(anchor) == 1
    text = text.replace(anchor, f"SKEW_SEED = {seed}\n" + _SKEW_HELPERS + anchor)
    path = tmp_path / f"topk_multicta_skew{seed}.py"
    path.write_text(text)
    name = f"{TM.__package__}.topk_multicta_skew{seed}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, sites


@requires_b200
@pytest.mark.parametrize("g, batch", [(8, 12), (4, 26)])
def test_exact_fallback_under_warp_skew(tmp_path, g, batch):
    """The levels of the radix fallback exchange counters through shared memory between the warps of a CTA and
    through distributed shared memory between the CTAs: with random delays after every barrier and wait of the kernel
    the result must stay exact. (A missing barrier between the reads at the end of a level and the zeroing at the top of
    the next was invisible at the normal schedule and gave half-empty outputs here and under compute-sanitizer.)"""
    module, sites = _skewed_module(tmp_path, seed=g)
    assert sites >= 25, "the kernel source changed: the delay sites are not found"
    kernel = module.compile_topk_multicta(module.TopkMultiCta(g, force_fallback=True))
    lengths = [262144, 131071, 65536, 20000, 16384, 8193, 4096, 600]
    for seed in (1, 2):
        rows = _make(KINDS, lengths, seed=700 + seed, batch=batch)
        lg, ctx, drows = _logits(rows)
        out = torch.full((batch, K_TOP), -9, dtype=torch.int32, device="cuda")
        kernel(lg, ctx, out)
        torch.cuda.synchronize()
        _check(drows, out)


# --------------------------------------------------------------------------------------------------------------------
# the hybrid: K1 logits -> multi-CTA top-K -> tail attention


def _hybrid_case(batch, rot, seed):
    """Tail-attention inputs of ``batch`` rows and logits of the compressed lengths kv_len // 4 for the top-K stage."""
    rows = T._rows(batch, rot)
    inp = T._make_case(rows, seed=seed)
    ctx = (inp["kv_lens"] // T.RATIO).to(torch.int32)
    gen = torch.Generator().manual_seed(seed + 5)
    kinds = ["gauss", "signed_tail", "lognorm"]
    lg, _, drows = _logits([_row(kinds[b % 3], int(n), gen) for b, n in enumerate(ctx.tolist())])
    return inp, lg, ctx, drows


@requires_b200
@pytest.mark.parametrize("batch", [5, 13, 17, MAX_ROWS])
def test_run_hybrid_with_the_multicta_top_k(batch):
    """K1 stand-in -> multi-CTA top-K -> tail attention: the selection is the exact top-512 of the logits, the attention
    matches the fp64 reference of that selection, and it is bit-equal to the tail attention launched on the same
    selection (the top-K output feeds the attention directly)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    inp, lg, ctx, drows = _hybrid_case(batch, 3, seed=2000 + batch)
    topk = torch.full((batch, K_TOP), -9, dtype=torch.int32, device="cuda")
    out = T._new_out(batch)
    events = []

    def k1():
        events.append("k1")
        return lg

    tail_op.run_hybrid(
        k1,
        None,
        topk,
        inp["q_att"],
        inp["swa_pool"],
        inp["bt_swa"],
        inp["c_pool"],
        inp["bt_c"],
        inp["kv_lens"],
        inp["sinks"],
        out,
        config=SPEC_FP4,
        ctx_lens=ctx,
    )
    torch.cuda.synchronize()
    assert events == ["k1"]
    _check(drows, topk)
    T._check_rows({**inp, "topk": topk}, out, "realistic")
    same = T._new_out(batch)
    T._launch({**inp, "topk": topk}, same, SPEC_FP4)
    torch.cuda.synchronize()
    assert torch.equal(out, same)


@requires_b200
def test_run_hybrid_with_the_multicta_top_k_in_a_cuda_graph():
    """Both kernels compiled before the capture, no warm-up launch; the replay runs the K1 stand-in, the top-K and the
    attention on refilled buffers."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    op = _op()
    batch = 16
    inp, lg, ctx, drows = _hybrid_case(batch, 3, seed=2100)
    topk = torch.full((batch, K_TOP), -9, dtype=torch.int32, device="cuda")
    out = T._new_out(batch)

    def hybrid():
        tail_op.run_hybrid(
            lambda: lg,  # the K1 stand-in: the static logits buffer
            None,
            topk,
            inp["q_att"],
            inp["swa_pool"],
            inp["bt_swa"],
            inp["c_pool"],
            inp["bt_c"],
            inp["kv_lens"],
            inp["sinks"],
            out,
            config=SPEC_FP4,
            ctx_lens=ctx,
        )

    # the default variants of run_hybrid with the multi-CTA top-K are the programmatic ones
    tail_op.precompile(torch.device("cuda"), batch, SPEC_FP4, pdl=True)
    op.precompile(torch.device("cuda"), batch, SPEC_FP4, trigger=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hybrid()
    graph.replay()
    torch.cuda.synchronize()
    _check(drows, topk)
    T._check_rows({**inp, "topk": topk}, out, "realistic")
    # new logits and lengths in place, the same attention inputs: the replay follows them
    ctx2 = ((inp["kv_lens"] // T.RATIO) // 2 + 1).to(torch.int32)
    gen = torch.Generator().manual_seed(2101)
    rows2 = [_row("gauss", int(n), gen) for n in ctx2.tolist()]
    lg2, _, drows2 = _logits(rows2)
    lg.copy_(lg2)
    ctx.copy_(ctx2)
    topk.fill_(-9)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _check(drows2, topk)
    T._check_rows({**inp, "topk": topk}, out, "realistic")


# --------------------------------------------------------------------------------------------------------------------
# programmatic dependent launch: the top-K triggers after its stream phase, the tail attention launches with it


def test_precompile_and_run_take_the_trigger_flag():
    op = _op()
    for fn in (op.precompile, op.run):
        p = inspect.signature(fn).parameters["trigger"]
        assert p.default is False and p.kind is not p.POSITIONAL_ONLY


def test_run_hybrid_pdl_needs_the_multicta_top_k(monkeypatch):
    """pdl=True with the caller's top-K kernel is refused before any stage runs (that kernel never triggers early)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    args = _stub_hybrid_args(3)
    called = []
    monkeypatch.setattr(tail_op, "run_attention", lambda *a, **k: called.append("attention"))
    names = ("topk", "q_att", "swa_pool", "bt_swa", "c_pool", "bt_c", "kv_lens", "sinks", "out")
    with pytest.raises(ValueError, match="pdl needs the multi-CTA top-K"):
        tail_op.run_hybrid(
            lambda: called.append("k1"), lambda lg: called.append("topk"),
            *(args[n] for n in names), pdl=True,
        )  # fmt: skip
    assert called == [], "refused before any stage ran"


@pytest.mark.parametrize("pdl, expected", [(None, True), (True, True), (False, False)])
def test_run_hybrid_pdl_selects_the_launch_variants_of_both_kernels(monkeypatch, pdl, expected):
    """With the multi-CTA top-K ``pdl`` (default: on) makes it trigger the dependent launch and launches the tail
    attention with the programmatic attribute; off, both are plain launches."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    op = _op()
    calls = []
    args = _stub_hybrid_args(3)
    monkeypatch.setattr(
        op,
        "run",
        lambda lg, topk, ctx, *, config=None, trigger=False: calls.append(("topk", trigger)),
    )
    monkeypatch.setattr(
        tail_op, "run_attention", lambda *a, pdl=False, **k: calls.append(("attention", pdl))
    )
    names = ("topk", "q_att", "swa_pool", "bt_swa", "c_pool", "bt_c", "kv_lens", "sinks", "out")
    kw = {} if pdl is None else {"pdl": pdl}
    tail_op.run_hybrid(lambda: object(), None, *(args[n] for n in names), ctx_lens=object(), **kw)
    assert calls == [("topk", expected), ("attention", expected)]


@requires_b200
def test_one_compile_per_cluster_size_and_trigger(monkeypatch):
    """The trigger is part of the compile: the 4 cluster sizes compile once for each value, none for a second pass."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import topk_multicta as TM

    op = _op()
    monkeypatch.setattr(op, "_compiled", {})
    compiles = []
    real = TM.compile_topk_multicta
    monkeypatch.setattr(
        TM, "compile_topk_multicta", lambda k: compiles.append((k.g, k.trigger)) or real(k)
    )
    dev = torch.device("cuda")
    for b in range(1, MAX_ROWS + 1):
        op.precompile(dev, b)
        op.precompile(dev, b, SPEC_FP4, trigger=True)
    for b in range(1, MAX_ROWS + 1):
        op.precompile(dev, b, SPEC, trigger=True)
    assert sorted(compiles) == sorted(
        [(g, False) for g in CLUSTER_SIZES] + [(g, True) for g in CLUSTER_SIZES]
    )


@requires_b200
@pytest.mark.parametrize("batch", [5, 9, 13, 17, MAX_ROWS])
def test_run_hybrid_pdl_is_bit_equal_to_the_plain_launches(batch):
    """K1 stand-in -> multi-CTA top-K (triggering) -> tail attention (programmatic launch) against the two plain
    launches of the same kernels: the same selection and the same attention output, bit for bit (several launches in a
    row)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    inp, lg, ctx, drows = _hybrid_case(batch, 3, seed=2200 + batch)
    runs = {}
    for pdl in (False, True, False, True):
        topk = torch.full((batch, K_TOP), -9, dtype=torch.int32, device="cuda")
        out = T._new_out(batch)
        tail_op.run_hybrid(
            lambda: lg, None, topk, inp["q_att"], inp["swa_pool"], inp["bt_swa"], inp["c_pool"],
            inp["bt_c"], inp["kv_lens"], inp["sinks"], out, config=SPEC_FP4, ctx_lens=ctx, pdl=pdl,
        )  # fmt: skip
        torch.cuda.synchronize()
        _check(drows, topk)
        T._check_rows({**inp, "topk": topk}, out, "realistic")
        runs.setdefault(pdl, []).append((topk.sort(dim=1).values, out))
    for pdl, lst in runs.items():
        for t2, o2 in lst[1:]:
            assert torch.equal(t2, lst[0][0]) and torch.equal(o2, lst[0][1]), (
                "deterministic per variant"
            )
    assert torch.equal(runs[True][0][0], runs[False][0][0])
    assert torch.equal(runs[True][0][1], runs[False][0][1])


@requires_b200
def test_run_hybrid_pdl_in_a_cuda_graph():
    """Both kernels compiled in their programmatic variants before the capture; the replay follows refilled buffers."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    op = _op()
    batch = 16
    inp, lg, ctx, drows = _hybrid_case(batch, 3, seed=2300)
    topk = torch.full((batch, K_TOP), -9, dtype=torch.int32, device="cuda")
    out = T._new_out(batch)

    def hybrid():
        tail_op.run_hybrid(
            lambda: lg, None, topk, inp["q_att"], inp["swa_pool"], inp["bt_swa"], inp["c_pool"],
            inp["bt_c"], inp["kv_lens"], inp["sinks"], out, config=SPEC_FP4, ctx_lens=ctx, pdl=True,
        )  # fmt: skip

    tail_op.precompile(torch.device("cuda"), batch, SPEC_FP4, pdl=True)
    op.precompile(torch.device("cuda"), batch, SPEC_FP4, trigger=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hybrid()
    for seed in (2301, 2302):
        ctx2 = ((inp["kv_lens"] // T.RATIO) // 2 + 1).to(torch.int32)
        gen = torch.Generator().manual_seed(seed)
        lg2, _, drows2 = _logits([_row("gauss", int(n), gen) for n in ctx2.tolist()])
        lg.copy_(lg2)
        ctx.copy_(ctx2)
        topk.fill_(-9)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _check(drows2, topk)
        T._check_rows({**inp, "topk": topk}, out, "realistic")
