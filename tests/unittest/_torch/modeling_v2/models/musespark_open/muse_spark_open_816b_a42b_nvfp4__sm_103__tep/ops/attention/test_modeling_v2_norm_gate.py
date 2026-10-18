# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open attention_norm_gate entry (the PDL norm-gate of ``boundary_early_release``).

The entry claims the built-in output gate's bits: ``MuseSparkOpenAttention.
apply_output_gate``, which launches ``_attention_norm_gate_kernel`` at these
shapes. Every cell is compared as raw bits against that method on a built-in
attention module at one tep8 rank's geometry, and against the entry's float64
reference within a stated band.

The PDL memory-order claims are tested directly:

* the PTX of every trigger variant reads no global memory before its
  ``griddepcontrol.wait`` (the PTX audit);
* a negative control: a PDL predecessor that releases its dependents at entry
  and then holds its output back for tens of microseconds, followed by this
  kernel and by the attention output projection (flashinfer split-K
  ``(64,8,1,12)``, which loads its weight before its own wait, and cuBLAS),
  eagerly and through CUDA-graph replays with changing inputs. The output
  buffer is poisoned with NaN before every replay, so a dependent that read
  before its predecessor's stores landed would see it. Every output must equal
  the plain (non-PDL) chain's bits;
* the same at rows 1-8 with the gate written late two kernels back, as the
  model's prep writes it ahead of the attention kernels.

``MODELING_V2_TIMING=1`` also runs the o_proj-site chain microbenchmark
(printed, never gated): 62 layers of [attention stand-in -> output gate ->
o_proj -> PDL successor] with cold weights, the built-in gate against the PDL
copy, cuBLAS against split-K, with and without the ``dense_gemm_prefetch`` L2 prefetch.
"""

import os
import re
import statistics

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    norm_gate as norm_gate_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.norm_gate import (  # noqa: E501
    HEAD_DIM,
    TRIGGERS,
    AttentionNormGate,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.norm_gate import (  # noqa: E501
    attention_norm_gate as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
    dense_gemm,
    kernel_error,
    l2_prefetch,
)

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "attention_norm_gate requires a CUDA device"

DEV = "cuda"
HEADS = 16
WIDTH = HEADS * HEAD_DIM
#: The attention output projection at one tep8 rank: [8192, 1024] BF16.
O_PROJ_N = 8192
SPLITK = "splitk:64,8,1,12"
EPS = 1e-5


def _attention():
    """The built-in attention module at one tep8 rank's geometry (16 query heads of 64)."""
    from tensorrt_llm._torch.configs.musespark_open import MuseSparkOpenConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenAttention
    from tensorrt_llm.mapping import Mapping

    config = MuseSparkOpenConfig(
        num_attention_heads=HEADS,
        num_key_value_heads=2,
        moe={"topk": 8, "num_experts": 256, "override_routed_hidden_dim": 4096},
    )
    model_config = ModelConfig(
        pretrained_config=config, mapping=Mapping(world_size=1, tp_size=1, rank=0)
    )
    with torch.device(DEV):
        return MuseSparkOpenAttention(model_config, 0)


_BUILTIN = {}


def _builtin():
    if "attention" not in _BUILTIN:
        _BUILTIN["attention"] = _attention()
    return _BUILTIN["attention"]


def _inputs(rows: int, gate_ndim: int, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    scale = torch.exp(torch.randn(rows, HEADS, 1, device=DEV, generator=gen) * 3.0)
    attention = (torch.randn(rows, HEADS, HEAD_DIM, device=DEV, generator=gen) * scale).to(
        torch.bfloat16
    )
    attention = attention.reshape(rows, WIDTH)
    gate = (torch.randn(rows, WIDTH, device=DEV, generator=gen) * 4.0).to(torch.bfloat16)
    if gate_ndim == 3:
        gate = gate.reshape(rows, HEADS, HEAD_DIM)
    return attention, gate


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype, what
    differ = (_bits(got) != _bits(want)).sum().item()
    assert differ == 0, f"{what}: {differ} values differ in bits from the built-in output gate"


@pytest.mark.parametrize("cell", AttentionNormGate.CELLS, ids=lambda c: c.why[:48])
@torch.inference_mode()
def test_cells_match_builtin_bits_and_reference(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    attention_module = _builtin()
    eps = attention_module.o_head_norm.variance_epsilon
    for seed in range(4):
        attention, gate = _inputs(spec["rows"], spec["gate_ndim"], 7 + seed)
        want = attention_module.apply_output_gate(attention, gate)
        got = op(attention, gate, eps, spec["trigger"])
        torch.cuda.synchronize()
        _assert_same_bits(got, want, f"{cell.why} seed {seed}")
        ref = op.reference(attention, gate, eps, spec["trigger"])
        # The kernel and the reference differ in the variance sum's order and the
        # approximate rsqrt (~2^-22 relative), which can move a normalized value
        # across a BF16 rounding threshold (one ulp of the row's scale), and then
        # the product's rounding once more: two ulps per element, well under one in RMS.
        assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.5)


#: BF16 bit patterns placed in heads: -0, the smallest subnormal, the largest
#: finite, the smallest normal, the most negative finite, +inf, NaN.
_SPECIAL = (-32768, 1, 32639, 128, -129, 0x7F80, 0x7FC0)


@pytest.mark.parametrize("trigger", sorted(TRIGGERS))
@torch.inference_mode()
def test_special_heads_match_builtin_bits(trigger):
    attention_module = _builtin()
    eps = attention_module.o_head_norm.variance_epsilon
    attention, gate = _inputs(8, 2, 99)
    attention.view(torch.int16)[0, : len(_SPECIAL)] = torch.tensor(_SPECIAL, dtype=torch.int16)
    gate.view(torch.int16)[1, : len(_SPECIAL)] = torch.tensor(_SPECIAL, dtype=torch.int16)
    attention[2, :HEAD_DIM] = 0  # an exact-zero head: rsqrt(eps) times zero
    gate[3, HEAD_DIM : 2 * HEAD_DIM] = 60.0  # sigmoid saturates to exactly one
    gate[3, 2 * HEAD_DIM : 3 * HEAD_DIM] = -60.0  # and to zero
    attention[4, 3 * HEAD_DIM : 4 * HEAD_DIM] = 3.0e38  # squares overflow FP32
    want = attention_module.apply_output_gate(attention, gate)
    got = op(attention, gate, eps, trigger)
    torch.cuda.synchronize()
    _assert_same_bits(got, want, f"special values, trigger {trigger}")


# --------------------------------------------------------------------------
# PDL memory order
# --------------------------------------------------------------------------
@triton.jit
def _globaltimer(dummy):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _late_writer_kernel(src, dst, n, delay_ns, BLOCK: tl.constexpr, TRIGGER_AT_ENTRY: tl.constexpr):
    """A PDL predecessor that releases its dependents first and writes its output late."""
    if TRIGGER_AT_ENTRY:
        tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def _late_writer(src: torch.Tensor, dst: torch.Tensor, delay_ns: int, entry: bool = True):
    n = src.numel()
    _late_writer_kernel[(triton.cdiv(n, 1024),)](
        src, dst, n, delay_ns, BLOCK=1024, TRIGGER_AT_ENTRY=entry, num_warps=4, launch_pdl=True
    )


def _plain_chain(src, gate, weight, kernel, eps):
    """The built-in order without PDL: copy, the built-in gate, the projection."""
    attention = src.clone()
    gated = _builtin().apply_output_gate(attention, gate)
    return gated, dense_gemm(gated, weight, kernel, False)


@pytest.mark.parametrize("trigger", sorted(TRIGGERS))
@pytest.mark.parametrize("kernel", [SPLITK, "cublas"])
@torch.inference_mode()
def test_a_late_predecessor_never_reaches_a_dependent(trigger, kernel):
    """Negative control: every dependent sees its predecessors' final values, eager and replayed."""
    eps = _builtin().o_head_norm.variance_epsilon
    rows = 8
    src = torch.empty(rows, WIDTH, device=DEV, dtype=torch.bfloat16)
    gate = torch.empty(rows, WIDTH, device=DEV, dtype=torch.bfloat16)
    attention = torch.empty_like(src)
    gated = torch.empty_like(src)
    weight = (torch.randn(O_PROJ_N, WIDTH, device=DEV) * 0.02).to(torch.bfloat16)

    def chain():
        _late_writer(src, attention, delay_ns=40_000)
        op(attention, gate, eps, trigger, out=gated)
        return dense_gemm(gated, weight, kernel, True)

    def load(seed):
        a, g = _inputs(rows, 2, seed)
        src.copy_(a)
        gate.copy_(g)
        attention.fill_(float("nan"))
        gated.fill_(float("nan"))

    load(1)
    out = chain()  # eager, also compiles
    torch.cuda.synchronize()
    want_gated, want_out = _plain_chain(src, gate, weight, kernel, eps)
    _assert_same_bits(gated, want_gated, f"eager gate, trigger {trigger}")
    _assert_same_bits(out, want_out, f"eager o_proj {kernel}, trigger {trigger}")

    graph = torch.cuda.CUDAGraph()
    load(2)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = chain()
    for seed in range(3, 9):
        load(seed)
        graph.replay()
        torch.cuda.synchronize()
        want_gated, want_out = _plain_chain(src, gate, weight, kernel, eps)
        _assert_same_bits(gated, want_gated, f"replay {seed} gate, trigger {trigger}")
        _assert_same_bits(captured, want_out, f"replay {seed} o_proj {kernel}, trigger {trigger}")


@pytest.mark.parametrize("trigger", sorted(TRIGGERS))
@pytest.mark.parametrize("kernel", ["cublas", SPLITK])
@pytest.mark.parametrize("rows", range(1, 9))
@torch.inference_mode()
def test_a_late_gate_two_kernels_back_never_reaches_a_dependent(rows, kernel, trigger):
    """Negative control on the gate operand, as the model writes it: two kernels back.

    In a decode step ``preattn_prep`` writes the gate and the attention kernels sit
    between it and the norm-gate, whose own wait covers only the attention's
    last kernel. Here a PDL writer releases its dependents at entry and writes
    the gate late, the next PDL writer the attention output a little later,
    then the norm-gate and the projection; every buffer is NaN-poisoned before
    each replay. Bits must equal the built-in gate and the projection launched
    without PDL, synchronized.
    """
    if kernel_error(kernel, rows, WIDTH, O_PROJ_N) is not None:
        pytest.skip(f"{kernel} does not serve {rows} rows of o_proj")
    eps = _builtin().o_head_norm.variance_epsilon
    src_attention = torch.empty(rows, WIDTH, device=DEV, dtype=torch.bfloat16)
    src_gate = torch.empty_like(src_attention)
    attention = torch.empty_like(src_attention)
    gate = torch.empty_like(src_attention)
    gated = torch.empty_like(src_attention)
    weight = (torch.randn(O_PROJ_N, WIDTH, device=DEV) * 0.02).to(torch.bfloat16)

    def chain():
        _late_writer(src_gate, gate, delay_ns=40_000)
        _late_writer(src_attention, attention, delay_ns=5_000)
        op(attention, gate, eps, trigger, out=gated)
        return dense_gemm(gated, weight, kernel, True)

    def load(seed):
        a, g = _inputs(rows, 2, seed)
        src_attention.copy_(a)
        src_gate.copy_(g)
        for t in (attention, gate, gated):
            t.fill_(float("nan"))

    def want():
        torch.cuda.synchronize()
        w_gated = _builtin().apply_output_gate(src_attention.clone(), src_gate.clone())
        torch.cuda.synchronize()
        w_out = dense_gemm(w_gated, weight, kernel, False)
        torch.cuda.synchronize()
        return w_gated, w_out

    load(1)
    out = chain()  # eager, also compiles
    torch.cuda.synchronize()
    want_gated, want_out = want()
    _assert_same_bits(gated, want_gated, f"eager gate, rows {rows}, trigger {trigger}")
    _assert_same_bits(out, want_out, f"eager o_proj {kernel}, rows {rows}, trigger {trigger}")

    graph = torch.cuda.CUDAGraph()
    load(2)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = chain()
    for seed in range(3, 11):
        load(seed)
        captured.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        want_gated, want_out = want()
        _assert_same_bits(gated, want_gated, f"replay {seed} gate, rows {rows}, {trigger}")
        _assert_same_bits(captured, want_out, f"replay {seed} o_proj {kernel}, rows {rows}")


def _ptx(trigger: str) -> str:
    attention, gate = _inputs(8, 2, 3)
    out = torch.empty_like(attention)
    compiled = norm_gate_module._attention_norm_gate_pdl_kernel[(8 * HEADS,)](
        out,
        attention,
        gate,
        attention.stride(0),
        gate.stride(0),
        HEAD_DIM,
        EPS,
        HEADS=HEADS,
        HEAD_DIM=HEAD_DIM,
        TRIGGER=TRIGGERS[trigger],
        num_warps=4,
        launch_pdl=True,
    )
    torch.cuda.synchronize()
    return compiled.asm["ptx"]


@pytest.mark.parametrize("trigger", sorted(TRIGGERS))
def test_ptx_reads_nothing_before_the_wait(trigger):
    """The audit: the wait precedes every global load; the trigger sits where it is named."""
    body = _ptx(trigger)
    lines = [line.strip() for line in body.splitlines()]
    wait = [i for i, line in enumerate(lines) if line.startswith("griddepcontrol.wait")]
    trig = [
        i for i, line in enumerate(lines) if line.startswith("griddepcontrol.launch_dependents")
    ]
    loads = [i for i, line in enumerate(lines) if re.match(r"(@%p\d+\s+)?ld\.global", line)]
    branches = [i for i, line in enumerate(lines) if re.match(r"(@%p\d+\s+)?bra", line)]
    assert len(wait) == 1 and loads, (wait, loads)
    # Straight-line code up to the first load: no branch can reach a load around the wait.
    assert not [b for b in branches if b < max(loads)], "the kernel branches; audit by hand"
    assert wait[0] < min(loads), "a global load precedes griddepcontrol.wait"
    if trigger == "exit":
        assert not trig
    elif trigger == "after_wait":
        assert len(trig) == 1 and wait[0] < trig[0] < min(loads)
    else:
        assert len(trig) == 1 and trig[0] < wait[0]
    print(
        f"\n[ptx] trigger={trigger}: wait at line {wait[0]}, trigger {trig}, first load {min(loads)}"
    )


# --------------------------------------------------------------------------
# Chain microbenchmark (MODELING_V2_TIMING=1)
# --------------------------------------------------------------------------
@triton.jit
def _spin_writer_kernel(dst, n, delay_ns, BLOCK: tl.constexpr):
    """Attention stand-in: a PDL kernel that waits, spins ``delay_ns`` and writes its output."""
    tl.extra.cuda.gdc_wait()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, (offsets % 7).to(tl.bfloat16) * 0.25, mask=mask)


@triton.jit
def _successor_kernel(src, dst, n, BLOCK: tl.constexpr):
    """Hop stand-in: a PDL kernel that waits and reads the projection's output."""
    tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask) * 2, mask=mask)


_TIMING = os.environ.get("MODELING_V2_TIMING") == "1"
LAYERS = 62


@pytest.mark.skipif(not _TIMING, reason="set MODELING_V2_TIMING=1 to run the microbenchmark")
@torch.inference_mode()
def test_bench_o_proj_site_chain():
    """[attention stand-in -> gate -> o_proj -> successor] x 62, cold weights, µs per layer."""
    eps = _builtin().o_head_norm.variance_epsilon
    rows = 8
    weights = [
        (torch.randn(O_PROJ_N, WIDTH, device=DEV) * 0.02).to(torch.bfloat16) for _ in range(LAYERS)
    ]
    gate = _inputs(rows, 2, 5)[1]
    attention = torch.empty(rows, WIDTH, device=DEV, dtype=torch.bfloat16)
    succ = torch.empty(rows, O_PROJ_N, device=DEV, dtype=torch.bfloat16)
    sink = torch.zeros(1, dtype=torch.int32, device=DEV)
    side = torch.cuda.Stream()
    attention_ns = int(os.environ.get("EARLY_RELEASE_TIMING_ATTN_NS", "6000"))
    module = _builtin()

    def layer(i, gate_mode, kernel, prefetch):
        main = torch.cuda.current_stream()
        if prefetch:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                l2_prefetch(weights[i], sink)
        _spin_writer_kernel[(triton.cdiv(rows * WIDTH, 1024),)](
            attention, rows * WIDTH, attention_ns, BLOCK=1024, num_warps=4, launch_pdl=True
        )
        if gate_mode == "builtin":
            gated = module.apply_output_gate(attention, gate)
        else:
            gated = op(attention, gate, eps, gate_mode)
        if prefetch:
            main.wait_stream(side)
        out = dense_gemm(gated, weights[i], kernel, True)
        _successor_kernel[(triton.cdiv(rows * O_PROJ_N, 1024),)](
            out, succ, rows * O_PROJ_N, BLOCK=1024, num_warps=4, launch_pdl=True
        )

    arms = [
        ("builtin", "cublas", True),  # baseline: o_proj on cuBLAS with the L2 prefetch
        ("builtin", "cublas", False),
        ("after_wait", "cublas", True),  # PDL norm-gate, o_proj prefetch kept
        ("after_wait", SPLITK, False),  # PDL norm-gate, split-K o_proj
        ("after_wait", SPLITK, True),  # PDL norm-gate, split-K o_proj, prefetch kept
        ("entry", SPLITK, False),
        ("exit", SPLITK, False),
        ("builtin", SPLITK, False),
    ]
    graphs = {}
    for arm in arms:
        for i in range(2):
            layer(i, *arm)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(LAYERS):
                layer(i, *arm)
        graphs[arm] = graph
    torch.cuda.synchronize()
    times = {arm: [] for arm in arms}
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(9):
        for arm in arms:  # interleaved rounds
            graphs[arm].replay()
            start.record()
            for _ in range(5):
                graphs[arm].replay()
            stop.record()
            torch.cuda.synchronize()
            times[arm].append(start.elapsed_time(stop) * 1000 / (5 * LAYERS))
    base = statistics.median(times[arms[0]])
    print(f"\n[bench o_proj site] attention stand-in {attention_ns / 1000:.1f} us, us per layer:")
    for arm in arms:
        med = statistics.median(times[arm])
        print(
            f"[bench] gate={arm[0]:<10} o_proj={arm[1]:<17} prefetch={'on ' if arm[2] else 'off'} "
            f"median {med:7.3f}  min {min(times[arm]):7.3f}  max {max(times[arm]):7.3f}  "
            f"vs baseline {med - base:+.3f}"
        )
