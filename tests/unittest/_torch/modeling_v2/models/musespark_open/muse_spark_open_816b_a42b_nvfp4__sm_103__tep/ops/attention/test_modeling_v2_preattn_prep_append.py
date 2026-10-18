# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open preattn_prep_append entry and the folded preprocess elision.

The entry's claim is bit identity with two built-in launches it replaces on a
full-attention (NoPE) layer: ``preattn_prep`` (itself the built-in prep, bit for
bit) followed by ``thop.trtllm_gen_generation_preprocess``
(``applyBiasRopeUpdateKVCacheV2``), called with the arguments
``FlashInferTrtllmGenFmha.run_generation`` passes on a decode step. Gated per
cell and on special values:

* ``qkv_out`` and ``gate_out`` against ``preattn_prep`` (and the cells' float64
  reference band);
* every byte of the paged FP8 pool (the step's slots and everything else),
  the E4M3 query bytes and both scale words against the built-in preprocess
  run on the same projection, pool and table;
* the trtllm-gen FMHA output computed from the entry's writes against the one
  computed from the built-in preprocess's, eager and replayed from a captured
  graph with programmatic dependent launch;
* ``full_attention.FoldedPreprocess``: for the call ``run_generation`` makes, its
  views are the real call's (same addresses, shapes, strides, dtypes, integers),
  and a call that differs in any checked field is refused;
* a delayed prep that releases the FMHA at its entry (the kernel's entry
  trigger, ``trigger_at_entry``, plus a test-only spin before the stores) still gives the
  same FMHA output -- the FMHA reads its inputs only after its
  ``griddepcontrol.wait`` -- while an FMHA on another stream with no dependency
  reads stale bytes (the control is able to fail);
* the ``trigger_at_entry`` launch (PDL, dependents released at entry): the same bytes and,
  with the FMHA launched behind it, the same FMHA output, eager and in a graph;
  its PTX triggers before the wait and the wait precedes every global load.

``MODELING_V2_TIMING=1`` also runs a microbenchmark (printed, never gated).
"""

import contextlib
import math
import os
import statistics
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    CKPT_EPS,
    CKPT_KV_HEADS,
    CKPT_Q_HEADS,
    CKPT_Q_SCALE,
    HEAD_DIM,
    preattn_prep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep_append import (  # noqa: E501
    _preattn_prep_append_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep_append import (  # noqa: E501
    preattn_prep_append as op,
)
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="preattn_prep_append is built for sm_103",
)

DEV = "cuda"
FP8 = torch.float8_e4m3fn
H, G = CKPT_Q_HEADS, CKPT_KV_HEADS
Q_WIDTH, KV_WIDTH = H * HEAD_DIM, G * HEAD_DIM
PROJ_WIDTH = 2 * (Q_WIDTH + KV_WIDTH)
OUT_WIDTH = Q_WIDTH + 2 * KV_WIDTH
#: The softmax scale ``run_generation`` passes (``get_bmm1_scale``: 1 / sqrt(64)).
BMM1_HOST = 1.0 / math.sqrt(HEAD_DIM)
QUANT_MODE = int(QuantMode.FP8_KV_CACHE)


def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _permutation() -> torch.Tensor:
    """Checkpoint column order -> low-latency layout ``[q | k | v | gate]``."""
    interleaved = torch.arange(2 * Q_WIDTH, device=DEV).reshape(H, 2 * HEAD_DIM)
    return torch.cat(
        (
            interleaved[:, :HEAD_DIM].reshape(-1),
            torch.arange(2 * Q_WIDTH, PROJ_WIDTH, device=DEV),
            interleaved[:, HEAD_DIM:].reshape(-1),
        )
    )


class _Step:
    """One decode step of a full-attention layer: projection, FP8 pool, table, scales.

    Logical page ``i`` is K block ``2 perm[i]`` and V block ``2 perm[i] + 1`` of the
    pool, ``perm`` a random permutation; the pool holds random FP8 everywhere, so a
    stray write anywhere shows up in the byte comparison.
    """

    def __init__(self, rows, ctx, *, page=32, seed=0, scale=1.0, gain=1.0, zero_heads=False):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx, self.page = rows, ctx, page
        pages = -(-ctx // page)
        spare = pages + 3
        perm = torch.randperm(spare, generator=g, device=DEV)[:pages].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = (
            torch.randn(2 * spare, G, page, HEAD_DIM, generator=g, device=DEV).mul_(4.0).to(FP8)
        )
        heads = PROJ_WIDTH // HEAD_DIM
        x = torch.randn(rows, heads, HEAD_DIM, generator=g, device=DEV)
        x = x * torch.exp(2.0 * torch.randn(rows, heads, 1, generator=g, device=DEV)) * gain
        if zero_heads:
            x[:, ::5] = 0.0
        self.projection = x.reshape(rows, PROJ_WIDTH).to(torch.bfloat16)
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.orig_quant = torch.tensor([scale], dtype=torch.float32, device=DEV)
        self.quant_orig = 1.0 / self.orig_quant

    def args(self, pool, q_fp8, bmm1, bmm2, permuted=False, pdl=False, operand=None, trigger=False):
        return (
            self.projection if operand is None else operand,
            H,
            G,
            CKPT_EPS,
            CKPT_Q_SCALE,
            permuted,
            pool,
            self.table,
            self.seq_lens,
            self.orig_quant,
            self.quant_orig,
            q_fp8,
            bmm1,
            bmm2,
            BMM1_HOST,
            1.0,
            pdl,
        ) + ((True,) if trigger else ())


def _workspace() -> torch.Tensor:
    return torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)


def _preprocess_args(step: _Step, qkv: torch.Tensor, pool: torch.Tensor, workspace) -> tuple:
    """``run_generation``'s positional arguments to the preprocess, for one sequence of ``step``."""
    return (
        qkv,
        workspace,
        step.seq_lens,
        None,
        None,
        step.table.view(1, 1, 2, -1),
        torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64),
        torch.tensor([[0, 0]], dtype=torch.int32),
        step.orig_quant,
        step.quant_orig,
        None,
        None,
        None,
        None,
        0,
        0,
        H,
        G,
        HEAD_DIM,
        step.page,
        QUANT_MODE,
        step.ctx,
        step.ctx,
        step.rows,
        1,
        step.rows,
        step.ctx,
        0,
        10000.0,
        0,
        1.0,
        1024,
        0,
        BMM1_HOST,
        1.0,
        True,
        1,
        0,
        _sm_count(),
        pool.shape[0],
        2,
        True,
        False,
    )


def _builtin(step: _Step, pool: torch.Tensor, permuted: bool = False):
    """``preattn_prep`` then the built-in generation preprocess on ``pool``; its outputs."""
    operand = step.projection[:, _permutation()].contiguous() if permuted else step.projection
    qkv, gate = preattn_prep(operand, None, None, H, G, CKPT_EPS, CKPT_Q_SCALE, permuted)
    workspace = _workspace()
    args = _preprocess_args(step, qkv, pool, workspace)
    outputs = thop.trtllm_gen_generation_preprocess(*args)
    return qkv, gate, args, outputs


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    a, b = _bytes(got), _bytes(want)
    assert a.numel() == b.numel(), (what, a.numel(), b.numel())
    diff = a != b
    if diff.any():
        first = int(diff.nonzero()[0])
        raise AssertionError(
            f"{what}: {int(diff.sum())} of {a.numel()} byte(s) differ, first at {first}: "
            f"{int(a[first])} vs {int(b[first])}"
        )


def _run_op(step: _Step, permuted: bool = False, pdl: bool = False, trigger: bool = False):
    pool = step.pool.clone()
    q8 = torch.full((step.rows * H * HEAD_DIM,), 0x7F, dtype=torch.uint8, device=DEV)
    bmm1 = torch.full((2,), float("nan"), device=DEV)
    bmm2 = torch.full((1,), float("nan"), device=DEV)
    operand = step.projection[:, _permutation()].contiguous() if permuted else step.projection
    qkv, gate = op(
        *step.args(
            pool, q8, bmm1, bmm2, permuted=permuted, pdl=pdl, operand=operand, trigger=trigger
        )
    )
    return qkv, gate, pool, q8, bmm1, bmm2


def _assert_matches_builtin(
    step: _Step, permuted: bool = False, pdl: bool = False, trigger: bool = False
):
    qkv, gate, pool, q8, bmm1, bmm2 = _run_op(step, permuted, pdl, trigger)
    want_pool = step.pool.clone()
    want_qkv, want_gate, _, outputs = _builtin(step, want_pool, permuted)
    _assert_bytes_equal(qkv, want_qkv, "qkv_out")
    _assert_bytes_equal(gate, want_gate, "gate_out")
    _assert_bytes_equal(pool, want_pool, "kv pool")
    q_processed, _, _, _, want_bmm1, want_bmm2 = outputs[:6]
    _assert_bytes_equal(q8, _bytes(q_processed)[: step.rows * H * HEAD_DIM], "E4M3 query")
    _assert_bytes_equal(bmm1, want_bmm1[:2], "bmm1 scale words")
    _assert_bytes_equal(bmm2, want_bmm2[:1], "bmm2 scale word")
    return qkv, gate, pool, q8, bmm1, bmm2


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    """Each cell: the built-in bytes exactly, and the float64 reference within preattn_prep's
    band."""
    certifying(op)
    spec = cell.spec
    step = _Step(
        spec["rows"],
        spec["ctx"],
        page=spec["page"],
        seed=zlib.crc32(cell.why.encode()),
        scale=spec.get("scale", 1.0),
        gain=spec.get("gain", 1.0),
        zero_heads=spec.get("zero_heads", False),
    )
    permuted = spec.get("permuted", False)
    qkv, gate, pool, q8, bmm1, bmm2 = _assert_matches_builtin(step, permuted)

    # The reference: preattn_prep's band on the normalized heads, the copies and the conversions exactly
    # where the reference's BF16 value is the op's.
    operand = step.projection[:, _permutation()].contiguous() if permuted else step.projection
    ref_pool = step.pool.clone()
    ref_q8 = torch.zeros_like(q8)
    ref_bmm1, ref_bmm2 = torch.zeros_like(bmm1), torch.zeros_like(bmm2)
    ref_qkv, ref_gate = op.reference(
        *step.args(ref_pool, ref_q8, ref_bmm1, ref_bmm2, permuted=permuted, operand=operand)
    )
    for lo, hi in ((0, Q_WIDTH), (Q_WIDTH, Q_WIDTH + KV_WIDTH)):
        assert_within_ulp(
            qkv[:, lo:hi].reshape(-1, HEAD_DIM),
            ref_qkv[:, lo:hi].reshape(-1, HEAD_DIM),
            element_ulp=4.0,
            rms_ulp=0.05,
        )
    _assert_bytes_equal(qkv[:, Q_WIDTH + KV_WIDTH :], ref_qkv[:, Q_WIDTH + KV_WIDTH :], "v")
    _assert_bytes_equal(gate, ref_gate, "gate")
    _assert_bytes_equal(bmm1, ref_bmm1, "bmm1 vs reference")
    _assert_bytes_equal(bmm2, ref_bmm2, "bmm2 vs reference")
    same_q = (qkv[:, :Q_WIDTH] == ref_qkv[:, :Q_WIDTH]).reshape(-1)
    assert torch.equal(q8[same_q], ref_q8[same_q]), "E4M3 query vs reference"
    differ = int((_bytes(pool) != _bytes(ref_pool)).sum())
    keys = slice(Q_WIDTH, Q_WIDTH + KV_WIDTH)
    k_moved = int((qkv[:, keys] != ref_qkv[:, keys]).sum())
    assert differ <= k_moved, f"{differ} cache bytes differ from the reference, {k_moved} K do"


def _special(kind: str, rows: int, g: torch.Generator) -> torch.Tensor:
    heads = PROJ_WIDTH // HEAD_DIM
    shape = (rows, heads, HEAD_DIM)
    sign = torch.where(torch.rand(shape, generator=g, device=DEV) < 0.5, -1.0, 1.0)
    unit = torch.randn(shape, generator=g, device=DEV)
    if kind == "zero":
        return torch.zeros(shape, device=DEV) * sign
    if kind == "subnormal":
        return sign * torch.randint(1, 128, shape, generator=g, device=DEV) * 2.0**-133
    if kind == "near-max":
        return sign * torch.empty(shape, device=DEV).uniform_(1e38, 3.3e38, generator=g)
    if kind == "e4m3-edges":
        # Values around the E4M3 rounding and saturation edges after the norm: 448, 464 (the
        # first value that rounds past 448), E4M3 subnormals (2^-9 .. 2^-7) and their halves.
        edges = torch.tensor(
            [448.0, 464.0, 480.0, 2.0**-9, 2.0**-10, 3 * 2.0**-11, 2.0**-7, 0.0], device=DEV
        )
        idx = torch.randint(0, edges.numel(), shape, generator=g, device=DEV)
        return sign * edges[idx]
    if kind == "inf":
        x = unit.clone()
        x[..., 7] = float("inf")
        x[..., 9] = float("-inf")
        return x
    if kind == "nan":
        x = unit.clone()
        x[..., 3] = float("nan")
        return x
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ("zero", "subnormal", "near-max", "e4m3-edges", "inf", "nan"))
@pytest.mark.parametrize("scale", (1.0, 0.03125), ids=("s1", "s1_32"))
def test_bit_identity_on_special_values(kind, scale) -> None:
    """Signed zeros, BF16 subnormals, values that saturate E4M3, the E4M3 subnormal and
    rounding edges, infinities and NaN: the same bytes as the built-in preprocess. The
    value heads (copied, not normalized) carry the special values straight into the cache."""
    step = _Step(8, 100003, seed=zlib.crc32(kind.encode()), scale=scale)
    g = torch.Generator(device=DEV).manual_seed(zlib.crc32(kind.encode()) + 1)
    step.projection = _special(kind, 8, g).reshape(8, PROJ_WIDTH).to(torch.bfloat16)
    _assert_matches_builtin(step)


@pytest.mark.parametrize("rows", (1, 2, 3, 5, 7, 8))
@pytest.mark.parametrize("ctx", (8, 31, 32, 33, 4095, 455000, (1 << 20) - 3))
def test_bit_identity_over_rows_and_contexts(rows, ctx) -> None:
    """M = 1..8 at contexts that start, end and straddle pages, up to 2^20 keys."""
    if ctx < rows:
        pytest.skip("the step's rows exceed the context")
    step = _Step(rows, ctx, seed=rows * 7919 + ctx)
    _assert_matches_builtin(step)


def _fmha(q8_bytes, pool, step: _Step, bmm1, bmm2, workspace, counter, pdl=False):
    """flashinfer's trtllm-gen decode, called as ``run_generation`` calls it."""
    import flashinfer.decode as fdec

    rows = step.rows
    out = torch.empty(rows, Q_WIDTH, dtype=torch.bfloat16, device=DEV)
    query = q8_bytes[: rows * H * HEAD_DIM].view(FP8).view(rows, H, HEAD_DIM)
    fdec.trtllm_batch_decode_with_kv_cache(
        query=query,
        kv_cache=(pool, pool),
        workspace_buffer=workspace,
        block_tables=step.table.view(1, 2, -1),
        seq_lens=step.seq_lens,
        max_seq_len=step.ctx,
        bmm1_scale=bmm1,
        bmm2_scale=bmm2,
        window_left=-1,
        out=out.view(rows, H, HEAD_DIM),
        kv_layout="HND",
        enable_pdl=pdl,
        backend="trtllm-gen",
        q_len_per_req=rows,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=bmm1.narrow(0, 1, 1),
        multi_ctas_kv_counter_buffer=counter,
    )
    return out


def _counter() -> torch.Tensor:
    return torch.zeros(4096, dtype=torch.int32, device=DEV)


@pytest.mark.parametrize("rows,ctx", ((8, 8192), (8, 100000), (8, 455000), (1, 100000), (5, 33)))
def test_the_fmha_output_is_the_builtin_one(rows, ctx) -> None:
    """The FMHA reading the entry's writes gives the bits it gives reading the built-in's."""
    step = _Step(rows, ctx, seed=ctx + rows)
    want_pool = step.pool.clone()
    _, _, _, outputs = _builtin(step, want_pool)
    q_processed, _, _, _, want_bmm1, want_bmm2, fmha_ws = outputs[:7]
    want = _fmha(_bytes(q_processed), want_pool, step, want_bmm1, want_bmm2, fmha_ws, _counter())
    _, _, pool, q8, bmm1, bmm2 = _run_op(step)
    got = _fmha(q8, pool, step, bmm1, bmm2, _workspace(), _counter())
    _assert_bytes_equal(got, want, "FMHA output")


@pytest.mark.parametrize(
    "pdl,trigger",
    ((False, False), (True, False), (True, True)),
    ids=("plain", "pdl", "pdl_trigger_at_entry"),
)
def test_a_captured_graph_replays_the_builtin_bits_at_other_contexts(pdl, trigger) -> None:
    """Entry + FMHA captured once (PDL as the target launches it, with and without the
    entry trigger), replayed with seq_lens moved: each replay gives the built-in path's FMHA
    output and cache bytes."""
    rows, base = 8, 100000
    step = _Step(rows, base + 64, seed=5)
    pool = step.pool.clone()
    q8 = torch.zeros(rows * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    bmm1 = torch.zeros(2, device=DEV)
    bmm2 = torch.zeros(1, device=DEV)
    workspace, counter = _workspace(), _counter()
    projection = step.projection.clone()
    seq_lens = torch.tensor([base], dtype=torch.int32, device=DEV)
    graph_step = _Step(rows, base + 64, seed=5)
    graph_step.seq_lens = seq_lens
    graph_step.ctx = base + 64

    def body():
        op(*graph_step.args(pool, q8, bmm1, bmm2, pdl=pdl, operand=projection, trigger=trigger))
        return _fmha(q8, pool, graph_step, bmm1, bmm2, workspace, counter, pdl=pdl)

    body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = body()
    for length in (base - 7, base + 1, base + 64):
        seq_lens.fill_(length)
        g = torch.Generator(device=DEV).manual_seed(length)
        projection.copy_(torch.randn(rows, PROJ_WIDTH, generator=g, device=DEV).to(torch.bfloat16))
        before = pool.clone()
        graph.replay()
        torch.cuda.synchronize()
        ref = _Step(rows, length, seed=5)
        ref.projection, ref.pool, ref.table = projection.clone(), before, step.table
        ref.ctx = length
        want_pool = before.clone()
        _, _, _, outputs = _builtin(ref, want_pool)
        q_processed, _, _, _, want_bmm1, want_bmm2, fmha_ws = outputs[:7]
        ref.ctx = base + 64
        want = _fmha(_bytes(q_processed), want_pool, ref, want_bmm1, want_bmm2, fmha_ws, _counter())
        _assert_bytes_equal(pool, want_pool, f"cache at {length}")
        _assert_bytes_equal(out, want, f"FMHA output at {length}")


def _delayed_launch(step, pool, q8, bmm1, bmm2, delay_ns, early_trigger, stream=None):
    """The entry's kernel releasing its dependents at entry (``trigger_at_entry``) when
    ``early_trigger``, then spinning ``delay_ns`` before any store (test-only switch)."""
    qkv_out = torch.empty(step.rows, OUT_WIDTH, dtype=torch.bfloat16, device=DEV)
    gate_out = torch.empty(step.rows, Q_WIDTH, dtype=torch.bfloat16, device=DEV)
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        _preattn_prep_append_kernel[(step.rows, H + G)](
            step.projection,
            qkv_out,
            gate_out,
            q8.view(FP8),
            pool,
            step.table,
            step.table.stride(0),
            step.seq_lens,
            step.orig_quant,
            step.quant_orig,
            bmm1,
            bmm2,
            BMM1_HOST,
            1.0,
            step.projection.stride(0),
            qkv_out.stride(0),
            gate_out.stride(0),
            CKPT_EPS,
            CKPT_Q_SCALE,
            NUM_Q_HEADS=H,
            NUM_KV_HEADS=G,
            QKV_PERMUTED=False,
            P=step.page,
            LAUNCH_WITH_PDL=True,
            TRIGGER_AT_ENTRY=early_trigger,
            TEST_DELAY_NS=delay_ns,
            num_warps=1,
            launch_pdl=True,
        )


def test_a_delayed_prep_is_still_observed_by_the_fmha() -> None:
    """Negative control for the prep -> FMHA ordering. The prep released its dependents at
    entry and then spun 200 us before its first store: the PDL-launched FMHA was resident
    long before the cache, the query and the scale words were written, and still gives
    the built-in bits, so it reads them only after its ``griddepcontrol.wait`` (full
    completion of the prep). The default launch never releases early, which only adds
    margin; the ``trigger_at_entry`` launch releases at entry exactly as here, without the
    spin. The control
    can fail: the same delayed prep on a second stream with no dependency leaves the FMHA
    reading the stale buffers."""
    step = _Step(8, 100000, seed=11)
    want_pool = step.pool.clone()
    _, _, _, outputs = _builtin(step, want_pool)
    q_processed, _, _, _, want_bmm1, want_bmm2, fmha_ws = outputs[:7]
    want = _fmha(_bytes(q_processed), want_pool, step, want_bmm1, want_bmm2, fmha_ws, _counter())

    def stale():
        q8 = torch.full((8 * H * HEAD_DIM,), 0x38, dtype=torch.uint8, device=DEV)  # 1.0 everywhere
        return step.pool.clone(), q8, torch.full((2,), 0.5, device=DEV), torch.ones(1, device=DEV)

    pool, q8, bmm1, bmm2 = stale()
    workspace, counter = _workspace(), _counter()
    torch.cuda.synchronize()
    # Nothing is launched between the prep and the FMHA: the FMHA is the prep's PDL dependent.
    _delayed_launch(step, pool, q8, bmm1, bmm2, delay_ns=200_000, early_trigger=True)
    got = _fmha(q8, pool, step, bmm1, bmm2, workspace, counter, pdl=True)
    torch.cuda.synchronize()
    _assert_bytes_equal(got, want, "FMHA output after a delayed, early-releasing prep")

    # The race spins 20 ms, far longer than the host takes to launch the FMHA behind it
    # (flashinfer's wrapper alone can take a few hundred microseconds), and is tried up to
    # three times: one stale read proves the point.
    raced_equal = []
    side = torch.cuda.Stream()
    for _ in range(3):
        pool, q8, bmm1, bmm2 = stale()
        workspace, counter = _workspace(), _counter()
        torch.cuda.synchronize()
        _delayed_launch(
            step, pool, q8, bmm1, bmm2, delay_ns=20_000_000, early_trigger=False, stream=side
        )
        raced = _fmha(q8, pool, step, bmm1, bmm2, workspace, counter, pdl=True)
        torch.cuda.synchronize()
        raced_equal.append(torch.equal(_bytes(raced), _bytes(want)))
        if not raced_equal[-1]:
            break
    assert not all(raced_equal), (
        "an FMHA with no dependency on the delayed prep read its outputs anyway (3 tries): "
        "the control cannot tell a missing dependency from a present one"
    )


@pytest.mark.parametrize(
    "pdl,trigger",
    ((False, False), (True, False), (True, True)),
    ids=("plain", "pdl", "pdl_trigger_at_entry"),
)
def test_the_compiled_kernel_waits_before_reading_and_never_releases_early(pdl, trigger) -> None:
    """PTX of the target's launches: with PDL ``griddepcontrol.wait`` precedes the first
    global load and, for the default launch, no ``launch_dependents`` is emitted; the
    ``trigger_at_entry`` launch releases its dependents at entry, before the wait; without PDL neither appears."""
    step = _Step(8, 4096, seed=3)
    pool = step.pool.clone()
    q8 = torch.zeros(8 * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    qkv_out = torch.empty(8, OUT_WIDTH, dtype=torch.bfloat16, device=DEV)
    gate_out = torch.empty(8, Q_WIDTH, dtype=torch.bfloat16, device=DEV)
    compiled = _preattn_prep_append_kernel[(8, H + G)](
        step.projection,
        qkv_out,
        gate_out,
        q8.view(FP8),
        pool,
        step.table,
        step.table.stride(0),
        step.seq_lens,
        step.orig_quant,
        step.quant_orig,
        torch.zeros(2, device=DEV),
        torch.zeros(1, device=DEV),
        BMM1_HOST,
        1.0,
        step.projection.stride(0),
        qkv_out.stride(0),
        gate_out.stride(0),
        CKPT_EPS,
        CKPT_Q_SCALE,
        NUM_Q_HEADS=H,
        NUM_KV_HEADS=G,
        QKV_PERMUTED=False,
        P=step.page,
        LAUNCH_WITH_PDL=pdl,
        num_warps=1,
        launch_pdl=pdl,
        **({"TRIGGER_AT_ENTRY": True} if trigger else {}),
    )
    ptx = compiled.asm["ptx"]
    assert "globaltimer" not in ptx
    first_load = ptx.index("ld.global")
    if pdl:
        assert ptx.index("griddepcontrol.wait") < first_load
    else:
        assert "griddepcontrol.wait" not in ptx
    if trigger:
        assert ptx.count("griddepcontrol.launch_dependents") == 1
        assert ptx.index("griddepcontrol.launch_dependents") < ptx.index("griddepcontrol.wait")
    else:
        assert "launch_dependents" not in ptx


@pytest.mark.parametrize("rows,ctx", ((8, 100000), (8, 455000), (1, 33), (5, 4100)))
def test_the_entry_trigger_changes_no_byte(rows, ctx) -> None:
    """The ``trigger_at_entry`` launch: the entry's writes are the built-in bytes, and the trtllm-gen FMHA
    launched straight behind it with PDL -- resident while the prep still runs -- gives the
    built-in path's FMHA output."""
    step = _Step(rows, ctx, seed=ctx * 3 + rows)
    _assert_matches_builtin(step, pdl=True, trigger=True)
    want_pool = step.pool.clone()
    _, _, _, outputs = _builtin(step, want_pool)
    q_processed, _, _, _, want_bmm1, want_bmm2, fmha_ws = outputs[:7]
    want = _fmha(_bytes(q_processed), want_pool, step, want_bmm1, want_bmm2, fmha_ws, _counter())
    pool = step.pool.clone()
    q8 = torch.full((rows * H * HEAD_DIM,), 0x7F, dtype=torch.uint8, device=DEV)
    bmm1 = torch.full((2,), float("nan"), device=DEV)
    bmm2 = torch.full((1,), float("nan"), device=DEV)
    workspace, counter = _workspace(), _counter()
    torch.cuda.synchronize()
    # Nothing is launched between the prep and the FMHA: the FMHA is the prep's PDL dependent.
    op(*step.args(pool, q8, bmm1, bmm2, pdl=True, trigger=True))
    got = _fmha(q8, pool, step, bmm1, bmm2, workspace, counter, pdl=True)
    torch.cuda.synchronize()
    _assert_bytes_equal(pool, want_pool, "kv pool behind the entry trigger")
    _assert_bytes_equal(got, want, "FMHA output behind the entry trigger")


def test_the_folded_preprocess_returns_the_real_calls_views() -> None:
    """For the call ``run_generation`` makes after the folded prep, ``FoldedPreprocess``
    returns what the real call returns -- the same tensors by address, shape, stride and
    dtype, the same integers -- and refuses a call that differs in a checked field."""
    rows, ctx = 8, 100000
    step = _Step(rows, ctx, seed=21)
    pool = step.pool.clone()
    qkv, _, args, real = _builtin(step, pool)
    workspace = args[1]
    layout = full_attention.generation_layout(torch.bfloat16, rows, H, G, 0)
    folded = full_attention.FoldedPreprocess(
        qkv_out=qkv,
        rows=rows,
        num_heads=H,
        num_kv_heads=G,
        workspace_ptr=workspace.data_ptr(),
        layout=layout,
        seq_lens_ptr=step.seq_lens.data_ptr(),
        block_offsets_ptr=args[5].data_ptr(),
        pool_pointers=args[6],
        pool_mapping=args[7],
        orig_quant_ptr=step.orig_quant.data_ptr(),
        quant_orig_ptr=step.quant_orig.data_ptr(),
        layer_idx=0,
        tokens_per_block=step.page,
        total_num_blocks=pool.shape[0],
        quant_mode=QUANT_MODE,
        bmm1_host=BMM1_HOST,
        rotary_dim=0,
    )
    assert folded.mismatch(args, {}) is None
    views = folded.outputs(args)
    assert len(views) == len(real)
    for index, (got, want) in enumerate(zip(views, real)):
        if isinstance(want, torch.Tensor):
            assert isinstance(got, torch.Tensor), index
            assert (got.data_ptr(), tuple(got.shape), got.stride(), got.dtype) == (
                want.data_ptr(),
                tuple(want.shape),
                want.stride(),
                want.dtype,
            ), index
        else:
            assert got == want, (index, got, want)

    def changed(name, value):
        perturbed = list(args)
        perturbed[full_attention.PREPROCESS_ARGS.index(name)] = value
        return folded.mismatch(tuple(perturbed), {})

    other = qkv.clone()
    for name, value in (
        ("qkv_input", other),
        ("workspace", _workspace()),
        ("sequence_lengths", step.seq_lens.clone()),
        ("kv_scale_orig_quant", step.orig_quant.clone()),
        ("attention_output_orig_quant", torch.ones(1, device=DEV)),
        ("layer_idx", 1),
        ("num_tokens", rows - 1),
        ("batch_beam", 2),
        ("position_embedding_type", 1),
        ("fp8_context_fmha", False),
        ("kv_cache_quant_mode", int(QuantMode.INT8_KV_CACHE)),
        ("bmm1_scale", 0.5),
        ("attention_chunk_size", 64),
        ("cross_attention", True),
    ):
        assert changed(name, value) is not None, name
    assert folded.mismatch(args[:-1], {}) is not None
    assert folded.mismatch(args, {"skip_fmha_workspace": True}) is not None


def test_the_target_can_skip_the_builtin_preprocess_here() -> None:
    """The install-time check: ``run_generation`` calls the preprocess as ``eliding`` expects."""
    assert full_attention.preprocess_reachable() is None


def test_eliding_swaps_only_the_preprocess_and_restores_on_error() -> None:
    from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi_gen

    step = _Step(8, 4096, seed=2)
    pool = step.pool.clone()
    qkv, _, args, _ = _builtin(step, pool)
    folded = full_attention.FoldedPreprocess(
        qkv_out=qkv, rows=8, num_heads=H, num_kv_heads=G, workspace_ptr=args[1].data_ptr(),
        layout=full_attention.generation_layout(torch.bfloat16, 8, H, G, 0),
        seq_lens_ptr=step.seq_lens.data_ptr(), block_offsets_ptr=args[5].data_ptr(),
        pool_pointers=args[6], pool_mapping=args[7], orig_quant_ptr=step.orig_quant.data_ptr(),
        quant_orig_ptr=step.quant_orig.data_ptr(), layer_idx=0, tokens_per_block=32,
        total_num_blocks=pool.shape[0], quant_mode=QUANT_MODE, bmm1_host=BMM1_HOST, rotary_dim=0,
    )  # fmt: skip
    with pytest.raises(RuntimeError, match="inside"):
        with full_attention.eliding(folded):
            assert fi_gen.thop is not thop
            metadata = thop.build_trtllm_gen_kv_cache_metadata
            assert fi_gen.thop.build_trtllm_gen_kv_cache_metadata is metadata
            fi_gen.thop.trtllm_gen_generation_preprocess(*args)
            assert folded.consumed
            raise RuntimeError("inside")
    assert fi_gen.thop is thop


_TIMING = os.environ.get("MODELING_V2_TIMING", "0") == "1"


@pytest.mark.skipif(not _TIMING, reason="MODELING_V2_TIMING=1 runs the microbenchmark")
def test_microbenchmark_prep_append_vs_prep_plus_builtin_preprocess() -> None:
    """Back-to-back graphs of 64 calls, PDL on: preattn_prep + the built-in preprocess vs the
    entry alone, each followed by nothing (launch + latency only)."""
    step = _Step(8, 100000, seed=9)
    pool_a, pool_b = step.pool.clone(), step.pool.clone()
    q8 = torch.zeros(8 * H * HEAD_DIM, dtype=torch.uint8, device=DEV)
    bmm1, bmm2 = torch.zeros(2, device=DEV), torch.zeros(1, device=DEV)
    workspace = _workspace()

    def builtin():
        qkv, _ = preattn_prep(
            step.projection, None, None, H, G, CKPT_EPS, CKPT_Q_SCALE, False, True
        )
        thop.trtllm_gen_generation_preprocess(*_preprocess_args(step, qkv, pool_b, workspace))

    def fused():
        op(*step.args(pool_a, q8, bmm1, bmm2, pdl=True))

    def graph_of(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(64):
                fn()
        return g

    graphs = {"prep + builtin preprocess": graph_of(builtin), "prep_append": graph_of(fused)}
    for name, g in graphs.items():
        times = []
        for _ in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) * 1000.0 / 64)
        print(f"\n  {name}: {statistics.median(times):.2f} us per layer")
