# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open unpool_gather_boundary entry (TP8 over two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The entry claims the built-in path's bits: the built-in unpool gather
(``ProjectionGatherWorkspace.gather``) followed by the built-in post-FFN join
(``MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm``, which runs the
cluster boundary kernel here). Every cell is driven through CUDA-graph replays
with changing inputs -- special BF16 payloads, an infinite input, a delayed
rank -- and the two outputs must be equal as raw bits on every rank; the cells
are also gated against the entry's float64 reference. Separate tests cover row
count changes with tag wrap-around and the re-classed join's two arms.

The ``capture`` cells (``capture_fold``) also hand the op a DFlash capture view;
the built-in side then runs the built-in capture
(``DFlashSpecMetadata.maybe_capture_hidden_states``: the ``float -> bfloat16``
copy and the 2-D slice copy) on its updated residual, and the two whole capture
buffers -- pre-filled with the same random bytes before every replay -- must
be equal as raw bits, so the op writes exactly the built-in capture's bytes
and nothing outside its rows and columns. Two more tests drive the capture's
cast on chosen FP32 residuals (NaN payloads, infinities, signed zeros,
subnormals, round-to-nearest-even ties, values that round to infinity) and
every (row count, capture slot) pair.

``MODELING_V2_TIMING=1`` also runs the microbenchmarks (printed, never gated).
"""

import os
import statistics
import time

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.projection_inbox import (  # noqa: E501
    ProjectionInbox,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.unpool_gather_boundary import (  # noqa: E501
    HIDDEN,
    LOCAL_COLUMNS,
    UnpoolGatherBoundary,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.unpool_gather_boundary import (  # noqa: E501
    unpool_gather_boundary as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

WORLD = 8
#: The checkpoint's norm epsilons: rms_post_norm_eps for the post-FFN norm, rms_norm_eps next.
POST_EPS, PRE_EPS = 1e-6, 1e-5
#: The served drafter's capture layers (Muse-Spark-Open-DFlash2 ``target_layer_ids``).
CAPTURE_LAYERS = (1, 16, 30, 44, 59)
#: Capture buffer rows (the engine sizes it to max_num_tokens; any count above 8 will do).
CAPTURE_ROWS = 16


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU GB300 nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("needs GB300 (sm_103)")
    return comm, rank


@pytest.fixture(scope="module")
def group():
    """One built-in gather workspace and one inbox for the whole module (collective)."""
    comm, rank = _world()
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    builtin = ProjectionGatherWorkspace(mapping)
    inbox = ProjectionInbox(mapping, row_words=LOCAL_COLUMNS // 2)
    comm.Barrier()
    return comm, rank, mapping, builtin, inbox


def _local(seed: int, rank: int, rows: int) -> torch.Tensor:
    """Rank ``rank``'s unpool slice under ``seed``; every rank can rebuild every slice."""
    gen = torch.Generator().manual_seed(seed * WORLD + rank)
    return (torch.randn(rows, LOCAL_COLUMNS, generator=gen) * 3.0).to(torch.bfloat16)


class _Join:
    """The built-in post-FFN join's modules: gated residual, next pre-QKV norm, gain."""

    def __init__(self, seed: int, with_gain: bool, norm_fp32_residual: bool, trigger: str = "exit"):
        from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        torch.manual_seed(seed)
        self.gate = MuseSparkOpenGatedResidual(HIDDEN, torch.bfloat16).cuda()
        self.gate.gate.normal_(0, 0.5)
        self.gate.cache_derived_state()
        assert self.gate._fused_enabled, "MUSESPARK_OPEN_FUSED_RESIDUAL must be on for this test"
        self.gate._preqkv_norm_fp32 = norm_fp32_residual
        self.norm = RMSNorm(hidden_size=HIDDEN, eps=PRE_EPS, dtype=torch.bfloat16, use_gemma=True)
        self.norm = self.norm.cuda()
        self.norm.weight.normal_(0, 0.02)
        self.gain = (
            torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)
            if with_gain
            else None
        )
        self.norm_fp32_residual = norm_fp32_residual
        self.trigger = trigger

    def builtin(self, workspace, local, residual):
        branch = workspace.gather(local)
        return self.gate._add_after_rmsnorm_and_norm(
            residual, branch, POST_EPS, self.norm, self.gain
        )

    def fused(self, inbox, local, residual, capture=None):
        alpha, beta = self.gate.coefficients()
        extra = {} if capture is None else {"capture": capture}
        return op(
            local,
            inbox,
            residual,
            self.gain,
            alpha,
            beta,
            self.norm.weight,
            POST_EPS,
            PRE_EPS,
            self.norm_fp32_residual,
            self.trigger,
            **extra,
        )

    def reference(self, locals_, residual, capture=None):
        alpha, beta = self.gate.coefficients()
        extra = {} if capture is None else {"capture": capture}
        return op.reference(
            locals_,
            None,
            residual,
            self.gain,
            alpha,
            beta,
            self.norm.weight,
            POST_EPS,
            PRE_EPS,
            self.norm_fp32_residual,
            **extra,
        )


def _capture_metadata():
    """The built-in DFlash metadata the engine builds, with its capture buffer."""
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

    return DFlashSpecMetadata(
        max_num_requests=1,
        max_draft_len=7,
        max_total_draft_tokens=7,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        layers_to_capture=list(CAPTURE_LAYERS),
        hidden_size=HIDDEN,
        max_num_tokens=CAPTURE_ROWS,
        dtype=torch.bfloat16,
    )


def _view(buffer: torch.Tensor, index: int, rows: int) -> torch.Tensor:
    """The slice the built-in capture of capture slot ``index`` writes at ``rows`` rows."""
    return buffer[:rows, index * HIDDEN : (index + 1) * HIDDEN]


def _same_buffer_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    differ = (got.view(torch.int16) != want.view(torch.int16)).sum().item()
    assert differ == 0, f"{what}: {differ} capture buffer values differ in bits from the built-in"


def _assert_same_bits(got, want, what: str):
    for g, w, name in zip(got, want, ("residual", "next input")):
        assert g.dtype == w.dtype and g.shape == w.shape, (what, name)
        width = torch.int32 if g.dtype == torch.float32 else torch.int16
        differ = (g.view(width) != w.view(width)).sum().item()
        assert differ == 0, f"{what}: {differ} {name} values differ in bits from the built-in path"


#: Special BF16 payloads placed in the first row of one replay: -0, the smallest
#: subnormal, the largest finite, the smallest normal and the most negative finite.
_SPECIAL = (-32768, 1, 32639, 128, -129)


@pytest.mark.parametrize("cell", UnpoolGatherBoundary.CELLS, ids=lambda c: c.why[:48])
@torch.inference_mode()
def test_cells_match_builtin_bits_and_reference(group, cell, certifying):  # noqa: F811
    comm, rank, _, builtin, inbox = group
    certifying(op)
    rows = cell.spec["rows"]
    join = _Join(
        seed=1234 + rows,
        trigger=cell.spec.get("trigger", "exit"),
        **{k: cell.spec[k] for k in ("with_gain", "norm_fp32_residual")},
    )
    local = torch.zeros(rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros(rows, HIDDEN, device="cuda", dtype=torch.float32)
    capture = cell.spec.get("capture", False)
    layer = CAPTURE_LAYERS[2]
    if capture:
        metadata = _capture_metadata()
        index = metadata._layer_to_idx[layer]
        want_buffer = metadata.captured_hidden_states
        got_buffer = torch.empty_like(want_buffer)
        view = _view(got_buffer, index, rows)

    def builtin_step():
        out = join.builtin(builtin, local, residual)
        if capture:
            metadata.maybe_capture_hidden_states(layer, out[0], None)
        return out

    def fused_step():
        return join.fused(inbox, local, residual, view if capture else None)

    # Warm both paths eagerly (compiles), then capture one graph per path.
    builtin_step()
    fused_step()
    torch.cuda.synchronize()
    comm.Barrier()
    builtin_graph, fused_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(builtin_graph):
        want = builtin_step()
    with torch.cuda.graph(fused_graph):
        got = fused_step()
    torch.cuda.synchronize()
    comm.Barrier()

    for repeat in range(8):
        seed = 1000 * rows + repeat
        slices = [_local(seed, r, rows) for r in range(WORLD)]
        if repeat == 1:
            for s in slices:
                s.view(torch.int16)[0, : len(_SPECIAL)] = torch.tensor(_SPECIAL, dtype=torch.int16)
        if repeat == 5:
            # An infinite input turns its whole row into NaNs on both paths.
            for s in slices:
                s[rows - 1, 7] = float("inf")
        local.copy_(slices[rank].cuda())
        residual.normal_().mul_(10.0 ** (repeat % 3 - 1))
        join.norm.weight.normal_(0, 0.02)
        if join.gain is not None:
            join.gain.normal_(1, 0.1)
        if capture:
            # The same random bytes in both buffers: whatever the op does not
            # write must still equal what the built-in capture left alone.
            torch.randint(
                -32768,
                32768,
                want_buffer.shape,
                dtype=torch.int16,
                device="cuda",
                out=want_buffer.view(torch.int16),
            )
            got_buffer.copy_(want_buffer)
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == repeat:
            time.sleep(0.002)
        builtin_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize()
        _assert_same_bits(got, want, f"rows={rows} replay {repeat}")
        if capture:
            _same_buffer_bits(got_buffer, want_buffer, f"rows={rows} replay {repeat}")
        if repeat in (1, 5):
            continue
        ref_capture = (
            torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16) if capture else None
        )
        ref = join.reference([s.cuda() for s in slices], residual, ref_capture)
        # The op and the reference differ only in the variance sums' order and the
        # approximate rsqrt (~2^-22 relative); that moves a few values across a
        # BF16 rounding threshold, each by one BF16 ulp (2^-8 relative). Bands of
        # the cluster boundary's own test: one dropped rounding stage exceeds them.
        # The capture is the residual's BF16 rounding: the next input's band.
        pairs = list(zip(got, ref, (2e-4, 1e-3)))
        if capture:
            pairs.append((view, ref_capture, 1e-3))
        for g, r, band in pairs:
            error = ((g.float() - r.float()).norm() / r.float().norm()).item()
            assert error < band, f"rows={rows} replay {repeat}: relative error {error:.2e}"
    comm.Barrier()


#: FP32 residual bit patterns whose BF16 rounding the capture must reproduce: quiet and
#: signalling NaNs with payloads, +-inf, +-0, subnormals, round-to-nearest-even ties
#: (down to even, up to even), one above a tie, the largest finite float and the largest
#: BF16 tie (both round to infinity), the largest finite BF16, and ordinary values.
_CAST_SPECIALS = (
    0x7FC00001,
    0x7F800001,
    0xFFC12345,
    0x7F800000,
    0xFF800000,
    0x00000000,
    0x80000000,
    0x00000001,
    0x007FFFFF,
    0x80000001,
    0x00408000,
    0x3F808000,
    0x3F818000,
    0x3F808001,
    0xBF808000,
    0x7F7FFFFF,
    0x7F7F8000,
    0x7F7F0000,
    0x3EAAAAAB,
    0xC2F6E979,
)


@torch.inference_mode()
def test_capture_cast_on_chosen_residuals(group):
    """alpha = 1, beta = 0: the updated residual is the input residual, so the capture
    is the BF16 rounding of chosen FP32 values. It must equal torch's cast of the op's
    own FP32 output and the built-in capture of it, bit for bit."""
    comm, rank, _, _, inbox = group
    rows = 8
    metadata = _capture_metadata()
    layer = CAPTURE_LAYERS[1]
    index = metadata._layer_to_idx[layer]
    got_buffer = torch.zeros_like(metadata.captured_hidden_states)
    metadata.captured_hidden_states.zero_()
    join = _Join(seed=5, with_gain=True, norm_fp32_residual=False)
    alpha = torch.ones(HIDDEN, device="cuda", dtype=torch.float32)
    beta = torch.zeros(HIDDEN, device="cuda", dtype=torch.float32)
    specials = torch.tensor(
        [v - (1 << 32) if v >= 1 << 31 else v for v in _CAST_SPECIALS], dtype=torch.int32
    )
    gen = torch.Generator().manual_seed(17)
    residual = torch.randn(rows, HIDDEN, generator=gen) * 100.0
    flat = residual.view(-1).view(torch.int32)
    # Every special in every row, at columns that fall into every rank's slice.
    for r in range(rows):
        for c in range(0, HIDDEN - len(specials), 409):
            flat[r * HIDDEN + c : r * HIDDEN + c + len(specials)] = specials
    residual = residual.cuda()
    local = _local(31, rank, rows).cuda()
    for norm_fp32 in (False, True):
        got_buffer.zero_()
        updated, _ = op(
            local,
            inbox,
            residual,
            join.gain,
            alpha,
            beta,
            join.norm.weight,
            POST_EPS,
            PRE_EPS,
            norm_fp32,
            capture=_view(got_buffer, index, rows),
        )
        metadata.captured_hidden_states.zero_()
        metadata.maybe_capture_hidden_states(layer, updated, None)
        torch.cuda.synchronize()
        cast = updated.to(torch.bfloat16)
        assert torch.equal(
            _view(got_buffer, index, rows).view(torch.int16), cast.view(torch.int16)
        ), f"norm_fp32={norm_fp32}: the capture is not torch's cast of the op's residual"
        _same_buffer_bits(got_buffer, metadata.captured_hidden_states, f"norm_fp32={norm_fp32}")
        # The specials reached the residual output unchanged where FP32 allows it.
        assert torch.isnan(updated).any() and torch.isinf(updated).any()
    comm.Barrier()


@torch.inference_mode()
def test_capture_every_row_count_and_slot(group):
    """Eager calls at rows 1..8 into every capture slot write the built-in capture's bytes
    and nothing else (the rest of the buffer keeps its random bytes)."""
    comm, rank, _, builtin, inbox = group
    metadata = _capture_metadata()
    want_buffer = metadata.captured_hidden_states
    got_buffer = torch.empty_like(want_buffer)
    join = _Join(seed=13, with_gain=True, norm_fp32_residual=False)
    for step, rows in enumerate((1, 2, 3, 4, 5, 6, 7, 8)):
        for layer in CAPTURE_LAYERS:
            index = metadata._layer_to_idx[layer]
            torch.randint(
                -32768,
                32768,
                want_buffer.shape,
                dtype=torch.int16,
                device="cuda",
                out=want_buffer.view(torch.int16),
            )
            got_buffer.copy_(want_buffer)
            local = _local(900 + 10 * step + index, rank, rows).cuda()
            residual = torch.randn(rows, HIDDEN, device="cuda")
            want = join.builtin(builtin, local, residual)
            metadata.maybe_capture_hidden_states(layer, want[0], None)
            got = join.fused(inbox, local, residual, _view(got_buffer, index, rows))
            torch.cuda.synchronize()
            _assert_same_bits(got, want, f"rows={rows} layer {layer}")
            _same_buffer_bits(got_buffer, want_buffer, f"rows={rows} layer {layer}")
    comm.Barrier()


@torch.inference_mode()
def test_row_count_changes_and_tag_wrap(group):
    """Eager calls of changing row counts across a tag wrap keep the built-in bits."""
    comm, rank, _, builtin, inbox = group
    inbox.counters[:, 0].fill_(-3)  # 0xFFFFFFFD: the third call on a row wraps
    builtin.counters.fill_(2**32 - 3)
    join = _Join(seed=7, with_gain=True, norm_fp32_residual=False)
    for step, rows in enumerate((1, 8, 3, 8, 1, 2, 8, 1, 8)):
        local = _local(500 + step, rank, rows).cuda()
        residual = torch.randn(rows, HIDDEN, device="cuda")
        want = join.builtin(builtin, local, residual)
        got = join.fused(inbox, local, residual)
        torch.cuda.synchronize()
        _assert_same_bits(got, want, f"step {step} rows={rows}")
    tags = inbox.counters[:, 0].cpu().tolist()
    assert 0 not in tags[:3], f"a row tag landed on zero: {tags}"
    comm.Barrier()


@pytest.mark.parametrize("fused_enabled", [True, False])
@torch.inference_mode()
def test_reclassed_join_takes_the_slice(group, fused_enabled):
    """``MuseSparkOpenTepGatedResidual`` given the slice equals the built-in join given the
    gathered branch: through the fused op when the built-in would run the cluster
    boundary, through the built-in gather and join when it would not."""
    comm, rank, _, builtin, inbox = group
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers.gathers import (  # noqa: E501
        MuseSparkOpenTepGatedResidual,
    )

    join = _Join(seed=11, with_gain=True, norm_fp32_residual=False)
    join.gate._fused_enabled = fused_enabled
    rows = 8
    local = _local(77, rank, rows).cuda()
    residual = torch.randn(rows, HIDDEN, device="cuda")
    want = join.builtin(builtin, local, residual)
    join.gate.__class__ = MuseSparkOpenTepGatedResidual
    join.gate._gather_inbox = inbox
    join.gate._gather = builtin
    calls = []
    original = type(op).__call__

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    type(op).__call__ = spy
    try:
        got = join.gate._add_after_rmsnorm_and_norm(residual, local, POST_EPS, join.norm, join.gain)
    finally:
        type(op).__call__ = original
    torch.cuda.synchronize()
    assert bool(calls) == fused_enabled
    _assert_same_bits(got, want, f"re-classed join, fused arm {fused_enabled}")
    assert not join.gate._capture_stored
    comm.Barrier()


@pytest.mark.parametrize("fused_enabled", [True, False])
@torch.inference_mode()
def test_reclassed_join_writes_the_capture_only_on_the_fused_arm(group, fused_enabled):
    """With ``_capture_fold`` set (``capture_fold``), the re-classed join's fused arm writes the capture
    and says so (``_capture_stored``); its built-in arm leaves the buffer alone and says not."""
    comm, rank, _, builtin, inbox = group
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers.gathers import (  # noqa: E501
        MuseSparkOpenTepGatedResidual,
    )

    join = _Join(seed=19, with_gain=True, norm_fp32_residual=False)
    join.gate._fused_enabled = fused_enabled
    rows = 8
    local = _local(78, rank, rows).cuda()
    residual = torch.randn(rows, HIDDEN, device="cuda")
    want = join.builtin(builtin, local, residual)
    metadata = _capture_metadata()
    layer = CAPTURE_LAYERS[4]
    index = metadata._layer_to_idx[layer]
    metadata.captured_hidden_states.zero_()
    metadata.maybe_capture_hidden_states(layer, want[0], None)
    buffer = torch.zeros_like(metadata.captured_hidden_states)
    join.gate.__class__ = MuseSparkOpenTepGatedResidual
    join.gate._gather_inbox = inbox
    join.gate._gather = builtin
    join.gate._capture_fold = _view(buffer, index, rows)
    join.gate._capture_stored = False
    got = join.gate._add_after_rmsnorm_and_norm(residual, local, POST_EPS, join.norm, join.gain)
    torch.cuda.synchronize()
    _assert_same_bits(got, want, f"re-classed join with a capture, fused arm {fused_enabled}")
    assert join.gate._capture_stored == fused_enabled
    if fused_enabled:
        _same_buffer_bits(buffer, metadata.captured_hidden_states, "re-classed join capture")
    else:
        assert not buffer.view(torch.int16).any(), "the built-in arm wrote the capture"
    comm.Barrier()


# ---------------------------------------------------------------------------
# Microbenchmark: MODELING_V2_TIMING=1 (printed, never gated).
# ---------------------------------------------------------------------------

_LAYERS = 61  # post-FFN boundaries per decode step (layer 61 has no next norm)


def _time_graphs(comm, graphs: dict, calls: int, rounds: int = 16):
    """Per-call microseconds of each graph: alternating order, rank-aligned starts."""
    samples = {name: [] for name in graphs}
    names = list(graphs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for warm in range(3):
        for name in names:
            comm.Barrier()
            graphs[name].replay()
            torch.cuda.synchronize()
    for r in range(rounds):
        for name in names if r % 2 == 0 else reversed(names):
            torch.cuda.synchronize()
            comm.Barrier()
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0 / calls)
    return {name: statistics.median(v) for name, v in samples.items()}


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@pytest.mark.parametrize("rows", [8, 1])
@torch.inference_mode()
def test_microbenchmark(group, rows):
    comm, rank, _, builtin, inbox = group
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
        default_tactic,
        run_splitk_dense,
    )

    join = _Join(seed=3, with_gain=True, norm_fp32_residual=False)
    residual = torch.randn(rows, HIDDEN, device="cuda")
    local = _local(9, rank, rows).cuda()

    # (1) the op alone: 64 back-to-back calls per graph.
    isolated = {}
    for name, body in (
        ("builtin gather+boundary", lambda: join.builtin(builtin, local, residual)),
        ("fused", lambda: join.fused(inbox, local, residual)),
    ):
        body()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(64):
                body()
        isolated[name] = graph
    alone = _time_graphs(comm, isolated, calls=64)

    # (2) the decode chain this op sits in: unpool split-K GEMM (PDL, distinct
    # weights per layer, so each reads HBM) then the gather+boundary, 61 times.
    combined = torch.randn(_LAYERS, rows, 4096, device="cuda").to(torch.bfloat16)
    weights = [
        torch.randn(LOCAL_COLUMNS, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(_LAYERS)
    ]
    tactic = default_tactic(rows, LOCAL_COLUMNS, 4096)
    outs = torch.empty(_LAYERS, rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)

    def chain(fused: bool):
        for layer in range(_LAYERS):
            projected = run_splitk_dense(
                combined[layer], weights[layer].t(), None, outs[layer], True, tactic
            )
            if fused:
                join.fused(inbox, projected, residual)
            else:
                join.builtin(builtin, projected, residual)

    chains = {}
    for name, fused in (("builtin chain", False), ("fused chain", True)):
        chain(fused)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain(fused)
        chains[name] = graph
    gemm_only = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gemm_only):
        for layer in range(_LAYERS):
            run_splitk_dense(combined[layer], weights[layer].t(), None, outs[layer], True, tactic)
    chains["gemm only"] = gemm_only
    per_layer = _time_graphs(comm, chains, calls=_LAYERS)

    results = comm.gather({**alone, **per_layer}, root=0)
    if rank == 0:
        print(
            f"\n[unpool_gather bench] rows={rows} (us per call / per layer; median over ranks, min..max)"
        )
        for name in results[0]:
            values = sorted(r[name] for r in results)
            print(
                f"[unpool_gather bench] rows={rows} {name:28s} {statistics.median(values):8.3f} "
                f"({values[0]:.3f}..{values[-1]:.3f})"
            )
    comm.Barrier()


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_capture_chain(group):
    """The capture fold in its decode chain, 8 rows: 61 times the unpool split-K GEMM, the
    boundary and the next QKV split-K GEMM (distinct weights per layer, PDL on). On the five
    capture layers the built-in capture runs after the boundary (``copies``), the boundary
    writes it (``fold``), or nothing is captured (``none``, the floor). Where the op takes
    ``trigger``, each arm runs at the ``exit`` and the ``entry`` trigger. Per-graph
    microseconds, printed (never gated)."""
    comm, rank, _, _, inbox = group
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
        default_tactic,
        run_splitk_dense,
    )

    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
        unpool_gather_boundary as module,
    )

    rows, qkv_columns = 8, 2304
    join = _Join(seed=23, with_gain=True, norm_fp32_residual=False)
    metadata = _capture_metadata()
    combined = torch.randn(_LAYERS, rows, 4096, device="cuda").to(torch.bfloat16)
    unpool_weights = [
        torch.randn(LOCAL_COLUMNS, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(_LAYERS)
    ]
    qkv_weights = [
        torch.randn(qkv_columns, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
        for _ in range(_LAYERS)
    ]
    unpool_tactic = default_tactic(rows, LOCAL_COLUMNS, 4096)
    qkv_tactic = default_tactic(rows, qkv_columns, HIDDEN)
    projected = torch.empty(_LAYERS, rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)
    qkv = torch.empty(_LAYERS, rows, qkv_columns, device="cuda", dtype=torch.bfloat16)
    residual0 = torch.randn(rows, HIDDEN, device="cuda")
    alpha, beta = join.gate.coefficients()
    triggers = ("exit", "entry") if hasattr(module, "TRIGGERS") else ("exit",)

    def chain(mode: str, trigger: str):
        residual = residual0
        for layer in range(_LAYERS):
            run_splitk_dense(
                combined[layer],
                unpool_weights[layer].t(),
                None,
                projected[layer],
                True,
                unpool_tactic,
            )
            extra = {} if trigger == "exit" else {"trigger": trigger}
            index = metadata._layer_to_idx.get(layer)
            if mode == "fold" and index is not None:
                extra["capture"] = _view(metadata.captured_hidden_states, index, rows)
            residual, next_input = op(
                projected[layer],
                inbox,
                residual,
                join.gain,
                alpha,
                beta,
                join.norm.weight,
                POST_EPS,
                PRE_EPS,
                False,
                **extra,
            )
            if mode == "copies" and index is not None:
                metadata.maybe_capture_hidden_states(layer, residual, None)
            run_splitk_dense(next_input, qkv_weights[layer].t(), None, qkv[layer], True, qkv_tactic)

    graphs = {}
    for trigger in triggers:
        for mode in ("copies", "fold", "none"):
            chain(mode, trigger)
            torch.cuda.synchronize()
            comm.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain(mode, trigger)
            graphs[f"{mode}/{trigger}"] = graph
    per_graph = _time_graphs(comm, graphs, calls=1, rounds=24)

    results = comm.gather(per_graph, root=0)
    if rank == 0:
        print(
            f"\n[capture_fold bench] rows={rows}, {_LAYERS} layers (us per graph; median over ranks)"
        )
        for name in results[0]:
            values = sorted(r[name] for r in results)
            print(
                f"[capture_fold bench] {name:12s} {statistics.median(values):9.2f} "
                f"({values[0]:.2f}..{values[-1]:.2f})"
            )
        for trigger in triggers:
            for mode in ("fold", "none"):
                deltas = sorted(r[f"{mode}/{trigger}"] - r[f"copies/{trigger}"] for r in results)
                print(
                    f"[capture_fold bench] {mode}/{trigger} - copies/{trigger}: "
                    f"{statistics.median(deltas):+.2f} ({deltas[0]:+.2f}..{deltas[-1]:+.2f})"
                )
    comm.Barrier()
