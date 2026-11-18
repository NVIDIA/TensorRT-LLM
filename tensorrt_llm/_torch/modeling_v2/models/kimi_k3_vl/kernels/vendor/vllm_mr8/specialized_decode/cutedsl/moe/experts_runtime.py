# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent routed experts for B<=64 (B>8: grouped units of <=8 tokens per expert)
over one shared core and model-owned workspace. The entry wait completes the
predecessor (incl. workspace rearm/counters) before shared state is read; output rows
keep the caller's top16 order."""

import functools

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...._compat import init_logger

from ..common import tensormap
from ..common.host import check_residency, fake, stream, tensor_maps, to_cute
from ..common.policy import batch_table
from .experts import (
    PLAN_BYTES,
    ExpertsGeometry,
    ExpertsGroupKernel,
    ExpertsKernel,
    ExpertsPlanKernel,
    fc2_pairs,
)
from .experts_rubin import (
    ROUTE_IDS,
    ROUTE_WORDS,
    ImagePlanKernel,
    RubinExpertsKernel,
    generation_bytes,
    peers_bytes,
    peers_offset,
)
from .experts_rubin64 import CENSUS_BYTES, RubinGroupedExpertsKernel
from .experts_vw import RubinVWExpertsKernel
from .experts_wide import WIDE_MAX, RubinWideExpertsKernel, WidePlanKernel

logger = init_logger(__name__)


def tensor(value, align=16):
    return to_cute(value.reshape(-1), align)


def streamed(batch):
    """B<=8 on the incremental producer: ~1-1.5% at B4-8, neutral at B1-2, which keep
    the previous producer."""
    return bool(batch_table("4-8:1").get(batch, 0))


def pregrouped(tp, batch):
    """B>8 experts read the unit table a one-CTA kernel builds after routing, beside
    the fused shared expert (hidden at B9-32; at B64 its ~2.3 us after routing outlast
    the shared expert)."""
    table = batch_table(PREGROUP[tp])
    return batch > 8 and bool(table.get(batch, 0))


PREGROUP = {8: "9-32:1", 16: "9-32:1"}
# With the shared join skipped (moe.prepared.JOIN), TP8 B16 releases at entry.
EARLY = {8: "9-16:1", 16: ""}


def early_group(tp, batch):
    """Pregrouped batches whose unit-table kernel releases the experts at its entry."""
    table = batch_table(EARLY[tp])
    return pregrouped(tp, batch) and bool(table.get(batch, 0))


def stealing(batch):
    """B>8 experts balance the FC2(B) tail by claims."""
    return batch > 8 and bool(batch_table("9-64:1").get(batch, 0))


def workspace_slots(device):
    """Per-CTA workspace entries of the device's experts: 160 covers every grid up to
    160 SMs; larger devices round their SM count up to a multiple of 32."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return max(160, -(-sms // 32) * 32)


@functools.cache
def compile_experts(tp, experts, batch, grid, device, publish=False, *knobs, slots=160):
    """Only code and geometry are cached; no owner tensors or descriptors. ``knobs``:
    (B<=8 stream, pregrouped, steal); ``slots``: ``workspace_slots``."""
    defaults = (streamed(batch), False, False)
    stream, grouped, steal = knobs + defaults[len(knobs) :]
    capacity = 1 << (batch - 1).bit_length()
    moe = ExpertsKernel(
        tp, capacity, experts, grid, publish, stream, grouped, steal, slots
    )
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    box = (c.Int64(0), c.Int32(0), c.Int32(0)) if publish else ()
    if publish == "push":  # + shared rows, the RS mailbox bases, capacity
        box += (flat(batch * 7168, c.BFloat16), fake((tp,), None, c.Int64), c.Int32(0))
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank
        box = (fake((3 * tp, 64, 3584 // tp)), fake((tp,), None, c.Int64), c.Int32(0))
    x, ids = fake((batch, 3584), (3584, 1)), flat(batch * 16, c.Int32)
    x = flat(PLAN_BYTES) if moe.grouped else x  # grouped: the MXFP8 plan
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), c.BFloat16)
    maps = flat(768 if moe.paired else 256, align=64)
    tensors = (x, ids, maps, *scales, flat(moe.workspace_bytes), out)
    scalars = (c.Int32(batch), *map(c.Float32, (100, 0.25, 0.04)), *box)
    with torch.accelerator.device_index(device):
        return cute.compile(moe, *tensors, *scalars, make_fake_stream())


# Rubin (cc 10.7) TP16 batches on ``RubinExpertsKernel``: the routing kernel writes
# the latent as the experts' MXFP8 SMEM image (``experts_rubin.image_chunk``), which
# they bulk-copy, and the last quarter of FC2(B) is claimed tile by tile. B8 only
# (the image needs every token row), on the grid that spans every SM (``GRID``).
RUBIN = {16: "8:1"}


def rubin(tp, batch, device):
    """Batches on the Rubin experts kernel; the routing kernel then writes the plan."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    if not every_sm(tp, batch, device):  # other SM counts: the 152-SM grids
        return False
    return bool(batch_table(RUBIN.get(tp, "")).get(batch, 0))


# Rubin TP16 batches whose routing kernel publishes the ids and the image plan's
# completion into the experts' route mailbox, so the experts dedupe, partition and
# stream their first weights while routing (and the shared expert) still run
# (``RubinExpertsKernel.polled``; the routing kernel's ``signal``).
POLLED = {16: "8:1"}
# Verify steps (q > 1), by TP: batches whose experts CTAs reserve only the named
# barriers they use (``ExpertsKernel.static_groups``; ``RubinExpertsKernel`` at 8 rows,
# ``RubinGroupedExpertsKernel`` above), so the CTAs already on an SM at their launch
# (routing's, the shared down GEMM's) no longer keep them off it.
STATIC_GROUPS = {16: "8-64:1"}
# Verify steps (q > 1), by TP: batches whose K-split exchange runs through the
# parity-buffered peer slots (``RubinExpertsKernel.fast_exchange``; the same sums).
FAST_EXCHANGE = {16: "8:1"}
# Verify steps (q > 1), by TP: rows on ``RubinGroupedExpertsKernel`` whose FC2 tail
# pool zone (the jobs at the end of every FC2(B) range that go into the pool) is not
# the default 12. A claim and a new unit's hidden image cost more than the pool saves
# when the FC2(B) ranges are short. One GPU, two-shot publish, real verify routes, vs
# zone 12: M16 zone 0 -1.42, M24 zone 6 -0.53, M32 zone 6 -0.20 us per call (M40-64:
# zone 6 within 0.15, zone 0 +0.7 to +1.0).
POOL_ZONE = {16: "9-16:0,17-32:6"}


def polled(tp, batch, device):
    """Batches whose experts poll the route mailbox (routing must then signal)."""
    if not rubin(tp, batch, device):
        return False
    return bool(batch_table(POLLED.get(tp, "")).get(batch, 0))


def route_mailbox(workspace):
    """The route mailbox words (int32) of a ``scratch`` workspace."""
    geometry = ExpertsGeometry(16, slots=workspace_slots(workspace.device))
    start = geometry.ws_xflag + ROUTE_IDS
    return workspace[start : start + 4 * ROUTE_WORDS].view(torch.int32)


@functools.cache
def compile_rubin(
    experts, batch, grid, device, publish, slots, polled=False, dies=(), static=False,
    fast=False,
):  # fmt: skip
    """``publish``: ``compile_experts``' modes False / True / "push" (not two-shot).
    The w13 / w2 descriptors are grid-constant parameters encoded at launch from the
    two weight addresses passed last (``RubinExpertsKernel.grid_maps``); ``polled``:
    the routing kernel signals the route mailbox (``polled``); ``dies``: the SMs' dies
    when w2 was ``place``d (FC2 die matching); ``static``: constant epilogue-group
    barrier ids (``RubinExpertsKernel.static_groups``); ``fast``: the peer-slot K-split
    exchange (``RubinExpertsKernel.fast_exchange``)."""
    assert batch == 8, "the image plan holds every token row only at capacity"
    assert publish != "twoshot"
    moe = RubinExpertsKernel(
        experts, grid, slots, publish=publish, grid_maps=True, polled=polled,
        dies=dies, static_groups=static, fast_exchange=fast,
    )  # fmt: skip
    assert tensormap.AVAILABLE, "grid-constant descriptors need CuTe DSL >= 4.8"
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    ids = flat(batch * 16, c.Int32)
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), c.BFloat16)
    box = (c.Int64(0), c.Int32(0), c.Int32(0)) if publish else ()
    if publish == "push":  # + shared rows, the RS mailbox bases, capacity
        box += (flat(batch * 7168, c.BFloat16), fake((16,), None, c.Int64), c.Int32(0))
    tensors = (flat(PLAN_BYTES), ids, flat(256, align=64), *scales)
    tensors += (flat(moe.workspace_bytes), out)
    scalars = (c.Int32(batch), *map(c.Float32, (100, 0.25, 0.04)), *box)
    scalars += (c.Int64(0), c.Int64(0))  # the w13 / w2 addresses
    with torch.accelerator.device_index(device):
        return cute.compile(moe, *tensors, *scalars, make_fake_stream())


# Rubin TP16 batches of 64 tokens (B8 verify steps) on ``RubinGroupedExpertsKernel``
# (packed weights, K64 MMAs, grid-constant maps, static FC2), with the plan written by
# the routing kernel. One GPU, cold weights, real DSpark verify routes (~385 distinct
# experts / call): 48.6 -> 45.1 us per call.
RUBIN64 = {16: "64:1"}
# Verify steps (q > 1), by TP: row counts below 64 on the same kernel (its grouping,
# partition and finalize take the row count at run time; routing writes the plan above
# 8 rows), when B64 runs it (the weights are then ``place``d). One GPU, real verify
# routes (first M rows), placed and die-matched vs the grouped ``ExpertsKernel``: M16
# 25.9 -> 22.4, M32 37.0 -> 30.9, M48 44.4 -> 35.1 us per call.
RUBIN64_VERIFY = {16: "9-63:1"}
# Verify steps (q > 1), by TP: row counts above 64 on the wide grouped engines at
# capacity ceil(rows / 64) x 64 (out: this rank's partial latent rows, ``publish``
# "local") when B64 runs the grouped kernel (the weights are placed), by stream width:
# 8 = ``RubinWideExpertsKernel`` (8-token units), 16 / 32 = ``RubinVWExpertsKernel``
# (each expert's weights once per up to 16 / 32 tokens). One GPU, WideExperts with the
# plan written in front, real verify routes of all 92 MoE layers, us per call, width 8
# / 16: M72 38.55 / 38.68, M96 44.86 / 44.34, M128 52.62 / 50.54, M160 60.08 / 56.32,
# M192 66.07 / 60.73, M224 72.25 / 65.69, M256 78.17 / 70.18. 513-768 rows (DL3 q4
# B129-192; capacity up to ``experts_wide.WIDE_MAX``): width 16, 1 GPU, q4 routes of
# independent real requests (all 92 layers x 2 draws), us per call: M576 115.0, M640
# 123.9, M704 131.9, M768 138.6 (M512 106.4).
WIDE = {16: "65-80:8,81-768:16"}


# Wide rows' FC2 tail pool zone (``RubinGroupedExpertsKernel``'s pool = (zone, grain 6):
# the jobs at the end of every FC2(B) range that go into the pool), by rows; other
# wide rows keep (12, 6). Pool claims move whole FC2 jobs: bitwise (zone 48 vs 12, 28
# real calls each at M256 / 512 / 768). Longer FC2 ranges leave larger late-CTA
# excesses. 1 GPU, W16 front-planned, real q4 routes (92 layers x 2 draws), two
# allocations per arm, us per call 12 -> 48: M192 66.2 -> 65.7, M256 73.9 / 73.8 ->
# 73.4 / 73.2, M384 88.7 / 88.2 -> 87.4 / 87.0, M512 106.1 / 107.4 -> 105.0 / 104.9,
# M640 123.8 -> 121.9, M768 138.6 / 138.6 -> 136.0 / 136.1 (36: equal to 48 at <= 512,
# +0.8 at 768; M128: -0.1).
WIDE_POOL = {16: "257-768:48"}


def wide_capacity(rows):
    return (rows + 63) // 64 * 64


def wide_kernel(experts, grid, slots, rows, width, **kwargs):
    """The wide engine for ``rows`` at stream ``width`` (``WIDE``)."""
    cap = wide_capacity(rows)
    if width == 8:
        return RubinWideExpertsKernel(experts, grid, slots, cap, **kwargs)
    return RubinVWExpertsKernel(experts, grid, slots, cap, width=width, **kwargs)


def wide_bytes(slots, rows, tp=16):
    """The wide kernels' workspace bytes for up to ``rows`` (``scratch``): the largest
    over ``WIDE``'s widths."""
    widths = set(batch_table(WIDE.get(tp, "")).values()) | {8}
    return max(wide_kernel(896, slots, slots, rows, w).workspace_bytes for w in widths)


@functools.cache
def compile_wide_plan(batch, device):
    """``WidePlanKernel``: the 64-row plans of ``batch`` rows from x (no routing kernel
    in front)."""
    plan = fake((wide_capacity(batch) // 64 * PLAN_BYTES,), (1,), c.Uint8)
    args = (fake((batch, 3584)), plan, make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(WidePlanKernel(batch), *args)


@functools.cache
def compile_wide(
    experts, batch, grid, device, slots, dies=(), static=False, width=8,
    publish="local", cap=0, pool=(12, 6),
):  # fmt: skip
    """The wide engine (``wide_kernel``) at ``batch`` rows: the plan (64-row blocks),
    ids, the maps, scales, workspace, the routing weights, the SiTU scalars, the
    partial latent's address (``publish`` "twoshot": the wide tail's receive slots
    [3 x 16, ``cap``, 224], the owners' bases and this rank's slot instead), then the
    w13 / w2 addresses."""
    moe = wide_kernel(
        experts, grid, slots, batch, width, publish=publish, dies=dies,
        static_groups=static, pool=pool,
    )  # fmt: skip
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    tensors = (flat(moe.plan_bytes), flat(batch * 16, c.Int32), flat(256, align=64))
    tensors += (*scales, flat(moe.workspace_bytes), flat(batch * 16, c.BFloat16))
    scalars = (c.Int32(batch), *map(c.Float32, (100, 0.25, 0.04)))
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank
        scalars += (fake((48, cap, 224)), fake((16,), None, c.Int64), c.Int32(0))
    else:
        scalars += (c.Int64(0),)
    scalars += (c.Int64(0), c.Int64(0))  # the w13 / w2 addresses
    with torch.accelerator.device_index(device):
        return cute.compile(moe, *tensors, *scalars, make_fake_stream())


def rubin64(tp, batch, device):
    """Batches on the Rubin grouped experts kernel (routing writes the plan)."""
    if torch.cuda.get_device_capability(device) != (10, 7):
        return False
    return bool(batch_table(RUBIN64.get(tp, "")).get(batch, 0))


# Chunks of each placed weight whose home die ``place`` measures; passes over the
# chunks measured off their requested die (4 timing SMs a die, then 8, then 16);
# sampled chunks a weight may still leave unclear after them (one timing probe can stay
# noisy, e.g. beside another process's startup probe, while a misplaced allocation
# shows on about half of them).
PLACE_SAMPLES, PLACE_PASSES, PLACE_UNCLEAR = 8, 5, 1


def place(routed):
    """Rubin TP16 with M64 experts (``RUBIN64``): re-home one MoE layer's routed w13
    and w2 every 2 MiB alternately on die 0 / 1 (``dies.localized_like``: the same
    bytes and layout, so every reader sees the same values; the originals are
    released) and return the die of every SM for the experts kernels' die matching
    (``dies``: FC1 + FC2 at M64, FC2 at M8), or () (no trusted die map, other
    devices or TP: parameters untouched)."""
    from ..common import dies

    device = routed.w13_weight.device
    if routed.w13_weight.shape[1] != 512 or not rubin64(16, 64, device):
        return ()
    if dies.domain_count(device.index) < 2:
        return ()
    try:
        table = tuple(dies.sm_domains(device.index))
    except RuntimeError:  # no trusted die map: stay die-oblivious
        return ()
    # The map must also hold for streaming: every SM reads its own die's memory
    # faster than the other die's (once per device).
    wrong = dies.stream_check(device.index)
    if wrong:
        logger.warning_once(
            "Kimi K3 MoE: die map and streaming disagree on SMs %s; die-oblivious "
            "routed experts", wrong, scope="process",
        )  # fmt: skip
        return ()
    placed = []
    for weight in (routed.w13_weight, routed.w2_weight):
        copy = dies.localized_like(weight.data, [0, 1])
        chunks = -(-copy.numel() * copy.element_size() // dies.CHUNK)
        picks = {chunks * i // PLACE_SAMPLES for i in range(PLACE_SAMPLES)}
        sample = tuple(sorted(picks))
        # A few percent of single measurements are unclear (or, rarely, wrong):
        # chunks off their requested die are measured again with more SMs a die.
        wrong, sms = sample, 4
        for attempt in range(PLACE_PASSES):
            homes = dies.chunk_homes(copy, wrong, sms)
            wrong = tuple(c for c, home in zip(wrong, homes) if home != c % 2)
            if not wrong:
                break
            sms = 8 if attempt < 2 else 16
        if len(wrong) > PLACE_UNCLEAR:
            logger.warning(
                "Kimi K3 MoE: die-localized weights not homed as requested (chunks "
                "%s of %s); die-oblivious routed experts for this layer", wrong, sample,
            )  # fmt: skip
            return ()
        if wrong:
            logger.info(
                "Kimi K3 MoE: die-localized chunk(s) %s of %s still unclear after %d "
                "passes; kept placed", wrong, sample, PLACE_PASSES,
            )  # fmt: skip
        placed.append(copy)
    for weight, copy in zip((routed.w13_weight, routed.w2_weight), placed):
        weight.data = copy
        torch.accelerator.empty_cache()  # the original
    return table


@functools.cache
def compile_rubin64(
    experts, batch, grid, device, publish, slots, dies=(), static=False, zone=12
):  # fmt: skip
    """``compile_experts``' arguments (plan, publish modes incl. two-shot) for
    ``RubinGroupedExpertsKernel``, then the w13 / w2 addresses; ``dies``: the SMs'
    dies when the weights were ``place``d; ``static``: constant epilogue-group
    barrier ids (``ExpertsKernel.static_groups``); ``zone``: the FC2 tail pool's
    (``POOL_ZONE``)."""
    moe = RubinGroupedExpertsKernel(
        experts, grid, slots, publish=publish, pool=(zone, 6), dies=dies,
        static_groups=static,
    )  # fmt: skip
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    box = (c.Int64(0), c.Int32(0), c.Int32(0)) if publish else ()
    if publish == "push":  # + shared rows, the RS mailbox bases, capacity
        box += (flat(batch * 7168, c.BFloat16), fake((16,), None, c.Int64), c.Int32(0))
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank
        box = (fake((48, 64, 224)), fake((16,), None, c.Int64), c.Int32(0))
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), c.BFloat16)
    tensors = (flat(PLAN_BYTES), flat(batch * 16, c.Int32), flat(256, align=64))
    tensors += (*scales, flat(moe.workspace_bytes), out)
    scalars = (c.Int32(batch), *map(c.Float32, (100, 0.25, 0.04)), *box)
    scalars += (c.Int64(0), c.Int64(0))  # the w13 / w2 addresses
    with torch.accelerator.device_index(device):
        return cute.compile(moe, *tensors, *scalars, make_fake_stream())


@functools.cache
def compile_image_plan(batch, device):
    """Image-plan writer for callers without the routing kernel in front."""
    args = (fake((batch, 3584)), fake((PLAN_BYTES,), (1,), c.Uint8))
    with torch.accelerator.device_index(device):
        return cute.compile(ImagePlanKernel(batch), *args, make_fake_stream())


@functools.cache
def compile_group(tp, experts, batch, grid, device, early=False, slots=160):
    """The pregrouped experts' unit-table kernel: (ids, workspace, batch); ``early``:
    it releases the experts at entry."""
    capacity = 1 << (batch - 1).bit_length()
    kernel = ExpertsGroupKernel(tp, capacity, experts, grid, early=early, slots=slots)
    flat = lambda n, dtype=c.Uint8: fake((n,), (1,), dtype, 16)
    args = (flat(batch * 16, c.Int32), flat(kernel.workspace_bytes), c.Int32(batch))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@functools.cache
def compile_plan(batch, device):
    """Plan writer for callers without the routing kernel's copy role in front."""
    args = (fake((batch, 3584)), fake((PLAN_BYTES,), (1,), c.Uint8), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(ExpertsPlanKernel(batch), *args)


def stage_homes(workspace, kernel):
    """The home die bit c of each 2 MiB chunk under the wide kernels' staged rows
    (``dies.page_domains``), into the two workspace words at ``kernel.ws_homes``: bit k
    for the k-th chunk from the one holding the first staged byte (word k // 32), 0 for
    a chunk without a clear answer. Only the die-local finalize's locality depends on
    them. Outside CUDA graph captures (probe launches)."""
    from ..common import dies

    bits = 0
    if dies.domain_count(workspace.device.index) > 1:
        span = 16 * WIDE_MAX * 7168  # the staged rows at the largest capacity
        stage = workspace[kernel.ws_stage : kernel.ws_stage + span]
        homes = dies.page_domains(stage)  # probes the staged rows' bytes only
        chunks = -(-(stage.data_ptr() % dies.CHUNK + span) // dies.CHUNK)
        assert chunks <= 63
        chunks = min(chunks, homes.numel() * dies.PAGE // dies.CHUNK)
        for k in range(chunks):
            if int(homes[k * (dies.CHUNK // dies.PAGE)]) == 1:  # page 0: parity 0
                bits |= 1 << k
    words = [
        (bits >> s & 0xFFFFFFFF) - ((bits >> s & 0x80000000) << 1) for s in (0, 32)
    ]
    homes = workspace[kernel.ws_homes : kernel.ws_homes + 8].view(torch.int32)
    homes.copy_(torch.tensor(words, dtype=torch.int32))
    return bits


def scratch(tp, device, rows=64):
    """Model-owned expert buffers: the workspace and the grouped plan (MXFP8 x, written
    by the routing kernel or ``compile_plan``); ``rows`` above 64: also the wide
    kernels' workspace and ceil(rows / 64) 64-row plans."""
    slots = workspace_slots(device)
    geometry = ExpertsGeometry(tp, slots=slots)
    size = geometry.workspace_bytes
    # The route mailbox lies inside the 64-row workspace, below every region past it.
    assert geometry.ws_xflag + ROUTE_IDS + 4 * ROUTE_WORDS <= size
    # RubinGroupedExpertsKernel's die census, then RubinExpertsKernel's peer slots
    # (``fast_exchange``; armed by the 0xFF fill), past the rest.
    size = peers_offset(size, CENSUS_BYTES) + peers_bytes(slots)
    size += generation_bytes(slots)  # ``RubinExpertsKernel.fast_exchange``'s
    plan_bytes = PLAN_BYTES
    if rows > 64:
        size = max(size, wide_bytes(slots, rows, tp))
        plan_bytes = wide_capacity(rows) // 64 * PLAN_BYTES
    u8 = torch.uint8
    # 2 MiB-aligned: the wide kernels' staged units sit on whole 4 KiB pages, each pair
    # on one die apiece (``RubinWideExpertsKernel.finalize``).
    chunk = 2 << 20
    workspace = torch.full((size + chunk,), 255, device=device, dtype=u8)
    start = -workspace.data_ptr() % chunk
    workspace = workspace[start : start + size]
    plan = torch.empty(plan_bytes, device=device, dtype=u8)
    return {"workspace": workspace, "plan": plan}


# Persistent grid and per-batch overrides by (TP, SMs); other SM counts take the 152-SM
# entries. 152 SMs: TP8 B3-4 on 148 CTAs; B5-8 publish on 128-136 CTAs, leaving SMs for
# the one-wave tail; grouped B9-64 on 152. 212 SMs (TP16): B5-64 on every SM (B5-8: the
# shared expert then finishes first; split batches leave their down SMs, see
# ``moe.prepared``); B2-4 on 192 with the shared expert's down rows beside them (B3-4
# ~0.7 us per layer faster than 212); B1 on 152 (TP16 B1-2 run the small-batch chain).
GRID = {(8, 152): 136, (16, 152): 152, (16, 212): 212}
GRIDS = {(8, 152): "3-4:148,5-6:128,7:132,9-64:152", (16, 212): "1:152,2-4:192"}


def grids(tp, device):
    """The persistent grid and its per-batch overrides for the device's SM count."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    key = (tp, sms) if (tp, sms) in GRID else (tp, 152)
    return GRID[key], batch_table(GRIDS.get(key, ""))


def every_sm(tp, batch, device):
    """Whether the persistent grid at ``batch`` spans every SM of the device (before
    ``moe.prepared`` gives a split shared expert's down rows their SMs, B>8)."""
    grid, overrides = grids(tp, device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return overrides.get(batch, grid) >= sms


class PersistentExperts:
    def __init__(
        self, w13, s13, w2, s2, *, beta=4.0, linear_beta=25.0, dies=(), q=1
    ):  # fmt: skip
        experts, rows, _ = w13.shape
        self.tp = 8 if rows == 768 else 16
        self.device, self.experts = w13.device, experts
        self.grid, self.grids = grids(self.tp, self.device)
        self.dies = tuple(dies)  # the weights are ``place``d (M64 die-aware experts)
        # verify steps: ``STATIC_GROUPS`` batches
        self.static = batch_table(STATIC_GROUPS.get(self.tp, "") if q > 1 else "")
        self.fast = batch_table(FAST_EXCHANGE.get(self.tp, "") if q > 1 else "")
        self.rows64 = batch_table(RUBIN64_VERIFY.get(self.tp, "") if q > 1 else "")
        self.zones = batch_table(POOL_ZONE.get(self.tp, "") if q > 1 else "")
        self.wide = batch_table(WIDE.get(self.tp, "") if q > 1 else "")
        self.wide_pool = batch_table(WIDE_POOL.get(self.tp, "") if q > 1 else "")
        self.slots = workspace_slots(self.device)
        self.weights = (w13, s13, w2, s2)
        scalars = (beta * linear_beta, 1 / beta, 1 / linear_beta)
        self.scalars = tuple(map(c.Float32, scalars))
        # The experts kernel uses maps 0-1 (L2 promotion on w13); the TP8 B1 cluster
        # experts use maps 2-4 (w13, w2 two-chunk and one-chunk boxes, promoted); TP8
        # FC2 pairs (B4-64) use map 5 (w2 chunk 2 of 256 rows).
        w2_tile = (192, 3, 64) if self.tp == 8 else (128, 2, 32)
        specs = [(w13, rows, 1792, 28, 64, 2, True), (w2, 3584, *w2_tile, 2, False)]
        if self.tp == 8:
            specs += [specs[0]] + [(w2, 3584, 192, 3, 64, box, True) for box in (2, 1)]
        dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
        swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        l2 = cuda.CUtensorMapL2promotion
        l2 = (l2.CU_TENSOR_MAP_L2_PROMOTION_NONE, l2.CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        maps = [
            (weight.data_ptr(), dtype, (128, experts * n, chunks), (row_bytes, stride))
            + ((128, 128, box), swizzle, l2[promote])
            for weight, n, row_bytes, chunks, stride, box, promote in specs
        ]
        if self.tp == 8:
            shape = (128, experts * 3584, 3), (192, 64), (128, 256, 1)
            maps.append((w2.data_ptr(), dtype, *shape, swizzle, l2[False]))
        self.maps = tensor_maps(maps, self.device)
        scales = (tensor(s.view(torch.uint8)) for s in (s13, s2))
        self.views = (to_cute(self.maps[:256], 64), *scales)
        self.all_maps = to_cute(self.maps, 64)  # TP8 FC2 pairs: maps 0-5
        self.rubin: set[int] = set()  # batches on RubinExpertsKernel
        self.addresses: dict[int, tuple] = {}  # their w13 / w2 descriptor sources
        self.prepared = {}

    def prepare(
        self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False
    ):
        """``publish``: (multicast, parity, rank) of the WaveMoeTail latent slot to
        publish (``twoshot``: the ``common.twoshot`` push box); ``planned``: calls find
        the plan written by the routing kernel (B>8, and ``rubin`` batches, which then
        run ``RubinExpertsKernel``), else B>8 run writes it from x first;
        ``early``: the caller runs ``group`` (pregrouped batches) before ``run``."""
        if batch > 64:
            self.prepare_wide(batch, buffers, out, publish, planned)
            return
        mode = "twoshot" if twoshot else "push" if len(publish) > 3 else bool(publish)
        if planned and not twoshot and rubin(self.tp, batch, self.device):
            args = (self.experts, batch, self.grid_for(batch), self.device.index)
            poll = polled(self.tp, batch, self.device)
            dies = self.dies if args[2] == len(self.dies) else ()  # one CTA per SM
            static, fast = (bool(self.static.get(batch)), bool(self.fast.get(batch)))
            if fast:
                need = ExpertsGeometry(self.tp, slots=self.slots).workspace_bytes
                need = peers_offset(need, CENSUS_BYTES) + peers_bytes(self.slots)
                need += generation_bytes(self.slots)
                assert buffers["workspace"].numel() >= need, "no peer slots (scratch)"
            code = compile_rubin(*args, mode, self.slots, poll, dies, static, fast)
            what = "Kimi K3 MoE experts"
            check_residency(code, self.device.index, args[2], 384, what)
            self.rubin.add(batch)
            ws, plan = (tensor(buffers[k]) for k in ("workspace", "plan"))
            work = (code, publish, (plan,), self.views[0], None)
            self.prepared[batch] = (out, tensor(out), ws, *work)
            weights = (self.weights[0], self.weights[2])
            self.addresses[batch] = tuple(c.Int64(w.data_ptr()) for w in weights)
            return
        if planned and batch > 8 and self.grouped64(batch):
            grid = self.grid_for(batch)
            code = compile_rubin64(
                self.experts, batch, grid, self.device.index, mode, self.slots,
                self.dies, bool(self.static.get(batch)), self.zones.get(batch, 12),
            )  # fmt: skip
            check_residency(code, self.device.index, grid, 384, "Kimi K3 MoE experts")
            ws = tensor(buffers["workspace"])
            plan = (tensor(buffers["plan"]),)
            work = (code, publish, plan, self.views[0], None)
            self.prepared[batch] = (out, tensor(out), ws, *work)
            weights = (self.weights[0], self.weights[2])
            self.addresses[batch] = tuple(c.Int64(w.data_ptr()) for w in weights)
            return
        grouped = pregrouped(self.tp, batch)
        args: tuple = (batch, self.grid_for(batch), self.device.index, mode)
        args += (streamed(batch), grouped, stealing(batch))
        code = compile_experts(self.tp, self.experts, *args, slots=self.slots)
        # CTAs wait on other clusters' partials: the whole grid must be resident.
        check_residency(code, self.device.index, args[1], 384, "Kimi K3 MoE experts")
        ws, maps = tensor(buffers["workspace"]), self.views[0]
        plan: tuple | None = None
        capacity = 1 << (batch - 1).bit_length()
        if fc2_pairs(self.tp, capacity, batch > 8 or streamed(batch)):
            maps = self.all_maps
        group = None
        if batch > 8:
            plan = (tensor(buffers["plan"]), compile_plan(batch, self.device.index))
            plan = plan[:1] if planned else plan
        if grouped:
            release = early_group(self.tp, batch)
            grouping = compile_group(
                self.tp, self.experts, *args[:3], release, slots=self.slots
            )
            group = grouping, ws, early
        self.prepared[batch] = (out, tensor(out), ws, code, publish, plan, maps, group)

    def group(self, ids):
        """Pregrouped batches prepared ``early``: the unit-table kernel (run it on the
        stream after routing; ``run`` then skips it)."""
        group = self.prepared[ids.shape[0]][7]
        if group is not None and group[2]:
            self.launch_group(ids, group)

    def launch_group(self, ids, group):
        code, ws, _ = group
        code(tensor(ids), ws, c.Int32(ids.shape[0]), stream(self.device))

    def grid_for(self, batch):
        return self.grids.get(batch, self.grid)

    def prepare_wide(self, batch, buffers, out, publish, planned):
        """``WIDE`` rows: the wide engine at ``WIDE``'s stream width with ``publish`` =
        ("local", partial): this rank's partial latent rows into partial[:batch] (BF16
        [>= batch, 3584]; ``run`` returns it), the routing weights passed to ``run``;
        the plan (64-row blocks) as the routing kernel wrote it (``planned``) or
        written by ``run`` from x first. ``publish`` (): the pair rows into ``out`` is
        not offered (16 x the bytes)."""
        if not (self.wide.get(batch) and rubin64(self.tp, 64, self.device)):
            raise ValueError(f"B{batch}: the routed experts take at most 64 tokens")
        twoshot = len(publish) == 4 and publish[0] == "twoshot"
        if not (twoshot or (len(publish) == 2 and publish[0] == "local")):
            raise ValueError(f"B{batch}: wide experts publish into a local box")
        partial = out if twoshot else publish[1]
        assert partial.shape[0] >= batch and partial.shape[1] == 3584
        grid = self.grid_for(batch)
        static = bool(self.static.get(batch))
        mode, cap = ("twoshot", publish[1].shape[1]) if twoshot else ("local", 0)
        pool = (self.wide_pool.get(batch, 12), 6)
        code = compile_wide(
            self.experts, batch, grid, self.device.index, self.slots, self.dies, static,
            self.wide[batch], mode, cap, pool,
        )  # fmt: skip
        check_residency(code, self.device.index, grid, 384, "Kimi K3 MoE experts")
        width, slots = (self.wide[batch], self.slots)
        kernel = wide_kernel(self.experts, slots, slots, batch, width)
        assert buffers["workspace"].numel() >= kernel.workspace_bytes, (
            "scratch too small"
        )
        if self.dies and "homes" not in buffers:  # once per scratch (every capacity)
            buffers["homes"] = stage_homes(buffers["workspace"], kernel)
        plan = buffers["plan"]
        assert plan.numel() >= wide_capacity(batch) // 64 * PLAN_BYTES
        ws = tensor(buffers["workspace"])
        box: tuple = (c.Int64(partial.data_ptr()),)
        if twoshot:  # ("twoshot", receive slots, owners' bases, this rank's slot)
            recv, peers, rslot = publish[1:]
            box = (to_cute(recv), to_cute(peers, 16), c.Int32(rslot))
        writer = () if planned else (compile_wide_plan(batch, self.device.index),)
        work = (code, box, (tensor(plan), *writer), self.views[0], None)
        self.prepared[batch] = (partial[:batch], tensor(out), ws, *work)
        weights = (self.weights[0], self.weights[2])
        self.addresses[batch] = tuple(c.Int64(w.data_ptr()) for w in weights)

    def grouped64(self, batch):
        """Rows on ``RubinGroupedExpertsKernel`` (``RUBIN64``, ``RUBIN64_VERIFY``)."""
        if rubin64(self.tp, batch, self.device):
            return True
        return bool(self.rows64.get(batch)) and rubin64(self.tp, 64, self.device)

    def run(self, x, ids, weights=None):
        batch = x.shape[0]
        out, output, workspace, code, box, plan, maps, group = self.prepared[batch]
        output = tensor(weights) if box else output
        source = to_cute(x, 16)
        if plan is not None:
            if len(plan) > 1:
                plan[1](source, plan[0], stream(self.device))
            source = plan[0]
        if group is not None and not group[2]:
            self.launch_group(ids, group)
        args = (source, tensor(ids), maps, *self.views[1:], workspace, output)
        extra = self.addresses.get(batch, ())
        code(*args, c.Int32(batch), *self.scalars, *box, *extra, stream(self.device))
        return out


class WideExperts:
    """Seam S5's routed experts (``stock_experts.StockExperts``' interface) on the wide
    engines for Rubin TP16 verify rows above 64 (``WIDE``): ``run`` writes this rank's
    routed partial [M, 3584] BF16 (the top-k weighted sum over the rank's share of the
    intermediate dimension, before the TP reduction; the finalize's FP32 FMA chain in
    slot order) into the partial buffer its row count was prepared with. Numerics: the
    64-row experts' (MXFP8 activations, K64 MMA grouping), not bitwise equal to the
    stock kernel. Prepare outside CUDA graph captures."""

    def __init__(self, experts, buffers, partials, *, planned, pushed=None):
        """``experts``: the layer's ``PersistentExperts`` (q > 1, weights ``place``d);
        ``buffers``: ``scratch(tp, device, rows=)`` for the largest row count;
        ``partials``: {rows: BF16 [>= rows, 3584], written at every call};
        ``planned``: the routing in front wrote the plan (MXFP8 latent, 64-row blocks)
        into buffers["plan"] (``front_rows.FrontRows``), else ``run`` writes it from
        ``latent`` first (``WidePlanKernel``)."""
        self.experts, self.partials = experts, dict(partials)
        # [axl wpush] ``pushed``: {rows: (receive slots, owners' bases, this rank's
        # slot)} of the wide tail's two-shot latent box (``tail_rows``'s
        # ``experts_box``): those rows' finalize pushes the latent columns to their
        # owners, so the tail only reduces; the partial buffer is then not written.
        pushed = pushed or {}
        for rows, partial in self.partials.items():
            publish = (
                ("twoshot", *pushed[rows]) if rows in pushed else ("local", partial)
            )
            experts.prepare_wide(rows, buffers, partial, publish, planned)

    def run(self, latent, topk_weights, topk_ids, out=None):
        """``latent`` [M, 3584] BF16, ``topk_weights`` [M, 16] BF16, ``topk_ids``
        [M, 16] int32 (as the front's routing writes them: no conversion launches)."""
        rows = latent.shape[0]
        partial = self.partials[rows][:rows]
        if out is not None and out.data_ptr() != partial.data_ptr():
            raise ValueError(f"B{rows}: wide experts write the prepared partial only")
        if topk_weights.dtype != torch.bfloat16 or topk_ids.dtype != torch.int32:
            raise ValueError("wide experts take BF16 top-k weights and int32 ids")
        return self.experts.run(latent, topk_ids, topk_weights.reshape(-1))
