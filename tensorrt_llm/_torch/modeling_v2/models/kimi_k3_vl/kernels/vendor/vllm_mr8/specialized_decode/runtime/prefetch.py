# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for the specialized decode.

Small-batch decode is bound by HBM bandwidth, and each kernel that starts on weights
not yet in L2 stalls on DRAM. Some kernels leave bandwidth idle, such as the attention
core and the all-reduces. At fixed points of each layer, this module launches a small
kernel (``cutedsl/common/prefetch.py``) on a side stream. It pulls the byte ranges of
the weights the next kernels read into L2, so that they start warm.

Prefetch points (``fork(point, layer, batch)``), each with its batches per TP:

- ``oproj_down``: after the o-projection (B3-8; none on Rubin-class devices, see
  ``WIDE``), the MoE front's packed router + latent-down rows.
- ``attention_moe``: after the KDA input projection or the MLA core, the same front
  rows plus the shared expert's weights (``ATTENTION_MOE``; at TP16 only on
  Rubin-class devices, see ``WIDE``; MLA layers at ``ATTENTION_MOE_MLA_LEAN``
  batches without the gate/up).
- ``next_attention``: once the routed experts are launched, the next layer's
  attention weights, capped in MiB per batch (``NEXT_ATTENTION``; ``WIDE`` on
  Rubin-class devices). Not above B8, where the HBM-bound experts lose more
  bandwidth than the attention gains, nor at the batches of ``BEFORE_HEAD`` before a
  KdaHead layer and of ``BEFORE_TMEM_MLA`` before an MLA layer whose in-projection
  stages its weights in tensor memory.

The front and shared-expert ranges are on the critical path. They launch with the
highest stream priority and the maximum shared-memory carveout, which needs a driver
launch because CuTe drops both attributes.

Each fork records an event on the main stream that the side stream waits on, so a
prefetch starts only after the work it overlaps. ``join()`` at the end of the step
makes the main stream wait for the side stream, so a captured graph always rejoins
it. This lives in ``runtime/`` rather than ``cutedsl/`` because it is host-side
orchestration across layers: a stream, events, and per-(point, layer, batch) tables
built from the model's layers.
"""

import ctypes
from functools import partial

import cuda.bindings.driver as cuda
import torch

from ...ops.cute_dsl.latent_moe_tail.primitives import to_cute
from ..cutedsl.common.host import _checked, resolve_kernel
from ..cutedsl.common.policy import batch_table
from ..cutedsl.common.prefetch import compile_kernel

# Per-TP batch -> MiB cap (or cap.carveout) of the next layer's attention weights,
# prefetched once the routed experts finish.
# Not at B>8: the HBM-bound experts lose more bandwidth than the attention gains.
NEXT_ATTENTION = {8: "1:96.100,2-3:96,4-7:64,8:32", 16: "1-8:1024"}
# Batches with the ``oproj_down`` front-rows prefetch, and the next-attention grid.
OPROJ_DOWN, GRID = "3-8:1", 32
# Per-TP batch -> grid of the ``attention_moe`` prefetch at (KDA, MLA) layers, launched
# with the top stream priority and carveout 100.
ATTENTION_MOE = {8: ("5-8:32", "8:32")}
# Devices with more SMs than GB300's 152 (Rubin: 212), per TP: (NEXT_ATTENTION spec,
# grid, OPROJ_DOWN spec, ATTENTION_MOE specs, launched plainly). There the front GEMM
# is resident beside the o-projection and preloads its rows itself, so the
# ``oproj_down`` prefetch only adds traffic. The next-attention prefetch is issue-bound
# per CTA: 32 single-warp CTAs need ~10 us, about the whole MoE tail + AttnRes gap, 96
# finish 48 MiB (the in-projection first, within the ~56 MiB of L2 one die keeps
# resident) in ~3.7 us. The front rows and the shared expert's gate/up and down
# (20.6 MB at TP16) are otherwise cold where the shared chain gates the routed experts
# (KDA B5-8, MLA B3-8); 64 plainly launched CTAs fetch them beside KdaSplit / the
# o-projection, before the front and shared GEMMs start. Not at KDA B8: forked after
# KdaHead it gains the front nothing (the front GEMM and the shared gate/up preload
# their weights during the o-projection's exchange), and its traffic delays the
# routed experts and the next AttnRes -> KdaHead launch (the same launch with 16 bytes
# of ranges costs nothing). KDA layer proxy (tail -> AttnRes -> KdaHead + F_A ->
# o-projection -> front + shared chain -> experts, side streams joined once), us/layer
# vs this prefetch: 1 GPU none -0.84, half -0.49; 4 ranks none -0.89 / -0.96;
# 16 ranks none -0.69 / -0.61, half -0.20 / -0.10.
# At B8 the published MoE tail leaves the next-attention prefetch less room: 48 MiB
# right after the routed experts contends with the tail's own weight and mailbox
# reads. One-GPU chain (experts -> tail -> AttnRes -> the cold 46.8 MB in-projection,
# the prefetch forked after the experts as here), us/layer: none 34.91-34.98, 8 MiB
# 34.77, 16 MiB 34.39, 24 MiB 34.35, 32 MiB 34.58-34.61, 48 MiB 35.69-35.89.
WIDE = {16: ("1-7:48,8:24", 96, "", ("5-7:64", "3-8:64"))}
# Per-TP batches whose MLA-layer ``attention_moe`` prefetch (``WIDE`` devices) fetches
# the front rows and the shared down only (9.6 of 19.7 MiB at TP16), not the shared
# gate/up, which requests its own 11 MB slice before its PDL wait beside the
# o-projection. The front rows keep the slowest rank's front GEMM warm (the route skew
# without any MLA prefetch); the down shortens the shared chain, which at 16 ranks
# ends only ~0.4 us before routing. 16-rank MLA layer proxy (every rank real, side
# streams joined once), us/layer: front + down -0.17 / -0.16 vs none and -0.37 vs all
# three ranges; front rows only -0.02, first half of the bytes -0.01.
ATTENTION_MOE_MLA_LEAN = {16: "8:1"}
# Per-TP batches whose ``attention_moe`` prefetch (``WIDE`` devices) spreads its
# 16-KiB chunks thread-major over the grid (``L2Prefetch`` spread): the 3.9 MiB of
# front rows come first in the table, and by default land on 8 of the 64 CTAs, which
# issue them in ~3.5 us while the front GEMM is already loading; spread, all 64 issue
# them at once.
ATTENTION_MOE_SPREAD = {16: "8:1"}
# Per-TP batches without the next-attention prefetch when the next layer runs KdaHead
# (Rubin TP16 B8, ``skip_before_heads``). KdaHead loads its own in-projection tiles
# before its PDL wait, and the prefetch's 96 single-warp CTAs hold ~72 SMs through the
# MoE tail while KdaHead needs whole free SMs to become resident. One-GPU chain
# (experts -> tail -> early-released AttnRes -> KdaHead, B8), us/layer: 48 MiB 38.6,
# 24 MiB 36.88-37.07, 12 MiB 36.84-36.87, 6 MiB 36.74, none 36.61-36.70.
BEFORE_HEAD = {16: "8:1"}
# Per-TP batches without the next-attention prefetch when the next layer is an MLA
# layer whose in-projection runs SplitKTmemGemm (Rubin TP16). Its 161 CTAs are resident
# beside the carve-out-100 AttnRes and request their whole weight slice before the PDL
# wait, so the prefetch only competes with them and with the MoE tail. Experts -> tail
# -> AttnRes -> in-projection chain, B8, side streams joined once as in serving,
# us/layer (mean over ranks): 4 ranks 24 MiB 35.53-35.64, 12 MiB 35.34-35.40, 6 MiB
# 35.19-35.32, 3 MiB 35.10-35.19, none 35.03; 16 ranks 24 MiB 35.02-35.07, 12 MiB
# 35.04-35.10, 6 MiB 34.95-35.06, 3 MiB 34.92-35.01, none 34.83-34.85.
BEFORE_TMEM_MLA = {16: "8:1"}


def attention_weights(attention):
    """Next-attention weights in their consumption order."""
    names = (
        *("in_proj_qkvgfab", "fused_qkv_a_g_proj", "fused_qkv_a_proj", "q_proj"),
        *("kv_a_proj_with_mqa", "f_b_proj", "q_b_proj", "g_proj"),
    )
    weights = [getattr(getattr(attention, n, None), "weight", None) for n in names]
    weights += [getattr(attention, "W_UK_T", None), getattr(attention, "W_UV", None)]
    weights = [w for w in weights if isinstance(w, torch.Tensor)]
    return (*weights, attention.o_proj.weight)


class PreparedWeightRanges:
    """One owner retains weights, immutable range table, executable and launch ABI."""

    def __init__(
        self, weights, *, byte_cap=0, grid=32, carveout=None, priority=False,
        spread=False,
    ):  # fmt: skip
        self.weights = weights = tuple(weights)
        device = weights[0].device
        ranges, remaining = [], byte_cap
        for weight in weights:
            if weight.numel() == 0:
                continue
            if weight.data_ptr() % 16:  # the kernel walks 16-byte aligned chunks
                raise ValueError("L2 prefetch ranges must start 16-byte aligned")
            span = 1 + sum((n - 1) * s for n, s in zip(weight.shape, weight.stride()))
            size = span * weight.element_size()
            if byte_cap:
                size = min(size, remaining) // 16 * 16
                remaining -= size
            if size:
                ranges.append((weight.data_ptr(), size))
            if byte_cap and remaining < 16:
                break
        self.table = torch.tensor(ranges, dtype=torch.int64, device=device)
        self.compiled = compile_kernel(
            len(ranges), grid, carveout, device.index, spread
        )
        self.config = None
        if not priority and carveout is None:
            self.views = to_cute(self.table, 16)
            return
        # Launch priority / carveout need a driver launch (CuTe drops both).
        self.executor, self.function = resolve_kernel(self.compiled, device.index)
        ids, self.attributes = cuda.CUlaunchAttributeID, []
        if priority:  # the greatest stream priority
            top = cuda.CUlaunchAttribute()
            top.id = ids.CU_LAUNCH_ATTRIBUTE_PRIORITY
            top.value.priority = _checked(cuda.cuCtxGetStreamPriorityRange())[1]
            self.attributes.append(top)
        if carveout is not None:
            smem = cuda.CUlaunchAttribute()
            smem.id = ids.CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            smem.value.sharedMemCarveout = carveout
            self.attributes.append(smem)
        config = self.config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 32, 1, 1
        config.sharedMemBytes, config.attrs = 0, self.attributes
        config.numAttrs = len(self.attributes)
        self.pointer = ctypes.c_void_p(self.table.data_ptr())
        self.arguments = (ctypes.c_void_p * 1)(ctypes.addressof(self.pointer))
        self.address = ctypes.addressof(self.arguments)

    def launch(self, stream):
        if self.config is None:
            self.compiled(self.views, stream)
        else:
            self.config.hStream = stream
            _checked(cuda.cuLaunchKernelEx(self.config, self.function, self.address, 0))


class PrefetchController:
    """Prefetch launches per (point, layer, batch) on one side stream."""

    def __init__(self, runtime, layers, packed):
        self.device, tp = runtime.device, runtime.tp
        self.tp = tp
        self.stream = torch.cuda.Stream(device=self.device)
        self.events, self.pending, self.tables = {}, False, {}
        urgent = partial(PreparedWeightRanges, carveout=100, priority=True)
        spec, self.grid, front = NEXT_ATTENTION[tp], GRID, OPROJ_DOWN
        shared_specs, shared_launch = ATTENTION_MOE.get(tp, ("", "")), urgent
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        lean: dict = {}  # MLA batches without the shared gate/up range
        spread: dict = {}  # batches whose chunks spread thread-major
        if sms > 152 and tp in WIDE:
            spec, self.grid, front, shared_specs = WIDE[tp]
            shared_launch = PreparedWeightRanges
            lean = batch_table(ATTENTION_MOE_MLA_LEAN.get(tp, ""))
            spread = batch_table(ATTENTION_MOE_SPREAD.get(tp, ""))
        following, front = batch_table(spec), sorted(batch_table(front))
        for position, layer in enumerate(layers):
            if not layer.is_moe_layer:
                continue
            index, moe, attention = layer.layer_idx, layer.mlp, layer.self_attn
            # The front GEMM reads [router shard; down shard] from the packed copy.
            if front:
                ranges = urgent((packed[index],), grid=8)
                self._add("oproj_down", index, front, ranges)
            kda = hasattr(attention, "in_proj_qkvgfab")
            grids = batch_table(shared_specs[0 if kda else 1])
            shared = moe.shared_experts
            gate_up, down = shared.gate_up_proj.weight, shared.down_proj.weight
            # (grid, MLA without the gate/up range, spread chunks) per batch
            key = {b: (g, not kda and bool(lean.get(b)), bool(spread.get(b)))
                   for b, g in grids.items()}  # fmt: skip
            for grid, slim, spreads in sorted(set(key.values())):
                weights = (packed[index], *((down,) if slim else (gate_up, down)))
                ranges = shared_launch(weights, grid=grid, spread=spreads)
                batches = sorted(b for b in grids if key[b] == (grid, slim, spreads))
                self._add("attention_moe", index, batches, ranges)
            if position + 1 < len(layers):
                weights = attention_weights(layers[position + 1].self_attn)
                self._add_capped("next_attention", index, weights, following)

    def skip_before(self, layer, batches, table):
        """No next-attention prefetch after ``layer`` at those of ``batches`` (where
        the next layer runs KdaHead or a tensor-memory MLA in-projection) listed in
        ``table`` (``BEFORE_HEAD`` or ``BEFORE_TMEM_MLA``)."""
        skip = batch_table(table.get(self.tp, ""))
        for batch in batches:
            if skip.get(batch):
                self.tables.pop(("next_attention", layer, batch), None)

    def _add_capped(self, point, layer, weights, table):
        groups: dict[tuple, list[int]] = {}
        for batch, spec in table.items():
            cap, carveout = spec if isinstance(spec, tuple) else (spec, None)
            if cap:
                groups.setdefault((cap, carveout), []).append(batch)
        for (cap, carveout), batches in groups.items():
            ranges = PreparedWeightRanges(
                weights, byte_cap=cap * 1024**2, grid=self.grid, carveout=carveout
            )
            self._add(point, layer, batches, ranges)

    def _add(self, point, layer, batches, ranges):
        key = point, layer
        if key not in self.events:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.events[key] = event
        for batch in batches:
            self.tables[point, layer, batch] = ranges

    def fork(self, point, layer, batch):
        ranges = self.tables.get((point, layer, batch))
        if ranges is None:
            return False
        event = self.events[point, layer]
        event.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(event)
        self.pending = True
        ranges.launch(cuda.CUstream(self.stream.cuda_stream))
        return True

    def join(self):
        if self.pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.pending = False
