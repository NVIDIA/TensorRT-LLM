# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``replicated_experts``: ring-buddy replicated experts with per-step balanced placement.

Under tep8 expert parallelism rank ``r`` owns experts ``[32r, 32r+32)``. At the
DFlash verify width the eight rows select ~30 distinct experts per layer, so
the per-rank count follows a balls-into-bins spread and every rank waits in
the finalize all-reduce for the one that streams the most expert weights.

``install`` gives every MoE layer (the 60 NVFP4 layers and the two BF16
layers the checkpoint leaves unquantized) a second copy of its ring
predecessor's processed expert tensors -- gathered from that rank once, after
the built-in loader finished, so the copy is bitwise the owner's -- stacked as
``[own block | predecessor's block]`` (64 local experts). The built-in
parameters are re-bound as views of the stack's first half, so the own block
is stored once.

Each decode step of at most ``max_rows`` rows (``MAX_ROWS``, 16; 32 on sm103 at
tep8, ``MAX_ROWS_BY_SM_TP``), the routing method's call additionally
runs ``ops/moe/ring_buddy_assign`` on the main stream. In the built-in MoE
input schedule it overlaps the pool/quantize work of the preparation stream;
where the main stream is the longer branch instead (the router after the pool
GEMM, or a shorter pool branch), it sits on the critical path in place of the
BF16 cast of the routing weights that it makes unnecessary, which is why it is
one small program. It places every selected expert
on its owner or its successor so the busiest rank runs as few experts as the
ring allows, and rewrites the top-k ids into this rank's stacked id space. The
expert call then runs over the stacked tensors with a 64-expert local window.
Its unfinalized output feeds the built-in fused finalize all-reduce unchanged:
the placement partitions the (token, slot) pairs across ranks, so the
cross-rank sum still adds every selected expert row exactly once.

Rounding-different from the built-in path: the finalize all-reduce rounds each
rank's partial sum to BF16 before adding the ranks, and the placement changes
which expert rows share a partial. Expert selection is unchanged.

Prefill rows and any call the stacked path does not cover take the built-in
expert call, which now reads the own-block views.

This module is a load-time installer, not an op: it patches the built-in
modules once the weights are final, so it sits beside ``modeling.py`` and is
imported only when ``replicated_experts`` is planned. Its one kernel is the
``ops/moe/ring_buddy_assign`` entry; with ``persistent_experts`` the NVFP4 layers' stacked
call is the ``ops/moe/persistent_experts`` entry instead of TRTLLM-gen's
runner, which hands the finalize the per-slot row map of
``expert_rows.py``; with ``bf16_routed_experts`` the BF16 layers' stacked call is the
``ops/moe/bf16_routed_experts`` entry, the same FlashInfer op on unpacked
routing.

``three_window_placement`` (``windows=3``, with ``persistent_experts`` and ``moe_prologue``)
gives every NVFP4 layer
whose stacked call is the persistent kernel a third window: the stack becomes
``[own | predecessor | second predecessor]`` (96 local experts), and on the
decode steps the prologue's routed pool kernel places (at most 8 rows) it places
over the three windows (``ring_windows.py``, ``PLACE_BALANCE3``) and records
the 96-expert window with the placement; the experts then run over the whole
stack. Every other call keeps the two-window placement over the stack's
first 64 experts, which are the two-window stack byte for byte. The two BF16
layers keep two windows.

``half_window_experts`` (``half=True`` with ``windows=1``, ``persistent``, tep4 only)
gives every NVFP4 layer the persistent kernel serves the half window: its stack is
``[own block | the predecessor's experts 32..63]`` (96 local experts, about +48 GiB
per rank over the own blocks of 60 layers), and on the decode steps the prologue's
routed kernel places (at most 8 rows) it places with the half-window flow
(``ring_windows.place_half``, ``PLACE_HALF``) and records the 96-expert window. The
other decode calls of those layers, up to ``HALF_MAX_ROWS`` rows (the 16..32-row steps
of two to four requests), take the same flow from ``ring_buddy_assign(half=True)`` and
run the whole stack too: the persistent kernel up to its row limit, TRTLLM-gen's
pre-routed call over the 96 local experts above it. Every call of the other layers
keeps the owner-only window over the own block.

Ring sizes (``ring_size``): tep8 (8 ranks, 32-expert blocks, the description
above) and tep4 (4 ranks on one tray, 64-expert blocks, ``ring_buddy_assign``
with ``ranks=4``). Any other mapping keeps the built-in path on every layer.

``windows=1`` is the owner-only window: no replication, the "stack" is the
rank's own parameters themselves (nothing is allocated or gathered), and the
placement keeps every expert on its owner (``balance=False``), so the stacked
call over ``BLOCK`` local experts reproduces plain expert parallelism. It is how
``persistent_experts`` and ``bf16_routed_experts`` are served without
``replicated_experts``, e.g. at tep4, where a second window costs about
113 GiB per rank (a 64-expert block of 62 layers). The prologue's routed kernel
reads such a layer as ``PLACE_OWNER`` (balance off, ``static_from`` the block
size), and ``early_expert_ids`` reads its stacks like any other. A third window
is refused at tep4 (it would not fit next to the KV cache), and two windows
there have no persistent-kernel build (128 experts): their NVFP4 stacked
call stays TRTLLM-gen's.
"""

from __future__ import annotations

import types
from typing import Dict, List, Optional

import torch

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ... import dispatch as _tp_agreement
from ...ops.moe.ring_buddy_assign import (
    BLOCK,
    EXPERTS,
    HALF_SPAN,
    MAX_SLOTS,
    RANKS,
    TOP_K,
    block_of,
    ring_buddy_assign,
)

#: Decode rows the stacked path serves (``install``'s ``max_rows`` default); larger
#: calls keep the built-in path.
MAX_ROWS = 16

#: The stacked path's row cap per ``(SM, tensor-parallel size)`` where it is not
#: ``MAX_ROWS`` (``max_rows_for``). sm103 tep8 also balances the decode steps of three
#: and four requests (17..32 rows at K = 7): 32 rows of ``TOP_K`` slots are the 256 slots
#: of one ``ring_buddy_assign`` program (``MAX_SLOTS``); measured at 6.4 / 8.3 us per layer
#: net faster than the built-in path at 24 / 32 rows.
MAX_ROWS_BY_SM_TP = {(103, 8): 32}

#: The widest window the persistent kernel's batch builds serve
#: (``ops/moe/persistent_experts`` asserts ``num_local <= 64`` for one). ``WINDOWS`` also holds
#: 96, the three-window placement's: its calls keep the 8-row and wide builds, as they did
#: before the batch builds existed.
BATCH_MAX_EXPERTS = 64

#: The window the batch builds' in-kernel placement serves: the tep8 ring's two-window 64
#: (``ops/moe/_persistent_experts/kernel`` compiles it only for ``num_local == 64``).
IN_KERNEL_PLACEMENT_EXPERTS = 64


def takes_batch_builds(window: int) -> bool:
    """Whether the persistent kernel's batch builds serve a ``window``-expert placement.

    One rule for all three sites that need it -- what ``prepare`` compiles, how many rows the
    stacked call serves, and whether it asks for a batch build -- so they cannot drift apart.
    Preparing a 96-expert window with batch rows fails the builds' own ``num_local <= 64``
    assert at install, which is how they were first found to disagree.
    """
    return window <= BATCH_MAX_EXPERTS


#: ``tep8_ring_buddy_wide``'s ``MAX_ROWS``, set by ``configure``: it replaces the tep8 entry of
#: ``MAX_ROWS_BY_SM_TP`` so the stacked path takes wider decode calls. None (the default) leaves
#: the target's own cap, and the ring is 8 ranks or it does not apply.
_TEP8_MAX_ROWS: Optional[int] = None


def max_rows_for(compute_capability, tp_size) -> int:
    """The stacked path's row cap on a device of ``compute_capability`` at ``tp_size``.

    ``tep8_ring_buddy_wide``'s ``MAX_ROWS`` at tep8 when it is set, else ``MAX_ROWS_BY_SM_TP``'s
    entry, else ``MAX_ROWS`` (also without a capability). Both callers read it here -- ``install``
    for the layers' caps and ``weights`` for the batch builds -- so one override reaches both.
    """
    if _TEP8_MAX_ROWS is not None and tp_size == 8:
        return _TEP8_MAX_ROWS
    sm = _tp_agreement.sm_of(compute_capability)
    return MAX_ROWS_BY_SM_TP.get((sm, tp_size), MAX_ROWS)


#: Decode rows the stacked path serves on a half-window layer (``half_window_experts``):
#: up to four requests' verify steps, the slots of one ``ring_buddy_assign`` program.
HALF_MAX_ROWS = 32

#: The static placement: local experts from this index up run on the successor at
#: every step (4 of 32 per block, about the share the balanced placement
#: moves in a verify step).
STATIC_FROM = 28

#: Per-expert tensors the expert call reads, per expert format, in the order
#: the stack is built. Every one is indexed by local expert first. ``nvfp4`` is
#: the native TRTLLM-gen NVFP4 leaf (60 layers); ``bf16`` the FlashInfer
#: TRTLLM-gen BF16 leaf of the two quantization-excluded layers (0 and 61).
STACKED = {
    "nvfp4": (
        "w3_w1_weight",
        "w3_w1_weight_scale",
        "w2_weight",
        "w2_weight_scale",
        "fc31_scale_c",
        "fc31_alpha",
        "fc2_alpha",
    ),
    "bf16": ("w3_w1_weight", "w2_weight"),
}

#: Experts gathered per collective while replicating; bounds the transient
#: buffer to RANKS * this many experts of the largest tensor.
_GATHER_CHUNK = 4


def _version(tensor: torch.Tensor) -> Optional[int]:
    """``tensor._version``, or None for an inference tensor, which has no counter.

    Serving runs the forward under ``torch.inference_mode()``, where reading
    ``_version`` raises.
    """
    return None if tensor.is_inference() else tensor._version


def _backend(moe):
    return getattr(moe.experts, "backend", moe.experts)


def expert_format(moe) -> Optional[str]:
    """``nvfp4`` or ``bf16`` for the two expert leaves this feature drives, else None."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.flashinfer_bf16 import (
        FlashinferTrtllmGenBf16Impl,
    )

    backend = _backend(moe)
    if getattr(backend, "has_nvfp4", False) and isinstance(
        getattr(backend, "op_backend", None), TRTLLMOpBackend
    ):
        return "nvfp4"
    if type(backend) is FlashinferTrtllmGenBf16Impl:
        return "bf16"
    return None


def ring_size(mapping) -> Optional[int]:
    """The expert-parallel ring ``mapping`` gives this feature: 8 ranks (tep8), 4 (tep4), else None."""
    tp = getattr(mapping, "tp_size", None)
    if tp == 8:
        return 8
    elif tp == 4:
        return 4
    return None


def eligible(moe, mapping: Mapping) -> Optional[str]:
    """Why this MoE layer cannot take the stacked path, or None if it can."""
    backend = _backend(moe)
    if getattr(moe.experts, "comm", None) is not None:
        return "the layer has an external MoE communication strategy"
    kind = expert_format(moe)
    if kind is None:
        return "the experts are neither the native NVFP4 nor the FlashInfer BF16 TRTLLM-gen leaf"
    ranks = ring_size(mapping)
    if ranks is None or mapping.moe_ep_size != ranks or mapping.moe_tp_size != 1:
        return "the mapping is neither tp8/ep8 nor tp4/ep4"
    if mapping.moe_ep_rank != mapping.tp_rank:
        return "the expert-parallel rank differs from the tensor-parallel rank"
    block = block_of(ranks)
    if (
        getattr(backend, "num_slots", None) != EXPERTS
        or getattr(backend, "expert_size_per_partition", None) != block
        or getattr(backend, "slot_start", None) != block * mapping.moe_ep_rank
    ):
        return f"the local expert window is not this rank's contiguous {block}-expert block"
    if getattr(backend, "bias", False):
        return "the experts carry biases"
    for name in ("act_alpha", "act_beta", "act_clamp", "fc31_act_scale"):
        if getattr(backend, name, None) is not None:
            return f"the experts carry {name}"
    for name in STACKED[kind]:
        tensor = getattr(backend, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != block:
            return f"{name} is not a [{block}, ...] per-expert tensor"
    return None


def _predecessor_block(local: torch.Tensor, mapping: Mapping) -> torch.Tensor:
    """The ring predecessor's copy of ``local`` ([BLOCK, ...]), gathered over the TP group."""
    return _ring_block(local, mapping, 1)


def _window_block(local: torch.Tensor, mapping: Mapping, back: int) -> torch.Tensor:
    """The block of stack window ``back``: the predecessor's, or (three windows) the one before it."""
    if back == 1:
        return _predecessor_block(local, mapping)
    return _ring_block(local, mapping, back)


def _ring_block(local: torch.Tensor, mapping: Mapping, back: int) -> torch.Tensor:
    """The copy of ``local`` ([BLOCK, ...]) held by the rank ``back`` places back on the ring."""
    source = local.detach().contiguous()
    flat = source.reshape(source.shape[0], -1).view(torch.uint8)
    out = torch.empty_like(flat)
    predecessor = (mapping.tp_rank - back) % mapping.tp_size
    for begin in range(0, flat.shape[0], _GATHER_CHUNK):
        part = flat[begin : begin + _GATHER_CHUNK].contiguous()
        rows = part.shape[0]
        gathered = allgather(part, mapping, dim=0)
        out[begin : begin + rows].copy_(gathered[predecessor * rows : (predecessor + 1) * rows])
        del gathered
    return out.view(source.dtype).reshape(source.shape)


class _State:
    """What one layer's stacked path holds: the stacks, the rank, the step's placement."""

    def __init__(
        self,
        kind: str,
        stacks: Dict[str, torch.Tensor],
        rank: int,
        balance: bool,
        static_from: Optional[int] = None,
        windows: int = 2,
        ranks: int = RANKS,
        half: bool = False,
        max_rows: int = MAX_ROWS,
    ):
        self.kind = kind
        self.stacks = stacks
        self.rank = rank
        # The ring (8 ranks of 32-expert blocks, or 4 of 64) and its block size.
        self.ranks = ranks
        self.block = block_of(ranks)
        # Ring windows the stacks hold (1: the owner-only window, 2, or 3 with
        # three_window_placement). The stacked call reads the stack prefix of the
        # placement's window: one or two blocks (``default_window``), or the whole
        # stack for the routed kernel's three-window placement.
        self.windows = windows
        self.default_window = min(windows, 2) * self.block
        self.by_window = {
            window * self.block: {
                name: stack[: window * self.block] for name, stack in stacks.items()
            }
            for window in range(2, windows)
        }
        self.by_window[windows * self.block] = stacks
        # The half window (``half_window_experts``): the owner-only placement over the
        # stack's own block, and the prologue's half-window placement over all of it.
        # Every other layer serves calls of up to ``max_rows`` rows (``install``'s).
        self.half = half
        self.max_rows = HALF_MAX_ROWS if half else max_rows
        if half:
            self.default_window = self.block
            self.by_window = {
                self.block: {name: stack[: self.block] for name, stack in stacks.items()},
                stacks["w3_w1_weight"].shape[0]: stacks,
            }
        self.active = True
        self.balance = balance
        # With balance off: local experts static_from.. of every block run on
        # the successor at every step (the static placement); the block size = none.
        self.static_from = self.block if static_from is None else static_from
        # (router_logits, its version, local_ids, weights_bf16, stats) of the
        # current layer call, plus its window (local experts) when it is not
        # the two-window 64 (the three-window placement in the prologue's routed kernel).
        self.placement = None
        # The same call's global expert ids (0..255). Only the batch builds' in-kernel
        # placement reads them; set and cleared with ``placement``, so the two never disagree.
        self.global_ids = None
        # The per-rank loads (before, after) of the last placement; kept across
        # memo clears for the shadow validation mode.
        self.last_stats = None

    def current(self, router_logits: torch.Tensor):
        """The placement computed from ``router_logits`` as it is now, or None."""
        placement = self.placement
        if (
            placement is None
            or placement[0] is not router_logits
            or placement[1] != _version(router_logits)
        ):
            return None
        return placement


def _replicate(
    moe, kind: str, mapping: Mapping, windows: int = 2, half: bool = False
) -> Dict[str, torch.Tensor]:
    """The layer's stacks of ``windows`` ring windows; ``windows=1`` is the parameters themselves.

    ``half``: the half window, ``[own block | the predecessor's upper half]``.
    """
    backend = _backend(moe)
    # The built-in loader finalizes the leaf's layout eagerly; finishing any
    # transform still pending here keeps the copies in the layout the kernels
    # read and stops a later transform from re-binding the parameters away
    # from the stack. Idempotent for both leaves.
    backend.quant_method.transform_weights(backend)
    stacks = {}
    for name in STACKED[kind]:
        param = getattr(backend, name)
        own = param.data
        if windows == 1 and not half:
            # The owner-only window: the stacked call reads the parameter itself.
            stacks[name] = own
            continue
        block = own.shape[0]
        if half:
            upper = block // 2
            stack = torch.empty((block + upper, *own.shape[1:]), dtype=own.dtype, device=own.device)
            stack[:block].copy_(own)
            predecessor = _predecessor_block(own[upper:], mapping)
            stack[block:].copy_(predecessor)
            del predecessor
            param.data = stack[:block]
            stacks[name] = stack
            continue
        stack = torch.empty((windows * block, *own.shape[1:]), dtype=own.dtype, device=own.device)
        stack[:block].copy_(own)
        for back in range(1, windows):
            predecessor = _window_block(own, mapping, back)
            stack[back * block : (back + 1) * block].copy_(predecessor)
            del predecessor
        # The built-in path keeps reading the parameter; make it the stack's
        # first half so the own block is stored once.
        param.data = stack[:block]
        stacks[name] = stack
    return stacks


def _refresh(moe, state: _State, mapping: Mapping) -> None:
    """Bring an installed layer's stacks up to date with its parameters, in place.

    A later weight load writes the built-in parameters, which are views of
    the stacks' first halves -- or, if the loader replaced a parameter's
    storage, a new tensor. Either way the own half is made the parameter's
    current bytes again and the predecessor half is gathered anew into the
    existing storage; the hooks are already in place and stay as they are.
    """
    backend = _backend(moe)
    backend.quant_method.transform_weights(backend)
    block = state.block
    for name, stack in state.stacks.items():
        param = getattr(backend, name)
        own = stack[:block]
        if param.data.shape != own.shape or param.data.dtype != own.dtype:
            raise RuntimeError(
                f"replicated_experts: {name} is now {tuple(param.data.shape)} {param.data.dtype}, "
                f"its stack half {tuple(own.shape)} {own.dtype}; cannot refresh in place"
            )
        if param.data.data_ptr() != own.data_ptr():
            own.copy_(param.data)
            param.data = own
        if state.half:
            stack[block:].copy_(_predecessor_block(own[block // 2 :], mapping))
            continue
        for back in range(1, state.windows):
            stack[back * block : (back + 1) * block].copy_(_window_block(own, mapping, back))
    state.placement = None


def _patch_routing(moe, state: _State) -> None:
    routing = moe.experts.routing_method
    builtin_apply = routing.apply
    builtin_clear = routing.clear_routing_memo

    def apply(self, router_logits, input_ids=None):
        selected, weights = builtin_apply(router_logits, input_ids)
        if (
            state.active
            and state.current(router_logits) is None
            and router_logits.is_cuda
            and 0 < router_logits.shape[0] <= state.max_rows
            and selected.dtype == torch.int32
            and weights.dtype == torch.float32
            and selected.is_contiguous()
            and weights.is_contiguous()
            and not torch.compiler.is_compiling()
        ):
            local_ids, weights_bf16, stats = ring_buddy_assign(
                selected,
                weights,
                state.rank,
                state.balance,
                state.static_from,
                ranks=state.ranks,
                half=state.half,
            )
            state.placement = (
                router_logits,
                _version(router_logits),
                local_ids,
                weights_bf16,
                stats,
            )
            if state.half:
                state.placement += (HALF_SPAN,)
            state.last_stats = stats
            state.global_ids = selected
        return selected, weights

    def clear_routing_memo(self):
        state.placement = None
        state.global_ids = None
        builtin_clear()

    routing.apply = types.MethodType(apply, routing)
    routing.clear_routing_memo = types.MethodType(clear_routing_memo, routing)


def _nvfp4_call(
    backend,
    routing,
    persistent: bool = False,
    stacks=None,
    persistent_rows=8,
    persistent_early_start: bool = False,
    persistent_place_rank: int = -1,
    state=None,
):
    """The stacked call for the native NVFP4 leaf, as its ``run_moe`` makes the built-in one.

    ``persistent`` (``persistent_experts``): calls of up to ``persistent_experts``'s
    row limit go to the persistent expert kernel instead, which returns the
    same unfinalized triple with the per-slot row map of
    ``expert_rows.py``. Only for the SwiGLU expert of the kernel's
    shapes and of the windows it is compiled for (``persistent_experts.WINDOWS``:
    a tep8 rank's 32-expert own block, the 64-expert two-window stack and the
    96-expert three-window one; a tep4 rank's 64-expert own block); any other
    leaf or window keeps TRTLLM-gen's call. The kernel is compiled and its
    workspace created here, from ``stacks``, so no graph capture ever does
    either.

    The call's window is the leading dim of the stacks it is given; the
    ``local_expert_offset=0`` TRTLLM-gen call runs over exactly that window.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
        fp4_block_scale_moe_runner,
    )
    from tensorrt_llm._torch.autotuner import AutoTuner

    from ...ops.moe import decode_expert_tactics as _decode_tactics

    intermediate = backend.w3_w1_weight.shape[-2] // 2
    tune_max_num_tokens = backend.max_num_tokens
    act_type = _act_type(backend)
    # Past two blocks only the three-window stack, which only the persistent kernel serves.
    two_windows = 2 * backend.expert_size_per_partition
    persistent_op = None
    persistent_windows = frozenset()
    if persistent:
        from ...ops.moe import persistent_experts as _persistent

        if persistent_eligible(backend):
            persistent_op = _persistent
            persistent_windows = frozenset(
                window for window in (stacks or {}) if window in _persistent.WINDOWS
            )
            # One compiled kernel per window the layer's placements address. Only a window
            # the batch builds serve is prepared with them; a wider one (96) would fail their
            # ``num_local <= 64`` assert at install.
            for window in sorted(persistent_windows):
                window_stacks = stacks[window]
                batch_here = persistent_rows if takes_batch_builds(window) else 0
                _persistent.persistent_experts.prepare(
                    window_stacks["w3_w1_weight"],
                    window_stacks["w3_w1_weight_scale"].view(torch.float8_e4m3fn),
                    window_stacks["w2_weight"],
                    window_stacks["w2_weight_scale"].view(torch.float8_e4m3fn),
                    batch_rows=batch_here,
                    early_start=persistent_early_start and batch_here > 0,
                    place_rank=(
                        persistent_place_rank
                        if persistent_place_rank >= 0 and window == IN_KERNEL_PLACEMENT_EXPERTS
                        else None
                    ),
                )

    def call(stacks, x, local_ids, weights_bf16):
        if not isinstance(x, Fp4QuantizedTensor) or x.is_sf_swizzled:
            return None
        x_sf = x.scaling_factor.reshape(-1).view(torch.float8_e4m3fn)
        window = stacks["w3_w1_weight"].shape[0]
        rows = x.fp4_tensor.shape[0]
        served = False
        if persistent_op is not None and window in persistent_windows:
            # The batch builds reach ``persistent_rows`` on a window of at most 64 experts; a
            # wider placement keeps the 8-row bound it had before them.
            cap = persistent_rows if takes_batch_builds(window) else persistent_op.MAX_ROWS
            served = rows <= cap
        if window > two_windows and not served:
            # The three-window placement runs only where the persistent kernel serves the
            # call (the prologue's steps of at most that window's bound); every rank shares
            # this shape.
            raise RuntimeError(
                f"three_window_placement: a {window}-expert placement reached a call "
                f"of {rows} rows the persistent expert kernel does not serve"
            )
        if served:
            ids, place = local_ids, None
            if (
                persistent_place_rank >= 0
                and window == IN_KERNEL_PLACEMENT_EXPERTS
                and rows > persistent_op.MAX_ROWS
                and state is not None
                and state.global_ids is not None
            ):
                # A batch build on the 64-expert window: hand it the step's global ids and let
                # every CTA derive this rank's balanced placement. ``local_ids`` are the assign
                # program's window ids, which that build does not read.
                ids, place = state.global_ids, persistent_place_rank
            return persistent_op.persistent_experts(
                x.fp4_tensor,
                x_sf,
                ids,
                weights_bf16,
                stacks["w3_w1_weight"],
                stacks["w3_w1_weight_scale"].view(torch.float8_e4m3fn),
                stacks["w2_weight"],
                stacks["w2_weight_scale"].view(torch.float8_e4m3fn),
                stacks["fc31_scale_c"],
                stacks["fc31_alpha"],
                stacks["fc2_alpha"],
                batch=takes_batch_builds(window),
                early_start=persistent_early_start,
                place_rank=place,
            )
        tactic = None
        if not AutoTuner.get().is_tuning_mode:
            tactic = _decode_tactics.fixed_tactic(
                x.fp4_tensor.device,
                act_type,
                TOP_K,
                2 * x.fp4_tensor.shape[1],
                intermediate,
                window,
                x.fp4_tensor.shape[0],
            )
        if tactic is not None:
            return _decode_tactics.run_fixed(
                tactic,
                [
                    None,
                    None,
                    x.fp4_tensor,
                    x_sf,
                    stacks["w3_w1_weight"],
                    stacks["w3_w1_weight_scale"].view(torch.float8_e4m3fn),
                    None,
                    None,
                    None,
                    None,
                    stacks["w2_weight"],
                    stacks["w2_weight_scale"].view(torch.float8_e4m3fn),
                    None,
                    stacks["fc31_scale_c"],
                    stacks["fc31_alpha"],
                    stacks["fc2_alpha"],
                    weights_bf16,
                    local_ids,
                ],
                EXPERTS,
                TOP_K,
                intermediate,
                window,
                routing.routing_method_type,
                act_type,
                tune_max_num_tokens,
            )
        return fp4_block_scale_moe_runner(
            None,
            None,
            x.fp4_tensor,
            x_sf,
            stacks["w3_w1_weight"],
            stacks["w3_w1_weight_scale"].view(torch.float8_e4m3fn),
            None,
            None,
            None,
            None,
            stacks["w2_weight"],
            stacks["w2_weight_scale"].view(torch.float8_e4m3fn),
            None,
            stacks["fc31_scale_c"],
            stacks["fc31_alpha"],
            stacks["fc2_alpha"],
            EXPERTS,
            TOP_K,
            # n_group, topk_group, routed_scaling_factor: inert for pre-routed
            # calls; the built-in backend passes None for this routing method.
            None,
            None,
            intermediate,
            0,
            window,
            None,
            routing.routing_method_type,
            False,
            act_type,
            topk_weights=weights_bf16,
            topk_ids=local_ids,
            output=None,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=False,
        )

    return call


def _bf16_call(backend, routing, unpacked: bool = False):
    """The stacked call for the FlashInfer BF16 leaf, as its ``run_moe`` makes the built-in one.

    ``unpacked`` (``bf16_routed_experts``): the same FlashInfer op on the ids and the
    BF16 weights as two tensors (``ops/moe/bf16_routed_experts``) instead of
    the leaf's packed ``(id << 16) | weight`` word, which takes three
    elementwise launches per call; every routed row is bitwise the packed
    call's.

    The call's window is the leading dim of the stacks it is given (one block for
    the owner-only window, two for the ring-buddy stack).
    """
    op_backend = backend.op_backend
    intermediate = backend.intermediate_size_per_partition
    act_type = _act_type(backend)
    use_shuffled_weight = getattr(backend.quant_method, "use_shuffled_weight", False)
    weight_layout = getattr(backend.quant_method, "weight_layout", 0)

    if unpacked:
        from ...ops.moe.bf16_routed_experts import bf16_routed_experts

        routing_method_type = int(op_backend.cvt_routing_method_type(routing.routing_method_type))
        activation_type = int(op_backend.cvt_activation_type(act_type))

        def call_unpacked(stacks, x, local_ids, weights_bf16):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
                return None
            return bf16_routed_experts(
                x,
                stacks["w3_w1_weight"],
                stacks["w2_weight"],
                local_ids,
                weights_bf16,
                intermediate,
                0,
                stacks["w3_w1_weight"].shape[0],
                routing_method_type,
                activation_type,
                use_shuffled_weight,
                weight_layout,
                EXPERTS,
                TOP_K,
            )

        return call_unpacked

    def call(stacks, x, local_ids, weights_bf16):
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
            return None
        return op_backend.run_bf16_moe(
            router_logits=None,
            routing_bias=None,
            hidden_states=x,
            gemm1_weights=stacks["w3_w1_weight"],
            gemm2_weights=stacks["w2_weight"],
            num_experts=EXPERTS,
            top_k=TOP_K,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate,
            local_expert_offset=0,
            local_num_experts=stacks["w3_w1_weight"].shape[0],
            routed_scaling_factor=None,
            routing_method_type=routing.routing_method_type,
            topk_weights=weights_bf16,
            topk_ids=local_ids,
            gated_act_type=act_type,
            output=None,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            do_finalize=False,
        )

    return call


def _patch_experts(
    moe,
    state: _State,
    persistent: bool = False,
    bf16_unpacked: bool = False,
    persistent_rows: int = 8,
    persistent_early_start: bool = False,
    persistent_in_kernel_placement: bool = False,
) -> None:
    experts = moe.experts
    backend = _backend(moe)
    # The rank whose balanced placement the batch builds derive. Only the tep8 ring: the
    # kernel's placement is 8 ranks of 32-expert blocks.
    place_rank = state.rank if persistent_in_kernel_placement and state.ranks == RANKS else -1
    builtin_forward = experts.forward
    routing = experts.routing_method
    if state.kind == "nvfp4" and persistent:
        stacked_call = _nvfp4_call(
            backend,
            routing,
            persistent=True,
            stacks=state.by_window,
            persistent_rows=persistent_rows,
            persistent_early_start=persistent_early_start,
            persistent_place_rank=place_rank,
            state=state,
        )
    elif state.kind == "nvfp4":
        stacked_call = _nvfp4_call(backend, routing)
    elif bf16_unpacked:
        stacked_call = _bf16_call(backend, routing, unpacked=True)
    else:
        stacked_call = _bf16_call(backend, routing)
    by_window = state.by_window
    default_window = state.default_window

    def forward(self, x, router_logits, *args, do_finalize=True, **kwargs):
        if not state.active or do_finalize or args or not 0 < x.shape[0] <= state.max_rows:
            return builtin_forward(x, router_logits, *args, do_finalize=do_finalize, **kwargs)
        routing.apply(router_logits)
        placement = state.current(router_logits)
        if placement is None:
            return builtin_forward(x, router_logits, *args, do_finalize=do_finalize, **kwargs)
        local_ids, weights_bf16 = placement[2], placement[3]
        window = placement[5] if len(placement) > 5 else default_window
        outputs = stacked_call(by_window[window], x, local_ids, weights_bf16)
        if outputs is None:
            return builtin_forward(x, router_logits, *args, do_finalize=do_finalize, **kwargs)
        return outputs

    experts.forward = types.MethodType(forward, experts)


def _act_type(backend) -> int:
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.kernel_inputs import to_trtllm_gen_act_type

    return int(to_trtllm_gen_act_type(backend.activation_type))


def persistent_eligible(backend) -> bool:
    """Whether the persistent expert kernel computes this NVFP4 leaf's experts (its shapes, SwiGLU)."""
    from ...ops.moe import persistent_experts as _persistent

    return (
        _is_plain_swiglu(backend)
        and backend.w3_w1_weight.shape[-2] // 2 == _persistent.INTER
        and backend.w3_w1_weight.shape[-1] * 2 == _persistent.HIDDEN
    )


def _layer_windows(moe, kind: str, windows: int, persistent: bool) -> int:
    """The ring windows ``install`` stacks on this layer: one on every layer for the
    owner-only window; else ``windows`` on an NVFP4 layer the persistent kernel
    serves, and two elsewhere (the BF16 layers, and any NVFP4 leaf the persistent
    kernel does not compute)."""
    if windows == 1:
        return 1
    if kind == "nvfp4" and persistent and persistent_eligible(_backend(moe)):
        return windows
    return 2


def _layer_half(moe, kind: str, half: bool, persistent: bool) -> bool:
    """Whether ``install`` gives this layer the half window: an NVFP4 layer the persistent
    kernel serves, when the install asks for it."""
    return half and kind == "nvfp4" and persistent and persistent_eligible(_backend(moe))


def _is_plain_swiglu(backend) -> bool:
    """SwiGLU without per-expert bias or clamp constants (what the persistent kernel computes)."""
    from tensorrt_llm._torch.utils import ActivationType

    return int(backend.activation_type) == int(ActivationType.Swiglu)


def _wide_layer(kind: str, windows: int, half: bool, balance: bool, ranks: Optional[int]) -> bool:
    """Whether ``tep8_wide_balance`` extends a layer: an NVFP4 layer's balanced two-window
    placement on the tep8 ring (the BF16 layers, the owner-only window and the half window
    keep their caps)."""
    return kind == "nvfp4" and windows >= 2 and not half and bool(balance) and ranks == 8


def _wide_rows_here(ranks: Optional[int]) -> Optional[int]:
    """``tep8_wide_balance``'s cap where it applies (the tep8 ring of this sm103 target), else
    None, with a warning when it is set elsewhere.

    Raises:
        ValueError: for a malformed value, on every rank alike.
    """
    from ...ops.moe.ring_buddy_assign import wide_balance_rows

    rows = wide_balance_rows()
    if rows is None:
        return None
    if ranks != 8:
        logger.warning(
            f"[musespark_open] tep8_wide_balance MAX_ROWS={rows} has no effect on a ring of "
            f"{ranks} ranks: it extends the tep8 balanced placement only"
        )
        return None
    return rows


def _apply_wide(state: "_State", wide_rows: Optional[int], widened: List[int]) -> None:
    """``tep8_wide_balance`` on one installed layer: its cap and the wide program, where
    ``_wide_layer`` admits it; any other layer, or the switch unset, is left as it is.

    Raising ``state.max_rows`` is the whole mechanism: the routing and expert patches gate on
    ``0 < rows <= state.max_rows``, so a layer admitted here serves the wider calls.
    """
    state.wide = False
    state.wide_base = state.max_rows
    if wide_rows is None or not _wide_layer(
        state.kind, state.windows, state.half, state.balance, state.ranks
    ):
        return
    state.wide = True
    state.max_rows = max(state.max_rows, wide_rows)
    widened.append(state.max_rows)


def configure(
    *, wide_balance_rows: Optional[int] = None, tep8_max_rows: Optional[int] = None
) -> None:
    """The one entry ``lever_settings`` calls for this module.

    ``wide_balance_rows`` is ``tep8_wide_balance``'s ``MAX_ROWS``; the band itself belongs to
    ``ring_buddy_assign``, so this forwards rather than storing a second copy.
    ``tep8_max_rows`` is ``tep8_ring_buddy_wide``'s, read back by ``max_rows_for``. None (the
    default) for either leaves the target's own caps in place.
    """
    global _TEP8_MAX_ROWS

    _TEP8_MAX_ROWS = None if tep8_max_rows is None else int(tep8_max_rows)
    _configure_wide_balance(wide_balance_rows)


def _configure_wide_balance(rows: Optional[int]) -> None:
    """Hand ``tep8_wide_balance``'s ``MAX_ROWS`` to the assign module, which owns the band.

    One ``configure`` reaches this module (``lever_settings``), and the band belongs to
    ``ring_buddy_assign``; this keeps both in step without a second entry in the settings
    table. ``_wide_rows_here`` reads it back.
    """
    from ...ops.moe.ring_buddy_assign import configure as _assign_configure

    _assign_configure(wide_balance_rows=rows)


def install(
    text_model,
    mapping: Mapping,
    balance: bool = True,
    static_from: Optional[int] = None,
    persistent: bool = False,
    bf16_unpacked: bool = False,
    persistent_rows: int = 8,
    persistent_early_start: bool = False,
    persistent_in_kernel_placement: bool = False,
    windows: int = 2,
    half: bool = False,
    max_rows: int = MAX_ROWS,
) -> List[int]:
    """Replicate and patch every eligible MoE layer of ``text_model``; returns their indices.

    Collective over the TP group (one all-gather sequence per layer), so every
    rank must call it at the same point, after the weights are final and
    before any CUDA graph is captured. The ring is the mapping's (``ring_size``:
    tep8 with 32-expert blocks, tep4 with 64-expert blocks; any other mapping
    skips every layer). ``balance=False`` keeps every expert on
    its owner: the stacked path then reproduces the built-in partials bitwise;
    with it, ``static_from`` below the block size instead runs local experts
    ``static_from..`` of every block on the successor at every step (a
    routing-independent regrouping with the same kind of rounding difference as
    ``balance``). ``static_from`` defaults to the block size and is the block
    size whenever ``balance`` is on.
    ``windows=1`` is the owner-only window (needs ``balance=False`` and no static
    move): nothing is replicated or gathered, the stacked calls run over the
    rank's own block with owner-only ids; it is how ``persistent`` and
    ``bf16_unpacked`` are served without replication.
    ``persistent=True``
    (``persistent_experts``) serves the NVFP4 layers' stacked calls of up to 8 rows with the
    persistent expert kernel (``ops/moe/persistent_experts.py``).
    ``bf16_unpacked=True`` (``bf16_routed_experts``) gives the two BF16 layers'
    stacked calls unpacked routing (``ops/moe/bf16_routed_experts.py``):
    bitwise the packed call, without its three packing launches.
    ``windows=3`` (``three_window_placement``; needs ``persistent`` and ``balance``, tep8
    only) stacks a third window on every NVFP4 layer the persistent kernel serves;
    the prologue's routed pool kernel then places over three windows (see the module
    docstring). The other layers keep two.
    ``half=True`` (``half_window_experts``; needs ``windows=1``, ``persistent`` and the
    owner-only placement, tep4 only) stacks the predecessor's upper half on every NVFP4
    layer the persistent kernel serves (see the module docstring); the other layers
    keep the owner-only window.
    ``max_rows`` (``MAX_ROWS``; ``max_rows_for`` of the device and the TP size) is the
    largest decode call the stacked path takes on every layer but a half-window one
    (``HALF_MAX_ROWS``); larger calls keep the built-in path. At most
    ``MAX_SLOTS // TOP_K`` (32), the rows of one ``ring_buddy_assign`` program.

    A layer installed by an earlier call is refreshed in place (``_refresh``):
    its stacks follow the newly loaded parameters, its hooks are not wrapped
    again (so it keeps the expert kernel it was installed with) and no second
    stack is allocated. Every rank takes the same branch,
    since the installed set is a function of the model and the mapping.

    The rank-local refusals are agreed over the TP group (``tp_agreement``):
    the arguments, a refresh that would change a layer's window count and, with
    ``bf16_unpacked``, the FlashInfer op it calls, before the first all-gather;
    the expert kernels' compiles (the persistent kernel, one per window) between
    the all-gathers are held, every later layer still takes its all-gather, and
    the outcome is agreed after the last one. A refusal one rank sees therefore
    raises on every rank instead of leaving its peers in a collective.

    Raises:
        ValueError: on every rank alike, for any of those refusals on some rank.
    """
    ranks = ring_size(mapping)
    refusal = _install_refusal(
        text_model, windows, persistent, balance, bf16_unpacked, ranks, static_from, half, max_rows
    )
    refusal = _tp_agreement.agreed_refusal(mapping, refusal)
    if refusal is not None:
        raise ValueError(refusal)
    # A mapping outside the served rings skips every layer (``eligible``); the
    # block only names the static placement in the log line then.
    block = block_of(ranks) if ranks is not None else BLOCK
    if static_from is None or balance:
        static_from = block
    failed: Optional[str] = None
    installed, refreshed = [], []
    # tep8_wide_balance: the cap that applies on this ring, from the band ``configure`` handed
    # to the assign module. None everywhere the lever is off, leaving every cap as it was.
    _wide_rows = _wide_rows_here(ranks)
    # Distinct from ``widened`` below, which collects the indices of three-window layers.
    _wide_widened: List[int] = []
    widened = []
    halved = []
    skipped = {}
    kinds: Dict[str, int] = {}
    added = 0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0]
    reserved_before = torch.cuda.memory_reserved()
    allocated_before = torch.cuda.memory_allocated()
    for index, layer in enumerate(text_model.layers):
        moe = layer.feed_forward
        state = getattr(moe, "_replicated_experts", None)
        if state is not None:
            _refresh(moe, state, mapping)
            state.balance = balance
            state.static_from = static_from
            state.max_rows = HALF_MAX_ROWS if state.half else max_rows
            refreshed.append(index)
            continue
        reason = eligible(moe, mapping)
        if reason is not None:
            skipped[index] = reason
            continue
        kind = expert_format(moe)
        kinds[kind] = kinds.get(kind, 0) + 1
        layer_windows = _layer_windows(moe, kind, windows, persistent)
        layer_half = _layer_half(moe, kind, half, persistent)
        if layer_windows == 3:
            widened.append(index)
        if layer_half:
            halved.append(index)
        stacks = _replicate(moe, kind, mapping, layer_windows, layer_half)
        added += sum(t[block:].numel() * t.element_size() for t in stacks.values())
        state = _State(
            kind,
            stacks,
            mapping.tp_rank,
            balance,
            static_from,
            layer_windows,
            ranks,
            layer_half,
            max_rows=max_rows,
        )
        _apply_wide(state, _wide_rows, _wide_widened)
        moe._replicated_experts = state
        if failed is None:
            _patch_routing(moe, state)
            try:
                _patch_experts(
                    moe,
                    state,
                    persistent,
                    bf16_unpacked,
                    persistent_rows,
                    persistent_early_start,
                    persistent_in_kernel_placement,
                )
            except Exception as error:  # noqa: BLE001 - every failure must reach the agreement
                failed = f"layer {index}: {error!r}"
        installed.append(index)
    failed = _tp_agreement.agreed_refusal(
        mapping,
        None
        if failed is None
        else f"[musespark_open] replicated_experts could not install: {failed}",
    )
    if failed is not None:
        raise ValueError(failed)
    torch.cuda.synchronize()
    # The all-gather's transient chunks are freed but still cached; hand them
    # back so the numbers below (and the KV-cache sizing that follows) see
    # what this fast path keeps.
    torch.cuda.empty_cache()
    device_added = free_before - torch.cuda.mem_get_info()[0]
    allocated_added = torch.cuda.memory_allocated() - allocated_before
    outside_torch = device_added - (torch.cuda.memory_reserved() - reserved_before)
    if balance:
        placement = "balanced"
    elif static_from < block:
        placement = f"static (local experts {static_from}..{block - 1} on the successor)"
    else:
        placement = "owner-only"
    if windows == 1 and half:
        what = (
            f"owner-only {block}-expert window with the predecessor's upper half on "
            f"{len(halved)} layers (half_window_experts)"
        )
    elif windows == 1:
        what = f"owner-only {block}-expert window (no replication)"
    else:
        what = "ring-buddy experts"
    logger.info(
        f"[musespark_open] replicated_experts: tep{ranks} {what} on {len(installed)} MoE layers "
        f"{'(persistent NVFP4 expert kernel) ' if persistent else ''}"
        f"{'(unpacked BF16 routing) ' if bf16_unpacked else ''}"
        f"{f'(three ring windows on {len(widened)} layers) ' if windows == 3 else ''}"
        f"({placement} placement, ring-buddy max_rows={max_rows}"
        f"{f' ({HALF_MAX_ROWS} on the half-window layers)' if halved else ''}"
        f"{f' ({_wide_rows} on {len(_wide_widened)} NVFP4 layers, tep8_wide_balance)' if _wide_widened else ''}, "
        f"{dict(sorted(kinds.items()))}), refreshed in place on {len(refreshed)}; "
        f"per rank: replicated experts +{added / 2**30:.2f} GiB, torch allocated "
        f"+{allocated_added / 2**30:.2f} GiB, device +{device_added / 2**30:.2f} GiB, of "
        f"which +{outside_torch / 2**30:.2f} GiB outside torch (the all-gather's "
        f"communicator buffers); built-in path kept for layers {sorted(skipped)}: "
        f"{sorted(set(skipped.values()))}"
    )
    return sorted(installed + refreshed)


def _install_refusal(
    text_model,
    windows: int,
    persistent: bool,
    balance: bool,
    bf16_unpacked: bool,
    ranks: Optional[int] = RANKS,
    static_from: Optional[int] = None,
    half: bool = False,
    max_rows: int = MAX_ROWS,
) -> Optional[str]:
    """``install``'s refusals known before its first all-gather, or None."""
    if not 0 < max_rows <= MAX_SLOTS // TOP_K:
        return (
            f"the stacked path's calls are placed by one ring_buddy_assign program of at most "
            f"{MAX_SLOTS} slots, {MAX_SLOTS // TOP_K} rows of {TOP_K}; got max_rows={max_rows}"
        )
    if windows not in (1, 2, 3):
        return (
            "the ring-buddy stacks hold the owner-only window, two ring windows, or three with "
            f"three_window_placement; got {windows}"
        )
    if windows == 1 and (
        balance or (static_from is not None and ranks is not None and static_from < block_of(ranks))
    ):
        return (
            "the owner-only window holds no copy of another rank's experts: it needs balance=False "
            "and no static move"
        )
    if windows == 3 and not (persistent and balance):
        return (
            "the third ring window is placed by the prologue's routed kernel for the persistent expert "
            "kernel: it needs persistent=True (persistent_experts) and the balanced placement"
        )
    if windows == 3 and ranks == 4:
        return (
            "three_window_placement at tep4 would hold three 64-expert blocks per rank (about +226 GiB "
            "per rank over the own block), which does not fit next to the KV cache on a GB300"
        )
    if half and not (windows == 1 and persistent and ranks == 4):
        return (
            "half_window_experts stacks the predecessor's upper half for the persistent expert kernel at "
            "tep4: it needs windows=1 (the owner-only placement elsewhere), persistent=True "
            f"(persistent_experts) and a tp4/ep4 mapping; got windows={windows}, "
            f"persistent={persistent}, ring {ranks}"
        )
    for index, layer in enumerate(text_model.layers):
        moe = layer.feed_forward
        state = getattr(moe, "_replicated_experts", None)
        if state is None:
            continue
        if state.half != _layer_half(moe, state.kind, half, persistent):
            return (
                f"half_window_experts: layer {index} {'holds' if state.half else 'lacks'} the half "
                "window and this install would change that; a refresh cannot"
            )
        want = _layer_windows(moe, state.kind, windows, persistent)
        if state.windows != want:
            return (
                f"three_window_placement: layer {index} holds {state.windows} ring windows and this install "
                f"stacks {want} on it; a refresh cannot add one or drop one"
            )
    if bf16_unpacked:
        try:
            from flashinfer.fused_moe import trtllm_bf16_routed_moe  # noqa: F401
        except ImportError as error:
            return f"bf16_routed_experts calls flashinfer.fused_moe.trtllm_bf16_routed_moe: {error}"
    return None
