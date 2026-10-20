# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One persistent launch for the NVFP4 experts of one window at M <= 8 rows (fast path ``persistent_experts``).

Replaces what ``fp4_block_scale_moe_runner`` launches for a pre-routed decode
step -- the routing-indices kernel, FC1 and FC2 -- with one CuTe DSL kernel
(``_persistent_experts/kernel.py``): a static tile queue over (expert, 128-row
tile) of FC1 and then FC2 across every SM, so the number of active experts
moves the time by the bytes it streams and nothing else (no 64-CTA-per-expert
waves), and FC2 starts per k-block as soon as the FC1 tiles that produce its
intermediate columns are published.

Contract (the unfinalized triple the MuseSpark MoE finalize reads):

* a slot ``(t, k)`` is local when ``0 <= topk_ids[t, k] < L`` and no earlier
  slot ``k' < k`` of the same token holds the same id: a token that repeats a
  local expert is routed to it once, through the first slot, as TRTLLM-gen's
  routing-indices kernel does.
* ``rows`` ``[M * 8, 4096]`` BF16: ``rows[t * 8 + k]`` is the FC2 output of
  local slot ``(t, k)``; other rows are not written.
* the second element is ``topk_weights`` itself (the model reads the FP32
  weights from its routing memo, never this one).
* ``index`` ``[M, 8]`` int32: ``t * 8 + k`` for a local slot, else ``-1`` --
  the row map ``FinalizeIpcWorkspace.finalize`` and ``post_expert_norm_combine``
  take as ``expanded_idx_to_permuted_idx`` (``expert_rows.slot_row_map``; the
  contract every finalize follows is ``expert_rows.py``'s).

The window is the leading dim of the weight tensors: ``L = 32`` is a tep8
rank's own expert block (ids rebased to it), ``L = 64`` the tep8 ring-buddy
stack or a tep4 rank's own 64-expert block (the owner-only window of
``ring_buddy.install(windows=1)``, ids rebased to it), ``L = 96`` the tep8
three-window stack ``[own | b-1 | b-2]``. The kernel's active-expert bit set
is over window ids (two 32-bit words up to 64), so the ring's block size does
not enter it. An expert's rows do not depend on the window it is read from.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    _E2M1_VALUES,
)

HIDDEN = 4096
INTER = 4096
TOP_K = 8
#: Rows one launch serves: the MMA's N (all token columns of a decode step).
MAX_ROWS = 8
#: Expert windows the kernel is compiled for.
WINDOWS = (32, 64, 96)
SF_VEC = 16

#: The kernel's compile-time options (``_persistent_experts/kernel.py``):
#:
#: * ``num_stages`` / ``acc_stages``: weight and accumulator pipeline depths;
#: * ``delay`` / ``delay_ns``: no injected delay, and the poll back-off;
#: * ``trace``: timeline stamps compiled out (1 stamps into ``_Workspace.trace``,
#:   2 also the first weight stage's landing, ``TR_W0``);
#: * ``balance``: each launch's tile rounds on as few CTAs as the round count allows;
#: * ``order``: early slots slot-major, then the rest row-major;
#: * ``publish``: FC1 flags published with a GPU fence and a relaxed store;
#: * ``handoff``: FC1 -> FC2 through epoch-tagged words;
#: * ``fastroute``: the weight TMA warp reads its routing through shared memory.
#:
#: The grid is the resident SM count.
OPTIONS: Dict[str, int] = {
    "num_stages": 6,
    "acc_stages": 4,
    "delay": 0,
    "delay_ns": 2000,
    "trace": 0,
    "balance": 1,
    "order": 1,
    "publish": 0,
    "handoff": 1,
    "fastroute": 0,
}

#: What each supported device compute capability changes in ``OPTIONS``: sm107 (212 SMs)
#: runs eight weight stages, which shortens a call there (six on sm103).
OPTIONS_BY_SM: Dict[Tuple[int, int], Dict[str, int]] = {
    (10, 3): {},
    (10, 7): {"num_stages": 8},
}

_LOCK = threading.Lock()
_COMPILED: Dict[tuple, object] = {}
_WORKSPACES: Dict[tuple, "_Workspace"] = {}
_RESIDENT_SMS: Dict[tuple, int] = {}
_DEVICE_OPTIONS: Dict[int, Dict[str, int]] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _resident_sms(device: torch.device) -> int:
    """SMs one grid can hold at once on ``device`` (the kernel runs 1 CTA per SM).

    The device's SM count, or fewer when the current context is limited to an
    SM partition (a green context). Refuses MPS: an MPS client's share of the
    SMs is not visible to it, so a grid sized to the device might never be
    fully resident.
    """
    from cuda.bindings import driver

    index = _device_index(device)
    ok = driver.CUresult.CUDA_SUCCESS
    err, ctx = driver.cuCtxGetCurrent()
    key = (index, int(ctx) if err == ok and ctx is not None else 0)
    sms = _RESIDENT_SMS.get(key)
    if sms is not None:
        return sms
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    mps = getattr(driver.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_MPS_ENABLED", None)
    if mps is not None:
        err, dev = driver.cuDeviceGet(index)
        if err == ok:
            err, enabled = driver.cuDeviceGetAttribute(mps, dev)
            if err == ok and enabled:
                raise RuntimeError(
                    "persistent_experts: refusing to run under MPS; every CTA of the "
                    "grid must be resident at once and an MPS client's SM share is unknown"
                )
    get_resource = getattr(driver, "cuCtxGetDevResource", None)
    if key[1] != 0 and get_resource is not None:
        err, res = get_resource(ctx, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
        if err == ok and int(res.sm.smCount) > 0:
            sms = min(sms, int(res.sm.smCount))
    _RESIDENT_SMS[key] = sms
    return sms


def _device_options(device: torch.device) -> Dict[str, int]:
    """``OPTIONS`` with ``device``'s compute-capability changes (``OPTIONS_BY_SM``) applied.

    The capability is read once per device.

    Raises:
        ValueError: on a device of another compute capability.
    """
    index = _device_index(device)
    changes = _DEVICE_OPTIONS.get(index)
    if changes is None:
        capability = tuple(torch.cuda.get_device_capability(index))
        changes = OPTIONS_BY_SM.get(capability)
        if changes is None:
            raise ValueError(
                f"persistent_experts: no kernel options for SM {capability} "
                f"(supported: {sorted(OPTIONS_BY_SM)})"
            )
        _DEVICE_OPTIONS[index] = changes
    return {**OPTIONS, **changes}


def _config(device: torch.device) -> Tuple[int, Dict[str, int]]:
    """The grid size and the kernel's compile-time options (``OPTIONS_BY_SM``) for ``device``.

    The grid is the resident SM count: CTAs spin on tiles of other CTAs, so every
    CTA of the grid must be resident at once.
    """
    return _resident_sms(device), _device_options(device)


def _refuse_under_capture(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"persistent_experts: the {what} must exist before CUDA-graph capture "
            "(call persistent_experts.prepare(...) at install time); a captured "
            "workspace zero-fill would reset every epoch on each replay"
        )


class _Workspace:
    """Graph-stable per-(device, grid) scratch shared by every launch on a stream.

    ``epoch`` holds one launch counter per CTA; ``flags`` one word per (slot,
    FC1 tile) carrying the epoch of the launch that published the tile;
    ``images`` one FC2 operand image per slot; ``trace`` the timeline-mode
    stamps (``TRACE_WORDS`` per CTA). Launches are stream-ordered
    and every access happens after the kernel's dependency wait, so one
    workspace serves every layer, and nothing is cleared between launches or
    graph replays. The grid size is fixed per workspace: each launch must
    advance every CTA's counter once. The two hand-off protocols read
    ``images`` differently, so each has its own workspace. It is created
    once, outside graph capture, and never re-initialized.
    """

    def __init__(self, device: torch.device, num_ctas: int):
        from ._persistent_experts.kernel import (
            FC1_MT,
            MAX_CTAS,
            MAX_SLOTS,
            TRACE_WORDS,
            WS_SLOT_BYTES,
        )

        assert num_ctas <= MAX_CTAS
        self.num_ctas = num_ctas
        self.epoch = torch.zeros(MAX_CTAS, dtype=torch.int32, device=device)
        self.flags = torch.zeros(MAX_SLOTS * FC1_MT, dtype=torch.int32, device=device)
        self.images = torch.zeros(MAX_SLOTS * WS_SLOT_BYTES, dtype=torch.uint8, device=device)
        self.trace = torch.zeros(num_ctas, TRACE_WORDS, dtype=torch.int64, device=device)


def _workspace(device: torch.device, num_ctas: int, handoff: Optional[int] = None) -> _Workspace:
    if handoff is None:
        handoff = _config(device)[1]["handoff"]
    key = (_device_index(device), num_ctas, int(handoff))
    with _LOCK:
        ws = _WORKSPACES.get(key)
        if ws is None:
            _refuse_under_capture("workspace")
            ws = _Workspace(device, num_ctas)
            _WORKSPACES[key] = ws
    return ws


#: ``early_expert_ids``: the early-ids buffer bound to each workspace that has one.
_EARLY: Dict[tuple, object] = {}


def early_expert_ids(device: torch.device):
    """The ``EarlyExpertIds`` buffer bound to ``device``'s current workspace (``early_expert_ids``).

    Created (with the workspace, if needed) and registered on first use; call it
    at install time, outside graph capture. Every launch on that workspace then
    disarms the buffer, and a launch whose ids are the armed ``local_ids`` runs
    the early-ids variant.

    Raises:
        ValueError: with ``OPTIONS["fastroute"]`` set (its weight TMA warp reads the
            ids from global memory, which the early variant's TMA warp does before
            any wait).
    """
    from .early_expert_ids import EarlyExpertIds

    num_ctas, options = _config(device)
    if options["fastroute"]:
        raise ValueError(
            "[musespark_open] early_expert_ids starts the persistent expert kernel before its dependency wait, which "
            "needs the "
            "routing through shared memory (fastroute 0)"
        )
    ws = _workspace(device, num_ctas, options["handoff"])
    key = (_device_index(device), num_ctas, int(options["handoff"]))
    with _LOCK:
        early = _EARLY.get(key)
        if early is None:
            _refuse_under_capture("early-ids buffer")
            early = EarlyExpertIds(device, ws.epoch, key)
            _EARLY[key] = early
    return early


def _pointers(w13, w13_sf, w2, w2_sf):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    return [
        make_ptr(cutlass.Float4E2M1FN, w13.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, w13_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, w2.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, w2_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    ]


_NUM_ADDRS = 12


def _early_options(early) -> Dict[str, int]:
    """The kernel options of the early-ids variant (``early_expert_ids``) bound to ``early``."""
    if early is None:
        return {}
    return dict(
        early_ids=early.words_address,
        early_fallbacks=early.fallbacks_address,
        early_deadline_ns=early.deadline_ns,
        early_poll_ns=early.poll_ns,
    )


def _compiled(device: torch.device, num_local: int, w13, w13_sf, w2, w2_sf, early=None):
    """(compiled launcher, grid size, options) for this device's configuration and window.

    With ``early`` (an ``EarlyExpertIds``), the variant that takes its ids from
    that buffer's tagged words before its dependency wait (``early_expert_ids``).
    """
    num_ctas, options = _config(device)
    extra = _early_options(early)
    key = (_device_index(device), num_local, num_ctas, tuple(sorted(options.items())))
    if extra:
        key = key + (tuple(sorted(extra.items())),)
    with _LOCK:
        fn = _COMPILED.get(key)
        if fn is not None:
            return fn, num_ctas, options
        _refuse_under_capture("compiled kernel")
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        from ._persistent_experts.kernel import PersistentExpertsKernel

        kern = PersistentExpertsKernel(num_ctas, num_local, **options, **extra)
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        fn = cute.compile(
            kern.run,
            *_pointers(w13, w13_sf, w2, w2_sf),
            *([cutlass.Int64(0)] * _NUM_ADDRS),
            cutlass.Int32(1),
            stream,
        )
        _COMPILED[key] = fn
        return fn, num_ctas, options


def _unswizzle_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """``[rows, cols]`` E4M3 block scales from the 128x4-interleaved byte layout.

    Atoms of 128 rows x 4 scale columns (512 bytes) in row-atom-major order;
    inside an atom, row ``m`` and column ``c`` sit at
    ``(m % 32) * 16 + (m // 32) * 4 + c``.
    """
    raw = sf.reshape(-1).view(torch.uint8)
    m = torch.arange(rows, device=sf.device)[:, None]
    c = torch.arange(cols, device=sf.device)[None, :]
    off = (
        (m // 128) * (cols // 4) * 512
        + (c // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (c % 4)
    )
    return raw[off].view(torch.float8_e4m3fn)


def _decode_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 bytes ``[..., K/2]`` -> FP32 values ``[..., K]`` (element 2i in the low nibble)."""
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    lo = table[(codes & 0xF).long()]
    hi = table[(codes >> 4).long()]
    return torch.stack((lo, hi), dim=-1).reshape(*codes.shape[:-1], codes.shape[-1] * 2)


def _shuffled_row_source(rows: int, device) -> torch.Tensor:
    """Physical row ``p`` of a TRTLLM-gen shuffled matrix holds pre-shuffle row ``[p]``."""
    p = torch.arange(rows, device=device)
    return (p // 32) * 32 + 4 * (p % 8) + (p % 32) // 8


def _dequant_rows(codes: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """One expert's ``[R, K]`` FP64 weight (physical row order) from codes and swizzled scales."""
    rows, k = codes.shape[0], codes.shape[1] * 2
    scales = _unswizzle_sf(sf, rows, k // SF_VEC).float()
    return (_decode_fp4(codes) * scales.repeat_interleave(SF_VEC, dim=1)).double()


def _requant_nvfp4(h: torch.Tensor) -> torch.Tensor:
    """NVFP4 quantize-dequantize of ``h`` ``[M, K]`` with unit global scale (exact IEEE math)."""
    rows, cols = h.shape
    blocks = h.reshape(rows, cols // SF_VEC, SF_VEC)
    sf = (blocks.abs().amax(dim=-1, keepdim=True) / 6.0).clamp(max=448.0)
    sf = sf.float().to(torch.float8_e4m3fn).double()
    scaled = torch.where(sf == 0, torch.zeros_like(blocks), blocks / sf)
    step = torch.where(scaled.abs() < 2.0, 0.5, torch.where(scaled.abs() < 4.0, 1.0, 2.0))
    q = torch.sign(scaled) * torch.clamp(torch.round(scaled.abs() / step) * step, max=6.0)
    return (q * sf).reshape(rows, cols)


class PersistentExperts(OpWrapper):
    """NVFP4 experts of one window for <= 8 pre-routed rows, one persistent launch.

    Arguments mirror the pre-routed ``fp4_block_scale_moe_runner`` call the
    MuseSpark layer makes: NVFP4 rows ``hidden_states`` ``[M, 2048]`` uint8 with
    linear E4M3 block scales ``hidden_states_scale`` (``M * 256``), window ids
    ``topk_ids`` ``[M, 8]`` int32 (local when ``0 <= id < L``, any value
    allowed; a repeated local id in one row counts once, at its first slot), the BF16
    ``topk_weights`` (passed through), the window's stock TRTLLM-gen tensors
    (``gemm1_weights`` ``[L, 8192, 2048]`` gate/up rows reordered and shuffled,
    ``gemm2_weights`` ``[L, 4096, 2048]`` shuffled, both scales 128x4
    interleaved) and the per-expert scalars (``output1_scale_scalar`` =
    ``fc31_scale_c``, ``output1_scale_gate_scalar`` = ``fc31_alpha``,
    ``output2_scale_scalar`` = ``fc2_alpha``). Returns
    ``(rows, topk_weights, index)``; see the module docstring.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DFlash K=7 verify width on the ring-buddy stack, ~4-5 active local experts: the "
            "typical two-window census",
            spec=dict(rows=8, window=64, active=5),
        ),
        Cell(
            why="the same width with no expert of this window selected: nothing to compute, "
            "every slot -1",
            spec=dict(rows=8, window=64, active=0),
        ),
        Cell(
            why="a rank's own 32-expert block (plain expert parallelism), 6 active experts",
            spec=dict(rows=8, window=32, active=6),
        ),
        Cell(
            why="tep4: a rank's own 64-expert block (the owner-only window), the busiest rank "
            "of a K=7 verify step (~28 distinct experts over 4 ranks, up to ~12 on one)",
            spec=dict(rows=8, window=64, active=12),
        ),
        Cell(
            why="a single decode row: one token, up to 8 local experts",
            spec=dict(rows=1, window=64, active=8),
        ),
        Cell(
            why="three rows: token columns past M must stay zero and unwritten",
            spec=dict(rows=3, window=64, active=7),
        ),
        Cell(
            why="every slot local and distinct: 64 active experts, the longest tile queue",
            spec=dict(rows=8, window=64, active=64),
        ),
        Cell(
            why="the three-window stack at the verify width, ~4 active experts spread "
            "over all three blocks (third-word bits of the active set)",
            spec=dict(rows=8, window=96, active=4),
        ),
        Cell(
            why="the three-window stack with 64 distinct active experts across the three blocks: "
            "every slot of the table, prefixes over three words",
            spec=dict(rows=8, window=96, active=64),
        ),
    )

    note = """
    Bitwise equal, row for row, to the routed FC2 rows of
    ``fp4_block_scale_moe_runner`` (any tactic) for the same inputs: the same
    block-scaled MMA instructions in the same K order with one accumulator,
    and TRTLLM-gen's FC1 epilogue operation for operation -- the order in
    which ``alpha1`` and ``scale_c`` enter, the constant 1/6, and FTZ only
    where TRTLLM-gen flushes (``kernel._swiglu_quant_to_stage``). Any other
    association of the scalars is mathematically equal and flips the FP4 code
    or block scale of values on a rounding midpoint, which arbitrary FP32
    scalars hit in about 1 % of decode calls. ``reference`` is the plain-torch
    layer in FP64 with exact IEEE division; it differs from both kernels by
    the approximate reciprocals and the FP4 code flips they cause.

    The CTAs of a launch spin on FC1 tiles of other CTAs of the same grid, so
    every CTA must be resident at once: the grid is the SMs the current
    context can use, MPS is refused, and the kernel must not run beside a kernel that waits on
    it. In particular it must not be launched ahead of its activation
    producer on another stream to stream weights early; that is the same
    deadlock shape. Weight loads before the activations are ready need a
    same-stream PDL dependent whose producer publishes the ids with
    release/acquire: the early-ids variant (``early_expert_ids``) takes
    them from epoch-tagged words that ``pool_gather_route`` stores, when it is
    called with exactly the ``local_ids`` that call armed the buffer with;
    otherwise this launch is the default kernel. The workspace is per
    process, device, grid size and
    hand-off protocol, and launches using it must be stream-ordered. Call
    ``prepare`` at install time, before any graph capture: it compiles the
    kernel (CuTe DSL JIT, about 1 s) and creates the workspace, and a
    captured call that would have to do either refuses.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import cutlass
        from cuda.bindings import driver as cuda

        device = hidden_states.device
        rows_in = hidden_states.shape[0]
        num_local = gemm1_weights.shape[0]
        fn, num_ctas, options = _compiled(
            device,
            num_local,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        )
        ws = _workspace(device, num_ctas, options["handoff"])
        if _EARLY:
            from . import early_expert_ids as _early_ids

            early = _EARLY.get((_device_index(device), num_ctas, int(options["handoff"])))
            if early is not None and _early_ids.launches_early(early, topk_ids):
                fn, _, _ = _compiled(
                    device,
                    num_local,
                    gemm1_weights,
                    gemm1_weights_scale,
                    gemm2_weights,
                    gemm2_weights_scale,
                    early=early,
                )
        rows = torch.empty(rows_in * TOP_K, HIDDEN, dtype=torch.bfloat16, device=device)
        index = torch.empty(rows_in, TOP_K, dtype=torch.int32, device=device)
        addrs = (
            hidden_states.data_ptr(),
            hidden_states_scale.data_ptr(),
            topk_ids.data_ptr(),
            output1_scale_scalar.data_ptr(),
            output1_scale_gate_scalar.data_ptr(),
            output2_scale_scalar.data_ptr(),
            rows.data_ptr(),
            index.data_ptr(),
            ws.epoch.data_ptr(),
            ws.flags.data_ptr(),
            ws.images.data_ptr(),
            ws.trace.data_ptr(),
        )
        fn(
            *_pointers(gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale),
            *[cutlass.Int64(a) for a in addrs],
            cutlass.Int32(rows_in),
            cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
        )
        return rows, topk_weights, index

    def prepare(
        self,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        early=None,
    ) -> None:
        """Compile the kernel for this window and create its workspace, now.

        Call it eagerly (at install time, before any graph capture): a call
        under capture that would have to do either refuses. With ``early`` (the
        workspace's ``EarlyExpertIds``) the early-ids variant
        is compiled too.
        """
        device = gemm1_weights.device
        _, num_ctas, options = _compiled(
            device,
            gemm1_weights.shape[0],
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
        )
        _workspace(device, num_ctas, options["handoff"])
        if early is not None:
            _compiled(
                device,
                gemm1_weights.shape[0],
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                early=early,
            )
        torch.cuda.synchronize(device)

    def reference(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer in plain torch, FP64, over the prepared (shuffled) operands.

        Undoes the 32-row shuffle and the gate/up interleave by index, decodes
        the swizzled scales, and requantizes the SwiGLU output to NVFP4 with
        exact division (the kernels use approximate reciprocals). Rows of
        non-local slots are zero here (the op leaves them unwritten); a local
        expert a token repeats is routed through its first slot only.
        """
        m = hidden_states.shape[0]
        num_local = gemm1_weights.shape[0]
        device = hidden_states.device
        x_sf = hidden_states_scale.reshape(m, HIDDEN // SF_VEC).float()
        x = (_decode_fp4(hidden_states) * x_sf.repeat_interleave(SF_VEC, dim=1)).double()
        ids = topk_ids.reshape(m, TOP_K)
        first = torch.zeros(m, TOP_K, dtype=torch.bool, device=device)
        for t, row in enumerate(ids.tolist()):
            for k, v in enumerate(row):
                first[t, k] = 0 <= v < num_local and v not in row[:k]
        rows = torch.zeros(m * TOP_K, HIDDEN, dtype=torch.float64, device=device)
        index = torch.full((m, TOP_K), -1, dtype=torch.int32, device=device)
        src1 = _shuffled_row_source(2 * INTER, device)  # pre-shuffle interleaved row
        up_rows = torch.empty(INTER, dtype=torch.long, device=device)
        gate_rows = torch.empty(INTER, dtype=torch.long, device=device)
        phys = torch.arange(2 * INTER, device=device)
        up_rows[src1[src1 % 2 == 0] // 2] = phys[src1 % 2 == 0]
        gate_rows[src1[src1 % 2 == 1] // 2] = phys[src1 % 2 == 1]
        src2 = _shuffled_row_source(HIDDEN, device)
        for e in sorted({int(v) for v in ids.reshape(-1).tolist() if 0 <= int(v) < num_local}):
            w13 = _dequant_rows(gemm1_weights[e], gemm1_weights_scale[e])
            acc1 = x @ w13.T  # [m, 8192] physical rows
            g = acc1[:, gate_rows] * float(output1_scale_gate_scalar[e])
            up = acc1[:, up_rows] * float(output1_scale_scalar[e])
            h = up * (g * torch.sigmoid(g))
            hq = _requant_nvfp4(h)
            w2 = torch.empty(HIDDEN, INTER, dtype=torch.float64, device=device)
            w2[src2] = _dequant_rows(gemm2_weights[e], gemm2_weights_scale[e])
            y = (hq @ w2.T) * float(output2_scale_scalar[e])
            for t, k in ((ids == e) & first).nonzero().tolist():
                rows[t * TOP_K + k] = y[t]
                index[t, k] = t * TOP_K + k
        return rows.to(torch.bfloat16), topk_weights, index

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
    ) -> None:
        """Inputs the kernel would read past or misread without an error."""
        m = hidden_states.shape[0]
        assert 1 <= m <= MAX_ROWS, f"one launch serves 1..{MAX_ROWS} rows, got {m}"
        assert hidden_states.dtype == torch.uint8 and hidden_states.is_contiguous(), (
            "hidden_states must be contiguous packed NVFP4 bytes"
        )
        assert tuple(hidden_states.shape) == (m, HIDDEN // 2), tuple(hidden_states.shape)
        assert hidden_states_scale.is_contiguous() and hidden_states_scale.numel() == m * (
            HIDDEN // SF_VEC
        ), "hidden_states_scale must be the linear (unswizzled) [M, 256] block scales"
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous(), (
            "topk_ids must be contiguous int32; the kernel reads raw words"
        )
        assert topk_ids.numel() == m * TOP_K, f"topk_ids must be [M, {TOP_K}]"
        num_local = gemm1_weights.shape[0]
        assert num_local in WINDOWS, f"the kernel is compiled for windows {WINDOWS}"
        for name, t, shape in (
            ("gemm1_weights", gemm1_weights, (num_local, 2 * INTER, HIDDEN // 2)),
            ("gemm2_weights", gemm2_weights, (num_local, HIDDEN, INTER // 2)),
        ):
            assert t.is_contiguous() and tuple(t.shape) == shape, f"{name} must be {shape}"
        for name, t, n in (
            ("gemm1_weights_scale", gemm1_weights_scale, num_local * 2 * INTER * HIDDEN // SF_VEC),
            ("gemm2_weights_scale", gemm2_weights_scale, num_local * HIDDEN * INTER // SF_VEC),
        ):
            assert t.is_contiguous() and t.numel() == n, f"{name} must hold {n} scales"
        for name, t in (
            ("output1_scale_scalar", output1_scale_scalar),
            ("output1_scale_gate_scalar", output1_scale_gate_scalar),
            ("output2_scale_scalar", output2_scale_scalar),
        ):
            assert t.dtype == torch.float32 and t.is_contiguous() and t.numel() == num_local, (
                f"{name} must be [{num_local}] float32"
            )


persistent_experts = PersistentExperts()
