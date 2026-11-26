# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A MoE layer's routed experts (this rank's 64-expert window) as the ops a Target calls in
order: `moe_l2_prefetch`, `moe_quantize_first`, `moe_router`, `moe_topk`, `moe_experts`,
`moe_combine`; and the layer's bookkeeping beside them (`moe_open`, `moe_exact`,
`mtp_moe_open`, `chunk_sizes`).

Router and top-k run on replicated hidden states, so every rank selects the same experts and
the four windows tile the routing space once. No stock fallback for the grouped GEMMs: the
weights are packed for these kernels alone. What the fast path leaves out at a row count --
the power-of-two padding, the staging copies, the routing launch, the NVFP4 quantize and the
output clear -- is `moe_experts`' fallback chain.

Every operand handed to these kernels lives in a buffer allocated at load (`build_kernels`),
in two slots alternating per layer. The engine allocates its tensors inside the CUDA-graph
capture; handed to these kernels directly, replays routed as though every id were zero. And
the kernels launch with PDL, so a layer's kernel can start while the previous layer's is still
reading: one slot measured every layer but the last at cosine 0.64-0.93. Which slot a layer
takes is runtime state (`SlotParity`), held by the core and passed in.

The MTP layer runs the same ops on its own handle (`MoeHandle.mtp_bf16`): with its experts
quantized to NVFP4 at load, exactly the trunk's path; otherwise router and top-k defer to
`moe_experts`, whose fallback is the MTP layer's bf16 pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

from .._build import Count, try_build
from ..elementwise.cute_dsl_copy import RowCopyHandle, row_copy
from ..elementwise.cute_dsl_hidden_add import HiddenAddHandle, hidden_add
from ..gemm.projection import DECODE_MAX_ROWS, ProjectionHandle, projection
from ..quantization import cute_dsl_moe_quantize_nvfp4 as _moe_quant_op
from ..quantization.cute_dsl_moe_quantize_nvfp4 import cute_dsl_moe_quantize_nvfp4
from . import cute_dsl_moe_l2_prefetch as _l2pf_op
from . import cute_dsl_moe_merge as _moe_merge_op
from . import cute_dsl_moe_nvfp4_experts as _moe_experts_op
from . import cute_dsl_moe_prefill_topk_routing_rep as _moe_topk_pre_rep_op
from . import cute_dsl_moe_router as _moe_router_op
from . import cute_dsl_moe_routing as _moe_routing_op
from . import cute_dsl_moe_topk_routing as _moe_topk_op
from . import cute_dsl_moe_topk_routing_rep as _moe_topk_rep_op
from . import cute_dsl_moe_topk_routing_wide_rep as _moe_topk_wide_rep_op
from . import cute_dsl_moe_zero as _moe_zero_op
from . import cute_dsl_mtp_moe_experts as _mtp_experts_op
from .cute_dsl_moe_l2_prefetch import cute_dsl_moe_l2_prefetch
from .cute_dsl_moe_merge import cute_dsl_moe_merge
from .cute_dsl_moe_nvfp4_experts import cute_dsl_moe_nvfp4_experts
from .cute_dsl_moe_prefill_topk_routing_rep import cute_dsl_moe_prefill_topk_routing_rep
from .cute_dsl_moe_router import cute_dsl_moe_router
from .cute_dsl_moe_routing import cute_dsl_moe_routing
from .cute_dsl_moe_sigmoid_topk import SigmoidTopkHandle, sigmoid_topk
from .cute_dsl_moe_topk_routing import cute_dsl_moe_topk_routing
from .cute_dsl_moe_topk_routing_rep import cute_dsl_moe_topk_routing_rep
from .cute_dsl_moe_topk_routing_wide_rep import cute_dsl_moe_topk_routing_wide_rep
from .cute_dsl_moe_zero import cute_dsl_moe_zero
from .cute_dsl_mtp_moe_experts import cute_dsl_mtp_moe_experts

#: The token budget: the routed buffers' rows; a layer above it runs in chunks of this many.
MAX_ROWS = 8192

TOP_K = _moe_routing_op.TOP_K

# The routed experts' row domains: the decode kernels (router, fused top-k, quantize, clear,
# decode grouped GEMMs) serve 1..64 rows -- `max_batch_size` 32 x MTP width 2 -- and the
# prefill kernels 65..8192. A decode step presents a power of two up to 64 after padding.
DECODE_WIDTHS = (1, 2, 4, 8, 16, 32, 64)

PREFILL_MIN = DECODE_MAX_ROWS + 1

#: From here up the prefill pair runs the exact row count. The routing schedule is exact per
#: `ceil(rows / 32)` up to 512 rows, so below this a power of two keeps the compiled
#: schedules few; above it the four compiled buckets cover every count.
PAIR_EXACT_FROM = 513

#: NVFP4: one e4m3 scale per 16 elements.
SF_VEC = 16

#: The widest wide decode step that takes the expert replicas (`build_kernels`): the wide
#: top-k's 65..128 range. 4-GPU decode chain with the real all-reduce on real routing (two
#: trunk MoE layers with replicas): -6.5 / -5.8 us per step at 128 rows (MTP3), -8.1 / -6.4 /
#: -4.2 us at 68 / 96 / 126; at 192 rows (MTP5) -2.7 / +2.8 / +0.4 us over 320 / 800 / 1600
#: replays a side, no gain past the replica top-k's +0.7 us, so 129..256 keep the wide top-k.
WIDE_REP_MAX_ROWS = 128


def pow2(rows: int) -> int:
    return 1 << max(0, (rows - 1).bit_length())


def chunk_sizes(total: int) -> list[int]:
    """The chunks a layer of `total` rows runs in: `MAX_ROWS` each, then the rest."""
    full, rest = divmod(total, MAX_ROWS)
    return [MAX_ROWS] * full + ([rest] if rest else [])


def _direct_ok(t: torch.Tensor) -> bool:
    """Read in place rather than staged. A staging copy is a memcpy node, which a PDL edge
    cannot pass through, so it is skipped for a 128-byte aligned contiguous bf16 tensor -- the
    collective's output, whose graph-pool address is stable."""
    return t.dtype is torch.bfloat16 and t.is_contiguous() and t.data_ptr() % 128 == 0


class SlotParity:
    """Which of the two buffer slots the next MoE call takes. Runtime state: the core holds
    one for the trunk (the MTP layer shares it on the NVFP4 path) and one for the MTP layer's
    bf16 pair, and the Targets pass it in."""

    def __init__(self) -> None:
        self.slot = 0

    def flip(self) -> int:
        """Advance, and return the slot a layer opened ahead of its shared branch takes."""
        self.slot ^= 1
        return self.slot

    def take(self) -> int:
        """Return the current slot, and advance for the next call (a chunk that did not open
        its layer ahead: above `MAX_ROWS`, or the MTP layer)."""
        slot = self.slot
        self.slot ^= 1
        return slot


@dataclass(frozen=True)
class MoeSelection:
    """`moe_topk`'s output: `w` / `ids` the routing, `maps` the tile maps a top-k kernel
    wrote for them (None: `moe_experts` runs the routing kernel), `signaled` the early-release
    top-k launched (its FC1 starts before the ids land: `moe_experts` must not pad), `rep`
    the maps address the replica slots (the replica prefill pair runs them)."""

    w: torch.Tensor
    ids: torch.Tensor
    maps: Any = None
    signaled: bool = False
    rep: bool = False


@dataclass(frozen=True)
class MoeHandle:
    """What the routed ops launch: `k` the compiled kernels and buffers (`build_kernels`);
    `router_proj` the router projection's handle and `route` the sigmoid top-k's (the
    fallbacks at row counts no fused front end takes); `copy` / `add` the row ops the
    fallback chains use; `mtp_bf16` the MTP layer's bf16 pair (`build_mtp_bf16`), set only
    on the MTP layer's handle when its experts did not quantize to NVFP4."""

    k: SimpleNamespace
    expert_offset: int
    routed_scale: float
    n_group: int
    topk_group: int
    topk: int
    num_layers: int
    wide_cells: bool
    dec_unpad: bool
    acc_padded: bool
    router_proj: ProjectionHandle
    route: SigmoidTopkHandle
    copy: RowCopyHandle
    add: HiddenAddHandle
    count: Count
    mtp_bf16: SimpleNamespace | None = None


def allocate_slot(
    *,
    mtp: bool,
    hidden: int,
    inter: int,
    num_experts: int,
    dec_replicas: int,
    ctx_replicas: int,
    wide_cells: bool,
    dev: int,
) -> SimpleNamespace:
    """One slot's buffers, trunk or MTP (`mtp`)."""
    rows = MAX_ROWS
    bf16 = {"dtype": torch.bfloat16, "device": dev}
    f32 = {"dtype": torch.float32, "device": dev}
    counts, barrier = _moe_routing_op.allocate_workspace(dev)
    widths = DECODE_WIDTHS if mtp else range(1, DECODE_MAX_ROWS + 1)
    ns = SimpleNamespace(
        # The routed output is returned to the caller, so it alternates too: one buffer would
        # be overwritten by the next layer before the caller used it.
        out=torch.zeros((rows, hidden), **bf16),
        x=torch.zeros((rows, hidden), **bf16),
        w=torch.zeros((rows, TOP_K), **f32),
        sel=torch.empty(rows * TOP_K, dtype=torch.int32, device=dev),
        # Context replicas: one more partial tile per replica slot (trunk only).
        maps=_moe_routing_op.allocate_maps(rows, dev, replicas=0 if mtp else ctx_replicas),
        counts=counts,
        barrier=barrier,
        router_out=torch.zeros((DECODE_MAX_ROWS, num_experts), **f32),
        router_partials=torch.zeros((DECODE_MAX_ROWS, _moe_router_op.SPLIT_K, num_experts), **f32),
        # The fused top-k checks its outputs against the exact row count: one set per width;
        # with decode expert replicas the push widths' sets hold the replica top-k's
        # `64 + replicas` tiles.
        topk_out={
            w: _moe_topk_rep_op.allocate_outputs(w, dec_replicas, dev)
            if not mtp and dec_replicas and _moe_topk_rep_op.uses_rep_path(w)
            else _moe_topk_op.allocate_decode_outputs(w, dev)
            for w in widths
        },
    )
    if mtp:
        ns.c = _mtp_experts_op.allocate_c(rows, inter, dev)
        return ns
    ns.c = _moe_experts_op.allocate_decode_c(rows, inter, dev)
    ns.sfc = _moe_experts_op.allocate_sfc(rows, inter, dev, ctx_replicas)
    ns.router_x = torch.zeros((DECODE_MAX_ROWS, hidden), **bf16)
    if wide_cells:
        # The wide decode router's `[rows, 4, 256]` partials (`wide_cells` only).
        ns.router_wide_out = torch.zeros(
            _moe_router_op.output_shape(_moe_router_op.WIDE_MAX_ROWS, wide=True), **f32
        )
    ns.logits = torch.zeros((rows, 1, num_experts), **f32)
    ns.topk_scratch = _moe_topk_op.allocate_prefill_scratch(dev)
    if ctx_replicas:
        ns.topk_rep_scratch = _moe_topk_pre_rep_op.allocate_scratch(ctx_replicas, dev)
    return ns


def build_kernels(
    *,
    hidden: int,
    inter: int,
    local_experts: int,
    num_experts: int,
    num_layers: int,
    dec_replicas: int,
    ctx_replicas: int,
    dec_hot: Mapping[int, Any],
    wide_cells: bool,
    wide_dec_replicas: bool,
    wide_topk_schedule: tuple[int, bool],
    fc1_kblk: bool,
    fc2_kblk: bool,
    l2_prefetch: bool,
    dec_unpad_signal: bool,
    max_draft_len: int,
    dev: int,
) -> SimpleNamespace:
    """The routed experts' buffers and compiled kernels, trunk (the MTP layer shares them on
    the NVFP4 path). The NVFP4 input (`xq` / `xsf`) is one buffer, written by the pre-MoE
    collective or the quantize launch and read by FC1 in the same layer.

    A decode kernel that fails to build raises: those steps have no stock path for the packed
    weights. A prefill kernel that fails to build is None, and the op takes its fallback (the
    decode grouped GEMMs for the prefill pair)."""
    rows = MAX_ROWS
    u8 = {"dtype": torch.uint8, "device": dev}

    def optional(build, what: str = "MoE prefill kernel"):
        return try_build(what, build)

    def slot() -> SimpleNamespace:
        return allocate_slot(
            mtp=False,
            hidden=hidden,
            inter=inter,
            num_experts=num_experts,
            dec_replicas=dec_replicas,
            ctx_replicas=ctx_replicas,
            wide_cells=wide_cells,
            dev=dev,
        )

    k = SimpleNamespace()
    k.slots = [slot() for _ in range(2)]
    # One byte per element, as the kernels' entries allocate: cute counts the shape in
    # elements and reads half as many bytes.
    k.xq = torch.zeros((rows, hidden // 2), **u8)
    k.xsf = torch.zeros((rows, hidden // SF_VEC), **u8)
    s0 = k.slots[0]
    # One routing function per schedule bucket any step reaches, shared with the MTP layer: a
    # bucket compiled on first use would compile inside a step.
    k.routing = _moe_routing_op.build(_moe_routing_op.ROW_COUNTS, dev)
    k.experts = _moe_experts_op.build_decode(
        c=s0.c,
        sfc=s0.sfc,
        maps=s0.maps,
        xq=k.xq,
        xsf=k.xsf,
        out=s0.out,
        hidden=hidden,
        inter=inter,
        # The decode expert replicas' slots follow this rank's 64 (0 when off).
        experts=local_experts + dec_replicas,
        max_rows=rows,
        fc1_kblk=fc1_kblk,
        fc2_kblk=fc2_kblk,
        # One FC2 tile counter per MoE call site: the trunk layers and the MTP layer.
        dyn_slots=num_layers + 1,
    )
    k.pair = optional(
        lambda: _moe_experts_op.build_prefill_pair(
            hidden, inter, local_experts, dev, fc1_kblk=fc1_kblk, fc2_kblk=fc2_kblk
        )
    )
    if k.pair is not None:
        for sl in k.slots:
            sl.pc = _moe_experts_op.allocate_pair_c(rows, inter, dev, ctx_replicas)
    # Context replicas: the replica top-k and the prefill pair on the `64 + R` deep stacks,
    # both or neither (the replica maps address slots past 64).
    k.topk_pre_rep = k.pair_rep = None
    k.ctx_hot_ids = {}
    if ctx_replicas and k.pair is not None:
        rep = optional(
            lambda: (
                _moe_topk_pre_rep_op.build_prefill(ctx_replicas, dev),
                _moe_experts_op.build_prefill_pair(
                    hidden,
                    inter,
                    local_experts + ctx_replicas,
                    dev,
                    fc1_kblk=fc1_kblk,
                    fc2_kblk=fc2_kblk,
                ),
            )
        )
        k.topk_pre_rep, k.pair_rep = rep if rep is not None else (None, None)
        k.ctx_hot_ids = {
            i: torch.tensor(ids[:ctx_replicas], dtype=torch.int32, device=dev)
            for i, ids in dec_hot.items()
            if len(ids) >= ctx_replicas
        }
    k.router_pre = optional(lambda: _moe_router_op.build_prefill(dev))
    k.topk_pre = optional(lambda: _moe_topk_op.build_prefill(dev))
    k.quant_pre = optional(lambda: _moe_quant_op.build_prefill(dev))
    k.zero_pre = optional(lambda: _moe_zero_op.build_prefill(dev))
    k.merge = optional(lambda: _moe_merge_op.build(dev))
    k.quant = {w: _moe_quant_op.build_decode(w, dev) for w in DECODE_WIDTHS}
    k.zero = {w: _moe_zero_op.build_decode(w, dev) for w in DECODE_WIDTHS}
    # The decode L2 prefetch of the routed FC1 weights: forked at a decode MoE layer's o_proj
    # end, ranked by the counts the same layer's previous decode step's top-k wrote into its
    # row of `hist` (one row per MoE call site, the MTP layer's included).
    k.pf = None
    pf = try_build("decode MoE L2 prefetch", lambda: _l2pf_op.compile(dev)) if l2_prefetch else None
    if pf is not None:
        k.pf = SimpleNamespace(
            compiled=pf,
            hist=torch.zeros((num_layers + 1, 64), dtype=torch.int32, device=dev),
            stream=torch.cuda.Stream(device=dev),
        )
    hist = k.pf is not None
    # The fused top-k releases FC1 early only at a power of two: other widths are padded and
    # run the routing kernel next, which does not wait for the ids; released early it reads
    # them mid-write (a 3-request step hung that way).
    k.topk = {
        w: _moe_topk_op.build_decode(
            w,
            _moe_router_op.decode_splits(w),
            dev,
            signal_gg1=(w & (w - 1)) == 0,
            expert_hist=hist and _moe_topk_op.uses_push_path(w),
        )
        for w in range(1, DECODE_MAX_ROWS + 1)
    }
    # Decode expert replicas: the push widths select on the replica top-k, with each trunk
    # layer's hot ids (the MTP layer passes all -1: the home mapping).
    k.topk_rep = {}
    k.dec_hot_ids = {}
    k.dec_hot_none = None
    if dec_replicas:
        k.topk_rep = {
            w: _moe_topk_rep_op.build_decode(
                w,
                _moe_router_op.decode_splits(w),
                dec_replicas,
                dev,
                signal_gg1=(w & (w - 1)) == 0,
                expert_hist=hist,
            )
            for w in range(1, DECODE_MAX_ROWS + 1)
            if _moe_topk_rep_op.uses_rep_path(w)
        }
        k.dec_hot_ids = {
            i: torch.tensor(ids, dtype=torch.int32, device=dev) for i, ids in dec_hot.items()
        }
        k.dec_hot_none = torch.full((dec_replicas,), -1, dtype=torch.int32, device=dev)
    # Early-release top-k handles for the unpadded trunk decode widths that are not powers of
    # two (`TRTLLM_GLM52_MOE_DEC_UNPAD_SIGNAL`): the decode step widths of an MTP3 / MTP5
    # engine, `requests * (max_draft_len + 1)` up to 64 rows. Trunk only: the MTP layer pads
    # those widths and runs the routing kernel after its top-k, which must not be released
    # early (it does not wait for the ids).
    k.topk_sig = {}
    if dec_unpad_signal and max_draft_len > 1:
        step = max_draft_len + 1
        sig_widths = [w for w in range(step, DECODE_MAX_ROWS + 1, step) if w & (w - 1)]

        def build_sig(w):
            if w in k.topk_rep:
                return _moe_topk_rep_op.build_decode(
                    w,
                    _moe_router_op.decode_splits(w),
                    dec_replicas,
                    dev,
                    signal_gg1=True,
                    expert_hist=hist,
                )
            return _moe_topk_op.build_decode(
                w,
                _moe_router_op.decode_splits(w),
                dev,
                signal_gg1=True,
                expert_hist=hist and _moe_topk_op.uses_push_path(w),
            )

        built = try_build(
            "MoE unpadded decode early-release top-k",
            lambda: {w: build_sig(w) for w in sig_widths},
        )
        k.topk_sig = built or {}
    # The MTP3 / MTP5 decode widths' front end (`decode_exact`, 65..256 rows): the wide decode
    # router's split-K partials, and the fused top-k summing them, one handle per row range,
    # writing the L2 prefetch's counts. MTP1 and no speculation build neither. The partials
    # need the wide top-k: without it the router is dropped too (the prefill router and top-k
    # serve those steps); without the router the top-k reads the prefill router's one plane.
    k.router_wide = None
    k.topk_wide = {}
    if wide_cells and k.topk_pre is not None:
        k.router_wide = optional(lambda: _moe_router_op.build_wide(dev), "wide decode MoE router")
        splits = _moe_router_op.SPLIT_K if k.router_wide is not None else 1
        wide = optional(
            lambda: {
                r: _moe_topk_op.build_wide(
                    r[1],
                    splits,
                    dev,
                    expert_hist=hist,
                    cluster=wide_topk_schedule[0],
                    trim=wide_topk_schedule[1],
                )
                for r in _moe_topk_op.WIDE_RANGES
            },
            "wide decode MoE top-k",
        )
        if wide is None:
            k.router_wide = None
        k.topk_wide = wide or {}
    _build_wide_rep(k, wide_dec_replicas, ctx_replicas, dev)
    # Shared with the MTP layer's front end: the same compiled kernel per row count.
    k.router = _moe_router_op.build_all(dev)
    return k


def _build_wide_rep(k: SimpleNamespace, wide_dec_replicas: bool, ctx_replicas: int, dev) -> None:
    """The MTP3 / MTP5 wide decode steps' expert replicas: `k.topk_wide_rep`, one handle per
    wide row range (keyed as `k.topk_wide`).

    At 65..256 decode rows the prefill pair streams one expert's weights per active local
    expert, so a rank's routed time follows its count of distinct active experts and the
    post-MoE all-reduce waits for the busiest rank (real routing, 75 layers: 2.66 / 2.09
    experts above the mean at 128 / 192 rows). The wide replica top-k moves each active hot
    expert whole to the EP rank with the fewest active experts (the decode replicas' greedy),
    on the context replicas' slots: the `64 + R` deep prefill pair (`k.pair_rep`) and each
    trunk layer's hot ids (`k.ctx_hot_ids`); the MTP layer has none and keeps the wide top-k.
    Built only with the wide top-k (MTP3 / MTP5), the context replicas and
    `wide_dec_replicas`, and only for the ranges up to `WIDE_REP_MAX_ROWS` (65..128); a failed
    build leaves every step on the wide top-k."""
    k.topk_wide_rep = {}
    if not (
        wide_dec_replicas
        and k.topk_wide
        and k.pair_rep is not None
        and ctx_replicas
        and k.ctx_hot_ids
    ):
        return
    splits = next(iter(k.topk_wide.values())).logit_splits
    built = try_build(
        "MoE wide decode replica top-k",
        lambda: {
            r: _moe_topk_wide_rep_op.build_wide_rep(
                r[1],
                splits,
                ctx_replicas,
                dev,
                expert_hist=h.expert_hist,
                cluster=h.cluster,
                trim=h.trim,
            )
            for r, h in k.topk_wide.items()
            if r[1] <= WIDE_REP_MAX_ROWS
        },
    )
    k.topk_wide_rep = built or {}


def build_mtp_bf16(
    k: SimpleNamespace,
    *,
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    hidden: int,
    inter: int,
    num_experts: int,
    dev: int,
) -> SimpleNamespace:
    """The MTP layer's bf16 pair, where its experts did not quantize to NVFP4: its own two
    slots, the grouped GEMM pair on its bf16 stacks, and the fused top-k per power of two
    (sharing the trunk's router and routing kernels in `k`)."""
    slots = [
        allocate_slot(
            mtp=True,
            hidden=hidden,
            inter=inter,
            num_experts=num_experts,
            dec_replicas=0,
            ctx_replicas=0,
            wide_cells=False,
            dev=dev,
        )
        for _ in range(2)
    ]
    m0 = slots[0]
    return SimpleNamespace(
        slots=slots,
        experts=_mtp_experts_op.build(
            x=m0.x,
            c=m0.c,
            out=m0.out,
            topk_w=m0.w,
            maps=m0.maps,
            fc1_w=fc1,
            fc2_w=fc2,
            max_rows=MAX_ROWS,
        ),
        # No quantize launch on the bf16 path to carry the output clear, so the top-k clears
        # it; every MTP width is a power of two, so it always releases FC1.
        topk={
            w: _moe_topk_op.build_decode(
                w, _moe_router_op.decode_splits(w), dev, signal_gg1=True, fuse_zero=True
            )
            for w in DECODE_WIDTHS
        },
    )


def build(
    k: SimpleNamespace,
    *,
    expert_offset: int,
    routed_scale: float,
    n_group: int,
    topk_group: int,
    topk: int,
    num_layers: int,
    wide_cells: bool,
    dec_unpad: bool,
    acc_padded: bool,
    router_proj: ProjectionHandle,
    route: SigmoidTopkHandle,
    copy: RowCopyHandle,
    add: HiddenAddHandle,
    count: Count,
    mtp_bf16: SimpleNamespace | None = None,
) -> MoeHandle:
    """The routed ops' handle over the kernels `build_kernels` built; compiles nothing."""
    return MoeHandle(
        k=k,
        expert_offset=expert_offset,
        routed_scale=routed_scale,
        n_group=n_group,
        topk_group=topk_group,
        topk=topk,
        num_layers=num_layers,
        wide_cells=wide_cells,
        dec_unpad=dec_unpad,
        acc_padded=acc_padded,
        router_proj=router_proj,
        route=route,
        copy=copy,
        add=add,
        count=count,
        mtp_bf16=mtp_bf16,
    )


# Row-count predicates: what each op takes at a row count, from what built at load.


def pair_ok(rows: int, h: MoeHandle) -> bool:
    """Whether `rows` takes the prefill grouped GEMM pair: a context row count above the
    decode kernels' 64."""
    return h.k.pair is not None and PREFILL_MIN <= rows <= MAX_ROWS


def decode_exact(rows: int, h: MoeHandle) -> bool:
    """Whether a wide decode step's routed experts run `rows` unpadded below 513 (an engine
    with the wide decode cells, `wide_cells`: 65..192 rows at MTP3 / MTP5 batch 32), on the
    prefill front end and pair. The prefill top-k writes the maps for any row count, so
    nothing is padded, re-routed, cleared or merged: 192 rows ran as 256 (a third more M, the
    top-k's maps discarded for a routing launch, a clear and a merge). Context steps keep the
    power-of-two padding."""
    return (
        h.wide_cells
        and PREFILL_MIN <= rows < PAIR_EXACT_FROM
        and h.k.topk_pre is not None
        and pair_ok(rows, h)
    )


def padded_rows(rows: int, h: MoeHandle, exact: bool = False) -> int:
    """The row count routing and the grouped GEMMs see for `rows`: a power of two, or the
    exact count on the prefill pair from `PAIR_EXACT_FROM` up, or on a wide decode step
    (`exact`, see `decode_exact`). `moe_open` and `moe_experts` both ask here, so the shared
    expert's rows and FC2's agree."""
    if (exact or rows >= PAIR_EXACT_FROM) and pair_ok(rows, h):
        return rows
    return pow2(rows)


def dec_unpadded(rows: int, h: MoeHandle) -> bool:
    """Whether a trunk MoE layer's decode width `rows` (1..64) runs unpadded on the fused
    top-k's maps (`TRTLLM_GLM52_MOE_DEC_UNPAD`). Only the trunk asks: there the collective
    writes the NVFP4 input (`allreduce_add_rmsnorm_quant`) and the shared expert the output's
    partial (`moe_open`), so no per-width quantize or clear runs (both are compiled for the
    powers of two only). The MTP layer quantizes and clears itself and keeps the padding.
    Every decode width has its own top-k handle (`k.topk`, `k.topk_rep`), whose maps the
    grouped GEMMs take at that width: they are compiled for the capacity and read the live
    tile count from the maps."""
    return h.dec_unpad and 1 <= rows <= DECODE_MAX_ROWS and rows in h.k.topk


def _topk_wide(rows: int, h: MoeHandle):
    """The wide decode top-k handle for `rows` (65..256), or None (not built: MTP1, no
    speculation, or a failed build; or rows outside its domain)."""
    if not _moe_topk_op.WIDE_MIN_ROWS <= rows <= _moe_topk_op.WIDE_MAX_ROWS:
        return None
    return h.k.topk_wide.get(_moe_topk_op.wide_range(rows))


def _topk_wide_rep(rows: int, layer: int, h: MoeHandle):
    """`(handle, hot ids)` of the wide replica top-k for a wide decode step's `rows` at trunk
    MoE layer `layer`, or None: not built (MTP1, no speculation, replicas off) or a layer
    without hot ids (the MTP layer)."""
    reps = h.k.topk_wide_rep
    hot = h.k.ctx_hot_ids.get(layer) if reps else None
    if hot is None:
        return None
    handle = reps.get(_moe_topk_op.wide_range(rows))
    return None if handle is None else (handle, hot)


def _quant_first_ok(x: torch.Tensor, rows: int, h: MoeHandle) -> bool:
    """Whether `moe_experts` would quantize `x` itself, unstaged and unpadded, on the prefill
    quantize: then `moe_quantize_first` may issue that quantize ahead of the router."""
    return (
        h.k.quant_pre is not None
        and pair_ok(rows, h)
        and padded_rows(rows, h) == rows
        and x.dtype is torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and not torch.cuda.is_current_stream_capturing()
    )


# The layer's bookkeeping: no kernel launches.


def moe_exact(rows: int, h: MoeHandle) -> bool:
    """`decode_exact` for an MTP3 / MTP5 wide decode step's trunk MoE layer, counted."""
    exact = decode_exact(rows, h)
    if exact:
        h.count("moe_dec_exact", True)
    return exact


def moe_open(rows: int, exact: bool, h: MoeHandle, parity: SlotParity):
    """Open a trunk MoE layer: `(slot, acc)`. Up to `MAX_ROWS` rows the layer advances the
    slot ahead of the shared branch (`slot`, which every routed op of the layer takes), and
    `acc` is the routed output rows the shared expert writes into (None where FC2 does not
    accumulate onto it); above, `slot` is None (each chunk takes its own, `parity.take()`)
    and `acc` None. FC2 accumulates at a width the GEMMs do not pad: a power of two, or any
    count on the prefill pair from 513 (or on a wide decode step, `exact`); and at a padded
    decode width up to 64 (`acc_padded`).

    A padded width is safe for the same reasons as the others: row `i` of the shared expert
    is token `i`, and FC2 adds into `out[expanded_idx // top_k]` for the selections the maps
    hold, nothing else. The padding tokens follow the real ones and route to a non-local
    expert (`pad_expert_id`), so no tile scatters into their rows, which are neither cleared
    nor written and are sliced off. The ordering is unchanged: the shared expert writes on
    the side stream, which FC2 joins (`moe_experts`), and the slot alternates per layer as at
    every other width. What a padded width adds (the staging copies, the routing kernel, the
    top-k's late FC1 release) does not touch the output."""
    if rows > MAX_ROWS:
        return None, None
    slot = parity.flip()
    if padded_rows(rows, h, exact) != rows and not dec_unpadded(rows, h):
        if rows > DECODE_MAX_ROWS or not h.acc_padded:
            return slot, None
        h.count("moe_acc_padded", True)
    return slot, h.k.slots[slot].out[:rows]


def mtp_moe_open(rows: int, wide: bool, h: MoeHandle) -> bool:
    """Open the MTP layer's routed experts on its handle; returns `exact`: a wide decode
    step's (`wide`) exact rows (`decode_exact`), only on the trunk's NVFP4 path. Counts the
    NVFP4 path (`mtp_moe_fp4`) and the exact rows."""
    if h.mtp_bf16 is not None:
        return False
    h.count("mtp_moe_fp4", True)
    exact = wide and decode_exact(rows, h)
    if exact:
        h.count("moe_dec_exact", True)
    return exact


# The ops.


class MoeL2Prefetch(OpWrapper):
    """Fork the L2 prefetch stream at a decode MoE layer's o_proj end and prefetch the FC1
    slices of the experts this layer's previous decode step routed the most tokens to;
    returns the stream `moe_experts` joins before FC2 (None: nothing launched). Only on the
    push top-k's widths (1..64 rows). Not on the MTP3 / MTP5 wide decode steps (`exact`):
    there the prefetch costs time (4-GPU chain with the real all-reduce, 2 prefetched MoE
    layers: +4.3 us at 128 rows, +2.6 us at 192, 200 alternating replays each)."""

    def __call__(self, layer: int, m, rows: int, exact: bool, h: MoeHandle):
        pf = h.k.pf
        if pf is None or exact:
            return None
        if not 1 <= rows <= DECODE_MAX_ROWS or not h.k.topk[rows].expert_hist:
            return None
        pf.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(pf.stream):
            # The home prefix: the prefetch ranks this rank's 64 home experts (replica slots,
            # when present, follow them in the same storage).
            e = _moe_routing_op.LOCAL_EXPERTS
            cute_dsl_moe_l2_prefetch(pf.hist[layer], m.fc1_w[:e], m.fc1_s[:e], pf.compiled)
        h.count("moe_l2_prefetch", True)
        return pf.stream

    def reference(self, layer: int, m, rows: int, exact: bool, h: MoeHandle):
        """Nothing: the prefetch only warms L2."""
        return None


class MoeQuantizeFirst(OpWrapper):
    """The NVFP4 quantize of a chunk `x` ahead of the router, on an exactly sized
    prefill-pair chunk: before the shared expert's GEMM on the side stream takes the SMs, so
    the latency-bound top-k overlaps the shared down instead (-16 to -22 us per MoE layer at
    7-8k rows on B300). Same kernel and bytes as `moe_experts`' own quantize. Returns whether
    the NVFP4 input now sits in the handle's `xq` / `xsf` (`pre`: the collective wrote it)."""

    def __call__(self, x: torch.Tensor, m, pre: bool, h: MoeHandle) -> bool:
        if pre:
            return True
        rows = int(x.shape[0])
        if h.mtp_bf16 is not None or not _quant_first_ok(x, rows, h):
            return False
        cute_dsl_moe_quantize_nvfp4(x, m.g1, h.k.xq[:rows], h.k.xsf[:rows], h.k.quant_pre)
        h.count("moe_quant_first", True)
        return True

    def reference(self, x: torch.Tensor, m, pre: bool, h: MoeHandle) -> bool:
        """Whether the input is quantized ahead: the reference quantizes nothing."""
        return pre


def _router_kernel(x: torch.Tensor, m, slot: int, exact: bool, h: MoeHandle):
    """Router logits in the slot's buffer on a router kernel, or None (none takes the call)."""
    rows = int(x.shape[0])
    k = h.k
    s = k.slots[slot]
    if rows > DECODE_MAX_ROWS:
        pre = k.router_pre
        wide = exact and k.router_wide is not None and _topk_wide(rows, h) is not None
        if (pre is None and not wide) or rows > MAX_ROWS:
            return None
        if not _direct_ok(x):
            x = x.contiguous()
        if x.dtype is not torch.bfloat16 or x.data_ptr() % 16:
            return None
        if wide:
            out = s.router_wide_out[:rows]
            cute_dsl_moe_router(x, m.router, out, k.router_wide)
            h.count("moe_router_wide", True)
            return out
        out = s.logits[:rows]
        cute_dsl_moe_router(x, m.router, out, pre)
        return out.view(rows, -1)
    if _direct_ok(x):
        xs = x
    else:
        xs = s.router_x[:rows]
        row_copy(xs, x, h.copy)
    out = s.router_out[:rows] if rows <= _moe_router_op.SIMT_MAX_ROWS else s.router_partials[:rows]
    cute_dsl_moe_router(xs, m.router, out, k.router[rows])
    return out


class MoeRouter(OpWrapper):
    """Router logits `(logits, in_slot)`, fp32: FC2 reads `token_final_scales` as fp32 and
    noaux_tc_op's weight dtype follows its logits (bf16 runs clean on garbage).

    Up to 64 rows on the decode router, where above 5 rows these are `[rows, 4, 256]` partials
    only the fused top-k reads; on a wide decode step (`exact`) the same partials from the
    wide decode router when it and the wide top-k built; above, on the prefill router
    (`[rows, 256]`). `in_slot`: the logits are in the slot's buffer. Where no router kernel
    takes the call, the router projection (`in_slot` False). On the MTP layer's bf16 pair: up
    to 64 rows nothing (`moe_experts` runs its front end on the padded rows, logits None),
    above the router projection."""

    def __call__(self, ph, x: torch.Tensor, m, slot: int, exact: bool, h: MoeHandle):
        rows = int(x.shape[0])
        if h.mtp_bf16 is not None:
            if rows <= DECODE_MAX_ROWS:
                return None, False
            return projection(ph, x, m.router_t, h.router_proj, torch.float32), False
        logits = _router_kernel(x, m, slot, exact, h)
        if logits is None:
            return projection(ph, x, m.router_t, h.router_proj, torch.float32), False
        return logits, True

    def reference(self, ph, x: torch.Tensor, m, slot: int, exact: bool, h: MoeHandle):
        """`x @ router_t` in fp32, `[rows, experts]`."""
        return (x.float() @ m.router_t.float()), False


def _topk_kernel(logits, in_slot: bool, m, slot: int, layer: int, exact: bool, h: MoeHandle):
    """A fused top-k's `MoeSelection`, or None (none takes the call)."""
    rows = int(logits.shape[0])
    k = h.k
    s = k.slots[slot]
    pf = k.pf
    sig = None
    if dec_unpadded(rows, h) and layer < h.num_layers:
        sig = k.topk_sig.get(rows)
    signaled = sig is not None
    if rows in k.topk_rep:
        # Decode expert replicas: this layer's hot ids (none on the MTP layer).
        handle = sig if sig is not None else k.topk_rep[rows]
        hot = k.dec_hot_ids.get(layer, k.dec_hot_none)
        if hot is not k.dec_hot_none:
            h.count("moe_dec_replica", True)
        w, ids, maps = cute_dsl_moe_topk_routing_rep(
            logits,
            m.router_bias,
            hot,
            h.expert_offset,
            h.routed_scale,
            s.topk_out[rows],
            handle,
            expert_hist=pf.hist[layer] if handle.expert_hist else None,
        )
        return MoeSelection(w, ids, maps, signaled)
    if rows <= DECODE_MAX_ROWS:
        handle = sig if sig is not None else k.topk[rows]
        w, ids, maps = cute_dsl_moe_topk_routing(
            logits,
            m.router_bias,
            h.expert_offset,
            h.routed_scale,
            s.topk_out[rows],
            None,
            handle,
            expert_hist=pf.hist[layer] if handle.expert_hist else None,
        )
        return MoeSelection(w, ids, maps, signaled)
    if k.topk_pre is None or rows > MAX_ROWS:
        return None
    if not in_slot:
        lg = s.logits[:rows].view(rows, -1)
        row_copy(lg, logits.reshape(rows, -1), h.copy)
        logits = lg
    wide = _topk_wide(rows, h) if exact else None
    planes = 1 if logits.dim() == 2 else int(logits.shape[1])
    if wide is not None and wide.logit_splits == planes:
        rep = _topk_wide_rep(rows, layer, h)
        if rep is not None and rep[0].logit_splits == planes:
            # Wide decode expert replicas: the same selection, maps over this rank's home and
            # replica slots; `moe_experts` runs them on the replica pair. Every rank takes this
            # branch for the same rows and layer.
            handle, hot = rep
            h.count("moe_wide_replica", True)
            if handle.cluster == 16:
                h.count("moe_topk_wide_cl16", True)
            outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.maps}
            w, ids, maps = cute_dsl_moe_topk_routing_wide_rep(
                logits, m.router_bias, hot, h.expert_offset, h.routed_scale, outputs,
                handle, expert_hist=pf.hist[layer] if handle.expert_hist else None,
            )  # fmt: skip
            return MoeSelection(w, ids, maps, rep=True)
        # One fused launch for the three of the prefill top-k (histogram, prefix, scatter);
        # the maps it writes are FC1's as they stand.
        h.count("moe_topk_wide", True)
        if wide.cluster == 16:
            h.count("moe_topk_wide_cl16", True)
        outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.maps}
        w, ids, maps = cute_dsl_moe_topk_routing(
            logits, m.router_bias, h.expert_offset, h.routed_scale, outputs, None, wide,
            expert_hist=pf.hist[layer] if wide.expert_hist else None,
        )  # fmt: skip
        return MoeSelection(w, ids, maps)
    hot = k.ctx_hot_ids.get(layer) if k.topk_pre_rep is not None else None
    if (
        hot is not None
        and not exact
        and _moe_topk_pre_rep_op.serves(rows)
        and padded_rows(rows, h) == rows
        and pair_ok(rows, h)
    ):
        # Context replicas: only where the GEMMs run these maps unpadded on the pair. Every
        # rank takes the same branch for the same rows and layer.
        outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.topk_rep_scratch, **s.maps}
        w, ids, maps = cute_dsl_moe_prefill_topk_routing_rep(
            logits, m.router_bias, hot, h.expert_offset, h.routed_scale, outputs,
            k.topk_pre_rep,
        )  # fmt: skip
        h.count("ctx_ar_rep", True)
        return MoeSelection(w, ids, maps, rep=True)
    outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.topk_scratch, **s.maps}
    w, ids, maps = cute_dsl_moe_topk_routing(
        logits, m.router_bias, h.expert_offset, h.routed_scale, outputs, None, k.topk_pre
    )
    return MoeSelection(w, ids, maps)


class MoeTopk(OpWrapper):
    """Sigmoid top-k with bias correction, writing this layer's tile maps: a `MoeSelection`
    in the slot's buffers. Up to 64 rows on the decode kernel; above, on the wide decode
    kernel for a wide decode step (`exact`, 65..256 rows, same ids, weights and tile layout
    as the prefill kernel's; a trunk layer with hot ids on its replica variant) and on the
    prefill one otherwise. `in_slot`: the logits are the slot's router output, read in place;
    others are staged into it first. Matches `noaux_tc_op` on this ungrouped routing: same
    sets, weights to 6e-8. A decode or wide handle built with `expert_hist` also writes
    `layer`'s row of the L2 prefetch's counts. Where no top-k kernel takes the call (and on
    the MTP layer's bf16 pair above 64 rows), `sigmoid_topk`, without maps; with no logits
    (the bf16 pair's front end, run by `moe_experts`), None."""

    def __call__(
        self, logits, in_slot: bool, m, slot: int, layer: int, exact: bool, h: MoeHandle
    ) -> MoeSelection | None:
        if logits is None:
            return None
        sel = (
            None
            if h.mtp_bf16 is not None
            else _topk_kernel(logits, in_slot, m, slot, layer, exact, h)
        )
        if sel is not None:
            return sel
        w, ids = sigmoid_topk(
            logits, m.router_bias, h.n_group, h.topk_group, h.topk, h.routed_scale, h.route
        )
        return MoeSelection(w, ids)

    def reference(
        self, logits, in_slot: bool, m, slot: int, layer: int, exact: bool, h: MoeHandle
    ) -> MoeSelection | None:
        """`sigmoid_topk`'s reference, without maps."""
        if logits is None:
            return None
        w, ids = sigmoid_topk.reference(
            logits, m.router_bias, h.n_group, h.topk_group, h.topk, h.routed_scale, h.route
        )
        return MoeSelection(w, ids)


def _pad_selection(ids, w, pad: int, h: MoeHandle):
    """`ids` / `w` with `pad` padding rows routed to a non-local expert at weight 0
    (`pad_expert_id`), so routing drops them."""
    dev = ids.device
    pad_id = _moe_routing_op.pad_expert_id(h.expert_offset)
    ids = torch.cat([ids, torch.full((pad, TOP_K), pad_id, dtype=ids.dtype, device=dev)])
    w = torch.cat([w, torch.zeros(pad, TOP_K, dtype=w.dtype, device=dev)])
    return ids, w


def _mtp_bf16_experts(x, sel: MoeSelection | None, m, slot: int, h: MoeHandle):
    """The MTP layer's bf16 round trip of one chunk; returns the slot's output rows. With no
    selection (up to 64 rows) the router and the fused top-k (which also clears the output)
    run here on the padded input in place; otherwise the selection's ids and weights select,
    staged with the input into the slot's buffers."""
    k, b = h.k, h.mtp_bf16
    rows = int(x.shape[0])
    dev = x.device
    padded = pow2(rows)
    topk_ids = topk_w = None
    if sel is not None:
        topk_ids, topk_w = sel.ids, sel.w
    if padded != rows:
        pad = padded - rows
        x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
        if sel is not None:
            topk_ids, topk_w = _pad_selection(topk_ids, topk_w, pad, h)
    s = b.slots[slot]
    out = s.out[:padded]
    if sel is None:
        # The router and gather GEMM read the activation in place; weights and maps come from
        # the top-k's outputs, and the top-k zeroed the output.
        if padded <= _moe_router_op.SIMT_MAX_ROWS:
            logits = s.router_out[:padded]
        else:
            logits = s.router_partials[:padded]
        cute_dsl_moe_router(x, m.router, logits, k.router[padded])
        topk_w, _, maps = cute_dsl_moe_topk_routing(
            logits,
            m.router_bias,
            h.expert_offset,
            h.routed_scale,
            s.topk_out[padded],
            out,
            b.topk[padded],
        )
        xs = x
    else:
        # The routing kernel does not wait before reading the ids, and `noaux_tc_op` releases
        # it before its stores are visible: these copies are what order them.
        xs = s.x[:padded]
        row_copy(xs, x, h.copy)
        w_s = s.w[:padded]
        row_copy(w_s, topk_w, h.copy)
        topk_w = w_s
        ids = s.sel[: padded * TOP_K]
        row_copy(ids, topk_ids.reshape(-1), h.copy)
        maps = cute_dsl_moe_routing(ids, h.expert_offset, s.maps, s.counts, s.barrier, k.routing)
        out.zero_()
    cute_dsl_mtp_moe_experts(xs, maps, topk_w, m.fc1, m.fc2, out, s.c, b.experts)
    return out[:rows]


class MoeExperts(OpWrapper):
    """One chunk's routed round trip (NVFP4 FC1 + FC2 grouped GEMMs); returns the slot's
    output rows.

    `sel`: `moe_topk`'s selection; its maps are trusted only at an unpadded row count (maps
    built for another row count are an illegal address in the grouped GEMMs, not a wrong
    answer). `pre`: the NVFP4 input already sits in `xq` / `xsf`. `acc`: `moe_open`'s output
    rows, already holding the shared expert's partial, or None (FC2 writes a cleared output).
    `join`: the shared branch's stream, waited on (only with `acc`: it writes that partial)
    with `prefetch`, the L2 prefetch stream (`moe_l2_prefetch`), between FC1 and FC2 so both
    overlap router, top-k and FC1. `layer` is this call site's slot of the decode FC2's tile
    counter: a launch of up to 64 tokens assigns its tiles dynamically, each layer once per
    forward. `exact`: a wide decode step (`decode_exact`), unpadded below 513 rows.

    Fallback chain where the fast path does not serve the rows: pad to a power of two below
    513 rows (the routing schedule is keyed by `ceil(rows / 32)` there, and arbitrary widths
    would compile about twenty; padding rows route to a non-local expert at weight 0 and are
    sliced off), stage the input and the selection into the slot, run the routing kernel,
    quantize (decode, prefill or stock quantize) and clear the output. The prefill pair takes
    513 rows and up exactly, and a wide decode step any count. On the MTP layer's bf16 pair
    the whole chunk runs on its bf16 path."""

    def __call__(
        self,
        x: torch.Tensor,
        sel: MoeSelection | None,
        m,
        pre: bool,
        slot: int,
        acc: torch.Tensor | None,
        join,
        prefetch,
        layer: int,
        exact: bool,
        h: MoeHandle,
    ) -> torch.Tensor:
        if h.mtp_bf16 is not None:
            return _mtp_bf16_experts(x, sel, m, slot, h)
        k = h.k
        rows = int(x.shape[0])
        dev = x.device
        topk_ids, topk_w, topk_maps = sel.ids, sel.w, sel.maps
        use_pair = pair_ok(rows, h)
        padded = padded_rows(rows, h, exact)
        acc = acc is not None
        if padded != rows and acc and pre and topk_maps is not None and dec_unpadded(rows, h):
            # A trunk decode width that is not a power of two: the top-k's maps at the exact
            # rows, no padding, staging or routing launch (`dec_unpadded`).
            padded = rows
            h.count("moe_dec_unpad", True)
        if sel.signaled:
            if padded != rows:
                # Fail fast: a routing launch after an early-release top-k reads the ids
                # mid-write (a hang, not a wrong answer).
                raise RuntimeError(
                    f"modeling_v2 GLM-5.2: early-release top-k at {rows} rows took the padded path"
                )
            h.count("moe_dec_unpad_signal", True)
        if padded != rows:
            pad = padded - rows
            if not pre:
                x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
            topk_ids, topk_w = _pad_selection(topk_ids, topk_w, pad, h)
        s = k.slots[slot]
        if not pre:
            # An activation the engine allocated inside a CUDA-graph capture is not safe to
            # hand over as a raw pointer, so a captured call stages it. An eager call (context
            # and mixed steps) whose `x` already is the contiguous, 16-byte aligned bf16
            # `[padded, hidden]` operand goes to the quantize directly, which takes it per
            # call: a `[T, 6144]` copy per MoE layer skipped (98 MB at T ~ 8k; 3.0-3.4 ms of
            # copies per context step, measured in a served run).
            if (
                x.dtype is torch.bfloat16
                and x.is_contiguous()
                and x.data_ptr() % 16 == 0
                and tuple(x.shape) == tuple(s.x[:padded].shape)
                and not torch.cuda.is_current_stream_capturing()
            ):
                h.count("moe_stage_skip", True)
            else:
                x_s = s.x[:padded]
                row_copy(x_s, x, h.copy)
                x = x_s
        # fp32, as both top-k kernels write it and FC2 reads it: a top-k kernel's own output
        # at an unpadded width goes in as it is.
        if topk_maps is None or padded != rows:
            w_s = s.w[:padded]
            row_copy(w_s, topk_w, h.copy)
            topk_w = w_s
        if topk_maps is not None and padded == rows:
            maps = topk_maps
        else:
            # Always staged: the routing kernels launch with PDL and do not wait for their
            # inputs, and the fused top-k signals its dependents before it writes the ids.
            # The copy puts a node no PDL edge can cross between the two.
            ids = s.sel[: padded * TOP_K]
            row_copy(ids, topk_ids.reshape(-1), h.copy)
            maps = cute_dsl_moe_routing(
                ids, h.expert_offset, s.maps, s.counts, s.barrier, k.routing
            )
        xq, xsf = k.xq[:padded], k.xsf[:padded]
        if pre:
            # Written by the collective; the padding rows are stale by design: routed to a
            # non-local expert, no grouped GEMM tile reads them.
            pass
        elif padded <= DECODE_MAX_ROWS:
            cute_dsl_moe_quantize_nvfp4(x, m.g1, xq, xsf, k.quant[padded])
        elif k.quant_pre is not None and padded <= MAX_ROWS:
            cute_dsl_moe_quantize_nvfp4(x, m.g1, xq, xsf, k.quant_pre)
        else:
            xq, xsf = fp4_quantize(x, m.g1, SF_VEC, False, False)
        # bf16: FC2 writes through a BFloat16 pointer whatever buffer it is handed.
        out = s.out[:padded]
        if not acc:
            if padded <= DECODE_MAX_ROWS:
                cute_dsl_moe_zero(out, k.zero[padded])
            elif k.zero_pre is not None and padded <= MAX_ROWS:
                cute_dsl_moe_zero(out, k.zero_pre)
            else:
                out.zero_()
        # Context replicas: maps the replica top-k wrote address slots up to 64 + R - 1 and
        # run on the replica pair; any other maps are the home mapping.
        rep = use_pair and sel.rep and maps is topk_maps and k.pair_rep is not None
        compiled, tiles = (k.pair_rep if rep else k.pair, s.pc) if use_pair else (k.experts, s.c)
        fc1_w, fc1_s, fc2_w, fc2_s = m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s
        alpha1, alpha2 = m.alpha1, m.alpha2
        e = compiled.experts if use_pair else fc1_w.shape[0]
        if fc1_w.shape[0] != e:
            # Decode expert replicas: the prefill pair maps home experts (or the first R
            # replica slots too, `rep`) and reads that prefix of the stacks (a contiguous view
            # of the same storage).
            fc1_w, fc1_s, fc2_w, fc2_s = fc1_w[:e], fc1_s[:e], fc2_w[:e], fc2_s[:e]
            alpha1, alpha2 = alpha1[:e], alpha2[:e]
        # The decode L2 prefetch forked at the o_proj end joins before FC2, with the shared
        # expert's stream where FC2 accumulates onto its partial.
        if not acc:
            join = None
        if prefetch is not None:
            join = (join, prefetch) if join is not None else prefetch
        cute_dsl_moe_nvfp4_experts(
            xq, xsf, maps, topk_w, fc1_w, fc1_s, fc2_w, fc2_s,
            m.norm_const, alpha1, alpha2, out, tiles, s.sfc, join, compiled,
            None if use_pair else layer,
        )  # fmt: skip
        return out[:rows]

    def reference(
        self,
        x: torch.Tensor,
        sel: MoeSelection | None,
        m,
        pre: bool,
        slot: int,
        acc: torch.Tensor | None,
        join,
        prefetch,
        layer: int,
        exact: bool,
        h: MoeHandle,
    ) -> torch.Tensor:
        """Not expressible in plain torch here: the experts' weights are packed NVFP4 for the
        grouped GEMMs alone."""
        raise NotImplementedError("moe_experts: the packed NVFP4 stacks have no torch reference")


class MoeCombine(OpWrapper):
    """The MoE layer's output: the routed rows as they stand where FC2 accumulated onto the
    shared expert's partial (`acc`), else joined with the shared branch (`side`) and added,
    on the merge kernel where it takes the rows, else `hidden_add`."""

    def __call__(self, routed, shared, acc, rows: int, main, side, h: MoeHandle):
        if acc is not None:
            return routed
        main.wait_stream(side)
        merge = h.k.merge
        if (
            merge is not None
            and PREFILL_MIN <= rows <= MAX_ROWS
            and routed.shape == shared.shape
            and routed.is_contiguous()
        ):
            if not shared.is_contiguous():
                packed = torch.empty_like(shared, memory_format=torch.contiguous_format)
                shared = row_copy(packed, shared, h.copy)
            return cute_dsl_moe_merge(routed, shared, merge)
        return hidden_add(routed, shared, h.add)

    def reference(self, routed, shared, acc, rows: int, main, side, h: MoeHandle):
        """`routed + shared` in fp32, rounded once (`routed` alone where it accumulated)."""
        if acc is not None:
            return routed
        return (routed.float() + shared.float()).to(routed.dtype)


moe_l2_prefetch = MoeL2Prefetch()
moe_quantize_first = MoeQuantizeFirst()
moe_router = MoeRouter()
moe_topk = MoeTopk()
moe_experts = MoeExperts()
moe_combine = MoeCombine()
