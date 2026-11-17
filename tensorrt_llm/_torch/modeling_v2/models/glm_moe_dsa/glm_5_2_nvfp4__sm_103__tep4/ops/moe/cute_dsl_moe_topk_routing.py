# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's sigmoid top-k with the routing maps fused in: the selected experts, their
weights and the grouped GEMMs' six tile maps in one CuTe DSL op. A decode kernel per row
count (1..64), a wide decode kernel per row range for the MTP3 / MTP5 decode widths
(65..256), and a three-stage prefill kernel (65..8192)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe import moe_prefill_frontend_schedule as _sched
from ...kernels.moe import moe_prefill_topk_routing as _prefill
from ...kernels.moe import moe_prefill_topk_routing_noraw as _prefill_noraw
from ...kernels.moe import moe_topk_routing as _decode
from ...kernels.moe import moe_topk_routing_push as _push
from ...kernels.moe import moe_topk_routing_push_wide as _wide
from .cute_dsl_moe_routing import (
    LOCAL_EXPERTS,
    MAP_KEYS,
    TILE,
    TOP_K,
    allocate_maps,
    reference_maps,
)

EXPERTS = 256
HIDDEN = 6144
DECODE_MAX_ROWS = 64
PREFILL_MIN_ROWS = DECODE_MAX_ROWS + 1
PREFILL_MAX_ROWS = 8192
#: The prefill histogram stage is compiled for this many producer CTAs, and its per-CTA
#: count slab (`partial_counts`) is sized by it. A kernel constant, not the SM count.
PREFILL_PRODUCER_CTAS = 148
#: The fused output-zero plane uses 128-bit stores.
_ZERO_ALIGN = 16

#: Whether a decode handle at `rows` rows runs the push cluster path (9..64 rows): the widths
#: that can also write the per-expert counts (`build_decode(expert_hist=True)`).
uses_push_path = _push.uses_push_path
#: What a decode `outputs` set holds, in the kernel's order (`allocate_decode_outputs`).
DECODE_OUTPUTS = _decode.OUTPUT_ORDER
#: What a prefill `outputs` set holds: the top-k pair, the histogram scratch, the maps.
PREFILL_OUTPUTS = ("topk_weights", "topk_indices", "partial_counts", "expert_bases", *MAP_KEYS)
#: The wide decode kernel's row domain (MTP3 / MTP5 decode widths): one 8-CTA cluster of 16,
#: 24 or 32 token-warps per CTA, one compile per row range (`WIDE_RANGES`), the row count a
#: runtime argument. Its `outputs` are the decode kernel's eight, sized like a prefill set's.
WIDE_MIN_ROWS = _wide.MIN_TOKENS
WIDE_MAX_ROWS = _wide.MAX_TOKENS
WIDE_RANGES = tuple((8 * (w - 8) + 1, 8 * w) for w in _wide.WARP_COUNTS)
#: The wide kernel's cluster sizes: the portable 8 CTAs, or 16 (half the token-warps per CTA
#: on twice the SMs; same selection, counts and tile layout).
WIDE_CLUSTERS = _wide.CLUSTERS
#: The wide kernel's schedules `(cluster, trim)`: the served 8-CTA one, and the 16-CTA cluster
#: with the trimmed winner lookup (`TRTLLM_GLM52_WIDE_TOPK_CLUSTER16`). Bitwise the same.
WIDE_SCHEDULES = ((8, False), (16, True))


def wide_range(rows: int) -> tuple[int, int]:
    """The `(first, last)` row range of the wide handle serving `rows` (65..256)."""
    w = _wide.warps_for_tokens(int(rows))
    return 8 * (w - 8) + 1, 8 * w


def allocate_decode_outputs(rows: int, device) -> dict[str, torch.Tensor]:
    """One decode output set for exactly `rows` tokens: the weights and ids, and the six
    maps sized for `rows` (at most 64 tiles, one per active expert)."""
    return _decode.allocate_outputs(int(rows), device)


def allocate_prefill_scratch(device) -> dict[str, torch.Tensor]:
    """The prefill kernel's histogram scratch for one `outputs` set: the per-producer-CTA
    count slab and the 64 expert bases. The caller adds the top-k pair and a map set
    (`allocate_maps`) sized for the row budget."""
    i32 = {"dtype": torch.int32, "device": device}
    return {
        "partial_counts": torch.zeros(PREFILL_PRODUCER_CTAS * LOCAL_EXPERTS, **i32),
        "expert_bases": torch.zeros(LOCAL_EXPERTS, **i32),
    }


@dataclass(frozen=True)
class CompiledTopkRouting:
    """A compiled top-k/routing kernel.

    Decode: `rows` is the exact row count, `logit_splits` the K partials it sums (1, or the
    router's 4), and the PDL roles and zero plane are compiled in. Wide: `rows` is the last
    row count of its range (`wide_range`), any count of which it runs; no zero plane.
    Prefill: `rows` None (any count in 65..8192), one logit plane, PDL as the engine sets
    it, no zero plane.
    `launch(logits, bias, local_expert_offset, routed_scale, outputs, zero_output,
    expert_hist)` is the kernel call; `expert_hist` is the int32 `[64]` row a handle built
    with `expert_hist=True` also writes (None otherwise).
    """

    kind: str
    rows: int | None
    logit_splits: int
    signal_gg1: bool
    fuse_zero: bool
    launch: Callable[..., None]
    expert_hist: bool = False
    cluster: int = 8
    trim: bool = False


def _decode_launch(
    compiled, fuse_zero: bool, push: bool = False, hist: bool = False
) -> Callable[..., None]:
    as_t = _decode._as_tensor
    # The push kernel loads its logits 128 bits at a time; its zero slot is the logits.
    zero_align = _ZERO_ALIGN if fuse_zero or push else 4
    logits_align = 16 if push else 4

    def launch(
        logits, bias, local_expert_offset, routed_scale, outputs, zero_output, expert_hist=None
    ):
        # Without the zero plane the kernel's ABI still takes a tensor there, unread;
        # the logits stand in, as the kernel's own runner passes them.
        zero_arg = zero_output if fuse_zero else logits
        compiled(
            as_t(logits.reshape(-1), assumed_align=logits_align),
            as_t(bias),
            as_t(zero_arg.reshape(-1), assumed_align=zero_align),
            *(as_t(outputs[name]) for name in DECODE_OUTPUTS),
            cutlass.Int32(int(logits.shape[0])),
            cutlass.Int32(local_expert_offset),
            cutlass.Float32(routed_scale),
            cutlass_torch.current_stream(),
            *((as_t(expert_hist, assumed_align=16),) if hist else ()),
        )

    return launch


def build_decode(
    rows: int,
    logit_splits: int,
    device,
    *,
    signal_gg1: bool,
    fuse_zero: bool = False,
    expert_hist: bool = False,
) -> CompiledTopkRouting:
    """Compile the fused decode top-k for exactly `rows` rows (1..64) reading
    `logit_splits` K partials (1: `[rows, 256]` logits; 4: the tensor-core router's
    `[rows, 4, 256]`). Call at load. Compiles only; nothing is launched.

    It always runs as the router's PDL dependent (waits before its first logits load).
    `signal_gg1` releases the next grouped GEMM early: only where nothing between the
    two reads what this writes (see `note`). `fuse_zero` also clears a bf16 `[rows,
    6144]` output (the MTP layer's, which has no quantize launch to carry the clear).
    `expert_hist` (the push widths, 9..64 rows): the handle also writes the step's token
    count per local expert into an int32 `[64]` row the call passes, which the next decode
    step's L2 prefetch of the same layer ranks experts by (`cute_dsl_moe_l2_prefetch`).
    """
    rows = int(rows)
    shape = (rows, EXPERTS) if logit_splits == 1 else (rows, int(logit_splits), EXPERTS)
    logits = torch.zeros(shape, dtype=torch.float32, device=device)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    zero = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device) if fuse_zero else None
    outputs = allocate_decode_outputs(rows, device)
    signal = bool(_USE_PDL and signal_gg1)
    # The cluster widths (9..64 rows) hand the histogram off by st.async instead of two
    # cluster barriers (`moe_topk_routing_push`, same outputs and maps bit for bit).
    push = _push.uses_push_path(rows, fuse_zero)
    if expert_hist and not push:
        raise ValueError(f"expert_hist is written on the push widths only, not at {rows} rows")
    hist = {"expert_hist": torch.zeros(64, dtype=torch.int32, device=device)} if expert_hist else {}
    with torch.cuda.device(device):
        compiled = (_push if push else _decode).compile_fused_topk_routing(
            logits,
            bias,
            zero,
            outputs,
            wait_for_router_pdl=_USE_PDL,
            signal_gg1_pdl=signal,
            fuse_zero=fuse_zero,
            **hist,
        )
    return CompiledTopkRouting(
        "decode",
        rows,
        int(logit_splits),
        signal,
        bool(fuse_zero),
        _decode_launch(compiled, fuse_zero, push, bool(expert_hist)),
        bool(expert_hist),
    )


def _wide_launch(compiled, hist: bool) -> Callable[..., None]:
    as_t = _decode._as_tensor

    def launch(
        logits, bias, local_expert_offset, routed_scale, outputs, zero_output, expert_hist=None
    ):
        # Flat views: the kernel indexes every output linearly, and the slot's weight
        # buffer is 2-D.
        compiled(
            as_t(logits.reshape(-1), assumed_align=16),
            as_t(bias),
            *(as_t(outputs[name].reshape(-1)) for name in DECODE_OUTPUTS),
            cutlass.Int32(int(logits.shape[0])),
            cutlass.Int32(local_expert_offset),
            cutlass.Float32(routed_scale),
            cutlass_torch.current_stream(),
            *((as_t(expert_hist, assumed_align=16),) if hist else ()),
        )

    return launch


def build_wide(
    rows: int,
    logit_splits: int,
    device,
    *,
    signal_gg1: bool = False,
    expert_hist: bool = False,
    cluster: int = 8,
    trim: bool = False,
) -> CompiledTopkRouting:
    """Compile the wide decode top-k for the row range holding `rows` (65..128, 129..192 or
    193..256, `wide_range`) reading `logit_splits` planes (1: the prefill router's logits; 2,
    4 or 8 K partials), on a `cluster`-CTA cluster (`WIDE_CLUSTERS`: 8, or 16 with half the
    token-warps per CTA), `trim`: with the trimmed winner lookup (one value fetch after the
    eight rounds). Call at load. Compiles only; nothing is launched.

    It waits for the router as its PDL dependent; `signal_gg1` releases the next grouped GEMM
    early (see `note`). `expert_hist`: also write the step's token count per local expert
    into an int32 `[64]` row (the L2 prefetch's, as the push widths do). Ids, weights and
    per-expert counts are the decode and prefill kernels' bit for bit; experts past 128 rows
    take two adjacent tiles.
    """
    _, hi = wide_range(rows)
    shape = (hi, EXPERTS) if logit_splits == 1 else (hi, int(logit_splits), EXPERTS)
    logits = torch.zeros(shape, dtype=torch.float32, device=device)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    outputs = {
        "topk_weights": torch.zeros(hi * TOP_K, dtype=torch.float32, device=device),
        "topk_indices": torch.zeros(hi * TOP_K, dtype=torch.int32, device=device),
        **allocate_maps(hi, device),
    }
    hist = torch.zeros(64, dtype=torch.int32, device=device) if expert_hist else None
    signal = bool(_USE_PDL and signal_gg1)
    with torch.cuda.device(device):
        compiled = _wide.compile_fused_topk_routing_wide(
            logits,
            bias,
            outputs,
            wait_for_router_pdl=_USE_PDL,
            signal_gg1_pdl=signal,
            expert_hist=hist,
            cluster=int(cluster),
            trim=bool(trim),
        )
    return CompiledTopkRouting(
        "wide",
        hi,
        int(logit_splits),
        signal,
        False,
        _wide_launch(compiled, bool(expert_hist)),
        bool(expert_hist),
        int(cluster),
        bool(trim),
    )


def _prefill_launch(handles: dict[int, Callable[..., None]]) -> Callable[..., None]:
    def launch(
        logits, bias, local_expert_offset, routed_scale, outputs, zero_output, expert_hist=None
    ):
        # The prefill kernels write no per-expert counts (`is_valid` refuses `expert_hist`).
        rows = int(logits.shape[0])
        h = handles[_sched.propose_schedule(rows).histogram_warps]
        n = rows * TOP_K
        # The tile records the kernel may write, its stated capacity `ceil(8T/128) + 63`:
        # at most 575 at 8192 rows, exactly `max_tiles(8192)`.
        q = (n + TILE - 1) // TILE + LOCAL_EXPERTS - 1
        h(
            logits.reshape(-1),
            bias,
            outputs["topk_weights"].reshape(-1)[:n],
            outputs["topk_indices"].reshape(-1)[:n],
            outputs["partial_counts"],
            outputs["expert_bases"],
            outputs["tile_idx_to_expert_idx"][:q],
            outputs["tile_idx_to_mn_limit"][:q],
            outputs["expanded_idx_to_permuted_idx"][:n],
            outputs["permuted_idx_to_expanded_idx"][: q * TILE],
            outputs["total_num_padded_tokens"],
            outputs["num_non_exiting_tiles"],
            rows,
            local_expert_offset,
            float(routed_scale),
        )

    return launch


def build_prefill(device) -> CompiledTopkRouting:
    """Compile the prefill top-k once per histogram width its row domain selects
    (`propose_schedule(rows).histogram_warps`: 4 warps up to 128 rows, 8 up to 512, 16
    above), one logit plane, PDL as the engine sets it; launch each once on random logits.
    Call at load. The handle picks the width per call by row count.

    The 16-warp width (above 512 rows) runs the histogram stage that recomputes each
    winner's sigmoid once per token instead of looking it up every selection round
    (`moe_prefill_topk_routing_noraw`): same arithmetic, prefix and scatter kernels, ids,
    weights and maps bit-identical; 11.4 -> 9.2 us at 7310 rows on B300.

    Raises if a compile fails: the target then selects with `noaux_tc_op` and builds the
    maps with `cute_dsl_moe_routing` at the call site.
    """
    first: dict[int, int] = {}
    for rows in range(PREFILL_MIN_ROWS, PREFILL_MAX_ROWS + 1):
        first.setdefault(_sched.propose_schedule(rows).histogram_warps, rows)
    handles: dict[int, Callable[..., None]] = {}
    launch = _prefill_launch(handles)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    with torch.cuda.device(device):
        for warps, rows in first.items():
            if warps == 16:
                handles[warps] = _prefill_noraw.compile_topk_routing_noraw(
                    logit_splits=1, warps=warps, noraw=True, use_pdl=_USE_PDL
                )
            else:
                handles[warps] = _prefill.compile_topk_routing(
                    logit_splits=1, warps=warps, use_pdl=_USE_PDL
                )
            outputs = {
                "topk_weights": torch.zeros(rows * TOP_K, dtype=torch.float32, device=device),
                "topk_indices": torch.zeros(rows * TOP_K, dtype=torch.int32, device=device),
                **allocate_prefill_scratch(device),
                # Sized for the budget, as the target's map sets are: the launch slices
                # the tile records to the kernel's stated capacity for these rows.
                **allocate_maps(PREFILL_MAX_ROWS, device),
            }
            logits = torch.randn(rows, 1, EXPERTS, dtype=torch.float32, device=device)
            launch(logits, bias, 0, 2.5, outputs, None)
    return CompiledTopkRouting("prefill", None, 1, False, False, launch)


def _sigmoid_topk(logits: torch.Tensor, bias: torch.Tensor, routed_scale: float):
    """`(weights fp32, ids int32, choice fp32)`, the kernels' selection arithmetic."""
    rows = int(logits.shape[0])
    lg = logits.float().reshape(rows, -1, EXPERTS)
    total = lg[:, 0]
    for s in range(1, lg.shape[1]):
        total = total + lg[:, s]
    scores = 0.5 * torch.tanh(0.5 * total) + 0.5
    choice = scores + bias.float()
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    w = torch.gather(scores, 1, ids).double()
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * float(routed_scale)
    return w.float(), ids.to(torch.int32), choice


def selection_margin(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Per row, the gap between the 8th and 9th selection scores, relative to the 8th.

    A row whose gap is within a few fp32 ulps can select either expert depending on the
    last bit of the sigmoid, which no reference fixes; the test exempts only those."""
    _, _, choice = _sigmoid_topk(logits, bias, 1.0)
    top = torch.topk(choice, TOP_K + 1, dim=-1).values
    return (top[:, TOP_K - 1] - top[:, TOP_K]) / top[:, TOP_K - 1].abs().clamp_min(1e-30)


class CuteDslMoeTopkRouting(OpWrapper):
    """Biased sigmoid top-8 of 256 experts, plus this rank's tile maps.

    Returns `(weights [rows, 8] fp32, ids [rows, 8] int32, maps)`: views into `outputs`.
    Selection uses `sigmoid(logit) + bias`; weights are the unbiased sigmoids of the
    winners, renormalized to sum to one and scaled by `routed_scale`. `ids` are global
    expert ids; `maps` are the six maps of `cute_dsl_moe_routing` for the window
    `[local_expert_offset, local_expert_offset + 64)`.

    `logits` is fp32 `[rows, 256]` or `[rows, splits, 256]` K partials (summed in
    increasing split order), `bias` fp32 `[256]`. `outputs` is the kernel's output set:

      decode   `allocate_decode_outputs(rows)`: the kernel's eight flat tensors.
      wide     `topk_weights` / `topk_indices` (>= rows * 8, flat or 2-D) and a map set
               from `allocate_maps` (other keys, a prefill set's scratch, are not read).
      prefill  `topk_weights` / `topk_indices` (>= rows * 8, flat or 2-D), the histogram
               scratch of `allocate_prefill_scratch`, and a map set from `allocate_maps`.

    `zero_output` is the bf16 `[rows, 6144]` buffer the decode kernel clears when compiled
    with `fuse_zero`, else None.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Decode: the block kernel (<= 8 rows) and the 8-CTA cluster at 64, the SIMT
    # router's 2-D logits and the tensor-core router's four partials, a width that does
    # not signal the grouped GEMM, and the MTP layer's zero-fusing variant. Wide: each
    # token-warp count at its row range's edges, MTP3's 128 and MTP5's 192 verify rows with
    # the per-expert counts, one and four logit planes, and a hot expert past 128 rows (two
    # tiles). Prefill: each histogram width at both edges of its row range, and the budget.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode row from the SIMT router's logits",
            spec=dict(rows=1, splits=1, signal_gg1=True, offset=0),
        ),
        Cell(
            why="3 rows, a padded width: compiled without the grouped GEMM's early release",
            spec=dict(rows=3, splits=1, signal_gg1=False, offset=64),
        ),
        Cell(
            why="8 rows, the block kernel's last count",
            spec=dict(rows=8, splits=4, signal_gg1=True, offset=0),
        ),
        Cell(
            why="12 rows summing four split-K partials on the 2-CTA cluster, not a power of two",
            spec=dict(rows=12, splits=4, signal_gg1=False, offset=128),
        ),
        Cell(
            why="the full decode batch on the 8-CTA cluster kernel",
            spec=dict(rows=64, splits=4, signal_gg1=True, offset=192),
        ),
        Cell(
            why="the full decode batch also writing the per-expert counts for the L2 prefetch",
            spec=dict(rows=64, splits=4, signal_gg1=True, offset=64, expert_hist=True),
        ),
        Cell(
            why="12 rows on the 2-CTA push path writing the per-expert counts",
            spec=dict(rows=12, splits=4, signal_gg1=False, offset=192, expert_hist=True),
        ),
        Cell(
            why="the MTP layer at width 2, clearing its output plane",
            spec=dict(rows=2, splits=1, signal_gg1=True, fuse_zero=True, offset=64),
        ),
        Cell(
            why="the MTP layer at 64 rows, clearing its output plane",
            spec=dict(rows=64, splits=4, signal_gg1=True, fuse_zero=True, offset=0),
        ),
        Cell(
            why="65 rows, the wide kernel's first: 16 token-warps per CTA",
            spec=dict(rows=65, splits=1, wide=True, offset=0),
        ),
        Cell(
            why="127 rows on the wide kernel summing four K partials",
            spec=dict(rows=127, splits=4, wide=True, offset=128),
        ),
        Cell(
            why="MTP3's 128 verify rows on the wide kernel writing the per-expert counts",
            spec=dict(rows=128, splits=1, wide=True, offset=64, expert_hist=True, hot=True),
        ),
        Cell(
            why="129 rows, 24 token-warps: a hot expert over two tiles",
            spec=dict(rows=129, splits=1, wide=True, offset=192, hot=True),
        ),
        Cell(
            why="MTP5's 192 verify rows on the wide kernel writing the per-expert counts",
            spec=dict(rows=192, splits=1, wide=True, offset=0, expert_hist=True, hot=True),
        ),
        Cell(
            why="200 rows on the wide kernel: 32 token-warps per CTA",
            spec=dict(rows=200, splits=4, wide=True, offset=64),
        ),
        Cell(
            why="256 rows, the wide kernel's ceiling, a hot expert over two tiles",
            spec=dict(rows=256, splits=1, wide=True, offset=128, hot=True),
        ),
        Cell(
            why="65 rows, the first prefill count: 4-warp histogram", spec=dict(rows=65, offset=0)
        ),
        Cell(why="128 rows, the 4-warp histogram's last count", spec=dict(rows=128, offset=64)),
        Cell(why="129 rows, the 8-warp histogram's first", spec=dict(rows=129, offset=128)),
        Cell(why="512 rows, the 8-warp histogram's last", spec=dict(rows=512, offset=192)),
        Cell(why="513 rows, the 16-warp (noraw) histogram's first", spec=dict(rows=513, offset=0)),
        Cell(
            why="7310 rows, a served context step on the noraw histogram",
            spec=dict(rows=7310, offset=128),
        ),
        Cell(why="8192 rows, the token budget", spec=dict(rows=8192, offset=64)),
    )

    note = """
    The trunk builds a decode handle for every row count 1..64: `logit_splits` 1 up to 5
    rows and 4 above (the router's), `signal_gg1` only at a power of two. A padded width
    runs `cute_dsl_moe_routing` next on staged ids, and that kernel does not wait for its
    input: released early, it reads the ids mid-write (a 3-request step hung that way).
    The MTP layer builds one per power-of-two width with `fuse_zero`. Only the trunk
    builds a prefill handle. The MTP layer's context rows, and the trunk's when that
    build failed, select with `noaux_tc_op` at the call site and build the maps with
    `cute_dsl_moe_routing`. At MTP3 / MTP5 (`max_draft_len > 1`) the target also builds
    one wide handle per row range, one logit plane (the prefill router's), which a decode
    step of 65..256 rows takes instead of the prefill handle; MTP1 builds none.

    The returned maps belong to these exact rows: the grouped GEMMs given maps built for
    another row count read an illegal address rather than return a wrong answer, so the
    target uses them only when the GEMMs run at this unpadded row count.

    The decode kernel's early release signals launch readiness, not data: the grouped
    GEMM waits before reading. Selection ties break toward the lower expert id. The row
    order within one expert of the maps is unspecified (atomic claims).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        zero_output: torch.Tensor | None,
        compiled: CompiledTopkRouting,
        expert_hist: torch.Tensor | None = None,
    ):
        compiled.launch(
            logits, bias, local_expert_offset, routed_scale, outputs, zero_output, expert_hist
        )
        rows = int(logits.shape[0])
        n = rows * TOP_K
        weights = outputs["topk_weights"].reshape(-1)[:n].view(rows, TOP_K)
        ids = outputs["topk_indices"].reshape(-1)[:n].view(rows, TOP_K)
        return weights, ids, {k: outputs[k] for k in MAP_KEYS}

    def reference(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        zero_output: torch.Tensor | None,
        compiled: CompiledTopkRouting,
        expert_hist: torch.Tensor | None = None,
    ):
        """`noaux_tc_op`'s arithmetic on the summed partials (tanh-form sigmoid, stable
        descending sort, fp64 renormalization), then `reference_maps` of the ids. Fresh
        tensors; `outputs`, `zero_output`, `compiled` and `expert_hist` are not read (the
        zero plane's clear and the counts are checked by the test)."""
        weights, ids, _ = _sigmoid_topk(logits, bias, routed_scale)
        return weights, ids, reference_maps(ids, local_expert_offset)

    def is_valid(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        zero_output: torch.Tensor | None,
        compiled: CompiledTopkRouting,
        expert_hist: torch.Tensor | None = None,
    ) -> None:
        # The compiled handles skip the shape checks the kernel modules' runners repeat
        # per call: a logits tensor with another split count is summed wrongly, and an
        # output set sized for other rows is written past its end.
        rows = int(logits.shape[0])
        splits = 1 if logits.dim() == 2 else int(logits.shape[1])
        assert logits.dtype == torch.float32 and logits.is_contiguous()
        assert logits.shape[-1] == EXPERTS, f"logits must score {EXPERTS} experts"
        assert tuple(bias.shape) == (EXPERTS,) and bias.dtype == torch.float32
        assert bias.is_contiguous(), "bias must be contiguous fp32 [256]"
        assert local_expert_offset in (0, 64, 128, 192), (
            f"local_expert_offset selects one EP4 window, got {local_expert_offset}"
        )
        assert splits == compiled.logit_splits, (
            f"compiled for {compiled.logit_splits} logit plane(s), got {splits}"
        )
        if compiled.kind == "decode":
            assert rows == compiled.rows, f"compiled for {compiled.rows} rows, got {rows}"
            ref = allocate_decode_outputs(rows, "meta")
            for name in DECODE_OUTPUTS:
                t = outputs[name]
                assert tuple(t.shape) == tuple(ref[name].shape) and t.dtype == ref[name].dtype, (
                    f"{name} must be {ref[name].dtype} {tuple(ref[name].shape)} for {rows} rows"
                )
                assert t.is_contiguous(), f"{name} must be contiguous"
            if compiled.expert_hist:
                assert (
                    expert_hist is not None
                    and expert_hist.dtype == torch.int32
                    and tuple(expert_hist.shape) == (64,)
                    and expert_hist.is_contiguous()
                    and expert_hist.data_ptr() % 16 == 0
                ), "this handle writes expert_hist: a contiguous 16-byte aligned int32 [64]"
            else:
                assert expert_hist is None, "this handle writes no expert counts"
            if compiled.fuse_zero:
                assert zero_output is not None and tuple(zero_output.shape) == (rows, HIDDEN)
                assert zero_output.dtype == torch.bfloat16 and zero_output.is_contiguous()
                assert zero_output.data_ptr() % _ZERO_ALIGN == 0, "zero plane: 16-byte aligned"
            else:
                assert zero_output is None, "this handle has no zero plane"
        elif compiled.kind == "wide":
            # One handle per row range: a count outside it runs too few token-warps (rows
            # dropped) or more than the slab it was compiled for.
            lo, hi = wide_range(compiled.rows)
            assert lo <= rows <= hi, f"this wide handle serves {lo}..{hi} rows, got {rows}"
            assert zero_output is None, "the wide top-k has no zero plane"
            assert logits.data_ptr() % 16 == 0, "logits: 16-byte aligned (128-bit loads)"
            n, q = rows * TOP_K, _wide.max_tiles(rows)
            need = {
                "topk_weights": (n, torch.float32),
                "topk_indices": (n, torch.int32),
                "tile_idx_to_expert_idx": (q, torch.int32),
                "tile_idx_to_mn_limit": (q, torch.int32),
                "expanded_idx_to_permuted_idx": (n, torch.int32),
                "permuted_idx_to_expanded_idx": (q * TILE, torch.int32),
                "total_num_padded_tokens": (1, torch.int32),
                "num_non_exiting_tiles": (1, torch.int32),
            }
            for name, (size, dtype) in need.items():
                t = outputs[name]
                assert t.dtype == dtype and t.is_contiguous(), f"{name} must be contiguous {dtype}"
                assert t.numel() >= size, f"{name} holds {t.numel()}; {rows} rows write {size}"
            if compiled.expert_hist:
                assert (
                    expert_hist is not None
                    and expert_hist.dtype == torch.int32
                    and tuple(expert_hist.shape) == (64,)
                    and expert_hist.is_contiguous()
                    and expert_hist.data_ptr() % 16 == 0
                ), "this handle writes expert_hist: a contiguous 16-byte aligned int32 [64]"
            else:
                assert expert_hist is None, "this handle writes no expert counts"
        else:
            assert PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS, (
                f"the prefill top-k serves {PREFILL_MIN_ROWS}..{PREFILL_MAX_ROWS} rows"
            )
            assert zero_output is None, "the prefill top-k has no zero plane"
            assert expert_hist is None, "the prefill top-k writes no expert counts"
            n = rows * TOP_K
            q = (n + TILE - 1) // TILE + LOCAL_EXPERTS - 1
            need = {
                "topk_weights": (n, torch.float32),
                "topk_indices": (n, torch.int32),
                "partial_counts": (PREFILL_PRODUCER_CTAS * LOCAL_EXPERTS, torch.int32),
                "expert_bases": (LOCAL_EXPERTS, torch.int32),
                "tile_idx_to_expert_idx": (q, torch.int32),
                "tile_idx_to_mn_limit": (q, torch.int32),
                "expanded_idx_to_permuted_idx": (n, torch.int32),
                "permuted_idx_to_expanded_idx": (q * TILE, torch.int32),
                "total_num_padded_tokens": (1, torch.int32),
                "num_non_exiting_tiles": (1, torch.int32),
            }
            for name, (size, dtype) in need.items():
                t = outputs[name]
                assert t.dtype == dtype and t.is_contiguous(), f"{name} must be contiguous {dtype}"
                assert t.numel() >= size, f"{name} holds {t.numel()}; {rows} rows write {size}"
                assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"


cute_dsl_moe_topk_routing = CuteDslMoeTopkRouting()
