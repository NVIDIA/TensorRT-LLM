# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTX boundary (DEP2CTX_NORMQ_V2 / DEP2CTX_NORMQ_STAGE): the launch geometry of the ring kernel and the
engage rules of the pre-MoE stage fold, in plain Python (no torch, no CUDA) so both run in CPU tests.

Geometry (``ctxBoundaryNormQuantSm107.cuh``): one CTA of ``hidden / 8`` threads per row, at most two CTAs per SM
(one on sm_107: 1024 threads per SM), a ring of (inter, residual) rows in shared memory ``STAGES`` deep in the
previous launch and ``L6_STAGES`` deep in the L6 launch (3: the previous launch; a deeper ring is stepped down to fit
the per-block opt-in and to keep the resident CTAs the thread limit admits), rows taken in
quads (rows ``m``, ``m + 32``, ``m + 64``, ``m + 96`` of a 128-row scale tile) so a quad's MXFP8 scales fill whole
16-byte groups of the swizzled 128x4 layout. CBNQ_P2 (``ctxBoundaryNormQuantP2Sm107.cuh``) runs the same rows at
``hidden / 16`` threads with two 8-element chunks each (thread ``t``, half ``h`` = the one-chunk kernel's thread
``t + hidden / 16 * h``), two CTAs per SM on sm_107, its ring ``P2_STAGES`` deep (3);
:func:`launch_form` picks the form the launcher takes. The defaults: P2 at 3 stages, and the one-chunk ring at 3
(deeper rings do not help).
The functions here mirror the C++ formulas one for one; the UT compares them with the op's launch shape and with the
served layout (``get_sf_out_offset_128x4``).

Engage rules: :func:`stage_decline_reason` orders the checks of the stage fold and names the first that fails;
:func:`drop_norm_decline_reason` decides whether the boundary may leave the BF16 norm out;
:func:`topk_fold_decline_reason` / :func:`topk_fold_plane_decline_reason` decide whether the next-layer boundary
takes the MegaMoE top-k reduce (DEP2CTX_TOPK_FOLD: ``ctxBoundaryTopkNormQuantSm107.cuh``, one CTA per SM, a ring of
(top-k plane rows, shared row, residual row, scores) stages).
"""

from typing import Iterable, List, Optional, Sequence, Tuple

STAGES = 3
L6_STAGES = 3  # kCbnqL6DefaultStages
STATIC_SMEM_BOUND = 2048
RESERVED_SMEM_PER_CTA = 1024  # cudaDevAttrReservedSharedMemoryPerBlock on sm_10x
MAX_THREADS = 768
MIN_BLOCKS = 2
PACKED = 8
TILE_ROWS = 128
QUAD_ROWS = 4


def supported(hidden: int) -> bool:
    """``ctxBoundaryNormQuantSupported``: hidden % 256 == 0 and 256 <= hidden <= 6144."""
    return hidden % 256 == 0 and hidden >= 256 and hidden // PACKED <= MAX_THREADS


def smem_bytes(stages: int, add: bool, hidden: int) -> int:
    """``cbnqSmemBytes``: ring + two swizzled-scale quad buffers + one NVFP4 scale row (16-byte rounded) + barriers."""
    return (
        stages * (2 if add else 1) * hidden * 2
        + 2 * (hidden // 128) * 16
        + (hidden // 16 + 15) // 16 * 16
        + stages * 8
    )


def launch_choice() -> Tuple[int, bool]:
    """``cbnqLaunchChoice``: (requested ring depth, max shared-memory carveout): L6_STAGES = 3; the carveout is set
    only for a ring deeper than STAGES, so this is the previous launch (3, no carveout)."""
    return L6_STAGES, L6_STAGES > STAGES


def launch_stages(
    requested: int,
    add: bool,
    hidden: int,
    smem_optin: int,
    max_threads_per_sm: int,
    smem_per_sm: int,
) -> int:
    """``cbnqLaunchStages``: ``requested`` stepped down 8 -> 6 -> 4 -> 3 until the ring fits the per-block opt-in and
    the CTAs the thread limit admits (:func:`ctas_per_sm`) still fit the SM's shared memory, so a deeper ring never
    costs a resident CTA (sm_100 / sm_103 at hidden 6144: 4 stages x 2 CTAs, not 6 x 1)."""
    resident = ctas_per_sm(max_threads_per_sm, hidden)
    for s in (8, 6, 4):
        per_cta = smem_bytes(s, add, hidden) + STATIC_SMEM_BOUND
        if (
            s <= requested
            and per_cta <= smem_optin
            and resident * (per_cta + RESERVED_SMEM_PER_CTA) <= smem_per_sm
        ):
            return s
    return STAGES


def ctas_per_sm(max_threads_per_sm: int, hidden: int) -> int:
    """Resident ring CTAs per SM the thread limit admits under the launch bound (registers <= 40 leave two on sm_100 /
    sm_103, and :func:`launch_stages` keeps the ring small enough for both; sm_107's 1024 threads per SM leave one
    768-thread CTA)."""
    return min(MIN_BLOCKS, max_threads_per_sm // (hidden // PACKED))


# CBNQ_P2 (the two-chunk boundary kernel): hidden / 16 threads x two chunks, two CTAs per SM.
P2_CHUNKS = 2
P2_MAX_THREADS = 384
P2_MIN_BLOCKS = 2
P2_STAGES = 3  # kCbnqP2DefaultStages
WARP = 32


def p2_supported(hidden: int) -> bool:
    """``cbnqP2Supported``: whole 32-chunk warps per half (hidden % 512 == 0) and at most 384 threads (so at most 24
    virtual warps), on a hidden size the ring kernel serves."""
    return (
        supported(hidden) and hidden % 512 == 0 and hidden // (P2_CHUNKS * PACKED) <= P2_MAX_THREADS
    )


def p2_block(hidden: int) -> int:
    """P2 threads per CTA: ``hidden / 16``."""
    return hidden // (P2_CHUNKS * PACKED)


def p2_chunk(tid: int, h: int, hidden: int) -> int:
    """The chunk (the one-chunk kernel's thread) that P2 thread ``tid`` owns in half ``h``."""
    return tid + p2_block(hidden) * h


def p2_virtual_warp(p: int, h: int, hidden: int) -> int:
    """The one-chunk kernel's warp that P2 physical warp ``p``'s half ``h`` holds (its shared reduction slot)."""
    return p + h * (p2_block(hidden) // WARP)


def p2_virtual_warps(hidden: int) -> int:
    """``cbnqBlockReduceSumP2``'s final xor reads this many shared slots: the one-chunk kernel's warp count
    (``hidden / 256``), not the P2 block's (half of it)."""
    return P2_CHUNKS * p2_block(hidden) // WARP


def p2_choice() -> Tuple[bool, int]:
    """``cbnqLaunchChoice``'s P2 part: (enabled, requested depth) = (True, P2_STAGES)."""
    return True, P2_STAGES


def p2_ctas_per_sm(
    stages: int,
    add: bool,
    hidden: int,
    max_threads_per_sm: int,
    smem_per_sm: int,
    static_smem: int = STATIC_SMEM_BOUND,
    reserved_per_cta: int = RESERVED_SMEM_PER_CTA,
) -> int:
    """P2 CTAs per SM the device holds (the occupancy API's answer, bounded by the launch bound): the thread limit
    and the shared memory of ``stages``-deep rings (dynamic + static + the per-CTA reservation)."""
    per_cta = smem_bytes(stages, add, hidden) + static_smem + reserved_per_cta
    return min(P2_MIN_BLOCKS, max_threads_per_sm // p2_block(hidden), smem_per_sm // per_cta)


def launch_form(
    hidden: int,
    add: bool,
    max_threads_per_sm: int,
    smem_optin: int,
    smem_per_sm: int,
) -> Tuple[int, int, int]:
    """``dispatch``: (chunks per thread, CTAs per SM, ring depth) of the launch. P2 at its requested depth, then at 3,
    when it is enabled, serves the hidden size, the thread limit holds the one-chunk kernel below two CTAs per SM
    (sm_107), the ring fits the opt-in and two P2 CTAs fit an SM; else the one-chunk kernel at :func:`launch_stages`'s
    depth."""
    enabled, requested_p2 = p2_choice()
    thread_limited = max_threads_per_sm // (hidden // PACKED) < MIN_BLOCKS
    if enabled and p2_supported(hidden) and thread_limited:
        for s in dict.fromkeys((requested_p2, STAGES)):  # 3 is tried once
            if smem_bytes(s, add, hidden) + STATIC_SMEM_BOUND > smem_optin:
                continue
            ctas = p2_ctas_per_sm(s, add, hidden, max_threads_per_sm, smem_per_sm)
            if ctas >= P2_MIN_BLOCKS:
                return P2_CHUNKS, ctas, s
    requested, _ = launch_choice()
    return (
        1,
        ctas_per_sm(max_threads_per_sm, hidden),
        launch_stages(requested, add, hidden, smem_optin, max_threads_per_sm, smem_per_sm),
    )


#: DEP2CTX_TOPK_FOLD: plane rows per token the fold reduces, and a row's FP32 scores (one 16-byte bulk copy).
TOPK = 4
TOPK_SCORE_BYTES = 16


def topk_stage_bytes(topk: int, hidden: int) -> int:
    """``cbnqTopkStageBytes``: one fold ring stage, [plane rows | shared row | residual row | scores]."""
    return (topk + 2) * hidden * 2 + TOPK_SCORE_BYTES


def topk_smem_bytes(stages: int, topk: int, hidden: int) -> int:
    """``cbnqTopkSmemBytes``: the fold's ring, then the same quad buffers / NVFP4 scale row / barriers as
    :func:`smem_bytes`."""
    return (
        stages * topk_stage_bytes(topk, hidden)
        + 2 * (hidden // 128) * 16
        + (hidden // 16 + 15) // 16 * 16
        + stages * 8
    )


def topk_fits(topk: int, hidden: int, smem_optin: int) -> bool:
    """``runTopkFlavor``'s check: the fold runs its ring STAGES deep (the only depth that fits at hidden 6144, K 4),
    when that ring plus the static bound fits the per-block opt-in."""
    return topk_smem_bytes(STAGES, topk, hidden) + STATIC_SMEM_BOUND <= smem_optin


# CBNQ_TOPK_DUAL: the fold at two hidden / 16-thread row groups per 768-thread CTA (sm_107).
TOPK_DUAL_GROUPS = 2


def topk_dual_choice() -> bool:
    """``cbnqTopkDualEnabled``: the dual fold is on."""
    return True


def topk_dual_form(enabled: bool, hidden: int, max_threads_per_sm: int) -> bool:
    """``cbnqTopkDualForm``: the fold takes the dual-group kernel where it is enabled, the hidden size splits into
    whole 32-chunk warps per half (:func:`p2_supported`) and the thread limit holds the 768-thread kernel to one CTA
    per SM (sm_107: 1024 threads per SM). Same grid, block (hidden / 8) and ring as the one-group fold."""
    return (
        enabled and p2_supported(hidden) and max_threads_per_sm // (hidden // PACKED) < MIN_BLOCKS
    )


def topk_dual_group(position: int) -> int:
    """The row group that consumes ring position ``position`` of a CTA's row sequence."""
    return position % TOPK_DUAL_GROUPS


def topk_dual_release_need(position: int, stages: int = STAGES) -> int:
    """The releases stage ``position % stages`` must count before its consumer waits on position ``position``: one
    per earlier position of the stage (the refill of ``position`` is issued before the release of
    ``position - stages`` is counted)."""
    return position // stages


def topk_dual_pos_row(p: int, cta: int, grid_size: int, rows: int) -> int:
    """``cbnqDualPosRow``: the row of position ``p`` of CTA ``cta``'s sequence (:func:`producer_rows` order), or -1
    past the end, in O(1) for the full tiles and a scan of the last tile's quads."""
    n = num_quads(rows)
    last_tile_quad = (n // 32 - 1) * 32
    full = (last_tile_quad - cta + grid_size - 1) // grid_size if cta < last_tile_quad else 0
    if p < QUAD_ROWS * full:
        return quad_row(cta + (p // QUAD_ROWS) * grid_size, p % QUAD_ROWS)
    p -= QUAD_ROWS * full
    q = cta + full * grid_size
    while q < n:
        count = 0
        while count < QUAD_ROWS and quad_row(q, count) < rows:
            count += 1
        if p < count:
            return quad_row(q, p)
        p -= count
        q += grid_size
    return -1


def num_quads(rows: int) -> int:
    """``cbnqNumQuads``: 32 quads per 128-row scale tile."""
    return (rows + TILE_ROWS - 1) // TILE_ROWS * 32


def quad_row(q: int, j: int) -> int:
    """``cbnqQuadRow``: row ``q % 32 + 32 j`` of tile ``q / 32``."""
    return (q // 32) * TILE_ROWS + j * 32 + (q % 32)


def grid(rows: int, ctas_per_sm: int, sms: int) -> int:
    """The launcher's grid: one CTA per quad, at most the resident CTAs."""
    return max(1, min(num_quads(rows), ctas_per_sm * sms))


def cta_rows(cta: int, grid_size: int, rows: int) -> List[Tuple[int, int, int]]:
    """(quad, j, row) of every quad slot CTA ``cta`` visits, in its order; row >= rows marks a pad slot."""
    out = []
    for q in range(cta, num_quads(rows), grid_size):
        for j in range(QUAD_ROWS):
            out.append((q, j, quad_row(q, j)))
    return out


def producer_rows(cta: int, grid_size: int, rows: int) -> List[int]:
    """The rows thread 0 loads into the ring, in order (``cbnqSettle`` / ``cbnqNext``): the CTA's existing rows."""
    n = num_quads(rows)
    q, j = cta, 0
    seq = []

    def settle(q, j):
        while q < n and quad_row(q, j) >= rows:
            j += 1
            if j == QUAD_ROWS:
                j, q = 0, q + grid_size
        return q, j

    q, j = settle(q, j)
    while q < n:
        seq.append(quad_row(q, j))
        j += 1
        if j == QUAD_ROWS:
            j, q = 0, q + grid_size
        q, j = settle(q, j)
    return seq


def ring_slots(count: int, stages: int = STAGES) -> List[Tuple[int, int]]:
    """(stage, mbarrier parity) of the ``count`` rows a CTA consumes, in order."""
    return [(c % stages, (c // stages) & 1) for c in range(count)]


def served_sf_offset(row: int, k: int, rows: int, hidden: int) -> int:
    """``get_sf_out_offset_128x4`` (batch 0): the swizzled byte of row ``row``'s ``k``-th 32-element scale."""
    num_col_vecs = hidden // 32
    num_k_tiles = (num_col_vecs + 3) // 4
    inner_k = k % 4
    inner_m = (row % 128) // 32
    outer_m = row % 32
    k_tile = k // 4
    m_tile = row // 128
    return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k


def quad_sf_offset(q: int, j: int, k: int, hidden: int) -> int:
    """Where the ring kernel's 16-byte group store puts that byte: staged at ``(k / 4) * 16 + j * 4 + k % 4`` of the
    quad buffer, stored as group ``k / 4`` at ``(mTile * numKTiles + k / 4) * 512 + (q % 32) * 16``."""
    num_k_tiles = hidden // 128
    staged = (k // 4) * 16 + j * 4 + k % 4
    group = staged // 16
    return (q // 32 * num_k_tiles + group) * 512 + (q % 32) * 16 + staged % 16


def ms_per_1k(delta_us: float, launches: int, rows: int) -> float:
    """A per-call saving of ``delta_us`` over ``launches`` boundary calls per ``rows``-row chunk, in ms per 1K new
    tokens (delta_us x launches / rows)."""
    return delta_us * launches / rows


#: The MegaMoE-local seat API the stage fold relies on (DEP_ROUTER_STOCK_BOUNDARY uses the same).
SEAT_API = (
    "_route_quant_decline_reason",
    "_select_launch_max_tokens",
    "_acquire_buffers",
    "_take_pad_mask",
    "_prestaged_route_quant",
    "_last_staged_T",
    "_active_launch_max_tokens",
    "fc31_input_scale",
    "routing_method",
)


def stage_decline_reason(
    *,
    enabled: bool,
    ops_built: bool,
    sm_major: int,
    hidden: int,
    seat_api: bool,
    router_f32: bool,
    lora: bool,
    rows: int,
    all_rank_num_tokens: Optional[Sequence[int]],
    ep_rank: int,
    moe_max_num_tokens: int,
    may_fuse: bool,
    appended_columns: int,
) -> Optional[str]:
    """Why the pre-MoE boundary must not stage the routed experts' NVFP4 input itself, or None.

    The stage is taken by the backend's ``try_fused_route_quant`` only when the fused-comm scheduler offers it the very
    rows the boundary wrote: one chunk (the chunk max within ``moe_max_num_tokens``), this rank's live rows equal to the
    boundary's rows (no ADP padding cut), and ``_may_fuse_route_quant`` true. The backend's own decline reason is
    checked by the caller with the step's bucket."""
    if not enabled:
        return "switched off"
    if not ops_built:
        return "the boundary / route-stage ops are not built"
    if sm_major != 10:
        return f"sm_{sm_major}x (the ring kernel runs on sm_10x)"
    if not supported(hidden):
        return f"hidden {hidden}"
    if not seat_api:
        return "the routed experts' backend has no MegaMoE-local seat"
    if not router_f32:
        return "the router does not take the boundary's FP32 copy (DEP2CTX_NORMQ_F32 off)"
    if lora:
        return "LoRA"
    if all_rank_num_tokens:
        if not 0 <= ep_rank < len(all_rank_num_tokens):
            return f"EP rank {ep_rank} outside all_rank_num_tokens"
        if int(all_rank_num_tokens[ep_rank]) != rows:
            return f"this rank's live rows {all_rank_num_tokens[ep_rank]} != the boundary's {rows}"
        chunk_max = max(int(v) for v in all_rank_num_tokens)
    else:
        chunk_max = rows
    if chunk_max > moe_max_num_tokens:
        return f"chunked ({chunk_max} > moe_max_num_tokens {moe_max_num_tokens})"
    if not may_fuse:
        return "the scheduler does not offer the fused route-quant"
    if appended_columns != 0:
        return f"{appended_columns} appended routing columns"
    return None


def chunk_max_tokens(all_rank_num_tokens: Optional[Iterable[int]]) -> Optional[int]:
    """What the fused-comm scheduler hands ``set_adaptive_launch_tokens`` for a one-chunk call."""
    vals = [int(v) for v in all_rank_num_tokens] if all_rank_num_tokens else []
    return max(vals) if vals else None


def drop_norm_decline_reason(
    *,
    enabled: bool,
    shared_takes_mxfp8: Optional[bool],
    lora: bool,
    router_reads_bf16: bool = False,
) -> Optional[str]:
    """Why a staged boundary must still write the BF16 norm, or None (left out: the router takes the FP32 copy, the
    routed experts the stage, the shared expert the MXFP8 slot; any consumer that misses its copy materializes the norm
    first, bit for bit). ``shared_takes_mxfp8`` is None without a shared expert. ``router_reads_bf16``: the router
    reads the BF16 norm itself (DEP2CTX_ROUTER_BF16S), so the norm is always written."""
    if router_reads_bf16:
        return "the router reads the BF16 norm (DEP2CTX_ROUTER_BF16S)"
    if not enabled:
        return "switched off"
    if lora:
        return "LoRA"
    if shared_takes_mxfp8 is False:
        return "the shared expert's gate_up does not take the published MXFP8"
    return None


def topk_fold_decline_reason(
    *,
    enabled: bool,
    op_built: bool,
    normq: bool,
    feeds_next_layer: bool,
    fused_comm: bool,
    capture_layer: bool,
    x_ok: bool,
    rows: int,
    min_rows: int,
    compiling: bool,
    capture_blocked: bool,
) -> Optional[str]:
    """Why a MoE layer's next-layer boundary must not take the top-k reduce (DEP2CTX_TOPK_FOLD), or None. Checked
    before the MoE runs; the unreduced shared rows never leave ``forward_MoE``, so every reader between the MoE and
    the boundary (the DSpark capture) declines here, as does anything the stock boundary
    (``_normq_boundary``) itself would not take. ``fused_comm`` (no MoE allreduce, a global routed output) is the
    tail-add deferral's own condition; it is checked here too so nothing after the tail add can read the parked rows."""
    if not enabled:
        return "switched off"
    if not op_built:
        return "the fold op is not built, not sm_10x, or rejected a launch"
    if not normq:
        return "the boundary is not DEP2CTX_NORMQ_V2"
    if not feeds_next_layer:
        return "the last layer (the final norm feeds lm_head) or no next-layer norm"
    if not fused_comm:
        return "not attention DP on a fused-communication MoE (an allreduce would read the unreduced rows)"
    if capture_layer:
        return "a speculative-decoding capture layer reads the MoE output"
    if not x_ok:
        return "the rows are not 2-D contiguous BF16"
    if rows < min_rows:
        return f"{rows} rows < {min_rows}"
    if compiling:
        return "torch.compile"
    if capture_blocked:
        return "a CUDA graph capture other than a breakable prefill capture"
    return None


def topk_fold_plane_decline_reason(*, deferred: bool, topk: int, aligned: bool) -> Optional[str]:
    """Why the MoE's routed output cannot be folded once it is known, or None: the launch reduced its own top-k, a
    plane of another K, or rows / scores the fold's 16-byte bulk copies cannot read."""
    if not deferred:
        return "the routed launch reduced its own top-k"
    if topk != TOPK:
        return f"top-{topk} plane (the fold reduces top-{TOPK})"
    if not aligned:
        return "the plane, scores or shared rows are not contiguous and 16-byte aligned"
    return None
