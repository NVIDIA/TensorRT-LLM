# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked gated delta rule on FlashInfer's Blackwell kernels: the GDN prefill core.

The FP8 target's copy of the family entry
`models/qwen4_exp/ops/ssm/flashinfer_chunk_gated_delta_rule`, with in-place paths. Every call
is the target's form: `has_initial_state` (a fresh sequence starts from zeros and its pool row
is not read, so the caller need not reset it), the in-place indexed state update, and
`flashinfer.chunk_gated_delta_rule` (CuTe-DSL tcgen05 kernels on sm_100/103) on one of the three
paths `__call__` picks.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.fused_state_io import cast_scatter_fp32_vk_to_vk

#: The Gated DeltaNet at tp1 and dep4: (H, HV, K, V).
QWEN38_GEOMETRY = (16, 48, 128, 128)
#: Slot count of the dense-pool cell.
ENGINE_SLOTS = 461


#: A tp4/tep4 rank: tp1's heads split four ways.
RANK4_GEOMETRY = (4, 12, 128, 128)
#: Per-rank slot counts the V2 hybrid manager allocates on this checkpoint (bf16 KV and state,
#: KV fraction 0.6). It sizes the pool from the rank's memory budget, so a dep4 rank (every GDN
#: head) holds fewer slots than a tep4 rank.
TEP4_SLOTS = 1874
DEP4_SLOTS = 850
#: Not measured: a tp4 rank's state per slot is tep4's.
TP4_SLOTS = TEP4_SLOTS
#: GDN layers whose states share one slot page: a layer's pool view has a slot stride of 36
#: `[HV, V, K]` states (a tep4 rank's last slot starts 1.3e10 elements past the base).
PAGE_LAYERS = 36

#: A tp2/tep2 rank: tp1's heads split two ways.
RANK2_GEOMETRY = (8, 24, 128, 128)
#: Not measured: a tp2/tep2 slot holds half a dep4 state, so twice dep4's count spans dep4's
#: page (2.4e10 elements, the largest certified).
TP2_SLOTS = 2 * DEP4_SLOTS


def _engine_slots(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots of the engine pool, slot 0 and the last live."""
    return ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs)))[:num_seqs]


def _rank_slots(num_seqs: int, slots: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots: the last, the first, then a stride-37 walk."""
    rest = tuple((i * 37 + 4) % (slots - 2) + 1 for i in range(max(num_seqs - 2, 0)))
    picked = ((slots - 1, 0) + rest)[:num_seqs]
    assert len(set(picked)) == num_seqs
    return picked


def _rank_cell(why: str, geometry, seqlens, slots: int, initial, fi_path: str) -> Cell:
    """The target's call at one rank: the layer's view of the bf16 state page at the member's
    slot count, a prefix view of a wider output buffer. `fi_path` is the path it reaches on
    sm_103 (`cp`, `chunked` or `inplace`)."""
    return Cell(
        why=why,
        spec=dict(
            geometry=geometry,
            seqlens=seqlens,
            slots=slots,
            page_layers=PAGE_LAYERS,
            indices=_rank_slots(len(seqlens), slots),
            pool_dtype=torch.bfloat16,
            output="slice",
            initial=initial,
            linear_gate=True,
            fi_path=fi_path,
        ),
    )


#: Elements of a `[V, K]` state plane a gather program copies.
_GATHER_BLOCK = 4096


# Slot ids and carry flags are scalar-read views of the step's metadata.
@triton.jit(do_not_specialize_on_alignment=["slot_ptr", "init_ptr"])
def _gather_carried_states_kernel(
    pool_ptr,
    dst_ptr,
    slot_ptr,
    init_ptr,
    stride_ps,
    stride_ph,
    stride_pv,
    stride_pk,
    H: tl.constexpr,
    K: tl.constexpr,
    PLANE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """`BLOCK` elements of sequence `n`'s head `h` plane into the packed `[N, H, V, K]` buffer:
    pool row `slot_ptr[n]` when `init_ptr[n]`, else +0.0 (the row is not read)."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = i < PLANE
    dst = dst_ptr + (n.to(tl.int64) * H + h) * PLANE + i
    if tl.load(init_ptr + n) != 0:
        slot = tl.load(slot_ptr + n).to(tl.int64)
        src = (
            pool_ptr + slot * stride_ps + h * stride_ph + (i // K) * stride_pv + (i % K) * stride_pk
        )
        tl.store(dst, tl.load(src, mask=mask), mask=mask)
    else:
        tl.store(dst, tl.zeros([BLOCK], dtype=dst_ptr.dtype.element_ty), mask=mask)


def _gather_carried_states(
    pool: torch.Tensor, indices: torch.Tensor, has_initial_state: torch.Tensor
) -> torch.Tensor:
    """`[N, H, V, K]` in the pool's dtype: `pool[indices[n]]` for a carried sequence, zeros
    for a fresh one."""
    num_seqs = indices.shape[0]
    _, h, v, k = pool.shape
    out = torch.empty(num_seqs, h, v, k, dtype=pool.dtype, device=pool.device)
    _gather_carried_states_kernel[(num_seqs, h, triton.cdiv(v * k, _GATHER_BLOCK))](
        pool, out, indices, has_initial_state, *pool.stride(),
        H=h, K=k, PLANE=v * k, BLOCK=_GATHER_BLOCK, num_warps=4,
    )  # fmt: skip
    return out


#: Context tokens up to which a call runs in place on the chunked kernel: at most one 64-token
#: chunk a sequence, one launch, where the CP pipeline buys nothing.
INPLACE_MAX_TOKENS = 64


@triton.jit(do_not_specialize_on_alignment=["slot_ptr", "init_ptr"])
def _zero_fresh_states_kernel(
    pool_ptr,
    slot_ptr,
    init_ptr,
    stride_ps,
    stride_ph,
    stride_pv,
    stride_pk,
    K: tl.constexpr,
    PLANE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero `BLOCK` elements of sequence `n`'s head `h` plane of pool row `slot_ptr[n]` when
    `init_ptr[n]` is false; a carried sequence's row is not touched."""
    n = tl.program_id(0)
    if tl.load(init_ptr + n) == 0:
        h = tl.program_id(1)
        i = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(slot_ptr + n).to(tl.int64)
        dst = (
            pool_ptr + slot * stride_ps + h * stride_ph + (i // K) * stride_pv + (i % K) * stride_pk
        )
        tl.store(dst, tl.zeros([BLOCK], dtype=pool_ptr.dtype.element_ty), mask=i < PLANE)


def _chunk_inplace(
    q, k, v, g, beta, scale, pool, indices, has_initial_state, cu_seqlens, output, linear_gate
) -> torch.Tensor:
    """The fresh rows zeroed, then the chunked kernel (`use_cp=False`) reading and writing the
    named pool rows itself (`state_indices`, the pool as `output_state`): no packed copies."""
    import flashinfer

    num_seqs = indices.shape[0]
    _, h, v_dim, k_dim = pool.shape
    _zero_fresh_states_kernel[(num_seqs, h, triton.cdiv(v_dim * k_dim, _GATHER_BLOCK))](
        pool, indices, has_initial_state, *pool.stride(),
        K=k_dim, PLANE=v_dim * k_dim, BLOCK=_GATHER_BLOCK, num_warps=4,
    )  # fmt: skip
    q3, k3, v3 = q.squeeze(0).contiguous(), k.squeeze(0).contiguous(), v.squeeze(0).contiguous()
    out = (
        output.squeeze(0)
        if output is not None
        else q3.new_empty(q3.shape[0], v3.shape[1], v3.shape[2])
    )
    flashinfer.chunk_gated_delta_rule(
        q=q3,
        k=k3,
        v=v3,
        g=g.squeeze(0) if linear_gate else torch.exp(g.squeeze(0)),
        beta=beta.squeeze(0).to(torch.float32),
        scale=scale,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out,
        output_state=pool,
        use_cp=False,
        state_indices=indices,
    )
    return out.unsqueeze(0)


#: FlashInfer's sm_10x rule for its context-parallel pipeline (`should_use_cp_host`): CP when
#: `num_seqs x num_v_heads x 4` is below the SM count.
_CP_PARALLELISM_DENOMINATOR = 4


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=1024)
def _cp_chunk_len_cached(seqlen: int, num_heads: int, device_index: int) -> int:
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import choose_cp_chunk_len_host
    from flashinfer.utils import get_compute_capability, get_device_name, get_device_sm_count

    device = torch.device("cuda", device_index)
    return int(
        choose_cp_chunk_len_host(
            seqlen,
            num_heads,
            get_device_sm_count(device),
            device_capability=get_compute_capability(device),
            total_seqlen=seqlen,
            num_seqs=1,
            device_name=get_device_name(device),
        )
    )


def cp_chunk_len_for(seqlen: int, num_heads: int, device: torch.device) -> int:
    """The CP chunk length FlashInfer picks by itself for one `seqlen`-token sequence over
    `num_heads` (max of q and v) heads. A call over a padded row count passes it
    (`_cp_chunk_len`) to cut the sequence as the unpadded call does."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _cp_chunk_len_cached(seqlen, num_heads, index)


def _reaches_cp(num_seqs: int, num_v_heads: int, device: torch.device) -> bool:
    """Whether FlashInfer's `use_cp="auto"` routes `num_seqs` sequences to the CP pipeline."""
    return num_seqs * num_v_heads * _CP_PARALLELISM_DENOMINATOR < _sm_count(device.index)


def _chunk_cp_inplace(
    q,
    k,
    v,
    g,
    beta,
    scale,
    pool,
    indices,
    has_initial_state,
    cu_seqlens,
    output,
    linear_gate,
    fi_kwargs=None,
) -> torch.Tensor:
    """The CP pipeline in place over the pool: the fresh rows zeroed, then the pipeline reading
    and writing the named rows itself. It keeps the state in its own workspace between its
    kernels and writes each final state once, so the result is bitwise `_chunk_carried`'s
    without its gather and scatter."""
    import flashinfer

    num_seqs = indices.shape[0]
    _, h, v_dim, k_dim = pool.shape
    _zero_fresh_states_kernel[(num_seqs, h, triton.cdiv(v_dim * k_dim, _GATHER_BLOCK))](
        pool, indices, has_initial_state, *pool.stride(),
        K=k_dim, PLANE=v_dim * k_dim, BLOCK=_GATHER_BLOCK, num_warps=4,
    )  # fmt: skip
    q3, k3, v3 = q.squeeze(0).contiguous(), k.squeeze(0).contiguous(), v.squeeze(0).contiguous()
    out = (
        output.squeeze(0)
        if output is not None
        else q3.new_empty(q3.shape[0], v3.shape[1], v3.shape[2])
    )
    flashinfer.chunk_gated_delta_rule(
        q=q3,
        k=k3,
        v=v3,
        g=g.squeeze(0) if linear_gate else torch.exp(g.squeeze(0)),
        beta=beta.squeeze(0).to(torch.float32),
        scale=scale,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out,
        output_state=pool,
        use_cp=True,
        state_indices=indices,
        **(fi_kwargs or {}),
    )
    return out.unsqueeze(0)


def _chunk_carried(
    q,
    k,
    v,
    g,
    beta,
    scale,
    pool,
    indices,
    has_initial_state,
    cu_seqlens,
    output,
    linear_gate,
    keep_rows=None,
    keep_slots=None,
    use_cp=None,
    fi_kwargs=None,
) -> torch.Tensor:
    """Packed path: the named states gathered in the pool's dtype (zeros for a fresh sequence),
    FlashInfer's kernel (`g` exponentiated unless `linear_gate`), and the scatter of the final
    states into the same slots, or with `keep_rows` only those sequences' into `keep_slots`.
    `use_cp` None is FlashInfer's heuristic; `fi_kwargs` are passed through (a padded call's
    `max_seqlen` and `_cp_chunk_len`)."""
    import flashinfer

    q3, k3, v3 = q.squeeze(0).contiguous(), k.squeeze(0).contiguous(), v.squeeze(0).contiguous()
    packed = _gather_carried_states(pool, indices, has_initial_state)
    out = (
        output.squeeze(0)
        if output is not None
        else q3.new_empty(q3.shape[0], v3.shape[1], v3.shape[2])
    )
    final = torch.empty_like(packed)
    flashinfer.chunk_gated_delta_rule(
        q=q3,
        k=k3,
        v=v3,
        g=g.squeeze(0) if linear_gate else torch.exp(g.squeeze(0)),
        beta=beta.squeeze(0).to(torch.float32),
        scale=scale,
        initial_state=packed,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=out,
        output_state=final,
        **({} if use_cp is None else {"use_cp": use_cp}),
        **(fi_kwargs or {}),
    )
    if keep_rows is None:
        cast_scatter_fp32_vk_to_vk(final, pool, indices)
    else:
        cast_scatter_fp32_vk_to_vk(final.index_select(0, keep_rows), pool, keep_slots)
    return out.unsqueeze(0)


def chunk_split_first_pass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    gather_slots: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    keep_rows: torch.Tensor,
    keep_slots: torch.Tensor,
    output: torch.Tensor,
    num_whole_seqs: int,
    whole: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """First pass of an in-forward snapshot split (`PrefillTarget._gdn_split`): `_chunk_carried`
    over `cu_seqlens` with extra sequence boundaries (`linear_gate`), only sequences `keep_rows`
    writing their final states, to pool rows `keep_slots`. Bitwise the unsplit two-iteration
    schedule: on the chunked kernel every sequence runs on its own; the CP pipeline segments a
    sequence by how many sequences the call has, so when FlashInfer picks CP for the
    `num_whole_seqs` unsplit sequences the pass runs over the kept rows alone, `whole` = (their
    rows, boundaries, carry flags), copied in and out, or on views when `rows` is an int (the
    kept rows lead the step)."""
    assert q.shape[1] > INPLACE_MAX_TOKENS, (
        f"a split first pass of {q.shape[1]} tokens would leave the carried path"
    )
    if whole is not None and _use_cp(num_whole_seqs * pool.shape[1], q.device):
        rows, cu, carry = whole
        if isinstance(rows, int):
            # The kept rows lead: the same operands as their gather, without the copies.
            _chunk_carried(
                *(x[:, :rows] for x in (q, k, v, g, beta)),
                None, pool, keep_slots, carry, cu, output[:, :rows], True,
            )  # fmt: skip
            return output
        o = _chunk_carried(
            *(x.index_select(1, rows) for x in (q, k, v, g, beta)),
            None, pool, keep_slots, carry, cu, None, True,
        )  # fmt: skip
        output[0].index_copy_(0, rows, o[0])
        return output
    return _chunk_carried(
        q, k, v, g, beta, None, pool, gather_slots, has_initial_state, cu_seqlens, output, True,
        keep_rows=keep_rows, keep_slots=keep_slots,
        use_cp=_use_cp(num_whole_seqs * pool.shape[1], q.device),
    )  # fmt: skip


@functools.lru_cache(maxsize=64)
def _use_cp_cached(units: int, device_index: int) -> bool:
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
    from flashinfer.utils import get_compute_capability, get_device_name, get_device_sm_count

    device = torch.device("cuda", device_index)
    return bool(
        should_use_cp_host(
            units,
            get_device_sm_count(device),
            get_device_name(device),
            device_capability=get_compute_capability(device),
        )
    )


def split_takes_cp(num_seqs: int, v_heads: int, device: torch.device) -> bool:
    """Whether `chunk_split_first_pass` runs over the kept rows alone (FlashInfer picks CP)."""
    return _use_cp(num_seqs * v_heads, device)


def _use_cp(units: int, device: torch.device) -> bool:
    """FlashInfer's CP choice (`should_use_cp_host`) for `units` = sequences x value heads."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _use_cp_cached(units, index)


class FlashinferChunkGatedDeltaRule(OpWrapper):
    """The gated delta rule over whole sequences, in place over the state pool.

    Per token `t` and value head, `h` the `[V, K]` state, `q_h`/`k_h` the key heads broadcast
    (value head `i` reads key head `i // (HV // H)`), fp32 accumulation:

        h *= exp(g[t]);  u = (v[t] - h k_h[t]) * beta[t];  h += u k_h[t]^T
        o[t] = h (scale * q_h[t])

    computed 64 tokens at a time on tensor cores (bf16 operands); `g` is in log space, or with
    `linear_gate` the factor `exp(g)`; `scale=None` is `K ** -0.5`. One packed row split at
    `cu_seqlens`; sequence `n` starts from `initial_state[initial_state_indices[n]]` (the pool
    `[S, HV, V, K]`), or from zeros when `has_initial_state[n]` is false, and its final state is
    written back into that slot. The conv, q/k/v/g/beta preparation and the gated norm are the
    caller's. Returns `(o, None)`, `o` `[1, T, HV, V]` in `q.dtype` (`output` when given).
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: geometry (H, HV, K, V); seqlens; `slots`-row pool with `indices`; pool_dtype;
    # output (None | "buffer" | "slice", a prefix view of a wider buffer); page_layers (the
    # pool is the layer's view of the state page); initial; linear_gate; fi_path. Each head
    # layout on each path `__call__` routes it to: in place (with the 64 / 65-token edge),
    # CP and chunked.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="target call: bf16 pool, has_initial_state [1, 0, 1], seqlens [300, 1024, 500]",
            spec=dict(
                geometry=QWEN38_GEOMETRY,
                seqlens=(300, 1024, 500),
                slots=ENGINE_SLOTS,
                indices=_engine_slots(3),
                pool_dtype=torch.bfloat16,
                output="buffer",
                initial=(True, False, True),
                linear_gate=True,
            ),
        ),
        _rank_cell(
            "tep4 4/12, [200, 37, 91] carried/fresh/carried, CP path, 1874 slots",
            RANK4_GEOMETRY, (200, 37, 91), TEP4_SLOTS, (True, False, True), "cp",
        ),
        _rank_cell(
            "tp4 4/12, carried 8192-token chunk, CP path, 1874-slot bf16 page",
            RANK4_GEOMETRY, (8192,), TP4_SLOTS, (True,), "cp",
        ),
        _rank_cell(
            "dep4 16/48, carried 512-token prompt, 850-slot bf16 page",
            QWEN38_GEOMETRY, (512,), DEP4_SLOTS, (True,), "chunked",
        ),
        _rank_cell(
            "tep4 4/12 envelope, 8192 rows over 128 seqs of 63/65, chunked path",
            RANK4_GEOMETRY, (63, 65) * 64, TEP4_SLOTS, (True, False) * 64, "chunked",
        ),
        _rank_cell(
            "tep4 4/12, one carried 64-token chunk (the in-place bound), 1874 slots",
            RANK4_GEOMETRY, (64,), TEP4_SLOTS, (True,), "inplace",
        ),
        _rank_cell(
            "tep4 4/12, one carried 65-token chunk, past the in-place bound: CP path",
            RANK4_GEOMETRY, (65,), TEP4_SLOTS, (True,), "cp",
        ),
        _rank_cell(
            "dep4 16/48, carried 2 + fresh 30 + carried 1 tokens, in place, 850 slots",
            QWEN38_GEOMETRY, (2, 30, 1), DEP4_SLOTS, (True, False, True), "inplace",
        ),
        # tp2/tep2: 24 v heads reach the CP pipeline for one context sequence only
        # (24 x 4 < 152 SMs, 48 x 4 is not).
        _rank_cell(
            "tp2/tep2 8/24, fresh 8192-token prompt, CP path, 1700-slot bf16 page",
            RANK2_GEOMETRY, (8192,), TP2_SLOTS, (False,), "cp",
        ),
        _rank_cell(
            "tp2/tep2 8/24, [200, 37, 91] carried/fresh/carried, chunked path, 1700 slots",
            RANK2_GEOMETRY, (200, 37, 91), TP2_SLOTS, (True, False, True), "chunked",
        ),
        _rank_cell(
            "tp2/tep2 8/24, fresh 5 + carried 2 tokens, in place, 1700 slots",
            RANK2_GEOMETRY, (5, 2), TP2_SLOTS, (False, True), "inplace",
        ),
    )  # fmt: skip

    note = """
    Source: `flashinfer.chunk_gated_delta_rule` (`flashinfer.gdn_prefill`), CuTe-DSL kernels
    JIT-compiled in-process at the first call per path and device; Triton kernels for the
    fresh-row zeroing, the packed gather and the scatter (`cast_scatter_fp32_vk_to_vk`).

    Paths (`__call__`): at most `INPLACE_MAX_TOKENS` context tokens run in place on the chunked
    kernel; calls FlashInfer routes to its CP pipeline (`_reaches_cp`: sequences x HV x 4 below
    the SM count, so on GB300 one sequence at 8/24, 1-3 at 4/12, never at 16/48) run in place
    on it; the rest take the packed path (`_chunk_carried`: a copy of the named states in and
    out). The state is fp32 inside and rounded once to the pool's dtype.

    Preconditions: q/k L2-normalized by the caller, `g` fp32, `cu_seqlens` given, slots
    distinct and in range (not checked), context rows only (single-token rows belong to the
    decode kernel). FlashInfer keeps one TMA workspace per device: two streams must not run
    this op concurrently. Slot offsets are int64 (every certified page's last slot is past
    int32).

    Certified on GB300 against the fp64 reference at the cells: bf16 pools at 16/48, 8/24 and
    4/12 heads on each path they reach. Not certified: fp32 / fp16 pools, `scale` other than the
    default, other head layouts, K or V other than 128, 2 context sequences on CP at 4/12, a
    carried sequence on CP at 8/24, the measured slot counts of tp4, tp2 and tep2 (stand-ins
    above), a call inside a CUDA graph (the target calls it eagerly).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        inplace_indexed_state_update: bool = False,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        output: Optional[torch.Tensor] = None,
        state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        linear_gate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert has_initial_state is not None, (
            "the target's call only: has_initial_state, the in-place indexed update"
        )
        assert (
            inplace_indexed_state_update
            and initial_state_indices is not None
            and not output_final_state
            and not head_first
            and not use_qk_l2norm_in_kernel
            and state_workspace is None
            and cu_seqlens is not None
            and g.dtype == torch.float32
        ), "has_initial_state: the in-place indexed call only, q/k normalized by the caller"
        if q.shape[1] <= INPLACE_MAX_TOKENS:
            path = _chunk_inplace
        elif _reaches_cp(initial_state_indices.shape[0], v.shape[2], q.device):
            path = _chunk_cp_inplace
        else:
            path = _chunk_carried
        o = path(
            q, k, v, g, beta, scale, initial_state, initial_state_indices,
            has_initial_state, cu_seqlens, output, linear_gate,
        )  # fmt: skip
        return o, None

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        inplace_indexed_state_update: bool = False,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        head_first: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        output: Optional[torch.Tensor] = None,
        state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        linear_gate: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """`(o, final_state, initial_state as the call leaves it)`: an fp64 token-at-a-time
        recurrence, `o` rounded once to `q.dtype`, the states to the pool's dtype. Mutates
        nothing; `output`, `head_first` and `state_workspace` are ignored."""
        dt = torch.float64
        _, _, num_k_heads, head_k = q.shape
        num_v_heads = v.shape[2]
        repeats = num_v_heads // num_k_heads
        scale = head_k**-0.5 if scale is None else scale
        bounds = cu_seqlens.tolist()
        slots = initial_state_indices.tolist()
        carried = [True] * len(slots) if has_initial_state is None else has_initial_state.tolist()
        out = torch.zeros(1, q.shape[1], num_v_heads, v.shape[3], dtype=dt, device=q.device)
        finals = []
        for n in range(len(bounds) - 1):
            state = initial_state[slots[n]].to(dt)
            if not carried[n]:
                state = torch.zeros_like(state)
            for t in range(bounds[n], bounds[n + 1]):
                q_t, k_t = q[0, t].to(dt), k[0, t].to(dt)
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / torch.sqrt(q_t.pow(2).sum(-1, keepdim=True) + 1e-6)
                    k_t = k_t / torch.sqrt(k_t.pow(2).sum(-1, keepdim=True) + 1e-6)
                q_t = q_t.repeat_interleave(repeats, dim=0) * scale
                k_t = k_t.repeat_interleave(repeats, dim=0)
                decay = g[0, t].to(dt) if linear_gate else g[0, t].to(dt).exp()
                state = state * decay[:, None, None]
                u = (v[0, t].to(dt) - (state * k_t[:, None, :]).sum(-1)) * beta[0, t].to(dt)[
                    :, None
                ]
                state = state + u[:, :, None] * k_t[:, None, :]
                out[0, t] = (state * q_t[:, None, :]).sum(-1)
            finals.append(state)
        pool = initial_state.clone()
        final_state = None
        if inplace_indexed_state_update:
            for slot, state in zip(slots, finals):
                pool[slot] = state.to(pool.dtype)
        elif output_final_state:
            final_state = torch.stack(finals).to(initial_state.dtype)
        return out.to(q.dtype), final_state, pool


flashinfer_chunk_gated_delta_rule = FlashinferChunkGatedDeltaRule()
