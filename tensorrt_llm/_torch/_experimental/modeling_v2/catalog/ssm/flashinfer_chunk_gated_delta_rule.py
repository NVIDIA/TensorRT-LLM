# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked gated delta rule on FlashInfer's Blackwell kernel: the GDN prefill core.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.fla.flashinfer_chunk.chunk_gated_delta_rule``
(tier-2 sourcing), the adapter in front of the installed FlashInfer's
``flashinfer.chunk_gated_delta_rule`` (CuTe-DSL tcgen05 kernel on sm_100/103).
One call is: ``g -> exp(g)``, a Triton gather of the indexed pool slots into a
packed ``[N, HV, V, K]`` buffer in the pool's dtype (the sm_100/103 kernel keeps
the state in fp32 inside and reads and writes it in that dtype), the FlashInfer
kernel, and a Triton scatter of the final states back into the same slots.
Call-compatible with the vendored Triton ``chunk_gated_delta_rule``
(``modules/fla/chunk``).

FlashInfer API: ``flashinfer.chunk_gated_delta_rule`` (``flashinfer.gdn_prefill``),
reached through the adapter above. With ``has_initial_state`` the call is the same
sequence in this module, its gather a Triton kernel that writes zeros for a sequence
carrying nothing instead of reading its slot, so the caller need not zero the slot first.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (
    chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
)
from tensorrt_llm._torch.modules.fla.fused_state_io import cast_scatter_fp32_vk_to_vk

from .._op import Arch, Cell, OpWrapper

#: qwen3.8-flash-next-nvfp4's Gated DeltaNet at tp1: (H, HV, K, V) -- 16 q/k
#: heads, 48 v heads, head dims 128.
QWEN38_GEOMETRY = (16, 48, 128, 128)
#: Largest slot count the cache manager was seen to allocate the recurrent pool
#: at for qwen3.8-flash-next-nvfp4 (10 .. 461 over twelve runtime-value dumps).
ENGINE_SLOTS = 461


#: A tp4/tep4 rank's Gated DeltaNet: tp1's 16 key / 48 value heads split four ways, the
#: key-to-value head ratio (3) unchanged. A dep4 rank keeps `QWEN38_GEOMETRY`.
RANK4_GEOMETRY = (4, 12, 128, 128)
#: Per-rank recurrent-pool slot counts at core's config (bf16 KV and state, KV fraction
#: 0.6, `max_batch_size` 128, `max_seq_len` 9216, TRTLLM MoE): the stock engine's
#: `Qwen4ExpHybridCacheManager` on this checkpoint, read off its "PLE state views bound"
#: log line (the PLE and GDN pools share one slot space) with the n-gram table
#: host-resident, the same count on all four ranks. The V2 manager sizes the pool from the
#: rank's memory budget, not from `max_batch_size`; a dep4 rank keeps every GDN head, so
#: it holds fewer slots than a tep4 rank.
TEP4_SLOTS = 1874
DEP4_SLOTS = 850
#: No engine has run tp4 (the stock MoE refuses its 160-row expert slice). A tp4 rank's
#: GDN state per slot is a tep4 rank's, so tep4's count stands in for it.
TP4_SLOTS = TEP4_SLOTS
#: GDN layers whose states share one slot page in the stock manager (48 layers, every
#: 4th full attention): the pool one layer receives is a view whose slot stride is 36
#: `[HV, V, K]` states, so a tep4 rank's last slot starts 1.3e10 elements past the base.
PAGE_LAYERS = 36


def _engine_slots(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots of the engine pool, slot 0 and the last live."""
    return ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs)))[:num_seqs]


def _rank_slots(num_seqs: int, slots: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots of a `slots`-slot pool: the last and the first, then a
    stride-37 walk over the interior (37 is coprime to 1874 - 2 and 850 - 2)."""
    rest = tuple((i * 37 + 4) % (slots - 2) + 1 for i in range(max(num_seqs - 2, 0)))
    picked = ((slots - 1, 0) + rest)[:num_seqs]
    assert len(set(picked)) == num_seqs
    return picked


def _rank_cell(why: str, geometry, seqlens, slots: int, initial, fi_path: str) -> Cell:
    """The target's call (`PrefillTarget._gdn_core`) at one rank's pool: the stock
    manager's bf16 state page at the member's slot count, the layer's view of it, a
    prefix view of a wider output buffer, `has_initial_state`, `linear_gate`. `fi_path` is
    the FlashInfer kernel the call reaches on sm_103: `cp` (the context-parallel pipeline)
    or `chunked`."""
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


@triton.jit
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
    """One program: `BLOCK` elements of sequence `n`'s head `h` plane, into the packed
    `[N, H, V, K]` buffer: the pool row `slot_ptr[n]` when `init_ptr[n]`, else +0.0 (the
    pool row is not read)."""
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


def _chunk_carried(
    q, k, v, g, beta, scale, pool, indices, has_initial_state, cu_seqlens, output, linear_gate
) -> torch.Tensor:
    """The in-place indexed call of the adapter, its gather `_gather_carried_states`: the
    packed states in the pool's dtype (the sm_100/103 kernel keeps fp32 inside), FlashInfer's
    kernel (`g` exponentiated first unless `linear_gate`), and the fused scatter of the final
    states into the same slots."""
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
    )
    cast_scatter_fp32_vk_to_vk(final, pool, indices)
    return out.unsqueeze(0)


class FlashinferChunkGatedDeltaRule(OpWrapper):
    """Run the gated delta rule over whole sequences, returning `(o, final_state)`.

    Per token `t` and value head, `h` the `[V, K]` state, `q_h`/`k_h` the key
    heads broadcast (value head `i` reads key head `i // (HV // H)`), fp32
    accumulation:

        h *= exp(g[t]);  u = (v[t] - h k_h[t]) * beta[t];  h += u k_h[t]^T
        o[t] = h (scale * q_h[t])

    computed 64 tokens at a time on tensor cores (WY form, bf16 operands) --
    the same recurrence as the Triton op, a different sum. `g` is in log space
    (the adapter exponentiates it), or with `linear_gate` (the
    `has_initial_state` call only) the factor `exp(g)` itself, as
    `gdn_prefill_conv_post(linear_gate=True)` returns it; `scale=None` is `K ** -0.5`;
    `use_qk_l2norm_in_kernel` normalizes q/k first (a Triton `l2norm_fwd`).

    One packed row (`q.shape[0] == 1`) split at `cu_seqlens` (required).
    `initial_state` is required: a pool `[S, HV, V, K]`; sequence `n` starts
    from `initial_state[initial_state_indices[n]]`, or, when `has_initial_state`
    (bool `[N]`) is given and `has_initial_state[n]` is false, from zeros, its
    slot not read (the in-place indexed call only). With
    `inplace_indexed_state_update=True` its final state is written back into
    that slot and `final_state` is `None`; otherwise `output_final_state`
    returns a fresh `[N, HV, V, K]` in the pool's dtype and the pool is not
    written.

    Fusion boundary: gate exponentiation, optional q/k norm, scale, slot
    gather (zeros for a fresh sequence with `has_initial_state`), recurrence,
    output, slot scatter. The conv, q/k/v/g/beta preparation, slot allocation
    and the gated norm on `o` are the caller's.

    Returns `(o, final_state)`: `o` `[1, T, HV, V]` in `q.dtype` (`output`
    itself when given).
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: geometry (H, HV, K, V); seqlens; `slots`-row pool with `indices`
    # (or _engine_slots); pool_dtype (fp32); output (None | "buffer" | "slice", a prefix
    # view of a wider buffer); page_layers (the pool is the layer's view of the stock
    # manager's state page, that many GDN states a slot); fi_path (the FlashInfer kernel
    # reached). Every cell is in place: tp1's at 16/48, then one rank's of tp4/tep4/dep4.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"qwen3.8 prefill into output=, 461-slot pool, seqlens {list(s)}",
            spec=dict(
                geometry=QWEN38_GEOMETRY,
                seqlens=s,
                slots=ENGINE_SLOTS,
                indices=_engine_slots(len(s)),
                output="buffer",
            ),
        )
        for s in ((200, 37, 91), (8192,))
    ) + (
        Cell(
            why="qwen3.8 context chunk + 1-token rows [8, 1, 1, 1], 461-slot pool",
            spec=dict(
                geometry=QWEN38_GEOMETRY,
                seqlens=(8, 1, 1, 1),
                slots=ENGINE_SLOTS,
                indices=_engine_slots(4),
            ),
        ),
        Cell(
            why="bf16 pool (mamba_ssm_cache_dtype), 461 slots, seqlens [200, 37, 91]",
            spec=dict(
                geometry=QWEN38_GEOMETRY,
                seqlens=(200, 37, 91),
                slots=ENGINE_SLOTS,
                indices=_engine_slots(3),
                pool_dtype=torch.bfloat16,
            ),
        ),
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
    ) + (
        _rank_cell(
            "tep4 4/12, fresh 8192-token prompt, CP path, 1874-slot bf16 page",
            RANK4_GEOMETRY, (8192,), TEP4_SLOTS, (False,), "cp",
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
    )  # fmt: skip

    note = """
    Metadata consumed: none.

    Call site as qwen3.8-flash-next-nvfp4's `_gdn_block` makes it (in place of
    the Triton `chunk_gated_delta_rule`):
    the same positional arguments, `(q, k, v, g, beta, None, ssm_states,
    si, True, False, qsl_long, False, False, None)` with `head_first=False`
    in the 12th slot, after `fused_gdn_post_conv` (q/k already normalized,
    contiguous `[1, T, H, 128]`; `g` fp32 log-space; `beta` fp32). Give it
    context rows only: single-token rows are slow here. Measured per layer
    at 8065 + 127 x 1 tokens: 1078.8 us with every row in this op, 399.4 us
    with the 127 decode rows sent to FlashInfer's decode kernel
    (`gdn_decode.gated_delta_rule_decode_pretranspose`) instead (both halves
    plus the concatenation).

    The target (`PrefillTarget._gdn_core`, `_gdn_core_verify`) passes the
    context rows by keyword after `gdn_prefill_conv_post(linear_gate=True)`,
    with `has_initial_state` and `linear_gate=True`.

    Call site with `has_initial_state` (`hi_p` / `hi_all`, the flags
    `reset_recurrent_state_rows` would zero the fresh slots by): a fresh
    sequence's slot is never read, and the scatter writes every named slot,
    so the reset of the recurrent pool before this call is not needed (the
    conv pass reads no carry for such a sequence and its publish writes the
    whole conv window). Bitwise the reset followed by the plain call; the
    gather moves only the carried sequences' rows.

    Certified at the 461-slot fp32 pool: three ragged context rows and one
    8192-token row, each into an `output=` buffer, and an 8-token context
    chunk with three one-token rows; and at the 461-slot bf16 pool: three
    ragged context rows, and the target's call (300 + 1024 + 500 tokens,
    the middle one fresh, `has_initial_state`, `linear_gate`, `output=`).

    Certified per rank, as `PrefillTarget._gdn_core` calls it (bf16 pool,
    `has_initial_state`, `linear_gate`, `output=` a prefix view of a wider
    buffer whose tail stays untouched). The pool is the layer's view of the
    stock manager's state page (`PAGE_LAYERS` states a slot), at the
    member's slot count:
      tp4/tep4 (`RANK4_GEOMETRY`, 4 k / 12 v heads), 1874 slots: one fresh
        8192-token prompt, a carried 8192-token chunk, [200, 37, 91]
        carried/fresh/carried, and 8192 rows over 128 sequences of 63/65
        tokens (the envelope: max_num_tokens over max_batch_size).
      dep4 (16 / 48), 850 slots: a carried 512-token prompt.
    The last slot is named at every one: offsets of 1.3e10 (tep4) and
    2.4e10 (dep4) elements, past int32; the gather and the scatter index the
    slot in int64.

    FlashInfer picks its kernel by `num_seqs x num_v_heads`: on sm_10x the
    context-parallel pipeline (`cp_delta_rule_dsl_sm100`: t precompute, mn
    precompute, fixup, prefill) when that times 4 is below the SM count
    (`should_use_cp_host`), else the chunked kernel. GB300 has 152 SMs, so a
    tp4/tep4 rank (12 v heads) runs CP for 1 to 3 context sequences and the
    chunked kernel from 4; a dep4 rank (48) and tp1 always run the chunked
    one. Both paths take the bf16 packed state. The cells hold the path
    they reach (`fi_path`), and a CP fallback warning fails them.

    Measured per rank cell against the fp64 reference (bf16 pool): o rel-L2
    3.72e-3 to 3.77e-3, cosine 0.999993, max abs 7.3e-4; named slots'
    states rel-L2 3.25e-3 to 3.47e-3, max abs 3.9e-3 -- the CP pipeline and
    the chunked kernel alike. A value head reading the wrong key head at
    4/12 (key heads rolled by one) is o rel-L2 1.41, max abs 7.6e-2, which
    the band refuses.

    Memory per call: a packed copy of the named slots in and out in the pool's
    dtype (2 x 3 MiB per sequence fp32, half that bf16), `exp(g)` (not with
    `linear_gate`), and the
    output. q/k/v/beta are not copied
    when already contiguous / fp32.

    JIT and streams: the first call per process and path compiles the
    CuTe-DSL kernels in-process for the device it runs on (seconds; the
    cells compiled and ran both paths for sm_103), and the installed
    FlashInfer keeps one TMA workspace per device (the CP pipeline also a
    cached initial-state workspace), so two streams must not run this op
    concurrently. Needs CUDA >= 13 torch (FlashInfer raises otherwise).

    Numerics (measured vs an fp64 recurrence, 1 x 8192 tokens, qwen3.8
    geometry): o rel-L2 4.06e-3, max abs 1.26e-3; state rel-L2 3.31e-3 --
    the Triton op's are
    4.00e-3 / 1.13e-3 / 3.26e-3. Not bitwise to it. A bf16 pool rounds each
    sequence's state to bf16 once per call, on the scatter. No cold-call
    defect: nothing in the call is autotuned.

    Loud, from the callable: `head_first=True`, `q.shape[0] != 1`, H_q != H_k,
    fp32 q, non-fp32 `g`, `cu_seqlens` or `initial_state` None, in-place
    without indices.

    Not checked: indices distinct and in `[0, S)`; q/k unit-norm (by the
    caller or the flag).

    Not certified: `state_workspace=`, an fp16 pool, `scale` other than the
    default, fp16 operands, K or V other than 128, head layouts other than
    16/48 and 4/12, and what the target never passes:
    `inplace_indexed_state_update=False` (final states returned) and
    `use_qk_l2norm_in_kernel`. At 4/12: 2 context sequences on CP and 4 to
    127 on the chunked kernel (the cells hold 1 and 3, and 128); a dep4 rank
    at more than 512 carried tokens; tp4's own slot count (no engine has
    run tp4; tep4's stands in) and any pool counted on the unified target's own
    engines or under a variant (MTP, kv_fp8/kv_nvfp4, chunked, reuse);
    a call inside a CUDA graph (the target calls it eagerly).

    Hard limits
      T: max_num_tokens (8192) rows a call is certified at; chunked
         prefill of a longer prompt is one call per chunk.
      slot offsets: past int32 at both members' last slots (certified).
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
        assert not linear_gate or has_initial_state is not None, (
            "linear_gate is taken with has_initial_state only"
        )
        if has_initial_state is not None:
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
            o = _chunk_carried(
                q, k, v, g, beta, scale, initial_state, initial_state_indices,
                has_initial_state, cu_seqlens, output, linear_gate,
            )  # fmt: skip
            return o, None
        return _fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            inplace_indexed_state_update=inplace_indexed_state_update,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output=output,
            state_workspace=state_workspace,
        )

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
        """`(o, final_state, initial_state as the call leaves it)`.

        fp64 token-at-a-time recurrence, rounded once: `o` to `q.dtype`, the
        states to the pool's dtype. `final_state` is `None` on the in-place
        path, as the callable returns it. Mutates nothing; `output`,
        `head_first` and `state_workspace` are ignored (they choose where
        results land)."""
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
