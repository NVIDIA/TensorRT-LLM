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

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.flashinfer_chunk import (
    chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
)
from tensorrt_llm._torch.modules.fla.fused_state_io import cast_scatter_fp32_vk_to_vk

#: qwen3.8-flash-next-nvfp4's Gated DeltaNet at tp1: (H, HV, K, V) -- 16 q/k
#: heads, 48 v heads, head dims 128.
QWEN38_GEOMETRY = (16, 48, 128, 128)
#: Largest slot count the cache manager was seen to allocate the recurrent pool
#: at for qwen3.8-flash-next-nvfp4 (10 .. 461 over twelve runtime-value dumps).
ENGINE_SLOTS = 461


def _engine_slots(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots of the engine pool, slot 0 and the last live."""
    return ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs)))[:num_seqs]


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
    # (or _engine_slots); pool_dtype (fp32); output (None | "buffer"). Every cell is
    # qwen3.8-flash-next-nvfp4's geometry, in place.
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
    )

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

    Memory per call: a packed copy of the named slots in and out in the pool's
    dtype (2 x 3 MiB per sequence fp32, half that bf16), `exp(g)` (not with
    `linear_gate`), and the
    output. q/k/v/beta are not copied
    when already contiguous / fp32.

    JIT and streams: the first call per process compiles the CuTe-DSL
    kernel in-process (seconds), and the installed FlashInfer keeps one TMA
    workspace per device, so two streams must not run this op concurrently.
    Needs CUDA >= 13 torch (FlashInfer raises otherwise).

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
    default, fp16 operands, K or V other than 128, other head ratios, and what the
    target never passes: `inplace_indexed_state_update=False` (final states
    returned) and `use_qk_l2norm_in_kernel`.

    Hard limits
      none probed.
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
