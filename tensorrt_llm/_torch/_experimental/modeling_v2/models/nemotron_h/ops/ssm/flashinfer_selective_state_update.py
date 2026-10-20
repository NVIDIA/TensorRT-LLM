# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 decode step through FlashInfer's CUDA kernel: one token per row, the SSM state
advanced in place in its pool slot -- the step `selective_state_update` (the Triton kernel)
takes, faster at large decode batches and slower at small ones."""

from __future__ import annotations

from typing import Optional

import torch
from flashinfer.mamba import selective_state_update as _fi_ssu

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Per-rank geometry of the Nemotron-3 Ultra BF16 targets (tep8, tp8ep2): 256 heads / 8, head_dim
#: 64, 8 groups / 8, d_state 128, the SSM state fp32 (config `mamba_ssm_cache_dtype`).
NHEADS, HEADDIM, NGROUPS, DSTATE = 32, 64, 1, 128
#: The engine's SSM pool for this checkpoint at max_batch_size 256: 271 slots, every slot holding
#: all 48 Mamba layers' states side by side, so a layer's view `[271, 32, 64, 128]` has slot stride
#: 48 * 32 * 64 * 128 = 12.6M elements and the slots from 171 on lie past 2^31 elements of the base.
#: At max_batch_size 32 the same pool has 35 slots.
ENGINE_SLOTS_256, ENGINE_SLOTS_32, ENGINE_SLOT_LAYERS = 271, 35, 48

#: 48 scattered slots of the 271-slot pool, slot 0, the last slot and slots past 2^31 included.
_SLOTS_48 = (ENGINE_SLOTS_256 - 1, 0, 170, 171, 269) + tuple(7 + 6 * i for i in range(43))


class FlashinferSelectiveStateUpdate(OpWrapper):
    """One recurrence step for every row of a decode batch, in one launch, fp32 inside, the
    state read from and written back to `state[slot]` -- the same contract as
    `selective_state_update`.

    For row `b`, `slot = state_batch_indices[b]`, head `h`, group `g = h // (nheads / ngroups)`,
    with `dt' = softplus(dt + dt_bias)`: `S = state[slot] * exp(A * dt') + dt' * outer(x[b,h],
    B[b,g])`, `out[b,h] = S @ C[b,g] + D * x[b,h]`, and `state[slot] = S` rounded once to the
    pool dtype. A row whose slot equals `pad_slot_id` reads a zero state, still writes `out`, and
    leaves the pool alone.

    Fusion boundary: dt bias + softplus, the decay, the update, the readout, the D skip and the
    state write are inside; the conv update that produces x/B/C is the caller's.

    Writes `out` (required, `x.shape`) and the named pool slots; returns `out`.
    """

    ARCHS = frozenset({Arch.SM_100})

    # spec keys: batch, pool_slots / pool_layers (the pool as `[slots, layers, H, P, N]` viewed at
    # one layer; default the 271-slot x 48-layer engine pool), slots (explicit slot per row,
    # default a seeded draw), pads (rows set to pad_slot_id). Always the targets' geometry: 32
    # heads of 64 over one group of 128 states, an fp32 pool, D and dt_bias bf16 as stored.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode batch 32, where the targets switch to this kernel, at the engine's "
            "35-slot pool (max_batch_size 32)",
            spec=dict(batch=32, pool_slots=ENGINE_SLOTS_32),
        ),
        Cell(
            why="decode batch 37: a row count no CUDA graph pads to (a mixed step's decode rows)",
            spec=dict(batch=37),
        ),
        Cell(
            why="decode batch 48, scattered slots: 0, the last, and both sides of 2^31 elements",
            spec=dict(batch=48, slots=_SLOTS_48),
        ),
        Cell(
            why="decode batch 64",
            spec=dict(batch=64),
        ),
        Cell(
            why="decode batch 128",
            spec=dict(batch=128),
        ),
        Cell(
            why="decode batch 256, the serving envelope's ceiling: 256 of the 271 slots",
            spec=dict(batch=256),
        ),
        Cell(
            why="pad_slot_id -1 rows: pool untouched, out still written",
            spec=dict(batch=32, pads=(1, 17, 31)),
        ),
    )

    note = """
    Call site (the decode step of both nemotron_h BF16 targets, from 32 rows on; below that,
    and for a mixed step's generation rows, they call `selective_state_update`, the Triton
    kernel):

        y = torch.empty(B, 32, 64, dtype=torch.bfloat16)
        flashinfer_selective_state_update(pool, x.view(B, 32, 64),
            dt[:, :, None].expand(B, 32, 64), A_b, Bm.view(B, 1, 128),
            Cm.view(B, 1, 128), D_b16, None, dt_bias_b16, True, si, -1, y)

    with, built once (post_load_weights): A_b = A[:, None, None].expand(32, 64, 128) (fp32
    `-exp(A_log)`), D_b16 = D16[:, None].expand(32, 64) and dt_bias_b16 the same, D16 and
    dt_bias16 the checkpoint's bf16 values. The kernel takes D and dt_bias only in dt's dtype
    and A, D, dt and dt_bias only as these stride-0 broadcasts (one dt and one decay per
    head), and refuses anything else loudly. x/B/C are views of the token-major conv output
    (last dim contiguous, row stride free) that must each start 128-byte aligned: the kernel
    reads them through TMA and refuses a misaligned start, loudly. A decode step's in_proj
    output satisfies that (x, B and C at 4096, 8192 and 8448 bytes into an aligned row 0);
    a mixed step's generation rows do not after an odd number of context rows (8768 bytes
    each), which is why the targets keep those on the Triton kernel. The pool is the layer's
    `[slots, 32, 64, 128]` view of the V2 pool (slot stride 48 layers, inner dims dense),
    fp32, updated in place with int64 slot offsets.

    The kernel picks its algorithm per call from the row count: below two blocks per SM
    (`batch * nheads`) the tiled "simple" kernel, above it the producer-consumer "vertical"
    one (Blackwell, fp32 state). Every cell from 32 rows on runs the vertical kernel on a
    GB200.

    CUDA graphs: capturable (no host sync; slots read on device at replay). The kernel is
    JIT-compiled on first call per dtype / shape specialization, so warmup must run each
    captured batch size before capture, as for the Triton kernel.

    Numerics: one fp32 step, rounded once, as `selective_state_update`, with the fast
    intrinsics: softplus is `__logf(1 + __expf(v))` and the decay `__expf(A dt)`. Their error
    in dt is larger than the Triton kernel's -- the test bands it from the intrinsics'
    documented bounds -- and reaches the fp32 state where the increment `dt x B` dominates.

    Not certified: z, multi-token rows (MTP), varlen, intermediate states, int16 state with
    `state_scale`, stochastic rounding (`rand_seed`), `dst_state_batch_indices`,
    `disable_state_update`, and any pool dtype other than fp32.
    """

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
        state_batch_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        _fi_ssu(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
        )
        return out

    def reference(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
        state_batch_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, state)` as the call leaves them; mutates nothing.

        The single-token form only (x `[batch, H, P]`), in fp32, rounded once: `out` to its
        dtype, each named slot to the pool's dtype.
        """
        assert x.dim() == 3, "only single-token rows [batch, H, P] are certified"
        batch = x.shape[0]
        rep = x.shape[1] // B.shape[1]
        if state_batch_indices is None:
            slots = torch.arange(batch, device=x.device)
        else:
            slots = state_batch_indices.long()
        live = slots != pad_slot_id if state_batch_indices is not None else slots >= 0
        s0 = state[slots.clamp_min(0)].float() * live.view(-1, 1, 1, 1)
        dtf = dt.float() + (0.0 if dt_bias is None else dt_bias.float())
        if dt_softplus:
            dtf = torch.where(dtf <= 20.0, torch.log1p(torch.exp(dtf)), dtf)
        xf = x.float()
        bf = B.float().repeat_interleave(rep, dim=1)  # [batch, H, N]
        cf = C.float().repeat_interleave(rep, dim=1)
        s1 = s0 * torch.exp(A.float() * dtf[..., None]) + (dtf * xf)[..., None] * bf[:, :, None, :]
        y = (s1 * cf[:, :, None, :]).sum(-1) + D.float() * xf
        if z is not None:
            y = y * torch.nn.functional.silu(z.float())
        after = state.clone()
        after[slots[live]] = s1[live].to(state.dtype)
        return y.to(out.dtype), after

    def is_valid(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
        state_batch_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        out: torch.Tensor,
    ) -> None:
        # Shapes, strides and dtypes the kernel checks itself, loudly. What it does not check
        # is where the slots point.
        if state_batch_indices is None:
            return
        slots = state_batch_indices.tolist()
        num_slots = state.shape[0]
        # A pad id that is also a real slot silently skips that slot's update.
        assert not 0 <= pad_slot_id < num_slots, (
            f"pad_slot_id {pad_slot_id} is a real slot of a {num_slots}-slot pool; a row "
            "naming that slot would be skipped"
        )
        live = [s for s in slots if s != pad_slot_id]
        # The kernel adds slot * stride to the pool's base without a bound: a slot past the
        # view lands in whatever memory follows it.
        assert all(0 <= s < num_slots for s in live), (
            f"state_batch_indices outside [0, {num_slots}) write outside the pool view"
        )
        # Two rows naming one slot race on the store: one row's update is lost.
        assert len(set(live)) == len(live), "two rows name one slot; one row's update is lost"


flashinfer_selective_state_update = FlashinferSelectiveStateUpdate()
