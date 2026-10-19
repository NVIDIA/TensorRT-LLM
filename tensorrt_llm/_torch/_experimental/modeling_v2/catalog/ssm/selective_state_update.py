# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 decode step: one token per row, the SSM state advanced in place in
its pool slot."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba import selective_state_update as _ssu

#: Per-rank geometry at tep4: 256 heads / 4, head_dim 64, 8 groups / 4, d_state 128.
NHEADS, HEADDIM, NGROUPS, DSTATE = 64, 64, 2, 128
#: At dep4 (attention DP) every rank runs the whole mixer: 256 heads, 8 groups.
DEP4_NHEADS, DEP4_NGROUPS = 256, 8
#: The engine's SSM pool for this checkpoint (V2 hybrid manager, max_batch_size 32): 35 slots,
#: every slot holding all 48 Mamba layers' states side by side, so a layer's view
#: `[35, 64, 64, 128]` has slot stride 48 * 64 * 64 * 128 and dense inner dims. At dep4
#: (the manager uses tp=1 for Mamba under attention DP) measured: `[35, 256, 64, 128]`,
#: slot stride 48 * 256 * 64 * 128 -- slots from 22 on lie past 2^31 elements.
ENGINE_SLOTS = 35
ENGINE_SLOT_LAYERS = 48
#: A pool deep enough that its last slot sits past 2^31 elements from the base:
#: 90 * 48 * 64 * 64 * 128 > 2^31.
DEEP_SLOTS = 90

_DEP4 = dict(nheads=DEP4_NHEADS, ngroups=DEP4_NGROUPS)
#: Seven scattered engine slots, both ends and slots past 2^31 elements included.
_DEP4_SLOTS_7 = (ENGINE_SLOTS - 1, 0, 22, 5, 30, 13, 27)


class SelectiveStateUpdate(OpWrapper):
    """One recurrence step for every row of a decode batch, in one launch,
    fp32 inside, the state read from and written back to `state[slot]`.

    For row `b`, `slot = state_batch_indices[b]` (or `b` when None), head `h`,
    group `g = h // (nheads / ngroups)`, with `dt' = softplus(dt + dt_bias)`:
    `S = state[slot] * exp(A * dt') + dt' * outer(x[b,h], B[b,g])`,
    `out[b,h] = S @ C[b,g] + D * x[b,h]` (times `silu(z)` when given), and
    `state[slot] = S` rounded once to the pool dtype. A row whose slot equals
    `pad_slot_id` reads a zero state, still writes `out`, and leaves the pool
    alone.

    Fusion boundary: dt bias + softplus, the decay, the update, the readout,
    the D skip and the state write are inside; the conv update that produces
    x/B/C is the caller's.

    Writes `out` (required, `x.shape`) and the named pool slots; returns `out`.
    """

    ARCHS = frozenset({Arch.SM_107})

    # spec keys: batch, pool_dtype (default fp16), pool_slots / pool_layers (the pool
    # as `[slots, layers, H, P, N]` viewed at one layer; default 40 x 3), slots
    # (explicit slot per row, default a seeded draw), pads (rows set to pad_slot_id),
    # param_dtype (D / dt_bias, default bf16 as stored), nheads / ngroups (default
    # the tep4 64 / 2; head_dim 64 and d_state 128 at both).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode batch 1 at the engine's fp16 35-slot x 48-layer pool",
            spec=dict(batch=1, pool_slots=ENGINE_SLOTS, pool_layers=ENGINE_SLOT_LAYERS),
        ),
        Cell(
            why="decode batch 7, scattered slots, fp16 pool",
            spec=dict(batch=7),
        ),
        Cell(
            why="decode batch 32 (the CUDA-graph ceiling) at the engine's fp16 pool",
            spec=dict(batch=32, pool_slots=ENGINE_SLOTS, pool_layers=ENGINE_SLOT_LAYERS),
        ),
        Cell(
            why="decode batch 1, fp32 pool",
            spec=dict(batch=1, pool_dtype=torch.float32),
        ),
        Cell(
            why="decode batch 7, fp32 pool",
            spec=dict(batch=7, pool_dtype=torch.float32),
        ),
        Cell(
            why="decode batch 32 at the engine's 35-slot x 48-layer pool, fp32",
            spec=dict(
                batch=32,
                pool_dtype=torch.float32,
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
            ),
        ),
        Cell(
            why="pad_slot_id -1 rows: pool untouched, out still written",
            spec=dict(batch=7, pads=(1, 4)),
        ),
        Cell(
            why="slots past 2^31 elements from the pool base (int64 slot offset)",
            spec=dict(
                batch=4,
                pool_slots=DEEP_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
                slots=(DEEP_SLOTS - 1, 0, DEEP_SLOTS - 2, 45),
            ),
        ),
        Cell(
            why="a dense pool (unit-layer slot stride)",
            spec=dict(batch=7, pool_layers=1),
        ),
        Cell(
            why="fp32 D and dt_bias instead of the checkpoint's bf16",
            spec=dict(batch=7, param_dtype=torch.float32),
        ),
        Cell(
            why="dep4 decode batch 1 into the engine fp16 pool's last slot (past 2^31)",
            spec=dict(
                _DEP4,
                batch=1,
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
                slots=(ENGINE_SLOTS - 1,),
            ),
        ),
        Cell(
            why="dep4 decode batch 7, scattered engine slots, fp16 pool",
            spec=dict(
                _DEP4,
                batch=7,
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
                slots=_DEP4_SLOTS_7,
            ),
        ),
        Cell(
            why="dep4 decode batch 32 (the CUDA-graph ceiling), engine fp16 pool",
            spec=dict(_DEP4, batch=32, pool_slots=ENGINE_SLOTS, pool_layers=ENGINE_SLOT_LAYERS),
        ),
        Cell(
            why="dep4 decode batch 7, engine fp32 pool",
            spec=dict(
                _DEP4,
                batch=7,
                pool_dtype=torch.float32,
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
                slots=_DEP4_SLOTS_7,
            ),
        ),
        Cell(
            why="dep4 decode batch 32, engine fp32 pool",
            spec=dict(
                _DEP4,
                batch=32,
                pool_dtype=torch.float32,
                pool_slots=ENGINE_SLOTS,
                pool_layers=ENGINE_SLOT_LAYERS,
            ),
        ),
        Cell(
            why="dep4 pad_slot_id -1 rows: pool untouched, out still written",
            spec=dict(_DEP4, batch=7, pads=(1, 4)),
        ),
    )

    note = """
    Call site (DecodeTarget, and the decode rows of PrefillTarget; si =
    mm.state_indices[rows], int32, contexts first in a mixed batch):

        y = torch.empty(B, 64, 64, dtype=torch.bfloat16)
        selective_state_update(pool, x.view(B, 64, 64),
            dt[:, :, None].expand(B, 64, 64), A_b, Bm.view(B, 2, 128),
            Cm.view(B, 2, 128), D_b, None, dt_bias_b, True, si, -1, y)

    with, built once (post_load_weights): A_b = A[:, None, None].expand(64, 64,
    128) (fp32 `-exp(A_log)`), D_b = D[:, None].expand(64, 64), dt_bias_b =
    dt_bias[:, None].expand(64, 64). A, dt and dt_bias all being stride-0
    broadcasts on their trailing dims is what selects the kernel's tied-head
    path (one dt and one decay per head); a dense copy computes the same
    numbers on the general path. D and dt_bias must be `[64, 64]`: a `[64]`
    is unsqueezed to `[1, 64]` and refused by a shape assert. dt_bias may not
    be None on the tied path (the wrapper reads its stride).

    x/B/C are views of the token-major `[B, 4608]` conv output and dt of the
    in_proj output; every stride is honoured. `pool` is the layer's
    `[slots, 64, 64, 128]` view of the V2 pool (slot stride 48 layers, inner
    dims dense), fp16 or fp32, updated in place.

    At dep4 the same call with 256 heads and 8 groups: x / dt / y
    `(B, 256, 64)`, B/C `(B, 8, 128)` views of the `[B, 18432]` conv output,
    A_b `(256, 64, 128)`, D_b / dt_bias_b `(256, 64)`, the pool
    `[slots, 256, 64, 128]` with slot stride 48 * 256 * 64 * 128, so slot
    offsets past 2^31 elements are the normal case (the `dep4` cells).

    CUDA graphs: capturable and certified so (one Triton launch, no autotune,
    no host sync, slots read on device at replay). The pool, the
    activation buffers and `si` must keep their addresses across replays
    (`state_indices` aliases a stable manager buffer). Triton compiles one
    kernel per specialization on first call -- batch 1, batch % 16 == 0, and
    other -- so warmup must run each captured batch size before capture.

    Numerics: one fp32 step, rounded once -- within one rounding of the fp32
    recurrence in bf16 y and in an fp16 pool (99.96% / 99.8% bitwise). The
    kernel computes softplus as log(1 + exp(v)) in fp32, so dt is off by up
    to one fp32 rounding of 1 + e^v (6e-8, 6% of dt at v = -14) and the
    increment dt * x B with it: an fp32 pool shows up to 3.7e-4 of the
    state's magnitude where the increment dominates. An fp16 pool rounds the
    state to nearest on every step (no stochastic rounding): a slow-decay head
    whose per-step increment is below half an fp16 ulp does not move. The
    prefill/decode chain test measures the drift from the fp32 recurrence
    (U = 2^-8: 0.45 U RMS on the state after 64 steps, 1.24 U after 256; fp32
    stays near 0.06 U).

    Not certified: z, multi-token rows (MTP), intermediate_states_buffer /
    cache_steps / retrieve_parent_token / intermediate_state_indices,
    disable_state_update, launch_with_pdl (the state is loaded before the
    PDL wait).
    """

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
        state_batch_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        out: torch.Tensor,
        disable_state_update: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        launch_with_pdl: bool = False,
    ) -> torch.Tensor:
        _ssu.selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
            launch_with_pdl=launch_with_pdl,
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
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
        state_batch_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        out: torch.Tensor,
        disable_state_update: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        launch_with_pdl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, state)` as the call leaves them; mutates nothing.

        The single-token form only (x `[batch, H, P]`), in fp32, rounded once:
        `out` to its dtype, each named slot to the pool's dtype. `launch_with_pdl`
        changes scheduling, not values; the MTP arguments are not mirrored.
        """
        assert x.dim() == 3, "only single-token rows [batch, H, P] are certified"
        assert intermediate_states_buffer is None and retrieve_parent_token is None, (
            "the MTP paths are not certified"
        )
        assert intermediate_state_indices is None and cache_steps is None
        batch, nheads, headdim = x.shape
        rep = nheads // B.shape[1]
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
        y = (s1 * cf[:, :, None, :]).sum(-1)
        if D is not None:
            y = y + D.float() * xf
        if z is not None:
            y = y * torch.nn.functional.silu(z.float())
        after = state.clone()
        if not disable_state_update:
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
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
        state_batch_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        out: torch.Tensor,
        disable_state_update: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        launch_with_pdl: bool = False,
    ) -> None:
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
        # The kernel adds slot * stride to the pool's base without a bound: a slot
        # past the view lands in whatever memory follows it.
        assert all(0 <= s < num_slots for s in live), (
            f"state_batch_indices outside [0, {num_slots}) write outside the pool view"
        )
        # Two rows naming one slot race on the store: one row's update is lost.
        assert len(set(live)) == len(live), "two rows name one slot; one row's update is lost"


selective_state_update = SelectiveStateUpdate()
