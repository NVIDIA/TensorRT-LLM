# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fold each selected slot's pending accepted drafts into its recurrent state: the KDA verify kernel's replay steps
alone (CuTe DSL, its replay-only variant)."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import (
    kda_mtp_commit_pending_drafts as _kda_mtp_commit_pending_drafts,
)


def _variant_key(tensors, ssm_state_indices, num_spec, conv_width) -> tuple:
    """What selects the kernel variant: the buffers' dtypes, shapes and strides, the slot count, the replay depth and
    the conv width (a superset of the kernel's compile key)."""
    return (
        tensors[0].device.index,
        tuple(None if t is None else (t.dtype, tuple(t.shape), tuple(t.stride())) for t in tensors),
        int(ssm_state_indices.shape[0]),
        int(num_spec),
        int(conv_width),
    )


#: Variant keys this process has run eagerly (their kernels are compiled).
_RUN_VARIANTS: set = set()


class KdaMtpCommitPendingDrafts(OpWrapper):
    """Replay the first `num_accepted_tokens[n]` cached drafts of slot `ssm_state_indices[n]` onto
    `recurrent_state[slot]` and commit the result in place. Returns None.

    The replay caches (`k/g/v/beta_cache`) and the conv caches (`cs_*`) are only read: the caller then
    stores the conv window after the drafts (`store_kda_replay_conv_window`) and clears the slot's count.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # The cache manager commits one slot per call (a suspended request), once per KDA layer. A slot with
    # nothing pending is committed unchanged. Two CTAs per (head, request) while HV * N <= 64, one above.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP1 / attention DP: one slot at 96 heads with all seven drafts of a w8 block pending",
            spec=dict(heads=96, num_spec=7, slots=(4,), accepted=(7,)),
        ),
        Cell(
            why="TP2: one slot at 48 heads, two of four drafts pending, two CTAs per head",
            spec=dict(heads=48, num_spec=4, slots=(1,), accepted=(2,)),
        ),
        Cell(
            why="TP4: one slot at 24 heads, three of seven drafts pending",
            spec=dict(heads=24, num_spec=7, slots=(5,), accepted=(3,)),
        ),
        Cell(
            why="TP8: one slot at 12 heads, all four drafts pending",
            spec=dict(heads=12, num_spec=4, slots=(0,), accepted=(4,)),
        ),
        Cell(
            why="TP16: one slot at 6 heads, one draft pending",
            spec=dict(heads=6, num_spec=7, slots=(3,), accepted=(1,)),
        ),
        Cell(
            why="TP32: one slot at 3 heads with nothing pending commits the state unchanged",
            spec=dict(heads=3, num_spec=4, slots=(2,), accepted=(0,)),
        ),
        Cell(
            why="eight slots at once, reverse order, 0..7 pending: the op's batched form",
            spec=dict(
                heads=6, num_spec=7, slots=(9, 7, 5, 3, 1, 0, 2, 4), accepted=tuple(range(8))
            ),
        ),
        Cell(
            why="four slots at 24 heads (HV * N = 96): one CTA per head",
            spec=dict(heads=24, num_spec=4, slots=(6, 0, 3, 8), accepted=(4, 0, 2, 1)),
        ),
        Cell(
            why="Rubin helix8: one slot at 12 heads, all five drafts of a w6 block pending",
            spec=dict(heads=12, num_spec=5, slots=(2,), accepted=(5,)),
        ),
        Cell(
            why="Rubin TEP8: one slot at 12 heads, three of seven drafts pending",
            spec=dict(heads=12, num_spec=7, slots=(1,), accepted=(3,)),
        ),
    )

    note = """
    The cache manager's commit (`commit_pending_drafts` via `commit_kda_pending_drafts`, no per-draft
    records), reached when a request is suspended with accepted drafts still pending, not per step:
    ssm_state_indices int32 [N], num_accepted_tokens request-indexed int32 [N] (the slot's gathered
    prev_num_accepted_tokens), num_spec the replay caches' depth, conv_width 4. Pools and caches as for
    kda_mtp_decode, carved per slot by the V2 manager with padded slot strides: recurrent_state fp32
    V-first [slots, H, 128, 128], k / v bf16 and g fp32 [slots, num_spec, H * 128], beta bf16
    [slots, num_spec, H] over rows padded to 16 bytes, cs_* bf16 dim-contiguous [slots, H * 128,
    3 + num_spec] (validated, never read). Token inputs, conv weights and gate constants are per
    (device, N, H) stand-ins the kernel never reads.

    In place: recurrent_state of the selected slots only; caches, unselected slots and slot padding are
    untouched, and a slot with nothing pending keeps its state bit for bit. The kernel computes the
    replayed chunk in one tensor-core pass, so the test holds the state within 2e-4 of an fp32 sequential
    replay (relative to its largest magnitude; measured 1.3e-5 on sm_103), not bit for bit; a verify that
    replays the drafts itself rounds differently by the same order. Compiled per layout key, warm before
    any graph capture (the K3.1 cache manager warms its pools' variant once they are bound; under capture a layout
    not run before raises). ARCHS lists only what the cells were driven on.
    """

    def __call__(
        self,
        recurrent_state: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        k_cache: torch.Tensor,
        g_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec: int,
        conv_width: int,
    ) -> None:
        tensors = (recurrent_state, cs_q, cs_k, cs_v, k_cache, g_cache, v_cache, beta_cache)
        key = _variant_key(tensors, ssm_state_indices, num_spec, conv_width)
        if key not in _RUN_VARIANTS and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "kda_mtp_commit_pending_drafts: a layout not run before this CUDA graph capture would compile "
                "its kernel inside it; warm it first (K31KDAReplayState.warm_commit_pending_drafts)"
            )
        _kda_mtp_commit_pending_drafts(
            recurrent_state,
            cs_q,
            cs_k,
            cs_v,
            k_cache,
            g_cache,
            v_cache,
            beta_cache,
            ssm_state_indices,
            num_accepted_tokens,
            num_spec,
            conv_width=conv_width,
        )
        _RUN_VARIANTS.add(key)

    def reference(
        self,
        recurrent_state: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        k_cache: torch.Tensor,
        g_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec: int,
        conv_width: int,
    ) -> None:
        """fp32 sequential replay of the cached drafts; cs_*, num_spec and conv_width are not needed."""
        heads = recurrent_state.shape[1]
        for slot, accepted in zip(ssm_state_indices.tolist(), num_accepted_tokens.tolist()):
            if accepted == 0:
                continue
            state = recurrent_state[slot].float()  # [H, V, K]
            for step in range(accepted):
                decay = g_cache[slot, step].float().reshape(heads, -1).exp()
                key = k_cache[slot, step].float().reshape(heads, -1)
                value = v_cache[slot, step].float().reshape(heads, -1)
                gain = beta_cache[slot, step].float().reshape(heads, 1)
                state = state * decay.unsqueeze(1)
                err = (value - (state * key.unsqueeze(1)).sum(-1)) * gain
                state = state + err.unsqueeze(-1) * key.unsqueeze(1)
            recurrent_state[slot] = state.to(recurrent_state.dtype)

    def is_valid(
        self,
        recurrent_state: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        k_cache: torch.Tensor,
        g_cache: torch.Tensor,
        v_cache: torch.Tensor,
        beta_cache: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec: int,
        conv_width: int,
    ) -> None:
        # Shapes, dtypes, strides, the count vector's length and the conv width the op checks itself, loudly.
        slots = ssm_state_indices.tolist()
        assert len(set(slots)) == len(slots), (
            "ssm_state_indices must name each slot once: requests sharing a slot race on its state"
        )
        assert all(0 <= slot < recurrent_state.shape[0] for slot in slots), (
            "a slot index outside the state pool is read and written without a bounds check"
        )
        assert all(0 <= a <= num_spec for a in num_accepted_tokens.tolist()), (
            "pending accepted drafts must lie in [0, num_spec]: the kernel clamps a larger count to "
            "num_spec and mis-addresses a negative one"
        )


kda_mtp_commit_pending_drafts = KdaMtpCommitPendingDrafts()
