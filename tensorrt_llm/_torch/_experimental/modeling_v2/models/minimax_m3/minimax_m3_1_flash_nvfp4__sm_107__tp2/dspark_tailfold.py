# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MEMCPY86K (the MEMCPY fold of the decode tail): the drafter's layer-0 residual stream starts in the
DRAFTER_INPUTS_FUSED noise block instead of a copy of it. The TAILFOLD item of the sm_107 TP4 target (its
``dspark_tailfold``: the MEMCPY86K part only).

The drafter forward's first layer copies the noise block into its residual stream (``residual =
hidden_states.clone()``, modeling_dflash.py): a graph memcpy node right before the first RMSNorm, 86,016 bytes at K7
batch 1 and 172,032 bytes at batch 2 ([B, 7, 6144] bf16). The DRAFTER_INPUTS_FUSED
drafter-inputs launch writes that block into a fresh tensor that only the drafter forward reads, so on a decode-only
step the worker names it to the drafter (``DFlashForCausalLM._dflash_owned_noise``, an in-tree hook that is off by
default) and the residual stream starts in it: layer 0 reads the block only through its input norm, before the
residual's first write, so every value read is the copy's. The name is exact (the very tensor DRAFTER_INPUTS_FUSED
returned this step) and lasts one forward.

The TP4 target names the block on its forked steps only (its TAIL_FORK's batch-1 bound); here every decode-only step
whose DRAFTER_INPUTS_FUSED launch wrote the block takes it, with or without TAIL_FORK (the fold does not touch the
fork's work).

The TP4 lane's two other items are not ported: TAILGLUE_A needs the TP4 target's ADV_PAR Triton advance (this target
runs the C++ advance), and MEMCPY4B measured +1.9 us per step slower in-model at TP4 (G6-B2: its event edge into
the sampling chain's PDL KV rewind costs more than the 4-byte copy it moves); this target's sampling chain has the
same copy -> PDL rewind edge.

The engagement line is logged once per rank, the first time the block is taken while a CUDA graph is captured (the
replayed path), with two more once-per-rank lines for the steps it does not cover: the first decode-only step run
eagerly after that captured take, and the first decode-only step captured with the copy.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from .m31_switches import m31_disabled

# Read once at import (the same on every rank and every step). Default ON; kill switch TRTLLM_M31_DISABLE_MEMCPY86K=1.
MEMCPY86K = not m31_disabled("MEMCPY86K")


def capturing() -> bool:
    """Whether the current stream is capturing a CUDA graph (the engagement lines report the captured path)."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class StepInputs:
    """One DRAFTER_INPUTS_FUSED drafter-inputs call of this step: its outputs (bonus, query positions, ctx positions,
    noise block)
    and the slots and context lengths it read."""

    __slots__ = ("outs", "slots", "ctx_len")

    def __init__(self, outs: tuple, slots, ctx_len) -> None:
        self.outs = outs
        self.slots = slots
        self.ctx_len = ctx_len


def recording_inputs(worker, inputs_fn):
    """The DRAFTER_INPUTS_FUSED hook ``inputs_fn`` with this step's call kept on ``worker`` (``_tfold_inputs``: a
    ``StepInputs``, or
    None when it declined); ``__wrapped__`` is ``inputs_fn`` (the work-counted hook it wraps)."""

    @functools.wraps(inputs_fn)
    def hook(*args, **kwargs):
        out = inputs_fn(*args, **kwargs)
        rec = None
        if out is not None:
            # The hook's contract: (embed_tokens, accepted, num_accepted, slots, ctx_len, max_ctx, ...).
            slots = args[3] if len(args) > 3 else kwargs.get("slots")
            ctx_len = args[4] if len(args) > 4 else kwargs.get("ctx_len")
            rec = StepInputs(tuple(out), slots, ctx_len)
        worker._tfold_inputs = rec
        return out

    hook.inner = inputs_fn
    return hook


def drafter_owns_noise_hook(draft_model) -> bool:
    """Whether the drafter's forward takes a worker-owned noise block (the modeling_dflash hook is present)."""
    return draft_model is not None and hasattr(draft_model, "_dflash_owned_noise")


def own_noise(draft_model, inputs, recorded: Optional[StepInputs]) -> bool:
    """Name the step's noise block to the drafter for its next forward when it is exactly the tensor this step's
    DRAFTER_INPUTS_FUSED launch wrote (fresh, read by nothing but that forward). Whether it was named."""
    if not isinstance(inputs, dict) or recorded is None or not drafter_owns_noise_hook(draft_model):
        return False
    noise = inputs.get("noise_embedding")
    if noise is None or len(recorded.outs) < 4 or noise is not recorded.outs[3]:
        return False
    draft_model._dflash_owned_noise = noise
    draft_model._dflash_owned_noise_taken = False
    return True


def memcpy86k_setup_line(rank, owned: bool, why: str = "") -> str:
    """The set-up line, in the worker's '<item> on|off on rank N' form."""
    return (
        f"M3.1 DSpark MEMCPY86K: drafter residual stream starts in the DRAFTER_INPUTS_FUSED noise block "
        f"{'on' if owned else 'off'} on rank {rank}{f' ({why})' if why and not owned else ''}"
    )


__all__ = [
    "MEMCPY86K",
    "StepInputs",
    "capturing",
    "drafter_owns_noise_hook",
    "memcpy86k_setup_line",
    "own_noise",
    "recording_inputs",
]
