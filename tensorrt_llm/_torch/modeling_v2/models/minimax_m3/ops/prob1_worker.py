# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared PROB1 worker protocol; targets retain only their S4 wiring.

The established TP8 adapter supplies topology-specific max_rows from the shared
DraftProb1 object (world2/4:8, world8:4). Collective readiness and verifier proposal
semantics are identical across targets. Native per-TP quality gates remain required."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN
from tensorrt_llm.logger import logger

from ..kernels.m3ll.prob1 import draft_prob1 as prob1


def prebuild(
    self,
    mapping,
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    offset: int,
    vocab: int,
    block: int,
    off: Optional[str] = None,
) -> None:
    """DRAFT_PROB1 (TP8): under rejection-sampling acceptance the S4-logits step
    runs the reference dk_mk_prob1_w8 draft + the draft_probs adapter instead of the greedy chain + biased logits +
    the stock gather / filter / draw (the FOLD_F1 renorm row store and TAIL16_G3 store are not reached: the
    adapter writes the slots). Default ON; kill switch TRTLLM_M31_DISABLE_DRAFT_PROB1=1 (read at import).
    Collective: the ranks agree first (shapes / environment only), as the chain's workspace. ``off`` is a
    target-level decline (its own kill switches); it still takes part in the agreement."""
    from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm
    from tensorrt_llm._torch.speculative import dspark

    self._p1 = None
    why = off or (None if self._s4_logits_on else "S4-logits off")
    if why is None:
        why = prob1.supported(mapping.tp_size, block, vocab, w1, w2_shard)
    setter = getattr(dspark, "set_ancestral_block_override", None)
    owner = getattr(dspark, "_ANCESTRAL_BLOCK_OVERRIDE", None)
    if (
        why is None
        and setter is not None
        and owner is not None
        and owner is not draw_ancestral_block
    ):
        why = "another ancestral block override is installed"
    if not self._s4_ranks_agree(why is None):
        logger.info(
            f"{prob1.TAG} not engaged on rank {mapping.tp_rank}: {why or 'another rank declined'}"
        )
        return
    self._p1 = prob1.DraftProb1.create(
        mapping, _get_mnnvl_workspace_comm(mapping), w1, w2_shard, offset, vocab, block
    )
    if setter is not None:
        setter(draw_ancestral_block)
    logger.info(self._p1.describe())


def engages(self, spec_metadata, num_gens: int, state) -> bool:
    return self._p1 is not None and prob1.engages(
        spec_metadata, num_gens, self._p1.k, state.block, self._p1.max_rows
    )


def step(self, state, base: torch.Tensor, inputs, spec_metadata) -> torch.Tensor:
    """One DRAFT_PROB1 draft: the tokens [G, K] int32 (sample_draft_tokens returns them) and draft_probs[slot]
    for the G gen requests; publishes draft_probs_last_dim (the context one-hot width and acceptance read it)."""
    nc, bs, _ = self._s4_step_meta
    g = bs - nc
    assert getattr(spec_metadata, "use_rejection_sampling", False), (
        "DRAFT_PROB1 engaged without rejection-sampling acceptance"
    )
    slots = spec_metadata.batch_slot_ids[nc:bs]
    tokens = self._p1.run(
        base.view(g * state.block, base.shape[-1]),
        inputs["first_prev_tokens"].view(-1)[:g].long(),
        slots if slots.dtype == torch.int64 else slots.long(),
        int(getattr(spec_metadata, "dummy_slot_row", 0) or 0),
        spec_metadata.request_offsets[nc:bs],
        int(spec_metadata.rng_draft_slot(0)) * _RNG_SLOT_SPAN,
        _RNG_SLOT_SPAN,
        spec_metadata.request_temperatures[nc:bs],
        spec_metadata.request_seeds[nc:bs],
        spec_metadata.draft_probs,
    )
    spec_metadata.draft_probs_last_dim = self._p1.vocab
    line = self._p1.taken_line(g)
    if line is not None:
        logger.info(line)
    return tokens


def draw_ancestral_block(
    self,
    draft_model,
    gen_logits: torch.Tensor,
    inputs,
    spec_metadata,
    num_contexts: int,
    batch_size: int,
):
    """Route BF16 sharded [G, K, V/TP] logits through the prepared ancestral proposal.

    Return None outside the active S4 step so the framework retains its stock
    ancestral sampler. Raw FP32 logits must keep their FP32 addition semantics.
    """
    state = getattr(self, "_s4_step", None)
    meta = getattr(self, "_s4_step_meta", None)
    if (
        state is None
        or getattr(self, "_s4_step_mode", None) != "logits"
        or getattr(self, "_s4_draft", None) is not draft_model
        or meta is None
        or meta[0] != num_contexts
        or meta[1] != batch_size
        or meta[2] is not spec_metadata
        or num_contexts < 0
        or batch_size <= num_contexts
    ):
        return _ancestral_decline(self, "outside the S4-logits step")
    num_gens = batch_size - num_contexts
    if not self._s4_common(num_gens, draft_model) or not engages(
        self, spec_metadata, num_gens, state
    ):
        return _ancestral_decline(self, "S4 / PROB1 eligibility")
    p1 = self._p1
    if (
        gen_logits.dtype != torch.bfloat16
        or not gen_logits.is_cuda
        or not gen_logits.is_contiguous()
        or gen_logits.device != p1.w1.device
        or tuple(gen_logits.shape) != (num_gens, p1.k, state.shard)
        or state.shard * p1.world != p1.vocab
    ):
        return _ancestral_decline(self, "raw logits not contiguous BF16 [G, K, V/TP]")
    probs = spec_metadata.draft_probs
    prev = inputs.get("first_prev_tokens")
    if (
        probs.dtype != torch.float32
        or probs.device != gen_logits.device
        or not probs.is_contiguous()
        or probs.ndim != 3
        or probs.shape[1] < p1.k
        or probs.shape[2] < p1.vocab
        or prev is None
        or prev.dtype not in (torch.int32, torch.int64)
        or prev.device != gen_logits.device
        or tuple(prev.shape) != (num_gens,)
    ):
        return _ancestral_decline(self, "proposal store or anchor layout")
    base = self._zero_padding_rows(
        gen_logits.view(num_gens * state.block, state.shard),
        spec_metadata,
        num_contexts,
        batch_size,
        state.block,
    ).view(num_gens, state.block, state.shard)
    _log_once(
        self,
        "taken",
        f"{prob1.TAG} ancestral block override taken on rank {getattr(p1, 'rank', None)} "
        f"(world {p1.world}, K {p1.k}): the native proposal replaces the per-position stock loop",
    )
    return step(self, state, base, inputs, spec_metadata), p1.vocab


def _log_once(self, key: str, line: str) -> None:
    """Host-only engagement reporting: one line per key and worker (Python runs at capture, not at replay)."""
    seen = getattr(self, "_p1_ancestral_logged", None)
    if seen is None:
        seen = set()
        self._p1_ancestral_logged = seen
    if key not in seen:
        seen.add(key)
        logger.info(line)


def _ancestral_decline(self, reason: str) -> None:
    """Report (once per reason) why a prepared PROB1 left the step to the stock ancestral loop; returns None."""
    if getattr(self, "_p1", None) is not None:
        _log_once(
            self,
            reason,
            f"{prob1.TAG} ancestral block override declined ({reason}); the stock ancestral loop runs",
        )
    return None


def log_fault(self) -> None:
    """Inspect the bounded peer-fault state only from host reporting, never graph capture."""
    if self._p1 is not None:
        try:
            line = self._p1.fault_line()
        except Exception:  # noqa: BLE001 - exit reporting may run after CUDA teardown
            line = None
        if line is not None:
            logger.warning(line)
