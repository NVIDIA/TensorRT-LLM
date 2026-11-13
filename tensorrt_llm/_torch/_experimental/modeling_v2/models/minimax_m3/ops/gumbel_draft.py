# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family GUMBEL_DRAFT (moved from the DEP target, which installs it as DEP_GUMBEL_DRAFT; the TP targets as
TP_GUMBEL_DRAFT): the DSpark ancestral draft block (``DSparkWorker._sample_block_ancestrally``) drawn by
``minimax_m3/ops/markov_sampling.sample_markov_block`` (Gumbel-max, one fused draw per position; the CUDA C++ ops
of NATIVE_MARKOV_SAMPLING, bit for bit the Triton sampler they replaced), through the ancestral block override
(``speculative/dspark.set_ancestral_block_override``).

The stock loop runs, at each of the K draft positions, the Markov bias, the biased logits, a softmax, a sampling
from the probabilities and a [rows, vocab] proposal store, with a [rows, vocab] materialization between each. The
block sampler draws position k as the argmax of ``(base_k + bias(d_{k-1})) / T + g``, ``g`` Gumbel noise from a
Philox stream keyed by the request's draft (seed, offset) of position k, and writes every
``q_k = softmax((base_k + bias) / T)`` into the slot-indexed proposal buffer after the block. The draws are temperature
only (the stock loop's contract: ``_samples_block_ancestrally``), so each token follows the stock loop's distribution
exactly. They are not the stock loop's tokens bit for bit (another random stream), and the stored proposals are the
same softmax by another reduction order: natural AL + GSM8K gate the switch, not a bitwise oracle. CUDA-graph padding
rows (the dummy slot) draw from the bias alone, as the stock loop's zeroed rows do.

Contract (``block_outside_contract``): CUDA float / bf16 [rows, K, vocab] logits; a 2-D Markov w1 with >= vocab rows
and a w2 of [vocab, rank]; float32 temperatures and int64 slot ids, one per row; a float32 [slots, >= K, >= vocab]
proposal buffer with a unit last stride, or none. Anything else returns None, logs a "not taken" line once per
reason, and the stock loop runs (SFP_MC_FI / DEP_MARKOV_GLUE stay its fast path); so does a call the sampler's ops
do not take (``markov_sampling.outside_contract``). A target installs it with ``install(tag, switch, enabled)``,
``enabled`` read from its own ``TRTLLM_M31_DISABLE_<switch>`` (this family module reads no target's switches); its
taken / not-taken / engagement lines carry the target's tag and switch name.
``TRTLLM_M31_DISABLE_NATIVE_MARKOV_SAMPLING=1`` (or a build without the ops) leaves no block sampler, so no target
installs it. Under tensor parallelism the drafter's vocab-sharded logits are gathered first
(``maybe_gather_sharded_draft_logits``, as the stock loop does), and every rank draws the same block from the same
Philox stream.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm.logger import logger

from . import markov_sampling

_FLOAT = (torch.float32, torch.bfloat16, torch.float16)
# fn / why: the block sampler or why there is none; tag / switch / enabled: the installing target's (see install).
_STATE: dict = {"tag": "[M3.1]", "switch": "GUMBEL_DRAFT", "enabled": True}
_TAKEN: set = set()
_NOT_TAKEN: set = set()


def block_sampler():
    """``markov_sampling.sample_markov_block`` when its ops are built and on, else None."""
    if "fn" not in _STATE:
        if markov_sampling.ops() is None:
            _STATE["why"] = (
                "markov_sampling unavailable: TRTLLM_M31_DISABLE_NATIVE_MARKOV_SAMPLING=1 or the ops are not built"
            )
            _STATE["fn"] = None
        else:
            _STATE["fn"] = markov_sampling.sample_markov_block
    return _STATE["fn"]


def block_outside_contract(
    gen_logits: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    slots: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
) -> Optional[str]:
    """Why the block sampler does not take these operands, or None."""
    if not all(
        isinstance(t, torch.Tensor)
        for t in (gen_logits, anchors, markov_w1, markov_w2, temperatures, slots)
    ):
        return "an operand is not a tensor"
    if gen_logits.dim() != 3 or gen_logits.dtype not in _FLOAT:
        return f"gen_logits {gen_logits.dtype} {tuple(gen_logits.shape)}"
    rows, num_pos, vocab = gen_logits.shape
    if markov_w1.dim() != 2 or markov_w1.shape[0] < vocab:
        return f"markov_w1 {tuple(markov_w1.shape)} for vocab {vocab}"
    if (
        markov_w2.dim() != 2
        or markov_w2.shape != (vocab, markov_w1.shape[1])
        or markov_w2.dtype != markov_w1.dtype
    ):
        return f"markov_w2 {markov_w2.dtype} {tuple(markov_w2.shape)} for w1 {markov_w1.dtype} {tuple(markov_w1.shape)}"
    if (
        anchors.dim() != 1
        or anchors.numel() != rows
        or anchors.dtype not in (torch.int32, torch.int64)
    ):
        return f"anchors {anchors.dtype} {tuple(anchors.shape)} for {rows} rows"
    if temperatures.dtype != torch.float32 or temperatures.numel() != rows:
        return f"temperatures {temperatures.dtype} {tuple(temperatures.shape)} for {rows} rows"
    if slots.dtype != torch.int64 or slots.numel() != rows:
        return f"slots {slots.dtype} {tuple(slots.shape)} for {rows} rows"
    if draft_probs is not None and (
        not isinstance(draft_probs, torch.Tensor)
        or draft_probs.dtype != torch.float32
        or draft_probs.dim() != 3
        or draft_probs.stride(2) != 1
        or draft_probs.shape[1] < num_pos
        or draft_probs.shape[2] < vocab
    ):
        return f"draft_probs {getattr(draft_probs, 'dtype', None)} {tuple(getattr(draft_probs, 'shape', ()))}"
    devices = {t.device for t in (gen_logits, anchors, markov_w1, markov_w2, temperatures, slots)}
    if draft_probs is not None:
        devices.add(draft_probs.device)
    if len(devices) != 1 or not gen_logits.is_cuda:
        return f"operands on {sorted(str(d) for d in devices)}, not one CUDA device"
    return None


def _not_taken(reason: str) -> None:
    """The not-taken line: once per reason."""
    if reason in _NOT_TAKEN:
        return
    _NOT_TAKEN.add(reason)
    logger.info(
        f"{_STATE['tag']} {_STATE['switch']} not taken ({reason}); the stock ancestral loop runs"
    )


def _taken(rows: int, num_pos: int) -> None:
    """The taken line: once per (rows, positions)."""
    if (rows, num_pos) in _TAKEN:
        return
    _TAKEN.add((rows, num_pos))
    logger.info(f"{_STATE['tag']} {_STATE['switch']} taken (at {rows} rows x {num_pos} positions)")


def draw_block(worker, draft_model, gen_logits, inputs, spec_metadata, num_contexts, batch_size):
    """The ancestral block override (see the module docstring): (int32 tokens [rows, K], vocab) or None."""
    from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN

    sample = block_sampler()
    if sample is None:
        _not_taken(_STATE.get("why", "markov_sampling unavailable"))
        return None
    # The chain indexes the full Markov table; verification reads full-vocab q.
    gen_logits = worker.maybe_gather_sharded_draft_logits(gen_logits, spec_metadata)
    temperatures = spec_metadata.request_temperatures[num_contexts:batch_size]
    slots = spec_metadata.batch_slot_ids[num_contexts:batch_size]
    anchors = inputs["first_prev_tokens"]
    w1, w2 = draft_model.markov_w1, draft_model.markov_w2
    draft_probs = spec_metadata.draft_probs
    reason = block_outside_contract(gen_logits, anchors, w1, w2, temperatures, slots, draft_probs)
    if reason is not None:
        _not_taken(reason)
        return None
    seed, offset = worker._rng_state_per_request(
        spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_draft_slot(0)
    )
    # A dummy_slot_row of 0 is not published yet: row 0 is then a live request's, not padding.
    padding_slot = getattr(spec_metadata, "dummy_slot_row", 0)
    rows, num_pos, vocab = gen_logits.shape
    tokens = sample(
        gen_logits.contiguous(),
        anchors,
        w1,
        w2,
        temperatures,
        seed,
        offset,
        _RNG_SLOT_SPAN,
        draft_probs,
        slots,
        padding_slot if padding_slot > 0 else -1,
    )
    if tokens is None:
        _not_taken(
            markov_sampling.outside_contract(
                gen_logits, anchors, w1, w2, temperatures, seed, offset, draft_probs, slots
            )
            or "declined"
        )
        return None
    if draft_probs is not None:
        spec_metadata.draft_probs_last_dim = vocab
    _taken(rows, num_pos)
    return tokens, vocab


def engagement(enabled: Optional[bool] = None) -> str:
    """The log line's state: what the override does in this process (``enabled``: the target's switch, default the
    installed one)."""
    if not (_STATE["enabled"] if enabled is None else enabled):
        return f"off (TRTLLM_M31_DISABLE_{_STATE['switch']}=1); the stock ancestral loop"
    if block_sampler() is None:
        return f"not installed ({_STATE.get('why')}); the stock ancestral loop"
    return (
        "engaged (CUDA C++ Gumbel-max block sampler, one fused draw per position; distribution-exact, not bitwise vs "
        "the stock loop)"
    )


def log_engagement() -> None:
    logger.info(f"{_STATE['tag']} {_STATE['switch']}: {engagement()}")


def install(tag: str, switch: str, enabled: bool) -> bool:
    """Install ``draw_block`` as the DSpark ancestral block override when ``enabled`` (the target's
    ``TRTLLM_M31_DISABLE_<switch>`` unset) and the block sampler's ops are built; logs the engagement line with the
    target's ``tag``. Returns whether it is installed."""
    from tensorrt_llm._torch.speculative import dspark

    _STATE.update(tag=tag, switch=switch, enabled=bool(enabled))
    # The DSpark worker's ancestral block registry; a runtime without it keeps the stock loop.
    setter = getattr(dspark, "set_ancestral_block_override", None)
    if setter is None and enabled:
        _STATE["why"] = "the DSpark ancestral block override is not available on this runtime"
        logger.info(f"{tag} {switch}: not installed ({_STATE['why']}); the stock ancestral loop")
        return False
    installed = bool(enabled) and block_sampler() is not None
    if installed:
        setter(draw_block)
    elif setter is not None and getattr(dspark, "_ANCESTRAL_BLOCK_OVERRIDE", None) is draw_block:
        setter(
            None
        )  # an earlier install in this process stops serving; another owner's override stays
    log_engagement()
    return installed
