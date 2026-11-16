# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""decode_kv_splits on the generation rows of mixed steps.

A mixed step (context rows and generation rows in one batch) runs ``PrefillTarget``,
the built-in text-model forward. Each layer's built-in attention runs the context
phase (flashinfer's trtllm-gen context FMHA) and then the generation phase of the
batch's generation sequences (``run_generation`` with ``seq_offset = num_contexts``),
which launches flashinfer's trtllm-gen decode FMHA over those sequences. flashinfer
picks that call's KV split from the SM count exactly as on a decode step (``splits``'s
module docstring), so a mixed step of 20 or more generation requests on a TEP4 rank
runs the persistent one-CTA-per-request kernel on every full-attention layer, and its
time is set by the longest of those requests.

This mixin, re-classed in front of the full-attention layers' attention, arms the
model's ``DecodeKvSplits`` for the duration of one attention call when the step has
context rows and ``takes_generation`` accepts its generation requests (9 to 64). The
split rule then acts on the generation phase's decode call exactly as on the decode
steps ``DecodeTarget`` arms: same SM-count rule, same own counter buffer, same
workspace check, same calls left alone. The context phase calls a different flashinfer
entry point (``flashinfer.prefill``), which the stand-in passes through unchanged.

Why at the attention call and not around ``PrefillTarget``'s forward: under the
breakable prefill CUDA graph a replay never runs the model's forward; it replays the
captured segments and re-runs the eager attention op between them, which reaches this
module's ``_attn_impl`` with the step's metadata. Arming here applies the rule on every
mixed step, captured or eager. Every read is host-side metadata of the step.

Decode steps (no context rows) are left to ``DecodeTarget``'s own arming; steps of one
to 8 generation requests and pure context steps run what they ran before.

Numerics: those of ``decode_kv_splits`` on decode steps (the same trtllm-gen kernel
family, multi-CTA KV with the in-kernel fp32 merge; only the KV partition of each
request changes).
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from .splits import MAX_ROWS, MIN_ROWS, ROWS_PER_REQUEST, DecodeKvSplits

#: Generation requests of a mixed step the rule is armed for.
MIN_REQUESTS = MIN_ROWS // ROWS_PER_REQUEST
MAX_REQUESTS = MAX_ROWS // ROWS_PER_REQUEST


def takes_generation(attn_metadata) -> bool:
    """Whether a step is a mixed step whose generation rows the rule is armed for.

    Context rows, and ``MIN_REQUESTS``..``MAX_REQUESTS`` generation sequences with beam
    width 1. Host-side reads only.
    """
    num_contexts = getattr(attn_metadata, "num_contexts", 0)
    num_generations = getattr(attn_metadata, "num_generations", 0)
    return (
        num_contexts > 0
        and MIN_REQUESTS <= num_generations <= MAX_REQUESTS
        and getattr(attn_metadata, "beam_width", 1) == 1
    )


class MixedStepKvSplitsAttention:
    """Mixin re-classed in front of a full-attention layer's attention class.

    A call on a step ``takes_generation`` accepts runs the class behind it with the model's
    ``DecodeKvSplits`` armed; any other call is exactly the class behind it. Adds one plain
    attribute (``_mixed_kv_splits``).
    """

    _mixed_kv_splits: Optional[DecodeKvSplits] = None

    def _attn_impl(self, q, k, v, attn_metadata, *args, **kwargs):
        rule = self._mixed_kv_splits
        if rule is None or not takes_generation(attn_metadata):
            return super()._attn_impl(q, k, v, attn_metadata, *args, **kwargs)
        split, kept = rule.split_calls, rule.kept_calls
        with rule.armed():
            result = super()._attn_impl(q, k, v, attn_metadata, *args, **kwargs)
        if rule.split_calls > split:
            outcome = "split"
        elif rule.kept_calls > kept:
            outcome = "kept flashinfer's split"
        else:
            outcome = "no trtllm-gen decode call reached (another FMHA library ran)"
        logger.info_once(
            f"[musespark_open] decode_kv_splits: a mixed step's generation rows "
            f"({attn_metadata.num_generations} requests, layer {self.layer_idx}): {outcome}",
            key=f"musespark_open.decode_kv_splits.mixed.{outcome}",
        )
        return result


@functools.lru_cache(maxsize=None)
def _mixed_class(base: type) -> type:
    """``base`` with ``MixedStepKvSplitsAttention`` in front, one class per base."""
    return type(f"{base.__name__}MixedKv", (MixedStepKvSplitsAttention, base), {})


def install(model: torch.nn.Module, rule: DecodeKvSplits) -> int:
    """Re-class the full-attention layers' attention in front of whatever class they carry,
    so their mixed-step attention calls arm ``rule``. Returns the number of layers re-classed.

    Run after every other attention re-class: the mixin goes in front of all of them and
    acts only on mixed steps, which none of the others intercept.
    """
    chosen = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    for attention in chosen:
        if not isinstance(attention, MixedStepKvSplitsAttention):
            attention.__class__ = _mixed_class(type(attention))
        attention._mixed_kv_splits = rule
    logger.info(
        f"[musespark_open] decode_kv_splits: the generation rows of mixed steps with "
        f"{MIN_REQUESTS}..{MAX_REQUESTS} generation requests run the split rule on "
        f"{len(chosen)} full-attention layers"
    )
    return len(chosen)


__all__ = ["MixedStepKvSplitsAttention", "install", "takes_generation"]
