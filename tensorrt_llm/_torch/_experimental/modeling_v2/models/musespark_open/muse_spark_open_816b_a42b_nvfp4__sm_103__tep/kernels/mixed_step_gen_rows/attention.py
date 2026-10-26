# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generation rows of mixed steps on ``multi_request_split_kv`` (sm_107 full-attention layers).

A mixed step (context rows and generation rows in one batch) runs ``PrefillTarget``, the
built-in text-model forward. On sm_107 the public FMHA library list leaves flashinfer's
trtllm-gen FMHA out; ``attention.select_decode_fmha`` puts a ``DecodeTrtllmGenFmha`` in front
of every layer's libraries, which offers the generation phase only. With nothing in front of
the thop fallback offering the context phase, the FMHA manager hands a mixed step's whole
batch to the thop op, so its generation rows run thop's generation FMHA, not the decode-step
paths (``DecodeTrtllmGenFmha`` and, in front of it, ``multi_request_split_kv``).

This path changes that on the full-attention layers in two parts:

* ``ThopContextPhaseFmha``, inserted in each full-attention layer's FMHA manager just before
  the thop fallback, offers the context phase of a mixed step that ``takes_generation``
  accepts (context rows, ``MIN_GENERATIONS``..``MAX_GENERATIONS`` generation sequences, beam
  width 1). The manager then forms ``CombinedFmha(context=ThopContextPhaseFmha,
  generation=DecodeTrtllmGenFmha)``: the context rows run the thop op with
  ``attention_input_type=context_only`` (the same op, the same context launch as before) and the
  generation rows run ``FlashInferTrtllmGenFmha.run_generation`` with ``seq_offset =
  num_contexts``, as on sm_103.
* ``MixedStepGenRowsAttention``, re-classed in front of the full-attention layers' attention,
  runs such a step's attention call with the decode FMHA call intercepted by a
  ``MixedStepCall`` (``attention_forward.intercepting``), so the generation phase launches
  ``multi_request_split_kv`` as a decode step's does. ``MixedStepCall`` is
  ``multi_request_split_kv.MultiRequestCall`` with the sequence lengths re-aligned: the
  generation phase's view of them starts at ``num_contexts``. A call the stand-in declines runs
  flashinfer's trtllm-gen decode FMHA.

The band is read on the host from the step's metadata. Every bound is a point of the FMHA
manager's batch-size cache grid, so a cached selection never crosses it. Sliding-window
layers, pure context steps, decode steps and mixed steps outside the band run what they ran
before. The arming sits at ``_attn_impl``, which the breakable prefill CUDA graph re-runs
eagerly on every replay (``eager_on_graph``), so captured and eager mixed steps both take it.

Numerics: the context rows are those of the thop op. The generation rows are those of
``multi_request_split_kv`` on decode steps (not bit-identical to thop's generation FMHA;
deterministic).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm.logger import logger

from ...attention import DecodeTrtllmGenFmha
from ...attention_forward import intercepting
from ..multi_request_split_kv import attention as _mr_attention

#: Generation sequences of a mixed step this path takes. Both bounds and their predecessors
#: are points of the FMHA cache's batch-size grid (``fmha.manager``).
MIN_GENERATIONS = 9
MAX_GENERATIONS = 64


def takes_generation(attn_metadata) -> bool:
    """Whether a step is a mixed step whose generation rows this path takes.

    Context rows, ``MIN_GENERATIONS``..``MAX_GENERATIONS`` generation sequences, beam width 1.
    Host-side reads only.
    """
    return (
        getattr(attn_metadata, "num_contexts", 0) > 0
        and MIN_GENERATIONS <= getattr(attn_metadata, "num_generations", 0) <= MAX_GENERATIONS
        and getattr(attn_metadata, "beam_width", 1) == 1
    )


class ThopContextPhaseFmha(PhasedFmha):
    """The thop op's context phase, offered on the mixed steps ``takes_generation`` accepts.

    It never takes a whole batch or a generation phase, so the manager pairs it with the
    generation library in front of it. ``run_context`` calls the thop fallback with the
    context rows and ``attention_input_type=context_only``.
    """

    supports_workspace_reclamation = True

    def __init__(self, attn, fallback: FallbackFmha):
        super().__init__(attn)
        self._fallback = fallback
        #: Context phases run, read by the served-path log.
        self.context_calls = 0

    def _is_supported(self, q, k, v, metadata, forward_args, *, phase=None) -> bool:
        if phase != FmhaPhase.CONTEXT or not takes_generation(metadata):
            return False
        if forward_args.attention_input_type != AttentionInputType.mixed:
            return False
        if forward_args.output is None or forward_args.output_sf is not None:
            return False
        return self._fallback.is_supported(q, k, v, metadata, forward_args)

    def run_context(self, params: FmhaParams) -> None:
        rows = params.num_tokens
        attention_input = params.qkv_input if params.qkv_input is not None else params.query_input
        forward_args = dataclasses.replace(
            params.fwd,
            output=params.fwd.output[:rows],
            attention_input_type=AttentionInputType.context_only,
        )
        self._fallback.forward(
            attention_input, params.key_input, params.value_input, params.meta, forward_args
        )
        self.context_calls += 1

    def run_generation(self, params: FmhaParams) -> None:
        raise RuntimeError("ThopContextPhaseFmha serves the context phase only")


def context_library(attn) -> Optional[ThopContextPhaseFmha]:
    """The ``ThopContextPhaseFmha`` in ``attn``'s FMHA manager, or None."""
    for fmha in attn._fmha_manager.fmha_libs:
        if isinstance(fmha, ThopContextPhaseFmha):
            return fmha
    return None


#: Tensors of the decode call ``multi_request_split_kv`` reads at a 16-byte aligned address.
_ALIGNED_KWARGS = ("query", "out", "block_tables")


class MixedStepCall(_mr_attention.MultiRequestCall):
    """A mixed step's generation-phase decode call for ``multi_request_split_kv``.

    The generation phase reads the sequence lengths from ``seq_offset = num_contexts`` on, so
    ``seq_lens`` is a view that starts 4 bytes per context sequence into the step's lengths.
    The kernel reads it at a 16-byte aligned address; ``launch`` hands it an aligned copy of
    those ``B`` lengths when the view is not aligned. The other tensors it reads are declined
    when not aligned.
    """

    def decline(self, kwargs: dict) -> Optional[str]:
        reason = super().decline(kwargs)
        if reason is not None:
            return reason
        for name in _ALIGNED_KWARGS:
            if kwargs[name].data_ptr() % 16:
                return f"a {name} not 16-byte aligned"
        return None

    def launch(self, kwargs: dict) -> None:
        seq_lens = kwargs["seq_lens"]
        if seq_lens.data_ptr() % 16:
            kwargs = dict(kwargs, seq_lens=seq_lens.clone())
        super().launch(kwargs)


class MixedStepGenRowsAttention:
    """Mixin re-classed in front of a full-attention layer's attention class.

    A call on a step ``takes_generation`` accepts runs the class behind it with the decode
    FMHA call intercepted by a ``MixedStepCall`` of the layer's KV heads; any other call is
    exactly the class behind it. Adds plain attributes only.
    """

    _mixed_kv_heads: int = 4

    def _attn_impl(self, q, k, v, attn_metadata, *args, **kwargs):
        if not takes_generation(attn_metadata):
            return super()._attn_impl(q, k, v, attn_metadata, *args, **kwargs)
        call = MixedStepCall(kv_heads=self._mixed_kv_heads, window=0)
        with intercepting(call):
            result = super()._attn_impl(q, k, v, attn_metadata, *args, **kwargs)
        if call.launched:
            outcome = "multi_request_split_kv"
        elif call.fallback is not None:
            outcome = f"flashinfer's trtllm-gen decode FMHA ({call.fallback})"
        else:
            outcome = "no trtllm-gen decode call reached (the thop op ran the whole batch)"
        logger.info_once(
            f"[musespark_open] mixed_step_gen_rows: a mixed step's generation rows "
            f"({attn_metadata.num_generations} requests, layer {self.layer_idx}): {outcome}",
            key=f"musespark_open.mixed_step_gen_rows.{outcome}",
        )
        return result


@functools.lru_cache(maxsize=None)
def _mixed_class(base: type) -> type:
    """``base`` with ``MixedStepGenRowsAttention`` in front, one class per base."""
    return type(f"{base.__name__}MixedGen", (MixedStepGenRowsAttention, base), {})


def _refusal(attention) -> Optional[str]:
    libs = attention.attn._fmha_manager.fmha_libs
    if not libs or not isinstance(libs[0], DecodeTrtllmGenFmha):
        return "no DecodeTrtllmGenFmha in front of its FMHA libraries"
    if not any(isinstance(fmha, FallbackFmha) for fmha in libs):
        return "no thop fallback among its FMHA libraries"
    if not isinstance(attention, _mr_attention.MultiRequestSplitKvAttention):
        return "its attention is not on multi_request_split_kv"
    return None


def install(model: torch.nn.Module) -> int:
    """Put a ``ThopContextPhaseFmha`` before the thop fallback of every full-attention layer's
    FMHA manager and re-class those layers' attention with ``MixedStepGenRowsAttention`` in
    front of whatever class they carry. Returns the number of layers changed.

    Run after ``attention.select_decode_fmha`` and the ``multi_request_split_kv`` install, and
    after every other attention re-class: the mixin acts only on mixed steps, which none of the
    others intercept. Every layer is checked before any changes; idempotent.

    Raises:
        ValueError: if a full-attention layer's FMHA manager does not start with
            ``DecodeTrtllmGenFmha``, holds no thop fallback, or its attention is not on
            ``multi_request_split_kv``.
    """
    chosen = [layer.self_attn for layer in model.layers if not layer.self_attn.is_sliding]
    refused = [
        f"layer {attention.layer_idx}: {reason}"
        for attention in chosen
        if (reason := _refusal(attention)) is not None
    ]
    if refused:
        raise ValueError(
            "[musespark_open] mixed_step_gen_rows cannot route mixed steps' generation rows: "
            + "; ".join(refused)
        )
    for attention in chosen:
        attn = attention.attn
        if context_library(attn) is None:
            libs = attn._fmha_manager.fmha_libs
            index = next(i for i, fmha in enumerate(libs) if isinstance(fmha, FallbackFmha))
            libs.insert(index, ThopContextPhaseFmha(attn, libs[index]))
        if not isinstance(attention, MixedStepGenRowsAttention):
            attention.__class__ = _mixed_class(type(attention))
        attention._mixed_kv_heads = attention._mr_kv_heads
    logger.info(
        f"[musespark_open] mixed_step_gen_rows: the generation rows of mixed steps with "
        f"{MIN_GENERATIONS}..{MAX_GENERATIONS} generation requests run multi_request_split_kv "
        f"on {len(chosen)} full-attention layers (context rows on the thop op)"
    )
    return len(chosen)


__all__ = [
    "MAX_GENERATIONS",
    "MIN_GENERATIONS",
    "MixedStepCall",
    "MixedStepGenRowsAttention",
    "ThopContextPhaseFmha",
    "context_library",
    "install",
    "takes_generation",
]
