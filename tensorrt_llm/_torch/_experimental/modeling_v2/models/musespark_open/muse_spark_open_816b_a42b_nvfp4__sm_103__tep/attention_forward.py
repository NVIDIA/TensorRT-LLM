# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time-scoped stand-ins for the flashinfer calls of the built-in decode FMHA path.

``FlashInferTrtllmGenFmha.run_generation`` reads ``thop`` and ``flashinfer`` as globals
of its own module. For the duration of one attention call, ``eliding`` swaps ``thop``
for a proxy whose generation preprocess stands down for a folded prep, and
``intercepting`` swaps ``flashinfer`` for a proxy whose trtllm-gen decode call launches
the split-KV FMHA. Any other call goes through to the real module.
"""

from __future__ import annotations

import contextlib
import inspect
import re
from typing import TYPE_CHECKING, Iterator, Optional

from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as _fi_gen
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from .kernels.full_attention_prep_append.attention import FoldedPreprocess
    from .kernels.full_attention_split_kv.attention import SplitKvCall

#: ``thop.trtllm_gen_generation_preprocess``'s parameters in order, as
#: ``run_generation`` passes them (positionally, all of them up to ``cross_attention``).
PREPROCESS_ARGS = (
    "qkv_input",
    "workspace",
    "sequence_lengths",
    "spec_decoding_generation_lengths",
    "spec_decoding_position_offsets",
    "kv_cache_block_offsets",
    "host_kv_cache_pool_pointers",
    "host_kv_cache_pool_mapping",
    "kv_scale_orig_quant",
    "kv_scale_quant_orig",
    "attention_output_orig_quant",
    "rotary_inv_freq",
    "rotary_cos_sin",
    "mrope_position_deltas",
    "layer_idx",
    "seq_offset",
    "num_heads",
    "num_kv_heads",
    "head_size",
    "tokens_per_block",
    "kv_cache_quant_mode",
    "max_attention_window_size",
    "cyclic_attention_window_size",
    "num_tokens",
    "batch_beam",
    "input_seq_length",
    "max_past_kv_length",
    "rotary_embedding_dim",
    "rotary_embedding_base",
    "rotary_embedding_scale_type",
    "rotary_embedding_scale",
    "rotary_embedding_max_positions",
    "position_embedding_type",
    "bmm1_scale",
    "bmm2_scale",
    "fp8_context_fmha",
    "predicted_tokens_per_seq",
    "attention_chunk_size",
    "multi_processor_count",
    "total_num_blocks",
    "kv_factor",
    "need_build_kv_cache_metadata",
    "cross_attention",
)


class _ElidingThop:
    """``thop`` whose generation preprocess stands down for one folded prep."""

    def __init__(self, real, folded: FoldedPreprocess) -> None:
        self._real = real
        self._folded = folded

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def trtllm_gen_generation_preprocess(self, *args, **kwargs):
        folded = self._folded
        reason = folded.mismatch(args, kwargs) if not folded.consumed else "a second call"
        if reason is not None:
            if getattr(folded, "fallback", None) is None:
                folded.fallback = reason
            logger.warning_once(
                f"[musespark_open] full_attention_prep_append: the generation preprocess runs after the "
                f"folded prep ({reason})",
                key=f"musespark_open.prep_append.mismatch_{reason}",
            )
            return self._real.trtllm_gen_generation_preprocess(*args, **kwargs)
        folded.consumed = True
        return folded.outputs(args)


@contextlib.contextmanager
def eliding(folded: FoldedPreprocess) -> Iterator[None]:
    """Have ``run_generation``'s preprocess call stand down for ``folded`` for the duration."""
    real = _fi_gen.thop
    _fi_gen.thop = _ElidingThop(real, folded)
    try:
        yield
    finally:
        _fi_gen.thop = real


def preprocess_reachable() -> Optional[str]:
    """Why ``eliding`` would not reach the built-in preprocess call, or ``None``.

    ``FlashInferTrtllmGenFmha.run_generation`` must read ``thop`` as a global of
    its own module (the bindings module the swap replaces for one call), and the
    binding must name its parameters ``PREPROCESS_ARGS`` in this order, which
    ``FoldedPreprocess.mismatch`` reads the positional arguments by.
    """
    if getattr(_fi_gen, "thop", None) is not thop:
        return "flashinfer_trtllm_gen does not hold the thop bindings module as a global"
    run_generation = inspect.unwrap(_fi_gen.FlashInferTrtllmGenFmha.run_generation)
    if getattr(run_generation, "__globals__", None) is not vars(_fi_gen):
        return "run_generation is not a function of flashinfer_trtllm_gen"
    doc = getattr(thop.trtllm_gen_generation_preprocess, "__doc__", "") or ""
    found = [re.search(rf"(?<!\w){name}:", doc) for name in PREPROCESS_ARGS]
    if not all(found) or [m.start() for m in found] != sorted(m.start() for m in found):
        return "the binding's parameters are not PREPROCESS_ARGS in this order"
    return None


class _DecodeStandIn:
    """``flashinfer.decode`` whose trtllm-gen decode call launches ``full_attention_split_kv`` for one call."""

    def __init__(self, real, call: SplitKvCall) -> None:
        self._real = real
        self._call = call

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def trtllm_batch_decode_with_kv_cache(self, *args, **kwargs):
        call = self._call
        reason = "a second call" if call.launched else None
        if reason is None:
            reason = "positional arguments" if args else call.decline(kwargs)
        if reason is not None:
            if call.fallback is None:
                call.fallback = reason
            logger.warning_once(
                f"[musespark_open] full_attention_split_kv: a full-attention decode call runs the "
                f"built-in FMHA ({reason})",
                key=f"musespark_open.split_kv.builtin_{reason}",
            )
            return self._real.trtllm_batch_decode_with_kv_cache(*args, **kwargs)
        call.launch(kwargs)
        return None


class _FlashinferStandIn:
    """``flashinfer`` whose ``decode`` is ``_DecodeStandIn`` for one call."""

    def __init__(self, real, call: SplitKvCall) -> None:
        self._real = real
        self.decode = _DecodeStandIn(real.decode, call)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


@contextlib.contextmanager
def intercepting(call: SplitKvCall) -> Iterator[None]:
    """Have ``run_generation``'s decode FMHA call go through ``call`` for the duration."""
    real = _fi_gen.flashinfer
    _fi_gen.flashinfer = _FlashinferStandIn(real, call)
    try:
        yield
    finally:
        _fi_gen.flashinfer = real


def decode_reachable() -> Optional[str]:
    """Why ``intercepting`` would not reach the built-in decode FMHA call, or ``None``.

    ``FlashInferTrtllmGenFmha.run_generation`` must read ``flashinfer`` as a global
    of its own module, the one the swap replaces for one call. A ``run_generation``
    that calls the decode FMHA some other way runs the built-in FMHA, and
    ``SplitKvCensus`` names the layers left on the built-in path.
    """
    if getattr(_fi_gen, "flashinfer", None) is None:
        return "flashinfer_trtllm_gen holds no flashinfer module"
    run_generation = inspect.unwrap(_fi_gen.FlashInferTrtllmGenFmha.run_generation)
    if getattr(run_generation, "__globals__", None) is not vars(_fi_gen):
        return "run_generation is not a function of flashinfer_trtllm_gen"
    return None
