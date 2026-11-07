# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The steps ``DecoderWarmup`` starts with that need no model forward.

``DecoderWarmup._warmup_scheduled`` starts with ``sampling_module_prewarm``. Three of its
builds need only the configuration: flashinfer's sampling module, its top-k module for beam
search and, for MiniMax-M3 with MSA sparse attention, the indexer's KV-split proxy variants.
The warmup still makes the same calls, and finds the modules built.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from .group import BuildGroup


def _msa_indexer_params(
    model_config: Any, llm_args: Any, mapping: Any
) -> Optional[Tuple[torch.dtype, int, int]]:
    """The arguments ``DecoderWarmup`` passes to ``prewarm_split_proxy_variants``.

    Derived from the configuration the KV cache manager is built from; None when that
    warmup step does not run (no MSA, or no KV cache manager V2).
    """
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    sparse = getattr(llm_args, "sparse_attention_config", None)
    if not isinstance(sparse, MiniMaxM3SparseAttentionConfig) or sparse.implementation != "msa":
        return None
    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    if not getattr(kv_cache_config, "use_kv_cache_manager_v2", False):
        return None
    params = sparse.to_sparse_metadata_params(pretrained_config=model_config.pretrained_config)
    dtype = (
        torch.float8_e4m3fn
        if str(getattr(sparse, "indexer_kv_dtype", "bf16")) == "fp8"
        else torch.bfloat16
    )
    return dtype, params.sharded_index_head_count(mapping), kv_cache_config.tokens_per_block


def forward_free_steps(
    model_config: Any, llm_args: Any, mapping: Any
) -> List[Tuple[str, Callable[[], None]]]:
    """The steps, as ``(name, call)``."""
    from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as sampler_ops

    steps = [("sampling_module", sampler_ops.warmup_sampling_module)]
    if getattr(llm_args, "max_beam_width", 1) > 1:
        steps.append(("topk_module", sampler_ops.warmup_topk_module))
    msa_args = _msa_indexer_params(model_config, llm_args, mapping)
    if msa_args is not None:
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
            prewarm_split_proxy_variants,
        )

        steps.append(("msa_indexer", lambda: prewarm_split_proxy_variants(*msa_args)))
    return steps


def _run_step(step: Tuple[str, Callable[[], None]], stop: threading.Event) -> int:
    name, call = step
    t0 = time.perf_counter()
    call()
    logger.info(f"[warmup] overlapped {name}: {time.perf_counter() - t0:.1f}s")
    return 1


def build_groups(
    model: torch.nn.Module, model_config: Any, llm_args: Any, mapping: Any
) -> List[BuildGroup]:
    """The steps, one thread each: each builds in subprocesses (ninja)."""
    steps = forward_free_steps(model_config, llm_args, mapping)
    return [BuildGroup("warmup_steps", steps, _run_step, len(steps))]
