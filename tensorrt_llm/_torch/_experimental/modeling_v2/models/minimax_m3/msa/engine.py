# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the M3.1 targets put the family's MSA attention in front of the engine.

The family's backend (:class:`~.msa_backend.MiniMaxM31MsaSparseAttention`), its step metadata and its KV cache
manager (:class:`~.cache_manager.MiniMaxM31KVCacheManagerV2`) derive from main's MiniMax-M3 MSA classes. A target
builds its attention layers from the family backend only when the engine can be told to build the matching step
metadata and KV cache manager (``RuntimeHooks.sparse_attention``); on an engine without that hook the targets keep
main's classes, which run the same tokens without the family's MSA paths.

Either way the targets run the family's vendored fmha_sm100 (``kernels/msa_kernels/fmha_sm100``), which carries two
correctness fixes the packaged module lacks: the forward kernel waits on its grid dependency before reading its inputs
(it is PDL-launched), and the NVFP4-KV forward zeroes the V rows at or past the valid columns. With main's classes,
:func:`route_stock_msa_to_vendored` points their fmha_sm100 lookups at the vendored copy.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Any, Optional

import torch

from tensorrt_llm.logger import logger

from .. import runtime_hooks_compat

__all__ = [
    "arm_target_paths",
    "create_msa_attention",
    "family_msa_enabled",
    "indexer_prewarm",
    "require_msa_module",
    "route_stock_msa_to_vendored",
    "sparse_attention_classes",
    "use_family_backend",
    "warmup_scope",
]

# Kill switch: TRTLLM_M31_DISABLE_FAMILY_MSA=1 keeps main's MSA classes on every engine.
_DISABLE_ENV = "TRTLLM_M31_DISABLE_FAMILY_MSA"


# main's modules that resolve fmha_sm100 for its MiniMax-M3 MSA classes, and the attribute each resolves it through.
_STOCK_MSA_LOOKUPS = (
    (
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils",
        "require_msa_module",
    ),
    ("tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend", "require_msa_module"),
    ("tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer", "require_msa_module"),
    ("tensorrt_llm._torch.attention.backends.fmha.msa_prefill", "_msa_sparse_api"),
)
_stock_routed = False


def _switch_off() -> bool:
    return os.environ.get(_DISABLE_ENV, "0") == "1"


def family_msa_enabled() -> bool:
    """Whether the targets build the family's MSA classes: the engine takes them and the switch is not set."""
    if _switch_off():
        return False
    return runtime_hooks_compat.has_hook("sparse_attention")


def route_stock_msa_to_vendored() -> bool:
    """Point main's MiniMax-M3 MSA classes' fmha_sm100 lookups (the module their prefill, indexer proxy pass and
    plans run) at the family's vendored copy, so the layers the targets build from main's classes run the fixed
    kernels too. Process-wide and idempotent; call it before the first forward. Only the module lookup changes:
    main's backend selection and code paths stay as they are. TRTLLM_M31_DISABLE_FAMILY_MSA=1 keeps the packaged
    module. Whether the lookups resolve the vendored copy."""
    global _stock_routed
    if _switch_off():
        return False
    if _stock_routed:
        return True
    from . import msa_prefill as family_prefill
    from .kernels import msa_utils as family_utils

    vendored = {
        "require_msa_module": family_utils.require_msa_module,
        "_msa_sparse_api": family_prefill._msa_sparse_api,
    }
    for module_name, attr in _STOCK_MSA_LOOKUPS:
        module = importlib.import_module(module_name)
        if hasattr(module, attr):
            setattr(module, attr, vendored[attr])
    _stock_routed = True
    logger.info("MiniMax-M3.1: main's MSA classes resolve the family's vendored fmha_sm100")
    return True


def sparse_attention_classes() -> Optional[Any]:
    """The ``RuntimeHooks.sparse_attention`` value naming the family's classes, or None on an engine without it."""
    if not family_msa_enabled():
        return None
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import SparseAttentionClasses

    from .cache_manager import MiniMaxM31KVCacheManagerV2
    from .msa_backend import MiniMaxM31MsaSparseAttention

    return SparseAttentionClasses(
        backend=MiniMaxM31MsaSparseAttention,
        kv_cache_manager=MiniMaxM31KVCacheManagerV2,
        warmup_scope=warmup_scope,
        indexer_prewarm=indexer_prewarm,
    )


def create_msa_attention(
    backend_name: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    *,
    sparse_params,
    quant_config=None,
    pos_embd_params=None,
    q_scaling: Optional[float] = None,
    skip_create_weights_in_init: bool = False,
):
    """``create_attention`` for a MiniMax-M3 MSA layer: the family backend when :func:`family_msa_enabled`, else
    whatever main's registry resolves."""
    from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend

    kwargs = dict(
        quant_config=quant_config,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        skip_create_weights_in_init=skip_create_weights_in_init,
        sparse_params=sparse_params,
    )
    if family_msa_enabled():
        from .msa_backend import MiniMaxM31MsaSparseAttention

        stock = get_attention_backend(backend_name, sparse_params=sparse_params)
        if issubclass(MiniMaxM31MsaSparseAttention, stock):
            arm_target_paths()
            return MiniMaxM31MsaSparseAttention(
                layer_idx,
                num_heads,
                head_dim,
                num_kv_heads,
                mla_params=None,
                attention_chunk_size=None,
                dtype=None,
                aux_stream=None,
                kv_cache_dtype="auto",
                skip_correction_threshold=0.0,
                gqa_decode=None,
                cache_context_plan=False,
                **kwargs,
            )
    route_stock_msa_to_vendored()
    return create_attention(backend_name, layer_idx, num_heads, head_dim, num_kv_heads, **kwargs)


def use_family_backend(module) -> bool:
    """Rebuild ``module.attn`` (an ``Attention`` whose backend main's registry resolved to the MiniMax-M3 MSA
    backend) as the family backend with the same arguments, when :func:`family_msa_enabled`; whether it did. When it
    does not, main's backend keeps the layer and :func:`route_stock_msa_to_vendored` points it at the vendored
    fmha_sm100."""
    if not family_msa_enabled():
        route_stock_msa_to_vendored()
        return False
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention

    from .msa_backend import MiniMaxM31MsaSparseAttention

    stock = getattr(module, "attn", None)
    if not isinstance(stock, MiniMaxM3MsaSparseAttention) or isinstance(
        stock, MiniMaxM31MsaSparseAttention
    ):
        return False
    backend = MiniMaxM31MsaSparseAttention(
        stock.layer_idx,
        stock.num_heads,
        stock.head_dim,
        stock.num_kv_heads,
        quant_config=stock.quant_config,
        q_scaling=stock.q_scaling,
        pos_embd_params=None,
        skip_create_weights_in_init=True,
        sparse_params=stock.sparse_params,
        mla_params=None,
        attention_chunk_size=stock.attention_chunk_size,
        dtype=None,
        aux_stream=None,
        kv_cache_dtype=stock.kv_cache_dtype,
        skip_correction_threshold=0.0,
        gqa_decode=None,
        cache_context_plan=bool(getattr(stock, "cache_context_plan", False)),
    )
    backend.update_quant_config(stock.quant_config)
    arm_target_paths()
    module.attn = backend
    return True


def require_msa_module():
    """The fmha_sm100 module the targets' MSA attention runs: the family's vendored copy (the family backend's, and
    main's backend's once :func:`route_stock_msa_to_vendored` ran), or with TRTLLM_M31_DISABLE_FAMILY_MSA=1 the
    package main's backend imports."""
    if not _switch_off():
        from .kernels.msa_utils import require_msa_module as family_require
    else:
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
            require_msa_module as family_require,
        )
    return family_require()


def arm_target_paths() -> None:
    """Arm, process-wide, the family MSA paths the M3.1 sm_107 targets run (each keeps its kill switch): the
    BF16X / E4M3X sparse-prefill fork and the indexer proxy max-score buffer reuse; and keep the context proxy plans
    unsplit, the stock context select and no MSA prewarm (msa_indexer.arm_sm107_msa_paths)."""
    from . import msa_indexer, msa_prefill

    msa_prefill.arm_prefill_fork()
    msa_indexer.arm_proxy_ms_reuse()
    msa_indexer.arm_sm107_msa_paths()


def warmup_scope(runner) -> contextlib.AbstractContextManager:
    """The MSA prewarm phase scope around the decoder warmup (``RuntimeHooks.sparse_attention.warmup_scope``)."""
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    sparse_config = getattr(runner, "sparse_attention_config", None)
    if (
        not isinstance(sparse_config, MiniMaxM3SparseAttentionConfig)
        or sparse_config.implementation != "msa"
    ):
        return contextlib.nullcontext()
    from .prewarm import hook as msa_prewarm_hook

    return msa_prewarm_hook.warmup_scope(runner)


def indexer_prewarm(runner, kv_cache_manager) -> None:
    """The family MSA indexer prewarm (``RuntimeHooks.sparse_attention.indexer_prewarm``): compile every fmha_sm100
    variant this rank's MSA plans can select, the KV-split proxy variants no warmup forward plans, and the armed
    prefill context scorer's kernels."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
        MiniMaxM3KVCacheManagerV2,
    )
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    if not isinstance(kv_cache_manager, MiniMaxM3KVCacheManagerV2):
        return
    sparse_config = runner.sparse_attention_config
    if (
        not isinstance(sparse_config, MiniMaxM3SparseAttentionConfig)
        or sparse_config.implementation != "msa"
    ):
        return

    from .msa_indexer import prewarm_prefill_context_scorer, prewarm_split_proxy_variants
    from .prewarm import hook as msa_prewarm_hook

    # Compile every fmha_sm100 variant this rank's MSA plans can select and start recording the sparse-decode call
    # shapes (a no-op unless enabled).
    msa_prewarm_hook.before_msa_indexer_prewarm(runner, kv_cache_manager)
    params = sparse_config.to_sparse_metadata_params(
        pretrained_config=runner.model.model_config.pretrained_config
    )
    num_index_heads = params.sharded_index_head_count(runner.mapping)
    # NVFP4 index-K pages are scored through their gathered E4M3 copy.
    index_dtype = (
        torch.float8_e4m3fn
        if kv_cache_manager.indexer_kv_dtype in ("fp8", "nvfp4")
        else torch.bfloat16
    )
    with torch.inference_mode():
        prewarm_split_proxy_variants(
            index_dtype, num_index_heads, kv_cache_manager.tokens_per_block
        )
        # A model's prefill context scorer runs only on steps longer than the warm-up's short prefills, so its kernels
        # compile or load here, not on the first long request.
        prewarm_prefill_context_scorer(num_index_heads)
    torch.cuda.synchronize()
    if runner.mapping.tp_size > 1 and runner.dist is not None:
        runner.dist.tp_allgather(1)
    logger.info("MiniMax-M3.1 MSA indexer prewarm complete")
