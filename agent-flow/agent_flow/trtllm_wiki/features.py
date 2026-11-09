# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine features an agent looks up, and what in the checkout each one is about.

This is the one hand-written table of the wiki: for each feature, the LLM API
fields that turn it on or shape it, the words that find its tests and code,
and the environment-variable words that belong to it. Everything else a
feature page shows is read from the checkout when it is asked for: each
field's type, default and status from the serve schema; the tests and
ModelingV2 files that use the words; the test lists that run those tests.
``tests/test_trtllm_wiki.py`` holds every field here to the checkout's schema,
so a renamed field fails there instead of going quiet.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    title: str
    fields: tuple[str, ...]
    words: tuple[str, ...]
    env_words: tuple[str, ...] = ()
    aliases: tuple[str, ...] = field(default=())


_SPEC = ("speculative_config.decoding_type", "speculative_config.max_draft_len")

FEATURES: dict[str, Feature] = {
    "reuse": Feature(
        "KV-cache block (prefix) reuse",
        (
            "kv_cache_config.enable_block_reuse",
            "kv_cache_config.enable_partial_reuse",
            "kv_cache_config.copy_on_partial_reuse",
            "kv_cache_config.block_reuse_config.policy",
            "kv_cache_config.mamba_state_config.periodic_snapshot_interval",
        ),
        ("enable_block_reuse", "mamba_state_config"),
        ("REUSE", "SNAPSHOT"),
        ("block_reuse", "prefix_reuse", "kv_cache_reuse"),
    ),
    "chunked": Feature(
        "Chunked prefill",
        ("enable_chunked_prefill", "max_num_tokens"),
        ("enable_chunked_prefill",),
        ("CHUNK",),
        ("chunked_prefill",),
    ),
    "kv_dtype": Feature(
        "KV-cache dtype (fp8, nvfp4) and its scales",
        ("kv_cache_config.dtype",),
        ("kv_cache_quant_algo", "kv_cache_dtype"),
        ("KV_CACHE_DTYPE", "FP8_KV", "NVFP4_KV"),
        ("kv_cache_dtype", "kv_quant"),
    ),
    "ssm_dtype": Feature(
        "Mamba/SSM state-cache dtype",
        ("kv_cache_config.mamba_ssm_cache_dtype", "kv_cache_config.mamba_ssm_stochastic_rounding"),
        ("mamba_ssm_cache_dtype",),
        ("MAMBA", "SSM"),
    ),
    "cuda_graph": Feature(
        "CUDA graphs",
        (
            "cuda_graph_config.batch_sizes",
            "cuda_graph_config.max_batch_size",
            "cuda_graph_config.enable_padding",
        ),
        ("cuda_graph_config", "CudaGraphConfig"),
        ("CUDA_GRAPH",),
        ("cuda_graphs", "graphs"),
    ),
    "overlap": Feature(
        "Overlap scheduler",
        ("disable_overlap_scheduler",),
        ("disable_overlap_scheduler",),
        ("OVERLAP",),
        ("overlap_scheduler",),
    ),
    "disagg": Feature(
        "Disaggregated serving (context/generation split, KV transfer)",
        ("cache_transceiver_config.backend", "cache_transceiver_config.max_tokens_in_buffer"),
        ("cache_transceiver_config", "launch_disaggregated_llm"),
        ("DISAGG", "TRANSCEIVER", "KV_TRANSFER", "NIXL", "UCX"),
        ("disaggregated",),
    ),
    "attention_dp": Feature(
        "Attention data parallelism (ADP)",
        ("enable_attention_dp", "attention_dp_config.enable_balance"),
        ("enable_attention_dp",),
        ("ADP", "ATTENTION_DP"),
        ("adp", "dep"),
    ),
    "moe": Feature(
        "MoE backends and expert parallelism",
        ("moe_config.backend", "moe_expert_parallel_size"),
        ("moe_config", "moe_expert_parallel_size"),
        ("MOE",),
        ("tep", "ep"),
    ),
    "mtp": Feature(
        "MTP speculative decoding",
        (
            *_SPEC,
            "speculative_config.num_nextn_predict_layers",
            "speculative_config.use_rejection_sampling",
        ),
        ("MTPDecodingConfig",),
        ("MTP", "SPEC_DECODE"),
    ),
    "eagle3": Feature(
        "EAGLE-3 speculative decoding",
        (*_SPEC, "speculative_config.speculative_model"),
        ("Eagle3DecodingConfig",),
        ("EAGLE", "SPEC_DECODE"),
    ),
    "dspark": Feature(
        "DSpark speculative decoding",
        (*_SPEC, "speculative_config.block_size"),
        ("DSparkDecodingConfig",),
        ("DSPARK", "SPEC_DECODE"),
    ),
    "dflash": Feature(
        "DFlash speculative decoding",
        (*_SPEC, "speculative_config.speculative_model"),
        ("DFlashDecodingConfig",),
        ("DFLASH", "SPEC_DECODE"),
    ),
    "host_offload": Feature(
        "KV-cache host (secondary) offload",
        ("kv_cache_config.host_cache_size",),
        ("host_cache_size",),
        ("OFFLOAD", "HOST_CACHE"),
    ),
    "kv_cache_v2": Feature(
        "KV-cache manager v2",
        ("kv_cache_config.use_kv_cache_manager_v2",),
        ("use_kv_cache_manager_v2", "KVCacheManagerV2"),
        ("KV_CACHE_MANAGER_V2", "KVCACHE_V2"),
    ),
    "multimodal": Feature(
        "Multimodal inputs",
        (),
        ("default_multimodal_input_loader", "MultimodalParams"),
        ("MULTIMODAL", "MM_"),
        ("mm", "vision"),
    ),
}


def resolve(name: str) -> str | None:
    """The feature key for ``name`` or one of its aliases."""
    key = name.strip().lower().replace("-", "_")
    if key in FEATURES:
        return key
    return next((k for k, f in FEATURES.items() if key in f.aliases), None)
