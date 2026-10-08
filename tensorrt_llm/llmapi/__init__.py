# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn

from .. import _BACKEND

if _BACKEND == "rocm":
    from ..rocm import LLM, CompletionOutput, RequestOutput, SamplingParams

    __all__ = ["LLM", "SamplingParams", "CompletionOutput", "RequestOutput"]
    _NVIDIA_EXPORTS = frozenset([
        'AsyncLLM', 'AttentionDpConfig', 'AutoDecodingConfig', 'BatchingType',
        'BlockReuseConfig', 'CacheTransceiverConfig', 'CalibConfig',
        'CapacitySchedulerPolicy', 'ColdPageQuantizationCompressionConfig',
        'ContextChunkingPolicy', 'ConversationParams', 'CudaGraphConfig',
        'DFlashDecodingConfig', 'DSparkDecodingConfig', 'DecodeCudaGraphConfig',
        'DeepSeekSparseAttentionConfig', 'DeepSeekV4SparseAttentionConfig',
        'DisaggScheduleStyle', 'DisaggregatedParams',
        'DraftTargetDecodingConfig', 'DynamicBatchConfig',
        'Eagle3DecodingConfig', 'EagleDecodingConfig', 'EncodeCudaGraphConfig',
        'EncodeExtraInputSpec', 'ExtendedRuntimePerfKnobConfig',
        'GuidedDecodingParams', 'KVEventsConfig', 'KvCacheConfig',
        'KvCacheRetentionConfig', 'LlmArgs', 'LoRARequest', 'MTPDecodingConfig',
        'MambaStateConfig', 'MiniMaxM3SparseAttentionConfig', 'MoeConfig',
        'MpiCommSession', 'MultimodalConfig', 'MultimodalEncoder',
        'NGramDecodingConfig', 'PARDDecodingConfig', 'PrefillCudaGraphBackend',
        'PrometheusMetricsConfig', 'QSASparseAttentionConfig', 'QuantAlgo',
        'QuantConfig', 'ReorderRequestPolicyConfig', 'RequestError',
        'RocketSparseAttentionConfig', 'SADecodingConfig', 'SAEnhancerConfig',
        'SaveHiddenStatesDecodingConfig', 'SchedulerConfig', 'SchedulingParams',
        'SkipSoftmaxAttentionConfig', 'ThinkingBudgetLogitsProcessor',
        'TorchCompileConfig', 'TorchLlmArgs',
        'TriAttentionKvCacheCompressionConfig', 'UserProvidedDecodingConfig',
        'add_thinking_budget_logits_processor'
    ])

    def __getattr__(name: str) -> NoReturn:
        if name in _NVIDIA_EXPORTS:
            raise NotImplementedError(
                f"{name} is not supported by the ROCm backend")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
else:
    from .._torch.async_llm import AsyncLLM
    from ..conversation_params import ConversationParams
    from ..disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
    from ..executor import CompletionOutput, LoRARequest, RequestError
    from ..sampling_params import GuidedDecodingParams, SamplingParams
    from ..scheduling_params import SchedulingParams
    from .llm import LLM, RequestOutput
    # yapf: disable
    from .llm_args import (AttentionDpConfig, AutoDecodingConfig, BatchingType,
                           BlockReuseConfig, CacheTransceiverConfig,
                           CalibConfig, CapacitySchedulerPolicy,
                           ColdPageQuantizationCompressionConfig,
                           ContextChunkingPolicy, CudaGraphConfig,
                           DecodeCudaGraphConfig, DeepSeekSparseAttentionConfig,
                           DeepSeekV4SparseAttentionConfig,
                           DFlashDecodingConfig, DraftTargetDecodingConfig,
                           DSparkDecodingConfig, DynamicBatchConfig,
                           Eagle3DecodingConfig, EagleDecodingConfig,
                           EncodeCudaGraphConfig, EncodeExtraInputSpec,
                           ExtendedRuntimePerfKnobConfig, KvCacheConfig,
                           KVEventsConfig, LlmArgs, MambaStateConfig,
                           MiniMaxM3SparseAttentionConfig, MoeConfig,
                           MTPDecodingConfig, MultimodalConfig,
                           NGramDecodingConfig, PARDDecodingConfig,
                           PrefillCudaGraphBackend, PrometheusMetricsConfig,
                           QSASparseAttentionConfig, ReorderRequestPolicyConfig,
                           RocketSparseAttentionConfig, SADecodingConfig,
                           SAEnhancerConfig, SaveHiddenStatesDecodingConfig,
                           SchedulerConfig, SkipSoftmaxAttentionConfig,
                           TorchCompileConfig, TorchLlmArgs,
                           TriAttentionKvCacheCompressionConfig,
                           UserProvidedDecodingConfig)
    from .llm_utils import KvCacheRetentionConfig, QuantAlgo, QuantConfig
    from .mm_encoder import MultimodalEncoder
    from .mpi_session import MpiCommSession
    from .thinking_budget import (ThinkingBudgetLogitsProcessor,
                                  add_thinking_budget_logits_processor)

    __all__ = [
        'LLM',
        'AsyncLLM',
        'MultimodalEncoder',
        'CompletionOutput',
        'RequestOutput',
        'GuidedDecodingParams',
        'SamplingParams',
        'DisaggregatedParams',
        'ConversationParams',
        'DisaggScheduleStyle',
        'BlockReuseConfig',
        'KvCacheConfig',
        'KVEventsConfig',
        'MambaStateConfig',
        'KvCacheRetentionConfig',
        'CudaGraphConfig',
        'DecodeCudaGraphConfig',
        'EncodeCudaGraphConfig',
        'EncodeExtraInputSpec',
        'MoeConfig',
        'EagleDecodingConfig',
        'Eagle3DecodingConfig',
        'MTPDecodingConfig',
        'SchedulerConfig',
        'CapacitySchedulerPolicy',
        'QuantConfig',
        'QuantAlgo',
        'CalibConfig',
        'RequestError',
        'MpiCommSession',
        'ExtendedRuntimePerfKnobConfig',
        'BatchingType',
        'ContextChunkingPolicy',
        'DynamicBatchConfig',
        'CacheTransceiverConfig',
        'NGramDecodingConfig',
        'PARDDecodingConfig',
        'DFlashDecodingConfig',
        'DSparkDecodingConfig',
        'SADecodingConfig',
        'SAEnhancerConfig',
        'UserProvidedDecodingConfig',
        'TorchCompileConfig',
        'DraftTargetDecodingConfig',
        'LlmArgs',
        'TorchLlmArgs',
        'AutoDecodingConfig',
        'AttentionDpConfig',
        'LoRARequest',
        'SaveHiddenStatesDecodingConfig',
        'RocketSparseAttentionConfig',
        'QSASparseAttentionConfig',
        'ReorderRequestPolicyConfig',
        'DeepSeekSparseAttentionConfig',
        'DeepSeekV4SparseAttentionConfig',
        'MiniMaxM3SparseAttentionConfig',
        'SchedulingParams',
        'SkipSoftmaxAttentionConfig',
        'ColdPageQuantizationCompressionConfig',
        'TriAttentionKvCacheCompressionConfig',
        'PrometheusMetricsConfig',
        'PrefillCudaGraphBackend',
        'ThinkingBudgetLogitsProcessor',
        'add_thinking_budget_logits_processor',
        'MultimodalConfig',
    ]
