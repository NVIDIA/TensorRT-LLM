# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
import os
from typing import Dict, List, Optional, Union

import torch

import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._utils import (confidential_compute_enabled, get_sm_version,
                                 prefer_pinned, str_dtype_to_binding,
                                 torch_dtype_to_str)
from tensorrt_llm.bindings.executor import DecodingMode
from tensorrt_llm.inputs.multimodal import MultimodalParams

# isort: off
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig, CapacitySchedulerPolicy, EagleDecodingConfig,
    KvCacheCompressionConfig, KvCacheConfig, MTPDecodingConfig, PeftCacheConfig,
    SamplerType, SchedulerConfig, SparseAttentionConfig, SpeculativeConfig,
    TorchLlmArgs, WaitingQueuePolicy)
# isort: on
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_helper import (LoraConfig,
                                      get_default_trtllm_modules_to_hf_modules)
from tensorrt_llm.lora_manager import load_torch_lora
from tensorrt_llm.mapping import CpType, Mapping

from ..attention_backend import get_sparse_attn_kv_cache_manager
from ..hostfunc import set_low_latency_dispatch
from ..model_config import ModelConfig
from ..models.modeling_multimodal_mixin import MultimodalModelMixin
from ..speculative import (get_num_extra_kv_tokens, get_num_spec_layers,
                           get_spec_decoder, should_use_separate_draft_kv_cache)
from .config_utils import (extract_mamba_kv_cache_params, is_gemma4_hybrid,
                           is_hybrid_linear, is_inkling, is_mla,
                           is_nemotron_hybrid, is_qwen3_hybrid)
from .connectors.kv_cache_connector import KvCacheConnectorManager
from .dwdp import DwdpManager
from .guided_decoder import GuidedDecoder
from .kv_cache_manager_v2 import KVCacheManagerV2
from .kv_cache_transceiver import AttentionTypeCpp, create_kv_cache_transceiver
from .llm_request import ExecutorResponse, LlmRequestState
from .mamba_cache_manager import (BaseMambaCacheManager,
                                  CppMambaHybridCacheManager,
                                  MixedMambaHybridCacheManager,
                                  use_py_mamba_cache_manager)
from .model_engine import PyTorchModelEngine
from .py_executor import PyExecutor
from .resource_manager import (BaseKVCacheCompressionManager, KVCacheManager,
                               PeftCacheManager, ResourceManager,
                               ResourceManagerType)
from .sampler import (EarlyStopSampler, EarlyStopWithMMResult, TorchSampler,
                      TRTLLMSampler)
from .scheduler import (BindCapacityScheduler, BindMicroBatchScheduler,
                        KVCacheV2Scheduler, SimpleScheduler,
                        SimpleUnifiedScheduler)
from .seq_slot_manager import SeqSlotManager

GB = 1 << 30


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _non_hybrid_kv_cache_manager_cls(config, kv_cache_config: KvCacheConfig):
    # Models with per-layer head_dim / num_kv_heads (e.g. Gemma4 hybrid
    # attention, or Inkling's local 16 / global 8 KV-head split) require
    # KVCacheManagerV2 for per-layer buffer sizes.
    needs_v2 = (kv_cache_config.use_kv_cache_manager_v2 is True
                or is_gemma4_hybrid(config) or is_inkling(config))
    return KVCacheManagerV2 if needs_v2 else KVCacheManager


def get_kv_cache_manager_cls(
        model_config: ModelConfig,
        kv_cache_config: KvCacheConfig,
        is_disagg: bool = False,
        cache_transceiver_config: Optional[CacheTransceiverConfig] = None):
    """Resolve the concrete KV cache manager class for ``model_config``.

    For hybrid mamba models the choice between ``Mixed`` (TRTLLM_USE_PY_MAMBA)
    and ``Cpp`` (unified pool with block reuse) is made here. Callers that
    don't care about disagg can omit ``is_disagg`` and get the unified-pool
    default.

    Env-var overrides (agg mode only — disagg picks its inner impl via
    ``cache_transceiver_config.transceiver_runtime``):
      * ``TRTLLM_USE_PY_MAMBA=1``  — Mixed manager with PythonMambaCacheManager.
    """
    config = model_config.pretrained_config
    sparse_attn_config = model_config.sparse_attention_config
    sparse_attn_algorithm = getattr(sparse_attn_config, "algorithm", None)
    if is_hybrid_linear(config):
        # Degenerate case: model is flagged as hybrid but the config has zero
        # mamba layers. Fall through to the standard non-hybrid routes.
        if model_config.get_num_mamba_layers() == 0:
            logger.info("Hybrid linear model has 0 mamba layers; using "
                        "KV cache manager without mamba caching")
            if sparse_attn_config is not None:
                return get_sparse_attn_kv_cache_manager(sparse_attn_config)
            return _non_hybrid_kv_cache_manager_cls(config, kv_cache_config)

        if (sparse_attn_config is not None
                and sparse_attn_algorithm != "skip_softmax"):
            raise ValueError(
                f"Sparse attention algorithm {sparse_attn_algorithm!r} is not "
                "supported with hybrid Mamba / linear-attention models.")

        # Skip Softmax only changes attention kernels. Hybrid models still
        # need a Mamba-capable cache manager for recurrent state.
        if use_py_mamba_cache_manager():
            if kv_cache_config.enable_block_reuse:
                raise ValueError(
                    "TRTLLM_USE_PY_MAMBA=1 forces "
                    "MixedMambaHybridCacheManager, which does not support "
                    "block reuse. Disable block reuse or unset "
                    "TRTLLM_USE_PY_MAMBA to use CppMambaHybridCacheManager.")
            logger.info(
                "Using MixedMambaHybridCacheManager for hybrid mamba model")
            return MixedMambaHybridCacheManager
        if kv_cache_config.enable_block_reuse:
            return CppMambaHybridCacheManager
        if (cache_transceiver_config is not None
                and cache_transceiver_config.transceiver_runtime == "PYTHON"):
            logger.info("Python transceiver detected; using "
                        "MixedMambaHybridCacheManager for hybrid mamba model")
            return MixedMambaHybridCacheManager
        default_cls = CppMambaHybridCacheManager
        env_override = os.environ.get('TLLM_MAMBA_MANAGER_PREFERENCE', None)
        if env_override is not None:
            if env_override.upper() == 'MIXED':
                logger.warning(
                    "Environment variable TLLM_MAMBA_MANAGER_PREFERENCE=MIXED overrides the default Mamba cache manager to MixedMambaHybridCacheManager. This may lead to increased memory usage due to lack of block reuse, but can be necessary for disaggregated setups or to avoid potential issues with the C++ manager. Set TLLM_MAMBA_MANAGER_PREFERENCE=CPP to use the CppMambaHybridCacheManager instead, which is the default for non-disaggregated setups without block reuse explicitly disabled."
                )
                return MixedMambaHybridCacheManager
            elif env_override.upper() == 'CPP':
                logger.warning(
                    "Environment variable TLLM_MAMBA_MANAGER_PREFERENCE=CPP overrides the default Mamba cache manager to CppMambaHybridCacheManager. This enables block reuse and can reduce memory usage, but may not be compatible with disaggregated setups. Set TLLM_MAMBA_MANAGER_PREFERENCE=MIXED to use the MixedMambaHybridCacheManager instead if you encounter issues with the C++ manager or are running in a disaggregated environment."
                )
                return CppMambaHybridCacheManager
            else:
                logger.warning(
                    f"Unrecognized value for TLLM_MAMBA_MANAGER_PREFERENCE: {env_override}. "
                    f"Expected 'CPP' or 'MIXED'. Using default {default_cls.__name__}."
                )
        return default_cls
    elif sparse_attn_config is not None:
        return get_sparse_attn_kv_cache_manager(sparse_attn_config)
    else:
        return _non_hybrid_kv_cache_manager_cls(config, kv_cache_config)


# --- KV cache cost model ------------------------------------------------------
#
# KVCacheManager.get_cache_size_per_token may return either an ``int``
# (legacy proportional model ``bytes = slope * tokens``) or an affine
# ``(slope, intercept)`` tuple (CppMambaHybridCacheManager, where mamba
# state introduces a per-batch fixed cost).  KVCacheManagerV2 reports
# sliding-window attention fixed cost in the tuple intercept.  CacheCost
# normalizes the combined shape so the rest of the file does plain attribute
# access and method calls instead of branching on type.


@dataclasses.dataclass(frozen=True)
class CacheCost:
    """Affine KV cache cost: ``bytes = slope * tokens + intercept``.

    The legacy proportional case is just ``intercept = 0``.
    """
    slope: int
    intercept: int = 0

    @classmethod
    def from_raw(cls, raw) -> "CacheCost":
        """Wrap an int / tuple / CacheCost result uniformly."""
        if isinstance(raw, CacheCost):
            return raw
        if isinstance(raw, tuple):
            slope, intercept = raw
            return cls(slope=int(slope), intercept=int(intercept))
        return cls(slope=int(raw))

    def __add__(self, other: "CacheCost") -> "CacheCost":
        if not isinstance(other, CacheCost):
            return NotImplemented
        return CacheCost(slope=self.slope + other.slope,
                         intercept=self.intercept + other.intercept)

    def __str__(self) -> str:
        if self.intercept == 0:
            return f"{self.slope} bytes/token"
        else:
            return f"{self.slope} bytes/token + {self.intercept} bytes fixed cost"

    def tokens_for_budget(self, budget: int) -> int:
        """Memory budget -> max tokens. Clamps a negative result to 0."""
        if self.slope <= 0:
            return 0
        tokens = max((budget - self.intercept) // self.slope, 0)
        return tokens

    def bytes_for_tokens(self, tokens: int) -> int:
        """Token count -> memory bytes."""
        return self.slope * tokens + self.intercept


def is_vswa_enabled(kv_cache_config):
    max_attention_window = kv_cache_config.max_attention_window
    return max_attention_window is not None and len(
        set(max_attention_window)) > 1


def _is_sliding_attention_layer(layer_type: object) -> bool:
    layer_type_name = getattr(layer_type, "name", str(layer_type)).lower()
    return "sliding" in layer_type_name


def _normalize_attention_windows(
    max_attention_window: List[int],
    max_seq_len: int,
) -> Optional[List[int]]:
    normalized = [min(max_seq_len, window) for window in max_attention_window]
    if all(window == max_seq_len for window in normalized):
        return None
    if len(set(normalized)) == 1:
        return [normalized[0]]
    return normalized


def _derive_draft_max_attention_window(
    kv_cache_config: KvCacheConfig,
    draft_pretrained_config: object,
    max_seq_len: int,
    num_draft_layers: int,
) -> Optional[List[int]]:
    if not is_vswa_enabled(kv_cache_config):
        max_attention_window = kv_cache_config.max_attention_window
        if max_attention_window is None:
            return None
        return _normalize_attention_windows(max_attention_window, max_seq_len)

    sliding_window = getattr(draft_pretrained_config, "sliding_window", None)
    layer_types = getattr(draft_pretrained_config, "layer_types", None)
    # HF configs today expose a single scalar `sliding_window`; `layer_types`
    # only marks sliding vs full. A draft with *multiple distinct* sliding window
    # sizes cannot be represented here — fail loudly instead of silently
    # collapsing every sliding layer to one size. Extension point: map each
    # sliding layer_type to its own window size.
    if isinstance(sliding_window, (list, tuple)):
        raise NotImplementedError(
            "Draft KV window derivation assumes a single sliding-window size, "
            f"got multiple: {sliding_window}")
    if sliding_window is not None and layer_types:
        layer_type_pattern = list(layer_types)
        if layer_type_pattern:
            draft_windows = []
            for layer_idx in range(num_draft_layers):
                layer_type = layer_type_pattern[layer_idx %
                                                len(layer_type_pattern)]
                draft_windows.append(
                    int(sliding_window)
                    if _is_sliding_attention_layer(layer_type) else max_seq_len)
            return _normalize_attention_windows(draft_windows, max_seq_len)

    use_sliding_window = getattr(draft_pretrained_config, "use_sliding_window",
                                 None)
    if sliding_window is not None and use_sliding_window is True:
        return _normalize_attention_windows([int(sliding_window)], max_seq_len)

    return None


class KvCacheCreator:
    """Groups together logic related to KV cache construction."""

    def __init__(
        self,
        *,
        model_engine: PyTorchModelEngine,
        draft_model_engine: Optional[PyTorchModelEngine],
        mapping: Mapping,
        net_max_seq_len: int,
        kv_connector_manager: Optional[KvCacheConnectorManager],
        max_num_tokens: int,
        max_beam_width: int,
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        llm_args: TorchLlmArgs,
        speculative_config: SpeculativeConfig,
        sparse_attention_config: SparseAttentionConfig,
        profiling_stage_data: Optional[dict],
        is_disagg: bool,
        execution_stream: Optional[torch.cuda.Stream] = None,
        draft_config: Optional[ModelConfig] = None,
        skip_est: bool = False,
    ):
        self._model_engine = model_engine
        self._draft_model_engine = draft_model_engine
        self._mapping = mapping
        self._kv_cache_config = kv_cache_config
        self._max_kv_tokens_in = self._kv_cache_config.max_tokens
        self._max_gpu_total_bytes_in = self._kv_cache_config.max_gpu_total_bytes
        self._pool_ratio_in = self._kv_cache_config.pool_ratio
        self._avg_seq_len_in = self._kv_cache_config.avg_seq_len
        self._max_num_tokens = max_num_tokens
        self._max_beam_width = max_beam_width
        self._kv_connector_manager = kv_connector_manager
        self._llm_args = llm_args
        self._speculative_config = speculative_config
        self._sparse_attention_config = sparse_attention_config
        self._tokens_per_block = tokens_per_block
        self._max_seq_len = max_seq_len
        self._max_batch_size = max_batch_size
        self._net_max_seq_len = net_max_seq_len
        self._dummy_reqs = None
        self._dummy_encoder_inputs: List[MultimodalParams] = []
        self._profiling_stage_data = profiling_stage_data
        self._is_disagg = is_disagg
        self._cache_transceiver_config = llm_args.cache_transceiver_config
        self._execution_stream = execution_stream
        self._kv_cache_manager_cls = self._get_model_kv_cache_manager_cls(
            model_engine)
        self._is_kv_cache_manager_v2 = issubclass(self._kv_cache_manager_cls,
                                                  KVCacheManagerV2)
        self._draft_config = draft_config
        self._skip_est = skip_est

    def _get_model_kv_cache_manager_cls(
        self,
        model_engine: PyTorchModelEngine,
        kv_cache_config_override: Optional[KvCacheConfig] = None,
    ):
        kv_cache_config = (kv_cache_config_override if kv_cache_config_override
                           is not None else self._kv_cache_config)
        model_config = model_engine.model.model_config
        cls = get_kv_cache_manager_cls(
            model_config,
            kv_cache_config,
            is_disagg=self._is_disagg,
            cache_transceiver_config=self._cache_transceiver_config)
        cls = self._fallback_if_unsupported_kv_cache_manager_v2(
            cls, model_config, kv_cache_config)
        # The V1-route hybrid mamba managers (disagg, TRTLLM_USE_CPP_MAMBA,
        # TRTLLM_USE_PY_MAMBA, or one-model speculative decoding) keep mamba
        # state in a separate cache that doesn't honor block reuse. Warn at
        # the routing site so users see the warning where the decision is
        # actually made.
        if is_hybrid_linear(model_engine.model.model_config.pretrained_config) \
                and kv_cache_config.enable_block_reuse:
            uses_v1_mamba_route = self._is_disagg \
                or os.environ.get('TRTLLM_USE_CPP_MAMBA', '0') == '1' \
                or os.environ.get('TRTLLM_USE_PY_MAMBA', '0') == '1' \
                or self._speculative_config is not None
            if uses_v1_mamba_route:
                logger.warning(
                    "Block reuse does not work with MTP for hybrid linear models "
                    "when using the legacy MambaCacheManager (TRTLLM_USE_CPP_MAMBA=1)"
                )
        return cls

    def _fallback_if_unsupported_kv_cache_manager_v2(
            self,
            kv_cache_manager_cls,
            model_config: ModelConfig,
            kv_cache_config: Optional[KvCacheConfig] = None):
        config = model_config.pretrained_config
        # Use ``issubclass`` rather than identity equality so V2 subclasses
        # (e.g. ``MiniMaxM3KVCacheManagerV2`` from the sparse-attention path)
        # also go through the V2-incompatible-feature gate below.
        if issubclass(kv_cache_manager_cls, KVCacheManagerV2):
            incompat: List[str] = []
            if self._kv_connector_manager is not None:
                incompat.append("kv_connector_manager")
            if self._max_beam_width is not None and self._max_beam_width > 1:
                incompat.append("beam_width > 1")
            if incompat:
                incompat_str = ", ".join(incompat)
                # Some models are structurally bound to V2 and cannot fall
                # back to V1 without producing wrong outputs:
                #   * Sparse-attention models (e.g. MiniMax-M3) need V2's
                #     per-layer split-pool to allocate the per-sparse-layer
                #     INDEX_KEY pool with a different stride than the main
                #     K/V pool. V1's unified pool cannot represent that.
                #   * Gemma4 hybrid uses per-layer head_dim that V1 would
                #     coerce to ``max(head_dim)``, changing per-layer KV
                #     byte sizes — correctness bug, not just efficiency.
                sparse_attn_config = model_config.sparse_attention_config
                if sparse_attn_config is not None:
                    raise NotImplementedError(
                        f"Sparse-attention models "
                        f"(algorithm={sparse_attn_config.algorithm!r}) require "
                        f"KVCacheManagerV2, which is not yet supported with "
                        f"{incompat_str}. Disable these KvCacheConfig features "
                        f"to run sparse-attention models.")
                if is_gemma4_hybrid(config):
                    raise NotImplementedError(
                        f"Gemma4 hybrid attention requires KVCacheManagerV2, "
                        f"which is not yet supported with {incompat_str}. "
                        f"Disable these features to run Gemma4 hybrid models.")
                if is_inkling(config):
                    # Inkling's per-layer KV-head split (local 16 / global 8)
                    # is the same structural V2 requirement as Gemma4's
                    # per-layer head_dim: V1's unified pool would coerce it to
                    # a single value, changing per-layer KV byte sizes -- a
                    # correctness bug, not just efficiency. Fail loudly rather
                    # than silently produce wrong outputs.
                    raise NotImplementedError(
                        f"Inkling hybrid attention requires KVCacheManagerV2, "
                        f"which is not yet supported with {incompat_str}. "
                        f"Disable these features to run Inkling.")
                # Plain V2 (explicitly enabled or selected by a model default):
                # V2 was a preference, not a structural requirement, so we can
                # safely fall back to V1.
                logger.warning(
                    "KVCacheManagerV2 is not supported with %s. "
                    "Falling back to KVCacheManager.", incompat_str)
                return KVCacheManager
        return kv_cache_manager_cls

    def _enable_kv_cache_stats(self) -> bool:
        return (self._llm_args.enable_iter_perf_stats
                or getattr(self._llm_args, "return_perf_metrics", False))

    def _per_manager_cache_cost(self,
                                manager_cls,
                                model_config,
                                kv_cache_config: Optional[KvCacheConfig] = None,
                                **extra_kwargs) -> CacheCost:
        kv_cache_config = (kv_cache_config if kv_cache_config is not None else
                           self._kv_cache_config)
        return CacheCost.from_raw(
            manager_cls.get_cache_size_per_token(
                model_config,
                self._mapping,
                tokens_per_block=self._tokens_per_block,
                max_seq_len=self._max_seq_len,
                max_batch_size=self._max_batch_size,
                kv_cache_config=kv_cache_config,
                spec_config=self._speculative_config,
                **extra_kwargs))

    def _get_kv_size_per_token(self,
                               kv_cache_config: Optional[KvCacheConfig] = None
                               ) -> CacheCost:
        """Aggregate KV cost across target + (optional) draft as a CacheCost.

        ``max_batch_size`` and ``kv_cache_config`` are passed unconditionally;
        managers that don't need them ignore via ``**kwargs``.
        """
        kv_cache_config = (kv_cache_config if kv_cache_config is not None else
                           self._kv_cache_config)
        model_config = self._model_engine.model.model_config
        total = self._per_manager_cache_cost(self._kv_cache_manager_cls,
                                             model_config, kv_cache_config)
        if self._is_encoder_decoder():
            total += CacheCost.from_raw(self._get_cross_kv_size_per_token())
        if self._draft_model_engine is not None:
            draft_model_config = self._draft_model_engine.model.model_config
            draft_kv_cache_manager_cls = self._get_model_kv_cache_manager_cls(
                self._draft_model_engine, kv_cache_config)
            total += self._per_manager_cache_cost(draft_kv_cache_manager_cls,
                                                  draft_model_config,
                                                  kv_cache_config)
        elif self._should_create_separate_draft_kv_cache():
            # One-model draft with separate KV cache layout.
            # Pass num_layers explicitly since the HF config may report a
            # different layer count than what is actually used at runtime
            # (e.g. EAGLE3: config says 1, runtime uses 4).
            # For PP, draft layers are only on the last rank (see
            # get_pp_layers), so only that rank should include draft cost.
            effective_draft_config = self._get_effective_draft_config()
            if self._speculative_config.spec_dec_mode.is_external_drafter():
                # External drafter: layers start from 0, normal PP distribution
                # Resolve draft manager class from draft config — may differ
                # from target (e.g. hybrid target + plain transformer draft).
                draft_kv_cache_manager_cls = get_kv_cache_manager_cls(
                    effective_draft_config,
                    kv_cache_config,
                    is_disagg=self._is_disagg)
                total += self._per_manager_cache_cost(
                    draft_kv_cache_manager_cls, effective_draft_config,
                    kv_cache_config)
            elif self._mapping.is_last_pp_rank():
                # EAGLE3/MTP: draft layers only on last PP rank
                total += self._per_manager_cache_cost(
                    self._kv_cache_manager_cls,
                    effective_draft_config,
                    kv_cache_config,
                    num_layers=self._get_num_draft_layers())
        return total

    def _cal_max_memory(self, peak_memory, total_gpu_memory, fraction,
                        allocated_bytes: int) -> int:
        """
        Calculate the max KV cache capacity.

        NOTE: `allocated_bytes` is the total KV-cache memory that must be pre-allocated during the estimation phase (for both the main and draft models) so the estimation run can complete successfully. When computing `available_kv_mem`, add this amount back in.
        """
        kv_size_per_token = self._get_kv_size_per_token()

        available_kv_mem = (total_gpu_memory - peak_memory +
                            allocated_bytes) * fraction
        logger.info(
            f"Peak memory during memory usage profiling (torch + non-torch): {peak_memory / (GB):.2f} GiB, "
            f"available KV cache memory when calculating max tokens: {available_kv_mem / (GB):.2f} GiB, "
            f"fraction is set {fraction}, kv size per token is {kv_size_per_token}. device total memory {total_gpu_memory / (GB):.2f} GiB, "
            f"temporary kv cache memory during profiling {allocated_bytes / (GB):.2f} GiB"
        )
        return int(available_kv_mem)

    def _create_dummy_context_requests(
            self, input_seq_len: int) -> List[trtllm.Request]:
        # Always text-only: this sizes the LLM-activation term at
        # ``max_num_tokens``. The multimodal encoder is profiled separately and
        # decoupled (``_encode_dummy_inputs`` in
        # ``configure_kv_cache_capacity``) by running the encoder on its own
        # worst-case dummy batch, so there is no multimodal dummy request here.
        requests = []
        vocab_size = self._model_engine.model.model_config.pretrained_config.vocab_size
        # These dummy warmup requests must pass the same token-range check real
        # requests do (PyExecutor._validate_token_range ->
        # request.check_token_id_range against lm_head.num_embeddings). When a
        # model's input-embedding vocab is padded ABOVE its (unpadded) output/head
        # vocab -- e.g. Inkling: embed 201024 vs lm_head 200058 -- sampling dummy
        # ids up to the padded vocab_size lands in the padded gap and fails
        # KV-cache capacity estimation with "Token ID out of range". Bound the
        # dummy range to the head so the warmup batch is always a valid request.
        lm_head = getattr(self._model_engine.model, "lm_head", None)
        head_vocab = getattr(lm_head, "num_embeddings", None)
        if head_vocab:
            vocab_size = min(vocab_size, head_vocab)
        max_num_tokens = self._max_num_tokens
        max_beam_width = self._max_beam_width

        input_seq_len = min(max_num_tokens, input_seq_len)
        remaining_tokens = max_num_tokens
        while remaining_tokens > 0:
            input_seq_len = min(input_seq_len, remaining_tokens)
            input_tokens = torch.randint(low=0,
                                         high=vocab_size,
                                         size=(input_seq_len, )).tolist()
            request = trtllm.Request(input_tokens,
                                     max_tokens=1,
                                     streaming=False,
                                     sampling_config=trtllm.SamplingConfig(
                                         beam_width=max_beam_width, ),
                                     output_config=trtllm.OutputConfig(),
                                     end_id=-1)
            if self._model_engine.use_mrope:
                request.py_multimodal_data = {
                    "mrope_config": {
                        "mrope_position_ids":
                        torch.zeros(3, 1, input_seq_len, dtype=torch.int32),
                        "mrope_position_deltas":
                        torch.zeros(1, 1, dtype=torch.int32)
                    }
                }
            request.py_conversation_params = None
            requests.append(request)
            remaining_tokens -= input_seq_len
        if self._mapping.enable_attention_dp:
            requests = requests * self._mapping.tp_size
        return requests

    def _create_dummy_encoder_inputs(self) -> List[MultimodalParams]:
        """Build the worst-case dummy multimodal batch for direct encoder
        profiling: the processor's worst-case dummy saturating
        ``encoder_max_num_tokens`` (one batched encoder forward), staged to GPU.

        Returns an empty list when the model is not a multimodal-encoder model
        or the processor has no dummy builder (then the encoder is not profiled
        directly). Whether the *processor* has opted into deterministic dummy
        sizing is detected below via ``NotImplementedError`` / empty demand — a
        model with the encoder entry but no dummy builder just yields an empty
        batch (no encoder profiling) until the builder is implemented.
        """
        # Gate on `MultimodalModelMixin`: the dummy-data sizing below only
        # needs the input processor, but `_encode_dummy_inputs` then calls
        # `model.encode_multimodal_inputs` (the mixin contract), so the model
        # must provide it. This also intentionally scopes direct encoder
        # profiling to mixin-migrated models (Qwen2-VL and Mistral are the
        # pilots; future models opt in by inheriting the mixin and implementing
        # the processor dummy hooks).
        if not isinstance(self._model_engine.model, MultimodalModelMixin):
            return []
        if isinstance(
                self._profiling_stage_data,
                dict) and not self._profiling_stage_data.get("enable_mm_reqs"):
            return []
        # No local multimodal encoder (disable_mm_encoder or MM E/P disagg):
        # nothing to profile.
        if getattr(self._model_engine.model, "mm_encoder", object()) is None:
            return []
        input_processor = self._model_engine.input_processor
        _, encoder_max_num_tokens = self._llm_args.get_encoder_runtime_sizes()
        # Modality-agnostic: the model declares each modality's per-item token
        # demand; split the shared ``encoder_max_num_tokens`` budget across them
        # in proportion to that demand (they share one encoder microbatch cap, so
        # the shares sum to the budget rather than each claiming all of it). The
        # processor then materializes a dummy per modality, merged into one batch
        # so a single encode profiles the combined peak. Empty demand /
        # NotImplementedError on the builder → text-only dummy fallback.
        demand = input_processor.get_mm_max_tokens_per_item()
        total_demand = sum(demand.values())
        if total_demand <= 0:
            return []
        max_tokens_per_modality = {
            m: max(1, encoder_max_num_tokens * d // total_demand)
            for m, d in demand.items()
        }
        try:
            multimodal_data = input_processor.get_dummy_mm_data_for_tokens(
                max_tokens_per_modality=max_tokens_per_modality,
                dtype=self._model_engine.model.dtype)
        except NotImplementedError:
            return []

        if not multimodal_data:
            return []
        params = MultimodalParams(multimodal_data=multimodal_data)
        params.to_device("multimodal_data",
                         "cuda",
                         pin_memory=prefer_pinned(),
                         target_keywords=getattr(
                             self._model_engine.model,
                             "multimodal_data_device_paths", None))
        return [params]

    def _encode_dummy_inputs(self):
        """Run the multimodal encoder(s) once on the pre-built worst-case dummy
        batch (``self._dummy_encoder_inputs``) and return its output so the
        embeddings stay resident while the peak is measured (the caller must hold
        the returned tensors so the peak accounts for the live embeddings).
        Returns ``None`` when direct profiling does not apply."""
        if not self._dummy_encoder_inputs:
            return None
        with torch.inference_mode():
            return self._model_engine.model.encode_multimodal_inputs(
                self._dummy_encoder_inputs)

    def _reserve_multimodal_encoder_cache_memory(
        self,
        peak_memory: int,
    ) -> int:
        """Reserve encoder-cache capacity when the model implements that cache."""
        model = self._model_engine.model
        if (not isinstance(model, MultimodalModelMixin)
                or not model.supports_encoder_cache):
            return peak_memory

        multimodal_config = model.model_config.multimodal_config
        if multimodal_config is None:
            return peak_memory
        return peak_memory + multimodal_config.encoder_cache_max_bytes

    def _get_token_num_for_estimation(self) -> int:
        """Compute KV cache capacity required for estimate_max_kv_cache_tokens to succeed."""
        if 'cp_type' in self._mapping.cp_config:
            raise ValueError(
                "KV cache size estimation not supported with context parallelism."
            )
        # estimate_max_kv_cache_tokens submits self._dummy_reqs
        num_cache_blocks = 0
        num_extra_tokens_per_seq = 1  # account for generated tokens
        spec_cfg = self._speculative_config
        if not self._llm_args.disable_overlap_scheduler and spec_cfg is not None:
            num_extra_tokens_per_seq += spec_cfg.tokens_per_gen_step - 1

        if spec_cfg is not None:
            num_extra_tokens_per_seq += spec_cfg.tokens_per_gen_step - 1
            num_extra_tokens_per_seq += get_num_extra_kv_tokens(spec_cfg)

        if self._dummy_reqs is None:
            self._dummy_reqs = self._create_dummy_context_requests(
                max(1, self._net_max_seq_len - 1))
            # Symmetric with `_dummy_reqs`: build the direct-encoder-profiling
            # batch here (empty for models without the uniform encoder entry);
            # `configure_kv_cache_capacity` runs it inside the peak window.
            self._dummy_encoder_inputs = self._create_dummy_encoder_inputs()
        for req in self._dummy_reqs:
            num_req_tokens = len(req.input_token_ids) + num_extra_tokens_per_seq
            # Requests cannot share KV cache blocks. Round up to nearest integer multiple of block size.
            num_cache_blocks += ceil_div(num_req_tokens, self._tokens_per_block)

        # With ADP enabled, _create_dummy_context_requests produces tp_size
        # copies so each rank gets work during the estimation warmup. But the
        # scheduler distributes them evenly (1 per rank), so each rank's KV
        # cache only needs capacity for its own share, not all of them.
        if self._mapping.enable_attention_dp and self._mapping.tp_size > 1:
            num_cache_blocks = (num_cache_blocks + self._mapping.tp_size -
                                1) // self._mapping.tp_size

        # Max cuda graph warmup required tokens
        max_cuda_graph_bs = min(self._model_engine.batch_size,
                                self._model_engine._max_cuda_graph_batch_size)
        # Round up the max seq len to the block size
        max_seq_len_blocks = ceil_div(self._model_engine.max_seq_len + 1,
                                      self._tokens_per_block)
        cuda_graph_warmup_block = max_seq_len_blocks + max_cuda_graph_bs - 1
        num_cache_blocks = max(cuda_graph_warmup_block, num_cache_blocks)

        # This is the minimal blocks required to run with max bs
        # If not able to allocate self._model_engine.batch_size blocks, the max batch size should be adjusted.
        num_cache_blocks = max(num_cache_blocks, self._model_engine.batch_size)

        # For VSWA (variable sliding window attention) models such as Gemma4
        # hybrid, KVCacheManagerV2 creates a separate pool group per distinct
        # attention window size. The quota passed via max_tokens is split
        # across pool groups proportionally, so each pool ends up with roughly
        # num_cache_blocks/num_pool_groups blocks in the worst case. A single
        # context request of max_seq_len tokens then exceeds the full-attention
        # pool's block budget and resize_context livelocks on suspend/retry
        # (observed for Gemma4 multimodal at max_seq_len>=8K, e.g. MMMU Pro).
        # Scale num_cache_blocks by the number of distinct pool groups so that
        # each pool has enough blocks for the dummy request even after the
        # proportional split. Inferred from the model config since the hybrid
        # max_attention_window hasn't been populated in kv_cache_config yet at
        # this stage (it's filled in later by _create_kv_cache_manager).
        # Only V2 has split-pool semantics — Mamba hybrid (which also has
        # heterogeneous layer_types) uses MambaHybridCacheManager and would
        # have its max_tokens estimate inflated incorrectly otherwise.
        num_pool_groups = 1
        if self._is_kv_cache_manager_v2:
            model_cfg = self._model_engine.model.model_config.pretrained_config
            layer_types = getattr(model_cfg, "layer_types", None)
            if isinstance(layer_types, (list, tuple)):
                distinct = len(set(layer_types))
                if distinct > 1:
                    num_pool_groups = distinct
            elif (self._kv_cache_config.max_attention_window is not None
                  and len(set(self._kv_cache_config.max_attention_window)) > 1):
                num_pool_groups = len(
                    set(self._kv_cache_config.max_attention_window))
        num_cache_blocks *= num_pool_groups

        # Multiply by beam width, to prevent rescaling of the max_seq_len caused by the influence of beam width during the preparation for kv_cache_estimation
        max_num_tokens_for_estimation = (
            num_cache_blocks * self._tokens_per_block *
            self._dummy_reqs[0].sampling_config.beam_width)
        # V2 capacity is controlled by max_gpu_total_bytes; max_tokens only
        # describes the dummy workload needed for estimation.
        if self._is_kv_cache_manager_v2:
            return max_num_tokens_for_estimation

        free_mem, _ = torch.cuda.mem_get_info()
        max_memory = self._kv_cache_config.free_gpu_memory_fraction * free_mem
        kv_size_per_token = self._get_kv_size_per_token()
        max_num_tokens_in_memory = (
            kv_size_per_token.tokens_for_budget(max_memory) //
            self._tokens_per_block * self._tokens_per_block)
        return min(max_num_tokens_for_estimation, max_num_tokens_in_memory)

    def try_prepare_estimation(self) -> bool:
        """Prepare for possible KV cache capacity estimation.

        This updates `kv_cache_config` and returns a boolean indicating whether KV cache
        estimation is to be performend.
        """
        if self._skip_est:
            return False

        estimating_kv_cache = True
        if 'cp_type' in self._mapping.cp_config:
            estimating_kv_cache = False
            logger.info(
                "KV cache size estimation is not supported for context parallelism, disable it."
            )
        model_config = self._model_engine.model.model_config
        if model_config.attn_backend == "VANILLA":
            estimating_kv_cache = False
            logger.info(
                "KV cache size estimation is not supported for Vanilla attention backend, disable it."
            )

        if estimating_kv_cache:
            estimate_max_tokens = self._get_token_num_for_estimation()
            max_tokens = min(
                estimate_max_tokens, self._kv_cache_config.max_tokens
            ) if self._kv_cache_config.max_tokens is not None else estimate_max_tokens
            # User-provided pool sizing can underprovision the temporary
            # estimation cache and cause warmup to hang or fail. Override it
            # for estimation, then restore it in configure_kv_cache_capacity().
            self._kv_cache_config.pool_ratio = None
            self._kv_cache_config.avg_seq_len = self._max_seq_len
            if self._is_kv_cache_manager_v2:
                free_mem, _ = torch.cuda.mem_get_info()
                max_gpu_total_bytes = int(
                    self._kv_cache_config.free_gpu_memory_fraction * free_mem)
                if (self._max_gpu_total_bytes_in is not None
                        and self._max_gpu_total_bytes_in > 0):
                    max_gpu_total_bytes = min(max_gpu_total_bytes,
                                              self._max_gpu_total_bytes_in)
                self._kv_cache_config.max_gpu_total_bytes = max_gpu_total_bytes
                self._kv_cache_config.max_tokens = max_tokens
            else:
                self._kv_cache_config.max_tokens = max_tokens
        return estimating_kv_cache

    def configure_kv_cache_capacity(self,
                                    py_executor: PyExecutor = None) -> None:
        """Perform KV cache capacity estimation.
        NOTE: for VSWA case, we calculate and set kv cache memory instead of using max_tokens in kv_cache_config.

        This updates `kv_cache_config`.
        """
        mapping = self._mapping

        # TODO: support CP by generating dummy requests for it.
        assert 'cp_type' not in mapping.cp_config

        fraction = self._kv_cache_config.free_gpu_memory_fraction

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        end, total_gpu_memory = torch.cuda.mem_get_info()
        total_used_bytes = total_gpu_memory - end
        model_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        logger.info(
            f"Memory used after loading model weights (inside torch) in memory usage profiling: {model_bytes / (GB):.2f} GiB"
        )
        logger.info(
            f"Memory used after loading model weights (outside torch) in memory usage profiling: {((total_used_bytes - model_bytes) if total_used_bytes > model_bytes else 0) / (GB):.2f} GiB"
        )

        if py_executor is not None and not self._skip_est:
            # Direct encoder profiling: run the vision encoder once
            # on the worst-case dummy batch and keep its embeddings resident so
            # the peak below captures the encoder activation + live embeddings,
            # while the LLM-activation term comes from the (text-only) dummy
            # requests. ``None`` for models without the uniform encoder entry.
            # Bound (not discarded) so the embeddings stay resident through the
            # peak read below; ``del`` frees them afterward.
            encoder_profile_output = self._encode_dummy_inputs()  # noqa: F841
            py_executor.set_gather_responses(True)
            origin_iter_stats = py_executor.enable_iter_perf_stats
            py_executor.enable_iter_perf_stats = False
            req_ids = []
            if py_executor.dist.mapping.rank == 0:
                req_ids = py_executor.enqueue_requests(self._dummy_reqs)
            req_ids = py_executor.dist.broadcast(req_ids, root=0)
            py_executor.is_warmup = True
            py_executor.start_worker()
            try:
                responses = py_executor.await_responses(req_ids)
                for response_or_list in responses:
                    response_list = [response_or_list] if isinstance(
                        response_or_list,
                        ExecutorResponse) else response_or_list
                    for response in response_list:
                        if response.has_error():
                            raise RuntimeError(response.error_msg)

                torch_peak_memory = torch.cuda.memory_stats(
                )["allocated_bytes.all.peak"]

                # Free the held encoder embeddings and the GPU-resident dummy
                # encoder inputs now that the peak (which they contributed to)
                # has been recorded, so the steady-state measurement below
                # doesn't count these transient dummies.
                del encoder_profile_output
                self._dummy_encoder_inputs = []

                # Clear the caching allocator before measuring the current memory usage
                torch.cuda.empty_cache()
                end, total_gpu_memory = torch.cuda.mem_get_info()
                torch_used_bytes = torch.cuda.memory_stats(
                )["allocated_bytes.all.current"]
            finally:
                # get kv cache stats for both model and draft model
                kv_stats = py_executor.resource_manager.resource_managers.get(
                    ResourceManagerType.KV_CACHE_MANAGER).get_kv_cache_stats()
                # Get draft KV cache stats if present (either from two-model mode or one-model
                # mode with separate draft KV cache)
                draft_kv_cache_manager = py_executor.resource_manager.resource_managers.get(
                    ResourceManagerType.DRAFT_KV_CACHE_MANAGER)
                kv_stats_draft = draft_kv_cache_manager.get_kv_cache_stats(
                ) if draft_kv_cache_manager is not None else None

                # get total allocated bytes
                allocated_bytes = kv_stats.allocated_bytes + (
                    kv_stats_draft.allocated_bytes
                    if kv_stats_draft is not None else 0)
                py_executor.is_warmup = False
                py_executor.shutdown()
                py_executor.enable_iter_perf_stats = origin_iter_stats
                py_executor.set_gather_responses(False)

            total_used_bytes = total_gpu_memory - end
            activation_bytes = torch_peak_memory - model_bytes
            extra_cost = max(total_used_bytes - torch_used_bytes, 0)
            peak_memory = torch_peak_memory + extra_cost
            logger.info(
                f"Memory dynamically allocated during inference (inside torch) in memory usage profiling: {activation_bytes / (GB):.2f} GiB"
            )
            logger.info(
                f"Memory used outside torch (e.g., NCCL and CUDA graphs) in memory usage profiling: {extra_cost / (GB):.2f} GiB"
            )

        else:
            peak_memory = total_used_bytes
            allocated_bytes = 0
            activation_bytes = 0

        peak_memory_without_encoder_cache = peak_memory
        peak_memory = self._reserve_multimodal_encoder_cache_memory(peak_memory)
        if peak_memory != peak_memory_without_encoder_cache:
            mem_gb = (peak_memory - peak_memory_without_encoder_cache) / GB
            logger.info(
                f"Reserving {mem_gb:.2f} GiB for the multimodal encoder cache while estimating KV cache "
                "capacity.", )

        # calculate max memory from peak memory and free gpu memory fraction
        kv_cache_max_memory = self._cal_max_memory(peak_memory,
                                                   total_gpu_memory, fraction,
                                                   allocated_bytes)

        # Estimation uses inferred pool sizing; the final manager uses the
        # user-provided configuration.
        self._kv_cache_config.pool_ratio = self._pool_ratio_in
        self._kv_cache_config.avg_seq_len = self._avg_seq_len_in

        # NOTE:
        # For KVCacheManager, KvCacheCreator currently controls capacity using two parameters in KVCacheConfig:
        #   • max_tokens
        #   • max_gpu_total_bytes
        # For KVCacheManagerV2, KvCacheCreator controls capacity using max_gpu_total_bytes only.
        # This leaves max_tokens as a user-defined constraint.

        # ---------------------------handle max_tokens---------------------------------
        if self._is_kv_cache_manager_v2:
            # KVCacheManagerV2 doesn't rely on max_tokens to control capacity, so restore user provided value
            self._kv_cache_config.max_tokens = self._max_kv_tokens_in
        else:
            # handle user provided max_tokens
            if self._max_kv_tokens_in is not None:
                # raise error if it is VSWA case
                is_vswa = is_vswa_enabled(self._kv_cache_config)

                # raise error if it is VSWA case
                if is_vswa:
                    logger.warning(
                        "max_tokens should not be set for VSWA case as it is ambiguous concept for VSWA."
                    )
                # calculate max memory from max_tokens
                kv_size_per_token = self._get_kv_size_per_token()
                kv_cache_max_memory_from_max_tokens = (
                    kv_size_per_token.bytes_for_tokens(self._max_kv_tokens_in))
                kv_cache_max_memory = min(kv_cache_max_memory,
                                          kv_cache_max_memory_from_max_tokens)
                logger.info(
                    f"max_tokens={self._max_kv_tokens_in} is provided. It limits max memory to {kv_cache_max_memory_from_max_tokens / (GB):.2f} GiB. "
                    f"New max_memory is set to {kv_cache_max_memory / (GB):.2f} GiB"
                )
            # For KvCacheManager, its logic still relies on max_tokens to control capacity
            self._kv_cache_config.max_tokens = (self._get_kv_size_per_token(
            ).tokens_for_budget(kv_cache_max_memory))
        # ---------------------------handle max_tokens---------------------------------

        # ---------------------------handle max_gpu_total_bytes---------------------------------
        # if user provided max_gpu_total_bytes, set max memory from max_gpu_total_bytes
        if (self._max_gpu_total_bytes_in is not None
                and self._max_gpu_total_bytes_in > 0):
            kv_cache_max_memory = min(kv_cache_max_memory,
                                      self._max_gpu_total_bytes_in)
            logger.info(
                f"max_gpu_total_bytes={self._max_gpu_total_bytes_in / (GB):.2f} GiB is provided. New max memory is {kv_cache_max_memory / (GB):.2f} GiB"
            )

        logger.info(
            f"Estimated max memory in KV cache : {kv_cache_max_memory / (GB):.2f} GiB"
        )
        # set max_gpu_total_bytes
        self._kv_cache_config.max_gpu_total_bytes = kv_cache_max_memory
        if isinstance(self._profiling_stage_data, dict):
            self._profiling_stage_data["activation_bytes"] = activation_bytes
        # ---------------------------handle max_gpu_total_bytes---------------------------------

    def _create_kv_cache_manager(
        self,
        model_engine: PyTorchModelEngine,
        estimating_kv_cache: bool = False,
        kv_cache_config_override: Optional[KvCacheConfig] = None
    ) -> KVCacheManager:
        mapping = self._mapping
        assert model_engine.model.model_config.is_generation, "Only construct KV cache for generation models."
        kv_cache_config = (kv_cache_config_override if kv_cache_config_override
                           is not None else self._kv_cache_config)
        kv_cache_manager_cls = self._get_model_kv_cache_manager_cls(
            model_engine, kv_cache_config)

        # When using separate draft KV cache in one-model speculative decoding,
        # use layer_mask to include only target layers. The draft layers should
        # only be in the separate draft KV cache manager.
        # We still pass spec_config so that num_extra_kv_tokens is calculated.
        spec_dec_layer_mask = None
        if self._should_create_separate_draft_kv_cache():
            num_target_layers = model_engine.model.model_config.pretrained_config.num_hidden_layers
            spec_dec_layer_mask = [True] * num_target_layers

        estimating_kv_cache = estimating_kv_cache and not self._skip_est
        kv_cache_manager = _create_kv_cache_manager(
            model_engine=model_engine,
            kv_cache_manager_cls=kv_cache_manager_cls,
            mapping=mapping,
            kv_cache_config=kv_cache_config,
            tokens_per_block=self._tokens_per_block,
            max_seq_len=self._max_seq_len,
            max_batch_size=self._max_batch_size,
            spec_config=self._speculative_config,
            sparse_attention_config=self._sparse_attention_config,
            max_num_tokens=self._max_num_tokens,
            max_beam_width=self._max_beam_width,
            kv_connector_manager=self._kv_connector_manager,
            estimating_kv_cache=estimating_kv_cache,
            enable_kv_cache_stats=self._enable_kv_cache_stats()
            and not estimating_kv_cache,
            execution_stream=self._execution_stream,
            layer_mask=spec_dec_layer_mask,
            is_disagg=self._is_disagg,
        )

        if not self._skip_est:
            # KVCacheManager (Non-draft) modifies the max_seq_len field, update it to self
            if model_engine.kv_cache_manager_key == ResourceManagerType.KV_CACHE_MANAGER:
                # When SWA is enabled, max_seq_len is updated inside kv_cache_manager.
                if kv_cache_manager is not None:
                    if kv_cache_manager.max_seq_len < self._max_seq_len:
                        self._dummy_reqs = self._create_dummy_context_requests(
                            max(
                                1, self._net_max_seq_len - 1 -
                                (self._max_seq_len -
                                 kv_cache_manager.max_seq_len)))
                    self._max_seq_len = kv_cache_manager.max_seq_len

                # When SWA is enabled, max_seq_len is updated inside kv_cache_manager.
                if kv_cache_manager is not None:
                    if kv_cache_manager.max_seq_len < self._max_seq_len:
                        self._dummy_reqs = self._create_dummy_context_requests(
                            max(1, kv_cache_manager.max_seq_len - 1))
                    self._max_seq_len = kv_cache_manager.max_seq_len
        else:
            if kv_cache_manager is not None:
                self._max_seq_len = kv_cache_manager.max_seq_len

        return kv_cache_manager

    def _should_create_separate_draft_kv_cache(self) -> bool:
        """
        Check if we need a separate draft KV cache manager for one-model mode.
        Returns True if the speculative config has use_separate_draft_kv_cache=True.

        Note: For MTP, _draft_config may be None since MTP layers are embedded
        in the target model and don't produce a separate ModelConfig. We fall
        back to the target model's config via _get_effective_draft_config().
        """
        if self._mapping.enable_attention_dp:
            logger.info(
                "Attention DP is enabled, separate draft KV cache is not supported."
            )
            return False
        return should_use_separate_draft_kv_cache(self._speculative_config)

    def _get_effective_draft_config(self) -> ModelConfig:
        """
        Return the ModelConfig to use for draft KV cache creation.

        For Eagle3 and draft-target one-model modes, a dedicated draft config
        is provided at construction time.  For MTP one-model mode, no separate
        draft config exists because the MTP layers share the same architecture
        as the target model.  In that case we fall back to the target model's
        config so that the draft KV cache is created with the correct layout.
        """
        if self._draft_config is not None:
            return self._draft_config
        # MTP: MTP layers reuse the target model architecture, so the target
        # model's config describes the correct KV cache layout for the draft
        # layers as well.
        return self._model_engine.model.model_config

    def _get_num_draft_layers(self) -> int:
        """Return the actual number of draft KV cache layers.

        This must stay in sync with the num_layers passed to the draft KV
        cache manager constructor in _create_one_model_draft_kv_cache_manager.
        """
        if self._speculative_config.spec_dec_mode.is_external_drafter():
            return self._draft_config.pretrained_config.num_hidden_layers
        return get_num_spec_layers(self._speculative_config)

    def _create_one_model_draft_kv_cache_manager(
        self,
        estimating_kv_cache: bool = False,
        kv_cache_config_override: Optional[KvCacheConfig] = None,
    ) -> Optional[KVCacheManager]:
        """
        Create a KV cache manager for draft model layers in one-model mode
        when target and draft have different KV cache layouts.
        """
        # Get target model's num_hidden_layers to compute correct layer indices.
        # Draft model layers in one-model mode start at target_num_layers.
        target_pretrained_config = self._model_engine.model.model_config.pretrained_config
        target_num_layers = target_pretrained_config.num_hidden_layers

        # PARD, External Drafter: draft is a separate model, layers start from 0.
        # Other methods (EAGLE3, MTP): draft layers are appended after target layers.
        num_draft_layers = self._get_num_draft_layers()
        if self._speculative_config.spec_dec_mode.is_external_drafter():
            spec_dec_layer_mask = [True] * num_draft_layers
        else:
            spec_dec_layer_mask = [False] * target_num_layers + [
                True
            ] * num_draft_layers

        # Get the effective draft config (explicit draft_config if available,
        # otherwise fall back to target model config for MTP).
        effective_draft_config = self._get_effective_draft_config()

        draft_kv_config = (kv_cache_config_override if kv_cache_config_override
                           is not None else self._kv_cache_config).model_copy()
        draft_kv_config.max_attention_window = _derive_draft_max_attention_window(
            self._kv_cache_config,
            effective_draft_config.pretrained_config,
            self._max_seq_len,
            num_draft_layers,
        )
        # A draft whose *own* config is VSWA (mixed sliding/full attention
        # layers, so the derived window has >1 distinct size) is envisioned but
        # not yet supported here. ``draft_kv_config`` inherits the target's
        # combined ``max_gpu_total_bytes`` via the ``model_copy()`` above; a
        # VSWA draft would route through ``calculate_max_num_blocks_for_vswa``,
        # which sizes pools from that full byte budget rather than the draft's
        # ``max_tokens`` share — so the separate draft manager would re-allocate
        # the whole KV budget and OOM. Only the non-SWA draft path (single/None
        # window, which falls back to the ``max_tokens``-partitioned allocation)
        # is exercised today; no draft model currently ships with mixed
        # ``layer_types``. When one does, partition the budget here before the
        # per-window split, e.g.:
        #     _, draft_cost = self._get_target_and_draft_cache_costs()
        #     draft_kv_config.max_gpu_total_bytes = draft_cost.bytes_for_tokens(
        #         self._kv_cache_config.max_tokens)
        if is_vswa_enabled(draft_kv_config):
            raise NotImplementedError(
                "A VSWA draft model (mixed sliding-window and full-attention "
                "layers) is not yet supported for one-model speculative "
                "decoding with a separate draft KV cache manager: its KV budget "
                "would not be partitioned from the target's and would overrun "
                "GPU memory. Derived draft max_attention_window="
                f"{draft_kv_config.max_attention_window}.")
        if is_vswa_enabled(self._kv_cache_config):
            logger.info(
                f"Derived draft KV cache max_attention_window for separate "
                f"draft manager: {draft_kv_config.max_attention_window}")
        # Get the appropriate KV cache manager class for the draft model
        draft_kv_cache_manager_cls = get_kv_cache_manager_cls(
            effective_draft_config, draft_kv_config, is_disagg=self._is_disagg)
        draft_kv_cache_manager_cls = self._fallback_if_unsupported_kv_cache_manager_v2(
            draft_kv_cache_manager_cls, effective_draft_config, draft_kv_config)

        estimating_kv_cache = estimating_kv_cache and not self._skip_est
        # For MTP with models using sparse attention (e.g., DeepSeek V3 with DSA),
        # the draft layers share the same architecture as the target model and need
        # the sparse_attention_config. Get it from effective_draft_config which
        # falls back to the target model's config for MTP mode.
        sparse_attn_config = effective_draft_config.sparse_attention_config
        return _create_kv_cache_manager(
            model_engine=None,
            kv_cache_manager_cls=draft_kv_cache_manager_cls,
            mapping=self._mapping,
            kv_cache_config=draft_kv_config,
            tokens_per_block=self._tokens_per_block,
            max_seq_len=self._max_seq_len,
            max_batch_size=self._max_batch_size,
            spec_config=self._speculative_config,
            sparse_attention_config=sparse_attn_config,
            max_num_tokens=self._max_num_tokens,
            max_beam_width=self._max_beam_width,
            kv_connector_manager=self._kv_connector_manager,
            estimating_kv_cache=estimating_kv_cache,
            enable_kv_cache_stats=self._enable_kv_cache_stats()
            and not estimating_kv_cache,
            execution_stream=self._execution_stream,
            # One-model draft specific overrides
            model_config=effective_draft_config,
            dtype=effective_draft_config.pretrained_config.torch_dtype,
            is_draft=True,
            layer_mask=spec_dec_layer_mask,
            num_layers=num_draft_layers,
            is_disagg=self._is_disagg,
        )

    def _get_target_and_draft_cache_costs(
        self,
        kv_cache_config: Optional[KvCacheConfig] = None,
    ) -> Optional[tuple[CacheCost, CacheCost]]:
        """Per-manager KV cache costs for target and draft layers."""
        target_kv_cache_config = (kv_cache_config if kv_cache_config is not None
                                  else self._kv_cache_config)
        total_kv = self._get_kv_size_per_token(target_kv_cache_config)
        target_kv = self._per_manager_cache_cost(
            self._kv_cache_manager_cls, self._model_engine.model.model_config,
            target_kv_cache_config)
        # The draft contribution is whatever the aggregate has on top of the
        # target. Both pieces are CacheCost; subtraction is component-wise.
        draft_kv = CacheCost(slope=total_kv.slope - target_kv.slope,
                             intercept=total_kv.intercept - target_kv.intercept)
        if target_kv.slope <= 0 or draft_kv.slope <= 0:
            return None
        return target_kv, draft_kv

    def _compute_draft_budget_shares(
        self,
        total_budget: int,
        target_kv: CacheCost,
        draft_kv: CacheCost,
    ) -> Optional[tuple[int, int]]:
        """Split *total_budget* into (target_budget, draft_budget) byte shares."""
        intercept_total = target_kv.intercept + draft_kv.intercept
        slope_budget = total_budget - intercept_total
        if slope_budget <= 0:
            logger.warning(
                f"KV cache budget {total_budget} is smaller than the fixed "
                f"mamba state cost {intercept_total}; cannot split between "
                f"target and draft.")
            return None
        slope_total = target_kv.slope + draft_kv.slope
        draft_slope_share = int(slope_budget * draft_kv.slope / slope_total)
        draft_budget = draft_kv.intercept + draft_slope_share
        target_budget = total_budget - draft_budget
        return target_budget, draft_budget

    def _split_kv_cache_budget_for_draft(
        self,
        budget_attr: str,
        target_kv_cache_config: Optional[KvCacheConfig] = None,
        draft_kv_cache_config: Optional[KvCacheConfig] = None,
    ) -> tuple[KvCacheConfig, Optional[KvCacheConfig]]:
        """Split a byte budget (attribute on ``KvCacheConfig``) between target
        and draft KV caches.

        Splits the value of ``target_kv_cache_config.<budget_attr>`` using the
        affine target/draft cache costs, then returns cloned target and draft
        configs containing their respective shares.

        The input target config and the creator's base config are not mutated.
        When the split is not applicable (the budget is not set, or the
        per-manager cache costs are unavailable), the input configs are returned
        unchanged.

        The affine fixed (intercept) cost models GPU-resident state (e.g. mamba
        SSM state). It is only charged against ``max_gpu_total_bytes``; for any
        other budget (e.g. ``host_cache_size``, which is host offload memory the
        GPU-resident state never occupies) the intercept is dropped so the split
        stays proportional to the per-token cost.

        When the split is *infeasible* (the combined fixed cost meets or exceeds
        the budget — only possible for ``max_gpu_total_bytes`` after the above)
        the shortfall is fatal: both managers need their fixed state resident in
        GPU memory, so the run would OOM. It raises ``ValueError`` rather than
        silently producing an unusable config. A defensive degrade-to-zero path
        for non-GPU budgets remains so the draft never silently inherits the full
        budget and double-allocates it.
        """
        target_kv_cache_config = (target_kv_cache_config
                                  if target_kv_cache_config is not None else
                                  self._kv_cache_config)
        total_budget = getattr(target_kv_cache_config, budget_attr) or 0
        if total_budget <= 0:
            return target_kv_cache_config, draft_kv_cache_config

        cache_costs = self._get_target_and_draft_cache_costs(
            target_kv_cache_config)
        if cache_costs is None:
            return target_kv_cache_config, draft_kv_cache_config
        target_kv, draft_kv = cache_costs

        # The fixed (intercept) cost models GPU-resident state such as mamba SSM
        # state; it does not consume host offload memory. When splitting a
        # non-GPU budget (e.g. host_cache_size), drop the intercept so the split
        # stays proportional to the per-token (slope) cost instead of being
        # spuriously starved by a GPU-only fixed cost.
        if budget_attr != "max_gpu_total_bytes":
            target_kv = CacheCost(slope=target_kv.slope)
            draft_kv = CacheCost(slope=draft_kv.slope)

        shares = self._compute_draft_budget_shares(total_budget, target_kv,
                                                   draft_kv)
        if shares is None:
            # The split is infeasible (combined fixed cost >= total budget).
            intercept_total = target_kv.intercept + draft_kv.intercept
            if budget_attr == "max_gpu_total_bytes":
                # A GPU budget that cannot even fit the combined fixed cost is
                # fatal: both managers need their fixed state resident in GPU
                # memory, so the run would OOM. Fail fast with actionable
                # guidance rather than producing an unusable zero-budget draft.
                raise ValueError(
                    f"KV cache GPU budget ({total_budget / GB:.2f} GiB) is "
                    f"smaller than the combined fixed cost "
                    f"({intercept_total / GB:.2f} GiB, e.g. mamba SSM state) "
                    f"for target+draft. Increase free_gpu_memory_fraction or "
                    f"max_gpu_total_bytes, or reduce max_batch_size (the fixed "
                    f"cost scales with batch size).")
            # Defensive: non-GPU budgets zero out the intercept above, so with a
            # positive budget this branch is currently unreachable for them. It
            # remains as a safety net guaranteeing that, should a non-GPU budget
            # ever carry a fixed cost it cannot fit, we degrade gracefully rather
            # than letting both managers inherit the full budget and
            # double-allocate it: keep the full budget on the target and zero the
            # draft's share for this attribute.
            logger.warning(
                f"Cannot split KV cache {budget_attr} between target and draft; "
                f"assigning the draft a zero {budget_attr} budget to avoid "
                f"double-allocating the full budget.")
            if draft_kv_cache_config is None:
                draft_kv_cache_config = target_kv_cache_config.model_copy()
            else:
                draft_kv_cache_config = draft_kv_cache_config.model_copy()
            setattr(draft_kv_cache_config, budget_attr, 0)
            return target_kv_cache_config, draft_kv_cache_config
        target_budget, draft_budget = shares

        logger.info(
            f"Splitting KV cache {budget_attr}: total={total_budget / GB:.2f} GiB, "
            f"target={target_budget / GB:.2f} GiB ({target_kv}), "
            f"draft={draft_budget / GB:.2f} GiB ({draft_kv})")

        split_target_kv_cache_config = target_kv_cache_config.model_copy()
        setattr(split_target_kv_cache_config, budget_attr, target_budget)
        if draft_kv_cache_config is None:
            split_draft_kv_cache_config = target_kv_cache_config.model_copy()
        else:
            split_draft_kv_cache_config = draft_kv_cache_config.model_copy()
        setattr(split_draft_kv_cache_config, budget_attr, draft_budget)
        return split_target_kv_cache_config, split_draft_kv_cache_config

    def _is_encoder_decoder(self) -> bool:
        return self._model_engine.model.model_config.is_encoder_decoder

    @staticmethod
    def _get_config_int_attr(config, names: tuple[str, ...]) -> Optional[int]:
        for name in names:
            value = getattr(config, name, None)
            if isinstance(value, int):
                return value
        return None

    def _get_cross_kv_cache_layout(
        self,
        fallback_max_seq_len: Optional[int] = None
    ) -> tuple[int, int, int, int]:
        """Return decoder-layer count and encoder KV geometry for cross cache."""
        config = self._model_engine.model.model_config.pretrained_config

        num_layers = self._get_config_int_attr(
            config,
            ("num_decoder_layers", "decoder_layers", "num_hidden_layers",
             "num_layers"),
        )
        if num_layers is None:
            raise ValueError(
                "Unable to determine decoder layer count for cross KV cache.")

        encoder_num_heads = self._get_config_int_attr(
            config,
            ("encoder_num_heads", "encoder_attention_heads", "num_heads",
             "num_attention_heads"),
        )
        if encoder_num_heads is None:
            raise ValueError(
                "Unable to determine encoder attention head count for cross KV cache."
            )

        num_kv_heads = self._get_config_int_attr(
            config,
            ("encoder_num_kv_heads", "encoder_num_key_value_heads",
             "encoder_attention_heads", "encoder_num_heads",
             "num_key_value_heads", "num_heads", "num_attention_heads"),
        )
        if num_kv_heads is None:
            num_kv_heads = encoder_num_heads

        encoder_hidden_size = self._get_config_int_attr(
            config, ("encoder_hidden_size", "d_model", "hidden_size"))
        if encoder_hidden_size is None:
            raise ValueError(
                "Unable to determine encoder hidden size for cross KV cache.")

        head_dim = self._get_config_int_attr(
            config,
            ("encoder_head_size", "encoder_head_dim", "d_kv"),
        )
        if head_dim is None:
            head_dim = encoder_hidden_size // encoder_num_heads

        max_seq_len = fallback_max_seq_len or self._max_seq_len
        max_input_len = getattr(self._llm_args, "max_input_len", None)
        if isinstance(max_input_len, int) and max_input_len > 0:
            max_seq_len = max_input_len
        encoder_limit = self._get_config_int_attr(
            config,
            ("max_encoder_input_len", "encoder_max_input_length",
             "max_encoder_position_embeddings",
             "encoder_max_position_embeddings", "max_position_embeddings",
             "n_positions"),
        )
        if encoder_limit is not None:
            max_seq_len = min(max_seq_len, encoder_limit)

        return num_layers, num_kv_heads, head_dim, max_seq_len

    def _get_cross_kv_size_per_token(self) -> int:
        """Estimate bytes/token for the encoder-decoder cross-attention pool."""
        from types import SimpleNamespace

        model_config = self._model_engine.model.model_config
        config = model_config.pretrained_config
        (num_layers, num_kv_heads, head_dim,
         _) = self._get_cross_kv_cache_layout()
        num_attention_heads = self._get_config_int_attr(
            config,
            ("encoder_num_heads", "encoder_attention_heads", "num_heads",
             "num_attention_heads"),
        )
        hidden_size = self._get_config_int_attr(
            config, ("encoder_hidden_size", "d_model", "hidden_size"))
        proxy_model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(
                num_key_value_heads=num_kv_heads,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                head_dim=head_dim,
            ),
            quant_config=model_config.quant_config,
        )
        return self._kv_cache_manager_cls.get_cache_size_per_token(
            proxy_model_config,
            self._mapping,
            tokens_per_block=self._tokens_per_block,
            num_layers=num_layers,
        )

    def _split_kv_cache_budget_for_cross(
        self,
        kv_cache_config: Optional[KvCacheConfig] = None,
    ) -> tuple[KvCacheConfig, KvCacheConfig]:
        """Split enc-dec KV cache budgets between self and cross pools.

        The cross manager must exist for every encoder-decoder runtime. During
        both estimation and final construction, split the same memory-derived
        budget sources used by the legacy TRT path: the free-memory fraction,
        any explicit ``max_gpu_total_bytes`` override, and any explicit host
        cache budget. ``max_tokens`` is a logical cap, not a memory split knob,
        so it is intentionally left unchanged. The creator's base config is not
        mutated.
        """
        base_kv_cache_config = (kv_cache_config if kv_cache_config is not None
                                else self._kv_cache_config)
        fraction = base_kv_cache_config.cross_kv_cache_fraction
        if fraction is None:
            raise ValueError("Encoder-decoder models require "
                             "cross_kv_cache_fraction to size the cross "
                             "KV cache pool.")

        self_kv_cache_config = base_kv_cache_config.model_copy()
        cross_kv_cache_config = base_kv_cache_config.model_copy()
        split_any_budget = False

        free_fraction = base_kv_cache_config.free_gpu_memory_fraction
        if free_fraction is not None:
            cross_fraction = free_fraction * fraction
            self_fraction = free_fraction - cross_fraction
            logger.info(
                "Splitting encoder-decoder free GPU memory fraction: "
                f"total={free_fraction:.3f}, self={self_fraction:.3f}, cross={cross_fraction:.3f}"
            )
            self_kv_cache_config.free_gpu_memory_fraction = self_fraction
            cross_kv_cache_config.free_gpu_memory_fraction = cross_fraction
            split_any_budget = True

        total_budget = base_kv_cache_config.max_gpu_total_bytes
        if total_budget is not None and total_budget > 0:
            cross_budget = int(total_budget * fraction)
            self_budget = total_budget - cross_budget
            logger.info(
                f"Splitting KV cache budget for encoder-decoder: "
                f"total={total_budget / GB:.2f} GiB, "
                f"self={self_budget / GB:.2f} GiB ({1 - fraction:.0%}), "
                f"cross={cross_budget / GB:.2f} GiB ({fraction:.0%})")
            self_kv_cache_config.max_gpu_total_bytes = self_budget
            cross_kv_cache_config.max_gpu_total_bytes = cross_budget
            split_any_budget = True

        host_cache_size = base_kv_cache_config.host_cache_size
        if host_cache_size is not None and host_cache_size > 0:
            cross_host_cache_size = int(host_cache_size * fraction)
            self_host_cache_size = host_cache_size - cross_host_cache_size
            logger.info(
                f"Splitting KV cache host budget for encoder-decoder: "
                f"total={host_cache_size / GB:.2f} GiB, "
                f"self={self_host_cache_size / GB:.2f} GiB ({1 - fraction:.0%}), "
                f"cross={cross_host_cache_size / GB:.2f} GiB ({fraction:.0%})")
            self_kv_cache_config.host_cache_size = self_host_cache_size
            cross_kv_cache_config.host_cache_size = cross_host_cache_size
            split_any_budget = True

        if not split_any_budget:
            raise ValueError("Unable to size the encoder-decoder cross KV "
                             "cache pool: neither free_gpu_memory_fraction nor "
                             "max_gpu_total_bytes nor host_cache_size is "
                             "available.")

        return self_kv_cache_config, cross_kv_cache_config

    def _create_cross_kv_cache_manager(
        self,
        cross_kv_cache_config: KvCacheConfig,
        estimating_kv_cache: bool = False,
        fallback_max_seq_len: Optional[int] = None,
    ) -> KVCacheManager:
        """Create a KV cache manager for the cross-attention pool.

        The cross pool stores encoder K/V projections that are written once
        during the first decoder context step and read on every subsequent
        decoder generation step. It uses ``CacheType.CROSS`` with decoder
        layer count but encoder-side KV geometry.

        The manager class mirrors the self pool (``KVCacheManager`` for V1,
        ``KVCacheManagerV2`` for V2) so that both pools share the same
        runtime ABI and scheduler integration. V1 is the default and the
        production target for encoder-decoder models.
        """
        (num_layers, num_kv_heads, head_dim,
         max_seq_len) = self._get_cross_kv_cache_layout(fallback_max_seq_len)
        estimating_kv_cache = estimating_kv_cache and not self._skip_est
        return _create_kv_cache_manager(
            model_engine=self._model_engine,
            kv_cache_manager_cls=self._kv_cache_manager_cls,
            mapping=self._mapping,
            kv_cache_config=cross_kv_cache_config,
            tokens_per_block=self._tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=self._max_batch_size,
            spec_config=None,
            sparse_attention_config=None,
            max_num_tokens=self._max_num_tokens,
            max_beam_width=1,
            kv_connector_manager=None,
            estimating_kv_cache=estimating_kv_cache,
            execution_stream=self._execution_stream,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.
            CacheType.CROSS,
        )

    def _needs_gpu_kv_cache_budget_split(
        self,
        kv_cache_config: Optional[KvCacheConfig] = None,
    ) -> bool:
        """Whether max_gpu_total_bytes must be split per manager."""
        if self._is_kv_cache_manager_v2:
            return self._should_create_separate_draft_kv_cache()
        kv_cache_config = (kv_cache_config if kv_cache_config is not None else
                           self._kv_cache_config)
        return is_vswa_enabled(kv_cache_config)

    def build_managers(self,
                       resources: Dict,
                       estimating_kv_cache: bool = False) -> None:
        """Construct KV caches for model and draft model (if applicable)."""
        if self._skip_est:
            self.configure_kv_cache_capacity()
        original_max_seq_len = self._max_seq_len

        # For encoder-decoder models, split the self/cross budgets first so
        # every enc-dec build creates a real cross pool.  This must happen
        # before any draft split so that the draft split operates on the
        # already-reduced self-pool budget.
        self_kv_cache_config = self._kv_cache_config
        cross_kv_cache_config = None
        if self._is_encoder_decoder():
            self_kv_cache_config, cross_kv_cache_config = self._split_kv_cache_budget_for_cross(
            )

        # Split combined KV cache budgets before creating managers. Skip during
        # estimation — estimation uses max_tokens-based logic and must not
        # mutate the config.
        has_draft = (
            self._draft_model_engine is not None  # two-model
            or self._should_create_separate_draft_kv_cache())  # one-model
        draft_kv_cache_config = None
        if not estimating_kv_cache and has_draft:
            # Used when each manager sizes pools from max_gpu_total_bytes (V2
            # and V1 VSWA). V1 non-VSWA GPU uses shared max_tokens instead.
            if self._needs_gpu_kv_cache_budget_split(self_kv_cache_config):
                self_kv_cache_config, draft_kv_cache_config = (
                    self._split_kv_cache_budget_for_draft(
                        "max_gpu_total_bytes", self_kv_cache_config,
                        draft_kv_cache_config))
            # KVCacheManagerV2 does not support two-model draft budget splitting.
            v2_two_model = (self._is_kv_cache_manager_v2
                            and self._draft_model_engine is not None)
            if not v2_two_model:
                # Each manager sizes its host pool from host_cache_size directly.
                self_kv_cache_config, draft_kv_cache_config = (
                    self._split_kv_cache_budget_for_draft(
                        "host_cache_size", self_kv_cache_config,
                        draft_kv_cache_config))

        kv_cache_manager = self._create_kv_cache_manager(
            self._model_engine,
            estimating_kv_cache,
            kv_cache_config_override=self_kv_cache_config)

        if (not estimating_kv_cache and self._kv_connector_manager is not None
                and self._draft_model_engine is not None):
            raise NotImplementedError(
                "Connector manager is not supported for draft model.")

        draft_kv_cache_manager = None
        draft_build_kv_cache_config = (draft_kv_cache_config
                                       if draft_kv_cache_config is not None else
                                       self_kv_cache_config)

        # Two-model speculative decoding: draft model has separate engine
        if self._draft_model_engine is not None:
            if self._is_kv_cache_manager_v2:
                assert draft_kv_cache_config is None, (
                    "KVCacheManagerV2 does not support two-model speculative "
                    "decoding with separate draft KV cache budget splitting.")
            draft_kv_cache_manager = self._create_kv_cache_manager(
                self._draft_model_engine,
                estimating_kv_cache,
                kv_cache_config_override=draft_build_kv_cache_config)
        # One-model speculative decoding with different KV layouts
        elif self._should_create_separate_draft_kv_cache():
            draft_kv_cache_manager = self._create_one_model_draft_kv_cache_manager(
                estimating_kv_cache,
                kv_cache_config_override=draft_build_kv_cache_config)

        # Encoder-decoder cross-attention pool
        cross_kv_cache_manager = None
        if cross_kv_cache_config is not None:
            cross_kv_cache_manager = self._create_cross_kv_cache_manager(
                cross_kv_cache_config, estimating_kv_cache,
                original_max_seq_len)

        resources[ResourceManagerType.KV_CACHE_MANAGER] = kv_cache_manager
        resources[
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER] = draft_kv_cache_manager
        resources[
            ResourceManagerType.CROSS_KV_CACHE_MANAGER] = cross_kv_cache_manager

    def teardown_managers(self, resources: Dict) -> None:
        """Clean up KV caches for model, draft model, and cross pool."""
        resources[ResourceManagerType.KV_CACHE_MANAGER].shutdown()
        del resources[ResourceManagerType.KV_CACHE_MANAGER]
        draft_kv_cache_manager = resources[
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER]
        if draft_kv_cache_manager:
            draft_kv_cache_manager.shutdown()
        del resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER]
        cross_kv_cache_manager = resources.get(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        if cross_kv_cache_manager is not None:
            cross_kv_cache_manager.shutdown()
        if ResourceManagerType.CROSS_KV_CACHE_MANAGER in resources:
            del resources[ResourceManagerType.CROSS_KV_CACHE_MANAGER]


def _build_per_layer_num_kv_heads(
    num_key_value_heads: int,
    num_hidden_layers: int,
    spec_config: Optional[SpeculativeConfig] = None,
    draft_config: Optional[ModelConfig] = None,
) -> Union[int, List[int]]:
    """
    Returns:
        An int when all layers share the same num_kv_heads (common case),
        or a list of num_kv_heads (one entry per layer) when target and
        draft models differ.
    """
    if spec_config is None or draft_config is None:
        return num_key_value_heads

    from ..speculative.utils import get_num_spec_layers
    draft_pretrained = draft_config.pretrained_config
    draft_num_kv_heads = getattr(
        draft_pretrained, 'num_key_value_heads',
        getattr(draft_pretrained, 'num_attention_heads', None))

    if draft_num_kv_heads is None or draft_num_kv_heads == num_key_value_heads:
        return num_key_value_heads

    num_spec_layers = get_num_spec_layers(spec_config)
    logger.info(f"Per-layer KV heads for speculative decoding: "
                f"target={num_key_value_heads} x {num_hidden_layers} layers, "
                f"draft={draft_num_kv_heads} x {num_spec_layers} layers, "
                f"total={num_hidden_layers + num_spec_layers} layers")
    return [num_key_value_heads] * num_hidden_layers + [draft_num_kv_heads
                                                        ] * num_spec_layers


def _create_kv_cache_manager(
        model_engine: Optional[PyTorchModelEngine],
        kv_cache_manager_cls,
        mapping: Mapping,
        kv_cache_config: KvCacheConfig,
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
        spec_config: Optional[SpeculativeConfig],
        sparse_attention_config: Optional[SparseAttentionConfig],
        max_num_tokens: int,
        max_beam_width: int,
        kv_connector_manager: Optional[KvCacheConnectorManager],
        estimating_kv_cache: bool = False,
        enable_kv_cache_stats: bool = False,
        execution_stream: Optional[torch.cuda.Stream] = None,
        # Optional overrides for one-model draft case (when model_engine is None)
        model_config: Optional[ModelConfig] = None,
        dtype: Optional[torch.dtype] = None,
        is_draft: Optional[bool] = None,
        layer_mask: Optional[List[bool]] = None,
        num_layers: Optional[int] = None,
        num_kv_heads: Optional[Union[int, List[int]]] = None,
        head_dim: Optional[int] = None,
        kv_cache_type=None,
        is_disagg: bool = False) -> KVCacheManager:
    """
    Returns:
        A KVCacheManager instance for the given model engine or model config
    """
    # Extract config from model_engine or use provided model_config
    if model_config is not None:
        config = model_config.pretrained_config
        quant_config = model_config.quant_config
        _model_config = model_config
    else:
        config = model_engine.model.model_config.pretrained_config
        quant_config = model_engine.model.model_config.quant_config
        _model_config = model_engine.model.model_config

    if dtype is None:
        dtype = model_engine.dtype

    if is_draft is None:
        is_draft = model_engine.is_draft_model

    if kv_cache_type is None:
        kv_cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF

    # Inkling: the KV cache is sized from the text tower. The top-level
    # inkling_mm_model config carries the decoder geometry in ``text_config``, so
    # route the whole sizing path through it (hidden_size / num_attention_heads /
    # head_dim / num_hidden_layers / vocab_size all live there).
    _is_inkling = is_inkling(config)
    if _is_inkling:
        config = getattr(config, "text_config", config)

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = num_kv_heads if num_kv_heads is not None else getattr(
        config, 'num_key_value_heads', num_attention_heads)
    if _is_inkling:
        # Hybrid per-layer geometry: local (sliding-window) layers use 16 KV
        # heads, global layers 8; head_dim is uniform (128). V2 divides each by
        # tp_size and allocates the paged pool per layer accordingly (the generic
        # V2 branch below consumes this list directly).
        num_key_value_heads = config.num_kv_heads_per_layer()
    if not isinstance(head_dim, int):
        head_dim = getattr(config, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = hidden_size // num_attention_heads

    # Gemma4: build per-layer head_dim, num_kv_heads, and sliding window
    # for hybrid attention. Different layer types need different KV cache
    # pool groups (via max_attention_window) so FlashInfer page indices
    # are consistent within each group.
    if is_gemma4_hybrid(config):
        layer_types = config.layer_types
        global_head_dim = config.global_head_dim
        attention_k_eq_v = getattr(config, 'attention_k_eq_v', False)
        num_global_kv_heads = (getattr(config, 'num_global_key_value_heads',
                                       None) or num_key_value_heads)
        sliding_window = getattr(config, 'sliding_window', None)
        head_dim_list = []
        kv_heads_list = []
        for lt in layer_types:
            is_sliding = (lt == "sliding_attention")
            if is_sliding:
                head_dim_list.append(head_dim)
                kv_heads_list.append(num_key_value_heads)
            else:
                head_dim_list.append(global_head_dim)
                use_k_eq_v = attention_k_eq_v and not is_sliding
                kv_heads_list.append(
                    num_global_kv_heads if use_k_eq_v else num_key_value_heads)
        head_dim = head_dim_list
        num_key_value_heads = kv_heads_list

        # Set per-layer max_attention_window so V2 creates separate pool
        # groups for sliding vs full attention layers (different page sizes).
        # Sliding layers use the model's sliding_window; full layers use
        # max_seq_len.  V2 uses this to evict old blocks when kv_len >
        # window, saving memory (only ~ceil(sliding_window/page_size)
        # blocks per sequence for sliding layers, vs the full kv_len for
        # full attention layers).  FlashInfer's prepare() reads the
        # currently-allocated block IDs per pool from the V2 manager,
        # so the smaller sliding-pool block count after eviction is
        # picked up automatically.
        if (kv_cache_config.max_attention_window is None
                and sliding_window is not None):
            kv_cache_config = copy.copy(kv_cache_config)
            kv_cache_config.max_attention_window = [
                int(sliding_window)
                if lt == "sliding_attention" else int(max_seq_len)
                for lt in layer_types
            ]

    # Note: Gemma4 KV sharing is handled at the model level — shared layers
    # use cache_layer_idx to read from the target layer's cache slot via
    # Gemma4Attention. No layer_mask exclusion needed here.

    if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    elif quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache(
    ):
        kv_cache_dtype = tensorrt_llm.bindings.DataType.NVFP4
    else:
        kv_cache_dtype = str_dtype_to_binding(torch_dtype_to_str(dtype))

    # Use provided num_layers if available, otherwise use config.
    # When layer_mask is set (e.g., KV sharing), num_layers for the cache
    # manager must equal the number of enabled (True) layers in the mask.
    if num_layers is not None:
        num_hidden_layers = num_layers
    elif layer_mask is not None:
        num_hidden_layers = sum(layer_mask)
    else:
        num_hidden_layers = config.num_hidden_layers
    # Only include draft KV heads in the per-layer list when draft layers
    # are NOT handled by a separate draft KV cache manager.  When layer_mask
    # is provided from the caller, it means the main KV cache covers only
    # the masked (target) layers and draft layers live in their own manager.
    draft_config_for_kv = None
    if layer_mask is None:
        draft_config_for_kv = (getattr(model_engine.model, 'draft_config', None)
                               if model_engine is not None else None)
    # If num_key_value_heads is already a per-layer list (e.g., Gemma4 hybrid),
    # use it directly; otherwise build from the scalar value.
    if isinstance(num_key_value_heads, list):
        per_layer_num_kv_heads = num_key_value_heads
    else:
        per_layer_num_kv_heads = _build_per_layer_num_kv_heads(
            num_key_value_heads, num_hidden_layers, spec_config,
            draft_config_for_kv)
    manager_extra_kwargs = {}
    if issubclass(kv_cache_manager_cls, KVCacheManagerV2):
        manager_extra_kwargs["enable_stats"] = enable_kv_cache_stats

    if is_mla(config):
        kv_cache_manager = kv_cache_manager_cls(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=num_hidden_layers,
            num_kv_heads=1,
            head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
            spec_config=spec_config,
            vocab_size=config.vocab_size,
            max_num_tokens=max_num_tokens,
            max_beam_width=max_beam_width,
            is_draft=is_draft,
            kv_connector_manager=kv_connector_manager
            if not estimating_kv_cache else None,
            sparse_attention_config=sparse_attention_config,
            pretrained_config=config,
            is_estimating_kv_cache=estimating_kv_cache,
            execution_stream=execution_stream,
            layer_mask=layer_mask,
            is_disagg=is_disagg,
            **manager_extra_kwargs,
        )
    elif is_nemotron_hybrid(config):
        if max_beam_width > 1:
            raise ValueError(
                "MambaHybridCacheManager + beam search is not supported yet.")

        if not estimating_kv_cache and kv_connector_manager is not None:
            raise NotImplementedError(
                "Connector manager is not supported for MambaHybridCacheManager."
            )

        mamba_params = extract_mamba_kv_cache_params(
            config,
            layer_mask=layer_mask,
            spec_config=spec_config,
            quant_config=quant_config,
        )

        # Replay state update kernel for MTP: default on for sm >= 80; gates
        # below disable it for incompatible feature combinations.  Cpp cache
        # manager doesn't expose use_replay_state_update, so the wrapper
        # property's getattr default keeps replay off there automatically.
        sm = get_sm_version()
        stochastic_rounding = getattr(
            quant_config, 'mamba_ssm_stochastic_rounding',
            False) if quant_config is not None else False

        use_replay = spec_config is not None and sm >= 80
        if spec_config is None:
            logger.info(
                "Replay kernel requires speculative decoding; using non-replay path"
            )

        # Block reuse (prefix caching): replay leaves SSM state at a
        # checkpoint after speculation. The next decode step replays forward
        # to correct it. If block reuse feeds that stale state into a new
        # prefill, the correction never happens.
        # Currently we only save and reuse context tokens so this does not affect.

        # Tree attention: replay assumes linear token sequence.
        if (spec_config is not None
                and (getattr(spec_config, 'eagle_choices', None) is not None
                     or getattr(spec_config, 'use_dynamic_tree', False))):
            logger.info("Replay kernel incompatible with tree attention; "
                        "using legacy MTP path")
            use_replay = False

        # Replay Philox uses PTX cvt.rs.f16x2.f32 which needs 100 <= sm < 120.
        # Flashinfer has a SW fallback at any SM.
        if (stochastic_rounding
                and mamba_params.mamba_ssm_cache_dtype == torch.float16
                and (sm < 100 or sm in (120, 121))):
            logger.info("Replay kernel Philox requires 100 <= sm < 120; "
                        "using legacy MTP path for stochastic rounding support")
            use_replay = False

        # Use replay algorithm for mamba (default is on).
        enforce_disable_replay = os.environ.get('TRTLLM_USE_MAMBA_REPLAY',
                                                '1') == '0'
        if enforce_disable_replay:
            logger.info(
                "Replay kernel is disabled by TRTLLM_USE_MAMBA_REPLAY=0")
            use_replay = False
        else:
            logger.info(
                "Replay kernel is not changed since TRTLLM_USE_MAMBA_REPLAY=1")

        # Stochastic-rounding seeds must live on the cache manager (not be
        # re-created with torch.randint per forward) whenever SR can fire
        # on the fp16 SSM cache.  This mirrors the predicate the mixer uses
        # internally (`_stochastic_rounding_for_flashinfer` /
        # `_stochastic_rounding_for_replay`) so allocation matches consumption.
        mamba_ssm_stochastic_rounding = (stochastic_rounding
                                         and mamba_params.mamba_ssm_cache_dtype
                                         == torch.float16)
        kv_cache_manager = kv_cache_manager_cls(
            # mamba cache parameters
            mamba_params.state_size,
            mamba_params.conv_kernel,
            mamba_params.num_heads,
            mamba_params.n_groups,
            mamba_params.head_dim,
            mamba_params.num_mamba_layers,
            mamba_params.mamba_layer_mask,
            mamba_params.dtype,
            mamba_params.mamba_ssm_cache_dtype,
            # kv cache parameters
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=mamba_params.num_full_attention_layers,
            layer_mask=mamba_params.full_attention_layer_mask,
            num_kv_heads=per_layer_num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            is_draft=is_draft,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
            spec_config=spec_config,
            is_estimating_kv_cache=estimating_kv_cache,
            execution_stream=execution_stream,
            model_type="nemotron_hybrid",
            use_replay_state_update=use_replay,
            mamba_ssm_stochastic_rounding=mamba_ssm_stochastic_rounding,
            **manager_extra_kwargs,
        )
    elif is_qwen3_hybrid(config):
        if max_beam_width > 1:
            raise ValueError(
                "MambaHybridCacheManager + beam search is not supported yet.")

        if not estimating_kv_cache and kv_connector_manager is not None:
            raise NotImplementedError(
                "Connector manager is not supported for MambaHybridCacheManager."
            )
        mamba_params = extract_mamba_kv_cache_params(
            config,
            layer_mask=layer_mask,
            spec_config=spec_config,
            quant_config=quant_config,
        )
        kv_cache_manager = kv_cache_manager_cls(
            # mamba cache parameters
            mamba_params.state_size,
            mamba_params.conv_kernel,
            mamba_params.num_heads,
            mamba_params.n_groups,
            mamba_params.head_dim,
            mamba_params.num_mamba_layers,
            mamba_params.mamba_layer_mask,
            mamba_params.dtype,
            mamba_params.mamba_ssm_cache_dtype,
            # kv cache parameters
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=mamba_params.num_full_attention_layers,
            layer_mask=mamba_params.full_attention_layer_mask,
            num_kv_heads=per_layer_num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            is_draft=is_draft,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
            spec_config=spec_config,
            is_estimating_kv_cache=estimating_kv_cache,
            execution_stream=execution_stream,
            model_type="qwen3_next",
            **manager_extra_kwargs,
        )
    else:
        # NOTE: this is a workaround for VSWA to switch to calculate_max_num_blocks_for_vswa in KVCahceManager
        # Only needed for V1; V2 handles per-layer windows natively via life cycles.
        is_vswa = is_vswa_enabled(kv_cache_config)
        binding_model_config = None
        if is_vswa and kv_cache_manager_cls.__name__ == "KVCacheManager":
            binding_model_config = _model_config.get_bindings_model_config(
                tokens_per_block=tokens_per_block,
                kv_cache_config=kv_cache_config,
                spec_config=spec_config)

        # KVCacheManager (V1) doesn't support per-layer head_dim lists;
        # use max for estimation. KVCacheManagerV2 handles lists natively.
        effective_head_dim = (
            max(head_dim) if isinstance(head_dim, list)
            and kv_cache_manager_cls.__name__ == "KVCacheManager" else head_dim)
        kv_cache_manager = kv_cache_manager_cls(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_hidden_layers,
            num_kv_heads=per_layer_num_kv_heads,
            head_dim=effective_head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
            spec_config=spec_config,
            vocab_size=config.vocab_size,
            max_num_tokens=max_num_tokens,
            model_config=binding_model_config,
            max_beam_width=max_beam_width,
            is_draft=is_draft,
            kv_connector_manager=kv_connector_manager
            if not estimating_kv_cache else None,
            sparse_attention_config=sparse_attention_config,
            pretrained_config=config,
            is_estimating_kv_cache=estimating_kv_cache,
            execution_stream=execution_stream,
            layer_mask=layer_mask,
            is_disagg=is_disagg,
            **manager_extra_kwargs,
        )
    # Note: Gemma4 KV sharing cache remapping is handled in Gemma4Attention
    # via cache_layer_idx — shared layers use target layer's index for
    # get_buffers(). No layer_offsets remapping needed here.

    return kv_cache_manager


def create_kv_cache_compression_manager(
    config: KvCacheCompressionConfig,
    kv_cache_manager: KVCacheManagerV2,
) -> Optional[BaseKVCacheCompressionManager]:
    """Build the KV-cache compression manager for ``config.algorithm``, or return
    None if no algorithm matches.

    Called from ``create_py_executor`` and registered as a resource manager,
    like the KV cache manager itself. Concrete algorithms add a dispatch branch
    here; the framework ships none.
    """
    logger.warning(
        "KV-cache compression algorithm '%s' is not registered; running without "
        "a compression manager.",
        config.algorithm,
    )
    return None


def compute_max_num_sequences(mapping: Mapping,
                              max_batch_size: int,
                              disable_overlap_scheduler: bool,
                              enable_overlap_headroom: bool = False) -> int:
    """Size the sequence-slot pool (and the sampler state it indexes).

    ``enable_overlap_headroom`` is intentionally opt-in. DeepSeek-V4 needs a
    second non-PP slot set because the V2 scheduler can backfill seats before
    the overlap scheduler releases the previous iteration's terminal slots.
    Other models retain their established sizing until that behavior is
    validated independently. Pipeline parallelism already sizes the pool by
    ``pp_size``.
    """
    if mapping.has_pp():
        num_micro_batches = mapping.pp_size
    else:
        num_micro_batches = (2 if enable_overlap_headroom
                             and not disable_overlap_scheduler else 1)
    return max_batch_size * num_micro_batches


def should_enable_dsv4_adp_dummy_fixes(model_type: Optional[str],
                                       mapping: Mapping) -> bool:
    """Gate DSv4 ADP dummy behavior while PP remains follow-up scope."""
    return model_type == "deepseek_v4" and not mapping.has_pp()


def should_enable_dsv4_overlap_headroom(
        model_type: Optional[str], spec_config: Optional[SpeculativeConfig],
        mapping: Mapping, disable_overlap_scheduler: bool) -> bool:
    """Gate extra sequence slots to the validated DSv4 MTP overlap path."""
    return (should_enable_dsv4_adp_dummy_fixes(model_type, mapping)
            and spec_config is not None
            and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
            and not disable_overlap_scheduler)


def create_py_executor_instance(
    *,
    dist,
    resources,
    mapping,
    llm_args,
    ctx_chunk_config,
    model_engine,
    start_worker,
    sampler,
    drafter,
    guided_decoder: Optional[GuidedDecoder] = None,
    lora_config: Optional[LoraConfig] = None,
    garbage_collection_gen0_threshold: Optional[int] = None,
    kv_connector_manager: Optional[KvCacheConnectorManager] = None,
    resource_governor_queue=None,
    max_seq_len: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_beam_width: Optional[int] = None,
    max_num_tokens: Optional[int] = None,
    peft_cache_config: Optional[PeftCacheConfig] = None,
    scheduler_config: Optional[SchedulerConfig] = None,
    cache_transceiver_config: Optional[CacheTransceiverConfig] = None,
    virtual_memory_pools: Optional[dict] = None,
    execution_stream: Optional[torch.cuda.Stream] = None,
    dwdp_manager: Optional[DwdpManager] = None,
    max_num_sequences: Optional[int] = None,
) -> PyExecutor:
    set_low_latency_dispatch(
        getattr(llm_args, 'enable_low_latency_host_dispatch', False))

    kv_cache_manager = resources.get(ResourceManagerType.KV_CACHE_MANAGER, None)

    spec_config = model_engine.spec_config

    if max_num_sequences is None:
        max_num_sequences = compute_max_num_sequences(
            mapping, max_batch_size, llm_args.disable_overlap_scheduler)

    logger.info(
        f"max_seq_len={max_seq_len}, max_num_requests={max_num_sequences}, max_num_tokens={max_num_tokens}, max_batch_size={max_batch_size}"
    )
    is_disagg = (cache_transceiver_config is not None
                 and cache_transceiver_config.backend is not None)
    for key, value in llm_args.extra_resource_managers.items():
        if key in resources:
            raise ValueError(
                f"Cannot overwrite existing resource manager {key}.")
        resources[key] = value

    peft_cache_manager = None
    if lora_config is not None:
        # TODO: Refactor dimension resolution into a LoraModuleDimensions
        # dataclass to avoid ad-hoc getattr + TP-division blocks per model type.
        from tensorrt_llm.bindings import LoraModule

        if len(lora_config.lora_dir) == 1:
            # Route to appropriate loader based on checkpoint source
            load_torch_lora(lora_config)
        else:
            assert len(lora_config.lora_target_modules
                       ) >= 1, "Expecting at least one lora target module"
            if not bool(lora_config.trtllm_modules_to_hf_modules):
                lora_config.trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules(
                )

        model_binding_config = model_engine.model.model_config.get_bindings_model_config(
            is_disagg=is_disagg)

        num_experts = _try_infer_num_experts(model_engine.model.model_config)

        num_kv_attention_heads_per_layer = model_binding_config.num_kv_heads_per_layer
        if max(num_kv_attention_heads_per_layer) != min(
                num_kv_attention_heads_per_layer):
            logger.warning(
                "Defining LORA with per-layer KV heads is not supported for LORA, using the max number of KV heads per layer"
            )
            num_kv_attention_heads = max(num_kv_attention_heads_per_layer)
        else:
            # all layers have the same number of KV heads
            num_kv_attention_heads = num_kv_attention_heads_per_layer[0]

        pretrained_config = model_engine.model.model_config.pretrained_config

        # Derive shared expert intermediate size from the LoRA adapter
        # weights, which are the source of truth for dimension validation.
        # The model config's shared_expert_intermediate_size may not match
        # the adapter (e.g., upcycled models).
        shared_expert_hidden_size = 0
        if lora_config.lora_dir:
            shared_expert_global = _infer_shared_expert_size_from_adapter(
                lora_config.lora_dir[0])
            if shared_expert_global > 0:
                shared_expert_hidden_size = shared_expert_global // mapping.tp_size

        moe_hidden_size = 0
        moe_intermediate = getattr(pretrained_config, 'moe_intermediate_size',
                                   None)
        if moe_intermediate is not None and moe_intermediate > 0:
            moe_hidden_size = moe_intermediate // mapping.tp_size

        # Mamba dimensions for hybrid models (e.g., Nemotron-H)
        # d_inner = mamba_head_dim * mamba_num_heads
        # d_in_proj = 2 * d_inner + 2 * n_groups * d_state + mamba_num_heads
        mamba_in_proj_size = 0
        mamba_inner_size = 0
        mamba_head_dim = getattr(pretrained_config, 'mamba_head_dim', 0)
        mamba_num_heads = getattr(pretrained_config, 'mamba_num_heads', 0)
        if mamba_head_dim > 0 and mamba_num_heads > 0:
            d_inner = mamba_head_dim * mamba_num_heads
            mamba_inner_size = d_inner // mapping.tp_size
            n_groups = getattr(pretrained_config, 'n_groups', 1)
            d_state = getattr(pretrained_config, 'ssm_state_size', 128)
            d_in_proj = 2 * d_inner + 2 * n_groups * d_state + mamba_num_heads
            mamba_in_proj_size = d_in_proj // mapping.tp_size

        # MoE latent size for latent MoE models (e.g., Nemotron-H SuperV3).
        # Latent projections are replicated (not TP-sharded), so pass the
        # raw config value without dividing by tp_size.
        moe_latent_size = getattr(pretrained_config, 'moe_latent_size', 0) or 0

        # For MoE models with shared experts: replace mlp_* target modules with
        # shared_expert_* equivalents. The shared expert uses different LoRA
        # module types with their own intermediate size. For pure MoE models
        # (no dense MLP layers), mlp_* modules don't exist in the model.
        target_modules = list(lora_config.lora_target_modules)
        if shared_expert_hidden_size > 0:
            has_dense_mlp = bool(
                getattr(pretrained_config, 'mlp_only_layers', None))
            mlp_to_shared_expert = {
                'mlp_h_to_4h': 'shared_expert_h_to_4h',
                'mlp_4h_to_h': 'shared_expert_4h_to_h',
                'mlp_gate': 'shared_expert_gate',
            }
            for mlp_mod, se_mod in mlp_to_shared_expert.items():
                if mlp_mod in target_modules:
                    if se_mod not in target_modules:
                        target_modules.append(se_mod)
                    if not has_dense_mlp:
                        target_modules.remove(mlp_mod)

        lora_modules = LoraModule.create_lora_modules(
            lora_module_names=target_modules,
            hidden_size=model_binding_config.hidden_size,
            mlp_hidden_size=model_binding_config.mlp_hidden_size,
            num_attention_heads=model_binding_config.num_heads,
            num_kv_attention_heads=num_kv_attention_heads,
            attention_head_size=model_binding_config.head_size,
            tp_size=mapping.tp_size,
            num_experts=num_experts,
            shared_expert_hidden_size=shared_expert_hidden_size,
            moe_hidden_size=moe_hidden_size,
            mamba_in_proj_size=mamba_in_proj_size,
            mamba_inner_size=mamba_inner_size,
            moe_latent_size=moe_latent_size)

        model_binding_config.use_lora_plugin = True
        model_binding_config.lora_modules = lora_modules
        model_binding_config.max_lora_rank = lora_config.max_lora_rank

        max_lora_rank = lora_config.max_lora_rank
        num_lora_modules = _compute_num_lora_modules(
            pretrained_config,
            target_modules + lora_config.missing_qkv_modules,
        )

        peft_cache_config_model = PeftCacheConfig(
        ) if peft_cache_config is None else peft_cache_config
        if lora_config.max_loras is not None:
            peft_cache_config_model.num_device_module_layer = \
                max_lora_rank * num_lora_modules * lora_config.max_loras
        if lora_config.max_cpu_loras is not None:
            peft_cache_config_model.num_host_module_layer = \
                max_lora_rank * num_lora_modules * lora_config.max_cpu_loras

        from tensorrt_llm.bindings import WorldConfig
        world_config = WorldConfig(
            tensor_parallelism=mapping.tp_size,
            pipeline_parallelism=mapping.pp_size,
            context_parallelism=mapping.cp_size,
            rank=dist.mapping.rank,
            gpus_per_node=dist.mapping.gpus_per_node,
        )
        peft_cache_manager = PeftCacheManager(
            peft_cache_config=peft_cache_config_model,
            lora_config=lora_config,
            model_config=model_binding_config,
            world_config=world_config,
            execution_stream=execution_stream,
            lora_target_modules=target_modules,
        )
        resources[ResourceManagerType.PEFT_CACHE_MANAGER] = peft_cache_manager
        model_engine.set_lora_model_config(
            target_modules, lora_config.trtllm_modules_to_hf_modules,
            lora_config.swap_gate_up_proj_lora_b_weight)
        if isinstance(model_engine, PyTorchModelEngine):
            model_engine._init_cuda_graph_lora_manager(lora_config)

    resources[ResourceManagerType.SEQ_SLOT_MANAGER] = SeqSlotManager(
        max_num_sequences)

    # Register the compression manager (if one is configured) with the other
    # managers, before building ResourceManager, so it is part of the manager
    # set from the start. Reads its own config, not the sparse-attention one.
    kv_cache_compression_config = getattr(llm_args,
                                          "kv_cache_compression_config", None)
    if kv_cache_compression_config is not None:
        compression_manager = create_kv_cache_compression_manager(
            kv_cache_compression_config, kv_cache_manager)
        if compression_manager is not None:
            resources[ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER] = (
                compression_manager)

    resource_manager = ResourceManager(resources)

    # KV cache manager runs last (others may depend on it), except the
    # compression manager which reconciles after it (below).
    if kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end(
            ResourceManagerType.KV_CACHE_MANAGER, last=True)
    # Compression manager runs after the cache manager: reconciles history once it's resized.
    if (ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER
            in resource_manager.resource_managers):
        resource_manager.resource_managers.move_to_end(
            ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER, last=True)

    cross_kv_cache_manager = resources.get(
        ResourceManagerType.CROSS_KV_CACHE_MANAGER)
    if cross_kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER, last=True)

    # When scheduler_capacity == 1, attention dp dummy request will prevent the scheduling of DISAGG_GENERATION_INIT.
    # Enlarge scheduler capacity to avoid DISAGG_GENERATION_INIT stuck in the scheduler.
    # V1 scheduler handles overlap via two_step_lookahead, so skip the
    # slot-pool overlap factor here.
    scheduler_capacity = max_batch_size * mapping.pp_size
    if scheduler_capacity == 1 and mapping.enable_attention_dp and kv_cache_manager:
        scheduler_capacity += 1

    # For encoder-decoder models, requests start in ENCODER_INIT and the
    # capacity scheduler must admit them already at that state so the
    # encoder loop can run. Decoder-only deployments keep the default
    # CONTEXT_INIT gating.
    no_schedule_until_state = (LlmRequestState.ENCODER_INIT
                               if cross_kv_cache_manager is not None else
                               LlmRequestState.CONTEXT_INIT)

    # V2 scheduler uses scheduler_capacity as the per-iteration request
    # budget (BudgetTracker.max_num_requests).  Unlike V1 which has a
    # separate CapacityScheduler (needs pp_size * max_batch_size to hold
    # requests across PP stages) and MicroBatchScheduler (uses
    # max_batch_size for per-forward batch limit), V2 merges both into
    # one loop.  PP on-the-fly is handled by inflight_request_ids
    # filtering, so its budget should be based on max_batch_size, not
    # max_num_sequences (which includes the pp_size multiplier).
    v2_scheduler_capacity = max_batch_size
    if v2_scheduler_capacity == 1 and mapping.enable_attention_dp and kv_cache_manager:
        v2_scheduler_capacity += 1

    if isinstance(kv_cache_manager, KVCacheManagerV2):
        # V2: interleaved scheduler handles both capacity and budget
        draft_kv_cache_manager = resources.get(
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER)
        scheduler_policy = (scheduler_config.capacity_scheduler_policy
                            if scheduler_config is not None else
                            CapacitySchedulerPolicy.MAX_UTILIZATION)
        enable_prefix_aware_scheduling = (
            scheduler_config.enable_prefix_aware_scheduling
            if scheduler_config is not None else True)
        scheduler = KVCacheV2Scheduler(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=kv_cache_manager,
            scheduler_policy=scheduler_policy,
            ctx_chunk_config=ctx_chunk_config,
            peft_cache_manager=peft_cache_manager.impl
            if peft_cache_manager is not None else None,
            scheduler_capacity=v2_scheduler_capacity,
            draft_kv_cache_manager=draft_kv_cache_manager,
            cross_kv_cache_manager=cross_kv_cache_manager,
            no_schedule_until_state=no_schedule_until_state,
            enable_prefix_aware_scheduling=enable_prefix_aware_scheduling,
        )
    elif (scheduler_config is not None
          and scheduler_config.use_python_scheduler):
        enable_prefix_aware_scheduling = scheduler_config.enable_prefix_aware_scheduling
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=kv_cache_manager.impl
            if kv_cache_manager is not None else None,
            peft_cache_manager=peft_cache_manager.impl
            if peft_cache_manager is not None else None,
            scheduler_policy=scheduler_config.capacity_scheduler_policy,
            ctx_chunk_config=ctx_chunk_config,
            cross_kv_cache_manager=cross_kv_cache_manager.impl
            if cross_kv_cache_manager is not None else None,
            two_step_lookahead=mapping.has_pp(),
            scheduler_capacity=scheduler_capacity,
            no_schedule_until_state=no_schedule_until_state,
            enable_prefix_aware_scheduling=enable_prefix_aware_scheduling,
        )
    else:
        enable_prefix_aware_scheduling = (
            scheduler_config.enable_prefix_aware_scheduling
            if scheduler_config is not None else True)
        capacity_scheduler = BindCapacityScheduler(
            scheduler_capacity,
            kv_cache_manager.impl if kv_cache_manager is not None else None,
            peft_cache_manager.impl if peft_cache_manager is not None else None,
            scheduler_config.capacity_scheduler_policy,
            cross_kv_cache_manager=cross_kv_cache_manager.impl
            if cross_kv_cache_manager is not None else None,
            two_step_lookahead=mapping.has_pp(),
            no_schedule_until_state=no_schedule_until_state,
            enable_prefix_aware_scheduling=enable_prefix_aware_scheduling,
        )

        mb_scheduler = BindMicroBatchScheduler(max_batch_size, max_num_tokens,
                                               ctx_chunk_config)

        reorder_policy_config = llm_args.reorder_policy_config
        if reorder_policy_config is not None:
            assert reorder_policy_config.policy_name == "AgentTree", "Reorder policy only supports AgentTree for now"
            capacity_scheduler.impl.set_agent_tree_reorder_policy(
                reorder_policy_config.policy_args.agent_percentage,
                reorder_policy_config.policy_args.agent_types,
                reorder_policy_config.policy_args.agent_inflight_seq_num)
        scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)

    config = model_engine.model.model_config.pretrained_config
    attention_type = AttentionTypeCpp.MLA if is_mla(
        config) else AttentionTypeCpp.DEFAULT

    # For hybrid models, this has both impl and mamba_impl
    mamba_cache_manager = None
    if isinstance(kv_cache_manager, BaseMambaCacheManager):
        mamba_cache_manager = kv_cache_manager

    kv_cache_transceiver = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager, attention_type,
        cache_transceiver_config, mamba_cache_manager)

    waiting_queue_policy = (scheduler_config.waiting_queue_policy
                            if scheduler_config is not None else
                            WaitingQueuePolicy.FCFS)

    return PyExecutor(
        resource_manager,
        scheduler,
        model_engine=model_engine,
        sampler=sampler,
        drafter=drafter,
        dist=dist,
        max_num_sequences=max_num_sequences,
        disable_overlap_scheduler=llm_args.disable_overlap_scheduler,
        enable_early_first_token_response=llm_args.
        enable_early_first_token_response,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        max_draft_len=spec_config.max_draft_len
        if spec_config is not None else 0,
        max_total_draft_tokens=(spec_config.tokens_per_gen_step -
                                1) if spec_config is not None else 0,
        kv_cache_transceiver=kv_cache_transceiver,
        guided_decoder=guided_decoder,
        start_worker=start_worker,
        garbage_collection_gen0_threshold=garbage_collection_gen0_threshold,
        kv_connector_manager=kv_connector_manager,
        resource_governor_queue=resource_governor_queue,
        max_seq_len=max_seq_len,
        peft_cache_config=peft_cache_config,
        virtual_memory_pools=virtual_memory_pools,
        execution_stream=execution_stream,
        waiting_queue_policy=waiting_queue_policy,
        dwdp_manager=dwdp_manager,
        enable_kv_pool_rebalance=llm_args.kv_cache_config.
        enable_kv_pool_rebalance,
    )


def create_torch_sampler_args(
    mapping: Mapping,
    *,
    max_seq_len: int,
    max_batch_size: int,
    speculative_config: SpeculativeConfig,
    max_beam_width: int,
    disable_overlap_scheduler: bool,
    enable_async_worker: bool,
    enable_speculative_beam_history_d2h: bool,
    max_num_sequences: Optional[int] = None,
):
    # The sampler's per-slot state is indexed by sequence slots, so it must
    # be sized identically to the executor's slot pool.
    if max_num_sequences is None:
        max_num_sequences = compute_max_num_sequences(
            mapping, max_batch_size, disable_overlap_scheduler)
    max_draft_len = (0 if speculative_config is None else
                     speculative_config.max_draft_len)
    max_total_draft_tokens = (0 if speculative_config is None else
                              speculative_config.tokens_per_gen_step - 1)

    return TorchSampler.Args(
        max_seq_len=max_seq_len,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_num_sequences=max_num_sequences,
        max_beam_width=max_beam_width,
        disable_overlap_scheduler=disable_overlap_scheduler,
        enable_async_worker=enable_async_worker,
        enable_speculative_beam_history_d2h=enable_speculative_beam_history_d2h,
    )


def instantiate_sampler(
    engine: PyTorchModelEngine,
    llm_args: TorchLlmArgs,
    mapping: Mapping,
    *,
    max_batch_size: int,
    max_beam_width: int,
    max_seq_len: int,
    mm_encoder_only: bool,
    speculative_config: SpeculativeConfig,
    decoding_config: trtllm.DecodingConfig,
    kv_cache_config: KvCacheConfig,
    max_num_sequences: Optional[int] = None,
):
    enable_async_worker = (confidential_compute_enabled()
                           or llm_args.sampler_force_async_worker)

    sampler_args = create_torch_sampler_args(
        mapping,
        max_seq_len=engine.max_seq_len,
        max_batch_size=max_batch_size,
        speculative_config=speculative_config,
        max_beam_width=max_beam_width,
        disable_overlap_scheduler=llm_args.disable_overlap_scheduler,
        enable_async_worker=enable_async_worker,
        enable_speculative_beam_history_d2h=llm_args.
        enable_speculative_beam_history_d2h,
        max_num_sequences=max_num_sequences,
    )
    decoding_mode = get_decoding_mode(decoding_config=decoding_config,
                                      max_beam_width=max_beam_width)
    if mapping.cp_config.get('cp_type') == CpType.STAR:
        assert llm_args.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        return TorchSampler(sampler_args)
    if engine.spec_config is not None and engine.spec_config.spec_dec_mode.has_spec_decoder(
    ):
        return get_spec_decoder(sampler_args, engine.spec_config)

    if mm_encoder_only:
        # NOTE: handle model outputs specially for mm encoder executor/engine
        return EarlyStopWithMMResult()
    if llm_args.sampler_type == SamplerType.TRTLLMSampler:
        logger.warning(
            "TRTLLMSampler is deprecated and will be removed in release 1.4. Please use TorchSampler instead."
        )
        logger.debug(f"DecodingMode: {decoding_mode.name}")
        return TRTLLMSampler(engine.model,
                             engine.dtype,
                             mapping,
                             decoding_mode,
                             llm_args.disable_overlap_scheduler,
                             max_seq_len=max_seq_len,
                             max_batch_size=max_batch_size,
                             max_beam_width=max_beam_width,
                             decoding_config=decoding_config,
                             kv_cache_config=kv_cache_config,
                             enable_async_worker=enable_async_worker,
                             max_num_sequences=max_num_sequences)
    if not engine.model.model_config.is_generation:
        # NOTE: choose sampler based on model type
        return EarlyStopSampler()
    return TorchSampler(sampler_args)


def get_decoding_mode(
    decoding_config: trtllm.DecodingConfig,
    max_beam_width: int,
) -> DecodingMode:
    '''This implementation is based off trtGptModelInflightBatching.cpp getDecodingMode().'''
    if decoding_config and decoding_config.decoding_mode and not decoding_config.decoding_mode.isAuto(
    ):
        decoding_mode = decoding_config.decoding_mode
    elif max_beam_width == 1:
        decoding_mode = DecodingMode.TopKTopP()
    else:
        decoding_mode = DecodingMode.BeamSearch()

    # Override decoding mode when beam width is one
    if max_beam_width == 1 and decoding_mode.isBeamSearch():
        logger.warning(
            "Beam width is set to 1, but decoding mode is BeamSearch. Overwriting decoding mode to TopKTopP."
        )
        decoding_mode = DecodingMode.TopKTopP()

    return decoding_mode


_ATTN_MODULES = frozenset({
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_dense",
    "cross_attn_q",
    "cross_attn_k",
    "cross_attn_v",
})
_MLP_MODULES = frozenset({
    "mlp_h_to_4h",
    "mlp_4h_to_h",
    "mlp_gate",
    "mlp_gate_up",
})


def _compute_num_lora_modules(pretrained_config,
                              all_target_modules: list[str]) -> int:
    """Compute the total number of LoRA module-layer slots for cache sizing.

    For models with per-layer block_configs (e.g. Nemotron-NAS / DeciLM),
    layers with no_op or replace_with_linear attention/FFN cannot host LoRA
    adapters, so they are excluded from the count.  For all other models,
    falls back to the uniform num_hidden_layers x len(target_modules).
    """
    num_layers = pretrained_config.num_hidden_layers
    block_configs = getattr(pretrained_config, "block_configs", None)

    if block_configs is None:
        return num_layers * len(all_target_modules)

    attn_modules = [m for m in all_target_modules if m in _ATTN_MODULES]
    mlp_modules = [m for m in all_target_modules if m in _MLP_MODULES]
    other_modules = [
        m for m in all_target_modules
        if m not in _ATTN_MODULES and m not in _MLP_MODULES
    ]

    def _has_lora_capable_attn(bc):
        return not bc.attention.no_op and not bc.attention.replace_with_linear

    def _has_lora_capable_ffn(bc):
        return not bc.ffn.no_op and not bc.ffn.replace_with_linear

    layers_with_attn = sum(1 for bc in block_configs
                           if _has_lora_capable_attn(bc))
    layers_with_mlp = sum(1 for bc in block_configs
                          if _has_lora_capable_ffn(bc))

    total = (layers_with_attn * len(attn_modules) +
             layers_with_mlp * len(mlp_modules) +
             num_layers * len(other_modules))

    logger.info(f"LoRA module-layer count: {total} "
                f"(attn: {layers_with_attn}x{len(attn_modules)}, "
                f"mlp: {layers_with_mlp}x{len(mlp_modules)}, "
                f"other: {num_layers}x{len(other_modules)}, "
                f"uniform would be {num_layers * len(all_target_modules)})")
    return total


def _infer_shared_expert_size_from_adapter(adapter_dir: str) -> int:
    """Infer shared expert intermediate size from LoRA adapter weights.

    Scans the adapter for shared_expert.down_proj lora_A weights and
    returns the global (unsharded) intermediate size. This is more reliable
    than the model config, which may not match the adapter (e.g., upcycled
    models).
    """
    import json

    try:
        from tensorrt_llm.lora_manager import get_model_path, load_state_dict
        model_path = get_model_path(adapter_dir, "adapter_model")
        if model_path is None:
            return 0
        adapter_weights = load_state_dict(model_path)
        if adapter_weights is None:
            return 0
        for key, tensor in adapter_weights.items():
            if 'shared_expert' in key and 'down_proj' in key and 'lora_A' in key:
                adapter_config_path = os.path.join(adapter_dir,
                                                   "adapter_config.json")
                with open(adapter_config_path) as f:
                    rank = json.load(f).get("r", 0)
                if rank > 0:
                    return tensor.shape[1] if tensor.shape[
                        0] == rank else tensor.shape[0]
    except Exception as e:
        logger.debug(f"Failed to infer shared expert size from adapter: {e}")
    return 0


def _try_infer_num_experts(model_config: ModelConfig) -> int:
    """
    Attempt to infer the number of experts from the model configuration.

    Different MoE models use different attribute names for storing the number of experts,
    so this function checks for various possible names and returns a default of 1 if none are found.
    However, this function is not exhaustive and may miss some cases, so it should be revised.
    """
    config = getattr(model_config, 'pretrained_config', model_config)

    expert_attr_names = [
        'num_experts', 'num_local_experts', 'moe_num_experts',
        'experts_per_router'
    ]
    num_experts = None
    for attr_name in expert_attr_names:
        if hasattr(config, attr_name):
            num_experts = getattr(config, attr_name)
            break

    # Default to 1 for non-MoE models or if no experts attribute is found
    if num_experts is None:
        return 1

    return num_experts


def _adjust_torch_mem_fraction():
    # If true, adjust PyTorch CUDA memory fraction to correspond to the
    # total GPU memory minus the statically allocated engine memory.
    # If false, set the PyTorch CUDA memory fraction to 1.0.
    _limit_torch_cuda_mem_fraction: bool = True

    # FIXME: PyTorch only uses the garbage_collection_threshold setting
    #        if a memory fraction is set, cf.
    #   https://github.com/pytorch/pytorch/blob/cd995bfb2aac8891465809be3ce29543bd524287/c10/cuda/CUDACachingAllocator.cpp#L1357
    logger.debug("Setting PyTorch memory fraction to 1.0")
    torch.cuda.set_per_process_memory_fraction(1.0)

    # FIXME: As soon as
    #     torch.cuda._set_allocator_settings (added in PyTorch 2.8.0-rc1)
    #   or a similar API is available, the warning below should be removed
    #   and the allocator GC threshold be set via the new API instead.
    torch_allocator_config = os.environ.get("PYTORCH_ALLOC_CONF", "")
    torch_mem_threshold_advised = (
        torch.cuda.get_allocator_backend() == "native"
        and "expandable_segments:True" not in torch_allocator_config)
    torch_mem_threshold_set = "garbage_collection_threshold:" in torch_allocator_config
    if torch_mem_threshold_advised and not torch_mem_threshold_set:
        logger.warning(
            "It is recommended to incl. 'garbage_collection_threshold:0.???' or 'backend:cudaMallocAsync'"
            " or 'expandable_segments:True' in PYTORCH_ALLOC_CONF.")

    # NOTE: Even if a memory threshold was not set (cf. warning above), setting a memory
    #       fraction < 1.0 is beneficial, because
    #         https://github.com/pytorch/pytorch/blob/5228986c395dc79f90d2a2b991deea1eef188260/c10/cuda/CUDACachingAllocator.cpp#L2719
    #       and
    #         https://github.com/pytorch/pytorch/blob/5228986c395dc79f90d2a2b991deea1eef188260/c10/cuda/CUDACachingAllocator.cpp#L1240
    #       lead PyTorch to release all unused memory before hitting the set fraction. This
    #       still mitigates OOM, although at a higher performance impact, because it
    #       effectively resets the allocator cache.
    if not _limit_torch_cuda_mem_fraction:
        return
    mem_reserved = torch.cuda.memory_reserved()
    mem_free, mem_total = torch.cuda.mem_get_info()
    safety_margin = 32 * 1024**2
    mem_torch_max = mem_free + mem_reserved - safety_margin
    mem_torch_fraction = mem_torch_max / mem_total
    logger.info(
        f"Setting PyTorch memory fraction to {mem_torch_fraction} ({mem_torch_max / 1024**3} GiB)"
    )
    torch.cuda.set_per_process_memory_fraction(mem_torch_fraction)


def validate_feature_combination(llm_args, model_engine, sampler_type):
    # Validate the flags for features' combination
    def init_feature_status(llm_args) -> Dict[str, bool]:
        assert isinstance(
            llm_args, TorchLlmArgs
        ), "Expect TorchLlmArgs used for feature status validation."
        feature_list = [
            "overlap_scheduler",
            "cuda_graph",
            "attention_dp",
            "disaggregated_serving",
            "chunked_prefill",
            "mtp",
            "eagle3_one_model",
            "eagle3_two_model",
            "torch_sampler",
            "trtllm_sampler",
            "kv_cache_reuse",
            "slide_window_attention",
            "guided_decoding",
        ]
        feature_status: Dict[str, bool] = dict.fromkeys(feature_list)
        feature_status[
            "overlap_scheduler"] = not llm_args.disable_overlap_scheduler
        feature_status["cuda_graph"] = llm_args.cuda_graph_config is not None
        feature_status["attention_dp"] = llm_args.enable_attention_dp
        feature_status[
            "disaggregated_serving"] = llm_args.cache_transceiver_config is not None
        feature_status["chunked_prefill"] = llm_args.enable_chunked_prefill
        feature_status["mtp"] = isinstance(llm_args.speculative_config,
                                           MTPDecodingConfig)
        feature_status["eagle3_one_model"] = (
            isinstance(llm_args.speculative_config, EagleDecodingConfig)
            and llm_args.speculative_config.eagle3_one_model)
        feature_status["eagle3_two_model"] = (
            isinstance(llm_args.speculative_config, EagleDecodingConfig)
            and not llm_args.speculative_config.eagle3_one_model)
        feature_status[
            "torch_sampler"] = sampler_type == SamplerType.TorchSampler
        feature_status[
            "trtllm_sampler"] = sampler_type == SamplerType.TRTLLMSampler

        feature_status[
            "kv_cache_reuse"] = llm_args.kv_cache_config is not None and llm_args.kv_cache_config.enable_block_reuse
        feature_status["slide_window_attention"] = (
            hasattr(model_engine.model.model_config.pretrained_config,
                    "layer_types") and "sliding_attention"
            in model_engine.model.model_config.pretrained_config.layer_types)
        feature_status[
            "guided_decoding"] = llm_args.guided_decoding_backend is not None
        assert all(v is not None for v in feature_status.values()
                   ), "feature status has not been fully initialized."
        assert all(
            k in feature_list for k in
            feature_status.keys()), "unexpected feature type in feature_status."
        return feature_status

    feature_status: Dict[str, bool] = init_feature_status(llm_args)

    ERR_MSG_TMPL = "{feature1} and {feature2} enabled together is not supported yet."

    CONFLICT_RULES = [
        {
            "features": ["trtllm_sampler", "mtp"],
            "message":
            ERR_MSG_TMPL.format(feature1="trtllm_sampler", feature2="mtp") +
            " Please use sampler type auto instead."
        },
        {
            "features": ["trtllm_sampler", "eagle3_one_model"],
            "message":
            ERR_MSG_TMPL.format(feature1="trtllm_sampler",
                                feature2="eagle3_one_model") +
            " Please use sampler type auto instead."
        },
        {
            "features": ["trtllm_sampler", "eagle3_two_model"],
            "message":
            ERR_MSG_TMPL.format(feature1="trtllm_sampler",
                                feature2="eagle3_two_model") +
            " Please use sampler type auto instead."
        },
        # Add new conflict rules here in the future
    ]
    for rule in CONFLICT_RULES:
        if all(feature_status[feature] for feature in rule["features"]):
            raise ValueError(rule["message"])
