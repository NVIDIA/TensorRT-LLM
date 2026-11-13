# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# dep2ctx target (sm_107): the tp2ctx target's modeling.py
#   models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2ctx/modeling.py
# under attention DP: the TP all-reduce boundaries, their fusions and the sequence-parallel boundaries removed. That
# file is derived from tensorrt_llm/_torch/models/modeling_minimaxm3.py with the context-role items of #20629 added.
# A change to a shared context item lands in both targets; every item is behind TRTLLM_M31_DISABLE_<ID>.
"""MiniMax-M3 text language-model.

Layers 0-2 are dense attention + dense MLP. Layers 3-59 are sparse
attention (MiniMax index-K block selector + sparse GQA,
``disable_index_value=True`` for every layer in the M3 checkpoint) +
MoE (top-4 of 128 routed experts plus one shared expert, sigmoid
routing with bias, ``routed_scaling_factor=2.0``, ``swigluoai``
activation). Q/K are per-head Gemma-RMSNormed before partial RoPE
(``rotary_dim=64`` of ``head_dim=128``, ``rope_theta=5e6``).
"""

from __future__ import annotations

import copy
import dataclasses
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from typing import Mapping as TMapping

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels.index_k_producer import (
    index_k_producer_cache as _index_k_producer_cache,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels.index_k_producer import (
    pack_nvfp4_index_k as _pack_nvfp4_index_k,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    bcg_ptr_guard as _bcg_ptr_guard,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    prefill_pad_rows as _ppr,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    router_bf16_split as _router_bf16_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_prefill_store as _dps,
)
from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import _aligned_nvfp4_dequant_scales
from tensorrt_llm._torch.attention.backends.fmha.registry import get_enabled_fmha_lib_classes
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    MiniMaxM3MsaSparseAttention,
    MiniMaxM3SparseRuntimeBackend,
    _gather_paged_batched,
    _write_main_kv_slots_to_pool,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import index_head_range
from tensorrt_llm._torch.attention.backends.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams, MiniMaxAllReduceRMS
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.minimaxm3_weight_mapper import (
    MINIMAX_M3_PARAMS_MAP,
    MiniMaxM3HfWeightMapper,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import (
    convert_compressed_tensors_nvfp4_weights,
)
from tensorrt_llm._torch.models.modeling_minimaxm3_vl import split_multimodal_weights
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, ModelConfig, filter_weights
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import (
    Linear,
    MXFP8LinearMethod,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
    copy_weight,
    load_weight_shard,
)
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe import (
    MiniMaxM3MoeRoutingMethod,
    SwigluBiasActivation,
    create_moe,
)
from tensorrt_llm._torch.moe.fused_moe import mega_moe as _mega_moe
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.fp4_block_scale import TRTLLMGenNvfp4Base
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph,
    get_current_replay_token,
    is_in_breakable_cuda_graph,
)
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import (
    AuxStreamType,
    EventType,
    Fp4QuantizedTensor,
    get_model_extra_attrs,
    is_torch_compiling,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import AllReduceStrategy, PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from .. import cute_aot, runtime_hooks_compat
from ..msa import engine as _msa_engine
from ..ops import swiglu_rows_plan
from . import ctx_boundary_plan as _cbplan
from . import idx_scorer as _idx_scorer
from . import tile_select as _tsel
from .m31_switches import large_prefill_graphs_disabled, m31_disabled

# This module's torch namespace (namespace_of): its Python custom ops register there.
_NS = namespace_of(__name__)

# The routed output of a MegaMoE launch that left its top-k reduce to the caller. None on a runtime whose MegaMoE
# backend has no deferred reduce: every routed output there is already reduced.
MegaMoEDeferredTopkReduce = getattr(_mega_moe, "MegaMoEDeferredTopkReduce", None)


def _is_deferred_topk_reduce(routed) -> bool:
    return MegaMoEDeferredTopkReduce is not None and isinstance(routed, MegaMoEDeferredTopkReduce)


def _route_quant_key(x: torch.Tensor, router_logits: torch.Tensor) -> Tuple:
    """What identifies a route-quant seat's inputs for the MegaMoE backend: the memory, shape, strides and dtype of
    both tensors."""
    return tuple((t.data_ptr(), tuple(t.shape), t.stride(), t.dtype) for t in (x, router_logits))


# Legacy dense fallback: cuDNN cannot consume this SDPA layout.
_DENSE_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def _moe_routed_output_is_global(experts: nn.Module) -> bool:
    """Return whether the selected MoE backend performs fused communication."""
    backend = getattr(experts, "backend", experts)
    return getattr(backend, "scheduler_kind", None) == MoESchedulerKind.FUSED_COMM


class MiniMaxM3QKVIndexerLinear(Linear):
    """Five-way MiniMax-M3 projection with vLLM-compatible TP sharding.

    Each rank emits ``[Q | K | V | index-Q | index-K]``. Q follows normal
    attention head sharding, K/V/index-Q follow KV-head sharding (including
    replication when TP exceeds the KV-head count), and the single index-K
    head is replicated. The underlying :class:`Linear` remains the standard
    quantized implementation; only checkpoint packing is model-specific.
    """

    _SHARD_NAMES = ("q", "k", "v", "index_q", "index_k")

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_dim: int,
        dtype: torch.dtype,
        mapping: Mapping,
        quant_config: Optional[QuantConfig],
        skip_create_weights_in_init: bool,
        force_dynamic_quantization: bool,
        disable_deep_gemm: bool,
        use_custom_cublas_mm: bool,
        use_cute_dsl_bf16_gemm: bool,
        use_cute_dsl_blockscaling_mm: bool,
    ) -> None:
        if total_num_index_heads != total_num_kv_heads:
            raise ValueError(
                "MiniMax-M3 fused QKV+index projection requires index heads "
                f"({total_num_index_heads}) to equal KV heads ({total_num_kv_heads})."
            )
        if index_head_dim != head_dim:
            raise ValueError(
                "MiniMax-M3 fused QKV+index projection requires index_head_dim "
                f"({index_head_dim}) to equal head_dim ({head_dim})."
            )

        tp_size = int(mapping.tp_size)
        if total_num_heads % tp_size != 0:
            raise ValueError(f"Q heads ({total_num_heads}) must be divisible by TP ({tp_size}).")
        if total_num_kv_heads >= tp_size:
            if total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"KV heads ({total_num_kv_heads}) must be divisible by TP ({tp_size})."
                )
            local_num_kv_heads = total_num_kv_heads // tp_size
        else:
            if tp_size % total_num_kv_heads != 0:
                raise ValueError(
                    f"TP ({tp_size}) must be divisible by KV heads "
                    f"({total_num_kv_heads}) for replication."
                )
            local_num_kv_heads = 1

        self.total_num_heads = int(total_num_heads)
        self.total_num_kv_heads = int(total_num_kv_heads)
        self.total_num_index_heads = int(total_num_index_heads)
        self.head_dim = int(head_dim)
        self.index_head_dim = int(index_head_dim)
        self.local_num_heads = total_num_heads // tp_size
        self.local_num_kv_heads = local_num_kv_heads
        self.local_num_index_heads = local_num_kv_heads
        self.local_output_sizes = (
            self.local_num_heads * head_dim,
            local_num_kv_heads * head_dim,
            local_num_kv_heads * head_dim,
            local_num_kv_heads * index_head_dim,
            index_head_dim,
        )
        local_out_features = sum(self.local_output_sizes)

        super().__init__(
            hidden_size,
            tp_size * local_out_features,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
            reduce_output=False,
            skip_create_weights_in_init=skip_create_weights_in_init,
            force_dynamic_quantization=force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm,
            use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
            use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
        )

    def _shard_geometry(self, shard_name: str) -> Tuple[int, int]:
        """Return effective (world size, rank) for one checkpoint shard."""
        if shard_name == "q":
            return self.tp_size, self.tp_rank
        if shard_name == "index_k":
            return 1, 0

        total_heads = (
            self.total_num_index_heads if shard_name == "index_q" else self.total_num_kv_heads
        )
        if self.tp_size <= total_heads:
            return self.tp_size, self.tp_rank
        replicas = self.tp_size // total_heads
        return total_heads, self.tp_rank // replicas

    def load_five_way_weights(self, shards: Dict[str, Dict]) -> None:
        """Load five checkpoint projections into this rank's packed MXFP8 matrix."""
        local_shards: Dict[str, Dict[str, torch.Tensor]] = {}
        for shard_name in self._SHARD_NAMES:
            shard = shards[shard_name]
            if "weight" not in shard:
                raise KeyError(f"Missing {shard_name} projection weight.")
            shard_world, shard_rank = self._shard_geometry(shard_name)
            local = {}
            for key in ("weight", "weight_scale_inv", "weight_scale", "bias"):
                if key in shard:
                    local[key] = load_weight_shard(
                        shard[key],
                        shard_world,
                        shard_rank,
                        TensorParallelMode.COLUMN,
                        device=torch.device("cuda"),
                    )
            local_shards[shard_name] = local

        combined: Dict[str, torch.Tensor] = {
            "weight": torch.cat(
                [local_shards[name]["weight"] for name in self._SHARD_NAMES], dim=0
            ).contiguous()
        }
        for key in ("weight_scale_inv", "weight_scale", "bias"):
            present = [key in local_shards[name] for name in self._SHARD_NAMES]
            if any(present):
                if not all(present):
                    raise KeyError(f"Incomplete {key} across fused QKV+index shards.")
                combined[key] = torch.cat(
                    [local_shards[name][key] for name in self._SHARD_NAMES], dim=0
                ).contiguous()

        # KV calibration is per tensor, independent of the TP row partition.
        for key in ("k_scale", "v_scale"):
            scales = [
                shards[name][key].reshape(()) for name in ("q", "k", "v") if key in shards[name]
            ]
            if scales:
                combined[key] = torch.stack(scales).amax()

        # The checkpoint tensors above are already rank-local and packed.
        # Temporarily select vanilla loading so Linear copies them without a
        # second TP split or the three-shard QKV loader.
        saved_tp_size = self.tp_size
        saved_tp_rank = self.tp_rank
        saved_tp_mode = self.tp_mode
        saved_loading_config = self.weights_loading_config
        try:
            self.tp_size = 1
            self.tp_rank = 0
            self.tp_mode = None
            self.weights_loading_config = WeightsLoadingConfig(weight_mode=WeightMode.VANILLA)
            self.load_weights([combined])
        finally:
            self.tp_size = saved_tp_size
            self.tp_rank = saved_tp_rank
            self.tp_mode = saved_tp_mode
            self.weights_loading_config = saved_loading_config


# ---------------------------------------------------------------------------
# Config normalization helpers
# ---------------------------------------------------------------------------


def is_minimax_m3_vl_config(pretrained_config: PretrainedConfig) -> bool:
    """Return True if ``pretrained_config`` is a MiniMax-M3 VL config."""
    model_type = getattr(pretrained_config, "model_type", None)
    if model_type == "minimax_m3_vl":
        return True
    if model_type == "minimax_m3":
        return False
    architectures = getattr(pretrained_config, "architectures", None) or []
    return any("ForConditionalGeneration" in a for a in architectures)


def _wrap_dict_as_config(d: Dict[str, Any]) -> PretrainedConfig:
    """Wrap a plain ``dict`` text_config as a ``PretrainedConfig`` instance.

    ``AutoConfig.from_pretrained`` on the M3 checkpoint returns the VL
    config, but the nested ``text_config`` arrives as a raw ``dict``
    because the M3 ``configuration_minimax_m3_vl.py`` does not register a
    typed ``text_config_class``. The rest of this module expects attribute
    access (``cfg.num_hidden_layers``), so we materialise a
    ``PretrainedConfig`` and copy every key onto it.
    """
    out = PretrainedConfig()
    for k, v in d.items():
        setattr(out, k, v)
    return out


def get_text_config(pretrained_config: PretrainedConfig) -> PretrainedConfig:
    """Extract the language-model text config from a possibly-VL config.

    Returns the text config unchanged when ``pretrained_config`` is
    already a text config; raises if a VL config is missing
    ``text_config``. When the VL wrapper stores ``text_config`` as a
    plain ``dict`` (the M3 checkpoint case), it is materialised into a
    ``PretrainedConfig`` so downstream attribute access works.
    """
    if not is_minimax_m3_vl_config(pretrained_config):
        return pretrained_config
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is None:
        raise ValueError(
            "MiniMax-M3 VL config has no `text_config`; expected the text "
            "language model under config.text_config"
        )
    if isinstance(text_config, dict):
        text_config = _wrap_dict_as_config(text_config)
    # Propagate fields that some loaders expect on the text config.
    if (
        getattr(text_config, "torch_dtype", None) is None
        and getattr(pretrained_config, "torch_dtype", None) is not None
    ):
        text_config.torch_dtype = pretrained_config.torch_dtype
    if (
        getattr(text_config, "tie_word_embeddings", None) is None
        and getattr(pretrained_config, "tie_word_embeddings", None) is not None
    ):
        text_config.tie_word_embeddings = pretrained_config.tie_word_embeddings
    return text_config


def get_text_model_config(
    model_config: "ModelConfig[PretrainedConfig]",
) -> "ModelConfig[PretrainedConfig]":
    """Return a ``ModelConfig`` whose pretrained_config is the M3 text config."""
    cfg = copy.deepcopy(model_config)
    text_cfg = get_text_config(cfg.pretrained_config)
    cfg = dataclasses.replace(cfg, pretrained_config=text_cfg)
    return cfg


def _nvfp4_cache_supports_fused_write(
    data: torch.Tensor, scales: torch.Tensor, num_heads: int
) -> bool:
    """Reject invalid HND pools; select fusion only for its packed layout."""
    if (
        data.dim() != 5
        or scales.dim() != 5
        or data.shape[0] <= 0
        or data.shape[0] != scales.shape[0]
        or tuple(data.shape[1:]) != (2, num_heads, 128, 64)
        or tuple(scales.shape[1:]) != (2, num_heads, 128, 8)
        or data.dtype not in (torch.int8, torch.uint8)
        or scales.dtype != torch.uint8
    ):
        raise ValueError("MiniMax-M3 NVFP4 data and scale caches must have matching HND geometry")
    for cache in (data, scales):
        if (
            cache.stride(4) != 1
            or cache.stride(3) != cache.shape[4]
            or cache.stride(2) < 128 * cache.stride(3)
            or cache.stride(1) < num_heads * cache.stride(2)
            or cache.stride(0) < 2 * cache.stride(1)
        ):
            raise ValueError(
                "MiniMax-M3 NVFP4 cache rows must be packed and heads/planes/pages must not overlap"
            )
    if scales.stride(2) != 128 * 8:
        # The MSA consumer, including the separate-producer path, requires
        # each head's scale block to follow the preceding head immediately.
        raise ValueError("MiniMax-M3 NVFP4 MSA scale cache must have packed head blocks")
    # Separate cache writes can use padded head strides. The packed PCG
    # boundary requires fusion and reports an error when this returns False.
    return data.stride(2) == 128 * 64 and data.stride(0) % 2 == 0 and data.stride(1) % 2 == 0


def _validate_sparse_attention_runtime_config(
    model_config: "ModelConfig[PretrainedConfig]",
) -> None:
    """Require the runtime backend that owns M3's metadata and index cache.

    Both dense and sparse M3 layers use that cache manager, independent of
    checkpoint precision or GPU architecture. Backend-specific constraints,
    such as MSA requiring SM100, are validated by the backend itself.
    """
    sparse_config = model_config.sparse_attention_config
    if sparse_config is None or sparse_config.algorithm != "minimax_m3":
        raise ValueError(
            "MiniMax-M3 requires sparse_attention_config.algorithm='minimax_m3' "
            "to create its KV-cache manager and prepare attn_metadata.minimax_m3. "
            "Set the following in the LLM API configuration:\n"
            "sparse_attention_config:\n  algorithm: minimax_m3"
        )
    implementation = sparse_config.implementation
    if implementation == "triton":
        logger.warning_once(
            "MiniMax-M3 is using implementation='triton'. "
            "Set sparse_attention_config.implementation='msa' to use the recommended "
            "MiniMax-M3 backend on SM100/SM103 GPUs.",
            key="minimax_m3_recommend_msa",
        )
    quant_config = model_config.quant_config
    if (
        quant_config is not None
        and quant_config.quant_mode is not None
        and quant_config.quant_mode.has_fp4_kv_cache()
        and implementation != "msa"
    ):
        raise ValueError(
            "MiniMax-M3 NVFP4 KV cache requires sparse_attention_config.implementation='msa'."
        )
    if getattr(sparse_config, "fuse_qkv_index_projection", False):
        if implementation != "msa":
            raise ValueError(
                "MiniMax-M3 fuse_qkv_index_projection=True requires the 'msa' implementation."
            )
        if getattr(sparse_config, "indexer_kv_dtype", None) not in ("fp8", "nvfp4"):
            raise ValueError(
                "MiniMax-M3 fuse_qkv_index_projection=True requires indexer_kv_dtype='fp8' or 'nvfp4'."
            )
        if (
            quant_config is None
            or quant_config.quant_mode is None
            or not (
                quant_config.quant_mode.has_fp8_kv_cache()
                or quant_config.quant_mode.has_fp4_kv_cache()
            )
        ):
            raise ValueError(
                "MiniMax-M3 fuse_qkv_index_projection=True requires an FP8 or NVFP4 main KV cache."
            )


def get_sparse_layer_ids(text_config: PretrainedConfig) -> Tuple[List[int], List[int]]:
    """Return ``(dense_layer_ids, sparse_layer_ids)`` for the M3 text model."""
    sparse_cfg = getattr(text_config, "sparse_attention_config", None)
    n_layers = int(text_config.num_hidden_layers)
    if sparse_cfg is None:
        return list(range(n_layers)), []
    if not sparse_cfg.get("use_sparse_attention", True):
        return list(range(n_layers)), []
    freq = sparse_cfg.get("sparse_attention_freq")
    if freq is None:
        return list(range(n_layers)), []
    if len(freq) != n_layers:
        raise ValueError(
            f"sparse_attention_freq length {len(freq)} does not match num_hidden_layers {n_layers}"
        )
    dense = [i for i, f in enumerate(freq) if int(f) == 0]
    sparse = [i for i, f in enumerate(freq) if int(f) != 0]
    return dense, sparse


def get_sparse_disable_index_value_layer_ids(
    text_config: PretrainedConfig,
) -> List[int]:
    """Return layer ids where the sparse index-value branch is disabled."""
    sparse_cfg = getattr(text_config, "sparse_attention_config", None)
    if sparse_cfg is None:
        return []
    flags = sparse_cfg.get("sparse_disable_index_value")
    if flags is None:
        return []
    return [i for i, f in enumerate(flags) if int(f) != 0]


def get_moe_layer_ids(text_config: PretrainedConfig) -> Tuple[List[int], List[int]]:
    """Return ``(dense_mlp_layer_ids, moe_layer_ids)`` for the M3 text model."""
    freq = getattr(text_config, "moe_layer_freq", None)
    n_layers = int(text_config.num_hidden_layers)
    if freq is None:
        return [], list(range(n_layers))
    if len(freq) != n_layers:
        raise ValueError(
            f"moe_layer_freq length {len(freq)} does not match num_hidden_layers {n_layers}"
        )
    dense = [i for i, f in enumerate(freq) if int(f) == 0]
    moe = [i for i, f in enumerate(freq) if int(f) != 0]
    return dense, moe


def _index_projection_quant_config(
    model_config: "ModelConfig[PretrainedConfig]", layer_idx: Optional[int]
) -> Optional[QuantConfig]:
    """The quantization a per-layer map declares for a layer's fused ``index_qk_proj``, or None.

    ``index_q_proj`` and ``index_k_proj`` load into one fused ``index_qk_proj`` (a
    FUSED_GATE_UP_LINEAR Linear) that ``apply_layerwise_quant_config`` matches only by
    ``gate_proj`` / ``gate_up_proj`` names, so its entry is read here, by the fused module's
    own name. Only the expansion of the MXFP8 export with packed NVFP4 experts declares it;
    ModelOpt maps name the checkpoint's ``index_q_proj`` / ``index_k_proj`` instead, so every
    other checkpoint keeps the unquantized projection, as before.
    """
    quant_config_dict = getattr(model_config, "quant_config_dict", None)
    if quant_config_dict is None or layer_idx is None:
        return None
    return quant_config_dict.get(f"model.layers.{layer_idx}.self_attn.index_qk_proj")


# Outside ``language_model.*``: deferred multimodal branches.
_VL_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _minimax_m3_swiglu_oai(gate_up: torch.Tensor, *, alpha: float, limit: float) -> torch.Tensor:
    """SwiGLU with asymmetric gate clamp + alpha scaling (SGLang's
    ``swiglu_no_interleaved_with_alpha_and_limit``)."""
    gate, up = gate_up.chunk(2, dim=-1)
    # Gate clamp is upper-side only (checkpoint contract — clamping
    # below -limit drifts vs SGLang for large-negative gates).
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


def _build_swiglu_oai_dense_mlp(
    model_config: "ModelConfig[PretrainedConfig]",
    intermediate_size: int,
    layer_idx: Optional[int] = None,
    *,
    is_shared_expert: bool = False,
) -> GatedMLP:
    """Build a dense MLP using the MiniMax-M3 ``swigluoai`` activation.

    Layers 0-2 use this with ``dense_intermediate_size`` (the
    ``is_shared_expert=False`` path); the MoE shared expert uses it
    with ``shared_intermediate_size`` and ``is_shared_expert=True``.

    Under Attention DP (``mapping.enable_attention_dp=True``) each rank
    processes rank-local tokens independently. Sharding ``gate_up_proj``
    (COLUMN) and ``down_proj`` (ROW) across the global TP group and then
    all-reducing ``down_proj`` would mix outputs across independent
    rank-local token sets, producing wrong results. We collapse TP to
    1 via ``overridden_tp_size=1`` so the MLP is replicated and operates
    purely rank-locally (no sharding, no allreduce on ``down_proj``).

    For the MoE shared-expert path (``is_shared_expert=True``)
    ``reduce_output`` is forced ``False`` regardless of ADP: the routed
    branch is also constructed with ``reduce_results=False`` and the
    composition runs a single external AllReduce on
    ``routed + shared``, matching the DeepSeekV3 / GLM convention.
    """
    config = model_config.pretrained_config
    swiglu_alpha = float(getattr(config, "swiglu_alpha", 1.702))
    swiglu_limit = float(getattr(config, "swiglu_limit", 7.0))
    enable_adp = model_config.mapping.enable_attention_dp
    # Shared experts inside the MoE block defer the cross-rank reduction
    # to the MoE's single external AllReduce. The dense MLP path (no
    # MoE composition) carries its own reduction unless ADP collapses
    # TP to 1.
    reduce_output = False if is_shared_expert else (not enable_adp)
    # SwiGLU-OAI is plain SwiGLU with an alpha gain and an (up + 1) offset, so
    # the fused silu_and_mul kernel runs it in one launch with an optional fp8
    # epilogue. Mirrors the routed-expert SwigluBias path; the math lives in
    # _minimax_m3_swiglu_oai.
    return GatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=intermediate_size,
        bias=False,
        activation=torch.nn.functional.silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=1.0,
        swiglu_limit=swiglu_limit,
        dtype=config.torch_dtype,
        config=model_config,
        overridden_tp_size=1 if enable_adp else None,
        reduce_output=reduce_output,
        is_shared_expert=is_shared_expert,
        layer_idx=layer_idx,
    )


_EAGER_MULTISTREAM = False  # EMS is not carried by this fork (sm_103 only)


# DEP2CTX_UNPERMUTE_ADD (CTX): the shared expert's output is added inside the routed experts' unpermute
# epilogue (moe_unpermute_inplace addend, C++), instead of a separate bf16 add kernel. The unpermuted sum is rounded to
# bf16 first and the add is done in float and rounded once, i.e. bitwise shared_output.add_(routed_output). The shared
# expert runs first (same stream, same kernels). Taken only on the CuTe DSL unfused-finalize path with an op that has
# the addend argument, eager, without EMS / MOE_GRAPH / LoRA; any other call keeps the separate add. Default ON; kill
# switch TRTLLM_M31_DISABLE_DEP2CTX_UNPERMUTE_ADD=1.
_UNPERMUTE_ADD = not m31_disabled("DEP2CTX_UNPERMUTE_ADD")
_UNPERMUTE_ADD_STATE: Dict[str, Any] = {}

# CTX_MOE_TAIL: the MegaMoE local backend leaves the routed launch's top-k reduce to the layer, which folds
# it into the shared-expert add (``megamoe_topk_reduce_shared_add``, bitwise TopkReduce then
# ``shared_output.add_(routed_output)``; 384-thread CTAs above 1024 rows). Every attention-DP MoE forward requests it,
# so each launch bucket always runs the same kernel class; a launch the backend does not defer returns the reduced
# rows and keeps the separate add. Default ON; kill switch TRTLLM_M31_DISABLE_CTX_MOE_TAIL=1 (the MegaMoE op reads it
# too and then reports no plane above the generation-phase range).
_CTX_MOE_TAIL = not m31_disabled("CTX_MOE_TAIL")
_CTX_MOE_TAIL_LOGGED: set = set()


def _ctx_moe_tail_log(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _CTX_MOE_TAIL_LOGGED:
        _CTX_MOE_TAIL_LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 dep2ctx] CTX_MOE_TAIL {outcome}: {detail}")


def _ctx_moe_tail_request(
    moe: nn.Module, hidden_states: torch.Tensor, all_rank_num_tokens, lora_params
) -> None:
    """Arm the routed launch of this forward to leave its top-k reduce to the layer, where the backend admits it."""
    if not _CTX_MOE_TAIL:
        return
    backend = getattr(moe.experts, "backend", None)
    if not hasattr(backend, "request_deferred_topk_reduce"):
        _ctx_moe_tail_log("stock", f"backend {type(backend).__name__} reduces its own top-k")
        return
    if moe.allreduce is not None or not moe.routed_output_is_global:
        reason = "not attention DP on a fused-communication MoE"
    elif lora_params:
        reason = "LoRA"
    elif hidden_states.dtype != torch.bfloat16:
        reason = "the activation is not BF16"
    elif is_torch_compiling() or torch.compiler.is_compiling():
        reason = "torch.compile"
    else:
        reason = backend.request_deferred_topk_reduce(all_rank_num_tokens, hidden_states.shape[0])
    if reason is not None:
        backend.cancel_deferred_topk_reduce()
        _ctx_moe_tail_log("stock", reason)


# DEP2CTX_TOPK_FOLD (K5 U2): where a MoE layer's routed launch deferred its top-k reduce (CTX_MOE_TAIL) and the next
# layer's input norm is the DEP2CTX_NORMQ_V2 boundary, the reduce + shared add runs inside that boundary
# (minimax_m31_ctx_boundary_normq_topk, ctxBoundaryTopkNormQuantSm107.cuh): one kernel instead of
# megamoe_topk_reduce_shared_add + minimax_m31_ctx_boundary_normq, every output bit for bit theirs (the moe rows are
# never written: the shared buffer ends holding moe + residual, as after the two kernels). forward_MoE arms
# _TOPK_FOLD_SLOT before the MoE only when nothing between the MoE and the boundary reads its output (no DSpark capture
# layer) and the stock boundary would take the step; _ctx_moe_tail_add then parks the deferred reduce
# instead of running it, and forward_MoE hands it to the fold (or, on any decline, runs the reduce and the stock
# boundary). Default ON; kill switch TRTLLM_M31_DISABLE_DEP2CTX_TOPK_FOLD=1 (the two kernels). An op rejection turns
# the fold off for the process.
_TOPK_FOLD = not m31_disabled("DEP2CTX_TOPK_FOLD")
_TOPK_FOLD_STATE: Dict[str, Any] = {}


class _TopkFoldSlot:
    """The deferred top-k reduce of one armed MoE call, between ``_ctx_moe_tail_add`` and the next-layer boundary.

    ``arm`` (forward_MoE, before the MoE) allows one park; ``park`` (the tail add) holds (deferred, shared) instead of
    reducing; ``take`` (forward_MoE, after the MoE, also on an exception) returns the pair and always disarms and
    clears, so an unreduced buffer can never outlive the forward that produced it."""

    def __init__(self):
        self.armed = False
        self.parked = None

    def arm(self) -> None:
        self.armed, self.parked = True, None

    def park(self, deferred: MegaMoEDeferredTopkReduce, shared: torch.Tensor) -> None:
        assert self.armed and self.parked is None, "DEP2CTX_TOPK_FOLD: park without arm"
        self.parked = (deferred, shared)

    def take(self):
        parked, self.parked, self.armed = self.parked, None, False
        return parked


_TOPK_FOLD_SLOT = _TopkFoldSlot()


def _ctx_moe_tail_add(routed_output, shared_output: torch.Tensor) -> torch.Tensor:
    """``shared_output.add_(routed_output)``, in one kernel with the top-k reduce when the launch deferred it; parked
    for the next-layer boundary (DEP2CTX_TOPK_FOLD) when forward_MoE armed the fold and the plane can be folded."""
    if _is_deferred_topk_reduce(routed_output):
        if not shared_output.is_contiguous() or shared_output.data_ptr() % 16:
            shared_output = shared_output.contiguous()
        if _TOPK_FOLD_SLOT.armed:
            reason = _topk_fold_plane_decline_reason(routed_output, shared_output)
            if reason is None:
                _TOPK_FOLD_SLOT.park(routed_output, shared_output)
                return shared_output
            _topk_fold_log("not taken", reason)
        _ctx_moe_tail_log("taken", "top-k reduce + shared add")
        return routed_output.add_into(shared_output)
    return shared_output.add_(routed_output)


def _topk_fold_log(outcome: str, detail: str) -> None:
    """Log ``outcome: detail`` once per message shape (numbers such as row counts do not make a new one, so the logged
    set stays bounded)."""
    key = ("log", outcome, re.sub(r"\d+", "#", detail))
    if key not in _TOPK_FOLD_STATE:
        _TOPK_FOLD_STATE[key] = True
        logger.info(f"[M3.1 dep2ctx] DEP2CTX_TOPK_FOLD {outcome}: {detail}")


def _topk_fold_plane_decline_reason(
    deferred: MegaMoEDeferredTopkReduce, shared: torch.Tensor
) -> Optional[str]:
    plane, scores = deferred.plane, deferred.scores
    aligned = all(
        t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (plane, scores, shared)
    ) and (
        shared.dim() == 2
        and plane.dim() == 3
        and plane.shape[0] == shared.shape[0]
        and tuple(scores.shape) == tuple(plane.shape[:2])
    )
    return _cbplan.topk_fold_plane_decline_reason(
        deferred=True, topk=int(plane.shape[1]) if plane.dim() == 3 else -1, aligned=aligned
    )


def _unpermute_add_supported() -> bool:
    ok = _UNPERMUTE_ADD_STATE.get("op")
    if ok is None:
        try:
            ok = "addend" in str(torch.ops.trtllm.moe_unpermute_inplace.default._schema)
        except (AttributeError, RuntimeError):
            ok = False
        _UNPERMUTE_ADD_STATE["op"] = ok
        if not ok:
            logger.info(
                "[M3.1 dep2ctx] DEP2CTX_UNPERMUTE_ADD not taken: moe_unpermute_inplace has no addend argument"
            )
    return ok


# DEP2CTX_ROUTER_CAST (the MiniMax-M3 router cast): from 2048 tokens the router input is cast to float32 by one
# vectorized Triton pass instead of aten's dtype-converting copy. bf16 -> fp32 is exact, so the cast and the router
# logits are bitwise unchanged. Small / non-contiguous / non-BF16 inputs and compiled or captured calls keep the aten
# cast. Default ON; kill switch TRTLLM_M31_DISABLE_DEP2CTX_GATE_CAST=1 (CTX's item ROUTER_CAST).
_ROUTER_CAST = not m31_disabled("DEP2CTX_GATE_CAST")
_ROUTER_CAST_MIN_TOKENS = 1024  # covers the 1160-row tail chunk
_ROUTER_CAST_STATE: Dict[str, Any] = {}


def _router_cast_kernel():
    kernel = _ROUTER_CAST_STATE.get("kernel")
    if kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _bf16_to_fp32_kernel(x_ptr, y_ptr, numel, BLOCK: tl.constexpr):
            offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < numel
            tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask).to(tl.float32), mask=mask)

        kernel = _ROUTER_CAST_STATE["kernel"] = (_bf16_to_fp32_kernel, triton.cdiv)
    return kernel


def _router_cast_op():
    """family_ops().minimax_m31_bf16_to_fp32 from the build; None when it is not built (the Triton prototype
    then runs)."""
    if "op" not in _ROUTER_CAST_STATE:
        _ROUTER_CAST_STATE["op"] = getattr(family_ops(), "minimax_m31_bf16_to_fp32", None)
    return _ROUTER_CAST_STATE["op"]


# DEP2CTX_ITEMS_IN_BCG (the breakable prefill graph path): under a breakable prefill CUDA graph capture the items below
# run instead of falling back to stock: NORMQ boundary (+ NF32 / N1b), MXFP8_CUBLASLT (M2), SWIGLU_MXFP8, UNPERMUTE_ADD
# and GATE_CAST. Each one's Python hand-off (the MXFP8 slot, the MoE prequant / router / addend attributes) is produced
# and consumed inside the same captured segment (the eager break is the attention core), and its lazy state (op lookup,
# M2 band self-check) is set by the bucket's eager warm-up, which runs the same shapes before capture. C1 (the combine's
# MXFP8 hand-off crosses the attention break) stays eager-only. Default ON (the captured steps' outputs are bitwise the
# eager steps' with A1_IN_BCG); kill switch TRTLLM_M31_DISABLE_DEP2CTX_ITEMS_IN_BCG=1.
_ITEMS_IN_BCG = not m31_disabled("DEP2CTX_ITEMS_IN_BCG")
_BCG_NOTED: set = set()


def _dep2ctx_capture_blocked() -> bool:
    """Whether a dep2ctx item must keep the stock path because the stream is capturing (any capture other than a
    breakable prefill graph capture with DEP2CTX_ITEMS_IN_BCG on, the default)."""
    return torch.cuda.is_current_stream_capturing() and not (
        _ITEMS_IN_BCG and is_in_breakable_cuda_graph()
    )


def _bcg_note(item: str, x: Any) -> None:
    """Engagement + call-site probe: one line per (item, captured bucket) when an item runs inside a breakable graph
    capture, with the dtype / shape it actually received there (never raises: a probe)."""
    if not torch.cuda.is_current_stream_capturing():
        return
    shape = tuple(getattr(x, "shape", ()))
    rows = int(shape[0]) if shape else -1
    if (item, rows) in _BCG_NOTED:
        return
    _BCG_NOTED.add((item, rows))
    logger.info(
        f"[M3.1 dep2ctx] DEP2CTX_ITEMS_IN_BCG captured {item} (rows {rows}) "
        f"input {getattr(x, 'dtype', type(x).__name__)} {shape}"
    )


# DEP2CTX_A1_IN_BCG (rides DEP2CTX_ITEMS_IN_BCG): inside a breakable prefill capture the QKV-index
# projection runs on A1's GEMM-only handle (the fused kernel's own mainloop with debug_gemm_out=True). That bf16 output
# is exactly what A1's epilogue producer consumes ("the outputs are bitwise those of running that producer on this
# kernel's own bf16 GEMM output", kernels/qkv_fusion/kernel.py), and the stock rows producer then runs in the eager
# attention break, so a captured step reproduces eager A1 bit for bit. Same row rule as A1 (> 1024 rows, here the
# captured bucket's rows): exact against eager only when no bucket straddles 1024 (1024 is a bucket boundary). The
# handle compiles on A1's first eager take (the bucket's warm-up), never inside a capture. Kill switch
# TRTLLM_M31_DISABLE_DEP2CTX_A1_IN_BCG=1.
_A1_IN_BCG = not m31_disabled("DEP2CTX_A1_IN_BCG")
_A1_BCG_GEMMS: Dict[int, Any] = {}


def _a1_bcg_warm(
    proj, act, act_sf, rows: int, n_total: int, num_heads_kv: int, num_heads_index: int
) -> None:
    """Compile and load A1's GEMM-only handle once per N, eagerly (on A1's first eager take)."""
    if not (_ITEMS_IN_BCG and _A1_IN_BCG) or n_total in _A1_BCG_GEMMS:
        return
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx.kernels.qkv_fusion.plain_gemm import (  # noqa: E501
        get_gemm,
    )

    gemm = get_gemm(n_total, 4, num_heads_kv, num_heads_index)
    gemm(act.view(rows, -1), act_sf, proj.weight, proj.weight_scale, proj._mxfp8_alpha)
    _A1_BCG_GEMMS[n_total] = gemm
    logger.info(f"[M3.1 dep2ctx] DEP2CTX_A1_IN_BCG GEMM-only handle ready (N={n_total})")


def _a1_bcg_gemm(attn, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    """The packed QKV-index projection on A1's GEMM-only handle inside a breakable capture, or None (stock)."""
    if not (
        _ITEMS_IN_BCG
        and _A1_IN_BCG
        and is_in_breakable_cuda_graph()
        and torch.cuda.is_current_stream_capturing()
    ):
        return None
    proj = attn.qkv_proj
    method = getattr(proj, "quant_method", None)
    rows = int(hidden_states.shape[0])
    n_total = (
        attn.num_heads + 2 * attn.num_key_value_heads + attn.sparse_num_index_heads + 1
    ) * 128
    gemm = _A1_BCG_GEMMS.get(n_total)
    if (
        gemm is None
        or not _CTX_QKV_FUSED
        or hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or not _ctx_qkv_fused_takes_rows(rows)
        or not isinstance(method, MXFP8LinearMethod)
        or not getattr(method, "use_cutlass", False)
        or getattr(proj, "bias", None) is not None
        or tuple(proj.weight.shape) != (n_total, hidden_states.shape[1])
    ):
        return None
    prequantized = _NORMQ_SLOT.take(hidden_states)
    act, act_sf = (
        prequantized
        if prequantized is not None
        else torch.ops.trtllm.mxfp8_quantize(hidden_states, True)
    )
    _bcg_note("A1_GEMM", hidden_states)
    return gemm(act.view(rows, -1), act_sf, proj.weight, proj.weight_scale, proj._mxfp8_alpha)


# Large prefill graphs (prefill breakable-graph buckets above 4096). Every item below is default ON;
# TRTLLM_M31_DISABLE_DEP2CTX_LARGE_PREFILL_GRAPHS=1 turns all of them off, TRTLLM_M31_DISABLE_<ID>=1 one of them
# (large_prefill_graphs_disabled).
#
# DEP2CTX_BCG_LARGE: the model adds the buckets ops.prefill_pad_rows.DEP2CTX_BCG_GRID (5120..16384, 1K grain
# plus the 12287 / 12288 band edge) to the configured breakable capture list (RuntimeHooks.prefill_capture_extra_num_
# tokens), so a step up to max_num_tokens replays a graph instead of running eager. Replay at a bucket is bitwise eager
# at the same padded rows. Against eager at the live rows no bucket above 4096 straddles a dep2ctx-pinned configuration
# edge (prefill_pad_rows.band_straddles), but the unpinned M-keyed tactics (the MXFP8 CUTLASS autotuner bands, the
# power-of-two tuning buckets: prefill_pad_rows.DEP2CTX_UNPINNED_EDGES) are straddled, so that comparison is not
# guaranteed bitwise (the O2 oracle decides; non-bitwise is labeled). The extras join only a served list that reaches
# 4096 (a shorter list would pad its gap up to 5120). The switch caps the engine's list at 4096 even when the served
# config lists larger buckets: steps above 4096 then run eager and unpadded, as before. It must ship with
# DEP2CTX_BCG_PAD_MASK (a light rank's pad rows otherwise run real expert work on the pacing rank).
_BCG_LARGE = not large_prefill_graphs_disabled("DEP2CTX_BCG_LARGE")
_BCG_BASE_MAX = 4096
# DEP2CTX_BCG_PAD_MASK: on a padded or graphed prefill step the pad rows go to no routed expert (ids -1,
# weights 0 through the MegaMoE local backend's set_pad_mask). The mask is written by one eager launch in the outer
# forward from the host live count (attn_metadata.num_tokens, the count the attention break slices to) and handed to
# the backend by each MoE block before its router seats the stage. Bitwise on the live rows (their grouped-GEMM rows
# and per-token reduce are independent of the masked rows). Kill switch TRTLLM_M31_DISABLE_DEP2CTX_BCG_PAD_MASK=1.
_BCG_PAD_MASK = not large_prefill_graphs_disabled("DEP2CTX_BCG_PAD_MASK")
# DEP2CTX_A1_FUSED_IN_BCG: inside a breakable capture the A1 fused QKV GEMM + NVFP4 producer + KV insert runs in
# the captured segment at the bucket's rows (static cache views and the persistent msa_out_cache_loc, whose -1 tail
# makes the pad rows' cache stores no-ops), so the break receives q / index-q and skips the GEMM-only round trip and the
# rows producer. Bitwise its kill switch (the GEMM-only + rows producer split is bitwise the fused kernel's epilogue).
# The captured addresses are guarded by the model's own StaticPointerGuard (wired into its attention layers; checked in
# the outer forward before the body replays, so before any captured producer writes; reset when the engine releases
# that model's prefill graphs); a layer without a guard keeps the stock projection. Rides on
# DEP2CTX_A1_IN_BCG and DEP2CTX_ITEMS_IN_BCG (either switch restores the stock projection under capture). Kill switch
# TRTLLM_M31_DISABLE_DEP2CTX_A1_FUSED_IN_BCG=1.
# Only the engine's release of the prefill graphs (RuntimeHooks.prefill_graphs_released) resets the guard, so on an
# engine without that hook the item stays off and the stock projection runs: a recapture on a new KV pool (after
# KV-cache estimation) would otherwise meet the released graphs' notes.
_A1_FUSED_IN_BCG_REQUESTED = not large_prefill_graphs_disabled("DEP2CTX_A1_FUSED_IN_BCG")
_A1_FUSED_IN_BCG = _A1_FUSED_IN_BCG_REQUESTED and runtime_hooks_compat.has_hook(
    "prefill_graphs_released"
)
_BCG_PTRS_WHAT = "[M3.1 dep2ctx] DEP2CTX_A1_FUSED_IN_BCG"

# The GPU architecture this target is written for (SM107).
_SM = (10, 7)


def _bcg_graphs_released(model: nn.Module) -> None:
    """RuntimeHooks.prefill_graphs_released: the released graphs' noted addresses go with them, so the next executor's
    captures (a new KV pool after KV-cache estimation) note theirs afresh. Only ``model``'s own guard is reset."""
    guard = getattr(model, "_bcg_ptrs", None)
    if guard is not None:
        guard.reset()


# DEP2CTX_BCG_RAW_BREAK: a breakable graph's attention break calls the attention core's Python body directly
# instead of through the torch custom-op dispatcher (torch.compile keeps the custom op). Host only, same calls. Kill
# switch TRTLLM_M31_DISABLE_DEP2CTX_BCG_RAW_BREAK=1.
_BCG_RAW_BREAK = not large_prefill_graphs_disabled("DEP2CTX_BCG_RAW_BREAK")
# DEP2CTX_K2Q_LPT_REUSE: the LPT-ordered k2q CSR build runs on msa_prefill's per-step workspace (sized once,
# reused by every layer) instead of allocating six buffers and re-sizing the schedule per layer. Same kernels and
# arguments, so the same bytes. Rides on DEP2CTX_SCHED_LPT; kill switch TRTLLM_M31_DISABLE_DEP2CTX_K2Q_LPT_REUSE=1.
_K2Q_LPT_REUSE = not large_prefill_graphs_disabled("DEP2CTX_K2Q_LPT_REUSE")


def _router_input_fp32(hidden_states: torch.Tensor) -> torch.Tensor:
    """``hidden_states.to(torch.float32)``; large contiguous BF16 eager inputs take the Triton cast (bitwise equal)."""
    if (
        not _ROUTER_CAST
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[0] < _ROUTER_CAST_MIN_TOKENS
        or not hidden_states.is_contiguous()
        or is_torch_compiling()
        or torch.compiler.is_compiling()
        or _dep2ctx_capture_blocked()
    ):
        return hidden_states.to(torch.float32)
    # Production: the compiled op (CUDA C++ kernel rule); the Triton pass is the prototype for images without it.
    _bcg_note("GATE_CAST", hidden_states)
    op = _router_cast_op()
    if op is not None:
        out = op(hidden_states)
        path = "C++ op"
    else:
        kernel, cdiv = _router_cast_kernel()
        out = torch.empty(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
        numel = hidden_states.numel()
        kernel[(cdiv(numel, 8192),)](hidden_states, out, numel, BLOCK=8192, num_warps=8)
        path = "Triton"
    if "logged" not in _ROUTER_CAST_STATE:
        _ROUTER_CAST_STATE["logged"] = True
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_GATE_CAST {path} router input cast taken ({hidden_states.shape[0]} rows)"
        )
    return out


# DEP2CTX_ROUTER_BF16S (not bitwise; default ON): the router reads the BF16 norm and an N-stacked BF16 split
# of its FP32 weight (minimax_m3/ops/router_bf16_split: one cuBLAS BF16 GEMM with FP32 accumulation + one reduce over
# the terms) instead of TF32 F.linear on the FP32 rows, and the pre-MoE boundary writes the BF16 norm (flavor <add,
# norm, mx, !f32, fp4> on the staged path) instead of the FP32 copy: 2 B instead of 4 B per element written and read.
# The weight carries 16 (2 terms) / 24 (3 terms) significant bits against TF32's 11, so the logits are closer to the
# FP32 / FP64 product, but not the bits of the TF32 router (top-4 flips only where TF32 noise already decides the
# order; the error against fp64 is at most TF32's and the top-4 agreement at least TF32's); +3.1 / 4.7 MB of weight per
# MoE layer. Kill switch TRTLLM_M31_DISABLE_DEP2CTX_ROUTER_BF16S=1: the TF32 router on the boundary's FP32 copy, bit
# for bit the path before this item.
# The weight is split into _router_bf16_split.DEFAULT_TERMS (2) terms. The router GEMM costs one more launch than
# F.linear and a 32 MiB cuBLAS workspace per stream (trtllm::cublas_mm, eager + capture).
_ROUTER_BF16S = not m31_disabled("DEP2CTX_ROUTER_BF16S")
_ROUTER_BF16S_TERMS = _router_bf16_split.DEFAULT_TERMS
_ROUTER_BF16S_LOGGED: set = set()


def _router_bf16s_log(key: str, message: str) -> None:
    if key not in _ROUTER_BF16S_LOGGED:
        _ROUTER_BF16S_LOGGED.add(key)
        logger.info(f"[M3.1 dep2ctx] DEP2CTX_ROUTER_BF16S {message}")


def _router_reads_bf16(moe: Any) -> bool:
    """Whether ``moe``'s gate takes the BF16 norm (DEP2CTX_ROUTER_BF16S, with its split weight available), so the
    pre-MoE boundary writes the BF16 norm for it instead of the FP32 copy."""
    gate = getattr(moe, "gate", None)
    return _ROUTER_BF16S and isinstance(gate, MiniMaxM3Gate) and gate.bf16s_weight() is not None


class MiniMaxM3Gate(nn.Module):
    """MiniMax-M3 router gate: float32 sigmoid scoring + per-expert bias correction.

    Owns the router projection ``weight`` and the per-expert
    ``e_score_correction_bias``, matching the DeepSeekV3 / Laguna /
    Qwen3 gate convention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        # Linear-style weight layout: ``(out_features, in_features)``
        # so the HF checkpoint's ``gate.weight`` (shape
        # ``[num_experts, hidden_size]``) loads as-is.
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=torch.float32),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((num_experts,), dtype=torch.float32),
            requires_grad=False,
        )
        # DEP2CTX_ROUTER_BF16S: ``weight`` as BF16 terms stacked along N (router_bf16_split.n_stacked_split); built by
        # load_weights, else on the first eager call (dummy weights); None while the item is off.
        self.register_buffer("wn", None, persistent=False)

    def bf16s_weight(self) -> Optional[torch.Tensor]:
        """DEP2CTX_ROUTER_BF16S's split weight, or None (switched off, or not built yet and this call cannot build it:
        a stream capture, torch.compile, a meta weight). A later in-place change of ``weight`` must go through
        load_weights (which rebuilds it)."""
        if not _ROUTER_BF16S:
            return None
        if self.wn is None and not (
            self.weight.is_meta
            or torch.cuda.is_current_stream_capturing()
            or is_torch_compiling()
            or torch.compiler.is_compiling()
        ):
            self.wn = _router_bf16_split.n_stacked_split(self.weight, _ROUTER_BF16S_TERMS)
        return self.wn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DEP2CTX_ROUTER_BF16S: contiguous [rows, hidden] BF16 rows (rows > 0) of an eager or breakable-capture call.
        wn = (
            self.bf16s_weight()
            if _ROUTER_BF16S
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[0] > 0
            and hidden_states.shape[1] == self.hidden_size
            and hidden_states.is_contiguous()
            and not (is_torch_compiling() or torch.compiler.is_compiling())
            and not _dep2ctx_capture_blocked()
            else None
        )
        if wn is not None:
            _bcg_note("ROUTER_BF16S", hidden_states)
            _router_bf16s_log(
                "taken",
                f"taken: BF16 split router GEMM (rows {hidden_states.shape[0]}, terms {_ROUTER_BF16S_TERMS}; NOT "
                f"bitwise vs the TF32 router)",
            )
            return _router_bf16_split.logits(hidden_states, wn, _ROUTER_BF16S_TERMS)
        # Router runs in fp32 to match SGLang.
        return torch.nn.functional.linear(_router_input_fp32(hidden_states), self.weight)

    def load_weights(self, weights: List[Dict]):
        """Load the router weight and the e_score_correction_bias.

        The HF MiniMax-M3 checkpoint stores the bias at
        ``block_sparse_moe.e_score_correction_bias`` (sibling of
        ``block_sparse_moe.gate.weight``), so
        :class:`MiniMaxM3ForCausalLM` rewrites it to
        ``block_sparse_moe.gate.e_score_correction_bias`` before the
        generic loader dispatches here. Both keys are tolerated as
        missing so callers can partial-load the gate (e.g. tests that
        only set the weight).
        """
        assert len(weights) == 1
        w = weights[0]
        if "weight" in w:
            self.weight.copy_(w["weight"][:].to(self.weight.dtype))
            # DEP2CTX_ROUTER_BF16S: (re)build the split weight from the loaded weight. A built one is refreshed in
            # place: graphs captured before a refit keep its storage, and a new allocation would leave their replay
            # reading the freed old split.
            if _ROUTER_BF16S and self.wn is not None and not self.weight.is_meta:
                self.wn.copy_(_router_bf16_split.n_stacked_split(self.weight, _ROUTER_BF16S_TERMS))
            else:
                self.wn = None
                self.bf16s_weight()
        if "e_score_correction_bias" in w:
            self.e_score_correction_bias.copy_(
                w["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)
            )

    @property
    def routing_method(self) -> MiniMaxM3MoeRoutingMethod:
        # Pass a callable so routing always sees the latest parameter
        # value — important for DWDP redistribution and post-load
        # adjustments that mutate ``e_score_correction_bias`` in place.
        return MiniMaxM3MoeRoutingMethod(
            top_k=self.top_k,
            num_experts=self.num_experts,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )


class MiniMaxM3MoE(nn.Module):
    """M3 routed MoE block + shared expert.

    The routed experts use the ``swigluoai`` activation:
    ``gate_clamped * sigmoid(alpha * gate_clamped) * (up_clamped + 1.0)``
    with asymmetric ``gate.clamp(max=limit)`` and symmetric
    ``up.clamp(min=-limit, max=limit)``. This matches SGLang's
    ``swiglu_no_interleaved_with_alpha_and_limit`` shape. We plumb
    ``alpha``, ``beta=1.0`` (the ``up + 1`` offset), and ``limit`` per
    local expert into ``create_moe`` together with
    ``ActivationType.SwigluBias``, which dispatches the CUTLASS-family
    ``SwigluBiasAdaptor`` operator inside the fused MoE kernel.

    The routed branch and the shared expert both produce local partial
    outputs (``reduce_results=False`` / ``reduce_output=False``); their
    sum runs through a single external AllReduce only when not under
    Attention DP and ``tp_size > 1``. This matches the
    DeepSeekV3 / GLM convention and avoids the duplicate communication
    that the previous independent-reduction wiring incurred.
    """

    @staticmethod
    def _get_experts_quant_config(model_config: "ModelConfig", layer_idx: int) -> QuantConfig:
        """Return the per-layer quant config for the routed experts.

        For MIXED_PRECISION checkpoints (MXFP8 base + NVFP4 experts),
        ``ModelConfig._set_minimax_m3_moe_quant_config`` pre-populates
        ``quant_config_dict`` with coarse entries keyed by
        ``model.layers.N.block_sparse_moe.experts``.  Falls back to the
        global ``quant_config`` when no per-layer entry exists (e.g. BF16
        or user-supplied global NVFP4 config).

        An exclusion outranks both the per-layer entry and the global config:
        ``create_weights`` treats an override as authoritative over anything
        ``__post_init__`` wrote, so this return value stands in for both
        quantization passes and exclusion is the one that runs second.

        The deepest name is the one tested, because exclusion walks ancestors
        -- a coarser entry is caught anyway, one naming the experts is not.
        """
        experts_name = f"model.layers.{layer_idx}.block_sparse_moe.experts"
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.is_module_excluded_from_quantization(
            experts_name
        ):
            return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        if getattr(model_config, "quant_config_dict", None) is None:
            return quant_config
        return model_config.quant_config_dict.get(experts_name, quant_config)

    def __init__(
        self,
        model_config: "ModelConfig[PretrainedConfig]",
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.use_dp = self.enable_attention_dp
        self.mapping = model_config.mapping

        self.swiglu_alpha_value = float(getattr(config, "swiglu_alpha", 1.702))
        self.swiglu_beta_value = 1.0  # SGLang's ``(up + 1)`` offset in swiglu_no_interleaved.
        self.swiglu_limit_value = float(getattr(config, "swiglu_limit", 7.0))

        # One value per layer, not per expert: the MoE backend broadcasts these
        # to whatever per-expert shape its kernels index (or bakes them as
        # scalars), sized by the slot count it actually resolved.
        self.moe_activation = SwigluBiasActivation(
            gate_sigmoid_scale=self.swiglu_alpha_value,
            linear_offset=self.swiglu_beta_value,
            clamp=self.swiglu_limit_value,
        )

        # Router gate owns the float32 projection weight, the per-expert
        # ``e_score_correction_bias``, and the ``routing_method`` it
        # exposes to ``create_moe`` — same convention as DeepSeekV3 /
        # Laguna / Qwen3.
        self.gate = MiniMaxM3Gate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        # Routed branch produces local partial outputs; the external
        # AllReduce below combines routed + shared in a single round
        # (matches DeepSeekV3). Under Attention DP the fused MoE already
        # skips its in-op all-reduce, so ``reduce_results=False`` is
        # also the correct flag there.
        experts_quant_config = MiniMaxM3MoE._get_experts_quant_config(model_config, layer_idx)
        # DEP2CTX_CUTEDSL_MOE: the routed experts on the CuTe DSL NVFP4 MoE (SwigluBias epilogue) when the installed
        # build implements it. Requested with degradation disallowed, so a layer it cannot serve raises instead of
        # running another backend; the role's LlmArgs keep their backend, so the kill switch restores it exactly.
        # The unfused finalize (FC2 to permuted rows + moe_unpermute_inplace: fixed top-k order, fp32 accumulate) is
        # run-to-run deterministic; the fused finalize's bf16 bulk scatter-add is not.
        cutedsl = _CUTEDSL_MOE and getattr(CuteDslFusedMoE, "SUPPORTS_SWIGLU_BIAS", False)
        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            aux_stream_dict=aux_stream_dict,
            reduce_results=False,
            model_config=dataclasses.replace(
                model_config, moe_backend="CUTEDSL", moe_disable_finalize_fusion=True
            )
            if cutedsl
            else model_config,
            layer_idx=layer_idx,
            override_quant_config=experts_quant_config,
            activation=self.moe_activation,
            allow_backend_degradation=not cutedsl,
        )
        if cutedsl:
            logger.info_once(
                f"[M3.1 dep2ctx] DEP2CTX_CUTEDSL_MOE routed experts on "
                f"{type(getattr(self.experts, 'backend', self.experts)).__name__}",
                key="dep2ctx_cutedsl_moe",
            )
        # DEP2CTX_MOE_RUNNER_CACHE: enabled on the first eager routed-experts call, once the weights and the
        # backend's quant state exist (not here: __init__ runs before create_weights / load_weights).
        self._moe_runner_cache_pending = bool(cutedsl and _MOE_RUNNER_CACHE)
        self.routed_output_is_global = _moe_routed_output_is_global(self.experts)
        # Shared expert: dense MLP fused into MoE output. Constructed
        # with ``is_shared_expert=True`` so ``reduce_output=False``
        # (the external AllReduce below performs the combined
        # reduction) and so the LoRA module types match the shared-
        # expert convention.
        n_shared = int(getattr(config, "n_shared_experts", 0) or 0)
        if n_shared > 0:
            shared_intermediate = (
                int(getattr(config, "shared_intermediate_size", None) or config.intermediate_size)
                * n_shared
            )
            self.shared_experts = _build_swiglu_oai_dense_mlp(
                model_config=model_config,
                intermediate_size=shared_intermediate,
                layer_idx=layer_idx,
                is_shared_expert=True,
            )
        else:
            self.shared_experts = None

        # External AllReduce on the combined routed + shared output.
        # Skipped under Attention DP (each rank's tokens are independent
        # so cross-rank reduction would mix them) and under tp_size==1
        # (nothing to reduce).
        self.allreduce = None
        if not self.use_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )

        # Aux stream + events for the routed-vs-shared parallel
        # execution path. The stream is only actually used when CUDA
        # graph multi-stream is enabled; otherwise
        # ``maybe_execute_in_parallel`` runs fn0/fn1 sequentially.
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}
        # DEP2CTX_NORMQ_FP4: one-shot (normed rows, their NVFP4 form) from the pre-MoE boundary.
        self._prequantized_fp4 = None
        # DEP2CTX_NORMQ_F32: one-shot (normed rows, the same rows as float32) for the router from the boundary.
        self._router_input_f32 = None
        # DEP2CTX_NORMQ_STAGE: one-shot _NormqStage of the MegaMoE-local stage the pre-MoE boundary wrote.
        self._normq_stage = None
        # DEP2CTX_BCG_PAD_MASK: hands the step's pad-row mask to the routed experts' next launch (ops.prefill_pad_rows
        # wire; None leaves the backend's mask alone).
        self.pad_rows_hand_over = None

    def fp4_input_scale(self, num_rows: int) -> Optional[torch.Tensor]:
        """The routed experts' fc31 input scale when a pre-quantized NVFP4 input stands in for their quantize_input,
        else None. Only for the TRT-LLM-op NVFP4 TRTLLM-gen backend, or the CuTe DSL NVFP4 backend
        (DEP2CTX_CUTEDSL_MOE), in one chunk, without token communication, DP, DWDP, an AWQ pre-scale or hidden
        padding: either quantize_input is then exactly fp4_quantize(x, fc31_input_scale, 16, False, False) and
        passes an unswizzled Fp4QuantizedTensor through."""
        experts = self.experts
        backend = getattr(experts, "backend", experts)
        trtllm_gen = isinstance(backend, TRTLLMGenNvfp4Base) and (
            type(getattr(backend, "op_backend", None)) is TRTLLMOpBackend
        )
        cutedsl = isinstance(backend, CuteDslFusedMoE) and getattr(backend, "has_nvfp4", False)
        w = getattr(backend, "w3_w1_weight", None)
        scale = getattr(backend, "fc31_input_scale", None)
        # The first failing condition names the reason in a one-shot 'not taken' line (engagement diagnosis).
        checks = (
            ("backend", not (trtllm_gen or cutedsl)),
            ("fused comm", self.routed_output_is_global),
            ("token comm", getattr(experts, "comm", None) is not None),
            ("dp", getattr(experts, "use_dp", False)),
            ("dwdp", getattr(experts, "enable_dwdp", False)),
            ("chunked", num_rows > getattr(experts, "moe_max_num_tokens", 0)),
            ("awq pre-scale", getattr(backend, "fc31_act_scale", None) is not None),
            ("sf vector", getattr(backend, "scaling_vector_size", None) != 16),
            # K in FP4 elements: two per byte, whatever the storage dtype (uint8 TRTLLM-gen, int64 CuTe DSL).
            ("weight shape", w is None or w.shape[-1] * w.element_size() * 2 != self.hidden_dim),
            (
                "input scale",
                not isinstance(scale, torch.Tensor)
                or scale.dtype != torch.float32
                or scale.numel() != 1,
            ),
        )
        reason = next((name for name, failed in checks if failed), None)
        if reason is None:
            return scale
        if "fp4_why" not in _NORMQ_STATE:
            _NORMQ_STATE["fp4_why"] = True
            logger.info(
                f"[M3.1 dep2ctx] DEP2CTX_NORMQ_FP4 not taken: {reason} (backend {type(backend).__name__}, "
                f"w {None if w is None else tuple(w.shape)}, scale "
                f"{None if not isinstance(scale, torch.Tensor) else (scale.dtype, scale.numel())}, rows {num_rows}, "
                f"moe_max_num_tokens {getattr(experts, 'moe_max_num_tokens', None)})"
            )
        return None

    def normq_stage_plan(
        self, x: torch.Tensor, all_rank_num_tokens: Optional[List[int]]
    ) -> Optional["_NormqStage"]:
        """DEP2CTX_NORMQ_STAGE: the MegaMoE-local stage the pre-MoE boundary on ``x`` (its input rows,
        [rows, hidden] bf16) can write the routed experts' NVFP4 input, scales and padded tail into, as
        ``try_fused_route_quant`` would at this call's bucket; None when the call will not take such a seat (the first
        failing check is logged once). ``drop_norm`` says whether the BF16 norm may be left out."""
        experts = self.experts
        backend = getattr(experts, "backend", experts)
        rows, hidden = x.shape
        routing_method = getattr(backend, "routing_method", None)
        base = getattr(routing_method, "route_quant_base", None)
        appended = (
            0
            if base is None
            else int(routing_method.experts_per_token) - int(base.experts_per_token)
        )
        may_fuse = getattr(getattr(experts, "scheduler", None), "_may_fuse_route_quant", None)
        reason = _cbplan.stage_decline_reason(
            enabled=_NORMQ_STAGE,
            ops_built=_normq_v2_out_op() is not None and _route_stage_op() is not None,
            sm_major=get_sm_version() // 10,
            hidden=hidden,
            seat_api=all(hasattr(backend, a) for a in _cbplan.SEAT_API),
            router_f32=_normq_f32_op() is not None,
            lora=False,
            rows=rows,
            all_rank_num_tokens=all_rank_num_tokens,
            ep_rank=int(getattr(getattr(experts, "mapping", None), "moe_ep_rank", 0) or 0),
            moe_max_num_tokens=int(getattr(experts, "moe_max_num_tokens", 0) or 0),
            may_fuse=bool(callable(may_fuse) and may_fuse(None)),
            appended_columns=appended,
        )
        stage = None
        if reason is None:
            chunk_max = _cbplan.chunk_max_tokens(all_rank_num_tokens)
            bias = routing_method.e_score_correction_bias
            # The backend's checks with a stand-in for the logits the router will return ([rows, E] fp32, contiguous;
            # nothing is launched); the seat re-checks them with the real logits before it is taken.
            standin = torch.empty((rows, bias.numel()), dtype=torch.float32, device=x.device)
            hint = backend._active_launch_max_tokens
            backend._active_launch_max_tokens = chunk_max
            try:
                reason = backend._route_quant_decline_reason(x, standin)
                if reason is None:
                    impl = routing_method.routing_impl
                    top_k = int(impl.top_k)
                    launch_max_t = backend._select_launch_max_tokens(chunk_max)
                    bufs = backend._acquire_buffers(
                        top_k=top_k,
                        hidden=hidden,
                        device=x.device,
                        output_dtype=x.dtype,
                        launch_max_T=launch_max_t,
                    )
                    stage = _NormqStage.build(
                        bufs,
                        rows=rows,
                        hidden=hidden,
                        top_k=top_k,
                        launch_max_t=launch_max_t,
                        chunk_max=chunk_max,
                        bias=bias,
                        scale=backend.fc31_input_scale,
                        routed_scaling_factor=float(impl.routed_scaling_factor),
                    )
                    if stage is None:
                        reason = "the bucket's stage buffers do not fit the boundary's layout"
            finally:
                backend._active_launch_max_tokens = hint
        if stage is None:
            key = ("stage_why", reason)
            if key not in _NORMQ_STATE:
                _NORMQ_STATE[key] = True
                logger.info(
                    f"[M3.1 dep2ctx] DEP2CTX_NORMQ_STAGE not taken: {reason} (backend {type(backend).__name__}, "
                    f"rows {rows})"
                )
            return None
        shared = self.shared_experts
        drop_reason = _cbplan.drop_norm_decline_reason(
            enabled=_NORMQ_STAGE_DROP_NORM,
            shared_takes_mxfp8=None if shared is None else _takes_published_mxfp8(shared),
            lora=False,
            router_reads_bf16=_router_reads_bf16(self),
        )
        stage.drop_norm = drop_reason is None
        if drop_reason is not None and ("drop_why", drop_reason) not in _NORMQ_STATE:
            _NORMQ_STATE[("drop_why", drop_reason)] = True
            logger.info(f"[M3.1 dep2ctx] DEP2CTX_NORMQ_STAGE_DROP_NORM not taken: {drop_reason}")
        return stage

    def _normq_stage_seat(
        self, stage: "_NormqStage", x: torch.Tensor, router_logits: torch.Tensor
    ) -> None:
        """DEP2CTX_NORMQ_STAGE: route the live rows of the stage the boundary wrote from the router's logits
        (``minimax_m31_route_stage``, bit for bit what ``noaux_tc_fp4_quantize_stage_out`` writes there) and hand the
        backend the seat, which its ``try_fused_route_quant`` takes for exactly (x, router_logits). If the backend now
        declines, it routes and quantizes itself from x, so a left-out norm is written first."""
        backend = getattr(self.experts, "backend", self.experts)
        hint = backend._active_launch_max_tokens
        backend._active_launch_max_tokens = stage.chunk_max
        try:
            reason = backend._route_quant_decline_reason(x, router_logits)
        finally:
            backend._active_launch_max_tokens = hint
        if reason is not None:
            if ("seat_why", reason) not in _NORMQ_STATE:
                _NORMQ_STATE[("seat_why", reason)] = True
                logger.info(
                    f"[M3.1 dep2ctx] DEP2CTX_NORMQ_STAGE declined at the router: {reason} (the backend stages itself)"
                )
            _NORMQ_DROPPED.materialize(x)
            return
        _route_stage_op()(
            router_logits,
            stage.bias,
            backend._take_pad_mask(stage.rows),
            stage.top_k,
            stage.routed_scaling_factor,
            -1,
            stage.ids,
            stage.weights,
            False,
            fp32_weight=_NOAUX_TC_FP32,
        )
        backend._prestaged_route_quant = (
            _route_quant_key(x, router_logits),
            stage.result,
            lambda: _NORMQ_DROPPED.materialize(x),
        )
        backend._last_staged_T[stage.launch_max_t] = stage.rows
        if "stage_seated" not in _NORMQ_STATE:
            _NORMQ_STATE["stage_seated"] = True
            logger.info(
                f"[M3.1 dep2ctx] DEP2CTX_NORMQ_STAGE seated: the boundary wrote the routed experts' NVFP4 input and "
                f"the stage tail, the route stage the ids / weights (rows {stage.rows}, bucket {stage.launch_max_t}, "
                f"BF16 norm {'left out' if stage.drop_norm else 'written'})"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        # DEP2CTX_BCG_PAD_MASK: first, so the seat below (_normq_stage_seat) and the backend's own staging take it.
        hand_over = getattr(self, "pad_rows_hand_over", None)
        if hand_over is not None:
            hand_over(hidden_states.shape[0])
        # DEP2CTX_NORMQ_STAGE: the stage the pre-MoE boundary wrote for exactly these rows (one-shot). A left-out BF16
        # norm lives only for this call: any reader that misses its quantized copy materializes it first.
        stage, self._normq_stage = self._normq_stage, None
        if stage is None:
            return self._forward(
                hidden_states, attn_metadata, final_all_reduce_params, lora_params, None
            )
        try:
            if stage.normed is not hidden_states or lora_params:
                _NORMQ_DROPPED.materialize(hidden_states)
                stage = None
            return self._forward(
                hidden_states, attn_metadata, final_all_reduce_params, lora_params, stage
            )
        finally:
            _NORMQ_DROPPED.clear()

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        final_all_reduce_params: Optional[AllReduceParams],
        lora_params: Optional[dict],
        normq_stage: Optional["_NormqStage"],
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        # DEP2CTX_NORMQ_FP4: the boundary's NVFP4 form of exactly these rows, if it wrote one (one-shot).
        prequantized, self._prequantized_fp4 = self._prequantized_fp4, None
        router_f32, self._router_input_f32 = self._router_input_f32, None
        if router_f32 is not None and router_f32[0] is not hidden_states:
            router_f32 = None
        experts_input, output_dtype = hidden_states, None
        if prequantized is not None and prequantized[0] is hidden_states and not lora_params:
            experts_input, output_dtype = prequantized[1], hidden_states.dtype

        def _compute_routed_output():
            if router_f32 is None:
                _NORMQ_DROPPED.materialize(hidden_states)
            router_logits = (
                torch.nn.functional.linear(router_f32[1], self.gate.weight)
                if router_f32 is not None
                else self.gate(hidden_states)
            )
            if normq_stage is not None:
                self._normq_stage_seat(normq_stage, hidden_states, router_logits)
            if self._moe_runner_cache_pending and not (
                is_torch_compiling() or torch.compiler.is_compiling()
            ):
                self._moe_runner_cache_pending = False
                enable = getattr(
                    getattr(self.experts, "backend", self.experts),
                    "enable_nvfp4_runner_cache",
                    None,
                )
                engaged = bool(enable is not None and enable())
                logger.info_once(
                    f"[M3.1 dep2ctx] DEP2CTX_MOE_RUNNER_CACHE {'on' if engaged else 'not applicable'} "
                    f"(routed experts, eager calls only)",
                    key="dep2ctx_moe_runner_cache",
                )

            def _run(logits):
                return self.experts(
                    experts_input,
                    logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=False,
                    lora_params=lora_params,
                    **({"output_dtype": output_dtype} if output_dtype is not None else {}),
                )

            return _run(router_logits)

        def _compute_shared_output():
            return self.shared_experts(hidden_states, lora_params=lora_params)

        if self.shared_experts is None:
            routed_output = _compute_routed_output()
            result = routed_output
        elif (
            _UNPERMUTE_ADD
            and not _EAGER_MULTISTREAM
            and not getattr(self, "_moe_graph_on", False)
            and not lora_params
            and not self.routed_output_is_global
            and hasattr(getattr(self.experts, "backend", self.experts), "moe_output_addend")
            and not getattr(
                getattr(self.experts, "backend", self.experts), "use_fused_finalize", True
            )
            and not (is_torch_compiling() or torch.compiler.is_compiling())
            and not _dep2ctx_capture_blocked()
            and _unpermute_add_supported()
        ):
            backend = getattr(self.experts, "backend", self.experts)
            _bcg_note("UNPERMUTE_ADD", hidden_states)
            shared_output = _compute_shared_output()
            backend.moe_output_addend = shared_output
            try:
                routed_output = _compute_routed_output()
            finally:
                leftover, backend.moe_output_addend = backend.moe_output_addend, None
            if (
                leftover is None
            ):  # the unpermute epilogue added it: routed_output is shared + routed
                result = routed_output
                if "logged" not in _UNPERMUTE_ADD_STATE:
                    _UNPERMUTE_ADD_STATE["logged"] = True
                    logger.info(
                        f"[M3.1 dep2ctx] DEP2CTX_UNPERMUTE_ADD shared expert added in the unpermute epilogue "
                        f"({routed_output.shape[0]} rows)"
                    )
            else:
                result = shared_output.add_(routed_output)
        else:
            _ctx_moe_tail_request(self, hidden_states, all_rank_num_tokens, lora_params)
            routed_output, shared_output = maybe_execute_in_parallel(
                _compute_routed_output,
                _compute_shared_output,
                self.event_dict[EventType.Main],
                self.event_dict[EventType.MoeShared],
                self.aux_stream,
                disable_on_compile=True,
            )
            if self.routed_output_is_global and self.allreduce is not None:
                # Fused-communication backends have already combined the routed
                # branch across ranks. Reduce only the still-local shared branch.
                shared_output = self.allreduce(
                    shared_output, all_reduce_params=final_all_reduce_params
                )
                return shared_output.add_(routed_output)
            # In-place add into ``shared_output`` to avoid allocating a temporary
            # (matches DeepSeekV3 / GLM convention).
            result = _ctx_moe_tail_add(routed_output, shared_output)

        if self.allreduce is not None and not self.routed_output_is_global:
            result = self.allreduce(result, all_reduce_params=final_all_reduce_params)
        return result


class MiniMaxRMSNorm(nn.Module):
    """All-reduce RMSNorm used for TP-sharded Q/K hidden states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        mapping: Mapping,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.mapping = mapping
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype), requires_grad=False)
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.all_reduce = AllReduce(mapping=self.mapping, strategy=AllReduceStrategy.NCCL)
        self.minimax_all_reduce_rms = MiniMaxAllReduceRMS(mapping=self.mapping)

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weight = load_weight_shard(
            weights[0]["weight"],
            tensor_parallel_size=self.mapping.tp_size,
            tensor_parallel_rank=self.mapping.tp_rank,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        copy_weight(self.weight, weight)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.contiguous()
        return self.minimax_all_reduce_rms(hidden_states, self.weight, self.eps)


def _extract_minimax_m3_attention_extra_attrs(layer_idx: str):
    """Resolve runtime metadata and the registered MiniMax-M3 layer."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata")
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata), "Invalid MiniMax-M3 attention metadata"

    attn_layers = extra_attrs.get("attn_layers")
    assert attn_layers is not None, "Attention layer is not registered"
    attn_layer_ref = attn_layers.get(layer_idx)
    assert attn_layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    attn_layer = attn_layer_ref()
    assert isinstance(attn_layer, MiniMaxM3Attention), "Invalid MiniMax-M3 attention layer"
    return metadata, attn_layer


# DEP2CTX_QKV_FUSED (#20629 QKV_FUSED, A1): one CuTe DSL kernel runs the QKV-index MXFP8 GEMM and the horizontal
# producer (per-head RMSNorm, RoPE, FP8 Q / index-Q, NVFP4 K/V + scales, FP8 index-K paged inserts) in its epilogue,
# bitwise the stock producer applied to its own BF16 GEMM output (the GEMM itself is deterministic, not bitwise to the
# stock MXFP8 GEMM). Steps above 1024 rows; eager only (the compiled prefill reaches it inside the fused-producer op).
# Kernel: SM107, #20629's kernel verbatim. Kill switch: TRTLLM_M31_DISABLE_DEP2CTX_QKV_FUSED=1.
_CTX_QKV_FUSED = not m31_disabled("DEP2CTX_QKV_FUSED")
_CTX_QKV_FUSED_MIN_ROWS = 1024
_CTX_QKV_FUSED_LOGGED = False
_CTX_QKV_FUSED_YIELD_LOGGED = False


_CTX_QKV_FUSED_COUNTS: Dict[str, int] = {}


def _ctx_qkv_fused_band(rows: int) -> str:
    return "<=2K" if rows <= 2048 else "<=4K" if rows <= 4096 else "<=8K" if rows <= 8192 else ">8K"


def _ctx_qkv_fused_final() -> None:
    logger.info(
        f"[M3.1 dep2ctx] DEP2CTX_QKV_FUSED final taken={dict(sorted(_CTX_QKV_FUSED_COUNTS.items()))}"
    )


def _ctx_qkv_fused_count(rows: int) -> None:
    """Per-call A1 engagement by step-size band, logged at counts 1, 2, 4, ... per band and once at exit."""
    if not _CTX_QKV_FUSED_COUNTS:
        import atexit

        atexit.register(_ctx_qkv_fused_final)
    band = _ctx_qkv_fused_band(rows)
    _CTX_QKV_FUSED_COUNTS[band] = n = _CTX_QKV_FUSED_COUNTS.get(band, 0) + 1
    if n & (n - 1) == 0:
        logger.info(f"[M3.1 dep2ctx] DEP2CTX_QKV_FUSED taken band={band} n={n} (rows {rows})")


def _ctx_qkv_fused_takes_rows(rows: int) -> bool:
    """Whether a step of rows QKV rows is large enough for the fused QKV GEMM + producer."""
    return rows > _CTX_QKV_FUSED_MIN_ROWS


def _autotuner_tuning() -> bool:
    """Whether the AutoTuner warmup pass is running. The fused QKV producer stays out of it, so the stock MXFP8 QKV
    GEMM profiles its shape there: the steps the producer does not take (< 1024 rows) otherwise run that GEMM on the
    CUTLASS runner's untuned default tactic."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    global _CTX_QKV_FUSED_YIELD_LOGGED
    tuning = AutoTuner.get().is_tuning_mode
    if tuning and not _CTX_QKV_FUSED_YIELD_LOGGED:
        _CTX_QKV_FUSED_YIELD_LOGGED = True
        logger.info(
            "[M3.1 dep2ctx] fused QKV yields to the AutoTuner tuning pass; plain QKV GEMM tuned"
        )
    return tuning


_CTX_QKV_FUSED_RUNNERS: Dict[Tuple[Any, ...], Any] = {}


# Every tile config a step can pick (_ctx_qkv_fused_cfg); all of them compile on a head geometry's first take, so a
# later chunk size (a prompt's short tail chunk) never compiles inside a timed request.
_CTX_QKV_FUSED_CFGS: Tuple[Dict[str, Any], ...] = ({},)


def _ctx_qkv_fused_cfg(rows: int) -> Dict[str, Any]:
    """The kernel's tile config for a step of rows QKV rows (every config is bitwise the same output)."""
    return {}


def _ctx_qkv_fused_runner(num_heads_q: int, num_heads_kv: int, num_heads_index: int, rows: int):
    """The compiled fused QKV runner for one per-rank head geometry and tile config. The first take for a geometry
    compiles every config in _CTX_QKV_FUSED_CFGS (eager, outside capture: the warmup's first long step)."""
    geometry = (num_heads_q, num_heads_kv, num_heads_index)
    cfg = _ctx_qkv_fused_cfg(rows)
    key = geometry + (tuple(sorted(cfg.items())),)
    runner = _CTX_QKV_FUSED_RUNNERS.get(key)
    if runner is None:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx.kernels.qkv_fusion.kernel import (  # noqa: E501
            M3QkvFusedRunner,
        )

        for each in _CTX_QKV_FUSED_CFGS:
            each_key = geometry + (tuple(sorted(each.items())),)
            if each_key not in _CTX_QKV_FUSED_RUNNERS:
                _CTX_QKV_FUSED_RUNNERS[each_key] = M3QkvFusedRunner(
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    num_heads_index=num_heads_index,
                    **each,
                )
        logger.info(
            f"[M3.1 dep2ctx] fused QKV runners compiled for heads {geometry}: {len(_CTX_QKV_FUSED_CFGS)} tile config(s)"
        )
        runner = _CTX_QKV_FUSED_RUNNERS[key]
    return runner


# DEP2CTX_QKV_ROWS (#20629 L2): chunks of at least 1024 tokens run the NVFP4 QKV + indexer producer as the rows
# kernel (minimax_m31_nvfp4_qkv_indexer_producer_rows: one warp per head slot and 4 consecutive tokens, the norm weight
# loaded once per warp), bitwise the per-token producer's outputs. A build without the op keeps the per-token producer.
# Kill switch: TRTLLM_M31_DISABLE_DEP2CTX_QKV_ROWS=1.
_QKV_ROWS = not m31_disabled("DEP2CTX_QKV_ROWS")
_QKV_ROWS_MIN_TOKENS = 1024


def _nvfp4_qkv_producer(num_tokens: int):
    """The NVFP4 QKV + indexer producer op for a step of num_tokens tokens (the rows producer for long chunks)."""
    if _QKV_ROWS and num_tokens >= _QKV_ROWS_MIN_TOKENS:
        rows = getattr(family_ops(), "minimax_m31_nvfp4_qkv_indexer_producer_rows", None)
        if rows is not None:
            return rows
    return torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert


# DEP2CTX_NORMQ (#20629 NORM_MXFP8, N1): a prefill layer boundary of at least 1024 rows runs
# minimax_m31_ctx_residual_rmsnorm_mxfp8 (residual add + RMSNorm that also writes the MXFP8 activation of the normed
# rows, bitwise mxfp8_quantize of them) on this rank's rows instead of the plain add + RMSNorm; the next MXFP8 linear
# reading the normed rows (the next layer's QKV, the shared expert's or dense MLP's gate_up) takes the activation and
# skips its quantize. Eager steps only (the slot compares data pointers). A build without the op keeps the plain norm.
# Kill switch: TRTLLM_M31_DISABLE_DEP2CTX_NORMQ=1.
_NORMQ = not m31_disabled("DEP2CTX_NORMQ")
# DEP2CTX_CUTEDSL_MOE: the routed experts on the CuTe DSL backend; not bitwise the stock backend, deterministic with the
# unfused finalize (identical token ids across server starts). Kill switch TRTLLM_M31_DISABLE_DEP2CTX_CUTEDSL_MOE=1.
_CUTEDSL_MOE = not m31_disabled("DEP2CTX_CUTEDSL_MOE")
# DEP2CTX_MOE_RUNNER_CACHE: the CuTe DSL routed experts' FC1 / FC2 run through pre-built runners with their tactic
# cached (CuteDslFusedMoE.enable_nvfp4_runner_cache): the same kernels without torch.library dispatch, runner
# construction or a tuner lookup per call -- bitwise unchanged, less host time per MoE call.
# Eager only. Kill switch TRTLLM_M31_DISABLE_DEP2CTX_MOE_RUNNER_CACHE=1.
_MOE_RUNNER_CACHE = not m31_disabled("DEP2CTX_MOE_RUNNER_CACHE")


_NORMQ_FP4 = not m31_disabled(
    "DEP2CTX_NORMQ_FP4"
)  # kill switch TRTLLM_M31_DISABLE_DEP2CTX_NORMQ_FP4=1
# the smallest boundary the fused pass takes
_NORMQ_MIN_ROWS = 1024
_NORMQ_STATE: Dict[str, Any] = {}


class _NormqSlot:
    """One pending MXFP8 activation for the next MXFP8 linear. It holds the normed tensor itself, so that storage can't
    be reused for another tensor while the entry waits; a take matches storage and shape, and every take clears it."""

    def __init__(self):
        self.pending = None
        # DEP2CTX_COMBINE_DROP_BF16: the pending entry's source was not written (the combine skipped its BF16 store);
        # this writes it. Set only by publish(..., materialize=); see _c1_materialize.
        self.pending_materialize = None
        # The taken entry's materialize, for the reader that took the pair but then runs the stock (BF16) path.
        self.taken_materialize = None

    def publish(
        self,
        x: torch.Tensor,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        materialize: Optional[Callable[[], None]] = None,
    ) -> None:
        if self.pending_materialize is not None:
            # An unread entry whose source was never written is replaced: write it first (no reader is left).
            _c1_materialize(self.pending_materialize)
        self.pending = (x, xq, xsf, torch.cuda.current_stream(x.device))
        self.pending_materialize = materialize

    def take(self, x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The pair for x's rows, or None."""
        pending, self.pending = self.pending, None
        materialize, self.pending_materialize = self.pending_materialize, None
        self.taken_materialize = None
        if pending is None:
            return None
        source, xq, xsf, producer = pending
        if x.dim() != 2 or x.data_ptr() != source.data_ptr() or x.shape != source.shape:
            if materialize is not None:
                _c1_materialize(materialize)
            return None
        self.taken_materialize = materialize
        consumer = torch.cuda.current_stream(x.device)
        if consumer != producer and not torch.cuda.is_current_stream_capturing():
            # The pair was allocated on the producing stream (a shared expert on the MoE aux stream reads it).
            xq.record_stream(consumer)
            xsf.record_stream(consumer)
        return xq, xsf


_NORMQ_SLOT = _NormqSlot()

# The combine writes the MXFP8 copy only for calls of at least _C1_MIN_ROWS rows (msa_prefill reads the sink's
# min_rows): at 8K rows the extra epilogue costs more than o_proj's quantize saves; full prefill chunks are 16K rows.
_C1_MIN_ROWS = 12288
# DEP2CTX_COMBINE_DROP_BF16: when the combine's MXFP8 copy is published for o_proj (DEP2CTX_COMBINE_MXFP8) and
# o_proj's input is exactly the combine's output (_combine_mxfp8_drop_bf16), the combine skips its BF16 store of that
# output: o_proj reads only the MXFP8 copy, whose bytes do not change. Any reader that misses the copy (a slot
# mismatch, a replaced entry, o_proj on its stock path) first writes the BF16 output through the entry's materialize
# (the same forward + combine without the MXFP8 copy: the bytes the skipped store would have written), so no path
# reads an unwritten output. Default ON; kill switch
# TRTLLM_M31_DISABLE_DEP2CTX_COMBINE_DROP_BF16=1.
_C1_DROP_BF16 = not m31_disabled("DEP2CTX_COMBINE_DROP_BF16")
_C1_MATERIALIZED = [0]


def _c1_publishes(out: torch.Tensor) -> bool:
    """Whether _combine_mxfp8_sink publishes the pair for this output (eager forwards of at least _NORMQ_MIN_ROWS)."""
    return not (
        out.shape[0] < _NORMQ_MIN_ROWS
        or is_torch_compiling()
        or is_in_breakable_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    )


def _combine_mxfp8_drop_bf16(out: torch.Tensor) -> bool:
    """DEP2CTX_COMBINE_DROP_BF16 (msa_prefill's sink ``drop_bf16`` hook): True when the combine may skip its BF16 store.
    The pair will be published (_c1_publishes) and ``out`` is the whole attention output buffer (offset 0, every byte
    of its storage), i.e. the tensor the attention hands o_proj, so o_proj's take matches it; a padded step's live
    slice (the buffer outruns the live rows) keeps the store."""
    if not _C1_DROP_BF16 or not _c1_publishes(out):
        return False
    whole = (
        out.dim() == 2
        and out.is_contiguous()
        and out.storage_offset() == 0
        and out.untyped_storage().nbytes() == out.numel() * out.element_size()
    )
    if whole and "drop_logged" not in _NORMQ_STATE:
        _NORMQ_STATE["drop_logged"] = True
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_COMBINE_DROP_BF16 engaged: the sparse prefill combine skips the BF16 "
            f"attention output store; o_proj takes the MXFP8 copy (rows {out.shape[0]})"
        )
    return whole


def _c1_materialize(materialize: Callable[[], None]) -> None:
    """Write a dropped BF16 combine output for a reader that misses its MXFP8 copy (DEP2CTX_COMBINE_DROP_BF16)."""
    materialize()
    _C1_MATERIALIZED[0] += 1
    if _C1_MATERIALIZED[0] == 1 or _C1_MATERIALIZED[0] % 1000 == 0:
        logger.warning(
            f"[M3.1 dep2ctx] DEP2CTX_COMBINE_DROP_BF16: a reader missed the MXFP8 copy; the BF16 attention output was "
            f"written for it ({_C1_MATERIALIZED[0]} so far; exact, but the drop saved nothing there)"
        )


def _combine_mxfp8_sink(
    out: torch.Tensor,
    xq: torch.Tensor,
    xsf: torch.Tensor,
    materialize: Optional[Callable[[], None]] = None,
) -> None:
    """DEP2CTX_COMBINE_MXFP8 (#7): the NVFP4 sparse prefill's MSA combine also wrote its bf16 output as MXFP8 (bitwise
    mxfp8_quantize(out, True)); publish the pair so o_proj (an MXFP8 linear) skips its own quantize. o_proj takes it
    only when its input is this very output (storage + shape); anything else falls back to the stock quantize. Eager
    forwards only: under a breakable / compiled graph o_proj replays without running this Python hand-off."""
    if (
        out.shape[0] < _NORMQ_MIN_ROWS
        or is_torch_compiling()
        or is_in_breakable_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):  # the same test as _c1_publishes
        if (
            materialize is not None
        ):  # the BF16 store was skipped (_combine_mxfp8_drop_bf16): write it
            _c1_materialize(materialize)
        return
    if "combine_logged" not in _NORMQ_STATE:
        _NORMQ_STATE["combine_logged"] = True
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_COMBINE_MXFP8 engaged: the sparse prefill combine publishes o_proj's MXFP8 input "
            f"(rows {out.shape[0]} >= {_NORMQ_MIN_ROWS})"
        )
    if "combine_take_logged" not in _NORMQ_STATE:
        _NORMQ_STATE["combine_src"] = (out.data_ptr(), tuple(out.shape))
    _NORMQ_SLOT.publish(out, xq, xsf, materialize=materialize)


_combine_mxfp8_sink.min_rows = _C1_MIN_ROWS
_combine_mxfp8_sink.drop_bf16 = _combine_mxfp8_drop_bf16


def _c1_allocate(out2d: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The combine's MXFP8 pair for C1 (msa_prefill's sink ``allocate`` hook), or None to decline the sink.

    Under a breakable / compiled graph the publish above declines anyway (o_proj replays its own quantize), so the
    combine must not write the copy at all: without this hook msa_prefill allocates a fresh pair and the combine pays
    the MXFP8 epilogue for nothing on a graphed step of at least _C1_MIN_ROWS live rows. Eager steps get exactly the
    pair msa_prefill builds without the hook (E4M3 data, zeroed swizzled scales over the 128-row padded tail)."""
    if is_torch_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    if is_in_breakable_cuda_graph():
        # DEP2CTX_C1_BCG: a bucket that takes the copy writes it into the persistent pair.
        if _c1_bcg_bucket() is None:
            return None
        return _c1_bcg_pair(int(out2d.shape[0]), int(out2d.shape[1]), out2d.device)
    rows, width = int(out2d.shape[0]), int(out2d.shape[1])
    rows_padded = (rows + 127) // 128 * 128
    return (
        torch.empty((rows, width), dtype=torch.uint8, device=out2d.device),
        torch.zeros((rows_padded * width // 32,), dtype=torch.uint8, device=out2d.device),
    )


# Combine MXFP8 copy declined under large prefill graphs (rides on DEP2CTX_COMBINE_MXFP8 and DEP2CTX_BCG_LARGE, which
# first graphs a step of >= 12288 rows): a graphed step's combine writes no MXFP8 copy. Bitwise: o_proj quantizes the
# same bf16 output itself.
if _BCG_LARGE:
    _combine_mxfp8_sink.allocate = _c1_allocate

# DEP2CTX_C1_BCG (C1 under breakable graphs): a graphed step's sparse prefill combine also writes o_proj's MXFP8 input,
# into one persistent pair sized for the largest prefill bucket (allocated once, outside every graph pool). The
# attention break knows its graph's bucket from the static output it fills (_attn_core_inplace); a bucket of at least
# _C1_MIN_ROWS rows takes the copy on its capture and on every replay alike (the choice depends on the bucket, never on
# the live rows), so the o_proj GEMM captured on the published pair reads what the replayed combine just wrote. Live
# rows are bit for bit the stock o_proj's: the pair holds mxfp8_quantize(out, True) of the live rows and every GEMM
# output row depends only on its own activation row. Pad rows read the pair's earlier finite bytes where the stock
# path quantizes the output's earlier pad rows; nothing reads either for its own result.
# Kill switch TRTLLM_M31_DISABLE_DEP2CTX_C1_BCG=1.
_C1_BCG = _BCG_LARGE and not m31_disabled("DEP2CTX_C1_BCG")
_C1_BCG_STATE: Dict[str, Any] = {"output": None}


def _c1_bcg_bucket() -> Optional[torch.Tensor]:
    """The static output of the attention break running now when its bucket takes DEP2CTX_C1_BCG, else None."""
    out = _C1_BCG_STATE["output"]
    if (
        out is None
        or not _C1_BCG
        or not is_in_breakable_cuda_graph()
        or out.dim() != 2
        or out.shape[0] < _C1_MIN_ROWS
    ):
        return None
    return out


def _c1_bcg_pair(
    rows: int, width: int, device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Views of the persistent pair over ``rows`` rows (E4M3 data [rows, width], swizzled scales over pad128(rows)),
    or None when it cannot hold them (the same answer for a bucket at capture and at replay)."""
    cap = max(_ppr.DEP2CTX_BCG_GRID)
    if "q" not in _C1_BCG_STATE:
        cap_padded = (cap + 127) // 128 * 128
        _C1_BCG_STATE["q"] = torch.zeros((cap * width,), dtype=torch.uint8, device=device)
        _C1_BCG_STATE["sf"] = torch.zeros(
            (cap_padded * width // 32,), dtype=torch.uint8, device=device
        )
        _C1_BCG_STATE["width"] = width
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_C1_BCG persistent MXFP8 pair: {cap} x {width} "
            f"({(cap * width + cap_padded * width // 32) / 2**20:.0f} MiB)"
        )
    # One pair per process: a model on another device declines it (and keeps the stock path).
    if width != _C1_BCG_STATE["width"] or _C1_BCG_STATE["q"].device != device or rows > cap:
        return None
    rows_padded = (rows + 127) // 128 * 128
    return (
        _C1_BCG_STATE["q"][: rows * width].view(rows, width),
        _C1_BCG_STATE["sf"][: rows_padded * width // 32],
    )


class _C1BcgSink:
    """msa_prefill's MXFP8 sink for DEP2CTX_COMBINE_MXFP8 with DEP2CTX_C1_BCG: eager calls are _combine_mxfp8_sink's;
    a graphed bucket that takes the copy ignores the live-row minimum and, on its capture, publishes the persistent
    pair keyed by the static output (the replays rewrite the pair; the captured o_proj reads it)."""

    @property
    def min_rows(self) -> int:
        return 0 if _c1_bcg_bucket() is not None else _C1_MIN_ROWS

    def allocate(self, out2d: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return _c1_allocate(out2d)

    def __call__(self, out: torch.Tensor, xq: torch.Tensor, xsf: torch.Tensor) -> None:
        full = _c1_bcg_bucket()
        if full is None:
            _combine_mxfp8_sink(out, xq, xsf)
            return
        if get_current_replay_token() is not None:
            return
        pair = _c1_bcg_pair(int(full.shape[0]), int(full.shape[1]), full.device)
        if pair is None:
            return
        rows = int(full.shape[0])
        noted = _C1_BCG_STATE.setdefault("noted", set())
        if rows not in noted:
            noted.add(rows)
            logger.info(
                f"[M3.1 dep2ctx] DEP2CTX_C1_BCG taken: bucket {rows} captures o_proj on the combine's MXFP8 pair"
            )
        _NORMQ_SLOT.publish(full, pair[0].view(torch.float8_e4m3fn), pair[1])


_C1_SINK = _C1BcgSink() if _C1_BCG else _combine_mxfp8_sink


# CTXNORM_LB2: the CTX residual add + RMSNorm + MXFP8 kernel launches at two CTAs per SM (a two-CTA launch bound) where
# hidden / 8 <= 768, the same outputs bit for bit. Default ON; kill switch TRTLLM_M31_DISABLE_CTXNORM_LB2=1 (one CTA per
# SM), passed to the op as its lb2 argument.
_CTXNORM_LB2 = not m31_disabled("CTXNORM_LB2")


def _normq_op():
    """family_ops().minimax_m31_ctx_residual_rmsnorm_mxfp8, or None when the build lacks it. Decided once."""
    if "op" not in _NORMQ_STATE:
        op = getattr(family_ops(), "minimax_m31_ctx_residual_rmsnorm_mxfp8", None)
        _NORMQ_STATE["op"] = op
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_NORMQ prefill boundary norm + MXFP8: "
            f"{'engaged' if op is not None else 'not taken: the op is not built'} (rows >= {_NORMQ_MIN_ROWS})"
        )
    return _NORMQ_STATE["op"]


# DEP2CTX_NORMQ_F32 (the router-input fold): at the pre-MoE boundary the NORMQ op variant
# minimax_m31_ctx_residual_rmsnorm_mxfp8_f32 also writes the normed rows as float32 (bit for bit norm_out.float()), and
# the router consumes them, so neither the router input cast (GATE_CAST / aten) nor a second read of the rows runs.
# The router GEMM's input is bitwise unchanged. A build without the op keeps the plain NORMQ op and the gate's own cast.
# Default ON; kill switch TRTLLM_M31_DISABLE_DEP2CTX_NORMQ_F32=1.
_NORMQ_F32 = not m31_disabled("DEP2CTX_NORMQ_F32")
_NORMQ_F32_STATE: Dict[str, Any] = {}


def _normq_f32_op():
    if "op" not in _NORMQ_F32_STATE:
        op = (
            getattr(family_ops(), "minimax_m31_ctx_residual_rmsnorm_mxfp8_f32", None)
            if _NORMQ_F32
            else None
        )
        _NORMQ_F32_STATE["op"] = op
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_NORMQ_F32 router fp32 input from the boundary: "
            f"{'engaged' if op is not None else 'not taken (switched off or the op is not built)'}"
        )
    return _NORMQ_F32_STATE["op"]


# DEP2CTX_NORMQ_V2: the boundary runs minimax_m31_ctx_boundary_normq (ctxBoundaryNormQuantSm107: the
# served kernel's arithmetic with two CTAs per SM, a cp.async.bulk ring three rows deep and coalesced scale-tile
# stores) instead of minimax_m31_ctx_residual_rmsnorm_mxfp8 / its _f32 fold: every output bit for bit the served op's
# (GPU UT test_modeling_v2_m31_dep2ctx_normq_v2_gpu.py). sm_10x builds that carry the op; else the served op.
# Default ON; kill switch TRTLLM_M31_DISABLE_DEP2CTX_NORMQ_V2=1 (the served op).
_NORMQ_V2 = not m31_disabled("DEP2CTX_NORMQ_V2")
# DEP2CTX_NORMQ_STAGE: where the routed experts run on the MegaMoE-local backend and the router takes the
# boundary's FP32 copy, the pre-MoE boundary also writes the experts' NVFP4 input (bitwise fp4_quantize(normed,
# fc31_input_scale, 16, False, False), the stage's own quantize) and the stage's padded tail straight into the bucket
# this call launches; after the router GEMM, minimax_m31_route_stage writes the live rows' ids / weights and the
# backend takes the seat (its try_fused_route_quant: no route-quant launch). The stage is bit for bit what
# noaux_tc_fp4_quantize_stage_out writes. Default ON; kill switch TRTLLM_M31_DISABLE_DEP2CTX_NORMQ_STAGE=1 (the
# backend's own route-quant).
_NORMQ_STAGE = not m31_disabled("DEP2CTX_NORMQ_STAGE")
# DEP2CTX_NORMQ_STAGE_DROP_NORM: a staged boundary leaves the BF16 norm out (-2 B / element) when every reader takes a
# quantized copy: the router the FP32 rows, the routed experts the stage, the shared expert the MXFP8 slot. A reader
# that misses its copy materializes the norm first (minimax_m31_ctx_boundary_normq_out on the summed rows: the same
# bits). Default ON; kill switch TRTLLM_M31_DISABLE_DEP2CTX_NORMQ_STAGE_DROP_NORM=1 (always written). With
# DEP2CTX_ROUTER_BF16S on (the default) the router reads the BF16 norm itself, so the plan keeps the norm written; the
# drop engages under TRTLLM_M31_DISABLE_DEP2CTX_ROUTER_BF16S=1.
_NORMQ_STAGE_DROP_NORM = not m31_disabled("DEP2CTX_NORMQ_STAGE_DROP_NORM")
# The ring kernel's launch choice, passed to every minimax_m31_ctx_boundary_normq* call (each form writes the same
# bits). CBNQ_P2: the two-chunk kernel at two CTAs per SM on sm_107, kill switch TRTLLM_M31_DISABLE_CBNQ_P2=1 (the
# one-chunk kernel). CBNQ_2CTA: the one-chunk ring at the requested depth, kill switch TRTLLM_M31_DISABLE_CBNQ_2CTA=1
# (three stages, no carveout attribute: only differs when a deeper ring is requested). CBNQ_TOPK_DUAL: the top-k fold
# at two row groups per CTA on sm_107, kill switch TRTLLM_M31_DISABLE_CBNQ_TOPK_DUAL=1 (one row group). Ring depths:
# ctx_boundary_plan's defaults.
_CBNQ_CHOICE: Dict[str, Any] = dict(
    p2=not m31_disabled("CBNQ_P2"),
    two_cta=not m31_disabled("CBNQ_2CTA"),
    stages=_cbplan.L6_STAGES,
    p2_stages=_cbplan.P2_STAGES,
)
_CBNQ_TOPK_DUAL = not m31_disabled("CBNQ_TOPK_DUAL")
_NORMQ_V2_STATE: Dict[str, Any] = {}


def _normq_v2_ops() -> Tuple[Any, Any]:
    """(minimax_m31_ctx_boundary_normq, minimax_m31_ctx_boundary_normq_out), each None when switched off, not built,
    or not an sm_10x device. Decided once."""
    if "ops" not in _NORMQ_V2_STATE:
        usable = _NORMQ_V2 and get_sm_version() // 10 == 10
        alloc = getattr(native_ops, "minimax_m31_ctx_boundary_normq", None) if usable else None
        out = getattr(native_ops, "minimax_m31_ctx_boundary_normq_out", None) if usable else None
        _NORMQ_V2_STATE["ops"] = (alloc, out)
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_NORMQ_V2 ring boundary kernel: "
            f"{'engaged' if alloc is not None else 'not taken (switched off, not sm_10x, or the op is not built)'}"
        )
    return _NORMQ_V2_STATE["ops"]


def _normq_v2_op():
    return _normq_v2_ops()[0]


def _normq_v2_out_op():
    return _normq_v2_ops()[1]


# NOAUX_TC_FP32: the route stage's routing weights take the FP32 form (the FP64 expression's values without FP64
# instructions on the common path). Kill switch TRTLLM_M31_DISABLE_NOAUX_TC_FP32=1 (the FP64 expression for every
# weight), passed as the op's fp32_weight argument.
_NOAUX_TC_FP32 = not m31_disabled("NOAUX_TC_FP32")


def _route_stage_op():
    """native_ops.minimax_m31_route_stage (DEP_ROUTER_STOCK_BOUNDARY's route-only stage op), or None."""
    if "route_stage" not in _NORMQ_V2_STATE:
        _NORMQ_V2_STATE["route_stage"] = getattr(native_ops, "minimax_m31_route_stage", None)
    return _NORMQ_V2_STATE["route_stage"]


def _takes_published_mxfp8(mlp: nn.Module) -> bool:
    """Whether a shared-expert MLP's first read of its input is its gate_up _NormqMXFP8LinearMethod (which takes the
    published MXFP8 pair): the unfused, unsplit GatedMLP path."""
    gate_up = getattr(mlp, "gate_up_proj", None)
    if type(getattr(gate_up, "quant_method", None)) is not _NormqMXFP8LinearMethod:
        return False
    if not isinstance(mlp, GatedMLP) or getattr(mlp, "split_gate_up", False):
        return False
    for name in ("_can_fuse_gate_up_swiglu_fp4out", "_can_fuse_gate_up_swiglu"):
        fused = getattr(mlp, name, None)
        if not callable(fused) or fused():
            return False
    return True


@dataclasses.dataclass
class _NormqStage:
    """DEP2CTX_NORMQ_STAGE: one MegaMoE-local stage a pre-MoE boundary writes into (the bucket's buffers)."""

    rows: int
    top_k: int
    launch_max_t: int
    chunk_max: Optional[int]
    bias: torch.Tensor
    scale: torch.Tensor
    routed_scaling_factor: float
    ids: (
        torch.Tensor
    )  # [launch rows, >= top_k] int32 (the stage tail is written through its first dimension)
    weights: torch.Tensor  # [launch rows, top_k] fp32 view
    act: torch.Tensor  # [rows, hidden / 2] NVFP4 rows
    act_sf: torch.Tensor  # [rows, hidden / 16] linear scales
    result: Tuple[torch.Tensor, ...]  # try_fused_route_quant's return value for the seat
    drop_norm: bool = False
    normed: Optional[torch.Tensor] = None

    @staticmethod
    def build(
        bufs, *, rows, hidden, top_k, launch_max_t, chunk_max, bias, scale, routed_scaling_factor
    ):
        """The stage over ``bufs`` (the bucket's _MegaMoeBuffers), or None when its layout does not fit the boundary
        op (byte rows of hidden / 2 and hidden / 16 with 4- / 16-byte aligned rows, int32 / fp32 ids / weights with
        unit column stride)."""
        act, act_sf, ids = bufs.activation, bufs.activation_sf, bufs.topk_idx_local
        weights = bufs.topk_weights[:, :top_k]
        ok = (
            act.dim() == 2
            and act.element_size() == 1
            and act.shape[0] >= rows
            and act.shape[1] == hidden // 2
            and act.stride(1) == 1
            and act.stride(0) % 4 == 0
            and act.data_ptr() % 4 == 0
            and act_sf.dim() == 2
            and act_sf.element_size() == 1
            and act_sf.shape[0] >= rows
            and act_sf.shape[1] == hidden // 16
            and act_sf.stride(1) == 1
            and act_sf.stride(0) % 16 == 0
            and act_sf.data_ptr() % 16 == 0
            and ids.dtype == torch.int32
            and ids.dim() == 2
            and ids.stride(1) == 1
            and ids.shape[1] >= top_k
            and weights.dtype == torch.float32
            and weights.stride(1) == 1
            and ids.shape[0] >= rows
            and weights.shape[0] >= rows
            and isinstance(scale, torch.Tensor)
            and scale.dtype == torch.float32
            and scale.numel() >= 1
        )
        if not ok:
            return None
        return _NormqStage(
            rows=rows,
            top_k=top_k,
            launch_max_t=launch_max_t,
            chunk_max=chunk_max,
            bias=bias,
            scale=scale,
            routed_scaling_factor=routed_scaling_factor,
            ids=ids,
            weights=weights,
            act=act[:rows],
            act_sf=act_sf[:rows],
            result=(ids[:rows], weights[:rows], act[:rows], act_sf[:rows]),
        )


class _DroppedNorms:
    """BF16 norms a DEP2CTX_NORMQ_STAGE boundary left out, for the one MoE call that follows. ``materialize(x)``
    writes the norm into x when x is one of them (from the boundary's summed rows, bit for bit) the first time a reader
    misses its quantized copy; a later reader on another stream waits for that write. The MoE call clears them."""

    def __init__(self):
        self.entries: Dict[int, list] = {}

    def add(self, normed: torch.Tensor, summed: torch.Tensor, norm: RMSNorm) -> None:
        self.entries[normed.data_ptr()] = [normed, summed, norm.weight, norm.variance_epsilon, None]

    def materialize(self, x) -> None:
        if not self.entries or not isinstance(x, torch.Tensor):
            return
        entry = self.entries.get(x.data_ptr())
        if entry is None:
            return
        normed, summed, weight, eps, done = entry
        if done is not None:
            torch.cuda.current_stream(normed.device).wait_event(done)
            return
        _normq_v2_out_op()(summed, None, weight, eps, normed, None, None, **_CBNQ_CHOICE)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(normed.device))
        entry[4] = done
        if "materialized" not in _NORMQ_STATE:
            _NORMQ_STATE["materialized"] = True
            logger.info(
                f"[M3.1 dep2ctx] DEP2CTX_NORMQ_STAGE_DROP_NORM: a reader missed its quantized copy; the BF16 norm was "
                f"written for it (rows {normed.shape[0]}; logged once)"
            )

    def clear(self) -> None:
        self.entries.clear()


_NORMQ_DROPPED = _DroppedNorms()


def _normq_stage_boundary(
    reduced: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
    moe: "MiniMaxM3MoE",
    all_rank_num_tokens: Optional[List[int]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """DEP2CTX_NORMQ_STAGE: the pre-MoE boundary that also stages the routed experts' input (see _NORMQ_STAGE); returns
    (normed, mxfp8, scales) as the NORMQ boundary does, or None when the stage does not apply."""
    stage = moe.normq_stage_plan(reduced, all_rank_num_tokens)
    if stage is None:
        return None
    rows = reduced.shape[0]
    normed = torch.empty_like(reduced)
    xq = torch.empty(reduced.shape, dtype=torch.float8_e4m3fn, device=reduced.device)
    xsf = torch.empty(
        (_cbplan.num_quads(rows) * 4 * (reduced.shape[1] // 32),),
        dtype=torch.uint8,
        device=reduced.device,
    )
    # DEP2CTX_ROUTER_BF16S: the router reads the BF16 norm (the plan then kept it: drop_norm is False), so no FP32 copy
    # (flavor <add, norm, mx, !f32, fp4>).
    bf16s = _router_reads_bf16(moe)
    f32 = None if bf16s else torch.empty(reduced.shape, dtype=torch.float32, device=reduced.device)
    _normq_v2_out_op()(
        reduced,
        residual.contiguous(),
        norm.weight,
        norm.variance_epsilon,
        None if stage.drop_norm else normed,
        xq,
        xsf,
        f32,
        stage.scale,
        stage.act,
        stage.act_sf,
        stage.ids,
        stage.weights,
        stage.top_k,
        **_CBNQ_CHOICE,
    )
    stage.normed = normed
    moe._normq_stage = stage
    if f32 is not None:
        moe._router_input_f32 = (normed, f32)
    if stage.drop_norm:
        _NORMQ_DROPPED.add(normed, reduced, norm)
    if "stage_logged" not in _NORMQ_STATE:
        _NORMQ_STATE["stage_logged"] = True
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_NORMQ_STAGE boundary: {'BF16 norm' if bf16s else 'FP32'} + MXFP8 + the MegaMoE "
            f"stage's NVFP4 rows and tail in one pass (rows {rows}, bucket {stage.launch_max_t}, BF16 norm "
            f"{'left out' if stage.drop_norm else 'written'})"
        )
    return normed, xq, xsf


def _normq_boundary(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
    moe: Optional["MiniMaxM3MoE"] = None,
    all_rank_num_tokens: Optional[List[int]] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(normed, updated residual) of the boundary as the fused residual add, RMSNorm and MXFP8 quantize of this rank's
    rows (the pair waits in _NORMQ_SLOT), or None for the stock residual add + RMSNorm (switched off, a short or
    compiled / captured step, the op not built). Attention DP: the rows are this rank's own, so nothing is reduced.
    At the pre-MoE boundary (moe set) ``all_rank_num_tokens`` is the MoE call's, for DEP2CTX_NORMQ_STAGE's bucket."""
    if (
        not _NORMQ
        or hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or hidden_states.shape[0] < _NORMQ_MIN_ROWS
        or hidden_states.dtype != torch.bfloat16
        or is_torch_compiling()
        or torch.compiler.is_compiling()
        or _dep2ctx_capture_blocked()
    ):
        return None
    op = _normq_op()
    if op is None:
        return None
    _bcg_note("NORMQ", hidden_states)
    # The op adds the residual into its first argument in place, which then holds the updated residual.
    reduced = hidden_states
    f32_op = _normq_f32_op() if moe is not None else None
    # DEP2CTX_ROUTER_BF16S: the router reads the BF16 norm, so no FP32 copy is written for it.
    write_f32 = f32_op is not None and not _router_reads_bf16(moe)
    # DEP2CTX_NORMQ_STAGE: on the MegaMoE-local backend the same pass stages the routed experts' NVFP4 input.
    staged = (
        _normq_stage_boundary(reduced, residual, norm, moe, all_rank_num_tokens)
        if f32_op is not None and _NORMQ_STAGE
        else None
    )
    # DEP2CTX_NORMQ_FP4: at the pre-MoE boundary the same pass writes the routed experts' NVFP4 input, bitwise
    # fp4_quantize(normed, fc31_input_scale, 16, False, False), which is their own quantize_input.
    fp4_scale = (
        moe.fp4_input_scale(hidden_states.shape[0])
        if moe is not None and _NORMQ_FP4 and staged is None
        else None
    )
    v2 = _normq_v2_op() if staged is None else None
    if staged is not None:
        outs = staged
    elif v2 is not None:
        # DEP2CTX_NORMQ_V2: the same outputs from the ring kernel (the FP32 copy only at the pre-MoE boundary).
        outs = v2(
            reduced,
            residual.contiguous(),
            norm.weight,
            norm.variance_epsilon,
            True,
            write_f32,
            fp4_scale,
            **_CBNQ_CHOICE,
        )
        if write_f32:
            moe._router_input_f32 = (outs[0], outs[5])
    elif write_f32:
        outs = f32_op(
            reduced,
            residual.contiguous(),
            norm.weight,
            norm.variance_epsilon,
            fp4_scale,
            lb2=_CTXNORM_LB2,
        )
        moe._router_input_f32 = (outs[0], outs[5])
    elif fp4_scale is None:
        outs = op(
            reduced,
            residual.contiguous(),
            norm.weight,
            norm.variance_epsilon,
            True,
            lb2=_CTXNORM_LB2,
        )
    else:
        outs = op(
            reduced,
            residual.contiguous(),
            norm.weight,
            norm.variance_epsilon,
            True,
            fp4_scale,
            lb2=_CTXNORM_LB2,
        )
    normed, xq, xsf = outs[0], outs[1], outs[2]
    if fp4_scale is not None:
        moe._prequantized_fp4 = (
            normed,
            Fp4QuantizedTensor(
                outs[3], outs[4], is_sf_swizzled=False, unquantized_hidden_states=normed
            ),
        )
        if "fp4_logged" not in _NORMQ_STATE:
            _NORMQ_STATE["fp4_logged"] = True
            logger.info(
                f"[M3.1 dep2ctx] DEP2CTX_NORMQ_FP4 NVFP4 MoE input from the boundary (rows {hidden_states.shape[0]})"
            )
    _NORMQ_SLOT.publish(normed, xq, xsf)
    if "logged" not in _NORMQ_STATE:
        _NORMQ_STATE["logged"] = True
        logger.info(f"[M3.1 dep2ctx] DEP2CTX_NORMQ boundary taken (rows {hidden_states.shape[0]})")
    return normed, reduced


def _topk_fold_op():
    """native_ops.minimax_m31_ctx_boundary_normq_topk, or None when switched off, not built, or not an sm_10x
    device. Decided once."""
    if "op" not in _TOPK_FOLD_STATE:
        usable = _TOPK_FOLD and get_sm_version() // 10 == 10
        op = getattr(native_ops, "minimax_m31_ctx_boundary_normq_topk", None) if usable else None
        _TOPK_FOLD_STATE["op"] = op
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_TOPK_FOLD top-k reduce in the next-layer boundary: "
            f"{'engaged' if op is not None else 'not taken (switched off, not sm_10x, or the op is not built)'}"
        )
    return _TOPK_FOLD_STATE["op"]


def _topk_fold_decline_reason(
    layer: "MiniMaxM3DecoderLayer", x: torch.Tensor, spec_metadata: Optional[SpecMetadata]
) -> Optional[str]:
    """Why ``layer.forward_MoE`` must not arm DEP2CTX_TOPK_FOLD for MoE input ``x``, or None (ctx_boundary_plan's
    order; the plane checks follow once the routed output is known)."""
    if not _TOPK_FOLD:
        return "switched off"
    moe = layer.block_sparse_moe
    return _cbplan.topk_fold_decline_reason(
        enabled=True,
        op_built=_topk_fold_op() is not None,
        normq=_NORMQ and _normq_op() is not None and _normq_v2_op() is not None,
        feeds_next_layer=bool(layer._normq_feeds_mxfp8) and layer.next_layer_layernorm is not None,
        fused_comm=getattr(moe, "allreduce", None) is None
        and bool(getattr(moe, "routed_output_is_global", False)),
        # DEP2CTX_TOPK_CAPTURE: a capture layer folds too and captures the fold's updated residual.
        capture_layer=not _TOPK_CAPTURE
        and spec_metadata is not None
        and spec_metadata.is_layer_capture(layer.layer_idx),
        x_ok=x.dim() == 2 and x.is_contiguous() and x.dtype == torch.bfloat16,
        rows=int(x.shape[0]) if x.dim() >= 1 else 0,
        min_rows=_NORMQ_MIN_ROWS,
        compiling=is_torch_compiling() or torch.compiler.is_compiling(),
        capture_blocked=_dep2ctx_capture_blocked(),
    )


# DEP2CTX_TOPK_CAPTURE: a DSpark / Eagle capture layer keeps DEP2CTX_TOPK_FOLD. The capture stores the MoE output +
# the residual, which is bit for bit the fold's updated residual (the same bf16 MoE output rows, one fp32 add with the
# residual, one rounding: the fold's check compares it with add_into + the boundary's residual add), so the capture
# reads that instead of keeping the separate top-k reduce + shared add, the add and the unfused boundary.
# Kill switch TRTLLM_M31_DISABLE_DEP2CTX_TOPK_CAPTURE=1.
_TOPK_CAPTURE = not m31_disabled("DEP2CTX_TOPK_CAPTURE")


def _topk_capture(spec_metadata: SpecMetadata, layer_idx: int, residual: torch.Tensor) -> None:
    """DEP2CTX_TOPK_CAPTURE: capture the fold's updated residual (see _TOPK_CAPTURE)."""
    rows = int(residual.shape[0])
    noted = _TOPK_FOLD_STATE.setdefault("capture_noted", set())
    if rows not in noted:
        noted.add(rows)
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_TOPK_CAPTURE taken: layer {layer_idx} captures the fold's updated residual "
            f"(rows {rows})"
        )
    spec_metadata.maybe_capture_hidden_states(layer_idx, residual, None)


def _topk_fold_arm(
    layer: "MiniMaxM3DecoderLayer", x: torch.Tensor, spec_metadata: Optional[SpecMetadata]
) -> bool:
    """Arm _TOPK_FOLD_SLOT for ``layer``'s MoE call on ``x`` when the fold may take it; logs a decline once."""
    reason = _topk_fold_decline_reason(layer, x, spec_metadata)
    if reason is not None:
        _topk_fold_log("not taken", reason)
        return False
    _TOPK_FOLD_SLOT.arm()
    return True


def _topk_fold_hand_off(
    layer: "MiniMaxM3DecoderLayer", parked, moe_out: torch.Tensor, residual: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """After an armed MoE call: the fused next-layer boundary of the parked reduce, or None for the stock path (the
    parked reduce, if any, has then run, so ``moe_out`` holds the MoE output as without the fold)."""
    if parked is None:
        _topk_fold_log(
            "not taken",
            "nothing parked (the routed launch reduced its own top-k, or its plane declined)",
        )
        return None
    deferred, shared = parked
    if moe_out is not shared:
        # Not reachable today (the tail add's buffer is the MoE output): reduce in place, keep the stock path.
        _topk_fold_log("not taken", "the MoE returned another tensor than the parked shared rows")
        deferred.add_into(shared)
        return None
    fused = _normq_topk_boundary(deferred, shared, residual, layer.next_layer_layernorm)
    if fused is None:
        deferred.add_into(shared)
    return fused


def _normq_topk_boundary(
    deferred: MegaMoEDeferredTopkReduce,
    shared: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """DEP2CTX_TOPK_FOLD: (normed, updated residual) of the next-layer boundary with the parked top-k reduce folded in
    (the MXFP8 pair waits in _NORMQ_SLOT, as after _normq_boundary), or None when the op rejects its arguments (the
    caller then reduces and runs the stock boundary; the fold stays off for the process, since the plane checks have
    passed and a rejection is the device's or the build's). ``shared`` holds the shared-expert rows and becomes the
    updated residual in place."""
    op = _topk_fold_op()
    if op is None:
        return None
    _bcg_note("TOPK_FOLD", shared)
    residual = residual.contiguous()
    try:
        normed, xq, xsf = op(
            shared,
            deferred.plane,
            deferred.scores,
            residual,
            norm.weight,
            norm.variance_epsilon,
            True,
            get_env_enable_pdl(),
            dual=_CBNQ_TOPK_DUAL,
        )
    except RuntimeError as err:
        # Argument checks run before the launch: shared is still unreduced. Latch the fold off (the decline then
        # names it) instead of raising again at every MoE layer of every step.
        _TOPK_FOLD_STATE["op"] = None
        _topk_fold_log(
            "not taken", f"the op rejected its arguments, fold off for the process ({err})"
        )
        return None
    _NORMQ_SLOT.publish(normed, xq, xsf)
    _topk_fold_log(
        "taken", f"top-k reduce + shared add + next-layer boundary (rows {shared.shape[0]})"
    )
    return normed, shared


# DEP2CTX_MXFP8_CUBLASLT (pinned cuBLASLt MXFP8 GEMMs): the dep2ctx prefill MXFP8 GEMMs (o_proj, dense MLP, shared
# expert) run on cuBLASLt with a PINNED algorithm (family_ops().minimax_m31_mxfp8_cublaslt_gemm). On sm_103 it
# is bitwise the CUTLASS mxfp8_mxfp8_gemm at every model shape. On sm_107 it is not bitwise CUTLASS: it equals
# torch._scaled_mm, and vs CUTLASS ~0.002% of outputs differ by > 2 bf16 ulp, up to ~30k ulp on near-zero outputs. Pins
# are per (N, K) and M band; each
# recorded configuration has splitk_num 1, so its bits are row-local and the same at any M it serves. A band is taken
# only after a first-use self-check: at both band edges, on the weights of the first module that reaches the band (the
# result is cached per (N, K, band), not per module) and random activations whose 32-wide blocks span 2^-12..2^12, the
# pinned GEMM runs and equals its reference bit for bit (the CUTLASS op on sm_103, torch._scaled_mm on sm_107)
# (cublasLtMatmulAlgoCheck is not the gate: it rejects every pin on sm_103 and sm_107, though the pinned launch runs).
# Otherwise, and for every shape outside the table (decode, other models), the CUTLASS op runs. Eager
# prefill steps only. Kill switch: TRTLLM_M31_DISABLE_DEP2CTX_MXFP8_CUBLASLT=1.
_MXFP8_CUBLASLT = not m31_disabled("DEP2CTX_MXFP8_CUBLASLT")
# (N, K) -> ((M_lo, M_hi, (algo_id, tile_id, stages_id, splitk_num, reduction_scheme, cta_swizzling, custom_option,
# inner_shape_id, cluster_shape_id)), ...), recorded on this arch (SM107, cuBLASLt 130800): heuristic-0 at M 1160 / 8192
# / 16384.
# Attention DP runs every GEMM at full width: o_proj (6144, 8192), dense gate_up (24576, 6144) / down (6144, 12288),
# shared expert gate_up (6144, 6144) / down (6144, 3072). A configuration is per (N, K), whatever the module; a shape
# or M outside the table runs the CUTLASS op (shared down below 4096 rows: the pin is not faster there).
_MXFP8_CUBLASLT_PINS: Dict[Tuple[int, int], Tuple[Tuple[int, int, Tuple[int, ...]], ...]] = {
    # shared expert gate_up (three bands)
    (6144, 6144): (
        (1024, 4095, (66, 513, 36, 1, 0, 0, 3, 0, 11)),
        (4096, 12287, (66, 513, 36, 1, 0, 0, 1, 0, 6)),
        (12288, 16384, (66, 513, 36, 1, 0, 0, 1, 0, 9)),
    ),
    # o_proj
    (6144, 8192): (
        (1024, 4095, (66, 513, 36, 1, 0, 0, 3, 0, 11)),
        (4096, 12287, (66, 513, 36, 1, 0, 0, 1, 0, 6)),
        (12288, 16384, (66, 513, 36, 1, 0, 0, 1, 0, 9)),
    ),
    # shared expert down
    (6144, 3072): (
        (4096, 12287, (66, 513, 36, 1, 0, 0, 1, 0, 6)),
        (12288, 16384, (66, 513, 36, 1, 0, 0, 1, 0, 9)),
    ),
    # dense gate_up
    (24576, 6144): (
        (1024, 4095, (66, 513, 36, 1, 0, 0, 1, 0, 6)),
        (4096, 12287, (66, 513, 36, 1, 0, 0, 3, 0, 9)),
        (12288, 16384, (66, 513, 36, 1, 0, 0, 3, 0, 9)),
    ),
    # dense down
    (6144, 12288): (
        (1024, 4095, (66, 513, 36, 1, 0, 0, 3, 0, 11)),
        (4096, 12287, (66, 513, 36, 1, 0, 0, 0, 0, 6)),
        (12288, 16384, (66, 513, 36, 1, 0, 0, 3, 0, 9)),
    ),
}
_MXFP8_CUBLASLT_STATE: Dict[Any, Any] = {}


def _mxfp8_cublaslt_op():
    """family_ops().minimax_m31_mxfp8_cublaslt_gemm from the build; None when it is not built."""
    if "op" not in _MXFP8_CUBLASLT_STATE:
        _MXFP8_CUBLASLT_STATE["op"] = getattr(family_ops(), "minimax_m31_mxfp8_cublaslt_gemm", None)
    return _MXFP8_CUBLASLT_STATE["op"]


# The self-check reference: "cutlass" (exact, sm_103) or "scaled_mm" (sm_107, not bitwise the CUTLASS op).
_MXFP8_CUBLASLT_REF = "scaled_mm"


def _mxfp8_cublaslt_self_check(
    module: Linear, band: Tuple[int, int, Tuple[int, ...]]
) -> Optional[str]:
    """None when, at both band edges on this module's weights, the pinned GEMM is deterministic and equals its
    reference bit for bit, else why not. The reference is the CUTLASS op on sm_103 (exact: the stock path's bits); on
    sm_107 the pinned cuBLASLt GEMM is not bitwise CUTLASS (a different fp32 accumulation; near-zero outputs can differ
    by many ulp), so the reference is torch._scaled_mm on the same operands (cuBLASLt's own numerics) and the item is
    not bitwise there. The caller caches the result per (N, K, band), so later modules of that shape reuse it."""
    n, k = (int(d) for d in module.weight.shape)
    gen = torch.Generator(device=module.weight.device).manual_seed(n * 31 + k)
    for m in (band[0], band[1]):
        x = torch.randn(m, k, device=module.weight.device, dtype=torch.float32, generator=gen)
        e = torch.randint(-12, 13, (m, k // 32, 1), device=x.device, generator=gen).float()
        act, act_sf = torch.ops.trtllm.mxfp8_quantize(
            (x.view(m, k // 32, 32) * torch.exp2(e)).view(m, k).to(torch.bfloat16), True
        )
        del x, e
        try:
            pinned = family_ops().minimax_m31_mxfp8_cublaslt_gemm(
                act, act_sf, module.weight, module.weight_scale, list(band[2])
            )
        except RuntimeError as ex:
            return f"the pinned launch failed at M={m}: {str(ex)[:120]}"
        again = family_ops().minimax_m31_mxfp8_cublaslt_gemm(
            act, act_sf, module.weight, module.weight_scale, list(band[2])
        )
        if not torch.equal(again.view(torch.int16), pinned.view(torch.int16)):
            return f"two pinned calls differ at M={m}"
        if _MXFP8_CUBLASLT_REF == "cutlass":
            ref = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                act, act_sf, module.weight, module.weight_scale, module._mxfp8_alpha, module.dtype
            )
            what = "the CUTLASS op"
        else:
            ref = torch._scaled_mm(
                act,
                module.weight.t(),
                act_sf.view(torch.float8_e8m0fnu),
                module.weight_scale.view(torch.float8_e8m0fnu),
                out_dtype=torch.bfloat16,
            )
            what = "torch._scaled_mm"
        differ = int((pinned.view(torch.int16) != ref.view(torch.int16)).sum())
        if differ:
            return f"{differ} bits differ from {what} at M={m}"
    return None


def _mxfp8_cublaslt_gemm(
    method: "MXFP8LinearMethod",
    module: Linear,
    input: torch.Tensor,
    prequant: Optional[Tuple[torch.Tensor, torch.Tensor]],
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """The pinned cuBLASLt MXFP8 GEMM of this linear, or None for the CUTLASS op (see _MXFP8_CUBLASLT)."""
    if (
        not _MXFP8_CUBLASLT
        or bias is not None
        or input.dim() != 2
        or not method.use_cutlass
        or is_torch_compiling()
        or torch.compiler.is_compiling()
        or _dep2ctx_capture_blocked()
    ):
        return None
    m = int(input.shape[0])
    n, k = (int(d) for d in module.weight.shape)
    band = next((b for b in _MXFP8_CUBLASLT_PINS.get((n, k), ()) if b[0] <= m <= b[1]), None)
    if band is None:
        return None
    key = (n, k, band[0], band[1])
    ready = _MXFP8_CUBLASLT_STATE.get(key)
    if ready is None:
        op = _mxfp8_cublaslt_op()
        if op is None:
            why = "the op is not built"
        elif get_sm_version() != 107 or module.dtype != torch.bfloat16:
            why = "not sm_107 bf16"
        elif float(module._mxfp8_alpha) != 1.0:
            why = "a module global scale != 1"
        else:
            why = _mxfp8_cublaslt_self_check(module, band)
        ready = why is None
        _MXFP8_CUBLASLT_STATE[key] = ready
        if not ready:
            logger.warning(
                f"[M3.1 dep2ctx] DEP2CTX_MXFP8_CUBLASLT not taken for N={n} K={k} M in [{band[0]}, {band[1]}]: "
                f"{why}; the CUTLASS op runs"
            )
        else:
            logger.info(
                f"[M3.1 dep2ctx] DEP2CTX_MXFP8_CUBLASLT band self-check: N={n} K={k} M in [{band[0]}, {band[1]}] "
                f"bitwise {'the CUTLASS op' if _MXFP8_CUBLASLT_REF == 'cutlass' else 'torch._scaled_mm'} at both edges"
            )
    if not ready:
        return None
    # The CUTLASS op scales by the module's global scale; the pinned matmul runs with alpha 1 (read once per module).
    alpha_one = getattr(module, "_m31_mxfp8_alpha_one", None)
    if alpha_one is None:
        alpha_one = bool(float(module._mxfp8_alpha) == 1.0)
        module._m31_mxfp8_alpha_one = alpha_one
    if not alpha_one:
        return None
    _bcg_note("MXFP8_CUBLASLT", input)
    act, act_sf = (
        prequant
        if prequant is not None
        else torch.ops.trtllm.mxfp8_quantize(input.contiguous(), True)
    )
    if "logged" not in _MXFP8_CUBLASLT_STATE:
        _MXFP8_CUBLASLT_STATE["logged"] = True
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_MXFP8_CUBLASLT pinned cuBLASLt MXFP8 GEMM taken (N={n} K={k} M={m})"
        )
    return family_ops().minimax_m31_mxfp8_cublaslt_gemm(
        act.view(m, -1), act_sf, module.weight, module.weight_scale, list(band[2])
    )


# DEP2CTX_SWIGLU_MXFP8: the swigluoai MLPs (shared expert, dense layers) whose down_proj is the
# DEP2CTX_NORMQ MXFP8 linear run their activation and down_proj's MXFP8 quantize as one C++ kernel
# (kernels/minimaxM31/swigluOaiMxfp8Sm103.cu: swigluOaiSm103's instruction sequence, bitwise the Triton silu_and_mul,
# then quantize_with_block_size's per-element MXFP8 math), and publish the pair so down_proj skips its quantize. Bitwise
# the stock silu_and_mul + mxfp8_quantize(h, True); the bf16 activation is still written, so any down_proj path that
# does not take the pair reads the same operand. From _NORMQ_MIN_ROWS rows, eager only (no compile / capture); kill
# switch TRTLLM_M31_DISABLE_DEP2CTX_SWIGLU_MXFP8=1.
_SWIGLU_MXFP8 = not m31_disabled("DEP2CTX_SWIGLU_MXFP8")
_SWIGLU_MXFP8_STATE: Dict[str, Any] = {}


# SWIGLU_MXFP8_ROWS: from ops/swiglu_rows_plan.ROWS_MIN rows the op keeps K_ROWS_DEFAULT rows in flight per CTA on an
# occupancy-true grid, every output byte the one-row launch's. Default ON; kill switch
# TRTLLM_M31_DISABLE_SWIGLU_MXFP8_ROWS=1 (the one-row launch at every row count), passed as the op's rows_form argument.
_SWIGLU_ROWS: Dict[str, Any] = dict(
    rows_form=not m31_disabled("SWIGLU_MXFP8_ROWS"), k_rows=swiglu_rows_plan.K_ROWS_DEFAULT
)


def _swiglu_mxfp8_op():
    """family_ops().minimax_m31_swiglu_oai_mxfp8 from the build; None when it is not built. Decided once."""
    if "op" not in _SWIGLU_MXFP8_STATE:
        op = getattr(family_ops(), "minimax_m31_swiglu_oai_mxfp8", None)
        _SWIGLU_MXFP8_STATE["op"] = op
        logger.info(
            f"[M3.1 dep2ctx] DEP2CTX_SWIGLU_MXFP8 swigluoai + down_proj MXFP8 quantize: "
            f"{'engaged' if op is not None else 'not taken: the op is not built'} (rows >= {_NORMQ_MIN_ROWS})"
        )
    return _SWIGLU_MXFP8_STATE["op"]


# DEP2CTX_SWIGLU_NOH: the down projection takes the published MXFP8 pair by identity (_NormqMXFP8LinearMethod), so the
# bf16 activation the SWIGLU_MXFP8 op would also write is never read: the op skips it (h=None, its kWriteH=false form,
# the same q / sf bytes) and h2 stays an unwritten key of the right shape. Taken only when down_proj's apply cannot fall
# back to quantizing h2 (the NORMQ method on its CUTLASS / pinned path, outside the FlashInfer decode
# graph capture). Kill switch TRTLLM_M31_DISABLE_DEP2CTX_SWIGLU_NOH=1.
_SWIGLU_NOH = not m31_disabled("DEP2CTX_SWIGLU_NOH")


def _swiglu_noh(mlp) -> bool:
    """Whether DEP2CTX_SWIGLU_NOH may skip the bf16 activation for this MLP's down projection (see above)."""
    method = getattr(getattr(mlp, "down_proj", None), "quant_method", None)
    taken = (
        _SWIGLU_NOH
        and type(method) is _NormqMXFP8LinearMethod
        and bool(getattr(method, "use_cutlass", False))
        and not (
            getattr(method, "tune_decode_graph_backends", False)
            and _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
        )
    )
    if taken and "noh_logged" not in _SWIGLU_MXFP8_STATE:
        _SWIGLU_MXFP8_STATE["noh_logged"] = True
        logger.info(
            "[M3.1 dep2ctx] DEP2CTX_SWIGLU_NOH taken: SWIGLU_MXFP8 writes only the MXFP8 pair (no bf16 activation)"
        )
    return taken


class _SwigluMxfp8GatedMLP(GatedMLP):
    """A swigluoai GatedMLP (limit 7, alpha 1.702, beta 1) whose large eager calls run DEP2CTX_SWIGLU_MXFP8; every other
    call is the stock forward."""

    def forward(
        self,
        x,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (
            lora_params
            or self.split_gate_up
            or not isinstance(x, torch.Tensor)
            or x.dim() != 2
            or x.shape[0] < _NORMQ_MIN_ROWS
            or is_torch_compiling()
            or torch.compiler.is_compiling()
            or _dep2ctx_capture_blocked()
            or self._can_fuse_gate_up_swiglu_fp4out()
            or self._can_fuse_gate_up_swiglu()
            or self._can_fuse_swiglu_fp8_quant()
            or _swiglu_mxfp8_op() is None
        ):
            return super().forward(
                x,
                all_rank_num_tokens=all_rank_num_tokens,
                final_all_reduce_params=final_all_reduce_params,
                lora_params=lora_params,
                **kwargs,
            )
        h1 = self.gate_up_proj(x)
        rows, d = h1.shape[0], h1.shape[-1] // 2
        if (
            h1.dim() != 2
            or h1.dtype != torch.bfloat16
            or not h1.is_contiguous()
            or d % 256
            or d > 8192
        ):
            h2 = self._apply_activation(h1)
        else:
            h2 = torch.empty((rows, d), dtype=torch.bfloat16, device=h1.device)
            q = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device=h1.device)
            sf = torch.empty(
                ((rows + 127) // 128 * 128 * d // 32,), dtype=torch.uint8, device=h1.device
            )
            _bcg_note("SWIGLU_MXFP8", h1)
            _swiglu_mxfp8_op()(h1, q, sf, None if _swiglu_noh(self) else h2, **_SWIGLU_ROWS)
            _NORMQ_SLOT.publish(h2, q, sf)
        return self.down_proj(
            h2, all_reduce_params=final_all_reduce_params, layer_idx=self.layer_idx
        )


class _NormqMXFP8LinearMethod(MXFP8LinearMethod):
    """MXFP8LinearMethod that runs its GEMM on the DEP2CTX_NORMQ activation when its input is the published normed rows
    (the operands the stock path would quantize, bit for bit); every other call is the stock apply."""

    def apply(self, module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor]):
        prequant = _NORMQ_SLOT.take(input) if _NORMQ_SLOT.pending is not None else None
        if prequant is None:
            # DEP2CTX_NORMQ_STAGE_DROP_NORM: this read misses the published pair, so a left-out norm is written first.
            _NORMQ_DROPPED.materialize(input)
        if (
            prequant is not None
            and "combine_take_logged" not in _NORMQ_STATE
            and _NORMQ_STATE.get("combine_src") == (input.data_ptr(), tuple(input.shape))
        ):
            _NORMQ_STATE["combine_take_logged"] = True
            logger.info(
                "[M3.1 dep2ctx] DEP2CTX_COMBINE_MXFP8 taken: o_proj consumed the combine's MXFP8 input "
                f"(rows {input.shape[0]}; no quantize)"
            )
        materialize = _NORMQ_SLOT.taken_materialize if prequant is not None else None
        _NORMQ_SLOT.taken_materialize = None  # the closure holds the forward's inputs; drop it now
        pinned = _mxfp8_cublaslt_gemm(self, module, input, prequant, bias)
        if pinned is not None:
            return pinned
        if (
            prequant is None
            or not self.use_cutlass
            or (
                self.tune_decode_graph_backends
                and not is_torch_compiling()
                and _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
            )
        ):
            if (
                materialize is not None
            ):  # DEP2CTX_COMBINE_DROP_BF16: the stock path quantizes the BF16 input itself
                _c1_materialize(materialize)
            return super().apply(module, input, bias)
        act_e4m3, act_sf = prequant
        use_flashinfer = self.backend == "flashinfer" or (
            self.backend == "auto"
            and not is_torch_compiling()
            and (
                _linear._FLASHINFER_MXFP8_AUTOTUNE_ACTIVE.get()
                or (
                    self._flashinfer_autotuned
                    and _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
                )
            )
        )
        if use_flashinfer:
            output = self._flashinfer_mxfp8(
                act_e4m3, act_sf, module.weight, module.weight_scale, module.dtype
            )
        else:
            gemm = (
                torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned
                if self.needs_native_autotune
                else torch.ops.trtllm.mxfp8_mxfp8_gemm
            )
            output = gemm(
                act_e4m3,
                act_sf,
                module.weight,
                module.weight_scale,
                module._mxfp8_alpha,
                module.dtype,
            )
        if bias is not None:
            output = output + bias
        return output


@torch.library.custom_op(f"{_NS}::m31_dep2ctx_sm107_qkv_index_proj", mutates_args=())
def minimax_m3_qkv_index_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> torch.Tensor:
    """Run the five-way projection as one CUDA-graph-capturable custom op."""
    del position_ids  # Symbolic token-shape carrier; projection itself is position agnostic.
    _, attn_layer = _extract_minimax_m3_attention_extra_attrs(layer_idx)
    packed = _a1_bcg_gemm(attn_layer, hidden_states)
    if packed is not None:
        return packed
    return attn_layer.qkv_proj(hidden_states)


@minimax_m3_qkv_index_proj.register_fake
def _minimax_m3_qkv_index_proj_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> torch.Tensor:
    """Preserve the symbolic token dimension across the opaque projection."""
    _, attn_layer = _extract_minimax_m3_attention_extra_attrs(layer_idx)
    qkv_proj = attn_layer.qkv_proj
    if not isinstance(qkv_proj, MiniMaxM3QKVIndexerLinear):
        raise RuntimeError(
            "MiniMax-M3 fused QKV/index projection custom op requires "
            f"MiniMaxM3QKVIndexerLinear, got {type(qkv_proj).__name__}."
        )
    # The real projection is token-major over ``hidden_states``. Keep that
    # exact output contract here so generation CUDA-graph batch sizes can
    # share Dynamo's dynamic-shape specialization. ``position_ids`` remains
    # an explicit custom-op input to carry the unpadded token symbol into
    # piecewise context segments; the attention output below uses that symbol.
    return hidden_states.new_empty((hidden_states.shape[0], sum(qkv_proj.local_output_sizes)))


# Projection and cache insertion inspect runtime shapes and layouts. Keep those
# checks opaque to avoid Dynamo specialization or graph breaks while PCG captures
# the fused kernels; explicit mutable cache inputs keep their writes visible.
@torch.library.custom_op(
    f"{_NS}::m31_dep2ctx_sm107_fused_sparse_qkv_producer",
    mutates_args=("kv_cache", "index_k_cache", "kv_scale_cache"),
)
def minimax_m3_fused_sparse_qkv_producer(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    kv_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    kv_scale_cache: Optional[torch.Tensor],
    out_cache_loc: torch.Tensor,
    layer_idx: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture projection, norm, RoPE and FP8/NVFP4 cache insertion together."""
    attn_metadata, attn_layer = _extract_minimax_m3_attention_extra_attrs(layer_idx)
    cache_tensors = (kv_cache, index_k_cache, kv_scale_cache, out_cache_loc)
    result = attn_layer._fused_fp8_qkv_indexer_norm_rope_kv_insert(
        None, position_ids, attn_metadata, cache_tensors=cache_tensors, hidden_states=hidden_states
    )
    if result is not None:
        return result
    packed = attn_layer.qkv_proj(hidden_states)
    result = attn_layer._fused_fp8_qkv_indexer_norm_rope_kv_insert(
        packed,
        position_ids,
        attn_metadata,
        cache_tensors=cache_tensors,
    )
    if result is None:
        raise RuntimeError("MiniMax-M3 piecewise graph requires the fused sparse QKV producer.")
    return result


@minimax_m3_fused_sparse_qkv_producer.register_fake
def _minimax_m3_fused_sparse_qkv_producer_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    kv_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    kv_scale_cache: Optional[torch.Tensor],
    out_cache_loc: torch.Tensor,
    layer_idx: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer FP8 query shapes while retaining the symbolic token dimension."""
    del position_ids
    _, attn_layer = _extract_minimax_m3_attention_extra_attrs(layer_idx)
    num_tokens = hidden_states.shape[0]
    return (
        hidden_states.new_empty((num_tokens, attn_layer.q_size), dtype=torch.float8_e4m3fn),
        hidden_states.new_empty((num_tokens, attn_layer.index_q_size), dtype=torch.float8_e4m3fn),
    )


def _dispatch_attention_over_live_tokens(
    attn_layer: "MiniMaxM3Attention",
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    idx_q: Optional[torch.Tensor],
    idx_k: Optional[torch.Tensor],
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
) -> None:
    """Run the attention core over the step's live tokens alone.

    A piecewise CUDA graph pads token-shaped inputs up to its capture bucket
    without adding requests to go with them (see _get_padding_params in
    model_engine), so q can outrun the rows the batch has. The kernels below
    read a request out of a token index, so the pad comes off here, once for
    both dispatch paths.

    The pad rows of output are left as the buffer supplied them. Nothing reads
    them for its own result, and clearing them would put a device launch behind
    a host-side count on every layer of every step.
    """
    num_tokens = int(attn_metadata.num_tokens)
    attn_layer._dispatch_attention_backend(
        q[:num_tokens],
        k[:num_tokens] if k is not None else None,
        v[:num_tokens] if v is not None else None,
        idx_q[:num_tokens] if idx_q is not None else None,
        idx_k[:num_tokens] if idx_k is not None else None,
        attn_metadata,
        output[:num_tokens],
    )


def _a1_bcg_ptr_reader(attn_metadata: AttentionMetadata) -> Callable[[int], Tuple[int, ...]]:
    """Per layer, the addresses a captured DEP2CTX_A1_FUSED_IN_BCG producer writes through: the persistent
    out-cache-loc slots, the layer's KV pool, index-K and KV-scale views (-1 when the layer has none) and the cache
    manager's FP8 index-K sink an NVFP4 index-K producer writes first (read live from the manager; -1 without one). The
    step-wide addresses are read once, so checking every layer costs a few ``data_ptr`` reads per layer."""
    caches = getattr(attn_metadata, "msa_layer_cache_tensors", None) or {}
    loc = getattr(attn_metadata, "msa_out_cache_loc", None)
    manager = getattr(attn_metadata, "kv_cache_manager", None)
    sink = getattr(manager, "get_nvfp4_index_k_sink", lambda: None)()
    shared = (
        (-1 if loc is None else loc.data_ptr()),
        (-1 if sink is None else sink.data_ptr()),
    )

    def read(layer_idx: int) -> Tuple[int, ...]:
        layer = caches.get(layer_idx)
        if layer is None or loc is None:
            return (-1, -1, -1, -1, -1)
        return (shared[0],) + tuple(-1 if t is None else t.data_ptr() for t in layer) + (shared[1],)

    return read


def _a1_bcg_ptrs(attn_metadata: AttentionMetadata, layer_idx: int) -> Tuple[int, ...]:
    """The addresses a captured DEP2CTX_A1_FUSED_IN_BCG producer writes through for ``layer_idx``
    (see _a1_bcg_ptr_reader)."""
    return _a1_bcg_ptr_reader(attn_metadata)(layer_idx)


def _attn_core_inplace(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    idx_q: Optional[torch.Tensor],
    idx_k: Optional[torch.Tensor],
    packed_qkv_index: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Run MiniMax-M3 cache and attention work behind a compile boundary (or a breakable graph's eager break).

    The captured horizontal producer can populate both caches before this
    boundary. The packed-projection fallback runs that producer here instead,
    while separate projections leave their cache writes here. Slice padded
    inputs to live tokens before request-dependent cache and MSA work.
    """
    attn_metadata, attn_layer = _extract_minimax_m3_attention_extra_attrs(layer_idx)
    num_tokens = attn_metadata.num_tokens
    if packed_qkv_index is not None:
        horizontal = attn_layer._fused_fp8_qkv_indexer_norm_rope_kv_insert(
            packed_qkv_index[:num_tokens],
            position_ids[..., :num_tokens] if position_ids is not None else None,
            attn_metadata,
        )
        if horizontal is None:
            raise RuntimeError(
                "MiniMax-M3 fused horizontal producer is required inside the "
                f"piecewise attention boundary for layer {layer_idx}, but its "
                "runtime geometry or cache layout was unsupported."
            )
        q, idx_q = horizontal
        k = v = idx_k = None
    if q is None:
        raise RuntimeError(f"MiniMax-M3 attention layer {layer_idx} received no query tensor.")
    # The live token count is a host value, so the compiled graph above must
    # not see it: it would guard on it and recapture per count.
    # DEP2CTX_C1_BCG: the combine's sink reads the bucket from this static output.
    _C1_BCG_STATE["output"] = output
    try:
        _dispatch_attention_over_live_tokens(
            attn_layer, q, k, v, idx_q, idx_k, attn_metadata, output
        )
    finally:
        _C1_BCG_STATE["output"] = None


@torch.library.custom_op(
    f"{_NS}::m31_dep2ctx_sm107_attn_custom_op_inplace", mutates_args=("output",)
)
def minimax_m3_attn_custom_op_inplace(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    idx_q: Optional[torch.Tensor],
    idx_k: Optional[torch.Tensor],
    packed_qkv_index: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Run MiniMax-M3 cache and attention work behind a compile boundary (see _attn_core_inplace)."""
    _attn_core_inplace(q, k, v, idx_q, idx_k, packed_qkv_index, position_ids, layer_idx, output)


def _bcg_break_attention(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    idx_q: Optional[torch.Tensor],
    idx_k: Optional[torch.Tensor],
    packed_qkv_index: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """DEP2CTX_BCG_RAW_BREAK: the attention break without the custom-op dispatcher; a compiled graph keeps the op."""
    if torch.compiler.is_compiling():
        minimax_m3_attn_custom_op_inplace(
            q, k, v, idx_q, idx_k, packed_qkv_index, position_ids, layer_idx, output
        )
        return
    _attn_core_inplace(q, k, v, idx_q, idx_k, packed_qkv_index, position_ids, layer_idx, output)


maybe_bcg_minimax_m3_attn_custom_op_inplace = eager_on_graph(
    _bcg_break_attention if _BCG_RAW_BREAK else minimax_m3_attn_custom_op_inplace
)


class MiniMaxM3Attention(Attention):
    """M3 attention: dense (layers 0-2) or sparse (layers 3-59).

    Both branches share the same dense GQA scaffolding (``qkv_proj`` +
    ``o_proj`` + per-head Gemma Q/K norm + partial RoPE). Sparse layers
    additionally carry the MiniMax index branch: a fused index_qk_proj that
    outputs [idx_q | idx_k], plus per-head index norms. The index value and
    output branch is omitted because the M3 checkpoint sets
    sparse_disable_index_value=True on every sparse layer.

    NOTE: This model-specific integration is kept for compatibility.
    New sparse algorithms should use the shared ``Attention``/``MLA``
    modules and sparse hooks; MiniMax-M3 should migrate in a future refactor.
    """

    def __init__(
        self,
        *,
        model_config: "ModelConfig[PretrainedConfig]",
        layer_idx: Optional[int] = None,
        is_sparse_attention_layer: bool = False,
        disable_index_value: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config
        self.pretrained_config = config

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            rope_fusion=False,  # partial RoPE; rotate only rotary_dim of head_dim
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )
        # The family's MSA backend in place of main's when the engine builds its step metadata (msa.engine).
        _msa_engine.use_family_backend(self)

        self.qk_norm_type = getattr(config, "qk_norm_type", "per_layer")
        if self.qk_norm_type != "per_head":
            raise ValueError(
                f"MiniMaxM3Attention only supports qk_norm_type='per_head', "
                f"got {self.qk_norm_type!r}"
            )
        self.use_gemma_norm = bool(getattr(config, "use_gemma_norm", False))
        self.head_dim_value = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )

        # Dtype of the q/k/v activations fed into norm and RoPE. KV-cache
        # quantization changes cache storage only, so this stays the compute dtype.
        self.attn_activation_dtype = config.torch_dtype

        # Whether the main K/V cache is stored as FP8 E4M3.
        quant_config = getattr(model_config, "quant_config", None)
        self.main_kv_is_fp8 = bool(
            quant_config is not None
            and quant_config.quant_mode is not None
            and quant_config.quant_mode.has_fp8_kv_cache()
        )
        self.main_kv_is_nvfp4 = bool(
            quant_config is not None
            and quant_config.quant_mode is not None
            and quant_config.quant_mode.has_fp4_kv_cache()
        )

        # Per-head Gemma RMSNorm — one set of weights shared across heads.
        self.q_norm = RMSNorm(
            hidden_size=self.head_dim_value,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=self.use_gemma_norm,
        )
        self.k_norm = RMSNorm(
            hidden_size=self.head_dim_value,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=self.use_gemma_norm,
        )

        # Stream and events used to overlap independent projection/norm/RoPE
        # work. The stream is shared across layers via aux_stream_dict, matching
        # DeepSeekV3; a private stream is used when no stream is supplied.
        self.aux_stream = aux_stream if aux_stream is not None else torch.cuda.Stream()
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.is_sparse_attention_layer = bool(is_sparse_attention_layer)
        if not self.is_sparse_attention_layer and isinstance(
            self.attn, MiniMaxM3SparseRuntimeBackend
        ):
            self._init_native_dense_attention(model_config)
        self.disable_index_value = bool(disable_index_value)
        sparse_runtime_cfg = getattr(model_config, "sparse_attention_config", None)
        self.enable_fused_qkv_index_projection = bool(
            self.is_sparse_attention_layer
            and sparse_runtime_cfg is not None
            and getattr(sparse_runtime_cfg, "fuse_qkv_index_projection", False)
        )
        if self.is_sparse_attention_layer:
            sparse_cfg = getattr(config, "sparse_attention_config", None) or {}
            total_num_index_heads = int(sparse_cfg.get("sparse_num_index_heads", 4))
            self.sparse_num_index_heads = total_num_index_heads
            self.sparse_index_dim = int(sparse_cfg.get("sparse_index_dim", 128))
            self.sparse_block_size = int(sparse_cfg.get("sparse_block_size", 128))
            self.sparse_topk_blocks = int(sparse_cfg.get("sparse_topk_blocks", 16))
            self.sparse_init_block = int(sparse_cfg.get("sparse_init_block", 0))
            self.sparse_local_block = int(sparse_cfg.get("sparse_local_block", 1))
            self.sparse_score_type = str(sparse_cfg.get("sparse_score_type", "max"))

            if self.enable_fused_qkv_index_projection:
                old_qkv_proj = self.qkv_proj
                self.qkv_proj = MiniMaxM3QKVIndexerLinear(
                    hidden_size=config.hidden_size,
                    head_dim=self.head_dim,
                    total_num_heads=config.num_attention_heads,
                    total_num_kv_heads=config.num_key_value_heads,
                    total_num_index_heads=total_num_index_heads,
                    index_head_dim=self.sparse_index_dim,
                    dtype=config.torch_dtype,
                    mapping=old_qkv_proj.mapping,
                    quant_config=old_qkv_proj.quant_config,
                    skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                    force_dynamic_quantization=old_qkv_proj.force_dynamic_quantization,
                    disable_deep_gemm=old_qkv_proj.disable_deep_gemm,
                    use_custom_cublas_mm=old_qkv_proj.use_custom_cublas_mm,
                    use_cute_dsl_bf16_gemm=old_qkv_proj.use_cute_dsl_bf16_gemm,
                    use_cute_dsl_blockscaling_mm=old_qkv_proj.use_cute_dsl_blockscaling_mm,
                )
                self.sparse_num_index_heads = self.qkv_proj.local_num_index_heads
            else:
                # Use the same KV-group ownership as the five-way projection.
                # Linear's existing per-shard override handles index-Q slicing
                # and index-K replication without a custom checkpoint loader.
                index_start, index_end = index_head_range(
                    total_num_index_heads, config.num_key_value_heads, self.qkv_proj.mapping
                )
                self.sparse_num_index_heads = index_end - index_start
                self.index_qk_proj = Linear(
                    config.hidden_size,
                    total_num_index_heads * self.sparse_index_dim + self.sparse_index_dim,
                    bias=False,
                    dtype=config.torch_dtype,
                    mapping=self.qkv_proj.mapping,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    quant_config=_index_projection_quant_config(model_config, layer_idx),
                    weights_loading_config=WeightsLoadingConfig(
                        weight_mode=WeightMode.FUSED_GATE_UP_LINEAR
                    ),
                    override_tp_sharding={
                        "gate": (
                            index_start * self.sparse_index_dim,
                            index_end * self.sparse_index_dim,
                        ),
                        "up": (0, self.sparse_index_dim),
                    },
                    skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                )

            self.index_q_size = self.sparse_num_index_heads * self.sparse_index_dim
            self.index_k_size = self.sparse_index_dim
            # Per-head Gemma RMSNorm of width ``sparse_index_dim``;
            # applied to the projected index Q/K before partial RoPE in
            # the sparse forward path.
            self.index_q_norm = RMSNorm(
                hidden_size=self.sparse_index_dim,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
                use_gemma=self.use_gemma_norm,
            )
            self.index_k_norm = RMSNorm(
                hidden_size=self.sparse_index_dim,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
                use_gemma=self.use_gemma_norm,
            )

    def _init_native_dense_attention(self, model_config: ModelConfig) -> None:
        """Use native attention for unquantized KV when TRTLLM-Gen is preferred."""
        # FP8 native FMHA also quantizes Q; SDPA preserves the activation dtype.
        # NVFP4 requires independent block-scale strides in its virtual pools.
        if self.main_kv_is_fp8 or self.main_kv_is_nvfp4:
            return
        enabled_libs = get_enabled_fmha_lib_classes()
        if FlashInferTrtllmGenFmha not in enabled_libs:
            return
        if FallbackFmha in enabled_libs and enabled_libs.index(FallbackFmha) < enabled_libs.index(
            FlashInferTrtllmGenFmha
        ):
            # A fallback-first policy cannot serve P128 cached prefill safely.
            return
        if not FlashInferTrtllmGenFmha.is_environment_available():
            return
        # Q/K norm and partial RoPE run in the model before backend dispatch.
        native_attn = TrtllmAttention(
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            quant_config=self.quant_config,
            q_scaling=self.q_scaling,
            pos_embd_params=None,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
        )
        if not FlashInferTrtllmGenFmha.is_available(native_attn):
            return
        self.attn = native_attn
        self.support_fused_qkv = self.attn.support_fused_qkv()

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head Gemma RMSNorm on the main attention Q/K.

        Reshapes ``q`` (shape ``[..., num_heads * head_dim]``) and ``k``
        (shape ``[..., num_kv_heads * head_dim]``) to per-head rows of
        width ``head_dim``, applies the per-head RMSNorm
        (:attr:`q_norm` / :attr:`k_norm`, both ``use_gemma=True``), and
        reshapes back. Matches the SGLang reference's
        ``_qk_norm`` for ``qk_norm_type='per_head'``.
        """
        q_shape = q.shape
        k_shape = k.shape

        def _q_norm():
            return self.q_norm(q.reshape(-1, self.head_dim_value)).reshape(q_shape)

        def _k_norm():
            return self.k_norm(k.reshape(-1, self.head_dim_value)).reshape(k_shape)

        # The q-norm and k-norm are independent, so overlap them on the aux
        # stream. Multi-stream is disabled under torch.compile.
        q, k = maybe_execute_in_parallel(
            _q_norm,
            _k_norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
            disable_on_compile=True,
        )
        return q, k

    def apply_index_qk_norm(
        self, idx_q: torch.Tensor, idx_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head Gemma RMSNorm on the sparse-attention index Q/K.

        Mirrors :meth:`apply_qk_norm` for the index branch: reshapes
        ``idx_q`` (shape ``[..., num_index_heads * sparse_index_dim]``,
        sharded with the KV heads) and ``idx_k`` (shape
        ``[..., sparse_index_dim]`` — single replicated head, not per
        index head) to per-head rows of width ``sparse_index_dim``,
        applies :attr:`index_q_norm` / :attr:`index_k_norm`, and reshapes
        back. The sparse forward path calls this on the projected index
        Q/K before partial RoPE.

        Raises:
            RuntimeError: if called on a dense layer (no index branch).
        """
        if not self.is_sparse_attention_layer:
            raise RuntimeError(
                f"apply_index_qk_norm is only valid on sparse attention layers "
                f"(layer_idx={self.layer_idx} is dense)"
            )
        idx_q_shape = idx_q.shape
        idx_k_shape = idx_k.shape

        def _idx_q_norm():
            return self.index_q_norm(idx_q.reshape(-1, self.sparse_index_dim)).reshape(idx_q_shape)

        def _idx_k_norm():
            return self.index_k_norm(idx_k.reshape(-1, self.sparse_index_dim)).reshape(idx_k_shape)

        idx_q, idx_k = maybe_execute_in_parallel(
            _idx_q_norm,
            _idx_k_norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
            disable_on_compile=True,
        )
        return idx_q, idx_k

    def _fused_qk_norm_rope(
        self,
        qkv: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        *,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        out_fp8: bool = False,
    ) -> Optional[torch.Tensor]:
        """Fuse per-head Gemma RMSNorm and partial RoPE into one kernel.

        Runs torch.ops.trtllm.fused_qk_norm_rope over the packed
        [Q heads, K heads, optional V heads] layout and returns the mutated
        tensor, which aliases qkv when qkv is already contiguous. The kernel
        norms the full head_dim and rotates only the leading rotary_dim
        channels, matching M3's whole-head norm with front partial RoPE, and
        leaves the V heads untouched.

        When out_fp8 is True, an out-of-place variant runs instead and returns a
        fresh FP8 E4M3 tensor with Q/K normed and roped and V copy-cast, leaving
        qkv untouched.

        Returns None with qkv unmodified when the kernel does not apply, so the
        caller runs norm and RoPE separately: non-bf16 activations (the kernel
        is bf16-only), missing position_ids, or no rotary embedding.
        """
        if position_ids is None or qkv.dtype != torch.bfloat16:
            return None
        if (
            self.rotary_emb is None
            or self.pos_embd_params is None
            or self.pos_embd_params.rope is None
        ):
            return None

        # Partial-RoPE dim comes from RopeParams (M3 rotates 64 of 128).
        rotary_dim = int(self.pos_embd_params.rope.dim)
        # The kernel assumes a contiguous [num_tokens, total_heads * head_dim].
        qkv = qkv.contiguous()
        position_ids_i32 = position_ids.reshape(-1).contiguous().to(torch.int32)
        if out_fp8:
            return torch.ops.trtllm.fused_qk_norm_rope_to_fp8(
                qkv,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_dim,
                rotary_dim,
                q_norm.variance_epsilon,
                q_norm.weight,
                k_norm.weight,
                self.pos_embd_params.rope.theta,
                self.pos_embd_params.is_neox,
                position_ids_i32,
                1.0,  # factor: no YARN (M3 has no rope_scaling)
                0.0,  # low
                0.0,  # high
                1.0,  # attention_factor
                True,  # is_qk_norm
                self.use_gemma_norm,  # use_gemma
                False,  # use_mrope
                0,  # mrope_section1
                0,  # mrope_section2
            )
        torch.ops.trtllm.fused_qk_norm_rope(
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            rotary_dim,
            q_norm.variance_epsilon,
            q_norm.weight,
            k_norm.weight,
            self.pos_embd_params.rope.theta,
            self.pos_embd_params.is_neox,
            position_ids_i32,
            1.0,  # factor: no YARN (M3 has no rope_scaling)
            0.0,  # low
            0.0,  # high
            1.0,  # attention_factor
            True,  # is_qk_norm
            self.use_gemma_norm,  # use_gemma
            False,  # use_mrope
            0,  # mrope_section1
            0,  # mrope_section2
        )
        return qkv

    def _fused_fp8_index_qk_norm_rope(
        self,
        idx_qk: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> Optional[torch.Tensor]:
        """Produce raw-E4M3 index-Q and insert index-K in one M3-only kernel.

        This path is deliberately narrower than the general fused QK kernel:
        it is enabled only for the MSA FP8-indexer configuration and the exact
        M3 Gemma-RMSNorm + NeoX partial-RoPE geometry. The kernel rounds the
        normalized/RoPE values through BF16 before E4M3 conversion, matching
        the former fused-BF16-kernel followed by ``Tensor.to(E4M3)`` contract.
        An NVFP4 index-K cache is written through the cache manager's FP8 sink
        and packed from it afterwards (_index_k_producer_cache).
        """
        if not isinstance(self.attn, MiniMaxM3MsaSparseAttention):
            return None
        if self.attn.indexer_kv_dtype not in ("fp8", "nvfp4"):
            return None
        if position_ids is None or idx_qk.dtype != torch.bfloat16:
            raise NotImplementedError(
                "MiniMax-M3 fused FP8 indexer requires BF16 activations and position_ids."
            )
        if (
            self.rotary_emb is None
            or self.pos_embd_params is None
            or self.pos_embd_params.rope is None
        ):
            raise NotImplementedError("MiniMax-M3 fused FP8 indexer requires partial RoPE.")
        if not self.pos_embd_params.is_neox:
            raise NotImplementedError("MiniMax-M3 fused FP8 indexer requires NeoX RoPE.")
        if not self.use_gemma_norm:
            raise NotImplementedError("MiniMax-M3 fused FP8 indexer requires Gemma RMSNorm.")
        if self.index_q_norm.variance_epsilon != self.index_k_norm.variance_epsilon:
            raise ValueError(
                "MiniMax-M3 fused FP8 indexer requires identical index Q/K "
                "RMSNorm epsilon values because the kernel accepts one epsilon."
            )

        rotary_dim = int(self.pos_embd_params.rope.dim)
        index_k_cache, nvfp4_index_cache = _index_k_producer_cache(
            attn_metadata.kv_cache_manager, attn_metadata.msa_idx_k_cache(self.layer_idx)
        )
        if index_k_cache.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "MiniMax-M3 fused FP8 indexer requires an E4M3 index-K cache, "
                f"got {index_k_cache.dtype}."
            )
        num_tokens = int(idx_qk.shape[0])
        out_cache_loc = attn_metadata.msa_out_cache_loc
        index_q = torch.ops.trtllm.minimax_m3_fp8_indexer_qk_norm_rope(
            idx_qk.contiguous(),
            index_k_cache,
            out_cache_loc[:num_tokens],
            self.sparse_num_index_heads,
            self.sparse_index_dim,
            rotary_dim,
            self.index_q_norm.variance_epsilon,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.pos_embd_params.rope.theta,
            position_ids.reshape(-1).contiguous().to(torch.int32),
        )
        _pack_nvfp4_index_k(nvfp4_index_cache, index_k_cache, out_cache_loc, num_tokens)
        return index_q.flatten(1)

    def _fused_fp8_qkv_indexer_norm_rope_kv_insert(
        self,
        packed: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        *,
        cache_tensors: Optional[
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
        ] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Run the vLLM-style horizontal producer for every sparse batch.

        With ``hidden_states`` (``packed`` None) only the fused SM107 QKV GEMM + producer kernel is tried; None
        means the caller projects with ``qkv_proj`` and calls again with ``packed``.

        The CUDA kernel is token-major and batch-type agnostic: per-token
        positions and cache slots cover pure prefill, mixed aggregate batches,
        and CUDA-graph decode.  FP8 caches receive FP8 K/V directly.  NVFP4
        caches receive packed E2M1 K/V plus their E4M3 SF16 scale bytes while
        Q, index-Q, and index-K remain FP8.  An NVFP4 index-K cache is written
        through the cache manager's FP8 sink and packed from it afterwards
        (_index_k_producer_cache).
        """
        if (
            not self.enable_fused_qkv_index_projection
            or not isinstance(self.attn, MiniMaxM3MsaSparseAttention)
            or not (self.main_kv_is_fp8 or self.main_kv_is_nvfp4)
            or self.attn.indexer_kv_dtype not in ("fp8", "nvfp4")
        ):
            return None
        rope = self.pos_embd_params.rope if self.pos_embd_params is not None else None
        rotary_dim = int(rope.dim) if rope is not None else 0
        source = packed if packed is not None else hidden_states
        if (
            source is None
            or source.dtype != torch.bfloat16
            or position_ids is None
            or self.head_dim != 128
            or self.sparse_index_dim != 128
            or self.sparse_num_index_heads != self.num_key_value_heads
            or not self.use_gemma_norm
            or self.pos_embd_params is None
            or not self.pos_embd_params.is_neox
            or self.rotary_emb is None
            or rope is None
            or rotary_dim != 64
        ):
            return None
        norm_eps = self.q_norm.variance_epsilon
        if any(
            module.variance_epsilon != norm_eps
            for module in (self.k_norm, self.index_q_norm, self.index_k_norm)
        ):
            return None
        norm_weights = (
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
        )
        if any(weight.dtype != torch.bfloat16 or not weight.is_cuda for weight in norm_weights):
            return None

        kv_cache_manager = attn_metadata.kv_cache_manager
        if cache_tensors is None:
            if kv_cache_manager is None:
                return None
            buffers = kv_cache_manager.get_buffers(self.layer_idx, kv_layout="HND")
            index_k_cache = attn_metadata.msa_idx_k_cache(self.layer_idx)
            scale_buffers = (
                kv_cache_manager.get_block_scale_buffers(self.layer_idx, kv_layout="HND")
                if kv_cache_manager.is_nvfp4_layer(self.layer_idx)
                else None
            )
            out_cache_loc = getattr(attn_metadata, "msa_out_cache_loc", None)
        else:
            buffers, index_k_cache, scale_buffers, out_cache_loc = cache_tensors
        index_k_cache, nvfp4_index_cache = _index_k_producer_cache(kv_cache_manager, index_k_cache)
        num_tokens = int(source.shape[0])
        layer_uses_nvfp4 = scale_buffers is not None
        kv_quant_scale = getattr(self.qkv_proj, "inv_kv_scales", None) if layer_uses_nvfp4 else None
        if layer_uses_nvfp4:
            supports_fused_layout = buffers is not None and _nvfp4_cache_supports_fused_write(
                buffers, scale_buffers, self.num_key_value_heads
            )
            supported_main_cache = (
                supports_fused_layout
                and buffers.is_cuda
                and scale_buffers.is_cuda
                and scale_buffers.device == buffers.device
                and kv_quant_scale is not None
                and kv_quant_scale.is_cuda
                and kv_quant_scale.dtype == torch.float32
                and kv_quant_scale.numel() >= 3
                and kv_quant_scale.is_contiguous()
            )
        else:
            supported_main_cache = (
                buffers is not None
                and buffers.is_cuda
                and buffers.dtype == torch.float8_e4m3fn
                and buffers.dim() == 5
                and tuple(buffers.shape[1:]) == (2, self.num_key_value_heads, 128, 128)
                and buffers.stride(4) == 1
                and buffers.stride(3) == 128
                and buffers.stride(2) == 128 * 128
                and buffers.stride(1) >= self.num_key_value_heads * buffers.stride(2)
                and buffers.stride(0) >= 2 * buffers.stride(1)
                and buffers.stride(0) % 4 == 0
                and buffers.stride(1) % 4 == 0
            )
        supported_index_cache = (
            index_k_cache.is_cuda
            and index_k_cache.dtype == torch.float8_e4m3fn
            and index_k_cache.dim() == 4
            and tuple(index_k_cache.shape[1:]) == (1, 128, 128)
            and index_k_cache.stride(3) == 1
            and index_k_cache.stride(2) == 128
            and index_k_cache.stride(1) >= 128 * 128
            and index_k_cache.stride(0) >= index_k_cache.stride(1)
            and index_k_cache.stride(0) % 4 == 0
            and index_k_cache.stride(1) % 4 == 0
            and buffers is not None
            and index_k_cache.shape[0] == buffers.shape[0]
        )
        rotary_cos_sin = self.rotary_emb.rotary_cos_sin
        supported_rope_cache = (
            rotary_cos_sin.is_cuda
            and rotary_cos_sin.dtype == torch.float32
            and rotary_cos_sin.is_contiguous()
            and rotary_cos_sin.dim() == 3
            and tuple(rotary_cos_sin.shape[1:]) == (2, 32)
        )
        if (
            not supported_main_cache
            or not supported_index_cache
            or not supported_rope_cache
            or out_cache_loc is None
            or not out_cache_loc.is_cuda
            or out_cache_loc.dtype != torch.int32
            or not out_cache_loc.is_contiguous()
            or out_cache_loc.numel() < num_tokens
        ):
            return None

        common_args = (
            out_cache_loc[:num_tokens],
            self.num_heads,
            self.num_key_value_heads,
            self.sparse_num_index_heads,
            self.head_dim,
            rotary_dim,
            norm_eps,
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            rotary_cos_sin,
            position_ids.reshape(-1).contiguous().to(torch.int32),
        )
        if packed is None:
            if not layer_uses_nvfp4:
                return None
            outputs = self._ctx_fused_qkv_gemm_producer(
                hidden_states,
                buffers,
                scale_buffers,
                index_k_cache,
                kv_quant_scale,
                common_args,
                norm_weights,
                norm_eps,
            )
            if outputs is not None:
                _pack_nvfp4_index_k(nvfp4_index_cache, index_k_cache, out_cache_loc, num_tokens)
            return outputs
        if layer_uses_nvfp4:
            q, index_q = _nvfp4_qkv_producer(num_tokens)(
                packed.contiguous(),
                buffers.view(torch.uint8),
                scale_buffers,
                index_k_cache,
                common_args[0],
                kv_quant_scale,
                *common_args[1:],
            )
        else:
            q, index_q = torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
                packed.contiguous(),
                buffers,
                index_k_cache,
                *common_args,
            )
        _pack_nvfp4_index_k(nvfp4_index_cache, index_k_cache, out_cache_loc, num_tokens)
        return q.flatten(1), index_q.flatten(1)

    def _ctx_fused_qkv_gemm_producer(
        self,
        hidden_states: torch.Tensor,
        buffers: torch.Tensor,
        scale_buffers: torch.Tensor,
        index_k_cache: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        common_args: tuple,
        norm_weights: Tuple[torch.Tensor, ...],
        norm_eps: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The QKV-index MXFP8 GEMM with the NVFP4 producer in its epilogue (see _CTX_QKV_FUSED), or None."""
        from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod

        proj = self.qkv_proj
        method = getattr(proj, "quant_method", None)
        num_tokens = int(hidden_states.shape[0])
        n_total = (
            self.num_heads + 2 * self.num_key_value_heads + self.sparse_num_index_heads + 1
        ) * 128
        if (
            not _CTX_QKV_FUSED
            or is_torch_compiling()
            or torch.compiler.is_compiling()
            or get_sm_version() != 107
            or hidden_states.dim() != 2
            or not hidden_states.is_contiguous()
            or not _ctx_qkv_fused_takes_rows(num_tokens)
            or not isinstance(method, MXFP8LinearMethod)
            or not getattr(method, "use_cutlass", False)
            or getattr(proj, "bias", None) is not None
            or tuple(proj.weight.shape) != (n_total, hidden_states.shape[1])
            or _autotuner_tuning()
        ):
            return None
        runner = _ctx_qkv_fused_runner(
            self.num_heads, self.num_key_value_heads, self.sparse_num_index_heads, num_tokens
        )
        q = hidden_states.new_empty(
            (num_tokens, self.num_heads, self.head_dim), dtype=torch.float8_e4m3fn
        )
        index_q = hidden_states.new_empty(
            (num_tokens, self.sparse_num_index_heads, self.head_dim), dtype=torch.float8_e4m3fn
        )
        out_cache_loc, *_, rotary_cos_sin, position_ids = common_args
        # DEP2CTX_NORMQ's MXFP8 activation of these rows when the boundary published it (else quantized here).
        taken = _NORMQ_SLOT.take(hidden_states)
        act, act_sf = (
            taken if taken is not None else torch.ops.trtllm.mxfp8_quantize(hidden_states, True)
        )
        act = act.view(num_tokens, -1)
        try:
            args = runner.make_args(
                act,
                act_sf,
                proj.weight,
                proj.weight_scale,
                proj._mxfp8_alpha,
                q,
                index_q,
                buffers.view(torch.uint8),
                scale_buffers,
                index_k_cache,
                out_cache_loc,
                kv_quant_scale,
                rotary_cos_sin,
                position_ids,
                torch.cat(norm_weights),
                norm_eps,
            )
        except AssertionError:
            # A cache geometry outside the kernel's alignment contract: the caller runs the stock projection +
            # producer (the MXFP8 activation it already took is re-quantized there, bitwise the same).
            return None
        runner.run_args(args)
        if not torch.cuda.is_current_stream_capturing():
            # Eager only: inside a capture (DEP2CTX_A1_FUSED_IN_BCG) a missing handle would record a wasted GEMM.
            _a1_bcg_warm(
                proj,
                act,
                act_sf,
                num_tokens,
                n_total,
                self.num_key_value_heads,
                self.sparse_num_index_heads,
            )
        global _CTX_QKV_FUSED_LOGGED
        if not _CTX_QKV_FUSED_LOGGED:
            _CTX_QKV_FUSED_LOGGED = True
            logger.info(f"MiniMax-M3 CTX fused QKV GEMM + producer taken (rows {num_tokens})")
        _ctx_qkv_fused_count(num_tokens)
        return q.flatten(1), index_q.flatten(1)

    def _expect_fused_qk_norm_rope(self, position_ids: Optional[torch.Tensor]) -> bool:
        """Whether the fused kernel is expected to run instead of the fallback.

        Every M3 config keeps bf16 attention activations, so with position_ids
        present a fallback means the fused kernel silently stopped applying.
        The forward paths assert on this.
        """
        return self.attn_activation_dtype == torch.bfloat16 and position_ids is not None

    def _msa_backend_active(self) -> bool:
        """Whether the MSA fmha_sm100 backend handles this layer's attention."""
        return isinstance(self.attn, MiniMaxM3MsaSparseAttention)

    def _emit_fp8_main_qkv(self) -> bool:
        """Whether the main-branch fused QK-norm+RoPE should emit FP8 q/k/v.

        Only the MSA backend consumes an FP8 paged K/V cache directly; the
        Triton/SDPA reference paths and the index branch stay bf16.
        """
        return self.main_kv_is_fp8 and self._msa_backend_active()

    def _split_main_qkv(
        self, fused_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused [q|k|v] buffer into per-tensor q/k/v.

        The MSA kernels read q with its real strides and scatter k/v into the
        paged cache with an indexed copy, so the split column-views can be
        handed over as-is. Other backends keep the previous contiguity.
        """
        q, k, v = fused_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self._msa_backend_active():
            return q, k, v
        return q.contiguous(), k.contiguous(), v

    def _split_index_qk(self, fused_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the fused [idx_q|idx_k] buffer into per-tensor idx_q/idx_k.

        The index analogue of _split_main_qkv; the index cache is bf16, so there
        is no FP8 variant here.
        """
        idx_q, idx_k = fused_idx.split([self.index_q_size, self.index_k_size], dim=-1)
        if self._msa_backend_active():
            return idx_q, idx_k
        return idx_q.contiguous(), idx_k.contiguous()

    def forward(
        self,
        position_ids: Optional[torch.IntTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ):
        """Dispatch sparse layers to the MiniMax-M3 sparse algorithm.

        Dense layers (0-2 in the M3 checkpoint) fall through to the base
        :class:`Attention` forward unchanged. Sparse layers (3+) drive
        the two-step block-sparse attention (index attention + top-k
        block selection + sparse GQA) implemented in
        :mod:`tensorrt_llm._torch.attention.backends.sparse.minimax_m3`.

        The sparse forward expects ``attn_metadata.kv_cache_manager`` to
        be a :class:`MiniMaxM3KVCacheManagerV2` (the cache manager
        produced by
        :func:`get_sparse_attn_kv_cache_manager` for
        ``algorithm='minimax_m3'``). The cache manager owns both the
        standard paged main K/V buffer and a per-sparse-layer paged
        side index-K buffer.
        """
        if not self.is_sparse_attention_layer:
            return self._dense_forward(
                position_ids,
                hidden_states,
                attn_metadata,
                all_reduce_params=all_reduce_params,
                **kwargs,
            )
        return self._sparse_forward(
            position_ids,
            hidden_states,
            attn_metadata,
            all_reduce_params=all_reduce_params,
            **kwargs,
        )

    def _dense_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply MiniMax Q/K norm and partial RoPE, then dispatch dense attention."""
        if attn_metadata is None:
            raise RuntimeError(
                f"MiniMax-M3 dense forward (layer {self.layer_idx}) requires "
                "attn_metadata; received None."
            )

        # Projections (no index branch).
        qkv = self.qkv_proj(hidden_states)

        # Per-head Gemma RMSNorm and partial RoPE on Q/K.
        fused_qkv = self._fused_qk_norm_rope(
            qkv,
            position_ids,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_key_value_heads,
            num_heads_v=self.num_key_value_heads,
            head_dim=self.head_dim,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            out_fp8=self._emit_fp8_main_qkv(),
        )
        if fused_qkv is not None:
            q, k, v = self._split_main_qkv(fused_qkv)
        else:
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 dense attention (layer {self.layer_idx}) expected the "
                f"fused QK-norm+RoPE kernel (bf16 activations, head_dim="
                f"{self.head_dim}) but fell back to the separate path; qkv dtype "
                f"is {qkv.dtype} (expected {self.attn_activation_dtype})."
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.apply_qk_norm(q, k)
            if self.rotary_emb is not None and position_ids is not None:
                q, k = self.rotary_emb(position_ids, [q, k])

        # Keep token-wise projections and the output projection visible to
        # torch.compile. Only the metadata/cache-dependent attention core is
        # hidden behind the inplace custom op.
        o = self._forward_attention_core(q, k, v, None, None, attn_metadata)
        # all_reduce_params lets the decoder defer the o_proj output AllReduce so
        # it can be fused with post_attention_layernorm (RESIDUAL_RMS_NORM).
        # Passing None preserves the standalone o_proj reduction used by the
        # single-GPU, attention-DP, and fusion-disabled paths.
        return self.o_proj(o, all_reduce_params=all_reduce_params)

    def _sdpa_dense_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run dense cache updates and attention into ``output`` (SDPA path)."""
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if kv_cache_manager is None:
            raise RuntimeError(
                f"MiniMax-M3 dense forward (layer {self.layer_idx}) requires "
                "attn_metadata.kv_cache_manager to be a MiniMaxM3KVCacheManagerV2."
            )

        num_tokens = int(q.shape[0])
        q_view = q.view(num_tokens, self.num_heads, self.head_dim)
        k_view = k.view(num_tokens, self.num_key_value_heads, self.head_dim)
        v_view = v.view(num_tokens, self.num_key_value_heads, self.head_dim)

        # Preserve the strided pool view so writes update the backing cache.
        # Flattening it can allocate a copy; the helpers address (page, slot).
        kv_pool = kv_cache_manager.get_buffers(self.layer_idx)
        k_cache_view = kv_pool[:, 0]
        v_cache_view = kv_pool[:, 1]

        # 5. Read the pre-built M3 metadata. Building is the
        # responsibility of ``MiniMaxM3AttentionMetadata.prepare`` (or
        # the test path), so the forward stays CUDA-graph-capture safe.
        m3_attachment = getattr(attn_metadata, "minimax_m3", None)
        if m3_attachment is None:
            raise RuntimeError(
                f"MiniMax-M3 dense forward (layer {self.layer_idx}) requires "
                "attn_metadata.minimax_m3 to be pre-built by "
                "MiniMaxM3AttentionMetadata.prepare(); the model_engine "
                "configures the M3 backend's Metadata class so this happens "
                "automatically. Test callers must attach minimax_m3 manually."
            )
        m3_meta = m3_attachment["metadata"]
        out_cache_loc = m3_attachment["out_cache_loc"]

        # 6. Write the new tokens' K/V to the pool. All inputs come
        # from prepare() which lands them on the cache device, so the
        # only ``.to(...)`` is a same-device dtype conversion (safe
        # under CUDA-graph capture).
        _write_main_kv_slots_to_pool(kv_pool, 0, out_cache_loc, k_view)
        _write_main_kv_slots_to_pool(kv_pool, 1, out_cache_loc, v_view)

        # 7. Gather padded K/V for every batch row and run dense GQA.
        batch = int(m3_meta.slot_ids.shape[0])
        # Under CUDA graphs bake a fixed gather/mask width: min(page-table width,
        # max_seq_len). The raw table width is inflated by the KV-estimation pass
        # and would OOM the gather; the seq_lens mask hides the slack.
        if attn_metadata.is_cuda_graph:
            capacity = int(m3_meta.req_to_token.shape[1])
            max_k = min(capacity, int(attn_metadata.max_seq_len or capacity))
        else:
            max_k = int(m3_meta.max_seqlen_k)
        if max_k <= 0:
            max_k = 1
        # ``_gather_paged_batched`` decomposes the flat slot id into
        # ``(page, within)`` when handed a 4-D pool view; the result is
        # ``[batch, max_k, num_kv_heads, head_dim]`` exactly as the
        # legacy flat-slot path produced.
        k_padded = _gather_paged_batched(
            k_cache_view, m3_meta.req_to_token, m3_meta.slot_ids, max_k
        )
        v_padded = _gather_paged_batched(
            v_cache_view, m3_meta.req_to_token, m3_meta.slot_ids, max_k
        )

        # SDPA expands KV heads to query heads for prefill. Decode processes
        # one KV head at a time to bound the temporary GQA expansion.
        if self.num_heads % max(self.num_key_value_heads, 1) != 0:
            raise RuntimeError(
                f"Dense GQA expects num_heads ({self.num_heads}) divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )
        group = self.num_heads // max(self.num_key_value_heads, 1)

        # Build the per-query attention mask that masks out padded KV
        # positions beyond each sequence's true ``seq_lens`` and (for
        # prefill) preserves causality. ``q_positions`` from the prefill
        # metadata names each Q token's K-side position.
        # The metadata tensors are produced by
        # :meth:`MiniMaxM3AttentionMetadata.prepare` on the cache
        # device, so ``.to(dtype=torch.long)`` is a same-device dtype
        # conversion (capture-safe).
        seq_lens_dev = m3_meta.seq_lens.to(dtype=torch.long)
        kv_positions = torch.arange(max_k, device=q.device).unsqueeze(0)  # [1, max_k]

        if m3_meta.is_prefill:
            if group > 1:
                k_padded = k_padded.repeat_interleave(group, dim=2)
                v_padded = v_padded.repeat_interleave(group, dim=2)
            # Prefill: build [total_q, max_k] mask using q_positions / q_batch_row.
            # Prefill never runs inside the CUDA-graph capture window
            # (capture is decode-only), so the per-batch Python loop and
            # the ``.tolist()`` below are acceptable.
            assert m3_meta.q_batch_row is not None and m3_meta.q_positions is not None, (
                "prefill metadata requires q_batch_row + q_positions; call .prepare()"
            )
            q_batch_row = m3_meta.q_batch_row.to(dtype=torch.long)
            q_positions = m3_meta.q_positions.to(dtype=torch.long)
            seq_lens_per_q = seq_lens_dev[q_batch_row]  # [total_q]
            # Each Q token attends to KV positions [0, q_position] (causal).
            # Also mask out KV positions >= seq_lens_per_q (padding).
            valid = (kv_positions <= q_positions.unsqueeze(-1)) & (
                kv_positions < seq_lens_per_q.unsqueeze(-1)
            )  # [total_q, max_k]
            # Build per-batch attention by routing each Q to its K via q_batch_row.
            # Easiest path: SDPA expects [batch, num_heads, q_len, head_dim].
            # We process each batch row independently to keep the math straightforward.
            output_view = output.view(-1, self.num_heads, self.head_dim)
            cu = m3_meta.cu_seqlens_q.to(torch.long).tolist()
            for b in range(batch):
                start, end = cu[b], cu[b + 1]
                if end <= start:
                    continue
                q_b = q_view[start:end].transpose(0, 1).unsqueeze(0)  # [1, H, q, d]
                k_b = k_padded[b].transpose(0, 1).unsqueeze(0)  # [1, H, k, d]
                v_b = v_padded[b].transpose(0, 1).unsqueeze(0)  # [1, H, k, d]
                mask_b = valid[start:end].unsqueeze(0).unsqueeze(0)  # [1, 1, q, k]
                # SDPA: when attn_mask is a bool tensor, True = attend, False = mask.
                with sdpa_kernel(_DENSE_SDPA_BACKENDS):
                    out_b = torch.nn.functional.scaled_dot_product_attention(
                        q_b.to(q.dtype),
                        k_b.to(q.dtype),
                        v_b.to(q.dtype),
                        attn_mask=mask_b,
                        dropout_p=0.0,
                        is_causal=False,
                    )  # [1, H, q, d]
                output_view[start:end].copy_(out_b.squeeze(0).transpose(0, 1))
        else:
            # Decode: one query token per request at position seq_lens - 1.
            valid = kv_positions < seq_lens_dev.unsqueeze(-1)  # [batch, max_k]
            q_b = q_view.view(batch, 1, self.num_heads, self.head_dim).transpose(
                1, 2
            )  # [batch, H, 1, d]
            mask_b = valid.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, k]
            # Expand K/V one KV head at a time: all heads at once needs an
            # O(batch * max_k * num_heads) temporary that overflows the CUDA-graph
            # pool under attention DP.
            out_b = q.new_empty(batch, self.num_heads, 1, self.head_dim)
            with sdpa_kernel(_DENSE_SDPA_BACKENDS):
                for h in range(max(self.num_key_value_heads, 1)):
                    qh = slice(h * group, (h + 1) * group)
                    k_h = k_padded[:, :, h : h + 1].repeat_interleave(group, dim=2)
                    v_h = v_padded[:, :, h : h + 1].repeat_interleave(group, dim=2)
                    out_b[:, qh] = torch.nn.functional.scaled_dot_product_attention(
                        q_b[:, qh].to(q.dtype),
                        k_h.transpose(1, 2).to(q.dtype),
                        v_h.transpose(1, 2).to(q.dtype),
                        attn_mask=mask_b,
                        dropout_p=0.0,
                        is_causal=False,
                    )  # [batch, group, 1, d]
            # Drop the singleton query axis; a transpose(1, 2).reshape would
            # scramble (head, head_dim) when H != head_dim.
            output.view(batch, self.num_heads, self.head_dim).copy_(out_b.squeeze(2))

        return output

    def _a1_fused_in_bcg_takes(self, attn_metadata: AttentionMetadata) -> bool:
        """DEP2CTX_A1_FUSED_IN_BCG: whether this sparse layer's QKV runs the A1 fused producer inside the breakable
        capture (a capture, NVFP4 main KV and indexer, the layer's static cache views staged). Replays run no Python;
        warm-ups are eager and take the eager fused path."""
        return (
            _A1_FUSED_IN_BCG
            and _ITEMS_IN_BCG
            and _A1_IN_BCG
            and self.register_to_config
            and is_in_breakable_cuda_graph()
            and torch.cuda.is_current_stream_capturing()
            and isinstance(self.attn, MiniMaxM3MsaSparseAttention)
            and self.main_kv_is_nvfp4
            and self.attn.indexer_kv_dtype == "nvfp4"
            and getattr(attn_metadata, "msa_out_cache_loc", None) is not None
            and self.layer_idx in (getattr(attn_metadata, "msa_layer_cache_tensors", None) or {})
            and getattr(self, "_bcg_ptrs", None) is not None
        )

    def _forward_attention_core(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # q may be FP8 on the MSA FP8-KV path, so pin the output to the compute
        # dtype rather than inheriting it from q.
        output = q.new_empty(
            (q.shape[0], self.num_heads * self.head_dim),
            dtype=self.attn_activation_dtype,
        )
        if self.register_to_config and (is_torch_compiling() or is_in_breakable_cuda_graph()):
            maybe_bcg_minimax_m3_attn_custom_op_inplace(
                q,
                k,
                v,
                idx_q,
                idx_k,
                None,
                None,
                self.layer_idx_str,
                output,
            )
        else:
            # A step that runs here rather than through compile is padded all
            # the same, since the bucket is agreed across ranks.
            _dispatch_attention_over_live_tokens(self, q, k, v, idx_q, idx_k, attn_metadata, output)
        return output

    def _dispatch_attention_backend(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Route the attention core to the configured backend.

        MSA handles both dense and sparse layers. The Triton configuration
        uses MiniMaxM3SparseRuntimeBackend for sparse layers and
        TrtllmAttention for unquantized dense layers when TRTLLM-Gen is available.
        FP8 and NVFP4 KV caches retain the SDPA dense path.

        * MSA → :meth:`_msa_attention_core`
        * Triton sparse → :meth:`_triton_sparse_attention_core`
        * Dense → generic TRTLLM paged attention
        """
        if isinstance(self.attn, MiniMaxM3MsaSparseAttention):
            return self._msa_attention_core(q, k, v, idx_q, idx_k, attn_metadata, output)
        assert k is not None and v is not None
        if self.is_sparse_attention_layer:
            assert idx_q is not None and idx_k is not None
            return self._triton_sparse_attention_core(q, k, v, idx_q, idx_k, attn_metadata, output)
        assert idx_q is None and idx_k is None
        if isinstance(self.attn, MiniMaxM3SparseRuntimeBackend):
            return self._sdpa_dense_attention_core(q, k, v, attn_metadata, output)
        if attn_metadata.kv_cache_manager is not None and attn_metadata.kv_cache_manager.dtype in (
            DataType.FP8,
            DataType.NVFP4,
        ):
            raise ValueError(
                "MiniMax-M3 native dense attention requires an unquantized KV cache; "
                "the model quantization config and KV cache manager disagree."
            )
        # TRTLLM-Gen self-attention requires packed QKV when updating the cache.
        # Pack only current tokens; separate K/V inputs are not supported.
        # Historical K/V stays in the paged cache and is never gathered here.
        qkv = torch.cat((q, k, v), dim=-1)
        self.attn.forward(
            qkv, None, None, attn_metadata, forward_args=AttentionForwardArgs(output=output)
        )
        return output

    def _msa_attention_core(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run the MSA backend (:class:`MiniMaxM3MsaSparseAttention`).

        The backend runs the sparse GQA or dense paged GQA through its inherited
        FMHA forward; this layer selects the top-k blocks (sparse only) and
        builds the forward_args the FMHA reads.

        Unless a producer has already populated the caches, write_layer_caches
        stores new-token K/V and any live index-K before the indexer's proxy
        pass reads the cache. FP8 index-K is then omitted from run_indexer;
        BF16 retains its live tensor with idx_k_prewritten=True. forward()
        receives k=v=None so neither FMHA phase writes them again.
        """
        assert (k is None) == (v is None)
        kv_scale_orig_quant = (
            getattr(self.qkv_proj, "inv_kv_scales", None) if self.main_kv_is_nvfp4 else None
        )
        kv_scale_quant_orig = (
            getattr(self.qkv_proj, "kv_scales", None) if self.main_kv_is_nvfp4 else None
        )
        if self.main_kv_is_nvfp4 and (kv_scale_orig_quant is None or kv_scale_quant_orig is None):
            raise RuntimeError("MiniMax-M3 NVFP4 KV cache requires QKV quantization scales")
        if self.is_sparse_attention_layer:
            assert idx_q is not None
            # Unfused PCG supplies live FP8 index-K; eager FP8 producers may
            # already have inserted it and supply None instead.
            if k is not None:
                self.attn.write_layer_caches(
                    k, v, idx_k, attn_metadata, kv_scale_orig_quant=kv_scale_orig_quant
                )
                if self.attn.indexer_kv_dtype == "fp8":
                    # The FP8 indexer accepts only an already-populated cache.
                    idx_k = None
            else:
                # The horizontal producer has already written both caches.
                assert idx_k is None
            # Publish the selected blocks so the FMHA runs the sparse path.
            # idx_k_prewritten: index-K is already in the cache (written here
            # or by an FP8 producer), so run_indexer must not write it.
            kv_block_indexes = self.attn.run_indexer(
                idx_q, idx_k, attn_metadata, idx_k_prewritten=True
            )
            forward_args = AttentionForwardArgs(
                output=output,
                sparse_backend_args=SparseBackendForwardArgs(topk_indices=kv_block_indexes),
            )
        else:
            assert idx_q is None and idx_k is None
            if k is not None:
                self.attn.write_layer_caches(
                    k, v, None, attn_metadata, kv_scale_orig_quant=kv_scale_orig_quant
                )
            # No top-k selection means the FMHA attends the full page table.
            forward_args = AttentionForwardArgs(output=output)
        forward_args.kv_scale_orig_quant = kv_scale_orig_quant
        forward_args.kv_scale_quant_orig = kv_scale_quant_orig
        self.attn.forward(q, None, None, attn_metadata, forward_args=forward_args)
        return output

    def _sparse_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run a MiniMax-M3 sparse attention forward end-to-end.

        Steps:
          1. Project ``hidden_states`` to Q/K/V (fused ``qkv_proj``)
             plus index Q (per-head) and index K (single replicated).
          2-3. Apply per-head Gemma RMSNorm and partial RoPE to the main and
             index branches, each fused into one kernel by
             :meth:`_fused_qk_norm_rope` when it applies. The index branch
             passes num_heads_v=0 because it carries no value heads.
          4. Pull paged main K/V cache (reshaped to flat-slot view) and
             paged side index-K cache from the
             :class:`MiniMaxM3KVCacheManagerV2`.
          5. Build a :class:`MiniMaxM3TritonSparseAttentionMetadata` from the
             standard :class:`AttentionMetadata` (using ``request_ids``
             + ``seq_lens`` + ``num_cached_tokens_per_seq``).
          6. Write the new token's K/V/idx_K to the slots named by the
             per-token ``out_cache_loc`` derived from
             ``req_to_token``.
          7. Dispatch to :meth:`MiniMaxM3SparseRuntimeBackend.forward`
             which runs the sparse path end-to-end.
          8. Apply ``o_proj`` and return.

        Production callers (the LLM API path) drive
        :meth:`MiniMaxM3AttentionMetadata.prepare` outside any
        CUDA-graph capture window; that method attaches a pre-built
        :class:`MiniMaxM3TritonSparseAttentionMetadata` and an
        ``out_cache_loc`` tensor as ``attn_metadata.minimax_m3``. Test
        callers attach the same dict manually.  This forward path
        always reads the pre-built attachment and never builds metadata
        itself — both would trigger
        ``cudaErrorStreamCaptureUnsupported`` for the
        ``cuda_graph=True`` hard path because the build does
        CPU->GPU copies.
        """
        if attn_metadata is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                "attn_metadata; received None."
            )

        # The opt-in projection emits [Q|K|V|index-Q|index-K] with one GEMM.
        # Its horizontal producer writes both paged caches directly and returns
        # only compact Q/index-Q. Unsupported compile/geometry/layout cases
        # split the packed output and retain the existing producer paths.
        packed_qkv = None
        packed_idx_qk = None
        if self.enable_fused_qkv_index_projection:
            if (
                self.register_to_config
                and is_torch_compiling()
                and isinstance(self.attn, MiniMaxM3MsaSparseAttention)
                and (self.main_kv_is_fp8 or self.main_kv_is_nvfp4)
                and self.attn.indexer_kv_dtype == "fp8"
            ):
                # Metadata stages these zero-copy views before compilation;
                # tracing the cache manager's native pointer access is unsafe.
                kv_cache, index_k_cache, kv_scale_cache = attn_metadata.msa_layer_cache_tensors[
                    self.layer_idx
                ]
                q, idx_q = minimax_m3_fused_sparse_qkv_producer(
                    hidden_states,
                    position_ids,
                    kv_cache,
                    index_k_cache,
                    kv_scale_cache,
                    attn_metadata.msa_out_cache_loc,
                    self.layer_idx_str,
                )
                o = self._forward_attention_core(q, None, None, idx_q, None, attn_metadata)
                return self.o_proj(o, all_reduce_params=all_reduce_params)
            if self._a1_fused_in_bcg_takes(attn_metadata):
                # DEP2CTX_A1_FUSED_IN_BCG: the fused producer in the captured segment at the bucket's rows; the break
                # then runs on q / index-q (pad rows' slots are -1, so their cache stores are skipped).
                caches = attn_metadata.msa_layer_cache_tensors[self.layer_idx]
                loc = attn_metadata.msa_out_cache_loc
                horizontal = self._fused_fp8_qkv_indexer_norm_rope_kv_insert(
                    None,
                    position_ids,
                    attn_metadata,
                    cache_tensors=(*caches, loc),
                    hidden_states=hidden_states,
                )
                if horizontal is not None:
                    self._bcg_ptrs.note(self.layer_idx, _a1_bcg_ptrs(attn_metadata, self.layer_idx))
                    _bcg_note("A1_FUSED", hidden_states)
                    q, idx_q = horizontal
                    o = self._forward_attention_core(q, None, None, idx_q, None, attn_metadata)
                    return self.o_proj(o, all_reduce_params=all_reduce_params)
            if self.register_to_config and (is_torch_compiling() or is_in_breakable_cuda_graph()):
                # Keep the projection in the captured segment while hiding
                # its shape-specializing MXFP8 internals behind a symbolic
                # fake implementation. Cache insertion and MSA remain in the
                # existing eager attention boundary below.
                packed = minimax_m3_qkv_index_proj(hidden_states, position_ids, self.layer_idx_str)
                num_tokens = (
                    position_ids.shape[-1] if position_ids is not None else hidden_states.shape[0]
                )
                output = packed.new_empty(
                    (num_tokens, self.num_heads * self.head_dim),
                    dtype=self.attn_activation_dtype,
                )
                maybe_bcg_minimax_m3_attn_custom_op_inplace(
                    None,
                    None,
                    None,
                    None,
                    None,
                    packed,
                    position_ids,
                    self.layer_idx_str,
                    output,
                )
                return self.o_proj(output, all_reduce_params=all_reduce_params)
            horizontal = self._fused_fp8_qkv_indexer_norm_rope_kv_insert(
                None, position_ids, attn_metadata, hidden_states=hidden_states
            )
            if horizontal is not None:
                q, idx_q = horizontal
                o = self._forward_attention_core(q, None, None, idx_q, None, attn_metadata)
                return self.o_proj(o, all_reduce_params=all_reduce_params)
            packed = self.qkv_proj(hidden_states)
            horizontal = self._fused_fp8_qkv_indexer_norm_rope_kv_insert(
                packed, position_ids, attn_metadata
            )
            if horizontal is not None:
                q, idx_q = horizontal
                o = self._forward_attention_core(q, None, None, idx_q, None, attn_metadata)
                return self.o_proj(o, all_reduce_params=all_reduce_params)
            main_size = self.q_size + 2 * self.kv_size
            packed_qkv, packed_idx_qk = packed.split(
                [main_size, self.index_q_size + self.index_k_size], dim=-1
            )

        # Project, norm, and apply RoPE for the compatibility/fallback main and
        # index branches. Both read only hidden_states and write disjoint
        # outputs, so they overlap on the aux stream and join before attention.
        def _main_norm_rope():
            qkv = packed_qkv if packed_qkv is not None else self.qkv_proj(hidden_states)
            fused_qkv = self._fused_qk_norm_rope(
                qkv,
                position_ids,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_key_value_heads,
                num_heads_v=self.num_key_value_heads,
                head_dim=self.head_dim,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                out_fp8=self._emit_fp8_main_qkv(),
            )
            if fused_qkv is not None:
                return self._split_main_qkv(fused_qkv)
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 sparse attention (layer {self.layer_idx}) expected the "
                f"fused QK-norm+RoPE kernel (bf16 activations, head_dim="
                f"{self.head_dim}) but fell back to the separate path; qkv dtype "
                f"is {qkv.dtype} (expected {self.attn_activation_dtype})."
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.apply_qk_norm(q, k)
            if self.rotary_emb is not None and position_ids is not None:
                q, k = self.rotary_emb(position_ids, [q, k])
            return q, k, v

        def _index_norm_rope():
            """Project and normalize index queries, with RoPE and cache updates."""
            idx_qk = (
                packed_idx_qk if packed_idx_qk is not None else self.index_qk_proj(hidden_states)
            )
            graph_fp8_indexer = (
                self.register_to_config
                and is_torch_compiling()
                and isinstance(self.attn, MiniMaxM3MsaSparseAttention)
                and self.attn.indexer_kv_dtype == "fp8"
            )
            if not graph_fp8_indexer:
                fp8_idx_q = self._fused_fp8_index_qk_norm_rope(idx_qk, position_ids, attn_metadata)
                if fp8_idx_q is not None:
                    # Index-K was inserted directly into the paged side cache.
                    return fp8_idx_q, None
            # During compilation keep norm/RoPE/FP8 conversion captured, but
            # leave dynamic index-K cache insertion in the attention boundary.
            fused_idx = self._fused_qk_norm_rope(
                idx_qk,
                position_ids,
                num_heads_q=self.sparse_num_index_heads,
                num_heads_k=1,
                num_heads_v=0,
                head_dim=self.sparse_index_dim,
                q_norm=self.index_q_norm,
                k_norm=self.index_k_norm,
                out_fp8=graph_fp8_indexer,
            )
            if fused_idx is not None:
                return self._split_index_qk(fused_idx)
            assert not self._expect_fused_qk_norm_rope(position_ids), (
                f"MiniMax-M3 sparse index branch (layer {self.layer_idx}) expected the "
                f"fused QK-norm+RoPE kernel (bf16 activations, index_dim="
                f"{self.sparse_index_dim}) but fell back to the separate path; idx "
                f"dtype is {idx_qk.dtype} (expected {self.attn_activation_dtype})."
            )
            idx_q, idx_k = idx_qk.split([self.index_q_size, self.index_k_size], dim=-1)
            idx_q, idx_k = self.apply_index_qk_norm(idx_q, idx_k)
            if self.rotary_emb is not None and position_ids is not None:
                idx_q, idx_k = self.rotary_emb(position_ids, [idx_q, idx_k])
            return idx_q, idx_k

        (q, k, v), (idx_q, idx_k) = maybe_execute_in_parallel(
            _main_norm_rope,
            _index_norm_rope,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
            disable_on_compile=True,
        )

        o = self._forward_attention_core(q, k, v, idx_q, idx_k, attn_metadata)
        # all_reduce_params lets the decoder defer the o_proj output AllReduce so
        # it can be fused with post_attention_layernorm (RESIDUAL_RMS_NORM).
        # Passing None preserves the standalone o_proj reduction used by the
        # single-GPU, attention-DP, and fusion-disabled paths.
        return self.o_proj(o, all_reduce_params=all_reduce_params)

    def _triton_sparse_attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse cache updates and attention into ``output`` (Triton path)."""
        kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if kv_cache_manager is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                "attn_metadata.kv_cache_manager to be a MiniMaxM3KVCacheManagerV2."
            )

        # 4. Get the paged-block main K/V cache + flat side index-K cache.
        # The base KVCacheManagerV2 layout is
        # ``[num_pages, kv_factor, tokens_per_block, num_kv_heads, head_dim]``
        # with NHD layout. Pass the multi-dim view ``kv_pool[:, kv_index]``
        # ``[num_pages, tokens_per_block, num_kv_heads, head_dim]`` directly:
        # ``_gather_paged_batched`` decomposes the flat slot id into
        # ``(page, within)`` for 4-D caches, and writes go through
        # :func:`_write_main_kv_slots_to_pool` which uses multi-dim
        # fancy assignment. The previously used reshape pattern
        # silently copied the K (or V) slice because ``kv_pool[:, 0]``
        # is non-contiguous, so writes never propagated to the pool —
        # the bug that drove the dense layer-0 decode attention to
        # near-zero output and produced the GSM8K-100 0.0 score.
        kv_pool = kv_cache_manager.get_buffers(self.layer_idx)
        # Index 0 = K, 1 = V on the kv_factor axis.
        k_cache = kv_pool[:, 0]
        v_cache = kv_pool[:, 1]
        if not hasattr(kv_cache_manager, "get_index_k_buffer"):
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires the "
                "kv_cache_manager to be a MiniMaxM3KVCacheManagerV2 "
                f"(got {type(kv_cache_manager).__name__})."
            )
        idx_k_cache = kv_cache_manager.get_index_k_buffer(self.layer_idx)
        idx_v_cache = (
            kv_cache_manager.get_index_v_buffer(self.layer_idx)
            if hasattr(kv_cache_manager, "get_index_v_buffer")
            else None
        )

        # 5. Read the pre-built M3 metadata. Building is the
        # responsibility of ``MiniMaxM3AttentionMetadata.prepare`` (or
        # the test path), so the forward stays CUDA-graph-capture safe.
        m3_attachment = getattr(attn_metadata, "minimax_m3", None)
        if m3_attachment is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                "attn_metadata.minimax_m3 to be pre-built by "
                "MiniMaxM3AttentionMetadata.prepare(); the model_engine "
                "configures the M3 backend's Metadata class so this happens "
                "automatically. Test callers must attach minimax_m3 manually."
            )
        m3_meta = m3_attachment["metadata"]
        out_cache_loc = m3_attachment["out_cache_loc"]

        if not self.disable_index_value and idx_v_cache is not None:
            # The shared idx_v_proj is not part of the M3 checkpoint
            # bring-up (disable_index_value=True everywhere); future
            # variants would project idx_v from hidden_states the same
            # way idx_k is projected.
            raise NotImplementedError(
                "Index V projection is not implemented (M3 checkpoint sets "
                "disable_index_value=True on every sparse layer)."
            )

        # 6-7. Dispatch to the registered MiniMax-M3 sparse runtime
        # backend, which executes the sparse path end-to-end including
        # the cache writes. Production construction (LLM API with
        # ``sparse_attention_config=MiniMaxM3SparseAttentionConfig()``)
        # registers :class:`MiniMaxM3SparseRuntimeBackend` as
        # ``self.attn``; any other backend on a sparse layer is a
        # configuration error.
        if not isinstance(self.attn, MiniMaxM3SparseRuntimeBackend):
            raise RuntimeError(
                f"MiniMax-M3 sparse forward (layer {self.layer_idx}) requires "
                f"self.attn to be a MiniMaxM3SparseRuntimeBackend, got "
                f"{type(self.attn).__name__}. Construct the model with "
                "sparse_attention_config=MiniMaxM3SparseAttentionConfig(...) on "
                "ModelConfig so the standard attention-backend dispatch selects "
                "the M3 sparse runtime."
            )
        return self.attn.forward(
            q,
            k,
            v,
            None,
            output=output,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_v=None,  # disable_index_value=True for M3 checkpoint
            k_cache=k_cache,
            v_cache=v_cache,
            idx_k_cache=idx_k_cache,
            idx_v_cache=idx_v_cache,
            out_cache_loc=out_cache_loc,
            m3_metadata=m3_meta,
        )


class MiniMaxM3DecoderLayer(DecoderLayer):
    """One M3 transformer block (dense or sparse, MLP or MoE)."""

    def __init__(
        self,
        model_config: "ModelConfig[PretrainedConfig]",
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        _, sparse_layer_ids = get_sparse_layer_ids(config)
        disable_index_value_ids = set(get_sparse_disable_index_value_layer_ids(config))
        is_sparse = layer_idx in sparse_layer_ids
        disable_index_value = layer_idx in disable_index_value_ids

        self.self_attn = MiniMaxM3Attention(
            model_config=model_config,
            layer_idx=layer_idx,
            is_sparse_attention_layer=is_sparse,
            disable_index_value=disable_index_value,
            aux_stream=aux_stream_dict[AuxStreamType.Attention],
        )

        _, moe_layer_ids = get_moe_layer_ids(config)
        if layer_idx in moe_layer_ids:
            self.block_sparse_moe = MiniMaxM3MoE(
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                layer_idx=layer_idx,
            )
            self.mlp = None
        else:
            dense_intermediate = int(
                getattr(config, "dense_intermediate_size", config.intermediate_size)
            )
            self.mlp = _build_swiglu_oai_dense_mlp(
                model_config=model_config,
                intermediate_size=dense_intermediate,
                layer_idx=layer_idx,
            )
            self.block_sparse_moe = None

        # Layer-boundary RMSNorms are plain (non-Gemma) norms so they can drive
        # the fused residual add + RMSNorm + MXFP8 boundary (DEP2CTX_NORMQ),
        # whose kernel applies a plain weight * x scaling with no Gemma
        # (1 + weight) offset. When the checkpoint stores Gemma norms
        # (use_gemma_norm=True), the loader folds (1 + weight) into the stored
        # weight at load time (see _fold_gemma_boundary_norm_weights), so the
        # runtime norm is numerically identical to the original Gemma norm on
        # every path. The per-head q/k/index norms keep use_gemma because they
        # are consumed by the separate fused_qk_norm_rope kernel, which handles
        # Gemma directly.
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )

        # Attention DP: every rank owns its tokens, so no layer boundary reduces
        # anything; both boundaries are the residual add + RMSNorm of this
        # rank's rows.

        # Wired by MiniMaxM3ForCausalLM.setup_aliases after weight load to the
        # next layer's input_layernorm (or the final model norm for the last
        # layer). None leaves the boundary norm to the caller.
        self.next_layer_layernorm: Optional[RMSNorm] = None
        # DEP2CTX_NORMQ: set by setup_aliases, False on the last layer (the final norm feeds lm_head).
        self._normq_feeds_mxfp8 = False

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Layer-0 prologue only. For every subsequent layer the input_layernorm
        # (an add+RMSNorm at the layer boundary) was already applied by the
        # previous layer as its next_layer_layernorm, so residual is not None
        # here and this block is skipped (matches DeepSeek-V3).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            lora_params=lora_params,
            **kwargs,
        )

        if self.block_sparse_moe is not None:
            hidden_states, residual = self.forward_MoE(
                hidden_states,
                attn_metadata,
                residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )
        else:
            hidden_states, residual = self.forward_mlp(
                hidden_states,
                residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )

        return hidden_states, residual

    def _apply_pre_feed_forward_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The residual add + RMSNorm between attention and the feed-forward: the NORMQ boundary (which also writes
        the feed-forward's MXFP8 input) when it takes the step, else the plain add + RMSNorm."""
        fused = _normq_boundary(
            hidden_states,
            residual,
            self.post_attention_layernorm,
            moe=self.block_sparse_moe,
            all_rank_num_tokens=getattr(attn_metadata, "all_rank_num_tokens", None),
        )
        if fused is not None:
            return fused
        return self.post_attention_layernorm(hidden_states, residual)

    def _apply_next_layer_layernorm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the next layer's input_layernorm (or the final model norm for the last layer) at the layer boundary.

        When next_layer_layernorm has not been wired (e.g. a standalone unit test that never ran setup_aliases) the
        (hidden_states, residual) pair is returned unchanged so the model can apply the final norm itself.
        """
        if self.next_layer_layernorm is None:
            return hidden_states, residual
        # Only a next decoder layer's QKV consumes the MXFP8 copy (not the final norm's lm_head).
        if self._normq_feeds_mxfp8:
            fused = _normq_boundary(hidden_states, residual, self.next_layer_layernorm)
            if fused is not None:
                return fused
        return self.next_layer_layernorm(hidden_states, residual)

    def forward_MoE(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self._apply_pre_feed_forward_norm(
            hidden_states, residual, attn_metadata
        )

        # DEP2CTX_TOPK_FOLD: armed, the MoE's tail add parks its deferred top-k reduce for this layer's boundary.
        fold = _TOPK_FOLD and _topk_fold_arm(self, hidden_states, spec_metadata)
        try:
            hidden_states = self.block_sparse_moe(
                hidden_states,
                attn_metadata,
                lora_params=lora_params,
            )
        finally:
            parked = _TOPK_FOLD_SLOT.take() if fold else None
        if fold:
            fused = _topk_fold_hand_off(self, parked, hidden_states, residual)
            if fused is not None:
                if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                    _topk_capture(spec_metadata, self.layer_idx, fused[1])
                return fused

        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
        hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self._apply_pre_feed_forward_norm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states, lora_params=lora_params)

        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
        hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)
        return hidden_states, residual


class MiniMaxM3Model(DecoderModel):
    """M3 text decoder model."""

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        _validate_sparse_attention_runtime_config(model_config)
        super().__init__(model_config)
        quant_config = model_config.quant_config
        if quant_config is None or (
            (not quant_config.quant_mode.has_fp8_kv_cache())
            and (not quant_config.quant_mode.has_fp4_kv_cache())
        ):
            model_config.pretrained_config.torch_dtype = torch.bfloat16
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        # Aux streams shared across layers, matching the DeepSeekV3 convention:
        # one for attention branch overlap, one for MoE shared/routed parallel
        # execution, and one for MoE chunking overlap inside the fused MoE kernel.
        self.aux_stream_dict = {
            AuxStreamType.Attention: torch.cuda.Stream(),
            AuxStreamType.MoeShared: torch.cuda.Stream(),
            AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.enable_torch_compile_for_embedding,
        )

        self.layers = nn.ModuleList(
            [
                MiniMaxM3DecoderLayer(model_config, layer_idx, self.aux_stream_dict)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # The executor owns the authoritative CUDA-graph configuration. Mark
        # this model as eligible here and let the executor enable the automatic
        # FlashInfer MXFP8 path only when its decode graph runner is active.
        self._use_flashinfer_mxfp8_decode_graph_default = True
        # Final norm is a plain (non-Gemma) RMSNorm for the same reason as the
        # layer-boundary norms (see MiniMaxM3DecoderLayer.__init__): it doubles
        # as the last layer's next_layer_layernorm, so the last MoE/MLP output
        # AllReduce folds into it via RESIDUAL_RMS_NORM. The Gemma (1 + weight)
        # offset is folded into the stored weight at load time (see
        # _fold_gemma_boundary_norm_weights).
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=False,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )

        # When setup_aliases has chained the final norm into the last decoder
        # layer (next_layer_layernorm = self.norm), the last layer's boundary
        # step already applied it (fused or plain), so hidden_states is normed
        # and this is skipped. The fallback covers paths that never ran
        # setup_aliases (e.g. standalone unit tests): there the last layer
        # returns the unnormed (hidden_states, residual) pair and the final
        # add+RMSNorm is applied here.
        if self.layers[-1].next_layer_layernorm is None:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _load_index_qk_proj_weights(model: nn.Module, weights) -> None:
    """Fuse checkpoint index_q_proj and index_k_proj into index_qk_proj.

    The shared weight loader fuses only qkv_proj and gate_up_proj by name, so
    load the sibling checkpoint tensors into each fused index module through the
    FUSED_GATE_UP_LINEAR row-concatenation path and mark the sources consumed.
    """
    for name, module in model.named_modules():
        if name.split(".")[-1] != "index_qk_proj":
            continue
        parent = name.rsplit(".", 1)[0]
        q_weights = filter_weights(f"{parent}.index_q_proj", weights)
        k_weights = filter_weights(f"{parent}.index_k_proj", weights)
        # Missing sources make Linear.load_weights assert rather than leave
        # the fused module silently uninitialized.
        module.load_weights(weights=[q_weights, k_weights])
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed(f"{parent}.index_q_proj")
            weights.mark_consumed(f"{parent}.index_k_proj")
        else:
            for key in list(weights.keys()):
                if key.startswith(f"{parent}.index_q_proj.") or key.startswith(
                    f"{parent}.index_k_proj."
                ):
                    del weights[key]


def _load_qkv_index_proj_weights(model: nn.Module, weights) -> List[str]:
    """Pack five checkpoint shards and return generic-loader module skips."""
    checkpoint_names = ("q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj")
    shard_names = MiniMaxM3QKVIndexerLinear._SHARD_NAMES
    loaded_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, MiniMaxM3QKVIndexerLinear):
            continue
        parent = name.rsplit(".", 1)[0]
        shards = {
            shard_name: filter_weights(f"{parent}.{checkpoint_name}", weights)
            for shard_name, checkpoint_name in zip(shard_names, checkpoint_names, strict=True)
        }
        module.load_five_way_weights(shards)
        loaded_modules.append(name)
        for checkpoint_name in checkpoint_names:
            prefix = f"{parent}.{checkpoint_name}"
            if hasattr(weights, "mark_consumed"):
                weights.mark_consumed(prefix)
            else:
                for key in list(weights.keys()):
                    if key.startswith(f"{prefix}."):
                        del weights[key]
    return loaded_modules


# Layer-boundary RMSNorms whose Gemma (1 + weight) scaling is folded into the
# stored weight at load time so the runtime norm is a plain RMSNorm (see
# MiniMaxM3DecoderLayer.__init__ / MiniMaxM3Model.__init__). These are exactly
# the norms that drive the DeepSeek-V3-style fused AllReduce+residual+RMSNorm
# epilogue, whose kernel has no Gemma offset. The per-head q/k/index norms are
# intentionally excluded: they feed the separate fused_qk_norm_rope kernel,
# which handles Gemma directly and stays use_gemma=True.
_M3_BOUNDARY_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
)
_M3_FINAL_NORM_KEY = "model.norm.weight"


def _fold_gemma_boundary_norm_weights(weights):
    """Fold Gemma (1 + weight) into the layer-boundary RMSNorm weights.

    MiniMax-M3 stores every RMSNorm as a Gemma norm (use_gemma_norm=True), which
    computes (1 + weight) * x. The layer-boundary norms are constructed as plain
    norms (use_gemma=False, weight * x) so they can drive the fused
    AllReduce+RMSNorm kernels, so their stored weights must be pre-incremented by
    1.0. This is a numerically exact, load-time-only rewrite; the resulting norm
    is identical to the original Gemma norm on every path.

    Only the decoder input_layernorm / post_attention_layernorm and the final
    model.norm are touched. A no-op for keys that are absent (partial load) so it
    is safe to call unconditionally, but it must only run when the checkpoint
    actually uses Gemma norms (guarded by the caller).
    """
    for key in list(weights.keys()):
        if key.endswith(_M3_BOUNDARY_NORM_SUFFIXES) or key == _M3_FINAL_NORM_KEY:
            w = weights[key]
            w = w[:] if hasattr(w, "__getitem__") else w
            weights[key] = w + 1.0
    return weights


class MiniMaxM3ForCausalLM(SpecDecOneEngineForCausalLM[MiniMaxM3Model, PretrainedConfig]):
    """Text-only M3 model."""

    # Preserve M3's hand-fused eager decode and above-ceiling prefill paths,
    # including MXFP8 decode-graph backend tuning, instead of the FX fallback.
    use_fx_for_pcg_fallback = False

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """Prefer KV cache manager V2 for MiniMax-M3.

        Sparse attention already routes M3 to a V2-core manager
        unconditionally; declaring the preference keeps
        ``kv_cache_config.use_kv_cache_manager_v2`` consistent with the
        manager actually in use.
        """
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config: Any = None) -> Literal["PYTHON"]:
        """Prefer the Python transceiver in disaggregated serving.

        M3 runs a V2-core cache manager, which the C++ transceiver cannot
        drive; the KV-transfer unit test exercises the Python transceiver
        directly. This routes the fully-'auto' path to that combination.
        """
        return "PYTHON"

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        raw_pretrained = model_config.pretrained_config
        if is_minimax_m3_vl_config(raw_pretrained):
            model_config = get_text_model_config(model_config)
        if model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4:
            # M3's 57 sparse target layers have an MSA NVFP4 consumer, but the
            # one-model Eagle layer has no shipped NVFP4 trtllm-gen cubin.
            # Keep its modules and shared draft cache on their established FP8
            # representation while the target remains NVFP4.
            model_config.extra_attrs["draft_kv_cache_quant_algo_override"] = QuantAlgo.FP8
        super().__init__(MiniMaxM3Model(model_config), model_config)

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional[BaseWeightMapper] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ) -> None:
        # compressed-tensors NVFP4 experts (MiniMax-M3.1) load under the modelopt names.
        weights = convert_compressed_tensors_nvfp4_weights(weights)
        # Pack the opt-in five-way projection with MiniMax-specific TP
        # sharding before the generic mapper handles the compatibility path.
        packed_projection_modules = _load_qkv_index_proj_weights(self, weights)
        _load_index_qk_proj_weights(self, weights)
        # Fold Gemma (1 + weight) into the layer-boundary RMSNorm weights so the
        # runtime norms can be plain (non-Gemma) and drive the fused
        # AllReduce+residual+RMSNorm epilogue. Only when the checkpoint actually
        # stores Gemma norms; otherwise the boundary norms are already plain.
        if bool(getattr(self.config, "use_gemma_norm", False)):
            weights = _fold_gemma_boundary_norm_weights(weights)
        if weight_mapper is None:
            weight_mapper = MiniMaxM3HfWeightMapper()
        # The generic mapper understands three-way QKV fusion, not the
        # already-loaded five-way Q/K/V/index-Q/index-K modules.
        weight_mapper.add_skip_modules(packed_projection_modules)
        weight_mapper.init_model_and_config(self, self.model_config)
        merged_params_map = {**MINIMAX_M3_PARAMS_MAP, **(params_map or {})}
        super().load_weights(
            weights=weights,
            weight_mapper=weight_mapper,
            params_map=merged_params_map,
            allow_partial_loading=allow_partial_loading,
        )
        # CUDA graphs retain the aligned MSA buffers created at warmup. Refresh
        # their contents after refit; a new allocation would leave replay stale.
        for layer in self.model.layers:
            attention = layer.self_attn
            if getattr(attention.attn, "_msa_nvfp4_dequant_scales", None) is not None:
                _aligned_nvfp4_dequant_scales(
                    attention.attn, attention.qkv_proj.kv_scales, refresh=True
                )

    def setup_aliases(self) -> None:
        """Chain each decoder layer's next_layer_layernorm: the next layer's input_layernorm, or the final model norm
        for the last layer, applied at the layer's output boundary.

        Wired after weight load (the generic loader skips next_layer_layernorm aliases).
        """
        super().setup_aliases()
        layers = self.model.layers
        num_layers = len(layers)
        for idx, layer in enumerate(layers):
            if idx == num_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = layers[idx + 1].input_layernorm
            layer._normq_feeds_mxfp8 = idx != num_layers - 1
        # DEP2CTX_NORMQ consumers: every MXFP8 linear takes a published activation for its own input.
        for module in self.modules():
            if type(getattr(module, "quant_method", None)) is MXFP8LinearMethod:
                module.quant_method.__class__ = _NormqMXFP8LinearMethod
        # DEP2CTX_SWIGLU_MXFP8 (#5): swigluoai MLPs at the kernel's constants whose down_proj is a DEP2CTX_NORMQ
        # consumer.
        if _SWIGLU_MXFP8:
            for module in self.modules():
                if (
                    type(module) is GatedMLP
                    and type(getattr(module.down_proj, "quant_method", None))
                    is _NormqMXFP8LinearMethod
                    and module.activation is torch.nn.functional.silu
                    and (module.swiglu_alpha, module.swiglu_beta, module.swiglu_limit)
                    == (1.702, 1.0, 7.0)
                ):
                    module.__class__ = _SwigluMxfp8GatedMLP
        # DEP2CTX_SPARSE_OUT (#20629 NVFP4_OUT_IN_PLACE): the NVFP4 sparse prefill's MSA combine writes the
        # attention output in place (msa_prefill: msa_sparse_out_in_place, bitwise the copied result) when the loaded
        # MSA build accepts a caller-owned output. Kill switch: TRTLLM_M31_DISABLE_DEP2CTX_SPARSE_OUT=1.
        if not m31_disabled("DEP2CTX_SPARSE_OUT"):
            for layer in layers:
                layer.self_attn.attn.msa_sparse_out_in_place = True
            # DEP2CTX_COMBINE_MXFP8 (#7): the in-place combine also writes o_proj's MXFP8 input (msa_prefill:
            # msa_sparse_out_mxfp8_sink, bitwise mxfp8_quantize(out, True)) when the loaded MSA build takes o_fp8 /
            # sf_out (the old NVFP4 sparse func, or the E4M3X / BF16X fork's combine); o_proj then skips its quantize.
            # Default ON (exact: bitwise the stock quantize). Kill switch: TRTLLM_M31_DISABLE_DEP2CTX_COMBINE_MXFP8=1.
            if not m31_disabled("DEP2CTX_COMBINE_MXFP8"):
                for layer in layers:
                    if (
                        type(getattr(layer.self_attn.o_proj, "quant_method", None))
                        is _NormqMXFP8LinearMethod
                    ):
                        layer.self_attn.attn.msa_sparse_out_mxfp8_sink = _C1_SINK
        # DEP2CTX_SCHED_LPT (R1-SCHED): the NVFP4 sparse prefill's k2q CSR schedule runs its work items
        # heaviest first (msa_prefill: msa_k2q_schedule_order, MSA build_k2q_csr schedule_order="lpt") when the
        # loaded MSA build accepts it. Exact: the order only changes which CTA runs which item (no cross-CTA
        # reduction; every partial lands at its CSR position); re-prove if the forward gains cross-CTA atomics.
        # Kill switch: TRTLLM_M31_DISABLE_DEP2CTX_SCHED_LPT=1.
        if not m31_disabled("DEP2CTX_SCHED_LPT"):
            for layer in layers:
                layer.self_attn.attn.msa_k2q_schedule_order = "lpt"
                # DEP2CTX_K2Q_LPT_REUSE: the LPT build on msa_prefill's reused per-step CSR workspace.
                layer.self_attn.attn.msa_k2q_lpt_reuse = _K2Q_LPT_REUSE


def _strip_language_model_prefix(
    weights: TMapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Strip the ``language_model.`` prefix and drop deferred multimodal/MTP keys.

    Returns ``(text_weights, ignored)`` where ``ignored`` maps each
    deferred-or-multimodal source key to the documented reason it was
    skipped. ``text_weights`` is suitable to pass to the underlying
    ``DecoderModelForCausalLM`` weight loader.
    """
    text_weights: Dict[str, Any] = {}
    ignored: Dict[str, str] = {}
    for key, value in weights.items():
        if key.startswith(_VL_PREFIXES):
            for prefix in _VL_PREFIXES:
                if key.startswith(prefix):
                    ignored[key] = f"deferred multimodal branch: {prefix.rstrip('.')}"
                    break
            continue
        if not key.startswith("language_model."):
            # Unknown top-level keys are kept for the inner loader so it can
            # surface a clear error rather than us silently dropping them.
            text_weights[key] = value
            continue
        rest = key[len("language_model.") :]
        if rest.startswith("model.mtp."):
            ignored[key] = "MTP (multi-token prediction) not implemented"
            continue
        text_weights[rest] = value
    return text_weights, ignored


class ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx(MiniMaxM3ForCausalLM):
    """MiniMax-M3.1 context (prefill) worker, sm_107, attention DP 2 + EP2 experts, text only.

    The disaggregated CTX role (tp 2, moe_ep 2, attention DP on): each rank runs the attention, the dense MLPs and the
    shared expert at full width on its own requests, and the routed experts exchange tokens over EP2 inside the MoE.
    No layer boundary reduces across ranks. The model is the tp2ctx target's (see the header), with its context items
    behind ``TRTLLM_M31_DISABLE_<ID>`` (``m31_switches``).

    The checkpoint declares the vision-language wrapper: weights load as the built-in VL class loads its text
    stream (vision branches split off and dropped, ``language_model.`` stripped), and a request carrying image or
    video data is refused.
    """

    # The DSpark draft window ships with the context KV in disaggregated serving; the step loop is otherwise the
    # built-in model's. DEP2CTX_PROXY_Q128: the prefill proxy (indexer max-score) plan uses 128-row query tiles with 4
    # KV splits on steps whose longest query span exceeds 128 rows (MSA maxscore_qo_tile: the same score table bits;
    # the OnlyScore proxy modes are native on SM107). Kill switch TRTLLM_M31_DISABLE_DEP2CTX_PROXY_Q128=1. The
    # late-compile detector arms after warmup, as on the generation targets: the server marks ready and logs any
    # compile after it.
    # Built for this engine: a requested hook it does not have is classified and dropped (runtime_hooks_compat).
    runtime_hooks = runtime_hooks_compat.build(
        dict(
            sparse_attention=_msa_engine.sparse_attention_classes(),
            disagg_draft_kv=True,
            # The windowed draft pool's sizing and resize fixes.
            draft_pool_fixes=True,
            late_compile_detect=True,
            # DEP2CTX_BCG_LARGE: the breakable prefill buckets above 4096; the kill switch caps the list at 4096.
            prefill_capture_extra_num_tokens=_ppr.DEP2CTX_BCG_GRID if _BCG_LARGE else (),
            prefill_capture_extra_min_top=_BCG_BASE_MAX,
            prefill_capture_max_num_tokens=None if _BCG_LARGE else _BCG_BASE_MAX,
            # DEP2CTX_A1_FUSED_IN_BCG: the pointer guard's notes end with the graphs that baked them.
            prefill_graphs_released=_bcg_graphs_released if _A1_FUSED_IN_BCG else None,
        ),
        msa=dict(
            prefill_proxy_qo_tile=None if m31_disabled("DEP2CTX_PROXY_Q128") else 128,
            prefill_proxy_kv_splits=None if m31_disabled("DEP2CTX_PROXY_Q128") else 4,
        ),
        target=__name__,
    )

    def __init__(self, model_config: "ModelConfig[PretrainedConfig]"):
        assert torch.cuda.get_device_capability() == _SM, (
            f"this target is built for sm_{_SM[0]}{_SM[1]}, the device is sm_"
            f"{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        mapping = model_config.mapping
        if mapping.tp_size > 1 and not mapping.enable_attention_dp:
            raise ValueError(
                f"the MiniMax-M3.1 dep2ctx target runs attention DP (no TP all-reduce); got tp {mapping.tp_size} "
                "with attention DP off"
            )
        # The CuTe DSL kernels' exported objects (TRTLLM_M31_CUTE_AOT_DIR) serve this process from here on.
        cute_aot.arm()
        if _A1_FUSED_IN_BCG_REQUESTED and not _A1_FUSED_IN_BCG:
            logger.info_once(
                f"{_BCG_PTRS_WHAT} not taken: this engine does not release prefill graphs to the model; stock "
                "projection under capture",
                key="dep2ctx_a1_fused_in_bcg",
            )
        super().__init__(model_config)
        # DEP2CTX_TSEL (#20629 TILE_SELECT): the prefill top-16 select as the tile select, on every layer's MSA indexer.
        _tsel.register(self)
        # DEP2CTX_IDX_SCORER (CTX, exact): the prefill indexer scores from the exact K=32 Q8KV8 scorer, not the proxy.
        _idx_scorer.register(self)
        # DEP2CTX_BCG_PAD_MASK: the step's pad-row mask, sized for the largest prefill step.
        self._pad_rows = _ppr.PrefillPadRows(
            int(getattr(model_config, "max_num_tokens", 0) or 0),
            enabled=_BCG_PAD_MASK,
            tag="[M3.1 dep2ctx] DEP2CTX_BCG_PAD_MASK",
            in_graph=is_in_breakable_cuda_graph,
        )
        self._bcg_reserved_logged: set = set()
        self._wire_pointer_guard()
        self._dspark_prefill: Dict[str, bool] = {}

    def _wire_pointer_guard(self) -> None:
        """DEP2CTX_A1_FUSED_IN_BCG (S4): this model's own pointer guard, shared by its attention layers (their captures
        note into it), checked by its outer forward and reset by its graph release, so two live instances never
        compare or reset each other's notes."""
        self._bcg_ptrs = _bcg_ptr_guard.StaticPointerGuard(_BCG_PTRS_WHAT)
        for layer in self.model.layers:
            attention = getattr(layer, "self_attn", None)
            if isinstance(attention, MiniMaxM3Attention):
                attention._bcg_ptrs = self._bcg_ptrs

    def cache_derived_state(self) -> None:
        """Connect the MoE blocks to the pad-row mask and allocate it (idempotent, the loader may call it twice),
        then install DEP2CTX_DSPARK_STORE.

        DEP2CTX_DSPARK_STORE: the DSpark drafter's prefill context
        K / V store per chunk through the decode store's C++ op (ops/dspark/dspark_prefill_store; bitwise the
        stock precompute_context_kv + paged append; the stock lines for any chunk it declines), with the lean drafter
        glue and the k_norm weight fold. This role runs no generation branch, so the process-wide fold is this store's
        alone (the install leaves it alone if a decode store is ever installed on this worker too). Kill switches:
        TRTLLM_M31_DISABLE_DEP2CTX_DSPARK_STORE=1 (nothing installed), TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE=1,
        TRTLLM_M31_DISABLE_DEP2CTX_DSPARK_STORE_FOLD=1, TRTLLM_M31_DISABLE_DSPARK_CTX_STORE_CPP=1 (the op). Only the
        first restores the stock launches as well as the stock bytes."""
        super().cache_derived_state()
        wired, left_out = _ppr.wire(self.model.layers, self._pad_rows)
        param = next(self.parameters(), None)
        if param is not None and param.device.type == "cuda":
            self._pad_rows.allocate(param.device)
        if not _BCG_PAD_MASK:
            state = "off (TRTLLM_M31_DISABLE_DEP2CTX_BCG_PAD_MASK=1 or _K3=1)"
        else:
            state = f"on {wired} MoE layers" + (f" ({left_out})" if left_out else "")
        logger.info_once(f"[M3.1 dep2ctx] DEP2CTX_BCG_PAD_MASK {state}", key="dep2ctx_bcg_pad_mask")
        logger.info_once(
            "[M3.1 dep2ctx] DEP2CTX_BCG_LARGE "
            + (
                f"buckets {list(_ppr.DEP2CTX_BCG_GRID)} (breakable backend, up to max_num_tokens)"
                if _BCG_LARGE
                else f"off: prefill capture capped at {_BCG_BASE_MAX} (TRTLLM_M31_DISABLE_DEP2CTX_BCG_LARGE=1 or _K3=1)"
            )
            + (
                f"; memory_reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB"
                if param is not None and param.device.type == "cuda"
                else ""
            ),
            key="dep2ctx_bcg_large",
        )
        self._dspark_prefill = _dps.install(
            getattr(self, "spec_worker", None),
            getattr(self, "draft_model", None),
            disabled=m31_disabled,
        )
        logger.info(f"[M3.1 dep2ctx] DEP2CTX_DSPARK_STORE {_dps.engagement(self._dspark_prefill)}")

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional[BaseWeightMapper] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ) -> None:
        text_or_other_weights, _ = split_multimodal_weights(weights)
        text_weights, ignored = _strip_language_model_prefix(text_or_other_weights)
        self.last_ignored_weight_keys: Dict[str, str] = ignored
        super().load_weights(
            weights=text_weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )

    def forward(
        self,
        attn_metadata: "AttentionMetadata",
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        for param in kwargs.pop("multimodal_params", None) or []:
            data = getattr(param, "multimodal_data", None) or {}
            if isinstance(data, dict) and (data.get("image") or data.get("video")):
                raise ValueError(
                    "the MiniMax-M3.1 dep2ctx target is text only: image / video input is not served"
                )
        # DEP2CTX_BCG_PAD_MASK: this forward's pad rows. The outer forward runs eagerly on every step, graph replays
        # included (BCG swaps only the decoder body), so the mask a replay reads is written here, before the body.
        tokens = input_ids if input_ids is not None else inputs_embeds
        graphed = is_in_breakable_cuda_graph()
        if graphed:
            # DEP2CTX_A1_FUSED_IN_BCG (S4): this forward replays the body, whose first captured segment already writes
            # the KV cache through the addresses noted at capture. Re-read every noted layer's first: a moved buffer
            # raises before any segment replays.
            self._bcg_ptrs.check(_a1_bcg_ptr_reader(attn_metadata))
        self._pad_rows.begin(
            int(attn_metadata.num_tokens),
            int(tokens.shape[0]),
            padded=attn_metadata.padded_num_tokens is not None,
            graphed=graphed,
            device=tokens.device,
        )
        try:
            out = super().forward(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_context_logits=return_context_logits,
                **kwargs,
            )
        finally:
            self._pad_rows.end()
        if graphed and _BCG_LARGE and tokens.is_cuda and len(self._bcg_reserved_logged) < 64:
            # DEP2CTX_BCG_LARGE receipt: the allocator's reserved bytes after the first replay of each bucket.
            rows = int(tokens.shape[0])
            if rows not in self._bcg_reserved_logged:
                self._bcg_reserved_logged.add(rows)
                logger.info(
                    f"[M3.1 dep2ctx] DEP2CTX_BCG_LARGE first replay of bucket {rows}: memory_reserved "
                    f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB"
                )
        return out
