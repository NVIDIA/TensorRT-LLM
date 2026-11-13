# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MLA weights, prefill, and general-batch attention on Rubin."""

from __future__ import annotations

import functools
import math
import os
import weakref
from typing import Dict, Optional, cast

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    TrtllmAttention,
    TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    MLAParams,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.locality_domain.policy import (
    LocalityDomainExecutionPlanner,
    LocalityDomainPolicy,
)
from tensorrt_llm._torch.locality_domain.runtime import LocalityDomainRuntime
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    is_static_nvfp4_input_eligible,
)
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import (
    AuxStreamType,
    Fp4QuantizedTensor,
    maybe_compiled_cat,
    maybe_compiled_copy_,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ._dsa import _get_short_seq_mha_threshold, forward_sparse_attn, forward_sparse_attn_custom_op
from ._dsa_backend import Glm52DSAAttention
from ._dsa_custom_ops import create_mla_outputs


def _create_attention(
    backend_name: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    pos_embd_params: Optional[PositionalEmbeddingParams] = None,
    quant_config: Optional[QuantConfig] = None,
    q_scaling: Optional[float] = None,
    is_mla_enable: bool = False,
    q_lora_rank: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    qk_nope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    rope_append: Optional[bool] = None,
    hidden_size: Optional[int] = None,
    predicted_tokens_per_seq: Optional[int] = 1,
    skip_create_weights_in_init: bool = False,
    attention_chunk_size: Optional[int] = None,
    sparse_params: Optional[DSAParams] = None,
    dtype: Optional[torch.dtype] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    kv_cache_dtype: str = "auto",
    skip_correction_threshold: float = 0.0,
    cache_context_plan: bool = False,
) -> AttentionBackend:
    if attention_chunk_size is not None and backend_name.upper() != "TRTLLM":
        raise ValueError(f"Backend {backend_name} does not support chunked attention.")
    if backend_name.upper() != "TRTLLM":
        raise ValueError("The GLM Rubin target requires the TRTLLM attention backend")
    attn_cls = Glm52DSAAttention if sparse_params is not None else TrtllmAttention

    if is_mla_enable:
        assert attn_cls.support_mla(), f"MLA is not supported for {backend_name} backend"
        assert (
            q_lora_rank > 0
            and kv_lora_rank > 0
            and qk_rope_head_dim >= 0
            and qk_nope_head_dim > 0
            and v_head_dim > 0
        )
        mla_params = MLAParams(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            rope_append=True if rope_append is None else rope_append,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            hidden_size=hidden_size,
        )
    else:
        mla_params = None

    kwargs = dict(
        quant_config=quant_config,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        skip_create_weights_in_init=skip_create_weights_in_init,
        attention_chunk_size=attention_chunk_size,
        dtype=dtype,
        aux_stream=aux_stream,
        sparse_params=sparse_params,
        kv_cache_dtype=kv_cache_dtype,
        skip_correction_threshold=skip_correction_threshold,
        cache_context_plan=cache_context_plan,
    )

    return attn_cls(
        layer_idx,
        num_heads,
        head_dim,
        num_kv_heads,
        **kwargs,
    )


def fp8_block_scaling_bmm_out(
    mat1, mat2_fp8, mat2_scale, out, mat2_dequant=None, use_cute_dsl_blockscaling_bmm=False
):
    """Run the Rubin absorption BMM with its existing block-scale layout."""
    if use_cute_dsl_blockscaling_bmm:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(mat1)
        torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, out)
    else:
        torch.bmm(mat1.transpose(0, 1), mat2_dequant.transpose(1, 2), out=out)


class Glm52MLA(nn.Module):
    """Target-owned MLA module for pure TP and an FP8 latent cache."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        bias: bool,
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
        rms_norm_eps: Optional[float] = None,
        enable_locality_domain_bf16_linear: bool = False,
    ) -> None:
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary
                CUDA streams by role.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
            rms_norm_eps (Optional[float]): Override the RMSNorm epsilon from
                the pretrained config. If neither source provides a value
                (e.g. config.pretrained_config is None), falls back to 1e-6.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype
        self._weights_transformed = False
        # Context-phase fused FP8-Q gate and its Q placeholder; both are
        # resolved once in create_weights(), never on the forward path.
        self._fused_q_fp8_ctx_ok = False
        self.register_buffer("_fused_q_placeholder", None, persistent=False)

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        self.register_to_config = False
        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            suffix = 0
            # ``layer_idx`` is local to an attention stack, while this registry is shared
            # by target and draft modules in one-model speculative decoding. Keep the first
            # registration under ``"<layer_idx>"`` and suffix later collisions as
            # ``"<layer_idx>_<n>"`` so custom ops resolve the originating module without
            # overwriting another stack's weak reference.
            while self.layer_idx_str in config.extra_attrs["mla_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

        config = config or ModelConfig()
        self.kv_cache_dtype = config.extra_attrs.get("kv_cache_dtype", "auto")
        sparse_attn_cfg = config.sparse_attention_config
        sparse_params = (
            sparse_attn_cfg.to_sparse_params(
                pretrained_config=config.pretrained_config,
                layer_idx=self.layer_idx,
            )
            if sparse_attn_cfg is not None
            else None
        )
        self.sparse_params = sparse_params
        self.short_seq_mha_threshold = _get_short_seq_mha_threshold()
        if sparse_params is None or sparse_params.algorithm != "dsa":
            raise ValueError("The GLM Rubin target requires DSA sparse attention")

        # Fold the residual-less q_a_layernorm -> q_b_proj NVFP4 input
        # quantization into one fused RMSNorm + FP4-quantize kernel. Resolve
        # lazily because q_b_proj.input_scale is finalized after weight loading.
        self._qa_fused_scale = None

        # tensor parallel
        if mapping_with_cp is not None:
            logger.warning_once(
                "[MLA::__init__] Overriding mapping with CP detected.",
                key="mla_init_mapping_with_cp",
            )
            self.mapping = mapping_with_cp
        else:
            self.mapping = config.mapping
        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        cp_size = self.mapping.cp_size
        dp_size = 1

        assert tp_size in (4, 8, 16) and pp_size == cp_size == 1
        assert self.kv_cache_dtype != "fp8_ds_mla"
        mapping = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )

        assert self.num_heads % (tp_size * cp_size) == 0
        self.num_heads_tp = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // cp_size
        self.num_context_heads = self.num_heads_tp
        self.num_key_value_heads_tp = (self.num_key_value_heads + tp_size - 1) // tp_size

        if rms_norm_eps is None:
            rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            # No explicit value and pretrained_config is None or lacks
            # rms_norm_eps (e.g. unit tests constructing MLA directly):
            # keep the historical default.
            rms_norm_eps = 1e-6
        quant_config = config.get_quant_config()
        if not quant_config.layer_quant_mode.has_fp8_kv_cache():
            raise ValueError("The GLM Rubin target requires an FP8 latent KV cache")
        self.quant_config = quant_config
        # Construction settings used by sparse module hooks.
        self.rms_norm_eps = rms_norm_eps
        self.reduce_output = reduce_output
        self.skip_create_weights_in_init = config.skip_create_weights_in_init
        self.allreduce_strategy = config.allreduce_strategy
        self.force_dynamic_quantization = config.force_dynamic_quantization
        self.num_hidden_layers = getattr(config.pretrained_config, "num_hidden_layers", None)
        self.pp_mapping = config.mapping

        self.use_cute_dsl_blockscaling_mm = config.use_cute_dsl_blockscaling_mm
        self.use_cute_dsl_blockscaling_bmm = config.use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = config.use_cute_dsl_bf16_bmm
        self._kvbproj_head_weights_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
            None
        )
        self.use_cute_dsl_bf16_gemm = config.use_cute_dsl_bf16_gemm
        locality_domain_policy = config.extra_attrs.get("locality_domain_policy")
        self._locality_domain_policy = (
            locality_domain_policy if locality_domain_policy is not None else LocalityDomainPolicy()
        )
        self._locality_domain_runtime: Optional[LocalityDomainRuntime] = None
        self._locality_domain_k_b_proj_trans_shards: Optional[tuple[torch.Tensor, torch.Tensor]] = (
            None
        )
        self._locality_domain_v_b_proj_shards: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        kv_a_out_features = self.kv_lora_rank + self.qk_rope_head_dim
        kv_a_out_features += self.q_lora_rank
        self.kv_a_proj_with_mqa = Linear(
            hidden_size,
            kv_a_out_features,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            use_custom_cublas_mm=True,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
        )

        self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank, eps=rms_norm_eps, dtype=dtype)

        self.q_b_proj = Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            enable_locality_domain_bf16_linear=enable_locality_domain_bf16_linear,
            locality_domain_policy=self._locality_domain_policy,
        )

        mapping_o = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = mapping_o

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.aux_stream_dict = aux_stream_dict or {}
        self.aux_stream = self.aux_stream_dict.get(AuxStreamType.Attention)
        mqa_aux_stream = self.aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.kv_a_layernorm = RMSNorm(hidden_size=self.kv_lora_rank, dtype=dtype, eps=rms_norm_eps)
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            enable_locality_domain_bf16_linear=enable_locality_domain_bf16_linear,
            locality_domain_policy=self._locality_domain_policy,
        )
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.mqa = _create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads_tp,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=self.pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            hidden_size=self.hidden_size,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            v_head_dim=self.kv_lora_rank,
            sparse_params=self.sparse_params,
            dtype=dtype,
            aux_stream=mqa_aux_stream,
            rope_append=True,
            kv_cache_dtype=self.kv_cache_dtype,
            skip_correction_threshold=config.skip_correction_threshold,
        )
        if self.mqa is None:
            raise RuntimeError("MLA requires a non-null MQA attention backend")

        self.indexer = self.mqa.indexer

        create_dense_mha = self.short_seq_mha_threshold > 0
        if create_dense_mha:
            mha_sparse_params = None
            self.mha = _create_attention(
                config.attn_backend,
                self.layer_idx,
                self.num_context_heads,
                head_dim=self.qk_head_dim,
                num_kv_heads=(self.num_key_value_heads_tp),
                pos_embd_params=self.pos_embd_params,
                quant_config=quant_config,
                q_scaling=q_scaling,
                is_mla_enable=True,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                predicted_tokens_per_seq=self.predicted_tokens_per_seq,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                sparse_params=mha_sparse_params,
                skip_correction_threshold=config.skip_correction_threshold,
            )
        else:
            self.mha = None

        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping_o,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            reduce_output=reduce_output,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            enable_locality_domain_bf16_linear=enable_locality_domain_bf16_linear,
            locality_domain_policy=self._locality_domain_policy,
        )
        self.attention_output_hidden_size = self.o_proj.in_features

        self.softmax_scale = 1.0 / (math.sqrt(self.qk_head_dim) * q_scaling)

        self.rope_fusion = self.mqa.support_fused_rope()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to
        # quant_config, which could be modified after __init__.
        if self.mha is not None:
            self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)
        self._fused_q_fp8_ctx_ok = False

        # Although we use FP8 MLA for context/generation phase, the output is still in BF16
        self.out_scale = None

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = bool(
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales()
        )
        mla_weight_dtype = torch.float8_e4m3fn if has_fp8_block_scales else self.dtype
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if has_fp8_block_scales:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp_cp,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if is_sm_100f() and not self.use_cute_dsl_blockscaling_bmm:
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None
        self._locality_domain_k_b_proj_trans_shards = None
        self._locality_domain_v_b_proj_shards = None
        self._weights_transformed = False
        self._init_fused_q_fp8_context()

    def create_output(self, hidden_states: torch.Tensor, num_contexts: int):
        if isinstance(hidden_states, Fp4QuantizedTensor):
            assert hidden_states.unquantized_hidden_states is not None, (
                "MLA.create_output received an Fp4QuantizedTensor without a "
                "unquantized_hidden_states view; the producing fusion must use "
                "return_norm_out=True"
            )
            hidden_states = hidden_states.unquantized_hidden_states
        num_tokens = hidden_states.shape[0]
        return hidden_states.new_empty(
            [num_tokens, self.attention_output_hidden_size], dtype=hidden_states.dtype
        )

    def _resolve_qa_fused_scale(self):
        """Resolve whether q_a_layernorm can fuse q_b_proj NVFP4 input quantization."""
        if self._qa_fused_scale is not None:
            return self._qa_fused_scale if self._qa_fused_scale is not False else None
        eligible = False
        if (
            getattr(self, "q_a_layernorm", None) is not None
            and getattr(self, "q_b_proj", None) is not None
        ):
            qb = self.q_b_proj
            qa = self.q_a_layernorm
            if is_static_nvfp4_input_eligible(qb) and not qa.use_gemma and not qa.is_nvfp4:
                eligible = True
        if eligible:
            self.q_a_layernorm.is_nvfp4 = True
            self.q_a_layernorm.nvfp4_scale = qb.input_scale
        self._qa_fused_scale = qb.input_scale if eligible else False
        return self._qa_fused_scale if eligible else None

    def _q_a_layernorm_maybe_fused(self, q: torch.Tensor, return_norm_out: bool = False):
        """Apply q_a_layernorm and fuse static NVFP4 input quantization when eligible."""
        scale = self._resolve_qa_fused_scale()
        if scale is None:
            normed = self.q_a_layernorm(q)
            return (normed, normed) if return_norm_out else normed
        return self.q_a_layernorm(q, return_norm_out=return_norm_out)

    def _use_kvbproj_strided_out(self) -> bool:
        """Whether the context KV expansion can skip the k_nope materialization
        copy by running per-head batched GEMMs that write directly into the
        strided K buffer (and the v half of a kv-layout buffer).

        Requires SM100f and an unquantized bf16 kv_b_proj without bias;
        other configs fall back to the wide-GEMM + copy path.
        """
        w = getattr(self.kv_b_proj, "weight", None)
        expected_shape = (
            self.num_heads_tp * (self.qk_nope_head_dim + self.v_head_dim),
            self.kv_lora_rank,
        )
        expected_numel = math.prod(expected_shape)
        return (
            w is not None
            and w.dtype == torch.bfloat16
            and tuple(w.shape) == expected_shape
            and w.numel() == expected_numel
            and self.kv_b_proj.bias is None
        )

    def _get_kvbproj_head_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head views into kv_b_proj.weight, layout [H, N, K] (contiguous).

        kv_b_proj.weight rows are [H*qk_nope (k_nope block) | H*v_head (v
        block)] with K = kv_lora_rank, so both blocks reshape to head-major
        [H, N, K] without any copy.
        """
        cached = getattr(self, "_kvbproj_head_weights_cache", None)
        w = self.kv_b_proj.weight
        if cached is not None and cached[0] is w:
            w_k, w_v = cached[1], cached[2]
        else:
            nope_rows = self.num_heads_tp * self.qk_nope_head_dim
            w_k = w[:nope_rows].view(self.num_heads_tp, self.qk_nope_head_dim, self.kv_lora_rank)
            w_v = w[nope_rows:].view(self.num_heads_tp, self.v_head_dim, self.kv_lora_rank)
            self._kvbproj_head_weights_cache = (w, w_k, w_v)
        return w_k, w_v

    def forward_context_default(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dense MHA context path: expand KV via kv_b_proj and run attention."""
        k = torch.empty_like(q).view(-1, self.num_context_heads, self.qk_head_dim)
        if self._use_kvbproj_strided_out():
            # Expand KV via per-head batched GEMMs that write k_nope directly
            # into the strided K buffer, skipping the [t, H*256] kv
            # intermediate and the k_nope materialization copy.
            num_tokens = compressed_kv.shape[0]
            w_k, w_v = self._get_kvbproj_head_weights()
            ckv = compressed_kv.unsqueeze(0).expand(
                self.num_context_heads, num_tokens, self.kv_lora_rank
            )
            # [H, t, qk_nope] view with strides (qk_head_dim, H*qk_head_dim, 1)
            k_nope_out = k[..., : self.qk_nope_head_dim].transpose(0, 1)
            # The MLA fp8 quantize kernel (quantizeCopyInputToFp8Kernel,
            # mlaKernels.cu) hardcodes the V source token stride as
            # H*(qk_nope+v) -- the layout of the kv.split() view. Keep that
            # contract: allocate a kv-shaped buffer and only write its v
            # half; the k_nope half is never written or read.
            nope_cols = self.num_context_heads * self.qk_nope_head_dim
            v_cols = self.num_context_heads * self.v_head_dim
            kv_buf = torch.empty(num_tokens, nope_cols + v_cols, dtype=q.dtype, device=q.device)
            v = kv_buf[:, nope_cols:]
            # [H, t, v_head] view with strides (v_head, H*(qk_nope+v), 1)
            v_out = v.unflatten(-1, (self.num_context_heads, self.v_head_dim)).transpose(0, 1)
            # cuBLAS consumes the transposed weights and writes strided
            # outputs directly, without materializing contiguous copies.
            torch.ops.trtllm.bmm_out(ckv, w_k.transpose(1, 2), k_nope_out)
            torch.ops.trtllm.bmm_out(ckv, w_v.transpose(1, 2), v_out)
        else:
            kv = self._project_context_kv(compressed_kv)
            k_nope, v = kv.split(
                [
                    self.num_context_heads * self.qk_nope_head_dim,
                    self.num_context_heads * self.v_head_dim,
                ],
                -1,
            )

            maybe_compiled_copy_(
                k[..., : self.qk_nope_head_dim],
                k_nope.view(-1, self.num_context_heads, self.qk_nope_head_dim),
            )
        # When rope_fusion=True (apply_rotary_emb=False), the rope portion
        # of k is left uninitialized here; the fused attention kernel
        # handles k_pe RoPE via latent_cache instead.
        k = k.view(-1, self.num_context_heads * self.qk_head_dim)

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def forward_context_with_cached_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert latent_cache is not None
        trtllm_attention = cast(TrtllmAttention, self.mha)

        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype
        )
        assert (
            full_compressed_kv.shape[0]
            == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        )
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert (
            full_k_pe.shape[0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        )
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self._project_context_kv(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_context_heads * self.qk_nope_head_dim,
                self.num_context_heads * self.v_head_dim,
            ],
            -1,
        )

        full_k_nope = full_k_nope.view(-1, self.num_context_heads, self.qk_nope_head_dim)
        full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
        full_k = maybe_compiled_cat(
            (full_k_nope, full_k_pe.expand(-1, self.num_context_heads, -1)), dim=-1
        )
        full_k = full_k.view(-1, self.num_context_heads * self.qk_head_dim)

        # release pytorch activation memory
        full_compressed_kv = None
        full_k_pe = None
        full_kv = None
        full_k_nope = None

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        attn_output = self.mha.forward(
            q,
            full_k,
            full_v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [total_token_q, num_heads, 2] -> [total_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_context_heads, 2),
            dtype=torch.float,
            device="cuda",
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_context_heads, 2),
            dtype=torch.float,
            device="cuda",
        )

        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_context_heads * self.v_head_dim), dtype=q.dtype
        )

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime
        # host_total_kv_lens is updated in place below, so copy the value out;
        # a 0-d view would alias it and make the restore a no-op.
        origin_ctx_total_kv_len = attn_metadata.host_total_kv_lens[0].item()

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts
            ]
            chunked_global_offset = attn_metadata.chunked_global_offset[loop_idx]
            chunked_max_seq_len = attn_metadata.max_chunk_len_per_loop[loop_idx]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                chunked_global_offset=chunked_global_offset,
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype,
            )

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self._project_context_kv(chunked_compressed_kv)
            chunked_k_nope, chunked_v = chunked_kv.split(
                [
                    self.num_context_heads * self.qk_nope_head_dim,
                    self.num_context_heads * self.v_head_dim,
                ],
                -1,
            )

            chunked_k_nope = chunked_k_nope.view(-1, self.num_context_heads, self.qk_nope_head_dim)
            chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
            chunked_k = maybe_compiled_cat(
                (chunked_k_nope, chunked_k_pe.expand(-1, self.num_context_heads, -1)), dim=-1
            )
            chunked_k = chunked_k.view(-1, self.num_context_heads * self.qk_head_dim)

            # release pytorch activation memory
            chunked_compressed_kv = None
            chunked_k_pe = None
            chunked_kv = None
            chunked_k_nope = None

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[loop_idx]
            attn_metadata.host_total_kv_lens[0] = total_ctx_chunked_tokens

            # do not apply mask for attention within loop
            # latent_cache must be None to differentiate from normal context phase,
            # so that we can skip applying RoPE and appending KV cache inside attention op
            temp_attn_output = self.mha.forward(
                q,
                chunked_k,
                chunked_v,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                    out_scale=self.out_scale,
                    attention_mask=PredefinedAttentionMask.FULL,
                    softmax_stats_tensor=self.temp_softmax_stats_tensor,
                    chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.chunked_prefill_buffer_batch_size,
                    output=temp_attn_output,
                ),
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output,
                temp_attn_output,
                self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor,
                temp_merge_op,
                attn_metadata,
            )

        # deal with the uncached kv
        kv = self._project_context_kv(compressed_kv)
        _, k_pe = latent_cache.view([-1, self.kv_lora_rank + self.qk_rope_head_dim]).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        # final round of attention

        k_nope, v = kv.split(
            [
                self.num_context_heads * self.qk_nope_head_dim,
                self.num_context_heads * self.v_head_dim,
            ],
            -1,
        )

        k_nope = k_nope.view(-1, self.num_context_heads, self.qk_nope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = maybe_compiled_cat((k_nope, k_pe.expand(-1, self.num_context_heads, -1)), dim=-1)
        k = k.view(-1, self.num_context_heads * self.qk_head_dim)

        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = (
            attn_metadata.prompt_lens_cpu_runtime[: attn_metadata.num_contexts].sum().item()
        )

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        temp_attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.chunked_prefill_buffer_batch_size,
                output=temp_attn_output,
            ),
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(
            attn_output,
            temp_attn_output,
            self.softmax_stats_tensor,
            self.temp_softmax_stats_tensor,
            temp_merge_op,
            attn_metadata,
        )
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = origin_ctx_total_kv_len

        return attn_output

    @staticmethod
    @functools.cache
    def cached_warmup_forward_context_with_cached_kv(
        num_heads_tp,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    ):
        """Warmup torch.compile for cat operations with different tensor layouts.

        Tensors are marked with torch._dynamo.maybe_mark_dynamic(..., 0) on the
        num_tokens dimension, so for num_tokens != 1 a single warmup run is
        enough and the compiled kernel generalizes across varying num_tokens at
        runtime. num_tokens=1 still triggers recompile (torch.compile specializes
        for it), so it is warmed up separately. Do not use torch.compile with
        dynamic=True here because it completely ignores tensor layout/stride
        information, resulting in significantly degraded performance.
        """

        def warmup(num_tokens):
            chunked_k_nope = k_nope = torch.empty(
                num_tokens,
                num_heads_tp * (qk_nope_head_dim + v_head_dim),
                dtype=dtype,
                device=device,
            )[:, : num_heads_tp * qk_nope_head_dim].view(num_tokens, num_heads_tp, qk_nope_head_dim)
            chunked_k_pe = torch.empty(
                num_tokens, 1, qk_rope_head_dim, dtype=dtype, device=device
            ).expand(-1, num_heads_tp, -1)
            k_pe = torch.empty(
                num_tokens,
                1,
                kv_lora_rank + qk_rope_head_dim,
                dtype=dtype,
                device=device,
            )[:, :, -qk_rope_head_dim:].expand(-1, num_heads_tp, -1)
            torch._dynamo.maybe_mark_dynamic(chunked_k_nope, 0)
            torch._dynamo.maybe_mark_dynamic(chunked_k_pe, 0)
            torch._dynamo.maybe_mark_dynamic(k_pe, 0)
            maybe_compiled_cat((chunked_k_nope, chunked_k_pe), dim=-1)
            maybe_compiled_cat((k_nope, k_pe), dim=-1)

        # With dim 0 (num_tokens) marked dynamic, one warmup suffices for all
        # num_tokens != 1 at runtime.
        warmup(2)

        # num_tokens=1 still triggers recompile; warm it separately.
        warmup(1)

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            # Warm up maybe_compiled_cat for both the chunked-prefill path and
            # the cached-kv path; without this, the cached-kv prefill
            # (block-reuse without chunked_prefill) recompiles per shape and
            # may stall inside inductor's compile worker.
            if trtllm_attention.has_cached_kv_for_mla_context_warmup(attn_metadata):
                self.cached_warmup_forward_context_with_cached_kv(
                    self.num_context_heads,
                    self.qk_nope_head_dim,
                    self.qk_rope_head_dim,
                    self.kv_lora_rank,
                    self.v_head_dim,
                    q.dtype,
                    q.device,
                )
            if (
                trtllm_attention.is_chunked_prefill_for_mla_context(attn_metadata)
                and get_sm_version() >= 100
            ):
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output
                )
            elif trtllm_attention.has_cached_kv_for_mla_context(
                attn_metadata
            ) or trtllm_attention.is_chunked_prefill_for_mla_context(attn_metadata):
                return self.forward_context_with_cached_kv(q, latent_cache, attn_metadata, output)
        return self.forward_context_default(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )

    def _bmm_bf16_out(
        self, a, b_no_transpose, b_transposed, output, locality_domain_weight_shards=None
    ):
        """BMM with optional CuTe DSL bf16 acceleration on Blackwell/SM107."""
        if (
            self.use_cute_dsl_bf16_bmm
            and get_sm_version() == 107
            and locality_domain_weight_shards is not None
        ):
            torch.ops.trtllm.cute_dsl_bf16_bmm_locality_domain_inplace_rubin(
                a, locality_domain_weight_shards[0], locality_domain_weight_shards[1], output
            )
        elif self.use_cute_dsl_bf16_bmm and is_sm_100f():
            bf16_bmm_op = torch.ops.trtllm.cute_dsl_bf16_bmm_rubin
            bf16_bmm_op(a, b_no_transpose, output)
        else:
            torch.ops.trtllm.bmm_out(a, b_transposed, output)

    def forward_absorption_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        num_seqs = attn_metadata.num_seqs

        # The rope kernel writes seqQOffset[batch_idx + 1] for every
        # batch_idx < batch_size, and CUB-scans total_s_len / seq_len entries
        # into seqKVOffsets (mlaKernels.cu:579, :650). `batch_size` there is the
        # count of presented rows, which equals num_seqs only while every
        # request contributes the same number of query tokens. Under the
        # token-major presentation a row is a token, so size these by the token
        # count. Uniform verification keeps the established request-major
        # allocation rather than scaling these temporary buffers by K.
        cu_seqlens_rows = (
            max(num_seqs, num_tokens) + 1
            if getattr(attn_metadata, "is_ragged_verify", False)
            else num_seqs + 1
        )
        cu_q_seqlens = torch.empty(cu_seqlens_rows, dtype=torch.int32, device=q.device)
        cu_kv_seqlens = torch.empty(cu_seqlens_rows, dtype=torch.int32, device=q.device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)
        use_fp8_mla = self.mqa.has_fp8_kv_cache

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if use_fp8_mla:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
            quant_q_buffer = torch.empty(
                num_tokens,
                self.num_heads_tp,
                (self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.uint8,
                device=q.device,
            )

        if hasattr(self, "k_b_proj_trans"):
            fused_q = torch.empty(
                [
                    num_tokens,
                    self.num_heads_tp,
                    (self.kv_lora_rank + self.qk_rope_head_dim),
                ],
                dtype=q.dtype,
                device=q.device,
            )

            def _mla_gen_rope():
                # Fused backend (TRTLLM): RoPE, latent-cache append and the
                # trtllm-gen scheduler buffers are all produced in-kernel.
                self.mqa.mla_rope_generation(
                    fused_q,
                    q_pe,
                    latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                )

            if use_fp8_mla:
                rope_stream = None
            else:
                rope_stream = self.aux_stream
            if self.k_b_proj_trans.dtype == torch.bfloat16:
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
                # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
                # The output of bmm is written directly into fused_q
                maybe_execute_in_parallel(
                    lambda: self._bmm_bf16_out(
                        q_nope_t,
                        self.k_b_proj_trans,
                        self.k_b_proj_trans.transpose(1, 2),
                        q_nope_out,
                        self._locality_domain_k_b_proj_trans_shards,
                    ),
                    _mla_gen_rope,
                    self.ln_events[0],
                    self.ln_events[1],
                    rope_stream,
                )

            elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                maybe_execute_in_parallel(
                    lambda: fp8_block_scaling_bmm_out(
                        q_nope,
                        self.k_b_proj_trans,
                        self.k_b_proj_trans_scale,
                        q_nope_out,
                        self.k_b_proj_trans_dequant,
                        self.use_cute_dsl_blockscaling_bmm,
                    ),
                    _mla_gen_rope,
                    self.ln_events[0],
                    self.ln_events[1],
                    rope_stream,
                )
            else:
                raise NotImplementedError(
                    f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}."
                )

            fused_q = fused_q.view(
                [
                    num_tokens,
                    self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )
        else:
            raise RuntimeError("MLA absorption requires k_b_proj_trans")

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
            cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
            fmha_scheduler_counter=fmha_scheduler_counter,  # used by `mlaGeneration`
            mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
            mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
            quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
            **kwargs,
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (
            attn_out_latent.shape[0] == q.shape[0]
            and attn_out_latent.shape[1] == self.num_heads_tp_cp * self.kv_lora_rank
        )

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent.transpose(0, 1),
                self.v_b_proj,
                self.v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
                self._locality_domain_v_b_proj_shards,
            )
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def _init_fused_q_fp8_context(self) -> None:
        """Resolve the static half of the context-phase fused FP8-Q gate (see
        `_use_fused_q_fp8_context`). Called once from `create_weights()`."""
        ok = (
            os.environ.get("TRTLLM_DISABLE_FUSED_Q_FP8_QUANT", "0") != "1"
            # Independent of use_cute_dsl_bf16_bmm (off by default outside PP):
            # this path is a memory-traffic win, not a GEMM-kernel preference.
            and IS_CUTLASS_DSL_AVAILABLE
            and is_sm_100f()
            # Layout the C++ RoPE kernel instantiates kOutputFp8Q for on this path
            # (mlaKernels.cu); the CuTe bmm itself has no shape requirement.
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and self.k_b_proj_trans.dtype == torch.bfloat16
            and bool(getattr(self.mqa, "has_fp8_kv_cache", False))
        )
        if ok:
            # The CuTe epilogue applies no scale, so both writers of the FP8 Q
            # buffer must agree on unit scale (the PyTorch backend never sets
            # another value; checked once here, at construction time).
            scale = getattr(self.mqa, "kv_scale_orig_quant", None)
            ok = isinstance(scale, torch.Tensor) and bool(torch.all(scale == 1.0).item())
        if ok:
            # The attention op takes Q only for num_tokens / dtype on the fused
            # path; a stride-0 expand of one row satisfies that without a
            # [num_tokens, heads * 576] bf16 allocation per layer. Registered
            # as a (non-persistent) buffer so meta-init / `.to(device)` place
            # it alongside the weights.
            self._fused_q_placeholder = torch.empty(
                1,
                self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=self.k_b_proj_trans.dtype,
                device=self.k_b_proj_trans.device,
            )
        self._fused_q_fp8_ctx_ok = ok
        logger.info(f"MLA fused FP8-Q (absorbed context) {'enabled' if ok else 'disabled'}")

    def _use_fused_q_fp8_context(
        self, attn_metadata: AttentionMetadata, rope_applied_in_python: bool
    ) -> bool:
        """Context-phase fused FP8-Q for the absorbed MLA path.

        The FP8 FMHA reads Q as FP8 [tokens, heads, kv_lora + rope]. Instead of
        a bf16 Q plus a standalone quantize pass, the CuTe-DSL bmm stores FP8
        nope columns and the RoPE kernel (kOutputFp8Q) stores FP8 rope columns
        into one buffer; the C++ op then skips invokeMLAContextFp8Quantize.
        Requires RoPE fused in the kernel and a context-only batch (generation
        still reads the bf16 Q). TRTLLM_DISABLE_FUSED_Q_FP8_QUANT=1 disables it.
        """
        return (
            not rope_applied_in_python
            and self._fused_q_fp8_ctx_ok
            and attn_metadata.num_generations == 0
        )

    def forward_absorption_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_rope_applied = kwargs.pop("q_rope_applied", False)

        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if not hasattr(self, "k_b_proj_trans"):
            raise RuntimeError("MLA absorption requires k_b_proj_trans")
        # RoPE on q_pe happened (or will happen) outside the attention kernel.
        rope_applied_in_python = q_rope_applied

        fused_q_fp8 = None
        if self._use_fused_q_fp8_context(attn_metadata, rope_applied_in_python):
            # FP8 Q written once: nope columns by the bmm epilogue, rope columns
            # by the RoPE kernel (kOutputFp8Q). The attention op still takes a
            # bf16 Q for num_tokens / dtype; nothing reads it, so pass a view.
            fused_q_fp8 = q.new_empty(
                [num_tokens, self.num_heads_tp, self.kv_lora_rank + self.qk_rope_head_dim],
                dtype=torch.float8_e4m3fn,
            )
            torch.ops.trtllm.cute_dsl_bf16_bmm_fp8out_blackwell(
                q_nope.transpose(0, 1),  # [heads, tokens, qk_nope_head_dim]
                self.k_b_proj_trans,  # [heads, kv_lora_rank, qk_nope_head_dim]
                fused_q_fp8[..., : self.kv_lora_rank].transpose(
                    0, 1
                ),  # [heads, tokens, kv_lora_rank]
            )
            fused_q = self._fused_q_placeholder.expand(num_tokens, -1)
        else:
            # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
            # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
            fused_q = torch.empty(
                [
                    num_tokens,
                    self.num_heads_tp,
                    (self.kv_lora_rank + self.qk_rope_head_dim),
                ],
                dtype=q.dtype,
                device=q.device,
            )

            if self.k_b_proj_trans.dtype == torch.bfloat16:
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
                # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
                # The output of bmm is written directly into fused_q
                self._bmm_bf16_out(
                    q_nope_t,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans.transpose(1, 2),
                    q_nope_out,
                    self._locality_domain_k_b_proj_trans_shards,
                )
            elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                fp8_block_scaling_bmm_out(
                    q_nope,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans_scale,
                    q_nope_out,
                    self.k_b_proj_trans_dequant,
                    self.use_cute_dsl_blockscaling_bmm,
                )
            else:
                raise NotImplementedError(
                    f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}."
                )

            if rope_applied_in_python:
                fused_q[..., self.kv_lora_rank :] = q_pe
            fused_q = fused_q.view(
                [
                    num_tokens,
                    self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by applyMLARopeAndAssignQKVKernelOptContext
            # fused FP8-Q path only (both None otherwise): the C++ op skips
            # invokeMLAContextFp8Quantize when both are set.
            quant_q_buffer=fused_q_fp8,
            quant_scale_qkv=self.mqa.kv_scale_orig_quant if fused_q_fp8 is not None else None,
            **kwargs,
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (
            attn_out_latent.shape[0] == q.shape[0]
            and attn_out_latent.shape[1] == self.num_heads_tp_cp * self.kv_lora_rank
        )

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent.transpose(0, 1),
                self.v_b_proj,
                self.v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
                self._locality_domain_v_b_proj_shards,
            )
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def _bind_v_b_proj_weight(self, weight: torch.Tensor) -> None:
        """Bind the V absorption weight without retaining a released full weight."""
        plan = self.kv_b_proj.partition_plan
        if plan.enabled and plan.op_kind == "bf16_linear":
            weight = weight.clone(memory_format=torch.contiguous_format)
        self.v_b_proj = nn.Parameter(weight, requires_grad=False)
        self._locality_domain_k_b_proj_trans_shards = None
        self._locality_domain_v_b_proj_shards = None
        self._weights_transformed = False

    def _has_full_absorption_weights(self) -> bool:
        if getattr(self, "kv_lora_rank", None) != 512 or getattr(self, "v_head_dim", None) != 128:
            return False
        # The Rubin BF16 BMM requires its K dimension to be 16-byte aligned.
        # qk_nope_head_dim is K for the absorbed K projection.
        if getattr(self, "qk_nope_head_dim", 0) % 8 != 0:
            return False

        k_weight = getattr(self, "k_b_proj_trans", None)
        v_weight = getattr(self, "v_b_proj", None)
        if not isinstance(k_weight, torch.Tensor) or not isinstance(v_weight, torch.Tensor):
            return False
        if k_weight.is_meta or v_weight.is_meta:
            return False

        expected_k_shape = (
            self.num_heads_tp,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
        )
        expected_v_shape = (
            self.num_heads_tp_cp,
            self.v_head_dim,
            self.kv_lora_rank,
        )
        return (
            k_weight.dtype == v_weight.dtype
            and tuple(k_weight.shape) == expected_k_shape
            and tuple(v_weight.shape) == expected_v_shape
            and k_weight.numel() == math.prod(expected_k_shape)
            and v_weight.numel() == math.prod(expected_v_shape)
        )

    def _copy_absorption_weight_shards(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._locality_domain_runtime is not None
        num_partitions = self._locality_domain_runtime.num_partitions
        assert weight.shape[1] % num_partitions == 0

        shard_n = weight.shape[1] // num_partitions
        shards = []
        for partition_id in range(num_partitions):
            weight_slice = weight[
                :,
                partition_id * shard_n : (partition_id + 1) * shard_n,
                :,
            ]
            with self._locality_domain_runtime.partition_weight_context(partition_id):
                shard = torch.empty_like(
                    weight_slice,
                    memory_format=torch.contiguous_format,
                )
                shard.copy_(weight_slice)
            shards.append(shard)
        assert len(shards) == 2
        return shards[0], shards[1]

    def _rebuild_locality_domain_bf16_bmm_weight_shards(self) -> None:
        """Rebuild localized BF16 absorption weights from the full parameters."""
        self._locality_domain_k_b_proj_trans_shards = None
        self._locality_domain_v_b_proj_shards = None

        if not self._has_full_absorption_weights():
            return

        locality_domain_policy = getattr(self, "_locality_domain_policy", None)
        if locality_domain_policy is None:
            return
        plan = LocalityDomainExecutionPlanner(locality_domain_policy).plan_bf16_bmm(
            dtype=self.k_b_proj_trans.dtype,
            use_cute_dsl_bf16_bmm=getattr(self, "use_cute_dsl_bf16_bmm", False),
        )
        if not plan.enabled:
            return

        if getattr(self, "_locality_domain_runtime", None) is None:
            self._locality_domain_runtime = LocalityDomainRuntime(plan.num_partitions)

        k_shards = self._copy_absorption_weight_shards(self.k_b_proj_trans)
        v_shards = self._copy_absorption_weight_shards(self.v_b_proj)
        self._locality_domain_runtime.prepare_for_capture(plan)
        self._locality_domain_k_b_proj_trans_shards = k_shards
        self._locality_domain_v_b_proj_shards = v_shards

    def _project_context_kv(self, compressed_kv: torch.Tensor) -> torch.Tensor:
        return self.kv_b_proj(compressed_kv)

    def _attn_forward_gen(self, attn_backend, q, k, v, position_ids, attn_metadata, **kwargs):
        return attn_backend.forward(
            q, k, v, attn_metadata, forward_args=AttentionForwardArgs(**kwargs)
        )

    def _create_outputs(self, hidden_states, attn_metadata) -> list[torch.Tensor]:
        return [self.create_output(hidden_states, attn_metadata.num_contexts)]

    def forward_impl(
        self, position_ids, hidden_states, attn_metadata, attn_output, latent_cache_gen=None
    ) -> None:
        forward_sparse_attn(self, position_ids, hidden_states, attn_metadata, attn_output)

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        all_reduce_params=None,
        latent_cache_gen=None,
    ):
        output_hidden_states = hidden_states
        if isinstance(hidden_states, Fp4QuantizedTensor):
            assert hidden_states.unquantized_hidden_states is not None
            output_hidden_states = hidden_states.unquantized_hidden_states
        if self.register_to_config:
            attn_output = [create_mla_outputs(output_hidden_states, self.layer_idx_str)]
            forward_sparse_attn_custom_op(
                self, hidden_states, position_ids, attn_output, latent_cache_gen
            )
        else:
            attn_output = self._create_outputs(hidden_states, attn_metadata)
            self.forward_impl(
                position_ids, hidden_states, attn_metadata, attn_output, latent_cache_gen
            )
        return self.o_proj(
            attn_output[0], all_reduce_params=all_reduce_params, layer_idx=self.layer_idx
        )

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            self._rebuild_locality_domain_bf16_bmm_weight_shards()
            self._weights_transformed = True

    def cache_derived_state(self) -> None:
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()
