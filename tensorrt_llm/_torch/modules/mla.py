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

import functools
import math
import weakref
from typing import Optional, cast

import torch
from torch import nn

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.utils.fp4_utils import NVFP4_SF_VEC_SIZE

from ..attention_backend import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    FlashInferAttentionMetadata,
    TrtllmAttention,
    TrtllmAttentionMetadata,
)
from ..attention_backend.interface import (
    AttentionBackend,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
)
from ..attention_backend.sparse.hooks import get_sparse_attn_hooks
from ..attention_backend.utils import create_attention
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..utils import (
    Fp4QuantizedTensor,
    compute_swizzled_sf_shape,
    maybe_compiled_cat,
    maybe_compiled_copy_,
)
from .attention import (
    _helix_cp_allgather_input,
    _helix_cp_output_projection,
    _helix_post_process,
    _helix_zero_kv_mask,
    extract_extra_attrs,
)
from .linear import Linear, TensorParallelMode, is_static_nvfp4_input_eligible
from .multi_stream_utils import maybe_execute_in_parallel
from .rms_norm import RMSNorm
from .rotary_embedding import RotaryEmbedding


def _slice_hidden_states_to_num_tokens(hidden_states, num_tokens: int):
    """Drop CUDA-graph padding by slicing hidden_states to num_tokens rows.

    For a plain tensor this is the usual row slice. For an Fp4QuantizedTensor
    (produced when the previous layer's boundary fusion pre-quantized this
    layer's input) the slice must act on the packed FP4 form:
      - fp4_tensor: row-major [m, k//2], so [:num_tokens] is a plain row slice;
      - scaling_factor: 1D swizzled buffer of padUp(m,128)*padUp(cols,4). NVFP4
        quant is per-row independent and the swizzle is laid out in independent
        128-row tiles, so the leading padUp(num_tokens,128)*padUp(cols,4) bytes
        are exactly the scale factors for the first num_tokens rows (verified by
        tests/unittest/_torch/modules/test_fp4_num_tokens_slice.py).
      - unquantized_hidden_states (when present): plain row slice.
    """
    if not isinstance(hidden_states, Fp4QuantizedTensor):
        return hidden_states[:num_tokens, ...]

    fp4 = hidden_states.fp4_tensor
    if fp4.shape[0] == num_tokens:
        return hidden_states
    if not hidden_states.is_sf_swizzled:
        raise ValueError(
            "_slice_hidden_states_to_num_tokens only supports swizzled FP4 scaling factors"
        )
    sf = hidden_states.scaling_factor
    sf_cols = fp4.shape[-1] * 2 // NVFP4_SF_VEC_SIZE
    padded_rows, padded_cols = compute_swizzled_sf_shape(num_tokens, sf_cols)
    sf_len = padded_rows * padded_cols
    sliced_unquant = (
        hidden_states.unquantized_hidden_states[:num_tokens]
        if hidden_states.unquantized_hidden_states is not None
        else None
    )
    return Fp4QuantizedTensor(
        fp4_tensor=fp4[:num_tokens].contiguous(),
        scaling_factor=sf.view(-1)[:sf_len].contiguous(),
        is_sf_swizzled=True,
        unquantized_hidden_states=sliced_unquant,
    )


def _extract_mla_extra_attrs(layer_idx: str):
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    assert isinstance(mla_layer, MLA), "MLA layer must be a subclass of MLA or an instance of MLA"
    return metadata, mla_layer


def create_mla_outputs_impl(hidden_states: torch.Tensor, layer_idx: str) -> list[torch.Tensor]:
    metadata, mla_layer = _extract_mla_extra_attrs(layer_idx)
    return mla_layer._create_outputs(hidden_states, metadata)


@torch.library.custom_op("trtllm::create_mla_outputs", mutates_args=())
def create_mla_outputs(hidden_states: torch.Tensor, layer_idx: str) -> list[torch.Tensor]:
    return create_mla_outputs_impl(hidden_states, layer_idx)


@create_mla_outputs.register_fake
def _create_mla_outputs_fake(hidden_states, layer_idx):
    return create_mla_outputs_impl(hidden_states, layer_idx)


@torch.library.custom_op(
    "trtllm::mla_custom_op_inplace",
    mutates_args=("attn_output",),
)
def mla_custom_op_inplace(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    attn_output: list[torch.Tensor],
    latent_cache_gen: Optional[torch.Tensor],
    hidden_states_fp4: Optional[torch.Tensor] = None,
    hidden_states_sf: Optional[torch.Tensor] = None,
) -> None:
    metadata, mla_layer = _extract_mla_extra_attrs(layer_idx)
    if hidden_states_fp4 is not None or hidden_states_sf is not None:
        assert hidden_states_fp4 is not None and hidden_states_sf is not None, (
            "hidden_states_fp4 and hidden_states_sf must be passed together"
        )
        hidden_states = Fp4QuantizedTensor(
            fp4_tensor=hidden_states_fp4,
            scaling_factor=hidden_states_sf,
            unquantized_hidden_states=hidden_states,
        )
    mla_layer.forward_impl(
        position_ids,
        hidden_states,
        metadata,
        attn_output=attn_output,
        latent_cache_gen=latent_cache_gen,
    )


def fp8_block_scaling_bmm_out(
    mat1: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
    mat2_dequant: Optional[torch.Tensor] = None,
    use_cute_dsl_blockscaling_bmm: bool = False,
) -> torch.Tensor:
    sm_version = get_sm_version()
    if sm_version == 90 or sm_version == 89:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(mat1)

        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, output
        )
        out.copy_(output)
    elif sm_version == 120:
        mat1_fp8, mat1_scale = fp8_utils.per_token_quant_and_transform(mat1, need_permute102=True)
        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, output
        )
        out.copy_(output)
    elif is_sm_100f(sm_version):
        if use_cute_dsl_blockscaling_bmm:
            mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(mat1)
            torch.ops.trtllm.cute_dsl_fp8_bmm_blackwell(
                mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, out
            )
            mat1_scale = None
        else:
            torch.bmm(mat1.transpose(0, 1), mat2_dequant.transpose(1, 2), out=out)
    else:
        raise NotImplementedError(f"SM{sm_version} is not supported")


class MLA(nn.Module):
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
        aux_stream: Optional[torch.cuda.Stream] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
        num_groups: int = 1,
        o_lora_rank: int = 1024,
    ):
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
            aux_stream (Optional[torch.cuda.Stream]): The auxiliary CUDA stream
                for running operations in two parallel streams.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
            num_groups (int): The number of groups.
            o_lora_rank (int): The dimension of the compressed output.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype
        self._weights_transformed = False

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
        self.num_groups = num_groups
        self.o_lora_rank = o_lora_rank
        if dense_bias is None:
            self.dense_bias = bias

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

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
        self.sparse_attn_hooks = get_sparse_attn_hooks(self)

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
        if self.mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1
        if self.mapping.has_cp_ulysses():
            raise NotImplementedError("MLA doesn't support CP Ulysses yet")
        if self.mapping.cp_size > 1:
            assert self.mapping.has_cp_helix(), (
                f"CP type must be HELIX for MLA, but got {self.mapping.cp_config['cp_type']}."
            )

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
        self.num_key_value_heads_tp = (self.num_key_value_heads + tp_size - 1) // tp_size

        rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", 1e-6)
        quant_config = config.get_quant_config()
        self.quant_config = quant_config

        self.use_cute_dsl_blockscaling_mm = config.use_cute_dsl_blockscaling_mm
        self.use_cute_dsl_blockscaling_bmm = config.use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = config.use_cute_dsl_bf16_bmm
        self.use_cute_dsl_bf16_gemm = config.use_cute_dsl_bf16_gemm

        if not self.is_lite:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            )

            self.q_a_layernorm = RMSNorm(
                hidden_size=self.q_lora_rank, eps=rms_norm_eps, dtype=dtype
            )

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
            )
        else:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
            )

            self.q_proj = Linear(
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
            )
            self.q_b_proj = self.q_proj

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

        self.aux_stream = aux_stream
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
        )
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )
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
        )
        self.attention_output_hidden_size = self.o_proj.in_features

        self.mqa = create_attention(
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
            aux_stream=aux_stream,
            rope_append=True,
        )

        # MHA is the dense expanded-KV path. Algorithms that do not use it
        # remove it in their initialization hook.
        self.mha = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads_tp,
            head_dim=self.qk_head_dim,
            num_kv_heads=self.num_key_value_heads_tp,
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
            sparse_params=None,
        )

        if initialize_sparse_attn := self.sparse_attn_hooks.initialize_sparse_attn:
            initialize_sparse_attn(
                self,
                config=config,
                mapping=mapping,
                mapping_o=mapping_o,
                rms_norm_eps=rms_norm_eps,
                quant_config=quant_config,
                q_scaling=q_scaling,
                bias=bias,
                dtype=dtype,
                reduce_output=reduce_output,
                aux_stream=aux_stream,
            )

        if self.mqa is None:
            raise RuntimeError("MLA requires a non-null MQA attention backend")

        self.softmax_scale = 1.0 / (math.sqrt(self.qk_head_dim) * q_scaling)

        self.rope_fusion = self.mqa.support_fused_rope()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        self.llama_4_scaling = False
        if hasattr(config.pretrained_config, "llama_4_scaling"):
            self.llama_4_scaling = True
            self.floor_scale = getattr(
                config.pretrained_config.llama_4_scaling,
                "original_max_position_embeddings",
                8192,
            )
            self.attn_scale = getattr(config.pretrained_config.llama_4_scaling, "beta", 0.1)

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to
        # quant_config, which could be modified after __init__.
        if self.mha is not None:
            self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)

        # Although we use FP8 MLA for context/generation phase, the output is still in BF16
        self.out_scale = None
        if create_sparse_attn_weights := self.sparse_attn_hooks.create_sparse_attn_weights:
            create_sparse_attn_weights(self)
            self._weights_transformed = False
            return

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
        self._weights_transformed = False

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads_tp, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim :].reshape(
            -1, self.num_heads_tp * self.qk_rope_head_dim
        )
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim :] = q_pe.view(-1, self.num_heads_tp, self.qk_rope_head_dim)
        return k_pe

    def _attn_forward_gen(
        self,
        attn_backend: AttentionBackend,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ):
        if self.mapping.has_cp_helix():
            # partial_o: [num_tokens, num_heads_tp * kv_lora_rank]
            # softmax_stats: [num_tokens, num_heads_tp, 2]
            softmax_stats = torch.empty(
                (q.shape[0], self.num_heads_tp, 2), device=q.device, dtype=torch.float32
            )
            kwargs["softmax_stats_tensor"] = softmax_stats
            partial_o = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            kv_lora_rank = partial_o.shape[-1] // self.num_heads_tp
            assert self.kv_lora_rank == kv_lora_rank

            # MLA processes only the generation token slice here, so build the
            # mask from the generation sequence range [num_contexts, num_seqs).
            zero_kv_mask = _helix_zero_kv_mask(
                attn_metadata,
                partial_o.shape[0],
                seq_start=attn_metadata.num_contexts,
                num_seqs=attn_metadata.num_generations,
            )
            return _helix_post_process(
                partial_o,
                softmax_stats,
                self.mapping,
                self.num_heads_tp_cp,
                kv_lora_rank,
                self.aux_stream,
                self.ln_events,
                zero_kv_mask=zero_kv_mask,
            )
        else:
            attn_output = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            return attn_output

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

    def _create_outputs(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> list[torch.Tensor]:
        """Create the standard output and any algorithm-specific output buffers."""
        if prepare_outputs := self.sparse_attn_hooks.prepare_sparse_attn_outputs:
            return prepare_outputs(self, hidden_states, attn_metadata)
        return [self.create_output(hidden_states, attn_metadata.num_contexts)]

    def _attention_scaling(self, q, position_ids):
        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def _resolve_qa_fused_scale(self):
        """Resolve whether q_a_layernorm can fuse q_b_proj NVFP4 input quantization."""
        if self._qa_fused_scale is not None:
            return self._qa_fused_scale if self._qa_fused_scale is not False else None
        eligible = False
        if (
            not self.is_lite
            and getattr(self, "q_a_layernorm", None) is not None
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

    def forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_output: list[torch.Tensor],
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> None:
        """Run the dense or sparse implementation of the shared MLA module."""
        if self.sparse_attn_hooks:
            self.sparse_attn_hooks.require("forward_sparse_attn")(
                self,
                position_ids,
                hidden_states,
                attn_metadata,
                attn_output,
            )
            return

        self._forward_impl(
            position_ids,
            hidden_states,
            attn_metadata,
            attn_output[0],
            latent_cache_gen=latent_cache_gen,
        )

    def _forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Forward pass for the MLA module. Writes result into output tensor in-place.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            output (torch.Tensor): The output tensor to write results into.
            latent_cache_gen (Optional[torch.Tensor]): The latent cache used in generation.
        """
        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        hidden_states = _slice_hidden_states_to_num_tokens(hidden_states, num_tokens)
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        if self.is_lite:
            compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1
            )
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
            q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
            )

            q, compressed_kv = maybe_execute_in_parallel(
                lambda: self._q_a_layernorm_maybe_fused(q),
                lambda: self.kv_a_layernorm(compressed_kv),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        q, latent_cache = maybe_execute_in_parallel(
            lambda: self.q_b_proj(q),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        assert q.shape[0] == num_tokens, (
            f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"
        )

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                # position_ids spans [ctx..., gen...] in mixed batches; slice to
                # match q_ctx/k_pe_ctx so external RoPE uses ctx positions.
                ctx_position_ids = position_ids[..., :num_ctx_tokens]
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, ctx_position_ids)
                # External RoPE is only used by backends that do not handle
                # fused RoPE internally, so keep latent_cache in sync.
                latent_cache_ctx = torch.cat([compressed_kv_ctx, k_pe_ctx], dim=-1)

            if self.llama_4_scaling:
                q_ctx = self._attention_scaling(q_ctx, position_ids[..., :num_ctx_tokens])

            self.forward_context(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                position_ids,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            if latent_cache_gen is None:
                latent_cache_gen = latent_cache[num_ctx_tokens:, ...]

            if self.llama_4_scaling:
                q_gen = self._attention_scaling(q_gen, position_ids[..., num_ctx_tokens:])

            self.forward_absorption_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                position_ids=position_ids,
                latent_cache=latent_cache_gen,
            )

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
        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim,
            ],
            -1,
        )

        k = torch.empty_like(q).view(-1, self.num_heads_tp, self.qk_head_dim)
        maybe_compiled_copy_(
            k[..., : self.qk_nope_head_dim],
            k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim),
        )
        # When rope_fusion=True (apply_rotary_emb=False), the rope portion
        # of k is left uninitialized here; the fused attention kernel
        # handles k_pe RoPE via latent_cache instead.
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim :] = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

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
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim,
            ],
            -1,
        )

        full_k_nope = full_k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
        full_k = maybe_compiled_cat(
            (full_k_nope, full_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1
        )
        full_k = full_k.view(-1, self.num_heads_tp * self.qk_head_dim)

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
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device="cuda",
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device="cuda",
        )

        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads_tp * self.v_head_dim), dtype=q.dtype
        )

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime
        origin_ctx_total_kv_len = attn_metadata.host_total_kv_lens[0]

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
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)
            chunked_k_nope, chunked_v = chunked_kv.split(
                [
                    self.num_heads_tp * self.qk_nope_head_dim,
                    self.num_heads_tp * self.v_head_dim,
                ],
                -1,
            )

            chunked_k_nope = chunked_k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
            chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
            chunked_k = maybe_compiled_cat(
                (chunked_k_nope, chunked_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1
            )
            chunked_k = chunked_k.view(-1, self.num_heads_tp * self.qk_head_dim)

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
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([-1, self.kv_lora_rank + self.qk_rope_head_dim]).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        # final round of attention

        k_nope, v = kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim,
            ],
            -1,
        )

        k_nope = k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = maybe_compiled_cat((k_nope, k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

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
        if isinstance(attn_metadata, FlashInferAttentionMetadata):
            if (
                attn_metadata.enable_context_mla_with_cached_kv
                and attn_metadata.num_ctx_cached_tokens > 0
            ):
                return self.forward_absorption_context(
                    q,
                    compressed_kv,
                    k_pe,
                    attn_metadata,
                    output,
                    position_ids=position_ids,
                    latent_cache=latent_cache,
                )

        if isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            # Warm up maybe_compiled_cat for both the chunked-prefill path and
            # the cached-kv path; without this, the cached-kv prefill
            # (block-reuse without chunked_prefill) recompiles per shape and
            # may stall inside inductor's compile worker.
            if trtllm_attention.has_cached_kv_for_mla_context_warmup(attn_metadata):
                self.cached_warmup_forward_context_with_cached_kv(
                    self.num_heads_tp,
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

    def _bmm_bf16_out(self, a, b_no_transpose, b_transposed, output):
        """BMM with optional CuTe DSL bf16 acceleration on Blackwell."""
        if self.use_cute_dsl_bf16_bmm and is_sm_100f():
            torch.ops.trtllm.cute_dsl_bf16_bmm_blackwell(a, b_no_transpose, output)
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
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        num_seqs = attn_metadata.num_seqs

        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)
        has_fp8_kv_cache = (
            self.mqa.has_fp8_kv_cache if hasattr(self.mqa, "has_fp8_kv_cache") else False
        )

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if has_fp8_kv_cache:
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
                if self.apply_rotary_emb:
                    # Non-fused backends (Vanilla / FlashInfer) do not fuse RoPE
                    # in the attention kernel. Reuse apply_rope, which rotates
                    # q's q_pe slice in place, then copy rotated k_pe into the
                    # latent cache that the backend appends.
                    assert position_ids is not None
                    assert latent_cache is not None
                    gen_position_ids = position_ids[..., attn_metadata.num_ctx_tokens :]
                    k_pe_rope = self.apply_rope(q, k_pe, gen_position_ids)
                    fused_q[..., self.kv_lora_rank :] = q_pe
                    latent_cache[..., self.kv_lora_rank :] = k_pe_rope
                else:
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

            rope_stream = self.aux_stream if not has_fp8_kv_cache else None
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

    def forward_absorption_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]

        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if hasattr(self, "k_b_proj_trans"):
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

            if self.apply_rotary_emb:
                fused_q[..., self.kv_lora_rank :] = q_pe
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
            attention_input_type=AttentionInputType.context_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by applyMLARopeAndAssignQKVKernelOptContext
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

    def _forward_custom_op(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_output: list[torch.Tensor],
        latent_cache_gen: Optional[torch.Tensor],
    ) -> None:
        """Run the dense or sparse registered custom-op implementation."""
        if custom_op := self.sparse_attn_hooks.forward_sparse_attn_custom_op:
            custom_op(
                self,
                hidden_states,
                position_ids,
                attn_output,
                latent_cache_gen,
            )
            return

        if isinstance(hidden_states, Fp4QuantizedTensor):
            torch.ops.trtllm.mla_custom_op_inplace(
                hidden_states.unquantized_hidden_states,
                position_ids,
                self.layer_idx_str,
                attn_output,
                latent_cache_gen,
                hidden_states.fp4_tensor,
                hidden_states.scaling_factor,
            )
        else:
            torch.ops.trtllm.mla_custom_op_inplace(
                hidden_states,
                position_ids,
                self.layer_idx_str,
                attn_output,
                latent_cache_gen,
            )

    def _project_output(
        self,
        attn_output: list[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams],
    ) -> torch.Tensor:
        """Apply the MLA output projection."""
        if project_output := self.sparse_attn_hooks.project_sparse_attn_output:
            return project_output(self, attn_output, position_ids, attn_metadata, all_reduce_params)

        return self._project_output_impl(
            attn_output[0], position_ids, attn_metadata, all_reduce_params
        )

    def _project_output_impl(
        self,
        attn_output: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams],
    ) -> torch.Tensor:
        """Apply the default MLA output projection implementation."""
        return _helix_cp_output_projection(
            self.o_proj,
            attn_output,
            attn_metadata,
            all_reduce_params,
            self.mapping,
            self.mapping_o,
            self.layer_idx,
        )

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = _helix_cp_allgather_input(
            hidden_states, attn_metadata, self.mapping, self.layer_idx
        )

        if self.register_to_config:
            output_hidden_states = hidden_states
            if isinstance(hidden_states, Fp4QuantizedTensor):
                assert hidden_states.unquantized_hidden_states is not None, (
                    "MLA.forward received an Fp4QuantizedTensor without a "
                    "unquantized_hidden_states view"
                )
                output_hidden_states = hidden_states.unquantized_hidden_states
            attn_output = torch.ops.trtllm.create_mla_outputs(
                output_hidden_states, self.layer_idx_str
            )
            self._forward_custom_op(
                hidden_states,
                position_ids,
                attn_output,
                latent_cache_gen,
            )
        else:
            attn_output = self._create_outputs(hidden_states, attn_metadata)
            self.forward_impl(
                position_ids,
                hidden_states,
                attn_metadata,
                attn_output=attn_output,
                latent_cache_gen=latent_cache_gen,
            )

        return self._project_output(attn_output, position_ids, attn_metadata, all_reduce_params)

    def resmooth_parameters(self, module_weight, module_weight_scale, recipe=(1, 128, 128)):
        weight, weight_scale = fp8_utils.resmooth_to_fp8_e8m0(module_weight, module_weight_scale)

        transfromed_scale = fp8_utils.transform_sf_into_required_layout(
            weight_scale,
            mn=weight.shape[1],
            k=weight.shape[2],
            recipe=recipe,
            num_groups=weight.shape[0],
            is_sfa=False,
        )

        weight_param = torch.nn.Parameter(weight, requires_grad=False)
        scale_param = torch.nn.Parameter(transfromed_scale, requires_grad=False)

        return weight_param, scale_param

    def transform_weights(self) -> None:
        if self._weights_transformed:
            return
        if transform_sparse_attn_weights := self.sparse_attn_hooks.transform_sparse_attn_weights:
            transform_sparse_attn_weights(self)
            self._weights_transformed = True
            return

        has_fp8_block_scales = bool(
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales()
        )
        if get_sm_version() == 120 and has_fp8_block_scales:
            self.k_b_proj_trans, self.k_b_proj_trans_scale = self.resmooth_parameters(
                self.k_b_proj_trans, self.k_b_proj_trans_scale, recipe=(1, 128, 128)
            )

            self.v_b_proj, self.v_b_proj_scale = self.resmooth_parameters(
                self.v_b_proj, self.v_b_proj_scale, recipe=(1, 128, 128)
            )
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
