import math
import weakref
from typing import List, Optional, Union, cast

import torch
from torch import nn

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import (get_sm_version, is_sm_100f, nvtx_range,
                                 nvtx_range_debug)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import (AttentionInputType, AttentionMetadata,
                                 FlashInferAttentionMetadata, TrtllmAttention,
                                 TrtllmAttentionMetadata)
from ..attention_backend.interface import (AttentionBackend, AttentionMask,
                                           CustomAttentionMask,
                                           PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.sparse.dsa import (
    DSAtrtllmAttentionMetadata, transform_local_topk_and_prepare_pool_view)
from ..attention_backend.utils import create_attention, get_attention_backend
from ..distributed import AllReduceParams, HelixAllToAllNative, alltoall_helix
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import (Fp4QuantizedTensor, get_model_extra_attrs,
                     is_torch_compiling, maybe_compiled_cat,
                     maybe_compiled_copy_)
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .multi_stream_utils import maybe_execute_in_parallel
from .rms_norm import RMSNorm
from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding

# Import FlashMLA sparse attention kernel
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None


def extract_extra_attrs(layer_idx: str, attn_type: str):
    assert attn_type in ["mla", "attn"], "Invalid attention type"
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    if attn_type == "mla":
        assert isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )
    else:
        assert isinstance(
            metadata,
            FlashInferAttentionMetadata,
        ) or isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )

    attn_layers = extra_attrs.get(attn_type + "_layers", None)
    assert attn_layers is not None, "Attention layer is not registered"
    attn_layer_ref = attn_layers.get(layer_idx, None)
    assert attn_layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    attn_layer = attn_layer_ref()

    if attn_type == "mla":
        assert isinstance(
            attn_layer,
            MLA), "MLA layer must be a subclass of MLA or an instance of MLA"
    elif attn_type == "attn":
        assert isinstance(
            attn_layer, Attention
        ), "Attention layer must be a subclass of Attention or an instance of Attention"

    return metadata, attn_layer


def create_attn_outputs_impl(q: torch.Tensor, attention_mask: str,
                             layer_idx: str) -> List[torch.Tensor]:
    metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
    return attn_layer.create_output(q, metadata, attention_mask)


@torch.library.custom_op("trtllm::create_attn_outputs", mutates_args=())
def create_attn_outputs(q: torch.Tensor, attention_mask: str,
                        layer_idx: str) -> List[torch.Tensor]:
    return create_attn_outputs_impl(q, attention_mask, layer_idx)


@create_attn_outputs.register_fake
def _(q, attention_mask, layer_idx):
    return create_attn_outputs_impl(q, attention_mask, layer_idx)


@torch.library.custom_op("trtllm::attn_custom_op_inplace",
                         mutates_args=("output", "output_sf"))
def attn_custom_op_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    attention_mask: str,
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    attention_window_size: Optional[int],
    attention_mask_data: Optional[torch.Tensor],
    attention_sinks: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
    output_sf: Optional[torch.Tensor],
) -> None:
    metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
    mask = PredefinedAttentionMask(
        attention_mask
    ) if attention_mask != CustomAttentionMask.CUSTOM else CustomAttentionMask(
        attention_mask)
    # NVFP4 output cannot be supported by torch compile for TRTLLM backend.
    attn_layer._attn_impl(q,
                          k,
                          v,
                          metadata,
                          mask,
                          mrope_rotary_cos_sin,
                          mrope_position_deltas,
                          attention_window_size,
                          attention_mask_data,
                          output=output,
                          output_sf=output_sf,
                          attention_sinks=attention_sinks)


class Attention(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rope_fusion: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
        attention_chunk_size: Optional[int] = None,
        disable_deep_gemm: bool = False,
        attn_output_gate: Optional[bool] = None,
        use_custom_cublas_mm: bool = False,
        reduce_output: bool = True,
    ):
        """
        Initialize the Attention module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            pos_embd_params (Optional[PositionalEmbeddingParams]): The positional embedding parameters.
            rope_fusion (Optional[bool]): Whether to fuse RoPE into the attention OP and skip applying unfused RoPE. If None, whether to fuse is decided by the capability of the attention backend.
            layer_idx (Optional[int]): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (Optional[bool]): Whether to use bias in the output projection layer.
            config (Optional[ModelConfig]): The model configuration.
            q_scaling (float): The scaling factor for the qk_scale. The definition is $O = softmax(QK^T * qk_scale) * V, qk_scale = 1 / (sqrt(head_dim) * q_scaling)$. The default value is 1.0.
            attention_chunk_size (Optional[int]): See [Chunked Attention] below.
            disable_deep_gemm (bool): Whether to disable the use of DeepGEMM in Linear layers (currently only matters on SM100 + FP8).
            attn_output_gate (Optional[bool]): Determines whether to use an output gate in the attention Op. If False, the decision is automatically handled by the attention backend based on its capabilities.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)

        self.register_to_config = False
        # We only register TRTLLM attention layers to config.
        if config is not None:
            if "attn_layers" not in config.extra_attrs:
                config.extra_attrs["attn_layers"] = {}
            suffix = 0
            # Makes sure there is no duplicate attention layer identifier.
            while self.layer_idx_str in config.extra_attrs["attn_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            config.extra_attrs["attn_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = getattr(config.pretrained_config, 'head_dim', None)
        if not isinstance(self.head_dim, int):
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.q_scaling = q_scaling
        self.attn_output_gate = attn_output_gate

        if self.attn_output_gate:
            logger.info_once("using attn output gate!", key="attn_output_gate")

        # [Chunked Attention]
        # Chunked attention is applied to context requests only. Chunked attention will be
        # applied when this field is specified and mMaskType == CAUSAL.
        #
        # In chunked attention, we break context requests into chunks of a specified size. Tokens can only
        # attend to tokens in the same chunk. So, for example, if the chunk size is 3, we might have a mask
        # that looks like this:
        #
        # 1 0 0 0 0 0
        # 1 1 0 0 0 0
        # 1 1 1 0 0 0
        # 0 0 0 1 0 0
        # 0 0 0 1 1 0
        # 0 0 0 1 1 1
        self.attention_chunk_size = attention_chunk_size

        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        cp_size = config.mapping.cp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size * cp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            cp_config=config.mapping.cp_config,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )
        self.tp_size = tp_size
        self.tp_rank = mapping.tp_rank
        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        qkv_shard_indices_mapping = {
            "q": (0, self.q_size * (2 if self.attn_output_gate else 1)),
            "k":
            (self.q_size * (2 if self.attn_output_gate else 1), self.kv_size),
            "v":
            (self.q_size * (2 if self.attn_output_gate else 1) + self.kv_size,
             self.kv_size),
        }

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size * (2 if self.attn_output_gate else 1) +
            2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm,
            fused_weight_shard_indices_mapping=qkv_shard_indices_mapping)

        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.o_lora,
            reduce_output=reduce_output,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm)

        self.quant_config = config.get_quant_config()
        self.attn_backend = config.attn_backend
        attn_cls = get_attention_backend(
            self.attn_backend,
            sparse_attn_config=config.sparse_attention_config)

        # These two modules are mutually exclusive - either splitted_qkv_lora or fused_qkv_lora will be used,
        # but never both at the same time. splitted_qkv_lora handles Q,K,V separately while fused_qkv_lora
        # handles them as a single fused operation.
        self.splitted_qkv_lora = LoraLayer([
            LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V
        ], [self.q_size, self.kv_size, self.kv_size])
        self.fused_qkv_lora = LoraLayer([LoraModuleType.ATTENTION_QKV],
                                        [self.q_size + 2 * self.kv_size])

        # Whether to fuse RoPE into the attention OP.
        # If true, RoPE will be applied in self.attn.forward.
        # If false, RoPE will be applied in self.apply_rope.
        self.rope_fusion = rope_fusion

        if config.sparse_attention_config is not None:
            # Log sparse attention configuration once
            algo = config.sparse_attention_config.algorithm
            cfg_dump = config.sparse_attention_config.model_dump(
                exclude_none=True)
            logger.info_once(f"Using sparse attention: {algo} {cfg_dump}",
                             key="sparse_attention_config")

            if config.sparse_attention_config.algorithm == "rocket":
                logger.warning("disable rope_fusion for RocketKV.")
                self.rope_fusion = False

        if self.rope_fusion and not attn_cls.support_fused_rope():
            logger.warning(
                "rope_fusion is true but the attention backend does not support it. Will disable rope_fusion."
            )
            self.rope_fusion = False
        # If rope_fusion is not specified, enable if the attention backend supports it.
        if self.rope_fusion is None:
            self.rope_fusion = attn_cls.support_fused_rope()

        self.rotary_emb = None
        if not self.rope_fusion and self.pos_embd_params is not None:
            if self.pos_embd_params.type.is_mrope():
                self.rotary_emb = MRotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                    mrope_section=self.pos_embd_params.mrope_section,
                    mrope_interleaved=self.pos_embd_params.mrope_interleaved)
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                )

        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=self.pos_embd_params if self.rope_fusion else None,
            quant_config=self.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            q_scaling=self.q_scaling,
            attention_chunk_size=self.attention_chunk_size,
            sparse_attention_config=config.sparse_attention_config,
        )

        self.support_fused_qkv = self.attn.support_fused_qkv()

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.attn has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.attn.update_quant_config(self.quant_config)

        self.o_proj.create_weights()
        self.has_quant_scale = (self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4
                                or self.o_proj.has_fp8_block_scales
                                or self.o_proj.has_fp8_rowwise
                                or self.o_proj.has_w4a8_nvfp4_fp8)

    def split_qkv(self, q, k=None, v=None):
        if k is None and v is None:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def convert_qkv(self, q, k, v):
        if k is None and v is None and not self.support_fused_qkv:
            q, k, v = self.split_qkv(q)
        elif k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def _use_quantize_output(self):
        has_awq_pre_quant_scale = hasattr(
            self.o_proj,
            'pre_quant_scale') and self.o_proj.pre_quant_scale is not None

        return self.has_quant_scale and not self.attn_output_gate and not has_awq_pre_quant_scale

    def create_output(self, q: torch.Tensor, attn_metadata: AttentionMetadata,
                      mask_type: str):
        # Attention is treated as mixed request by default.
        return self.attn.create_output(
            q,
            is_quantize_output=self._use_quantize_output(),
            metadata=attn_metadata,
            attention_mask=mask_type,
            is_gen_only=False)

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        has_lora: bool = False,
    ):
        num_tokens = attn_metadata.num_tokens

        q = q[:num_tokens, :]
        if k is not None:
            k = k[:num_tokens, :]
        if v is not None:
            v = v[:num_tokens, :]

        out_scale = None
        out_scale_sf = None
        # Don't set out_scale if o_proj has pre_quant_scale - this prevents FP8/FP4 output
        # and keeps attention output in BF16 for better precision when applying pre_quant_scale
        # Also don't set out_scale if LoRA is active - LoRA grouped_gemm doesn't support FP8
        if self._use_quantize_output() and not has_lora:
            out_scale = self.o_proj.inv_input_scale
            out_scale_sf = self.o_proj.input_scale

        kv_scales_sf = None
        kv_scales_sf_inv = None
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp4_kv_cache(
        ):
            kv_scales_sf = self.qkv_proj.kv_scales
            kv_scales_sf_inv = self.qkv_proj.inv_kv_scales

        mrope_config = None
        if mrope_rotary_cos_sin is not None or mrope_position_deltas is not None:
            mrope_config = dict()
            if mrope_rotary_cos_sin is not None:
                mrope_config["mrope_rotary_cos_sin"] = mrope_rotary_cos_sin
            if mrope_position_deltas is not None:
                mrope_config["mrope_position_deltas"] = mrope_position_deltas

        attn_output = self.attn.forward(
            q,
            k,
            v,
            attn_metadata,
            out_scale=out_scale,
            out_scale_sf=out_scale_sf,
            kv_scales_sf=kv_scales_sf,
            kv_scales_sf_inv=kv_scales_sf_inv,
            attention_mask=attention_mask,
            mrope_config=mrope_config,
            attention_window_size=attention_window_size,
            attention_mask_data=attention_mask_data,
            output=output[:num_tokens, :] if output is not None else None,
            output_sf=output_sf,
            attention_sinks=attention_sinks)
        if isinstance(attn_output, tuple):
            assert len(
                attn_output
            ) == 2, "attn_output should be a tuple of (output, output_sf)"
            return attn_output[0], attn_output[1]
        return attn_output, None

    def forward_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        mrope_config: Optional[dict],
        attention_sinks: Optional[torch.Tensor] = None,
        has_lora: bool = False,
    ):
        mrope_rotary_cos_sin = None
        mrope_position_deltas = None
        if mrope_config is not None:
            if "mrope_rotary_cos_sin" in mrope_config:
                mrope_rotary_cos_sin = mrope_config["mrope_rotary_cos_sin"]
            if "mrope_position_deltas" in mrope_config:
                mrope_position_deltas = mrope_config["mrope_position_deltas"]

        # Currently only TRTLLM and FLASHINFER are torch compile compatible backends.
        # Only enable custom inplace op when torch compiling.
        use_custom_inplace_op = (self.register_to_config
                                 and (self.attn_backend == "TRTLLM"
                                      or self.attn_backend == "FLASHINFER")
                                 and is_torch_compiling())

        if use_custom_inplace_op:
            outputs = create_attn_outputs(q, attention_mask, self.layer_idx_str)
            assert len(outputs) == 1 or len(outputs) == 2
            output = outputs[0]
            output_sf = outputs[1] if len(outputs) == 2 else None
            attn_custom_op_inplace(
                q,
                k,
                v,
                attention_mask,
                mrope_rotary_cos_sin,
                mrope_position_deltas,
                attention_window_size,
                attention_mask_data,
                attention_sinks,
                self.layer_idx_str,
                output,
                output_sf,
            )
        else:
            output, output_sf = self._attn_impl(q,
                                                k,
                                                v,
                                                attn_metadata,
                                                attention_mask,
                                                mrope_rotary_cos_sin,
                                                mrope_position_deltas,
                                                attention_window_size,
                                                attention_mask_data,
                                                attention_sinks=attention_sinks,
                                                has_lora=has_lora)
        if output_sf is not None:
            output = Fp4QuantizedTensor(output, output_sf)

        return output

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            attention_mask (AttentionMask): The attention mask type.
            mrope_config (Optional[dict]): The MROPE configuration.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.
            lora_params (Optional[dict]): The LoRA parameters.
            attention_window_size (Optional[int]): The attention window size.
            attention_mask_data (Optional[torch.Tensor]): The attention mask data.
        Returns:
            torch.Tensor: The output tensor.
        """
        qkv = self.qkv_proj(hidden_states)

        if bool(lora_params):
            qkv_lora = self.splitted_qkv_lora(hidden_states, lora_params,
                                              self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

            qkv_lora = self.fused_qkv_lora(hidden_states, lora_params,
                                           self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            # Single line: view -> chunk -> reshape both q and gate
            q, gate = [
                t.reshape(*orig_shape, -1) for t in torch.chunk(
                    q_gate.view(*orig_shape, self.num_heads, -1), 2, dim=-1)
            ]
        else:
            q, k, v = qkv, None, None

        q, k, v = self.apply_rope(q, k, v, position_ids)
        q, k, v = self.convert_qkv(q, k, v)

        if attention_sinks is not None:
            assert self.attn_backend == "TRTLLM", "Attention sinks are only supported for TRTLLM backend."

        attn_output = self.forward_impl(q,
                                        k,
                                        v,
                                        attn_metadata,
                                        attention_mask,
                                        attention_window_size,
                                        attention_mask_data,
                                        mrope_config=mrope_config,
                                        attention_sinks=attention_sinks,
                                        has_lora=bool(lora_params))

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params,
                                  lora_params=lora_params,
                                  layer_idx=self.layer_idx)
        return attn_output

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        Apply RoPE to the query and key.
        Depending on the implementation, q, k, v could be either fused (q, k, v = concat(q, k, v), None, None) or unfused (none of q, k, v is None).
        Before self.attn.forward, convert_qkv will be called to make sure that the format of (q, k, v) satisfies the requirement of self.attn.
        This method could be overridden in the subclass, in which extra functionalities such as q_norm/k_norm could be added.
        Args:
            q (torch.Tensor): The query tensor.
            k (Optional[torch.Tensor]): The key tensor.
            v (Optional[torch.Tensor]): The value tensor.
            position_ids (torch.Tensor): The position IDs of each token for RoPE.
        Returns:
            tuple: A tuple of (q, k, v).
        """
        # If RoPE is fused into the attention OP, do not apply RoPE here.
        if not self.rope_fusion and position_ids is not None:
            q, k, v = self.split_qkv(q, k, v)
            q, k = self.rotary_emb(position_ids, [q, k])
        return q, k, v

    def apply_qk_norm(self, q, k):
        raise NotImplementedError(
            f"QK norm is not implemented for {self.__class__.__name__}."
            "Please override the `apply_qk_norm` method in the subclass.")


@torch.library.custom_op("trtllm::mla_custom_op_inplace",
                         mutates_args=("output", ))
def mla_custom_op_inplace(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
    latent_cache_gen: Optional[torch.Tensor],
) -> None:
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    mla_layer.forward_impl(position_ids,
                           hidden_states,
                           metadata,
                           output=output,
                           latent_cache_gen=latent_cache_gen)


def fp8_block_scaling_bmm_out(
    mat1: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
    mat2_dequant: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sm_version = get_sm_version()
    if sm_version == 90 or sm_version == 89:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            mat1)

        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                   mat1_scale, mat2_scale,
                                                   output)
        out.copy_(output)
    elif sm_version == 120:
        mat1_fp8, mat1_scale = fp8_utils.per_token_quant_and_transform(
            mat1, need_permute102=True)
        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                   mat1_scale, mat2_scale,
                                                   output)
        out.copy_(output)
    elif is_sm_100f(sm_version):
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
        enable_helix_test: bool = False,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
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
            aux_stream (Optional[torch.cuda.Stream]): The auxiliary CUDA stream for running operations in two parallel streams.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
            enable_helix_test (bool): Whether to enable helix unit test.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype

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
        self.enable_helix_test = enable_helix_test
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
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        # only support one kind of sparse attention, dsa now.
        if config is not None and config.sparse_attention_config is not None and config.sparse_attention_config.algorithm == "dsa":
            self.is_dsa = True
        else:
            self.is_dsa = False

        # tensor parallel
        config = config or ModelConfig()
        if mapping_with_cp is not None:
            logger.warning(
                "[MLA::__init__] Overriding mapping with CP detected.")
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
            raise NotImplementedError("MLA doesn't support CP Ulyssees yet")
        if self.mapping.cp_size > 1:
            assert self.mapping.has_cp_helix(
            ), f"CP type must be HELIX for MLA, but got {self.mapping.cp_config['cp_type']}."

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
        self.num_key_value_heads_tp = (self.num_key_value_heads + tp_size -
                                       1) // tp_size

        if self.enable_helix_test:
            rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps",
                                   1e-6)
        else:
            rms_norm_eps = config.pretrained_config.rms_norm_eps
        quant_config = config.get_quant_config()
        self.quant_config = quant_config

        if not self.is_lite:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

            self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank,
                                         eps=rms_norm_eps,
                                         dtype=dtype)

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
                force_dynamic_quantization=config.force_dynamic_quantization)
        else:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

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
                force_dynamic_quantization=config.force_dynamic_quantization)
            self.q_b_proj = self.q_proj

        self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
                                      dtype=dtype,
                                      eps=rms_norm_eps)

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
            force_dynamic_quantization=config.force_dynamic_quantization)
        # This parameter will view into self.kv_b_proj.weight after loading weights.
        # For dummy weight initialization, this parameter is initialized with empty tensor.
        # Used in forward_absorption only
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
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
            force_dynamic_quantization=config.force_dynamic_quantization)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        if not self.is_dsa:
            self.mha = create_attention(
                config.attn_backend,
                self.layer_idx,
                self.num_heads_tp,
                head_dim=self.qk_head_dim,
                num_kv_heads=self.num_key_value_heads_tp,
                pos_embd_params=pos_embd_params,
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
                sparse_attention_config=config.sparse_attention_config,
            )
        else:
            self.mha = None

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads_tp,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            hidden_size=self.hidden_size,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            sparse_attention_config=config.sparse_attention_config,
            dtype=dtype,
            aux_stream=aux_stream,
        )

        self.softmax_scale = 1.0 / (math.sqrt(self.qk_head_dim) * q_scaling)

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

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
        if hasattr(config.pretrained_config, 'llama_4_scaling'):
            self.llama_4_scaling = True
            self.floor_scale = getattr(config.pretrained_config.llama_4_scaling,
                                       'original_max_position_embeddings', 8192)
            self.attn_scale = getattr(config.pretrained_config.llama_4_scaling,
                                      'beta', 0.1)

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        if not self.is_dsa:
            self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)

        # Although we use FP8 MLA for context/generation phase, the output is still in BF16
        self.out_scale = None

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales())

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
            if is_sm_100f():
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp, self.kv_lora_rank,
                         self.qk_nope_head_dim),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp_cp, self.v_head_dim,
                         self.kv_lora_rank),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads_tp, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim:].reshape(
            -1, self.num_heads_tp * self.qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim:] = q_pe.view(-1, self.num_heads_tp,
                                                   self.qk_rope_head_dim)
        return k_pe

    def _attn_forward_gen(self, attn_backend: AttentionBackend, q: torch.Tensor,
                          k: torch.Tensor, v: torch.Tensor,
                          position_ids: Optional[torch.Tensor],
                          attn_metadata: AttentionMetadata, **kwargs):
        if self.mapping.has_cp_helix():
            # partial_o: [num_tokens, num_heads_tp * kv_lora_rank]
            # softmax_stats: [num_tokens, num_heads_tp, 2]
            softmax_stats = torch.empty((q.shape[0], self.num_heads_tp, 2),
                                        device=q.device,
                                        dtype=torch.float32)
            partial_o = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                softmax_stats_tensor=softmax_stats,
                **kwargs,
            )
            kv_lora_rank = partial_o.shape[-1] // self.num_heads_tp
            assert self.kv_lora_rank == kv_lora_rank

            # Switch between NCCL-based and FIFO-based (MNNVL) all-to-all based on cp_config.
            if self.mapping.cp_config.get("use_nccl_for_alltoall", True):
                # NCCL-based implementation using alltoall_helix.
                # This is the post-processing of helix parallel attention,
                # similar to the post-processing of ring attention.
                # Transpose the tensors to make the split across cp_size contiguous
                # For both tensors, we need to split across the second dimension.
                chunks = []
                for t in [partial_o, softmax_stats]:
                    t = t.transpose(1, 0).contiguous()
                    chunks.extend(
                        torch.split(t, t.shape[0] // self.mapping.cp_size))
                gathered = alltoall_helix(chunks, self.mapping.cp_group)
                # Transpose the tensors back to ensure dimensions are ordered correctly.
                # Note: an additional dimension was added at the first index for all-to-all,
                # so the transpose dimensions are shifted by 1.
                gathered = [t.transpose(1, 2).contiguous() for t in gathered]
                return torch.ops.trtllm.helix_post_process(
                    gathered[0], gathered[1], 1.0)
            else:
                # FIFO-based implementation using MNNVL workspace and LL128 Proto.
                # Get or create Helix All-to-All instance.
                helix = HelixAllToAllNative.get(self.mapping)

                # Get dimensions.
                num_tokens = partial_o.shape[0]
                cp_size = self.mapping.cp_size

                # Reshape for FIFO-based all-to-all.
                # partial_o: [num_tokens, num_heads * kv_lora_rank] -> [num_tokens, cp_size, num_heads_tp_cp, kv_lora_rank]
                # softmax_stats: [num_tokens, num_heads, 2] -> [num_tokens, cp_size, num_heads_tp_cp, 2]

                partial_o = partial_o.view(
                    num_tokens, cp_size, self.num_heads_tp_cp,
                    kv_lora_rank).transpose(1, 2).contiguous()
                softmax_stats = softmax_stats.view(num_tokens, cp_size,
                                                   self.num_heads_tp_cp,
                                                   2).transpose(1,
                                                                2).contiguous()

                # Call FIFO-based helixAllToAll.
                partial_o_out, softmax_stats_out = helix.alltoall_native(
                    partial_o, softmax_stats)

                # partial_o_out: [num_tokens, num_heads_tp_cp, cp_size, kv_lora_rank]
                # softmax_stats_out: [num_tokens, num_heads_tp_cp, cp_size, 2]
                # cp_dim = 2 (the dimension where cp_size is located)

                # Call helix_post_process_native with cp_dim=2.
                return torch.ops.trtllm.helix_post_process_native(
                    partial_o_out, softmax_stats_out, 1.0, 2)
        else:
            attn_output = attn_backend.forward(q, k, v, attn_metadata, **kwargs)
            return attn_output

    def create_output(self, hidden_states: torch.Tensor, num_contexts: int):
        num_tokens = hidden_states.shape[0]
        hidden_size = self.o_proj.in_features
        if self.enable_helix_test and num_contexts > 0:
            # note: for testing Helix parallelism, we ensure that the output is
            # large enough for the context phase, but we then cut it again in
            # `forward_context`
            hidden_size *= self.mapping.cp_size
        return hidden_states.new_empty([num_tokens, hidden_size],
                                       dtype=hidden_states.dtype)

    def _attention_scaling(self, q, position_ids):

        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def forward_impl(self,
                     position_ids: Optional[torch.Tensor],
                     hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     output: torch.Tensor,
                     latent_cache_gen: Optional[torch.Tensor] = None) -> None:
        """
        Forward pass for the MLA module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.
            latent_cache_gen (Optional[torch.Tensor]): The latent cache used in generation.

        Returns:
            torch.Tensor: The output tensor.
        """
        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        hidden_states = hidden_states[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        if self.is_lite:
            compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1)
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
            q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(
                hidden_states).split([
                    self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
                ], -1)

            q, compressed_kv = maybe_execute_in_parallel(
                lambda: self.q_a_layernorm(q),
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

        assert q.shape[
            0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            if self.llama_4_scaling:
                q_ctx = self._attention_scaling(
                    q_ctx, position_ids[..., :num_ctx_tokens])

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
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            if self.llama_4_scaling:
                q_gen = self._attention_scaling(
                    q_gen, position_ids[..., num_ctx_tokens:])

            self.forward_absorption_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                position_ids=position_ids,
                latent_cache=latent_cache_gen,
            )

    def forward_impl_with_dsa(self, position_ids: Optional[torch.Tensor],
                              hidden_states: torch.Tensor,
                              attn_metadata: AttentionMetadata,
                              output: torch.Tensor) -> None:
        """
        Forward pass for the MLA module with DSA (always in MQA mode).

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.

        Returns:
            torch.Tensor: The output tensor.
        """
        assert self.mha is None and self.mqa is not None, "DSA is only supported in MQA mode"
        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        hidden_states = hidden_states[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        q, compressed_kv, k_pe, indexer_k = self.kv_a_proj_with_mqa(
            hidden_states).split([
                self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim,
                self.indexer.head_dim
            ], -1)

        # TODO: possibly overlap/fuse q_a_rmsnorm + kv_a_rmsnorm + indexer.k_layernorm?
        q, compressed_kv = maybe_execute_in_parallel(
            lambda: self.q_a_layernorm(q),
            lambda: self.kv_a_layernorm(compressed_kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        qr = q
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

        # TODO: fuse wq_b + (indexer) wlq here
        q = self.q_b_proj(q)
        # Indexer
        topk_indices = self.indexer(
            qr,
            hidden_states,
            attn_metadata,
            position_ids,
            indexer_k=indexer_k,  # indexer K proj
        )

        assert q.shape[
            0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            self.forward_context_dsa(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
                topk_indices=topk_indices[:num_ctx_tokens, :],
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            self.forward_generation_dsa(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                latent_cache_gen,
                topk_indices=topk_indices[num_ctx_tokens:num_tokens, :],
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
        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim
            ],
            -1,
        )

        if self.enable_helix_test:
            # While helix parallelism is mainly meant for generation, we set the
            # helix position offsets for the context phase to get the math right
            # in test_mla_helix.py.
            attn_metadata.helix_position_offsets = position_ids

        k = torch.empty_like(q).view(-1, self.num_heads_tp, self.qk_head_dim)
        maybe_compiled_copy_(
            k[..., :self.qk_nope_head_dim],
            k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim))
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1,
                                                       self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
            out_scale=self.out_scale,
            output=output,
        )

        return attn_output

    def forward_context_dsa(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_sm_version() >= 100:
            return self.forward_absorption_context(q,
                                                   compressed_kv,
                                                   k_pe,
                                                   attn_metadata,
                                                   output,
                                                   latent_cache=latent_cache,
                                                   topk_indices=topk_indices)
        else:
            return self.forward_sparse_mla_kvcache_bf16(q,
                                                        latent_cache,
                                                        attn_metadata,
                                                        output,
                                                        topk_indices,
                                                        is_generation=False)

    def forward_generation_dsa(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_sm_version() >= 100:
            return self.forward_absorption_generation(q,
                                                      compressed_kv,
                                                      k_pe,
                                                      attn_metadata,
                                                      output,
                                                      latent_cache=latent_cache,
                                                      topk_indices=topk_indices)
        else:
            return self.forward_sparse_mla_kvcache_bf16(q,
                                                        latent_cache,
                                                        attn_metadata,
                                                        output,
                                                        topk_indices,
                                                        is_generation=True)

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
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype)
        assert full_compressed_kv.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert full_k_pe.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim
            ],
            -1,
        )

        full_k_nope = full_k_nope.view(-1, self.num_heads_tp,
                                       self.qk_nope_head_dim)
        full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
        full_k = maybe_compiled_cat(
            (full_k_nope, full_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1)
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
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=self.out_scale,
            output=output,
        )

        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.
        Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [toal_token_q, num_heads, 2] -> [toal_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device='cuda',
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device='cuda',
        )

        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads_tp * self.v_head_dim), dtype=q.dtype)

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime
        origin_ctx_total_kv_len = attn_metadata.host_total_kv_lens[0]

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts]
            chunked_global_offset = attn_metadata.chunked_global_offset[
                loop_idx]
            chunked_max_seq_len = attn_metadata.max_chunk_len_per_loop[loop_idx]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                chunked_global_offset=chunked_global_offset,
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype)

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)
            chunked_k_nope, chunked_v = chunked_kv.split(
                [
                    self.num_heads_tp * self.qk_nope_head_dim,
                    self.num_heads_tp * self.v_head_dim
                ],
                -1,
            )

            chunked_k_nope = chunked_k_nope.view(-1, self.num_heads_tp,
                                                 self.qk_nope_head_dim)
            chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
            chunked_k = maybe_compiled_cat(
                (chunked_k_nope, chunked_k_pe.expand(-1, self.num_heads_tp,
                                                     -1)),
                dim=-1)
            chunked_k = chunked_k.view(-1, self.num_heads_tp * self.qk_head_dim)

            # release pytorch activation memory
            chunked_compressed_kv = None
            chunked_k_pe = None
            chunked_kv = None
            chunked_k_nope = None

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[
                loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[
                loop_idx]
            attn_metadata.host_total_kv_lens[0] = total_ctx_chunked_tokens

            # do not apply mask for attention within loop
            # latent_cache must be None to differentiate from normal context phase,
            # so that we can skip applying RoPE and appending KV cache inside attention op
            temp_attn_output = self.mha.forward(
                q,
                chunked_k,
                chunked_v,
                attn_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                attention_mask=PredefinedAttentionMask.FULL,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                chunked_prefill_buffer_batch_size=attn_metadata.
                runtime_features.chunked_prefill_buffer_batch_size,
                output=temp_attn_output,
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output, temp_attn_output, self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor, temp_merge_op, attn_metadata)

        # deal with the uncached kv
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([
            -1, self.kv_lora_rank + self.qk_rope_head_dim
        ]).split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        # final round of attention

        k_nope, v = kv.split(
            [
                self.num_heads_tp * self.qk_nope_head_dim,
                self.num_heads_tp * self.v_head_dim
            ],
            -1,
        )

        k_nope = k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = maybe_compiled_cat((k_nope, k_pe.expand(-1, self.num_heads_tp, -1)),
                               dim=-1)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[
            0] = attn_metadata.prompt_lens_cpu_runtime[:attn_metadata.
                                                       num_contexts].sum().item(
                                                       )

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        temp_attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=self.out_scale,
            softmax_stats_tensor=self.temp_softmax_stats_tensor,
            chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.
            chunked_prefill_buffer_batch_size,
            output=temp_attn_output,
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(attn_output, temp_attn_output,
                                                 self.softmax_stats_tensor,
                                                 self.temp_softmax_stats_tensor,
                                                 temp_merge_op, attn_metadata)
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = origin_ctx_total_kv_len

        return attn_output

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
            if trtllm_attention.is_chunked_prefill_for_mla_context(
                    attn_metadata):
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output)
            elif trtllm_attention.has_cached_kv_for_mla_context(attn_metadata):
                return self.forward_context_with_cached_kv(
                    q, latent_cache, attn_metadata, output)
        return self.forward_context_default(q, compressed_kv, k_pe,
                                            position_ids, attn_metadata, output,
                                            latent_cache)

    def forward_absorption_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        num_seqs = attn_metadata.kv_lens_cuda_runtime.size(0)

        cu_q_seqlens = torch.empty(num_seqs + 1,
                                   dtype=torch.int32,
                                   device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1,
                                    dtype=torch.int32,
                                    device=q.device)
        fmha_scheduler_counter = torch.empty(1,
                                             dtype=torch.uint32,
                                             device=q.device)
        has_fp8_kv_cache = self.mqa.has_fp8_kv_cache if hasattr(
            self.mqa, 'has_fp8_kv_cache') else False

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if has_fp8_kv_cache:
            mla_bmm1_scale = torch.empty(2,
                                         dtype=torch.float32,
                                         device=q.device)
            mla_bmm2_scale = torch.empty(1,
                                         dtype=torch.float32,
                                         device=q.device)
            quant_q_buffer = torch.empty(
                num_tokens,
                self.num_heads_tp, (self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.uint8,
                device=q.device)

        fused_q = torch.empty(
            [
                num_tokens, self.num_heads_tp,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        rope_stream = self.aux_stream if not has_fp8_kv_cache else None
        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            maybe_execute_in_parallel(
                lambda: torch.ops.trtllm.bmm_out(
                    q_nope_t, self.k_b_proj_trans.transpose(1, 2), q_nope_out),
                lambda: self.mqa.mla_rope_generation(
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
                ),
                self.ln_events[0],
                self.ln_events[1],
                rope_stream,
            )

        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            maybe_execute_in_parallel(
                lambda: fp8_block_scaling_bmm_out(
                    q_nope,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans_scale,
                    q_nope_out,
                    self.k_b_proj_trans_dequant,
                ),
                lambda: self.mqa.mla_rope_generation(
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
                ),
                self.ln_events[0],
                self.ln_events[1],
                rope_stream,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        fused_q = fused_q.view([
            num_tokens,
            self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        # Use generation_only for generation phase and context_only for context phase in DSA attention
        attention_input_type = AttentionInputType.generation_only

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=attention_input_type,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            topk_indices=topk_indices,  # used by DSA attention
            is_generation=True,  # used by DSA attention
            cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
            cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
            fmha_scheduler_counter=
            fmha_scheduler_counter,  # used by `mlaGeneration`
            mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
            mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
            quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (attn_out_latent.shape[0] == q.shape[0]
                and attn_out_latent.shape[1]
                == self.num_heads_tp_cp * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view(
            [num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

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
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [
                num_tokens, self.num_heads_tp,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            torch.ops.trtllm.bmm_out(q_nope_t,
                                     self.k_b_proj_trans.transpose(1, 2),
                                     q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            fp8_block_scaling_bmm_out(
                q_nope,
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                q_nope_out,
                self.k_b_proj_trans_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        if self.apply_rotary_emb:
            fused_q[..., self.kv_lora_rank:] = q_pe
        fused_q = fused_q.view([
            num_tokens,
            self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        # Use generation_only for generation phase and context_only for context phase in DSA attention
        attention_input_type = AttentionInputType.context_only
        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=attention_input_type,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            topk_indices=topk_indices,  # used by DSA attention
            is_generation=False,  # used by DSA attention
        )
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (attn_out_latent.shape[0] == q.shape[0]
                and attn_out_latent.shape[1]
                == self.num_heads_tp_cp * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view(
            [num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    @nvtx_range("forward_sparse_mla_kvcache_bf16")
    def forward_sparse_mla_kvcache_bf16(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: DSAtrtllmAttentionMetadata,
        output: torch.Tensor,
        topk_indices: torch.Tensor,
        is_generation: bool = False,
    ) -> torch.Tensor:
        """
        Forward sparse MLA (DSA) for BF16 KV cache for both context and generation phases using FlashMLA kernels

        To form the input for FlashMLA kernel and adapt our KV cache manager, we need to:
        1. Append current tokens to paged cache and apply rope to q/k via mla_rope_append_paged_kv_assign_q
        2. Load full kv cache from paged memory (with k rope applied)
        3. Call FlashMLA sparse attention kernel for sparse prefill/decode
        """
        assert isinstance(attn_metadata, DSAtrtllmAttentionMetadata), \
            "DSA requires DSAtrtllmAttentionMetadata"
        # Append current tokens to paged cache and apply RoPE to q
        # This writes latent_cache to paged KV and modifies q in-place
        trtllm_attention = self.mqa
        with nvtx_range_debug(
                f"mla_rope_append_paged_kv_assign_q_is_generation={is_generation}"
        ):
            trtllm_attention.mla_rope_append_paged_kv_assign_q(
                q, latent_cache, attn_metadata, is_generation=is_generation)

        num_tokens = q.shape[0]
        q_nope, q_rope = q.view(-1, self.num_heads_tp, self.qk_head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope_out = torch.empty(
            [num_tokens, self.num_heads_tp, (self.kv_lora_rank)],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            torch.ops.trtllm.bmm_out(q_nope_t,
                                     self.k_b_proj_trans.transpose(1, 2),
                                     q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

            fp8_block_scaling_bmm_out(
                q_nope,
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                q_nope_out,
                self.k_b_proj_trans_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        q_nope_out = q_nope_out.transpose(0, 1)
        q_concat = torch.cat([q_nope_out, q_rope], dim=-1)

        sm_version = get_sm_version()
        # FlashMLA sparse kernel (bf16) requires num_heads=128 on sm100 or multiple of 64 on sm90
        if sm_version >= 100:
            padding = 128
            assert self.num_heads_tp <= padding, (
                f"SM100 FlashMLA sparse kernel requires exactly {padding} heads, "
                f"got {self.num_heads_tp}. Padding from values > {padding} is not supported."
            )
        else:  # SM90
            padding = ((self.num_heads_tp + 63) // 64) * 64  # multiple of 64

        if self.num_heads_tp != padding:
            logger.warning_once(
                f"Padding num_heads from {self.num_heads_tp} to {padding} "
                f"due to FlashMLA sparse attention kernel requirement",
                key="sparse_mla_padding_warning")

            # Create padded tensor with zeros for extra heads
            q_padded = q_concat.new_empty(
                (num_tokens, padding, q_concat.shape[2]))
            q_padded[:, :self.num_heads_tp, :] = q_concat
            q_concat = q_padded

        # Convert indices and return all-layer KV pool
        # Note: underlying pool is layer-interleaved: [num_blocks, num_layers, kv_factor, tokens_per_block, num_kv_heads, head_dim]
        # to avoid reshape(copy) per-layer KV cache, we return all-layer KV pool w/ topk indices adjusted by stride_factor=num_layers*tokens_per_block
        topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
            topk_indices,
            attn_metadata,
            layer_idx=self.layer_idx,
            is_generation=is_generation,
        )
        topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
        if flash_mla_sparse_fwd is not None:
            attn_out_latent = flash_mla_sparse_fwd(q_concat, kv_cache_pool,
                                                   topk_indices_pool,
                                                   self.softmax_scale)[0]
        else:
            raise RuntimeError(
                "flash_mla_sparse_fwd not available. Please ensure FlashMLA module is built."
            )

        # [seq, num_heads, kv_lora_rank], account for padding
        attn_out_latent = attn_out_latent[:, :self.num_heads_tp, :]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads_tp, self.kv_lora_rank])
        if self.num_heads_tp != padding:
            attn_out_latent = attn_out_latent.contiguous()

        assert (attn_out_latent.shape[0] == q.shape[0]
                and attn_out_latent.shape[1] == self.num_heads_tp)

        attn_output = output.view(
            [num_tokens, self.num_heads_tp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")
        return output

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        attn_output = self.create_output(hidden_states,
                                         attn_metadata.num_contexts)
        if self.is_dsa:
            self.forward_impl_with_dsa(position_ids,
                                       hidden_states,
                                       attn_metadata,
                                       output=attn_output)
        elif self.register_to_config:
            torch.ops.trtllm.mla_custom_op_inplace(hidden_states, position_ids,
                                                   self.layer_idx_str,
                                                   attn_output,
                                                   latent_cache_gen)
        else:
            self.forward_impl(position_ids,
                              hidden_states,
                              attn_metadata,
                              output=attn_output,
                              latent_cache_gen=latent_cache_gen)

        if self.enable_helix_test and self.mapping.has_cp_helix():
            # note: for allowing testing Helix parallelism, we ensure that
            # the output is compatible with o_proj even in the context phase,
            # thus we cut it to num_heads_tp_cp * v_head_dim
            attn_output = attn_output[:, :self.num_heads_tp_cp *
                                      self.v_head_dim].contiguous()

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output

    def resmooth_parameters(self,
                            module_weight,
                            module_weight_scale,
                            recipe=(1, 128, 128)):
        weight, weight_scale = fp8_utils.resmooth_to_fp8_e8m0(
            module_weight, module_weight_scale)

        transfromed_scale = fp8_utils.transform_sf_into_required_layout(
            weight_scale,
            mn=weight.shape[1],
            k=weight.shape[2],
            recipe=recipe,
            num_groups=weight.shape[0],
            is_sfa=False)

        weight_param = torch.nn.Parameter(weight, requires_grad=False)
        scale_param = torch.nn.Parameter(transfromed_scale, requires_grad=False)

        return weight_param, scale_param

    def post_load_weights(self):
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales())
        is_sm120 = get_sm_version() == 120
        if is_sm120 and has_fp8_block_scales:
            self.k_b_proj_trans, self.k_b_proj_trans_scale = self.resmooth_parameters(
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                recipe=(1, 128, 128))

            self.v_b_proj, self.v_b_proj_scale = self.resmooth_parameters(
                self.v_b_proj, self.v_b_proj_scale, recipe=(1, 128, 128))
