# Adapted from: https://github.com/thu-ml/SageAttention
# @inproceedings{
#   title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration},
#   author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
#   booktitle={International Conference on Learning Representations (ICLR)},
#   year={2025}
# }
#
# Adapted from: https://github.com/svg-project/Sparse-VideoGen/tree/main
# @article{xi2025sparse,
#   title={Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity},
#   author={Xi, Haocheng and Yang, Shuo and Zhao, Yilong and Xu, Chenfeng and Li, Muyang and Li, Xiuyu and Lin, Yujun and Cai, Han and Zhang, Jintao and Li, Dacheng and others},
#   journal={arXiv preprint arXiv:2502.01776},
#   year={2025}
# }
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from visual_gen.configs.op_manager import AttentionOpManager, SparseVideogenConfig, SparseVideogenConfig2
from visual_gen.configs.parallel import get_dit_parallel_config
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.utils.auto_tuner import TunableParam
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from sageattention import sageattn
except ImportError:
    sageattn = None
    logger.warning("SageAttention is not installed.")

try:
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None
    logger.warning("FlashAttn3 is not installed.")

try:
    from flash_attn.cute.interface import _flash_attn_fwd
except ImportError:
    _flash_attn_fwd = None
    logger.warning("FlashAttn4 is not installed.")

try:
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
except ImportError:
    get_attention_backend = None
    logger.warning("TensorRT-LLM is not installed.")

try:
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import DotProductAttention, fp8_autocast
except ImportError:
    DotProductAttention = None
    logger.warning("TransformerEngine is not installed.")

try:
    import flashinfer_vx
except ImportError:
    flashinfer_vx = None
    logger.warning("SageAttention for Blackwell not available")


class BaseAttn:
    def __init__(self):
        pass

    def _convert_qkv_layout(self, q, k, v, src_layout, dst_layout):
        if src_layout == "HND" and dst_layout == "NHD":
            # [B, H, S, D] -> [B, S, H, D]
            q = q.permute(0, 2, 1, 3).contiguous()
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()
        elif src_layout == "NHD" and dst_layout == "HND":
            # [B, S, H, D] -> [B, H, S, D]
            q = q.permute(0, 2, 1, 3).contiguous()
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()
        else:
            raise NotImplementedError(f"Unsupported tensor layout conversion: {src_layout} -> {dst_layout}")
        return q, k, v

    def _convert_output_layout(self, out, src_layout, dst_layout):
        if src_layout == "HND" and dst_layout == "NHD":
            # [B, S, H, D] -> [B, H, S, D]
            out = out.permute(0, 2, 1, 3).contiguous()
        elif src_layout == "NHD" and dst_layout == "HND":
            # [B, H, S, D] -> [B, S, H, D]
            out = out.permute(0, 2, 1, 3).contiguous()
        else:
            raise NotImplementedError(f"Unsupported tensor layout conversion: {src_layout} -> {dst_layout}")
        return out

    def register_tunable_params(
        self, value: Any, param_range: Optional[List[Any]], name: Optional[str], description: Optional[str]
    ):
        return TunableParam(value, param_range, name, description)

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        valid_query_length=None,
        valid_kv_length=None,
    ):
        raise NotImplementedError("BaseAttn is not implemented, please implement it in the subclass")


@AttentionOpManager.register_attn("default")
class DefaultAttn(BaseAttn):
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        """
        Default attention is implemented with `F.scaled_dot_product_attention`

        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
            attn_mask (optional Tensor): Attention mask; shape :math:`(N, ..., L, S)`. Two types of masks are supported.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If true, assumes upper left causal attention masking and errors if both attn_mask and is_causal are set.
            scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set
                to 1/sqrt(E)


        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.

        Shape legend:
            - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
            - :math:`S: \text{Source sequence length}`
            - :math:`L: \text{Target sequence length}`
            - :math:`E: \text{Embedding dimension of the query and key}`
            - :math:`Ev: \text{Embedding dimension of the value}`
        """
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError("Tensor layout not supported for DefaultAttn")
        if return_lse:
            raise NotImplementedError("Return LSE is not supported for DefaultAttn")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for DefaultAttn")

        if get_dit_parallel_config().ring_size() > 1:
            raise NotImplementedError("Default attention not implemented for ring parallel")
        logger.debug("Using DefaultAttn")
        if tensor_layout == "NHD":
            query, key, value = self._convert_qkv_layout(query, key, value, src_layout="NHD", dst_layout="HND")

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

        if tensor_layout == "NHD":
            output = self._convert_output_layout(output, src_layout="HND", dst_layout="NHD")
        return output


@AttentionOpManager.register_attn("sage-attn")
class SageAttn(BaseAttn):
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        logger.debug("Using SageAttn")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError(f"Tensor layout {tensor_layout} not supported for SageAttn")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for Sage attention")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for Sage attention")
        if enable_gqa:
            raise NotImplementedError("GQA is not supported for Sage attention")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for SageAttn")

        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
            logger.debug(f"SageAttn: query.dtype: {query.dtype}, converting to bfloat16")
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        if query.dtype == torch.float16 or query.dtype == torch.bfloat16:
            assert query.shape[-1] % 8 == 0, "hidden size per attention head should be multiple of 8"
        elif query.dtype == torch.float8_e4m3fn or query.dtype == torch.float8_e5m2:
            assert query.shape[-1] % 16 == 0, "hidden size per attention head should be multiple of 16"

        if sageattn is None:
            raise ImportError("SageAttention is not installed")

        output = sageattn(
            query,
            key,
            value,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=scale,
            enable_gqa=False,
            return_lse=return_lse,
        )

        if not return_lse and output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        elif return_lse:
            output = list(output)
            if output[0].dtype != origin_dtype:
                output[0] = output[0].to(origin_dtype)
            if tensor_layout == "NHD":
                # lse always in [B, H, S], permute it to [B, S, H]
                output[1] = output[1].permute(0, 2, 1).contiguous()  # [B, S, H] -> [B, H, S]
            output = tuple(output)

        return output


@AttentionOpManager.register_attn("sparse-videogen")
class SparseVideoGenAttn(BaseAttn):
    def __init__(self):
        try:
            from svg.models.wan.attention import WanAttn_SVGAttn_Processor2_0
        except ImportError:
            raise ImportError("Sparse VideoGen is not installed.")
        if PipelineConfig.transformer_type == "ditWanTransformer3DModel":
            SVG_ATTN_CLASS = WanAttn_SVGAttn_Processor2_0
        else:
            raise NotImplementedError("Sparse VideoGen attention not implemented for this model")
        self.svg_attn = SVG_ATTN_CLASS(layer_idx=0)  # layer_idx is not used in our implementation
        self.svg_attn.num_sampled_rows = SparseVideogenConfig.num_sampled_rows()
        self.svg_attn.sample_mse_max_row = SparseVideogenConfig.sample_mse_max_row()
        self.svg_attn.attention_masks = SparseVideogenConfig.attention_masks()
        self.svg_attn.context_length = SparseVideogenConfig.context_length()
        self.svg_attn.num_frame = SparseVideogenConfig.num_frame()
        self.svg_attn.frame_size = SparseVideogenConfig.frame_size()
        self.svg_attn.block_mask = SparseVideogenConfig.block_mask()

        # when to fallback to high precision attention is managed by ditAttnProcessor, so these values are unused
        self.svg_attn.first_layers_fp = 0.0  # not used in our implementation
        self.svg_attn.first_times_fp = float("-inf")  # not used in our implementation
        self.svg_attn.num_layers = 0  # not used in our implementation

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        logger.debug("Using SparseVideoGenAttn")
        if return_lse:
            raise NotImplementedError("Return LSE is not supported for SparseVideoGenAttn")
        if get_dit_parallel_config().ring_size() > 1:
            # todo: we can set `return_lse=True` to get the log-sum-exp value and then compute for ring parallel
            raise NotImplementedError("Sparse VideoGen attention not implemented for ring parallel")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for Sparse VideoGen attention")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for Sparse VideoGen attention")
        if enable_gqa:
            raise NotImplementedError("GQA is not supported for Sparse VideoGen attention")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError(f"Tensor layout {tensor_layout} not supported for SparseVideoGenAttn")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for SparseVideoGenAttn")

        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16]:
            logger.debug(f"SparseVideoGenAttn: query.dtype: {query.dtype}, converting to bfloat16")
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        if tensor_layout == "NHD":
            query, key, value = self._convert_qkv_layout(query, key, value, src_layout="NHD", dst_layout="HND")

        cfg, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = (
            self.svg_attn.context_length,
            self.svg_attn.num_frame,
            self.svg_attn.frame_size,
        )

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        sampled_mses = self.svg_attn.sample_mse(query, key, value)
        best_mask_idx = torch.argmin(sampled_mses, dim=0)

        output_hidden_states = torch.zeros_like(query)
        query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

        query_out, key_out, value_out = self.svg_attn.fast_sparse_head_placement(
            query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
        )

        hidden_states = self.svg_attn.sparse_flex_attention(
            query_out, key_out, value_out, block_mask=self.svg_attn.block_mask
        )

        self.svg_attn.fast_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

        output = output_hidden_states.reshape(cfg, num_heads, seq_len, dim).contiguous()

        if tensor_layout == "NHD":
            output = self._convert_output_layout(output, src_layout="HND", dst_layout="NHD")

        output = output.to(origin_dtype)

        return output


@AttentionOpManager.register_attn("sparse-videogen2")
class SparseVideoGenAttn2(BaseAttn):
    def __init__(self):
        try:
            from svg.models.wan.attention import WanAttn_SAPAttn_Processor
        except ImportError:
            raise ImportError("Sparse VideoGen2 is not installed.")
        if PipelineConfig.transformer_type == "ditWanTransformer3DModel":
            SAP_ATTN_CLASS = WanAttn_SAPAttn_Processor
        else:
            raise NotImplementedError("Sparse VideoGen2 attention not implemented for this model")

        self.sap_attn = SAP_ATTN_CLASS(layer_idx=0)  # layer_idx is not used in our implementation
        self.sap_attn.num_q_centroids = SparseVideogenConfig2.num_q_centroids()
        self.sap_attn.num_k_centroids = SparseVideogenConfig2.num_k_centroids()
        self.sap_attn.top_p_kmeans = SparseVideogenConfig2.top_p_kmeans()
        self.sap_attn.min_kc_ratio = SparseVideogenConfig2.min_kc_ratio()
        self.sap_attn.kmeans_iter_init = SparseVideogenConfig2.kmeans_iter_init()
        self.sap_attn.kmeans_iter_step = SparseVideogenConfig2.kmeans_iter_step()
        self.sap_attn.logging_file = SparseVideogenConfig2.logging_file()
        self.sap_attn.zero_step_kmeans_init = SparseVideogenConfig2.zero_step_kmeans_init()
        self.sap_attn.context_length = SparseVideogenConfig2.context_length()
        self.sap_attn.num_frame = SparseVideogenConfig2.num_frame()
        self.sap_attn.frame_size = SparseVideogenConfig2.frame_size()

        # when to fallback to high precision attention is managed by ditAttnProcessor, so these values are unused
        self.sap_attn.first_layers_fp = 0.0  # not used in our implementation
        self.sap_attn.first_times_fp = float("-inf")  # not used in our implementation
        self.sap_attn.num_layers = 0  # not used in our implementation

    @torch.compiler.disable
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
    ):
        logger.debug("Using SparseVideoGen2Attn")
        if return_lse:
            raise NotImplementedError("Return LSE is not supported for SparseVideoGen2Attn")
        if get_dit_parallel_config().ring_size() > 1:
            # todo: we can set `return_lse=True` to get the log-sum-exp value and then compute for ring parallel
            raise NotImplementedError("Sparse VideoGen2 attention not implemented for ring parallel")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for Sparse VideoGen2 attention")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for Sparse VideoGen2 attention")
        if enable_gqa:
            raise NotImplementedError("GQA is not supported for Sparse VideoGen2 attention")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError(f"Tensor layout {tensor_layout} not supported for SparseVideoGen2Attn")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for SparseVideoGen2Attn")

        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16]:
            logger.debug(f"SparseVideoGenAttn: query.dtype: {query.dtype}, converting to bfloat16")
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        if tensor_layout == "NHD":
            query, key, value = self._convert_qkv_layout(query, key, value, src_layout="NHD", dst_layout="HND")

        cfg, num_heads, seq_len, dim = query.size()
        assert cfg == 1, "Batch size must be 1 for kmeans block sparse attention"

        context_length, num_frame, frame_size = (
            self.sap_attn.context_length,
            self.sap_attn.num_frame,
            self.sap_attn.frame_size,
        )

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.sap_attn.semantic_aware_permutation(
            query, key, value
        )

        from svg.kernels.triton.permute import apply_inverse_permutation_triton
        from svg.kmeans_utils import dynamic_block_sparse_fwd_flashinfer

        output_permuted = dynamic_block_sparse_fwd_flashinfer(
            q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
        )

        attn_output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)

        output = attn_output.reshape(cfg, num_heads, seq_len, dim).contiguous()

        if tensor_layout == "NHD":
            output = self._convert_output_layout(output, src_layout="HND", dst_layout="NHD")

        output = output.to(origin_dtype)

        return output


@AttentionOpManager.register_attn("trtllm-attn")
class TRTLLMAttn(BaseAttn):
    def _lazy_init(self):
        AttentionCls = get_attention_backend(self.attn_backend)

        assert self.num_heads is not None, "num_heads must be set"
        assert self.head_dim is not None, "head_dim must be set"
        assert self.num_key_value_heads is not None, "num_key_value_heads must be set"

        self.attn = AttentionCls(
            layer_idx=self.layer_idx,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_key_value_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.support_fused_qkv = self.attn.support_fused_qkv()

    def _set_attention_metadata(self, batch_size, seq_len):
        if get_attention_backend is None:
            raise ImportError("TensorRT-LLM is not installed")

        AttentionCls = get_attention_backend(self.attn_backend)
        if self.attn_backend == "trtllm":
            self.attention_metadata = AttentionCls.Metadata(
                # AttentionMetadata
                max_num_requests=batch_size,
                max_num_tokens=batch_size * seq_len,
                kv_cache_manager=None,
                mapping=None,
                runtime_features=None,
            )
            sequence_lengths = [seq_len] * batch_size
            self.attention_metadata.seq_lens = torch.tensor(sequence_lengths, dtype=torch.int)
            self.attention_metadata.num_contexts = batch_size
            self.attention_metadata.request_ids = torch.tensor(range(batch_size), dtype=torch.int)
            self.attention_metadata.max_seq_len = seq_len
            self.attention_metadata.prepare()
        else:
            raise NotImplementedError(f"Attention backend {self.attn_backend} is not supported")

    def split_qkv(self, q, k=None, v=None):
        if k is None and v is None:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    @torch.compile
    def convert_qkv(self, q, k, v, tensor_layout):
        if k is None and v is None and not self.support_fused_qkv:
            if tensor_layout == "HND":
                # [B, H, S, D] -> [B, S, H, D]
                q = q.permute(0, 2, 1, 3)
            q = q.reshape(-1, self.num_heads * self.head_dim).contiguous()
            q, k, v = self.split_qkv(q)
        elif k is not None and v is not None and self.support_fused_qkv:
            if tensor_layout == "HND":
                # [B, H, S, D] -> [B, S, H, D]
                q, k, v = self._convert_qkv_layout(q, k, v, src_layout="HND", dst_layout="NHD")
            q = q.reshape(-1, self.num_heads * self.head_dim).contiguous()
            k = k.reshape(-1, self.num_key_value_heads * self.head_dim).contiguous()
            v = v.reshape(-1, self.num_key_value_heads * self.head_dim).contiguous()
            qkv = torch.concat([q, k, v], dim=-1).contiguous()
            q, k, v = qkv, None, None
        return q, k, v

    @torch.compile
    def convert_output(self, out, batch_size, tensor_layout):
        out = out.reshape(batch_size, -1, self.num_heads, self.head_dim).contiguous()
        if tensor_layout == "HND":
            out = self._convert_output_layout(out, src_layout="NHD", dst_layout="HND")
        return out

    @torch.compiler.disable
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        if return_lse:
            raise NotImplementedError("Return LSE is not supported for TRTLLMAttn")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError(f"Tensor layout {tensor_layout} not supported for TRTLLMAttn")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for TRTLLMAttn")

        if tensor_layout == "NHD":
            # input shape: [B, S, H, D]
            batch_size = query.size(0)
            seq_len = query.size(1)
        else:
            # input shape: [B, H, S, D]
            batch_size = query.size(0)
            seq_len = query.size(2)
        if not hasattr(self, "attn"):
            self.attn_backend = "trtllm"  # todo: support other backends
            self.layer_idx = PipelineConfig.current_dit_block_id
            if tensor_layout == "NHD":
                _, _, self.num_heads, self.head_dim = query.shape
            else:
                _, self.num_heads, _, self.head_dim = query.shape
            self.num_key_value_heads = self.num_heads  # the number of key/value heads
            self.pos_embd_params = None
            self.rope_fusion = False
            if scale is None:
                self.q_scaling = 1.0 / math.sqrt(self.head_dim)
            else:
                self.q_scaling = scale
            self._lazy_init()
        self._set_attention_metadata(batch_size=batch_size, seq_len=seq_len)

        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for TRTLLMAttn")
        else:
            from tensorrt_llm._torch.modules.attention import PredefinedAttentionMask

            if is_causal:
                attention_mask = PredefinedAttentionMask.CAUSAL
            else:
                attention_mask = PredefinedAttentionMask.FULL
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for TRTLLMAttn")
        if enable_gqa:
            raise NotImplementedError("GQA is not supported for TRTLLMAttn")

        q, k, v = self.convert_qkv(query, key, value, tensor_layout=tensor_layout)
        attn_output = self.attn.forward(
            q,
            k,
            v,
            metadata=self.attention_metadata,
            attention_mask=attention_mask,
        )
        attn_output = self.convert_output(attn_output, batch_size=batch_size, tensor_layout=tensor_layout)
        return attn_output


@AttentionOpManager.register_attn("flash-attn3")
class FlashAttn3(BaseAttn):
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
    ):

        if flash_attn_interface is None:
            raise ImportError("FlashAttn3 is not installed")

        logger.debug("Using FlashAttn3")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for FlashAttn3")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for FlashAttn3")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError("Tensor layout not supported for FlashAttn3")

        if tensor_layout == "HND":
            query, key, value = self._convert_qkv_layout(query, key, value, src_layout="HND", dst_layout="NHD")

        # FA3 only supports float16 and bfloat16
        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16]:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        if cu_seqlens_q is None:
            output = flash_attn_interface.flash_attn_func(
                q=query,
                k=key,
                v=value,
                softmax_scale=scale,
                causal=is_causal,
                qv=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                deterministic=False,
                sm_margin=0,
                return_attn_probs=return_lse,
            )
        else:
            query = torch.squeeze(query, dim=0)    
            key = torch.squeeze(key, dim=0)
            value = torch.squeeze(value, dim=0)
            output = flash_attn_interface.flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqused_q=None,
                seqused_k=None,
                softmax_scale=scale,
                causal=is_causal,
                qv=None,
                q_descale=None, k_descale=None, v_descale=None,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                deterministic=False,
                sm_margin=0,
                return_attn_probs=False,
            )
            output = torch.unsqueeze(output, dim=0)

        lse = None
        if isinstance(output, tuple):
            lse = output[1]
            output = output[0]

        if tensor_layout == "HND":
            output = self._convert_output_layout(output, src_layout="NHD", dst_layout="HND")

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
 
        if return_lse:
            assert lse is not None, "lse is not returned by FlashAttn3"
            return output, lse
        else:
            return output


@AttentionOpManager.register_attn("flash-attn3-fp8")
class FlashAttn3FP8(BaseAttn):
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        if flash_attn_interface is None:
            raise ImportError("FlashAttn3 is not installed")

        logger.debug("Using FlashAttn3")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for FlashAttn3")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for FlashAttn3")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError("Tensor layout not supported for FlashAttn3")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for FlashAttn3FP8")

        if tensor_layout == "HND":
            query, key, value = self._convert_qkv_layout(query, key, value, src_layout="HND", dst_layout="NHD")

        query, key, value = query.to(torch.float8_e4m3fn), key.to(torch.float8_e4m3fn), value.to(torch.float8_e4m3fn)
        output = flash_attn_interface.flash_attn_func(
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            qv=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            deterministic=False,
            sm_margin=0,
            return_attn_probs=return_lse,
        )

        lse = None
        if isinstance(output, tuple):
            lse = output[1]
            output = output[0]

        if tensor_layout == "HND":
            output = self._convert_output_layout(output, src_layout="NHD", dst_layout="HND")

        if return_lse:
            assert lse is not None, "lse is not returned by FlashAttn3"
            return output, lse
        else:
            return output


@AttentionOpManager.register_attn("flash-attn4")
class FlashAttn4(BaseAttn):
    @torch.compiler.disable
    def _flash_attn_fwd(self, *args, **kwargs):
        return _flash_attn_fwd(*args, **kwargs)

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        if _flash_attn_fwd is None:
            raise ImportError("FlashAttn4 is not installed")

        logger.debug("Using FlashAttn4")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for FlashAttn4")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for FlashAttn4")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError("Tensor layout not supported for FlashAttn4")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for FlashAttn4")

        if tensor_layout == "HND":
            query, key, value = self._convert_qkv_layout(query, key, value, src_layout="HND", dst_layout="NHD")

        # FA4 only supports float16 and bfloat16
        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16]:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        output, lse = self._flash_attn_fwd(
            query,
            key,
            value,
            softmax_scale=None,
            causal=is_causal,
            window_size_left=None,
            window_size_right=None,
            learnable_sink=None,
            softcap=0.0,
            pack_gqa=None,
            mask_mod=None,
            block_sparse_tensors=None,
            return_lse=True,
        )

        if tensor_layout == "HND":
            output = self._convert_output_layout(output, src_layout="NHD", dst_layout="HND")

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)

        if return_lse:
            if tensor_layout == "NHD":
                lse = lse.permute(0, 2, 1)
            return output, lse
        else:
            return output


@AttentionOpManager.register_attn("te")
class TransformerEngineAttn(BaseAttn):
    def __init__(self):
        self.traits = (0, 0, None, "no_mask", None)
        self.attn_op = None
        self.enable_fp8 = False
        self.recipe = DelayedScaling()

    def _enable_fp8(self):
        # This method will be called by its derived class TransformerEngineAttnFP8
        self.enable_fp8 = True
        self.recipe = DelayedScaling(fp8_dpa=True, fp8_mha=True)

    def _lazy_init(
        self, num_attention_heads, kv_channels, num_gqa_groups, attn_mask_type, softmax_scale, warn_once_hnd
    ):
        if DotProductAttention is None:
            raise ImportError("TransformerEngine is not installed")

        if (num_attention_heads, kv_channels, num_gqa_groups, attn_mask_type, softmax_scale) != self.traits:
            if warn_once_hnd:
                logger.warning("Potential performance loss: bhsd->bshd transposition will happen at attn op.")
            self.attn_op = DotProductAttention(
                num_attention_heads,
                kv_channels,
                num_gqa_groups=num_gqa_groups,
                attn_mask_type=attn_mask_type,
                softmax_scale=softmax_scale,
                qkv_format="bshd",
            )

    @torch.compiler.disable
    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported by TransformerEngines")
        if return_lse:
            raise NotImplementedError("TransformerEngine does not support returning LSE")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for TransformerEngineAttn")

        if tensor_layout.upper() == "HND":
            warn_once_hnd = True
            # TE only supports sbhd / bshd. h before s always needs transposing
            query, key, value = map(lambda t: t.transpose(1, 2).contiguous(), (query, key, value))
            if attn_mask is not None and len(attn_mask.shape) == 4:
                attn_mask = attn_mask.transpose(1, 2).contiguous()
        else:
            warn_once_hnd = False
        num_attention_heads = query.shape[-2]
        kv_channels = value.shape[-1]
        if enable_gqa:
            num_gqa_groups = key.shape[-2]
        else:
            num_gqa_groups = None
        if is_causal:
            attn_mask_type = "causal"
        elif attn_mask is None:
            attn_mask_type = "no_mask"
        else:
            attn_mask_type = "arbitrary"
        self._lazy_init(num_attention_heads, kv_channels, num_gqa_groups, attn_mask_type, scale, warn_once_hnd)

        with fp8_autocast(enabled=self.enable_fp8, fp8_recipe=self.recipe):
            out = self.attn_op(query, key, value, attention_mask=attn_mask).unflatten(-1, (-1, kv_channels))
        if tensor_layout.upper() == "HND":
            out = out.transpose(1, 2)
        return out


@AttentionOpManager.register_attn("te-fp8")
class TransformerEngineAttnFP8(TransformerEngineAttn):
    def __init__(self):
        super(TransformerEngineAttnFP8, self).__init__()
        super(TransformerEngineAttnFP8, self)._enable_fp8()


@AttentionOpManager.register_attn("fivx")
class FlashInferVx(BaseAttn):
    @torch.compiler.disable()
    def _sageattn(self, *args, **kwargs):
        return flashinfer_vx.sageattn(*args, **kwargs)

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,  # WILL BE IGNORED
        enable_gqa=False,
        return_lse=False,
        tensor_layout="HND",
        cu_seqlens_q = None,
        cu_seqlens_k = None,
        max_seqlen_q = 0,
        max_seqlen_k = 0,
    ):
        if flashinfer_vx is None:
            raise ImportError("FlashInferVx is not installed")

        logger.debug("Using SageAttn for Blackwell")
        if tensor_layout not in ["HND", "NHD"]:
            raise NotImplementedError(f"Tensor layout {tensor_layout} not supported for SageAttn for Blackwell")
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not supported for Sage attention")
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not supported for Sage attention")
        if enable_gqa:
            raise NotImplementedError("GQA is not supported for Sage attention")
        if is_causal:
            raise NotImplementedError("Caucal mask is not supported by SageAttn for Blackwell yet")
        if cu_seqlens_q is not None:
            raise NotImplementedError("cu_seqlens_q is not supported for FlashInferVx")

        origin_dtype = query.dtype
        if query.dtype not in [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
            logger.debug(f"SageAttn for Blackwell: query.dtype: {query.dtype}, converting to bfloat16")
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        if tensor_layout == "HND":
            logger.debug("SageAttn for Blackwell: transposing input")
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

        if query.shape[-1] not in [128, 256]:
            raise RuntimeError(f"SageAttn for Blackwell only supports D=128 and D=256, got shape {query.shape}")

        output = self._sageattn(query, key, value, smooth=not return_lse, returnLse=return_lse)
        if return_lse:
            output, lse = output

        if tensor_layout == "HND":
            logger.debug("SageAttn for Blackwell: transposing output")
            output = output.transpose(1, 2)

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)

        if return_lse:
            return output.contiguous(), lse
        else:
            return output.contiguous()
