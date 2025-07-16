from typing import List, Optional, Union

import deep_gemm
import torch
import torch.nn.functional as F

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import nvtx_range

from ...model_config import ModelConfig
from ...utils import Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .quantization import MoEWeightLoadingMode
from .routing import BaseMoeRoutingMethod


@nvtx_range("[DG] act")
@torch.compile(dynamic=True)
def swiglu_fused_moe(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


@nvtx_range("[DG] indexing")
@torch.compile(dynamic=True)
def indexing(x, mask):
    return x[mask > 0, :].contiguous()


@nvtx_range("[DG] copy after permute")
@torch.compile(dynamic=True)
def copy_after(
    expert_first_token_offset_tensor,
    permuted_data_tensor,
    base_indices,
    hidden_size,
):
    token_per_expert = expert_first_token_offset_tensor[
        1:] - expert_first_token_offset_tensor[:-1]
    token_per_expert_padded = (token_per_expert + 127) // 128 * 128
    expert_first_token_offset_tensor_padded = torch.cat(
        (torch.zeros(1, dtype=torch.int32,
                     device='cuda'), torch.cumsum(token_per_expert_padded,
                                                  dim=0)))

    token_num = token_per_expert.sum()
    total_tokens_padded = token_per_expert_padded.sum()
    m_indices = torch.repeat_interleave(base_indices,
                                        token_per_expert_padded,
                                        dim=0,
                                        output_size=total_tokens_padded)
    src_offsets = torch.repeat_interleave(expert_first_token_offset_tensor[:-1],
                                          token_per_expert,
                                          dim=0,
                                          output_size=token_num)
    dest_starts = torch.repeat_interleave(
        expert_first_token_offset_tensor_padded[:-1],
        token_per_expert,
        dim=0,
        output_size=token_num)
    token_j_offset_in_expert = torch.arange(token_num,
                                            device='cuda') - src_offsets
    dest_indices = dest_starts + token_j_offset_in_expert

    permuted_data_tensor_padded = torch.empty(total_tokens_padded,
                                              hidden_size,
                                              dtype=permuted_data_tensor.dtype,
                                              device='cuda')
    src_indices = torch.arange(dest_indices.shape[0], device='cuda')
    permuted_data_tensor_padded.index_copy_(0, dest_indices,
                                            permuted_data_tensor[src_indices])

    return permuted_data_tensor_padded, m_indices, dest_indices


@nvtx_range("[DG]")
def deepgemm_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:

    torch.cuda.synchronize()
    d = torch.empty((a.shape[0], b.shape[1]),
                    device=b.device,
                    dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a, a_sf), (b, b_sf), d,
                                               m_indices)
    torch.cuda.synchronize()
    return d


class DeepGemmFusedMoE(CutlassFusedMoE):
    """
    Python Flow of Fused Mixture of Experts (MoE) Layer.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream (Optional[torch.cuda.Stream]): Auxiliary CUDA stream to overlap chunks.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.

    This backend is composed of multiple custom ops:
    1. moe_permute_op: permute the input tensor and the expert selected tensor.
    2. cute_dsl_fp8_group_blockwise_gemm_ref: a reference implementation of the cute_dsl_fp8_group_blockwise_gemm.
    3. moe_finalize_scale_op: finalize the scale of the output tensor.
    """

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream: Optional[torch.cuda.Stream] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
    ):

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream=aux_stream,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
        )

        self.base_indices = torch.arange(self.expert_size_per_partition,
                                         device="cuda",
                                         dtype=torch.int32)

    @nvtx_range("[DG] forward")
    def forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
    ) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
            output_dtype = output_dtype
        else:
            output_dtype = x.dtype

        # apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)
        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current workaround only supports top-1 routing"
            assert x.dtype != torch.float8_e4m3fn, "Current workaround for apply_router_weight_on_input does not support fp8 input"
            x = x * token_final_scales.to(x.dtype)
            # TODO: remove this once we have correct fusedmoe kernel ready
            token_final_scales = None

        # quantize inputs
        use_deepseek_fp8_block_scale = False
        x_sf = None
        if self.has_any_quant:
            if self.has_deepseek_fp8_block_scales:
                use_deepseek_fp8_block_scale = True
            else:
                raise ValueError(
                    f"unsupported quantization mode for CUTEDSL backend: {self.quant_config.quant_mode}"
                )

        (
            permuted_row_to_unpermuted_row_tensor,
            permuted_token_selected_experts_tensor,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            permuted_token_final_scales_tensor,
            unpermuted_row_to_permuted_row_tensor,
        ) = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_experts,
            token_final_scales,
            None,  # w3_w1_weight.view(weight_dtype),
            None,  # w2_weight.view(weight_dtype),
            None,  # quant_scales,
            input_sf=x_sf,
            num_experts_on_rank=self.expert_size_per_partition,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            min_latency_mode=False,
            use_fp8_block_scaling=use_deepseek_fp8_block_scale,
        )

        permuted_data_tensor_padded, m_indices, dest_indices = copy_after(
            expert_first_token_offset_tensor,
            permuted_data_tensor,
            self.base_indices,
            self.hidden_size,
        )

        if permuted_data_tensor_padded.numel() == 0:
            return torch.zeros_like(x)
        act_input_fp8, act_input_sf = fp8_utils.per_token_cast_to_fp8_e8m0(
            permuted_data_tensor_padded)
        h1 = deepgemm_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w3_w1_weight,
            a_sf=act_input_sf,
            b_sf=self.quant_scales[0],
            m_indices=m_indices,
        )
        h2 = swiglu_fused_moe(h1)
        act_input_fp8, act_input_sf = fp8_utils.per_token_cast_to_fp8_e8m0(h2)
        h3 = deepgemm_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w2_weight,
            a_sf=act_input_sf,
            b_sf=self.quant_scales[1],
            m_indices=m_indices,
        )

        permuted_data_tensor[0:dest_indices.shape[0]].copy_(h3[dest_indices])
        final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
            permuted_data_tensor,
            None,  # biases
            token_final_scales,
            unpermuted_row_to_permuted_row_tensor,
            permuted_row_to_unpermuted_row_tensor,
            token_selected_experts,
            expert_first_token_offset_tensor,
            False,  # enable_alltoall
            x.shape[0],  # num_rows
            x.shape[1],  # hidden_size
            self.routing_method.top_k,
            self.expert_size_per_partition,  # num_experts_per_node
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
        )

        return final_hidden_states
