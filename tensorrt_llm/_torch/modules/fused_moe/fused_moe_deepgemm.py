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
def swiglu_fused_moe(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


@nvtx_range("[DG]")
def deepgemm_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    m_indices: torch.Tensor,
) -> torch.Tensor:

    # m, k = a.shape
    # num_groups, n, _ = b.shape

    # m_padded = (m + 127) // 128 * 128
    torch.cuda.synchronize()
    d = torch.empty((a.shape[0], b.shape[1]),
                    device=b.device,
                    dtype=torch.bfloat16)
    # m_indices = torch.empty(a.shape[0], device=b.device, dtype=torch.int32)
    # for idx in range(offset_array.numel() - 1):
    #     m_indices[offset_array[idx]:offset_array[idx + 1]] = idx

    # for g in range(num_groups):
    #     aa = a[offset_array[g]:offset_array[g + 1], :].to(torch.bfloat16)
    #     aa_sf = a_sf[offset_array[g]:offset_array[g + 1], :]
    #     aa_dq = aa * aa_sf.repeat_interleave(128, dim=1)[:aa.shape[0], :aa.shape[1]]
    #     bb = b[g, :, :].to(torch.bfloat16)
    #     bb_sf = b_sf[g, :, :]
    #     bb_dq = bb * bb_sf.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:bb.shape[0], :bb.shape[1]]
    #     if aa_dq.numel() == 0:
    #         continue
    #     d[offset_array[g]:offset_array[g + 1], :] = (aa_dq @ bb_dq.t())
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

        experts = torch.arange(self.ep_rank * self.expert_size_per_partition,
                               (self.ep_rank + 1) *
                               self.expert_size_per_partition,
                               device=x.device).view(-1, 1, 1)
        matches = (token_selected_experts == experts).cpu()
        token_per_expert = matches.sum(dim=[-1, -2]).flatten()
        token_per_expert_padded = (token_per_expert + 127) // 128 * 128
        token_per_expert_offset_padded = torch.cat(
            (torch.tensor([0], dtype=torch.int32),
             torch.cumsum(token_per_expert_padded, dim=0)))

        permuted_data_tensor = torch.empty(token_per_expert_padded.sum(),
                                           x.shape[1],
                                           dtype=x.dtype,
                                           device=x.device)
        m_indices = torch.empty(permuted_data_tensor.shape[0],
                                dtype=torch.int32)
        token_map = torch.zeros(permuted_data_tensor.shape[0],
                                dtype=torch.int32)
        m = matches.nonzero()
        m_indices = torch.cat([
            torch.full((l, ), i, dtype=torch.int32)
            for i, l in enumerate(token_per_expert_padded)
        ])
        for idx in range(experts.numel()):
            token_map[token_per_expert_offset_padded[idx]:
                      token_per_expert_offset_padded[idx] +
                      token_per_expert[idx]] = 1
        permuted_data_tensor[token_map > 0, :] = x[m[:, 1], :]

        # token_final_scales_padded = []
        # token_map = []
        # expert_first_token_offset_tensor = torch.zeros(
        #     self.expert_size_per_partition + 1, dtype=torch.int32)

        # t_idx = 0
        # accum_t_idx = 0
        # for e_idx in range(self.ep_rank * self.expert_size_per_partition, (self.ep_rank + 1) * self.expert_size_per_partition):
        #     for idx, token in enumerate(x):
        #         if e_idx in token_selected_experts[idx]:
        #             token_final_scales_padded.append(
        #                 token_final_scales[idx][torch.where(
        #                     token_selected_experts[idx] == e_idx)[0].item()])
        #             token_map.append(idx)
        #             t_idx += 1
        #     ceil_t_idx = (t_idx + 127) // 128 * 128
        #     for _ in range(ceil_t_idx - t_idx):
        #         token_final_scales_padded.append(0)
        #         token_map.append(-1)
        #     t_idx = ceil_t_idx
        #     accum_t_idx += idx
        #     expert_first_token_offset_tensor[e_idx - self.ep_rank * self.expert_size_per_partition + 1] = t_idx
        # # print(self.ep_rank, x.shape, expert_first_token_offset_tensor[-1])
        # # print("-------------------")
        # permuted_data_tensor = torch.zeros(expert_first_token_offset_tensor[-1], x.shape[1], dtype=x.dtype, device=x.device)
        # for idx, line in enumerate(permuted_data_tensor):
        #     token_idx = token_map[idx]
        #     if token_idx >= 0:
        #         line.copy_(x[token_idx, :])
        # if len(permuted_data_tensor) == 0:
        #     # for e_idx in range(self.ep_rank * self.expert_size_per_partition, (self.ep_rank + 1) * self.expert_size_per_partition):
        #     #     for idx, token in enumerate(x):
        #     #         if e_idx in token_selected_experts[idx]:
        #     #             print("Yes!")
        #     return torch.zeros_like(x)
        #     # assert False
        # # permuted_data_tensor = torch.stack(permuted_data_tensor).contiguous()
        # token_final_scales_padded = torch.Tensor(token_final_scales_padded).contiguous()

        # print(permuted_data_tensor.shape, token_final_scales_padded.shape)
        # print(permuted_data_tensor[:, 0])
        # print(x[:, 0])
        # print(token_final_scales_padded)
        # print(token_final_scales)
        # print(token_selected_experts)
        # print(expert_first_token_offset_tensor)
        # print(token_map)

        if permuted_data_tensor.numel() == 0:
            return torch.zeros_like(x)
        act_input_fp8, act_input_sf = fp8_utils.per_token_cast_to_fp8_e8m0(
            permuted_data_tensor)
        # print(f"act_input_fp8, shape: {act_input_fp8.shape}, type: {act_input_fp8.dtype}")
        # print(f"act_input_sf, shape: {act_input_sf.shape}, type: {act_input_sf.dtype}")
        h1 = deepgemm_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w3_w1_weight,
            a_sf=act_input_sf,
            b_sf=self.quant_scales[0],
            m_indices=m_indices,
        )
        h2 = swiglu_fused_moe(h1)
        # print(f"h2, shape: {h2.shape}, type: {h2.dtype}")
        act_input_fp8, act_input_sf = fp8_utils.per_token_cast_to_fp8_e8m0(h2)
        # print(f"act_input_fp8, shape: {act_input_fp8.shape}, type: {act_input_fp8.dtype}")
        # print(f"act_input_sf, shape: {act_input_sf.shape}, type: {act_input_sf.dtype}")

        h3 = deepgemm_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w2_weight,
            a_sf=act_input_sf,
            b_sf=self.quant_scales[1],
            m_indices=m_indices,
        )

        # print(m_indices[token_map > 0])
        # for ss in [permuted_data_tensor, h1, h2, h3]:
        #     print("--")
        #     print(ss[token_map > 0, 0])

        # print(111, m.shape, token_final_scales[m[:, 1], m[:, 2]].unsqueeze(1).shape, h3[token_map, :].shape)
        res = (h3[token_map > 0, :] *
               token_final_scales[m[:, 1], m[:, 2]].unsqueeze(1)).to(h3.dtype)

        final_hidden_states = torch.zeros_like(x)
        indices = m[:, 1].unsqueeze(1).expand(-1, res.size(1)).cuda()  # [N, D]

        # 使用scatter_add_进行累加
        # print(final_hidden_states.dtype, res.dtype)
        # final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
        #     h3,
        #     None,  # biases
        #     token_final_scales,
        #     unpermuted_row_to_permuted_row_tensor,
        #     permuted_row_to_unpermuted_row_tensor,
        #     token_selected_experts,
        #     expert_first_token_offset_tensor,
        #     False,  # enable_alltoall
        #     x.shape[0],  # num_rows
        #     x.shape[1],  # hidden_size
        #     self.routing_method.top_k,
        #     self.expert_size_per_partition,  # num_experts_per_node
        #     self.tp_size,
        #     self.tp_rank,
        #     self.ep_size,
        #     self.ep_rank,
        # )
        final_hidden_states.scatter_add_(0, indices, res)
        # final_hidden_states = torch.zeros_like(x)

        return final_hidden_states
