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
        if self.has_any_quant:
            if self.has_deepseek_fp8_block_scales:
                pass
            else:
                raise ValueError(
                    f"unsupported quantization mode for DEEPGEMM backend: {self.quant_config.quant_mode}"
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

        if permuted_data_tensor.numel() == 0:
            return torch.zeros_like(x)
        act_input_fp8, act_input_sf = fp8_utils.per_token_cast_to_fp8_e8m0(
            permuted_data_tensor)
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

        res = (h3[token_map > 0, :] *
               token_final_scales[m[:, 1], m[:, 2]].unsqueeze(1)).to(h3.dtype)

        final_hidden_states = torch.zeros_like(x)
        indices = m[:, 1].unsqueeze(1).expand(-1, res.size(1)).cuda()  # [N, D]
        final_hidden_states.scatter_add_(0, indices, res)

        return final_hidden_states
