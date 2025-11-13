import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._utils import is_sm_100f

from ...distributed import allgather
from ...model_config import ModelConfig
from ...utils import AuxStreamType, Fp4QuantizedTensor, ceil_div
from .fused_moe_cutlass import CutlassFusedMoE
from .quantization import MoEWeightLoadingMode
from .routing import BaseMoeRoutingMethod


@torch.compile(options={"max-autotune": True})
def swiglu_fused_moe(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


def cute_dsl_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    offset_array: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1]
    l, n, k = b.shape[0], b.shape[1], b.shape[2]
    num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]

    # Note: view(int8) will cause error.
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k))
    b_tmp = b.permute(1, 2, 0)

    # Note: we have different output scale shape for fp8_quantize_1x128, so we need to handle it differently for sm100 and other archs.
    if is_sm_100f():
        input_scale_tmp = a_sf.permute(1, 0).as_strided((m, w_k, 1),
                                                        (1, m, m * w_k))
    else:
        m_padded = (m + 3) // 4 * 4
        input_scale_tmp = a_sf[0:m_padded * w_k]
        input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
        input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
        input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1),
                                                     (1, m, m * w_k))

    weight_scale_tmp = b_sf.permute(1, 2, 0)

    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)

    len_offset_array = offset_array.shape[0]
    for i in range(len_offset_array - 1):
        start = offset_array[i]
        end = offset_array[i + 1]
        # assert start <= end, f"Invalid group boundaries: start={start} > end={end}"
        ref[start:end, :] = torch.einsum("mk,nk->mn", updated_a[start:end, :,
                                                                0],
                                         updated_b[:, :, i])
    ref = ref.to(torch.bfloat16)
    return ref


def cute_dsl_nvfp4_grouped_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_size: int,
    output_dtype: torch.dtype,
    scaling_vector_size: int = 16,
):
    assert a.dtype == torch.float4_e2m1fn_x2
    assert a.dim() == 2
    assert b.dtype == torch.float4_e2m1fn_x2
    assert b.dim() == 3
    assert a_sf.dtype == torch.uint8
    assert a_sf.dim() == 1
    assert b_sf.dtype == torch.uint8
    assert b_sf.dim() == 3
    assert alpha.dtype == torch.float32
    assert alpha.dim() == 1

    m, k = a.size(0), a.size(1) * 2
    l, n = b.size(0), b.size(1)
    scale_k = k // scaling_vector_size
    assert m % tile_size == 0
    assert k % (scaling_vector_size * 4) == 0
    assert b.size(2) * 2 == k
    assert a_sf.size(0) == m * scale_k
    assert b_sf.size(0) == l
    assert b_sf.size(1) == n
    assert b_sf.size(2) == scale_k
    assert alpha.size(0) == l

    num_tiles = m // tile_size
    assert tile_idx_to_group_idx.dtype == torch.int32
    assert tile_idx_to_group_idx.size() == (num_tiles, )
    assert num_non_exiting_tiles.dtype == torch.int32
    assert num_non_exiting_tiles.size() == (1, )

    num_tiles_per_expert = torch.bincount(
        tile_idx_to_group_idx[:num_non_exiting_tiles[0].item()], minlength=l)
    offsets = [0] + num_tiles_per_expert.cumsum(dim=0).tolist()

    ref = torch.empty(m, n, dtype=output_dtype, device="cuda")
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if end <= start:
            continue
        a_sliced = a[start * tile_size:end * tile_size]
        a_sf_sliced = a_sf[start * tile_size * k // scaling_vector_size:end *
                           tile_size * k // scaling_vector_size]
        ref[start * tile_size:end * tile_size] = torch.ops.trtllm.nvfp4_gemm(
            a_sliced.view(torch.uint8), b[i].view(torch.uint8), a_sf_sliced,
            b_sf[i], alpha[i], output_dtype)

    return ref


class CuteDslFusedMoE(CutlassFusedMoE):
    """
    Python Flow of Fused Mixture of Experts (MoE) Layer.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
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
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
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
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
        )

    def forward_chunk_unquantized(
            self,
            x: Union[torch.Tensor, Fp4QuantizedTensor],
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        assert not self.has_any_quant
        return super().forward_chunk(x,
                                     router_logits,
                                     output_dtype=output_dtype,
                                     all_rank_num_tokens=all_rank_num_tokens,
                                     use_dp_padding=use_dp_padding,
                                     repeating_info=repeating_info)

    def forward_chunk_fp8_block_scales(
            self,
            x: Union[torch.Tensor, Fp4QuantizedTensor],
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        assert self.has_deepseek_fp8_block_scales

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

        weight_dtype = self.w3_w1_weight.dtype

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
            input_sf=None,
            num_experts_on_rank=self.expert_size_per_partition,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            min_latency_mode=False,
            use_fp8_block_scaling=True,
        )
        act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(
            permuted_data_tensor)
        h1 = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w3_w1_weight.view(weight_dtype),
            a_sf=act_input_sf,
            b_sf=self.quant_scales[0],
            offset_array=expert_first_token_offset_tensor,
        )
        h2 = swiglu_fused_moe(h1)
        act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(h2)
        h3 = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=act_input_fp8,
            b=self.w2_weight.view(weight_dtype),
            a_sf=act_input_sf,
            b_sf=self.quant_scales[1],
            offset_array=expert_first_token_offset_tensor,
        )
        h4 = torch.ops.trtllm.moe_finalize_scale_op(
            h3,
            None,  # biases
            token_final_scales,
            unpermuted_row_to_permuted_row_tensor,
            permuted_row_to_unpermuted_row_tensor,
            token_selected_experts,
            expert_first_token_offset_tensor,
            False,  # enable_alltoall
            x.shape[0],  # num_rows
            x.shape[1],  # (possibly padded) hidden_size
            self.unpadded_hidden_size,  # original hidden size
            self.routing_method.top_k,
            self.expert_size_per_partition,  # num_experts_per_node
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
        )
        return h4

    def forward_chunk_nvfp4(
            self,
            x: Union[torch.Tensor, Fp4QuantizedTensor],
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        assert self.has_nvfp4

        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
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

        run_post_quant_allgather = self.use_dp and self.parallel_size > 1
        if run_post_quant_allgather:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False,
                    False)
            # note: we use uint8 to store 2 fp4 values
            x_row, x_col = x.size(0), x.size(1) * 2
        else:
            if not isinstance(x, Fp4QuantizedTensor):
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False,
                    False)

        if run_post_quant_allgather:
            # Original allgather logic
            if x_sf is not None:
                x_sf = x_sf.view(x_row, ceil_div(x_col,
                                                 self.scaling_vector_size))
                assert x_sf.dim(
                ) == 2, "The hidden states scaling factor should be 2D tensor before allgather"

            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)

        tile_size = 128
        tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            local_expert_offset=self.slot_start,
            local_num_experts=self.expert_size_per_partition,
            tile_tokens_dim=tile_size,
        )

        x, x_sf = torch.ops.trtllm.moe_permute(
            input=x.view(torch.float4_e2m1fn_x2),
            input_sf=x_sf,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_tokens_dim=tile_size,
            top_k=self.routing_method.experts_per_token,
        )
        x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=self.w3_w1_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale=self.quant_scales.fc1_weight_block.view(torch.uint8),
            alpha=self.quant_scales.fc1_global,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_dtype=output_dtype,
        )
        x, x_sf = torch.ops.trtllm.moe_swiglu_nvfp4_quantize(
            input=x,
            global_sf=self.fc2_input_scale,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_tokens_dim=tile_size,
        )
        x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
            input=x.view(torch.float4_e2m1fn_x2),
            weight=self.w2_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            weight_scale=self.quant_scales.fc2_weight_block.view(torch.uint8),
            alpha=self.quant_scales.fc2_global,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=self.num_slots,
            top_k=self.routing_method.experts_per_token,
            num_local_experts=self.expert_size_per_partition,
            local_expert_offset=self.slot_start,
            tile_size=tile_size,
            output_dtype=output_dtype,
        )
        x = torch.ops.trtllm.moe_unpermute(
            permuted_input=x,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            topk_scales=token_final_scales,
        )
        return x

    def forward_chunk(
            self,
            x: Union[torch.Tensor, Fp4QuantizedTensor],
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        if self.has_any_quant:
            if self.has_nvfp4:
                return self.forward_chunk_nvfp4(
                    x,
                    router_logits,
                    output_dtype=output_dtype,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=use_dp_padding,
                    repeating_info=repeating_info)
            elif self.has_deepseek_fp8_block_scales:
                return self.forward_chunk_fp8_block_scales(
                    x,
                    router_logits,
                    output_dtype=output_dtype,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=use_dp_padding,
                    repeating_info=repeating_info)
            else:
                raise ValueError(
                    f"unsupported quantization mode for CUTEDSL backend: {self.quant_config.quant_mode}"
                )
        else:
            return self.forward_chunk_unquantized(
                x,
                router_logits,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                repeating_info=repeating_info)
