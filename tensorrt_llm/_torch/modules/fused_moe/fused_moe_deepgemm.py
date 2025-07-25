from typing import List, Optional, Union

import deep_gemm
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import nvtx_range

from ...distributed import allgather
from ...model_config import ModelConfig
from ...utils import Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .quantization import MoEWeightLoadingMode
from .routing import BaseMoeRoutingMethod


@triton.jit
def masked_index_copy_kernel(output_ptr, input_ptr, start_offsets_ptr,
                             row_indices_ptr, row_size, col_size, dim_size,
                             BLOCK_SIZE: tl.constexpr):
    # get program id and block offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # compute mask and pointers
    num_tokens = tl.load(start_offsets_ptr + row_size)
    token_idx = offsets // dim_size
    valid = token_idx < num_tokens
    row_idx = tl.load(row_indices_ptr + token_idx)
    start_offset = tl.load(start_offsets_ptr + row_idx, mask=valid)
    col_idx = token_idx - start_offset
    elem_idx = offsets % dim_size

    # load input data
    input = tl.load(input_ptr + offsets, mask=valid)

    # write output
    output_offsets = row_idx * col_size * dim_size + col_idx * dim_size + elem_idx
    tl.store(output_ptr + output_offsets, input, mask=valid)


def triton_masked_index_copy(output, input, start_offsets, row_indices):
    assert output.ndim == 3, "Input must be a 3D tensor, [row, col, dim]"
    assert input.ndim == 2, "Input must be a 2D tensor"
    assert start_offsets.shape[
        0] == output.shape[0] + 1, "Start offsets must be (num_experts + 1)"

    num_tokens = input.shape[0]
    row_size = output.shape[0]
    col_size = output.shape[1]
    dim_size = output.shape[2]
    total_elems = num_tokens * dim_size

    # launch kernel
    grid = lambda meta: (triton.cdiv(total_elems, meta['BLOCK_SIZE']), )
    masked_index_copy_kernel[grid](output,
                                   input,
                                   start_offsets,
                                   row_indices,
                                   row_size,
                                   col_size,
                                   dim_size,
                                   BLOCK_SIZE=1024)
    return


@triton.jit
def masked_index_gather_kernel(output_ptr, input_ptr, start_offsets_ptr,
                               row_indices_ptr, row_size, col_size, dim_size,
                               BLOCK_SIZE: tl.constexpr):
    # get program id and block offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # compute mask and pointers
    num_tokens = tl.load(start_offsets_ptr + row_size)
    token_idx = offsets // dim_size
    valid = token_idx < num_tokens
    row_idx = tl.load(row_indices_ptr + token_idx)
    start_offset = tl.load(start_offsets_ptr + row_idx, mask=valid)
    col_idx = token_idx - start_offset
    elem_idx = offsets % dim_size

    # input data
    input_offsets = row_idx * col_size * dim_size + col_idx * dim_size + elem_idx
    input_vals = tl.load(input_ptr + input_offsets, mask=valid)

    # get gather indices and store to output
    tl.store(output_ptr + offsets, input_vals, mask=valid)


@torch.no_grad()
def triton_masked_index_gather(output, input, start_offsets, row_indices):
    assert output.ndim == 2, "Output must be a 2D tensor"
    assert input.ndim == 3, "Input must be a 3D tensor, [row, col, dim]"
    assert start_offsets.shape[
        0] == input.shape[0] + 1, "Start offsets must be (num_experts + 1)"

    row_size = input.shape[0]
    col_size = input.shape[1]
    dim_size = input.shape[2]
    num_tokens = output.shape[0]
    total_elems = num_tokens * dim_size

    # launch kernel
    grid = lambda meta: (triton.cdiv(total_elems, meta['BLOCK_SIZE']), )
    masked_index_gather_kernel[grid](output,
                                     input,
                                     start_offsets,
                                     row_indices,
                                     row_size,
                                     col_size,
                                     dim_size,
                                     BLOCK_SIZE=1024)
    return


@nvtx_range("[DG] act")
@torch.compile(dynamic=True)
def swiglu_fused_moe(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


@nvtx_range("[DG] indexing")
@torch.compile(dynamic=True)
def indexing(x, mask):
    return x[mask > 0, :].contiguous()


@nvtx_range("[DG] preprocess_after_permute")
def preprocess_after_permute(expert_first_token_offset_tensor,
                             permuted_data_tensor):
    # get tokens per expert
    masked_m = expert_first_token_offset_tensor[
        1:] - expert_first_token_offset_tensor[:-1]
    token_to_expert_map = torch.searchsorted(
        expert_first_token_offset_tensor[1:],
        torch.arange(permuted_data_tensor.shape[0], device='cuda'),
        right=True)
    return masked_m.to(torch.int32), token_to_expert_map


@nvtx_range("[DG]")
def deepgemm_fp8_group_blockwise_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> torch.Tensor:
    d = torch.empty((a.shape[0], a.shape[1], b.shape[1]),
                    device=b.device,
                    dtype=torch.bfloat16)

    # NOTES: shape must be `[G, M, K] @ [G, N, K].mT`
    assert a.stride(-1) == 1
    assert b.stride(-1) == 1
    assert masked_m.is_contiguous()

    num_groups, m, k = a.shape
    num_groups_, n, k_ = b.shape
    num_groups__, m_, n_ = d.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    assert d.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32

    # D must be N-major
    assert d.stride(-1) == 1

    # Transform SFA and SFB into compute-required layout
    recipe = (1, 128, 128)
    sfa = fp8_utils.transform_sf_into_required_layout(sfa,
                                                      mn=m,
                                                      k=k,
                                                      recipe=recipe,
                                                      num_groups=num_groups,
                                                      is_sfa=True)
    sfb = fp8_utils.transform_sf_into_required_layout(sfb,
                                                      mn=n,
                                                      k=k,
                                                      recipe=recipe,
                                                      num_groups=num_groups,
                                                      is_sfa=False)

    deep_gemm.fp8_m_grouped_gemm_nt_masked((a, sfa), (b, sfb),
                                           d,
                                           masked_m,
                                           expected_m,
                                           disable_ue8m0_cast=True)
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

        use_allgather = self.use_dp and self.parallel_size > 1
        if use_allgather:
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)

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

        if permuted_data_tensor.numel() == 0:
            return torch.zeros_like(x)

        masked_m, token_to_expert_map = preprocess_after_permute(
            expert_first_token_offset_tensor, permuted_data_tensor)

        m_max = (x.shape[0] + 127) // 128 * 128
        expected_m = (token_selected_experts.numel() +
                      self.expert_size_per_partition -
                      1) // self.expert_size_per_partition
        permuted_data_tensor_padded = torch.empty(
            self.expert_size_per_partition,
            m_max,
            self.hidden_size,
            dtype=self.dtype,
            device='cuda')
        triton_masked_index_copy(permuted_data_tensor_padded,
                                 permuted_data_tensor,
                                 expert_first_token_offset_tensor,
                                 token_to_expert_map)

        act_input_fp8, act_input_sf = fp8_utils.per_token_cast_to_fp8_e8m0(
            permuted_data_tensor_padded)
        h1 = deepgemm_fp8_group_blockwise_gemm(
            a=act_input_fp8,
            b=self.w3_w1_weight,
            sfa=act_input_sf,
            sfb=self.quant_scales[0],
            masked_m=masked_m,
            expected_m=expected_m,
        )
        h2 = swiglu_fused_moe(h1)
        act_input_fp8, act_input_sf = fp8_utils.per_token_cast_to_fp8_e8m0(h2)
        h3 = deepgemm_fp8_group_blockwise_gemm(
            a=act_input_fp8,
            b=self.w2_weight,
            sfa=act_input_sf,
            sfb=self.quant_scales[1],
            masked_m=masked_m,
            expected_m=expected_m,
        )

        triton_masked_index_gather(permuted_data_tensor, h3,
                                   expert_first_token_offset_tensor,
                                   token_to_expert_map)

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
