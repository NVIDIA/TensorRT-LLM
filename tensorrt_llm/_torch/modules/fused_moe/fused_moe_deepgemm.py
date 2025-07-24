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
def _masked_index_copy_group_quant_fp8(
    input_ptr,
    out_q_ptr,
    out_s_ptr,
    # mask indices
    start_offsets_ptr,
    row_indices_ptr,
    # dimensions
    row_size,
    col_size,
    dim_size,
    group_size,
    # output scale factor size
    aligned_col,
    aligned_dim,
    # quantization parameters
    eps,
    fp8_max,
    # block size
    BLOCK: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    group_block = tl.program_id(0)
    token_block = tl.program_id(1)
    token_block_num = tl.num_programs(1)

    # calculate group and element offsets
    num_tokens = tl.load(start_offsets_ptr + row_size)
    group_start = group_block * group_size
    elem_offsets = group_start + tl.arange(0, BLOCK)
    valid_elem = elem_offsets < (group_start + group_size)
    input_ptr_offs = input_ptr + elem_offsets
    output_ptr_offs = out_q_ptr + elem_offsets
    output_s_offs = out_s_ptr + (group_block // 4) * aligned_col
    shift = (group_block % 4) * 8

    # process tokens
    for token_index in tl.range(token_block,
                                num_tokens,
                                token_block_num,
                                num_stages=NUM_STAGE):
        # load input and indices
        input_data = tl.load(input_ptr_offs + token_index * dim_size,
                             mask=valid_elem,
                             other=0.0)
        row_idx = tl.load(row_indices_ptr + token_index)
        start_offset = tl.load(start_offsets_ptr + row_idx)
        idx = row_idx * col_size + token_index - start_offset
        idx_s = row_idx * aligned_dim * aligned_col + token_index - start_offset

        # quantization
        _absmax = tl.maximum(tl.max(tl.abs(input_data)), eps)
        output_s = _absmax / fp8_max
        output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(input_data / output_s, -fp8_max,
                            fp8_max).to(out_q_ptr.dtype.element_ty)
        output_s = (output_s.to(tl.int32, bitcast=True) >> 23).to(tl.uint8)

        # store quantized values and scaling factor
        tl.store(output_ptr_offs + idx * dim_size, output_q, mask=valid_elem)
        tl.atomic_or(output_s_offs + idx_s, output_s << shift)


def masked_index_copy_group_quant_fp8(
    output: torch.Tensor,
    # output_s: torch.Tensor,
    input: torch.Tensor,
    start_offsets: torch.Tensor,
    row_indices: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
):
    assert (
        input.shape[-1] % group_size == 0
    ), "the last dimension of `input` cannot be divisible by `group_size`"
    assert input.is_contiguous(), "`input` is not contiguous"
    assert input.ndim == 2, "Input must be a 2D tensor"
    assert output.ndim == 3, "Output must be a 3D tensor, [row, col, dim]"
    assert start_offsets.shape[
        0] == output.shape[0] + 1, "Start offsets must be (num_experts + 1)"

    num_tokens = input.shape[0]
    row_size = output.shape[0]
    col_size = output.shape[1]
    dim_size = output.shape[2]
    num_groups = (dim_size + group_size - 1) // group_size

    # create padded output_s
    alignment = 4
    scale_dim = (dim_size + group_size - 1) // group_size
    padded_dim_size = (scale_dim + alignment - 1) // alignment * alignment
    padded_col_size = (col_size + alignment - 1) // alignment * alignment
    output_s = torch.zeros((row_size, padded_dim_size // 4, padded_col_size),
                           dtype=torch.int32,
                           device='cuda')

    # get block/grid/stage/warp
    BLOCK = group_size
    if num_tokens <= 4096:
        TOKEN_BLOCK_NUM = 128
        NUM_STAGES = 4
        num_warps = 2
    else:
        TOKEN_BLOCK_NUM = 64
        NUM_STAGES = 6
        num_warps = 1
    grid = (
        num_groups,
        TOKEN_BLOCK_NUM,
    )

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max

    _masked_index_copy_group_quant_fp8[grid](
        input,
        output,
        output_s,
        start_offsets,
        row_indices,
        row_size,
        col_size,
        dim_size,
        group_size,
        padded_col_size,
        padded_dim_size // 4,
        eps,
        fp8_max,
        BLOCK=BLOCK,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )
    output_s = output_s.transpose(1, 2)[:, :col_size, :]
    return output_s


@triton.jit
def masked_index_gather_kernel(output_ptr, input_ptr, start_offsets_ptr,
                               row_indices_ptr, row_size, col_size, dim_size,
                               BLOCK_SIZE: tl.constexpr):
    # get program id and block offset
    pid = tl.program_id(0)
    num_tokens = tl.load(start_offsets_ptr + row_size)

    token_idx = pid
    valid_token = token_idx < num_tokens
    if not valid_token:
        return

    row_idx = tl.load(row_indices_ptr + token_idx)
    start_offset = tl.load(start_offsets_ptr + row_idx)
    col_idx = token_idx - start_offset

    # Process elements in blocks
    for hidden_start in tl.range(0, dim_size, BLOCK_SIZE):
        hidden_indices = hidden_start + tl.arange(0, BLOCK_SIZE)
        valid_hidden = hidden_indices < dim_size

        input_offset = row_idx * col_size * dim_size + col_idx * dim_size + hidden_indices
        input_val = tl.load(input_ptr + input_offset,
                            mask=valid_hidden,
                            other=0.0)

        output_offset = pid * dim_size + hidden_indices
        tl.store(output_ptr + output_offset, input_val, mask=valid_hidden)


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

    grid = (num_tokens, )
    # launch kernel
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
        act_input_fp8 = torch.empty(
            (self.expert_size_per_partition, m_max, self.hidden_size),
            dtype=torch.float8_e4m3fn,
            device='cuda')
        act_input_sf = masked_index_copy_group_quant_fp8(
            act_input_fp8,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            token_to_expert_map,
            group_size=128)

        h1 = deepgemm_fp8_group_blockwise_gemm(
            a=act_input_fp8,
            b=self.w3_w1_weight,
            sfa=act_input_sf,
            sfb=self.quant_scales[0],
            masked_m=masked_m,
            expected_m=expected_m,
        )
        act_input_fp8 = torch.empty(h1.shape[0],
                                    h1.shape[1],
                                    h1.shape[2] // 2,
                                    dtype=torch.float8_e4m3fn,
                                    device='cuda')
        act_input_sf = torch.empty(h1.shape[0],
                                   h1.shape[1],
                                   h1.shape[2] // 256,
                                   dtype=torch.float32,
                                   device='cuda')
        fp8_utils.silu_and_mul_masked_post_quant_fwd(input=h1,
                                                     output=act_input_fp8,
                                                     output_scale=act_input_sf,
                                                     quant_group_size=128,
                                                     masked_m=masked_m,
                                                     scale_ue8m0=True)
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
