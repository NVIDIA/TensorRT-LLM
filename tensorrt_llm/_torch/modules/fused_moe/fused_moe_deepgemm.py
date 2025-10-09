from typing import Dict, List, Optional, Union

import torch
import triton
import triton.language as tl

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm import deep_gemm
from tensorrt_llm._utils import nvtx_range

from ...distributed import allgather
from ...memory_buffer_utils import get_memory_buffers
from ...model_config import ModelConfig
from ...utils import AuxStreamType, EventType, Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm,
                           MoEWeightLoadingMode, UnquantizedFusedMoEMethod)
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
    elem_offsets = group_block * group_size * 4 + tl.arange(0, BLOCK)
    output_s_offs = out_s_ptr + group_block * aligned_col

    # process tokens
    for token_index in tl.range(token_block,
                                num_tokens,
                                token_block_num,
                                num_stages=NUM_STAGE):
        # load indices
        row_idx = tl.load(row_indices_ptr + token_index)
        start_offset = tl.load(start_offsets_ptr + row_idx)
        idx = row_idx * col_size + token_index - start_offset
        idx_s = row_idx * aligned_dim * aligned_col + token_index - start_offset

        output_s_int32 = 0
        for group_index in tl.range(4):
            # load input data
            dim_offset = elem_offsets + group_index * group_size
            valid = dim_offset < dim_size
            input_data = tl.load(input_ptr + token_index * dim_size +
                                 dim_offset,
                                 mask=valid,
                                 other=0.0)
            # quantization
            _absmax = tl.maximum(tl.max(tl.abs(input_data)), eps)
            output_s = _absmax / fp8_max
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
            output_q = tl.clamp(input_data / output_s, -fp8_max,
                                fp8_max).to(out_q_ptr.dtype.element_ty)
            output_s = output_s.to(tl.int32, bitcast=True) >> 23
            output_s_int32 += output_s << (group_index * 8)

            # store quantized values and scaling factor
            tl.store(out_q_ptr + idx * dim_size + dim_offset,
                     output_q,
                     mask=valid)
        tl.store(output_s_offs + idx_s, output_s_int32)


def masked_index_copy_group_quant_fp8(
    output: torch.Tensor,
    output_s: torch.Tensor,
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

    alignment = 4
    scale_dim = (dim_size + group_size - 1) // group_size
    padded_dim_size = (scale_dim + alignment - 1) // alignment * alignment
    padded_col_size = (col_size + alignment - 1) // alignment * alignment

    # get block/grid/stage/warp
    num_groups = (dim_size + group_size - 1) // group_size
    BLOCK = group_size
    if num_tokens <= 1000 or col_size <= 256:  # Small workload
        TOKEN_BLOCK_NUM = 256
        NUM_STAGES = 4
        num_warps = 2
    elif num_tokens <= 10000 or col_size <= 2048:  # Medium workload
        TOKEN_BLOCK_NUM = 1024
        NUM_STAGES = 2
        num_warps = 1
    else:  # Large workload
        TOKEN_BLOCK_NUM = 2048
        NUM_STAGES = 2
        num_warps = 1
    grid = (
        (num_groups + 3) // 4,
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


@triton.jit
def _preprocess_after_permute_kernel(
    expert_offsets_ptr,
    masked_m_ptr,
    token_map_ptr,
    total_tokens,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    if pid_y == 0:
        token_offsets = pid_x * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        token_mask = token_offsets < total_tokens
        # get expert_id for each token in the block
        expert_ids = tl.full((BLOCK_SIZE_M, ), NUM_EXPERTS - 1, dtype=tl.int32)
        found_mask = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.int1)
        for i in tl.static_range(NUM_EXPERTS):
            boundary = tl.load(expert_offsets_ptr + i + 1)
            cond = (token_offsets < boundary) & ~found_mask
            expert_ids = tl.where(cond, i, expert_ids)
            found_mask = found_mask | cond
        tl.store(token_map_ptr + token_offsets,
                 expert_ids.to(tl.int64),
                 mask=token_mask)
    elif pid_y == 1:
        # get num_tokens for each expert
        expert_mask = pid_x < NUM_EXPERTS
        next_offset = tl.load(expert_offsets_ptr + pid_x + 1,
                              mask=expert_mask,
                              other=0)
        current_offset = tl.load(expert_offsets_ptr + pid_x,
                                 mask=expert_mask,
                                 other=0)
        tokens_per_expert = next_offset - current_offset
        tl.store(masked_m_ptr + pid_x,
                 tokens_per_expert.to(tl.int32),
                 mask=expert_mask)


@nvtx_range("[DG] preprocess_after_permute")
def preprocess_after_permute(expert_first_token_offset_tensor,
                             permuted_data_tensor):
    """
    Python wrapper that launches a single fused kernel to get the token-to-expert map
    and the number of tokens per expert.
    """
    total_tokens = permuted_data_tensor.shape[0]
    num_experts = expert_first_token_offset_tensor.shape[0] - 1

    # create output tensors
    masked_m = torch.empty(num_experts, dtype=torch.int32, device='cuda')
    token_to_expert_map = torch.empty(total_tokens,
                                      dtype=torch.int64,
                                      device='cuda')

    # calculate the grid size
    DEFAULT_BLOCK_SIZE_M = 256
    grid_m_size = triton.cdiv(total_tokens, DEFAULT_BLOCK_SIZE_M)
    if grid_m_size >= num_experts:
        BLOCK_SIZE_M = DEFAULT_BLOCK_SIZE_M
        grid = (grid_m_size, 2)
    else:
        block_size_m = triton.cdiv(total_tokens, num_experts)
        BLOCK_SIZE_M = triton.next_power_of_2(block_size_m)
        grid = (num_experts, 2)

    # launch the kernel
    _preprocess_after_permute_kernel[grid](
        expert_first_token_offset_tensor,
        masked_m,
        token_to_expert_map,
        total_tokens,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    return masked_m, token_to_expert_map


@nvtx_range("[DG]")
def deepgemm_fp8_group_blockwise_gemm(
    d: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> torch.Tensor:
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

    deep_gemm.fp8_m_grouped_gemm_nt_masked((a, sfa), (b, sfb),
                                           d,
                                           masked_m,
                                           expected_m,
                                           disable_ue8m0_cast=True)
    return


def set_strides(workspace: torch.Tensor, g: int, m: int, k: int):
    workspace = workspace[0:g * m * k]
    workspace = workspace.as_strided(
        size=(g, m, k),
        stride=(m * k, k, 1),
    )
    return workspace


class DeepGemmFusedMoE(CutlassFusedMoE):
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

    # To reuse pytorch memory segments allocated during graph capture.
    buffers = get_memory_buffers()

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
        if model_config.moe_max_num_tokens is None:
            moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size
            # The default moe_max_num_tokens is calculated from the following formula:
            # max_isl = 8196, max_batch_size = 1024, mtp = 0
            # max_num_tokens = ((mtp+1)*max_batch_size+max_isl+128+63)//64*64 = 9344
            # moe_max_num_tokens = max_num_tokens * 2 = 18688
            # It can avoid OOM for 8k/1k cases.
            default_moe_max_num_tokens = 18688
            if moe_max_num_tokens > default_moe_max_num_tokens:
                model_config._frozen = False
                model_config.moe_max_num_tokens = default_moe_max_num_tokens
                model_config._frozen = True

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

    def get_workspace(self, m_max: int, group_size: int):
        capture_graph = torch.cuda.is_current_stream_capturing()
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size_per_partition
        num_experts = self.expert_size_per_partition

        # create workspace
        fp8_dim = max(hidden_size, intermediate_size)
        workspace_0 = DeepGemmFusedMoE.buffers.get_buffer(
            (num_experts * m_max * fp8_dim, ),
            dtype=torch.float8_e4m3fn,
            buffer_name='workspace_0',
            reserve_buffer=capture_graph)
        workspace_1 = DeepGemmFusedMoE.buffers.get_buffer(
            (num_experts * m_max * max(intermediate_size * 2, hidden_size), ),
            dtype=torch.bfloat16,
            buffer_name='workspace_1',
            reserve_buffer=capture_graph)

        # create workspace for scaling factors
        m_padded = fp8_utils.align(m_max, 4)
        scale_k = fp8_utils.ceil_div(fp8_dim, group_size)
        scale_k_padded = fp8_utils.align(scale_k, 4)

        workspace_sf = DeepGemmFusedMoE.buffers.get_buffer(
            (num_experts * (scale_k_padded // 4) * m_padded, ),
            dtype=torch.int32,
            buffer_name='workspace_sf',
            reserve_buffer=capture_graph)

        workspace = {
            "workspace_0": workspace_0,
            "workspace_1": workspace_1,
            "workspace_sf": workspace_sf,
        }
        return workspace

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm()
            else:
                raise ValueError(
                    f"Unsupported quantization mode: {self.quant_config.quant_mode}"
                )
        else:
            return UnquantizedFusedMoEMethod()

    @nvtx_range("[DG] forward")
    def forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        workspace: Optional[dict] = None,
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

        expected_m = (token_selected_experts.numel() +
                      self.expert_size_per_partition -
                      1) // self.expert_size_per_partition

        # padding and quantization
        m_max = fp8_utils.align(x.shape[0], 128)
        act_input_fp8 = set_strides(workspace["workspace_0"],
                                    self.expert_size_per_partition, m_max,
                                    self.hidden_size)

        m_padded = fp8_utils.align(m_max, 4)
        scale_k = fp8_utils.ceil_div(self.hidden_size, 128)
        scale_k_padded = fp8_utils.align(scale_k, 4)
        act_input_sf = set_strides(workspace["workspace_sf"],
                                   self.expert_size_per_partition,
                                   scale_k_padded // 4, m_padded)

        act_input_sf = masked_index_copy_group_quant_fp8(
            act_input_fp8,
            act_input_sf,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            token_to_expert_map,
            group_size=128)

        # grouped gemm 1
        h1 = set_strides(workspace["workspace_1"],
                         self.expert_size_per_partition, m_max,
                         self.intermediate_size_per_partition * 2)

        deepgemm_fp8_group_blockwise_gemm(
            d=h1,
            a=act_input_fp8,
            b=self.w3_w1_weight,
            sfa=act_input_sf,
            sfb=self.quant_scales[0],
            masked_m=masked_m,
            expected_m=expected_m,
        )

        # activation and quantization
        act_input_fp8 = set_strides(workspace["workspace_0"],
                                    self.expert_size_per_partition, m_max,
                                    self.intermediate_size_per_partition)

        scale_k = fp8_utils.ceil_div(self.intermediate_size_per_partition, 128)
        scale_k_padded = fp8_utils.align(scale_k, 4)
        act_input_sf = set_strides(workspace["workspace_sf"],
                                   self.expert_size_per_partition,
                                   scale_k_padded // 4, m_padded)

        act_input_sf = fp8_utils.silu_and_mul_masked_post_quant_fwd(
            output=act_input_fp8,
            output_scale=act_input_sf,
            input=h1,
            quant_group_size=128,
            masked_m=masked_m,
            scale_ue8m0=True)

        # grouped gemm 2
        h3 = set_strides(workspace["workspace_1"],
                         self.expert_size_per_partition, m_max,
                         self.hidden_size)

        deepgemm_fp8_group_blockwise_gemm(
            d=h3,
            a=act_input_fp8,
            b=self.w2_weight,
            sfa=act_input_sf,
            sfb=self.quant_scales[1],
            masked_m=masked_m,
            expected_m=expected_m,
        )

        # gather and finalize
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
            x.shape[1],  # (possibly padded) hidden_size
            self.unpadded_hidden_size,  # original hidden size
            self.routing_method.top_k,
            self.expert_size_per_partition,  # num_experts_per_node
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
        )

        return final_hidden_states

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,  # used by other MoE backends
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert do_finalize, "CutlassFusedMoE does not support do_finalize=False"
        if self.use_dp and self.parallel_size > 1:
            assert all_rank_num_tokens is not None
            assert use_dp_padding is not None
            num_rows = sum(all_rank_num_tokens)
        else:
            num_rows = x.shape[0]

        # In case of num_rows is larger than max_chunk_size * 2, we need to split the input into multiple chunks.
        # Because we will use two streams in chunked moe and preallocate two workspaces.
        num_chunks = 1
        if num_rows > self.moe_max_num_tokens * 2:
            num_chunks = (num_rows + self.moe_max_num_tokens -
                          1) // self.moe_max_num_tokens

        if use_dp_padding:
            all_rank_num_tokens_padded = [max(all_rank_num_tokens)
                                          ] * len(all_rank_num_tokens)
        else:
            all_rank_num_tokens_padded = all_rank_num_tokens

        if num_chunks == 1:
            # create workspace
            num_rows = x.shape[0]
            if self.use_dp:
                num_rows = sum(all_rank_num_tokens_padded)
            m_max = fp8_utils.align(num_rows, 128)
            workspace = self.get_workspace(m_max, 128)
            outputs = self.forward_chunk(
                x,
                router_logits,
                output_dtype,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding,
                workspace=workspace)
            outputs = self.reducescatter_or_allreduce(
                outputs,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding)
        else:
            if self.use_dp:
                all_rank_chunk_size_list = [
                    self.split_chunk(val, num_chunks)
                    for val in all_rank_num_tokens_padded
                ]
                all_rank_num_tokens_list = [[
                    val[idx_chunk] for val in all_rank_chunk_size_list
                ] for idx_chunk in range(num_chunks)]
                chunk_size_list = all_rank_chunk_size_list[self.rank]
            else:
                all_rank_num_tokens_list = [None] * num_chunks
                chunk_size_list = self.split_chunk(x.shape[0], num_chunks)

            # create workspace
            chunk_size_0 = sum(all_rank_num_tokens_list[0]
                               ) if self.use_dp else chunk_size_list[0]
            chunk_size_1 = sum(all_rank_num_tokens_list[1]
                               ) if self.use_dp else chunk_size_list[1]
            workspace_0 = self.get_workspace(fp8_utils.align(chunk_size_0, 128),
                                             128)
            workspace_1 = self.get_workspace(fp8_utils.align(chunk_size_1, 128),
                                             128)

            x_list = x.split(chunk_size_list)
            router_logits_list = router_logits.split(chunk_size_list)

            self.event_dict[EventType.Main].record()
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.Main].wait()

            def _forward_chunk(x_, router_logits_, idx, workspace):
                return self.forward_chunk(
                    x_,
                    router_logits_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx]
                    if self.use_dp else None,
                    use_dp_padding=use_dp_padding,
                    workspace=workspace)

            def _reducescatter_or_allreduce(x_, idx):
                return self.reducescatter_or_allreduce(
                    x_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx],
                    use_dp_padding=use_dp_padding)

            outputs_list = []
            # Postpone reduce-scatter/all-reduce to the next iteration to achieve better overlap
            for idx_chunk, (x, router_logits) in enumerate(
                    zip(x_list, router_logits_list)):

                if idx_chunk % 2 == 0:
                    with torch.cuda.stream(self.aux_stream):
                        outputs = _forward_chunk(x, router_logits, idx_chunk,
                                                 workspace_0)
                    if idx_chunk > 0:
                        outputs_list[-1] = _reducescatter_or_allreduce(
                            outputs_list[-1], idx_chunk - 1)
                else:
                    outputs = _forward_chunk(x, router_logits, idx_chunk,
                                             workspace_1)
                    with torch.cuda.stream(self.aux_stream):
                        outputs_list[-1] = _reducescatter_or_allreduce(
                            outputs_list[-1], idx_chunk - 1)

                outputs_list.append(outputs)

            if num_chunks % 2 == 0:
                outputs_list[-1] = _reducescatter_or_allreduce(
                    outputs_list[-1], -1)
            else:
                with torch.cuda.stream(self.aux_stream):
                    outputs_list[-1] = _reducescatter_or_allreduce(
                        outputs_list[-1], -1)
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.MoeChunkingOverlap].record()
            self.event_dict[EventType.MoeChunkingOverlap].wait()

            outputs = torch.cat(outputs_list)

        if self.use_dp and self.parallel_size > 1:
            rank = self.mapping.tp_rank
            outputs = outputs[:all_rank_num_tokens[rank]]
        return outputs
