from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import nvtx_range


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


@nvtx_range("[DG] quantization")
@torch.compile(dynamic=True)
def per_token_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        assert x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
            m, n), sf
    else:
        assert x.size(2) % 128 == 0
        g, m, n = x.shape
        x_view = x.view(g, m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=3).view(g, m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(3))).to(torch.float8_e4m3fn).view(
            g, m, n), sf


def per_block_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        m, n = x.shape
        x_padded = torch.zeros((align(m, 128), align(n, 128)),
                               dtype=x.dtype,
                               device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
            x_view.size(0), x_view.size(2))
    else:
        g, m, n = x.shape
        x_padded = torch.zeros((g, align(m, 128), align(n, 128)),
                               dtype=x.dtype,
                               device=x.device)
        x_padded[:, :m, :n] = x
        x_view = x_padded.view(g, -1, 128, x_padded.size(-1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), sf.view(
            x_view.size(0), x_view.size(1), x_view.size(3))


def resmooth_to_fp8_e8m0(weight: torch.Tensor,
                         sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = weight.cuda()
    sf = sf.cuda()
    if weight.dim() == 2:
        x = weight.float() * sf.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1)[:weight.shape[0], :weight.shape[1]]
    else:
        x = weight.float() * sf.repeat_interleave(128, dim=1).repeat_interleave(
            128, dim=2)[:weight.shape[0], :weight.shape[1], :weight.shape[2]]
    return per_block_cast_to_fp8_e8m0(x)


def get_m_alignment_for_contiguous_layout():
    return 128


def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return align(x, alignment)


def get_col_major_tma_aligned_packed_tensor(x: torch.Tensor) -> torch.Tensor:
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k),
                         device=x.device,
                         dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn,
                                                        aligned_k // 4)

    # Finally, transpose
    transposed = torch.transpose(
        torch.empty((b, aligned_k // 4, aligned_mn),
                    device=x.device,
                    dtype=torch.int), 1, 2)
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def check_sf_layout(sf: torch.Tensor,
                    mn: int,
                    k: int,
                    gran: Tuple[int, int],
                    num_groups: Optional[int],
                    tma_stride_check: bool = False,
                    type_check: Optional[torch.dtype] = None) -> torch.Tensor:
    # Type check
    if type_check is not None:
        assert sf.dtype == type_check

    # Always do shape checks
    assert sf.dtype in (torch.float, torch.int)
    assert sf.dim() == int(num_groups is not None) + 2
    if num_groups is not None:
        assert sf.size(-3) == num_groups
    assert sf.size(-2) == ceil_div(mn, gran[0])
    assert sf.size(-1) == ceil_div(
        k, gran[1] * (1 if sf.dtype == torch.float else 4))

    # TMA stride checks: TMA aligned and MN-major
    if tma_stride_check:
        if num_groups is not None:
            assert sf.stride(-3) == sf.stride(-1) * sf.size(-1)
        assert sf.stride(-2) == 1
        assert sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size())

    return sf


@nvtx_range("[DG] transform_sf_into_required_layout")
def transform_sf_into_required_layout(sf: torch.Tensor,
                                      mn: int,
                                      k: int,
                                      recipe: Tuple[int, int, int],
                                      num_groups: Optional[int] = None,
                                      is_sfa: bool = False):
    gran = (recipe[0 if is_sfa else 1], recipe[2])

    should_skip_transform = ((sf.dtype == torch.int and gran == (1, 128))
                             or (sf.dtype == torch.int and gran == (128, 128)))

    if not should_skip_transform:
        # Pre-transform checks
        check_sf_layout(sf, mn=mn, k=k, gran=gran, num_groups=num_groups)

    # (FP32, 1, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (1, 128):
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=(1, 128),
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    # (FP32, 128, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (128, 128):
        sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=(1, 128),
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    if should_skip_transform:
        # TODO: add transpose kernel if SF layout is not satisfied
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=(1, 128),
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    assert False, f'Unknown cases: {sf.dtype=}, {gran=}'


# copy from https://github.com/ModelTC/lightllm/blob/a000ab69098654df4731f5b12587dd4e7f0a4f41/lightllm/common/fused_moe/moe_silu_and_mul_mix_quant_ep.py
@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_k,
    fp8_max,
    fp8_min,
    BLOCK: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK + tl.arange(0, BLOCK // 4)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (output_scale_ptr + expert_id * stride_output_scale_0 +
                         hidden_dim_block_index * stride_output_scale_1)

    for token_index in tl.range(token_id,
                                token_num_cur_expert,
                                block_num_per_expert,
                                num_stages=NUM_STAGE):
        output_s_int32 = 0
        for pack_index in tl.range(4):
            local_mask = offs_in_d + pack_index * 128
            up = tl.load(
                input_ptr_offs + token_index * stride_input_1 +
                pack_index * 128,
                mask=local_mask < size_k,
                other=0.0,
            )
            gate = tl.load(
                input_ptr_offs + token_index * stride_input_1 + size_k +
                pack_index * 128,
                mask=local_mask < size_k,
                other=0.0,
            ).to(tl.float32)
            gate = gate / (1 + tl.exp(-gate))
            gate = gate.to(input_ptr.dtype.element_ty)
            gate_up = up * gate
            _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
            output_s = _absmax / fp8_max
            if SCALE_UE8M0:
                output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
            output_q = tl.clamp(gate_up / output_s, fp8_min,
                                fp8_max).to(output_ptr.dtype.element_ty)
            output_s_int32 += ((output_s.to(tl.int32, bitcast=True) >> 23) <<
                               (8 * pack_index))
            tl.store(
                output_ptr_offs + token_index * stride_output_1 +
                pack_index * 128,
                output_q,
                mask=local_mask < size_k,
            )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_2,
            output_s_int32,
        )


def silu_and_mul_masked_post_quant_fwd(
    output: torch.Tensor,
    output_scale: torch.Tensor,
    input: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
):
    """
    input shape [g, m, k]
    output shape [g, m, k // 2], dtype fp8
    output_scale [g, k // 4, m // 2 // 128], dtype int32
    quant_group_size int
    masked_m shape [g]
    """

    assert input.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = finfo.min

    g, m, k = input.shape
    k = k // 2

    # Get block/grid/stage/warp
    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 128

    BLOCK = quant_group_size * 4
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(k, BLOCK)
    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )
    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        k,
        fp8_max,
        fp8_min,
        BLOCK=BLOCK,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    output_scale = output_scale.transpose(1, 2)[:, :m, :]
    check_sf_layout(
        output_scale,
        m,
        k,
        (1, 128),
        g,
        tma_stride_check=True,
    )
    return output_scale


@triton.jit
def _per_token_quant_and_transform_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    output_ptr,
    stride_output_0,
    stride_output_1,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    token_num_cur_expert,
    size_k,
    fp8_max,
    fp8_min,
    BLOCK: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK + tl.arange(0, BLOCK // 4)
    input_ptr_offs = input_ptr + offs_in_d
    output_ptr_offs = output_ptr + offs_in_d
    output_scale_offs = (output_scale_ptr +
                         hidden_dim_block_index * stride_output_scale_0)

    for token_index in tl.range(token_id,
                                token_num_cur_expert,
                                block_num_per_expert,
                                num_stages=NUM_STAGE):
        output_s_int32 = 0
        for pack_index in tl.range(4):
            local_mask = offs_in_d + pack_index * 128
            act = tl.load(
                input_ptr_offs + token_index * stride_input_0 +
                pack_index * 128,
                mask=local_mask < size_k,
                other=0.0,
            ).to(tl.float32)
            _absmax = tl.maximum(tl.max(tl.abs(act)), 1e-10)
            output_s = _absmax / fp8_max
            if SCALE_UE8M0:
                output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
            output_q = tl.clamp(act / output_s, fp8_min,
                                fp8_max).to(output_ptr.dtype.element_ty)
            output_s_int32 += ((output_s.to(tl.int32, bitcast=True) >> 23) <<
                               (8 * pack_index))
            tl.store(
                output_ptr_offs + token_index * stride_output_0 +
                pack_index * 128,
                output_q,
                mask=local_mask < size_k,
            )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s_int32,
        )


# TODO: Add more comments and tests for this function for future reuse
def per_token_quant_and_transform(
    input: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
):
    """
    input shape [g, m, k]
    output shape [g, m, k // 2], dtype fp8
    output_scale [g, k // 4, m // 2 // 128], dtype int32
    quant_group_size int
    masked_m shape [g]
    """

    assert input.is_contiguous()
    assert len(input.shape) == 2
    assert input.shape[-1] % 2 == 0

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    m, k = input.shape

    # Create output
    output = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")

    # Create output scale
    alignment = 4
    scale_k = ceil_div(k, quant_group_size)
    m_padded = align(m, alignment)
    scale_k_padded = align(scale_k, alignment)
    output_scale = torch.empty((scale_k_padded // 4, m_padded),
                               dtype=torch.int32,
                               device='cuda')

    # Get block/grid/stage/warp
    BLOCK_NUM_PER_EXPERT = 64

    BLOCK = quant_group_size * 4
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(k, BLOCK)
    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        1,
    )
    _per_token_quant_and_transform_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        m,
        k,
        fp8_max,
        fp8_min,
        BLOCK=BLOCK,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    output_scale = output_scale.transpose(0, 1)[:m, :]
    check_sf_layout(
        output_scale,
        m,
        k,
        (1, 128),
        num_groups=None,
        tma_stride_check=True,
    )
    return output, output_scale
