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


@triton.jit
def _resmooth_kernel(
    w_ptr,
    s_ptr,
    M,
    K,
    stride_wb,
    stride_wm,
    stride_wk,
    stride_sb,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    curr_w_ptr = w_ptr + batch_idx * stride_wb
    curr_s_ptr = s_ptr + batch_idx * stride_sb

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    s_offset = pid_m * stride_sm + pid_k * stride_sk
    old_scale = tl.load(curr_s_ptr + s_offset)

    w_mask = (rm[:, None] < M) & (rk[None, :] < K)
    w_offsets = rm[:, None] * stride_wm + rk[None, :] * stride_wk
    w_fp8 = tl.load(curr_w_ptr + w_offsets, mask=w_mask, other=0.0)
    w_fp32 = w_fp8.to(tl.float32)

    w_val = w_fp32 * old_scale
    block_amax = tl.maximum(tl.max(tl.abs(w_val)), 1e-4)

    # UE8M0 sf = 2 ^ ceil(log2(sf))
    new_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(block_amax / 448.0)))
    w_requant = w_val * (1.0 / new_scale)

    tl.store(curr_w_ptr + w_offsets, w_requant, mask=w_mask)
    tl.store(curr_s_ptr + s_offset, new_scale)


def resmooth_to_fp8_e8m0(
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: tuple[int, int] = (128, 128),
):
    assert weight.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.float32

    weight = weight.cuda()
    weight_scale = weight_scale.cuda()

    orig_shape = weight.shape
    M, K = orig_shape[-2:]
    w_view = weight.view(-1, M, K)
    s_view = weight_scale.view(-1, weight_scale.shape[-2],
                               weight_scale.shape[-1])

    num_batches = w_view.shape[0]
    BLOCK_M, BLOCK_K = block_size

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_k = triton.cdiv(K, BLOCK_K)
    blocks_per_batch = grid_m * grid_k

    # Workaround for Triton compiler/runtime bug on SM100f (Blackwell):
    # CUDA illegal memory access when total grid blocks exceed ~65K.
    # Split launches along the batch dimension to stay under the limit.
    MAX_GRID_BLOCKS = 65536
    if blocks_per_batch > 0:
        max_batches_per_launch = max(1, MAX_GRID_BLOCKS // blocks_per_batch)
    else:
        max_batches_per_launch = num_batches

    for batch_offset in range(0, num_batches, max_batches_per_launch):
        batch_count = min(max_batches_per_launch, num_batches - batch_offset)
        grid = (batch_count, grid_m, grid_k)

        _resmooth_kernel[grid](
            w_view[batch_offset:],
            s_view[batch_offset:],
            M,
            K,
            w_view.stride(0),
            w_view.stride(1),
            w_view.stride(2),
            s_view.stride(0),
            s_view.stride(1),
            s_view.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
    # this is an in-place operation, however, we return for simplicity
    return weight, weight_scale


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


def unpack_col_major_tma_aligned_packed_tensor(
    packed: torch.Tensor,
    mn: int,
    k: int,
) -> torch.Tensor:
    """Inverse of :func:`get_col_major_tma_aligned_packed_tensor`.

    Recovers a ``(mn, k)`` float32 UE8M0 scale tensor from the packed
    int32 col-major layout produced by the forward transform.

    Only valid when the original scales were UE8M0 (all power-of-two),
    which is guaranteed after :func:`resmooth_to_fp8_e8m0`.
    """
    assert packed.dtype == torch.int

    remove_dim = False
    if packed.dim() == 2:
        packed = packed.unsqueeze(0)
        remove_dim = True
    b = packed.shape[0]

    aligned_mn = get_tma_aligned_size(mn, 4)  # int32 element_size = 4
    aligned_k = align(k, 4)

    # The packed tensor may have col-major strides from the forward
    # transform.  Flatten to 1-D via clone() so dtype views always work.
    packed.shape[-2] * packed.shape[-1]
    flat_int = packed.reshape(b, -1).clone()  # (b, mn * cols) contiguous

    # Pad mn back to aligned_mn (forward sliced [:mn])
    if mn < aligned_mn:
        extra = (aligned_mn - mn) * packed.shape[-1]
        pad = torch.zeros(b, extra, device=packed.device, dtype=torch.int)
        flat_int = torch.cat([flat_int, pad], dim=1)
        aligned_mn * packed.shape[-1]

    # int32 → uint8: 4 bytes per element
    flat_bytes = flat_int.view(torch.uint8)  # (b, n_int32 * 4)
    unpacked = flat_bytes.view(b, aligned_mn, aligned_k)

    # Slice away padding, reconstruct float32 from UE8M0 exponent byte
    ue8m0 = unpacked[:, :mn, :k]  # (b, mn, k) uint8
    float_bits = ue8m0.to(torch.int32) << 23
    result = float_bits.reshape(b, -1).view(torch.float32).reshape(b, mn, k)

    return result.squeeze(0) if remove_dim else result


def inverse_transform_sf(
    sf: torch.Tensor,
    mn: int,
    k: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Recover a ``(nb_m, nb_k)`` float32 block-scale grid from a packed SF.

    This reverses the ``(FP32, 128, 128) → (INT, 1, 128)`` path in
    :func:`transform_sf_into_required_layout` (the path taken by
    ``FP8BlockScalesLinearMethod.post_load_weights`` on SM100f / SM120).
    """
    nb_k = ceil_div(k, block_size)

    # Step 1: unpack to per-row float32 UE8M0 grid  (mn, nb_k)
    per_row = unpack_col_major_tma_aligned_packed_tensor(sf, mn, nb_k)

    # Step 2: collapse replicated rows back to (nb_m, nb_k).
    # Forward did index_select with indices = arange(mn) // block_size,
    # so rows within a 128-block are identical.  Take one per block.
    per_block = per_row[::block_size]  # (nb_m, nb_k)
    return per_block


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
    HAS_SWIGLU_LIMIT: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
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
            if HAS_SWIGLU_LIMIT:
                # gate is fp32; clamp directly. up is in input dtype (bf16/fp16);
                # cast the limit constant to that dtype to avoid a fp32 round-trip
                # on every element.
                gate = tl.minimum(gate, SWIGLU_LIMIT)
                limit_native = tl.cast(SWIGLU_LIMIT, input_ptr.dtype.element_ty)
                neg_limit_native = tl.cast(-SWIGLU_LIMIT,
                                           input_ptr.dtype.element_ty)
                up = tl.maximum(tl.minimum(up, limit_native), neg_limit_native)
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
    swiglu_limit: Optional[float] = None,
):
    """
    input shape [g, m, k]
    output shape [g, m, k // 2], dtype fp8
    output_scale [g, k // 4, m // 2 // 128], dtype int32
    quant_group_size int
    masked_m shape [g]
    swiglu_limit optional Python float (uniform across experts); when provided,
        the gate input is clamped to (-inf, limit] before silu and the up
        branch is clamped to [-limit, limit] before the multiply. Baked into
        the kernel as a constexpr — caller supplies a host-side float, no
        per-expert tensor / global load.
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
    has_swiglu_limit = swiglu_limit is not None
    swiglu_limit_value = float(swiglu_limit) if has_swiglu_limit else 0.0
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
        HAS_SWIGLU_LIMIT=has_swiglu_limit,
        SWIGLU_LIMIT=swiglu_limit_value,
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
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    token_num_cur_expert,
    size_k,
    fp8_max,
    fp8_min,
    BLOCK: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    batch_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_output_scale_0 = tl.cast(stride_output_scale_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)
    stride_output_scale_1 = tl.cast(stride_output_scale_1, dtype=tl.int64)
    stride_input_2 = tl.cast(stride_input_2, dtype=tl.int64)
    stride_output_2 = tl.cast(stride_output_2, dtype=tl.int64)
    stride_output_scale_2 = tl.cast(stride_output_scale_2, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK + tl.arange(0, BLOCK // 4)
    input_ptr_offs = input_ptr + batch_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + batch_id * stride_output_0 + offs_in_d
    output_scale_offs = (output_scale_ptr + batch_id * stride_output_scale_0 +
                         hidden_dim_block_index * stride_output_scale_1)

    for token_index in tl.range(token_id,
                                token_num_cur_expert,
                                block_num_per_expert,
                                num_stages=NUM_STAGE):
        output_s_int32 = 0
        for pack_index in tl.range(4):
            local_mask = offs_in_d + pack_index * 128
            act = tl.load(
                input_ptr_offs + token_index * stride_input_1 +
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
                output_ptr_offs + token_index * stride_output_1 +
                pack_index * 128,
                output_q,
                mask=local_mask < size_k,
            )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_2,
            output_s_int32,
        )


def per_token_quant_and_transform(
    input: torch.Tensor,
    quant_group_size: int = 128,
    scale_ue8m0: bool = True,
    need_permute102: bool = False,
):
    """
    input shape [g, m, k]
    output shape [g, m, k // 2], dtype fp8
    output_scale [g, k // 4, m // 2 // 128], dtype int32
    quant_group_size int
    masked_m shape [g]
    """

    assert input.shape[-1] % 2 == 0

    # FP8 quantization parameters
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    b = 1
    original_input_rank = len(input.shape)
    if (original_input_rank == 2):
        assert input.is_contiguous()
        input = input.unsqueeze(0)
        b, m, k = input.shape
    elif (original_input_rank == 3):
        if need_permute102:
            input = input.transpose(0, 1)
        b, m, k = input.shape
    else:
        raise AssertionError(
            f"Unsupported input shape rank: {original_input_rank}")

    # Create output
    output = torch.empty((b, m, k), dtype=torch.float8_e4m3fn, device="cuda")

    # Create output scale
    alignment = 4
    scale_k = ceil_div(k, quant_group_size)
    m_padded = align(m, alignment)
    scale_k_padded = align(scale_k, alignment)
    output_scale = torch.empty((b, scale_k_padded // 4, m_padded),
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
        b,
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
    if (original_input_rank == 2):
        output = output.squeeze(0)
        output_scale = output_scale.squeeze(0)
        output_scale = output_scale.transpose(0, 1)[:m, :]
    else:
        output_scale = output_scale.transpose(1, 2)[:, :m, :]

    check_sf_layout(
        output_scale,
        m,
        k,
        (1, 128),
        num_groups=b if original_input_rank == 3 else None,
        tma_stride_check=True,
    )
    return output, output_scale


def fp8_quantize_1x128_sf_transpose(
        x: torch.Tensor,
        use_ue8m0: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    x_fp8, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x, use_ue8m0=use_ue8m0)
    if x_scale.ndim == 1:  # Handle SM version differences (SM90: 1D padded, SM100+: 2D)
        x_padded = (x.shape[0] + 3) // 4 * 4
        num_blocks = (x.shape[1] + 127) // 128
        x_scale = x_scale[:x_padded * num_blocks].view(num_blocks,
                                                       x_padded)[:, :x.shape[0]]
    x_scale = x_scale.contiguous().transpose(0, 1)
    return x_fp8, x_scale
