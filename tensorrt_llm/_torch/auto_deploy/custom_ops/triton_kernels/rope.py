import triton
import triton.language as tl


@triton.jit
def rope_fwd_kernel(
    x_ptr,
    input_pos_ptr,
    f_ptr,
    output_ptr,
    N,
    L,
    H,
    D: tl.constexpr,
    stride_n,
    stride_l,
    stride_h,
    stride_d,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    """Grids(N, H // BLOCK_SIZE_H, BLOCK_SIZE_L).

    Each block gets 2 blocks of (1, H, BLOCK_SIZE_L, D) input data.
    """
    D2: tl.constexpr = D // 2
    D2_PADDED: tl.constexpr = triton.next_power_of_2(D2)
    batch = tl.program_id(0)
    x_ptr += batch * stride_n
    output_ptr += batch * stride_n

    # frequencies tensor is not sliced.
    # layout: [1,max_seq_len,D//2,2]
    input_offset = tl.load(input_pos_ptr + batch) * D
    head_offsets = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    head_mask = head_offsets < H

    # input is interleaved as [2,D//2].
    col_offsets1 = tl.arange(0, D2_PADDED)
    col_mask1 = col_offsets1 < D2
    col_offsets2 = col_offsets1 + D2
    col_mask2 = col_offsets2 < D

    row_start = tl.program_id(2) * BLOCK_SIZE_L
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_L)
    row_mask = row_offsets < L

    offsets1 = (
        head_offsets[:, None, None] * stride_h
        + row_offsets[None, :, None] * stride_l
        + col_offsets1[None, None, :] * stride_d
    )
    offsets2 = (
        head_offsets[:, None, None] * stride_h
        + row_offsets[None, :, None] * stride_l
        + col_offsets2[None, None, :] * stride_d
    )

    nohead_mask1 = row_mask[None, :, None] * col_mask1[None, None, :]
    nohead_mask2 = row_mask[None, :, None] * col_mask2[None, None, :]

    mask1 = head_mask[:, None, None] * nohead_mask1
    mask2 = head_mask[:, None, None] * nohead_mask2

    a = tl.load(x_ptr + offsets1, mask=mask1).to(tl.float32)
    b = tl.load(x_ptr + offsets2, mask=mask2).to(tl.float32)
    # -----------------------------------
    # torch version sin/cos
    # cos and sin values are interleaved in frequencies tensor.
    col_offsets = tl.arange(0, D2_PADDED)
    offsets = row_offsets[None, :, None] * D2 + col_offsets[None, None, :]
    cos_ref = tl.load(f_ptr + input_offset + offsets * 2, mask=nohead_mask1).to(dtype=tl.float32)
    sin_ref = tl.load(f_ptr + input_offset + offsets * 2 + 1, mask=nohead_mask2).to(
        dtype=tl.float32
    )

    y1 = cos_ref * a - sin_ref * b
    y2 = sin_ref * a + cos_ref * b

    # -----------------------------------
    # triton version sin/cos
    # m = row_offsets + 1.
    # theta = tl.exp(-2. * (col_start + tl.arange(0, BLOCK_SIZE_D)) / D * LOG_BASE)
    # mtheta = m[None, :, None] * theta[None, None, :]
    # cos = tl.cos(mtheta)
    # sin = tl.sin(mtheta)

    # y1 = cos * a - sin * b
    # y2 = sin * a + cos * b
    # -----------------------------------

    tl.store(output_ptr + offsets1, y1, mask=mask1)
    tl.store(output_ptr + offsets2, y2, mask=mask2)


@triton.jit
def rope_fwd_flattened_kernel(
    x_ptr,  # [B*S, N, D]
    seq_lens_ptr,  # [B]
    seq_start_indices_ptr,  # [B]
    input_pos_ptr,  # [B]
    f_ptr,
    output_ptr,
    H: tl.constexpr,  # number of heads
    D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    """Rope that works with flattened Q/K."""
    D2: tl.constexpr = D // 2
    D2_PADDED: tl.constexpr = triton.next_power_of_2(D2)
    batch = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + batch)
    seq_start_index = tl.load(seq_start_indices_ptr + batch)

    x_ptr += seq_start_index * H * D
    output_ptr += seq_start_index * H * D

    # frequencies tensor is not sliced.
    # layout: [1,max_seq_len,D//2,2]
    input_offset = tl.load(input_pos_ptr + batch) * D
    head_offsets = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    head_mask = head_offsets < H

    # input is interleaved as [2,D//2].
    col_offsets1 = tl.arange(0, D2_PADDED)
    col_mask1 = col_offsets1 < D2
    col_offsets2 = col_offsets1 + D2
    col_mask2 = col_offsets2 < D

    row_start = tl.program_id(2) * BLOCK_SIZE_L
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_L)
    row_mask = row_offsets < seq_len

    offsets1 = (
        head_offsets[:, None, None] * D
        + row_offsets[None, :, None] * D * H
        + col_offsets1[None, None, :]
    )
    offsets2 = (
        head_offsets[:, None, None] * D
        + row_offsets[None, :, None] * D * H
        + col_offsets2[None, None, :]
    )

    nohead_mask1 = row_mask[None, :, None] * col_mask1[None, None, :]
    nohead_mask2 = row_mask[None, :, None] * col_mask2[None, None, :]

    mask1 = head_mask[:, None, None] * nohead_mask1
    mask2 = head_mask[:, None, None] * nohead_mask2

    a = tl.load(x_ptr + offsets1, mask=mask1).to(tl.float32)
    b = tl.load(x_ptr + offsets2, mask=mask2).to(tl.float32)
    # -----------------------------------
    # torch version sin/cos
    # cos and sin values are interleaved in frequencies tensor.
    col_offsets = tl.arange(0, D2_PADDED)
    offsets = row_offsets[None, :, None] * D2 + col_offsets[None, None, :]
    cos_ref = tl.load(f_ptr + input_offset + offsets * 2, mask=nohead_mask1).to(dtype=tl.float32)
    sin_ref = tl.load(f_ptr + input_offset + offsets * 2 + 1, mask=nohead_mask2).to(
        dtype=tl.float32
    )

    y1 = cos_ref * a - sin_ref * b
    y2 = sin_ref * a + cos_ref * b

    # -----------------------------------
    # triton version sin/cos
    # m = row_offsets + 1.
    # theta = tl.exp(-2. * (col_start + tl.arange(0, BLOCK_SIZE_D)) / D * LOG_BASE)
    # mtheta = m[None, :, None] * theta[None, None, :]
    # cos = tl.cos(mtheta)
    # sin = tl.sin(mtheta)

    # y1 = cos * a - sin * b
    # y2 = sin * a + cos * b
    # -----------------------------------

    tl.store(output_ptr + offsets1, y1, mask=mask1)
    tl.store(output_ptr + offsets2, y2, mask=mask2)
