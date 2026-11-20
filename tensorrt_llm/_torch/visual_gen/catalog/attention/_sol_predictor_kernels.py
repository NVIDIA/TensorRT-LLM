# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _block_pool_kernel(
    x_ptr,
    out_ptr,
    seq_len,
    num_blocks,
    num_chunks,
    row_width,
    stride_x_batch,
    stride_x_token,
    stride_out_batch,
    stride_out_block,
    BLOCK: tl.constexpr,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    chunk = pid % num_chunks
    batch_block = pid // num_chunks
    block = batch_block % num_blocks
    batch = batch_block // num_blocks
    columns = chunk * WIDTH + tl.arange(0, WIDTH)
    in_row = columns < row_width
    first_token = block * BLOCK
    total = tl.zeros([WIDTH], dtype=tl.float32)
    for start in range(0, BLOCK, TOKENS):
        offsets = start + tl.arange(0, TOKENS)
        tokens = first_token + offsets
        values = tl.load(
            x_ptr + batch * stride_x_batch + tokens[:, None] * stride_x_token + columns[None, :],
            mask=((offsets < BLOCK) & (tokens < seq_len))[:, None] & in_row[None, :],
            other=0.0,
        )
        total += tl.sum(values.to(tl.float32), axis=0)
    total = total / tl.minimum(seq_len - first_token, BLOCK).to(tl.float32)
    tl.store(
        out_ptr + batch * stride_out_batch + block * stride_out_block + columns,
        total.to(out_ptr.dtype.element_ty),
        mask=in_row,
    )


@triton.jit
def _block_statistics_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    moment_ptr,
    num_blocks,
    num_heads,
    stride_x_batch,
    stride_x_block,
    stride_x_head,
    stride_s_batch,
    stride_s_head,
    stride_m_batch,
    stride_m_head,
    EXACT: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    cols = chunk * COLS + tl.arange(0, COLS)
    dims = tl.arange(0, HEAD_DIM)
    base = x_ptr + batch * stride_x_batch + head * stride_x_head
    total = tl.zeros([COLS], dtype=tl.float32)
    total_sq = tl.zeros([COLS], dtype=tl.float32)
    if EXACT:
        moment = tl.zeros([HEAD_DIM, COLS], dtype=tl.float32)
    for start in range(0, num_blocks, ROWS):
        rows = start + tl.arange(0, ROWS)
        row_valid = rows < num_blocks
        tile = tl.load(
            base + rows[:, None] * stride_x_block + cols[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )
        values = tile.to(tl.float32)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
        if EXACT:
            full = tl.load(
                base + rows[:, None] * stride_x_block + dims[None, :],
                mask=row_valid[:, None],
                other=0.0,
            )
            moment += tl.dot(tl.trans(full), tile)
    count = num_blocks.to(tl.float32)
    mean = total / count
    stats = batch * stride_s_batch + head * stride_s_head + cols
    tl.store(mean_ptr + stats, mean)
    tl.store(var_ptr + stats, tl.maximum(total_sq / count - mean * mean, 0.0))
    if EXACT:
        tl.store(
            moment_ptr
            + batch * stride_m_batch
            + head * stride_m_head
            + dims[:, None] * HEAD_DIM
            + cols[None, :],
            moment / count,
        )


@triton.jit
def _block_thresholds_kernel(
    centroid_ptr,
    mean_ptr,
    var_ptr,
    moment_ptr,
    threshold_ptr,
    num_q_blocks,
    num_heads,
    tau,
    log2_scale,
    epsilon,
    stride_c_batch,
    stride_c_block,
    stride_c_head,
    stride_s_batch,
    stride_s_head,
    stride_m_batch,
    stride_m_head,
    stride_t_batch,
    stride_t_head,
    EXACT: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    q_blocks = tile * Q_BLOCKS + tl.arange(0, Q_BLOCKS)
    q_valid = q_blocks < num_q_blocks
    dims = tl.arange(0, HEAD_DIM)
    centroid = tl.load(
        centroid_ptr
        + batch * stride_c_batch
        + q_blocks[:, None] * stride_c_block
        + head * stride_c_head
        + dims[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    stats = batch * stride_s_batch + head * stride_s_head + dims
    mean = tl.load(mean_ptr + stats)
    projected_mean = tl.sum(centroid * mean[None, :], axis=1)
    if EXACT:
        moment = tl.load(
            moment_ptr
            + batch * stride_m_batch
            + head * stride_m_head
            + dims[:, None] * HEAD_DIM
            + dims[None, :]
        )
        projected = tl.dot(centroid, moment, input_precision="ieee")
        projected_var = tl.maximum(
            tl.sum(projected * centroid, axis=1) - projected_mean * projected_mean, 0.0
        )
    else:
        var = tl.load(var_ptr + stats)
        projected_var = tl.sum(centroid * centroid * var[None, :], axis=1)
    threshold = projected_mean * log2_scale + tau * tl.sqrt(
        projected_var * (log2_scale * log2_scale) + epsilon
    )
    tl.store(
        threshold_ptr + batch * stride_t_batch + head * stride_t_head + q_blocks,
        threshold,
        mask=q_valid,
    )


@triton.jit
def _select_exact_blocks_kernel(
    centroid_ptr,
    keys_ptr,
    threshold_ptr,
    bits_ptr,
    num_q_blocks,
    num_kv_blocks,
    num_words,
    num_heads,
    local_radius,
    log2_scale,
    stride_c_batch,
    stride_c_block,
    stride_c_head,
    stride_k_batch,
    stride_k_block,
    stride_k_head,
    stride_t_batch,
    stride_t_head,
    stride_b_batch,
    stride_b_head,
    stride_b_block,
    Q_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1).to(tl.int64)
    batch = batch_head // num_heads
    head = batch_head % num_heads
    q_blocks = tile * Q_BLOCKS + tl.arange(0, Q_BLOCKS)
    q_valid = q_blocks < num_q_blocks
    dims = tl.arange(0, HEAD_DIM)

    centroid = tl.load(
        centroid_ptr
        + batch * stride_c_batch
        + q_blocks[:, None] * stride_c_block
        + head * stride_c_head
        + dims[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    threshold = tl.load(
        threshold_ptr + batch * stride_t_batch + head * stride_t_head + q_blocks,
        mask=q_valid,
        other=0.0,
    )

    # Three BF16 terms retain the FP32 query centroid for tensor-core products.
    high = centroid.to(keys_ptr.dtype.element_ty)
    rest = centroid - high.to(tl.float32)
    mid = rest.to(keys_ptr.dtype.element_ty)
    low = (rest - mid.to(tl.float32)).to(keys_ptr.dtype.element_ty)

    lanes = tl.arange(0, 32)
    lane_bits = 1 << lanes.to(tl.int64)
    for word in range(num_words):
        kv_blocks = word * 32 + lanes
        kv_valid = kv_blocks < num_kv_blocks
        keys = tl.load(
            keys_ptr
            + batch * stride_k_batch
            + kv_blocks[:, None] * stride_k_block
            + head * stride_k_head
            + dims[None, :],
            mask=kv_valid[:, None],
            other=0.0,
        )
        keys_t = tl.trans(keys)
        scores = (tl.dot(high, keys_t) + tl.dot(mid, keys_t) + tl.dot(low, keys_t)) * log2_scale
        distance = q_blocks[:, None] - kv_blocks[None, :]
        is_local = (distance >= -local_radius) & (distance <= local_radius)
        exact = kv_valid[None, :] & ((scores > threshold[:, None]) | is_local)
        packed = tl.sum(tl.where(exact, lane_bits[None, :], 0), axis=1)
        tl.store(
            bits_ptr
            + batch * stride_b_batch
            + head * stride_b_head
            + q_blocks * stride_b_block
            + word,
            packed.to(tl.int32),
            mask=q_valid,
        )


def sol_predictor_kernel(q, k, v, block_size, tau, sm_scale, thresh_type):
    batch, tokens, heads, dim = q.shape
    blocks = triton.cdiv(tokens, block_size)
    words = triton.cdiv(blocks, 32)
    shape = (batch, blocks, heads, dim)
    queries = torch.empty(shape, dtype=torch.float32, device=q.device)
    keys = k.new_empty(shape)
    values = v.new_empty(shape)

    width = min(1024, triton.next_power_of_2(heads * dim))
    chunks = triton.cdiv(heads * dim, width)
    for source, summary in ((q, queries), (k, keys), (v, values)):
        _block_pool_kernel[(batch * blocks * chunks,)](
            source,
            summary,
            tokens,
            blocks,
            chunks,
            heads * dim,
            *source.stride()[:2],
            *summary.stride()[:2],
            BLOCK=block_size,
            TOKENS=8,
            WIDTH=width,
            num_warps=4,
        )

    log2_scale = sm_scale * math.log2(math.e)
    exact = thresh_type == "exact"
    mean = queries.new_empty((batch, heads, dim))
    variance = torch.empty_like(mean)
    moment = queries.new_empty((batch, heads, dim, dim)) if exact else mean
    _block_statistics_kernel[(dim // 64, batch * heads)](
        keys,
        mean,
        variance,
        moment,
        blocks,
        heads,
        *keys.stride()[:3],
        *mean.stride()[:2],
        *moment.stride()[:2],
        EXACT=exact,
        ROWS=32,
        COLS=64,
        HEAD_DIM=dim,
        num_warps=4,
    )
    threshold = queries.new_empty((batch, heads, blocks))
    _block_thresholds_kernel[(triton.cdiv(blocks, 32), batch * heads)](
        queries,
        mean,
        variance,
        moment,
        threshold,
        blocks,
        heads,
        tau,
        log2_scale,
        1.0e-6,
        *queries.stride()[:3],
        *mean.stride()[:2],
        *moment.stride()[:2],
        *threshold.stride()[:2],
        EXACT=exact,
        Q_BLOCKS=32,
        HEAD_DIM=dim,
        num_warps=4,
    )
    bits = torch.empty((batch, heads, blocks, words), device=q.device, dtype=torch.uint32)
    _select_exact_blocks_kernel[(triton.cdiv(blocks, 64), batch * heads)](
        queries,
        keys,
        threshold,
        bits.view(torch.int32),
        blocks,
        blocks,
        words,
        heads,
        1,
        log2_scale,
        *queries.stride()[:3],
        *keys.stride()[:3],
        *threshold.stride()[:2],
        *bits.stride()[:3],
        Q_BLOCKS=64,
        HEAD_DIM=dim,
        num_warps=4,
    )
    return bits, keys, values
