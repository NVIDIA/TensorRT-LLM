"""Multi-head attention kernel that can operate with kv-caches."""

import triton
from triton import language as tl


@triton.jit
def update_kv_cache(
    k_ptr,  # [B*S, N, D]
    v_ptr,  # [B*S, N, D]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    input_pos_ptr,  # Specifies the sequence index in the caches at which to write the provided kv
    cache_loc_ptr,  # Specifies the batch index for each of the input sequences
    MAX_SEQ_LENGTH: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    Q_D_HEAD: tl.constexpr,
    V_D_HEAD: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    GENERATE_ONLY: tl.constexpr,
):
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    if GENERATE_ONLY:
        seq_start_index = batch_id
        seq_len: tl.constexpr = 1
    else:
        seq_start_index = tl.load(seq_start_indices_ptr + batch_id)
        seq_len = tl.load(seq_len_ptr + batch_id)

    # cache is [bsnd]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)

    kv_position = tl.load(input_pos_ptr + batch_id)

    K_D_HEAD: tl.constexpr = Q_D_HEAD
    k_cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH * K_D_HEAD
    v_cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH * V_D_HEAD

    k_dhead_offsets = tl.arange(0, triton.next_power_of_2(K_D_HEAD))
    k_dhead_mask = k_dhead_offsets < K_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    k_load_mask = seq_mask[:, None] * k_dhead_mask[None, :]
    v_load_mask = seq_mask[:, None] * v_dhead_mask[None, :]

    k_batch_offset = seq_start_index * N_KV_HEADS * K_D_HEAD
    v_batch_offset = seq_start_index * N_KV_HEADS * V_D_HEAD
    # Write back to kv-caches
    ks = tl.load(
        k_ptr
        + k_batch_offset
        + seq_offsets[:, None] * N_KV_HEADS * K_D_HEAD
        + head_id * K_D_HEAD
        + k_dhead_offsets[None, :],
        mask=k_load_mask,
    )
    vs = tl.load(
        v_ptr
        + v_batch_offset
        + seq_offsets[:, None] * N_KV_HEADS * V_D_HEAD
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        mask=v_load_mask,
    )

    kv_writeback_seq_offsets = seq_offsets + kv_position

    k_cache_offset = (
        k_cache_batch_offset
        + kv_writeback_seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
        + head_id * K_D_HEAD
        + k_dhead_offsets[None, :]
    )

    v_cache_offset = (
        v_cache_batch_offset
        + kv_writeback_seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :]
    )
    tl.store(k_cache_ptr + k_cache_offset, ks, k_load_mask)
    tl.store(v_cache_ptr + v_cache_offset, vs, v_load_mask)


@triton.jit
def gqa_attention_kv_stage1(
    q_ptr,  # [Batch, 1, N_HEADS, D_HEAD]
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    cache_loc_ptr,  # [Batch] # Specifies the batch index for each of the generate tokens.
    input_pos_ptr,  # [Batch]
    output_values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    output_logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    num_blocks,
    SCALE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,  # Maximum supported sequence length
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    Q_D_HEAD: tl.constexpr,  # Dimension of each query head.
    V_D_HEAD: tl.constexpr,  # Dimension of each key/value head
    SEQ_BLOCK_SIZE: tl.constexpr,  # Block size used for tiling the sequence dim.
    HEAD_BLOCK_SIZE: tl.constexpr,  # pad to 16 if HEAD_RATIO is < 16 to invoke tensor cores.
    SLIDING_WINDOW: tl.constexpr,
):
    """Attention kernel to be used for generate-only batches.

    Specialized for GQA.

    Assumes that kv caches have been updated.

    Supports non-power-of-2 D_HEAD

    Uses flash decoding.
    KV-cache layout is assumed to be [Batch, Seq, Head, Dim]
    1. Fetch the K-cache from 0 to input_pos
    2. Fetch the V-cache from 0 to input_pos
    3. A = Q*K^T [1,D_HEAD] * [1,seq_len,D_HEAD] -> [1, seq_len]
    4. S = softmax(A)
    5. O = S*V [1, seq_len] * [1, seq_len, D_HEAD] -> [1, D_HEAD]
    """
    # Assume KV-cache layout: [Batch, Seq, Head, Dim]
    # A program is responsible for 1 batch, 1 head and a block of sequences.
    batch_id = tl.program_id(axis=0)
    kv_head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    kv_position = tl.load(input_pos_ptr + batch_id)
    kv_batch_id = tl.load(cache_loc_ptr + batch_id)
    K_D_HEAD: tl.constexpr = Q_D_HEAD
    batch_offset = kv_batch_id * N_KV_HEADS * MAX_SEQ_LEN

    # Offsets for the block of sequences this program processes.
    seq_start_pos = seq_block_id * SEQ_BLOCK_SIZE

    # The number of Q heads that map to each KV head.
    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS  # This needs to be a power-of-2

    # Apply sliding window constraints
    if SLIDING_WINDOW > 0:
        # For sliding window, limit the sequence range
        sliding_start = tl.maximum(0, kv_position - SLIDING_WINDOW + 1)
        if seq_start_pos + SEQ_BLOCK_SIZE <= sliding_start or seq_start_pos > kv_position:
            return
        seq_offsets = seq_start_pos + tl.arange(0, SEQ_BLOCK_SIZE)
        seq_mask = (seq_offsets <= kv_position) & (seq_offsets >= sliding_start)
    else:
        if seq_start_pos > kv_position:
            return
        seq_offsets = seq_start_pos + tl.arange(0, SEQ_BLOCK_SIZE)
        seq_mask = seq_offsets <= kv_position

    # Need to pad the head dim to 16 if HEAD_RATIO is < 16 so that tensor cores can be invoked
    #
    head_offsets = kv_head_id * HEAD_RATIO + tl.arange(0, HEAD_BLOCK_SIZE)
    head_mask = head_offsets < (kv_head_id * HEAD_RATIO + HEAD_RATIO)
    # Assuming D_HEAD is a power of 2
    q_dhead_offsets = tl.arange(0, triton.next_power_of_2(Q_D_HEAD))
    q_dhead_mask = q_dhead_offsets < Q_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    # Program loads the entire Q for the head assigned to it.
    # [NUM_HEADS, Q_D_HEAD]
    q_batch_offset = batch_id * N_HEADS * Q_D_HEAD
    q_head_offsets = head_offsets * Q_D_HEAD

    # Q layout : BSND
    q = tl.load(
        q_ptr + q_batch_offset + q_head_offsets[:, None] + q_dhead_offsets[None, :],
        mask=head_mask[:, None] * q_dhead_mask[None, :],
        other=0.0,
    )

    # [BSND]
    k_block_offsets = (
        batch_offset * K_D_HEAD
        + seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
        + kv_head_id * K_D_HEAD
        + q_dhead_offsets[None, :]
    )
    k_mask = seq_mask[:, None] * q_dhead_mask[None, :]  # K and Q share the same head dim
    k = tl.load(k_cache_ptr + k_block_offsets, mask=k_mask, other=0.0)

    v_block_offsets = (
        batch_offset * V_D_HEAD
        + seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
        + kv_head_id * V_D_HEAD
        + v_dhead_offsets[None, :]
    )
    v_mask = seq_mask[:, None] * v_dhead_mask[None, :]

    # [seq_block, V_D_HEAD]
    v = tl.load(v_cache_ptr + v_block_offsets, mask=v_mask, other=0.0)

    # Note: check the output precision of the sum.
    # compute q*K^T
    # [NUM_HEADS, Q_D_HEAD] * [seq_block, Q_D_HEAD], sum along axis 1
    attn = tl.dot(q, k.trans())  # [N, seq_block]
    attn = attn.to(tl.float32)
    attn *= SCALE
    # Set to -inf attn values where mask is not set. This forces exp(attn) to 0.
    attn = tl.where(head_mask[:, None] * seq_mask[None, :], attn, float("-inf"))
    # compute max_attn only when invalid attn values are masked out.
    max_attn = tl.max(attn, axis=1)  # [N, 1]

    exp_attn = tl.exp(attn - max_attn[:, None])
    sumexp = tl.sum(exp_attn, axis=1)  # [N, 1]

    # [NUM_HEADS, seq_len] * [seq_len, V_D_HEAD], sum along axis 0
    output = tl.dot(exp_attn.to(v.dtype), v)

    output = output / sumexp[:, None]  # [N, D_HEAD]

    # We store the log-sum-exp after removing the max.
    logsumexp = tl.log(sumexp) + max_attn
    # when seq_mask is all false, max_attn will be -inf and sumexp is zero

    tl.store(
        output_values_ptr
        + batch_id * N_HEADS * V_D_HEAD * num_blocks
        + head_offsets[:, None] * V_D_HEAD * num_blocks
        + seq_block_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        output,
        mask=head_mask[:, None] * v_dhead_mask[None, :],
    )
    tl.store(
        output_logsumexp_ptr
        + batch_id * N_HEADS * num_blocks
        + head_offsets * num_blocks
        + seq_block_id,
        logsumexp,
        mask=head_mask,
    )


@triton.jit
def attention_kv_stage1(
    q_ptr,  # [Batch, 1, N_HEADS, D_HEAD]
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    cache_loc_ptr,  # [Batch] # Specifies the batch index for each of the generate tokens.
    input_pos_ptr,  # [Batch]
    output_values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    output_logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    num_blocks,
    MAX_SEQ_LEN: tl.constexpr,  # Maximum supported sequence length
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    SEQ_BLOCK_SIZE: tl.constexpr,  # Block size used for tiling the sequence dim.
):
    """Attention kernel to be used for generate-only batches.

    Assumes that kv caches have been updated.

    Uses flash decoding.
    KV-cache layout is assumed to be [Batch,Seq, Head, Dim]
    1. Fetch the K-cache from 0 to input_pos
    2. Fetch the V-cache from 0 to input_pos
    3. A = Q*K^T [1,D_HEAD] * [1,seq_len,D_HEAD] -> [1, seq_len]
    4. S = softmax(A)
    5. O = S*V [1, seq_len] * [1, seq_len, D_HEAD] -> [1, D_HEAD]
    """
    # Assume KV-cache layout: [Batch, Seq, Head, Dim]
    # A program is responsible for 1 batch, 1 head and a block of sequences.
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)
    epsilon: tl.constexpr = 1e-38  # float32 smallest positive number

    kv_position = tl.load(input_pos_ptr + batch_id)
    kv_batch_id = tl.load(cache_loc_ptr + batch_id)
    kv_batch_offset = kv_batch_id * N_KV_HEADS * MAX_SEQ_LEN * D_HEAD
    # Offsets for the block of sequences this program processes.
    seq_start_pos = seq_block_id * SEQ_BLOCK_SIZE

    if seq_start_pos > kv_position:
        return
    seq_offsets = seq_start_pos + tl.arange(0, SEQ_BLOCK_SIZE)
    seq_mask = seq_offsets <= kv_position
    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, triton.next_power_of_2(D_HEAD))
    dhead_mask = dhead_offsets < D_HEAD

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    kv_head_offset = (head_id // HEAD_RATIO) * D_HEAD

    sm_scale: tl.constexpr = 1.0 / (D_HEAD**0.5)

    # Program loads the entire Q for the head assigned to it.
    # [D_HEAD]
    q_batch_offset = batch_id * N_HEADS * D_HEAD
    q_head_offset = head_id * D_HEAD
    q = tl.load(q_ptr + q_batch_offset + q_head_offset + dhead_offsets, mask=dhead_mask)

    kv_block_offsets = (
        kv_batch_offset
        + seq_offsets[:, None] * D_HEAD * N_KV_HEADS
        + kv_head_offset
        + dhead_offsets[None, :]
    )  # [BSND]
    kv_mask = seq_mask[:, None] * dhead_mask[None, :]

    # [seq_block, D_HEAD]
    k = tl.load(k_cache_ptr + kv_block_offsets, mask=kv_mask, other=0.0)
    v = tl.load(v_cache_ptr + kv_block_offsets, mask=kv_mask, other=0.0)

    # Note: check the output precision of the sum.
    # compute q*K^T
    # [D_HEAD] * [seq_block, D_HEAD], sum along axis 1
    attn = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1)  # [seq_block]

    attn *= sm_scale
    max_attn = tl.max(attn)
    # Set to -inf attn values where mask is not set. This forces exp(attn) to 0.
    attn = tl.where(seq_mask, attn, float("-inf"))
    exp_attn = tl.exp(attn - max_attn)
    exp_attn = tl.where(exp_attn == 0, epsilon, exp_attn)
    sumexp = tl.sum(exp_attn, axis=0)  # scalar.

    # [seq_len] * [seq_len, D_HEAD], sum along axis 0
    output = tl.sum(exp_attn[:, None] * v, axis=0)  # [D_HEAD]

    output = output / sumexp

    # We store the log-sum-exp after removing the max.
    logsumexp = tl.log(sumexp) + max_attn
    # when seq_mask is all false, max_attn will be -inf and sumexp is zero

    tl.store(
        output_values_ptr
        + batch_id * N_HEADS * D_HEAD * num_blocks
        + head_id * D_HEAD * num_blocks
        + seq_block_id * D_HEAD
        + dhead_offsets,
        output,
        mask=dhead_mask,
    )
    tl.store(
        output_logsumexp_ptr
        + batch_id * N_HEADS * num_blocks
        + head_id * num_blocks
        + seq_block_id,
        logsumexp,
    )


@triton.jit
def attention_kv_stage2(
    values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    output_ptr,  # [Batch, N_HEADS, D_HEAD]
    input_pos_ptr,
    NUM_BLOCKS: tl.constexpr,
    N_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    SEQ_BLOCK_SIZE: tl.constexpr,  # Nearest power of 2 for num_blocks
    HAS_SINKS: tl.constexpr,
    sinks_ptr,
):
    # There are batch * N_HEADS programs
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    dhead_offsets = tl.arange(0, triton.next_power_of_2(D_HEAD))
    dhead_mask = dhead_offsets < D_HEAD

    kv_position = tl.load(input_pos_ptr + batch_id)
    block_id = kv_position // SEQ_BLOCK_SIZE + 1

    NUM_BLOCKS_POW2: tl.constexpr = triton.next_power_of_2(NUM_BLOCKS)
    block_offsets = tl.arange(0, NUM_BLOCKS_POW2)

    block_mask = block_offsets < block_id
    logsumexp = tl.load(
        logsumexp_ptr + batch_id * N_HEADS * NUM_BLOCKS + head_id * NUM_BLOCKS + block_offsets,
        mask=block_mask,
        other=float("-inf"),
    )
    max_logsumexp = tl.max(logsumexp)
    sumexp = tl.exp(logsumexp - max_logsumexp)  # [NUM_BLOCKS_POW2]

    aggregate_sumexp = tl.sum(sumexp, axis=0)
    # Add sinks contribution to the softmax denominator
    if HAS_SINKS:
        sinks_val = tl.load(sinks_ptr + batch_id * N_HEADS + head_id)
        sinks_exp = tl.exp(sinks_val - max_logsumexp)
        aggregate_sumexp += sinks_exp

    values_offsets = block_offsets[:, None] * D_HEAD + dhead_offsets[None, :]
    values_mask = block_mask[:, None] * dhead_mask[None, :]

    values = tl.load(
        values_ptr
        + batch_id * N_HEADS * D_HEAD * NUM_BLOCKS
        + head_id * D_HEAD * NUM_BLOCKS
        + values_offsets,
        mask=values_mask,
        other=0.0,
    )  # [BLOCK_SIZE, D_HEAD]
    values *= sumexp[:, None]
    values /= aggregate_sumexp

    output = tl.sum(values, axis=0)  # [DHEAD]

    tl.store(
        output_ptr + batch_id * N_HEADS * D_HEAD + head_id * D_HEAD + dhead_offsets,
        output,
        mask=dhead_mask,
    )


@triton.jit
def context_attention_kv(
    q_ptr,  # [bsnd]
    k_ptr,  # [bsnd]
    v_ptr,  # [bsnd]
    k_cache_ptr,  # [bsnd]
    v_cache_ptr,  # [bsnd]
    seq_len,
    o_ptr,
    SCALE: tl.constexpr,
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    Q_D_HEAD: tl.constexpr,  # Dimension of each query head.
    V_D_HEAD: tl.constexpr,  # Dimension of each value head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
):
    """Kernel for context phase.

    Assuming:
    1. Self-attention [seqlen(Q) == seqlen(K)]
    2. Causal attention
    3. QKV layout: [bsnd]
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    K_D_HEAD: tl.constexpr = Q_D_HEAD

    q_dhead_offsets = tl.arange(0, triton.next_power_of_2(Q_D_HEAD))
    q_dhead_mask = q_dhead_offsets < Q_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    q_load_mask = seq_mask[:, None] * q_dhead_mask[None, :]

    q_batch_offset = batch_id * seq_len * N_HEADS
    kv_batch_offset = batch_id * seq_len * N_KV_HEADS

    k_head_offset = (head_id // HEAD_RATIO) * K_D_HEAD
    v_head_offset = (head_id // HEAD_RATIO) * V_D_HEAD

    # Q will stay in SRAM
    q = tl.load(
        q_ptr
        + q_batch_offset * Q_D_HEAD
        + seq_offsets[:, None] * N_HEADS * Q_D_HEAD
        + head_id * Q_D_HEAD
        + q_dhead_offsets[None, :],
        mask=q_load_mask,
    )
    acc = tl.zeros([SEQ_BLOCK, triton.next_power_of_2(V_D_HEAD)], dtype=tl.float32)
    lse_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")

    for s in range(0, seq_block_id + 1, 1):
        kv_seq_offsets = s * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
        kv_seq_mask = kv_seq_offsets < seq_len
        k_load_mask = kv_seq_mask[:, None] * q_dhead_mask[None, :]

        k = tl.load(
            k_ptr
            + kv_batch_offset * K_D_HEAD
            + kv_seq_offsets[:, None] * N_KV_HEADS * K_D_HEAD
            + k_head_offset
            + q_dhead_offsets[None, :],
            mask=k_load_mask,
        )
        qk = tl.zeros([SEQ_BLOCK, SEQ_BLOCK], dtype=tl.float32)
        qk += tl.dot(q, k.trans())
        # causal mask
        qk = tl.where(seq_offsets[:, None] >= kv_seq_offsets[None, :], qk, float("-inf"))
        qk *= SCALE
        # rowmax
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])  # [S,S]
        v = tl.load(
            v_ptr
            + kv_batch_offset * V_D_HEAD
            + kv_seq_offsets[:, None] * N_KV_HEADS * V_D_HEAD
            + v_head_offset
            + v_dhead_offsets[None, :],
            mask=kv_seq_mask[:, None] * v_dhead_mask[None, :],
        )

        l_ij = tl.sum(p, 1)
        acc_scale = tl.exp(m_i - m_ij)
        acc = acc * acc_scale[:, None]
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)

    acc = acc * o_scale[:, None]

    tl.store(
        o_ptr
        + batch_id * seq_len * N_HEADS * V_D_HEAD
        + seq_offsets[:, None] * N_HEADS * V_D_HEAD
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        acc,
        mask=seq_mask[:, None] * v_dhead_mask[None, :],
    )

    # Write back to kv-caches

    ks = tl.load(
        k_ptr
        + kv_batch_offset * K_D_HEAD
        + seq_offsets[:, None] * N_KV_HEADS * K_D_HEAD
        + k_head_offset
        + q_dhead_offsets[None, :],
        mask=seq_mask[:, None] * q_dhead_mask[None, :],
    )
    vs = tl.load(
        v_ptr
        + kv_batch_offset * V_D_HEAD
        + seq_offsets[:, None] * N_KV_HEADS * V_D_HEAD
        + v_head_offset
        + v_dhead_offsets[None, :],
        mask=seq_mask[:, None] * v_dhead_mask[None, :],
    )
    # cache is [bsnd]
    k_cache_offset = (
        batch_id * N_KV_HEADS * MAX_SEQ_LENGTH * K_D_HEAD
        + seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
        + k_head_offset
        + q_dhead_offsets[None, :]
    )

    v_cache_offset = (
        batch_id * N_KV_HEADS * MAX_SEQ_LENGTH * V_D_HEAD
        + seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
        + v_head_offset
        + v_dhead_offsets[None, :]
    )
    tl.store(k_cache_ptr + k_cache_offset, ks, seq_mask[:, None] * q_dhead_mask[None, :])
    tl.store(v_cache_ptr + v_cache_offset, vs, seq_mask[:, None] * v_dhead_mask[None, :])


@triton.jit
def context_attention_kv_flattened(
    q_ptr,  # [b*s,nd]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [bsnd]
    v_cache_ptr,  # [bsnd]
    input_pos_ptr,  # [b] # specifies the location in the sequence where kv must be written back.
    cache_loc_ptr,  # [b] # location of the sequence in the cache.
    o_ptr,
    SCALE: tl.constexpr,
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    Q_D_HEAD: tl.constexpr,  # Dimension of each query head.
    V_D_HEAD: tl.constexpr,  # Dimension of each value head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,  # Sliding window size, -1 means no sliding window
    HAS_SINKS: tl.constexpr,
    sinks_ptr,
):
    """Kernel for context phase.

    Assumes that kv caches have been updated.
    Assuming QKV layout: [b*s,n,d]
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    seq_start_index = tl.load(seq_start_indices_ptr + batch_id)
    seq_len = tl.load(seq_len_ptr + batch_id)
    K_D_HEAD: tl.constexpr = Q_D_HEAD
    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS

    # cache is [bsnd]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)

    cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH
    cache_head_offset = head_id // HEAD_RATIO

    q_dhead_offsets = tl.arange(0, triton.next_power_of_2(Q_D_HEAD))
    q_dhead_mask = q_dhead_offsets < Q_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    # Q will stay in SRAM
    q = tl.load(
        q_ptr
        + seq_start_index * N_HEADS * Q_D_HEAD
        + seq_offsets[:, None] * N_HEADS * Q_D_HEAD
        + head_id * Q_D_HEAD
        + q_dhead_offsets[None, :],
        mask=seq_mask[:, None] * q_dhead_mask[None, :],
    )

    acc = tl.zeros([SEQ_BLOCK, triton.next_power_of_2(V_D_HEAD)], dtype=tl.float32)
    lse_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")

    # Loop over the entire KV-history
    # input_pos_ptr stores the location at which kv must be written back for the given batch.
    kv_position = tl.load(input_pos_ptr + batch_id)
    num_blocks = (kv_position + seq_len + SEQ_BLOCK - 1) // SEQ_BLOCK
    start = 0
    if SLIDING_WINDOW > 0:
        # Use the LAST query in this block for more conservative start calculation
        last_q_pos = (
            (seq_block_id + 1) * SEQ_BLOCK - 1 + kv_position
        )  # Last query's absolute position
        earliest_kv_pos = max(0, last_q_pos - SLIDING_WINDOW + 1)
        start = max(0, earliest_kv_pos // SEQ_BLOCK)
    for s in range(start, num_blocks + 1):
        kv_seq_offsets = s * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
        kv_seq_mask = kv_seq_offsets < (kv_position + seq_len)

        k = tl.load(
            k_cache_ptr
            + cache_batch_offset * K_D_HEAD
            + kv_seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
            + cache_head_offset * K_D_HEAD
            + q_dhead_offsets[None, :],
            mask=kv_seq_mask[:, None] * q_dhead_mask[None, :],
        )
        qk = tl.zeros([SEQ_BLOCK, SEQ_BLOCK], dtype=tl.float32)
        qk += tl.dot(q, k.trans())
        # Apply causal mask
        causal_mask = (seq_offsets[:, None] + kv_position) >= kv_seq_offsets[None, :]
        # Apply sliding window mask if enabled
        if SLIDING_WINDOW > 0:
            sliding_window_mask = kv_seq_offsets[None, :] >= (
                seq_offsets[:, None] + kv_position - SLIDING_WINDOW + 1
            )
            combined_mask = sliding_window_mask & causal_mask
        else:
            combined_mask = causal_mask
        qk = tl.where(combined_mask, qk, float("-inf"))
        qk *= SCALE
        # rowmax
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
        v = tl.load(
            v_cache_ptr
            + cache_batch_offset * V_D_HEAD
            + kv_seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
            + cache_head_offset * V_D_HEAD
            + v_dhead_offsets[None, :],
            mask=kv_seq_mask[:, None] * v_dhead_mask[None, :],
        )

        l_ij = tl.sum(p, 1)
        acc_scale = tl.exp(m_i - m_ij)
        acc = acc * acc_scale[:, None]
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # Add sinks contribution to the final softmax calculation
    if HAS_SINKS:
        sinks_val = tl.load(sinks_ptr + batch_id * N_HEADS + head_id)
        m_sinks = tl.maximum(m_i, sinks_val)
        acc_scale = tl.exp(m_i - m_sinks)
        acc = acc * acc_scale[:, None]
        l_sinks = tl.exp(lse_i - m_sinks) + tl.exp(sinks_val - m_sinks)
        lse_i = m_sinks + tl.log(l_sinks)
        m_i = m_sinks

    o_scale = tl.exp(m_i - lse_i)

    acc = acc * o_scale[:, None]

    tl.store(
        o_ptr
        + seq_start_index * N_HEADS * V_D_HEAD
        + seq_offsets[:, None] * N_HEADS * V_D_HEAD
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        acc,
        mask=seq_mask[:, None] * v_dhead_mask[None, :],
    )


@triton.jit
def update_kv_cache_rope_fusion(
    q_ptr,  # [B*S, N, D]
    k_ptr,  # [B*S, N, D]
    v_ptr,  # [B*S, N, D]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    q_rope_ptr,  # [B*S, N, D], roped q result
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    input_pos_ptr,  # Specifies the sequence index in the caches at which to write the provided kv
    cache_loc_ptr,  # Specifies the batch index for each of the input sequences
    f_ptr,  # [MAX_SEQ_LEN, D_HEAD//2, 2] # frequencies for rope embadding.
    MAX_SEQ_LENGTH: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,  # pad to 16 if HEAD_RATIO is < 16 to invoke tensor cores.
    GENERATE_ONLY: tl.constexpr,
):
    """Fuse q and k rope with update_kv_cache kernel.

    The input is interleaved as [2, D//2] in D_HEAD dim.
    Update q_rope with the post-rope-embadding q values.
    Update k_cache with the post-rope-embadding k values.
    For rope computation, q and k need to load and store in tensors pair of 2 * [D//2].
    Update v_cache with v.
    """
    batch_id = tl.program_id(axis=0)
    kv_head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    if GENERATE_ONLY:
        seq_start_index = batch_id
        seq_len: tl.constexpr = 1
    else:
        seq_start_index = tl.load(seq_start_indices_ptr + batch_id)
        seq_len = tl.load(seq_len_ptr + batch_id)

    # cache is [bsnd]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)

    kv_position = tl.load(input_pos_ptr + batch_id)

    cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH * D_HEAD
    cache_head_offset = kv_head_id * D_HEAD

    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    load_mask = seq_mask[:, None] * dhead_mask[None, :]

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS  # This needs to be a power-of-2
    q_head_offsets = kv_head_id * HEAD_RATIO + tl.arange(0, HEAD_BLOCK_SIZE)
    q_head_mask = q_head_offsets < (kv_head_id * HEAD_RATIO + HEAD_RATIO)

    q_batch_offset = seq_start_index * N_HEADS * D_HEAD

    kv_batch_offset = seq_start_index * N_KV_HEADS * D_HEAD
    kv_head_offset = cache_head_offset

    D2: tl.constexpr = D_HEAD // 2
    # input is interleaved as [2, D//2] in dim [D_HEAD].
    d2_offsets = tl.arange(0, D2)
    dhead_offsets1 = d2_offsets
    dhead_offsets2 = d2_offsets + D2
    d2_mask = dhead_offsets2 < D_HEAD
    d2_load_mask = seq_mask[:, None] * d2_mask[None, :]

    # offsets of [bsn]
    q_offsets_base = (
        q_batch_offset
        + seq_offsets[:, None, None] * N_HEADS * D_HEAD
        + q_head_offsets[None, :, None] * D_HEAD
    )
    q_offsets1 = q_offsets_base + dhead_offsets1[None, None, :]
    q_offsets2 = q_offsets_base + dhead_offsets2[None, None, :]
    q_mask = d2_load_mask[:, None, :] * q_head_mask[None, :, None]

    q1 = tl.load(q_ptr + q_offsets1, mask=q_mask).to(tl.float32)
    q2 = tl.load(q_ptr + q_offsets2, mask=q_mask).to(tl.float32)

    k_offsets_base = kv_batch_offset + seq_offsets[:, None] * N_KV_HEADS * D_HEAD + kv_head_offset
    k_offsets1 = k_offsets_base + dhead_offsets1[None, :]
    k_offsets2 = k_offsets_base + dhead_offsets2[None, :]

    k1 = tl.load(k_ptr + k_offsets1, mask=d2_load_mask).to(tl.float32)
    k2 = tl.load(k_ptr + k_offsets2, mask=d2_load_mask).to(tl.float32)

    # -----------------------------------
    # torch version sin/cos
    # cos and sin values are interleaved in frequencies tensor.
    f_offsets = seq_offsets[:, None] * D2 + d2_offsets[None, :]
    cos_ref = tl.load(f_ptr + kv_position * D_HEAD + f_offsets * 2, mask=d2_load_mask).to(
        dtype=tl.float32
    )
    sin_ref = tl.load(f_ptr + kv_position * D_HEAD + f_offsets * 2 + 1, mask=d2_load_mask).to(
        dtype=tl.float32
    )

    qs1 = cos_ref[:, None, :] * q1 - sin_ref[:, None, :] * q2
    qs2 = sin_ref[:, None, :] * q1 + cos_ref[:, None, :] * q2

    tl.store(q_rope_ptr + q_offsets1, qs1, mask=q_mask)
    tl.store(q_rope_ptr + q_offsets2, qs2, mask=q_mask)

    ks1 = cos_ref * k1 - sin_ref * k2
    ks2 = sin_ref * k1 + cos_ref * k2

    # Write back to kv-caches
    vs = tl.load(
        v_ptr
        + kv_batch_offset
        + seq_offsets[:, None] * N_KV_HEADS * D_HEAD
        + kv_head_offset
        + dhead_offsets[None, :],
        mask=load_mask,
    )

    kv_writeback_seq_offsets = seq_offsets + kv_position

    cache_offset_base = (
        cache_batch_offset
        + kv_writeback_seq_offsets[:, None] * D_HEAD * N_KV_HEADS
        + cache_head_offset
    )

    k_cache_offset1 = cache_offset_base + dhead_offsets1[None, :]
    k_cache_offset2 = cache_offset_base + dhead_offsets2[None, :]
    tl.store(k_cache_ptr + k_cache_offset1, ks1, mask=d2_load_mask)
    tl.store(k_cache_ptr + k_cache_offset2, ks2, mask=d2_load_mask)

    v_cache_offset = cache_offset_base + dhead_offsets[None, :]
    tl.store(v_cache_ptr + v_cache_offset, vs, load_mask)
