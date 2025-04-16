import triton
from triton import language as tl

"""
Kernels based on paged KV Cache.
Parameter infos:
    tensors:
    - q: [b*s, n, d], flattened queries.
    - k/v: [b*s, n, d], flattened key/value.
    - seq_len: [b], length of each sequence in the batch.
        `seq_len` can be 1 (generate) or larger (context).
    - seq_start: [b], start index of each sequence in b*s dim of q/k/v.
    - k_cache/v_cache: [num_pages, PAGE_SIZE, n, d], paged KV Cache.
        New-coming k/v is split into small group of PAGE_SIZE, and then
        mapped to incontinuous memory in KV Cache.
    - page_table: [b, max_num_pages_per_seq], mapping logic of each sequence.
    - cache_loc: [b], mapping logic of `batch_id` in q/k/v to index in `page_table`.
    - cache_len: [b], existing cached k/v length of each sequence.

    constexpr:
    - N_HEADS/N_KV_HEADS: shape of dim [n] in q or k/v.
    - D_HEAD: shape of dim [d] in q/k/v.
        Assuming power of 2.
    - SEQ_BLOCK: block size to split dim [s].
        Assuming power of 2.
        Split k/v in update kernel and split q in context/generate kernel.
    - MAX_SEQ_LENGTH: seq_len <= MAX_SEQ_LENGTH.
    - PAGE_SIZE: shape of each kv cache page,
        Assuming power of 2 and SEQ_BLOCK % PAGE_SIZE = 0.
    - PAGE_TABLE_STIDE: stride of dim [b] in `page_table`.

KV Cache access logic in update kernel:
    1. batch_id i access k[seq_start[i] : seq_start[i] + seq_len[i]]
        and can be split into pages [a:b] in the sequence.
    2. Look up cache_len[i] to find if the sequence has cached k/v.
    3. Look up page_table[cache_loc[i], cache_len[i] + a : cache_len[i] + b]
       to get the corresponding pages in the k_cache, with result [c:d].
    4. Then update k_cache[c:d] with the k value.

"""


@triton.jit
def update_paged_kv_cache(
    k_ptr,  # [B*S, N, D]
    v_ptr,  # [B*S, N, D]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [num_pages, page_size, n, d]
    v_cache_ptr,  # [num_pages, page_size, n, d]
    cache_loc_ptr,  # [b] # index of the sequence in the page table.
    cache_len_ptr,  # [b] # length of the sequence already in kv cache.
    page_table_ptr,  # [b, max_num_pages_per_seq] # loc of the block page in the cache.
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_STRIDE: tl.constexpr,
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

    cache_len = tl.load(cache_len_ptr + batch_id)

    # cache is [num_pages, page_size, n, d]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)
    cache_head_offset = head_id * D_HEAD

    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    load_mask = seq_mask[:, None] * dhead_mask[None, :]

    kv_batch_offset = seq_start_index * N_KV_HEADS * D_HEAD
    kv_head_offset = cache_head_offset

    # Write back to kv-caches
    ks = tl.load(
        k_ptr
        + kv_batch_offset
        + seq_offsets[:, None] * N_KV_HEADS * D_HEAD
        + kv_head_offset
        + dhead_offsets[None, :],
        mask=load_mask,
    )
    vs = tl.load(
        v_ptr
        + kv_batch_offset
        + seq_offsets[:, None] * N_KV_HEADS * D_HEAD
        + kv_head_offset
        + dhead_offsets[None, :],
        mask=load_mask,
    )

    # assuming SEQ_BLOCK can be divided by PAGE_SIZE and PAGE_SIZE is a power of 2.
    SEQ_BLOCK_PAGE: tl.constexpr = SEQ_BLOCK // PAGE_SIZE
    MAX_NUM_PAGES: tl.constexpr = (MAX_SEQ_LENGTH + PAGE_SIZE - 1) // PAGE_SIZE
    # cache_len // PAGE_SIZE means history pages
    # if decode sequence, then seq_len = 1 and only seq_block_id = 0 works,
    kv_pages = seq_block_id * SEQ_BLOCK_PAGE + tl.arange(0, SEQ_BLOCK_PAGE) + cache_len // PAGE_SIZE
    cache_pages = tl.load(
        page_table_ptr + cache_loc * PAGE_TABLE_STRIDE + kv_pages, mask=kv_pages < MAX_NUM_PAGES
    )

    page_offsets = tl.arange(0, PAGE_SIZE)
    # shape [SEQ_BLOCK], means [cache_pages, page_offsets]
    cache_seq_offset = tl.reshape(
        cache_pages[:, None] * PAGE_SIZE + page_offsets[None, :], [SEQ_BLOCK]
    )
    # write offset inside the page
    cache_seq_offset += cache_len % PAGE_SIZE

    cache_offsets = (
        cache_seq_offset[:, None] * N_KV_HEADS * D_HEAD + kv_head_offset + dhead_offsets[None, :]
    )
    tl.store(k_cache_ptr + cache_offsets, ks, load_mask)
    tl.store(v_cache_ptr + cache_offsets, vs, load_mask)


# TODO: Write a doc describing the 2 stage algorithm
@triton.jit
def attention_kv_paged_stage1(
    q_ptr,  # [Batch, 1, N_HEADS, D_HEAD]
    k_cache_ptr,  # [NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD]
    v_cache_ptr,  # [NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD]
    cache_loc_ptr,  # [Batch] # Specifies the batch index for each of the generate tokens.
    page_table_ptr,  # [Batch, num_pages_per_seq]
    cache_len_ptr,  # [Batch] # Number of tokens in kv cache.
    output_values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    output_logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    num_blocks,
    MAX_SEQ_LEN: tl.constexpr,  # Maximum supported sequence length
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    # Block size used for tiling the sequence dim.
    SEQ_BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_STRIDE: tl.constexpr,
):
    """Attention kernel to be used during the generate phase.

    Uses flash decoding.
    KV-cache layout is assumed to be [Batch, Head, Seq, Dim]
    1. Fetch the K-cache from 0 to input_pos
    2. Fetch the V-cache from 0 to input_pos
    3. A = Q*K^T [1,D_HEAD] * [1,seq_len,D_HEAD] -> [1, seq_len]
    4. S = softmax(A)
    5. O = S*V [1, seq_len] * [1, seq_len, D_HEAD] -> [1, D_HEAD]
    """
    # Assume KV-cache layout: [Batch, Head, Seq, Dim]
    # A program is responsible for 1 batch, 1 head and a block of sequences.
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    SEQ_BLOCK_PAGE: tl.constexpr = SEQ_BLOCK_SIZE // PAGE_SIZE
    MAX_NUM_PAGES: tl.constexpr = MAX_SEQ_LEN // PAGE_SIZE

    cache_loc = tl.load(cache_loc_ptr + batch_id)
    seq_len = tl.load(cache_len_ptr + batch_id)
    # Offsets for the block of sequences this program processes.
    seq_start_pos = seq_block_id * SEQ_BLOCK_SIZE

    if seq_start_pos > seq_len:
        return
    seq_offsets = seq_start_pos + tl.arange(0, SEQ_BLOCK_SIZE)
    seq_mask = seq_offsets <= seq_len
    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    cache_head_offset = (head_id // HEAD_RATIO) * D_HEAD

    sm_scale: tl.constexpr = 1 / (D_HEAD**0.5)

    # Program loads the entire Q for the head assigned to it.
    # [D_HEAD]
    q_batch_offset = batch_id * N_HEADS * D_HEAD
    q_head_offset = head_id * D_HEAD
    q = tl.load(q_ptr + q_batch_offset + q_head_offset + dhead_offsets)

    kv_mask = seq_mask[:, None] * dhead_mask[None, :]

    kv_pages = seq_block_id * SEQ_BLOCK_PAGE + tl.arange(0, SEQ_BLOCK_PAGE)
    cache_pages = tl.load(
        page_table_ptr + cache_loc * PAGE_TABLE_STRIDE + kv_pages, mask=kv_pages < MAX_NUM_PAGES
    )

    page_offsets = tl.arange(0, PAGE_SIZE)
    # shape [SEQ_BLOCK], means [cache_pages, page_offsets]
    # token offsets in the paged kv cache
    cache_seq_offset = tl.reshape(
        cache_pages[:, None] * PAGE_SIZE + page_offsets[None, :], [SEQ_BLOCK_SIZE]
    )

    cache_offsets = (
        cache_seq_offset[:, None] * N_KV_HEADS * D_HEAD + cache_head_offset + dhead_offsets[None, :]
    )

    k = tl.load(k_cache_ptr + cache_offsets, mask=kv_mask)
    v = tl.load(v_cache_ptr + cache_offsets, mask=kv_mask)

    # Note: check the output precision of the sum.
    # compute q*K^T
    # [D_HEAD] * [seq_block, D_HEAD], sum along axis 1
    attn = tl.sum(q[None, :] * k, axis=1)  # [seq_block]
    attn = attn.to(tl.float32)
    attn *= sm_scale
    max_attn = tl.max(attn)
    # Set to -inf attn values where mask is not set. This forces exp(attn) to 0.
    attn = tl.where(seq_mask, attn, float("-inf"))
    exp_attn = tl.exp(attn - max_attn)

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
    )
    tl.store(
        output_logsumexp_ptr
        + batch_id * N_HEADS * num_blocks
        + head_id * num_blocks
        + seq_block_id,
        logsumexp,
    )


@triton.jit
def context_attention_kv_paged(
    q_ptr,  # [b*s,nd]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [num_pages, page_size, n, d]
    v_cache_ptr,  # [num_pages, page_size, n, d]
    cache_loc_ptr,  # [b] # index of the sequence in the page table.
    cache_len_ptr,  # [Batch] # Number of tokens in kv cache.
    page_table_ptr,  # [b, max_num_pages_per_seq] # loc of the block page in the cache.
    softmax_scale,
    o_ptr,
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_STRIDE: tl.constexpr,
):
    """Kernel for context phase.

    Fuses rope
    Assuming:
    1. Self-attention [seqlen(Q) == seqlen(K)]
    2. Causal attention
    3. QKV layout: [b*s,n,d]
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    seq_start_index = tl.load(seq_start_ptr + batch_id)
    seq_len = tl.load(seq_len_ptr + batch_id)

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS

    # assuming SEQ_BLOCK can be divided by PAGE_SIZE and PAGE_SIZE is a power of 2.
    SEQ_BLOCK_PAGE: tl.constexpr = SEQ_BLOCK // PAGE_SIZE
    MAX_NUM_PAGES: tl.constexpr = (MAX_SEQ_LENGTH + PAGE_SIZE - 1) // PAGE_SIZE

    # cache is [num_pages, page_size, n, d]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)
    table_batch_offset = cache_loc * PAGE_TABLE_STRIDE

    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    seq_offsets = tl.arange(0, SEQ_BLOCK)
    q_seq_offsets = seq_block_id * SEQ_BLOCK + seq_offsets
    seq_mask = q_seq_offsets < seq_len

    load_mask = seq_mask[:, None] * dhead_mask[None, :]

    q_batch_offset = seq_start_index * N_HEADS * D_HEAD
    q_head_offset = head_id * D_HEAD
    cache_head_offset = (head_id // HEAD_RATIO) * D_HEAD

    # Q will stay in SRAM
    q = tl.load(
        q_ptr
        + q_batch_offset
        + q_seq_offsets[:, None] * N_HEADS * D_HEAD
        + q_head_offset
        + dhead_offsets[None, :],
        mask=load_mask,
    )
    acc = tl.zeros([SEQ_BLOCK, D_HEAD], dtype=tl.float32)
    lse_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")

    cache_len = tl.load(cache_len_ptr + batch_id)
    total_len = cache_len + seq_len
    num_blocks = (total_len + SEQ_BLOCK - 1) // SEQ_BLOCK
    for s in range(0, num_blocks + 1, 1):
        kv_pages = s * SEQ_BLOCK_PAGE + tl.arange(0, SEQ_BLOCK_PAGE)
        cache_pages = tl.load(
            page_table_ptr + table_batch_offset + kv_pages, mask=kv_pages < MAX_NUM_PAGES
        )

        page_offsets = tl.arange(0, PAGE_SIZE)
        # shape [SEQ_BLOCK], means [cache_pages, page_offsets]
        # physical token offsets in the paged kv cache
        cache_seq_offset = tl.reshape(
            cache_pages[:, None] * PAGE_SIZE + page_offsets[None, :], [SEQ_BLOCK]
        )
        cache_offsets = (
            cache_seq_offset[:, None] * N_KV_HEADS * D_HEAD
            + cache_head_offset
            + dhead_offsets[None, :]
        )

        # logical kv tokens offsets
        kv_seq_offsets = s * SEQ_BLOCK + seq_offsets
        kv_seq_mask = kv_seq_offsets < total_len
        kv_load_mask = kv_seq_mask[:, None] * dhead_mask[None, :]

        k = tl.load(k_cache_ptr + cache_offsets, mask=kv_load_mask)
        qk = tl.zeros([SEQ_BLOCK, SEQ_BLOCK], dtype=tl.float32)
        qk += tl.dot(q, k.trans())
        # causal mask, need to use kv_seq_offsets
        qk = tl.where(
            (q_seq_offsets[:, None] + cache_len) >= kv_seq_offsets[None, :], qk, float("-inf")
        )

        qk *= softmax_scale
        # rowmax
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
        v = tl.load(v_cache_ptr + cache_offsets, mask=kv_load_mask)

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
        + q_batch_offset
        + q_seq_offsets[:, None] * N_HEADS * D_HEAD
        + q_head_offset
        + dhead_offsets[None, :],
        acc,
        mask=load_mask,
    )
