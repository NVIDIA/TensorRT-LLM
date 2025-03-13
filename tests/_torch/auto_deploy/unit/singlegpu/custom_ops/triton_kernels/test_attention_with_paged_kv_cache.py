import math
import random

import pytest
import torch
import triton
from _custom_op_utils import torch_reference_mha_stage2
from _model_test_utils import repeat_kv

from tensorrt_llm._torch.auto_deploy.custom_ops.triton_kernels.attention_with_paged_kv_cache import (
    attention_kv_paged_stage1,
    context_attention_kv_paged,
    update_paged_kv_cache,
)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [16, 8, 9, 21],  # context only sequences
        [1, 1, 1, 1, 1, 1],  # decode only sequences
        [5, 10, 4, 1, 1, 1],  # context + decode sequences
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_update_paged_kv_cache(seq_lens, dtype):
    DEVICE = "cuda"
    DTYPE = dtype
    N_KV_HEADS = 8
    D_HEAD = 16
    MAX_SEQ_LEN = 64
    SEQ_LENS = seq_lens  # 2 context
    BATCH_SIZE = len(SEQ_LENS)
    CACHE_LOCS = list(range(BATCH_SIZE))
    random.shuffle(CACHE_LOCS)
    NUM_PAGES = 256
    PAGE_SIZE = 4

    k = []
    v = []
    for i, s in enumerate(SEQ_LENS):
        k.append(torch.randn(1, s, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE))
        v.append(torch.randn(1, s, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE))

    (k_f, v_f) = tuple(map(lambda x: torch.cat(x, 1), (k, v)))

    k_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)

    # length of kv cache
    # if context batch then 0
    CACHE_LENS = []
    for b in range(BATCH_SIZE):
        CACHE_LENS.append(random.randint(0, 4 * PAGE_SIZE))

    # allocate pages for kv cache
    # pages of each batch is continuous
    PAGE_TABLE = [None] * BATCH_SIZE
    PAGES_PER_SEQ = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    cnt = 0
    for b in range(BATCH_SIZE):
        # allocate pages for history kv cache and new coming kv
        length = CACHE_LENS[b] + SEQ_LENS[b]
        allocated_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        table = []
        for p in range(PAGES_PER_SEQ):
            if p < allocated_pages:
                table.append(cnt)
                cnt = cnt + 1
            else:
                table.append(0)
        PAGE_TABLE[CACHE_LOCS[b]] = table
    page_table = torch.tensor(PAGE_TABLE, device=DEVICE, dtype=torch.int32)

    GENERATE_ONLY = all(s == 1 for s in SEQ_LENS)
    SEQ_BLOCK = PAGE_SIZE if GENERATE_ONLY else 32
    grid = (BATCH_SIZE, N_KV_HEADS, (max(SEQ_LENS) + SEQ_BLOCK - 1) // SEQ_BLOCK)

    seq_len = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)
    seq_start_indices = torch.zeros(BATCH_SIZE, device=DEVICE, dtype=torch.int32)
    seq_start_indices[1:] = torch.cumsum(seq_len[:-1], 0)

    update_paged_kv_cache[grid](
        k_f,
        v_f,
        seq_len,
        seq_start_indices,
        k_cache,
        v_cache,
        torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
        torch.tensor(CACHE_LENS, device=DEVICE, dtype=torch.int32),
        page_table,
        N_KV_HEADS,
        D_HEAD,
        SEQ_BLOCK,
        MAX_SEQ_LEN,
        PAGE_SIZE,
        page_table.stride(0),
        GENERATE_ONLY,
    )

    # Check if the cache was correctly updated:
    for batch, kv_batch in enumerate(CACHE_LOCS):
        batch_page_table = page_table[kv_batch]
        cache_len = CACHE_LENS[batch]
        update_len = SEQ_LENS[batch]
        if cache_len == 0:
            # context batch
            for seq_page, kv_page in enumerate(batch_page_table):
                if seq_page * PAGE_SIZE >= update_len:
                    break
                start = 0
                end = min(update_len - seq_page * PAGE_SIZE, PAGE_SIZE)
                assert torch.equal(
                    k_cache[kv_page, start:end].squeeze(),
                    k[batch][:, seq_page * PAGE_SIZE : (seq_page + 1) * PAGE_SIZE].squeeze(),
                )
                assert torch.equal(
                    v_cache[kv_page, start:end].squeeze(),
                    v[batch][:, seq_page * PAGE_SIZE : (seq_page + 1) * PAGE_SIZE].squeeze(),
                )
        else:
            # decode batch, only check one token in one page
            check_page = cache_len // PAGE_SIZE
            kv_page = batch_page_table[check_page]
            start = cache_len % PAGE_SIZE
            end = start + 1
            assert torch.equal(
                k_cache[kv_page, start:end].squeeze(),
                k[batch][:, 0:1].squeeze(),
            )
            assert torch.equal(
                v_cache[kv_page, start:end].squeeze(),
                v[batch][:, 0:1].squeeze(),
            )


def test_attention_kv_paged_flash_decoding():
    DEVICE = "cuda"
    DTYPE = torch.float16
    N_HEADS = 32
    D_HEAD = 32
    MAX_SEQ_LEN = 64
    NUM_PAGES = 256
    PAGE_SIZE = 4

    CACHE_LEN = [44, 33, 18, 11, 25]
    BATCH_SIZE = len(CACHE_LEN)
    SEQ_LENS = []
    for _ in range(BATCH_SIZE):
        SEQ_LENS.append(1)
    # only use for page table index
    CACHE_LOCS = list(range(0, BATCH_SIZE))
    random.shuffle(CACHE_LOCS)

    # Q,K,V are computed using GEMM.
    qkv = torch.randn(BATCH_SIZE, 3, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE) * 2
    q, k, v = torch.split(qkv, [1, 1, 1], dim=1)
    q, k, v = (x.contiguous() for x in (q, k, v))
    k_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)

    # allocate pages for kv cache
    # pages of each batch is continuous
    PAGE_TABLE = [None] * BATCH_SIZE
    PAGES_PER_SEQ = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    cnt = 0
    for b in range(BATCH_SIZE):
        length = CACHE_LEN[b]
        allocated_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        table = []
        for p in range(PAGES_PER_SEQ):
            if p < allocated_pages:
                table.append(cnt)
                cnt = cnt + 1
            else:
                table.append(0)
        PAGE_TABLE[CACHE_LOCS[b]] = table
    page_table = torch.tensor(PAGE_TABLE, device=DEVICE, dtype=torch.int32)

    # prepare kv-cache
    for b in range(BATCH_SIZE):
        pages = PAGE_TABLE[CACHE_LOCS[b]]
        cache_l = CACHE_LEN[b]
        page_num = cache_l // PAGE_SIZE
        page_off = cache_l % PAGE_SIZE
        for p in range(page_num):
            k_cache[pages[p]] = torch.randn(
                (1, PAGE_SIZE, N_HEADS, D_HEAD), dtype=DTYPE, device=DEVICE
            )
            v_cache[pages[p]] = torch.randn(
                (1, PAGE_SIZE, N_HEADS, D_HEAD), dtype=DTYPE, device=DEVICE
            )
        k_cache[pages[page_num], 0:page_off] = torch.randn(
            (1, page_off, N_HEADS, D_HEAD), dtype=DTYPE, device=DEVICE
        )
        v_cache[pages[page_num], 0:page_off] = torch.randn(
            (1, page_off, N_HEADS, D_HEAD), dtype=DTYPE, device=DEVICE
        )

    SEQ_BLOCK_SIZE = PAGE_SIZE
    # Input position 0 implies that kv-cache is empty
    num_blocks = MAX_SEQ_LEN // SEQ_BLOCK_SIZE
    output_tensor = torch.zeros(
        BATCH_SIZE, N_HEADS, num_blocks, D_HEAD, device=DEVICE, dtype=torch.float32
    )
    output_logsumexp = torch.zeros(
        BATCH_SIZE, N_HEADS, num_blocks, device=DEVICE, dtype=torch.float32
    ) - float("inf")

    grid = (BATCH_SIZE, N_HEADS, (max(SEQ_LENS) + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE)

    seq_len = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)
    seq_start_indices = torch.zeros(len(SEQ_LENS), device=DEVICE, dtype=torch.int32)
    seq_start_indices[1:] = torch.cumsum(seq_len[:-1], 0)
    update_paged_kv_cache[grid](
        k,
        v,
        seq_len,
        seq_start_indices,
        k_cache,
        v_cache,
        torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
        torch.tensor(CACHE_LEN, device=DEVICE, dtype=torch.int32),
        page_table,
        N_HEADS,
        D_HEAD,
        SEQ_BLOCK_SIZE,
        MAX_SEQ_LEN,
        PAGE_SIZE,
        page_table.stride(0),
        GENERATE_ONLY=True,
    )

    # Check if the cache was correctly updated:
    for batch, kv_batch in enumerate(CACHE_LOCS):
        batch_page_table = page_table[kv_batch]
        # decode batch, only check one token in one page
        cache_len = CACHE_LEN[batch]
        check_page = cache_len // PAGE_SIZE
        kv_page = batch_page_table[check_page]
        start = cache_len % PAGE_SIZE
        end = start + 1
        assert torch.equal(
            k_cache[kv_page, start:end].squeeze(),
            k[batch].squeeze(),
        )
        assert torch.equal(
            v_cache[kv_page, start:end].squeeze(),
            v[batch].squeeze(),
        )

    def run():
        attention_kv_paged_stage1[
            (
                BATCH_SIZE,
                N_HEADS,
                num_blocks,
            )
        ](
            q,
            k_cache,
            v_cache,
            torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
            page_table,
            torch.tensor(CACHE_LEN, device=DEVICE, dtype=torch.int32),
            output_tensor,
            output_logsumexp,
            num_blocks,
            MAX_SEQ_LEN,
            N_HEADS,
            N_HEADS,
            D_HEAD,
            SEQ_BLOCK_SIZE,
            PAGE_SIZE,
            page_table.stride(0),
        )

    run()

    # This needs to be another kernel if torch-trt doesn't support broadcast + div.
    output = torch_reference_mha_stage2(output_tensor, output_logsumexp)

    _ref = []
    for b in range(BATCH_SIZE):
        pages = PAGE_TABLE[CACHE_LOCS[b]]
        cache_l = CACHE_LEN[b]
        page_num = cache_l // PAGE_SIZE
        page_off = cache_l % PAGE_SIZE
        _k = []
        _v = []
        for p in range(page_num):
            _k.append(k_cache[pages[p]].reshape([-1, N_HEADS, D_HEAD]))
            _v.append(v_cache[pages[p]].reshape([-1, N_HEADS, D_HEAD]))
        _k.append(k_cache[pages[page_num], 0 : page_off + 1].reshape([-1, N_HEADS, D_HEAD]))
        _v.append(v_cache[pages[page_num], 0 : page_off + 1].reshape([-1, N_HEADS, D_HEAD]))
        _ref.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[b].reshape([1, N_HEADS, 1, D_HEAD]),
                torch.cat(_k, 0).reshape(1, -1, N_HEADS, D_HEAD).transpose(1, 2),
                torch.cat(_v, 0).reshape(1, -1, N_HEADS, D_HEAD).transpose(1, 2),
            ).transpose(2, 1)
        )
    ref = torch.cat(_ref, 1)

    assert torch.allclose(
        ref.squeeze().cpu().to(torch.float32),
        output.squeeze().cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: run(),
        quantiles=quantiles,
    )

    def compute_flops():
        flops = BATCH_SIZE * N_HEADS * (D_HEAD * D_HEAD * num_blocks * SEQ_BLOCK_SIZE)  # S = q*K
        flops += (
            BATCH_SIZE
            * N_HEADS
            * (D_HEAD * num_blocks * SEQ_BLOCK_SIZE * num_blocks * SEQ_BLOCK_SIZE)
        )  # S*V
        return flops

    print("Time: %0.2f ms" % ms)
    print("GFLOPs: %0.2f" % (compute_flops() / ms / 1e6))


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "bfloat16"],
)
@pytest.mark.parametrize("n_heads, n_kv_heads", [(8, 8), (8, 1)])
def test_context_attention_kv_paged(n_heads, n_kv_heads, dtype):
    DEVICE = "cuda"
    DTYPE = getattr(torch, dtype)
    N_HEADS = n_heads
    N_KV_HEADS = n_kv_heads
    D_HEAD = 16
    MAX_SEQ_LEN = 64
    SEQ_LENS = [36, 43, 21, 14, 18, 1, 1]
    BATCH_SIZE = len(SEQ_LENS)
    CACHE_LOCS = list(range(BATCH_SIZE))
    random.shuffle(CACHE_LOCS)
    NUM_PAGES = 256
    PAGE_SIZE = 4
    SEQ_BLOCK = 32

    q = []
    k = []
    v = []
    for i, s in enumerate(SEQ_LENS):
        q.append(torch.randn(1, s, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE) + i)
        k.append(torch.randn(1, s, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE) + i)
        v.append(torch.randn(1, s, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE) + i)

    (q_f, k_f, v_f) = tuple(map(lambda x: torch.cat(x, 1).contiguous(), (q, k, v)))

    k_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(NUM_PAGES, PAGE_SIZE, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)

    CACHE_LENS = []
    for b in range(BATCH_SIZE):
        CACHE_LENS.append(random.randint(0, 4 * PAGE_SIZE))

    # allocate pages for kv cache
    # pages of each batch is continuous
    PAGE_TABLE = [None] * BATCH_SIZE
    PAGES_PER_SEQ = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    cnt = 0
    for b in range(BATCH_SIZE):
        # allocate pages for history kv cache and new coming kv
        length = CACHE_LENS[b] + SEQ_LENS[b]
        allocated_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        table = []
        for p in range(PAGES_PER_SEQ):
            if p < allocated_pages:
                table.append(cnt)
                cnt = cnt + 1
            else:
                table.append(0)
        PAGE_TABLE[CACHE_LOCS[b]] = table
        # prepare value for kv cache of decode batch
        cache_pages = CACHE_LENS[b] // PAGE_SIZE
        cache_page_off = CACHE_LENS[b] % PAGE_SIZE
        k_cache[table[0] : table[cache_pages]] = torch.randn(
            cache_pages, PAGE_SIZE, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE
        )
        v_cache[table[0] : table[cache_pages]] = torch.randn(
            cache_pages, PAGE_SIZE, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE
        )
        k_cache[table[cache_pages], 0:cache_page_off] = torch.randn(
            cache_page_off, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE
        )
        v_cache[table[cache_pages], 0:cache_page_off] = torch.randn(
            cache_page_off, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE
        )

    page_table = torch.tensor(PAGE_TABLE, device=DEVICE, dtype=torch.int32)

    seq_len = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)
    seq_start_indices = torch.zeros(BATCH_SIZE, device=DEVICE, dtype=torch.int32)
    seq_start_indices[1:] = torch.cumsum(seq_len[:-1], 0)

    softmax_scale = 1.0 / math.sqrt(D_HEAD)
    output_tensor = torch.empty_like(q_f)

    grid = (BATCH_SIZE, N_KV_HEADS, (max(SEQ_LENS) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    update_paged_kv_cache[grid](
        k_f,
        v_f,
        seq_len,
        seq_start_indices,
        k_cache,
        v_cache,
        torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
        torch.tensor(CACHE_LENS, device=DEVICE, dtype=torch.int32),
        page_table,
        N_KV_HEADS,
        D_HEAD,
        SEQ_BLOCK,
        MAX_SEQ_LEN,
        PAGE_SIZE,
        page_table.stride(0),
        GENERATE_ONLY=False,
    )

    # Check if the cache was correctly updated:
    for batch, kv_batch in enumerate(CACHE_LOCS):
        batch_page_table = page_table[kv_batch]
        cache_len = CACHE_LENS[batch]
        update_len = SEQ_LENS[batch]
        if cache_len == 0:
            # context batch
            for seq_page, kv_page in enumerate(batch_page_table):
                if seq_page * PAGE_SIZE >= update_len:
                    break
                start = 0
                end = min(update_len - seq_page * PAGE_SIZE, PAGE_SIZE)
                assert torch.equal(
                    k_cache[kv_page, start:end].squeeze(),
                    k[batch][:, seq_page * PAGE_SIZE : (seq_page + 1) * PAGE_SIZE].squeeze(),
                )
                assert torch.equal(
                    v_cache[kv_page, start:end].squeeze(),
                    v[batch][:, seq_page * PAGE_SIZE : (seq_page + 1) * PAGE_SIZE].squeeze(),
                )
        else:
            # decode batch, only check one token in one page
            check_page = cache_len // PAGE_SIZE
            kv_page = batch_page_table[check_page]
            start = cache_len % PAGE_SIZE
            end = start + 1
            assert torch.equal(
                k_cache[kv_page, start:end].squeeze(),
                k[batch][:, 0:1].squeeze(),
            )
            assert torch.equal(
                v_cache[kv_page, start:end].squeeze(),
                v[batch][:, 0:1].squeeze(),
            )

    grid = (BATCH_SIZE, N_HEADS, (max(SEQ_LENS) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_paged[grid](
        q_f,
        seq_len,
        seq_start_indices,
        k_cache,
        v_cache,
        torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
        torch.tensor(CACHE_LENS, device=DEVICE, dtype=torch.int32),
        page_table,
        softmax_scale,
        output_tensor,
        N_HEADS,
        N_KV_HEADS,
        D_HEAD,
        SEQ_BLOCK,
        MAX_SEQ_LEN,
        PAGE_SIZE,
        page_table.stride(0),
        num_stages=2,
    )

    def compute_reference(q, k_cache, v_cache):
        ref = []
        for batch in range(BATCH_SIZE):
            table = page_table[CACHE_LOCS[batch]]
            length = CACHE_LENS[batch] + SEQ_LENS[batch]
            cache_pages = length // PAGE_SIZE
            cache_page_off = length % PAGE_SIZE
            kk = []
            vv = []
            kk.append(
                k_cache[table[0] : table[0] + cache_pages].reshape(
                    1, cache_pages * PAGE_SIZE, N_KV_HEADS, D_HEAD
                )
            )
            kk.append(
                k_cache[table[0] + cache_pages, 0:cache_page_off].reshape(
                    1, cache_page_off, N_KV_HEADS, D_HEAD
                )
            )
            k_f = torch.cat(kk, 1)
            vv.append(
                v_cache[table[0] : table[0] + cache_pages].reshape(
                    1, cache_pages * PAGE_SIZE, N_KV_HEADS, D_HEAD
                )
            )
            vv.append(
                v_cache[table[0] + cache_pages, 0:cache_page_off].reshape(
                    1, cache_page_off, N_KV_HEADS, D_HEAD
                )
            )
            v_f = torch.cat(vv, 1)
            if N_HEADS != N_KV_HEADS:
                k_f = repeat_kv(q[batch], k_f)
                v_f = repeat_kv(q[batch], v_f)
            mask = torch.tril(
                torch.ones(q[batch].shape[1], k_f.shape[1], dtype=torch.bool),
                diagonal=k_f.shape[1] - q[batch].shape[1],
            )
            ref.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[batch].transpose(1, 2),
                    k_f.transpose(1, 2),
                    v_f.transpose(1, 2),
                    attn_mask=mask.to(DEVICE),
                ).transpose(2, 1)
            )
        return torch.cat(ref, 1)

    ref = compute_reference(q, k_cache, v_cache)
    assert torch.allclose(ref, output_tensor, atol=1e-2, rtol=1e-2)
