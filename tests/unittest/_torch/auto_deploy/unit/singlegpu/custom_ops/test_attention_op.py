import pytest
import torch
from _custom_op_utils import torch_rope_reference
from torch_attention_reference import TorchAttentionReference

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def test_attention_op():
    DEVICE = "cuda"
    DTYPE = torch.float16
    BATCH_SIZE = 2
    N_HEADS = 8
    D_HEAD = 32
    MAX_SEQ_LEN = 128

    # Q,K,V are computed using GEMM.
    qkv = torch.randn(BATCH_SIZE, 3, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros((BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros((BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=DEVICE)
    input_positions = torch.zeros(BATCH_SIZE, device=DEVICE, dtype=torch.int) + 1

    q, k, v = (x.contiguous() for x in torch.split(qkv, 1, dim=1))

    output = torch.ops.auto_deploy.triton_attention_fused_mha_with_cache(
        q, k, v, input_positions, k_cache, v_cache, None
    )
    # Use torch backend as clean reference
    ref = TorchAttentionReference.basic_mha_with_cache(q, k, v, k_cache, v_cache, input_positions)
    assert torch.allclose(
        ref.cpu().to(torch.float32),
        output.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("seq_len", [1, 8])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("device", ["cuda"])
def test_gqa_op(device, dtype, n_heads, group_size, seq_len):
    BATCH_SIZE = 2
    D_HEAD = 16
    CACHE_SEQ_LEN = 8

    dtype = getattr(torch, dtype)
    n_kv_heads = n_heads // group_size

    if seq_len == 1:
        offset = seq_len // 2
        input_positions = torch.zeros(BATCH_SIZE, device=device, dtype=torch.int) + offset
    else:
        input_positions = torch.zeros(BATCH_SIZE, device=device, dtype=torch.int)

    q = torch.randn(BATCH_SIZE, seq_len, n_heads, D_HEAD, dtype=dtype, device=device)
    k = torch.randn(BATCH_SIZE, seq_len, n_kv_heads, D_HEAD, dtype=dtype, device=device)
    v = torch.randn(BATCH_SIZE, seq_len, n_kv_heads, D_HEAD, dtype=dtype, device=device)

    # setup kv-cache
    k_cache = torch.randn(BATCH_SIZE, CACHE_SEQ_LEN, n_kv_heads, D_HEAD, dtype=dtype, device=device)
    v_cache = torch.randn(BATCH_SIZE, CACHE_SEQ_LEN, n_kv_heads, D_HEAD, dtype=dtype, device=device)

    # run custom op
    output = torch.ops.auto_deploy.triton_attention_fused_mha_with_cache(
        q, k, v, input_positions, k_cache, v_cache, None
    )

    # Use torch backend as clean reference
    ref = TorchAttentionReference.basic_mha_with_cache(q, k, v, k_cache, v_cache, input_positions)

    assert torch.allclose(
        ref.cpu().to(torch.float32),
        output.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("num_generate_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("max_seq_len", [0, 1, 16])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("device", ["cuda"])
def test_flat_gqa_op(
    device, dtype, batch_size, n_heads, group_size, max_seq_len, num_generate_ratio
):
    n_heads = n_heads
    n_kv_heads = n_heads // group_size
    D_HEAD = 16
    dtype = getattr(torch, dtype)
    int_kwargs = {"device": device, "dtype": torch.int32}
    dtype_kwargs = {"device": device, "dtype": dtype}

    # setup caches with 2*batch_size, 2*max_seq_len since we also randomize input_pos
    cache_max_seq_len = 2 * (max_seq_len + 1)
    cache_max_batch_size = 2 * batch_size
    cache_size = (cache_max_batch_size, cache_max_seq_len, n_kv_heads, D_HEAD)
    cache_loc = torch.randperm(cache_max_batch_size, **int_kwargs)[:batch_size]

    k_cache = torch.randn(cache_size, **dtype_kwargs)
    v_cache = torch.randn(cache_size, **dtype_kwargs)

    # randomize num_context vs num_generate; NOTE: we can use context kernel for generate as well
    num_generate = torch.tensor(num_generate_ratio * batch_size, **int_kwargs)
    num_context = batch_size - num_generate

    # construct random input_positions
    input_positions = torch.randint(0, max_seq_len + 1, (batch_size,), **int_kwargs)

    # construct seq_len, seq_start;
    seq_len = torch.cat(
        [
            torch.randint(0, max_seq_len + 1, (num_context,), **int_kwargs),  # context
            torch.zeros(num_generate, **int_kwargs) + (max_seq_len > 0),  # generate
        ]
    )
    seq_start = seq_len.cumsum(0) - seq_len

    # get fake input
    q = torch.randn(1, seq_len.sum(), n_heads * D_HEAD, **dtype_kwargs)
    k = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)
    v = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)

    # run op
    output = torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache(
        # Q, K, V
        q,
        k,
        v,
        # METADATA
        seq_len,
        input_positions,
        cache_loc,
        seq_start,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        # <none>
        # CONSTANTS
        scale=None,
    )

    # Use torch backend as clean reference
    ref_flat = TorchAttentionReference.flattened_mha_with_cache(
        q, k, v, seq_len, input_positions, cache_loc, seq_start, k_cache, v_cache
    )

    assert torch.allclose(
        ref_flat.cpu().to(torch.float32),
        output.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("num_generate_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("max_seq_len", [0, 1, 16])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("device", ["cuda"])
def test_flat_gqa_op_with_rope(
    device, dtype, batch_size, n_heads, group_size, max_seq_len, num_generate_ratio
):
    n_heads = n_heads
    n_kv_heads = n_heads // group_size
    D_HEAD = 16
    dtype = getattr(torch, dtype)
    int_kwargs = {"device": device, "dtype": torch.int32}
    dtype_kwargs = {"device": device, "dtype": dtype}

    # setup caches with 2*batch_size, 2*max_seq_len since we also randomize input_pos
    cache_max_seq_len = 2 * (max_seq_len + 1)
    cache_max_batch_size = 2 * batch_size
    cache_size = (cache_max_batch_size, cache_max_seq_len, n_kv_heads, D_HEAD)
    cache_loc = torch.randperm(cache_max_batch_size, **int_kwargs)[:batch_size]

    k_cache = torch.randn(cache_size, **dtype_kwargs)
    v_cache = torch.randn(cache_size, **dtype_kwargs)

    # randomize num_context vs num_generate; NOTE: we can use context kernel for generate as well
    num_generate = torch.tensor(num_generate_ratio * batch_size, **int_kwargs)
    num_context = batch_size - num_generate

    # construct random input_positions
    input_positions = torch.randint(0, max_seq_len + 1, (batch_size,), **int_kwargs)

    # construct seq_len, seq_start;
    seq_len = torch.cat(
        [
            torch.randint(0, max_seq_len + 1, (num_context,), **int_kwargs),  # context
            torch.zeros(num_generate, **int_kwargs) + (max_seq_len > 0),  # generate
        ]
    )
    seq_start = seq_len.cumsum(0) - seq_len

    # get fake input
    q = torch.randn(1, seq_len.sum(), n_heads * D_HEAD, **dtype_kwargs)
    k = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)
    v = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)

    # rope can modify the original tensor value
    q_o = q.clone()
    k_o = k.clone()

    freqs_cis = torch.rand([cache_max_seq_len, D_HEAD // 2, 2], device=device, dtype=torch.float32)

    # run op
    source = 1
    if source == 1:
        # call rope fusion kernels
        output = torch.ops.auto_deploy.triton_attention_fused_flattened_mha_with_cache_rope_fusion(
            q,
            k,
            v,
            input_positions,
            cache_loc,
            seq_len,
            seq_start,
            k_cache,
            v_cache,
            freqs_cis,
        )
    else:
        # call stand-alone rope embedding kernel
        output = torch.ops.auto_deploy.triton_attention_fused_flattened_mha_with_cache(
            q,
            k,
            v,
            input_positions,
            cache_loc,
            seq_len,
            seq_start,
            k_cache,
            v_cache,
            freqs_cis,
        )

    # prep batched tensors for comparison
    q_b = torch.zeros(batch_size, n_heads, max_seq_len, D_HEAD, **dtype_kwargs)
    k_cache_b = k_cache[cache_loc].transpose(1, 2)
    v_cache_b = v_cache[cache_loc].transpose(1, 2)

    def _store(t_batched, t_flat):
        # batched layout: [n,s,d]; flat layout: [s,n*d]
        n_h, _, d_h = t_batched.shape
        t_batched[:] = t_flat.view(-1, n_h, d_h).transpose(0, 1)

    def _store_rope(t_batched, t_flat, input_pos):
        # batched layout: [n,s,d];
        # flat layout: [s,n*d], and in interleaved order
        # need to reorder to normal for torch_rope_reference and then reorder back
        n_h, _, d_h = t_batched.shape
        t_i = t_flat.view(-1, n_h, d_h).unsqueeze(0)
        t = t_i.unflatten(-1, (2, D_HEAD // 2)).transpose(-1, -2).flatten(-2).contiguous()
        t_rope = torch_rope_reference(t, freqs_cis, input_pos)
        t_rope = t_rope.unflatten(-1, (D_HEAD // 2, 2)).transpose(-1, -2).flatten(-2).contiguous()
        t_batched[:] = t_rope[0].transpose(0, 1)

    for i_b, (i_pos, s_start, s_len) in enumerate(zip(input_positions, seq_start, seq_len)):
        # fill roped q, k in a batched manner
        _store_rope(q_b[i_b, :, :s_len], q_o[0, s_start : s_start + s_len], input_positions[i_b])
        _store_rope(
            k_cache_b[i_b, :, i_pos : i_pos + s_len],
            k_o[0, s_start : s_start + s_len],
            input_positions[i_b],
        )
        # fill v in a batched manner
        _store(v_cache_b[i_b, :, i_pos : i_pos + s_len], v[0, s_start : s_start + s_len])

    k_cache_b = torch.repeat_interleave(k_cache_b, group_size, dim=1)  # [b,n,s,d]
    v_cache_b = torch.repeat_interleave(v_cache_b, group_size, dim=1)  # [b,n,s,d]

    # run comparison
    refs = []
    for i_b, (i_pos, s_start, s_len) in enumerate(zip(input_positions, seq_start, seq_len)):
        mask = torch.cat(
            [
                torch.ones(s_len, i_pos, device=device, dtype=torch.bool),
                torch.tril(torch.ones(s_len, s_len, device=device, dtype=torch.bool)),
            ],
            dim=1,
        )
        ref_i = torch.nn.functional.scaled_dot_product_attention(
            q_b[i_b, :, :s_len],
            k_cache_b[i_b, :, : i_pos + s_len],
            v_cache_b[i_b, :, : i_pos + s_len],
            attn_mask=mask,
        )  # [n,s,d]
        ref_i = ref_i.transpose(0, 1).contiguous().view(s_len, n_heads * D_HEAD)  # [s,n*d]
        refs.append(ref_i)

    # flatten output for comparison
    ref_flat = torch.cat(refs, dim=0)[None]  # [1,s_total,n*d]

    assert torch.allclose(
        ref_flat.cpu().to(torch.float32),
        output.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("num_generate_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("max_seq_len", [0, 1, 16])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("device", ["cuda"])
def test_paged_gqa_op(
    device, dtype, batch_size, n_heads, group_size, max_seq_len, num_generate_ratio
):
    n_heads = n_heads
    n_kv_heads = n_heads // group_size
    D_HEAD = 16
    dtype = getattr(torch, dtype)
    int_kwargs = {"device": device, "dtype": torch.int32}
    dtype_kwargs = {"device": device, "dtype": dtype}
    PAGE_SIZE = 4

    # setup caches with 2*batch_size, 2*max_seq_len since we also randomize input_pos
    cache_max_seq_len = 2 * (max_seq_len + 1)
    cache_max_batch_size = 2 * batch_size
    cache_max_pages = (cache_max_batch_size * cache_max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    batch_max_pages = (cache_max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    cache_size = (cache_max_pages, PAGE_SIZE, n_kv_heads, D_HEAD)
    cache_loc = torch.randperm(cache_max_batch_size, **int_kwargs)[:batch_size]

    k_cache = torch.zeros(cache_size, **dtype_kwargs)
    v_cache = torch.randn(cache_size, **dtype_kwargs)

    # randomize num_context vs num_generate; NOTE: we can use context kernel for generate as well
    num_generate = torch.tensor(num_generate_ratio * batch_size, **int_kwargs)
    num_context = batch_size - num_generate

    # construct seq_len, seq_start;
    # Context seq_len = 0 can result in wrong view and disorder infos like seq_start,
    #   only check seq_len > 0.
    #   i.e. num_context = 1, num_generate = 1 and seq_len = [0, 1],
    #   but the op might mistake the batch as batch_size = 1 and use the context batch infos.
    seq_len = torch.cat(
        [
            torch.randint(
                1 if max_seq_len > 0 else 0,
                max_seq_len + 1,
                (num_context,),
                **int_kwargs,
            ),  # context
            torch.zeros(num_generate, **int_kwargs) + (max_seq_len > 0),  # generate
        ]
    )

    # construct random input_positions(cache_len)
    input_positions = torch.cat(
        [
            torch.zeros(num_context, **int_kwargs),  # context
            torch.randint(0, max_seq_len + 1, (num_generate,), **int_kwargs),  # generate
        ]
    )

    seq_start = (seq_len.cumsum(0) - seq_len).to(torch.int32)

    # allocate pages for kv cache
    # pages of each batch is continuous
    PAGE_TABLE = [[0] * batch_max_pages] * cache_max_batch_size
    cnt = 0
    for b in range(batch_size):
        # allocate pages for history kv cache and new coming kv
        length = input_positions[b] + seq_len[b]
        allocated_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        table = []
        for p in range(batch_max_pages):
            if p < allocated_pages:
                table.append(cnt)
                cnt = cnt + 1
            else:
                table.append(0)
        PAGE_TABLE[cache_loc[b]] = table
        # prepare value for kv cache of decode batch
        cache_pages = input_positions[b] // PAGE_SIZE
        cache_page_off = input_positions[b] % PAGE_SIZE
        k_cache[table[0] : table[cache_pages]] = torch.randn(
            cache_pages, PAGE_SIZE, n_kv_heads, D_HEAD, **dtype_kwargs
        )
        v_cache[table[0] : table[cache_pages]] = torch.randn(
            cache_pages, PAGE_SIZE, n_kv_heads, D_HEAD, **dtype_kwargs
        )
        k_cache[table[cache_pages], 0:cache_page_off] = torch.randn(
            cache_page_off, n_kv_heads, D_HEAD, **dtype_kwargs
        )
        v_cache[table[cache_pages], 0:cache_page_off] = torch.randn(
            cache_page_off, n_kv_heads, D_HEAD, **dtype_kwargs
        )

    page_table = torch.tensor(PAGE_TABLE, **int_kwargs)

    # get fake input
    q = torch.randn(1, seq_len.sum(), n_heads * D_HEAD, **dtype_kwargs)
    k = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)
    v = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)

    # run op
    output = torch.ops.auto_deploy.triton_attention_fused_mha_with_paged_cache(
        q,
        k,
        v,
        input_positions,
        cache_loc,
        seq_len,
        seq_start,
        page_table,
        cache_max_seq_len,
        k_cache,
        v_cache,
        None,
    )

    # TODO (nvchenghaoz): Replace this with torch backend reference.

    # prep batched tensors for comparison
    def compute_reference(q, k_cache, v_cache):
        ref = []
        for batch in range(batch_size):
            table = page_table[cache_loc[batch]]
            s_len = seq_len[batch]
            c_len = input_positions[batch]
            length = c_len + s_len
            cache_pages = length // PAGE_SIZE
            cache_page_off = length % PAGE_SIZE
            s_start = seq_start[batch]
            # [bsnd]
            qq = q[0, s_start : s_start + s_len].view(1, -1, n_heads, D_HEAD)
            kk = []
            vv = []
            kk.append(
                k_cache[table[0] : table[0] + cache_pages].reshape(
                    1, cache_pages * PAGE_SIZE, n_kv_heads, D_HEAD
                )
            )
            kk.append(
                k_cache[table[0] + cache_pages, 0:cache_page_off].reshape(
                    1, cache_page_off, n_kv_heads, D_HEAD
                )
            )
            # [bsnd]
            k_f = torch.cat(kk, 1)
            vv.append(
                v_cache[table[0] : table[0] + cache_pages].reshape(
                    1, cache_pages * PAGE_SIZE, n_kv_heads, D_HEAD
                )
            )
            vv.append(
                v_cache[table[0] + cache_pages, 0:cache_page_off].reshape(
                    1, cache_page_off, n_kv_heads, D_HEAD
                )
            )
            v_f = torch.cat(vv, 1)
            if n_heads != n_kv_heads:
                k_f = torch.repeat_interleave(k_f, group_size, dim=2)
                v_f = torch.repeat_interleave(v_f, group_size, dim=2)
            mask = torch.tril(
                torch.ones(s_len, length, dtype=torch.bool),
                diagonal=c_len,
            )
            ref.append(
                torch.nn.functional.scaled_dot_product_attention(
                    qq.transpose(1, 2),
                    k_f.transpose(1, 2),
                    v_f.transpose(1, 2),
                    attn_mask=mask.to(device),
                )
                .transpose(2, 1)
                .contiguous()
                .view(1, s_len, n_heads * D_HEAD)  # [b,s,n*d]
            )
        return torch.cat(ref, 1)

    ref = compute_reference(q, k_cache, v_cache)
    assert torch.allclose(ref, output, atol=1e-2, rtol=1e-2)
