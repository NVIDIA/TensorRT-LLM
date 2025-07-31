import math
import random

import pytest
import torch
import triton
from _custom_op_utils import torch_rope_reference
from _model_test_utils import repeat_kv

from tensorrt_llm._torch.auto_deploy.custom_ops.triton_kernels.attention_with_kv_cache import (
    attention_kv_stage1,
    attention_kv_stage2,
    context_attention_kv,
    context_attention_kv_flattened,
    gqa_attention_kv_stage1,
    update_kv_cache,
    update_kv_cache_rope_fusion,
)


def torch_reference_stage2(values, logsumexp, sinks=None):
    max_logsumexp = torch.max(logsumexp, axis=-1, keepdim=True)[0]  # [b, n_heads, 1]
    sumexp = torch.exp(logsumexp - max_logsumexp)  # [b, n_heads, num_blocks]
    aggregate_sumexp = torch.sum(sumexp, axis=-1)  # [b, n_heads]
    # Add sinks contribution to the softmax denominator
    if sinks is not None:
        sinks_exp = torch.exp(sinks - max_logsumexp.squeeze(-1))  # [b, n_heads]
        aggregate_sumexp += sinks_exp
    output = values * sumexp[:, :, :, None]  # [b, n_heads, num_blocks, d_head]
    output = output / aggregate_sumexp[:, :, None, None]
    output = torch.sum(output, axis=2)
    return output


@pytest.mark.parametrize("k_d_head", [16, 96])
@pytest.mark.parametrize("v_d_head", [16, 96])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [16, 8, 9, 21],  # context only sequences
        [1, 1, 1, 1, 1, 1],  # decode only sequences
        [5, 10, 4, 1, 1, 1],  # context + decode sequences
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_update_kv_cache(k_d_head, v_d_head, seq_lens, dtype):
    DEVICE = "cuda"
    DTYPE = dtype
    N_KV_HEADS = 8
    K_D_HEAD = k_d_head
    V_D_HEAD = v_d_head
    MAX_SEQ_LEN = 64
    MAX_BATCH_SIZE = 16
    SEQ_LENS = seq_lens
    CACHE_LOCS = list(range(0, len(SEQ_LENS)))
    random.shuffle(CACHE_LOCS)
    INPUT_POS = [
        random.randint(0, 16) for _ in range(len(SEQ_LENS))
    ]  # The starting position for in the cache for each of the sequences.
    k = []
    v = []
    for i, s in enumerate(SEQ_LENS):
        k.append(torch.ones(1, s, N_KV_HEADS, K_D_HEAD, dtype=DTYPE, device=DEVICE))
        v.append(torch.ones(1, s, N_KV_HEADS, V_D_HEAD, dtype=DTYPE, device=DEVICE))

    (k_f, v_f) = tuple(map(lambda x: torch.cat(x, 1), (k, v)))
    k_f = k_f.contiguous()
    v_f = v_f.contiguous()
    k_cache = torch.zeros(
        MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, K_D_HEAD, dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.zeros(
        MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, V_D_HEAD, dtype=DTYPE, device=DEVICE
    )

    GENERATE_ONLY = all(s == 1 for s in SEQ_LENS)
    SEQ_BLOCK = 1 if GENERATE_ONLY else 32
    grid = (len(SEQ_LENS), N_KV_HEADS, (max(SEQ_LENS) + SEQ_BLOCK - 1) // SEQ_BLOCK)

    seq_len = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)
    seq_start_indices = torch.zeros(len(SEQ_LENS), device=DEVICE, dtype=torch.int32)
    seq_start_indices[1:] = torch.cumsum(seq_len[:-1], 0)

    update_kv_cache[grid](
        k_f,
        v_f,
        seq_len,
        seq_start_indices,
        k_cache,
        v_cache,
        torch.tensor(INPUT_POS, device=DEVICE, dtype=torch.int32),
        torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
        MAX_SEQ_LEN,
        N_KV_HEADS,
        K_D_HEAD,
        V_D_HEAD,
        SEQ_BLOCK,
        GENERATE_ONLY,
    )

    # Check if the cache was correctly updated:
    for i, cache_loc in enumerate(CACHE_LOCS):
        assert torch.equal(
            k_cache[cache_loc, INPUT_POS[i] : INPUT_POS[i] + SEQ_LENS[i], :N_KV_HEADS, :].squeeze(),
            k[i].squeeze(),
        )
        assert torch.equal(
            v_cache[cache_loc, INPUT_POS[i] : INPUT_POS[i] + SEQ_LENS[i], :N_KV_HEADS, :].squeeze(),
            v[i].squeeze(),
        )


@pytest.mark.parametrize("d_head", [16, 96])
def test_attention_kv_flash_decoding(d_head):
    DEVICE = "cuda"
    DTYPE = torch.float16
    BATCH_SIZE = 1
    N_HEADS = 1
    D_HEAD = d_head
    MAX_SEQ_LEN = 64
    CACHE_LOCS = list(range(0, BATCH_SIZE))
    random.shuffle(CACHE_LOCS)
    INPUT_POS = [0] * BATCH_SIZE
    # Q,K,V are computed using GEMM.
    q = torch.randn(BATCH_SIZE, 1, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(BATCH_SIZE, 1, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(BATCH_SIZE, 1, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE)

    grid = (BATCH_SIZE, N_HEADS, 1)  #
    update_kv_cache[grid](
        k,
        v,
        torch.tensor([1] * BATCH_SIZE, device=DEVICE, dtype=torch.int32),
        torch.arange(0, BATCH_SIZE, 1, device=DEVICE, dtype=torch.int32),
        k_cache,
        v_cache,
        torch.tensor(INPUT_POS, device=DEVICE, dtype=torch.int32),
        torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
        MAX_SEQ_LEN,
        N_HEADS,
        D_HEAD,
        D_HEAD,
        SEQ_BLOCK=1,
        GENERATE_ONLY=True,
    )
    SEQ_BLOCK_SIZE = 8
    num_blocks = MAX_SEQ_LEN // SEQ_BLOCK_SIZE
    output_tensor = torch.zeros(
        BATCH_SIZE, N_HEADS, num_blocks, D_HEAD, device=DEVICE, dtype=torch.float32
    )
    output_logsumexp = torch.zeros(
        BATCH_SIZE, N_HEADS, num_blocks, device=DEVICE, dtype=torch.float32
    ) - float("inf")

    def run(q, k_cache, v_cache, output_tensor, output_logsumexp):
        attention_kv_stage1[
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
            torch.tensor(INPUT_POS, device=DEVICE, dtype=torch.int32),
            output_tensor,
            output_logsumexp,
            num_blocks,
            MAX_SEQ_LEN,
            N_HEADS,
            N_HEADS,
            D_HEAD,
            SEQ_BLOCK_SIZE,
        )

    run(q, k_cache, v_cache, output_tensor, output_logsumexp)
    output = torch_reference_stage2(output_tensor, output_logsumexp)

    ref = []
    for i in range(BATCH_SIZE):
        ref.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[i, :, :, :].unsqueeze(0).transpose(1, 2),  # [BSND]->[BNSD]
                k_cache[CACHE_LOCS[i], 0 : INPUT_POS[i] + 1, :, :].unsqueeze(0).transpose(1, 2),
                v_cache[CACHE_LOCS[i], 0 : INPUT_POS[i] + 1, :, :].unsqueeze(0).transpose(1, 2),
            )
        )
    ref = torch.cat(ref, dim=0)
    print(ref)
    torch.allclose(
        ref.squeeze().cpu().to(torch.float32),
        output.squeeze().cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("q_d_head", [16, 96])
@pytest.mark.parametrize("v_d_head", [16, 96])
@pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (8, 1)])
@pytest.mark.parametrize("sliding_window", [-1, 16])
def test_gqa_attention_kv_flash_decoding(q_d_head, v_d_head, n_heads, n_kv_heads, sliding_window):
    DEVICE = "cuda"
    DTYPE = torch.float16
    BATCH_SIZE = 64
    N_HEADS = n_heads
    N_KV_HEADS = n_kv_heads
    Q_D_HEAD = q_d_head
    V_D_HEAD = v_d_head
    MAX_SEQ_LEN = 64
    CACHE_LOCS = list(range(0, BATCH_SIZE))
    INPUT_POS = [0] * BATCH_SIZE
    # Q,K,V are computed using GEMM.
    q = torch.randn(BATCH_SIZE, 1, N_HEADS, Q_D_HEAD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(BATCH_SIZE, 1, N_KV_HEADS, Q_D_HEAD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(BATCH_SIZE, 1, N_KV_HEADS, V_D_HEAD, dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, Q_D_HEAD, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn(BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, V_D_HEAD, dtype=DTYPE, device=DEVICE)

    cache_loc = torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32)
    input_pos = torch.tensor(INPUT_POS, device=DEVICE, dtype=torch.int32)

    grid = (BATCH_SIZE, N_KV_HEADS, 1)  #
    update_kv_cache[grid](
        k,
        v,
        torch.tensor([1] * BATCH_SIZE, device=DEVICE, dtype=torch.int32),
        torch.arange(0, BATCH_SIZE, 1, device=DEVICE, dtype=torch.int32),
        k_cache,
        v_cache,
        input_pos,
        cache_loc,
        MAX_SEQ_LEN,
        N_KV_HEADS,
        Q_D_HEAD,
        V_D_HEAD,
        SEQ_BLOCK=1,
        GENERATE_ONLY=True,
    )
    SEQ_BLOCK_SIZE = 16
    num_blocks = MAX_SEQ_LEN // SEQ_BLOCK_SIZE
    output_tensor = torch.zeros(
        BATCH_SIZE, N_HEADS, num_blocks, V_D_HEAD, device=DEVICE, dtype=torch.float32
    )
    output_logsumexp = torch.zeros(
        BATCH_SIZE, N_HEADS, num_blocks, device=DEVICE, dtype=torch.float32
    ) - float("inf")

    def run(q, k_cache, v_cache, output_tensor, output_logsumexp):
        HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(N_HEADS // N_KV_HEADS))

        gqa_attention_kv_stage1[
            (
                BATCH_SIZE,
                N_KV_HEADS,
                num_blocks,
            )
        ](
            q,
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            output_tensor,
            output_logsumexp,
            num_blocks,
            1.0 / math.sqrt(Q_D_HEAD),
            MAX_SEQ_LEN,
            N_HEADS,
            N_KV_HEADS,
            Q_D_HEAD,
            V_D_HEAD,
            SEQ_BLOCK_SIZE,
            HEAD_BLOCK_SIZE,
            sliding_window,  # SLIDING_WINDOW: parameterized
        )

    run(q, k_cache, v_cache, output_tensor, output_logsumexp)
    # This needs to be another kernel if torch-trt doesn't support broadcast + div.
    output = torch_reference_stage2(output_tensor, output_logsumexp)

    ref = []
    for i in range(BATCH_SIZE):
        kk = k_cache[CACHE_LOCS[i], 0 : INPUT_POS[i] + 1, :, :].unsqueeze(0)
        vv = v_cache[CACHE_LOCS[i], 0 : INPUT_POS[i] + 1, :, :].unsqueeze(0)

        if N_HEADS != N_KV_HEADS:
            kk = repeat_kv(q[i, :, :, :].unsqueeze(0), kk)
            vv = repeat_kv(q[i, :, :, :].unsqueeze(0), vv)
        ref.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[i, :, :, :].unsqueeze(0).transpose(1, 2),  # [BSND]->[BNSD]
                kk.transpose(1, 2),
                vv.transpose(1, 2),
            )
        )
    ref = torch.cat(ref, dim=0)
    assert torch.allclose(
        ref.squeeze().cpu().to(torch.float32),
        output.squeeze().cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("has_sinks", [False, True])
def test_attention_with_kv_stage2(has_sinks):
    DEVICE = "cuda"
    BATCH_SIZE = 4
    N_HEADS = 32
    D_HEAD = 96
    SEQ_BLOCK_SIZE = 8
    input_positions = torch.zeros(BATCH_SIZE, device=DEVICE, dtype=torch.int32) + 10
    num_blocks = 3
    # Produced by stage1
    values = torch.randn(
        BATCH_SIZE, N_HEADS, num_blocks, D_HEAD, device=DEVICE, dtype=torch.float32
    )
    logsumexp = torch.randn(BATCH_SIZE, N_HEADS, num_blocks, device=DEVICE, dtype=torch.float32)
    output = torch.zeros(BATCH_SIZE, N_HEADS, D_HEAD, device=DEVICE, dtype=torch.float32)
    # Create sink tokens if needed - kernel expects [BATCH_SIZE, N_HEADS] shape
    sinks = (
        torch.randn(BATCH_SIZE, N_HEADS, device=DEVICE, dtype=torch.float32) if has_sinks else None
    )

    def run():
        attention_kv_stage2[
            (
                BATCH_SIZE,
                N_HEADS,
            )
        ](
            values,
            logsumexp,
            output,
            input_positions,
            num_blocks,
            N_HEADS,
            D_HEAD,
            SEQ_BLOCK_SIZE,
            has_sinks,
            sinks,
        )

    run()
    ref = []
    for i in range(BATCH_SIZE):
        block_id = input_positions[i].item() // SEQ_BLOCK_SIZE + 1
        batch_sinks = sinks[i : i + 1, :] if has_sinks else None  # [1, N_HEADS]
        ref.append(
            torch_reference_stage2(
                values[i, :, :block_id, :].unsqueeze(0),
                logsumexp[i, :, :block_id].unsqueeze(0),
                batch_sinks,
            )
        )
    ref = torch.cat(ref, dim=0)
    assert torch.allclose(
        ref.squeeze().cpu().to(torch.float32),
        output.squeeze().cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("q_d_head", [32, 96])
@pytest.mark.parametrize("v_d_head", [32, 96])
@pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (8, 1)])
@pytest.mark.parametrize(
    "dtype_cache",
    [
        torch.bfloat16,
        pytest.param(
            torch.float8_e4m3fn,
            marks=pytest.mark.skipif(
                torch.cuda.get_device_capability(0) < (8, 9), reason="Requires fp8 support"
            ),
        ),
    ],
)
def test_context_attention_kv(batch_size, q_d_head, v_d_head, n_heads, n_kv_heads, dtype_cache):
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    BATCH_SIZE = batch_size
    N_HEADS = n_heads
    N_KV_HEADS = n_kv_heads
    Q_D_HEAD = K_D_HEAD = q_d_head
    V_D_HEAD = v_d_head
    MAX_SEQ_LEN = 64
    SEQ = 36
    # Q,K,V are computed using GEMM.
    q = torch.randn(BATCH_SIZE, SEQ, N_HEADS, Q_D_HEAD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(BATCH_SIZE, SEQ, N_KV_HEADS, K_D_HEAD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(BATCH_SIZE, SEQ, N_KV_HEADS, V_D_HEAD, dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros(
        BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, K_D_HEAD, dtype=dtype_cache, device=DEVICE
    )
    v_cache = torch.zeros(
        BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, V_D_HEAD, dtype=dtype_cache, device=DEVICE
    )

    SEQ_BLOCK = 16
    output_tensor = torch.empty((BATCH_SIZE, SEQ, N_HEADS, V_D_HEAD), dtype=DTYPE, device=DEVICE)
    grid = (BATCH_SIZE, N_HEADS, (SEQ + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        SEQ,
        output_tensor,
        1.0 / math.sqrt(Q_D_HEAD),
        N_HEADS,
        N_KV_HEADS,
        Q_D_HEAD,
        V_D_HEAD,
        SEQ_BLOCK,
        MAX_SEQ_LEN,
        num_stages=2,
    )

    if N_HEADS != N_KV_HEADS:
        k = repeat_kv(q, k)
        v = repeat_kv(q, v)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(2, 1)
    assert torch.allclose(ref, output_tensor, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "bfloat16"],
)
@pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (8, 1)])
@pytest.mark.parametrize("q_d_head", [32, 96])
@pytest.mark.parametrize("v_d_head", [32, 96])
@pytest.mark.parametrize("sliding_window", [-1, 16])
def test_context_attention_kv_flattened(
    q_d_head, v_d_head, n_heads, n_kv_heads, dtype, sliding_window
):
    DEVICE = "cuda"
    DTYPE = getattr(torch, dtype)
    N_HEADS = n_heads
    N_KV_HEADS = n_kv_heads
    K_D_HEAD = Q_D_HEAD = q_d_head
    V_D_HEAD = v_d_head
    MAX_SEQ_LEN = 64
    SEQ_LENS = [36, 44, 12, 1, 1]
    CACHE_LOCS = list(range(0, len(SEQ_LENS)))
    random.shuffle(CACHE_LOCS)
    INPUT_POS = [2, 4, 8, 0, 1]  # The starting position for in the cache for each of the sequences.
    q = []
    k = []
    v = []
    for i, s in enumerate(SEQ_LENS):
        q.append(torch.ones(1, s, N_HEADS, Q_D_HEAD, dtype=DTYPE, device=DEVICE))
        k.append(torch.ones(1, s, N_KV_HEADS, K_D_HEAD, dtype=DTYPE, device=DEVICE))
        v.append(torch.ones(1, s, N_KV_HEADS, V_D_HEAD, dtype=DTYPE, device=DEVICE))

    (q_f, k_f, v_f) = tuple(map(lambda x: torch.cat(x, 1).contiguous(), (q, k, v)))

    k_cache = torch.zeros(
        len(SEQ_LENS), MAX_SEQ_LEN, N_KV_HEADS, K_D_HEAD, dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.zeros(
        len(SEQ_LENS), MAX_SEQ_LEN, N_KV_HEADS, V_D_HEAD, dtype=DTYPE, device=DEVICE
    )

    def compute_reference(q, k_cache, v_cache):
        ref = []
        for i in range(len(SEQ_LENS)):
            kk = k_cache[CACHE_LOCS[i], : INPUT_POS[i] + SEQ_LENS[i], :N_KV_HEADS, :].view(
                1, INPUT_POS[i] + SEQ_LENS[i], N_KV_HEADS, K_D_HEAD
            )
            vv = v_cache[CACHE_LOCS[i], : INPUT_POS[i] + SEQ_LENS[i], :N_KV_HEADS, :].view(
                1, INPUT_POS[i] + SEQ_LENS[i], N_KV_HEADS, V_D_HEAD
            )

            if N_HEADS != N_KV_HEADS:
                kk = repeat_kv(q[i], kk)
                vv = repeat_kv(q[i], vv)

            mask = torch.tril(
                torch.ones(q[i].shape[1], kk.shape[1], dtype=torch.bool),
                diagonal=kk.shape[1] - q[i].shape[1],
            )

            # Apply sliding window constraints if enabled
            if sliding_window > 0:
                seq_len_q = q[i].shape[1]  # Current sequence length
                seq_len_k = kk.shape[1]  # Total KV sequence length

                # Create sliding window mask
                sliding_mask = torch.zeros_like(mask)
                for q_pos in range(seq_len_q):
                    # For each query position, determine its absolute position in the cache
                    abs_q_pos = INPUT_POS[i] + q_pos
                    # Calculate sliding window range
                    sliding_start = max(0, abs_q_pos - sliding_window + 1)
                    sliding_end = abs_q_pos + 1
                    # Apply to KV cache positions
                    k_start = max(0, sliding_start)
                    k_end = min(seq_len_k, sliding_end)
                    if k_start < k_end:
                        sliding_mask[q_pos, k_start:k_end] = True

                # Combine causal and sliding window masks
                mask = mask & sliding_mask

            ref.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[i].transpose(1, 2),
                    kk.transpose(1, 2),
                    vv.transpose(1, 2),
                    attn_mask=mask.to(DEVICE),
                ).transpose(2, 1)
            )
        return torch.cat(ref, 1)

    seq_len = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)
    seq_start_indices = torch.zeros(len(SEQ_LENS), device=DEVICE, dtype=torch.int32)
    seq_start_indices[1:] = torch.cumsum(seq_len[:-1], 0)
    input_pos = torch.tensor(INPUT_POS, device=DEVICE, dtype=torch.int32)
    cache_loc = torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32)
    SEQ_BLOCK = 32
    output_tensor = torch.empty((1, sum(SEQ_LENS), N_HEADS, V_D_HEAD), dtype=DTYPE, device=DEVICE)
    grid = (len(SEQ_LENS), N_KV_HEADS, (max(SEQ_LENS) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    update_kv_cache[grid](
        k_f,
        v_f,
        seq_len,
        seq_start_indices,
        k_cache,
        v_cache,
        input_pos,
        cache_loc,
        MAX_SEQ_LEN,
        N_KV_HEADS,
        K_D_HEAD,
        V_D_HEAD,
        SEQ_BLOCK,
        GENERATE_ONLY=False,
    )

    # Check if the cache was correctly updated:
    for i, cl in enumerate(CACHE_LOCS):
        assert torch.equal(
            k_cache[cl, INPUT_POS[i] : INPUT_POS[i] + SEQ_LENS[i], :N_KV_HEADS, :].squeeze(),
            k[i].squeeze(),
        )
        assert torch.equal(
            v_cache[cl, INPUT_POS[i] : INPUT_POS[i] + SEQ_LENS[i], :N_KV_HEADS, :].squeeze(),
            v[i].squeeze(),
        )
    ref = compute_reference(q, k_cache, v_cache)
    grid = (len(SEQ_LENS), N_HEADS, (max(SEQ_LENS) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_flattened[grid](
        q_f,
        seq_len,
        seq_start_indices,
        k_cache,
        v_cache,
        input_pos,
        cache_loc,
        output_tensor,
        1.0 / math.sqrt(Q_D_HEAD),
        N_HEADS,
        N_KV_HEADS,
        K_D_HEAD,
        V_D_HEAD,
        SEQ_BLOCK,
        MAX_SEQ_LEN,
        sliding_window,  # SLIDING_WINDOW: parameterized
        False,  # HAS_SINKS: no sink tokens used
        None,  # sinks_ptr: no sink tokens used
    )
    assert torch.allclose(ref, output_tensor, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [16, 8, 9, 21],  # context only sequences
        [1, 1, 1, 1, 1, 1],  # decode only sequences
        [5, 10, 4, 1, 1, 1],  # context + decode sequences
    ],
)
@pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (8, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_update_kv_cache_rope_fusion(seq_lens, n_heads, n_kv_heads, dtype):
    DEVICE = "cuda"
    DTYPE = dtype
    N_HEADS = n_heads
    N_KV_HEADS = n_kv_heads
    D_HEAD = 16
    MAX_SEQ_LEN = 64
    MAX_BATCH_SIZE = 16
    SEQ_LENS = seq_lens
    BATCH_SIZE = len(SEQ_LENS)
    CACHE_LOCS = list(range(0, BATCH_SIZE))
    random.shuffle(CACHE_LOCS)
    INPUT_POS = [
        random.randint(0, 16) for _ in range(BATCH_SIZE)
    ]  # The starting position for in the cache for each of the sequences.
    q = []
    k = []
    v = []
    for i, s in enumerate(SEQ_LENS):
        q.append(torch.randn(1, s, N_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE))
        k.append(torch.randn(1, s, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE))
        v.append(torch.randn(1, s, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE))

    (q_f, k_f, v_f) = tuple(map(lambda x: torch.cat(x, 1), (q, k, v)))
    # kernel handles interleaved input
    q_f = q_f.unflatten(-1, (D_HEAD // 2, 2)).transpose(-1, -2).flatten(-2).contiguous()
    k_f = k_f.unflatten(-1, (D_HEAD // 2, 2)).transpose(-1, -2).flatten(-2).contiguous()

    q_rope = torch.zeros_like(q_f)
    k_cache = torch.zeros(
        MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.zeros(
        MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, D_HEAD, dtype=DTYPE, device=DEVICE
    )

    GENERATE_ONLY = all(s == 1 for s in SEQ_LENS)
    SEQ_BLOCK = 1 if GENERATE_ONLY else 32
    grid = (BATCH_SIZE, N_KV_HEADS, (max(SEQ_LENS) + SEQ_BLOCK - 1) // SEQ_BLOCK)

    seq_len = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)
    seq_start_indices = torch.zeros(BATCH_SIZE, device=DEVICE, dtype=torch.int32)
    seq_start_indices[1:] = torch.cumsum(seq_len[:-1], 0)

    freqs = torch.rand([MAX_SEQ_LEN, D_HEAD // 2, 2], device=DEVICE, dtype=torch.float32)
    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(N_HEADS // N_KV_HEADS))

    update_kv_cache_rope_fusion[grid](
        q_f,
        k_f,
        v_f,
        seq_len,
        seq_start_indices,
        q_rope,
        k_cache,
        v_cache,
        torch.tensor(INPUT_POS, device=DEVICE, dtype=torch.int32),
        torch.tensor(CACHE_LOCS, device=DEVICE, dtype=torch.int32),
        freqs,
        MAX_SEQ_LEN,
        N_HEADS,
        N_KV_HEADS,
        D_HEAD,
        SEQ_BLOCK,
        HEAD_BLOCK_SIZE,
        GENERATE_ONLY,
    )

    q_ref = []
    k_ref = []
    for i in range(BATCH_SIZE):
        # result of torch reference is in normal order
        # simulate interleaved rope result
        q_batch = q[i]
        k_batch = k[i]
        q_ref.append(
            torch_rope_reference(q_batch, freqs, INPUT_POS[i])
            .unflatten(-1, (D_HEAD // 2, 2))
            .transpose(-1, -2)
            .flatten(-2)
            .contiguous()
        )
        k_ref.append(
            torch_rope_reference(k_batch, freqs, INPUT_POS[i])
            .unflatten(-1, (D_HEAD // 2, 2))
            .transpose(-1, -2)
            .flatten(-2)
            .contiguous()
        )

    # Check if q_rope was correctly updated:
    start = 0
    for i in range(BATCH_SIZE):
        end = start + SEQ_LENS[i]
        assert torch.allclose(
            q_rope[0, start:end].squeeze(),
            q_ref[i].squeeze(),
            atol=1e-2,
            rtol=1e-2,
        )
        start = end

    # Check if the cache was correctly updated:
    for i, cl in enumerate(CACHE_LOCS):
        assert torch.allclose(
            k_cache[cl, INPUT_POS[i] : INPUT_POS[i] + SEQ_LENS[i]].squeeze(),
            k_ref[i].squeeze(),
            atol=1e-2,
            rtol=1e-2,
        )
        assert torch.equal(
            v_cache[cl, INPUT_POS[i] : INPUT_POS[i] + SEQ_LENS[i]].squeeze(),
            v[i].squeeze(),
        )
