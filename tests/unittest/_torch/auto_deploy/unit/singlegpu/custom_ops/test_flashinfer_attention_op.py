import flashinfer
import pytest
import torch
from torch_attention_reference import TorchAttentionReference

from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_attention import _GlobalFlashInferPlanner


def _attention_with_fp8_kv_cache(
    q, k, v, k_cache, v_cache, k_scale, v_scale, prefill_seq_len, causal, mask
):
    """Simulates attention for fp8 kv cache with q,k,v outputs of GEMM in fp16"""
    batch_size, seq_len, _ = k.shape
    # Quantize k and v
    # k = (k / k_scale).to(torch.float8_e4m3fn)
    # v = (v / v_scale).to(torch.float8_e4m3fn)
    # Append to kv cache
    # k_cache[0:batch_size, prefill_seq_len : prefill_seq_len + seq_len, :, :] = k
    # v_cache[0:batch_size, prefill_seq_len : prefill_seq_len + seq_len, :, :] = v

    # Compute attention
    # Step 1: Retrieve KV cache
    k_ref = k_cache[0:batch_size, : prefill_seq_len + seq_len, :, :]
    v_ref = v_cache[0:batch_size, : prefill_seq_len + seq_len, :, :]
    # Step 2: Dequantize KV cache
    k_ref = k_ref.to(torch.float16) * k_scale
    v_ref = v_ref.to(torch.float16) * v_scale
    # Step 3: Apply RoPE
    # Step 4: Compute attention
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        is_causal=causal,
        attn_mask=mask,
    )

    return ref.transpose(1, 2)


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5095416")
@pytest.mark.parametrize("seq_length", [8, 32, 2048])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_attention_op_context(seq_length, n_heads, batch_size, dtype, device):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32
    DTYPE = dtype
    BATCH_SIZE = batch_size
    N_HEADS = n_heads
    SEQ_LEN = seq_length

    # metadata
    seq_len_tensor = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.int32, device=device)

    offsets = torch.zeros(BATCH_SIZE, device=device, dtype=torch.int)

    qo_indptr = torch.cat(
        (torch.zeros_like(seq_len_tensor[:1]), torch.cumsum(seq_len_tensor, 0))
    ).to(torch.int32)
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(BATCH_SIZE).int().to(device)
    paged_kv_last_page_len = offsets + seq_len_tensor

    # Q,K,V are computed using GEMM.
    q = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # Setup KV Cache. KV cache is empty, context phase
    k_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )
    v_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )

    # make sure planner is initialized
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _GlobalFlashInferPlanner.init_workspace(workspace)

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            paged_kv_indptr, paged_kv_last_page_len, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * SEQ_LEN,
    )
    flashinfer_output = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q,
        k,
        v,
        # METADATA
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        1.0,
        1.0,
    )

    # Use torch backend as clean reference
    q_reshaped = q.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)
    k_reshaped = k.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)
    v_reshaped = v.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)

    ref = TorchAttentionReference.basic_mha_with_cache(
        q_reshaped,
        k_reshaped,
        v_reshaped,
        k_cache,
        v_cache,
        torch.zeros(BATCH_SIZE, device=device, dtype=torch.int),
    )

    assert torch.allclose(
        flashinfer_output.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("seq_length", [1])
@pytest.mark.parametrize("prefill_seq_length", [0, 1, 4, 2047])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_attention_op_decode(
    prefill_seq_length, seq_length, batch_size, n_heads, dtype, device
):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32
    DTYPE = dtype
    BATCH_SIZE = batch_size
    N_HEADS = n_heads
    SEQ_LEN = seq_length
    PREFILL_SEQ_LEN = prefill_seq_length

    seq_len_tensor = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.int32).to(device)

    offsets = torch.tensor([PREFILL_SEQ_LEN] * BATCH_SIZE, device=device, dtype=torch.int)

    qo_indptr = torch.cat(
        (torch.zeros_like(seq_len_tensor[:1]), torch.cumsum(seq_len_tensor, 0))
    ).to(torch.int32)
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(BATCH_SIZE).int().to(device)
    paged_kv_last_page_len = offsets + seq_len_tensor

    # Q,K,V are computed using GEMM.
    q = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k = torch.ones(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # Setup KV Cache. KV cache is partially filled from the context phase
    k_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )
    v_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )

    k_cache[0:BATCH_SIZE, 0:PREFILL_SEQ_LEN, :, :] = torch.zeros(
        BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS, D_HEAD
    )
    v_cache[0:BATCH_SIZE, 0:PREFILL_SEQ_LEN, :, :] = torch.zeros(
        BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS, D_HEAD
    )

    # Generate reference cache
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()

    # Apply RoPE to k_cache
    k_cache_ref[0:BATCH_SIZE, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + SEQ_LEN, :, :] = k.view(
        BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD
    )
    v_cache_ref[0:BATCH_SIZE, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + SEQ_LEN, :, :] = v.view(
        BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD
    )

    assert not torch.allclose(
        k_cache_ref.to(torch.float32),
        k_cache.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    assert not torch.allclose(
        v_cache_ref.to(torch.float32),
        v_cache.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # make sure planner is initialized
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _GlobalFlashInferPlanner.init_workspace(workspace)

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            paged_kv_indptr, paged_kv_last_page_len, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * SEQ_LEN,
    )
    flashinfer_output = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q,
        k,
        v,
        # METADATA
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        1.0,
        1.0,
    )

    assert torch.allclose(
        k_cache_ref.to(torch.float32),
        k_cache.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.allclose(
        v_cache_ref.to(torch.float32),
        v_cache.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # Generate reference outputs
    q_ref = q.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)

    k_ref = k_cache[:BATCH_SIZE, : PREFILL_SEQ_LEN + SEQ_LEN, :, :]
    v_ref = v_cache[:BATCH_SIZE, : PREFILL_SEQ_LEN + SEQ_LEN, :, :]
    k_ref[:, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + SEQ_LEN, :, :] = k.view(
        BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD
    )
    v_ref[:, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + SEQ_LEN, :, :] = v.view(
        BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD
    )

    # Use torch backend as clean reference for decode with prefilled cache
    ref = TorchAttentionReference.decode_with_prefilled_cache(
        q_ref,
        k_ref,
        v_ref,
        k_cache,
        v_cache,
        torch.tensor([PREFILL_SEQ_LEN] * BATCH_SIZE, device=device, dtype=torch.int),
    )

    assert torch.allclose(
        flashinfer_output.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5095416")
@pytest.mark.parametrize("prefill_seq_length", [4, 10, 2047])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_attention_context_and_generate(
    prefill_seq_length, n_heads, batch_size, dtype, device
):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32
    DTYPE = dtype
    BATCH_SIZE = batch_size
    N_HEADS = n_heads
    PREFILL_SEQ_LEN = prefill_seq_length

    # Prefill phase
    seq_len_tensor = torch.tensor([prefill_seq_length] * BATCH_SIZE, dtype=torch.int32).to(device)

    offsets = torch.zeros(BATCH_SIZE, device=device, dtype=torch.int)

    qo_indptr = torch.cat(
        (torch.zeros_like(seq_len_tensor[:1]), torch.cumsum(seq_len_tensor, 0))
    ).to(torch.int32)
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(BATCH_SIZE).int().to(device)
    paged_kv_last_page_len = offsets + seq_len_tensor

    # Q,K,V for prefill phase
    q_1 = torch.randn(BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k_1 = torch.randn(BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v_1 = torch.randn(BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # Setup KV Cache. KV cache is empty, context phase
    k_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )
    v_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )

    # make sure planner is initialized
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _GlobalFlashInferPlanner.init_workspace(workspace)

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            paged_kv_indptr, paged_kv_last_page_len, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * PREFILL_SEQ_LEN,
    )
    flashinfer_output_1 = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q_1,
        k_1,
        v_1,
        # METADATA
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        1.0,
        1.0,
    )

    # Generate reference outputs
    q_ref = q_1
    k_ref = k_cache[:BATCH_SIZE, 0:PREFILL_SEQ_LEN, :, :]
    v_ref = v_cache[:BATCH_SIZE, 0:PREFILL_SEQ_LEN, :, :]

    # Use torch backend as clean reference
    ref = TorchAttentionReference.basic_mha_with_cache(
        q_ref.view(BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS, D_HEAD),
        k_ref.transpose(1, 2).transpose(2, 3),  # Convert [B,N,S,D] to [B,S,N,D]
        v_ref.transpose(1, 2).transpose(2, 3),  # Convert [B,N,S,D] to [B,S,N,D]
        k_cache,
        v_cache,
        torch.zeros(BATCH_SIZE, device=device, dtype=torch.int),
    )
    flashinfer_output_1 = flashinfer_output_1.view(BATCH_SIZE, -1, N_HEADS, D_HEAD)

    assert torch.allclose(
        flashinfer_output_1.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    k_cache[:BATCH_SIZE, 0:PREFILL_SEQ_LEN, :, :] = k_ref

    # Generate output
    seq_len_tensor = torch.tensor([1] * BATCH_SIZE, dtype=torch.int32).to(device)

    offsets = torch.tensor([PREFILL_SEQ_LEN] * BATCH_SIZE, device=device, dtype=torch.int)

    qo_indptr = torch.cat(
        (torch.zeros_like(seq_len_tensor[:1]), torch.cumsum(seq_len_tensor, 0))
    ).to(torch.int32)
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(BATCH_SIZE).int().to(device)
    paged_kv_last_page_len = offsets + seq_len_tensor

    # Q,K,V are computed using GEMM.
    q_3 = torch.randn(BATCH_SIZE, 1, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k_3 = torch.randn(BATCH_SIZE, 1, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v_3 = torch.randn(BATCH_SIZE, 1, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # Create FlashInferAttention class before calling the custom op
    _GlobalFlashInferPlanner.reset()

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            paged_kv_indptr, paged_kv_last_page_len, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * 1,
    )
    flashinfer_output_3 = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q_3,
        k_3,
        v_3,
        # METADATA
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        1.0,
        1.0,
    )

    # Generate reference outputs
    q_ref = torch.cat([q_1, q_3], dim=-2)
    k_ref = k_cache[:BATCH_SIZE, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + 1, :, :]
    v_ref = v_cache[:BATCH_SIZE, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + 1, :, :]

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_3.view(BATCH_SIZE, 1, N_HEADS, D_HEAD).transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        is_causal=True,
    )

    ref = ref.transpose(1, 2)
    ref = ref[0:BATCH_SIZE, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + 1, :, :]
    flashinfer_output_3 = flashinfer_output_3.view(BATCH_SIZE, -1, N_HEADS, D_HEAD)

    assert torch.allclose(
        flashinfer_output_3.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "seq",
    [
        (2, 1),
        (2, 64),
        (2, 2046),
        (16, 1),
        (16, 2022),
        (2047, 1),
        (1984, 64),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_attention_op_context_input_pos(seq, batch_size, n_heads, dtype, device):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32
    DTYPE = dtype
    BATCH_SIZE = batch_size
    N_HEADS = n_heads
    SEQ_LEN = seq[0]
    PREFILL_SEQ_LEN = seq[1]

    seq_len_tensor = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.int32).to(device)

    offsets = torch.tensor([PREFILL_SEQ_LEN] * BATCH_SIZE, device=device, dtype=torch.int)

    qo_indptr = torch.cat(
        (torch.zeros_like(seq_len_tensor[:1]), torch.cumsum(seq_len_tensor, 0))
    ).to(torch.int32)
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(BATCH_SIZE).int().to(device)
    paged_kv_last_page_len = offsets + seq_len_tensor

    # Q,K,V are computed using GEMM.
    q = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # Setup KV Cache. KV cache is partially filled from the context phase
    k_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )
    v_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )

    # make sure planner is initialized
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _GlobalFlashInferPlanner.init_workspace(workspace)

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            paged_kv_indptr, paged_kv_last_page_len, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * SEQ_LEN,
    )
    flashinfer_output = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q,
        k,
        v,
        # METADATA
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        1.0,
        1.0,
    )

    # Generate ref
    q_ref = q.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)
    k_ref = k_cache[0:BATCH_SIZE, 0 : PREFILL_SEQ_LEN + SEQ_LEN, :, :]
    v_ref = v_cache[0:BATCH_SIZE, 0 : PREFILL_SEQ_LEN + SEQ_LEN, :, :]

    q_ref = q_ref.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)
    k_ref = k_ref.view(BATCH_SIZE, PREFILL_SEQ_LEN + SEQ_LEN, N_HEADS, D_HEAD)
    mask = torch.cat(
        [
            torch.ones(SEQ_LEN, PREFILL_SEQ_LEN, device=device, dtype=torch.bool),
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=device, dtype=torch.bool)),
        ],
        dim=1,
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        attn_mask=mask,
    )

    ref = ref.transpose(1, 2).contiguous()
    ref = ref.view(BATCH_SIZE, -1, N_HEADS * D_HEAD)
    assert torch.allclose(
        flashinfer_output.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


# TODO: Get fp8 kv cache unit tests working with RoPE
# TODO: fix fp8 kv cache for sm90+
@pytest.mark.skipif(
    torch.cuda.get_device_capability(0) >= (9, 0),
    reason="flashinfer fp8 kv cache not supported on sm90",
)
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("kv_scales", [(1.0, 1.0)])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize(
    "seq_length", [(0, 1), (0, 8), (0, 2048), (1, 1), (8, 1), (2047, 1), (1024, 1024)]
)
def test_flashinfer_attention_with_fp8_cache(
    seq_length, n_heads, batch_size, dtype, device, kv_scales
):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32
    DTYPE = dtype
    BATCH_SIZE = batch_size
    N_HEADS = n_heads
    SEQ_LEN = seq_length[1]
    PREFILL_SEQ_LEN = seq_length[0]
    K_SCALE = kv_scales[0]
    V_SCALE = kv_scales[1]

    seq_len_tensor = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.int32).to(device)

    offsets = torch.tensor([PREFILL_SEQ_LEN] * BATCH_SIZE, device=device, dtype=torch.int)

    qo_indptr = torch.cat(
        (torch.zeros_like(seq_len_tensor[:1]), torch.cumsum(seq_len_tensor, 0))
    ).to(torch.int32)
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(BATCH_SIZE).int().to(device)
    paged_kv_last_page_len = offsets + seq_len_tensor

    # Q,K,V are computed using GEMM, in fp16
    q = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v = torch.randn(BATCH_SIZE, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # Setup KV Cache. KV cache is empty, context phase
    k_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )
    v_cache = torch.zeros(
        (MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD), dtype=DTYPE, device=device
    )

    if PREFILL_SEQ_LEN > 0:
        k_cache[0:BATCH_SIZE, 0:PREFILL_SEQ_LEN, :, :] = torch.randn(
            BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS, D_HEAD
        )
        v_cache[0:BATCH_SIZE, 0:PREFILL_SEQ_LEN, :, :] = torch.randn(
            BATCH_SIZE, PREFILL_SEQ_LEN, N_HEADS, D_HEAD
        )

        k_cache = k_cache / K_SCALE
        v_cache = v_cache / V_SCALE

        # Set causal mask to false if its a partially filled kv_cache
        causal = False
        mask = None
        if SEQ_LEN > 1:
            # Set custom mask
            mask = torch.cat(
                [
                    torch.ones(SEQ_LEN, PREFILL_SEQ_LEN, device=device, dtype=torch.bool),
                    torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=device, dtype=torch.bool)),
                ],
                dim=1,
            )
    else:
        causal = True
        mask = None

    k_cache = k_cache.to(torch.float8_e4m3fn)
    v_cache = v_cache.to(torch.float8_e4m3fn)

    # make sure planner is initialized
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _GlobalFlashInferPlanner.init_workspace(workspace)

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            paged_kv_indptr, paged_kv_last_page_len, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * SEQ_LEN,
    )
    flashinfer_output = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q,
        k,
        v,
        # METADATA
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        K_SCALE,
        V_SCALE,
    )

    y = flashinfer_output.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)
    q = q.view(BATCH_SIZE, SEQ_LEN, N_HEADS, D_HEAD)

    ref = _attention_with_fp8_kv_cache(
        q, k, v, k_cache, v_cache, K_SCALE, V_SCALE, PREFILL_SEQ_LEN, causal, mask
    )

    assert torch.allclose(
        y.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5095416")
@pytest.mark.parametrize("seq_lengths", [[8, 14], [11, 19, 22, 49]])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_attention_with_paged_kvcache(seq_lengths, n_heads, dtype, device):
    PAGE_SIZE = 8
    D_HEAD = 64
    MAX_SEQ_LEN = 128
    MAX_BATCH_SIZE = 32
    DTYPE = dtype
    BATCH_SIZE = len(seq_lengths)
    N_HEADS = n_heads
    SEQ_LEN = sum(seq_lengths)

    MAX_NUM_PAGES = MAX_BATCH_SIZE * MAX_SEQ_LEN // PAGE_SIZE
    seq_len_tensor = torch.tensor(seq_lengths, dtype=torch.int32).to(device)

    q = torch.randn(1, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k = torch.randn(1, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v = torch.randn(1, SEQ_LEN, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # Setup KV Cache. KV cache is empty, context phase
    k_cache = torch.zeros((MAX_NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD), dtype=DTYPE, device=device)
    v_cache = torch.zeros((MAX_NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD), dtype=DTYPE, device=device)
    offsets = torch.zeros(BATCH_SIZE, device=device, dtype=torch.int)

    # assign pages
    free_pages = torch.randperm(MAX_NUM_PAGES).int().tolist()

    pages_per_seq = [(s - 1) // PAGE_SIZE + 1 for s in seq_lengths]
    page_assignments = [[free_pages.pop() for _ in range(np)] for np in pages_per_seq]

    num_pages_per_seq = torch.tensor(
        [len(p) for p in page_assignments], device=device, dtype=torch.int
    )
    assert sum(num_pages_per_seq) < MAX_NUM_PAGES

    # convert to flashinfer convention
    qo_indptr = torch.zeros(BATCH_SIZE + 1, device=device, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(seq_len_tensor, 0)
    paged_kv_indptr = torch.zeros_like(qo_indptr)
    paged_kv_indptr[1:] = torch.cumsum(num_pages_per_seq, 0)
    paged_kv_indices = torch.tensor(
        [p for ps in page_assignments for p in ps], device=device, dtype=torch.int
    )
    paged_kv_last_page_len = ((offsets + seq_len_tensor - 1) % PAGE_SIZE) + 1

    # make sure planner is initialized
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _GlobalFlashInferPlanner.init_workspace(workspace)

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            paged_kv_indptr, paged_kv_last_page_len, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * SEQ_LEN,
    )
    flashinfer_output = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q,
        k,
        v,
        # METADATA
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        1.0,
        1.0,
    )

    # Compute reference
    ref = []
    for i, s in enumerate(seq_lengths):
        qq = q[0, qo_indptr[i] : qo_indptr[i + 1], :].view(1, s, N_HEADS, D_HEAD)
        kk = k[0, qo_indptr[i] : qo_indptr[i + 1], :].view(1, s, N_HEADS, D_HEAD)
        vv = v[0, qo_indptr[i] : qo_indptr[i + 1], :].view(1, s, N_HEADS, D_HEAD)
        oo = torch.nn.functional.scaled_dot_product_attention(
            qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), is_causal=True
        )
        ref.append(oo.transpose(1, 2).contiguous().view(s, N_HEADS * D_HEAD))

    ref = torch.cat(ref, dim=0)

    assert torch.allclose(
        flashinfer_output.squeeze().cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # Now let's generate 1 more token
    seq_len2 = [1] * BATCH_SIZE
    seq_len_tensor2 = torch.tensor(seq_len2, dtype=torch.int32).to(device)
    # Start index of each query in each batch/request
    offsets2 = torch.tensor(seq_lengths, device=device, dtype=torch.int)

    q_gen = torch.randn(BATCH_SIZE, 1, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    k_gen = torch.randn(BATCH_SIZE, 1, N_HEADS * D_HEAD, dtype=DTYPE).to(device)
    v_gen = torch.randn(BATCH_SIZE, 1, N_HEADS * D_HEAD, dtype=DTYPE).to(device)

    # update assignment
    for pages, last_p_len in zip(page_assignments, paged_kv_last_page_len):
        if last_p_len == PAGE_SIZE:
            pages.append(free_pages.pop())

    num_pages_per_seq2 = torch.tensor(
        [len(p) for p in page_assignments], device=device, dtype=torch.int
    )
    assert sum(num_pages_per_seq2) < MAX_NUM_PAGES

    # convert to flashinfer convention
    qo_indptr2 = torch.zeros(BATCH_SIZE + 1, device=device, dtype=torch.int32)
    qo_indptr2[1:] = torch.cumsum(seq_len_tensor2, 0)
    paged_kv_indptr2 = torch.zeros_like(qo_indptr)
    paged_kv_indptr2[1:] = torch.cumsum(num_pages_per_seq2, 0)
    paged_kv_indices2 = torch.tensor(
        [p for ps in page_assignments for p in ps], device=device, dtype=torch.int
    )
    paged_kv_last_page_len2 = ((offsets2 + seq_len_tensor2 - 1) % PAGE_SIZE) + 1

    # Create FlashInferAttention class before calling the custom op
    _GlobalFlashInferPlanner.reset()

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr2,
        flashinfer.get_seq_lens(
            paged_kv_indptr2, paged_kv_last_page_len2, page_size=k_cache.shape[1]
        ),
        BATCH_SIZE * 1,
    )
    flashinfer_output_gen = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        # Q, K, V
        q_gen,
        k_gen,
        v_gen,
        # METADATA
        qo_indptr2,
        paged_kv_indptr2,
        paged_kv_indices2,
        paged_kv_last_page_len2,
        batch_indices,
        positions,
        # CACHES
        k_cache,
        v_cache,
        # BUFFERS
        workspace,
        # CONSTANTS
        None,
        1.0,
        1.0,
    )

    # Compute reference
    # Here we compute the output for the new query using all the previous keys and values.
    ref = []
    for i, s in enumerate(seq_lengths):
        qq = q_gen[i : i + 1, :, :].view(1, 1, N_HEADS, D_HEAD)

        kk = k[0, qo_indptr[i] : qo_indptr[i + 1], :].view(1, s, N_HEADS, D_HEAD)
        kk = torch.cat([kk, k_gen[i : i + 1, :, :].view(1, 1, N_HEADS, D_HEAD)], dim=1)

        vv = v[0, qo_indptr[i] : qo_indptr[i + 1], :].view(1, s, N_HEADS, D_HEAD)
        vv = torch.cat([vv, v_gen[i : i + 1, :, :].view(1, 1, N_HEADS, D_HEAD)], dim=1)
        oo = torch.nn.functional.scaled_dot_product_attention(
            qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), is_causal=False
        )
        ref.append(oo.transpose(1, 2).contiguous().view(1, N_HEADS * D_HEAD))
    ref = torch.cat(ref, dim=0)

    assert torch.allclose(
        flashinfer_output_gen.squeeze().cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
