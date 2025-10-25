import pytest
import torch
from utils.util import skip_pre_hopper

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.load_moe_align import moe_align_block_size


def _pack_routed_tokens_reference(
    topk_ids: torch.Tensor,
    M: int,
    E: int,
    top_k: int,
    block_size_m: int,
):
    """
    Reference implementation based on the provided previous algorithm.
    Produces used-region outputs (excluding sentinel-E blocks) for ground-truth comparison.
    Returns: (sorted_token_ids_used[int64], expert_ids_used[int32], num_tokens_post_padded[int])
    """
    device = topk_ids.device
    T = M * top_k

    # Flatten and clamp invalid experts to sentinel E
    experts_raw = topk_ids.reshape(-1).to(torch.int64)
    within = (experts_raw >= 0) & (experts_raw < E)
    experts = torch.where(within, experts_raw, torch.full_like(experts_raw, E))

    tokens = torch.arange(T, device=device, dtype=torch.int64)

    # Sort by expert id to group tokens
    perm = torch.argsort(experts)
    e_sorted = experts[perm]
    t_sorted = tokens[perm]

    # Counts per expert including sentinel E
    counts = torch.zeros(E + 1, device=device, dtype=torch.int64)
    ones = torch.ones_like(e_sorted, dtype=torch.int64)
    counts.scatter_add_(0, e_sorted, ones)

    # Compute per-expert number of blocks and offsets
    blocks_per_expert = (counts + (block_size_m - 1)) // block_size_m
    block_offsets = torch.cumsum(blocks_per_expert, dim=0) - blocks_per_expert

    # Positions within each expert run
    idx_range = torch.arange(T, device=device, dtype=torch.int64)
    cum_counts = torch.cumsum(counts, dim=0)
    group_end = cum_counts[e_sorted]
    group_start = group_end - counts[e_sorted]
    pos_in_run = idx_range - group_start

    block_idx_in_e = pos_in_run // block_size_m
    idx_in_block = pos_in_run % block_size_m

    # Used blocks exclude sentinel E
    used_blocks = int(blocks_per_expert[:E].sum().item())
    used_len = used_blocks * block_size_m

    # Map to contiguous used-block ids by ignoring sentinel E region
    global_block_id = block_offsets[e_sorted] + block_idx_in_e
    mask_real = e_sorted < E
    used_block_id = global_block_id[mask_real]
    used_token_ids = t_sorted[mask_real]

    # Outputs sized to used region
    sorted_token_ids_used = torch.full((used_len,), T, device=device, dtype=torch.int64)
    expert_ids_used = torch.full((used_blocks,), -1, device=device, dtype=torch.int32)

    out_index = used_block_id * block_size_m + idx_in_block[mask_real]
    sorted_token_ids_used[out_index] = used_token_ids

    e_for_block = e_sorted[mask_real]
    expert_ids_used.index_copy_(0, used_block_id, e_for_block.to(torch.int32))

    return sorted_token_ids_used, expert_ids_used, used_len


def test_triton_moe_matches_torch_moe_mlp_relu2():
    torch.manual_seed(0)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for triton_moe fused MLP test")
    device = "cuda"
    dtype = torch.bfloat16

    M = 8  # tokens
    HIDDEN_SIZE = 8
    INTERMEDIATE_SIZE = 16
    E = 8  # experts
    top_k = 2

    x = torch.randn(M, HIDDEN_SIZE, device=device, dtype=dtype)

    w_up_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) for _ in range(E)
    ]
    w_down_list = [
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype) for _ in range(E)
    ]

    # Triton fused kernel expects stacked weights
    w_up_stacked = torch.stack(w_up_list, dim=0).contiguous()  # [E, I, H]
    w_down_stacked = torch.stack(w_down_list, dim=0).contiguous()  # [E, H, I]

    # Create routing with top-k normalization
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_full = torch.softmax(router_logits, dim=-1)
    routing_weights, selected_experts = torch.topk(routing_full, k=top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(torch.float32)

    # Triton fused MoE (mlp with relu^2 activation between two GEMMs)
    out_triton = torch.ops.auto_deploy.triton_moe_fused(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w_up_stacked,
        w_down_stacked,
    )

    # Reference Torch MoE in mlp mode with relu2 activation
    out_torch = torch.ops.auto_deploy.torch_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight=w_up_list,
        w2_weight=w_down_list,
        w3_weight=[],
        mlp_style="mlp",
        act_fn="relu2",
    )
    torch.testing.assert_close(out_triton, out_torch, rtol=5e-2, atol=5e-2)


def test_moe_align_kernel_groups_tokens_by_expert_and_block_padding():
    torch.manual_seed(0)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for moe_align kernel test")

    device = "cuda"

    # Problem size
    M = 8
    E = 8
    top_k = 2
    block_size = 4  # small block to exercise padding across experts

    # Generate routing ids in-range [0, E)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_ids_flat = topk_ids.reshape(-1)
    T = topk_ids_flat.numel()

    # Allocate capacity buffers as the CUDA kernel expects
    max_num_tokens_padded = T + E * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_token_ids = torch.empty(max_num_tokens_padded, device=device, dtype=torch.int32)
    expert_ids = torch.empty(max_num_m_blocks, device=device, dtype=torch.int32)
    num_tokens_post_pad = torch.empty(1, device=device, dtype=torch.int32)

    # Invoke CUDA kernel
    moe_align_block_size(
        topk_ids_flat,
        E,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    # Reference computation on CPU (order within an expert is not enforced)
    topk_ids_cpu = topk_ids_flat.cpu().to(torch.int64)
    counts = torch.bincount(topk_ids_cpu, minlength=E)
    blocks_per_expert = (counts + (block_size - 1)) // block_size
    num_tokens_post_pad_ref = int((blocks_per_expert * block_size).sum().item())
    used_blocks_ref = int(blocks_per_expert.sum().item())

    # 1) Padded token count must match and be a multiple of block_size
    num_tokens_post_pad_val = int(num_tokens_post_pad.item())
    assert num_tokens_post_pad_val == num_tokens_post_pad_ref
    assert num_tokens_post_pad_val % block_size == 0

    # 2) expert_ids for used blocks must match expected distribution per expert
    expected_expert_ids = torch.cat(
        [torch.full((int(blocks_per_expert[e].item()),), e, dtype=torch.int32) for e in range(E)]
    )
    assert expert_ids[:used_blocks_ref].cpu().tolist() == expected_expert_ids.tolist()

    # 3) sorted_token_ids semantics
    #    - Each token index [0, T) appears exactly once in the used region
    #    - Sentinel value T pads the remainder (inside used region and beyond)
    used_sorted = sorted_token_ids[:num_tokens_post_pad_val].cpu().to(torch.int64)
    counts_all = torch.bincount(used_sorted, minlength=T + 1)
    # All real tokens appear exactly once
    assert torch.all(counts_all[:T] == 1)
    # Sentinel count equals total padding
    sentinel_pad = int(num_tokens_post_pad_val - T)
    assert int(counts_all[T].item()) == sentinel_pad

    # 4) Per-block expert grouping: every non-sentinel token in a block belongs to that block's expert
    for block_idx in range(used_blocks_ref):
        e = int(expert_ids[block_idx].item())
        block_vals = used_sorted[block_idx * block_size : (block_idx + 1) * block_size]
        non_sentinel = block_vals[block_vals < T]
        if non_sentinel.numel() == 0:
            continue
        assert torch.all(topk_ids_cpu[non_sentinel] == e)

    # 5) Compare against reference implementation (ground truth)
    ref_sorted_used, ref_expert_ids, ref_used_len = _pack_routed_tokens_reference(
        topk_ids, M, E, top_k, block_size
    )
    assert ref_used_len == num_tokens_post_pad_val
    assert ref_expert_ids.numel() == used_blocks_ref
    assert expert_ids[:used_blocks_ref].cpu().tolist() == ref_expert_ids.cpu().tolist()

    ref_counts_all = torch.bincount(ref_sorted_used.cpu().to(torch.int64), minlength=T + 1)
    assert torch.all(ref_counts_all == counts_all)


@skip_pre_hopper
def test_triton_quant_fp8_moe_matches_torch_quant_fp8_moe():
    """Test triton_quant_fp8_moe against torch_quant_fp8_moe reference."""
    torch.manual_seed(0)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for triton_quant_fp8_moe test")
    device = "cuda"
    dtype = torch.bfloat16

    M = 32  # tokens
    HIDDEN_SIZE = 16  # Must be multiple of 16 for FP8 linear
    INTERMEDIATE_SIZE = 32  # Must be multiple of 16 for FP8 linear
    E = 4  # experts
    top_k = 2

    # Use small normalized values to avoid FP8 range issues
    x = torch.randn(M, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1

    # Create BF16 weights for each expert (normalized to small values)
    w_up_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1
        for _ in range(E)
    ]
    w_down_list = [
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype) * 0.1
        for _ in range(E)
    ]

    # Stack weights [E, ...]
    w_up_stacked = torch.stack(w_up_list, dim=0).contiguous()  # [E, I, H]
    w_down_stacked = torch.stack(w_down_list, dim=0).contiguous()  # [E, H, I]

    # Quantize weights to FP8 with per-expert scales
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

    # Per-expert weight scales (use max absolute value per expert)
    w1_weight_scale = torch.tensor(
        [w_up_stacked[e].abs().max().item() / FP8_MAX for e in range(E)],
        device=device,
        dtype=torch.float32,
    )
    w2_weight_scale = torch.tensor(
        [w_down_stacked[e].abs().max().item() / FP8_MAX for e in range(E)],
        device=device,
        dtype=torch.float32,
    )

    # Quantize weights and stack
    w1_fp8_list = [
        (w_up_stacked[e] / w1_weight_scale[e]).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)
        for e in range(E)
    ]
    w2_fp8_list = [
        (w_down_stacked[e] / w2_weight_scale[e]).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)
        for e in range(E)
    ]
    w1_fp8_stacked = torch.stack(w1_fp8_list).contiguous()
    w2_fp8_stacked = torch.stack(w2_fp8_list).contiguous()

    # Input scales (tensor-wise, replicated per expert for interface compatibility)
    x_scale = x.abs().max().item() / FP8_MAX
    w1_input_scale_tensor = torch.full((E,), x_scale, device=device, dtype=torch.float32)

    # Compute intermediate activation scale by simulating first GEMM + ReLU^2
    # This ensures w2_input_scale matches the actual activation magnitude
    with torch.no_grad():
        # Simulate the first GEMM: quantize input, do FP8 matmul, apply ReLU^2
        x_q = (x / w1_input_scale_tensor[0]).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)
        # Dequantize and compute output for a sample
        x_dq = x_q[:8].to(torch.float32) * w1_input_scale_tensor[0].item()
        w1_dq = w1_fp8_stacked[0].to(torch.float32) * w1_weight_scale[0].item()
        sample_out = torch.nn.functional.linear(x_dq.to(dtype), w1_dq.to(dtype))
        sample_act = torch.square(torch.nn.functional.relu(sample_out))
        intermediate_scale = sample_act.abs().max().item() / FP8_MAX
        # Ensure scale is not too small
        intermediate_scale = max(intermediate_scale, 1e-6)

    w2_input_scale_tensor = torch.full((E,), intermediate_scale, device=device, dtype=torch.float32)

    # Convert scales to lists for torch_quant_fp8_moe reference
    w1_input_scale_list = [w1_input_scale_tensor[0].clone() for _ in range(E)]
    w2_input_scale_list = [w2_input_scale_tensor[0].clone() for _ in range(E)]
    w1_weight_scale_list = [w1_weight_scale[e].clone() for e in range(E)]
    w2_weight_scale_list = [w2_weight_scale[e].clone() for e in range(E)]

    # Dummy w3 tensors (unused for mlp style)
    w3_fp8_list = [torch.empty((1, 1), device=device, dtype=torch.float8_e4m3fn) for _ in range(E)]
    w3_fp8_stacked = torch.stack(w3_fp8_list).contiguous()
    w3_input_scale_list = [torch.ones((), device=device, dtype=torch.float32) for _ in range(E)]
    w3_input_scale_tensor = torch.ones((E,), device=device, dtype=torch.float32)
    w3_weight_scale_list = [torch.ones((), device=device, dtype=torch.float32) for _ in range(E)]
    w3_weight_scale_tensor = torch.ones((E,), device=device, dtype=torch.float32)

    # Create controlled routing to ensure even token distribution across experts
    selected_experts = torch.zeros((M, top_k), dtype=torch.int64, device=device)
    for i in range(M):
        # Distribute tokens evenly: token i goes to experts (i % E) and ((i+1) % E)
        selected_experts[i, 0] = i % E
        selected_experts[i, 1] = (i + 1) % E

    # Create equal routing weights
    routing_weights = torch.ones((M, top_k), device=device, dtype=torch.float32) / top_k

    # Triton FP8 quantized MoE (uses stacked tensors)
    out_triton = torch.ops.auto_deploy.triton_quant_fp8_moe(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w1_fp8_stacked,
        w2_fp8_stacked,
        w3_fp8_stacked,
        w1_input_scale_tensor,
        w2_input_scale_tensor,
        w3_input_scale_tensor,
        w1_weight_scale,
        w2_weight_scale,
        w3_weight_scale_tensor,
        mlp_style="mlp",
        act_fn="relu2",
    )

    # Reference: Torch quantized FP8 MoE (uses lists of tensors and scales)
    out_torch = torch.ops.auto_deploy.torch_quant_fp8_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight=w1_fp8_list,
        w2_weight=w2_fp8_list,
        w3_weight=w3_fp8_list,
        w1_input_scale=w1_input_scale_list,
        w2_input_scale=w2_input_scale_list,
        w3_input_scale=w3_input_scale_list,
        w1_weight_scale=w1_weight_scale_list,
        w2_weight_scale=w2_weight_scale_list,
        w3_weight_scale=w3_weight_scale_list,
        mlp_style="mlp",
        act_fn="relu2",
    )

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-2, atol=1e-2)
