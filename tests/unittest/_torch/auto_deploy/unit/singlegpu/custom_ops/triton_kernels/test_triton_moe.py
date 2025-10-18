import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.load_moe_align import moe_align_block_size


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
