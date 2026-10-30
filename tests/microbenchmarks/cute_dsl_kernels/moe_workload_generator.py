import json
import os
from typing import List, Optional

import click
import safetensors.torch
import torch

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
from tensorrt_llm.tools.layer_wise_benchmarks.runner import (
    get_balanced_selection_impl_default,
    get_balanced_selection_impl_random,
)


def get_balanced_selection_impl_default_legacy(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    dp_size: int,
    dp_rank: int,
    ep_size: int,
):
    world_size = ep_size
    rank = dp_rank
    # First, each sender selects target rank
    target_rank_before_mod = torch.arange(num_tokens * world_size * top_k).view(
        num_tokens, world_size, top_k
    )
    target_rank_before_mod += top_k * torch.arange(num_tokens).view(
        num_tokens, 1, 1
    )  # Shift `top_k` ranks for the next token on each rank, to balance network traffic
    target_rank = target_rank_before_mod % world_size
    # Second, each receiver selects target expert
    target_expert = torch.empty_like(target_rank)
    for reciever_rank in range(world_size):
        mask = target_rank == reciever_rank
        experts_per_rank = num_experts // world_size
        local_expert = torch.arange(num_tokens * top_k) % experts_per_rank
        target_expert[mask] = (reciever_rank * experts_per_rank) + local_expert
    token_selected_experts = target_expert[:, rank].sort(dim=-1).values
    return token_selected_experts.contiguous().to(dtype=dtype, device=device)


def gen_moe_workload(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    num_tokens_per_expert: Optional[List[int]],
    tile_size: int,
    method: str = "balanced_random",
):
    if num_tokens_per_expert is not None:
        num_local_experts = len(num_tokens_per_expert)
        assert num_local_experts * ep_size == num_experts
        helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
        token_selected_experts = helper.generate_token_selected_experts(
            num_tokens * ep_size, num_tokens_per_expert
        )
        token_selected_experts = token_selected_experts.cuda()
    else:
        if method == "balanced_random":
            get_balanced_selection_impl = get_balanced_selection_impl_random
        elif method == "balanced_default":
            get_balanced_selection_impl = get_balanced_selection_impl_default
        elif method == "balanced_default_legacy":
            get_balanced_selection_impl = get_balanced_selection_impl_default_legacy
        else:
            raise ValueError(f"Invalid method: {method}.")

        token_selected_experts = [
            get_balanced_selection_impl(
                num_tokens=num_tokens,
                top_k=top_k,
                num_experts=num_experts,
                dtype=torch.int32,
                device="cuda",
                dp_size=ep_size,
                dp_rank=i,
                ep_size=ep_size,
            )
            for i in range(ep_size)
        ]
        token_selected_experts = torch.cat(token_selected_experts, dim=0)

    assert token_selected_experts.size() == (num_tokens * ep_size, top_k)
    token_final_scales = torch.ones_like(token_selected_experts, dtype=torch.float32)
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=0,
        local_num_experts=num_experts // ep_size,
        tile_tokens_dim=tile_size,
    )


@click.command("moe_workload_generator")
@click.option("--num_tokens", type=int, default=128)
@click.option("--top_k", type=int, default=8)
@click.option("--num_experts", type=int, default=256)
@click.option("--ep_size", type=int, default=32)
@click.option("--num_tokens_per_expert", type=str, default=None)
@click.option("--tile_size", type=click.Choice([128, 256]), default=128)
@click.option(
    "--method",
    type=click.Choice(["balanced_random", "balanced_default", "balanced_default_legacy"]),
    default="balanced_random",
)
@click.option("--seed", type=int, default=515)
@click.option("--output_path", type=str, default="./moe_workload")
def main(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    num_tokens_per_expert: str,
    tile_size: int,
    method: str,
    seed: int,
    output_path: str,
):
    torch.manual_seed(seed)

    if num_tokens_per_expert is not None:
        num_tokens_per_expert = [int(x) for x in num_tokens_per_expert.split(",")]

    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = gen_moe_workload(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        ep_size=ep_size,
        num_tokens_per_expert=num_tokens_per_expert,
        tile_size=tile_size,
        method=method,
    )

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    metadata = {
        "num_tokens": num_tokens,
        "top_k": top_k,
        "num_experts": num_experts,
        "ep_size": ep_size,
        "tile_size": tile_size,
        "method": method,
        "seed": seed,
    }
    with open(f"{output_path}/metadata.json", "w") as f:
        json.dump(metadata, f)

    workload = {
        "tile_idx_to_group_idx": tile_idx_to_group_idx,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx,
        "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
        "total_num_padded_tokens": total_num_padded_tokens,
        "num_non_exiting_tiles": num_non_exiting_tiles,
    }
    safetensors.torch.save_file(workload, f"{output_path}/workload.safetensors")


if __name__ == "__main__":
    main()
