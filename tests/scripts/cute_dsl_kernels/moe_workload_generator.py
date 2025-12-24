import click
import safetensors.torch
import torch

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import GroupedGemmInputsHelper
from tensorrt_llm.tools.layer_wise_benchmarks.runner_utils import get_balanced_selection_no_cache


def gen_moe_workload_layer_wise_benchmark(
    num_tokens: int, top_k: int, num_experts: int, ep_size: int, tile_size: int
):
    token_selected_experts = [
        get_balanced_selection_no_cache(
            num_tokens, top_k, num_experts, torch.int32, "cuda", ep_size, i
        )
        for i in range(ep_size)
    ]
    token_selected_experts = torch.cat(token_selected_experts, dim=0)
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


def gen_moe_workload_balanced_random(
    num_tokens: int, top_k: int, num_experts: int, ep_size: int, tile_size: int
):
    num_local_experts = num_experts // ep_size
    num_total_tokens = num_tokens * ep_size
    helper = GroupedGemmInputsHelper(num_experts, top_k, num_local_experts, 0, tile_size)
    num_tokens_per_expert = helper.generate_num_tokens_per_expert(num_total_tokens)
    if sum(num_tokens_per_expert) != num_total_tokens * top_k:
        num_tokens_per_expert[0] -= 1
    assert sum(num_tokens_per_expert) == num_total_tokens * top_k // ep_size

    token_expert_selection = torch.zeros(num_total_tokens, num_local_experts, dtype=torch.int32)
    token_selected_experts = -torch.ones(num_total_tokens, top_k, dtype=torch.int32)
    for j, curr_num_tokens in enumerate(num_tokens_per_expert):
        curr_num_selected_tokens = 0
        for i in torch.randperm(num_total_tokens).tolist():
            if (curr_num_selected_experts := token_expert_selection[i].sum()) < top_k:
                token_selected_experts[i, curr_num_selected_experts] = j
                token_expert_selection[i, j] = 1
                curr_num_selected_tokens += 1
                if curr_num_selected_tokens >= curr_num_tokens:
                    break
    assert (
        ((token_selected_experts >= 0).sum(dim=-1) == token_expert_selection.sum(dim=-1))
        .all()
        .item()
    )

    token_selected_experts = token_selected_experts.cuda()
    token_final_scales = torch.ones_like(token_selected_experts, dtype=torch.float32)
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        tile_tokens_dim=tile_size,
    )


@click.command("moe_workload_generator")
@click.option("--num_tokens", type=int, default=128)
@click.option("--top_k", type=int, default=8)
@click.option("--num_experts", type=int, default=256)
@click.option("--ep_size", type=int, default=32)
@click.option("--tile_size", type=click.Choice([128, 256]), default=128)
@click.option(
    "--method",
    type=click.Choice(["balanced_random", "balanced_layer_wise_benchmark"]),
    default="balanced_random",
)
@click.option("--seed", type=int, default=515)
@click.option("--output_file", type=str, default="./moe_workload.safetensors")
def main(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    tile_size: int,
    method: str,
    seed: int,
    output_file: str,
):
    torch.manual_seed(seed)

    if method == "balanced_random":
        generator = gen_moe_workload_balanced_random
    elif method == "balanced_layer_wise_benchmark":
        generator = gen_moe_workload_layer_wise_benchmark
    else:
        raise ValueError(f"Invalid method: {method}.")

    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = generator(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        ep_size=ep_size,
        tile_size=tile_size,
    )

    workload = {
        "tile_idx_to_group_idx": tile_idx_to_group_idx,
        "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
        "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx,
        "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
        "total_num_padded_tokens": total_num_padded_tokens,
        "num_non_exiting_tiles": num_non_exiting_tiles,
    }
    safetensors.torch.save_file(workload, output_file)


if __name__ == "__main__":
    main()
