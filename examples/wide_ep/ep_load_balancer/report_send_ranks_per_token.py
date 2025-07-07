import argparse

import pandas as pd
import torch
import yaml
from utils import load_expert_statistic


def calculate_load_statistics(load_iters: torch.Tensor):
    # calculate the statistics for each iteration and average over iterations
    mean = load_iters.mean(dim=-1).mean().item()
    std = load_iters.std(dim=-1).mean().item()
    imbalance_ratio = load_iters.max(dim=-1).values / load_iters.mean(
        dim=-1) - 1
    imbalance_ratio = imbalance_ratio.mean().item()
    return {"mean": mean, "std": std, "imbalance-ratio": imbalance_ratio}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert_statistic_path",
        type=str,
        required=True,
        help="The directory path to the expert statistic files.")
    parser.add_argument("--iter_start",
                        type=int,
                        default=None,
                        help="The start iteration of used iterations.")
    parser.add_argument("--iter_stop",
                        type=int,
                        default=None,
                        help="The end iteration of used iterations.")
    parser.add_argument("--per_expert",
                        default=False,
                        action="store_true",
                        help="Report the load statistics per expert.")
    parser.add_argument("--projection_eplb_config_path", type=str, default=None)
    args = parser.parse_args()

    meta_info, statistic, cooccurrence, token_selected_experts_stats = load_expert_statistic(
        args.expert_statistic_path)
    num_experts = meta_info["num_experts"]
    num_experts_per_token = meta_info["num_experts_per_token"]
    if args.iter_start is None:
        args.iter_start = meta_info["iter_start"]
    if args.iter_stop is None:
        args.iter_stop = meta_info["iter_stop"]
    num_iters = args.iter_stop - args.iter_start

    eplb_config = None
    if args.projection_eplb_config_path is not None:
        with open(args.projection_eplb_config_path, "r") as f:
            eplb_config = yaml.safe_load(f)

    stats = {}
    for layer_idx in meta_info["layers"]:
        expert_token_count_iters = [
            data for key, data in statistic.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        expert_token_count_iters = torch.stack(expert_token_count_iters, dim=0)
        assert expert_token_count_iters.size(0) == num_iters

        token_selected_experts_iters = [
            data for key, data in token_selected_experts_stats.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        token_selected_experts = torch.cat(token_selected_experts_iters,
                                           dim=0).cuda()
        token_selected_experts = torch.zeros(
            (token_selected_experts.size(0), num_experts),
            dtype=torch.int32,
            device=token_selected_experts.device).scatter(
                -1, token_selected_experts.to(torch.int64), 1)

        if eplb_config is not None:
            num_slots = eplb_config["num_slots"]
            ep_size = eplb_config["ep_size"]
            layer_initial_global_assignments = eplb_config[
                "initial_global_assignments"][layer_idx]
            layer_initial_global_assignments = torch.tensor(
                layer_initial_global_assignments,
                device=token_selected_experts.device)
            expert_replica_count = layer_initial_global_assignments.bincount(
                minlength=num_experts)
            token_selected_experts = token_selected_experts / expert_replica_count
            token_selected_experts = token_selected_experts[:,
                                                            layer_initial_global_assignments]
            num_slots_per_rank = num_slots // ep_size
        else:
            ep_size = meta_info["ep_size"]
            num_slots_per_rank = num_experts // ep_size

        token_selected_experts = token_selected_experts.view(
            -1, ep_size, num_slots_per_rank)
        num_send_ranks_per_token = (
            1 - (1 - token_selected_experts).prod(dim=-1)).sum(
                dim=-1).float().mean().item()
        stats[layer_idx] = {
            'num_send_ranks_per_token': num_send_ranks_per_token
        }

    stats = pd.DataFrame(stats)
    stats["average"] = stats.mean(axis=1)
    stats = stats.T
    print(f"Load statistics:\n{stats}")
