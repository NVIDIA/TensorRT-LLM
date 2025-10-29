import argparse

import pandas as pd
import torch
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
    args = parser.parse_args()

    meta_info, statistic = load_expert_statistic(args.expert_statistic_path)
    num_experts = meta_info["num_experts"]
    num_experts_per_token = meta_info["num_experts_per_token"]

    if args.iter_start is None:
        args.iter_start = meta_info["iter_start"]
    if args.iter_stop is None:
        args.iter_stop = meta_info["iter_stop"]
    num_iters = args.iter_stop - args.iter_start

    load_stats = {}
    for layer_idx in meta_info["layers"]:
        expert_token_count_iters = [
            data for key, data in statistic.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        expert_token_count_iters = torch.stack(expert_token_count_iters, dim=0)
        assert expert_token_count_iters.size(0) == num_iters

        if args.per_expert:
            load_iters = expert_token_count_iters
        else:
            load_iters = expert_token_count_iters.reshape(
                num_iters, meta_info["ep_size"], -1).sum(dim=-1)
        load_stats[layer_idx] = calculate_load_statistics(load_iters.float())

    load_stats = pd.DataFrame(load_stats)
    load_stats["average"] = load_stats.mean(axis=1)
    load_stats = load_stats.T
    print(f"Load statistics:\n{load_stats}")
