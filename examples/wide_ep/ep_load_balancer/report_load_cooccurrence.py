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

    meta_info, statistic, cooccurrence = load_expert_statistic(
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

    load_cooccurrence_stats = {}
    for layer_idx in meta_info["layers"]:
        expert_token_count_iters = [
            data for key, data in statistic.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        expert_token_count_iters = torch.stack(expert_token_count_iters, dim=0)
        assert expert_token_count_iters.size(0) == num_iters

        cooccurrence_iters = [
            data for key, data in cooccurrence.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        cooccurrence_iters = torch.stack(cooccurrence_iters, dim=0)
        assert cooccurrence_iters.size(0) == num_iters
        cooccurrence_matrix = cooccurrence_iters.sum(dim=0).float()

        expert_ids = torch.arange(num_experts)
        cooccurrence_matrix = cooccurrence_matrix.masked_fill(
            expert_ids.unsqueeze(1) == expert_ids, 0)

        if eplb_config is not None:
            num_slots = eplb_config["num_slots"]
            ep_size = eplb_config["ep_size"]
            layer_initial_global_assignments = eplb_config[
                "initial_global_assignments"][layer_idx]
            layer_initial_global_assignments = torch.tensor(
                layer_initial_global_assignments)
            expert_replica_count = layer_initial_global_assignments.bincount(
                minlength=num_experts)
            cooccurrence_matrix = cooccurrence_matrix / expert_replica_count / expert_replica_count.unsqueeze(
                -1)
            cooccurrence_matrix = cooccurrence_matrix[
                layer_initial_global_assignments][:,
                                                  layer_initial_global_assignments]
            num_slots_per_rank = num_slots // ep_size
            slot_ids = torch.arange(num_slots)
        else:
            ep_size = meta_info["ep_size"]
            num_slots_per_rank = num_experts // ep_size
            slot_ids = expert_ids

        rank_ids = slot_ids // num_slots_per_rank
        within_rank_cnt = cooccurrence_matrix[rank_ids.unsqueeze(1) ==
                                              rank_ids].sum().item()
        between_rank_cnt = cooccurrence_matrix[rank_ids.unsqueeze(1) !=
                                               rank_ids].sum().item()

        load_cooccurrence_stats[layer_idx] = {
            'within_rank_cnt': within_rank_cnt,
            'between_rank_cnt': between_rank_cnt
        }

    load_cooccurrence_stats = pd.DataFrame(load_cooccurrence_stats)
    load_cooccurrence_stats = load_cooccurrence_stats.T
    load_cooccurrence_stats['total_cnt'] = load_cooccurrence_stats[
        'within_rank_cnt'] + load_cooccurrence_stats['between_rank_cnt']
    load_cooccurrence_stats['within_rank_ratio'] = load_cooccurrence_stats[
        'within_rank_cnt'] / load_cooccurrence_stats['total_cnt']
    load_cooccurrence_stats = load_cooccurrence_stats.T
    load_cooccurrence_stats["average"] = load_cooccurrence_stats.mean(axis=1)
    load_cooccurrence_stats = load_cooccurrence_stats.T
    print(f"Load cooccurrence statistics:\n{load_cooccurrence_stats}")
