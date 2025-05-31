import argparse
import glob
import json
import os

import pandas as pd
import safetensors
import torch
import yaml

from tensorrt_llm.bindings.internal.runtime import (MoeLoadBalanceMetaInfo,
                                                    MoePlacementCpuInfo,
                                                    do_placement,
                                                    do_replication)
from tensorrt_llm.logger import logger

logger.set_level("info")


def calculate_load_statistics(load_iters: torch.Tensor):
    # sum the loads over iterations and calculate the statistics
    load_total = load_iters.sum(dim=0)
    mean = load_total.mean().item()
    std = load_total.std().item()
    imbalance_ratio = (load_total.max().item() - load_total.min().item()) / mean
    stats = {
        "mean-total": mean,
        "std-total": std,
        "imbalance-ratio-total": imbalance_ratio
    }

    # calculate the statistics for each iteration and average over iterations
    mean = load_iters.mean(dim=-1).mean().item()
    std = load_iters.std(dim=-1).mean().item()
    imbalance_ratio = (load_iters.max(dim=-1).values -
                       load_iters.min(dim=-1).values) / load_iters.mean(dim=-1)
    imbalance_ratio = imbalance_ratio.mean().item()
    stats.update({
        "mean-iter": mean,
        "std-iter": std,
        "imbalance-ratio-iter": imbalance_ratio
    })
    return stats


def save_eplb_config(config: dict, path: str):

    def represent_list_inline(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq',
                                         data,
                                         flow_style=True)

    yaml.add_representer(list, represent_list_inline)

    with open(path, "w") as f:
        yaml.dump(config, f, width=float('inf'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_statistic_path",
                        type=str,
                        default=os.environ.get("EXPERT_STATISTIC_PATH",
                                               "expert_statistic"))
    parser.add_argument("--iter_start", type=int, default=None)
    parser.add_argument("--iter_stop", type=int, default=None)
    parser.add_argument("--output_path",
                        type=str,
                        default="moe_load_balancer.yaml")
    parser.add_argument("--ep_size", type=int, default=8)
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--layer_updates_per_iter", type=int, default=0)
    args = parser.parse_args()

    with open(f"{args.expert_statistic_path}/meta_info.json", "r") as f:
        meta_info = json.load(f)
    num_experts = meta_info["num_experts"]
    num_experts_per_token = meta_info["num_experts_per_token"]

    statistic = {}
    for statistic_file in glob.glob(
            f"{args.expert_statistic_path}/rank*.safetensors"):
        rank_statistic = safetensors.torch.load_file(statistic_file)
        for key, data in rank_statistic.items():
            if key not in statistic:
                statistic[key] = torch.zeros_like(data)
            statistic[key] += data

    def parse_key(key: str) -> tuple[int, int]:
        iter_idx, layer_idx = key.split("_")
        return int(iter_idx), int(layer_idx)

    statistic = {parse_key(key): data for key, data in statistic.items()}

    iters = sorted(list(set(iter_idx for iter_idx, _ in statistic)))
    layers = sorted(list(set(layer_idx for _, layer_idx in statistic)))
    num_iters = len(iters)
    num_layers = len(layers)
    assert iters[-1] + 1 - iters[0] == num_iters
    assert len(statistic) == num_iters * num_layers

    if args.iter_start is None:
        args.iter_start = iters[0]
    if args.iter_stop is None:
        args.iter_stop = iters[-1] + 1
    logger.info(f"Statistic iterations: {iters[0]} - {iters[-1] + 1}")
    logger.info(f"Used iterations: {args.iter_start} - {args.iter_stop}")
    logger.info(f"Statistic layers: {layers}")

    num_local_slots = args.num_slots // args.ep_size
    initial_global_assignments = {}
    load_stats = {}
    load_stats_rebalanced = {}

    for layer_idx in layers:
        expert_token_count_iters = [
            data for key, data in statistic.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        expert_token_count_iters = torch.stack(expert_token_count_iters, dim=0)
        assert expert_token_count_iters.size(
            0) == args.iter_stop - args.iter_start
        expert_load_factor = expert_token_count_iters.sum(dim=0).float()

        meta_info = MoeLoadBalanceMetaInfo(expert_count=num_experts,
                                           top_k=num_experts_per_token,
                                           ep_rank=0,
                                           ep_size=args.ep_size,
                                           slot_count_per_rank=num_local_slots)
        placement_info = MoePlacementCpuInfo()
        placement_info.expert_replica_count = [0] * num_experts
        placement_info.rank_expert_ids = [[0] * num_local_slots
                                          for _ in range(args.ep_size)]

        do_replication(meta_info, expert_load_factor.tolist(), placement_info)
        do_placement(meta_info, expert_load_factor.tolist(), placement_info)

        initial_global_assignments[layer_idx] = []
        for local_expert_ids in placement_info.rank_expert_ids:
            initial_global_assignments[layer_idx].extend(local_expert_ids)

        # Report load statistics
        rank_load_iters = expert_token_count_iters.reshape(
            expert_token_count_iters.size(0), args.ep_size, -1).sum(dim=-1)
        load_stats[layer_idx] = calculate_load_statistics(
            rank_load_iters.float())

        token_load_iters_rebalanced = expert_token_count_iters / torch.tensor(
            placement_info.expert_replica_count)
        token_load_iters_rebalanced = token_load_iters_rebalanced[:,
                                                                  torch.
                                                                  tensor(initial_global_assignments[
                                                                      layer_idx]
                                                                         )]
        rank_load_iters_rebalanced = token_load_iters_rebalanced.reshape(
            expert_token_count_iters.size(0), args.ep_size, -1).sum(dim=-1)
        load_stats_rebalanced[layer_idx] = calculate_load_statistics(
            rank_load_iters_rebalanced.float())

    eplb_config = {
        "num_slots": args.num_slots,
        "initial_global_assignments": initial_global_assignments,
        "layer_updates_per_iter": args.layer_updates_per_iter,
    }
    save_eplb_config(eplb_config, args.output_path)

    load_stats = pd.DataFrame(load_stats).T
    logger.info(f"Load statistics:\n{load_stats}")

    load_stats_rebalanced = pd.DataFrame(load_stats_rebalanced).T
    logger.info(
        f"Load statistics after rebalance (estimated):\n{load_stats_rebalanced}"
    )
