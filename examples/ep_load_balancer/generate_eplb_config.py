import argparse
import glob
import json
import os

import safetensors
import torch
import yaml

from tensorrt_llm.bindings.internal.runtime import (MoeLoadBalanceMetaInfo,
                                                    MoePlacementCpuInfo,
                                                    do_placement,
                                                    do_replication)
from tensorrt_llm.logger import logger

logger.set_level("info")

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
    assert len(statistic) == num_iters * num_layers

    if args.iter_start is None:
        args.iter_start = iters[0]
    if args.iter_stop is None:
        args.iter_stop = iters[-1] + 1
    logger.info(f"Statistic iterations: {iters}")
    logger.info(f"Used iteration range: {args.iter_start} - {args.iter_stop}")
    logger.info(f"Statistic layers: {layers}")

    num_local_slots = args.num_slots // args.ep_size
    initial_global_assignments = {}

    for layer_idx in layers:
        expert_token_count = [
            data for key, data in statistic.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        assert len(expert_token_count) == args.iter_stop - args.iter_start
        expert_token_count = sum(expert_token_count)
        expert_load_factor = expert_token_count.float().tolist()

        meta_info = MoeLoadBalanceMetaInfo(expert_count=num_experts,
                                           top_k=num_experts_per_token,
                                           ep_rank=0,
                                           ep_size=args.ep_size,
                                           slot_count_per_rank=num_local_slots)
        placement_info = MoePlacementCpuInfo()
        placement_info.expert_replica_count = [0] * num_experts
        placement_info.rank_expert_ids = [[0] * num_local_slots
                                          for _ in range(args.ep_size)]

        do_replication(meta_info, expert_load_factor, placement_info)
        do_placement(meta_info, expert_load_factor, placement_info)

        initial_global_assignments[layer_idx] = []
        for local_expert_ids in placement_info.rank_expert_ids:
            initial_global_assignments[layer_idx].extend(local_expert_ids)

    eplb_config = {
        "num_slots": args.num_slots,
        "initial_global_assignments": initial_global_assignments,
        "layer_updates_per_iter": args.layer_updates_per_iter,
    }

    def represent_list_inline(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq',
                                         data,
                                         flow_style=True)

    yaml.add_representer(list, represent_list_inline)

    with open("moe_load_balancer.yaml", "w") as f:
        yaml.dump(eplb_config, f, width=float('inf'))
