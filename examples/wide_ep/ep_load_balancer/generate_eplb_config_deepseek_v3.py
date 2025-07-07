import argparse

import torch
from utils import (do_placement_with_cooccurrence, load_expert_statistic,
                   save_eplb_config)

from tensorrt_llm.bindings.internal.runtime import (MoeLoadBalanceMetaInfo,
                                                    MoePlacementCpuInfo,
                                                    do_replication)


def create_cooccurrence_matrix(expert_load_factor: torch.Tensor,
                               expert_replica_count: torch.Tensor,
                               num_groups: int = 32):
    num_experts = expert_replica_count.size(0)
    num_experts_per_group = num_experts // num_groups
    slot_load_factor = expert_load_factor / expert_replica_count
    cooccurrence_matrix = slot_load_factor.unsqueeze(
        1) * slot_load_factor / slot_load_factor.sum()
    expert_ids = torch.arange(num_experts)
    expert_group_ids = expert_ids // num_experts_per_group
    cooccurrence_matrix = cooccurrence_matrix.masked_fill(
        expert_group_ids.unsqueeze(1) != expert_group_ids, 0)
    cooccurrence_matrix = cooccurrence_matrix.masked_fill(
        expert_ids.unsqueeze(1) == expert_ids, -1e9)
    assert (cooccurrence_matrix
            > 0).sum() == num_experts_per_group**2 * num_groups - num_experts
    return cooccurrence_matrix


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
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="The output path to the eplb config file.")
    parser.add_argument(
        "--ep_size",
        type=int,
        default=None,
        help="The expert parallelism size after load rebalance.")
    parser.add_argument(
        "--num_slots",
        type=int,
        default=None,
        help="The total number of expert slots after load rebalance.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="The alpha value for the load balancing algorithm.")
    parser.add_argument("--layer_updates_per_iter",
                        type=int,
                        default=0,
                        help="The number of layers to update per iteration.")
    args = parser.parse_args()

    meta_info, statistic, _, _ = load_expert_statistic(
        args.expert_statistic_path)
    num_experts = meta_info["num_experts"]
    num_experts_per_token = meta_info["num_experts_per_token"]

    if args.ep_size is None:
        args.ep_size = meta_info["ep_size"]
    if args.num_slots is None:
        args.num_slots = num_experts
    if args.iter_start is None:
        args.iter_start = meta_info["iter_start"]
    if args.iter_stop is None:
        args.iter_stop = meta_info["iter_stop"]
    num_iters = args.iter_stop - args.iter_start

    num_local_slots = args.num_slots // args.ep_size
    initial_global_assignments = {}
    for layer_idx in meta_info["layers"]:
        expert_token_count_iters = [
            data for key, data in statistic.items() if
            args.iter_start <= key[0] < args.iter_stop and key[1] == layer_idx
        ]
        expert_token_count_iters = torch.stack(expert_token_count_iters, dim=0)
        assert expert_token_count_iters.size(0) == num_iters
        expert_load_factor = expert_token_count_iters.sum(dim=0).float()

        moelb_info = MoeLoadBalanceMetaInfo(expert_count=num_experts,
                                            top_k=num_experts_per_token,
                                            ep_rank=0,
                                            ep_size=args.ep_size,
                                            slot_count_per_rank=num_local_slots)
        placement_info = MoePlacementCpuInfo()
        placement_info.expert_replica_count = [0] * num_experts
        placement_info.rank_expert_ids = [[0] * num_local_slots
                                          for _ in range(args.ep_size)]

        do_replication(moelb_info, expert_load_factor.tolist(), placement_info)

        num_groups = 32
        expert_replica_count = torch.tensor(placement_info.expert_replica_count)
        cooccurrence_matrix = create_cooccurrence_matrix(expert_load_factor,
                                                         expert_replica_count,
                                                         num_groups=num_groups)
        rank_expert_ids = do_placement_with_cooccurrence(expert_load_factor,
                                                         expert_replica_count,
                                                         cooccurrence_matrix,
                                                         ep_size=args.ep_size,
                                                         alpha=args.alpha)

        initial_global_assignments[layer_idx] = []
        for local_expert_ids in rank_expert_ids:
            initial_global_assignments[layer_idx].extend(local_expert_ids)

    eplb_config = {
        "num_slots": args.num_slots,
        "ep_size": args.ep_size,
        "initial_global_assignments": initial_global_assignments,
        "layer_updates_per_iter": args.layer_updates_per_iter,
    }
    save_eplb_config(eplb_config, args.output_path)
