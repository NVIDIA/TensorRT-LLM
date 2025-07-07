import glob
import json
from queue import PriorityQueue

import safetensors.torch
import torch
import yaml


def load_expert_statistic(path: str):
    with open(f"{path}/meta_info.json", "r") as f:
        meta_info = json.load(f)

    statistic_files = glob.glob(f"{path}/rank*.safetensors")
    statistic = {}
    for statistic_file in statistic_files:
        rank_statistic = safetensors.torch.load_file(statistic_file)
        for key, data in rank_statistic.items():
            if key not in statistic:
                statistic[key] = torch.zeros_like(data)
            statistic[key] += data

    cooccurrence_files = glob.glob(f"{path}/cooccurrence_rank*.safetensors")
    cooccurrence = {}
    for cooccurrence_file in cooccurrence_files:
        rank_cooccurrence = safetensors.torch.load_file(cooccurrence_file)
        for key, data in rank_cooccurrence.items():
            if key not in cooccurrence:
                cooccurrence[key] = torch.zeros_like(data)
            cooccurrence[key] += data

    token_selected_experts_files = glob.glob(
        f"{path}/token_selected_experts_rank*.safetensors")
    token_selected_experts = {}
    for token_selected_experts_file in token_selected_experts_files:
        rank_token_selected_experts = safetensors.torch.load_file(
            token_selected_experts_file)
        for key, data in rank_token_selected_experts.items():
            if key not in token_selected_experts:
                token_selected_experts[key] = data
            else:
                token_selected_experts[key] = torch.cat(
                    [token_selected_experts[key], data], dim=0)

    def parse_key(key: str) -> tuple[int, int]:
        iter_idx, layer_idx = key.split("_")
        return int(iter_idx), int(layer_idx)

    statistic = {parse_key(key): data for key, data in statistic.items()}
    cooccurrence = {parse_key(key): data for key, data in cooccurrence.items()}
    token_selected_experts = {
        parse_key(key): data
        for key, data in token_selected_experts.items()
    }

    iters = sorted(list(set(iter_idx for iter_idx, _ in statistic)))
    layers = sorted(list(set(layer_idx for _, layer_idx in statistic)))
    num_iters = len(iters)
    num_layers = len(layers)
    assert iters[-1] + 1 - iters[0] == num_iters
    assert len(statistic) == num_iters * num_layers
    meta_info["ep_size"] = len(statistic_files)
    meta_info["iter_start"] = iters[0]
    meta_info["iter_stop"] = iters[-1] + 1
    meta_info["layers"] = layers
    return meta_info, statistic, cooccurrence, token_selected_experts


def select_expert(sorted_expert_ids, expert_replica_assigned,
                  expert_replica_count):
    for expert_id in sorted_expert_ids.tolist():
        if expert_replica_assigned[expert_id] < expert_replica_count[expert_id]:
            return expert_id
    raise ValueError("No available expert replica.")


def do_placement_with_cooccurrence(expert_load_factor: torch.Tensor,
                                   expert_replica_count: torch.Tensor,
                                   cooccurrence_matrix: torch.Tensor,
                                   ep_size=36,
                                   alpha=0.8):
    num_experts = expert_replica_count.size(0)
    num_slots = expert_replica_count.sum().item()
    num_slots_per_rank = num_slots // ep_size

    slot_load_factor = expert_load_factor / expert_replica_count
    sorted_expert_ids = slot_load_factor.sort(descending=True).indices

    average_rank_load_factor = expert_load_factor.sum().item() / ep_size
    rank_expert_ids = [[] for _ in range(ep_size)]
    rank_load_factor = [0.0 for _ in range(ep_size)]
    expert_replica_assigned = [0 for _ in range(num_experts)]

    # Step 1
    for rank in range(ep_size):
        current_sorted_expert_ids = sorted_expert_ids
        while True:
            expert_id = select_expert(current_sorted_expert_ids,
                                      expert_replica_assigned,
                                      expert_replica_count)
            if len(rank_expert_ids[rank]) + 1 > num_slots_per_rank * alpha:
                break
            if rank_load_factor[rank] + slot_load_factor[expert_id].item(
            ) > average_rank_load_factor * alpha:
                break

            rank_expert_ids[rank].append(expert_id)
            rank_load_factor[rank] += slot_load_factor[expert_id].item()
            expert_replica_assigned[expert_id] += 1
            current_sorted_expert_ids = cooccurrence_matrix[
                rank_expert_ids[rank]].sum(dim=0).sort(descending=True).indices

    # Step 2
    pq = PriorityQueue()
    for rank in range(ep_size):
        if len(rank_expert_ids[rank]) < num_slots_per_rank:
            pq.put((rank_load_factor[rank], rank))

    for expert_id in sorted_expert_ids.tolist():
        while expert_replica_assigned[expert_id] < expert_replica_count[
                expert_id]:
            _, rank = pq.get()
            rank_expert_ids[rank].append(expert_id)
            rank_load_factor[rank] += slot_load_factor[expert_id].item()
            expert_replica_assigned[expert_id] += 1
            if len(rank_expert_ids[rank]) < num_slots_per_rank:
                pq.put((rank_load_factor[rank], rank))

    return rank_expert_ids


def save_eplb_config(config: dict, path: str):

    def represent_list_inline(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq',
                                         data,
                                         flow_style=True)

    yaml.add_representer(list, represent_list_inline)

    with open(path, "w") as f:
        yaml.dump(config, f, width=float('inf'))
