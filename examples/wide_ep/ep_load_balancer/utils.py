import glob
import json

import safetensors.torch
import torch


def load_expert_statistic(path: str):
    """Load expert statistics (bincount results) from saved files.

    Loads from .safetensors files which contain pre-computed bincount results.
    This is the same format as the original implementation.

    Returns:
        meta_info: Dictionary containing metadata (num_experts, num_experts_per_token, etc.)
        statistic: Dictionary mapping (iter_id, layer_id) to expert token counts
    """
    with open(f"{path}/meta_info.json", "r") as f:
        meta_info = json.load(f)

    statistic_files = glob.glob(f"{path}/rank*.safetensors")
    # Exclude *_raw.safetensors files
    statistic_files = [f for f in statistic_files if not f.endswith('_raw.safetensors')]
    statistic = {}

    for statistic_file in statistic_files:
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
    meta_info["ep_size"] = len(statistic_files)
    meta_info["iter_start"] = iters[0]
    meta_info["iter_stop"] = iters[-1] + 1
    meta_info["layers"] = layers
    return meta_info, statistic


def load_raw_expert_statistic(path: str):
    """Load raw token_selected_experts tensors from *_raw.safetensors files.

    Returns:
        meta_info: Dictionary containing metadata
        raw_statistic: Dictionary mapping (iter_id, layer_id, rank_id)
                       to raw token_selected_experts tensors
    """
    import re

    with open(f"{path}/meta_info.json", "r") as f:
        meta_info = json.load(f)

    statistic_files = glob.glob(f"{path}/rank*_raw.safetensors")
    raw_statistic = {}

    for statistic_file in statistic_files:
        match = re.search(r'rank(\d+)_raw\.safetensors', statistic_file)
        rank_id = int(match.group(1)) if match else 0

        rank_statistic = safetensors.torch.load_file(statistic_file)
        for key, data in rank_statistic.items():
            iter_idx, layer_idx = key.split("_")
            full_key = (int(iter_idx), int(layer_idx), rank_id)
            raw_statistic[full_key] = data

    meta_info["ep_size"] = len(statistic_files)
    return meta_info, raw_statistic
