import glob
import json

import safetensors.torch
import torch


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
