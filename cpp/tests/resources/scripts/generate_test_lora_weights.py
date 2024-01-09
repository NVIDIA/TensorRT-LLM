import argparse
import os
from pathlib import Path

import numpy as np
import torch


def generate_source_weights(mod_id,
                            num_layers,
                            adapter_dim,
                            in_dim,
                            out_dim,
                            dtype=torch.float32):
    weights = torch.rand((num_layers, adapter_dim * (in_dim + out_dim)),
                         dtype=dtype)
    config = []
    for layer_idx in range(num_layers):
        config.append((mod_id, layer_idx, adapter_dim))
    config = torch.tensor(config, dtype=torch.int32)
    return weights, config


def format_tensors(weights,
                   adapter_dim,
                   in_dim,
                   out_dim,
                   tp_size=1,
                   split_in=False):
    target_weights = torch.zeros_like(weights)
    if tp_size == 1:
        return weights
    num_layers = weights.shape[0]
    for layer_idx in range(num_layers):
        in_size = adapter_dim * in_dim
        if split_in:
            target_weights[layer_idx, 0:in_size] = torch.concatenate(
                torch.split(torch.reshape(weights[layer_idx, 0:in_size],
                                          (adapter_dim, in_dim)),
                            in_dim // tp_size,
                            dim=1),
                dim=0).contiguous().flatten()
            target_weights[layer_idx, in_size:] = weights[layer_idx, in_size:]
        else:
            target_weights[layer_idx] = weights[layer_idx]
    return target_weights


def pad_tensors(weights_list):
    max_size = 0
    for w in weights_list:
        max_size = max(w.shape[1], max_size)

    padded_weights = []
    for w in weights_list:
        padded_weights.append(
            torch.nn.functional.pad(w, (0, max_size - w.shape[1])))
    return torch.concatenate(padded_weights, dim=0)


def main():
    torch.manual_seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--out-dir', type=Path, required=True)

    args = parser.parse_args()

    num_layers = 2
    adapter_size = 8
    hidden_size = 16
    mlp_hidden_size = 32
    configs = [
        (0, num_layers, adapter_size, hidden_size, 3 * hidden_size),  # attn_qkv
        (1, num_layers, adapter_size // 2, hidden_size, hidden_size),  # attn_q
        (2, num_layers, adapter_size // 2, hidden_size, hidden_size),  # attn_k
        (3, num_layers, adapter_size // 2, hidden_size, hidden_size),  # attn_v
        (4, num_layers, adapter_size, hidden_size, hidden_size),  # attn_dense
        (5, num_layers, adapter_size, hidden_size,
         mlp_hidden_size),  # mlp_h_to_4h
        (6, num_layers, adapter_size, mlp_hidden_size,
         hidden_size),  # mlp_4h_to_h
        (7, num_layers, adapter_size, hidden_size, mlp_hidden_size),  # mlp_gate
    ]

    all_source = []
    all_config = []

    all_target = []
    for c in configs:
        source_weights, config = generate_source_weights(*c)
        all_source.append(source_weights)
        all_config.append(config)

        mod_id, _, adapter_size, in_dim, out_dim = c
        split_in = mod_id in (4, 6)

        target_weights = format_tensors(source_weights, adapter_size, in_dim,
                                        out_dim, args.tp_size, split_in)
        all_target.append(target_weights)

    all_source = pad_tensors(all_source)
    all_config = pad_tensors(all_config)
    all_target = pad_tensors(all_target)

    source_out_path = args.out_dir / 'source.npy'
    config_out_path = args.out_dir / 'config.npy'
    target_out_path = args.out_dir / 'target.npy'

    os.makedirs(args.out_dir, exist_ok=True)

    np.save(source_out_path, all_source)
    np.save(config_out_path, all_config)
    np.save(target_out_path, all_target)


if __name__ == "__main__":
    main()
