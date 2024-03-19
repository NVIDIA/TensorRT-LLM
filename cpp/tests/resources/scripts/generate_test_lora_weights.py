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


def copy_to_cache_pages(weights,
                        lora_config,
                        page_blocks,
                        configs,
                        tp_rank=0,
                        tp_size=1):
    page_slots = page_blocks.shape[1]
    page_width = page_blocks.shape[2]

    curr_page = 0
    curr_slot = 0
    for i in range(lora_config.shape[0]):
        module = configs[lora_config[i, 0]]
        adapter_size = module[2]
        in_dim = module[3]
        out_dim = module[4]
        mod_id = module[0]
        split_in = mod_id in (4, 6, 12)

        local_in_dim = in_dim // tp_size
        local_out_dim = out_dim // tp_size

        local_size = 0
        if split_in:
            local_size = adapter_size * (local_in_dim + out_dim)
        else:
            local_size = adapter_size * (in_dim + local_out_dim)

        num_slots = (local_size + page_width - 1) // page_width
        if num_slots + curr_slot > page_slots:
            curr_slot = 0
            curr_page += 1

        flattend_size = adapter_size * (in_dim + out_dim)

        if split_in:
            in_weights = weights[i, :adapter_size * in_dim].reshape(
                (adapter_size, tp_size,
                 local_in_dim))[:, tp_rank, :].contiguous().flatten()
            out_weights = weights[i, adapter_size *
                                  in_dim:flattend_size].contiguous().flatten()
        else:
            in_weights = weights[i, :adapter_size *
                                 in_dim].contiguous().flatten()
            out_weights = weights[i,
                                  adapter_size * in_dim:flattend_size].reshape(
                                      (tp_size, local_out_dim, adapter_size
                                       ))[tp_rank, :, :].contiguous().flatten()

        page_blocks[curr_page, curr_slot:curr_slot + num_slots, :].view(
            -1)[0:in_weights.shape[0] +
                out_weights.shape[0]] = torch.concatenate(
                    (in_weights, out_weights)).contiguous().flatten()
        curr_slot += num_slots


def main():
    torch.manual_seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--num-loras', type=int, default=1)

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
        (8, num_layers, adapter_size, hidden_size,
         3 * hidden_size),  # cross_attn_qkv
        (9, num_layers, adapter_size // 2, hidden_size,
         hidden_size),  # cross_attn_q
        (10, num_layers, adapter_size // 2, hidden_size,
         hidden_size),  # cross_attn_k
        (11, num_layers, adapter_size // 2, hidden_size,
         hidden_size),  # cross_attn_v
        (12, num_layers, adapter_size, hidden_size,
         hidden_size),  # cross_attn_dense
    ]

    for lora_idx in range(args.num_loras):
        all_source = []
        all_config = []

        all_target = []
        for c in configs:
            source_weights, config = generate_source_weights(*c)
            all_source.append(source_weights)
            all_config.append(config)

            mod_id, _, adapter_size, in_dim, out_dim = c
            split_in = mod_id in (4, 6, 12)

            target_weights = format_tensors(source_weights, adapter_size,
                                            in_dim, out_dim, args.tp_size,
                                            split_in)
            all_target.append(target_weights)

        all_source = pad_tensors(all_source)
        all_config = pad_tensors(all_config)
        all_target = pad_tensors(all_target)

        output_dir = Path(args.out_dir)
        if args.num_loras > 1:
            output_dir = output_dir / str(lora_idx)

        os.makedirs(output_dir, exist_ok=True)
        # copy weights into cache pages
        for rank in range(args.tp_size):
            page_block = torch.zeros((8, 18, 128),
                                     dtype=torch.float32,
                                     device='cpu')
            copy_to_cache_pages(all_source,
                                all_config,
                                page_block,
                                configs,
                                tp_rank=rank,
                                tp_size=args.tp_size)

            out_path = output_dir / f'cache_pages_rank{rank}.npy'
            np.save(out_path, page_block)

        source_out_path = output_dir / 'source.npy'
        config_out_path = output_dir / 'config.npy'
        target_out_path = output_dir / 'target.npy'

        np.save(source_out_path, all_source)
        np.save(config_out_path, all_config)
        np.save(target_out_path, all_target)


if __name__ == "__main__":
    main()
