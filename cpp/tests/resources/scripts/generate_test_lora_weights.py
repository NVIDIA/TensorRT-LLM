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
                            dtype=torch.float32,
                            is_dora: bool | None = None):
    weights = torch.rand((num_layers, adapter_dim * (in_dim + out_dim) +
                          (out_dim if is_dora else 0)),
                         dtype=dtype)
    config = []
    for layer_idx in range(num_layers):
        if is_dora is not None:
            config.append((mod_id, layer_idx, adapter_dim, is_dora))
        else:
            # test old config format for backwards compatibility
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
                        tp_size=1,
                        is_dora: bool | None = None):
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
            local_size = adapter_size * (local_in_dim + out_dim) + (
                out_dim if is_dora else 0)
        else:
            local_size = adapter_size * (in_dim + local_out_dim) + (
                local_out_dim if is_dora else 0)

        num_slots = (local_size + page_width - 1) // page_width
        if num_slots + curr_slot > page_slots:
            curr_slot = 0
            curr_page += 1

        flattend_size = adapter_size * (in_dim + out_dim) + (out_dim
                                                             if is_dora else 0)

        dora_mag = None
        if split_in:
            in_weights = weights[i, :adapter_size * in_dim].reshape(
                (adapter_size, tp_size,
                 local_in_dim))[:, tp_rank, :].contiguous().flatten()
            out_weights = weights[i, adapter_size *
                                  in_dim:flattend_size].contiguous().flatten()
            if is_dora:
                dora_mag = out_weights[-out_dim:]
                out_weights = out_weights[:-out_dim]

        else:
            in_weights = weights[i, :adapter_size *
                                 in_dim].contiguous().flatten()
            out_weights = weights[i, adapter_size * in_dim:flattend_size]

            if is_dora:
                dora_mag = out_weights[-out_dim:]
                out_weights = out_weights[:-out_dim]
                dora_mag = dora_mag.reshape(
                    (tp_size, local_out_dim))[tp_rank].contiguous().flatten()

            out_weights = out_weights.reshape(
                (tp_size, local_out_dim,
                 adapter_size))[tp_rank, :, :].contiguous().flatten()

        page_blocks[curr_page, curr_slot:curr_slot + num_slots, :].view(
            -1)[0:in_weights.shape[0] + out_weights.shape[0] +
                (dora_mag.shape[0] if is_dora else 0)] = torch.concatenate(
                    (in_weights, out_weights) +
                    ((dora_mag, ) if is_dora else ())).contiguous().flatten()

        curr_slot += num_slots


def main():
    torch.manual_seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--num-loras', type=int, default=1)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--adapter-size', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--mlp-hidden-size', type=int, default=32)
    parser.add_argument('--no-generate-cache-pages',
                        action='store_true',
                        default=False)
    parser.add_argument(
        '--config-ids-filter',
        type=str,
        default=None,
        help=
        "Comma separated list of ids to include. For example, use --config-ids-filter=0 for attn_qkv only."
    )
    parser.add_argument('--target-file-name', type=str, default="target.npy")
    parser.add_argument('--config-file-name', type=str, default="config.npy")

    args = parser.parse_args()

    num_layers = args.num_layers
    adapter_size = args.adapter_size
    hidden_size = args.hidden_size
    mlp_hidden_size = args.mlp_hidden_size
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
    if args.config_ids_filter:
        config_ids_filter = [int(x) for x in args.config_ids_filter.split(",")]
        configs = [c for c in configs if c[0] in config_ids_filter]

    for lora_idx in range(args.num_loras):
        for is_dora in [None, False, True]:
            all_source = []
            all_config = []

            all_target = []
            for c in configs:
                source_weights, config = generate_source_weights(
                    *c, is_dora=is_dora)
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
            suffix = "" if is_dora is None else (
                "_dora" if is_dora else "_no_dora")
            # copy weights into cache pages
            if not args.no_generate_cache_pages:
                for rank in range(args.tp_size):
                    page_block = torch.zeros((8, 18, 128),
                                             dtype=torch.float32,
                                             device='cpu')
                    copy_to_cache_pages(all_source,
                                        all_config,
                                        page_block,
                                        configs,
                                        tp_rank=rank,
                                        tp_size=args.tp_size,
                                        is_dora=is_dora)

                    out_path = output_dir / f'cache_pages_rank{rank}{suffix}.npy'
                    np.save(out_path, page_block)

            config_file_name = Path(args.config_file_name).stem
            target_file_name = Path(args.target_file_name).stem

            source_out_path = output_dir / f'source{suffix}.npy'
            config_out_path = output_dir / f'{config_file_name}{suffix}.npy'
            target_out_path = output_dir / f'{target_file_name}{suffix}.npy'

            np.save(source_out_path, all_source)
            np.save(config_out_path, all_config)
            np.save(target_out_path, all_target)


if __name__ == "__main__":
    main()
