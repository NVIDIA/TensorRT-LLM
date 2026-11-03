# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the runtime YAMLs of CONFIDENCE_REPRODUCTION.md for 8 GPUs (attention DP + 8-way EP).

For every global batch size BS it writes bs<BS>-fixed-k<K>.yaml (DSpark with a fixed draft length)
and bs<BS>-confidence-k<K>.yaml (confidence scheduling with maximum draft length K). The rank-local
batch size is BS/8, and the decode CUDA graphs are captured at 16, 32, 64, 128 and 256 below BS/8
plus BS/8 itself.
"""

import argparse
from pathlib import Path

import yaml

GPUS = 8


def base_config(batch_size: int) -> dict:
    local = batch_size // GPUS
    return {
        "gpus_per_node": GPUS,
        "tensor_parallel_size": GPUS,
        "moe_expert_parallel_size": GPUS,
        "moe_tensor_parallel_size": 1,
        "enable_attention_dp": True,
        "attention_dp_config": {
            "enable_balance": True,
            "batching_wait_iters": 10,
            "timeout_iters": 50,
        },
        "max_batch_size": local,
        "max_num_tokens": 4096,
        "max_input_len": 1536,
        "max_seq_len": 2560,
        "enable_chunked_prefill": False,
        "custom_tokenizer": "deepseek_v4",
        "kv_cache_config": {
            "dtype": "fp8",
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False,
            "tokens_per_block": 128,
            "host_cache_size": 0,
        },
        "moe_config": {"backend": "MEGAMOE_DEEPGEMM"},
        "cuda_graph_config": {
            "enable_padding": True,
            "max_batch_size": local,
            "batch_sizes": [g for g in (16, 32, 64, 128, 256) if g < local] + [local],
        },
        "prefill_cuda_graph_backend": "breakable",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cost-table",
        type=Path,
        required=True,
        help="cost table covering every captured graph batch size",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 384, 512, 768, 1024, 2048]
    )
    parser.add_argument("--fixed-k", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--confidence-k", type=int, nargs="+", default=[3, 4, 5])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for batch_size in args.batch_sizes:
        for k in args.fixed_k:
            config = base_config(batch_size)
            config["speculative_config"] = {
                "decoding_type": "DSpark",
                "max_draft_len": k,
                "block_size": k,
            }
            path = args.output_dir / f"bs{batch_size}-fixed-k{k}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
        for k in args.confidence_k:
            config = base_config(batch_size)
            config["speculative_config"] = {
                "decoding_type": "DSpark",
                "max_draft_len": k,
                "block_size": k,
                "enable_confidence_scheduling": True,
                "confidence_cost_table_path": str(args.cost_table.resolve()),
                "confidence_calibration_path": str(args.calibration.resolve()),
            }
            path = args.output_dir / f"bs{batch_size}-confidence-k{k}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(
        f"wrote {len(args.batch_sizes) * (len(args.fixed_k) + len(args.confidence_k))} configs "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
