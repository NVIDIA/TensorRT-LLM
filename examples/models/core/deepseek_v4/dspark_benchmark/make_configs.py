# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the runtime YAMLs of CONFIDENCE_REPRODUCTION.md for 8 GPUs with 8-way EP.

For every global batch size BS it writes bs<BS>-fixed-k<K>.yaml (DSpark with a fixed draft length)
and bs<BS>-confidence-k<K>.yaml (confidence scheduling with maximum draft length K and device
re-ranking, the default). With --host-ranking it also writes bs<BS>-confidence-host-k<K>.yaml, the
same with re-ranking off. Confidence configs are skipped where the cost table has no row for a
captured graph batch size.

--layout dep8 (the default) uses attention DP: the rank-local batch size is BS/8, and the decode
CUDA graphs are captured at 16, 32, 64, 128 and 256 below BS/8 plus BS/8 itself. --layout tep8 uses
tensor-parallel attention: every rank runs all BS requests, the graphs are captured at powers of two
below BS plus BS itself, and prefill requests are batched (section 9 of the guide).
"""

import argparse
import json
from pathlib import Path

import yaml

GPUS = 8


def base_config(batch_size: int, layout: str) -> dict:
    local = batch_size if layout == "tep8" else batch_size // GPUS
    config = {
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
    if layout == "tep8":
        config["enable_attention_dp"] = False
        del config["attention_dp_config"]
        config["max_num_tokens"] = 8192
        config["moe_config"] = {"backend": "TRTLLM", "use_low_precision_moe_combine": True}
        config["cuda_graph_config"]["batch_sizes"] = [
            g for g in (1, 2, 4, 8, 16, 32, 64, 128) if g < local
        ] + [local]
        # Batch prefill requests instead of running each in its own mixed step (section 9).
        config["batch_wait_timeout_iters"] = 10
        config["batch_wait_max_tokens_ratio"] = 0.2
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cost-table",
        type=Path,
        required=True,
        help="cost table for the confidence configs",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=["dep8", "tep8"],
        default="dep8",
        help="dep8: attention DP (default); tep8: tensor-parallel attention",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="global batch sizes (default: 64 to 2048 for dep8, 8 to 64 for tep8)",
    )
    parser.add_argument("--fixed-k", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--confidence-k", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument(
        "--host-ranking",
        action="store_true",
        help="also write confidence configs without device re-ranking (enable_fused_confidence_scheduler)",
    )
    args = parser.parse_args()
    if args.batch_sizes is None:
        args.batch_sizes = (
            [8, 16, 32, 64] if args.layout == "tep8" else [64, 128, 256, 384, 512, 768, 1024, 2048]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    costs = json.loads(args.cost_table.read_text())["costs"]
    variants = [("confidence", True)] + ([("confidence-host", False)] if args.host_ranking else [])
    count, skipped = 0, []
    for batch_size in args.batch_sizes:
        for k in args.fixed_k:
            config = base_config(batch_size, args.layout)
            config["speculative_config"] = {
                "decoding_type": "DSpark",
                "max_draft_len": k,
                "block_size": k,
            }
            path = args.output_dir / f"bs{batch_size}-fixed-k{k}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            count += 1
        for k in args.confidence_k:
            # The runtime rejects a config whose captured graph sizes have no cost-table row.
            ladder = base_config(batch_size, args.layout)["cuda_graph_config"]["batch_sizes"]
            missing = [g for g in ladder if str(g) not in costs.get(str(k), {})]
            if missing:
                skipped.append(f"bs{batch_size}-k{k} (G {', '.join(map(str, missing))})")
                continue
            for name, rerank in variants:
                config = base_config(batch_size, args.layout)
                config["speculative_config"] = {
                    "decoding_type": "DSpark",
                    "max_draft_len": k,
                    "block_size": k,
                    "enable_confidence_scheduling": True,
                    "confidence_cost_table_path": str(args.cost_table.resolve()),
                    "confidence_calibration_path": str(args.calibration.resolve()),
                    "enable_fused_confidence_scheduler": rerank,
                }
                path = args.output_dir / f"bs{batch_size}-{name}-k{k}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                count += 1
    print(f"wrote {count} configs to {args.output_dir}")
    if skipped:
        print(f"no confidence configs for {', '.join(skipped)}: the cost table lacks those rows")


if __name__ == "__main__":
    main()
