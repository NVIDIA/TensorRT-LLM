#!/usr/bin/env python3
"""Generate TensorRT-LLM autodeploy configuration files.

Dynamically loads:
- Transforms from tensorrt_llm/_torch/auto_deploy/config/default.yaml
- Config fields from tensorrt_llm/_torch/auto_deploy/llm_args.py
"""

import argparse
import sys
from pathlib import Path

import yaml

# Minimal docs for common transforms
COMMON_TRANSFORM_DOCS = {
    "fuse_fp8_gemms": "Fuse FP8 GEMM operations",
    "multi_stream_moe": "Multi-stream MoE execution",
    "insert_cached_ssm_attention": "Cached SSM attention (Mamba)",
    "detect_sharding": "Auto-detect TP/EP parallelism",
}

# Key config fields from llm_args.py
LLMARGS_CONFIG_FIELDS = {
    # Model/Tokenizer
    "model_factory": "Model factory to use (default: AutoModelForCausalLM)",
    "model_kwargs": "Extra kwargs for model config (dict)",
    "skip_loading_weights": "Skip loading weights (bool, default: false)",
    "tokenizer_kwargs": "Extra kwargs for tokenizer (dict)",
    # Runtime
    "world_size": "Number of GPUs for auto-sharding (int, default: 1)",
    "runtime": "Runtime to use: trtllm or demollm (default: trtllm)",
    "device": "Device to use (default: cuda)",
    "sampler_type": "Sampler type: TorchSampler or TRTLLMSampler",
    "gpus_per_node": "GPUs per node (default: auto-detect)",
    # Inference Optimizer
    "mode": "Optimizer mode: graph or transformers (default: graph)",
    "transforms": "Transform configurations (dict)",
    "attn_backend": "Attention backend (default: flashinfer)",
    "compile_backend": "Compile backend (default: torch-compile)",
    "cuda_graph_batch_sizes": "Batch sizes for CUDA graphs (list[int])",
    # Sequence Interface
    "max_seq_len": "Maximum sequence length (int, default: 512)",
    "max_batch_size": "Maximum batch size (int, default: 8)",
    # Advanced
    "enable_chunked_prefill": "Enable chunked prefill (bool)",
    "kv_cache_config": "KV cache configuration (dict)",
    "speculative_config": "Speculative decoding config",
    "draft_checkpoint_loader": "Draft model checkpoint loader",
}


def find_trtllm_repo():
    """Find TensorRT-LLM repository."""
    search_paths = [
        Path.cwd(),
        Path.cwd().parent,
        Path.home() / "dev" / "TensorRT-LLM",
        Path.home() / "TensorRT-LLM",
        Path("/workspace/TensorRT-LLM"),
    ]
    for path in search_paths:
        if (path / "tensorrt_llm" / "_torch" / "auto_deploy").exists():
            return path
    return None


def find_default_yaml():
    """Find default.yaml."""
    repo = find_trtllm_repo()
    if repo:
        return repo / "tensorrt_llm" / "_torch" / "auto_deploy" / "config" / "default.yaml"
    return None


def load_available_transforms(default_yaml_path=None):
    """Load transforms from default.yaml."""
    if default_yaml_path is None:
        default_yaml_path = find_default_yaml()

    if not default_yaml_path or not default_yaml_path.exists():
        print("Warning: Could not find default.yaml.")
        return {}

    try:
        with open(default_yaml_path) as f:
            config = yaml.safe_load(f)
        transforms = config.get("transforms", {})
        print(f"Loaded {len(transforms)} transforms from: {default_yaml_path}")
        return transforms
    except Exception as e:
        print(f"Warning: Could not load default.yaml: {e}")
        return {}


def get_base_config(max_seq_len=512):
    """Minimal base config (dashboard_default.yaml)."""
    return {
        "runtime": "trtllm",
        "attn_backend": "flashinfer",
        "compile_backend": "torch-cudagraph",
        "model_factory": "AutoModelForCausalLM",
        "skip_loading_weights": False,
        "max_seq_len": max_seq_len,
    }


def add_world_size(config, world_size):
    """Add world_size for multi-GPU."""
    if world_size > 1:
        config["world_size"] = world_size


def add_kv_cache(config, dtype="auto", mem_fraction=None, mamba_cache_dtype=None):
    """Add KV cache config."""
    kv_config = {}
    if dtype != "auto":
        kv_config["dtype"] = dtype
    if mem_fraction is not None:
        kv_config["free_gpu_memory_fraction"] = mem_fraction
    if mamba_cache_dtype is not None:
        kv_config["mamba_ssm_cache_dtype"] = mamba_cache_dtype

    if kv_config:
        config["kv_cache_config"] = kv_config


def add_transform(config, transform_name, **kwargs):
    """Add transform."""
    if "transforms" not in config:
        config["transforms"] = {}
    config["transforms"][transform_name] = kwargs


def print_available_configs():
    """Print available config fields from llm_args.py."""
    print("\n" + "=" * 80)
    print("AVAILABLE CONFIG FIELDS (from llm_args.py)")
    print("=" * 80 + "\n")

    categories = {
        "Model and Tokenizer": [
            "model_factory",
            "model_kwargs",
            "skip_loading_weights",
            "tokenizer_kwargs",
        ],
        "Runtime Features": ["world_size", "runtime", "device", "sampler_type", "gpus_per_node"],
        "Inference Optimizer": [
            "mode",
            "transforms",
            "attn_backend",
            "compile_backend",
            "cuda_graph_batch_sizes",
        ],
        "Sequence Interface": ["max_seq_len", "max_batch_size", "enable_chunked_prefill"],
        "Cache Configuration": ["kv_cache_config"],
        "Advanced": ["speculative_config", "draft_checkpoint_loader"],
    }

    for category, fields in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for field in fields:
            desc = LLMARGS_CONFIG_FIELDS.get(field, "")
            print(f"  {field}")
            if desc:
                print(f"    {desc}")


def print_available_transforms(available_transforms):
    """Print transforms from default.yaml."""
    if not available_transforms:
        print("No transforms loaded.")
        return

    print("\n" + "=" * 80)
    print(f"AVAILABLE TRANSFORMS ({len(available_transforms)} from default.yaml)")
    print("=" * 80 + "\n")

    # Group by stage
    by_stage = {}
    for name, props in available_transforms.items():
        stage = props.get("stage", "unknown")
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append(name)

    stage_order = [
        "factory",
        "export",
        "post_export",
        "pattern_matcher",
        "sharding",
        "weight_load",
        "post_load_fusion",
        "visualize",
        "cache_init",
        "compile",
    ]

    for stage in stage_order:
        if stage not in by_stage:
            continue

        print(f"\n{stage.upper().replace('_', ' ')}:")
        print("-" * 80)

        for transform in sorted(by_stage[stage]):
            props = available_transforms[transform]
            doc = COMMON_TRANSFORM_DOCS.get(transform, "")

            prop_str = ", ".join(f"{k}={v}" for k, v in props.items() if k != "stage")
            if prop_str:
                print(f"  {transform} ({prop_str})")
            else:
                print(f"  {transform}")

            if doc:
                print(f"    → {doc}")

    other_stages = set(by_stage.keys()) - set(stage_order)
    if other_stages:
        print("\n\nOTHER STAGES:")
        for stage in sorted(other_stages):
            print(f"\n{stage}:")
            for transform in sorted(by_stage[stage]):
                print(f"  {transform}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TensorRT-LLM autodeploy config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all config fields
  %(prog)s --list-configs

  # List all transforms
  %(prog)s --list-transforms

  # Generate minimal config
  %(prog)s -o config.yaml

  # Add common settings
  %(prog)s -o config.yaml \\
    --world-size 8 \\
    --kv-cache-dtype fp8 \\
    --mem-fraction 0.9

  # Add any config key:value
  %(prog)s -o config.yaml \\
    --set max_batch_size=256 \\
    --set enable_chunked_prefill=true \\
    --set cuda_graph_batch_sizes=[1,2,4,8,16,32,64,128,256]

  # Add transforms
  %(prog)s -o config.yaml \\
    --add-transform fuse_fp8_gemms enabled=true stage=post_load_fusion \\
    --add-transform multi_stream_moe enabled=true stage=compile

  # Complex example
  %(prog)s -o config.yaml \\
    --max-seq-len 65536 \\
    --world-size 4 \\
    --kv-cache-dtype fp8 \\
    --set max_batch_size=384 \\
    --set mode=graph \\
    --add-transform fuse_fp8_gemms enabled=true

  # Fast debug config (DO NOT USE FOR ACCURACY/PERFORMANCE TESTS)
  # Option 1: Skip loading weights (random weights)
  %(prog)s -o debug_fast.yaml \\
    --set skip_loading_weights=true \\
    --set max_batch_size=8

  # Option 2: Reduce layers (2-10 layers for fast iteration)
  %(prog)s -o debug_fast.yaml \\
    --set model_kwargs={num_hidden_layers:2}

  # Combined: random weights + fewer layers
  %(prog)s -o debug_fast.yaml \\
    --set skip_loading_weights=true \\
    --set model_kwargs={num_hidden_layers:2} \\
    --set max_batch_size=8
        """,
    )

    parser.add_argument("--default-yaml", type=Path, help="Path to default.yaml (auto-detected)")

    parser.add_argument(
        "--list-configs", action="store_true", help="List all config fields from llm_args.py"
    )

    parser.add_argument(
        "--list-transforms", action="store_true", help="List all transforms from default.yaml"
    )

    parser.add_argument("-o", "--output", help="Output YAML file path")

    # Base config
    parser.add_argument(
        "--max-seq-len", type=int, default=512, help="Maximum sequence length (default: 512)"
    )

    # Common helpers
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs (default: 1)")

    parser.add_argument(
        "--kv-cache-dtype",
        choices=["fp8", "fp16", "auto", "int8"],
        default="auto",
        help="KV cache dtype (default: auto)",
    )
    parser.add_argument("--mem-fraction", type=float, help="GPU memory fraction for KV cache")
    parser.add_argument(
        "--mamba-cache-dtype", choices=["auto", "float32"], help="Mamba cache dtype"
    )

    # Generic config setter
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Set config key:value (e.g., --set max_batch_size=256)",
    )

    # Transforms
    parser.add_argument(
        "--add-transform",
        action="append",
        nargs="+",
        metavar=("TRANSFORM", "KEY=VALUE"),
        help="Add transform with params",
    )

    args = parser.parse_args()

    # Load transforms
    available_transforms = load_available_transforms(args.default_yaml)

    # List configs and exit
    if args.list_configs:
        print_available_configs()
        return 0

    # List transforms and exit
    if args.list_transforms:
        print_available_transforms(available_transforms)
        return 0

    # Require output
    if not args.output:
        parser.print_help()
        print("\nError: --output required")
        return 1

    # Generate config
    config = get_base_config(args.max_seq_len)
    add_world_size(config, args.world_size)
    add_kv_cache(config, args.kv_cache_dtype, args.mem_fraction, args.mamba_cache_dtype)

    # Add generic configs
    if args.set:
        for setting in args.set:
            if "=" not in setting:
                print(f"Error: Invalid --set format: {setting}")
                continue

            key, value = setting.split("=", 1)
            key = key.strip()

            # Parse value
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.startswith("[") and value.endswith("]"):
                try:
                    value = eval(value)
                except Exception:
                    pass
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").replace("-", "").isdigit():
                value = float(value)

            config[key] = value

    # Add transforms
    if args.add_transform:
        for transform_args in args.add_transform:
            transform_name = transform_args[0]

            if available_transforms and transform_name not in available_transforms:
                print(f"Warning: '{transform_name}' not in default.yaml")

            transform_params = {}
            for param in transform_args[1:]:
                if "=" in param:
                    key, value = param.split("=", 1)
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "").replace("-", "").isdigit():
                        value = float(value)
                    transform_params[key] = value

            add_transform(config, transform_name, **transform_params)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfiguration written to: {output_path}")
    print("\nGenerated config preview:")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
