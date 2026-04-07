#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and submit DWDP reproduction configs.

This script combines a user-provided environment YAML and a reproduction
matrix YAML, writes full benchmark configs, and forwards them to
``examples/disaggregated/slurm/benchmark/submit_dwdp.py``.
"""

import argparse
import math
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ModuleNotFoundError as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent / "disaggregated" / "slurm" / "benchmark"
SUBMIT_DWDP_SCRIPT = BENCHMARK_DIR / "submit_dwdp.py"
DEFAULT_ENV_CONFIG = SCRIPT_DIR / "env.yaml"
DEFAULT_REPRODUCE_CONFIG = SCRIPT_DIR / "dwdp_reproduce.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
TOTAL_EXPERTS = 256

DEFAULT_WORKER_ENV_VAR = (
    "TLLM_LOG_LEVEL=INFO TRTLLM_SERVER_DISABLE_GC=1 "
    "TRTLLM_WORKER_DISABLE_GC=1 TRTLLM_ENABLE_PDL=1 "
    "ENROOT_ALLOW_DEV=yes NCCL_GRAPH_MIXING_SUPPORT=0"
)
DEFAULT_SERVER_ENV_VAR = "TRTLLM_SERVER_DISABLE_GC=1"

REQUIRED_EXPERIMENT_FIELDS = {
    "isl",
    "osl",
    "num_ctx_servers",
    "ctx_tp",
    "num_gen_servers",
    "gen_tp",
    "batch",
    "gen_max_tokens",
    "ctx_max_bs",
    "ctx_max_num_tokens",
    "mtp",
    "eplb",
    "prefetch",
    "dwdp",
    "dwdp_group",
    "ratio",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate DWDP reproduction configs and submit them"
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG),
        help="Path to the environment YAML",
    )
    parser.add_argument(
        "--reproduce-config",
        default=str(DEFAULT_REPRODUCE_CONFIG),
        help="Path to the DWDP reproduction YAML",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated benchmark configs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and invoke submit_dwdp.py with --dry-run",
    )
    return parser.parse_args()


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level in {path}")
    return data


def merge_nested_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _as_bool(value: Any) -> bool:
    """Convert booleans and boolean-like strings to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _slugify(value: str) -> str:
    """Convert arbitrary text into a file-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "experiment"


def _cuda_graph_batch_sizes(gen_max_num_tokens: int) -> List[int]:
    """Build CUDA graph batch sizes following the internal benchmark defaults."""
    sizes = [1, 2, 4]
    stages = [(8, 1024, 8), (1040, 2048, 16)]
    for start, limit, step in stages:
        if gen_max_num_tokens >= start:
            end = min(gen_max_num_tokens, limit)
            sizes.extend(range(start, end + 1, step))
    if gen_max_num_tokens > 2048:
        sizes.append(gen_max_num_tokens)
    return sizes


def calc_seq_lens(isl: int, osl: int, ratio: float = 0) -> List[int]:
    """Return ``[ctx_max_seq_len, gen_max_seq_len]``."""
    if ratio != 0:
        if isl == 1024:
            ctx_max_seq_len = 1044
            gen_max_seq_len = 2068
        else:
            ctx_max_seq_len = isl + 1024
            gen_max_seq_len = ctx_max_seq_len + osl
    else:
        ctx_max_seq_len = isl + 1024
        gen_max_seq_len = ctx_max_seq_len + osl
    return [ctx_max_seq_len, gen_max_seq_len]


def build_worker_config(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``worker_config`` section for the benchmark launcher."""
    isl = int(experiment["isl"])
    osl = int(experiment["osl"])
    ratio = float(experiment.get("ratio", 0))
    ctx_tp = int(experiment["ctx_tp"])
    gen_tp = int(experiment["gen_tp"])
    batch = int(experiment["batch"])
    gen_max_tokens = int(experiment["gen_max_tokens"])
    ctx_max_bs = int(experiment.get("ctx_max_bs", 4))
    ctx_max_num_tokens = int(experiment.get("ctx_max_num_tokens", isl + 1024))
    mtp = int(experiment.get("mtp", 0))
    eplb = int(experiment.get("eplb", 0))
    enable_dp = _as_bool(experiment.get("enable_dp", True))
    prefetch = int(experiment.get("prefetch", 0))
    dwdp_enabled = _as_bool(experiment.get("dwdp", False))
    dwdp_group = int(experiment.get("dwdp_group", 1))
    num_ctx_servers = int(experiment["num_ctx_servers"])

    if dwdp_group <= 0:
        raise ValueError("dwdp_group must be greater than zero")
    if prefetch < 0:
        raise ValueError("prefetch must be greater than or equal to zero")
    if dwdp_enabled and num_ctx_servers % dwdp_group != 0:
        raise ValueError("num_ctx_servers must be divisible by dwdp_group when DWDP is enabled")

    ctx_max_seq_len, gen_max_seq_len = calc_seq_lens(isl, osl, ratio)
    max_tokens_in_buffer = math.ceil(ctx_max_seq_len / 64) * 64
    dwdp_size = num_ctx_servers // dwdp_group if dwdp_enabled else 1
    experts_per_worker = int(TOTAL_EXPERTS - (dwdp_size - 1) * prefetch)
    if dwdp_enabled and experts_per_worker <= 0:
        raise ValueError(
            "Invalid DWDP configuration: "
            f"TOTAL_EXPERTS={TOTAL_EXPERTS}, dwdp_size={dwdp_size}, and prefetch={prefetch} "
            "produce a non-positive num_experts_per_worker. "
            "Reduce prefetch or use a smaller DWDP size."
        )

    gen_cfg: Dict[str, Any] = {
        "tensor_parallel_size": gen_tp,
        "moe_expert_parallel_size": gen_tp,
        "enable_attention_dp": True,
        "enable_lm_head_tp_in_adp": True,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "max_batch_size": batch,
        "max_num_tokens": gen_max_tokens,
        "max_seq_len": gen_max_seq_len,
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": _cuda_graph_batch_sizes(gen_max_tokens),
        },
        "print_iter_log": True,
        "trust_remote_code": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "dtype": "fp8",
            "free_gpu_memory_fraction": 0.75,
        },
        "stream_interval": 100,
        "moe_config": {
            "backend": "CUTEDSL",
        },
        "cache_transceiver_config": {
            "backend": "UCX",
            "max_tokens_in_buffer": max_tokens_in_buffer,
        },
        "num_postprocess_workers": 4,
    }
    if mtp > 0:
        gen_cfg["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": mtp,
        }
    if eplb > 0:
        gen_cfg["moe_config"]["load_balancer"] = {
            "num_slots": eplb,
            "layer_updates_per_iter": 1,
        }

    ctx_cfg: Dict[str, Any] = {
        "max_batch_size": ctx_max_bs,
        "max_num_tokens": ctx_max_num_tokens,
        "max_seq_len": ctx_max_seq_len,
        "tensor_parallel_size": ctx_tp,
        "context_parallel_size": 1,
        "moe_expert_parallel_size": ctx_tp,
        "enable_attention_dp": enable_dp,
        "pipeline_parallel_size": 1,
        "print_iter_log": True,
        "trust_remote_code": True,
        "cuda_graph_config": None,
        "disable_overlap_scheduler": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "dtype": "fp8",
            "free_gpu_memory_fraction": 0.3,
        },
        "cache_transceiver_config": {
            "backend": "UCX",
            "max_tokens_in_buffer": max_tokens_in_buffer,
        },
        "moe_config": {
            "backend": "CUTEDSL",
        },
    }
    if dwdp_enabled:
        ctx_cfg["dwdp_config"] = {
            "dwdp_size": dwdp_size,
            "num_groups": dwdp_group,
            "num_experts_per_worker": experts_per_worker,
            "num_prefetch_experts": prefetch,
        }
    if mtp > 0:
        ctx_cfg["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": mtp,
        }

    return {"gen": gen_cfg, "ctx": ctx_cfg}


def _build_sub_file(experiment: Dict[str, Any]) -> str:
    """Build the benchmark config identifier."""
    batch = int(experiment["batch"])
    gen_tp = int(experiment["gen_tp"])
    isl = int(experiment["isl"])
    osl = int(experiment["osl"])
    mtp = int(experiment.get("mtp", 0))
    eplb = int(experiment.get("eplb", 0))
    ratio = experiment.get("ratio", 0)
    num_ctx_servers = int(experiment["num_ctx_servers"])
    ctx_tp = int(experiment["ctx_tp"])
    num_gen_servers = int(experiment["num_gen_servers"])
    dwdp = str(experiment.get("dwdp", "false")).lower()
    return (
        f"isl{isl}_osl{osl}_sa{ratio}_lbz{batch}_"
        f"{num_ctx_servers}ctx{ctx_tp}_{num_gen_servers}gen{gen_tp}_"
        f"dwdp{dwdp}_mtp{mtp}_eplb{eplb}"
    )


def resolve_dataset_file(env_config: Dict[str, Any], experiment: Dict[str, Any]) -> str:
    """Resolve the dataset path for an experiment."""
    dataset_file = experiment.get("dataset_file")
    if dataset_file:
        return str(dataset_file)

    dataset_key = experiment.get("dataset_key")
    datasets = env_config.get("datasets", {})
    if dataset_key:
        if dataset_key not in datasets:
            raise ValueError(f"dataset_key '{dataset_key}' is not defined in env.yaml datasets")
        return str(datasets[dataset_key])

    raise ValueError("Each experiment must define either dataset_file or dataset_key")


def build_job_name(slurm_config: Dict[str, Any], experiment: Dict[str, Any], sub_file: str) -> str:
    """Build a short, descriptive Slurm job name."""
    prefix = str(slurm_config.get("job_name_prefix", "dwdp-reproduce"))
    experiment_name = experiment.get("name")
    if experiment_name:
        return f"{prefix}_{_slugify(str(experiment_name))}"
    return f"{prefix}_{sub_file}"


def build_accuracy_config(env_config: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Build the accuracy section."""
    base_accuracy = env_config.get("accuracy", {})
    experiment_accuracy = experiment.get("accuracy", {})
    accuracy = merge_nested_dicts(base_accuracy, experiment_accuracy)
    # Use env.yaml as the default; experiment-level override takes precedence.
    if "enable_accuracy_test" in experiment_accuracy:
        accuracy["enable_accuracy_test"] = _as_bool(experiment_accuracy["enable_accuracy_test"])
    else:
        accuracy["enable_accuracy_test"] = _as_bool(
            base_accuracy.get("enable_accuracy_test", False)
        )
    accuracy.setdefault("tasks", {})
    return accuracy


def build_full_config(env_config: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Build a full config for submit_dwdp.py."""
    slurm_config = env_config.get("slurm", {})
    hardware_config = env_config.get("hardware", {})
    environment_config = env_config.get("environment", {})
    benchmark_defaults = env_config.get("benchmark_defaults", {})
    profiling_defaults = env_config.get("profiling", {})

    missing_slurm = [key for key in ("partition", "account", "time") if key not in slurm_config]
    if missing_slurm:
        raise ValueError(
            f"env.yaml missing required slurm keys: {', '.join(sorted(missing_slurm))}"
        )

    if "gpus_per_node" not in hardware_config:
        raise ValueError("env.yaml missing required hardware.gpus_per_node")

    missing_environment = [
        key
        for key in ("container_image", "container_mount", "model_path")
        if key not in environment_config
    ]
    if missing_environment:
        raise ValueError(
            "env.yaml missing required environment keys: " + ", ".join(sorted(missing_environment))
        )

    missing_fields = sorted(REQUIRED_EXPERIMENT_FIELDS - set(experiment))
    if missing_fields:
        raise ValueError("Experiment is missing required fields: " + ", ".join(missing_fields))

    isl = int(experiment["isl"])
    osl = int(experiment["osl"])
    batch = int(experiment["batch"])
    gen_tp = int(experiment["gen_tp"])
    if batch <= 0 or gen_tp <= 0:
        raise ValueError("batch and gen_tp must be greater than zero")
    num_prompts = int(experiment.get("num_prompts", 4096 if osl == 1 else 20000))
    concurrency_list = str(experiment.get("concurrency_list", batch * gen_tp))
    multi_round = int(experiment.get("multi_round", max(1, num_prompts // (batch * gen_tp))))
    sub_file = _build_sub_file(experiment)
    dataset_file = resolve_dataset_file(env_config, experiment)
    launcher_work_dir = str(environment_config.get("work_dir", BENCHMARK_DIR))

    full_environment = {
        "container_mount": environment_config["container_mount"],
        "container_image": environment_config["container_image"],
        "model_path": environment_config["model_path"],
        "trtllm_repo": environment_config.get("trtllm_repo", ""),
        "build_wheel": _as_bool(environment_config.get("build_wheel", False)),
        "cuda_architectures": environment_config.get("cuda_architectures", ""),
        "trtllm_wheel_path": environment_config.get("trtllm_wheel_path", ""),
        "work_dir": launcher_work_dir,
        "worker_env_var": environment_config.get("worker_env_var", DEFAULT_WORKER_ENV_VAR),
        "server_env_var": environment_config.get("server_env_var", DEFAULT_SERVER_ENV_VAR),
    }
    for key in ("ctx_worker_env_var", "gen_worker_env_var"):
        if key in environment_config:
            full_environment[key] = environment_config[key]
    full_environment = merge_nested_dicts(full_environment, experiment.get("environment", {}))
    if not full_environment.get("log_dir"):
        full_environment.pop("log_dir", None)
    benchmark_config = {
        "mode": benchmark_defaults.get("mode", "e2e"),
        "use_nv_sa_benchmark": _as_bool(benchmark_defaults.get("use_nv_sa_benchmark", False)),
        "multi_round": multi_round,
        "benchmark_ratio": float(experiment.get("ratio", 0)),
        "streaming": _as_bool(benchmark_defaults.get("streaming", True)),
        "concurrency_list": concurrency_list,
        "input_length": isl,
        "output_length": osl,
        "dataset_file": dataset_file,
    }
    benchmark_config = merge_nested_dicts(benchmark_config, experiment.get("benchmark", {}))

    profiling_config = {
        "nsys_on": _as_bool(profiling_defaults.get("nsys_on", False)),
        "ctx_profile_range": profiling_defaults.get("ctx_profile_range", "10-15"),
        "gen_profile_range": profiling_defaults.get("gen_profile_range", "200-250"),
    }
    profiling_config = merge_nested_dicts(profiling_config, experiment.get("profiling", {}))

    config = {
        "slurm": {
            "script_file": "disaggr_torch_dwdp.slurm",
            "partition": slurm_config["partition"],
            "account": slurm_config["account"],
            "job_time": slurm_config["time"],
            "job_name": build_job_name(slurm_config, experiment, sub_file),
            "extra_args": slurm_config.get("extra_args", ""),
            "set_segment": _as_bool(slurm_config.get("set_segment", True)),
            "numa_bind": _as_bool(slurm_config.get("numa_bind", True)),
        },
        "benchmark": benchmark_config,
        "hardware": {
            "gpus_per_node": int(hardware_config["gpus_per_node"]),
            "num_ctx_servers": int(experiment["num_ctx_servers"]),
            "num_gen_servers": int(experiment["num_gen_servers"]),
        },
        "environment": full_environment,
        "profiling": profiling_config,
        "accuracy": build_accuracy_config(env_config, experiment),
        "worker_config": build_worker_config(experiment),
    }
    config["slurm"] = merge_nested_dicts(config["slurm"], experiment.get("slurm", {}))
    return config


def write_config_file(config: Dict[str, Any], output_path: Path) -> None:
    """Write the generated config to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, default_flow_style=False, sort_keys=False)


def build_output_path(output_dir: Path, experiment: Dict[str, Any]) -> Path:
    """Build the config output path for one experiment."""
    sub_file = _build_sub_file(experiment)
    name = experiment.get("name")
    if name:
        filename = f"{_slugify(str(name))}_{sub_file}_config.yaml"
    else:
        filename = f"{sub_file}_config.yaml"
    return output_dir / filename


def load_experiments(reproduce_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load experiments after applying top-level defaults."""
    experiment_defaults = reproduce_config.get("experiment_defaults", {})
    if not isinstance(experiment_defaults, dict):
        raise ValueError("experiment_defaults must be a mapping when provided")
    experiments = reproduce_config.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("dwdp_reproduce.yaml must contain a non-empty experiments list")

    merged_experiments = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment entry must be a mapping")
        merged_experiment = merge_nested_dicts(experiment_defaults, experiment)
        merged_experiment.setdefault("isl_std", 0)
        merged_experiments.append(merged_experiment)
    return merged_experiments


def submit_config(submit_dwdp_script: Path, config_path: Path, dry_run: bool) -> None:
    """Forward a generated config to submit_dwdp.py."""
    command = [sys.executable, str(submit_dwdp_script), "-c", str(config_path)]
    if dry_run:
        command.append("--dry-run")

    print("  " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    """Entry point."""
    args = parse_args()

    if YAML_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing Python dependency: 'pyyaml'. Install it with: python3 -m pip install pyyaml"
        ) from YAML_IMPORT_ERROR

    if not SUBMIT_DWDP_SCRIPT.is_file():
        raise FileNotFoundError(f"submit_dwdp.py not found at expected path: {SUBMIT_DWDP_SCRIPT}")

    env_config = load_yaml_file(Path(args.env_config))
    reproduce_config = load_yaml_file(Path(args.reproduce_config))
    experiments = load_experiments(reproduce_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, experiment in enumerate(experiments, start=1):
        name = experiment.get("name", _build_sub_file(experiment))
        print(f"[{index}/{len(experiments)}] {name}")
        config = build_full_config(env_config, experiment)
        output_path = build_output_path(output_dir, experiment)
        write_config_file(config, output_path)
        print(f"  Generated config: {output_path}")
        submit_config(SUBMIT_DWDP_SCRIPT, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
