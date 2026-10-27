#!/usr/bin/env python3
"""Parse a trtllm-test-specialist YAML config file.

Validates that the file is well-formed YAML, filters to known input keys, warns
about unrecognized keys, fills in scope-specific default values for absent
optional keys, and prints the resolved parameters as JSON to stdout.

Usage:
    parse_config.py --config-file <path>

Output (JSON on stdout):
    {
      "params": { <key>: <value>, ... },
      "missing_required": [ <key>, ... ],
      "defaults_applied": [ <key>, ... ]
    }

    "params"           — recognised keys found in the config file plus any
                         defaults that were filled in for absent optional keys.
    "missing_required" — required keys still absent after defaults were applied,
                         determined by test scope:
                           * test_cmd present  → module-test required keys
                           * test_type present → type-specific required keys
                           * otherwise         → universally required keys
    "defaults_applied" — keys whose values were filled in from _DEFAULTS_BY_TYPE
                         (not present in the config file).

Exit codes:
    0  success — JSON printed to stdout
    1  file not found, bad extension, or YAML parse error
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# All recognised input keys from the trtllm-test-specialist Inputs section.
# Keys are grouped by section for readability; the set is what matters at runtime.
KNOWN_KEYS = {
    # Common
    "model_name",
    "checkpoint_path",
    "repo_path",
    "report_file",
    # Module test
    "test_cmd",
    "class_name",
    "required_devices",
    # Model test — classification
    "test_type",
    # Model test — benchmark
    "mtp_layers",
    "benchmark_model_name",
    # Model test — parallelism
    "tp_size",
    "ep_size",
    "dp_size",
    "pp_size",
    # Model test — hardware
    "device_type",
    # Model test — dataset / paths
    "dataset_path",
    "benchmark_config_yaml",
    "extra_llm_api_options_yaml",
    # Model test — trtllm-bench
    "bench_subcommand",
    "backend",
    "workspace",
    "num_requests",
    "input_mean",
    "input_stdev",
    "output_mean",
    "output_stdev",
    "kv_cache_free_gpu_mem_fraction",
    "max_seq_len",
    "beam_width",
    "warmup",
    "streaming",
    "no_chunked_context",
    "scheduler_policy",
    "cluster_size",
    "concurrency",
    "max_batch_size",
    "max_num_tokens",
    "iteration_log",
    # Model test — trtllm-eval (global options)
    "eval_tasks",
    "eval_tokenizer",
    "eval_trust_remote_code",
    "eval_disable_kv_cache_reuse",
    "eval_kv_cache_free_gpu_memory_fraction",
    # Model test — trtllm-eval (per-task options, common across most tasks)
    "eval_num_samples",
    "eval_apply_chat_template",
    "eval_chat_template_kwargs",
    "eval_system_prompt",
    "eval_max_input_length",
    "eval_max_output_length",
    "eval_log_samples",
    "eval_output_path",
    # Model test — trtllm-eval (task-specific options)
    "eval_check_accuracy",
    "eval_accuracy_threshold",
    "eval_num_fewshot",
}

# Required keys for the module-test scope (test_cmd present).
_REQUIRED_MODULE = {"test_cmd"}

# Required keys when no test_cmd and no test_type — most conservative baseline.
_REQUIRED_ALWAYS = {"model_name", "checkpoint_path"}

# Required keys per explicit test_type value.
_REQUIRED_BY_TYPE: dict[str, set[str]] = {
    "functionality": {"model_name", "checkpoint_path"},
    "benchmark": {"model_name", "checkpoint_path", "tp_size", "pp_size"},
    "evaluation": {"checkpoint_path", "eval_tasks"},
}

# Default values for absent optional (and some required) keys, keyed by test_type.
# Applied before the missing_required check so defaults can satisfy required fields
# (e.g. tp_size=1 satisfies the benchmark requirement without forcing explicit declaration).
_DEFAULTS_BY_TYPE: dict[str, dict] = {
    "benchmark": {
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 0,
        "bench_subcommand": "throughput",
        "backend": "pytorch",
        "num_requests": 100,
        "input_mean": 128,
        "input_stdev": 0,
        "output_mean": 128,
        "output_stdev": 0,
        "beam_width": 1,
        "warmup": 2,
        "kv_cache_free_gpu_mem_fraction": 0.90,
    },
    "evaluation": {
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 0,
        "dp_size": 1,
    },
    "functionality": {
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 0,
        "dp_size": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config-file", required=True, metavar="PATH", help="Path to the YAML config file"
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: config file not found: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        sys.exit(f"ERROR: config file must have a .yaml or .yml extension, got: {path.suffix!r}")
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        sys.exit(f"ERROR: failed to parse YAML: {exc}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        sys.exit(f"ERROR: config file must be a YAML mapping (got {type(data).__name__})")
    return data


def apply_defaults(params: dict) -> tuple[dict, list[str]]:
    """Fill in default values for absent keys based on test_type.

    Only keys absent from *params* are touched; explicit values are never overridden.
    Returns (updated_params, sorted list of keys that received a default value).
    """
    scope_defaults = _DEFAULTS_BY_TYPE.get(params.get("test_type"), {})
    result = dict(params)
    defaulted: list[str] = []
    for key, value in scope_defaults.items():
        if key not in result:
            result[key] = value
            defaulted.append(key)
    return result, sorted(defaulted)


def get_missing_required(params: dict) -> list[str]:
    """Return sorted list of required keys absent from *params*."""
    if "test_cmd" in params:
        required = _REQUIRED_MODULE
    elif "test_type" in params:
        required = _REQUIRED_BY_TYPE.get(params["test_type"], _REQUIRED_ALWAYS)
    else:
        required = _REQUIRED_ALWAYS
    return sorted(required - set(params.keys()))


def main() -> None:
    args = parse_args()
    path = Path(args.config_file)
    raw = load_yaml(path)

    unknown = sorted(raw.keys() - KNOWN_KEYS)
    if unknown:
        print(f"WARNING: unrecognized keys ignored: {', '.join(unknown)}", file=sys.stderr)

    params = {k: v for k, v in raw.items() if k in KNOWN_KEYS}
    params, defaults_applied = apply_defaults(params)
    missing_required = get_missing_required(params)

    print(
        json.dumps(
            {
                "params": params,
                "missing_required": missing_required,
                "defaults_applied": defaults_applied,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
