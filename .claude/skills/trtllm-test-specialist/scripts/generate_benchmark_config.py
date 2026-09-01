#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate a perf-sanity compatible benchmark config YAML (aggregated or disaggregated).

Takes user-provided benchmark parameters and produces a YAML file that can be
passed to submit.py via --config-file for Slurm job submission.

Config types (--config-type):
  aggr    Aggregated mode: single-server deployment with server_configs[] format.
          Use for standard single-role benchmarks (trtllm-serve aggregated).
  disagg  Disaggregated mode (default): separate ctx/gen workers with worker_config format.
          Use for disaggregated prefill/decode serving benchmarks.

Supports two generation modes:
  1. Generate from scratch: provide --model-name and parameters.
  2. Load from existing config: provide --from-config to load a base YAML and
     optionally override specific fields with CLI arguments. Config type is
     auto-detected from the loaded file (server_configs → aggr, worker_config → disagg).

Usage examples:

  # Aggregated: generate from scratch
  python3 generate_benchmark_config.py \
    --config-type aggr \
    --model-name gpt_oss_120b_fp4 \
    --tp 4 --ep 4 --pp 2 \
    --max-batch-size 256 --max-num-tokens 20000 \
    --mtp-layers 1 \
    --concurrency 256 --iterations 10 \
    --input-length 1024 --output-length 1024 \
    --output config.yaml

  # Aggregated: load template and override MTP layers
  python3 generate_benchmark_config.py \
    --from-config references/agg_config_template.yaml \
    --mtp-layers 3 \
    --output config.yaml

  # Disaggregated: model + hardware essentials
  python3 generate_benchmark_config.py \
    --model-name deepseek_r1_0528_fp4_v2 \
    --gen-tp 8 --ctx-tp 4 \
    --output config.yaml

  # Disaggregated: from existing config with overrides
  python3 generate_benchmark_config.py \
    --from-config examples/disaggregated/slurm/benchmark/config.yaml \
    --gen-tp 4 --concurrency-list "16,32" \
    --output config.yaml

  # Disaggregated: full control
  python3 generate_benchmark_config.py \
    --model-name deepseek_r1_0528_fp4_v2 \
    --benchmark-mode e2e \
    --gpus-per-node 8 \
    --num-ctx-servers 1 --num-gen-servers 1 \
    --gen-tp 8 --gen-pp 1 --gen-cp 1 \
    --gen-max-batch-size 256 --gen-max-num-tokens 512 \
    --ctx-tp 4 --ctx-pp 1 --ctx-cp 1 \
    --ctx-max-batch-size 4 --ctx-max-num-tokens 4608 \
    --input-length 1024 --output-length 1024 \
    --concurrency-list "16,32,64" \
    --streaming \
    --kv-cache-dtype fp8 \
    --mtp-layers 1 \
    --cache-backend UCX \
    --output config.yaml
"""

import argparse
import os
import sys

import yaml


def _sanitize_placeholders(obj):
    """Recursively replace '<placeholder>' strings with None in a config dict.

    Template files (e.g., benchmark_config_template.yaml) contain angle-bracket
    placeholder strings such as "<dataset_file>" or "<model_path>" for fields the
    user is expected to fill in.  After loading the template we strip those so they
    do not propagate as literal strings into the generated config.

    A string is considered a placeholder when it matches the pattern ^<[^>]+>$
    (entire value is a single angle-bracketed token).
    """
    import re

    _placeholder_re = re.compile(r"^<[^>]+>$")

    if isinstance(obj, dict):
        return {k: _sanitize_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_placeholders(v) for v in obj]
    if isinstance(obj, str) and _placeholder_re.match(obj):
        return None
    return obj


def load_and_normalize_config(config_path):
    """Load an existing config YAML and normalize it to perf-sanity format.

    Handles both perf-sanity style configs (which have metadata/benchmark/hardware/worker_config)
    and examples/disaggregated/slurm/benchmark/config.yaml format (which has slurm/environment
    sections). Strips non-perf-sanity sections (slurm, environment) and keeps the rest.
    Placeholder strings of the form "<name>" (used in template files) are replaced with None.

    Returns:
        dict: Normalized config dict compatible with perf-sanity format.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _sanitize_placeholders(config)

    # If it already has the perf-sanity structure, return as-is (minus extra sections)
    normalized = {}

    # Copy over the sections we care about
    if "metadata" in config:
        normalized["metadata"] = config["metadata"]
    elif "model_name" in config.get("environment", {}):
        # Derive metadata from environment.model_path if available
        model_path = config["environment"].get("model_path", "unknown")
        normalized["metadata"] = {
            "model_name": os.path.basename(model_path),
            "benchmark_type": "",
        }

    if "benchmark" in config:
        normalized["benchmark"] = config["benchmark"]

    if "hardware" in config:
        normalized["hardware"] = config["hardware"]

    if "worker_config" in config:
        normalized["worker_config"] = config["worker_config"]

    # Aggregated format: pass server_configs through unchanged
    if "server_configs" in config:
        normalized["server_configs"] = config["server_configs"]

    if "environment" in config:
        # Only keep worker_env_var and server_env_var
        env = config["environment"]
        normalized["environment"] = {
            "worker_env_var": env.get("worker_env_var", ""),
            "server_env_var": env.get("server_env_var", ""),
        }

    if "profiling" in config:
        normalized["profiling"] = config["profiling"]

    if "accuracy" in config:
        normalized["accuracy"] = _normalize_accuracy_section(config["accuracy"])

    return normalized


def _normalize_accuracy_section(acc):
    """Convert flat accuracy format (perf-sanity) to nested tasks format (examples).

    Flat format (perf-sanity):
        accuracy:
          enable_accuracy_test: false
          model: local-completions
          tasks: gsm8k
          model_args_extra: "..."

    Nested format (examples / target):
        accuracy:
          enable_accuracy_test: false
          tasks:
            gsm8k:
              model: local-completions
              model_args_extra: "..."
              extra_kwargs: {}
    """
    if not acc:
        return acc
    tasks = acc.get("tasks")
    # Already nested format — tasks is a dict mapping task name → config
    if isinstance(tasks, dict):
        return acc
    # Flat format — tasks is a string like "gsm8k"
    task_name = tasks or "gsm8k"
    return {
        "enable_accuracy_test": acc.get("enable_accuracy_test", False),
        "tasks": {
            task_name: {
                "model": acc.get("model", "local-completions"),
                "model_args_extra": acc.get("model_args_extra", ""),
                "extra_kwargs": {},
            }
        },
    }


def apply_overrides(config, args, explicitly_set):
    """Apply CLI argument overrides to a loaded config.

    Only overrides fields for arguments that were explicitly provided on the
    command line (tracked via explicitly_set set).

    Args:
        config: The base config dict to modify in place.
        args: Parsed argparse namespace.
        explicitly_set: Set of argument dest names that were explicitly provided.
    """
    # --- metadata overrides ---
    if "model_name" in explicitly_set and args.model_name:
        config.setdefault("metadata", {})["model_name"] = args.model_name

    # --- benchmark overrides ---
    bm = config.setdefault("benchmark", {})
    bm_map = {
        "benchmark_mode": "mode",
        "input_length": "input_length",
        "output_length": "output_length",
        "concurrency_list": "concurrency_list",
        "multi_round": "multi_round",
        "streaming": "streaming",
        "dataset_file": "dataset_file",
        "benchmark_ratio": "benchmark_ratio",
    }
    for arg_name, config_key in bm_map.items():
        if arg_name in explicitly_set:
            bm[config_key] = getattr(args, arg_name)

    # Update benchmark_type in metadata if input/output lengths changed
    if "input_length" in explicitly_set or "output_length" in explicitly_set:
        isl = bm.get("input_length", args.input_length)
        osl = bm.get("output_length", args.output_length)
        config.setdefault("metadata", {})["benchmark_type"] = (
            f"{isl // 1000}k{osl // 1000}k" if isl >= 1000 and osl >= 1000 else f"{isl}-{osl}"
        )

    # --- hardware overrides ---
    hw = config.setdefault("hardware", {})
    hw_map = {
        "gpus_per_node": "gpus_per_node",
        "num_ctx_servers": "num_ctx_servers",
        "num_gen_servers": "num_gen_servers",
    }
    for arg_name, config_key in hw_map.items():
        if arg_name in explicitly_set:
            hw[config_key] = getattr(args, arg_name)

    # --- environment overrides ---
    env = config.setdefault("environment", {})
    if "worker_env_var" in explicitly_set:
        env["worker_env_var"] = args.worker_env_var
    if "server_env_var" in explicitly_set:
        env["server_env_var"] = args.server_env_var

    # --- worker_config gen overrides ---
    wc = config.setdefault("worker_config", {})
    gen = wc.setdefault("gen", {})
    gen_simple = {
        "gen_tp": "tensor_parallel_size",
        "gen_pp": "pipeline_parallel_size",
        "gen_cp": "context_parallel_size",
        "gen_max_batch_size": "max_batch_size",
        "gen_max_num_tokens": "max_num_tokens",
        "gen_enable_attention_dp": "enable_attention_dp",
        "stream_interval": "stream_interval",
        "num_postprocess_workers": "num_postprocess_workers",
    }
    for arg_name, config_key in gen_simple.items():
        if arg_name in explicitly_set:
            gen[config_key] = getattr(args, arg_name)

    if "gen_ep" in explicitly_set:
        if args.gen_ep > 0:
            gen["moe_expert_parallel_size"] = args.gen_ep
        else:
            gen.pop("moe_expert_parallel_size", None)

    if "gen_max_seq_len" in explicitly_set and args.gen_max_seq_len > 0:
        gen["max_seq_len"] = args.gen_max_seq_len

    if "gen_enable_lm_head_tp_in_adp" in explicitly_set:
        if args.gen_enable_lm_head_tp_in_adp:
            gen["enable_lm_head_tp_in_adp"] = True
        else:
            gen.pop("enable_lm_head_tp_in_adp", None)

    if "gen_disable_overlap_scheduler" in explicitly_set:
        if args.gen_disable_overlap_scheduler:
            gen["disable_overlap_scheduler"] = True
        else:
            gen.pop("disable_overlap_scheduler", None)

    if "trust_remote_code" in explicitly_set:
        if args.trust_remote_code:
            gen["trust_remote_code"] = True
        else:
            gen.pop("trust_remote_code", None)

    # Gen CUDA graph config
    if "gen_cuda_graph_padding" in explicitly_set:
        if args.gen_cuda_graph_padding:
            cuda_graph = gen.get("cuda_graph_config") or {}
            cuda_graph["enable_padding"] = True
            if "gen_cuda_graph_batch_sizes" in explicitly_set and args.gen_cuda_graph_batch_sizes:
                cuda_graph["batch_sizes"] = [
                    int(x.strip()) for x in args.gen_cuda_graph_batch_sizes.split(",")
                ]
            gen["cuda_graph_config"] = cuda_graph
        else:
            gen["cuda_graph_config"] = None

    # Gen KV cache config
    gen_kv = gen.setdefault("kv_cache_config", {})
    if "kv_cache_dtype" in explicitly_set:
        gen_kv["dtype"] = args.kv_cache_dtype
    if "gen_kv_free_fraction" in explicitly_set:
        gen_kv["free_gpu_memory_fraction"] = args.gen_kv_free_fraction
    if "kv_tokens_per_block" in explicitly_set and args.kv_tokens_per_block > 0:
        gen_kv["tokens_per_block"] = args.kv_tokens_per_block

    # Gen MoE config
    if "moe_backend" in explicitly_set:
        if args.moe_backend:
            moe = gen.get("moe_config") or {}
            moe["backend"] = args.moe_backend
            if "moe_low_precision_combine" in explicitly_set and args.moe_low_precision_combine:
                moe["use_low_precision_moe_combine"] = True
            if "eplb_num_slots" in explicitly_set and args.eplb_num_slots > 0:
                moe["load_balancer"] = {"num_slots": args.eplb_num_slots}
            gen["moe_config"] = moe
        else:
            gen.pop("moe_config", None)

    # Gen cache transceiver
    gen_ct = gen.setdefault("cache_transceiver_config", {})
    if "cache_backend" in explicitly_set:
        gen_ct["backend"] = args.cache_backend
    if "cache_max_tokens_in_buffer" in explicitly_set:
        gen_ct["max_tokens_in_buffer"] = args.cache_max_tokens_in_buffer

    # Gen speculative config
    if "mtp_layers" in explicitly_set:
        if args.mtp_layers > 0:
            gen["speculative_config"] = {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": args.mtp_layers,
            }
        else:
            gen.pop("speculative_config", None)

    # --- worker_config ctx overrides ---
    ctx = wc.setdefault("ctx", {})
    ctx_simple = {
        "ctx_tp": "tensor_parallel_size",
        "ctx_pp": "pipeline_parallel_size",
        "ctx_cp": "context_parallel_size",
        "ctx_max_batch_size": "max_batch_size",
        "ctx_max_num_tokens": "max_num_tokens",
        "ctx_enable_attention_dp": "enable_attention_dp",
    }
    for arg_name, config_key in ctx_simple.items():
        if arg_name in explicitly_set:
            ctx[config_key] = getattr(args, arg_name)

    if "ctx_ep" in explicitly_set:
        if args.ctx_ep > 0:
            ctx["moe_expert_parallel_size"] = args.ctx_ep
        else:
            ctx.pop("moe_expert_parallel_size", None)

    if "ctx_max_seq_len" in explicitly_set and args.ctx_max_seq_len > 0:
        ctx["max_seq_len"] = args.ctx_max_seq_len

    if "ctx_enable_lm_head_tp_in_adp" in explicitly_set:
        if args.ctx_enable_lm_head_tp_in_adp:
            ctx["enable_lm_head_tp_in_adp"] = True
        else:
            ctx.pop("enable_lm_head_tp_in_adp", None)

    if "trust_remote_code" in explicitly_set:
        if args.trust_remote_code:
            ctx["trust_remote_code"] = True
        else:
            ctx.pop("trust_remote_code", None)

    # Ctx KV cache config
    ctx_kv = ctx.setdefault("kv_cache_config", {})
    if "kv_cache_dtype" in explicitly_set:
        ctx_kv["dtype"] = args.kv_cache_dtype
    if "ctx_kv_free_fraction" in explicitly_set:
        ctx_kv["free_gpu_memory_fraction"] = args.ctx_kv_free_fraction
    if "kv_tokens_per_block" in explicitly_set and args.kv_tokens_per_block > 0:
        ctx_kv["tokens_per_block"] = args.kv_tokens_per_block

    # Ctx MoE config
    if "moe_backend" in explicitly_set:
        if args.moe_backend:
            moe = ctx.get("moe_config") or {}
            moe["backend"] = args.moe_backend
            ctx["moe_config"] = moe
        else:
            ctx.pop("moe_config", None)

    # Ctx cache transceiver
    ctx_ct = ctx.setdefault("cache_transceiver_config", {})
    if "cache_backend" in explicitly_set:
        ctx_ct["backend"] = args.cache_backend
    if "cache_max_tokens_in_buffer" in explicitly_set:
        ctx_ct["max_tokens_in_buffer"] = args.cache_max_tokens_in_buffer

    # Ctx speculative config
    if "mtp_layers" in explicitly_set:
        if args.mtp_layers > 0:
            ctx["speculative_config"] = {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": args.mtp_layers,
            }
        else:
            ctx.pop("speculative_config", None)

    # --- accuracy overrides ---
    if any(
        k in explicitly_set
        for k in (
            "enable_accuracy_test",
            "accuracy_task",
            "accuracy_model",
            "accuracy_model_args_extra",
            "accuracy_trust_remote_code",
        )
    ):
        acc = config.setdefault(
            "accuracy",
            {
                "enable_accuracy_test": False,
                "tasks": {},
            },
        )
        # Normalize to nested format first
        acc = _normalize_accuracy_section(acc)
        config["accuracy"] = acc

        if "enable_accuracy_test" in explicitly_set:
            acc["enable_accuracy_test"] = args.enable_accuracy_test

        # Build / update the task entry
        task_name = getattr(args, "accuracy_task", "gsm8k") or "gsm8k"
        tasks = acc.setdefault("tasks", {})
        task_cfg = tasks.setdefault(
            task_name,
            {
                "model": "local-completions",
                "model_args_extra": "",
                "extra_kwargs": {},
            },
        )
        if "accuracy_model" in explicitly_set:
            task_cfg["model"] = args.accuracy_model
        if "accuracy_model_args_extra" in explicitly_set:
            task_cfg["model_args_extra"] = args.accuracy_model_args_extra
        if "accuracy_trust_remote_code" in explicitly_set and args.accuracy_trust_remote_code:
            task_cfg.setdefault("extra_kwargs", {})["trust_remote_code"] = True

    return config


def _detect_config_type(config):
    """Detect if a loaded config is aggregated or disaggregated."""
    if "server_configs" in config:
        return "aggr"
    if "worker_config" in config:
        return "disagg"
    return None


def _apply_single_server_config_overrides(sc, args, explicitly_set):
    """Apply CLI overrides to a single server_config dict (aggregated mode)."""
    sc_map = {
        "tp": "tensor_parallel_size",
        "pp": "pipeline_parallel_size",
        "cp": "context_parallel_size",
        "max_batch_size": "max_batch_size",
        "max_num_tokens": "max_num_tokens",
        "attn_backend": "attn_backend",
        "enable_attention_dp": "enable_attention_dp",
        "num_postprocess_workers": "num_postprocess_workers",
        "stream_interval": "stream_interval",
    }
    for arg_name, config_key in sc_map.items():
        if arg_name in explicitly_set:
            sc[config_key] = getattr(args, arg_name)

    if "ep" in explicitly_set:
        if args.ep > 0:
            sc["moe_expert_parallel_size"] = args.ep
        else:
            sc.pop("moe_expert_parallel_size", None)

    if "max_seq_len" in explicitly_set and args.max_seq_len > 0:
        sc["max_seq_len"] = args.max_seq_len

    if "enable_adp_balance" in explicitly_set:
        if args.enable_adp_balance and sc.get("enable_attention_dp", False):
            sc["attention_dp_config"] = {"enable_balance": True}
        elif not args.enable_adp_balance:
            sc.pop("attention_dp_config", None)

    if "moe_backend" in explicitly_set:
        if args.moe_backend:
            moe = sc.get("moe_config") or {}
            moe["backend"] = args.moe_backend
            if "moe_low_precision_combine" in explicitly_set and args.moe_low_precision_combine:
                moe["use_low_precision_moe_combine"] = True
            if "eplb_num_slots" in explicitly_set and args.eplb_num_slots > 0:
                moe["load_balancer"] = {"num_slots": args.eplb_num_slots}
            sc["moe_config"] = moe
        else:
            sc.pop("moe_config", None)

    if "kv_cache_dtype" in explicitly_set:
        sc.setdefault("kv_cache_config", {})["dtype"] = args.kv_cache_dtype
    if "kv_free_fraction" in explicitly_set:
        sc.setdefault("kv_cache_config", {})["free_gpu_memory_fraction"] = args.kv_free_fraction
    if "kv_tokens_per_block" in explicitly_set and args.kv_tokens_per_block > 0:
        sc.setdefault("kv_cache_config", {})["tokens_per_block"] = args.kv_tokens_per_block

    if "trust_remote_code" in explicitly_set:
        if args.trust_remote_code:
            sc["trust_remote_code"] = True
        else:
            sc.pop("trust_remote_code", None)

    if "mtp_layers" in explicitly_set:
        if args.mtp_layers > 0:
            sc["speculative_config"] = {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": args.mtp_layers,
            }
        else:
            sc.pop("speculative_config", None)

    # Client config overrides (apply to all client_configs in this server)
    for cc in sc.get("client_configs", []):
        if "input_length" in explicitly_set:
            cc["isl"] = args.input_length
        if "output_length" in explicitly_set:
            cc["osl"] = args.output_length
        if "concurrency" in explicitly_set and args.concurrency > 0:
            cc["concurrency"] = args.concurrency
        if "iterations" in explicitly_set:
            cc["iterations"] = args.iterations
        if "dataset_file" in explicitly_set and args.dataset_file:
            cc["dataset_file"] = args.dataset_file


def apply_aggr_overrides(config, args, explicitly_set):
    """Apply CLI argument overrides to a loaded aggregated config.

    Applies scalar overrides to all server_configs unless --server-name
    is specified, in which case only the matching entry is updated.
    """
    if "model_name" in explicitly_set and args.model_name:
        config.setdefault("metadata", {})["model_name"] = args.model_name
    if "supported_gpus" in explicitly_set and args.supported_gpus:
        config.setdefault("metadata", {})["supported_gpus"] = [
            g.strip() for g in args.supported_gpus.split(",")
        ]

    hw = config.setdefault("hardware", {})
    if "gpus_per_node" in explicitly_set:
        hw["gpus_per_node"] = args.gpus_per_node

    server_configs = config.get("server_configs", [])
    targets = server_configs
    if "server_name" in explicitly_set and args.server_name:
        targets = [sc for sc in server_configs if sc.get("name") == args.server_name]
        if not targets:
            print(
                f"Warning: no server_config named '{args.server_name}' found; "
                "applying overrides to all server_configs",
                file=sys.stderr,
            )
            targets = server_configs

    for sc in targets:
        _apply_single_server_config_overrides(sc, args, explicitly_set)

    return config


def build_server_config(args):
    """Build a server_config entry dict for aggregated mode."""
    isl = args.input_length
    osl = args.output_length

    # Auto-derive server name if not provided
    server_name = args.server_name
    if not server_name:
        parts = [args.model_name, f"tp{args.tp}"]
        if args.ep > 0:
            parts.append(f"ep{args.ep}")
        if args.pp > 1:
            parts.append(f"pp{args.pp}")
        if args.mtp_layers > 0:
            parts.append(f"mtp{args.mtp_layers}")
        server_name = "_".join(parts)

    sc = {
        "name": server_name,
        "model_name": args.model_name,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "max_batch_size": args.max_batch_size,
        "max_num_tokens": args.max_num_tokens,
        "attn_backend": args.attn_backend,
        "enable_attention_dp": args.enable_attention_dp,
    }

    if args.cp > 1:
        sc["context_parallel_size"] = args.cp
    if args.ep > 0:
        sc["moe_expert_parallel_size"] = args.ep
    if args.max_seq_len > 0:
        sc["max_seq_len"] = args.max_seq_len

    if args.enable_attention_dp and args.enable_adp_balance:
        sc["attention_dp_config"] = {"enable_balance": True}

    if args.moe_backend:
        moe = {"backend": args.moe_backend}
        if args.moe_low_precision_combine:
            moe["use_low_precision_moe_combine"] = True
        if args.eplb_num_slots > 0:
            moe["load_balancer"] = {"num_slots": args.eplb_num_slots}
        sc["moe_config"] = moe

    if args.gen_cuda_graph_padding:
        cuda_graph = {"enable_padding": True, "max_batch_size": args.max_batch_size}
        if args.gen_cuda_graph_batch_sizes:
            cuda_graph["batch_sizes"] = [
                int(x.strip()) for x in args.gen_cuda_graph_batch_sizes.split(",")
            ]
        sc["cuda_graph_config"] = cuda_graph

    kv_cache = {
        "dtype": args.kv_cache_dtype,
        "enable_block_reuse": False,
        "free_gpu_memory_fraction": args.kv_free_fraction,
    }
    if args.kv_tokens_per_block > 0:
        kv_cache["tokens_per_block"] = args.kv_tokens_per_block
    sc["kv_cache_config"] = kv_cache

    if args.mtp_layers > 0:
        sc["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp_layers,
        }

    if args.trust_remote_code:
        sc["trust_remote_code"] = True

    sc["num_postprocess_workers"] = args.num_postprocess_workers
    sc["stream_interval"] = args.stream_interval

    # Auto-derive client config name
    client_name = args.client_name
    if not client_name:
        isl_str = f"{isl // 1000}k" if isl >= 1000 else str(isl)
        osl_str = f"{osl // 1000}k" if osl >= 1000 else str(osl)
        client_name = f"con{args.concurrency}_iter{args.iterations}_{isl_str}{osl_str}"

    client = {
        "name": client_name,
        "concurrency": args.concurrency,
        "iterations": args.iterations,
        "isl": isl,
        "osl": osl,
        "backend": args.client_backend,
    }
    if args.dataset_file:
        client["dataset_file"] = args.dataset_file

    sc["client_configs"] = [client]
    return sc


def build_aggr_config(args):
    """Build a complete aggregated benchmark config dict."""
    config = {
        "metadata": {
            "model_name": args.model_name,
        },
        "hardware": {
            "gpus_per_node": args.gpus_per_node,
        },
        "server_configs": [build_server_config(args)],
    }

    if args.supported_gpus:
        config["metadata"]["supported_gpus"] = [g.strip() for g in args.supported_gpus.split(",")]

    return config


def build_gen_worker_config(args):
    """Build generation worker configuration dict."""
    gen = {
        "tensor_parallel_size": args.gen_tp,
        "pipeline_parallel_size": args.gen_pp,
        "context_parallel_size": args.gen_cp,
        "max_batch_size": args.gen_max_batch_size,
        "max_num_tokens": args.gen_max_num_tokens,
        "enable_attention_dp": args.gen_enable_attention_dp,
        "print_iter_log": True,
        "stream_interval": args.stream_interval,
        "num_postprocess_workers": args.num_postprocess_workers,
    }

    if args.gen_ep > 0:
        gen["moe_expert_parallel_size"] = args.gen_ep
    if args.gen_max_seq_len > 0:
        gen["max_seq_len"] = args.gen_max_seq_len
    if args.gen_enable_lm_head_tp_in_adp:
        gen["enable_lm_head_tp_in_adp"] = True
    if args.gen_disable_overlap_scheduler:
        gen["disable_overlap_scheduler"] = True
    if args.trust_remote_code:
        gen["trust_remote_code"] = True

    # CUDA graph config
    if args.gen_cuda_graph_padding:
        cuda_graph = {"enable_padding": True}
        if args.gen_cuda_graph_batch_sizes:
            cuda_graph["batch_sizes"] = [
                int(x.strip()) for x in args.gen_cuda_graph_batch_sizes.split(",")
            ]
        gen["cuda_graph_config"] = cuda_graph
    else:
        gen["cuda_graph_config"] = None

    # KV cache config
    kv_cache = {
        "enable_block_reuse": False,
        "free_gpu_memory_fraction": args.gen_kv_free_fraction,
        "dtype": args.kv_cache_dtype,
    }
    if args.kv_tokens_per_block > 0:
        kv_cache["tokens_per_block"] = args.kv_tokens_per_block
    gen["kv_cache_config"] = kv_cache

    # MoE config
    if args.moe_backend:
        moe = {"backend": args.moe_backend}
        if args.moe_low_precision_combine:
            moe["use_low_precision_moe_combine"] = True
        if args.eplb_num_slots > 0:
            moe["load_balancer"] = {"num_slots": args.eplb_num_slots}
        gen["moe_config"] = moe

    # Cache transceiver config
    gen["cache_transceiver_config"] = {
        "max_tokens_in_buffer": args.cache_max_tokens_in_buffer,
        "backend": args.cache_backend,
    }

    # Speculative config
    if args.mtp_layers > 0:
        gen["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp_layers,
        }

    return gen


def build_ctx_worker_config(args):
    """Build context worker configuration dict."""
    ctx = {
        "tensor_parallel_size": args.ctx_tp,
        "pipeline_parallel_size": args.ctx_pp,
        "context_parallel_size": args.ctx_cp,
        "max_batch_size": args.ctx_max_batch_size,
        "max_num_tokens": args.ctx_max_num_tokens,
        "enable_attention_dp": args.ctx_enable_attention_dp,
        "print_iter_log": True,
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
    }

    if args.ctx_ep > 0:
        ctx["moe_expert_parallel_size"] = args.ctx_ep
    if args.ctx_max_seq_len > 0:
        ctx["max_seq_len"] = args.ctx_max_seq_len
    if args.ctx_enable_lm_head_tp_in_adp:
        ctx["enable_lm_head_tp_in_adp"] = True
    if args.trust_remote_code:
        ctx["trust_remote_code"] = True

    # KV cache config
    kv_cache = {
        "enable_block_reuse": False,
        "free_gpu_memory_fraction": args.ctx_kv_free_fraction,
        "dtype": args.kv_cache_dtype,
    }
    if args.kv_tokens_per_block > 0:
        kv_cache["tokens_per_block"] = args.kv_tokens_per_block
    ctx["kv_cache_config"] = kv_cache

    # MoE config (ctx may use different backend)
    if args.moe_backend:
        moe = {"backend": args.moe_backend}
        ctx["moe_config"] = moe

    # Cache transceiver config
    ctx["cache_transceiver_config"] = {
        "max_tokens_in_buffer": args.cache_max_tokens_in_buffer,
        "backend": args.cache_backend,
    }

    # Speculative config (same as gen if set)
    if args.mtp_layers > 0:
        ctx["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp_layers,
        }

    return ctx


def build_config(args):
    """Build the complete config dict."""
    isl = args.input_length
    osl = args.output_length

    config = {
        "metadata": {
            "model_name": args.model_name,
            "benchmark_type": f"{isl // 1000}k{osl // 1000}k"
            if isl >= 1000 and osl >= 1000
            else f"{isl}-{osl}",
        },
        "benchmark": {
            "mode": args.benchmark_mode,
            "use_nv_sa_benchmark": False,
            "multi_round": args.multi_round,
            "benchmark_ratio": args.benchmark_ratio,
            "streaming": args.streaming,
            "concurrency_list": args.concurrency_list,
            "input_length": isl,
            "output_length": osl,
        },
        "hardware": {
            "gpus_per_node": args.gpus_per_node,
            "num_ctx_servers": args.num_ctx_servers,
            "num_gen_servers": args.num_gen_servers,
        },
        "environment": {
            "worker_env_var": args.worker_env_var,
            "server_env_var": args.server_env_var,
        },
        "profiling": {
            "nsys_on": False,
        },
        "accuracy": {
            "enable_accuracy_test": args.enable_accuracy_test,
            "tasks": {
                args.accuracy_task: {
                    "model": args.accuracy_model,
                    "model_args_extra": args.accuracy_model_args_extra,
                    "extra_kwargs": {
                        **({"trust_remote_code": True} if args.accuracy_trust_remote_code else {}),
                    },
                }
            }
            if args.enable_accuracy_test
            else {},
        },
        "worker_config": {
            "gen": build_gen_worker_config(args),
            "ctx": build_ctx_worker_config(args),
        },
    }

    if args.dataset_file:
        config["benchmark"]["dataset_file"] = args.dataset_file

    return config


def get_explicitly_set_args(parser, argv=None):
    """Determine which arguments were explicitly provided on the command line.

    Parses argv twice: once normally, once with all defaults suppressed.
    Arguments present in the suppressed parse were explicitly provided.

    Returns:
        tuple: (args, explicitly_set) where explicitly_set is a set of dest names.
    """
    args = parser.parse_args(argv)

    # Parse again with defaults suppressed to find what was explicitly set
    parser_no_defaults = argparse.ArgumentParser(parents=[], add_help=False)
    for action in parser._actions:
        if action.dest == "help":
            continue
        kwargs = {
            "dest": action.dest,
            "default": argparse.SUPPRESS,
        }
        if isinstance(action, argparse._StoreTrueAction):
            kwargs["action"] = "store_true"
        elif isinstance(action, argparse._StoreFalseAction):
            kwargs["action"] = "store_false"
        elif isinstance(action, argparse._StoreAction):
            kwargs["type"] = action.type
            kwargs["nargs"] = action.nargs
            if action.choices:
                kwargs["choices"] = action.choices

        # Use the same option strings
        if action.option_strings:
            parser_no_defaults.add_argument(*action.option_strings, **kwargs)
        elif action.dest != "help":
            kwargs.pop("dest", None)
            parser_no_defaults.add_argument(action.dest, **kwargs)

    ns_explicit, _ = parser_no_defaults.parse_known_args(argv)
    explicitly_set = set(vars(ns_explicit).keys())

    return args, explicitly_set


def main():
    p = argparse.ArgumentParser(
        description="Generate aggregated or disaggregated benchmark config YAML for perf-sanity submission"
    )

    p.add_argument(
        "--model-name",
        default=None,
        help="Model identifier (must match MODEL_PATH_DICT in test_perf_sanity.py)",
    )
    p.add_argument("--output", "-o", required=True, help="Output YAML file path")
    p.add_argument(
        "--from-config",
        default=None,
        help="Path to an existing benchmark config YAML to use as base. "
        "Config type is auto-detected (server_configs → aggr, worker_config → disagg).",
    )
    p.add_argument(
        "--config-type",
        default="disagg",
        choices=["aggr", "disagg"],
        help="Config type to generate from scratch: 'aggr' (aggregated) or "
        "'disagg' (disaggregated, default). Ignored when --from-config auto-detects the type.",
    )

    # ── Aggregated server config ──────────────────────────────────────────────
    aggr = p.add_argument_group("aggregated server config (--config-type aggr)")
    aggr.add_argument("--tp", type=int, default=8, help="Tensor parallel size (aggr server)")
    aggr.add_argument("--pp", type=int, default=1, help="Pipeline parallel size (aggr server)")
    aggr.add_argument(
        "--ep", type=int, default=0, help="Expert (MoE) parallel size (aggr server; 0 = disabled)"
    )
    aggr.add_argument(
        "--cp", type=int, default=1, help="Context parallel size (aggr server; omitted when 1)"
    )
    aggr.add_argument(
        "--max-batch-size", type=int, default=256, help="Server max_batch_size (aggr)"
    )
    aggr.add_argument(
        "--max-num-tokens", type=int, default=20000, help="Server max_num_tokens (aggr)"
    )
    aggr.add_argument(
        "--max-seq-len", type=int, default=0, help="Server max_seq_len (aggr; 0 = omit)"
    )
    aggr.add_argument(
        "--attn-backend", default="TRTLLM", help="Attention backend (aggr; default: TRTLLM)"
    )
    aggr.add_argument(
        "--enable-attention-dp",
        action="store_true",
        default=False,
        help="Enable attention data parallelism (aggr)",
    )
    aggr.add_argument(
        "--enable-adp-balance",
        action="store_true",
        default=False,
        help="Set attention_dp_config.enable_balance=true (aggr, requires --enable-attention-dp)",
    )
    aggr.add_argument(
        "--server-name",
        default="",
        help="Name for the server_config entry (auto-derived if empty). "
        "When used with --from-config, limits overrides to this named entry.",
    )
    aggr.add_argument(
        "--supported-gpus",
        default="",
        help="Comma-separated supported GPU types, e.g. 'B200,GB200' (aggr metadata)",
    )
    aggr.add_argument(
        "--kv-free-fraction",
        type=float,
        default=0.8,
        help="KV cache free GPU memory fraction (aggr)",
    )

    # ── Aggregated client config ───────────────────────────────────────────────
    client = p.add_argument_group("aggregated client config (--config-type aggr)")
    client.add_argument(
        "--concurrency",
        type=int,
        default=256,
        help="Benchmark concurrency for client config (aggr)",
    )
    client.add_argument(
        "--iterations", type=int, default=10, help="Benchmark iterations for client config (aggr)"
    )
    client.add_argument(
        "--client-backend", default="openai", help="Client backend (aggr; default: openai)"
    )
    client.add_argument(
        "--client-name", default="", help="Name for the client_config entry (auto-derived if empty)"
    )

    # ── Disaggregated / shared ────────────────────────────────────────────────
    p.add_argument(
        "--benchmark-mode",
        default="e2e",
        choices=["e2e"],
        help="Benchmark mode. Only 'e2e' is supported for benchmark tests.",
    )
    p.add_argument("--input-length", type=int, default=1024)
    p.add_argument("--output-length", type=int, default=1024)
    p.add_argument("--concurrency-list", default="16")
    p.add_argument("--multi-round", type=int, default=8)
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--no-streaming", dest="streaming", action="store_false")
    p.add_argument("--dataset-file", default="")
    p.add_argument("--benchmark-ratio", type=float, default=0.8)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--num-ctx-servers", type=int, default=1)
    p.add_argument("--num-gen-servers", type=int, default=1)
    p.add_argument("--gen-tp", type=int, default=8)
    p.add_argument("--gen-pp", type=int, default=1)
    p.add_argument("--gen-cp", type=int, default=1)
    p.add_argument("--gen-ep", type=int, default=0)
    p.add_argument("--gen-max-batch-size", type=int, default=256)
    p.add_argument("--gen-max-num-tokens", type=int, default=512)
    p.add_argument("--gen-max-seq-len", type=int, default=0)
    p.add_argument("--gen-enable-attention-dp", action="store_true", default=False)
    p.add_argument("--gen-enable-lm-head-tp-in-adp", action="store_true", default=False)
    p.add_argument("--gen-disable-overlap-scheduler", action="store_true", default=False)
    p.add_argument("--ctx-tp", type=int, default=4)
    p.add_argument("--ctx-pp", type=int, default=1)
    p.add_argument("--ctx-cp", type=int, default=1)
    p.add_argument("--ctx-ep", type=int, default=0)
    p.add_argument("--ctx-max-batch-size", type=int, default=4)
    p.add_argument("--ctx-max-num-tokens", type=int, default=4608)
    p.add_argument("--ctx-max-seq-len", type=int, default=0)
    p.add_argument("--ctx-enable-attention-dp", action="store_true", default=True)
    p.add_argument("--ctx-no-attention-dp", dest="ctx_enable_attention_dp", action="store_false")
    p.add_argument("--ctx-enable-lm-head-tp-in-adp", action="store_true", default=False)
    p.add_argument("--kv-cache-dtype", default="fp8")
    p.add_argument("--gen-kv-free-fraction", type=float, default=0.8)
    p.add_argument("--ctx-kv-free-fraction", type=float, default=0.85)
    p.add_argument("--kv-tokens-per-block", type=int, default=0)
    p.add_argument("--cache-backend", default="DEFAULT", choices=["DEFAULT", "UCX", "NIXL"])
    p.add_argument("--cache-max-tokens-in-buffer", type=int, default=4608)
    p.add_argument("--moe-backend", default="")
    p.add_argument("--moe-low-precision-combine", action="store_true", default=False)
    p.add_argument("--eplb-num-slots", type=int, default=0)
    p.add_argument("--mtp-layers", type=int, default=0)
    p.add_argument("--gen-cuda-graph-padding", action="store_true", default=True)
    p.add_argument(
        "--gen-no-cuda-graph-padding", dest="gen_cuda_graph_padding", action="store_false"
    )
    p.add_argument("--gen-cuda-graph-batch-sizes", default="")
    p.add_argument(
        "--worker-env-var",
        default="TLLM_LOG_LEVEL=INFO TRTLLM_SERVER_DISABLE_GC=1 TRTLLM_WORKER_DISABLE_GC=1 TRTLLM_ENABLE_PDL=1",
    )
    p.add_argument("--server-env-var", default="TRTLLM_SERVER_DISABLE_GC=1")
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--stream-interval", type=int, default=20)
    p.add_argument("--num-postprocess-workers", type=int, default=4)
    # Accuracy testing (mirrors examples/disaggregated/slurm/benchmark/config.yaml structure)
    p.add_argument(
        "--enable-accuracy-test",
        action="store_true",
        default=False,
        help="Enable accuracy evaluation after benchmark",
    )
    p.add_argument("--accuracy-task", default="gsm8k", help="lm_eval task name (default: gsm8k)")
    p.add_argument(
        "--accuracy-model",
        default="local-completions",
        choices=["local-completions", "local-chat-completions"],
        help="lm_eval model type (default: local-completions)",
    )
    p.add_argument(
        "--accuracy-model-args-extra",
        default=(
            "num_concurrent=512,max_retries=3,tokenized_requests=false,"
            "timeout=7200,max_gen_toks=16384"
        ),
        help="Extra model_args passed to lm_eval",
    )
    p.add_argument(
        "--accuracy-trust-remote-code",
        action="store_true",
        default=False,
        help="Pass --trust_remote_code to lm_eval",
    )

    args, explicitly_set = get_explicitly_set_args(p)

    if args.from_config:
        if not os.path.exists(args.from_config):
            print(f"Error: config file not found: {args.from_config}", file=sys.stderr)
            sys.exit(1)

        config = load_and_normalize_config(args.from_config)
        # Auto-detect type from loaded config; fall back to --config-type flag
        detected = _detect_config_type(config)
        config_type = detected or args.config_type
        if detected:
            print(f"Auto-detected config type: {config_type}")

        explicitly_set.discard("output")
        explicitly_set.discard("from_config")
        explicitly_set.discard("config_type")

        if config_type == "aggr":
            apply_aggr_overrides(config, args, explicitly_set)
        else:
            apply_overrides(config, args, explicitly_set)

        print(f"Loaded base config from: {args.from_config}")
        if explicitly_set:
            print(f"Applied overrides for: {', '.join(sorted(explicitly_set))}")
    else:
        if not args.model_name:
            print(
                "Error: --model-name is required when generating from scratch "
                "(not using --from-config)",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.config_type == "aggr":
            config = build_aggr_config(args)
        else:
            config = build_config(args)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config written to: {args.output}")


if __name__ == "__main__":
    main()
