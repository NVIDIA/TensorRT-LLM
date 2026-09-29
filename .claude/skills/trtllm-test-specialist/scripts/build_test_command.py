# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
r"""Build CLI commands for trtllm test workflows.

Supports three command types:
  bench        -- trtllm-bench prepare-dataset + throughput/latency steps
  eval         -- trtllm-eval accuracy evaluation
  perf_sanity  -- pytest test_perf_sanity.py with aggregated or disaggregated test ID

Non-perf pytest commands, python scripts, and custom commands do not require
this script — pass them directly to trtllm-case-executor as test_cmd / custom_cmd.

Prints the fully-constructed command string to stdout (suitable for passing
to trtllm-case-executor).  Exit code is non-zero on error.

Examples:
--------
# trtllm-bench single GPU:
python3 build_test_command.py --type bench \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-path /models/Llama-3.1-8B-Instruct \
  --tp 1 --work-dir /tmp/work

# trtllm-bench multi-GPU:
python3 build_test_command.py --type bench \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-path /models/Llama-3.1-8B-Instruct \
  --tp 4 --backend pytorch \
  --num-requests 512 --input-mean 1024 --output-mean 1024 \
  --concurrency 32 --work-dir /tmp/work

# trtllm-eval single GPU:
python3 build_test_command.py --type eval \
  --model /models/Llama-3.1-8B-Instruct \
  --tp-size 1 --tasks gsm8k

# trtllm-eval multi-GPU:
python3 build_test_command.py --type eval \
  --model /models/Llama-3.1-8B-Instruct \
  --tp-size 4 --tasks gsm8k,mmlu

# perf_sanity aggregated — config name and test name from args:
python3 build_test_command.py --type perf_sanity \
  --serving-type aggr \
  --config-name gpt_oss \
  --test-name gpt_oss_tp4ep4pp2_mtp1 \
  --repo-root /path/to/TensorRT-LLM

# perf_sanity aggregated — config name derived from a generated YAML:
python3 build_test_command.py --type perf_sanity \
  --serving-type aggr \
  --config-file /work/gpt_oss.yaml \
  --test-name gpt_oss_tp4ep4pp2_mtp1 \
  --repo-root /path/to/TensorRT-LLM

# List available test names inside a config YAML:
python3 build_test_command.py --type perf_sanity \
  --config-file /work/gpt_oss.yaml \
  --list-test-names

# perf_sanity disaggregated:
python3 build_test_command.py --type perf_sanity \
  --serving-type disagg --benchmark-mode e2e \
  --config-name deepseek_r1 \
  --repo-root /path/to/TensorRT-LLM

Multi-node benchmark/evaluation workflow
-----------------------------------------
For workloads that exceed one node, generate a perf-sanity config YAML first
(using generate_benchmark_config.py from this skill),
then build the test command from it:

  # 1. Generate the config (aggr, benchmark):
  python3 <skill_dir>/scripts/generate_benchmark_config.py \\
    --config-type aggr --model-name gpt_oss_120b_fp4 \\
    --tp 4 --ep 4 --pp 2 --mtp-layers 1 \\
    --output /work/gpt_oss_120b_fp4.yaml

  # 2. List available server config names:
  python3 build_test_command.py --type perf_sanity \\
    --config-file /work/gpt_oss_120b_fp4.yaml --list-test-names

  # 3. Build the pytest command:
  python3 build_test_command.py --type perf_sanity \\
    --config-file /work/gpt_oss_120b_fp4.yaml \\
    --test-name gpt_oss_120b_fp4_tp4ep4pp2_mtp1 \\
    --repo-root /path/to/TensorRT-LLM

  # For evaluation (accuracy), add --enable-accuracy-test when generating:
  python3 <skill_dir>/scripts/generate_benchmark_config.py \\
    --config-type aggr --model-name gpt_oss_120b_fp4 \\
    --tp 4 --ep 4 --pp 2 \\
    --enable-accuracy-test --accuracy-task gsm8k \\
    --output /work/gpt_oss_120b_fp4.yaml
"""

import argparse
import os
import re
import shlex
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(s):
    """Shell-quote a string only when it contains characters that need it."""
    if re.search(r"[^\w@%+=:,./-]", s):
        return shlex.quote(s)
    return s


def _join(*parts):
    return " ".join(_q(str(p)) for p in parts if p is not None)


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def build_bench(args):
    """Return 'prepare-dataset && <subcommand>' command string."""
    if not args.model:
        sys.exit("--model is required for --type bench")

    work_dir = args.work_dir or "."
    dataset_path = args.dataset or os.path.join(work_dir, "bench_dataset.txt")
    report_json = args.report_json or os.path.join(work_dir, "bench_report.json")

    num_requests = args.num_requests if args.num_requests is not None else 100
    input_mean = args.input_mean if args.input_mean is not None else 128
    input_stdev = args.input_stdev if args.input_stdev is not None else 0
    output_mean = args.output_mean if args.output_mean is not None else 128
    output_stdev = args.output_stdev if args.output_stdev is not None else 0

    # --- prepare-dataset step ---
    prep = ["trtllm-bench", "--model", args.model]
    if args.workspace:
        prep += ["--workspace", args.workspace]
    if args.model_path:
        prep += ["--model_path", args.model_path]
    prep += [
        "prepare-dataset",
        "--output",
        dataset_path,
        "token-norm-dist",
        "--num-requests",
        str(num_requests),
        "--input-mean",
        str(input_mean),
        "--input-stdev",
        str(input_stdev),
        "--output-mean",
        str(output_mean),
        "--output-stdev",
        str(output_stdev),
    ]

    # --- benchmark run step ---
    subcommand = args.subcommand or "throughput"
    backend = args.backend or "pytorch"
    tp = args.tp  # argparse default: 1
    pp = args.pp  # argparse default: 1
    ep = args.ep  # argparse default: 0

    run = ["trtllm-bench", "--model", args.model]
    if args.workspace:
        run += ["--workspace", args.workspace]
    if args.model_path:
        run += ["--model_path", args.model_path]
    run += [subcommand, "--backend", backend, "--dataset", dataset_path]
    run += ["--tp", str(tp)]
    if pp > 1:
        run += ["--pp", str(pp)]
    if ep > 0:
        run += ["--ep", str(ep)]
    if args.cluster_size is not None:
        run += ["--cluster_size", str(args.cluster_size)]
    if args.max_seq_len is not None:
        run += ["--max_seq_len", str(args.max_seq_len)]
    if args.kv_cache_free_gpu_mem_fraction is not None:
        run += ["--kv_cache_free_gpu_mem_fraction", str(args.kv_cache_free_gpu_mem_fraction)]
    if args.beam_width is not None:
        run += ["--beam_width", str(args.beam_width)]
    if args.warmup is not None:
        run += ["--warmup", str(args.warmup)]
    if args.concurrency is not None:
        run += ["--concurrency", str(args.concurrency)]
    if args.max_batch_size is not None:
        run += ["--max_batch_size", str(args.max_batch_size)]
    if args.max_num_tokens is not None:
        run += ["--max_num_tokens", str(args.max_num_tokens)]
    if args.config:
        run += ["--config", args.config]
    if subcommand == "throughput":
        if args.streaming:
            run += ["--streaming"]
        if args.no_chunked_context:
            run += ["--disable_chunked_context"]
        if args.scheduler_policy:
            run += ["--scheduler_policy", args.scheduler_policy]
    run += ["--report_json", report_json]
    if args.iteration_log:
        run += ["--iteration_log", args.iteration_log]

    world_size = tp * pp
    prep_cmd = _join(*prep)
    run_cmd = _join(*run) if world_size == 1 else _join("trtllm-llmapi-launch", *run)
    return f"{prep_cmd} && {run_cmd}"


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def build_eval(args):
    """Return trtllm-eval command string(s). One invocation per task, joined with &&.

    trtllm-eval is a Click group without chain=True, so multiple tasks cannot be
    chained in a single invocation. Each task gets its own command.

    Option placement:
      - Global options go before the task subcommand name.
      - Common per-task options (num_samples, apply_chat_template, etc.) are
        appended after each task name.
      - mmlu-specific options (check_accuracy, accuracy_threshold, num_fewshot)
        are appended only to the mmlu task invocation.
    """
    if not args.model:
        sys.exit("--model is required for --type eval")
    if not args.tasks:
        sys.exit("--tasks is required for --type eval")

    tp_size = args.tp_size  # argparse default: 1
    pp_size = args.pp_size  # argparse default: 1
    ep_size = args.ep_size  # argparse default: 0

    # --- Global options (placed before the task subcommand name) ---
    # trtllm-eval's --model accepts an HF ID, a local checkpoint path, or a
    # TensorRT engine path; the tokenizer is loaded from whatever is passed
    # here (unless --tokenizer overrides it). When a local checkpoint is
    # available via --model-path, prefer it over a short label for --model so
    # the tokenizer resolves locally instead of being treated as an HF ID
    # (which fails for short names like "gpt_oss_120b").
    model_arg = args.model_path if args.model_path else args.model
    global_parts = ["trtllm-eval", "--model", model_arg]
    global_parts += ["--tp_size", str(tp_size)]
    if pp_size > 1:
        global_parts += ["--pp_size", str(pp_size)]
    if ep_size > 0:
        global_parts += ["--ep_size", str(ep_size)]
    if args.tokenizer:
        global_parts += ["--tokenizer", args.tokenizer]
    if args.backend and args.backend != "pytorch":
        global_parts += ["--backend", args.backend]
    if args.trust_remote_code:
        global_parts += ["--trust_remote_code"]
    if args.disable_kv_cache_reuse:
        global_parts += ["--disable_kv_cache_reuse"]
    if args.eval_kv_cache_free_gpu_memory_fraction is not None:
        global_parts += [
            "--kv_cache_free_gpu_memory_fraction",
            str(args.eval_kv_cache_free_gpu_memory_fraction),
        ]
    if args.max_batch_size is not None:
        global_parts += ["--max_batch_size", str(args.max_batch_size)]
    if args.max_num_tokens is not None:
        global_parts += ["--max_num_tokens", str(args.max_num_tokens)]
    if args.max_seq_len is not None:
        global_parts += ["--max_seq_len", str(args.max_seq_len)]
    if args.config:
        global_parts += ["--config", args.config]
    if args.extra_eval_flags:
        global_parts += args.extra_eval_flags.split()

    # --- Common per-task options (appended after each task name) ---
    common_task_parts = []
    if args.num_samples is not None:
        common_task_parts += ["--num_samples", str(args.num_samples)]
    if args.apply_chat_template:
        common_task_parts += ["--apply_chat_template"]
    if args.chat_template_kwargs:
        common_task_parts += ["--chat_template_kwargs", args.chat_template_kwargs]
    if args.system_prompt:
        common_task_parts += ["--system_prompt", args.system_prompt]
    if args.max_input_length is not None:
        common_task_parts += ["--max_input_length", str(args.max_input_length)]
    if args.max_output_length is not None:
        common_task_parts += ["--max_output_length", str(args.max_output_length)]
    if args.log_samples:
        common_task_parts += ["--log_samples"]
    if args.output_path:
        common_task_parts += ["--output_path", args.output_path]

    # --- mmlu-specific options (only appended to the mmlu task invocation) ---
    mmlu_parts = []
    if args.check_accuracy:
        mmlu_parts += ["--check_accuracy"]
    if args.accuracy_threshold is not None:
        mmlu_parts += ["--accuracy_threshold", str(args.accuracy_threshold)]
    if args.num_fewshot is not None:
        mmlu_parts += ["--num_fewshot", str(args.num_fewshot)]

    world_size = tp_size * pp_size
    launcher = ["trtllm-llmapi-launch"] if world_size > 1 else []

    task_cmds = []
    for task in args.tasks.split(","):
        task = task.strip()
        if not task:
            continue
        task_parts = list(common_task_parts)
        if task == "mmlu":
            task_parts += mmlu_parts
        full_parts = launcher + global_parts + [task] + task_parts
        task_cmds.append(_join(*full_parts))

    if not task_cmds:
        sys.exit("--tasks produced no valid task names")
    return " && ".join(task_cmds)


# ---------------------------------------------------------------------------
# perf_sanity
# ---------------------------------------------------------------------------


def _load_yaml_config(path):
    """Load a YAML file, returning the parsed dict. Exits on error."""
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except Exception as e:
        sys.exit(f"Failed to read {path}: {e}")


def _aggr_server_names(config):
    """Return list of server_config names from an aggregated config dict."""
    return [sc.get("name", "") for sc in config.get("server_configs", [])]


def build_perf_sanity(args):
    """Return pytest command string for test_perf_sanity.py."""
    repo_root = args.repo_root or "."
    test_file = os.path.join(repo_root, "tests/integration/defs/perf/test_perf_sanity.py")

    # Resolve config_name: prefer explicit --config-name; fall back to YAML filename.
    config_file_data = None
    if args.config_file:
        config_file_data = _load_yaml_config(args.config_file)
        basename = os.path.basename(args.config_file)
        derived_name = re.sub(r"\.(yaml|yml)$", "", basename)
        config_name = args.config_name or derived_name
    else:
        config_name = args.config_name

    if not config_name:
        sys.exit(
            "--config-name (or --config-file whose basename is used as the name) "
            "is required for --type perf_sanity"
        )

    # --list-test-names: print available server_config names and exit.
    if args.list_test_names:
        if config_file_data is None:
            sys.exit("--config-file is required with --list-test-names")
        names = _aggr_server_names(config_file_data)
        if not names:
            print("No server_configs found in the config file.", file=sys.stderr)
            sys.exit(1)
        for name in names:
            print(name)
        return None  # signal to main() not to print a command

    serving_type = args.serving_type or "aggr"

    # Build the pytest test ID from serving type and config
    if serving_type == "aggr":
        test_name = args.test_name
        if not test_name:
            # If config file is available, suggest valid names.
            hint = ""
            if config_file_data is not None:
                names = _aggr_server_names(config_file_data)
                if names:
                    hint = f"  Available names: {', '.join(names)}"
            sys.exit(f"--test-name is required for --serving-type aggr.{hint}")
        test_id = f"test_e2e[aggr-{config_name}-{test_name}]"
    else:
        # disaggregated: mode determines the prefix segment
        mode = args.benchmark_mode or "e2e"
        if mode == "ctx_only":
            test_id = f"test_e2e[aggr-ctx_only-{config_name}]"
        else:
            test_id = f"test_e2e[disagg-{mode}-{config_name}]"

    pytest_flags = args.pytest_flags or "-v"
    parts = ["pytest", _q(test_file), pytest_flags]
    parts.append(f'"{test_id}"')
    if args.output_dir:
        parts += ["--output-dir", _q(args.output_dir)]
    if args.extra_pytest_flags:
        parts += args.extra_pytest_flags.split()
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--type",
        required=True,
        choices=["bench", "eval", "perf_sanity"],
        dest="cmd_type",
        help="Command type to build",
    )

    # ---- Common ----
    g_common = p.add_argument_group("Common")
    g_common.add_argument("--model", help="HuggingFace model ID or local path (bench/eval)")
    g_common.add_argument(
        "--model-path",
        dest="model_path",
        help="Local checkpoint path. For bench, passed as --model_path. "
        "For eval, used as --model when set so the tokenizer loads "
        "from the local checkpoint instead of being treated as an HF ID.",
    )
    g_common.add_argument(
        "--work-dir", dest="work_dir", help="Work directory for default output paths"
    )
    g_common.add_argument(
        "--repo-root", dest="repo_root", help="TensorRT-LLM repo root (perf_sanity/pytest)"
    )
    g_common.add_argument("--output-dir", dest="output_dir", help="Pytest --output-dir")

    # ---- bench ----
    g_bench = p.add_argument_group("bench (trtllm-bench)")
    g_bench.add_argument(
        "--subcommand",
        choices=["throughput", "latency"],
        default="throughput",
        help="trtllm-bench subcommand (default: throughput)",
    )
    g_bench.add_argument(
        "--backend", default="pytorch", help="trtllm-bench --backend (default: pytorch)"
    )
    g_bench.add_argument("--tp", type=int, default=1, help="Tensor parallel size (default: 1)")
    g_bench.add_argument("--pp", type=int, default=1, help="Pipeline parallel size (default: 1)")
    g_bench.add_argument("--ep", type=int, default=0, help="Expert parallel size (default: 0)")
    g_bench.add_argument(
        "--dataset", help="Dataset file path (default: <work-dir>/bench_dataset.txt)"
    )
    g_bench.add_argument(
        "--num-requests",
        dest="num_requests",
        type=int,
        default=100,
        help="Synthetic dataset size (default: 100)",
    )
    g_bench.add_argument(
        "--input-mean",
        dest="input_mean",
        type=int,
        default=128,
        help="Mean input token length (default: 128)",
    )
    g_bench.add_argument(
        "--input-stdev",
        dest="input_stdev",
        type=int,
        default=0,
        help="Input length std dev (default: 0)",
    )
    g_bench.add_argument(
        "--output-mean",
        dest="output_mean",
        type=int,
        default=128,
        help="Mean output token length (default: 128)",
    )
    g_bench.add_argument(
        "--output-stdev",
        dest="output_stdev",
        type=int,
        default=0,
        help="Output length std dev (default: 0)",
    )
    g_bench.add_argument("--concurrency", type=int, help="--concurrency for benchmark run")
    g_bench.add_argument(
        "--max-batch-size", dest="max_batch_size", type=int, help="--max_batch_size"
    )
    g_bench.add_argument(
        "--max-num-tokens", dest="max_num_tokens", type=int, help="--max_num_tokens"
    )
    g_bench.add_argument("--config", help="Extra LLM API options YAML (--config flag)")
    g_bench.add_argument(
        "--report-json",
        dest="report_json",
        help="Output report path (default: <work-dir>/bench_report.json)",
    )
    g_bench.add_argument(
        "--workspace", help="trtllm-bench --workspace: directory for benchmark intermediate files"
    )
    g_bench.add_argument(
        "--kv-cache-free-gpu-mem-fraction",
        dest="kv_cache_free_gpu_mem_fraction",
        type=float,
        help="Fraction of GPU memory reserved for KV cache after model load (default: 0.90)",
    )
    g_bench.add_argument(
        "--max-seq-len",
        dest="max_seq_len",
        type=int,
        help="Maximum total sequence length (input + output tokens)",
    )
    g_bench.add_argument(
        "--beam-width",
        dest="beam_width",
        type=int,
        help="Number of search beams for beam search decoding (default: 1)",
    )
    g_bench.add_argument(
        "--warmup", type=int, help="Number of warmup requests before benchmarking (default: 2)"
    )
    g_bench.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming mode (throughput subcommand only)",
    )
    g_bench.add_argument(
        "--no-chunked-context",
        dest="no_chunked_context",
        action="store_true",
        help="Disable chunked prefill — passes --disable_chunked_context (throughput only; chunked is on by default)",
    )
    g_bench.add_argument(
        "--scheduler-policy",
        dest="scheduler_policy",
        choices=["guaranteed_no_evict", "max_utilization"],
        help="KV cache scheduler policy (throughput only)",
    )
    g_bench.add_argument(
        "--cluster-size",
        dest="cluster_size",
        type=int,
        help="Expert cluster parallelism size (MoE models)",
    )
    g_bench.add_argument(
        "--iteration-log",
        dest="iteration_log",
        help="Path to write per-iteration benchmark log (--iteration_log)",
    )

    # ---- eval ----
    # Note: --max-batch-size, --max-num-tokens, --max-seq-len, --config, and --backend
    # are defined in the bench group but are read by build_eval as well — no duplicates needed.
    g_eval = p.add_argument_group("eval (trtllm-eval)")
    g_eval.add_argument(
        "--tasks",
        help="Comma-separated eval task names (e.g. gsm8k,mmlu). "
        "Each task becomes a separate trtllm-eval invocation joined with &&.",
    )
    g_eval.add_argument(
        "--tp-size", dest="tp_size", type=int, default=1, help="TP size (default: 1)"
    )
    g_eval.add_argument(
        "--pp-size", dest="pp_size", type=int, default=1, help="PP size (default: 1)"
    )
    g_eval.add_argument(
        "--ep-size", dest="ep_size", type=int, default=0, help="EP size (default: 0)"
    )
    # Global eval options
    g_eval.add_argument(
        "--tokenizer",
        dest="tokenizer",
        help="Tokenizer path or HF name (--tokenizer; only needed for TensorRT engine)",
    )
    g_eval.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Pass --trust_remote_code to trtllm-eval",
    )
    g_eval.add_argument(
        "--disable-kv-cache-reuse",
        dest="disable_kv_cache_reuse",
        action="store_true",
        help="Pass --disable_kv_cache_reuse to trtllm-eval",
    )
    g_eval.add_argument(
        "--kv-cache-free-gpu-memory-fraction",
        dest="eval_kv_cache_free_gpu_memory_fraction",
        type=float,
        help="Fraction of free GPU memory for KV cache after model weights "
        "(--kv_cache_free_gpu_memory_fraction; note: 'memory' not 'mem')",
    )
    # Common per-task options (appended after each task name)
    g_eval.add_argument(
        "--num-samples",
        dest="num_samples",
        type=int,
        help="Number of evaluation samples per task (--num_samples)",
    )
    g_eval.add_argument(
        "--apply-chat-template",
        dest="apply_chat_template",
        action="store_true",
        help="Apply the model's chat template to each prompt (--apply_chat_template)",
    )
    g_eval.add_argument(
        "--chat-template-kwargs",
        dest="chat_template_kwargs",
        help="Chat template kwargs as a JSON string (--chat_template_kwargs)",
    )
    g_eval.add_argument(
        "--system-prompt",
        dest="system_prompt",
        help="System prompt prepended to every request (--system_prompt)",
    )
    g_eval.add_argument(
        "--max-input-length",
        dest="max_input_length",
        type=int,
        help="Maximum input prompt length in tokens (--max_input_length)",
    )
    g_eval.add_argument(
        "--max-output-length",
        dest="max_output_length",
        type=int,
        help="Maximum output generation length in tokens (--max_output_length)",
    )
    g_eval.add_argument(
        "--log-samples",
        dest="log_samples",
        action="store_true",
        help="Log per-sample outputs to stdout for debugging (--log_samples)",
    )
    g_eval.add_argument(
        "--output-path",
        dest="output_path",
        help="Path to save per-sample evaluation results JSON (--output_path)",
    )
    # mmlu-specific options (only appended when the task is 'mmlu')
    g_eval.add_argument(
        "--check-accuracy",
        dest="check_accuracy",
        action="store_true",
        help="Assert accuracy meets minimum threshold — mmlu only (--check_accuracy)",
    )
    g_eval.add_argument(
        "--accuracy-threshold",
        dest="accuracy_threshold",
        type=float,
        help="Minimum accuracy fraction to pass — mmlu only (--accuracy_threshold)",
    )
    g_eval.add_argument(
        "--num-fewshot",
        dest="num_fewshot",
        type=int,
        help="Number of few-shot examples prepended to each prompt — mmlu only (--num_fewshot)",
    )
    g_eval.add_argument(
        "--extra-eval-flags",
        dest="extra_eval_flags",
        help="Additional global flags appended verbatim to trtllm-eval (before task name)",
    )

    # ---- perf_sanity ----
    g_perf = p.add_argument_group("perf_sanity (test_perf_sanity.py)")
    g_perf.add_argument(
        "--serving-type",
        dest="serving_type",
        choices=["aggr", "disagg"],
        default="aggr",
        help="Serving topology (default: aggr)",
    )
    g_perf.add_argument(
        "--config-name",
        dest="config_name",
        help="Base name of the perf-sanity config YAML (without .yaml). "
        "Auto-derived from --config-file basename when omitted.",
    )
    g_perf.add_argument(
        "--config-file",
        dest="config_file",
        help="Path to a generated or existing perf-sanity config YAML. "
        "config_name is derived from its basename. "
        "Use with --list-test-names to inspect available server configs.",
    )
    g_perf.add_argument(
        "--list-test-names",
        dest="list_test_names",
        action="store_true",
        help="Print available server_config names from --config-file and exit.",
    )
    g_perf.add_argument(
        "--test-name",
        dest="test_name",
        help="Server config entry name in the YAML (aggr only). "
        "Required for aggr; if omitted and --config-file is given, "
        "available names are shown.",
    )
    g_perf.add_argument(
        "--benchmark-mode",
        dest="benchmark_mode",
        choices=["e2e", "gen_only", "ctx_only"],
        default="e2e",
        help="Benchmark mode (disagg, default: e2e)",
    )
    g_perf.add_argument(
        "--pytest-flags", dest="pytest_flags", default="-v", help="Pytest flags (default: -v)"
    )
    g_perf.add_argument(
        "--extra-pytest-flags",
        dest="extra_pytest_flags",
        help="Additional pytest flags appended verbatim",
    )

    return p


def main():
    parser = make_parser()
    args = parser.parse_args()

    builders = {
        "bench": build_bench,
        "eval": build_eval,
        "perf_sanity": build_perf_sanity,
    }
    cmd = builders[args.cmd_type](args)
    # build_perf_sanity returns None when --list-test-names is used
    # (output already printed to stdout by the function).
    if cmd is not None:
        print(cmd)


if __name__ == "__main__":
    main()
