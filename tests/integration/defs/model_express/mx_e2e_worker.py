# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Run one process role for the ModelExpress donor/receiver E2E test."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

try:  # Run as a script: the worker directory is `sys.path[0]`.
    from mx_evidence import (
        check_receiver_transfer_logs,
        summarize_transfer_logs,
        transfer_logs_by_rank,
    )
except ImportError:  # Imported by pytest as part of the `defs` package.
    from defs.model_express.mx_evidence import (
        check_receiver_transfer_logs,
        summarize_transfer_logs,
        transfer_logs_by_rank,
    )

# Eight prompts of distinct lengths (4-12 tokens). Every ID stays below 30000
# so the same probe is valid for TinyLlama (32000-token vocabulary), Mistral,
# and Qwen tokenizers; the leading `1` mirrors a BOS token.
_PROMPT_TOKEN_IDS = (
    (1, 42, 7, 9),
    (1, 17, 23, 5, 11),
    (1, 306, 626, 263, 4086, 1904),
    (1, 450, 4996, 17354, 1701, 29916, 432),
    (1, 3, 4, 5, 6, 7, 8, 9),
    (1, 1724, 338, 278, 7483, 310, 3444, 29973, 13),
    (1, 15043, 3186, 29991, 1128, 526, 366, 2599, 9826, 29973),
    (1, 12, 34, 56, 78, 910, 1112, 1314, 1516, 1718, 1920, 2122),
)
# Greedy tokens generated per prompt; `end_id=-1` keeps every output exactly
# this long so the orchestrator can compare fixed-length token-ID lists.
_MAX_NEW_TOKENS = 32
# Engine limits sized for the probe: the longest prompt plus the generated
# tokens fits in `_MAX_SEQ_LEN`, and one context iteration schedules the
# whole batch within `_MAX_NUM_TOKENS`.
_MAX_SEQ_LEN = 128
_MAX_NUM_TOKENS = 256
# Exit status of a receiver whose own transfer-log self-check failed before
# any accuracy evaluation was attempted.
SELF_CHECK_FAILED_EXIT_CODE = 3
_EVAL_TASKS = ("MMLU", "GSM8K")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("baseline", "donor", "receiver"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mx-url")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--max-serve-seconds", type=float, default=1800.0)
    eval_group = parser.add_argument_group(
        "eval", "Run an accuracy evaluator on the loaded model instead of the token probe"
    )
    eval_group.add_argument("--eval-task", choices=_EVAL_TASKS)
    eval_group.add_argument("--eval-num-samples", type=int)
    eval_group.add_argument("--eval-dataset-path")
    eval_group.add_argument("--eval-max-input-len", type=int)
    eval_group.add_argument("--eval-max-output-len", type=int)
    eval_group.add_argument("--eval-max-batch-size", type=int, default=32)
    eval_group.add_argument("--eval-kv-fraction", type=float, default=0.6)
    eval_group.add_argument("--eval-random-seed", type=int, default=0)
    eval_group.add_argument("--eval-apply-chat-template", action="store_true")
    eval_group.add_argument("--eval-system-prompt")
    args = parser.parse_args(argv)
    if args.eval_task is not None:
        if args.role == "donor":
            parser.error("--eval-task is not supported for the donor role")
        required = (
            "eval_num_samples",
            "eval_dataset_path",
            "eval_max_input_len",
            "eval_max_output_len",
        )
        absent = [name for name in required if getattr(args, name) is None]
        if absent:
            flags = ", ".join("--" + name.replace("_", "-") for name in absent)
            parser.error(f"--eval-task requires {flags}")
    return args


def _llm_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": args.model,
        "backend": "pytorch",
        "checkpoint_format": "HF" if args.role == "baseline" else "MX",
        "tensor_parallel_size": args.tp_size,
        "dtype": "bfloat16",
        "attn_backend": "TRTLLM",
        # Evaluators feed string prompts and need the tokenizer; the token
        # probe works on token IDs only.
        "skip_tokenizer_init": args.eval_task is None,
        "max_batch_size": len(_PROMPT_TOKEN_IDS),
        "max_num_tokens": _MAX_NUM_TOKENS,
        "max_seq_len": _MAX_SEQ_LEN,
        "kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=0.15),
    }
    if args.eval_task is not None:
        # Size the engine for the evaluator instead of the short probe. The
        # donor keeps the probe sizing; SourceIdentity does not hash engine
        # limits, so donor and receiver identities still match.
        max_seq_len = args.eval_max_input_len + args.eval_max_output_len
        kwargs.update(
            {
                "max_batch_size": args.eval_max_batch_size,
                "max_num_tokens": max(8192, max_seq_len),
                "max_seq_len": max_seq_len,
                "kv_cache_config": KvCacheConfig(free_gpu_memory_fraction=args.eval_kv_fraction),
            }
        )
    env_overrides = _rank_process_env_overrides(args)
    if env_overrides:
        kwargs["env_overrides"] = env_overrides
    if args.role != "baseline":
        if not args.mx_url:
            raise ValueError("MX donor and receiver roles require --mx-url")
        kwargs["mx_config"] = {
            "server_url": args.mx_url,
            # ModelExpress 0.4.1 skips polling at zero but still sleeps once
            # for five seconds before disk fallback. The receiver starts only
            # after donor readiness, so it can use a bounded discovery window.
            "server_query_timeout_s": 0 if args.role == "donor" else 30,
        }
    return kwargs


def _rank_process_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Environment for the executor rank processes.

    MPI-spawned ranks inherit only selected `TRTLLM*`/`TLLM*` variables, so
    everything the loaders read must travel through `LLM(env_overrides=...)`.
    The weight-manifest variables apply to every role (the HF baseline too);
    the MX transfer log only exists for MX roles.
    """
    overrides: dict[str, str] = {}
    manifest_dir = os.environ.get("MX_WEIGHT_MANIFEST_DIR")
    if manifest_dir:
        overrides["MX_WEIGHT_MANIFEST_DIR"] = manifest_dir
        overrides["MX_WEIGHT_MANIFEST_ROLE"] = args.role
    if args.role != "baseline":
        transfer_log_dir = os.environ.get("MX_TRANSFER_LOG_DIR")
        if transfer_log_dir:
            overrides["MX_TRANSFER_LOG_DIR"] = transfer_log_dir
    return overrides


def _receiver_transfer_evidence() -> dict[str, object] | None:
    """Summarize this receiver's own MX rank logs with the shared evidence rules."""
    transfer_log_dir = os.environ.get("MX_TRANSFER_LOG_DIR")
    if not transfer_log_dir:
        return None
    try:
        summaries = summarize_transfer_logs(Path(transfer_log_dir))
    except ValueError as error:
        return {"error": str(error)}
    return {str(rank): summary.to_dict() for rank, summary in sorted(summaries.items())}


def _self_check_receiver_transfer(
    transfer_log_dir: Path, tp_size: int, timeout_s: float = 5.0
) -> None:
    """Abort before spending eval time when this receiver's own MX logs show a fallback.

    The rank logs are complete once `LLM(...)` returned; the short poll only
    covers handler flushing. Problems are printed and the process exits with
    `SELF_CHECK_FAILED_EXIT_CODE` so the orchestrator can label the failure.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            rank_logs = transfer_logs_by_rank(transfer_log_dir)
        except ValueError as error:
            problems = [str(error)]
            rank_logs = {}
        else:
            problems = check_receiver_transfer_logs(rank_logs, tp_size)
        if not problems or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if problems:
        print("MX_ACCURACY_SELF_CHECK_FAILED: " + "; ".join(problems), flush=True)
        for rank, text in sorted(rank_logs.items()):
            tail = "\n".join(text.splitlines()[-40:])
            print(f"--- rank{rank}.log tail ---\n{tail}", flush=True)
        sys.exit(SELF_CHECK_FAILED_EXIT_CODE)


def _run_eval(llm: LLM, args: argparse.Namespace) -> dict[str, object]:
    """Run the requested `tensorrt_llm.evaluate` task with the reference sampling params."""
    import tensorrt_llm.evaluate as evaluate

    evaluator_cls = getattr(evaluate, args.eval_task)
    evaluator = evaluator_cls(
        dataset_path=args.eval_dataset_path,
        num_samples=args.eval_num_samples,
        random_seed=args.eval_random_seed,
        apply_chat_template=args.eval_apply_chat_template,
        system_prompt=args.eval_system_prompt,
    )
    # Same construction as `accuracy_core.AccuracyTask.evaluate`: greedy by
    # default, prompts truncated to the task's input length.
    sampling_params = SamplingParams(
        max_tokens=args.eval_max_output_len,
        truncate_prompt_tokens=args.eval_max_input_len,
    )
    evaluate_kwargs: dict[str, object] = {}
    if "scores_filter" in inspect.signature(evaluator.evaluate).parameters:
        evaluate_kwargs["scores_filter"] = None
    started = time.perf_counter()
    score = float(evaluator.evaluate(llm, sampling_params=sampling_params, **evaluate_kwargs))
    return {
        "eval_task": args.eval_task,
        "eval_num_samples": args.eval_num_samples,
        "eval_seconds": time.perf_counter() - started,
        "score": score,
        "engine": {
            "max_batch_size": args.eval_max_batch_size,
            "max_seq_len": args.eval_max_input_len + args.eval_max_output_len,
            "max_num_tokens": max(8192, args.eval_max_input_len + args.eval_max_output_len),
            "kv_fraction": args.eval_kv_fraction,
        },
        "sampling": {
            "max_tokens": args.eval_max_output_len,
            "truncate_prompt_tokens": args.eval_max_input_len,
        },
    }


def main() -> None:
    args = _parse_args()
    if args.role == "donor" and (args.ready_file is None or args.stop_file is None):
        raise ValueError("The donor role requires --ready-file and --stop-file")
    if args.role == "donor" and args.max_serve_seconds <= 0:
        raise ValueError("The donor role requires --max-serve-seconds > 0")

    llm_kwargs = _llm_kwargs(args)
    started = time.perf_counter()
    with LLM(**llm_kwargs) as llm:
        load_seconds = time.perf_counter() - started
        mx_config = llm_kwargs.get("mx_config")
        server_query_timeout_s = (
            mx_config.get("server_query_timeout_s") if isinstance(mx_config, dict) else None
        )
        if args.eval_task is not None:
            self_check_started = time.perf_counter()
            if args.role == "receiver":
                transfer_log_dir = os.environ.get("MX_TRANSFER_LOG_DIR")
                if not transfer_log_dir:
                    raise ValueError("The receiver eval role requires MX_TRANSFER_LOG_DIR")
                _self_check_receiver_transfer(Path(transfer_log_dir), args.tp_size)
            self_check_seconds = time.perf_counter() - self_check_started
            payload: dict[str, object] = {
                "role": args.role,
                "tp_size": args.tp_size,
                "load_seconds": load_seconds,
                "self_check_seconds": self_check_seconds,
                "server_query_timeout_s": server_query_timeout_s,
            }
            payload.update(_run_eval(llm, args))
            if args.role == "receiver":
                payload["transfer_evidence"] = _receiver_transfer_evidence()
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print("MX_WORKER_METRICS " + json.dumps(payload), flush=True)
            return

        sampling_params = SamplingParams(
            max_tokens=_MAX_NEW_TOKENS,
            temperature=0.0,
            top_k=1,
            end_id=-1,
            pad_id=0,
        )
        generate_started = time.perf_counter()
        results = list(
            llm.generate(
                [list(prompt) for prompt in _PROMPT_TOKEN_IDS],
                sampling_params=sampling_params,
            )
        )
        generate_seconds = time.perf_counter() - generate_started
        payload = {
            "role": args.role,
            "tp_size": args.tp_size,
            "load_seconds": load_seconds,
            "generate_seconds": generate_seconds,
            "server_query_timeout_s": server_query_timeout_s,
            "max_new_tokens": _MAX_NEW_TOKENS,
            "prompt_count": len(_PROMPT_TOKEN_IDS),
            "prompt_lengths": [len(prompt) for prompt in _PROMPT_TOKEN_IDS],
            "token_ids": [list(result.outputs[0].token_ids) for result in results],
        }
        if args.role == "receiver":
            payload["transfer_evidence"] = _receiver_transfer_evidence()
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        if args.role == "donor":
            assert args.ready_file is not None
            assert args.stop_file is not None
            args.ready_file.write_text("ready\n", encoding="utf-8")
            deadline = time.monotonic() + args.max_serve_seconds
            while not args.stop_file.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"The stop file {args.stop_file} did not appear within "
                        f"{args.max_serve_seconds}s"
                    )
                time.sleep(0.2)


if __name__ == "__main__":
    main()
