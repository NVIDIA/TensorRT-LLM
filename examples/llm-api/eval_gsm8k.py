### :title Evaluate GSM8K through a chat endpoint
### :order 8
### :section Customization

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
r"""Evaluate the first N questions of the pinned GSM8K test split over a chat endpoint.

Existing endpoint (any OpenAI-compatible server: aggregated or disaggregated ``trtllm-serve``,
or a Dynamo frontend). Needs only ``requests`` and ``pyarrow``; the server is never started,
stopped or signalled::

    python examples/llm-api/eval_gsm8k.py --endpoint http://HOST:PORT \
        --dataset-dir DATASET --num-questions 100 --output-dir RESULTS

Managed server: launch ``trtllm-serve MODEL --config CONFIG`` on this machine, evaluate, and
stop the processes this run started::

    python examples/llm-api/eval_gsm8k.py --model MODEL --config CONFIG \
        [--chat-template TEMPLATE] [--server-env NAME=VALUE ...] \
        --dataset-dir DATASET --num-questions 100 --output-dir RESULTS

``DATASET`` holds ``main/train-00000-of-00001.parquet`` and ``main/test-00000-of-00001.parquet``
of ``openai/gsm8k`` at revision ``cc7b047b6e5bb11b4f1af84efc572db110a51b3c``. The output
directory receives ``results.jsonl`` (one line per question attempt), ``run_manifest.json`` and
``summary.json``; running again with the same directory and settings resumes, provided the
server still serves the model that the recorded replies came from. The exit status is 0 when
every selected question completed without error, timeout or truncation (and, with
``--min-correct``, enough answers are correct), 1 when the run finished but did not pass, and
2 when it was refused before asking.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
from typing import Optional, Sequence

from gsm8k_eval import protocol, runner

DEFAULT_CONCURRENCY = 64
DEFAULT_TIME_BUDGET_S = 2700.0


def _count(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    try:
        return protocol.check_count(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _positive_int(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _env_assignment(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See examples/llm-api/gsm8k_eval/README.md for the protocol and the outputs.",
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--endpoint", help="base URL of an existing OpenAI-compatible server")
    where.add_argument("--model", help="model directory for a managed trtllm-serve launch")
    parser.add_argument(
        "--served-model-name",
        help="model id to request (default: the only one listed; required for endpoints "
        "without /v1/models, such as the trtllm-serve disaggregated router)",
    )
    parser.add_argument("--config", help="managed mode: the trtllm-serve --config YAML")
    parser.add_argument(
        "--chat-template", help="managed mode: the trtllm-serve --chat_template file"
    )
    parser.add_argument(
        "--server-env",
        type=_env_assignment,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="managed mode: an extra server environment variable (repeatable)",
    )
    parser.add_argument(
        "--monitor-gpus",
        default="",
        help="managed mode: comma-separated GPU indices to sample memory",
    )
    parser.add_argument("--dataset-dir", required=True, help="the staged openai/gsm8k dataset")
    parser.add_argument(
        "--num-questions",
        type=_count,
        default=protocol.NUM_QUESTIONS,
        help=f"evaluate test rows 0..N-1, 1 <= N <= {protocol.TEST_SPLIT_ROWS} (default %(default)s)",
    )
    parser.add_argument("--output-dir", required=True, help="results directory (resumable)")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        help=f"requests in flight (default: managed mode the config's max_batch_size, endpoint "
        f"mode {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--time-budget-s",
        type=_positive_float,
        default=DEFAULT_TIME_BUDGET_S,
        help="whole-run budget in seconds, startup included (default %(default)s)",
    )
    parser.add_argument(
        "--request-timeout-s", type=_positive_float, help="per-request read timeout"
    )
    parser.add_argument(
        "--max-seq-len",
        type=_positive_int,
        help="the server's max_seq_len, to reject prompts that cannot hold the full allowance "
        "(default: managed mode reads it from the config)",
    )
    parser.add_argument(
        "--min-correct",
        type=int,
        help="also require at least this many correct answers (regression gate)",
    )
    return parser


def _managed(args: argparse.Namespace) -> tuple[runner.ManagedServer, Optional[int], Optional[int]]:
    """The managed launch and the concurrency/max_seq_len its config implies."""
    import yaml

    with open(args.config, "rb") as f:
        raw = f.read()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise runner.EvaluationError(f"{args.config} is not a YAML mapping")
    argv = ["trtllm-serve", args.model, "--config", args.config]
    if args.chat_template:
        argv += ["--chat_template", args.chat_template]
    env = dict(os.environ)
    env.update(dict(args.server_env))
    gpus = tuple(token.strip() for token in args.monitor_gpus.split(",") if token.strip())
    settings = {
        "config": os.path.abspath(args.config),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "chat_template": os.path.abspath(args.chat_template) if args.chat_template else None,
        "chat_template_sha256": protocol.sha256_file(args.chat_template)
        if args.chat_template
        else None,
        "server_env": dict(args.server_env),
    }
    batch, seq = config.get("max_batch_size"), config.get("max_seq_len")
    return (
        runner.ManagedServer(argv=tuple(argv), env=env, gpu_ids=gpus, settings=settings),
        batch if type(batch) is int and batch > 0 else None,
        seq if type(seq) is int and seq > 0 else None,
    )


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.endpoint is not None:
        for flag in ("config", "chat_template"):
            if getattr(args, flag):
                parser.error(f"--{flag.replace('_', '-')} applies to managed mode (--model) only")
        if args.server_env or args.monitor_gpus:
            parser.error("--server-env and --monitor-gpus apply to managed mode (--model) only")
    elif not args.config:
        parser.error("managed mode (--model) needs --config")
    if args.min_correct is not None and not 0 <= args.min_correct <= args.num_questions:
        parser.error(f"--min-correct must be in 0..{args.num_questions}")

    managed = None
    concurrency = args.concurrency
    max_seq_len = args.max_seq_len
    identity_extra = {}
    try:
        if args.model is not None:
            managed, config_batch, config_seq = _managed(args)
            if concurrency is None:
                if config_batch is None:
                    parser.error("the config sets no max_batch_size; pass --concurrency")
                concurrency = config_batch
            max_seq_len = max_seq_len if max_seq_len is not None else config_seq
            identity_extra = {
                "model": os.path.abspath(args.model),
                "server_settings": dict(managed.settings),
            }
        elif concurrency is None:
            concurrency = DEFAULT_CONCURRENCY
        signal.signal(signal.SIGTERM, _interrupt)
        summary = runner.run(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            num_questions=args.num_questions,
            concurrency=concurrency,
            time_budget_s=args.time_budget_s,
            endpoint=args.endpoint,
            served_model=args.served_model_name,
            managed=managed,
            max_seq_len=max_seq_len,
            min_correct=args.min_correct,
            request_timeout_s=args.request_timeout_s,
            identity_extra=identity_extra,
            attempt_extra={"client_concurrency": concurrency},
        )
    except (runner.EvaluationError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; completed replies are kept and the run can be resumed", file=sys.stderr)
        return 130
    for problem in summary["verdict"]["problems"]:
        print(f"problem: {problem}", file=sys.stderr)
    return 0 if summary["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
