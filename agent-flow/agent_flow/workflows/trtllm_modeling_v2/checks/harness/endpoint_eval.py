# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score a benchmark against a served endpoint with the tree's own evaluator.

The tree's evaluators drive an ``LLM``. Here they drive a stand-in that has the three
things they use (``tokenizer``, ``generate_async``, ``shutdown``) and sends each
prompt to an OpenAI-compatible completions endpoint, as the tree's own disaggregated
accuracy tests do. The request states its sampling outright, because a server's
defaults are not the ``LLM``'s: temperature 0, ``top_k`` 1, the truncation and stop
strings the evaluator asks for.

Runs under the interpreter of the tree being measured: nothing here imports
``agent_flow``. The score the evaluator returns is printed as ``trtllm-eval`` logs it.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

#: The benchmarks whose evaluator the tree has driven through an endpoint: evaluator class
#: name in ``tensorrt_llm.evaluate``. Any other is not judged through the hop.
EVALUATORS: dict[str, str] = {"gsm8k": "GSM8K"}
#: How many samples each scores when the command names no count: the whole split.
SAMPLES: dict[str, int] = {"gsm8k": 1319}
#: The exit status of an evaluator whose own loader cannot read the benchmark's data.
NO_DATA = 4


def request_body(model: str, prompt: str, sampling: Any) -> dict[str, Any]:
    """One completion request, every sampling setting stated."""
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(getattr(sampling, "max_tokens", None) or 256),
        "temperature": 0.0,
        "top_k": 1,
        "n": 1,
        "stream": False,
        # Stated even when empty: a server's own default stop is not the evaluator's.
        "stop": list(getattr(sampling, "stop", None) or []),
    }
    truncate = getattr(sampling, "truncate_prompt_tokens", None)
    if truncate:
        body["truncate_prompt_tokens"] = int(truncate)
    return body


class EndpointLLM:
    """What an evaluator needs of an ``LLM``, answered by a served endpoint."""

    def __init__(self, url: str, model: str, tokenizer: Any, workers: int) -> None:
        self.url, self.model, self.tokenizer = url.rstrip("/"), model, tokenizer
        self.args = None
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def _complete(self, prompt: str, sampling: Any) -> Any:
        import urllib.request

        from tensorrt_llm.executor.result import GenerationResultBase
        from tensorrt_llm.llmapi import CompletionOutput, RequestOutput

        class Result(GenerationResultBase):
            """A finished generation built from an endpoint's answer."""

            def __init__(self, outputs: list) -> None:
                super().__init__(0, sampling)
                self._outputs = outputs
                self._streaming = False

            @property
            def outputs(self) -> list:
                return self._outputs

            def result(self):
                return self

        request = urllib.request.Request(
            f"{self.url}/v1/completions",
            data=json.dumps(request_body(self.model, prompt, sampling)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=3600) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
        result = Result(
            [
                CompletionOutput(text=choice["text"], index=index)
                for index, choice in enumerate(payload["choices"])
            ]
        )
        output = RequestOutput._from_generation_result(result, prompt=prompt)
        output.result = result.result
        return output

    def generate_async(self, prompt: str, sampling_params: Any = None, streaming: bool = False):
        return self._pool.submit(self._complete, prompt, sampling_params)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


def build_parser() -> argparse.ArgumentParser:
    """What this evaluator takes: the whole of its protocol, and nothing it would ignore."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="", help="http://<host>:<port> of the endpoint")
    parser.add_argument("--model", default="", help="the checkpoint directory")
    parser.add_argument(
        "--dataset_only",
        action="store_true",
        help="only build the evaluator, which loads the benchmark's data: exit 0 when it "
        "reads it. Asked before any server is started; needs no --url and no --model",
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(EVALUATORS))
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--stop", default=None, help="a JSON list of stop strings")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--apply_chat_template", action="store_true")
    return parser


def not_carried(benchmark: str, arguments: list[str]) -> str:
    """Why this evaluator cannot run ``benchmark`` with ``arguments``; ``""`` when it can.

    Asked before any load, with the evaluator's own parser: an argument it does not take
    (one of ``trtllm-eval``'s own, a few-shot count) or a value it cannot read is a
    protocol it does not carry, and must not be found out after the servers are up.
    """
    if benchmark not in EVALUATORS:
        return f"it scores {', '.join(sorted(EVALUATORS))}, not `{benchmark}`"

    class _Quiet(argparse.ArgumentParser):
        def error(self, message: str):  # type: ignore[override]
            raise ValueError(message)

    parser = build_parser()
    parser.__class__ = _Quiet
    words = ["--url", "http://endpoint", "--model", "model", "--benchmark", benchmark]
    try:
        _, unknown = parser.parse_known_args([*words, *(str(a) for a in arguments)])
    except ValueError as exc:
        return f"it cannot read the benchmark's arguments: {exc}"
    if unknown:
        return f"it does not take {' '.join(unknown)}"
    return ""


def looks(dataset_path: str | None) -> str:
    """Where the evaluator's loader reads the benchmark's data, as its own libraries say."""
    if dataset_path:
        return f"`--dataset_path` {dataset_path}"
    try:
        import datasets
        from huggingface_hub import constants

        caches = f"{constants.HF_HUB_CACHE} and {datasets.config.HF_DATASETS_CACHE}"
    except Exception:  # an install that names them otherwise still reads under HF_HOME
        caches = "under `HF_HOME`"
    return f"the task's own dataset in the HuggingFace caches ({caches}), offline"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.dataset_only or (args.url and args.model)):
        parser.error("--url and --model are required to score")

    import tensorrt_llm.evaluate as evaluate
    from tensorrt_llm.llmapi import SamplingParams
    from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

    stop = json.loads(args.stop) if args.stop else None
    samples = args.num_samples or SAMPLES[args.benchmark]
    # Stop strings a command states replace the task's own, in the task itself: the
    # evaluator copies the task's `until` over whatever the sampling parameters hold.
    stated = {"stop_strings": stop} if stop is not None else {}
    try:
        # Building the evaluator loads the benchmark's data: its loader's answer is the
        # one that counts, not a guess at where a cache keeps what.
        evaluator = getattr(evaluate, EVALUATORS[args.benchmark])(
            dataset_path=args.dataset_path,
            num_samples=samples,
            random_seed=0,
            apply_chat_template=args.apply_chat_template,
            **stated,
        )
    except Exception as exc:  # whatever the loader raises is why it cannot read
        said = (str(exc).strip().splitlines() or [""])[0]
        print(
            f"endpoint-eval: the evaluator of `{args.benchmark}` cannot load its data on "
            f"{socket.gethostname()} ({type(exc).__name__}: {said}). It reads "
            f"{looks(args.dataset_path)}: put the dataset there, or state the benchmark's "
            f"`dataset_path`",
            file=sys.stderr,
            flush=True,
        )
        return NO_DATA
    if args.dataset_only:
        print(f"endpoint-eval: the evaluator of `{args.benchmark}` loads its data here")
        return 0
    sampling = SamplingParams(
        max_tokens=args.max_output_length,
        truncate_prompt_tokens=args.max_input_length,
        stop=stop,
        temperature=0.0,
        top_k=1,
    )
    print(
        f"endpoint-eval: {args.benchmark} on {samples} samples, greedy, stop "
        f"{'the task own' if stop is None else stop}, input cut at {args.max_input_length} "
        f"tokens, {args.max_output_length} new tokens",
        flush=True,
    )
    llm = EndpointLLM(args.url, args.model, load_hf_tokenizer(args.model), args.concurrency)
    try:
        score = evaluator.evaluate(llm, sampling)
    finally:
        llm.shutdown()
    if score is None:
        print("endpoint-eval: the evaluator returned no score", file=sys.stderr)
        return 1
    # The tree logs the score at INFO, which this script does not show: printed here in
    # the tree's own words, which are what the check reads.
    task = getattr(evaluator, "task_name", args.benchmark)
    print(f"lm-eval {task} average accuracy: {float(score):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
