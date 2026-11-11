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
``agent_flow``. The evaluator prints its score as it does for ``trtllm-eval``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

#: The benchmarks whose evaluator the tree has driven through an endpoint: evaluator class
#: name in ``tensorrt_llm.evaluate``. Any other is not judged through the hop.
EVALUATORS: dict[str, str] = {"gsm8k": "GSM8K"}


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
    }
    stop = getattr(sampling, "stop", None)
    if stop:
        body["stop"] = stop
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="http://<host>:<port> of the endpoint")
    parser.add_argument("--model", required=True, help="the checkpoint directory")
    parser.add_argument("--benchmark", required=True, choices=sorted(EVALUATORS))
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--stop", default=None, help="a JSON list of stop strings")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--apply_chat_template", action="store_true")
    args = parser.parse_args(argv)

    import tensorrt_llm.evaluate as evaluate
    from tensorrt_llm.llmapi import SamplingParams
    from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

    evaluator = getattr(evaluate, EVALUATORS[args.benchmark])(
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        random_seed=0,
        apply_chat_template=args.apply_chat_template,
    )
    sampling = SamplingParams(
        max_tokens=args.max_output_length,
        truncate_prompt_tokens=args.max_input_length,
        stop=json.loads(args.stop) if args.stop else None,
        temperature=0.0,
        top_k=1,
    )
    llm = EndpointLLM(args.url, args.model, load_hf_tokenizer(args.model), args.concurrency)
    try:
        evaluator.evaluate(llm, sampling)
    finally:
        llm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
