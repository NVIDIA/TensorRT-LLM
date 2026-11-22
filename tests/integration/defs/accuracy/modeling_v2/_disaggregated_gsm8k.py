# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full GSM8K through the frontend of a multi-node perf-sanity deployment."""

import contextlib
import os
from concurrent.futures import Future
from typing import Iterator, Optional

import openai
import requests

from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

from ..accuracy_core import GSM8K
from ..test_disaggregated_serving import DuckLLM, MyThreadPoolExecutor, Result


@contextlib.contextmanager
def _frontend_llm(base_url: str, args: LlmArgs, max_workers: int) -> Iterator[DuckLLM]:
    """An LLM stand-in that sends GSM8K's requests to the disaggregated frontend."""
    response = requests.get(f"{base_url}/models", timeout=60)
    response.raise_for_status()
    served_model = response.json()["data"][0]["id"]
    # Seconds per request attempt; a stalled request ends well inside the test timeout.
    client = openai.OpenAI(api_key="unused", base_url=base_url, timeout=3600)
    tokenizer = load_hf_tokenizer(args.model, trust_remote_code=True)

    with MyThreadPoolExecutor(max_workers=max_workers) as thread_pool:

        def send_request(
            prompt: str, sampling_params: Optional[SamplingParams], streaming: bool
        ) -> RequestOutput:
            if sampling_params is None:
                sampling_params = SamplingParams()
            response = client.completions.create(
                model=served_model,
                prompt=prompt,
                stream=streaming,
                max_tokens=sampling_params.max_tokens,
                n=sampling_params.n,
                temperature=(
                    sampling_params.temperature if sampling_params.top_p is not None else 0
                ),
                top_p=sampling_params.top_p,
                stop=sampling_params.stop,
                seed=sampling_params.seed,
            )
            result = Result(
                id=0,
                sampling_params=sampling_params,
                outputs=[
                    CompletionOutput(text=choice.text, index=index)
                    for index, choice in enumerate(response.choices)
                ],
            )
            output = RequestOutput._from_generation_result(result, prompt=prompt)
            setattr(output, "result", result.result)
            return output

        def generate_async(
            prompt: str, sampling_params: Optional[SamplingParams] = None, streaming: bool = False
        ) -> Future:
            future = thread_pool.submit(send_request, prompt, sampling_params, streaming)
            thread_pool.futures.append(future)
            return future

        yield DuckLLM(args, tokenizer, generate_async)


def run_disaggregated_gsm8k(
    output_dir: str, test_case: str, reference_model: str, args: LlmArgs
) -> None:
    """Run ``test_case``'s deployment in the role this process was launched as.

    The client role runs GSM8K against the frontend and asserts the score;
    the worker and frontend roles serve until the client is done.
    """
    from ...perf.test_perf_sanity import AccuracyConfig, PerfSanityTestConfig

    assert "DISAGG_SERVING_TYPE" in os.environ, (
        "launch through the multi-node disaggregated harness (jenkins/scripts/perf/submit.py)"
    )

    class Gsm8kGate(AccuracyConfig):
        def __init__(self, max_workers: int) -> None:
            super().__init__({"enable_accuracy_test": True})
            self.max_workers = max_workers
            self.evaluated = False

        def run(
            self,
            model_name: str,
            server_hostname: str,
            server_port: int,
            output_dir: str,
            server_idx: int,
        ) -> None:
            base_url = f"http://{server_hostname}:{server_port}/v1"
            with _frontend_llm(base_url, args, self.max_workers) as llm:
                GSM8K(reference_model).evaluate(llm)
            self.evaluated = True

    config = PerfSanityTestConfig(test_case, output_dir)
    config.parse_config_file()
    commands = config.get_commands()
    gates = []
    for clients in commands.client_configs.values():
        for client in clients:
            gate = Gsm8kGate(max_workers=client.concurrency)
            client.only_run_accuracy = True
            client.accuracy_config = gate
            gates.append(gate)
    assert gates, f"{test_case} defines no client to run GSM8K from"
    config.run_ex(commands)
    if commands.disagg_serving_type == "BENCHMARK":
        assert all(gate.evaluated for gate in gates), "GSM8K did not run against every frontend"
