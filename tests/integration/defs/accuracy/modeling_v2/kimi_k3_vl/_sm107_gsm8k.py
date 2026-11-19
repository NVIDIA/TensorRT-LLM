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
"""Shared full-GSM8K gate of the NVFP4 Kimi K3 targets on SM107.

The targets run in disaggregated serving: one-GPU DWDP context servers
(``kimi_k3_nvfp4__sm_107__tp1``) and a generation server laid out as attention
DP16 with expert parallel EP16 (``__dep16``) or Helix CP8 with the experts over
EP8 (``__helix8_moeep8``) or MoE TP2 x EP4 (``__helix8_moetp2ep4``). Each
target's ``test_accuracy.py`` lists its cases. Each case runs one multi-node deployment
through the perf-sanity disaggregated harness (``jenkins/scripts/perf/submit.py``;
``jenkins/scripts/perf/local/submit.py`` outside CI) from a
``tests/scripts/perf-sanity/disaggregated/vr200_*`` config, and replaces the
benchmark client with GSM8K against the disaggregated frontend. Every role of the
harness runs the same test id; only the client role evaluates and asserts.

The configs require modeling_v2 on every worker, so an unsupported deployment
cannot silently test the built-in model.
"""

import contextlib
import os
from concurrent.futures import Future
from typing import Iterator, Optional

import openai
import pytest
import requests

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CompletionOutput,
    DSparkDecodingConfig,
    RequestOutput,
    SamplingParams,
)
from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

from ....conftest import llm_models_root
from ...accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ...test_disaggregated_serving import DuckLLM, MyThreadPoolExecutor, Result

MODEL_NAME = "moonshotai/Kimi-K3"
# The perf-sanity configs name the same directories (model_dir_name and
# speculative_model, both under the models root).
NVFP4_MODEL_PATH = f"{llm_models_root()}/Kimi-K3-NVFP4"
DSPARK_MODEL_PATH = f"{llm_models_root()}/Inferact-Kimi-K3-DSpark"

# Deployment config -> DSpark draft length used by its generation server.
CASES = {
    # Context on 16 one-GPU DWDP servers; generation on Helix CP8, MoE TP2 x EP4, batch 20.
    "helix8_moetp2ep4": (
        "disagg-e2e-vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_helix8_bs20_dspark5",
        5,
    ),
    # Generation on Helix CP8 with experts over EP8, batch 8.
    "helix8_moeep8": (
        "disagg-e2e-vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_helix8_bs8_dspark7",
        7,
    ),
    # Generation with attention DP over 16 ranks and experts over EP16.
    "dep16": (
        "disagg-e2e-vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_dep16_bs16_dspark5",
        5,
    ),
}


@contextlib.contextmanager
def _frontend_llm(base_url: str, max_draft_len: int, max_workers: int) -> Iterator[DuckLLM]:
    """An LLM stand-in that sends GSM8K's requests to the disaggregated frontend."""
    response = requests.get(f"{base_url}/models", timeout=60)
    response.raise_for_status()
    served_model = response.json()["data"][0]["id"]
    # Seconds per request attempt; a stalled request ends well inside the test timeout.
    client = openai.OpenAI(api_key="unused", base_url=base_url, timeout=3600)
    # The fields that pick the GSM8K reference row: the NVFP4 export declares
    # MIXED_PRECISION, the workers run an NVFP4 KV cache and DSpark.
    args = LlmArgs(model=NVFP4_MODEL_PATH)
    args.quant_config.quant_algo = "MIXED_PRECISION"
    args.quant_config.kv_cache_quant_algo = "NVFP4"
    args.speculative_config = DSparkDecodingConfig(
        max_draft_len=max_draft_len, speculative_model=DSPARK_MODEL_PATH
    )
    tokenizer = load_hf_tokenizer(NVFP4_MODEL_PATH, trust_remote_code=True)

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


def run_disaggregated_gsm8k(output_dir: str, test_case: str, max_draft_len: int) -> None:
    """Run ``test_case``'s deployment in the role this process was launched as.

    The client role runs GSM8K against the frontend and asserts the score;
    the worker and frontend roles serve until the client is done.
    """
    from ....perf.test_perf_sanity import AccuracyConfig, PerfSanityTestConfig

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
            with _frontend_llm(base_url, max_draft_len, self.max_workers) as llm:
                GSM8K(MODEL_NAME).evaluate(llm)
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


requires_sm107 = pytest.mark.skipif(
    get_sm_version() != 107, reason="The modeling_v2 Kimi K3 SM107 targets require SM107 (Rubin)."
)


class Sm107Gsm8kHarness(LlmapiAccuracyTestHarness):
    """GSM8K of one SM107 Kimi K3 target's disaggregated deployments."""

    MODEL_NAME = MODEL_NAME
    MODEL_PATH = NVFP4_MODEL_PATH
