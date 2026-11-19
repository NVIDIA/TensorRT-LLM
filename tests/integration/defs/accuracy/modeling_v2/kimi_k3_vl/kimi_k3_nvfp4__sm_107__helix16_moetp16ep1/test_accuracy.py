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
"""Full-GSM8K gate for kimi_k3_nvfp4__sm_107__helix16_moetp16ep1, NVFP4 Kimi K3 on SM107.

The target is the ultra-low-latency generation server of disaggregated serving:
Helix CP16 with the routed experts over MoE TP16, DSpark with 7 draft tokens and
at most 4 requests, fed by one-GPU DWDP context servers (which run the
kimi_k3_nvfp4__sm_107__tp1 target). The case runs one multi-node deployment
through the perf-sanity disaggregated harness (``jenkins/scripts/perf/submit.py``;
``jenkins/scripts/perf/local/submit.py`` outside CI) from a
``tests/scripts/perf-sanity/disaggregated/vr200_*`` config, and replaces the
benchmark client with GSM8K against the disaggregated frontend. Every role of the
harness runs the same test id; only the client role evaluates and asserts.

The config requires modeling_v2 on every worker, so an unsupported deployment
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

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ....test_disaggregated_serving import DuckLLM, MyThreadPoolExecutor, Result

_MODEL_NAME = "moonshotai/Kimi-K3"
# The perf-sanity configs name the same directories (model_dir_name and
# speculative_model, both under the models root).
_NVFP4_MODEL_PATH = f"{llm_models_root()}/Kimi-K3-NVFP4"
_DSPARK_MODEL_PATH = f"{llm_models_root()}/Inferact-Kimi-K3-DSpark"

# Deployment config -> DSpark draft length used by its generation server.
_CASES = {
    # Generation on Helix CP16 with experts over MoE TP16, batch 4; context on 12 one-GPU
    # DWDP servers.
    "disagg-e2e-vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx12_dwdp12_gen1_helix16_moetp16ep1_bs4_dspark7": 7,
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
    args = LlmArgs(model=_NVFP4_MODEL_PATH)
    args.quant_config.quant_algo = "MIXED_PRECISION"
    args.quant_config.kv_cache_quant_algo = "NVFP4"
    args.speculative_config = DSparkDecodingConfig(
        max_draft_len=max_draft_len, speculative_model=_DSPARK_MODEL_PATH
    )
    tokenizer = load_hf_tokenizer(_NVFP4_MODEL_PATH, trust_remote_code=True)

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


def _run_disaggregated_gsm8k(output_dir: str, test_case: str, max_draft_len: int) -> None:
    """Run ``test_case``'s deployment in the role this process was launched as.

    The client role runs GSM8K against the frontend and asserts the score;
    the worker and frontend roles serve until the client is done.
    """
    from .....perf.test_perf_sanity import AccuracyConfig, PerfSanityTestConfig

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
                GSM8K(_MODEL_NAME).evaluate(llm)
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


@pytest.mark.skipif(
    get_sm_version() != 107, reason="This modeling_v2 Kimi K3 target requires SM107 (Rubin)."
)
@pytest.mark.timeout(7200)
class TestModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1(LlmapiAccuracyTestHarness):
    """GSM8K against the target's disaggregated deployment."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _NVFP4_MODEL_PATH

    @pytest.mark.parametrize("test_case", list(_CASES))
    def test_gsm8k(self, output_dir: str, test_case: str) -> None:
        _run_disaggregated_gsm8k(output_dir, test_case, max_draft_len=_CASES[test_case])
