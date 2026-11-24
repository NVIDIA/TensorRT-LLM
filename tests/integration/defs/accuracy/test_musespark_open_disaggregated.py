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
"""HTTP accuracy clients for Slurm-launched 1P1D TEP4/TEP8 deployments.

The disaggregated perf launcher owns the workers and invokes this suite via
accuracy.pytest_nodeid. The clients run on one rank and require no local model
engine. Missing checkpoints, datasets, workers or speculative stats fail the run.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import openai
import pytest
import requests

from tensorrt_llm.llmapi import CompletionOutput, DFlashDecodingConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, assert_acceptance_length
from .test_disaggregated_serving import compute_disagg_acceptance_length


@pytest.fixture
def disagg_client():
    url = os.environ.get("DISAGG_ACCURACY_URL")
    if not url:
        pytest.skip("requires the Slurm disaggregated accuracy launcher")
    assert os.environ.get("INTEGRATION_TEST", "0") != "1", "accuracy verification must be enabled"
    assert os.environ.get("TRTLLM_ACCURACY_NO_REFERENCE", "0") != "1"
    assert os.environ.get("TRTLLM_POPULATE_ACCEPTANCE_LENGTH", "0") != "1"
    model = os.environ["DISAGG_ACCURACY_MODEL"]
    with openai.OpenAI(
        base_url=f"{url}/v1", api_key="unused", timeout=1800, max_retries=0
    ) as client:
        yield url, model, client


@pytest.mark.timeout(7200)
def test_gsm8k(disagg_client):
    """The disaggregated target and DFlash verification preserve GSM8K accuracy."""
    _, model, client = disagg_client
    args = LlmArgs(model=model)
    args.quant_config.quant_algo = "NVFP4"
    args.quant_config.kv_cache_quant_algo = "FP8"
    args.speculative_config = DFlashDecodingConfig(
        max_draft_len=7, speculative_model=f"{llm_models_root()}/Muse-Spark-Open-816B-A42B-DFlash2"
    )

    def complete(prompt, sampling_params, streaming=False):
        assert not streaming
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=sampling_params.max_tokens,
            temperature=0,
            stop=sampling_params.stop,
        )
        return SimpleNamespace(
            prompt=prompt, outputs=[CompletionOutput(text=response.choices[0].text, index=0)]
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        llm = SimpleNamespace(
            args=args,
            tokenizer=load_hf_tokenizer(model),
            generate_async=lambda prompt, sampling_params, streaming=False: pool.submit(
                complete, prompt, sampling_params, streaming
            ),
        )
        GSM8K("musespark_open").evaluate(
            llm, sampling_params=SamplingParams(max_tokens=256, temperature=0)
        )


@pytest.mark.timeout(7200)
def test_speed_bench_coding_golden_acceptance(disagg_client):
    """Natural DFlash2 acceptance clears the existing K=7 golden floor."""
    url, model, client = disagg_client
    dataset = os.environ.get("MUSESPARK_SPEED_BENCH_CODING") or (
        f"{llm_models_root()}/datasets/nvidia/SPEED-Bench/qualitative_coding_turn0.jsonl"
    )
    with open(dataset, encoding="utf-8") as source:
        texts = [json.loads(line)["text"] for line in source if line.strip()]
    assert len(texts) == 80, f"expected 80 coding prompts, found {len(texts)}"
    # Drain GSM8K / warmup stats so the golden protocol is measured by itself.
    response = requests.get(f"{url}/cluster_info", timeout=10)
    response.raise_for_status()
    workers = response.json()["current_workers"]["generation_servers"]
    assert workers, "no generation workers registered"
    for worker in workers:
        response = requests.get(f"http://{worker['host']}:{worker['port']}/metrics", timeout=60)
        response.raise_for_status()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for seed in (0, 1):
            futures = [
                pool.submit(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=4096,
                    temperature=1.0,
                    top_p=0.95,
                    seed=seed,
                    extra_body={"cache_salt": f"golden-{seed}-{index}"},
                )
                for index, text in enumerate(texts)
            ]
            for future in futures:
                future.result()
    acceptance_length = compute_disagg_acceptance_length(url)
    print(f"[AL] disagg DFlash2 SPEED-Bench coding: {acceptance_length:.4f}")
    assert_acceptance_length(
        "ModelingV2MuseSparkOpenNvfp4Tep4::dflash2_speed_bench_coding_golden", acceptance_length
    )
