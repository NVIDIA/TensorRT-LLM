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
"""Disaggregated latency with and without pipelined KV transfer.

Run the same workload against different builds to compare implementations; the
config knob has the same name in each, so only the installed wheel changes.
PIPELINED=0|1 selects the variant, LABEL tags the printed result line.
"""

import os
import statistics
import time

import pytest

from tensorrt_llm.llmapi import SamplingParams

from ..conftest import llm_models_root, skip_pre_hopper
from .accuracy_core import LlmapiAccuracyTestHarness
from .test_disaggregated_serving import launch_disaggregated_llm

# Long enough that a 256-token prefill chunk really splits the prompt, which is
# the only regime where overlapping transfer with prefill can pay off.
# The model caps max_seq_len at max_position_embeddings (4096), and this
# vocabulary tokenizes at ~4.3 tokens/word, so keep the prompt near 2.8k tokens.
_PROMPT_WORDS = 650
_MAX_SEQ_LEN = 4096
_NUM_REQUESTS = 16
_MAX_NUM_TOKENS = 256


def _long_prompt(idx: int) -> str:
    """Distinct prompts so prefix reuse cannot hide the transfer cost."""
    head = f"Document {idx}. "
    body = " ".join(f"item{idx}_{i}" for i in range(_PROMPT_WORDS))
    return head + body + "\nSummarize the document above in one sentence."


class TestPipelinedTransferPerf(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    @skip_pre_hopper
    @pytest.mark.skip_less_device(2)
    @pytest.mark.timeout(3600)
    def test_disagg_latency(self):
        pipelined = os.environ.get("PIPELINED", "0") == "1"
        label = os.environ.get("LABEL", "unlabelled")

        cache_transceiver_config = {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 4096,
            "enable_pipelined_transfer": pipelined,
        }
        ctx_server_config = {
            "max_seq_len": _MAX_SEQ_LEN,
            "max_num_tokens": _MAX_NUM_TOKENS,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": dict(cache_transceiver_config),
            "enable_chunked_prefill": True,
        }
        gen_server_config = {
            "max_seq_len": _MAX_SEQ_LEN,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": dict(cache_transceiver_config),
            "enable_chunked_prefill": True,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "backend": "pytorch",
            # Required by pipelined transfer; kept for the baseline too so the
            # only difference between runs is the flag itself.
            "schedule_style": "generation_first",
            "context_servers": {"num_instances": 1},
            "generation_servers": {"num_instances": 1},
        }

        # One CSV row per slice, written by the worker itself, so chunk counts
        # and per-slice timestamps survive independently of stdout capture.
        perf_dir = os.environ.get("KV_PERF_DIR")
        extra_env = {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": perf_dir} if perf_dir else None

        sampling_params = SamplingParams(max_tokens=8, temperature=0)
        with launch_disaggregated_llm(
            disaggregated_server_config,
            ctx_server_config,
            gen_server_config,
            self.MODEL_PATH,
            extra_env=extra_env,
        ) as llm:
            # Warm up so autotuning and the first-touch KV allocation do not
            # land in the measurement.
            llm.generate_async(_long_prompt(999), sampling_params).result()

            latencies = []
            wall_start = time.perf_counter()
            futures = []
            for i in range(_NUM_REQUESTS):
                start = time.perf_counter()
                fut = llm.generate_async(_long_prompt(i), sampling_params)
                futures.append((start, fut))
            for start, fut in futures:
                fut.result()
                latencies.append(time.perf_counter() - start)
            wall = time.perf_counter() - wall_start

        latencies.sort()
        print(
            f"PERFRESULT label={label} pipelined={int(pipelined)} "
            f"requests={_NUM_REQUESTS} prompt_words={_PROMPT_WORDS} "
            f"wall_s={wall:.3f} "
            f"mean_s={statistics.mean(latencies):.3f} "
            f"median_s={statistics.median(latencies):.3f} "
            f"p90_s={latencies[int(len(latencies) * 0.9) - 1]:.3f} "
            f"min_s={latencies[0]:.3f} max_s={latencies[-1]:.3f}"
        )
