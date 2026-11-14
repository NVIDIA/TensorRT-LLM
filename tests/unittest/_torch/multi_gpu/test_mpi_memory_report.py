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
"""``LLM.get_memory_report()`` across MPI ranks (TP=2).

Rank 0 answers report requests and, in the memory monitor's full mode,
gathers the other ranks' reports over its own communicator; without full
mode the other ranks are listed as not gathered. Full mode is requested
through ``memory_monitor_config``, which has to reach every worker. Either
way the executor shuts down promptly afterwards: a rank's report listener
must not keep it in MPI.
"""

import time

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, MemoryMonitorConfig, SamplingParams

_MODEL_PATH = str(llm_models_root() / "Qwen3/Qwen3-0.6B")
_SHUTDOWN_BOUND_S = 60.0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("full_mode", [True, False], ids=["full_config", "default"])
def test_every_rank_has_an_entry_and_shutdown_is_prompt(monkeypatch, full_mode):
    # Read by the MPI workers, which inherit TLLM* variables.
    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    extra = {}
    if full_mode:
        extra["memory_monitor_config"] = MemoryMonitorConfig(mode="full")
    llm = LLM(
        model=_MODEL_PATH,
        tensor_parallel_size=2,
        max_batch_size=4,
        max_seq_len=256,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.2),
        **extra,
    )
    try:
        llm.generate(["Hello"], SamplingParams(max_tokens=4))
        reports = llm.get_memory_report()
        assert len(reports) == 2, reports
        assert "rank=0" in reports[0] and "This process uses" in reports[0]
        if full_mode:
            assert "rank=1" in reports[1] and "This process uses" in reports[1]
        else:
            assert reports[1].startswith("rank 1: not gathered")
    finally:
        started = time.monotonic()
        llm.shutdown()
        elapsed = time.monotonic() - started
    assert elapsed < _SHUTDOWN_BOUND_S, f"shutdown took {elapsed:.0f} s"
