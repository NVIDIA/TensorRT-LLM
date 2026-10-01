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
"""Tests for the PyTorch thread cap of multimodal input-processor workers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.serve.openai_server import _multimodal_processor_torch_initializer

pytestmark = pytest.mark.cpu_only

_MAX_THREADS = 16


def _generator(gather_generation_logits: bool = False) -> SimpleNamespace:
    return SimpleNamespace(args=SimpleNamespace(gather_generation_logits=gather_generation_logits))


@pytest.fixture
def multimodal_processor(monkeypatch) -> Mock:
    # The GPU worker runs in a separate process unless this is set.
    monkeypatch.delenv("TLLM_WORKER_USE_SINGLE_PROCESS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    processor = Mock(spec=BaseMultimodalInputProcessor)
    processor.frontend_torch_threads = _MAX_THREADS
    return processor


def test_caps_multimodal_processor_threads(monkeypatch, multimodal_processor):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 72)
    observed = []
    monkeypatch.setattr(torch, "set_num_threads", observed.append)

    initializer = _multimodal_processor_torch_initializer(_generator(), multimodal_processor)

    assert initializer is not None
    initializer()
    assert observed == [_MAX_THREADS]


@pytest.mark.parametrize(
    "single_process,gather_generation_logits,current_threads,frontend_torch_threads,omp_threads",
    [
        (True, False, 72, _MAX_THREADS, None),
        (False, True, 72, _MAX_THREADS, None),
        (False, False, _MAX_THREADS, _MAX_THREADS, None),
        (False, False, 72, None, None),
        (False, False, 72, _MAX_THREADS, "72"),
    ],
    ids=[
        "single_process_env",
        "gather_generation_logits",
        "within_cap",
        "processor_not_opted_in",
        "omp_num_threads_set",
    ],
)
def test_keeps_thread_count(
    monkeypatch,
    multimodal_processor,
    single_process,
    gather_generation_logits,
    current_threads,
    frontend_torch_threads,
    omp_threads,
):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1" if single_process else "0")
    if omp_threads is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", omp_threads)
    multimodal_processor.frontend_torch_threads = frontend_torch_threads
    monkeypatch.setattr(torch, "get_num_threads", lambda: current_threads)
    observed = []
    monkeypatch.setattr(torch, "set_num_threads", observed.append)

    initializer = _multimodal_processor_torch_initializer(
        _generator(gather_generation_logits), multimodal_processor
    )

    assert initializer is None
    assert observed == []
