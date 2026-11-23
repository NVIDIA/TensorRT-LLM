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
"""The opt-in TLLM_EXECUTOR_PROXY_SKIP_CUDA_INIT switch of the executor proxy (no GPU).

The proxy creates a CUDA stream before it starts the workers unless the switch is "1". The stream
constructor is replaced by a recorder, so no device is touched.
"""

import pytest

from tensorrt_llm.executor import proxy as proxy_module

_ENV = "TLLM_EXECUTOR_PROXY_SKIP_CUDA_INIT"


@pytest.fixture
def stream_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(proxy_module.torch.cuda, "Stream", lambda *a, **k: calls.append((a, k)))
    return calls


@pytest.mark.parametrize(
    "raw, skip",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("true", False),
        (" 1", False),
        ("1", True),
    ],
)
def test_skip_proxy_cuda_init_parse(monkeypatch, raw, skip):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    assert proxy_module._skip_proxy_cuda_init() is skip


def test_unset_creates_one_stream_as_before(monkeypatch, stream_calls):
    monkeypatch.delenv(_ENV, raising=False)
    proxy_module._init_leader_cuda_stream()
    assert stream_calls == [((), {})]


def test_other_value_creates_one_stream(monkeypatch, stream_calls):
    monkeypatch.setenv(_ENV, "0")
    proxy_module._init_leader_cuda_stream()
    assert stream_calls == [((), {})]


def test_one_skips_the_stream(monkeypatch, stream_calls):
    monkeypatch.setenv(_ENV, "1")
    proxy_module._init_leader_cuda_stream()
    assert stream_calls == []


def test_start_executor_workers_goes_through_the_switch():
    # The worker start path must call the switch, not torch.cuda.Stream directly, so the knob
    # covers the only stream the proxy creates.
    import inspect

    src = inspect.getsource(proxy_module.GenerationExecutorProxy._start_executor_workers)
    assert "_init_leader_cuda_stream()" in src
    assert "torch.cuda.Stream(" not in src
    module_src = inspect.getsource(proxy_module)
    assert module_src.count("torch.cuda.Stream(") == 1
