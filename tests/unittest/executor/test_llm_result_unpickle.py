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
"""LlmResult on the worker -> proxy response path.

PyTorch-backend responses cross the RPC / IPC boundary pickled as LlmResponse
objects carrying an LlmResult. Unpickling looks up __setstate__ on every
LlmResult it rebuilds; the lookup must resolve on the class instead of falling
into LlmResult.__getattr__, which raises AttributeError reading _result on the
half-built object (CPU-only; no engine needed).
"""

import pickle

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse, LlmResult

pytestmark = pytest.mark.cpu_only


def test_llm_result_unpickles_without_reaching_getattr(monkeypatch):
    result = LlmResult(result=b"serialized-result", py_result=None, is_final=True)
    result.cached_tokens = 7
    payload = pickle.dumps([LlmResponse(request_id=3, result=result, client_id=5)])

    seen = []
    original = LlmResult.__getattr__

    def recording_getattr(self, item):
        seen.append(item)
        return original(self, item)

    monkeypatch.setattr(LlmResult, "__getattr__", recording_getattr)
    (response,) = pickle.loads(payload)

    assert seen == []
    assert (response.request_id, response.client_id) == (3, 5)
    restored = response.result
    assert restored.is_final is True
    assert restored.cached_tokens == 7
    assert restored.__dict__["_result"] == b"serialized-result"
