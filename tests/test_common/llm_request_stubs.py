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

from types import SimpleNamespace
from unittest.mock import Mock

from test_common.interface_checks import verify_instance


def make_llm_request_stub(**fields: object) -> SimpleNamespace:
    """Supply the request fields used by cache retention and LoRA handling."""
    request = SimpleNamespace(kv_cache_retention_config=None, py_lora_ckpt_source=None)
    verify_instance(
        request,
        "tensorrt_llm._torch.pyexecutor.llm_request:LlmRequest",
        members=("kv_cache_retention_config", "py_lora_ckpt_source"),
    )
    # The check covers the two shared fields. Tests supply the rest of the fixture.
    vars(request).update(fields)
    return request


def make_llm_request_mock() -> Mock:
    """Supply those request fields for a test that also needs mock methods."""
    request = Mock()
    request.kv_cache_retention_config = None
    request.py_lora_ckpt_source = None
    return verify_instance(
        request,
        "tensorrt_llm._torch.pyexecutor.llm_request:LlmRequest",
        members=("kv_cache_retention_config", "py_lora_ckpt_source"),
    )
