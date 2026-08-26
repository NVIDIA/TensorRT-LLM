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

import pytest

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


@pytest.mark.cpu_only
def test_nanojet_environment_does_not_enable_torch_compile(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_ENABLE_NANOJET", "1")

    args = TorchLlmArgs(model="/tmp/dummy_model")

    assert args.torch_compile_config is None
