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

from typing import Literal

import pytest

from tensorrt_llm.llmapi.llm_args import TorchCompileConfig, TorchLlmArgs


class _NonPytorchLlmArgs(TorchLlmArgs):
    backend: Literal["test"] = "test"


@pytest.mark.cpu_only
def test_nanojet_configures_prefill_only_compilation() -> None:
    args = TorchLlmArgs(model="/tmp/dummy_model", use_nanojet=True)

    assert args.encode_only is True
    assert args.attn_backend == "TRTLLM"
    assert args.use_nanojet is True
    assert isinstance(args.torch_compile_config, TorchCompileConfig)


@pytest.mark.cpu_only
def test_nanojet_rejects_multi_gpu() -> None:
    with pytest.raises(ValueError, match="NanoJet currently supports a single GPU only"):
        TorchLlmArgs(
            model="/tmp/dummy_model",
            use_nanojet=True,
            tensor_parallel_size=2,
        )


@pytest.mark.cpu_only
def test_nanojet_rejects_non_pytorch_backend() -> None:
    with pytest.raises(ValueError, match="NanoJet requires backend='pytorch'"):
        _NonPytorchLlmArgs(model="/tmp/dummy_model", use_nanojet=True)
