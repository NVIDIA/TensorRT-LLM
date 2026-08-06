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

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor_creator import _get_draft_llm_args
from tensorrt_llm.llmapi.llm_args import LoadFormat, MoeConfig


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("draft_backend", "expected_backend"),
    [
        (None, "TRTLLM"),
        ("AUTO", "AUTO"),
        ("CUTLASS", "CUTLASS"),
    ],
)
def test_get_draft_llm_args_moe_backend(draft_backend, expected_backend):
    llm_args = SimpleNamespace(
        moe_config=MoeConfig(backend="TRTLLM"),
        load_format=LoadFormat.AUTO,
    )
    spec_config = SimpleNamespace(
        moe_backend=draft_backend,
        load_format=None,
    )

    draft_llm_args = _get_draft_llm_args(llm_args, spec_config)

    assert draft_llm_args is not llm_args
    assert draft_llm_args.moe_config.backend == expected_backend
    assert llm_args.moe_config.backend == "TRTLLM"
    if draft_backend is None:
        assert draft_llm_args.moe_config is llm_args.moe_config
    else:
        assert draft_llm_args.moe_config is not llm_args.moe_config


@pytest.mark.cpu_only
def test_get_draft_llm_args_load_format():
    llm_args = SimpleNamespace(moe_config=MoeConfig(), load_format=LoadFormat.AUTO)
    spec_config = SimpleNamespace(moe_backend=None, load_format="dummy")

    draft_llm_args = _get_draft_llm_args(llm_args, spec_config)

    assert draft_llm_args.load_format == LoadFormat.DUMMY
    assert llm_args.load_format == LoadFormat.AUTO
