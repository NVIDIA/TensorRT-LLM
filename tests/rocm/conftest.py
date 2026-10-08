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
"""Standalone ROCm tests: invoke pytest with --confcutdir=tests/rocm."""

import os

os.environ["TRTLLM_BACKEND"] = "rocm"
os.environ["TRTLLM_NO_USAGE_STATS"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

from tensorrt_llm.rocm.llm import LLM  # noqa: E402
from tensorrt_llm.rocm.validation import tiny_model_and_tokenizer  # noqa: E402

# Keep the tiny CPU fixtures fast and deterministic on shared CI hosts.
torch.set_num_threads(1)


@pytest.fixture
def engine():
    model, tokenizer = tiny_model_and_tokenizer()
    with LLM(model, tokenizer, device="cpu", dtype="float32", max_batch_size=2) as instance:
        yield instance
