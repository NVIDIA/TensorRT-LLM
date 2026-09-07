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

import torch

from tensorrt_llm._torch.attention.backends.fmha.triton_custom_mask import TritonCustomMaskFmha
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs


def test_triton_custom_mask_rejects_whole_request_probe() -> None:
    fmha = object.__new__(TritonCustomMaskFmha)

    assert not fmha.is_supported(
        torch.empty((1, 4)),
        None,
        None,
        SimpleNamespace(),
        AttentionForwardArgs(),
    )
