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
"""The masked multi-head attention library of the native attention op."""

import torch

from tensorrt_llm.bindings.internal import thop

from .native import NativeFmha
from .phased import FmhaParams


class MmhaFmha(NativeFmha):
    """Masked multi-head attention: generation for what XQA and trtllm-gen decline."""

    op_is_supported = staticmethod(thop.mmha_is_supported)

    @torch.compiler.disable
    def run_generation(self, params: FmhaParams) -> None:
        thop.mmha_run_generation(params)


__all__ = ["MmhaFmha"]
