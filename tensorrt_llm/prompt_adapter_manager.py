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
"""Compatibility shim for ``tensorrt_llm.prompt_adapter_manager``.

Will be removed once all usages are migrated to
``tensorrt_llm._torch.peft.prompt_adapter``.

DO NOT ADD ANYTHING TO THIS FILE.
"""

import warnings

from tensorrt_llm._torch.peft.prompt_adapter import PromptAdapterManager  # noqa: F401

warnings.warn(
    "tensorrt_llm.prompt_adapter_manager has moved to "
    "tensorrt_llm._torch.peft.prompt_adapter and will be removed in a "
    "future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "PromptAdapterManager",
]
