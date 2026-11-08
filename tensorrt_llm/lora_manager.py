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
"""Compatibility shim for ``tensorrt_llm.lora_manager``.

Will be removed once all usages are migrated to
``tensorrt_llm._torch.peft.lora.config`` and
``tensorrt_llm._torch.peft.lora.loaders`` and
``tensorrt_llm._torch.peft.lora.manager``.

DO NOT ADD ANYTHING TO THIS FILE.
"""

import warnings

from tensorrt_llm._torch.peft.lora.config import LoraConfig  # noqa: F401
from tensorrt_llm._torch.peft.lora.loaders import HfLoraLoader, NemoLoraLoader  # noqa: F401
from tensorrt_llm._torch.peft.lora.manager import (  # noqa: F401
    LoraManager,
    LoraModelConfig,
    load_torch_lora,
    load_torch_nemo_lora,
)

warnings.warn(
    "tensorrt_llm.lora_manager has moved to "
    "tensorrt_llm._torch.peft.lora.config and "
    "tensorrt_llm._torch.peft.lora.loaders and "
    "tensorrt_llm._torch.peft.lora.manager and will be removed in a "
    "future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "HfLoraLoader",
    "LoraConfig",
    "LoraManager",
    "LoraModelConfig",
    "NemoLoraLoader",
    "load_torch_lora",
    "load_torch_nemo_lora",
]
