# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Backend that uses torch.compile only."""

import torch
import torch.nn as nn

from ...utils.logger import ad_logger
from ..compiler import CompileBackendRegistry, CompilerBackend


@CompileBackendRegistry.register("torch-compile")
class TorchCompileCompiler(CompilerBackend):
    def __init__(self, *args_for_init, **kwargs_for_init):
        super().__init__(*args_for_init, **kwargs_for_init)
        ad_logger.info(f"Torch Dynamo cache size limit {torch._dynamo.config.cache_size_limit=}")

    def compile(self) -> nn.Module:
        """Compile the model using torch.compile."""
        return torch.compile(self.model, dynamic=True)
