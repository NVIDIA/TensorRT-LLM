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
"""Mixed backend with torch.compile + Cudagraph."""

import torch
import torch.nn as nn

from ...utils.logger import ad_logger
from ..compiler import CompileBackendRegistry
from .torch_cudagraph import TorchCudagraphCompiler


@CompileBackendRegistry.register("torch-opt")
class TorchOptCompiler(TorchCudagraphCompiler):
    """Compiler that uses both torch.compile and CUDA graphs."""

    def __init__(self, *args_for_init, **kwargs_for_init):
        super().__init__(*args_for_init, **kwargs_for_init)
        torch._dynamo.config.recompile_limit = max(
            len(self.cuda_graph_batch_sizes), torch._dynamo.config.recompile_limit
        )
        ad_logger.info(
            f"Setting Torch Dynamo recompile limit {torch._dynamo.config.recompile_limit=}; "
            f"{torch._dynamo.config.cache_size_limit=}"
        )

    def compile(self) -> nn.Module:
        self.model = torch.compile(self.model, dynamic=True)
        return super().compile()
