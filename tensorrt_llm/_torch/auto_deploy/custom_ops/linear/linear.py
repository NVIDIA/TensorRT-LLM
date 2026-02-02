# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom ops for linear layers."""

from typing import Optional

import torch


@torch.library.custom_op("auto_deploy::torch_linear_simple", mutates_args=())
def simple(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """A wrapper for the linear functional to control how it is exposed.

    By default F.linear (used in linear layers) will be represented as a call to
    torch.ops.aten.linear.default wrapped with two view ops to flatten/unflatten multiple batch
    dimensions into one batch dimension.

    This wrapper avoids exposing this view op during the export graph.
    """
    return torch.ops.aten.linear(input, weight, bias)


@simple.register_fake
def simple_fake(input, weight, bias):
    """Fake implementation of simple_linear."""
    return torch.ops.aten.linear(input, weight, bias)
