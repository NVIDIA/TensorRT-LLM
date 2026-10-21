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
"""Optional capabilities of bias-free projection modules.

A projection module declares a capability by deriving from its class here;
attention modules test for the class rather than for a concrete projection
type, so they need not know which module implements it.
"""

import torch
from torch import nn


class GatedEpilogueProjection(nn.Module):
    """A projection that applies an output gate inside its own GEMMs.

    ``forward_sigmoid(x)`` equals ``self(x).sigmoid()`` and
    ``forward_gated(x, gate)`` equals ``self(x * gate)``.
    """

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
