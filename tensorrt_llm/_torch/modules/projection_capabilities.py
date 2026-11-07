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

from typing import Optional

import torch
from torch import nn


class OutputAllReduceProjection(nn.Module):
    """A row-parallel projection that may be built with the all-reduce of its partial sums
    (``output_all_reduce``).

    One built with it returns the reduced output from every forward (or, where it was built
    to, the all-reduce left in its workspace, ``MnnvlWorkspaceResult``), so the module
    that made that all-reduce does not run it again (:func:`reduces_output`).
    """

    @property
    def output_all_reduce(self) -> Optional[nn.Module]:
        raise NotImplementedError


def reduces_output(projection: nn.Module) -> bool:
    """Whether ``projection`` returns the all-reduce of its partial sums."""
    return (
        isinstance(projection, OutputAllReduceProjection)
        and projection.output_all_reduce is not None
    )


class GatedEpilogueProjection(nn.Module):
    """A projection that applies an output gate inside its own GEMMs.

    ``forward_sigmoid(x)`` equals ``self(x).sigmoid()`` and
    ``forward_gated(x, gate)`` equals ``self(x * gate)``.
    """

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PrequantizedInputProjection(nn.Module):
    """A projection that can consume an input its producer already quantized.

    ``takes_prequantized_input(num_rows)`` says whether a producer should hand
    ``forward_prequantized`` the quantized form of ``num_rows`` rows (an
    activation and its scale, in the layout the projection documents) instead
    of the rows themselves; there ``forward_prequantized`` equals ``self(rows)``.
    ``stages_weights(num_rows)`` says whether ``forward_prequantized`` with
    ``stage_weights`` holds its weights on chip before it waits for its input,
    so that the producer may release it at the producer's entry (same result).
    """

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether ``forward_prequantized`` serves every row count."""
        raise NotImplementedError

    def takes_prequantized_input(self, num_rows: int) -> bool:
        raise NotImplementedError

    def stages_weights(self, num_rows: int) -> bool:
        return False

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        stage_weights: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError
