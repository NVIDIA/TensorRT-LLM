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
"""Abstract base class for MoE execution units."""

import abc
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .impl_contract import MoEDeployment, MoEEligibility, MoEEplbBinding, MoEProblem, MoERunContext

if TYPE_CHECKING:
    from ...utils import Fp4QuantizedTensor
    from .impl_identity import MoEImplDescriptor


class MoEImplBase(nn.Module, abc.ABC):
    """An execution unit. NOT a complete MoE layer.

    Deliberately does NOT inherit ``MoE``. Three consequences the design relies
    on:

    - no ``forward`` / ``forward_impl``, so it cannot be mistaken for a layer;
    - no ``_register_layer``, so double EPLB registration is impossible at the
      type level and needs no runtime guard;
    - ``ABCMeta`` is real, so a missing method fails at CONSTRUCTION -- unlike
      ``MoE`` today, whose ``@abstractmethod`` markers do not bite because
      ``MoE`` is declared without ``ABCMeta``.
    """

    descriptor: "MoEImplDescriptor"  # set by every concrete subclass

    def __init__(self, *, eplb: MoEEplbBinding) -> None:
        super().__init__()
        # Layout is known BEFORE create_weights, because weight shapes depend
        # on it. Passing it here is what makes post-hoc setattr unnecessary.
        self.eplb = eplb

    # ---- selection (pure; no GPU, no env, no import probe) ----------------
    @classmethod
    @abc.abstractmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility: ...

    # ---- weight lifecycle -------------------------------------------------
    @abc.abstractmethod
    def create_weights(self) -> None: ...

    @abc.abstractmethod
    def load_weights(self, weights: list[dict], allow_partial_loading: bool = False) -> None: ...

    # ---- execution --------------------------------------------------------
    @abc.abstractmethod
    def quantize_input(
        self, x: "torch.Tensor | Fp4QuantizedTensor", **kwargs: object
    ) -> "tuple[torch.Tensor, torch.Tensor | None] | dict": ...

    @abc.abstractmethod
    def run_moe(self, ctx: MoERunContext) -> torch.Tensor: ...

    # ---- impl-owned resources: produced here, never passed in -------------
    def get_workspaces(self, *args: object, **kwargs: object) -> "list[dict] | None":
        return None
