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

from .impl_blocks import MoEEplbWeightLayoutMixin, MoEWeightOwnerMixin
from .impl_contract import MoEDeployment, MoEEligibility, MoEEplbBinding, MoEProblem, MoERunContext

if TYPE_CHECKING:
    from ...utils import Fp4QuantizedTensor
    from .impl_identity import MoEImplDescriptor


class MoEImplBase(MoEWeightOwnerMixin, MoEEplbWeightLayoutMixin, nn.Module, abc.ABC):
    """An execution unit. NOT a complete MoE layer.

    Takes the concrete halves of the two blocks an expert-weight owner needs --
    the weights themselves and the weight-side EPLB layout -- from the mixins,
    which ``MoE`` includes as well. The abstract contract is restated here rather
    than shared, because the two bases do not promise the same thing:
    ``run_moe`` below is deliberately narrower than ``MoE.run_moe``, and only
    this class enforces the contract at construction.

    Method resolution is all the mixins carry across. ``MoE.__init__`` also
    establishes the construction state every backend then reads off ``self``
    (``hidden_size``, ``quant_config``, ``mapping``,
    ``intermediate_size_per_partition``, and the rest), and this class
    establishes none of it -- it takes an ``eplb`` binding and nothing else.
    Swapping a backend's base class therefore needs matching constructor work
    in the same change; it is not a one-line edit.

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
        # Project the binding onto the attribute names the quantization layer
        # reads off the weight owner (``module.expert_size_per_partition`` and
        # friends). Plain attributes rather than properties, because the DWDP
        # fixup rewrites the layout in place after construction.
        self.layer_idx = eplb.layer_idx
        self.num_slots = eplb.num_slots
        self.slot_start = eplb.slot_start
        self.slot_end = eplb.slot_end
        self.expert_size_per_partition = eplb.expert_size_per_partition
        # Lists, not tuples: call sites slice and index these.
        self.initial_local_expert_ids = list(eplb.initial_local_expert_ids)
        self.initial_global_assignments = list(eplb.initial_global_assignments)
        self.layer_load_balancer = eplb.layer_load_balancer

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

    # Narrower than ``MoE.run_moe``, which also takes a keyword-only
    # ``workspace``. Not a drift: the impl already allocates that scratch
    # itself, through ``get_workspaces`` below. What the scheduler still owns is
    # its LIFETIME -- one allocation reused across chunks and alternated
    # between streams, so it outlives a single call and travels back in through
    # the signature. This signature is the state after that lifetime moves
    # inside the impl; impls arriving on this base (TRTLLM-14958,
    # TRTLLM-14960..14969) drop the parameter as they do.
    @abc.abstractmethod
    def run_moe(self, ctx: MoERunContext) -> torch.Tensor: ...

    # ---- impl-owned resources: produced here, never passed in -------------
    def get_workspaces(self, *args: object, **kwargs: object) -> "list[dict] | None":
        return None
