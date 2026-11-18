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
"""Bind the MoE layers of a model to one execution state as it loads, under Wide-EP fault tolerance.

The model loader calls ``prepare_model`` once the model is built, before any weight loads.
"""

from __future__ import annotations

from typing import List

import torch

from ..moe.fused_moe.communication.execution_state import MoeExecutionState
from ..moe.fused_moe.ep_group_health import EPGroupHealth
from ..moe.fused_moe.fused_moe_vanilla import VanillaMoE
from ..moe.fused_moe.interface import MoE

# The alternative that each error about an unmet fault tolerance requirement offers.
_TURN_FAULT_TOLERANCE_OFF = "turn fault tolerance off (enable_wide_ep_fault_tolerance=False)"


def prepare_model(model: torch.nn.Module) -> None:
    """Bind every MoE layer of ``model`` to one execution state over all EP ranks.

    Raises:
        ValueError: A MoE layer's communication cannot be bound, or the model has no MoE layer.
    """
    comms = moe_comms(model)
    mapping = model.model_config.mapping
    state = MoeExecutionState(
        EPGroupHealth(mapping.moe_ep_size),
        mapping.moe_ep_rank,
        torch.device("cuda", torch.cuda.current_device()),
    )
    for comm in comms:
        comm.bind_execution_state(state)


def moe_comms(model: torch.nn.Module) -> List[object]:
    """Return the communication strategies that ``model``'s MoE layers use, each once.

    Raises:
        ValueError: A MoE layer's communication cannot be bound to an execution state.
    """
    comms = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, (MoE, VanillaMoE)):
            continue
        comm = getattr(layer, "comm", None)
        if not hasattr(comm, "bind_execution_state"):
            uses = "no communication strategy" if comm is None else type(comm).__name__
            raise ValueError(
                f"MoE layer {name} ({type(layer).__name__}) uses {uses}, but Wide-EP fault "
                "tolerance requires NVLinkOneSided, the only MoE communication that can leave "
                f"out a failed rank. Select it or {_TURN_FAULT_TOLERANCE_OFF}."
            )
        comms[id(comm)] = comm
    if not comms:
        raise ValueError(
            f"Wide-EP fault tolerance requires a model with MoE layers; {_TURN_FAULT_TOLERANCE_OFF}"
        )
    return list(comms.values())
