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

from typing import TYPE_CHECKING

import torch

from ..moe.fused_moe.communication.execution_state import MoeExecutionState
from ..moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from ..moe.fused_moe.ep_group_health import EPGroupHealth
from ..moe.fused_moe.fused_moe_vanilla import VanillaMoE
from ..moe.fused_moe.interface import MoE

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

# The alternative that each error about an unmet fault tolerance requirement offers.
_TURN_FAULT_TOLERANCE_OFF = "turn fault tolerance off (enable_wide_ep_fault_tolerance=False)"


def prepare_model(model: torch.nn.Module) -> None:
    """Bind every MoE layer of ``model`` to one execution state over all EP ranks.

    Raises:
        ValueError: A MoE layer's communication cannot be bound, two MoE layers communicate over
            different EP groups, or the model has no MoE layer.
    """
    # A model may run its MoE layers over an EP group other than its mapping's MoE parallelism;
    # moe_comms checks that every layer runs over the same one.
    comms = moe_comms(model)
    state = MoeExecutionState(
        EPGroupHealth(comms[0].ep_size),
        comms[0].ep_rank,
        torch.device("cuda", torch.cuda.current_device()),
    )
    for comm in comms:
        comm.bind_execution_state(state)


def moe_comms(model: torch.nn.Module) -> list[NVLinkOneSided]:
    """Return the communication strategies that ``model``'s MoE layers use, each once.

    Raises:
        ValueError: A MoE layer's communication cannot be bound to an execution state, or two
            MoE layers communicate over different EP groups.
    """
    comms = {}
    first = None  # The name and communication of the first MoE layer.
    for name, layer in model.named_modules():
        if not isinstance(layer, (MoE, VanillaMoE)):
            continue
        comm = getattr(layer, "comm", None)
        if not isinstance(comm, NVLinkOneSided):
            uses = "no communication strategy" if comm is None else type(comm).__name__
            raise ValueError(
                f"MoE layer {name} ({type(layer).__name__}) uses {uses}, but Wide-EP fault "
                "tolerance requires NVLinkOneSided, the only MoE communication that can leave "
                f"out a failed rank. {_no_nvlink_one_sided_cause(layer.mapping)}; "
                f"{_TURN_FAULT_TOLERANCE_OFF}."
            )
        if first is None:
            first = name, comm
        first_name, first_comm = first
        if (comm.ep_size, comm.ep_rank) != (first_comm.ep_size, first_comm.ep_rank):
            raise ValueError(
                f"MoE layer {name} communicates as EP rank {comm.ep_rank} of {comm.ep_size}, but "
                f"MoE layer {first_name} as EP rank {first_comm.ep_rank} of {first_comm.ep_size}. "
                "Wide-EP fault tolerance runs every MoE layer over one EP group; give the layers "
                f"the same MoE parallelism or {_TURN_FAULT_TOLERANCE_OFF}."
            )
        comms[id(comm)] = comm
    if not comms:
        raise ValueError(
            f"Wide-EP fault tolerance requires a model with MoE layers; {_TURN_FAULT_TOLERANCE_OFF}"
        )
    return list(comms.values())


def _no_nvlink_one_sided_cause(mapping: Mapping) -> str:
    """Name the setting with which a MoE layer of ``mapping`` gets no NVLinkOneSided."""
    if not mapping.enable_attention_dp or mapping.dp_size == 1:
        return (
            "A MoE layer gets it only under attention DP over more than one rank "
            "(enable_attention_dp=True and tensor_parallel_size above 1)"
        )
    if mapping.moe_tp_size != 1:
        return (
            "A MoE layer gets it only without MoE TP (moe_tensor_parallel_size=1, not "
            f"{mapping.moe_tp_size})"
        )
    return (
        "The layer's MoE backend, or the communication method that TRTLLM_FORCE_COMM_METHOD or "
        "the model forces, selected another; use a MoE backend that supports NVLinkOneSided and "
        "leave TRTLLM_FORCE_COMM_METHOD unset or set it to NVLINK_ONE_SIDED"
    )
