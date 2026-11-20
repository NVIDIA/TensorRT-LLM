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
"""Check that a model supports Wide-EP fault tolerance and bind its MoE layers as it loads.

The model loader calls ``check_supported`` before it builds the model and ``prepare_model`` before
any weight loads. ``create_moe_load_balancer`` creates the EPLB load balancer that restores failed
ranks' experts.
"""

from __future__ import annotations

import contextlib
import functools
import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._wide_ep_fault_tolerance import (
    TURN_FAULT_TOLERANCE_OFF,
    check_fault_tolerance_requirements,
    check_model_architectures,
)

from ..distributed import Distributed
from ..distributed.membership import HANG_TIMEOUT_S
from ..moe.fused_moe.communication.execution_state import MoeExecutionState
from ..moe.fused_moe.communication.nvlink_one_sided import (
    SYNC_PROTOCOL_ENV,
    NVLinkOneSided,
    get_sync_protocol,
)
from ..moe.fused_moe.ep_group_health import EPGroupHealth
from ..moe.fused_moe.eplb_expert_restore import FaultTolerantMoeLoadBalancer
from ..moe.fused_moe.fused_moe_vanilla import VanillaMoE
from ..moe.fused_moe.interface import MoE
from ..moe.fused_moe.moe_comm_timeout_guard import (
    SERVING_TIMEOUT_ENV,
    resolve_moe_comm_timeout_budgets,
)
from ..moe.fused_moe.moe_load_balancer import BaseMoeLoadBalancer, maybe_create_moe_load_balancer

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

    from ..model_config import ModelConfig


def check_supported(llm_args: TorchLlmArgs, mapping: Mapping) -> None:
    """Check, before the model loads, the requirements that LLM argument validation cannot check.

    Raises:
        ValueError: The configuration is not supported with fault tolerance.
    """
    # The authority excludes a hung rank about a hang timeout after it stopped. The all-to-all
    # kernels that wait on the rank must not time out before that.
    kernel_timeout_s = resolve_moe_comm_timeout_budgets(os.environ).serving_seconds
    if kernel_timeout_s is None:
        kernel_timeout_s = torch.ops.trtllm.moe_a2a_get_timeout()
    checks = [
        # The executor creator turns it off after argument validation in these modes.
        (
            not llm_args.disable_overlap_scheduler,
            "the overlap scheduler, which speculative decoding without overlap support (such as "
            "two-model Eagle3 or draft-target) or multimodal encoder-only mode turned off, "
            "because only its loop accepts or discards each step when the next one starts",
        ),
        (
            mapping.moe_ep_size == mapping.world_size,
            f"expert parallelism over every rank (moe_expert_parallel_size={mapping.world_size}, "
            f"not {mapping.moe_ep_size}), because a failed rank must take only whole experts with it",
        ),
        # Resolved from "auto" only as the model loads, after argument validation.
        (
            llm_args.kv_cache_config.use_kv_cache_manager_v2 is True,
            "KV cache manager V2 (kv_cache_config.use_kv_cache_manager_v2=True, not "
            f"{llm_args.kv_cache_config.use_kv_cache_manager_v2!r}), because only it holds a "
            "step's KV reuse commits until the step is accepted",
        ),
        (
            get_sync_protocol() != "sentinel",
            f"the fence path ({SYNC_PROTOCOL_ENV} unset, auto or fence), because only it stops "
            "waiting on a failed rank",
        ),
        (
            kernel_timeout_s >= 2 * HANG_TIMEOUT_S,
            f"an all-to-all timeout of at least twice the hang timeout ({SERVING_TIMEOUT_ENV}"
            f">={2 * HANG_TIMEOUT_S:.0f} s, not {kernel_timeout_s:.0f} s), because a wait on a "
            "hung rank must last until the rank is dropped",
        ),
    ]
    check_fault_tolerance_requirements(checks)


def prepare_model(model: torch.nn.Module) -> None:
    """Check that the built model supports fault tolerance and bind its MoE layers before loading.

    Raises:
        ValueError: The model is not supported with fault tolerance.
    """
    check_model_architectures(model.model_config.pretrained_config.architectures)
    # A model may run its MoE layers over an EP group other than its mapping's MoE parallelism;
    # moe_comms checks that every layer runs over the same one.
    comms = moe_comms(model)
    health = EPGroupHealth(comms[0].ep_size)
    ep_rank = comms[0].ep_rank
    if Distributed.startup_record.replacement:
        # A replacement starts alone. The change that admits it installs the active ranks.
        for rank in range(health.moe_world_size):
            if rank != ep_rank:
                health.mark_failed(rank)
    moe_execution_state = MoeExecutionState(
        health, ep_rank, torch.device("cuda", torch.cuda.current_device())
    )
    for comm in comms:
        comm.bind_execution_state(moe_execution_state)


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
                f"{TURN_FAULT_TOLERANCE_OFF}."
            )
        if first is None:
            first = name, comm
        first_name, first_comm = first
        if (comm.ep_size, comm.ep_rank) != (first_comm.ep_size, first_comm.ep_rank):
            raise ValueError(
                f"MoE layer {name} communicates as EP rank {comm.ep_rank} of {comm.ep_size}, but "
                f"MoE layer {first_name} as EP rank {first_comm.ep_rank} of {first_comm.ep_size}. "
                "Wide-EP fault tolerance runs every MoE layer over one EP group; give the layers "
                f"the same MoE parallelism or {TURN_FAULT_TOLERANCE_OFF}."
            )
        comms[id(comm)] = comm
    if not comms:
        raise ValueError(
            f"Wide-EP fault tolerance requires a model with MoE layers; {TURN_FAULT_TOLERANCE_OFF}"
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


def create_moe_load_balancer(
    model_config: ModelConfig, mapping: Mapping
) -> contextlib.AbstractContextManager[Optional[BaseMoeLoadBalancer]]:
    """Create this rank's EPLB load balancer, which restores failed ranks' experts."""
    return maybe_create_moe_load_balancer(
        model_config,
        mapping,
        functools.partial(FaultTolerantMoeLoadBalancer, mapping=model_config.mapping),
    )
