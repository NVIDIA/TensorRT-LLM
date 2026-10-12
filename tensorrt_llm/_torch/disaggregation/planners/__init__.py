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
"""Which disaggregated KV transfer planner serves a deployment.

A planner owns a compatibility set: the (context deployment, generation
deployment) pairs it can move bytes between, declared on the class as
``COVERS``. A :class:`Deployment` is the architecture plus the parallel
topology, so one entry covers the built-in model and any modeling_v2 target
alike. Models declare nothing. This module is the one place that knows every
planner, and it answers one question: given the deployment a model instance
was built for, which planner covers it? None means the engine keeps the
framework's default plan.

Binding happens where the model is constructed (``AutoModelForCausalLM``),
by installing the covering planner's factory as the instance's
``build_transfer_planner``; the framework reads that hook and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

from tensorrt_llm._torch.disaggregation.base.planner import TransferPlanner

from .deepseek_mla_latent import DeepseekMlaLatentTransferPlanner
from .deployment import Deployment

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig

#: Every planner a deployment may be served by. Order carries no meaning: a
#: deployment covered by two planners is a configuration error, not a priority
#: question.
PLANNERS: tuple = (DeepseekMlaLatentTransferPlanner,)


def planner_for(deployment: Deployment) -> Optional[Type[TransferPlanner]]:
    """The planner whose matrix mentions ``deployment`` on either side, or None."""
    found = [p for p in PLANNERS if any(deployment in pair for pair in p.COVERS)]
    if len(found) > 1:
        names = ", ".join(p.__name__ for p in found)
        raise ValueError(f"{deployment} is covered by more than one planner: {names}")
    return found[0] if found else None


def bind_transfer_planner(model, model_config: "ModelConfig") -> Optional[Type[TransferPlanner]]:
    """Install the covering planner's factory on ``model``; return the planner or None.

    The factory has the ``build_transfer_planner`` signature the framework
    calls, closing over the deployment so the planner can publish it.
    """
    deployment = Deployment.of(model_config)
    if deployment is None:
        return None
    planner_cls = planner_for(deployment)
    if planner_cls is None:
        return None

    def build_transfer_planner(kv_cache_manager, self_rank_info, self_extractor):
        return planner_cls(self_rank_info, self_extractor, self_deployment=deployment)

    model.build_transfer_planner = build_transfer_planner
    return planner_cls


__all__ = ["PLANNERS", "Deployment", "bind_transfer_planner", "planner_for"]
