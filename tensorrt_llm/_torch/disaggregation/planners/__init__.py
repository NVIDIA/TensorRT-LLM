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

Each planner class judges deployments itself through
``supports_deployment(architecture, mapping)``: the model architecture and the
parallel topology are the whole input, so the built-in model and any
modeling_v2 target of the same architecture are judged alike. Models declare
nothing. This module is the one place that knows every planner, and it answers
one question: which planner supports the deployment a model instance was built
for? None means the engine keeps the framework's default plan.

Binding happens where the model is constructed (``AutoModelForCausalLM``),
by installing the covering planner's factory as the instance's
``build_transfer_planner``; the framework reads that hook and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

from tensorrt_llm._torch.disaggregation.base.planner import TransferPlanner

from .deepseek_mla_latent import DeepseekMlaLatentTransferPlanner

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

#: Every planner a deployment may be served by. Order carries no meaning: a
#: deployment two planners support is a configuration error, not a priority
#: question.
PLANNERS: tuple = (DeepseekMlaLatentTransferPlanner,)


def planner_for(architecture: str, mapping: "Mapping") -> Optional[Type[TransferPlanner]]:
    """The planner that supports ``architecture`` sharded as ``mapping`` says, or None."""
    found = [p for p in PLANNERS if p.supports_deployment(architecture, mapping)]
    if len(found) > 1:
        names = ", ".join(p.__name__ for p in found)
        raise ValueError(f"{architecture} is supported by more than one planner: {names}")
    return found[0] if found else None


def bind_transfer_planner(model, model_config: "ModelConfig") -> Optional[Type[TransferPlanner]]:
    """Install the supporting planner's factory on ``model``; return the planner or None.

    The factory has the ``build_transfer_planner`` signature the framework calls.
    """
    architectures = getattr(model_config.pretrained_config, "architectures", None)
    if not architectures:
        return None
    planner_cls = planner_for(architectures[0], model_config.mapping)
    if planner_cls is None:
        return None

    def build_transfer_planner(kv_cache_manager, self_rank_info, self_extractor):
        return planner_cls(self_rank_info, self_extractor)

    model.build_transfer_planner = build_transfer_planner
    return planner_cls


__all__ = ["PLANNERS", "bind_transfer_planner", "planner_for"]
