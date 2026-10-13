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
"""The KV transfer planner the user asks for, bound to the model it will serve.

The user names a planner by setting ``TRTLLM_KV_TRANSFER_PLANNER`` to its
``planner_id``; ``PLANNERS`` maps every id to its class. Unset, every model
keeps the framework's default plan. The named planner must support the
deployment, the model architecture under the parallel topology it is built
for, or binding fails with a clear reason. Both sides of a disaggregated pair
must name the same planner; the framework refuses to connect ranks whose
``planner_id`` differs.

Binding happens where the model is constructed (``AutoModelForCausalLM``),
by installing the planner's factory as the instance's
``build_transfer_planner``; the framework reads that hook and nothing else.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Optional, Type

from tensorrt_llm._torch.disaggregation.base.planner import TransferPlanner

from .deepseek_mla_latent import DeepseekMlaLatentTransferPlanner

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig

PLANNER_ENV_VAR = "TRTLLM_KV_TRANSFER_PLANNER"

#: Every planner a user may name, keyed by its ``planner_id``.
PLANNERS: Dict[str, Type[TransferPlanner]] = {
    p.planner_id: p for p in (DeepseekMlaLatentTransferPlanner,)
}


def bind_transfer_planner(model, model_config: "ModelConfig") -> Optional[Type[TransferPlanner]]:
    """Install the planner named by the environment on ``model``; return it, or None if none is named.

    Raises ``ValueError`` when the name is unknown or the planner does not
    support this deployment.
    """
    key = os.environ.get(PLANNER_ENV_VAR, "")
    if not key:
        return None
    planner_cls = PLANNERS.get(key)
    if planner_cls is None:
        raise ValueError(
            f"{PLANNER_ENV_VAR}={key!r} names no KV transfer planner; "
            f"known planners: {sorted(PLANNERS)}"
        )
    architectures = getattr(model_config.pretrained_config, "architectures", None)
    if not architectures:
        raise ValueError(
            f"{PLANNER_ENV_VAR}={key!r} needs a model architecture, "
            "but the pretrained config names none"
        )
    mapping = model_config.mapping
    if not planner_cls.supports_deployment(architectures[0], mapping):
        raise ValueError(
            f"{planner_cls.__name__} does not support {architectures[0]} with "
            f"tp={mapping.tp_size} pp={mapping.pp_size} cp={mapping.cp_size} "
            f"attention_dp={bool(mapping.enable_attention_dp)}"
        )

    def build_transfer_planner(kv_cache_manager, self_rank_info, self_extractor):
        return planner_cls(self_rank_info, self_extractor)

    model.build_transfer_planner = build_transfer_planner
    return planner_cls


__all__ = ["PLANNER_ENV_VAR", "PLANNERS", "bind_transfer_planner"]
