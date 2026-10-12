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
"""What a transfer planner identifies a deployment by.

A KV transfer plan depends on the model architecture and on how each side is
sharded, not on which implementation computed the KV: a built-in
``DeepseekV3ForCausalLM`` and a modeling_v2 target for the same checkpoint on
the same topology write the same bytes into the same pools. ``Deployment``
therefore names exactly those two things, so a planner's compatibility matrix
covers both implementations at once and a context on one can feed a
generation on the other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig


@dataclass(frozen=True)
class Deployment:
    """The architecture of a model and the parallel topology it runs under."""

    architecture: str
    tp: int
    pp: int
    cp: int
    moe_ep: int
    moe_tp: int
    attention_dp: bool

    @classmethod
    def of(cls, model_config: "ModelConfig") -> Optional["Deployment"]:
        """The deployment ``model_config`` describes, or None when it names no architecture."""
        architectures = getattr(model_config.pretrained_config, "architectures", None)
        if not architectures:
            return None
        m = model_config.mapping
        return cls(
            architecture=architectures[0],
            tp=m.tp_size,
            pp=m.pp_size,
            cp=m.cp_size,
            moe_ep=m.moe_ep_size,
            moe_tp=m.moe_tp_size,
            attention_dp=bool(m.enable_attention_dp),
        )

    def to_bytes(self) -> bytes:
        """Wire form for ``RankInfo.planner_blob``."""
        return json.dumps(asdict(self), sort_keys=True).encode()

    @classmethod
    def from_bytes(cls, blob: bytes) -> Optional["Deployment"]:
        """Inverse of :meth:`to_bytes`; None for anything that is not one."""
        try:
            return cls(**json.loads(blob.decode()))
        except (ValueError, TypeError, UnicodeDecodeError):
            return None
