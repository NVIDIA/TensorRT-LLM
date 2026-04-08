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

"""Registry for model-owned semantic attention mask lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.fx import Node

from ..custom_ops.attention import multimodal_mask  # noqa: F401
from ..custom_ops.attention_interface import Constant, PrepareMetadataCallable
from ..utils.node_utils import is_op


@dataclass(frozen=True)
class SemanticMaskLoweringSpec:
    prepare_op: PrepareMetadataCallable
    num_outputs: int
    const_args: Tuple[Constant, ...] = ()


class SemanticMaskRegistry:
    """Registry mapping source semantic mask ops to backend-specific prep ops."""

    _registry: Dict[Tuple[object, str], SemanticMaskLoweringSpec] = {}

    @classmethod
    def register(cls, source_op: object, backend: str, spec: SemanticMaskLoweringSpec) -> None:
        cls._registry[(source_op, backend)] = spec

    @classmethod
    def get(cls, node: Optional[Node], backend: str) -> Optional[SemanticMaskLoweringSpec]:
        if node is None:
            return None
        for (source_op, registered_backend), spec in cls._registry.items():
            if registered_backend == backend and is_op(node, source_op):
                return spec
        return None

    @classmethod
    def get_source_op(cls, node: Optional[Node]) -> Optional[object]:
        if node is None:
            return None
        for source_op, _ in cls._registry:
            if is_op(node, source_op):
                return source_op
        return None

    @classmethod
    def get_supported_backends(cls, node: Optional[Node]) -> List[str]:
        source_op = cls.get_source_op(node)
        if source_op is None:
            return []
        return sorted(
            {backend for registered_op, backend in cls._registry if registered_op == source_op}
        )

    @classmethod
    def required_inputs(cls, spec: SemanticMaskLoweringSpec) -> List[str]:
        return [arg.name for arg in spec.prepare_op._schema.arguments]


_GEMMA4_PREP_SPEC = SemanticMaskLoweringSpec(
    prepare_op=torch.ops.auto_deploy.gemma4_prepare_multimodal_mask.default,
    num_outputs=1,
)

SemanticMaskRegistry.register(
    torch.ops.auto_deploy.gemma4_multimodal_mask,
    "torch",
    _GEMMA4_PREP_SPEC,
)
SemanticMaskRegistry.register(
    torch.ops.auto_deploy.gemma4_multimodal_mask,
    "triton_paged",
    _GEMMA4_PREP_SPEC,
)
