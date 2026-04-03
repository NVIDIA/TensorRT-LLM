# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Inject backend-native custom attention masks into ``torch_attention`` nodes."""

from typing import Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..attention_mask_provider import (
    AttentionMaskProviderContext,
    AttentionMaskProviderRegistry,
    infer_model_type,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    Stages,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class InjectCustomAttentionMaskConfig(TransformConfig):
    """Configuration for injecting backend-native custom attention masks."""

    stage: Stages = Field(default=Stages.PATTERN_MATCHER)
    backend: str = Field(
        default="torch_attention",
        description="Backend key used to resolve the attention mask provider.",
    )
    model_type: Optional[str] = Field(
        default=None,
        description="Optional explicit model_type override used for provider lookup.",
    )
    override_existing_mask: bool = Field(
        default=False,
        description="Whether to override an attention node that already has an attn_mask input.",
    )


@TransformRegistry.register("inject_custom_attention_mask")
class InjectCustomAttentionMask(BaseTransform):
    """Inject backend-native masks into ``torch_attention`` calls."""

    config: InjectCustomAttentionMaskConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return InjectCustomAttentionMaskConfig

    @staticmethod
    def _get_attn_mask_arg(node: Node):
        if "attn_mask" in node.kwargs:
            return node.kwargs["attn_mask"]
        if len(node.args) > 3:
            return node.args[3]
        return None

    def _set_attn_mask_arg(self, node: Node, attn_mask: Node) -> None:
        if len(node.args) > 3:
            node.update_arg(3, attn_mask)
            return

        kwargs = dict(node.kwargs)
        kwargs["attn_mask"] = attn_mask
        node.kwargs = kwargs

    def _apply(
        self,
        gm: GraphModule,
        cm: Optional[CachedSequenceInterface],
        factory: Optional[ModelFactory],
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        attn_nodes = [n for n in gm.graph.nodes if is_op(n, torch.ops.auto_deploy.torch_attention)]
        if not attn_nodes:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        model_type = self.config.model_type or infer_model_type(factory)
        provider = AttentionMaskProviderRegistry.get(model_type, self.config.backend)
        if provider is None:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        assert model_type is not None
        ctx = AttentionMaskProviderContext(
            gm=gm,
            cm=cm,
            factory=factory,
            shared_config=shared_config,
            model_type=model_type,
            backend=self.config.backend,
        )

        num_matches = 0
        for attn_node in attn_nodes:
            if (
                not self.config.override_existing_mask
                and self._get_attn_mask_arg(attn_node) is not None
            ):
                continue

            with gm.graph.inserting_before(attn_node):
                attn_mask = provider(ctx, attn_node)

            if attn_mask is None:
                continue

            self._set_attn_mask_arg(attn_node, attn_mask)
            num_matches += 1

        if num_matches == 0:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        return gm, TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )
