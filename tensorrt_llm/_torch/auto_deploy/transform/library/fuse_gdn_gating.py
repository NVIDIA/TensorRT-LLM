# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Graph transform to fuse GDN gating ops from torch source to Triton kernel."""

from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.fla import gdn_gating as _gdn_gating_ops  # noqa: F401 (registers ops)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


@TransformRegistry.register("fuse_gdn_gating")
class FuseGdnGating(BaseTransform):
    """Replaces torch_fused_gdn_gating ops with triton_fused_gdn_gating.

    This transform runs in the post_load_fusion stage and swaps the pure-torch
    source op with a single-kernel Triton implementation, eliminating ~5 kernel
    launches per GDN layer.

    Args:
        gm: Input graph module to transform.

    Returns:
        Transformed graph module with Triton-fused GDN gating operations.
    """

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        target_op = torch.ops.auto_deploy.triton_fused_gdn_gating.default
        cnt = 0

        for node in list(graph.nodes):
            if is_op(node, torch.ops.auto_deploy.torch_fused_gdn_gating):
                with graph.inserting_after(node):
                    new_node: Node = graph.call_function(
                        target_op,
                        args=node.args,
                        kwargs=node.kwargs,
                    )
                    new_node.meta = node.meta.copy()
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
                    cnt += 1

        info = TransformInfo(
            skipped=False,
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )

        return gm, info
