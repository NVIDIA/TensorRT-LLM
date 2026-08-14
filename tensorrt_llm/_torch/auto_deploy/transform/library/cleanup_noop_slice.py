# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("cleanup_noop_slice")
class CleanupNoopSlice(BaseTransform):
    """Remove no-op slice nodes from the graph.

    Those will be nodes that are used to represent a slice operation like ``t[:, :5]``. The graph IR
    will represent it as ``t[:][:5]``, i.e., two nodes and the first slice being a no-op. This
    function gets rid of such instances.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0
        for node in gm.graph.nodes:
            # looking for slice nodes
            if not is_op(node, torch.ops.aten.slice):
                continue
            # only handling this parameter combination for now
            # 4 args will be (input, dim, start, end)
            if len(node.args) != 4 or len(node.kwargs) != 0:
                continue
            # check if dim is just an integer
            if not isinstance(node.args[1], int):
                continue
            # check if the slice op is indeed a no-op
            if node.args[2] != 0 or node.args[3] != torch.iinfo(torch.long).max:
                continue
            # extract input tensor node and remove the slice node
            in_node = node.args[0]
            assert [in_node] == node.all_input_nodes, "Slice node has unexpected input nodes."
            node.replace_all_uses_with(in_node)
            gm.graph.erase_node(node)
            num_matches += 1

        # store info object about the transform
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )

        return gm, info
