# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The transform passes to capture the hidden states of the target model."""

from typing import Dict, List, Optional, Set, Tuple, Type

import torch
from torch._ops import OpOverloadPacket
from torch.fx import GraphModule, Node

from .....llmapi.llm_args import KvCacheConfig
from ...custom_ops.attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    MHACallable,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import get_all_layer_subgraphs, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from .kvcache import _InsertCachedOperator


@torch.library.custom_op("auto_deploy::residual_add_for_capture", mutates_args=())
def residual_add_for_capture(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.add(t1, t2)


@residual_add_for_capture.register_fake
def residual_add_for_capture_fake(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.add(t1, t2)


@torch.library.custom_op("auto_deploy::cached_residual_add", mutates_args=())
def cached_residual_add(
    t1: torch.Tensor, t2: torch.Tensor, hidden_states_cache: torch.Tensor
) -> torch.Tensor:
    ret = torch.ops.aten.add(t1, t2)
    b, s, _ = ret.shape
    num_tokens = b * s

    hidden_states_cache[:num_tokens].copy_(ret.view(num_tokens, -1), non_blocking=True)
    return ret


@cached_residual_add.register_fake
def cached_residual_add_fake(
    t1: torch.Tensor, t2: torch.Tensor, hidden_states_cache: torch.Tensor
) -> torch.Tensor:
    return torch.ops.aten.add(t1, t2)


class DetectHiddenStatesForCaptureConfig(TransformConfig):
    """Configuration for the hidden states detection transform."""

    # Whether to capture hidden states at all. If False we will not capture any layers.
    capture_hidden_states: bool = False

    # TODO: figure out how to get layers to capture.
    # We should consider if we can use the layer indices stored in eagle checkpoints, e.g.
    # https://huggingface.co/nvidia/gpt-oss-120b-Eagle3/blob/main/config.json#L9-L14
    eagle3_layers_to_capture: Optional[Set[int]] = None

    def set_default_eagle3_layers_to_capture(self, num_hidden_layers: int):
        """
        Used to set default layers to capture when we want to capture hidden states, but
        no layers to capture are provided.
        """
        if num_hidden_layers <= 6:
            raise ValueError("Not enough hidden layers for default EAGLE3 capture")
        self.eagle3_layers_to_capture = {1, num_hidden_layers // 2 - 1, num_hidden_layers - 4}


@TransformRegistry.register("detect_hidden_states_for_capture")
class DetectHiddenStatesForCapture(BaseTransform):
    """Detect the hidden states we should capture in the graph."""

    config: DetectHiddenStatesForCaptureConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return DetectHiddenStatesForCaptureConfig

    def collect_residual_add_nodes(self, gm: GraphModule) -> Dict[int, Node]:
        def _get_layer_number(lin_node: Node) -> Optional[int]:
            weight = lin_node.args[1]
            if weight.op == "get_attr":
                subnames = weight.target.split(".")
                for subname in subnames:
                    if subname.isdigit():
                        return int(subname)

            return None

        # find last closing linear node of each layer
        # from there we will find the residual add node for that layer
        layer_subgraphs, unprocessed_linear_nodes = get_all_layer_subgraphs(gm)
        residual_add_nodes: Dict[int, Node] = {}
        for layer_subgraph in layer_subgraphs:
            lin_node_closing = layer_subgraph.terminating_node
            # need layer number to correctly identify the residual add node
            layer_number = _get_layer_number(lin_node_closing)
            if layer_number is None:
                continue

            # Conditions to identify as the hidden states after the residual
            # The first node after the linear closing node that satisfies:
            # 1. is an add node with > 1 users (hidden states before the last are used directly by next layer
            # as well as having a residual add to the next hidden state). Stopping here prevents us from
            # using a future residual add node for the next layer.
            # 2. is the last add node in a 1 user chain (for last layer or layers with no following residual add)
            # This stops us before we go to the next layer.
            res_node = lin_node_closing
            while len(res_node.users) == 1:
                user_node = list(res_node.users)[0]
                if not is_op(user_node, torch.ops.aten.add):
                    break
                res_node = user_node

            if is_op(res_node, torch.ops.aten.add):
                # this stores the last residual add node encountered for each layer
                residual_add_nodes[layer_number] = res_node

        return residual_add_nodes

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.capture_hidden_states:
            info = TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
            return gm, info

        if gm.graph.find_nodes(
            op="call_function", target=torch.ops.auto_deploy.residual_add_for_capture.default
        ):
            info = TransformInfo(skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True)
            return gm, info

        residual_add_nodes = self.collect_residual_add_nodes(gm)

        if self.config.eagle3_layers_to_capture is None:
            num_hidden_layers = len(residual_add_nodes)
            self.config.set_default_eagle3_layers_to_capture(num_hidden_layers)

        residual_add_nodes = {
            k: v for k, v in residual_add_nodes.items() if k in self.config.eagle3_layers_to_capture
        }

        assert residual_add_nodes.keys() == self.config.eagle3_layers_to_capture, (
            f"Unable to find residual add nodes for layers. Expected: {self.config.eagle3_layers_to_capture}, \
            Found: {residual_add_nodes.keys()}"
        )

        # replace residual add nodes with special placeholder nodes
        for layer_number, res_node in residual_add_nodes.items():
            with gm.graph.inserting_before(res_node):
                new_node = gm.graph.call_function(
                    torch.ops.auto_deploy.residual_add_for_capture.default,
                    args=res_node.args,
                    kwargs=res_node.kwargs,
                )
            res_node.replace_all_uses_with(new_node)
            gm.graph.erase_node(res_node)

        cnt = len(residual_add_nodes)
        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=(cnt == 0), has_valid_shapes=(cnt == 0)
        )
        return gm, info


class HiddenStatesResourceHandler(ResourceHandler):
    """A resource handler for hidden states."""

    def __init__(self, hidden_size: int, dtype: torch.dtype) -> None:
        """Initialize the HiddenStatesResourceHandler.

        Args:
            hidden_size: The size of the hidden states resource.
            dtype: The dtype of the hidden states resource.
        """
        self.hidden_size = hidden_size
        self.dtype = dtype

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        return torch.empty(
            sequence_info.max_num_tokens,
            self.hidden_size,
            device=sequence_info.device,
            dtype=self.dtype,
        )


@AttentionRegistry.register("cached_residual_add")
class CachedResidualAdd(AttentionDescriptor):
    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 2

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.residual_add_for_capture

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.cached_residual_add

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        hidden_size = source_attn_node.meta["val"].shape[-1]
        hidden_type = source_attn_node.meta["val"].dtype

        return {"hidden_states_cache": HiddenStatesResourceHandler(hidden_size, dtype=hidden_type)}

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return []


@TransformRegistry.register("insert_cached_residual_add")
class InsertCachedResidualAdd(_InsertCachedOperator):
    """A transform to handle residual add cache operations."""
