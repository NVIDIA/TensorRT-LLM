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


"""Graph transformation to automatically add kv cache into fused MHA op."""

import inspect
import operator
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import (
    AttentionDescriptor,
    AttentionRegistry,
    Constant,
    PrepareMetadataCallable,
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import add_graph_input
from ...utils.cuda_mem_tracker import get_mem_info
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class InsertCachedAttentionConfig(TransformConfig):
    """Configuration for the insert cached attention transform."""

    backend: Optional[str] = Field(default=None, description="The attention backend to use.")


class _InsertCachedOperator(BaseTransform):
    """A generic base transform to insert cached operators into the graph module."""

    config: InsertCachedAttentionConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return InsertCachedAttentionConfig

    @property
    def attn_descriptor(self) -> Type[AttentionDescriptor]:
        return AttentionRegistry.get(self.config.backend)

    def _process_metadata_std(self, gm: GraphModule, cm: CachedSequenceInterface) -> List[Node]:
        """Process the standard metadata nodes."""
        return [
            self._add_or_retrieve_input(gm, cm, arg_name)
            for arg_name in self.attn_descriptor.get_standard_metadata_args()
        ]

    def _insert_extra_metadata_op(
        self,
        gm: GraphModule,
        prep_meta_op: PrepareMetadataCallable,
        inputs_for_prep_meta: List[Node],
        const_args: List[Constant],
        num_meta_out: int,
    ) -> List[Node]:
        # add the computed extra metadata nodes to the graph and add to meta for cached attention op
        meta_nodes_extra = []
        node_last_input = gm.graph.find_nodes(op="placeholder", sort=True)[-1]
        with gm.graph.inserting_before(node_last_input.next):
            ret_node = gm.graph.call_function(
                prep_meta_op, args=(*inputs_for_prep_meta, *const_args)
            )
            for idx in range(num_meta_out):
                meta_extra_node = gm.graph.call_function(operator.getitem, args=(ret_node, idx))
                meta_nodes_extra.append(meta_extra_node)

        return meta_nodes_extra

    def _process_metadata_extra(
        self, gm: GraphModule, cm: CachedSequenceInterface, any_source_attn_node: Node
    ) -> List[Node]:
        """Process the get_metadata function into an op and return node references."""
        # get the metadata op for extra metadata and number of return values
        prep_meta_op, num_meta_out, const_args = (
            self.attn_descriptor.get_prepare_extra_metadata_info(any_source_attn_node)
        )

        # if there is no extra metadata op or no return values, we can return early
        if prep_meta_op is None or num_meta_out == 0:
            return []

        # check what inputs the extra metadata op expects
        inputs_for_prep_meta = [
            self._add_or_retrieve_input(gm, cm, arg.name)
            for arg in prep_meta_op._schema.arguments
            if arg.name in cm.info.available_args
        ]

        return self._insert_extra_metadata_op(
            gm, prep_meta_op, inputs_for_prep_meta, const_args, num_meta_out
        )

    def _process_metadata_host(self, cm: CachedSequenceInterface):
        """Process the host-side prepare metadata function."""
        prep_meta_host_op = self.attn_descriptor.get_host_prepare_metadata_function()
        if prep_meta_host_op is None:
            return

        # analyze the args of the host-side prepare metadata function using inspect
        sig = inspect.signature(prep_meta_host_op)
        args = sig.parameters.keys()

        # check if all args are available in the cached sequence interface
        unavailable_args = args - cm.info.available_args
        assert not unavailable_args, f"Missing args in SequenceInfo: {unavailable_args=}"

        # add the host-side prepare metadata function to the graph
        cm.info.register_host_prepare_for_attention_forward(prep_meta_host_op, list(args))

    def _process_cache_node(self, gm: GraphModule, cache_name: str) -> Node:
        """Process the cache nodes by inserting a cached attention replacement op."""
        return add_graph_input(gm, cache_name)

    def _insert_cached_attn_node(
        self,
        gm: GraphModule,
        attn_node: Node,
        qkv_nodes: List[Node],
        meta_nodes_std: List[Node],
        meta_nodes_extra: List[Node],
        cache_nodes: List[Node],
        constants: List[Constant],
    ):
        """Insert a cached attention node into the graph."""
        with gm.graph.inserting_before(attn_node):
            cached_attn_node = gm.graph.call_function(
                self.attn_descriptor.get_cached_attention_op(),
                args=(
                    *qkv_nodes,
                    *meta_nodes_std,
                    *meta_nodes_extra,
                    *cache_nodes,
                    *constants,
                ),
            )
        attn_node.replace_all_uses_with(cached_attn_node)
        gm.graph.erase_node(attn_node)

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        """Replace uncached source attention node with corresponding cached attn node."""
        attn_descriptor = self.attn_descriptor

        # Get all attention nodes and their info objects
        source_op = attn_descriptor.get_source_attention_op()

        # look for relevant source attention nodes
        source_attn_nodes = [n for n in gm.graph.nodes if is_op(n, source_op)]

        if not source_attn_nodes:
            # If there are no nodes for kv cache insertion found, return current graph
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # get standard metadata nodes for all source attention nodes
        meta_nodes_std = self._process_metadata_std(gm, cm)

        # insert metadata computation and extract each argument as a node
        meta_nodes_extra = self._process_metadata_extra(gm, cm, source_attn_nodes[0])

        # Register host-side prepare_metadata function for attention descriptor.
        self._process_metadata_host(cm)

        # replace fused attention node with attention node that has kv cache
        num_cached_attn_replacements = 0
        for idx, attn_node in enumerate(source_attn_nodes):
            # pick out GEMMs
            qkv = attn_node.args[: attn_descriptor.get_num_qkv_args()]

            # setup + store cache initializers and caches as input nodes
            cache_in_nodes = []
            for k, resource_handler in attn_descriptor.get_cache_initializers(
                attn_node, cm.kv_cache_config
            ).items():
                k_indexed = f"{k}_{idx}"
                cm.add_resource(k_indexed, resource_handler)
                cache_in_nodes.append(self._process_cache_node(gm, k_indexed))

            # retrieve constants for attention_op
            constants = attn_descriptor.get_constants(attn_node)

            # insert cached attention replacement op
            self._insert_cached_attn_node(
                gm,
                attn_node,
                qkv,
                meta_nodes_std,
                meta_nodes_extra,
                cache_in_nodes,
                constants,
            )

            num_cached_attn_replacements += 1

        info = TransformInfo(
            skipped=False,
            num_matches=num_cached_attn_replacements,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info


@TransformRegistry.register("insert_cached_attention")
class InsertCachedAttention(_InsertCachedOperator):
    """A transform to insert cached attention into the graph module."""

    def _apply(self, *args, **kwargs):
        if self.config.backend == "triton":
            self._log_warning(
                "'triton' backend only supports GREEDY sampling (top_k=1). "
                "Please set top_k=1 in the sampling parameters to ensure cohesive output."
            )
        return super()._apply(*args, **kwargs)


@TransformRegistry.register("insert_cached_mla_attention")
class InsertCachedMLAAttention(_InsertCachedOperator):
    """A transform to insert cached MLA attention into the graph module."""


@TransformRegistry.register("resize_kv_cache")
class ResizeKVCache(BaseTransform):
    """Resize the KV cache to occupy available GPU memory.

    This implements the two-phase approach:
    1. Run a forward pass to allocate intermediate memory (activations, workspaces, etc.)
    2. Call resize_kv_cache_manager() to recreate KVCacheManager with optimal capacity
    """

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # check if we need a resize or not
        if not cm.needs_resize():
            return mod, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Run a forward pass to get the extra memory usage
        cm.info.set_max_num_tokens_sample()
        try:
            mod(**cm.named_args)
        except torch.OutOfMemoryError as e:
            self._log_info(
                f"OutOfMemoryError in forward pass while trying to resize the kv-cache:\n{e}"
            )
            raise e

        # NOTE: use fragmented memory without empty cache (peak forward memory + fragmented memory)
        # as a proxy for the memory reserved for the forward pass. This is a rough estimate and
        # may not be accurate.
        *_, mem_reserved_for_forward = get_mem_info(empty_cache=False, unit="B")

        # Resize - KVCacheManager will compute optimal capacity based on free memory
        cm.resize_kv_cache_manager(mem_reserved_for_forward)

        info = TransformInfo(
            skipped=False,
            num_matches=0,
            is_clean=True,
            has_valid_shapes=True,
        )

        return mod, info


@TransformRegistry.register("initialize_cache")
class InitializeCache(BaseTransform):
    """Initialize KV caches using KVCacheManager.

    Gets kv_cache_config from shared_config.ad_config and creates the KVCacheManager
    in estimation mode with conservative capacity. The ResizeKVCache transform will
    later recreate it with optimal capacity after measuring memory usage.
    """

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # Initialize with estimation mode
        # This allows resize_kv_cache to recreate with correct capacity after measuring memory
        num_caches = cm.initialize_resources()

        info = TransformInfo(
            skipped=False, num_matches=num_caches, is_clean=True, has_valid_shapes=True
        )
        return mod, info
