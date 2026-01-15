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
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import (
    AttentionDescriptor,
    AttentionRegistry,
    CacheConfig,
    Constant,
    PrepareMetadataCallable,
)
from ...distributed.common import all_gather_object, get_world_size
from ...distributed.common import is_initialized as is_distributed_initialized
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import add_graph_input
from ...utils.cuda_mem_tracker import get_mem_info_in_mb
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
    cache_config: CacheConfig = Field(
        default_factory=CacheConfig, description="The custom cache configuration to use."
    )


@TransformRegistry.register("insert_cached_attention")
class InsertCachedAttention(BaseTransform):
    """
    A transform to insert cached attention into the graph module."""

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
        buffer_nodes: List[Node],
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
                    *buffer_nodes,
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

        # run field-wise or to combine the cache config from the transform and the factory
        # the transform config takes precedence over the factory config
        cache_config = self.config.cache_config | factory.get_cache_config()

        # Get all attention nodes and their info objects
        source_op = attn_descriptor.get_source_attention_op()

        # look for relevant source attention nodes
        source_attn_nodes = [n for n in gm.graph.nodes if is_op(n, source_op)]

        if not source_attn_nodes:
            # If there are no nodes for kv cache insertion found, return current graph
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Sanity check
        if cm.info.is_paged:
            assert attn_descriptor.is_paged(), "Paged sequence info requires paged attention op."

        # get standard metadata nodes for all source attention nodes
        meta_nodes_std = self._process_metadata_std(gm, cm)

        # insert metadata computation and extract each argument as a node
        meta_nodes_extra = self._process_metadata_extra(gm, cm, source_attn_nodes[0])

        # Register host-side prepare_metadata function for attention descriptor.
        self._process_metadata_host(cm)

        buffer_in_lookup: Dict[str, Node] = {}

        # replace fused attention node with attention node that has kv cache
        num_cached_attn_replacements = 0
        for idx, attn_node in enumerate(source_attn_nodes):
            # pick out GEMMs
            qkv = attn_node.args[: attn_descriptor.get_num_qkv_args()]

            # setup + store cache initializers and caches as input nodes
            cache_in_nodes = []
            for k, get_cache in attn_descriptor.get_cache_initializers(
                attn_node, cache_config
            ).items():
                k_indexed = f"{k}_{idx}"
                cm.add_cache(k_indexed, get_cache)
                cache_in_nodes.append(self._process_cache_node(gm, k_indexed))

            # setup + store global buffer initializers and buffers as input nodes
            # NOTE: we have to check against existing keys to make sure nothing is registered twice...
            buffer_in_nodes = []
            for k, get_buffer in attn_descriptor.get_global_buffer_initializers(attn_node).items():
                if k not in buffer_in_lookup:
                    cm.add_cache(k, get_buffer)
                    buffer_in_lookup[k] = self._process_cache_node(gm, k)
                buffer_in_nodes.append(buffer_in_lookup[k])  # store buffer nodes for this op

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
                buffer_in_nodes,
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


@TransformRegistry.register("insert_cached_mla_attention")
class InsertCachedMLAAttention(InsertCachedAttention):
    """
    A transform to insert cached MLA attention into the graph module.

    This class is identical to InsertCachedAttention and inherits all its behavior.
    """

    pass


class ResizeKVCacheConfig(TransformConfig):
    """Configuration for the resize kv cache transform."""

    free_mem_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="The fraction of available memory to occupy."
    )


@TransformRegistry.register("resize_kv_cache")
class ResizeKVCache(BaseTransform):
    """Inflate the kv cache to occupy the available GPU memory.

    free_mem_ratio specifies the fraction of available memory to occupy.
    """

    config: ResizeKVCacheConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ResizeKVCacheConfig

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        free_mem_ratio = self.config.free_mem_ratio

        free_mem, total_mem = get_mem_info_in_mb(empty_cache=True)
        self._log_info(f"Free memory (MB): {free_mem}, Total memory (MB): {total_mem}")
        current_cache_size = cm.current_cache_size_bytes()
        current_kv_cache_size = getattr(cm, "current_kv_cache_size_bytes", None)
        current_kv_cache_size = (
            current_kv_cache_size() if callable(current_kv_cache_size) else current_cache_size
        )
        current_num_pages = cm.info.num_pages
        self._log_info(
            f"Current cache size (MB): {current_cache_size // 1024**2}, "
            f"Current num pages: {current_num_pages}"
        )
        if current_kv_cache_size != current_cache_size:
            self._log_info(
                f"Current KV-only cache size (MB): {current_kv_cache_size // 1024 // 1024}"
            )

        if free_mem_ratio == 0.0:
            self._log_info(f"Skipping cache resize for {free_mem_ratio=}")
            return mod, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # TODO: the manual PyTorch workflow respects max_num_tokens if set and does _NOT_ resize
        # the cache in this case. Should we do the same here?

        # Let's run a forward pass to get the memory usage
        cm.info.set_max_num_tokens_sample()
        free_mem_pre, _ = get_mem_info_in_mb(empty_cache=True)
        self._log_info(f"Free memory before forward pass (MB): {free_mem_pre}")

        # Reset peak memory stats to get the extra memory used during the forward pass
        torch.cuda.reset_peak_memory_stats()
        memory_allocated_before_forward_pass_mb = torch.cuda.memory_allocated() // 1024**2
        try:
            mod(**cm.named_args)
        except torch.OutOfMemoryError as e:
            self.ad_logger.error(
                f"OutOfMemoryError in forward pass while trying to resize the kv-cache:\n{e}"
            )
            raise e

        peak_memory_during_forward_pass_mb = torch.cuda.max_memory_allocated() // 1024**2
        mem_used_during_forward_pass_mb = (
            peak_memory_during_forward_pass_mb - memory_allocated_before_forward_pass_mb
        )
        self._log_info(
            f"Peak memory uasge during forward pass (MB): {peak_memory_during_forward_pass_mb}"
        )
        self._log_info(
            f"Extra memory used during forward pass (MB): {mem_used_during_forward_pass_mb}"
        )

        free_mem_post, _ = get_mem_info_in_mb(empty_cache=True)
        self._log_info(f"Free memory after forward pass (MB): {free_mem_post}")

        memory_for_forward_pass = free_mem_pre - free_mem_post
        self._log_info(f"Memory for forward pass (MB): {memory_for_forward_pass}")

        # Compute new pages using KV-only bytes to avoid SSM/conv inflating per-page cost
        # Reserve headroom to avoid OOM from other allocations (workspaces, cudagraph pools, etc.)
        reserve_mb = max(1024, (total_mem * 5) // 100)  # at least 1 GiB or 5% of total
        available_mb = max(0, free_mem_post - reserve_mb)

        new_kv_total_bytes = int(
            available_mb * 1024 * 1024 * free_mem_ratio + current_kv_cache_size
        )
        per_page_bytes = max(1, current_kv_cache_size // max(1, current_num_pages))
        new_num_pages = int(new_kv_total_bytes // per_page_bytes)

        # Need to sync all the GPUs if distributed group is initialized
        log_msg = f"Using local new_num_pages: {new_num_pages}"
        if is_distributed_initialized():
            gathered_num_pages = [None] * get_world_size()
            all_gather_object(gathered_num_pages, new_num_pages)
            new_num_pages = min(gathered_num_pages)
            log_msg = f"After all_gather - new_num_pages: {new_num_pages}"

        self._log_info(log_msg)
        cm.resize_cache(new_num_pages)

        # Log the final cache size for performance measurement, do not remove this log.
        final_cache_size_bytes = cm.current_cache_size_bytes()
        final_cache_size_gb = final_cache_size_bytes / (1024**3)  # Convert to GiB
        self._log_info(
            f"Final KV cache size after resize: {final_cache_size_gb:.2f} GiB ({new_num_pages} pages)"
        )

        # Free memory
        torch.cuda.empty_cache()

        info = TransformInfo(
            skipped=False,
            num_matches=0,
            is_clean=True,
            has_valid_shapes=True,
        )

        return mod, info


@TransformRegistry.register("initialize_cache")
class InitializeCache(BaseTransform):
    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        num_caches = cm.initialize_caches()
        self._log_info(f"Initialized {num_caches} caches for cached attention")

        info = TransformInfo(
            skipped=False, num_matches=num_caches, is_clean=True, has_valid_shapes=True
        )
        return mod, info
