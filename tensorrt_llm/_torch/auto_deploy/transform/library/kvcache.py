"""Graph transformation to automatically add kv cache into fused MHA op."""

import operator
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import AttentionDescriptor, AttentionRegistry, Constant
from ...distributed.common import all_gather_object, get_world_size
from ...distributed.common import is_initialized as is_distributed_initialized
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import add_graph_input
from ...utils.node_utils import get_all_input_output_nodes, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


@TransformRegistry.register("update_in_out_nodes")
class UpdateInOutNodes(BaseTransform):
    """Modify the graph module by adding new input nodes.

    The new input nodes correspond to the extra arguments needed for cached and flattened attention.

    Args:
        egm: The graph module to analyze and modify.
        cm: Cached sequence interface containing extra argument information.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # loop through nodes to get input, output, and get_attr nodes
        input_nodes, output_nodes = get_all_input_output_nodes(gm.graph)

        # NOTE: for now, we wanna make sure we *only* return the final output and no hidden states.
        # Later on, we can revisit how to support returning hidden states.
        assert len(output_nodes) == 1, "Expected exactly one output node!"
        assert len(output_nodes[0].all_input_nodes) == 1, (
            "Expected to only return final tensor output!"
        )

        # Activate and add extra argument nodes
        new_args = cm.info.switch_to_cached_attn_inputs()
        for name in new_args:
            input_nodes.append(add_graph_input(gm, name))

        info = TransformInfo(skipped=False, num_matches=1, is_clean=False, has_valid_shapes=False)

        return gm, info


class InsertCachedAttentionConfig(TransformConfig):
    """Configuration for the insert cached attention transform."""

    backend: Optional[str] = Field(default=None, description="The attention backend to use.")


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

    def _process_get_metadata(
        self, gm: GraphModule, m_args: List[str], const_args: List[Constant]
    ) -> List[Node]:
        """Process the get_metadata function into an op and return node references."""
        # retrieve input nodes
        input_nodes, _ = get_all_input_output_nodes(gm.graph)
        input_nodes_mapping = {n.target: n for n in input_nodes}

        # filtered and sorted for SequenceInfo arguments + constants (input_ids, position_ids, etc.)
        inputs_from_info = [input_nodes_mapping[k] for k in m_args]

        # insert metadata computation and extract each argument as a node
        get_metadata, num_metadata = self.attn_descriptor.get_prepare_metadata_op()
        with gm.graph.inserting_before(input_nodes[-1].next):
            ret_node = gm.graph.call_function(get_metadata, args=(*inputs_from_info, *const_args))
            metadata_nodes = [
                gm.graph.call_function(operator.getitem, args=(ret_node, idx))
                for idx in range(num_metadata)
            ]
        return metadata_nodes

    def _process_cache_node(self, gm: GraphModule, cache_name: str) -> Node:
        """Process the cache nodes by inserting a cached attention replacement op."""
        return add_graph_input(gm, cache_name)

    def _insert_cached_attn_node(
        self,
        gm: GraphModule,
        attn_node: Node,
        qkv_nodes: List[Node],
        meta_nodes: List[Node],
        cache_nodes: List[Node],
        buffer_nodes: List[Node],
        constants: List[Constant],
    ):
        """Insert a cached attention node into the graph."""
        with gm.graph.inserting_before(attn_node):
            cached_attn_node = gm.graph.call_function(
                self.attn_descriptor.get_cached_attention_op(),
                args=(*qkv_nodes, *meta_nodes, *cache_nodes, *buffer_nodes, *constants),
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

        cache_config = factory.get_cache_config()

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

        # insert metadata computation and extract each argument as a node
        metadata_nodes = self._process_get_metadata(
            gm, cm.info.args_for_prepare_metadata, cm.info.const_args_for_prepare_metadata
        )

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
                gm, attn_node, qkv, metadata_nodes, cache_in_nodes, buffer_in_nodes, constants
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

        def _get_mem_info_in_mb():
            free_mem, total_mem = torch.cuda.mem_get_info()
            return free_mem // 1024**2, total_mem // 1024**2

        free_mem, total_mem = _get_mem_info_in_mb()
        self._log_info(f"Free memory (MB): {free_mem}, Total memory (MB): {total_mem}")
        current_cache_size = cm.current_cache_size_bytes()
        current_kv_cache_size = getattr(cm, "current_kv_cache_size_bytes", None)
        current_kv_cache_size = (
            current_kv_cache_size() if callable(current_kv_cache_size) else current_cache_size
        )
        current_num_pages = cm.info.num_pages
        self._log_info(
            f"Current cache size (MB): {current_cache_size // 1024 // 1024}, "
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
        free_mem_pre, _ = _get_mem_info_in_mb()
        self._log_info(f"Free memory before forward pass (MB): {free_mem_pre}")

        mod(**cm.named_args)

        free_mem_post, _ = _get_mem_info_in_mb()
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
