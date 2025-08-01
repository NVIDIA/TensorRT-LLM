"""Graph transformation to automatically add kv cache into fused MHA op."""

import operator
from typing import Dict, Type

import torch
from torch.fx import Graph, GraphModule, Node

from ...custom_ops.attention_interface import AttentionDescriptor, CacheConfig
from ...distributed.common import all_gather_object, get_world_size
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import get_all_input_output_nodes, is_op
from .._graph import add_graph_input, canonicalize_graph


def update_in_out_nodes(egm: GraphModule, cm: CachedSequenceInterface) -> None:
    """Modify the graph module by adding new input nodes and canonicalizing the graph.

    The new input nodes correspond to the extra arguments needed for cached and flattened attention.

    Args:
        egm: The graph module to analyze and modify.
        cm: Cached sequence interface containing extra argument information.
    """
    # loop through nodes to get input, output, and get_attr nodes
    input_nodes, output_nodes = get_all_input_output_nodes(egm.graph)

    # we only expect one input node
    assert len(input_nodes) == 2, "Expected exactly two input nodes (input_ids, position_ids)."

    # NOTE: for now, we wanna make sure we *only* return the final output and no hidden states.
    # Later on, we can revisit how to support returning hidden states.
    assert len(output_nodes) == 1, "Expected exactly one output node!"
    assert len(output_nodes[0].all_input_nodes) == 1, "Expected to only return final tensor output!"

    ad_logger.info(f"Found {len(input_nodes)} input nodes and {len(output_nodes)} output nodes")

    # Activate and add extra argument nodes
    new_args = cm.info.switch_to_cached_attn_inputs()
    for name in new_args:
        input_nodes.append(add_graph_input(egm, name))
    ad_logger.info(f"Added {len(new_args)} new input nodes for cached attention metadata")

    canonicalize_graph(egm)


def insert_cached_attention(
    egm: GraphModule,
    cm: CachedSequenceInterface,
    attn_descriptor: Type[AttentionDescriptor],
    cache_config: CacheConfig,
) -> None:
    """Replace uncached source attention node with corresponding cached attn node."""
    # Get all attention nodes and their info objects
    source_op = attn_descriptor.get_source_attention_op()

    # pick up graph
    graph: Graph = egm.graph

    # look for relevant source attention nodes
    source_attn_nodes = [n for n in graph.nodes if is_op(n, source_op)]

    if not source_attn_nodes:
        # If there are no nodes for kv cache insertion found, return current graph
        return

    # Sanity check
    if cm.info.is_paged:
        assert attn_descriptor.is_paged(), "Paged sequence info requires paged attention op."

    ad_logger.debug(f"Before inserting {attn_descriptor=} with cache: {egm}")

    # retrieve input nodes
    input_nodes, _ = get_all_input_output_nodes(egm.graph)

    # insert metadata computation and extract each argument as a node
    get_metadata, num_metadata = attn_descriptor.get_prepare_metadata_op()
    with graph.inserting_before(input_nodes[-1].next):
        ret_node = graph.call_function(
            get_metadata,
            args=(
                *input_nodes,
                cm.info.page_size,
            ),
        )
        metadata_nodes = [
            graph.call_function(operator.getitem, args=(ret_node, idx))
            for idx in range(num_metadata)
        ]

    buffer_in_lookup: Dict[str, Node] = {}

    # replace fused attention node with attention node that has kv cache
    num_cached_attn_replacements = 0
    for idx, attn_node in enumerate(source_attn_nodes):
        # pick out GEMMs
        qkv = attn_node.args[: attn_descriptor.get_num_qkv_args()]

        # setup + store cache initializers and caches as input nodes
        cache_in_nodes = []
        for k, get_cache in attn_descriptor.get_cache_initializers(attn_node, cache_config).items():
            k_indexed = f"{k}_{idx}"
            cm.add_cache(k_indexed, get_cache)
            cache_in_nodes.append(add_graph_input(egm, k_indexed))

        # setup + store global buffer initializers and buffers as input nodes
        # NOTE: we have to check against existing keys to make sure nothing is registered twice...
        buffer_in_nodes = []
        for k, get_buffer in attn_descriptor.get_global_buffer_initializers(attn_node).items():
            if k not in buffer_in_lookup:
                cm.add_cache(k, get_buffer)
                buffer_in_lookup[k] = add_graph_input(egm, k)
            buffer_in_nodes.append(buffer_in_lookup[k])  # store buffer nodes for this op

        # retrieve constants for attention_op
        constants = attn_descriptor.get_constants(attn_node)

        # insert cached attention replacement op
        with graph.inserting_before(attn_node):
            cached_attn_node = graph.call_function(
                attn_descriptor.get_cached_attention_op(),
                args=(*qkv, *metadata_nodes, *cache_in_nodes, *buffer_in_nodes, *constants),
            )
        attn_node.replace_all_uses_with(cached_attn_node)
        graph.erase_node(attn_node)
        num_cached_attn_replacements += 1

    canonicalize_graph(egm)
    ad_logger.info(
        f"Replaced {num_cached_attn_replacements} {source_op} ops "
        f"with {attn_descriptor.get_cached_attention_op()}"
    )
    ad_logger.debug(f"After inserting {attn_descriptor=} with cache: {egm}")


def resize_kv_cache(
    egm: GraphModule,
    cm: CachedSequenceInterface,
    free_mem_ratio: float = 0.8,
) -> None:
    """Inflate the kv cache to occupy the available GPU memory.

    free_mem_ratio specifies the fraction of available memory to occupy.
    """

    def _get_mem_info_in_mb():
        free_mem, total_mem = torch.cuda.mem_get_info()
        return free_mem // 1024**2, total_mem // 1024**2

    free_mem, total_mem = _get_mem_info_in_mb()
    ad_logger.info(f"Free memory (MB): {free_mem}, Total memory (MB): {total_mem}")
    current_cache_size = cm.current_cache_size_bytes()
    current_num_pages = cm.info.num_pages
    ad_logger.info(
        f"Current cache size: {current_cache_size}, Current num pages: {current_num_pages}"
    )

    if free_mem_ratio == 0.0:
        ad_logger.info(f"Skipping cache resize for {free_mem_ratio=}")
        return

    try:
        # Let's run a forward pass to get the memory usage
        cm.info._set_max_num_tokens_sample()
        free_mem_pre, _ = _get_mem_info_in_mb()
        ad_logger.info(f"Free memory before forward pass (MB): {free_mem_pre}")

        egm(*cm.args)

        free_mem_post, _ = _get_mem_info_in_mb()
        ad_logger.info(f"Free memory after forward pass (MB): {free_mem_post}")

        memory_for_forward_pass = free_mem_pre - free_mem_post
        ad_logger.info(f"Memory for forward pass (MB): {memory_for_forward_pass}")

        new_cache_size = free_mem_post * 1024 * 1024 * free_mem_ratio + current_cache_size
        new_num_pages = int(new_cache_size // (current_cache_size // current_num_pages))

        # Need to sync all the GPUs
        gathered_num_pages = [None] * get_world_size()
        all_gather_object(gathered_num_pages, new_num_pages)
        new_num_pages = min(gathered_num_pages)
        ad_logger.info(f"After all_gather - new_num_pages: {new_num_pages}")

        cm.resize_cache(new_num_pages)
    except Exception as e:
        ad_logger.warning(
            f"Error encountered while resizing kv cache: {e}.\nSkipping cache resize."
        )

    # Free memory
    torch.cuda.empty_cache()
