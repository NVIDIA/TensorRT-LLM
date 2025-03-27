"""Graph transformation to automatically add kv cache into fused MHA op."""

import operator
from typing import Dict

import torch
from torch.fx import Graph, GraphModule, Node

from ...custom_ops.attention_interface import (
    AttentionDescriptor,
    AttentionInfo,
    CacheConfig,
    GetAttentionInfo,
    PositionalEmbeddingConfig,
)
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import get_all_input_output_nodes, is_op
from .._graph import add_graph_input, canonicalize_graph


def _collect_mha_nodes_and_info(
    egm: GraphModule, cache_config: CacheConfig
) -> Dict[Node, GetAttentionInfo]:
    graph: Graph = egm.graph

    # list of MHA kernels we would want to detect and replace
    fused_mha_ops = {
        torch.ops.attention.fused_mha,
    }

    # get all mha nodes and extract information needed for AttentionInfo
    mha_info: Dict[Node, GetAttentionInfo] = {}
    for mha_node in graph.nodes:
        if mha_node.op != "call_function" or not is_op(mha_node, fused_mha_ops):
            continue

        # get q and k weights from the mha node
        q_weight = egm.get_parameter(mha_node.all_input_nodes[0].args[1].target)  # [nd,hidden_size]
        k_weight = egm.get_parameter(mha_node.all_input_nodes[1].args[1].target)  # [nd,hidden_size]

        # get some more information from mha node
        head_dim = mha_node.args[3]
        if len(mha_node.args) > 4:
            pos_embd_mode, rope_theta, rope_scale = mha_node.args[4:7]
            pos_embd_config = PositionalEmbeddingConfig(
                mode=pos_embd_mode, rope_theta=rope_theta, rope_scale=rope_scale
            )
        else:
            pos_embd_config = PositionalEmbeddingConfig()

        # get fake q tensor that is an MHA input node to retrieve head_dim, num_heads, and dtype
        # also retrieve fake tensor corresponding to output of k GEMM to infer number of kv heads
        def _get_info():
            return AttentionInfo(
                num_heads=q_weight.shape[0] // head_dim,
                num_kv_heads=k_weight.shape[0] // head_dim,
                head_dim=head_dim,
                dtype=q_weight.dtype,
                cache_config=cache_config,
                pos_embd_config=pos_embd_config,
            )

        mha_info[mha_node] = _get_info

    return mha_info


def _check_in_out_nodes(egm: GraphModule) -> Node:
    """Check for input and output nodes in the graph and return 1st input node."""
    # loop through nodes to get input, output, and get_attr nodes
    input_nodes, output_nodes = get_all_input_output_nodes(egm.graph)

    # we only expect one input node
    assert len(input_nodes) == 1, "Expected exactly one input node."

    # NOTE: for now, we wanna make sure we *only* return the final output and no hidden states.
    # Later on, we can revisit how to support returning hidden states.
    assert len(output_nodes) == 1, "Expected exactly one output node!"
    assert len(output_nodes[0].all_input_nodes) == 1, "Expected to only return final tensor output!"

    return input_nodes[0]


def _insert_cached_nodes(
    egm: GraphModule,
    cm: CachedSequenceInterface,
    attention_op: AttentionDescriptor,
    attn_lookup: Dict[Node, GetAttentionInfo],
    input_node: Node,
) -> None:
    """Insert all input nodes and cached attention nodes."""
    # pick up graph
    graph: Graph = egm.graph

    # insert metadata computation and extract each argument as a node
    attn_node_0 = next(iter(attn_lookup.keys()))
    get_metadata, num_metadata = attention_op.get_prepare_metadata_op()
    with graph.inserting_before(attn_node_0):
        ret_node = graph.call_function(
            get_metadata,
            args=(
                input_node,
                *(add_graph_input(egm, name) for name in cm.info.extra_arg_names),
                cm.info.page_size,
            ),
        )
        metadata_nodes = [
            graph.call_function(operator.getitem, args=(ret_node, idx))
            for idx in range(num_metadata)
        ]

    buffer_in_lookup: Dict[str, Node] = {}

    # replace fused attention node with attention node that has kv cache
    for idx, (attn_node, get_info) in enumerate(attn_lookup.items()):
        # pick out GEMMs
        qkv = attn_node.args[:3]

        # setup + store cache initializers and caches as input nodes
        cache_in_nodes = []
        for k, get_cache in attention_op.get_cache_initializers(get_info).items():
            k_indexed = f"{k}_{idx}"
            cm.add_cache(k_indexed, get_cache)
            cache_in_nodes.append(add_graph_input(egm, k_indexed))

        # setup + store global buffer initializers and buffers as input nodes
        # NOTE: we have to check against existing keys to make sure nothing is registered twice...
        buffer_in_nodes = []
        for k, get_buffer in attention_op.get_global_buffer_initializers(get_info).items():
            if k not in buffer_in_lookup:
                cm.add_cache(k, get_buffer)
                buffer_in_lookup[k] = add_graph_input(egm, k)
            buffer_in_nodes.append(buffer_in_lookup[k])  # store buffer nodes for this op

        # retrieve constants for attention_op
        constants = attention_op.get_constants(get_info())

        # insert fused replacement op
        with graph.inserting_before(attn_node):
            node_with_cache = graph.call_function(
                attention_op.get_attention_op(),
                args=(*qkv, *metadata_nodes, *cache_in_nodes, *buffer_in_nodes, *constants),
            )
        attn_node.replace_all_uses_with(node_with_cache)
        graph.erase_node(attn_node)


def insert_mha_with_kv_cache(
    egm: GraphModule,
    cm: CachedSequenceInterface,
    attention_op: AttentionDescriptor,
    cache_config: CacheConfig,
) -> GraphModule:
    """Replaces the vanilla fused_mha node in the graph with a custom MHA op with KV cache."""
    # sanity check
    if cm.info.is_paged:
        assert attention_op.is_paged(), "Paged sequence info requires paged attention op."

    ad_logger.info(f"Inserting MHA with KV cache and AttentionOp as {attention_op.__name__}")
    ad_logger.debug(f"Before inserting MHA with KV cache: {egm}")

    # check for input node
    input_node = _check_in_out_nodes(egm)

    # get all attention nodes and their info objects
    attn_node_lookup = _collect_mha_nodes_and_info(egm, cache_config)

    _insert_cached_nodes(egm, cm, attention_op, attn_node_lookup, input_node)

    egm = canonicalize_graph(egm, shape_prop=False)
    ad_logger.debug("After inserting MHA with KV cache: " + str(egm))
    return egm


def resize_kv_cache(
    egm: GraphModule, cm: CachedSequenceInterface, free_mem_ratio: float = 0.8
) -> None:
    """Inflate the kv cache to occupy the available GPU memory.

    free_mem_ratio specifies the fraction of available memory to occupy.
    """
    free_mem, total_mem = torch.cuda.mem_get_info()
    ad_logger.info(f"Free memory: {free_mem}, Total memory: {total_mem}")
    current_cache_size = cm.current_cache_size_bytes()
    current_num_pages = cm.info.num_pages
    ad_logger.info(
        f"Current cache size: {current_cache_size}, Current num pages: {current_num_pages}"
    )

    try:
        # Let's run a forward pass to get the memory usage
        cm.info._set_max_num_tokens_sample()
        free_mem_pre, _ = torch.cuda.mem_get_info()
        ad_logger.info(f"Free memory before forward pass: {free_mem_pre}")
        egm(*cm.args)
        free_mem_post, _ = torch.cuda.mem_get_info()
        ad_logger.info(f"Free memory after forward pass: {free_mem_post}")

        memory_for_forward_pass = free_mem_pre - free_mem_post
        ad_logger.info(f"Memory for forward pass: {memory_for_forward_pass}")

        new_cache_size = free_mem_post * free_mem_ratio + current_cache_size
        new_num_pages = int(new_cache_size // (current_cache_size // current_num_pages))
        ad_logger.info(f"New cache size: {new_cache_size}, New num pages: {new_num_pages}")
        cm.resize_cache(new_num_pages)
    except Exception as e:
        ad_logger.warning(
            f"Error encountered while resizing kv cache: {e}.\nSkipping cache resize."
        )

    # Free memory
    torch.cuda.empty_cache()
