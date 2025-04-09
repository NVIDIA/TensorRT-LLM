"""Graph transformation to automatically detect+replace SDPA op with cached attention op."""

from typing import Callable, List

import torch
from torch._subclasses import FakeTensor
from torch.fx import Graph, GraphModule, Node

from ...custom_ops.attention_interface import (
    AttentionDescriptor,
    AttentionInfo,
    CacheConfig,
    PositionalEmbeddingConfig,
)
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from .._graph import canonicalize_graph
from .kvcache import _insert_with_kv_cache


def _extract_sdpa_info(node: Node, egm: GraphModule, cache_config: CacheConfig) -> Callable:
    """Extract information from SDPA node."""
    # SDPA attention node args:
    # (query, key, value, attention_mask, dropout, scaling, is_causal, num_key_value_groups)

    # Get fake tensors to extract shape information
    # In the updated wrapped_sdpa_attention_forward function, args are:
    # query=args[0], key=args[1], value=args[2]
    q_fake: FakeTensor = node.args[0].meta["val"]
    k_fake: FakeTensor = node.args[1].meta["val"]

    # Extract shapes - SDPA uses [batch, num_heads, seq_len, head_dim]
    num_heads = q_fake.shape[1]
    head_dim = q_fake.shape[3]

    # Determine num_kv_heads
    num_kv_heads = k_fake.shape[1]

    # Get positional embedding config with default parameters
    pos_embd_config = PositionalEmbeddingConfig()

    def _get_info():
        return AttentionInfo(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=q_fake.dtype,
            cache_config=cache_config,
            pos_embd_config=pos_embd_config,
        )

    return _get_info


def _identify_attention_ops(egm: GraphModule, attention_op: AttentionDescriptor) -> List[Node]:
    """Identify attention operation nodes in the graph."""
    graph: Graph = egm.graph
    attention_nodes = []

    # Get the attention op function
    cached_attn_op, _ = attention_op.get_attention_op()

    # Find all nodes that use the attention op
    for node in graph.nodes:
        if node.op == "call_function" and is_op(node, cached_attn_op):
            attention_nodes.append(node)

    return attention_nodes


def _insert_transposes(egm: GraphModule, attention_nodes: List[Node]) -> GraphModule:
    """Insert transpose operations for q, k, v inputs to the attention nodes."""
    graph: Graph = egm.graph

    for attn_node in attention_nodes:
        # Get the q, k, v inputs (first 3 arguments)
        q, k, v = attn_node.args[:3]

        # Create transpose nodes before the attention node
        with graph.inserting_before(attn_node):
            q_transposed = graph.call_function(torch.transpose, args=(q, 1, 2))
            k_transposed = graph.call_function(torch.transpose, args=(k, 1, 2))
            v_transposed = graph.call_function(torch.transpose, args=(v, 1, 2))

        # Replace the q, k, v inputs with their transposed versions
        new_args = (q_transposed, k_transposed, v_transposed) + attn_node.args[3:]
        attn_node.args = new_args

    return canonicalize_graph(egm, shape_prop=False)


# TODO (lliebenwein): we completely ignore the other arguments to the SDPA op for now...
# need support....


def insert_unfused_mha_with_kv_cache(
    egm: GraphModule,
    cm: CachedSequenceInterface,
    attention_op: AttentionDescriptor,
    cache_config: CacheConfig,
    input_nodes: List[Node],
) -> GraphModule:
    """Identify SDPA op and replace it with the unfused_mha op.

    This pattern matcher will look for the SDPA op and replace them with a cached op. The cached op
    will take post-processed Q, K, V as inputs and output the MHA output which will be fed to the
    output GEMM.
    """
    ad_logger.info("Detecting SDPA pattern for caching.")
    ad_logger.debug(f"Before detecting SDPA pattern: {egm}")

    # List of MHA kernels we would want to detect and replace
    sdpa_ops = {torch.ops.sdpa.attention}

    # First use the shared logic from kvcache.py to insert the attention op
    egm = _insert_with_kv_cache(
        egm, cm, attention_op, cache_config, sdpa_ops, _extract_sdpa_info, "sdpa", input_nodes
    )

    # Now post-process the graph to insert transpose operations for q, k, v
    # SDPA uses [batch, num_heads, seq_len, head_dim]
    # Our attention op expects [batch, seq_len, num_heads, head_dim]
    attention_nodes = _identify_attention_ops(egm, attention_op)

    if attention_nodes:
        ad_logger.debug(
            f"Post-processing graph to insert transposes for {len(attention_nodes)} attention nodes"
        )
        egm = _insert_transposes(egm, attention_nodes)
        ad_logger.debug(f"After inserting transposes: {egm}")

    return egm
