"""Graph transformation to automatically add kv cache into MHA kernels."""

import operator
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from torch._subclasses import FakeTensor
from torch.fx import Graph, GraphModule, Node

from ...custom_ops.attention_interface import AttentionDescriptor, AttentionInfo
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import get_all_input_output_nodes, is_dist_op, is_linear_op, is_op
from .._graph import add_graph_input, canonicalize_graph


def _is_dist_lin_op(node: Node, exclude: Optional[List[Node]] = None) -> bool:
    return node not in (exclude or []) and (
        is_linear_op(node, include_quantization=True)
        or (is_dist_op(node) and is_linear_op(node.all_input_nodes[0], include_quantization=True))
    )


def _bfs(node: Node, target: Callable, attr_next: str = "users") -> Node:
    queue = [node]
    while queue:
        cur_node = queue.pop(0)
        if target(cur_node):
            return cur_node
        queue.extend(getattr(cur_node, attr_next))
    raise RuntimeError(f"Could not find node with target condition {target}.")


def insert_mha_with_kv_cache(
    egm: GraphModule,
    cm: CachedSequenceInterface,
    attention_op: AttentionDescriptor,
    rope_theta: Optional[float] = None,
    kv_cache_dtype: Optional[torch.dtype] = None,
) -> GraphModule:
    """Perform insertion of kv-caches and attention kernel."""
    # sanity check
    if cm.info.is_paged:
        assert attention_op.is_paged(), "Paged sequence info requires paged attention op."

    graph: Graph = egm.graph

    ad_logger.info(f"Inserting MHA with KV cache and AttentionOp as {attention_op.__name__}")
    ad_logger.debug(f"Before inserting MHA with KV cache: {egm}")

    # list of MHA kernels we would want to detect and replace
    mha_ops = {
        torch.ops.attention.scaled_dot_product_attention,
    }

    # loop through nodes to get input, output, and get_attr nodes
    input_nodes, output_nodes = get_all_input_output_nodes(graph)

    # we only expect one input node
    assert len(input_nodes) == 1, "Expected exactly one input node."

    # NOTE: for now, we wanna make sure we *only* return the final output and no hidden states.
    # Later on, we can revisit how to support returning hidden states.
    assert len(output_nodes) == 1, "Expected exactly one output node!"
    assert len(output_nodes[0].all_input_nodes) == 1, "Expected to only return final tensor output!"

    # get all mha nodes and their GEMMs as well as sanity checks and shape information
    mha_gemms: Dict[Node, List[Node]] = defaultdict(list)
    mha_info: Dict[Node, AttentionInfo] = {}
    for mha_node in graph.nodes:
        if mha_node.op != "call_function" or not is_op(mha_node, mha_ops):
            continue
        # do some sanity checks on the args of the node
        assert mha_node.kwargs == {}, "We don't handle kwargs for mha nodes right now."
        assert len(mha_node.args) >= 3, "MHA nodes should have at least 3 args: q, k, v."
        args_other = mha_node.args[3:]
        args_other_expected = (None, 0.0, True)[: len(args_other)]  # other args expected
        if args_other != args_other_expected:
            ad_logger.debug(f"Unexpected args for MHA node: {args_other}.")

        # from the sdpa node, identify q, k, v, and out GEMMs via BFS
        for arg in mha_node.args[:3]:
            mha_gemms[mha_node].append(
                _bfs(arg, lambda n: _is_dist_lin_op(n, mha_gemms[mha_node]), "all_input_nodes")
            )
        mha_gemms[mha_node].append(_bfs(mha_node, _is_dist_lin_op, "users"))

        # get fake q tensor that is an MHA input node to retrieve head_dim, num_heads, and dtype
        # also retrieve fake tensor corresponding to output of k GEMM to infer number of kv heads
        q_fake: FakeTensor = mha_node.args[0].meta["val"]
        kv_gemm_fake: FakeTensor = mha_gemms[mha_node][1].meta["val"]
        mha_info[mha_node] = AttentionInfo(
            num_heads=q_fake.shape[1],
            num_kv_heads=kv_gemm_fake.shape[-1] // q_fake.shape[3],
            head_dim=q_fake.shape[3],
            dtype=q_fake.dtype,
            cache_dtype=kv_cache_dtype,
            rope_theta=rope_theta,
        )

    # insert metadata computation and extract each argument as a node
    mha_0 = next(iter(mha_info.keys()))
    get_metadata, num_metadata = attention_op.get_prepare_metadata_op()
    with graph.inserting_before(mha_0):
        ret_node = graph.call_function(
            get_metadata,
            args=(
                input_nodes[0],
                *(add_graph_input(egm, name) for name in cm.info.extra_arg_names),
                cm.info.page_size,
            ),
        )
        metadata_nodes = [
            graph.call_function(operator.getitem, args=(ret_node, idx))
            for idx in range(num_metadata)
        ]

    buffer_in_lookup: Dict[str, Node] = {}

    # replace SDPA with custom MHA kernel that takes q, k, v GEMMs directly and is fed to out GEMM
    # all other nodes will be pruned during recompile below
    for idx, (mha_node, gemms) in enumerate(mha_gemms.items()):
        # retrieve some MHA GEMM info
        qkv, out = gemms[:3], gemms[3]

        # setup + store cache initializers and caches as input nodes
        cache_in_nodes = []
        for k, fn in attention_op.get_cache_initializers(mha_info[mha_node]).items():
            k_indexed = f"{k}_{idx}"
            cm.add_cache(k_indexed, fn)
            cache_in_nodes.append(add_graph_input(egm, k_indexed))

        # setup + store global buffer initializers and buffers as input nodes
        # NOTE: we have to check against existing keys to make sure nothing is registered twice...
        buffer_in_nodes = []
        for k, fn in attention_op.get_global_buffer_initializers(mha_info[mha_node]).items():
            if k not in buffer_in_lookup:
                cm.add_cache(k, fn)
                buffer_in_lookup[k] = add_graph_input(egm, k)
            buffer_in_nodes.append(buffer_in_lookup[k])  # store buffer nodes for this op

        # retrieve constants for attention_op
        constants = attention_op.get_constants(mha_info[mha_node])

        # insert fused replacement op
        with graph.inserting_before(mha_node):
            mha_node_with_cache = graph.call_function(
                attention_op.get_attention_op(),
                args=(*qkv, *metadata_nodes, *cache_in_nodes, *buffer_in_nodes, *constants),
            )
        mha_node.replace_all_uses_with(mha_node_with_cache)
        graph.erase_node(mha_node)

        # hook mha output directly to GEMM
        out.args = (mha_node_with_cache, *out.args[1:])

    egm = canonicalize_graph(egm, shape_prop=False)
    ad_logger.debug("After inserting MHA with KV cache: " + str(egm))
    return egm
