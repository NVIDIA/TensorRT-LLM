"""Pattern matcher to detect MHA pattern and replace with simple fused_mha op."""

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch._subclasses import FakeTensor
from torch.fx import Graph, GraphModule, Node

from ...models.factory import PositionalEmbeddingConfig
from ...utils.logger import ad_logger
from ...utils.node_utils import bfs, is_dist_op, is_linear_op, is_op
from .._graph import canonicalize_graph


def _is_dist_lin_op(node: Node, exclude: Optional[List[Node]] = None) -> bool:
    return node not in (exclude or []) and (
        is_linear_op(node, include_quantization=True)
        or (is_dist_op(node) and is_linear_op(node.all_input_nodes[0], include_quantization=True))
    )


def identify_and_fuse_mha(
    egm: GraphModule, pos_embd_config: PositionalEmbeddingConfig
) -> GraphModule:
    """Identify MHA pattern and fuse them together by replacing them with the fused_mha op.

    This pattern matcher will look for Q, K, V GEMMs followed by a SDPA op and replace them with
    a fused_mha op. The fused_mha op will take Q, K, V GEMMs as inputs and output the MHA output
    which will be fed to the output GEMM. Rope information is provided manually as an argument.
    """
    graph: Graph = egm.graph

    ad_logger.info("Fusing MHA pattern.")
    ad_logger.debug(f"Before fusing MHA pattern: {egm}")

    # list of MHA kernels we would want to detect and replace
    mha_ops = {
        torch.ops.attention.scaled_dot_product_attention,
    }

    # get all mha nodes and their GEMMs as well as sanity checks and shape information
    mha_gemms: Dict[Node, List[Node]] = defaultdict(list)
    mha_head_dim: Dict[Node, int] = {}
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
                bfs(arg, lambda n: _is_dist_lin_op(n, mha_gemms[mha_node]), "all_input_nodes")
            )
        mha_gemms[mha_node].append(bfs(mha_node, _is_dist_lin_op, "users"))

        # get fake q tensor that is an MHA input node to retrieve head_dim
        q_fake: FakeTensor = mha_node.args[0].meta["val"]
        mha_head_dim[mha_node] = q_fake.shape[3]

    # replace SDPA with fused MHA op that takes q, k, v GEMMs directly and is fed to out GEMM
    # all other nodes will be pruned during recompile below
    for mha_node, gemms in mha_gemms.items():
        # retrieve some MHA GEMM info
        qkv, out = gemms[:3], gemms[3]
        head_dim = mha_head_dim[mha_node]

        # insert fused replacement op
        with graph.inserting_before(mha_node):
            fused_mha_node = graph.call_function(
                torch.ops.attention.fused_mha,
                args=(
                    *qkv,
                    head_dim,
                    pos_embd_config.mode,
                    pos_embd_config.rope_theta,
                    pos_embd_config.rope_scale,
                ),
            )
        mha_node.replace_all_uses_with(fused_mha_node)
        graph.erase_node(mha_node)

        # hook mha output directly to GEMM
        out.args = (fused_mha_node, *out.args[1:])

    egm = canonicalize_graph(egm, shape_prop=False)
    ad_logger.debug("After inserting MHA with KV cache: " + str(egm))
    return egm
