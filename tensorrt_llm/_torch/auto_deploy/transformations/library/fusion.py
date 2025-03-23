import operator
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    extract_param_names_from_lin_node,
    get_op_overload_packet,
    is_linear_op,
)
from ...utils.quantization_utils import QuantizationImpl
from .._graph import canonicalize_graph


def _insert_fused_gemm(gm: GraphModule, idx: int, parent_node: Node, linear_nodes: List[Node]):
    """Fuse GEMMs that have the same input activation.

    Below, is a simple example of how the fusion works:

    # before fusion:
    w1 = out1 x in
    w2 = out2 x in
    x = b x in
    y1 = x @ w1.T = b x out1
    y2 = x @ w2.T = b x out2

    # after fusion
    w = out1+out2 x in
    y = x @ w.T = b x (out1+out2)
    y1 = y[:, :out1]
    y2 = y[:, out1:out1+out2]
    """
    # some info we need
    keys_unfused = [extract_param_names_from_lin_node(n)[0] for n in linear_nodes]
    params_unfused = [gm.get_parameter(k) for k in keys_unfused]
    sizes_unfused = [p.size(0) for p in params_unfused]
    key_fused = f"fused_weight_{idx}"

    quantization_impl = QuantizationImpl.create(linear_nodes[0])

    def fuse_weights(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Fuse weights of linear nodes."""
        return torch.cat(tensors, dim=0)

    def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split the output tensor of the fused linear node to obtain the original outputs."""
        return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

    if quantization_impl:
        scales = {}
        for weight_key in keys_unfused:
            key = weight_key.rsplit(".", 1)[0]

            for scale_name in quantization_impl.scale_names():
                buffer_name = key + "." + scale_name
                scales.setdefault(scale_name, []).append(gm.get_buffer(buffer_name))

        try:
            weights_fused, buffer_fused = quantization_impl.fuse_linear_weights(
                params_unfused, **scales
            )
        except NotImplementedError as e:
            ad_logger.warning(f"Cannot fuse ops {keys_unfused}, skipping: {e}")
            return
        param_fused = nn.Parameter(weights_fused, requires_grad=False)

        for scale_name, buffer in buffer_fused.items():
            fused_buffer_name = key_fused + "_" + scale_name
            gm.register_buffer(fused_buffer_name, buffer)

    else:
        param_fused = nn.Parameter(fuse_weights([gm.get_parameter(k) for k in keys_unfused]))

    setattr(gm, key_fused, param_fused)

    # Handle fused_kwargs for quantized fused gemm.
    fused_kwargs = dict(linear_nodes[0].kwargs)

    with gm.graph.inserting_before(linear_nodes[0]):
        get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)
        if quantization_impl:
            for scale_name in quantization_impl.scale_names():
                # Creates new nodes for the fused scales so the unfused linear ops can be fully erased.
                fused_kwargs[scale_name] = gm.graph.create_node(
                    "get_attr", key_fused + "_" + scale_name
                )

    # add new linear node + split node
    with gm.graph.inserting_before(linear_nodes[0]):
        fused_linear_node = gm.graph.call_function(
            get_op_overload_packet(linear_nodes[0].target),
            args=(parent_node, get_param_node, None),
            kwargs=fused_kwargs,  # Assuming the scaling factors are the same
        )
        split_node = gm.graph.call_function(split_output, args=(fused_linear_node,))

    # now we need to replace all the linear nodes with the correct index of the split node
    for i, n in enumerate(linear_nodes):
        with gm.graph.inserting_before(n):
            get_split_node = gm.graph.call_function(operator.getitem, args=(split_node, i))
        n.replace_all_uses_with(get_split_node)

    # Clean up deleted modules to save GPU memory
    gm.graph.eliminate_dead_code()
    gm.delete_all_unused_submodules()


def fuse_gemms(gm: GraphModule) -> GraphModule:
    ad_logger.info("GEMM fusion")
    ad_logger.debug("Before GEMM fusion: " + str(gm))
    # sort linear nodes by parent node
    linear_nodes = defaultdict(list)
    for node in gm.graph.nodes:
        # TODO: we don't handle bias for now...
        if is_linear_op(node, include_quantization=True) and node.args[2] is None:
            linear_nodes[node.args[0]].append(node)

    # fuse linear nodes
    idx = -1
    with cuda_memory_tracker():
        for parent_node, lin_children in linear_nodes.items():
            if len(lin_children) < 2:
                continue
            # linear nodes to fuse
            ad_logger.debug(
                f"Found linear nodes to fuse: {lin_children} with parent node: {parent_node}"
            )
            _insert_fused_gemm(gm, idx := idx + 1, parent_node, lin_children)

        # clean up and return
        gm = canonicalize_graph(gm)

    ad_logger.debug("After GEMM fusion: " + str(gm))
    torch.cuda.empty_cache()
    return gm
