import operator
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    add_new_attribute_to_submodule,
    extract_param_names_from_lin_node,
    get_op_overload_packet,
    is_linear_op,
)
from ...utils.quantization_utils import QuantizationImpl
from .._graph import canonicalize_graph


def _insert_fused_gemm(gm: GraphModule, parent_node: Node, linear_nodes: List[Node]):
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

    # Register each fused_weight in a uniquely-named submodule so unused ones
    # can be cleaned up by `gm.delete_all_unused_submodules()`. Direct attributes
    # are not deleted automatically.
    new_weight_key = _fuse_keys(keys_unfused)
    new_module_name, new_param_name = new_weight_key.rsplit(".", 1)

    quantization_impl = QuantizationImpl.create(linear_nodes[0])

    def fuse_weights(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Fuse weights from multiple linear nodes by concatenation.

        Note:
            This function may slightly increase CUDA memory usage due to PyTorch's caching allocator behavior,
            which rounds allocations to reduce fragmentation. Allocations are typically rounded up to powers of two
            with subdivisions controlled by the environment variable:
            `CUDA_PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:N`.

            Models with irregular parameter sizes, an odd number of fused weights, or
            unusual num_ranks sharding across GPUs are more likely to trigger rounding up, increasing memory usage.
            For example, with `CUDA_PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:8`,
            4608 * 3072 * 2 bytes = 28311552 bytes = 2^20 * 27 bytes will be rounded to 29360128 = 2^24 * 7

        References:
            - PyTorch CUDA caching allocator implementation:
            https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp
            #L2046 for the rounding algorithm
            - Overview of the CUDA caching allocator:
            https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html"""

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
            fused_buffer_name = new_param_name + "_" + scale_name
            add_new_attribute_to_submodule(
                gm, new_module_name, fused_buffer_name, buffer, is_buffer=True
            )

    else:
        param_fused = nn.Parameter(fuse_weights([gm.get_parameter(k) for k in keys_unfused]))

    # Register fused parameters to new submodule
    full_new_param_name = add_new_attribute_to_submodule(
        gm, new_module_name, new_param_name, param_fused
    )

    # Handle fused_kwargs for quantized fused gemm.
    fused_kwargs = dict(linear_nodes[0].kwargs)

    with gm.graph.inserting_before(linear_nodes[0]):
        get_param_node = gm.graph.get_attr(full_new_param_name, torch.Tensor)
        if quantization_impl:
            for scale_name in quantization_impl.scale_names():
                full_new_buffer_name = full_new_param_name + "_" + scale_name
                fused_kwargs[scale_name] = gm.graph.create_node("get_attr", full_new_buffer_name)

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
    for key_unfused in keys_unfused:
        submodule_name, _ = key_unfused.rsplit(".", 1)
        gm.delete_submodule(submodule_name)


def _fuse_keys(keys: list[str]) -> str:
    """
    Fuse multiple dot-separated keys into a single key with a common prefix and fused middle segments.

    For Example, ["model.layers.0.q.weight", "model.layers.0.k.weight", "model.layers.1.v.weight"]
    is fused as 'model.layers.fused__0_q__0_k__1_v.weight'

    """
    token_lists = [key.split(".") for key in keys]
    # Check that all keys have same suffix
    assert len(set(tokens[-1] for tokens in token_lists)) == 1

    common_prefix = []
    for tokens in zip(*token_lists):
        if len(set(tokens)) == 1:
            common_prefix.append(tokens[0])
        else:
            break

    fused_parts = ["_".join(tokens[len(common_prefix) : -1]) for tokens in token_lists]

    fused_str = "__".join(fused_parts)
    final_str = f"{'.'.join(common_prefix)}.fused__{fused_str}.{token_lists[0][-1]}"

    return final_str


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
    with cuda_memory_tracker():
        for parent_node, lin_children in linear_nodes.items():
            if len(lin_children) < 2:
                continue
            # linear nodes to fuse
            ad_logger.debug(
                f"Found linear nodes to fuse: {lin_children} with parent node: {parent_node}"
            )
            _insert_fused_gemm(gm, parent_node, lin_children)

        # clean up and return
        gm = canonicalize_graph(gm)

    ad_logger.debug("After GEMM fusion: " + str(gm))
    return gm
