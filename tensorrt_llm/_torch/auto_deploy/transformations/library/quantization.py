from functools import partial
from typing import List

import torch.nn as nn
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import (
    extract_param_names_from_lin_node,
    get_quantization_params_from_linear_node,
    is_linear_op,
    is_match,
)
from ...utils.quantization_utils import (
    QuantizationImpl,
    get_quantization_from_linear_node,
    is_quantized_op,
    remove_output_quantizers,
)
from .._graph import canonicalize_graph


def _insert_quantized_linear(
    gm: GraphModule,
    node: Node,
    quantization_impl: QuantizationImpl,
    is_quantized_graph: bool = False,
):
    """Replaces the matmul node with a new quantized matmul node.

    The state_dict is also updated to contain the sharded weights.
    """
    param_name, _ = extract_param_names_from_lin_node(node)
    original_weight = gm.get_parameter(param_name)
    new_param = nn.Parameter(
        quantization_impl.quantize_weight(original_weight), requires_grad=False
    )
    modname, _, attrname = param_name.rpartition(".")

    submod = gm.get_submodule(modname)
    setattr(submod, attrname, new_param)

    # check modelopt quantizers from graph
    if is_quantized_graph:
        input_params, weight_params, output_params = get_quantization_params_from_linear_node(node)
        # redirect to input and weight
        node.args = (input_params.input_node, weight_params.input_node, *node.args[2:])

        # redirect output to skip output quantizer if any
        user = list(node.users.keys())[0]
        if len(node.users) == 1 and is_quantized_op(user):
            user.replace_all_uses_with(node)

        # when loading the state_dict, we need to convert input amax to input scale
        input_scale_name = quantization_impl.scale_names()[0]
        gm._register_load_state_dict_pre_hook(
            partial(
                quantization_impl.convert_amax_hook,
                scale_name=modname + "." + input_scale_name,
                amax_name=input_params.amax.target,
            )
        )
        # Note: canonicalize_graph() will remove input/weight/output quantizer

    for scale_name, scale in quantization_impl.default_scales(original_weight.shape).items():
        submod.register_buffer(scale_name, scale)

    gm._register_load_state_dict_pre_hook(
        partial(quantization_impl.load_hook, weight_name=param_name)
    )

    node.target = quantization_impl.target_op()

    with gm.graph.inserting_before(node):
        scales = {}
        for scale_name in quantization_impl.scale_names():
            scales[scale_name] = gm.graph.create_node("get_attr", modname + "." + scale_name)

    node.kwargs = {**node.kwargs, **scales}


def quantize(
    gm: GraphModule, quantization: str, skip: List[str] = [], is_quantized_graph: bool = False
):
    """Quantize the GraphModule and replace linear with quantized linear."""
    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    # tracking quantized linears in the graph
    quantized_nodes = {}
    for n in gm.graph.nodes:
        if is_match(n, skip):
            ad_logger.debug(f"Skipping node for quantization: {n}")
            continue

        if is_linear_op(n, include_quantization=False):
            # get per-layer quantization format from the node
            if is_quantized_graph:
                quantization = get_quantization_from_linear_node(n)
            if quantization:
                _insert_quantized_linear(
                    gm, n, QuantizationImpl.create(quantization), is_quantized_graph
                )
                quantized_nodes[quantization] = quantized_nodes.get(quantization, -1) + 1

    if is_quantized_graph:
        remove_output_quantizers(gm)

    gm = canonicalize_graph(gm)
    for quantization in quantized_nodes:
        ad_logger.info(f"Found {quantized_nodes[quantization]} {quantization} quantized nodes.")
    ad_logger.debug("After quantization: " + str(gm))

    return gm
