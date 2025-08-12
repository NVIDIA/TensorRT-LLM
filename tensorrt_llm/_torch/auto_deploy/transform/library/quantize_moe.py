from functools import partial
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ...utils.quantization_utils import QuantizationImpl, should_skip_quantization
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

quantized_moe_op_map = {
    "FP8": torch.ops.auto_deploy.torch_quant_fp8_moe,
    "NVFP4": torch.ops.auto_deploy.torch_quant_fp4_moe,
}


def _quantize_moe_node(
    gm: GraphModule,
    node: Node,
    quant_impl: QuantizationImpl,
    quantized_op: Callable[..., Node],
):
    """
    Replace a torch.ops.auto_deploy.torch_moe node with its quantized version,
    quantizing each expert weight list and registering scales + hooks.
    Automatically handles different scale configurations per quantization type.
    """
    w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)

    scale_keys = quant_impl.scale_names()

    def quantize_param_list(weight_names: List[str]) -> Tuple[List[Node], List[List[Node]]]:
        new_attrs = []
        scale_nodes_group = []
        for name in weight_names:
            orig_weight = gm.get_parameter(name)
            new_weight = quant_impl.quantize_weight(orig_weight)

            # Replace parameter in submodule
            modname, _, attrname = name.rpartition(".")
            submod = gm.get_submodule(modname)
            setattr(submod, attrname, nn.Parameter(new_weight, requires_grad=False))

            # Register new scale buffers
            for scale_name, scale_val in quant_impl.default_scales(orig_weight.shape).items():
                submod.register_buffer(scale_name, scale_val)

            # Register load hook
            gm._register_load_state_dict_pre_hook(partial(quant_impl.load_hook, weight_name=name))

            # Create get_attr nodes for new param and each scale
            with gm.graph.inserting_before(node):
                new_weight_attr = gm.graph.get_attr(name)
                new_attrs.append(new_weight_attr)
                scales = [gm.graph.get_attr(modname + "." + s) for s in scale_keys]
                scale_nodes_group.append(scales)

        return new_attrs, scale_nodes_group

    # Quantize all three expert weights
    w1_attrs, w1_scales = quantize_param_list(w1_names)
    w2_attrs, w2_scales = quantize_param_list(w2_names)
    w3_attrs, w3_scales = quantize_param_list(w3_names)

    # Collect scale tensors per scale type across w1, w2, w3
    def collect_scales(index: int) -> Tuple[List[Node], List[Node], List[Node]]:
        return (
            [s[index] for s in w1_scales],
            [s[index] for s in w2_scales],
            [s[index] for s in w3_scales],
        )

    # Prepare args
    args = [
        node.args[0],  # x
        node.args[1],  # selected_experts
        node.args[2],  # routing_weights
        w1_attrs,
        w2_attrs,
        w3_attrs,
    ]

    for idx in range(len(scale_keys)):
        s1, s2, s3 = collect_scales(idx)
        args.extend([s1, s2, s3])

    # Replace the current node with the quantized version
    with gm.graph.inserting_after(node):
        new_node = gm.graph.call_function(
            quantized_op,
            args=tuple(args),
        )
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)


# TODO(Fridah-nv): robust handling similar to `extract_param_names_from_lin_node` or expand it
def _extract_moe_weight_param_lists(moe_node: Node) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a torch.ops.moe.torch_moe node in gm.graph, extract three lists of
    the parameter names for w1_weight, w2_weight, and w3_weight.

    Returns:
      (w1_names, w2_names, w3_names), each a list of strings like 'layer.expert_0.w1.weight'
    """
    # args layout: (x, selected_experts, routing_weights, w1_list, w2_list, w3_list)
    try:
        w1_list, w2_list, w3_list = moe_node.args[3:6]
    except ValueError:
        raise RuntimeError(
            f"Expected moe_node.args to have at least 6 entries, got {len(moe_node.args)}"
        )

    def _unwrap_list(arg) -> List[str]:
        if not isinstance(arg, (list, tuple)):
            raise TypeError(f"Expected a Python list/tuple of get_attr Nodes, got {type(arg)}")
        names: List[str] = []
        for elt in arg:
            if not isinstance(elt, Node) or elt.op != "get_attr":
                raise RuntimeError(f"Expected each list element to be a get_attr Node, got {elt}")
            names.append(elt.target)
        return names

    w1_names = _unwrap_list(w1_list)
    w2_names = _unwrap_list(w2_list)
    w3_names = _unwrap_list(w3_list)

    return w1_names, w2_names, w3_names


@TransformRegistry.register("quantize_moe")
class QuantizeMOE(BaseTransform):
    """
    Traverse gm, find every torch.ops.auto_deploy.torch_moe, and replace it with the
    quantized version using the quant_algo from quant_config.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        quant_config = factory.get_quant_config()
        quant_algo = quant_config.get("quant_algo") if quant_config else None

        if not quant_config or not quant_algo:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        excluded_patterns = quant_config.get("exclude_modules", [])

        quant_impl = QuantizationImpl.create(quant_algo)
        quantized_op = quantized_moe_op_map[quant_algo]

        count = 0

        for node in list(gm.graph.nodes):
            if is_op(node, torch.ops.auto_deploy.torch_moe):
                # Check that all expert weights should be quantized
                w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)
                if any(
                    should_skip_quantization(n, excluded_patterns)
                    for n in w1_names + w2_names + w3_names
                ):
                    continue
                _quantize_moe_node(gm, node, quant_impl, quantized_op)
                count += 1

        if count == 0:
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        info = TransformInfo(
            skipped=False, num_matches=count, is_clean=False, has_valid_shapes=False
        )

        return gm, info
