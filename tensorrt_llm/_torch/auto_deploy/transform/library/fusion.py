import operator
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    extract_param_names_from_lin_node,
    get_op_overload_packet,
    is_linear_op,
    is_op,
)
from ...utils.quantization_utils import QuantizationImpl
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


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

    quantization_impls = [QuantizationImpl.create(n) for n in linear_nodes]

    def fuse_weights(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Fuse weights of linear nodes."""
        return torch.cat(tensors, dim=0)

    def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split the output tensor of the fused linear node to obtain the original outputs."""
        return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

    if all(
        q is not None and quantization_impls[0].target_op() == q.target_op()
        for q in quantization_impls
    ):
        scales = {}
        for weight_key in keys_unfused:
            key = weight_key.rsplit(".", 1)[0]

            for scale_name in quantization_impls[0].scale_names():
                buffer_name = key + "." + scale_name
                scales.setdefault(scale_name, []).append(gm.get_buffer(buffer_name))

        try:
            weights_fused, buffer_fused = quantization_impls[0].fuse_linear_weights(
                params_unfused, **scales
            )
        except NotImplementedError as e:
            ad_logger.warning(f"Cannot fuse ops {keys_unfused}, skipping: {e}")
            return
        param_fused = nn.Parameter(weights_fused, requires_grad=False)

        for scale_name, buffer in buffer_fused.items():
            fused_buffer_name = key_fused + "_" + scale_name
            gm.register_buffer(fused_buffer_name, buffer)

    elif all(q is None for q in quantization_impls):
        param_fused = nn.Parameter(fuse_weights([gm.get_parameter(k) for k in keys_unfused]))
    else:
        ad_logger.warning(f"Cannot fuse ops {keys_unfused} for mixed-precision linear nodes.")
        return

    setattr(gm, key_fused, param_fused)

    # Handle fused_kwargs for quantized fused gemm.
    fused_kwargs = dict(linear_nodes[0].kwargs)

    with gm.graph.inserting_before(linear_nodes[0]):
        get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)
        if quantization_impls[0]:
            for scale_name in quantization_impls[0].scale_names():
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


# difference between this function and _insert_fused_fp4_gemm is `fuse_fp8_weights` function and scale_names list
def _insert_fused_fp8_gemm(gm: GraphModule, idx: int, parent_node: Node, linear_nodes: List[Node]):
    """Fuse FP8 GEMMs that have the same input activation.

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

    def fuse_fp8_weights(
        weights, weight_scale, input_scale
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not all(s == input_scale[0] for s in input_scale):
            raise NotImplementedError(f"Cannot fuse due to mismatched input_scale {input_scale}")

        # Handle quantized weights with weight_scale.
        # First we upcast to FP32 precision and then downcast back to the original precision (FP8)
        assert weights[0].dtype == torch.float8_e4m3fn, "Only support FP8 quantized weights fusion."
        fused_fp32_weights = torch.cat(
            [t.to(torch.float) * s for t, s in zip(weights, weight_scale)], dim=0
        )
        new_weight_scale = torch.max(torch.stack(weight_scale))
        fused_fp8_weights = (fused_fp32_weights / new_weight_scale).to(weights[0].dtype)

        return fused_fp8_weights, {
            "weight_scale": new_weight_scale,
            "input_scale": input_scale[0].clone(),
        }

    def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split the output tensor of the fused linear node to obtain the original outputs."""
        return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

    if all(is_op(n, torch.ops.auto_deploy.torch_quant_linear_fp8) for n in linear_nodes):
        scales = {}
        for weight_key in keys_unfused:
            key = weight_key.rsplit(".", 1)[0]

            for scale_name in ["input_scale", "weight_scale"]:
                buffer_name = key + "." + scale_name
                scales.setdefault(scale_name, []).append(gm.get_buffer(buffer_name))

        try:
            weights_fused, buffer_fused = fuse_fp8_weights(params_unfused, **scales)
        except NotImplementedError as e:
            ad_logger.warning(f"Cannot fuse ops {keys_unfused}, skipping: {e}")
            return
        param_fused = nn.Parameter(weights_fused, requires_grad=False)

        for scale_name, buffer in buffer_fused.items():
            fused_buffer_name = key_fused + "_" + scale_name
            gm.register_buffer(fused_buffer_name, buffer)

    else:
        ad_logger.warning(f"Cannot fuse ops {keys_unfused} for mixed-precision linear nodes.")
        return

    setattr(gm, key_fused, param_fused)

    # Handle fused_kwargs for quantized fused gemm.
    fused_kwargs = dict(linear_nodes[0].kwargs)

    with gm.graph.inserting_before(linear_nodes[0]):
        get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)
        scale_groups = [["input_scale"], ["weight_scale"]]
        for group, kwarg_name in zip(scale_groups, ["input_scale", "weight_scale"]):
            fused_kwargs[kwarg_name] = [
                gm.graph.create_node("get_attr", f"{key_fused}_{name}") for name in group
            ]
        # for scale_name in ["input_scale", "weight_scale"]:
        #     # Creates new nodes for the fused scales so the unfused linear ops can be fully erased.
        #     fused_kwargs[scale_name] = gm.graph.create_node(
        #         "get_attr", key_fused + "_" + scale_name
        #     )

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


def _insert_fused_fp4_gemm(gm: GraphModule, idx: int, parent_node: Node, linear_nodes: List[Node]):
    """Fuse FP8 GEMMs that have the same input activation.

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

    def fuse_fp4_weights(
        weights, weight_scale, alpha, input_scale
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not all(s == input_scale[0] for s in input_scale):
            raise NotImplementedError(f"Cannot fuse due to mismatched input_scale {input_scale}")

        if not all(s == alpha[0] for s in alpha):
            raise NotImplementedError(f"Cannot fuse due to mismatched alpha {alpha}")

        fused_weights = torch.cat(weights, dim=0)
        fused_weight_scale = torch.cat(weight_scale, dim=0)

        return fused_weights, {
            "weight_scale": fused_weight_scale,
            "alpha": alpha[0],
            "input_scale": input_scale[0].clone(),
        }

    def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split the output tensor of the fused linear node to obtain the original outputs."""
        return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

    if all(is_op(n, torch.ops.auto_deploy.torch_quant_linear_fp4) for n in linear_nodes):
        scales = {}
        for weight_key in keys_unfused:
            key = weight_key.rsplit(".", 1)[0]

            for scale_name in ["input_scale", "weight_scale", "alpha"]:
                buffer_name = key + "." + scale_name
                scales.setdefault(scale_name, []).append(gm.get_buffer(buffer_name))

        try:
            weights_fused, buffer_fused = fuse_fp4_weights(params_unfused, **scales)
        except NotImplementedError as e:
            ad_logger.warning(f"Cannot fuse ops {keys_unfused}, skipping: {e}")
            return
        param_fused = nn.Parameter(weights_fused, requires_grad=False)

        for scale_name, buffer in buffer_fused.items():
            fused_buffer_name = key_fused + "_" + scale_name
            gm.register_buffer(fused_buffer_name, buffer)

    else:
        ad_logger.warning(f"Cannot fuse ops {keys_unfused} for mixed-precision linear nodes.")
        return

    setattr(gm, key_fused, param_fused)

    # Handle fused_kwargs for quantized fused gemm.
    fused_kwargs = dict(linear_nodes[0].kwargs)

    with gm.graph.inserting_before(linear_nodes[0]):
        get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)
        scale_groups = [["input_scale"], ["weight_scale", "alpha"]]
        for group, kwarg_name in zip(scale_groups, ["input_scale", "weight_scale"]):
            fused_kwargs[kwarg_name] = [
                gm.graph.create_node("get_attr", f"{key_fused}_{name}") for name in group
            ]

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


@TransformRegistry.register("fuse_gemms")
class FuseGemms(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # sort linear nodes by parent node
        linear_nodes = defaultdict(list)
        for node in gm.graph.nodes:
            # TODO: we don't handle bias for now...
            if is_linear_op(node, include_quantization=True) and node.args[2] is None:
                linear_nodes[node.args[0]].append(node)

        # fuse linear nodes
        idx = -1
        num_matches = 0
        with cuda_memory_tracker():
            for parent_node, lin_children in linear_nodes.items():
                if len(lin_children) < 2:
                    continue
                # linear nodes to fuse
                _insert_fused_gemm(gm, idx := idx + 1, parent_node, lin_children)
                num_matches += 1

        torch.cuda.empty_cache()

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )
        return gm, info


@TransformRegistry.register("fuse_fp8_gemms")
class FuseFP8Gemms(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # sort linear nodes by parent node
        linear_nodes = defaultdict(list)
        for node in gm.graph.nodes:
            # TODO: we don't handle bias for now...
            # if is_linear_op(node, include_quantization=True) and node.args[2] is None:
            #     linear_nodes[node.args[0]].append(node)
            if is_op(node, torch.ops.auto_deploy.torch_quant_linear_fp8):
                linear_nodes[node.args[0]].append(node)

        # fuse linear nodes
        idx = -1
        num_matches = 0
        with cuda_memory_tracker():
            for parent_node, lin_children in linear_nodes.items():
                if len(lin_children) < 2:
                    continue
                # linear nodes to fuse
                _insert_fused_fp8_gemm(gm, idx := idx + 1, parent_node, lin_children)
                num_matches += 1

        torch.cuda.empty_cache()

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )
        return gm, info


@TransformRegistry.register("fuse_fp4_gemms")
class FuseFP4Gemms(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # sort linear nodes by parent node
        linear_nodes = defaultdict(list)
        for node in gm.graph.nodes:
            # TODO: we don't handle bias for now...
            # if is_linear_op(node, include_quantization=True) and node.args[2] is None:
            #     linear_nodes[node.args[0]].append(node)
            if is_op(node, torch.ops.auto_deploy.torch_quant_linear_fp4):
                linear_nodes[node.args[0]].append(node)

        # fuse linear nodes
        idx = -1
        num_matches = 0
        with cuda_memory_tracker():
            for parent_node, lin_children in linear_nodes.items():
                if len(lin_children) < 2:
                    continue
                # linear nodes to fuse
                _insert_fused_fp4_gemm(gm, idx := idx + 1, parent_node, lin_children)
                num_matches += 1

        torch.cuda.empty_cache()

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )
        return gm, info
