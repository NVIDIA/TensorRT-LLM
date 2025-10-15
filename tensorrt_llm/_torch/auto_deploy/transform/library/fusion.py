import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_param_names_from_lin_node, is_linear_op, is_op
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

    def fuse_weights(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Fuse weights of linear nodes."""
        return torch.cat(tensors, dim=0)

    def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split the output tensor of the fused linear node to obtain the original outputs."""
        return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

    param_fused = nn.Parameter(fuse_weights([gm.get_parameter(k) for k in keys_unfused]))

    setattr(gm, key_fused, param_fused)

    # Handle fused_kwargs for quantized fused gemm.
    fused_kwargs = dict(linear_nodes[0].kwargs)

    with gm.graph.inserting_before(linear_nodes[0]):
        get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)

    # add new linear node + split node
    with gm.graph.inserting_before(linear_nodes[0]):
        fused_linear_node = gm.graph.call_function(
            linear_nodes[0].target,
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


def check_same_children(parent_node: Node, is_desired_child: Callable[[Node], bool]) -> bool:
    """
    Return True iff *all* direct users of `parent_node` satisfy `is_desired_child`.
    """
    users_dict = getattr(parent_node, "users", None)
    if not users_dict:
        return False
    users = list(users_dict.keys()) if isinstance(users_dict, dict) else list(users_dict)
    if not users:
        return False
    return all(is_desired_child(u) for u in users)


class QuantizationFusionMixin(ABC):
    """
    Mixin that factors out the shared logic for fusing quantized GEMMs
    that share the same input activation (parent node).

    Subclasses must define:
      - target_op: the torch op identifying the quantized linear
      - scale_groups: List[List[str]] describing how kwargs should be grouped, e.g.
            FP8 -> [["input_scale"], ["weight_scale"]]
            FP4 -> [["input_scale"], ["weight_scale", "alpha"]]
      - fuse_rule(weights, **scales) -> (fused_weight, fused_buffers: Dict[str, Tensor])
        which takes:
            weights: List[Tensor]       # unfused per-out features (stacked along dim=0)
            **scales: Dict[str, List[Tensor]] with keys = flattened(scale_groups)
        and returns:
            fused_weight: Tensor
            fused_buffers: Dict[str, Tensor] to register as buffers on the fused module
    """

    target_op: Callable
    scale_groups: List[List[str]]

    @abstractmethod
    def fuse_rule(
        self, weights: List[torch.Tensor], **scales
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def build_custom_args_for_linear(self, scale_getattrs: Dict[str, Node]) -> Tuple[object, ...]:
        """Return the *positional* tail after bias for the fused call."""
        raise NotImplementedError

    def _insert_fused_quant_gemm(
        self, gm: GraphModule, idx: int, parent_node: Node, linear_nodes: List[Node]
    ):
        keys_unfused = [extract_param_names_from_lin_node(n)[0] for n in linear_nodes]
        params_unfused = [gm.get_parameter(k) for k in keys_unfused]
        sizes_unfused = [p.size(0) for p in params_unfused]
        key_fused = f"fused_weight_{idx}"

        def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            """Split the output tensor of the fused linear node to obtain the original outputs."""
            return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

        # Load scale buffers grouped by flattened scale names
        flat_scale_names = list(chain.from_iterable(self.scale_groups))
        scales: Dict[str, List[torch.Tensor]] = {}
        for weight_key in keys_unfused:
            key = weight_key.rsplit(".", 1)[0]
            for scale_name in flat_scale_names:
                buffer_name = key + "." + scale_name
                scales.setdefault(scale_name, []).append(gm.get_buffer(buffer_name))

        try:
            weights_fused, buffers_fused = self.fuse_rule(params_unfused, **scales)
        except NotImplementedError as e:
            ad_logger.warning(f"Cannot fuse ops {keys_unfused}, skipping: {e}")
            return
        param_fused = nn.Parameter(weights_fused, requires_grad=False)
        setattr(gm, key_fused, param_fused)
        for name, buf in buffers_fused.items():
            gm.register_buffer(f"{key_fused}_{name}", buf)

        # Handle fused_kwargs for quantized fused gemm.
        fused_kwargs = dict(linear_nodes[0].kwargs)
        with gm.graph.inserting_before(linear_nodes[0]):
            get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)

            # For each kwarg group (e.g., input_scale, weight_scale[, alpha]),
            # create a list of get_attr nodes in the same structure the op expects.
            scale_getattrs: Dict[str, Node] = {
                name: gm.graph.create_node("get_attr", f"{key_fused}_{name}")
                for name in flat_scale_names
            }
            custom_tail_args = self.build_custom_args_for_linear(scale_getattrs)

        # add new linear node + split node
        with gm.graph.inserting_before(linear_nodes[0]):
            fused_linear_node = gm.graph.call_function(
                linear_nodes[0].target,
                args=(parent_node, get_param_node, None, *custom_tail_args),
                kwargs=fused_kwargs,
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

    def _apply_fusion_pass(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Group quantized linear nodes by their parent (same activation)
        quant_linear_nodes = defaultdict(list)
        for node in gm.graph.nodes:
            if is_op(node, self.target_op) and node.args[2] is None:
                quant_linear_nodes[node.args[0]].append(node)

        idx = -1
        num_matches = 0
        with cuda_memory_tracker():
            for parent_node, lin_children in quant_linear_nodes.items():
                if len(lin_children) < 2:
                    continue
                if not check_same_children(parent_node, partial(is_op, ops=self.target_op)):
                    # Mixed children (e.g., quantized or non-linear) — skip fusion
                    continue
                self._insert_fused_quant_gemm(gm, idx := idx + 1, parent_node, lin_children)
                num_matches += 1

        torch.cuda.empty_cache()
        return gm, TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )


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
            if is_linear_op(node) and node.args[2] is None:
                linear_nodes[node.args[0]].append(node)

        # fuse linear nodes
        idx = -1
        num_matches = 0
        with cuda_memory_tracker():
            for parent_node, lin_children in linear_nodes.items():
                if len(lin_children) < 2:
                    continue
                if not check_same_children(parent_node, is_linear_op):
                    # Mixed children (e.g., quantized or non-linear) — skip fusion
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
class FuseFP8Gemms(QuantizationFusionMixin, BaseTransform):
    target_op = torch.ops.auto_deploy.torch_fake_quant_fp8_linear
    scale_groups = [["input_scale"], ["weight_scale"]]

    def fuse_rule(
        self, weights: List[torch.Tensor], **scales
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weight_scale: List[torch.Tensor] = scales["weight_scale"]
        input_scale: List[torch.Tensor] = scales["input_scale"]

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

    def build_custom_args_for_linear(self, scale_getattrs: Dict[str, Node]) -> Tuple[object, ...]:
        # (..., bias, input_scale(list), weight_scale(list), input_zp(list), weight_zp(list))
        return (
            [scale_getattrs["input_scale"]],
            [scale_getattrs["weight_scale"]],
            [],
            [],
        )

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        return self._apply_fusion_pass(gm, cm, factory, shared_config)


@TransformRegistry.register("fuse_fp4_gemms")
class FuseFP4Gemms(QuantizationFusionMixin, BaseTransform):
    target_op = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear
    scale_groups = [["input_scale"], ["weight_scale", "alpha"]]

    def fuse_rule(
        self, weights: List[torch.Tensor], **scales
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weight_scale: List[torch.Tensor] = scales["weight_scale"]
        input_scale: List[torch.Tensor] = scales["input_scale"]
        alpha: List[torch.Tensor] = scales["alpha"]

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

    def build_custom_args_for_linear(self, scale_getattrs: Dict[str, Node]) -> Tuple[object, ...]:
        # (..., bias, input_scale(list), weight_scale(list-with-alpha), input_zp(list), weight_zp(list))
        return (
            [scale_getattrs["input_scale"]],
            [scale_getattrs["weight_scale"], scale_getattrs["alpha"]],
            [],
            [],
        )

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        return self._apply_fusion_pass(gm, cm, factory, shared_config)
