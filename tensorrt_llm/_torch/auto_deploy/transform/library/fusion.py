import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache, partial
from itertools import chain
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import delete_all_unused_submodules, eliminate_dead_code
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    WeightBiasInfoCache,
    extract_weight_name,
    is_fake_quantized_linear_op,
    is_linear_op,
    is_op,
)
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _insert_fused_gemm(
    gm: GraphModule,
    idx: int,
    parent_node: Node,
    linear_nodes: List[Node],
    allow_not_contigous: bool = True,
) -> bool:
    """Fuse GEMMs sharing the same input activation.

    Args:
        allow_not_contigous: If True, split output via torch.narrow (zero-copy view).
            If False, split via torch.split + .contiguous() (independent copies).

    # before fusion:
    w1 = out1 x in,  w2 = out2 x in
    y1 = x @ w1.T,   y2 = x @ w2.T

    # after fusion (allow_not_contigous=True):
    w = (out1+out2) x in
    y = x @ w.T
    y1 = y.narrow(-1, 0, out1)          # view, no copy
    y2 = y.narrow(-1, out1, out2)        # view, no copy

    # after fusion (allow_not_contigous=False):
    w = (out1+out2) x in
    y = x @ w.T
    y1, y2 = split(y)                   # contiguous copies

    Bias handling:
        All children must have uniform bias state (all with bias or none).
        Each bias must be 1D per-channel matching its weight's out_features.
        Stacked bias is the dim=0 concatenation, mirroring weight stacking.
    """
    keys_unfused = [extract_weight_name(n) for n in linear_nodes]
    params_unfused = [gm.get_parameter(k) for k in keys_unfused]
    sizes_unfused = [p.size(0) for p in params_unfused]

    dtypes = {p.dtype for p in params_unfused}
    if len(dtypes) != 1:
        ad_logger.warning(f"Skipping GEMM fusion for {keys_unfused}: mixed dtypes {dtypes}")
        return False
    weight_dtype = dtypes.pop()

    # --- Bias fusibility check (all-or-none + 1D per-channel + size match) ---
    bias_args = [n.args[2] for n in linear_nodes]
    bias_present = [b is not None for b in bias_args]
    if any(bias_present) and not all(bias_present):
        # Mixed bias state — would require padding with zeros; bail out.
        return False
    has_bias = bias_present[0]
    bias_params: List[torch.Tensor] = []
    if has_bias:
        for n, w_param in zip(linear_nodes, params_unfused):
            bnode = n.args[2]
            # Only fuse statically known biases (get_attr nodes).
            if bnode.op != "get_attr":
                ad_logger.warning(
                    f"Skipping GEMM fusion for {keys_unfused}: bias is not a get_attr node"
                )
                return False
            bp = gm.get_parameter(bnode.target)
            # Reject anything other than per-channel 1D bias matching out_features.
            if bp.dim() != 1 or bp.size(0) != w_param.size(0):
                ad_logger.warning(
                    f"Skipping GEMM fusion for {keys_unfused}: non per-channel bias "
                    f"(weight out={w_param.size(0)}, bias shape={tuple(bp.shape)})"
                )
                return False
            bias_params.append(bp)
        bias_dtypes = {p.dtype for p in bias_params}
        if len(bias_dtypes) != 1:
            ad_logger.warning(
                f"Skipping GEMM fusion for {keys_unfused}: mixed bias dtypes {bias_dtypes}"
            )
            return False

    key_fused = f"fused_weight_{idx}"
    fused_weight = torch.cat(params_unfused, dim=0).to(weight_dtype)
    param_fused = nn.Parameter(fused_weight, requires_grad=False)
    setattr(gm, key_fused, param_fused)

    bias_key_fused = None
    if has_bias:
        bias_key_fused = f"fused_bias_{idx}"
        bias_dtype = bias_params[0].dtype
        fused_bias = torch.cat(bias_params, dim=0).to(bias_dtype)
        bias_param_fused = nn.Parameter(fused_bias, requires_grad=False)
        setattr(gm, bias_key_fused, bias_param_fused)

    ad_logger.warning(
        f"Fusing {len(linear_nodes)} GEMMs ({keys_unfused}) into {key_fused} "
        f"(dtype={weight_dtype}, bias={'yes' if has_bias else 'no'})"
    )

    fused_kwargs = dict(linear_nodes[0].kwargs)
    ref_val = linear_nodes[0].meta.get("val")

    with gm.graph.inserting_before(linear_nodes[0]):
        get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)
        get_bias_node = gm.graph.get_attr(bias_key_fused, torch.Tensor) if has_bias else None

    with gm.graph.inserting_before(linear_nodes[0]):
        fused_linear_node = gm.graph.call_function(
            linear_nodes[0].target,
            args=(parent_node, get_param_node, get_bias_node),
            kwargs=fused_kwargs,
        )
        if ref_val is not None:
            fused_out_shape = (*ref_val.shape[:-1], sum(sizes_unfused))
            fused_linear_node.meta["val"] = torch.empty(
                fused_out_shape, dtype=ref_val.dtype, device="meta"
            )

    if allow_not_contigous:
        offset = 0
        for i, n in enumerate(linear_nodes):
            size = sizes_unfused[i]
            with gm.graph.inserting_before(n):
                narrow_node = gm.graph.call_function(
                    torch.narrow, args=(fused_linear_node, -1, offset, size)
                )
                if ref_val is not None:
                    narrow_node.meta["val"] = torch.empty(
                        (*ref_val.shape[:-1], size), dtype=ref_val.dtype, device="meta"
                    )
            n.replace_all_uses_with(narrow_node)
            offset += size
    else:

        def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

        with gm.graph.inserting_before(linear_nodes[0]):
            split_node = gm.graph.call_function(split_output, args=(fused_linear_node,))

        for i, n in enumerate(linear_nodes):
            with gm.graph.inserting_before(n):
                get_split_node = gm.graph.call_function(operator.getitem, args=(split_node, i))
            n.replace_all_uses_with(get_split_node)

    # Clean up deleted modules to save GPU memory
    eliminate_dead_code(gm)
    delete_all_unused_submodules(gm)
    return True


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
        self,
        gm: GraphModule,
        idx: int,
        parent_node: Node,
        linear_nodes: List[Node],
        allow_not_contigous: bool = True,
    ) -> bool:
        """Fuse quantized GEMMs sharing the same input activation.

        Args:
            allow_not_contigous: If True, split output via torch.narrow (zero-copy view).
                If False, split via torch.split + .contiguous() (independent copies).
        """
        keys_unfused = [extract_weight_name(n) for n in linear_nodes]
        params_unfused = [gm.get_parameter(k) for k in keys_unfused]
        sizes_unfused = [p.size(0) for p in params_unfused]
        key_fused = f"fused_weight_{idx}"

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
            return False
        param_fused = nn.Parameter(weights_fused, requires_grad=False)
        setattr(gm, key_fused, param_fused)
        for name, buf in buffers_fused.items():
            gm.register_buffer(f"{key_fused}_{name}", buf)

        ad_logger.debug(
            f"Fusing {len(linear_nodes)} quantized GEMMs ({keys_unfused}) into {key_fused}"
        )

        # Handle fused_kwargs for quantized fused gemm.
        fused_kwargs = dict(linear_nodes[0].kwargs)
        with gm.graph.inserting_before(linear_nodes[0]):
            get_param_node = gm.graph.get_attr(key_fused, torch.Tensor)
            get_param_node.meta["val"] = torch.empty(
                param_fused.shape, dtype=param_fused.dtype, device="meta"
            )

            # For each kwarg group (e.g., input_scale, weight_scale[, alpha]),
            # create a list of get_attr nodes in the same structure the op expects.
            scale_getattrs: Dict[str, Node] = {}
            for name in flat_scale_names:
                attr_node = gm.graph.create_node("get_attr", f"{key_fused}_{name}")
                buf = buffers_fused[name]
                attr_node.meta["val"] = torch.empty(buf.shape, dtype=buf.dtype, device="meta")
                scale_getattrs[name] = attr_node
            custom_tail_args = self.build_custom_args_for_linear(scale_getattrs)

        ref_val = linear_nodes[0].meta.get("val")

        # add new linear node + output splitting
        with gm.graph.inserting_before(linear_nodes[0]):
            fused_linear_node = gm.graph.call_function(
                linear_nodes[0].target,
                args=(parent_node, get_param_node, None, *custom_tail_args),
                kwargs=fused_kwargs,
            )
            if ref_val is not None:
                fused_out_shape = (*ref_val.shape[:-1], sum(sizes_unfused))
                fused_linear_node.meta["val"] = torch.empty(
                    fused_out_shape, dtype=ref_val.dtype, device="meta"
                )

        if allow_not_contigous:
            offset = 0
            for i, n in enumerate(linear_nodes):
                size = sizes_unfused[i]
                with gm.graph.inserting_before(n):
                    narrow_node = gm.graph.call_function(
                        torch.narrow, args=(fused_linear_node, -1, offset, size)
                    )
                    if ref_val is not None:
                        narrow_node.meta["val"] = torch.empty(
                            (*ref_val.shape[:-1], size), dtype=ref_val.dtype, device="meta"
                        )
                n.replace_all_uses_with(narrow_node)
                offset += size
        else:

            def split_output(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
                """Split the output tensor of the fused linear node to obtain the original outputs."""
                return tuple(t.contiguous() for t in torch.split(tensor, sizes_unfused, dim=-1))

            with gm.graph.inserting_before(linear_nodes[0]):
                split_node = gm.graph.call_function(split_output, args=(fused_linear_node,))

            for i, n in enumerate(linear_nodes):
                with gm.graph.inserting_before(n):
                    get_split_node = gm.graph.call_function(operator.getitem, args=(split_node, i))
                n.replace_all_uses_with(get_split_node)

        # Clean up deleted modules to save GPU memory
        eliminate_dead_code(gm)
        delete_all_unused_submodules(gm)
        return True

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
                if self._insert_fused_quant_gemm(
                    gm, idx := idx + 1, parent_node, lin_children, allow_not_contigous=False
                ):
                    num_matches += 1

        torch.cuda.empty_cache()
        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
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
            if is_linear_op(node):
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

                # Group siblings by uniform bias state so each subgroup
                # (all-no-bias / all-with-bias) can fuse independently.
                no_bias_group = [c for c in lin_children if c.args[2] is None]
                with_bias_group = [c for c in lin_children if c.args[2] is not None]

                for group in (no_bias_group, with_bias_group):
                    if len(group) < 2:
                        continue
                    # linear nodes to fuse (split+copy for contiguous outputs)
                    if _insert_fused_gemm(
                        gm, idx := idx + 1, parent_node, group, allow_not_contigous=False
                    ):
                        num_matches += 1

        torch.cuda.empty_cache()

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


def _get_op_key(node: Node):
    """Get canonical op key for grouping nodes by quantization scheme.

    Resolves specific overloads (e.g., torch_fake_quant_fp8_linear.default) back
    to their overload packet so that all overloads of the same op are grouped together.
    """
    target = node.target
    return target.overloadpacket if hasattr(target, "overloadpacket") else target


@lru_cache(maxsize=None)
def _get_quant_fuser(op_key):
    """Get or create a lightweight QuantizationFusionMixin adapter for quantized GDN fusion.

    Reuses fuse_rule and build_custom_args_for_linear from the existing FP8/FP4
    fusion classes without requiring BaseTransform config.
    """
    # Lazily resolved: fusion classes are defined later in this module.
    # Use getattr to avoid AttributeError when an op is not yet registered.
    _OP_TO_CLS = {
        torch.ops.auto_deploy.torch_fake_quant_fp8_linear: FuseFP8Gemms,
        torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear: FuseFP4Gemms,
    }
    # TODO: once the finegrained FP8 custom op is always registered,
    # move this into _OP_TO_CLS above and remove the getattr guard.
    _fg_fp8_op = getattr(torch.ops.auto_deploy, "torch_fake_quant_finegrained_fp8_linear", None)
    if _fg_fp8_op is not None:
        _OP_TO_CLS[_fg_fp8_op] = FuseFineGrainedFP8Gemms
    src_cls = _OP_TO_CLS.get(op_key)
    if src_cls is None:
        return None

    return type(
        f"_MixedChildren{src_cls.__name__}",
        (QuantizationFusionMixin,),
        {
            "target_op": src_cls.target_op,
            "scale_groups": src_cls.scale_groups,
            "fuse_rule": src_cls.fuse_rule,
            "build_custom_args_for_linear": src_cls.build_custom_args_for_linear,
        },
    )()


@TransformRegistry.register("fuse_gemms_mixed_children")
class FuseGemmsMixedChildren(BaseTransform):
    """Fuse linear projections sharing the same input, even when the parent has
    non-linear users (e.g., shape access).

    This is a relaxed variant of FuseGemms: it does NOT require all children of
    the parent to be linear ops — only that at least 2 linear children exist.
    The fused output is split via torch.narrow (zero-copy view).

    Handles both non-quantized and quantized (FP8, FP4) linear ops. Nodes are
    grouped by (parent, quantization scheme) so only linears with the same parent
    AND the same op target are fused together.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        grouped_nodes: Dict[tuple, List[Node]] = defaultdict(list)
        for node in gm.graph.nodes:
            if (is_linear_op(node) or is_fake_quantized_linear_op(node)) and node.args[2] is None:
                # Skip linears with a unit dimension (e.g., [1, H] scalar gates).
                # A weight with dim=1 is effectively a lower-order tensor and
                # should not be fused with proper matrix projections.
                try:
                    w = gm.get_parameter(extract_weight_name(node))
                    if any(d == 1 for d in w.shape):
                        continue
                except (AttributeError, KeyError):
                    pass
                grouped_nodes[(node.args[0], _get_op_key(node))].append(node)

        idx = -1
        num_matches = 0
        with WeightBiasInfoCache(), cuda_memory_tracker():
            for (parent_node, op_key), lin_children in grouped_nodes.items():
                if len(lin_children) < 2:
                    continue
                if is_linear_op(lin_children[0]):
                    if _insert_fused_gemm(
                        gm, idx := idx + 1, parent_node, lin_children, allow_not_contigous=False
                    ):
                        num_matches += 1
                else:
                    fuser = _get_quant_fuser(op_key)
                    if fuser is None:
                        ad_logger.warning(
                            f"No quantized fuser for {op_key}, skipping mixed-children fusion"
                        )
                        continue
                    if fuser._insert_fused_quant_gemm(
                        gm, idx := idx + 1, parent_node, lin_children, allow_not_contigous=False
                    ):
                        num_matches += 1

        torch.cuda.empty_cache()

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
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


@TransformRegistry.register("fuse_finegrained_fp8_gemms")
class FuseFineGrainedFP8Gemms(QuantizationFusionMixin, BaseTransform):
    """Fuse FineGrained (block-wise) FP8 GEMMs sharing the same input activation.

    FineGrained FP8 uses per-block weight scales (weight_scale_inv) and dynamic
    input quantization, so fusion simply concatenates weights and their block scales
    along the output dimension.
    """

    target_op = getattr(torch.ops.auto_deploy, "torch_fake_quant_finegrained_fp8_linear", None)
    scale_groups = [["weight_scale_inv"]]

    def fuse_rule(
        self, weights: List[torch.Tensor], **scales
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weight_scale_inv: List[torch.Tensor] = scales["weight_scale_inv"]

        # The kernel infers block_n = N // scale_n at runtime.  After catting
        # weights and scales along dim-0 the fused block_n must still be
        # consistent, i.e.  sum(N_i) // sum(scale_n_i) == each per-weight
        # block_n_i.  This only holds when every N_i is an exact multiple of
        # the true block size (typically 128).
        block_ns = [w.size(0) // ws.size(0) for w, ws in zip(weights, weight_scale_inv)]
        if len(set(block_ns)) != 1:
            raise NotImplementedError(
                f"Cannot fuse finegrained FP8: inconsistent per-weight block sizes {block_ns}"
            )
        block_n = block_ns[0]

        total_N = sum(w.size(0) for w in weights)
        total_scale_n = sum(ws.size(0) for ws in weight_scale_inv)
        if total_scale_n == 0 or total_N // total_scale_n != block_n:
            raise NotImplementedError(
                f"Cannot fuse finegrained FP8: fused block_n "
                f"({total_N // total_scale_n if total_scale_n else 'N/A'}) != {block_n}"
            )

        fused_weights = torch.cat(weights, dim=0)
        fused_weight_scale_inv = torch.cat(weight_scale_inv, dim=0)

        return fused_weights, {"weight_scale_inv": fused_weight_scale_inv}

    def build_custom_args_for_linear(self, scale_getattrs: Dict[str, Node]) -> Tuple[object, ...]:
        return (
            [],
            [scale_getattrs["weight_scale_inv"]],
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
