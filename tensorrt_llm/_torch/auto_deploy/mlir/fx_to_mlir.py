# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FX graph → MLIR (xDSL) converter.

Walks an FX ``GraphModule`` topologically and emits ``ad`` dialect ops.
Tensor shapes/dtypes are extracted from ``node.meta["val"]`` (FakeTensor).
FX metadata is stored in a Python-side dict keyed by a unique node name
for round-trip fidelity.
"""

import operator
from typing import Any, Dict, List, Optional

import torch
from torch.fx import GraphModule, Node
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.ir import Block, Region, SSAValue

from ..utils.node_utils import (
    collect_terminal_users_through_passthrough,
    extract_op_args,
    extract_output_tuple,
    is_dtype_cast_op,
    is_op,
    unwrap_input_through_passthrough,
)
from .dialect import (
    AdAdd,
    AdDiv,
    AdEq,
    AdExp,
    AdFloorDiv,
    AdGatedRMSNorm,
    AdGelu,
    AdGraphInput,
    AdGraphOutput,
    AdMul,
    AdNeg,
    AdOpaque,
    AdPow,
    AdQuantFP8,
    AdReduceMean,
    AdReduceSum,
    AdRelu,
    AdRMSNorm,
    AdRsqrt,
    AdSigmoid,
    AdSilu,
    AdSlice,
    AdSoftplus,
    AdSplat,
    AdSqrt,
    AdSub,
    AdTanh,
    AdToDtype,
    Float8E4M3FNType,
    register_ad_dialect,
    tensor_type_from_meta,
)

# All rmsnorm FX targets that map to the single ad.rmsnorm op
_RMSNORM_TARGETS = []


def _init_rmsnorm_targets():
    """Lazily collect rmsnorm op targets (avoids import-time torch.ops resolution)."""
    global _RMSNORM_TARGETS
    if _RMSNORM_TARGETS:
        return
    for attr_name in [
        "flashinfer_rms_norm",
        "triton_rms_norm",
        "torch_rmsnorm",
    ]:
        op = getattr(torch.ops.auto_deploy, attr_name, None)
        if op is not None:
            _RMSNORM_TARGETS.append(op)


def _is_rmsnorm(node: Node) -> bool:
    """Return True if *node* calls any registered rmsnorm variant."""
    _init_rmsnorm_targets()
    return is_op(node, _RMSNORM_TARGETS) if _RMSNORM_TARGETS else False


# All gated rmsnorm FX targets
_GATED_RMSNORM_TARGETS = []


def _init_gated_rmsnorm_targets():
    """Lazily collect gated rmsnorm op targets."""
    global _GATED_RMSNORM_TARGETS
    if _GATED_RMSNORM_TARGETS:
        return
    for attr_name in [
        "triton_rmsnorm_gated",
        "torch_rmsnorm_gated",
    ]:
        op = getattr(torch.ops.auto_deploy, attr_name, None)
        if op is not None:
            _GATED_RMSNORM_TARGETS.append(op)


def _is_gated_rmsnorm(node: Node) -> bool:
    """Return True if *node* calls any registered gated rmsnorm variant."""
    _init_gated_rmsnorm_targets()
    return is_op(node, _GATED_RMSNORM_TARGETS) if _GATED_RMSNORM_TARGETS else False


_L2NORM_TARGETS = []


def _init_l2norm_targets():
    """Lazily collect l2norm op targets."""
    global _L2NORM_TARGETS
    if _L2NORM_TARGETS:
        return
    for attr_name in [
        "torch_l2norm",
        "fla_l2norm",
    ]:
        op = getattr(torch.ops.auto_deploy, attr_name, None)
        if op is not None:
            _L2NORM_TARGETS.append(op)


def _is_l2norm(node: Node) -> bool:
    """Return True if *node* calls any registered l2norm variant."""
    _init_l2norm_targets()
    return is_op(node, _L2NORM_TARGETS) if _L2NORM_TARGETS else False


def _get_fake_tensor(node: Node) -> Optional[torch.Tensor]:
    """Extract the FakeTensor from node metadata, if available."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val
    return None


def _result_type_for_node(node: Node) -> TensorType:
    """Return an MLIR TensorType for *node*'s output tensor."""
    ft = _get_fake_tensor(node)
    if ft is not None:
        return tensor_type_from_meta(ft)
    # Fallback: unranked f32 tensor (should not happen with proper shape prop)
    from xdsl.dialects.builtin import Float32Type

    return TensorType(Float32Type(), [-1])


def _normalize_last_dim_reduction(dim: int, input_val: SSAValue) -> int:
    """Normalize positive last-dimension reductions to ``-1`` for codegen."""
    if isinstance(input_val.type, TensorType):
        rank = len(input_val.type.get_shape())
        if dim == rank - 1:
            return -1
    return dim


def _dtype_id_from_torch_dtype(dtype: torch.dtype) -> int:
    """Return the stable dtype id used by ``ad.to_dtype``."""
    dtype_to_id = {
        torch.float16: 5,
        torch.bfloat16: 15,
        torch.float32: 6,
        torch.float64: 7,
    }
    return dtype_to_id.get(dtype, hash(str(dtype)) % (2**31))


def _is_getitem(node: Node) -> bool:
    """Return True if *node* is an ``operator.getitem`` call."""
    return node.op == "call_function" and node.target is operator.getitem


def _is_split_with_sizes(node: Node) -> bool:
    """Return True for supported split-with-sizes ops."""
    targets = [torch.ops.aten.split_with_sizes.default]
    ad_ops = getattr(torch.ops, "auto_deploy", None)
    ad_split = getattr(ad_ops, "split_with_sizes", None) if ad_ops is not None else None
    if ad_split is not None:
        targets.append(getattr(ad_split, "default", ad_split))
    return is_op(node, targets)


def _last_dim_from_node_meta(node: Node) -> Optional[int]:
    """Return the static last dimension from FX tensor metadata, if present."""
    val = node.meta.get("val")
    if hasattr(val, "shape") and len(val.shape) > 0:
        last_dim = val.shape[-1]
        return int(last_dim) if isinstance(last_dim, int) else None
    tensor_meta = node.meta.get("tensor_meta")
    if hasattr(tensor_meta, "shape") and len(tensor_meta.shape) > 0:
        last_dim = tensor_meta.shape[-1]
        return int(last_dim) if isinstance(last_dim, int) else None
    return None


def _is_supported_fp8_linear(node: Node) -> bool:
    """Return True for FP8 quantized linear ops that MLIR can prequantize."""
    return is_op(
        node,
        [
            torch.ops.auto_deploy.trtllm_quant_fp8_linear,
            torch.ops.auto_deploy.torch_quant_fp8_linear,
        ],
    )


def _is_supported_nvfp4_linear(node: Node) -> bool:
    """Return True for NVFP4 quantized linear ops that MLIR can prequantize."""
    return is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear)


def _extract_nvfp4_linear_args(node: Node) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Extract canonical NVFP4 quant-linear arguments from FX args/kwargs."""

    def _arg(pos: int, name: str, default=None):
        if len(node.args) > pos:
            return node.args[pos]
        return node.kwargs.get(name, default)

    return (
        _arg(0, "input"),
        _arg(1, "weight_fp4"),
        _arg(2, "bias"),
        _arg(3, "input_scale"),
        _arg(4, "weight_scale"),
        _arg(5, "alpha"),
    )


def _same_scale_node(lhs: Node, rhs: Node) -> bool:
    """Return True if two scale nodes are the same stable FX value."""
    if lhs is rhs:
        return True
    return lhs.op == "get_attr" and rhs.op == "get_attr" and lhs.target == rhs.target


def _unwrap_post_norm_nodes(node: Node) -> tuple[Node, list[Node]]:
    """Walk through dtype/view passthroughs from a norm output toward quant consumers."""
    return unwrap_input_through_passthrough(node, allow_dtype_cast=True)


def _has_unsupported_post_norm_nodes(post_nodes: list[Node]) -> bool:
    """NVFP4 norm quant can absorb dtype casts, but not arbitrary post-norm ops."""
    return any(not is_dtype_cast_op(node) for node in post_nodes)


def _nvfp4_linear_out_dtype(linear_node: Node, source_node: Node) -> torch.dtype:
    """Return the output dtype expected by ``trtllm_nvfp4_prequant_linear``."""
    input_arg, _, _, _, _, _ = _extract_nvfp4_linear_args(linear_node)
    input_dtype = (
        _extract_tensor_dtype_from_meta(input_arg) if isinstance(input_arg, Node) else None
    )
    return (
        _extract_tensor_dtype_from_meta(linear_node)
        or input_dtype
        or _extract_tensor_dtype_from_meta(source_node)
        or torch.bfloat16
    )


def _is_getitem_index(node: Node, index: int) -> bool:
    """Return True if *node* is ``operator.getitem(..., index)``."""
    return node.op == "call_function" and node.target is operator.getitem and node.args[1] == index


def _extract_nonquant_allreduce_norm(node: Node):
    """Extract the non-quant allreduce+residual+RMSNorm producer for output 0."""
    if not _is_getitem_index(node, 0):
        return None

    source_node = node.args[0]
    if not isinstance(source_node, Node) or not is_op(
        source_node, torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm
    ):
        return None

    tensor, residual, norm_weight, eps, strategy = source_node.args
    _, residual_out = extract_output_tuple(source_node, count=2)
    return source_node, tensor, residual, norm_weight, eps, strategy, residual_out


def _extract_add_rmsnorm(node: Node):
    """Extract add(+optional cast)+RMSNorm inputs for NVFP4 quant rewrites."""
    if not is_op(
        node,
        [
            torch.ops.auto_deploy.flashinfer_rms_norm,
            torch.ops.auto_deploy.torch_rmsnorm,
            torch.ops.auto_deploy.triton_rms_norm,
        ],
    ):
        return None

    norm_input, norm_weight, eps = node.args
    pre_norm_cast = None
    if isinstance(norm_input, Node) and is_dtype_cast_op(norm_input):
        pre_norm_cast = norm_input
        norm_input = norm_input.args[0]

    if not isinstance(norm_input, Node) or not is_op(norm_input, torch.ops.aten.add.Tensor):
        return None

    add_lhs, add_rhs = norm_input.args[:2]
    if not isinstance(add_lhs, Node) or not isinstance(add_rhs, Node):
        return None

    return norm_input, pre_norm_cast, add_lhs, add_rhs, norm_weight, eps


def _extract_gated_rmsnorm(node: Node):
    """Extract supported gated RMSNorm inputs for NVFP4 quant rewrites."""
    if not is_op(
        node,
        [
            torch.ops.auto_deploy.torch_rmsnorm_gated,
            torch.ops.auto_deploy.triton_rmsnorm_gated,
        ],
    ):
        return None

    x, weight, gate, eps, group_size, norm_before_gate = extract_op_args(
        node, "x", "weight", "gate", "eps", "group_size", "norm_before_gate"
    )
    if gate is None or norm_before_gate:
        return None
    return x, weight, gate, eps, group_size


def _collect_grouped_nvfp4_linear_users(
    source_node: Node,
    seed_user: Node,
    seed_scale: Node,
    processed_users: set[int],
) -> list[Node]:
    """Collect terminal NVFP4 linear users sharing one norm source and scale."""
    terminal_users, traversal_ok = collect_terminal_users_through_passthrough(
        source_node, allow_dtype_cast=True
    )
    if not traversal_ok:
        return []

    grouped_users: list[Node] = []
    for user in terminal_users:
        if not _is_supported_nvfp4_linear(user) or id(user) in processed_users:
            continue

        input_arg, _, _, input_scale, _, _ = _extract_nvfp4_linear_args(user)
        if not isinstance(input_arg, Node) or not isinstance(input_scale, Node):
            continue

        user_source, post_nodes = _unwrap_post_norm_nodes(input_arg)
        if user_source is not source_node or _has_unsupported_post_norm_nodes(post_nodes):
            continue
        if not _same_scale_node(seed_scale, input_scale):
            continue

        grouped_users.append(user)

    if seed_user not in grouped_users:
        grouped_users.append(seed_user)

    return grouped_users


def _all_terminal_users_are_grouped(source_node: Node, grouped_users: list[Node]) -> bool:
    """Return True if every terminal source user is in the supplied group."""
    terminal_users, traversal_ok = collect_terminal_users_through_passthrough(
        source_node, allow_dtype_cast=True
    )
    if not traversal_ok:
        return False
    grouped_user_set = set(grouped_users)
    return all(user in grouped_user_set for user in terminal_users)


def _has_terminal_users_outside_group(source_node: Node, grouped_users: list[Node]) -> bool:
    """Return True if terminal source users include non-group consumers."""
    terminal_users, traversal_ok = collect_terminal_users_through_passthrough(
        source_node, allow_dtype_cast=True
    )
    if not traversal_ok:
        return True
    grouped_user_set = set(grouped_users)
    return any(user not in grouped_user_set for user in terminal_users)


def _extract_tensor_dtype_from_meta(node: Node) -> Optional[torch.dtype]:
    """Return tensor dtype from FX metadata, if available."""
    val = node.meta.get("val")
    if hasattr(val, "dtype"):
        return val.dtype
    tensor_meta = node.meta.get("tensor_meta")
    if hasattr(tensor_meta, "dtype"):
        return tensor_meta.dtype
    return None


def _torch_dtype_name(dtype: Optional[torch.dtype]) -> str:
    """Return the torch dtype name expected by prequant custom ops."""
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return str(dtype).removeprefix("torch.")
    return "bfloat16"


def _is_relu(node: Node) -> bool:
    """Return True for supported ReLU ops."""
    return is_op(node, [torch.ops.aten.relu, torch.ops.aten.relu.default])


def _extract_relu2_source_for_nvfp4_quant(node: Node) -> Optional[Node]:
    """Return pre-ReLU input for an exclusive ReLU2 -> NVFP4 quant-linear path."""
    if not isinstance(node, Node) or node.op != "call_function":
        return None
    if len(node.users) != 1:
        return None
    quant_user = next(iter(node.users))
    if not _is_supported_nvfp4_linear(quant_user):
        return None

    relu_node = None
    if is_op(node, [torch.ops.aten.square, torch.ops.aten.square.default]):
        relu_node = node.args[0]
    elif is_op(node, torch.ops.aten.pow.Tensor_Scalar):
        exponent = node.args[1] if len(node.args) > 1 else None
        if exponent != 2:
            return None
        relu_node = node.args[0]
    elif is_op(node, torch.ops.aten.mul.Tensor):
        lhs, rhs = node.args[:2]
        if lhs is not rhs:
            return None
        relu_node = lhs

    if not isinstance(relu_node, Node) or not _is_relu(relu_node):
        return None
    if len(relu_node.users) != 1:
        return None

    relu_input = relu_node.args[0] if relu_node.args else None
    return relu_input if isinstance(relu_input, Node) else None


def _is_relu2_nvfp4_quant_node(node: Node) -> bool:
    """Return True if *node* is the ReLU2 node consumed by an NVFP4 quant-linear."""
    return _extract_relu2_source_for_nvfp4_quant(node) is not None


def _is_relu_nvfp4_quant_node(node: Node) -> bool:
    """Return True if *node* is the ReLU feeding an exclusive ReLU2 NVFP4 path."""
    if not _is_relu(node) or len(node.users) != 1:
        return False
    relu2_node = next(iter(node.users))
    return _is_relu2_nvfp4_quant_node(relu2_node)


def _nvfp4_quant_output_shapes(input_shape: tuple[int, ...]) -> Optional[tuple[list[int], int]]:
    """Return packed FP4 shape and swizzled scale-factor size for static shapes."""
    if not input_shape or input_shape[-1] <= 0 or input_shape[-1] % 16 != 0:
        return None
    m = 1
    for dim in input_shape[:-1]:
        if dim <= 0:
            return None
        m *= dim

    n = input_shape[-1]
    packed_shape = list(input_shape)
    packed_shape[-1] //= 2
    padded_m = ((m + 127) // 128) * 128
    padded_cols = (((n // 16) + 3) // 4) * 4
    sf_size = padded_m * padded_cols
    return packed_shape, sf_size


class FXToMLIRConverter:
    """Convert an FX ``GraphModule`` to an xDSL ``ModuleOp`` using the ``ad`` dialect.

    After conversion:
    - ``self.mlir_module``: the xDSL ``ModuleOp``
    - ``self.metadata``: dict mapping ``node_name → node.meta`` for round-trip
    """

    def __init__(self, gm: GraphModule, enabled_rewrites: Optional[set[str]] = None):
        self._gm = gm
        self._enabled_rewrites = enabled_rewrites
        # node name → MLIR SSAValue (for wiring operands)
        self._value_map: Dict[str, SSAValue] = {}
        # node name → node.meta (for round-trip)
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.mlir_module: Optional[ModuleOp] = None
        self.num_direct_rewrites = 0

    def convert(self) -> ModuleOp:
        """Run the conversion and return the MLIR module."""
        from xdsl.context import Context as MLContext

        ctx = MLContext()
        register_ad_dialect(ctx)

        block = Block()
        region = Region([block])

        for node in self._gm.graph.nodes:
            self._convert_node(node, block)
        if self._rewrite_enabled("rmsnorm_quant_nvfp4"):
            self._run_nvfp4_rmsnorm_quant_rewrites(block)

        self.mlir_module = ModuleOp(region)
        return self.mlir_module

    def _rewrite_enabled(self, rewrite_name: str) -> bool:
        return self._enabled_rewrites is None or rewrite_name in self._enabled_rewrites

    # ------------------------------------------------------------------
    # Per-node conversion
    # ------------------------------------------------------------------

    def _convert_node(self, node: Node, block: Block) -> None:
        """Dispatch to the appropriate handler based on FX node op type."""
        if node.op == "placeholder":
            self._convert_placeholder(node, block)
        elif node.op == "call_function":
            self._convert_call_function(node, block)
        elif node.op == "output":
            self._convert_output(node, block)
        elif node.op == "get_attr":
            # Parameters/buffers — treat as graph inputs for MLIR purposes
            self._convert_get_attr(node, block)
        elif node.op == "call_method" and node.target == "contiguous":
            self._convert_passthrough(node, block)
        elif node.op in ("call_module", "call_method"):
            self._convert_opaque(node, block)

    def _convert_placeholder(self, node: Node, block: Block) -> None:
        """Convert FX placeholder → ``ad.graph_input``."""
        result_type = _result_type_for_node(node)
        op = AdGraphInput.build(
            attributes={"input_name": StringAttr(node.name)},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)

    def _convert_get_attr(self, node: Node, block: Block) -> None:
        """Convert FX get_attr → ``ad.graph_input`` (parameters as inputs)."""
        result_type = _result_type_for_node(node)
        op = AdGraphInput.build(
            attributes={"input_name": StringAttr(node.name)},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        # Tag as get_attr so MLIR→FX can reconstruct it correctly
        self.metadata[node.name]["_fx_op"] = "get_attr"
        self.metadata[node.name]["_fx_target"] = node.target

    def _convert_output(self, node: Node, block: Block) -> None:
        """Convert FX output → ``ad.graph_output``."""
        # FX output args can be a nested tuple; flatten for MLIR but store
        # the original structure template for faithful reconstruction.
        flat_args = self._flatten_args(node.args[0])
        operands = [self._resolve_operand(a) for a in flat_args if isinstance(a, Node)]
        op = AdGraphOutput.build(operands=[operands])
        block.add_op(op)
        # Store the original output structure as metadata so MLIR→FX can rebuild it
        self.metadata["__output_structure__"] = node.args[0]

    def _convert_call_function(self, node: Node, block: Block) -> None:
        """Route call_function nodes to precise or opaque lowering."""
        import operator as op_module

        if _is_relu_nvfp4_quant_node(node) or _is_relu2_nvfp4_quant_node(node):
            if self._rewrite_enabled("relu2_quant_nvfp4"):
                self._store_meta(node)
            else:
                self._convert_opaque(node, block)
        elif is_op(node, torch.ops.aten.add.Tensor):
            self._convert_add(node, block)
        elif is_op(node, torch.ops.aten.mul.Tensor):
            self._convert_binary_elementwise(node, block, AdMul)
        elif is_op(node, torch.ops.aten.sub.Tensor):
            self._convert_binary_elementwise(node, block, AdSub)
        elif is_op(node, [torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar]):
            self._convert_binary_elementwise(node, block, AdDiv)
        elif is_op(node, [torch.ops.aten.neg, torch.ops.aten.neg.default]):
            self._convert_unary_elementwise(node, block, AdNeg)
        elif is_op(node, [torch.ops.aten.silu, torch.ops.aten.silu.default]):
            self._convert_unary_elementwise(node, block, AdSilu)
        elif is_op(node, [torch.ops.aten.gelu, torch.ops.aten.gelu.default]):
            self._convert_unary_elementwise(node, block, AdGelu)
        elif is_op(node, [torch.ops.aten.relu, torch.ops.aten.relu.default]):
            self._convert_unary_elementwise(node, block, AdRelu)
        elif is_op(node, [torch.ops.aten.tanh, torch.ops.aten.tanh.default]):
            self._convert_unary_elementwise(node, block, AdTanh)
        elif is_op(node, [torch.ops.aten.sigmoid, torch.ops.aten.sigmoid.default]):
            self._convert_unary_elementwise(node, block, AdSigmoid)
        elif is_op(node, [torch.ops.aten.exp, torch.ops.aten.exp.default]):
            self._convert_unary_elementwise(node, block, AdExp)
        elif is_op(node, [torch.ops.aten.softplus, torch.ops.aten.softplus.default]):
            self._convert_unary_elementwise(node, block, AdSoftplus)
        elif is_op(node, [torch.ops.aten.rsqrt, torch.ops.aten.rsqrt.default]):
            self._convert_unary_elementwise(node, block, AdRsqrt)
        elif is_op(node, [torch.ops.aten.sqrt, torch.ops.aten.sqrt.default]):
            self._convert_unary_elementwise(node, block, AdSqrt)
        elif is_op(node, [torch.narrow, torch.ops.aten.narrow.default]):
            self._convert_narrow(node, block)
        elif is_op(node, torch.ops.aten.slice.Tensor):
            self._convert_slice(node, block)
        elif _is_split_with_sizes(node):
            self._convert_split(node, block)
        elif is_op(node, torch.ops.aten.contiguous.default):
            self._convert_passthrough(node, block)
        elif is_op(node, [torch.ops.aten.square, torch.ops.aten.square.default]):
            self._convert_square(node, block)
        elif is_op(node, torch.ops.aten.pow.Tensor_Scalar):
            self._convert_pow(node, block)
        elif is_op(node, torch.ops.aten.sum.dim_IntList):
            self._convert_sum(node, block)
        elif is_op(node, torch.ops.aten.mean.dim):
            self._convert_mean(node, block)
        elif _is_rmsnorm(node):
            self._convert_rmsnorm(node, block)
        elif _is_gated_rmsnorm(node):
            self._convert_gated_rmsnorm(node, block)
        elif _is_l2norm(node) and self._rewrite_enabled("l2norm"):
            self._convert_l2norm(node, block)
        elif _is_l2norm(node):
            self._convert_opaque(node, block)
        elif (
            _is_supported_fp8_linear(node)
            and self._rewrite_enabled("rmsnorm_quant_fp8")
            and (self._enabled_rewrites is None or self._is_rmsnorm_quant_fp8_linear(node))
        ):
            self._convert_fp8_quant_linear(node, block)
        elif _is_supported_fp8_linear(node):
            self._convert_opaque(node, block)
        elif (
            _is_supported_nvfp4_linear(node)
            and self._rewrite_enabled("relu2_quant_nvfp4")
            and self._has_relu2_nvfp4_source(node)
        ):
            self._convert_nvfp4_relu2_quant_linear(node, block)
        elif _is_supported_nvfp4_linear(node):
            self._convert_opaque(node, block)
        elif is_op(node, torch.ops.aten.to.dtype):
            self._convert_to_dtype(node, block)
        elif node.target is op_module.floordiv:
            self._convert_floordiv(node, block)
        elif node.target is op_module.eq or getattr(node.target, "__name__", "") == "eq":
            self._convert_eq(node, block)
        elif node.target is op_module.getitem:
            self._convert_getitem(node, block)
        else:
            self._convert_opaque(node, block)

    # ------------------------------------------------------------------
    # Precise lowerings
    # ------------------------------------------------------------------

    def _convert_add(self, node: Node, block: Block) -> None:
        """``aten.add.Tensor`` → ``ad.add``. Handles scalar second arg via splat."""
        result_type = _result_type_for_node(node)
        lhs = self._resolve_operand(node.args[0], block, result_type)
        rhs = self._resolve_operand(node.args[1], block, result_type)
        op = AdAdd.build(operands=[lhs, rhs], result_types=[result_type])
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_binary_elementwise(self, node: Node, block: Block, op_cls) -> None:
        """Generic binary elementwise: two inputs, one output. Handles scalar args via splat."""
        result_type = _result_type_for_node(node)
        lhs = self._resolve_operand(node.args[0], block, result_type)
        rhs = self._resolve_operand(node.args[1], block, result_type)
        op = op_cls.build(operands=[lhs, rhs], result_types=[result_type])
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_pow(self, node: Node, block: Block) -> None:
        """``aten.pow.Tensor_Scalar`` → ``ad.pow``."""
        base_val = self._resolve_operand(node.args[0])
        exponent = float(node.args[1])
        result_type = _result_type_for_node(node)
        op = AdPow.build(
            operands=[base_val],
            attributes={"exponent": FloatAttr(exponent, Float64Type())},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_square(self, node: Node, block: Block) -> None:
        """``aten.square`` → ``ad.mul(x, x)``."""
        result_type = _result_type_for_node(node)
        input_val = self._resolve_operand(node.args[0])
        op = AdMul.build(operands=[input_val, input_val], result_types=[result_type])
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_floordiv(self, node: Node, block: Block) -> None:
        """``operator.floordiv(tensor, scalar)`` → ``ad.floordiv``."""
        input_val = self._resolve_operand(node.args[0])
        divisor = int(node.args[1])
        result_type = _result_type_for_node(node)
        op = AdFloorDiv.build(
            operands=[input_val],
            attributes={"divisor": IntegerAttr(divisor, IntegerType(64))},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_narrow(self, node: Node, block: Block) -> None:
        """``torch.narrow(x, -1, start, length)`` → ``ad.slice``."""
        input_val = self._resolve_operand(node.args[0])
        dim = int(node.args[1])
        start = int(node.args[2])
        length = int(node.args[3])
        if dim != -1:
            self._convert_opaque(node, block)
            return
        result_type = _result_type_for_node(node)
        op = AdSlice.build(
            operands=[input_val],
            attributes={
                "dim": IntegerAttr(dim, IntegerType(64)),
                "start": IntegerAttr(start, IntegerType(64)),
                "length": IntegerAttr(length, IntegerType(64)),
            },
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_slice(self, node: Node, block: Block) -> None:
        """``aten.slice.Tensor(x, -1, start, end, 1)`` → ``ad.slice``."""
        input_val = self._resolve_operand(node.args[0])
        dim = int(node.args[1])
        start = int(node.args[2])
        end = int(node.args[3])
        step = int(node.args[4]) if len(node.args) > 4 else 1
        if dim != -1 or step != 1:
            self._convert_opaque(node, block)
            return
        result_type = _result_type_for_node(node)
        op = AdSlice.build(
            operands=[input_val],
            attributes={
                "dim": IntegerAttr(dim, IntegerType(64)),
                "start": IntegerAttr(start, IntegerType(64)),
                "length": IntegerAttr(end - start, IntegerType(64)),
            },
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_split(self, node: Node, block: Block) -> None:
        """Skip static split nodes whose getitems lower to ``ad.slice``."""
        split_info = self._extract_static_last_dim_split(node)
        if split_info is not None and all(_is_getitem(user) for user in node.users):
            self._store_meta(node)
            return

        self._convert_opaque(node, block)

    def _convert_passthrough(self, node: Node, block: Block) -> None:
        """Lower view/layout no-ops that do not change values for fusion purposes."""
        input_arg = node.args[0] if node.args else None
        if isinstance(input_arg, Node) and input_arg.name in self._value_map:
            self._value_map[node.name] = self._value_map[input_arg.name]
            self._store_meta(node)
            self.metadata[node.name]["_fx_node_name"] = node.name
            return
        self._convert_opaque(node, block)

    def _convert_eq(self, node: Node, block: Block) -> None:
        """``operator.eq(tensor, scalar)`` → ``ad.eq``."""
        input_val = self._resolve_operand(node.args[0])
        value = int(node.args[1])
        result_type = _result_type_for_node(node)
        op = AdEq.build(
            operands=[input_val],
            attributes={"value": IntegerAttr(value, IntegerType(64))},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_mean(self, node: Node, block: Block) -> None:
        """``aten.mean.dim`` → ``ad.reduce_mean``."""
        input_val = self._resolve_operand(node.args[0])
        dims = node.args[1]  # list of ints
        keepdim = node.args[2] if len(node.args) > 2 else False
        # Only single-dim reductions are supported; fall back to opaque for multi-axis.
        if isinstance(dims, (list, tuple)):
            if len(dims) != 1:
                self._convert_opaque(node, block)
                return
            dim = dims[0]
        else:
            dim = dims
        dim = _normalize_last_dim_reduction(int(dim), input_val)
        result_type = _result_type_for_node(node)
        op = AdReduceMean.build(
            operands=[input_val],
            attributes={
                "dim": IntegerAttr(dim, IntegerType(64)),
                "keepdim": IntegerAttr(1 if keepdim else 0, IntegerType(1)),
                "group_size": IntegerAttr(0, IntegerType(64)),
            },
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_sum(self, node: Node, block: Block) -> None:
        """``aten.sum.dim_IntList`` → ``ad.reduce_sum``."""
        input_val = self._resolve_operand(node.args[0])
        dims = node.args[1]
        keepdim = node.args[2] if len(node.args) > 2 else False
        if isinstance(dims, (list, tuple)):
            if len(dims) != 1:
                self._convert_opaque(node, block)
                return
            dim = dims[0]
        else:
            dim = dims
        dim = _normalize_last_dim_reduction(int(dim), input_val)
        result_type = _result_type_for_node(node)
        op = AdReduceSum.build(
            operands=[input_val],
            attributes={
                "dim": IntegerAttr(dim, IntegerType(64)),
                "keepdim": IntegerAttr(1 if keepdim else 0, IntegerType(1)),
            },
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_unary_elementwise(self, node: Node, block: Block, op_cls) -> None:
        """Generic unary elementwise: one input, one output."""
        input_val = self._resolve_operand(node.args[0])
        result_type = _result_type_for_node(node)
        op = op_cls.build(operands=[input_val], result_types=[result_type])
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _convert_rmsnorm(self, node: Node, block: Block) -> None:
        """Any rmsnorm variant → ``ad.rmsnorm`` (backend-agnostic)."""
        input_val = self._resolve_operand(node.args[0])
        weight_val = self._resolve_operand(node.args[1])
        eps_val = float(node.args[2]) if len(node.args) > 2 else 1e-5
        eps_attr = FloatAttr(eps_val, Float64Type())
        result_type = _result_type_for_node(node)
        # If no FakeTensor metadata (e.g., from match_rmsnorm_pattern),
        # use the input tensor's type — rmsnorm preserves shape.
        if result_type.get_shape() == (-1,) and isinstance(input_val.type, TensorType):
            result_type = input_val.type
        op = AdRMSNorm.build(
            operands=[input_val, weight_val],
            attributes={"eps": eps_attr},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)

    def _convert_gated_rmsnorm(self, node: Node, block: Block) -> None:
        """Gated rmsnorm variant → ``ad.gated_rmsnorm``.

        FX signature: ``(x, weight, gate, eps, group_size, norm_before_gate)``
        """
        input_val = self._resolve_operand(node.args[0])
        weight_val = self._resolve_operand(node.args[1])
        gate_val = self._resolve_operand(node.args[2])
        eps_val = float(node.args[3]) if len(node.args) > 3 else 1e-5
        group_size = int(node.args[4]) if len(node.args) > 4 else 0
        norm_before_gate = bool(node.args[5]) if len(node.args) > 5 else False
        eps_attr = FloatAttr(eps_val, Float64Type())
        result_type = _result_type_for_node(node)
        if result_type.get_shape() == (-1,) and isinstance(input_val.type, TensorType):
            result_type = input_val.type
        op = AdGatedRMSNorm.build(
            operands=[input_val, weight_val, gate_val],
            attributes={
                "eps": eps_attr,
                "group_size": IntegerAttr(group_size, IntegerType(64)),
                "norm_before_gate": IntegerAttr(int(norm_before_gate), IntegerType(1)),
            },
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)

    def _convert_l2norm(self, node: Node, block: Block) -> None:
        """``auto_deploy.torch_l2norm`` → elementwise MLIR primitives."""
        input_val = self._resolve_operand(node.args[0])
        eps_val = float(node.args[1]) if len(node.args) > 1 else 1e-6
        output_type = _result_type_for_node(node)
        shape = output_type.get_shape()
        f32_type = TensorType(Float32Type(), shape)
        reduced_shape = [s if i < len(shape) - 1 else 1 for i, s in enumerate(shape)]
        f32_reduced_type = TensorType(Float32Type(), reduced_shape)

        cast_in = AdToDtype.build(
            operands=[input_val],
            attributes={"target_dtype": IntegerAttr(6, IntegerType(64))},
            result_types=[f32_type],
        )
        block.add_op(cast_in)

        square = AdMul.build(operands=[cast_in.output, cast_in.output], result_types=[f32_type])
        block.add_op(square)

        sum_sq = AdReduceSum.build(
            operands=[square.output],
            attributes={
                "dim": IntegerAttr(-1, IntegerType(64)),
                "keepdim": IntegerAttr(1, IntegerType(1)),
            },
            result_types=[f32_reduced_type],
        )
        block.add_op(sum_sq)

        eps_t = AdSplat.build(
            attributes={"value": FloatAttr(eps_val, Float64Type())},
            result_types=[f32_reduced_type],
        )
        block.add_op(eps_t)

        sum_eps = AdAdd.build(
            operands=[sum_sq.output, eps_t.output], result_types=[f32_reduced_type]
        )
        block.add_op(sum_eps)

        inv = AdRsqrt.build(operands=[sum_eps.output], result_types=[f32_reduced_type])
        block.add_op(inv)

        normed = AdMul.build(operands=[cast_in.output, inv.output], result_types=[f32_type])
        block.add_op(normed)

        target_dtype = _get_fake_tensor(node)
        dtype_id = _dtype_id_from_torch_dtype(target_dtype.dtype) if target_dtype is not None else 0
        cast_out = AdToDtype.build(
            operands=[normed.output],
            attributes={"target_dtype": IntegerAttr(dtype_id, IntegerType(64))},
            result_types=[output_type],
        )
        block.add_op(cast_out)

        self._value_map[node.name] = cast_out.output
        self._store_meta(node)
        self.metadata[node.name]["_fx_node_name"] = node.name

    def _is_rmsnorm_quant_fp8_linear(self, node: Node) -> bool:
        input_arg, _, _, _, _ = self._extract_fp8_linear_args(node)
        if not isinstance(input_arg, Node):
            return False
        norm_node, post_nodes = _unwrap_post_norm_nodes(input_arg)
        if _has_unsupported_post_norm_nodes(post_nodes):
            return False
        return _is_rmsnorm(norm_node) or _is_gated_rmsnorm(norm_node)

    def _has_relu2_nvfp4_source(self, node: Node) -> bool:
        input_arg, _, _, _, _, _ = _extract_nvfp4_linear_args(node)
        return _extract_relu2_source_for_nvfp4_quant(input_arg) is not None

    def _convert_fp8_quant_linear(self, node: Node, block: Block) -> None:
        """Lower FP8 quant-linear to MLIR quant + prequant linear."""
        input_arg, weight_arg, bias_arg, input_scale_arg, weight_scale_arg = (
            self._extract_fp8_linear_args(node)
        )
        required_args = (input_arg, weight_arg, input_scale_arg, weight_scale_arg)
        if not all(isinstance(arg, Node) and arg.name in self._value_map for arg in required_args):
            self._convert_opaque(node, block)
            return
        if bias_arg is not None and not (
            isinstance(bias_arg, Node) and bias_arg.name in self._value_map
        ):
            self._convert_opaque(node, block)
            return

        input_dtype = _extract_tensor_dtype_from_meta(input_arg)
        if input_dtype == torch.float8_e4m3fn:
            self._convert_opaque(node, block)
            return

        input_val = self._resolve_operand(input_arg)
        input_scale_val = self._resolve_operand(input_scale_arg)
        if not isinstance(input_val.type, TensorType):
            self._convert_opaque(node, block)
            return

        quant_type = TensorType(Float8E4M3FNType(), input_val.type.get_shape())
        quant_op = AdQuantFP8.build(
            operands=[input_val, input_scale_val],
            result_types=[quant_type],
        )
        block.add_op(quant_op)

        operands = [
            quant_op.output,
            self._resolve_operand(weight_arg),
        ]
        args_template = [
            ("__mlir_operand__", 0),
            ("__mlir_operand__", 1),
        ]
        if bias_arg is None:
            args_template.append(None)
        else:
            operands.append(self._resolve_operand(bias_arg))
            args_template.append(("__mlir_operand__", len(operands) - 1))

        operands.append(input_scale_val)
        args_template.append(("__mlir_operand__", len(operands) - 1))
        operands.append(self._resolve_operand(weight_scale_arg))
        args_template.append(("__mlir_operand__", len(operands) - 1))

        result_type = _result_type_for_node(node)
        prequant_op = AdOpaque.build(
            operands=[operands],
            attributes={
                "op_name": StringAttr("trtllm_fp8_prequant_linear"),
                "node_key": StringAttr(node.name),
            },
            result_types=[[result_type]],
        )
        block.add_op(prequant_op)
        self._value_map[node.name] = prequant_op.outputs[0]
        self._store_meta(node)
        self.metadata[node.name]["_original_target"] = (
            torch.ops.auto_deploy.trtllm_fp8_prequant_linear.default
        )
        self.metadata[node.name]["_args_template"] = tuple(args_template)
        self.metadata[node.name]["_kwargs_template"] = {
            "out_dtype": _torch_dtype_name(input_dtype),
        }

    @staticmethod
    def _extract_fp8_linear_args(node: Node) -> tuple[Any, Any, Any, Any, Any]:
        """Extract canonical FP8 quant-linear arguments from FX args/kwargs."""

        def _arg(pos: int, name: str, default=None):
            if len(node.args) > pos:
                return node.args[pos]
            return node.kwargs.get(name, default)

        return (
            _arg(0, "input"),
            _arg(1, "weight_fp8"),
            _arg(2, "bias"),
            _arg(3, "input_scale"),
            _arg(4, "weight_scale"),
        )

    def _convert_nvfp4_relu2_quant_linear(self, node: Node, block: Block) -> None:
        """Lower ReLU2 -> NVFP4 quant-linear to fused quant + prequant linear."""
        input_arg, weight_arg, bias_arg, input_scale_arg, weight_scale_arg, alpha_arg = (
            self._extract_nvfp4_linear_args(node)
        )
        if not isinstance(input_arg, Node):
            self._convert_opaque(node, block)
            return

        relu2_source = _extract_relu2_source_for_nvfp4_quant(input_arg)
        required_args = (relu2_source, weight_arg, input_scale_arg, weight_scale_arg, alpha_arg)
        if not all(isinstance(arg, Node) and arg.name in self._value_map for arg in required_args):
            self._convert_opaque(node, block)
            return
        if bias_arg is not None and not (
            isinstance(bias_arg, Node) and bias_arg.name in self._value_map
        ):
            self._convert_opaque(node, block)
            return

        input_fake = _get_fake_tensor(input_arg)
        if input_fake is None:
            self._convert_opaque(node, block)
            return
        quant_shapes = _nvfp4_quant_output_shapes(tuple(int(dim) for dim in input_fake.shape))
        if quant_shapes is None:
            self._convert_opaque(node, block)
            return
        fp4_shape, sf_size = quant_shapes

        fp4_type = TensorType(IntegerType(8), fp4_shape)
        sf_type = TensorType(IntegerType(8), [sf_size])
        relu2_source_val = self._resolve_operand(relu2_source)
        input_scale_val = self._resolve_operand(input_scale_arg)
        fused_quant_key = f"{node.name}__mlir_nvfp4_relu2_quant"
        fused_quant = AdOpaque.build(
            operands=[[relu2_source_val, input_scale_val]],
            attributes={
                "op_name": StringAttr("trtllm_fused_relu2_quant_nvfp4"),
                "node_key": StringAttr(fused_quant_key),
            },
            result_types=[[fp4_type, sf_type]],
        )
        block.add_op(fused_quant)

        fp4_meta = input_fake.new_empty(fp4_shape, dtype=torch.uint8)
        sf_meta = input_fake.new_empty((sf_size,), dtype=torch.uint8)
        self.metadata[fused_quant_key] = {
            "val": (fp4_meta, sf_meta),
            "_original_target": torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default,
            "_args_template": (
                ("__mlir_operand__", 0),
                ("__mlir_operand__", 1),
            ),
            "_kwargs_template": {},
        }

        operands = [
            fused_quant.outputs[0],
            self._resolve_operand(weight_arg),
            fused_quant.outputs[1],
            self._resolve_operand(weight_scale_arg),
            self._resolve_operand(alpha_arg),
        ]
        args_template = [
            ("__mlir_operand__", 0),
            ("__mlir_operand__", 1),
            ("__mlir_operand__", 2),
            ("__mlir_operand__", 3),
            ("__mlir_operand__", 4),
        ]
        kwargs_template: dict[str, Any] = {
            "out_dtype": _extract_tensor_dtype_from_meta(input_arg) or torch.bfloat16
        }
        if bias_arg is not None:
            operands.append(self._resolve_operand(bias_arg))
            kwargs_template["bias"] = ("__mlir_operand__", len(operands) - 1)

        result_type = _result_type_for_node(node)
        prequant_op = AdOpaque.build(
            operands=[operands],
            attributes={
                "op_name": StringAttr("trtllm_nvfp4_prequant_linear"),
                "node_key": StringAttr(node.name),
            },
            result_types=[[result_type]],
        )
        block.add_op(prequant_op)
        self._value_map[node.name] = prequant_op.outputs[0]
        self._store_meta(node)
        self.metadata[node.name]["_original_target"] = (
            torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default
        )
        self.metadata[node.name]["_args_template"] = tuple(args_template)
        self.metadata[node.name]["_kwargs_template"] = kwargs_template
        self.num_direct_rewrites += 1

    def _run_nvfp4_rmsnorm_quant_rewrites(self, block: Block) -> None:
        """Move RMSNorm/GatedRMSNorm -> NVFP4 quant-linear rewrites into MLIR."""
        processed_users: set[int] = set()
        original_nodes = list(self._gm.graph.nodes)
        node_order = {node: idx for idx, node in enumerate(original_nodes)}

        for node in original_nodes:
            if not _is_supported_nvfp4_linear(node) or id(node) in processed_users:
                continue

            input_arg, _, _, input_scale, _, _ = _extract_nvfp4_linear_args(node)
            if not isinstance(input_arg, Node) or not isinstance(input_scale, Node):
                continue
            if _extract_relu2_source_for_nvfp4_quant(input_arg) is not None:
                continue

            norm_node, post_nodes = _unwrap_post_norm_nodes(input_arg)
            if _has_unsupported_post_norm_nodes(post_nodes):
                continue

            nvfp4_linear_users = _collect_grouped_nvfp4_linear_users(
                norm_node, node, input_scale, processed_users
            )
            if not nvfp4_linear_users:
                continue
            if not self._nvfp4_prequant_operands_available(nvfp4_linear_users):
                continue

            earliest_user = min(
                nvfp4_linear_users, key=lambda user: node_order.get(user, float("inf"))
            )

            num_matches = self._try_rewrite_allreduce_rmsnorm_quant_nvfp4(
                block,
                norm_node,
                input_scale,
                nvfp4_linear_users,
                earliest_user,
                node_order,
                processed_users,
            )
            if num_matches is not None:
                self.num_direct_rewrites += num_matches
                continue

            num_matches = self._try_rewrite_add_rmsnorm_quant_nvfp4(
                block,
                norm_node,
                input_scale,
                nvfp4_linear_users,
                node_order,
                processed_users,
            )
            if num_matches is not None:
                self.num_direct_rewrites += num_matches
                continue

            num_matches = self._try_rewrite_gated_rmsnorm_quant_nvfp4(
                block,
                norm_node,
                input_scale,
                nvfp4_linear_users,
                earliest_user,
                processed_users,
            )
            if num_matches is not None:
                self.num_direct_rewrites += num_matches

    def _try_rewrite_allreduce_rmsnorm_quant_nvfp4(
        self,
        block: Block,
        norm_node: Node,
        input_scale: Node,
        nvfp4_linear_users: list[Node],
        earliest_user: Node,
        node_order: dict[Node, int],
        processed_users: set[int],
    ) -> int | None:
        allreduce_info = _extract_nonquant_allreduce_norm(norm_node)
        if allreduce_info is None:
            return None

        (
            allreduce_node,
            tensor,
            residual,
            norm_weight,
            eps,
            strategy,
            residual_out_node,
        ) = allreduce_info
        needs_norm_output = _has_terminal_users_outside_group(norm_node, nvfp4_linear_users)
        insertion_op = self._op_for_node(earliest_user)
        if needs_norm_output:
            insertion_op = self._earliest_op_for_nodes(list(norm_node.users), block)
        if insertion_op is None:
            return 0
        old_allreduce_op = self._op_for_node(allreduce_node)
        old_norm_op = self._op_for_node(norm_node)
        old_residual_op = self._op_for_node(residual_out_node)

        tensor_val = self._value_for_node(tensor)
        residual_val = self._value_for_node(residual)
        norm_weight_val = self._value_for_node(norm_weight)
        scale_val = self._operand_defined_before_or_clone_get_attr(input_scale, insertion_op, block)
        if any(val is None for val in (tensor_val, residual_val, norm_weight_val, scale_val)):
            return 0

        quant_meta = self._nvfp4_quant_meta(norm_node)
        if quant_meta is None:
            return 0
        fp4_type, sf_type, fp4_meta, sf_meta = quant_meta
        norm_val = self._value_for_node(norm_node)
        if norm_val is None:
            return 0
        norm_type = norm_val.type
        residual_out_val = self._value_for_node(residual_out_node)
        residual_type = (
            residual_out_val.type
            if isinstance(residual_out_node, Node) and residual_out_val is not None
            else norm_type
        )

        operands = [tensor_val, residual_val, norm_weight_val, scale_val]
        args_template = (
            ("__mlir_operand__", 0),
            ("__mlir_operand__", 1),
            ("__mlir_operand__", 2),
            ("__mlir_operand__", 3),
            eps,
            strategy,
        )
        fused_key = f"{earliest_user.name}__mlir_nvfp4_allreduce_rmsnorm_quant"
        if needs_norm_output:
            fused_op = self._insert_direct_opaque_before(
                block,
                insertion_op,
                fused_key,
                "trtllm_fused_allreduce_residual_rmsnorm_out_quant_nvfp4",
                torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm_out_quant_nvfp4.default,
                operands,
                args_template,
                {},
                [norm_type, fp4_type, sf_type, residual_type],
                (self._fake_like(norm_node), fp4_meta, sf_meta, self._fake_like(residual_out_node)),
            )
            self._replace_node_value(norm_node, fused_op.outputs[0])
            fp4_val = fused_op.outputs[1]
            sf_val = fused_op.outputs[2]
            residual_val_new = fused_op.outputs[3]
        else:
            fused_op = self._insert_direct_opaque_before(
                block,
                insertion_op,
                fused_key,
                "trtllm_fused_allreduce_residual_rmsnorm_quant_nvfp4",
                torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm_quant_nvfp4.default,
                operands,
                args_template,
                {},
                [fp4_type, sf_type, residual_type],
                (fp4_meta, sf_meta, self._fake_like(residual_out_node)),
            )
            fp4_val = fused_op.outputs[0]
            sf_val = fused_op.outputs[1]
            residual_val_new = fused_op.outputs[2]

        if isinstance(residual_out_node, Node):
            self._replace_node_value(residual_out_node, residual_val_new)

        num_matches = self._replace_nvfp4_linear_users(
            block, nvfp4_linear_users, fp4_val, sf_val, norm_node, processed_users
        )
        self._erase_op_if_dead(block, old_norm_op)
        self._erase_op_if_dead(block, old_residual_op)
        self._erase_op_if_dead(block, old_allreduce_op)
        return num_matches

    def _try_rewrite_add_rmsnorm_quant_nvfp4(
        self,
        block: Block,
        norm_node: Node,
        input_scale: Node,
        nvfp4_linear_users: list[Node],
        node_order: dict[Node, int],
        processed_users: set[int],
    ) -> int | None:
        add_norm_info = _extract_add_rmsnorm(norm_node)
        if add_norm_info is None:
            return None

        if not self._supports_add_rmsnorm_nvfp4_quant(norm_node):
            return 0

        add_node, pre_norm_cast, add_lhs, add_rhs, norm_weight, eps = add_norm_info
        insertion_candidates = list(add_node.users) + list(norm_node.users)
        if pre_norm_cast is not None:
            insertion_candidates.extend(list(pre_norm_cast.users))
        insertion_op = self._earliest_op_for_nodes(insertion_candidates, block)
        if insertion_op is None:
            return 0

        add_lhs_val = self._value_for_node(add_lhs)
        add_rhs_val = self._value_for_node(add_rhs)
        norm_weight_val = self._operand_defined_before_or_clone_get_attr(
            norm_weight, insertion_op, block
        )
        scale_val = self._operand_defined_before_or_clone_get_attr(input_scale, insertion_op, block)
        if any(val is None for val in (add_lhs_val, add_rhs_val, norm_weight_val, scale_val)):
            return 0

        quant_meta = self._nvfp4_quant_meta(norm_node)
        if quant_meta is None:
            return 0
        fp4_type, sf_type, fp4_meta, sf_meta = quant_meta
        norm_val = self._value_for_node(norm_node)
        add_val = self._value_for_node(add_node)
        if norm_val is None or add_val is None:
            return 0
        norm_type = norm_val.type
        add_type = add_val.type
        needs_norm_output = _has_terminal_users_outside_group(norm_node, nvfp4_linear_users)
        old_norm_op = norm_val.owner
        old_add_op = add_val.owner
        old_cast_op = self._op_for_node(pre_norm_cast)

        operands = [add_lhs_val, add_rhs_val, norm_weight_val, scale_val]
        args_template = (
            ("__mlir_operand__", 0),
            ("__mlir_operand__", 1),
            ("__mlir_operand__", 2),
            ("__mlir_operand__", 3),
            eps,
        )
        fused_key = f"{norm_node.name}__mlir_nvfp4_add_rmsnorm_quant"
        if needs_norm_output:
            fused_op = self._insert_direct_opaque_before(
                block,
                insertion_op,
                fused_key,
                "trtllm_fused_add_rmsnorm_out_quant_nvfp4",
                torch.ops.auto_deploy.trtllm_fused_add_rmsnorm_out_quant_nvfp4.default,
                operands,
                args_template,
                {},
                [norm_type, fp4_type, add_type, sf_type],
                (self._fake_like(norm_node), fp4_meta, self._fake_like(add_node), sf_meta),
            )
            self._replace_node_value(norm_node, fused_op.outputs[0])
            fp4_val = fused_op.outputs[1]
            add_val_new = fused_op.outputs[2]
            sf_val = fused_op.outputs[3]
        else:
            fused_op = self._insert_direct_opaque_before(
                block,
                insertion_op,
                fused_key,
                "trtllm_fused_add_rmsnorm_quant_nvfp4",
                torch.ops.auto_deploy.trtllm_fused_add_rmsnorm_quant_nvfp4.default,
                operands,
                args_template,
                {},
                [fp4_type, add_type, sf_type],
                (fp4_meta, self._fake_like(add_node), sf_meta),
            )
            fp4_val = fused_op.outputs[0]
            add_val_new = fused_op.outputs[1]
            sf_val = fused_op.outputs[2]

        self._replace_node_value(add_node, add_val_new)
        num_matches = self._replace_nvfp4_linear_users(
            block, nvfp4_linear_users, fp4_val, sf_val, norm_node, processed_users
        )
        self._erase_op_if_dead(block, old_norm_op)
        self._erase_op_if_dead(block, old_cast_op)
        self._erase_op_if_dead(block, old_add_op)
        return num_matches

    def _try_rewrite_gated_rmsnorm_quant_nvfp4(
        self,
        block: Block,
        norm_node: Node,
        input_scale: Node,
        nvfp4_linear_users: list[Node],
        earliest_user: Node,
        processed_users: set[int],
    ) -> int | None:
        gated_info = _extract_gated_rmsnorm(norm_node)
        if gated_info is None:
            return None
        if not _all_terminal_users_are_grouped(norm_node, nvfp4_linear_users):
            return 0

        x, weight, gate, eps, group_size = gated_info
        insertion_op = self._op_for_node(earliest_user)
        if insertion_op is None:
            return 0

        x_val = self._value_for_node(x)
        gate_val = self._value_for_node(gate)
        weight_val = self._value_for_node(weight)
        scale_val = self._operand_defined_before_or_clone_get_attr(input_scale, insertion_op, block)
        if any(val is None for val in (x_val, gate_val, weight_val, scale_val)):
            return 0

        quant_meta = self._nvfp4_quant_meta(norm_node)
        if quant_meta is None:
            return 0
        fp4_type, sf_type, fp4_meta, sf_meta = quant_meta
        fused_op = self._insert_direct_opaque_before(
            block,
            insertion_op,
            f"{norm_node.name}__mlir_nvfp4_gated_rmsnorm_quant",
            "trtllm_fused_gated_rmsnorm_quant_nvfp4",
            torch.ops.auto_deploy.trtllm_fused_gated_rmsnorm_quant_nvfp4.default,
            [x_val, gate_val, weight_val, scale_val],
            (
                ("__mlir_operand__", 0),
                ("__mlir_operand__", 1),
                ("__mlir_operand__", 2),
                ("__mlir_operand__", 3),
                eps,
                group_size,
            ),
            {},
            [fp4_type, sf_type],
            (fp4_meta, sf_meta),
        )

        num_matches = self._replace_nvfp4_linear_users(
            block,
            nvfp4_linear_users,
            fused_op.outputs[0],
            fused_op.outputs[1],
            norm_node,
            processed_users,
        )
        self._erase_dead_ops_for_nodes(block, [norm_node])
        return num_matches

    def _replace_nvfp4_linear_users(
        self,
        block: Block,
        nvfp4_linear_users: list[Node],
        fp4_val: SSAValue,
        sf_val: SSAValue,
        source_node: Node,
        processed_users: set[int],
    ) -> int:
        cnt = 0
        for user in nvfp4_linear_users:
            old_val = self._value_for_node(user)
            if old_val is None:
                continue
            old_op = old_val.owner
            input_arg, weight_arg, bias_arg, _, weight_scale_arg, alpha_arg = (
                _extract_nvfp4_linear_args(user)
            )
            weight_val = self._value_for_node(weight_arg)
            weight_scale_val = self._value_for_node(weight_scale_arg)
            alpha_val = self._value_for_node(alpha_arg)
            if any(val is None for val in (weight_val, weight_scale_val, alpha_val)):
                continue

            operands = [fp4_val, weight_val, sf_val, weight_scale_val, alpha_val]
            args_template = [
                ("__mlir_operand__", 0),
                ("__mlir_operand__", 1),
                ("__mlir_operand__", 2),
                ("__mlir_operand__", 3),
                ("__mlir_operand__", 4),
            ]
            kwargs_template: dict[str, Any] = {
                "out_dtype": _nvfp4_linear_out_dtype(user, source_node)
            }
            if bias_arg is not None:
                bias_val = self._value_for_node(bias_arg)
                if bias_val is None:
                    continue
                operands.append(bias_val)
                kwargs_template["bias"] = ("__mlir_operand__", len(operands) - 1)

            prequant_op = self._insert_direct_opaque_before(
                block,
                old_op,
                user.name,
                "trtllm_nvfp4_prequant_linear",
                torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default,
                operands,
                tuple(args_template),
                kwargs_template,
                [old_val.type],
                user.meta.get("val"),
            )
            old_val.replace_by(prequant_op.outputs[0])
            self._value_map[user.name] = prequant_op.outputs[0]
            block.erase_op(old_op, safe_erase=False)

            if isinstance(input_arg, Node):
                _, post_nodes = _unwrap_post_norm_nodes(input_arg)
                self._erase_dead_ops_for_nodes(block, post_nodes)
            processed_users.add(id(user))
            cnt += 1
        return cnt

    def _nvfp4_prequant_operands_available(self, nvfp4_linear_users: list[Node]) -> bool:
        for user in nvfp4_linear_users:
            _, weight_arg, bias_arg, _, weight_scale_arg, alpha_arg = _extract_nvfp4_linear_args(
                user
            )
            required_args = (user, weight_arg, weight_scale_arg, alpha_arg)
            if not all(
                isinstance(arg, Node) and arg.name in self._value_map for arg in required_args
            ):
                return False
            if bias_arg is not None and not (
                isinstance(bias_arg, Node) and bias_arg.name in self._value_map
            ):
                return False
        return True

    def _nvfp4_quant_meta(self, source_node: Node):
        source_fake = _get_fake_tensor(source_node)
        if source_fake is None:
            return None
        try:
            source_shape = tuple(int(dim) for dim in source_fake.shape)
        except TypeError:
            return None
        quant_shapes = _nvfp4_quant_output_shapes(source_shape)
        if quant_shapes is None:
            return None
        fp4_shape, sf_size = quant_shapes
        fp4_type = TensorType(IntegerType(8), fp4_shape)
        sf_type = TensorType(IntegerType(8), [sf_size])
        fp4_meta = source_fake.new_empty(fp4_shape, dtype=torch.uint8)
        sf_meta = source_fake.new_empty((sf_size,), dtype=torch.uint8)
        return fp4_type, sf_type, fp4_meta, sf_meta

    def _insert_direct_opaque_before(
        self,
        block: Block,
        insertion_op,
        node_key: str,
        op_name: str,
        target,
        operands: list[SSAValue],
        args_template,
        kwargs_template: dict[str, Any],
        result_types: list,
        val_meta,
    ) -> AdOpaque:
        op = AdOpaque.build(
            operands=[operands],
            attributes={
                "op_name": StringAttr(op_name),
                "node_key": StringAttr(node_key),
            },
            result_types=[result_types],
        )
        block.insert_op_before(op, insertion_op)
        self.metadata[node_key] = {
            "val": val_meta,
            "_original_target": target,
            "_args_template": args_template,
            "_kwargs_template": kwargs_template,
        }
        return op

    def _operand_defined_before_or_clone_get_attr(
        self, node: Node, insertion_op, block: Block
    ) -> SSAValue | None:
        val = self._value_for_node(node)
        if val is None:
            return None

        order = {op: idx for idx, op in enumerate(block.ops)}
        owner = val.owner
        if owner not in order or insertion_op not in order or order[owner] < order[insertion_op]:
            return val
        if node.op != "get_attr":
            return None

        clone = AdGraphInput.build(
            attributes={"input_name": StringAttr(node.name)},
            result_types=[val.type],
        )
        block.insert_op_before(clone, insertion_op)
        return clone.output

    def _value_for_node(self, node: Node | None) -> SSAValue | None:
        if not isinstance(node, Node):
            return None
        return self._value_map.get(node.name)

    def _op_for_node(self, node: Node | None):
        val = self._value_for_node(node)
        return val.owner if val is not None else None

    def _earliest_op_for_nodes(self, nodes: list[Node], block: Block):
        order = {op: idx for idx, op in enumerate(block.ops)}
        candidates = []
        for node in nodes:
            op = self._op_for_node(node)
            if op in order:
                candidates.append(op)
        if not candidates:
            return None
        return min(candidates, key=lambda op: order[op])

    def _replace_node_value(self, node: Node, new_val: SSAValue) -> None:
        old_val = self._value_for_node(node)
        if old_val is not None:
            old_val.replace_by(new_val)
        self._value_map[node.name] = new_val

    def _erase_dead_ops_for_nodes(self, block: Block, nodes: list[Node | None]) -> None:
        for node in nodes:
            op = self._op_for_node(node)
            self._erase_op_if_dead(block, op)

    @staticmethod
    def _erase_op_if_dead(block: Block, op) -> None:
        if op is None or op.name in {"ad.graph_input", "ad.graph_output"}:
            return
        if op not in set(block.ops):
            return
        if all(not any(True for _ in result.uses) for result in op.results):
            block.erase_op(op, safe_erase=False)

    @staticmethod
    def _fake_like(node: Node | None):
        if not isinstance(node, Node):
            return None
        return node.meta.get("val")

    @staticmethod
    def _supports_add_rmsnorm_nvfp4_quant(node: Node) -> bool:
        hidden_size = _last_dim_from_node_meta(node)
        return hidden_size is not None and 2048 <= hidden_size <= 16384 and hidden_size % 16 == 0

    @staticmethod
    def _extract_nvfp4_linear_args(node: Node) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Extract canonical NVFP4 quant-linear arguments from FX args/kwargs."""
        return _extract_nvfp4_linear_args(node)

    def _convert_to_dtype(self, node: Node, block: Block) -> None:
        """``aten.to.dtype`` → ``ad.to_dtype``."""
        input_val = self._resolve_operand(node.args[0])
        # args[1] is the target dtype — can be torch.dtype or int
        target_dtype = node.args[1]
        if isinstance(target_dtype, int):
            dtype_int = target_dtype
        else:
            # torch.dtype → store a stable identifier (str hash works for round-trip)
            _DTYPE_TO_INT = {
                torch.float16: 5,
                torch.bfloat16: 15,
                torch.float32: 6,
                torch.float64: 7,
            }
            dtype_int = _DTYPE_TO_INT.get(target_dtype, hash(str(target_dtype)) % (2**31))
        dtype_attr = IntegerAttr(dtype_int, IntegerType(64))
        result_type = _result_type_for_node(node)
        op = AdToDtype.build(
            operands=[input_val],
            attributes={"target_dtype": dtype_attr},
            result_types=[result_type],
        )
        block.add_op(op)
        self._value_map[node.name] = op.output
        self._store_meta(node)

    def _convert_getitem(self, node: Node, block: Block) -> None:
        """Handle ``operator.getitem`` nodes that index into tuple results.

        If the container op in MLIR has multiple results, map this getitem to
        the corresponding result. Otherwise, fall through to opaque handling.
        """
        container = node.args[0]
        index = node.args[1]

        if isinstance(container, Node):
            split_info = self._extract_static_last_dim_split(container)
            if split_info is not None and isinstance(index, int):
                parent, sizes = split_info
                if 0 <= index < len(sizes):
                    start = sum(sizes[:index])
                    length = sizes[index]
                    input_val = self._resolve_operand(parent)
                    result_type = _result_type_for_node(node)
                    op = AdSlice.build(
                        operands=[input_val],
                        attributes={
                            "dim": IntegerAttr(-1, IntegerType(64)),
                            "start": IntegerAttr(start, IntegerType(64)),
                            "length": IntegerAttr(length, IntegerType(64)),
                        },
                        result_types=[result_type],
                    )
                    block.add_op(op)
                    self._value_map[node.name] = op.output
                    self._store_meta(node)
                    self.metadata[node.name]["_fx_node_name"] = node.name
                    return

            # Check if the container op has a multi-result mapping
            multi_key = f"{container.name}__result_{index}"
            if multi_key in self._value_map:
                self._value_map[node.name] = self._value_map[multi_key]
                self._store_meta(node)
                return

            # If container has a single result, pass through to opaque
            # (getitem on a single-result op is valid for tensor indexing)
            if container.name in self._value_map:
                # Treat as opaque — it may be tensor indexing
                self._convert_opaque(node, block)
                return

        # Fallback to opaque
        self._convert_opaque(node, block)

    def _extract_static_last_dim_split(self, node: Node) -> Optional[tuple[Node, list[int]]]:
        """Extract ``(parent, sizes)`` for static last-dim splits."""
        if not isinstance(node, Node) or node.op != "call_function":
            return None

        if _is_split_with_sizes(node):
            if len(node.args) < 2:
                return None
            parent = node.args[0]
            sizes = node.args[1]
            dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
            if not isinstance(parent, Node) or dim != -1:
                return None
            if not isinstance(sizes, (list, tuple)) or not all(isinstance(s, int) for s in sizes):
                return None
            sizes = list(sizes)
        else:
            if len(node.args) != 1 or not isinstance(node.args[0], Node):
                return None
            parent = node.args[0]
            indexed_sizes: list[tuple[int, int]] = []
            for user in node.users:
                if not _is_getitem(user):
                    return None
                idx = user.args[1]
                val = user.meta.get("val")
                if not isinstance(idx, int) or not hasattr(val, "shape") or len(val.shape) == 0:
                    return None
                last_dim = val.shape[-1]
                if not isinstance(last_dim, int):
                    return None
                indexed_sizes.append((idx, int(last_dim)))
            if not indexed_sizes:
                return None
            indexed_sizes.sort(key=lambda item: item[0])
            expected_indices = list(range(len(indexed_sizes)))
            if [idx for idx, _ in indexed_sizes] != expected_indices:
                return None
            sizes = [size for _, size in indexed_sizes]

        parent_last_dim = _last_dim_from_node_meta(parent)
        if parent_last_dim is not None and sum(sizes) != parent_last_dim:
            return None
        return parent, sizes

    # ------------------------------------------------------------------
    # Opaque (catch-all)
    # ------------------------------------------------------------------

    def _convert_opaque(self, node: Node, block: Block) -> None:
        """Unmodeled FX op → ``ad.opaque``."""
        # Build MLIR operands from Node args, tracking arg structure for round-trip.
        # _args_template stores the original args structure with Node positions replaced
        # by integer indices into the MLIR operands list.
        operands = []
        operand_idx = [0]  # mutable counter

        def _build_template(arg):
            if isinstance(arg, Node) and arg.name in self._value_map:
                operands.append(self._value_map[arg.name])
                idx = operand_idx[0]
                operand_idx[0] += 1
                return ("__mlir_operand__", idx)
            elif isinstance(arg, Node):
                return ("__fx_node_ref__", arg.name)
            elif isinstance(arg, (tuple, list)):
                return type(arg)(_build_template(a) for a in arg)
            else:
                return arg

        args_template = tuple(_build_template(a) for a in node.args)
        kwargs_template = {k: _build_template(v) for k, v in node.kwargs.items()}

        # Determine result types
        val = node.meta.get("val")
        result_types: List[TensorType] = []
        if isinstance(val, torch.Tensor):
            result_types = [tensor_type_from_meta(val)]
        elif isinstance(val, (tuple, list)):
            for v in val:
                if isinstance(v, torch.Tensor):
                    result_types.append(tensor_type_from_meta(v))
        # If still empty but node has users, create a dummy result so downstream
        # nodes can reference it (prevents "value not found" errors).
        if not result_types and node.users:
            from xdsl.dialects.builtin import Float32Type

            result_types = [TensorType(Float32Type(), [-1])]

        # Build a human-readable op name from the FX target
        op_name = (
            str(node.target)
            if not callable(node.target)
            else getattr(node.target, "__name__", str(node.target))
        )

        op = AdOpaque.build(
            operands=[operands],
            attributes={
                "op_name": StringAttr(op_name),
                "node_key": StringAttr(node.name),
            },
            result_types=[result_types],
        )
        block.add_op(op)

        # Map results
        if len(op.outputs) == 1:
            self._value_map[node.name] = op.outputs[0]
        elif len(op.outputs) > 1:
            for i, res in enumerate(op.outputs):
                self._value_map[f"{node.name}__result_{i}"] = res
            # Also map the base name to first result for simple consumers
            self._value_map[node.name] = op.outputs[0]

        self._store_meta(node)
        # Store original target and args/kwargs templates for round-trip reconstruction
        # (must be after _store_meta which initializes the metadata dict)
        self.metadata[node.name]["_original_target"] = node.target
        self.metadata[node.name]["_args_template"] = args_template
        self.metadata[node.name]["_kwargs_template"] = kwargs_template
        if node.op != "call_function":
            self.metadata[node.name]["_fx_op"] = node.op

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_operand(self, arg, block: Block = None, result_type=None) -> SSAValue:
        """Look up the MLIR SSAValue for a given FX node or constant.

        For scalar constants (int, float), emits an ``AdSplat`` op to the given
        block and returns its result. This enables ``aten.add(tensor, 1e-5)``
        to lower to ``ad.splat(1e-5) → ad.add(tensor, splat)`` instead of
        falling through to ``ad.opaque``.
        """
        if isinstance(arg, Node):
            val = self._value_map.get(arg.name)
            if val is None:
                raise ValueError(
                    f"MLIR value not found for FX node '{arg.name}'. "
                    "This may indicate an unsupported node type or ordering issue."
                )
            return val
        if isinstance(arg, (int, float)) and block is not None and result_type is not None:
            # Scalar constant → emit AdSplat
            scalar_val = float(arg)
            splat_type = result_type  # broadcast to match the result shape
            splat_op = AdSplat.build(
                attributes={"value": FloatAttr(scalar_val, Float64Type())},
                result_types=[splat_type],
            )
            block.add_op(splat_op)
            return splat_op.output
        raise TypeError(f"Cannot resolve non-Node operand to SSAValue: {type(arg)}")

    def _store_meta(self, node: Node) -> None:
        """Store FX node metadata in the side-table for round-trip."""
        self.metadata[node.name] = dict(node.meta)

    @staticmethod
    def _flatten_args(args) -> list:
        """Flatten potentially nested args (tuples/lists) into a flat list."""
        if isinstance(args, (tuple, list)):
            flat = []
            for a in args:
                flat.extend(FXToMLIRConverter._flatten_args(a))
            return flat
        return [args]

    # Keep old name as alias for backward compat within the class
    _flatten_output_args = _flatten_args
