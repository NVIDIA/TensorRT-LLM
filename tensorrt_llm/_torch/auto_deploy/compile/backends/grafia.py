# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoDeploy compile backend that lowers canonical FX directly to Grafia CTM."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten, tree_unflatten

from ...utils.node_utils import is_op
from ..compiler import CompileBackendRegistry, CompilerBackend

_SUPPORTED_RMSNORM_ROWS = 107
_SUPPORTED_RMSNORM_HIDDEN = 2880
_RMSNORM_OP_KIND = "grafia.fast_low_latency_rms_norm"


class GrafiaCompileError(RuntimeError):
    """Base error for ``compile_backend='grafia'``."""


class GrafiaUnsupportedError(GrafiaCompileError):
    """Raised when the canonical FX graph contains unsupported work."""


def _ensure_auto_deploy_ops_registered() -> None:
    try:
        importlib.import_module(
            "tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm"
        )
    except Exception as exc:
        raise GrafiaCompileError(
            "compile_backend='grafia' requires AutoDeploy normalization custom "
            "ops to be importable so canonical torch_rmsnorm can be identified"
        ) from exc


def _is_torch_rmsnorm(node: Node) -> bool:
    return is_op(node, torch.ops.auto_deploy.torch_rmsnorm)


def _is_grafia_rmsnorm(node: Node) -> bool:
    return is_op(node, torch.ops.auto_deploy.grafia_rms_norm)


def _validate_supported_fx(gm: GraphModule) -> None:
    _ensure_auto_deploy_ops_registered()
    unsupported: list[str] = []
    has_rmsnorm = False
    for node in gm.graph.nodes:
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        if _is_grafia_rmsnorm(node):
            raise GrafiaUnsupportedError(
                "compile_backend='grafia' consumes canonical "
                "auto_deploy::torch_rmsnorm. The graph already contains "
                "auto_deploy::grafia_rms_norm, which is the old per-op custom "
                "op path and is not used by this backend."
            )
        if node.op == "call_function" and _is_torch_rmsnorm(node):
            has_rmsnorm = True
            continue
        unsupported.append(f"{node.name}: op={node.op}, target={node.target}")

    if unsupported:
        raise GrafiaUnsupportedError(
            "compile_backend='grafia' has no fallback path. Unsupported FX "
            f"node(s): {unsupported}. Initial support is limited to canonical "
            "auto_deploy::torch_rmsnorm.default lowered to "
            f"{_RMSNORM_OP_KIND}."
        )
    if not has_rmsnorm:
        raise GrafiaUnsupportedError(
            "compile_backend='grafia' found no canonical "
            "auto_deploy::torch_rmsnorm.default node to lower."
        )


def _import_ctm_spec_deps():
    try:
        spec_mod = importlib.import_module("backends.ctm.spec")
        types_mod = importlib.import_module("graph.types")
    except ImportError as exc:
        raise GrafiaCompileError(
            "compile_backend='grafia' requires CTM spec modules to be "
            "importable. Add $GRAFIA_ARM/thin_ir to PYTHONPATH."
        ) from exc
    return spec_mod, types_mod


def _import_grafia_runtime_deps():
    try:
        importlib.import_module("grafia_runtime")
        importlib.import_module("backends.ctm.factories.rmsnorm_rts")
        ctm = importlib.import_module("backends.ctm")
        factory_mod = importlib.import_module("backends.ctm.factories.rmsnorm_rts")
    except ImportError as exc:
        raise GrafiaCompileError(
            "compile_backend='grafia' requires Grafia runtime and CTM/ThinIR "
            "modules to be importable. Set GRAFIA_ARM, add "
            "$GRAFIA_ARM/grafia/python and $GRAFIA_ARM/thin_ir to PYTHONPATH, "
            "and set LD_LIBRARY_PATH for the Grafia CUDA toolkit."
        ) from exc
    return ctm, factory_mod


def _require_cuda_device(device: torch.device | str) -> str:
    if not torch.cuda.is_available():
        raise GrafiaCompileError("compile_backend='grafia' requires CUDA to be available")
    device = torch.device(device)
    if device.type != "cuda":
        raise GrafiaCompileError(
            f"compile_backend='grafia' requires a CUDA compile device, got {device}"
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    major, minor = torch.cuda.get_device_capability(index)
    if major < 10:
        raise GrafiaCompileError(
            "compile_backend='grafia' RMSNorm MVP requires a Blackwell-class "
            f"GPU for the sm100 cubin; got sm{major}{minor} on cuda:{index}"
        )
    return f"cuda:{index}"


def _torch_dtype_to_ctm(dtype: torch.dtype, types_mod):
    mapping = {
        torch.float32: types_mod.DType.FP32,
        torch.float16: types_mod.DType.FP16,
        torch.bfloat16: types_mod.DType.BF16,
        torch.int64: types_mod.DType.INT64,
        torch.int32: types_mod.DType.INT32,
        torch.int8: types_mod.DType.INT8,
        torch.uint8: types_mod.DType.UINT8,
        torch.bool: types_mod.DType.BOOL,
    }
    try:
        return mapping[dtype]
    except KeyError:
        raise GrafiaUnsupportedError(
            f"compile_backend='grafia' does not support dtype {dtype}"
        ) from None


def _ctm_dtype_to_torch(dtype) -> torch.dtype:
    name = getattr(dtype, "name", None)
    mapping = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
        "INT64": torch.int64,
        "INT32": torch.int32,
        "INT8": torch.int8,
        "UINT8": torch.uint8,
        "BOOL": torch.bool,
    }
    try:
        return mapping[name]
    except KeyError:
        raise GrafiaUnsupportedError(
            f"compile_backend='grafia' cannot map CTM dtype {dtype} to torch"
        ) from None


def _node_meta_val(node: Node) -> Any:
    val = node.meta.get("val")
    if val is None:
        raise GrafiaUnsupportedError(
            f"FX node {node.name!r} is missing meta['val']; "
            "compile_backend='grafia' requires canonical AutoDeploy FX with "
            "static tensor metadata."
        )
    return val


def _shape_from_meta(node: Node) -> tuple[int, ...]:
    val = _node_meta_val(node)
    try:
        return tuple(int(d) for d in val.shape)
    except Exception as exc:
        raise GrafiaUnsupportedError(
            f"FX node {node.name!r} has unsupported or symbolic shape metadata: "
            f"{getattr(val, 'shape', None)!r}"
        ) from exc


def _dtype_from_meta(node: Node) -> torch.dtype:
    val = _node_meta_val(node)
    dtype = getattr(val, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise GrafiaUnsupportedError(
            f"FX node {node.name!r} has invalid dtype metadata: {dtype!r}"
        )
    return dtype


def _is_contiguous_meta(node: Node) -> bool:
    val = _node_meta_val(node)
    is_contiguous = getattr(val, "is_contiguous", None)
    if callable(is_contiguous):
        return bool(is_contiguous())
    shape = _shape_from_meta(node)
    stride_fn = getattr(val, "stride", None)
    if not callable(stride_fn):
        return False
    expected = 1
    for dim in range(len(shape) - 1, -1, -1):
        if int(stride_fn(dim)) != expected:
            return False
        expected *= shape[dim]
    return True


def _get_attr(gm: GraphModule, target: str) -> Any:
    obj: Any = gm
    for atom in target.split("."):
        obj = getattr(obj, atom)
    return obj


def _infer_compile_device(gm: GraphModule, compiler_kwargs: dict[str, Any]) -> str:
    for arg in compiler_kwargs.get("args", ()):
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            return _require_cuda_device(arg.device)
    for tensor in list(gm.parameters()) + list(gm.buffers()):
        if tensor.is_cuda:
            return _require_cuda_device(tensor.device)
    return _require_cuda_device(torch.device("cuda"))


def _check_rmsnorm_contract(x: Node, weight: Node, out: Node) -> None:
    x_shape = _shape_from_meta(x)
    weight_shape = _shape_from_meta(weight)
    out_shape = _shape_from_meta(out)
    x_dtype = _dtype_from_meta(x)
    weight_dtype = _dtype_from_meta(weight)
    out_dtype = _dtype_from_meta(out)

    if x_dtype is not torch.bfloat16 or weight_dtype is not torch.bfloat16:
        raise GrafiaUnsupportedError(
            "compile_backend='grafia' RMSNorm supports only BF16 input and "
            f"BF16 weight; got input={x_dtype}, weight={weight_dtype}"
        )
    if out_dtype is not torch.bfloat16:
        raise GrafiaUnsupportedError(
            f"compile_backend='grafia' RMSNorm output must be BF16, got {out_dtype}"
        )
    if len(x_shape) < 1:
        raise GrafiaUnsupportedError("compile_backend='grafia' RMSNorm input must be ranked")
    if x_shape[-1] != _SUPPORTED_RMSNORM_HIDDEN:
        raise GrafiaUnsupportedError(
            "compile_backend='grafia' RMSNorm supports hidden size "
            f"{_SUPPORTED_RMSNORM_HIDDEN}, got {x_shape[-1]}"
        )
    rows = 1
    for dim in x_shape[:-1]:
        rows *= dim
    if rows != _SUPPORTED_RMSNORM_ROWS:
        raise GrafiaUnsupportedError(
            "compile_backend='grafia' RMSNorm supports flattened rows "
            f"{_SUPPORTED_RMSNORM_ROWS}, got {rows}"
        )
    if weight_shape != (_SUPPORTED_RMSNORM_HIDDEN,):
        raise GrafiaUnsupportedError(
            "compile_backend='grafia' RMSNorm weight shape must be "
            f"({_SUPPORTED_RMSNORM_HIDDEN},), got {weight_shape}"
        )
    if out_shape != x_shape:
        raise GrafiaUnsupportedError(
            f"compile_backend='grafia' RMSNorm output shape {out_shape} "
            f"must match input shape {x_shape}"
        )
    if not _is_contiguous_meta(x) or not _is_contiguous_meta(weight):
        raise GrafiaUnsupportedError(
            "compile_backend='grafia' RMSNorm requires contiguous input and weight metadata"
        )


def _validate_rmsnorm_contracts_fx(gm: GraphModule) -> None:
    for node in gm.graph.nodes:
        if not _is_torch_rmsnorm(node):
            continue
        if len(node.args) != 3:
            raise GrafiaUnsupportedError(
                f"torch_rmsnorm node {node.name!r} must have args=(x, weight, eps)"
            )
        x_node, weight_node, eps = node.args
        if not isinstance(x_node, Node) or not isinstance(weight_node, Node):
            raise GrafiaUnsupportedError(
                f"torch_rmsnorm node {node.name!r} requires tensor node inputs"
            )
        if not isinstance(eps, (float, int)):
            raise GrafiaUnsupportedError(
                f"torch_rmsnorm node {node.name!r} requires numeric eps, got {eps!r}"
            )
        _check_rmsnorm_contract(x_node, weight_node, node)


def _validate_outputs_are_lowered_kernel_fx(gm: GraphModule) -> None:
    for node in gm.graph.nodes:
        if node.op != "output":
            continue
        flat_outputs, _output_tree_spec = tree_flatten(node.args[0])
        for leaf in flat_outputs:
            if not isinstance(leaf, Node) or not _is_torch_rmsnorm(leaf):
                raise GrafiaUnsupportedError(
                    "compile_backend='grafia' requires every graph output to be "
                    "produced by a lowered Grafia kernel op. Outputs that alias "
                    "placeholders, get_attr constants, or other unsupported nodes "
                    "are rejected."
                )
        return
    raise GrafiaUnsupportedError("FX graph has no output node")


class _FxToCTMSpec:
    def __init__(self, gm: GraphModule, spec_mod, types_mod) -> None:
        self.gm = gm
        self.spec_mod = spec_mod
        self.types_mod = types_mod
        self.env: dict[Node, Any] = {}
        self.ops: list[Any] = []
        self.inputs: list[Any] = []
        self.constants: dict[Any, torch.Tensor] = {}

    def _tensor_for_node(self, node: Node, *, producer_id: int | None = None):
        dtype = _torch_dtype_to_ctm(_dtype_from_meta(node), self.types_mod)
        tensor = self.spec_mod.CTMTensorSpec(
            spec=self.types_mod.TensorSpec.contiguous(
                shape=_shape_from_meta(node),
                dtype=dtype,
                storage_id=-1,
            ),
            name=node.name,
            producer_id=producer_id,
            producer_idx=0,
        )
        self.env[node] = tensor
        return tensor

    def _handle_placeholder(self, node: Node) -> None:
        tensor = self._tensor_for_node(node)
        tensor.name = str(node.target)
        self.inputs.append(tensor)

    def _handle_get_attr(self, node: Node) -> None:
        value = _get_attr(self.gm, str(node.target))
        if isinstance(value, torch.nn.Parameter):
            value = value.detach()
        if not isinstance(value, torch.Tensor):
            raise GrafiaUnsupportedError(
                f"compile_backend='grafia' get_attr node {node.name!r} "
                f"must resolve to a tensor, got {type(value).__name__}"
            )
        tensor = self._tensor_for_node(node)
        self.constants[tensor] = value

    def _handle_rmsnorm(self, node: Node) -> None:
        x_node, weight_node, eps = node.args

        op_id = len(self.ops)
        out = self._tensor_for_node(node, producer_id=op_id)
        self.ops.append(
            self.spec_mod.CTMOpSpec(
                op_kind=_RMSNORM_OP_KIND,
                id=op_id,
                inputs=[self.env[x_node], self.env[weight_node]],
                outputs=[out],
                attrs={"hidden_size": _SUPPORTED_RMSNORM_HIDDEN, "eps": float(eps)},
            )
        )

    def lower(self):
        output_arg = None
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self._handle_placeholder(node)
            elif node.op == "get_attr":
                self._handle_get_attr(node)
            elif node.op == "call_function" and _is_torch_rmsnorm(node):
                self._handle_rmsnorm(node)
            elif node.op == "output":
                output_arg = node.args[0]
            elif node.op != "output":
                raise AssertionError(f"unexpected node after validation: {node}")

        if output_arg is None:
            raise GrafiaUnsupportedError("FX graph has no output node")
        flat_outputs, output_tree_spec = tree_flatten(output_arg)
        outputs = []
        for leaf in flat_outputs:
            if not isinstance(leaf, Node):
                raise GrafiaUnsupportedError(
                    "compile_backend='grafia' supports tensor-node outputs only"
                )
            out = self.env[leaf]
            if out.producer_id is None:
                raise GrafiaUnsupportedError(
                    "compile_backend='grafia' requires every graph output to be "
                    "produced by a lowered Grafia kernel op."
                )
            outputs.append(out)

        return (
            self.spec_mod.CTMGraphSpec(
                name=getattr(self.gm, "__class__", type(self.gm)).__name__,
                ops=self.ops,
                inputs=self.inputs,
                outputs=outputs,
                constant_data=self.constants,
            ),
            output_tree_spec,
        )


class GrafiaCompiledGraph(nn.Module):
    def __init__(
        self,
        gm: GraphModule,
        ctm_backend: Any,
        artifact: Any,
        input_tensors: list[Any],
        output_tree_spec: Any,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "gm", gm)
        self.ctm_backend = ctm_backend
        self.artifact = artifact
        self.input_tensors = input_tensors
        self.input_names = [t.name for t in input_tensors]
        self.output_tree_spec = output_tree_spec
        self.lowered_op_kinds = [op.op_kind for op in artifact.compiled_nodes.keys()]
        self._signature = inspect.signature(gm.forward)

    def _flatten_inputs(self, *args, **kwargs) -> tuple[Any, ...]:
        if kwargs:
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                return tuple(bound.arguments[name] for name in self.input_names)
            except KeyError as exc:
                raise GrafiaCompileError(
                    f"missing runtime input {exc.args[0]!r} for compile_backend='grafia'"
                ) from exc
        if len(args) != len(self.input_tensors):
            raise GrafiaCompileError(
                f"compile_backend='grafia' expected {len(self.input_tensors)} "
                f"runtime input(s) {self.input_names}, got {len(args)}"
            )
        return tuple(args)

    def _validate_runtime_inputs(self, args: tuple[Any, ...]) -> None:
        expected_device = None
        for idx, (arg, spec) in enumerate(zip(args, self.input_tensors, strict=True)):
            if not isinstance(arg, torch.Tensor):
                raise GrafiaCompileError(
                    f"compile_backend='grafia' input {idx} must be a tensor, "
                    f"got {type(arg).__name__}"
                )
            if not arg.is_cuda:
                raise GrafiaCompileError(
                    f"compile_backend='grafia' input {idx} must be CUDA, got {arg.device}"
                )
            if expected_device is None:
                expected_device = arg.device
            elif arg.device != expected_device:
                raise GrafiaCompileError(
                    "compile_backend='grafia' requires all runtime inputs on the "
                    f"same CUDA device; got {expected_device} and {arg.device}"
                )
            expected_shape = tuple(int(d) for d in spec.spec.shape)
            if tuple(arg.shape) != expected_shape:
                raise GrafiaCompileError(
                    f"compile_backend='grafia' input {idx} shape mismatch: "
                    f"expected {expected_shape}, got {tuple(arg.shape)}"
                )
            expected_dtype = _ctm_dtype_to_torch(spec.spec.dtype)
            if arg.dtype is not expected_dtype:
                raise GrafiaCompileError(
                    f"compile_backend='grafia' input {idx} dtype mismatch: "
                    f"expected {expected_dtype}, got {arg.dtype}"
                )
            if not arg.is_contiguous():
                raise GrafiaCompileError(
                    f"compile_backend='grafia' input {idx} must be contiguous"
                )

    def forward(self, *args, **kwargs):
        flat_args = self._flatten_inputs(*args, **kwargs)
        self._validate_runtime_inputs(flat_args)
        outputs = self.ctm_backend.launch(self.artifact, *flat_args)
        return tree_unflatten(list(outputs), self.output_tree_spec)


@CompileBackendRegistry.register("grafia")
class GrafiaCompiler(CompilerBackend):
    """Strict AutoDeploy-native Grafia backend."""

    def __init__(self, model: nn.Module, **compiler_kwargs):
        super().__init__(model, **compiler_kwargs)
        self.compiler_kwargs = dict(compiler_kwargs)

    def compile(self) -> nn.Module:
        if not isinstance(self.model, GraphModule):
            raise GrafiaUnsupportedError(
                "compile_backend='grafia' requires an AutoDeploy FX GraphModule; "
                f"got {type(self.model).__name__}"
            )

        _validate_supported_fx(self.model)
        _validate_rmsnorm_contracts_fx(self.model)
        _validate_outputs_are_lowered_kernel_fx(self.model)
        spec_mod, types_mod = _import_ctm_spec_deps()
        spec, output_tree_spec = _FxToCTMSpec(self.model, spec_mod, types_mod).lower()

        ctm, factory_mod = _import_grafia_runtime_deps()
        cubin_path = factory_mod._default_cubin_path()
        if cubin_path is None:
            raise GrafiaCompileError(
                "compile_backend='grafia' could not find the rmsnorm_rts cubin. "
                "Set GRAFIA_RMSNORM_RTS_CUBIN or DKG_HOME."
            )

        device = _infer_compile_device(self.model, self.compiler_kwargs)
        backend = ctm.CTMBackend(device=device)
        artifact = backend.compile_spec(spec)
        compiled = GrafiaCompiledGraph(
            self.model,
            backend,
            artifact,
            input_tensors=spec.inputs,
            output_tree_spec=output_tree_spec,
        )
        compiled.eval()
        return compiled


__all__ = [
    "GrafiaCompileError",
    "GrafiaCompiler",
    "GrafiaCompiledGraph",
    "GrafiaUnsupportedError",
]
