# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic Grafia backend adapter implementation over CTM graph specs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
from torch.fx import Node
from torch.utils._pytree import tree_flatten

from ...lowering import BoundaryValue, ProgramData, RegionSpec, ValueType
from .constants import BACKEND_NAME
from .deps import _import_ctm_spec_deps, _import_grafia_runtime_deps, _infer_compile_device
from .errors import GrafiaCompileError, GrafiaUnsupportedError
from .metadata import _dtype_from_meta, _shape_from_meta, _torch_dtype_to_ctm
from .runtime import GrafiaLoweredArtifact


class GrafiaBackendContext:
    """Compiler-owned cache for CTM deps, device backends, and artifacts."""

    def __init__(
        self,
        program: ProgramData,
        compiler_kwargs: dict[str, Any] | None = None,
        *,
        compile_artifacts: bool = True,
        spec_deps: tuple[Any, Any] | None = None,
        runtime_deps: Any | None = None,
    ) -> None:
        self.program = program
        self.compiler_kwargs = dict(compiler_kwargs or {})
        self.compile_artifacts = compile_artifacts
        self._spec_deps = spec_deps
        self._runtime_deps = runtime_deps
        self._compile_device: str | None = None
        self._backend_by_device: dict[str, Any] = {}
        self._artifact_cache: dict[Any, tuple[Any, Any]] = {}

    def spec_deps(self) -> tuple[Any, Any]:
        if self._spec_deps is None:
            self._spec_deps = _import_ctm_spec_deps()
        return self._spec_deps

    def runtime_deps(self) -> Any:
        if self._runtime_deps is None:
            self._runtime_deps = _import_grafia_runtime_deps()
        return self._runtime_deps

    def compile_device(self) -> str:
        if self._compile_device is None:
            self._compile_device = _infer_compile_device(
                self.program.graph_module, self.compiler_kwargs
            )
        return self._compile_device

    def ctm_backend(self) -> Any:
        device = self.compile_device()
        backend = self._backend_by_device.get(device)
        if backend is None:
            ctm = self.runtime_deps()
            backend = ctm.CTMBackend(device=device)
            self._backend_by_device[device] = backend
        return backend

    def compile_spec(
        self,
        spec: Any,
        spec_cache_key: Any,
        *,
        resource_cache_key: Sequence[Any | Callable[[], Any]] = (),
        configure_backend: Callable[[Any], None] | None = None,
    ) -> tuple[Any | None, Any | None]:
        if not self.compile_artifacts:
            return None, None
        resource_key = tuple(item() if callable(item) else item for item in resource_cache_key)
        key = (self.compile_device(), resource_key, spec_cache_key)
        cached = self._artifact_cache.get(key)
        if cached is not None:
            return cached
        backend = self.ctm_backend()
        if configure_backend is not None:
            configure_backend(backend)
        artifact = backend.compile_spec(spec)
        cached = (backend, artifact)
        self._artifact_cache[key] = cached
        return cached


class GrafiaCTMBackendAdapter:
    """Backend adapter that turns semantic Grafia operations into CTM specs."""

    backend_name = BACKEND_NAME

    def __init__(self, context: GrafiaBackendContext) -> None:
        self.context = context
        self.spec_mod, self.types_mod = context.spec_deps()
        self.program: ProgramData | None = None
        self.region: RegionSpec | None = None
        self.ops: list[Any] = []
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []
        self.constants: dict[Any, torch.Tensor] = {}
        self.constant_names: list[str] = []
        self.op_kinds: list[str] = []
        self._op_cache_records: list[Any] = []
        self._compile_resource_ids: set[Any] = set()
        self._compile_resource_cache_key: list[Any | Callable[[], Any]] = []
        self._compile_resource_configurers: list[Callable[[Any], None]] = []
        self._output_tree_spec: Any | None = None

    @classmethod
    def from_program(
        cls,
        program: ProgramData,
        region: RegionSpec,
        compiler_kwargs: dict[str, Any] | None = None,
        *,
        compile_artifacts: bool = True,
        spec_deps: tuple[Any, Any] | None = None,
        runtime_deps: Any | None = None,
    ) -> "GrafiaCTMBackendAdapter":
        context = GrafiaBackendContext(
            program,
            compiler_kwargs,
            compile_artifacts=compile_artifacts,
            spec_deps=spec_deps,
            runtime_deps=runtime_deps,
        )
        adapter = cls(context)
        adapter.begin_region(program, region)
        return adapter

    def begin_region(self, program: ProgramData, region: RegionSpec) -> None:
        self.program = program
        self.region = region
        self.ops = []
        self.inputs = []
        self.outputs = []
        self.constants = {}
        self.constant_names = []
        self.op_kinds = []
        self._op_cache_records = []
        self._compile_resource_ids = set()
        self._compile_resource_cache_key = []
        self._compile_resource_configurers = []
        self._output_tree_spec = None

    def input(self, boundary: BoundaryValue) -> Any:
        tensor = self._tensor_for_node(boundary.node, name=self._input_name(boundary.node))
        self.inputs.append(tensor)
        return tensor

    def constant(self, boundary: BoundaryValue, value: object) -> Any:
        if isinstance(value, torch.nn.Parameter):
            value = value.detach()
        if not isinstance(value, torch.Tensor):
            raise GrafiaUnsupportedError(
                f"compile_backend='grafia' get_attr node {boundary.node_name!r} "
                f"must resolve to a tensor, got {type(value).__name__}"
            )
        tensor = self._tensor_for_node(boundary.node, name=boundary.node_name)
        self.constants[tensor] = value
        self.constant_names.append(boundary.node_name)
        return tensor

    def output(self, values: Sequence[Any], outputs: Sequence[BoundaryValue]) -> None:
        if not values:
            raise GrafiaUnsupportedError(
                f"{self._region_id}: compile_backend='grafia' requires a region output"
            )
        for value, boundary in zip(values, outputs, strict=True):
            if getattr(value, "producer_id", None) is None:
                raise GrafiaUnsupportedError(
                    f"{self._region_id}: region output {boundary.node_name!r} is not "
                    "produced by a lowered Grafia op"
                )
        self.outputs = list(values)
        self._output_tree_spec = self._infer_output_tree_spec(outputs)

    def emit(
        self,
        op_name: str,
        operands: Sequence[Any],
        attrs: Mapping[str, object],
        result_types: Sequence[ValueType],
        *,
        loc: Any | None = None,
    ) -> Any:
        raise GrafiaUnsupportedError(
            f"compile_backend='grafia' does not support generic emit({op_name!r}); "
            "use a typed semantic adapter method"
        )

    def emit_rms_norm(
        self,
        x: Any,
        weight: Any,
        *,
        eps: float,
        result_meta: ValueType,
        loc: Any | None = None,
    ) -> Any:
        from .ops.rmsnorm import emit

        return emit(self, x, weight, eps=eps, result_meta=result_meta, loc=loc)

    def register_compile_resource(
        self,
        resource_id: Any,
        *,
        cache_key: Any | Callable[[], Any],
        configure_backend: Callable[[Any], None],
    ) -> None:
        if resource_id in self._compile_resource_ids:
            return
        self._compile_resource_ids.add(resource_id)
        self._compile_resource_cache_key.append(cache_key)
        self._compile_resource_configurers.append(configure_backend)

    def finalize(self) -> GrafiaLoweredArtifact:
        if self.program is None or self.region is None:
            raise GrafiaCompileError("GrafiaCTMBackendAdapter.finalize called before begin_region")
        if self._output_tree_spec is None:
            raise GrafiaCompileError(
                f"{self._region_id}: GrafiaCTMBackendAdapter.finalize called before output"
            )
        spec = self.spec_mod.CTMGraphSpec(
            name=self._spec_name(),
            ops=self.ops,
            inputs=self.inputs,
            outputs=self.outputs,
            constant_data=self.constants,
        )
        spec_cache_key = self._spec_cache_key()
        backend, artifact = self.context.compile_spec(
            spec,
            spec_cache_key,
            resource_cache_key=tuple(self._compile_resource_cache_key),
            configure_backend=self._configure_backend,
        )
        return GrafiaLoweredArtifact(
            program=self.program,
            region=self.region,
            spec=spec,
            backend=backend,
            artifact=artifact,
            input_tensors=tuple(self.inputs),
            output_tree_spec=self._output_tree_spec,
            op_kinds=tuple(self.op_kinds),
            constant_names=tuple(self.constant_names),
            spec_cache_key=spec_cache_key,
        )

    def _configure_backend(self, backend: Any) -> None:
        for configure_backend in self._compile_resource_configurers:
            configure_backend(backend)

    @property
    def _region_id(self) -> str:
        return self.region.region_id if self.region is not None else "<uninitialized>"

    def _input_name(self, node: Node) -> str:
        return str(node.target) if node.op == "placeholder" else node.name

    def _tensor_for_node(
        self, node: Node, *, name: str | None = None, producer_id: int | None = None
    ) -> Any:
        return self._tensor_spec(
            name=name or node.name,
            shape=_shape_from_meta(node),
            dtype=_dtype_from_meta(node),
            producer_id=producer_id,
        )

    def _tensor_spec(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        producer_id: int | None = None,
    ) -> Any:
        ctm_dtype = _torch_dtype_to_ctm(dtype, self.types_mod)
        return self.spec_mod.CTMTensorSpec(
            spec=self.types_mod.TensorSpec.contiguous(
                shape=tuple(shape),
                dtype=ctm_dtype,
                storage_id=-1,
            ),
            name=name,
            producer_id=producer_id,
            producer_idx=0,
        )

    def _shape_dtype_from_result_type(
        self, result_type: ValueType, loc: Any | None
    ) -> tuple[tuple[int, ...], torch.dtype]:
        if result_type.shape is None or result_type.dtype is None:
            raise GrafiaUnsupportedError(
                f"{self._region_id}: result {loc!r} is missing static shape/dtype metadata"
            )
        if not isinstance(result_type.dtype, torch.dtype):
            raise GrafiaUnsupportedError(
                f"{self._region_id}: result {loc!r} has invalid dtype {result_type.dtype!r}"
            )
        try:
            shape = tuple(int(dim) for dim in result_type.shape)
        except Exception as exc:
            raise GrafiaUnsupportedError(
                f"{self._region_id}: result {loc!r} has unsupported shape {result_type.shape!r}"
            ) from exc
        return shape, result_type.dtype

    def _infer_output_tree_spec(self, outputs: Sequence[BoundaryValue]) -> Any:
        output_nodes = tuple(output.node for output in outputs)
        if len(output_nodes) == 1:
            _flat, tree_spec = tree_flatten(output_nodes[0])
            return tree_spec
        _flat, tree_spec = tree_flatten(output_nodes)
        return tree_spec

    def _graph_output(self) -> tuple[tuple[Node, ...], Any]:
        if self.program is None:
            raise GrafiaCompileError("GrafiaCTMBackendAdapter has no ProgramData")
        for node in self.program.graph_module.graph.nodes:
            if node.op == "output":
                flat, tree_spec = tree_flatten(node.args[0])
                if all(isinstance(leaf, Node) for leaf in flat):
                    return tuple(flat), tree_spec
                return (), tree_spec
        raise GrafiaUnsupportedError("FX graph has no output node")

    def _spec_name(self) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", self._region_id).strip("_")
        return safe or "grafia_region"

    def _spec_cache_key(self) -> Any:
        return (
            tuple(self._tensor_cache_record(tensor) for tensor in self.inputs),
            tuple(
                self._constant_cache_record(tensor, value)
                for tensor, value in self.constants.items()
            ),
            tuple(self._op_cache_records),
            tuple(self._tensor_cache_record(tensor) for tensor in self.outputs),
        )

    def _tensor_cache_record(self, tensor: Any) -> Any:
        spec = tensor.spec
        return (
            tensor.name,
            tuple(int(dim) for dim in spec.shape),
            self._dtype_name(spec.dtype),
            getattr(tensor, "producer_id", None),
            getattr(tensor, "producer_idx", None),
        )

    def _constant_cache_record(self, tensor: Any, value: torch.Tensor) -> Any:
        return (
            self._tensor_cache_record(tensor),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
            int(value.data_ptr()) if value.device.type != "meta" else 0,
        )

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        return str(getattr(dtype, "name", dtype))
