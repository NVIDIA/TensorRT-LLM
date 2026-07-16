# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Component-offloading utilities for visual generation pipelines.

Terminology used throughout this module:

- **component**: a named, public sub-model of the pipeline (e.g. ``text_encoder``,
  ``transformer``, ``vae``).
- **stage**: one step of the offload schedule — a single component, or a group
  of components that are co-resident on the GPU and run together before being
  evicted back to CPU.

The offload path keeps model loading and quantization unchanged: weights are
loaded into the modules first, then the components of each stage are copied into
packed CPU storage. At runtime one stage at a time is brought onto a reusable GPU
arena and the original module parameters/buffers are rebound to views of that
storage.
"""

import time
import weakref
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


def _align_offset(offset: int, alignment: int = 256) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def transformer_component_offload_name(component_name: str, transformer_prefix: str) -> str:
    """Map an internal transformer attribute to its public offload component name.

    Transformer component names follow the diffusers ``model_index.json`` keys
    such as ``transformer`` and ``transformer_2``.
    """
    del transformer_prefix
    return component_name


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


# FlashInfer and other custom kernels can require tensor data pointers to be at
# least 16-byte aligned even for smaller dtypes such as BF16.
_PACKED_TENSOR_ALIGNMENT = 16


OffloadPipelineStage = tuple[str, ...]


@dataclass
class _FlatTensorSpec:
    owner: nn.Module
    name: str
    qualified_name: str
    is_parameter: bool
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    persistent: bool
    offset: int
    nbytes: int


@dataclass
class _StageLayout:
    """Packed storage layout and rebound views for one offload stage."""

    name: str
    nbytes: int
    specs: list[_FlatTensorSpec]
    cpu_storage: torch.Tensor | None = None
    cpu_views: tuple[nn.Parameter | torch.Tensor, ...] = ()
    gpu_views: tuple[nn.Parameter | torch.Tensor, ...] = ()


class ModuleOffloadManager:
    """Pack component stages into CPU storage and bring one stage onto the GPU.

    The manager owns packed byte buffers:
    - each layout owns ``cpu_storage`` for one offloaded stage.
    - ``gpu_arena`` is reused for whichever stage is currently active.

    Initializing the manager packs and rebinds one stage at a time. This
    requires enough host memory to allocate the current stage's packed CPU
    storage before that stage's original tensors are released.
    """

    def __init__(
        self,
        stages: Mapping[str, nn.Module],
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("At least one offload stage must be provided")

        self.stages = dict(stages)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.gpu_arena: torch.Tensor | None = None
        self.layouts: dict[str, _StageLayout] = {}
        self.active_stage_name: str | None = None

        for name, module in self.stages.items():
            if not name:
                raise ValueError("Offload stage names must be non-empty")
            if not isinstance(module, nn.Module):
                raise TypeError(f"Offload stage '{name}' must contain an nn.Module")

    @staticmethod
    def _owner_and_name(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
        if "." not in qualified_name:
            return root, qualified_name
        module_path, name = qualified_name.rsplit(".", 1)
        return root.get_submodule(module_path), name

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[int, int] | None:
        if tensor.numel() == 0:
            return None
        storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
        return tensor.untyped_storage().data_ptr(), storage_offset_bytes

    def _get_alias_spec(
        self,
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec],
        tensor: torch.Tensor,
        display_name: str,
    ) -> _FlatTensorSpec | None:
        key = self._storage_key(tensor)
        if key is None:
            return None
        canonical = seen_tensors.get(key)
        if canonical is None:
            return None
        if self._tensor_nbytes(tensor) != canonical.nbytes or tensor.dtype != canonical.dtype:
            raise ValueError(
                "Shared parameters or buffers with different sizes or dtypes are "
                f"not supported by ModuleOffloadManager: '{display_name}' aliases "
                f"'{canonical.qualified_name}'"
            )
        return canonical

    def _build_spec(
        self,
        stage_name: str,
        stage_module: nn.Module,
        qualified_name: str,
        tensor: torch.Tensor,
        is_parameter: bool,
        offset: int,
    ) -> _FlatTensorSpec:
        display_name = f"{stage_name}.{qualified_name}"
        if not tensor.is_contiguous():
            raise ValueError(
                f"Cannot offload non-contiguous tensor '{display_name}' "
                f"with stride {tuple(tensor.stride())}"
            )

        owner, name = self._owner_and_name(stage_module, qualified_name)
        return _FlatTensorSpec(
            owner=owner,
            name=name,
            qualified_name=display_name,
            is_parameter=is_parameter,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad if is_parameter else False,
            persistent=True if is_parameter else name not in owner._non_persistent_buffers_set,
            offset=offset,
            nbytes=self._tensor_nbytes(tensor),
        )

    def _iter_stage_tensors(
        self, stage_module: nn.Module
    ) -> Iterator[tuple[str, torch.Tensor, bool]]:
        for qualified_name, param in stage_module.named_parameters(
            recurse=True,
            remove_duplicate=False,
        ):
            yield qualified_name, param.detach(), True
        for qualified_name, buffer in stage_module.named_buffers(
            recurse=True,
            remove_duplicate=False,
        ):
            yield qualified_name, buffer.detach(), False

    def _append_layout_spec(
        self,
        stage_name: str,
        stage_module: nn.Module,
        qualified_name: str,
        tensor: torch.Tensor,
        is_parameter: bool,
        offset: int,
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec],
        specs: list[_FlatTensorSpec],
    ) -> int:
        """Append a tensor spec and return the next stage-local byte offset.

        This handles three layout concerns in one place: alias reuse, packed
        tensor alignment, and spec construction. The offset is local to this
        stage and is shared by the CPU storage and reusable GPU arena views.
        """
        display_name = f"{stage_name}.{qualified_name}"
        alias = self._get_alias_spec(seen_tensors, tensor, display_name)
        if alias is None:
            offset = _align_offset(offset, _PACKED_TENSOR_ALIGNMENT)

        spec = self._build_spec(
            stage_name=stage_name,
            stage_module=stage_module,
            qualified_name=qualified_name,
            tensor=tensor,
            is_parameter=is_parameter,
            offset=alias.offset if alias is not None else offset,
        )
        specs.append(spec)

        if alias is not None:
            return offset

        key = self._storage_key(tensor)
        if key is not None:
            seen_tensors[key] = spec
        return offset + spec.nbytes

    def _collect_stage_layout(self, stage_name: str, stage_module: nn.Module) -> _StageLayout:
        """Build the packed storage layout for one named component stage."""
        offset = 0
        specs: list[_FlatTensorSpec] = []
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec] = {}

        for qualified_name, tensor, is_parameter in self._iter_stage_tensors(stage_module):
            offset = self._append_layout_spec(
                stage_name=stage_name,
                stage_module=stage_module,
                qualified_name=qualified_name,
                tensor=tensor,
                is_parameter=is_parameter,
                offset=offset,
                seen_tensors=seen_tensors,
                specs=specs,
            )

        if not specs:
            raise ValueError(f"Offload stage '{stage_name}' has no parameters or buffers")

        return _StageLayout(
            name=stage_name,
            nbytes=_align_offset(offset),
            specs=specs,
        )

    def _copy_stage_to_cpu_storage(self, layout: _StageLayout) -> None:
        if layout.cpu_storage is None:
            raise RuntimeError(
                f"CPU storage for offload stage '{layout.name}' has not been allocated"
            )
        for spec in layout.specs:
            if spec.nbytes == 0:
                continue
            try:
                tensor = getattr(spec.owner, spec.name).detach()
                tensor_bytes = tensor.reshape(-1).view(torch.uint8).cpu()
                layout.cpu_storage.narrow(0, spec.offset, spec.nbytes).copy_(tensor_bytes)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to copy offload tensor '{spec.qualified_name}' "
                    f"({_format_bytes(spec.nbytes)}, shape={spec.shape}, dtype={spec.dtype}) "
                    f"to packed CPU storage at offset {spec.offset}."
                ) from e

    def _stage_size_summary(self) -> str:
        return ", ".join(
            f"{name}={_format_bytes(layout.nbytes)}" for name, layout in self.layouts.items()
        )

    def _cuda_allocation_hint(self) -> str:
        if self.device.type != "cuda":
            return ""
        return (
            " If this is due to CUDA memory fragmentation, try setting "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting the process."
        )

    def _allocate_cpu_storage(self, num_bytes: int, stage_name: str | None = None) -> torch.Tensor:
        try:
            return torch.empty(
                num_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        except RuntimeError as e:
            stage_context = f", stage='{stage_name}'" if stage_name is not None else ""
            raise RuntimeError(
                "Failed to allocate packed CPU storage for visual generation offload "
                f"({_format_bytes(num_bytes)}, {num_bytes} bytes{stage_context}, "
                f"pin_memory={self.pin_memory}, stages=[{self._stage_size_summary()}])."
            ) from e

    def _allocate_gpu_arena(self, num_bytes: int) -> torch.Tensor:
        try:
            return torch.empty(num_bytes, dtype=torch.uint8, device=self.device)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to allocate GPU arena for visual generation offload "
                f"({_format_bytes(num_bytes)}, {num_bytes} bytes, "
                f"device={self.device}, stages=[{self._stage_size_summary()}])."
                f"{self._cuda_allocation_hint()}"
            ) from e

    def _make_views(
        self,
        layout: _StageLayout,
        storage: torch.Tensor,
    ) -> tuple[nn.Parameter | torch.Tensor, ...]:
        views: list[nn.Parameter | torch.Tensor] = []
        for spec in layout.specs:
            view = storage.narrow(0, spec.offset, spec.nbytes).view(spec.dtype)
            view = view.as_strided(spec.shape, spec.stride)
            if spec.is_parameter:
                views.append(nn.Parameter(view, requires_grad=spec.requires_grad))
            else:
                views.append(view)
        return tuple(views)

    def _bind_views(
        self,
        layout: _StageLayout,
        views: tuple[nn.Parameter | torch.Tensor, ...],
    ) -> None:
        for spec, view in zip(layout.specs, views, strict=True):
            if spec.is_parameter:
                if not isinstance(view, nn.Parameter):
                    raise TypeError(
                        f"Expected offload view '{spec.name}' to be an nn.Parameter, "
                        f"got {type(view).__name__}"
                    )
                spec.owner.register_parameter(spec.name, view)
            else:
                if not isinstance(view, torch.Tensor):
                    raise TypeError(
                        f"Expected offload view '{spec.name}' to be a torch.Tensor, "
                        f"got {type(view).__name__}"
                    )
                spec.owner.register_buffer(spec.name, view, persistent=spec.persistent)

    def initialize(self) -> None:
        """Allocate packed storage, copy current tensors, and bind CPU views."""
        if self.layouts:
            raise RuntimeError("ModuleOffloadManager has already been initialized")

        start_time = time.time()
        for name, module in self.stages.items():
            layout = self._collect_stage_layout(name, module)
            self.layouts[name] = layout

        total_cpu_bytes = sum(layout.nbytes for layout in self.layouts.values())
        max_gpu_bytes = max(layout.nbytes for layout in self.layouts.values())
        logger.info(
            "Module offload storage layout: "
            f"cpu_total={_format_bytes(total_cpu_bytes)}, "
            f"gpu_arena={_format_bytes(max_gpu_bytes)}, "
            f"stages=[{self._stage_size_summary()}], device={self.device}"
        )

        # Pack and rebind one stage at a time. This keeps setup simple and fast:
        # offloading requires enough host memory to allocate one stage's packed
        # CPU storage before that stage's original tensors are released.
        for layout in self.layouts.values():
            logger.info(
                "Module offload packing stage into CPU storage: "
                f"{layout.name} ({_format_bytes(layout.nbytes)})"
            )
            layout.cpu_storage = self._allocate_cpu_storage(layout.nbytes, stage_name=layout.name)
            self._copy_stage_to_cpu_storage(layout)
            layout.cpu_views = self._make_views(layout, layout.cpu_storage)
            self._rebind_to_cpu(layout.name)

        self.gpu_arena = self._allocate_gpu_arena(max_gpu_bytes)
        for layout in self.layouts.values():
            layout.gpu_views = self._make_views(layout, self.gpu_arena)

        logger.info(f"Module offload setup completed in {time.time() - start_time:.2f}s")

    def _get_layout(self, name: str) -> _StageLayout:
        try:
            return self.layouts[name]
        except KeyError as e:
            raise KeyError(
                f"Unknown offload stage '{name}'. Available stages: {sorted(self.layouts)}"
            ) from e

    def stage(self, name: str) -> None:
        """Bring one offload stage onto the GPU arena and rebind its tensors."""
        layout = self._get_layout(name)
        if self.active_stage_name == name:
            return
        if layout.cpu_storage is None or self.gpu_arena is None:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")

        if self.active_stage_name is not None:
            self._rebind_to_cpu(self.active_stage_name)
            self.active_stage_name = None

        src = layout.cpu_storage.narrow(0, 0, layout.nbytes)
        dst = self.gpu_arena.narrow(0, 0, layout.nbytes)
        try:
            dst.copy_(src, non_blocking=layout.cpu_storage.is_pinned())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to bring offload stage '{name}' ({_format_bytes(layout.nbytes)}) "
                f"onto {self.device}"
            ) from e
        self._rebind_to_gpu(name)
        self.active_stage_name = name

    def _rebind_to_cpu(self, name: str) -> None:
        layout = self._get_layout(name)
        if not layout.cpu_views:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")
        self._bind_views(layout, layout.cpu_views)

    def _rebind_to_gpu(self, name: str) -> None:
        layout = self._get_layout(name)
        if not layout.gpu_views:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")
        self._bind_views(layout, layout.gpu_views)


class OffloadPipeline:
    """Bring offload stages onto the GPU explicitly from model call-site contexts.

    This class intentionally does not use forward hooks. Pipeline code must wrap
    the relevant call site with ``with self.offloader.context("stage")`` so staging
    happens before the model invocation and outside any later CUDA graph capture.
    """

    def __init__(
        self,
        stages: Sequence[Sequence[str] | str],
        components: Mapping[str, nn.Module],
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("At least one offload pipeline stage must be provided")

        self.stages = tuple(
            (stage,) if isinstance(stage, str) else tuple(stage) for stage in stages
        )
        self.components = dict(components)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.manager = ModuleOffloadManager(
            stages=self._build_stage_modules(),
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _build_stage_modules(self) -> dict[str, nn.Module]:
        stage_modules: dict[str, nn.Module] = {}
        for stage in self.stages:
            stage_name = self._stage_name(stage)
            if not stage:
                raise ValueError("Offload pipeline stages must have at least one component")
            if stage_name in stage_modules:
                raise ValueError(f"Duplicate offload pipeline stage: {stage_name}")

            modules: list[nn.Module] = []
            for component_name in stage:
                try:
                    component = self.components[component_name]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown offload pipeline component '{component_name}' for stage "
                        f"'{stage_name}'. Available components: {sorted(self.components)}"
                    ) from e
                modules.append(component)

            stage_module = modules[0] if len(modules) == 1 else nn.ModuleList(modules)
            stage_modules[stage_name] = stage_module

        return stage_modules

    def initialize(self) -> None:
        """Allocate and populate backing storage for all configured stages."""
        self.manager.initialize()

    @staticmethod
    def _stage_name(stage: Sequence[str] | str) -> str:
        return stage if isinstance(stage, str) else "+".join(stage)

    def context(self, stage_name: str):
        """Bring stage ``stage_name`` onto the GPU and return a no-op context manager."""
        self.manager.stage(stage_name)
        # The active stage intentionally stays resident after the call site.
        # The next stage() call rebinds it back to CPU before bringing another
        # stage on, and cleanup() handles the final rebind when the pipeline exits.
        return nullcontext()

    def cleanup(self) -> None:
        """Return the active stage to CPU-backed views."""
        if self.manager.active_stage_name is not None:
            self.manager._rebind_to_cpu(self.manager.active_stage_name)
            self.manager.active_stage_name = None


class PipelineOffloader:
    """Resolve, build, and drive component offloading for a visual-gen pipeline.

    Keeps ``BasePipeline`` free of offload bookkeeping: the pipeline only supplies
    model-specific hooks (``offload_pipeline_components``, ``default_offload_stages``,
    ``extra_offload_component_names``), while this coordinator resolves configured
    stages, validates/filters them, builds the backing :class:`OffloadPipeline`,
    and brings stages on/off the GPU at explicit call sites.
    """

    def __init__(self, pipeline: "BasePipeline") -> None:
        # Hold the owning pipeline weakly: the pipeline owns this offloader
        # (``pipeline.offloader``), so a strong back-reference would form a
        # reference cycle and keep the whole pipeline (including its GPU
        # weights) alive until a ``gc.collect()`` runs, defeating prompt
        # release on ``del pipeline``.
        self._pipeline_ref = weakref.ref(pipeline)
        self._offload_pipeline: OffloadPipeline | None = None
        self._stage_name_by_component: dict[str, str] = {}

    @property
    def _pipeline(self) -> "BasePipeline":
        pipeline = self._pipeline_ref()
        if pipeline is None:
            raise ReferenceError("PipelineOffloader's owning pipeline has already been released")
        return pipeline

    @property
    def offload_pipeline(self) -> OffloadPipeline | None:
        """The backing :class:`OffloadPipeline`, or ``None`` if not initialized."""
        return self._offload_pipeline

    @staticmethod
    def _normalize_stages(
        stages: list[str | list[str]],
    ) -> tuple[OffloadPipelineStage, ...]:
        """Normalize user-configured offload stages to tuple form."""
        normalized_stages: list[OffloadPipelineStage] = []
        for stage in stages:
            components = (stage,) if isinstance(stage, str) else tuple(stage)
            if not components:
                raise ValueError("Offload stages must contain at least one component")
            normalized_stages.append(components)
        return tuple(normalized_stages)

    def stages(self) -> tuple[OffloadPipelineStage, ...]:
        """Return offload stages resolved from the pipeline configuration."""
        cpu_offload_config = getattr(self._pipeline.pipeline_config, "cpu_offload_config", None)
        if cpu_offload_config is None or not cpu_offload_config.enable:
            return ()
        configured_stages = getattr(cpu_offload_config, "stages", None)
        if configured_stages is not None:
            return self._normalize_stages(configured_stages)
        return self._pipeline.default_offload_stages()

    def requested_components(self) -> set[str]:
        """Return all component names referenced by the configured stages."""
        return {component for stage in self.stages() for component in stage}

    def validate_configured_stages(
        self,
        stages: tuple[OffloadPipelineStage, ...],
        available_components: Mapping[str, nn.Module],
    ) -> None:
        """Validate user-configured stages against this pipeline's component names."""
        known = dict(available_components)
        for name in self._pipeline.extra_offload_component_names():
            known.setdefault(name, None)
        unknown_components = sorted(
            {component for stage in stages for component in stage if component not in known}
        )
        if unknown_components:
            available_names = ", ".join(sorted(known)) or "<none>"
            unknown_names = ", ".join(unknown_components)
            raise ValueError(
                f"Unknown cpu_offload_config.stages entries for this model: {unknown_names}. "
                f"Available components: {available_names}."
            )

    @staticmethod
    def filter_available_stages(
        stages: tuple[OffloadPipelineStage, ...],
        available_components: Mapping[str, nn.Module],
    ) -> tuple[OffloadPipelineStage, ...]:
        """Drop stage components that are unavailable for the loaded pipeline."""
        return tuple(
            tuple(component for component in stage if component in available_components)
            for stage in stages
            if any(component in available_components for component in stage)
        )

    def initialize(self) -> None:
        """Create and initialize the offload pipeline after weights are loaded."""
        configured_stages = self.stages()
        if not configured_stages or self._offload_pipeline is not None:
            return

        pipeline = self._pipeline
        available_components = pipeline.offload_pipeline_components()
        cpu_offload_config = getattr(pipeline.pipeline_config, "cpu_offload_config", None)
        has_user_stages = (
            cpu_offload_config is not None
            and getattr(cpu_offload_config, "stages", None) is not None
        )
        if has_user_stages:
            self.validate_configured_stages(configured_stages, available_components)

        stages = self.filter_available_stages(configured_stages, available_components)
        if not stages:
            return

        if pipeline._cuda_graph_runners:
            raise NotImplementedError(
                "CUDA graphs are not supported with visual generation offloading yet. "
                "Disable either cuda_graph_config.enable or cpu_offload_config.enable."
            )

        pin_memory = bool(getattr(cpu_offload_config, "pin_memory", True))
        stage_summary = " -> ".join("+".join(stage) for stage in stages)
        logger.info(f"{pipeline.__class__.__name__} offload pipeline enabled: {stage_summary}")
        offload_pipeline = OffloadPipeline(
            stages=stages,
            components=available_components,
            device=torch.device(pipeline.device),
            pin_memory=pin_memory,
        )
        offload_pipeline.initialize()
        self._offload_pipeline = offload_pipeline
        self._stage_name_by_component = {
            component: "+".join(stage) for stage in stages for component in stage
        }

    def cleanup(self) -> None:
        """Tear down the offload pipeline and reset state."""
        if self._offload_pipeline is not None:
            self._offload_pipeline.cleanup()
            self._offload_pipeline = None
            self._stage_name_by_component = {}

    def context(self, component_name: str, enable: bool = True) -> AbstractContextManager:
        """Bring the stage containing ``component_name`` onto the GPU for one call site."""
        if not enable:
            return nullcontext()
        if self.stages() and self._offload_pipeline is None:
            raise RuntimeError(
                "Visual generation offloading is configured but the offload pipeline "
                "has not been initialized. Call initialize_offload_pipeline() after "
                "loading weights."
            )
        if self._offload_pipeline is None:
            return nullcontext()
        stage_name = self._stage_name_by_component.get(component_name, component_name)
        return self._offload_pipeline.context(stage_name)

    def context_if_requested(self, component_name: str) -> AbstractContextManager:
        """Return an offload context for ``component_name``, active only if configured."""
        return self.context(component_name, enable=component_name in self.requested_components())
