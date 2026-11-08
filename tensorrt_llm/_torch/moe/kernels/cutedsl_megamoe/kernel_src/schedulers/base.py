"""Scheduler façade and architecture-independent work-tile transport."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values

from ...api import ImplDesc, KernelComponent, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.smem_workspace import SmemWorkspace


WorkIdAcquisitionMode = Literal["grid_stride", "atomic_counter", "cluster_launch_control"]


class SchedulerWorkTileBase(ABC):
    """Register ABI for one work tile transported through scheduler SMEM."""

    storage_dtype: ClassVar[type] = cutlass.Int32
    storage_field_count: ClassVar[int]

    @property
    @abstractmethod
    def is_valid_tile(self):
        """Return whether this tile names executable work."""
        ...

    @abstractmethod
    def to_rmem(self) -> cute.Tensor:
        """Serialize this tile into its one-dimensional register ABI."""
        ...

    @classmethod
    @abstractmethod
    def from_rmem(cls, registers: cute.Tensor) -> "SchedulerWorkTileBase":
        """Deserialize one tile from its register ABI."""
        ...


class SchedulerConsumer:
    """Per-consumer state for the common scheduler SMEM transport."""

    def __init__(
        self,
        scheduler_pipeline: pipeline.PipelineAsync,
        smem_buffer: cute.Tensor,
        num_stages: int,
        work_tile_type: Type[SchedulerWorkTileBase],
    ) -> None:
        self._pipeline = scheduler_pipeline
        self._smem_buffer = smem_buffer
        self._work_tile_type = work_tile_type
        self._consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, num_stages)

    def __extract_mlir_values__(self) -> list:
        return extract_mlir_values(self._consumer_state)

    def __new_from_mlir_values__(self, values: list) -> "SchedulerConsumer":
        expected_value_count = len(extract_mlir_values(self._consumer_state))
        if len(values) != expected_value_count:
            raise ValueError(
                f"SchedulerConsumer MLIR value count mismatch: expected {expected_value_count}, got {len(values)}."
            )
        result = type(self).__new__(type(self))
        result._pipeline = self._pipeline
        result._smem_buffer = self._smem_buffer
        result._work_tile_type = self._work_tile_type
        result._consumer_state = new_from_mlir_values(self._consumer_state, values)
        return result

    @cute.jit
    def consume_work(self) -> SchedulerWorkTileBase:
        """Block until the next work tile is available."""
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._work_tile_type.storage_dtype, num_bits_per_copy=128
        )
        self._pipeline.consumer_wait(self._consumer_state)
        registers = cute.make_rmem_tensor(
            (self._work_tile_type.storage_field_count,), self._work_tile_type.storage_dtype
        )
        cute.copy(copy_atom, self._smem_buffer[(None, self._consumer_state.index)], registers)
        work_tile = self._work_tile_type.from_rmem(registers)
        cute.arch.fence_acq_rel_cta()
        self._pipeline.consumer_release(self._consumer_state)
        self._consumer_state.advance()
        return work_tile


class SchedulerBase(KernelComponent):
    """Common work-tile transport and façade protocol for schedulers."""

    pipeline_mbarriers_region = "scheduler.pipeline_mbarriers"
    work_tiles_region = "scheduler.work_tiles"
    num_scheduler_stages = 2

    @classmethod
    def problem_desc_require(cls) -> dict:
        return {}

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {"num_scheduler_consumer_threads": int}

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)
        self.num_scheduler_consumer_threads = impl_desc["num_scheduler_consumer_threads"]
        if self.num_scheduler_consumer_threads <= 0:
            raise ValueError("num_scheduler_consumer_threads must be positive.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register the common work-tile transport regions."""
        if not hasattr(self, "work_tile_type"):
            raise AttributeError(
                f"{type(self).__name__} must bind work_tile_type before registering scheduler SMEM regions."
            )
        work_tile_type = self.work_tile_type
        work_tile_field_count = work_tile_type.storage_field_count
        smem_workspace.register_mbarrier(self.pipeline_mbarriers_region, self.num_scheduler_stages * 2)
        smem_workspace.register_tensor(
            self.work_tiles_region,
            work_tile_type.storage_dtype,
            (work_tile_field_count, self.num_scheduler_stages),
            stride=(1, work_tile_field_count),
            byte_alignment=16,
        )

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register scheduler-specific GMEM regions when needed."""
        pass

    @cute.jit
    def create_scheduler_pipelines(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Create CTA-lifetime scheduler pipelines and transport state."""
        self._pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_scheduler_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_scheduler_consumer_threads),
            barrier_storage=smem_workspace.ptr(self.pipeline_mbarriers_region, smem_base),
            defer_sync=True,
        )
        self._smem_buffer = smem_workspace.tensor(self.work_tiles_region, smem_base)
        self._producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_scheduler_stages
        )

    def make_consumer(self) -> SchedulerConsumer:
        """Create a consumer with an independent pipeline state."""
        return SchedulerConsumer(
            scheduler_pipeline=self._pipeline,
            smem_buffer=self._smem_buffer,
            num_stages=self.num_scheduler_stages,
            work_tile_type=self.work_tile_type,
        )

    @cute.jit
    def publish_work(self, work_tile: SchedulerWorkTileBase) -> None:
        """Publish one work tile through the common transport pipeline."""
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), work_tile.storage_dtype, num_bits_per_copy=128)
        self._pipeline.producer_acquire(self._producer_state)
        cute.copy(copy_atom, work_tile.to_rmem(), self._smem_buffer[(None, self._producer_state.index)])
        cute.arch.fence_proxy("async.shared", space="cta")
        self._pipeline.producer_commit(self._producer_state)
        self._producer_state.advance()

    @cute.jit
    def produce_tail(self) -> None:
        """Wait until every published work tile has been consumed."""
        self._pipeline.producer_tail(self._producer_state)

    def __extract_mlir_values__(self) -> list:
        return extract_mlir_values(self._producer_state)

    def __new_from_mlir_values__(self, values: list) -> "SchedulerBase":
        expected_value_count = len(extract_mlir_values(self._producer_state))
        if len(values) != expected_value_count:
            raise ValueError(
                f"SchedulerBase MLIR value count mismatch: expected {expected_value_count}, got {len(values)}."
            )
        result = type(self).__new__(type(self))
        result.num_scheduler_consumer_threads = self.num_scheduler_consumer_threads
        result.work_tile_type = self.work_tile_type
        result._pipeline = self._pipeline
        result._smem_buffer = self._smem_buffer
        result._producer_state = new_from_mlir_values(self._producer_state, values)
        return result

    @abstractmethod
    def get_grid_shape(self, *, max_active_clusters: Optional[int] = None, problem_desc: Any = None):
        """Return the launch grid from static scheduler policy."""
        ...

    @abstractmethod
    def assign_device_members(self, *args, **kwargs) -> None:
        """Initialize device members whose ownership spans one CTA lifetime."""
        ...

    @abstractmethod
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim, map, and return the next work tile."""
        ...


__all__ = ["SchedulerBase", "SchedulerConsumer", "SchedulerWorkTileBase", "WorkIdAcquisitionMode"]
