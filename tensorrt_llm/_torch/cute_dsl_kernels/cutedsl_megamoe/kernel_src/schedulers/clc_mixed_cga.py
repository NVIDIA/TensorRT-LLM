"""Hardware CLC work-ID acquisition for preferred/fallback (mixed-CGA) cluster launches."""

from typing import List, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Boolean, Int32, extract_mlir_values, new_from_mlir_values

from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.smem_workspace import SmemWorkspace
from .non_clc_mixed_cga import NonClcMixedCgaConfig
from .work_id_claim import ClcMixedCgaWorkIdState, claim_work_id


class ClcMixedCgaSchedulerWorker:
    """Claim linear work IDs from hardware cluster launch control on a mixed-CGA launch.

    Same surface as ``NonClcMixedCgaSchedulerWorker`` plus ``claimed_is_valid``: a failed
    ``try_cancel`` means the grid is exhausted and the caller must emit the DONE tile.
    """

    cluster_pipeline_mbarriers_region = "clc_mixed_cga.cluster_pipeline_mbarriers"
    clc_response_region = "clc_mixed_cga.clc_response"

    def __init__(self, *, config: NonClcMixedCgaConfig, stream_count: int = 1) -> None:
        if stream_count != 1:
            raise ValueError("Cluster launch control exposes exactly one hardware work-ID stream.")
        self.config = config
        self.work_id_mode = "cluster_launch_control"
        self.stream_count = stream_count

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register the cluster claim barriers and the 16-byte CLC response slot."""
        smem_workspace.register_mbarrier(self.cluster_pipeline_mbarriers_region, 2)
        smem_workspace.register_tensor(self.clc_response_region, cutlass.Int32, (4,), byte_alignment=16)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """CLC keeps no GMEM scheduler state."""
        pass

    @cute.jit
    def assign_device_members(
        self,
        *,
        is_fallback_cluster: Optional[Boolean],
        block_idx,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
    ) -> None:
        """Bind this CTA's active cluster to the CLC claim pipeline and seed the bootstrap ticket."""
        if cutlass.const_expr(self.config.is_mixed and is_fallback_cluster is None):
            raise ValueError("is_fallback_cluster is required for a true mixed-CGA launch.")

        preferred_m, preferred_n = self.config.preferred_cluster_shape
        active_cluster_m = preferred_m
        active_cluster_n = preferred_n
        if cutlass.const_expr(self.config.is_mixed):
            if is_fallback_cluster:
                active_cluster_m = Int32(self.config.fallback_cluster_shape[0])
                active_cluster_n = Int32(self.config.fallback_cluster_shape[1])

        cta_coord_in_active_cluster = (
            Int32(block_idx[0]) % Int32(active_cluster_m),
            Int32(block_idx[1]) % Int32(active_cluster_n),
            Int32(0),
        )
        active_cluster_size = active_cluster_m * active_cluster_n
        cluster_pipeline = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * active_cluster_size),
            barrier_storage=smem_workspace.ptr(self.cluster_pipeline_mbarriers_region, smem_base),
            defer_sync=True,
        )
        self._work_id_state = ClcMixedCgaWorkIdState(
            response_pending=Boolean(True),
            grid_m=Int32(block_idx[0]),
            grid_n=Int32(block_idx[1]),
            grid_l=Int32(block_idx[2]),
            response_is_valid=Boolean(True),
            cta_coord_in_cluster=cta_coord_in_active_cluster,
            cluster_pipeline=cluster_pipeline,
            producer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1),
            consumer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1),
            is_leader_cta=(cta_coord_in_active_cluster[0] + cta_coord_in_active_cluster[1]) == Int32(0),
            response_pointer=smem_workspace.ptr(self.clc_response_region, smem_base),
            cluster_size=active_cluster_size,
        )
        self.cta_coord_in_preferred_cluster = (
            Int32(block_idx[0]) % Int32(preferred_m),
            Int32(block_idx[1]) % Int32(preferred_n),
            Int32(0),
        )
        self.claimed_stream_index = Int32(0)
        self.claimed_is_valid = Boolean(True)

    @cute.jit
    def initialize_fallback_group(self) -> None:
        """No fallback registration: hardware assigns fallback clusters to slots."""
        pass

    @cute.jit
    def claim_next_work(self, stream_index=0) -> Int32:
        """Claim the next ticket; publish its validity and this CTA's preferred-cluster coordinate."""
        work_id, self._work_id_state = claim_work_id(self._work_id_state)
        preferred_m, preferred_n = self.config.preferred_cluster_shape
        self.cta_coord_in_preferred_cluster = (
            work_id.grid_m % Int32(preferred_m),
            work_id.grid_n % Int32(preferred_n),
            Int32(0),
        )
        self.claimed_is_valid = work_id.is_valid
        self.claimed_stream_index = Int32(0)
        # Exhausted grid: hand the mapper a harmless in-range ID; the caller substitutes the DONE tile.
        linear_work_id = Int32(0)
        if work_id.is_valid:
            linear_work_id = work_id.grid_l
        return linear_work_id

    @cute.jit
    def produce_tail(self) -> None:
        """Leader waits for the cluster's final release, since peers release on rank 0's empty barrier."""
        work_id_state = self._work_id_state
        if self._work_id_state.is_leader_cta:
            self._work_id_state.cluster_pipeline.producer_tail(self._work_id_state.producer_state)
        else:
            self._work_id_state = work_id_state

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self._work_id_state))
        values.extend(extract_mlir_values(self.cta_coord_in_preferred_cluster))
        values.extend(extract_mlir_values(self.claimed_stream_index))
        values.extend(extract_mlir_values(self.claimed_is_valid))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "ClcMixedCgaSchedulerWorker":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            value_index += field_value_count
            return result

        result = type(self)(config=self.config, stream_count=self.stream_count)
        result._work_id_state = rebuild(self._work_id_state)
        result.cta_coord_in_preferred_cluster = rebuild(self.cta_coord_in_preferred_cluster)
        result.claimed_stream_index = rebuild(self.claimed_stream_index)
        result.claimed_is_valid = rebuild(self.claimed_is_valid)
        if value_index != len(values):
            raise ValueError(
                f"ClcMixedCgaSchedulerWorker MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return result


__all__ = ["ClcMixedCgaSchedulerWorker"]
