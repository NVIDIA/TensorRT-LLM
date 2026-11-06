"""Warp execution and scratch storage for dedicated token communication CTAs."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

from ...helpers.smem_workspace import SmemWorkspace
from .token_comm import TokenCommNonDeterministic


class TokenCommCtaMainloop:
    """Run token-in and optional concurrent token-back on a dedicated CTA.

    Device buffers and counters belong to the shared TokenComm protocol. This
    mainloop owns the communication CTA's warp branches and SMEM placement;
    each transfer operation initializes and drains its own asynchronous copies.
    The enclosing kernel performs the all-CTA reset after both sides finish.
    """

    def __init__(
        self,
        token_comm: TokenCommNonDeterministic,
        *,
        token_back_on_compute_ctas: bool,
    ) -> None:
        self.token_comm = token_comm
        self.token_back_on_compute_ctas = token_back_on_compute_ctas
        self.dispatch_warp_start = token_comm.dispatch_warp_start
        self.dispatch_warp_count = token_comm.dispatch_warp_count
        self.combine_warp_start = token_comm.combine_warp_start
        self.combine_warp_count = token_comm.combine_warp_count
        if token_comm.token_back_mode != "standalone_warps":  # nosec B105
            raise ValueError("Dedicated communication CTAs require standalone token-back warps.")
        if not token_back_on_compute_ctas:
            dispatch_end = self.dispatch_warp_start + self.dispatch_warp_count
            combine_end = self.combine_warp_start + self.combine_warp_count
            if self.dispatch_warp_start < combine_end and self.combine_warp_start < dispatch_end:
                raise ValueError("Concurrent token-in and token-back warp intervals must not overlap.")

    def register_smem_regions(self, workspace: SmemWorkspace) -> None:
        """Declare communication CTA scratch independently of fused placement."""
        self.register_token_in_smem(workspace)
        if not self.token_back_on_compute_ctas:
            self.register_token_back_smem(workspace)

    def register_token_in_smem(self, workspace: SmemWorkspace) -> None:
        """Declare input scratch for the dedicated dispatch warp group."""
        comm = self.token_comm
        workspace.register_mbarrier(
            comm.token_in_mbarrier_region,
            self.dispatch_warp_count * comm.dispatch_stage_depth,
        )
        workspace.register_tensor(
            comm.expert_sizes_smem_region,
            cutlass.Int32,
            (comm.experts_per_rank,),
            byte_alignment=16,
        )
        staging = workspace.create_overlay("nvlink.token_comm.main_smem.transfer").add_lifetime("token_in")
        comm._register_token_in_staging(staging)

    def register_token_back_smem(self, workspace: SmemWorkspace) -> None:
        """Declare output scratch on whichever CTA executes standalone token-back."""
        comm = self.token_comm
        if not comm.token_back_enabled:
            return
        workspace.register_mbarrier(
            comm.token_back_mbarrier_region,
            self.combine_warp_count * comm.combine_stage_depth,
        )
        staging = workspace.create_overlay("nvlink.token_comm.main_smem.standalone_token_back").add_lifetime(
            "token_back"
        )
        comm._register_token_back_staging(staging)

    @cute.jit
    def run(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
    ) -> None:
        """Run independent warp groups without joining them before their drains."""
        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        if (warp_idx >= Int32(self.dispatch_warp_start)) & (
            warp_idx < Int32(self.dispatch_warp_start + self.dispatch_warp_count)
        ):
            TokenCommNonDeterministic.token_in(self.token_comm, smem_workspace, smem_base)
        elif cutlass.const_expr(not self.token_back_on_compute_ctas):
            if (warp_idx >= Int32(self.combine_warp_start)) & (
                warp_idx < Int32(self.combine_warp_start + self.combine_warp_count)
            ):
                TokenCommNonDeterministic.token_back(self.token_comm, smem_workspace, smem_base)
