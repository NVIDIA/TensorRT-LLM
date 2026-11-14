# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
from typing import Any, Optional, Tuple

import cutlass
import torch

from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as shared
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
    _LOCKED_KERNEL_KWARGS,
    _SF_PADDING_BLOCK,
    Sm100MegaMoENvfp4Runner,
    _default_megamoe_tactic_for_key,
    _launch_cluster_configuration,
    _unpack_tactic,
    validate_megamoe_tactic,
)
from tensorrt_llm.logger import logger

from ...kernels.moe.constants import GEN_TOKEN_BUCKETS


def _construct_rubin_kernel(
    *, sm_version: int, common: dict, combine_format: str
) -> Tuple[Any, Optional[Exception]]:
    """Construct the active architecture kernel and report a narrow gen rejection."""
    from cutlass.cute.nvgpu import OperandMajorMode

    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import ImplDesc, ProblemDesc
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def import CombineFormat, QuantKind

    from ...kernels.moe.block_scaled_swap_ab_mega_moe_kernel_gen_specialized import (
        BlockScaledSwapAbGenphaseMoeKernel,
    )

    local_experts, intermediate_gateup_size, hidden_size = common["static_expert_shape"]
    problem_desc = ProblemDesc(
        {
            "expert_count": int(local_experts) * int(common["world_size"]),
            "intermediate_gateup_size": int(intermediate_gateup_size),
            "hidden_size": int(hidden_size),
            "quant_kind": "nvfp4",
            "a_major_mode": OperandMajorMode.K,
            "b_major_mode": OperandMajorMode.K,
            "combine_format": CombineFormat.parse(combine_format),
            "gate_up_clamp": common["gate_up_clamp"],
            "swiglu_alpha": common["swiglu_alpha"],
            "swiglu_beta": common["swiglu_beta"],
            "situ_beta": common["situ_beta"],
            "situ_linear_beta": common["situ_linear_beta"],
            "world_size": int(common["world_size"]),
            "topk": int(common["num_topk"]),
            "topk_index_dtype": cutlass.Int32 if sm_version == 107 else cutlass.Int64,
            "max_tokens_per_rank": int(common["max_tokens_per_rank"]),
            "apply_topk_at_fc1": bool(common["apply_topk_in_fc1"]),
        }
    )
    mma_tiler = tuple(common["mma_tiler_mnk"])
    impl_values = {
        "mma_tiler_mnk": mma_tiler,
        "cluster_shape_mn": tuple(common["cluster_shape_mnk"][:2]),
        "use_2cta_instrs": bool(common["use_2cta_instrs"]),
        "schedule_policy": tuple(common["schedule_policy"]),
        "token_padding_block": int(common["token_padding_block"]),
        "sf_padding_block": int(common["sf_padding_block"]),
        "work_id_mode": str(common["work_id_mode"]),
        "fc2_use_bulk": not bool(common["non_ubulk_fc2_store"]),
        "epi_flag_batches": tuple(common["epi_flag_batch"]),
        # Kernel construction and READY arguments must use the same helper count.
        "helper_expert_count": int(common.get("helper_expert_count", 0)),
        "launch_cluster_count": int(common["launch_cluster_count"]),
        "fallback_cluster_shape_mn": common["fallback_cluster_shape_mn"],
        "preferred_cluster_count": common["preferred_cluster_count"],
        "fallback_cluster_count": common["fallback_cluster_count"],
        "token_in_flag_batch": int(common["flag_batch"]),
        "token_back_mode": str(common["token_back_mode"]),
        "token_back_ready_granularity": str(common.get("token_back_ready_granularity", "expert")),
        "reduce_topk_in_kernel": bool(common["in_kernel_fc2_reduce"]),
    }
    if sm_version == 107:
        impl_values.update(
            mma_instruction_mnk=(
                mma_tiler[0],
                mma_tiler[1],
                QuantKind.nvfp4.instruction_k("2x"),
            ),
            mma_k_mode="2x",
        )
    if common["fc2_tma_stages"] is not None:
        impl_values["fc2_tma_stages"] = int(common["fc2_tma_stages"])
    impl_values["defer_topk_reduce"] = bool(common.get("defer_topk_reduce", False))
    impl_desc = ImplDesc(impl_values)

    def construct(kernel_cls):
        # ProblemDesc.validate ignores undeclared keys, so a kernel without the SwiGLU
        # constants would silently run plain SwiGLU.
        if (
            common["swiglu_alpha"] is not None
            and "swiglu_alpha" not in kernel_cls.problem_desc_require()
        ):
            raise NotImplementedError(f"{kernel_cls.__name__} does not support swiglu_alpha.")
        return kernel_cls(problem_desc, impl_desc)

    return construct(BlockScaledSwapAbGenphaseMoeKernel), None


class RubinMegaMoERunner(Sm100MegaMoENvfp4Runner):
    kernel_cache = {}
    _kernel_descriptor_cache = {}
    defer_topk_reduce = False

    def _build_unique_id(self):
        # The shape/tactic keeps the shared runner's positional key contract.
        return (*super()._build_unique_id(), "glm-rubin-high-throughput")

    def _build_kernel_once(self, tactic: Tuple):
        validate_megamoe_tactic(tactic, sm_version=self.sm_version)
        cached = self.__class__.kernel_cache.get(self._tactic_cache_key(tactic))
        if cached is not None:
            return cached[1]
        (
            mma_tiler,
            cluster_shape,
            fallback_cluster_shape,
            schedule_policy,
            work_id_mode,
            token_back_mode,
            use_bulk_fc2_store,
            fc2_tma_stages,
            flag_batch,
            epi_flag_batch,
            token_back_ready_granularity,
        ) = _unpack_tactic(tactic)
        (
            launch_cluster_count,
            preferred_cluster_count,
            fallback_cluster_count,
            _,
        ) = _launch_cluster_configuration(
            cluster_shape,
            fallback_cluster_shape,
            self.sm_version,
        )
        schedule_hint = schedule_policy[1]
        common = dict(
            mma_tiler_mnk=tuple(mma_tiler),
            cluster_shape_mnk=tuple(cluster_shape),
            use_2cta_instrs=bool(mma_tiler[0] == 256),
            group_hint=(512 if schedule_hint is None else int(schedule_hint)),
            token_padding_block=64,
            sf_padding_block=_SF_PADDING_BLOCK,
            load_balance_mode=("atomic_counter" if work_id_mode == "atomic_counter" else "static"),
            fallback_cluster_shape_mn=(
                None if fallback_cluster_shape is None else tuple(fallback_cluster_shape[:2])
            ),
            schedule_policy=tuple(schedule_policy),
            work_id_mode=str(work_id_mode),
            launch_cluster_count=launch_cluster_count,
            preferred_cluster_count=preferred_cluster_count,
            fallback_cluster_count=fallback_cluster_count,
            fc2_tma_stages=fc2_tma_stages,
            static_expert_shape=(
                self.num_experts_per_rank,
                self.expand_intermediate_size_per_partition,
                self.hidden_size,
            ),
            world_size=self.world_size,
            num_topk=self.num_topk,
            max_tokens_per_rank=self.max_tokens_per_rank,
            hidden=self.hidden_size,
            fc2_output_dtype=cutlass.BFloat16,
            in_kernel_fc2_reduce=self.in_kernel_fc2_reduce,
            token_back_mode=str(token_back_mode),
            token_back_ready_granularity=str(token_back_ready_granularity),
            non_ubulk_fc2_store=(not bool(use_bulk_fc2_store)),
            flag_batch=int(flag_batch),
            epi_flag_batch=tuple(epi_flag_batch),
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            gate_up_clamp=self.gate_up_clamp,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            helper_expert_count=self.helper_expert_count,
            defer_topk_reduce=self.defer_topk_reduce,
            **_LOCKED_KERNEL_KWARGS,
        )
        kernel, rejection = _construct_rubin_kernel(
            sm_version=self.sm_version,
            common=common,
            combine_format=self.combine_format,
        )
        if rejection is not None:
            logger.debug(
                f"[MegaMoE] gen specialization rejected tactic {tactic!r}: "
                f"{type(rejection).__name__}: {rejection}"
            )
        return kernel


class RubinDeferredMegaMoERunner(RubinMegaMoERunner):
    # Deferred and reduced launchers share device workspace/fence identity, but
    # their compiled launchers and descriptors must never alias.
    kernel_cache = {}
    _kernel_descriptor_cache = {}
    defer_topk_reduce = True


def query_rubin_kernel_properties(**kwargs):
    if kwargs["max_tokens_per_rank"] not in GEN_TOKEN_BUCKETS:
        return shared.query_megamoe_kernel_properties(**kwargs)
    kwargs = dict(kwargs)
    tactic = kwargs.pop("tactic", None)
    runner = RubinMegaMoERunner(local_rank=0, output_dtype=torch.bfloat16, **kwargs)
    if tactic is None:
        tactic = _default_megamoe_tactic_for_key(
            runner.unique_id(), num_tokens=runner.max_tokens_per_rank
        )
    kernel, _ = runner._resolve_kernel_descriptor(tactic, allow_fallback=True)
    return int(kernel.get_workspace_sizes()[1]), True


def runner_kernel(runner):
    tactic = _default_megamoe_tactic_for_key(
        runner.unique_id(), num_tokens=runner.max_tokens_per_rank
    )
    return runner._resolve_kernel_descriptor(tactic, allow_fallback=True)[0]


@lru_cache(maxsize=None)
def get_rubin_runner(
    *,
    world_size: int,
    local_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    expand_intermediate_size_per_partition: int,
    max_tokens_per_rank: int,
    apply_topk_in_fc1: bool,
    gate_up_clamp: Optional[float],
    in_kernel_fc2_reduce: bool,
    combine_format: str,
    defer_topk_reduce: bool,
):
    runner_cls = RubinDeferredMegaMoERunner if defer_topk_reduce else RubinMegaMoERunner
    return runner_cls(
        world_size=world_size,
        local_rank=local_rank,
        num_topk=num_topk,
        num_experts_per_rank=num_experts_per_rank,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        expand_intermediate_size_per_partition=expand_intermediate_size_per_partition,
        max_tokens_per_rank=max_tokens_per_rank,
        output_dtype=torch.bfloat16,
        apply_topk_in_fc1=apply_topk_in_fc1,
        gate_up_clamp=gate_up_clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        combine_format=combine_format,
    )


def run_rubin_megamoe(
    activation: torch.Tensor,
    activation_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    fc1_norm_const: torch.Tensor,
    combine_output: torch.Tensor,
    shared_workspace: torch.Tensor,
    peer_offsets: list[int],
    world_size: int,
    local_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    expand_intermediate_size_per_partition: int,
    max_tokens_per_rank: int,
    apply_topk_in_fc1: bool,
    gate_up_clamp: Optional[float],
    in_kernel_fc2_reduce: bool,
    combine_format: str,
    defer_topk_reduce: bool,
) -> None:
    runner = get_rubin_runner(
        world_size=world_size,
        local_rank=local_rank,
        num_topk=num_topk,
        num_experts_per_rank=num_experts_per_rank,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        expand_intermediate_size_per_partition=expand_intermediate_size_per_partition,
        max_tokens_per_rank=max_tokens_per_rank,
        apply_topk_in_fc1=apply_topk_in_fc1,
        gate_up_clamp=gate_up_clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        combine_format=combine_format,
        defer_topk_reduce=defer_topk_reduce,
    )
    runner(
        [
            activation,
            activation_sf,
            topk_idx,
            topk_weights,
            fc1_weight,
            fc1_weight_sf,
            fc2_weight,
            fc2_weight_sf,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            combine_output,
        ],
        tactic=-1,
        peer_offsets=peer_offsets,
        shared_workspace=shared_workspace,
    )


# Compiler analysis must see both the output write and the peer-visible exchange
# workspace mutation. Eager launches call the runner directly to avoid dispatcher overhead.
run_rubin_megamoe_op = torch.library.custom_op(
    "trtllm::glm_rubin_megamoe", mutates_args=("combine_output", "shared_workspace")
)(run_rubin_megamoe)


@run_rubin_megamoe_op.register_fake
def _fake_run_rubin_megamoe(*args, **kwargs):
    return None
