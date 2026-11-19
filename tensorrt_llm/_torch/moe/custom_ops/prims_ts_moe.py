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
"""``trtllm::prims_ts_moe``: the vendored PrimsTS MoE GEMMs over TRT-LLM routing.

Routing and finalize are TRT-LLM's own ops (``moe_sort``,
``moe_unpermute_inplace``); only the FC1 and FC2 batched GEMMs come from the
FlashInfer PrimsTS sources vendored under ``moe/kernels/prims_ts``. Their
launch arguments are built the way FlashInfer's own runner builds them
(``prims_ts/moe/runner.py`` upstream), and the tactic is chosen by the TRT-LLM
autotuner.
"""

from dataclasses import replace
from functools import lru_cache
from typing import Callable, NamedTuple, Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.math_utils import ceil_div

from ...autotuner import AutoTuner, TunableRunner, TuningConfig
from ...custom_ops.trtllm_gen_custom_ops import (
    FP4BlockScaleMoEInputs,
    FP4BlockScaleMoERunner,
    MxE4m3MxE2m1BlockScaleMoERunner,
    prepare_dummy_topk_and_hook,
)

#: TMA descriptors need at least this much backing storage per tensor.
_MIN_TMA_BYTES = 128 * 1024

#: Version tag of the tactic space, part of the autotuner cache key.
_TACTIC_SPACE_VERSION = "prims_ts_moe_v1"


class _GemmPlan(NamedTuple):
    """The FC1/FC2 configs one tactic resolves to."""

    tile_n: int
    fc1_cfg: object
    fc2_cfg: object


def _input_format(hidden_states: torch.Tensor) -> str:
    """The MoE format the activation storage selects."""
    if hidden_states.dtype == torch.bfloat16:
        return "bf16"
    if hidden_states.dtype == torch.uint8:
        return "nvfp4"
    if hidden_states.dtype == torch.float8_e4m3fn:
        return "mxfp4_mxfp8"
    raise ValueError(f"PrimsTS MoE has no kernel for {hidden_states.dtype} activations")


def _tactic_functions(fmt: str) -> tuple[Callable, Callable]:
    """``(enumerate tactics, map one tactic to a GEMM pair)`` for ``fmt``."""
    from ..kernels.prims_ts.moe import config_mapper

    return {
        "bf16": (
            config_mapper.valid_prims_ts_bf16_moe_tactics,
            config_mapper.map_trtllm_bf16_moe_tactic,
        ),
        "nvfp4": (
            config_mapper.valid_prims_ts_nvfp4_moe_tactics,
            config_mapper.map_trtllm_nvfp4_moe_tactic,
        ),
        "mxfp4_mxfp8": (
            config_mapper.valid_prims_ts_mxfp4_mxfp8_moe_tactics,
            config_mapper.map_trtllm_mxfp4_mxfp8_moe_tactic,
        ),
    }[fmt]


def _weight_geometry(fmt: str, gemm2_weights: torch.Tensor) -> tuple[int, int]:
    """``(weight layout, per-rank intermediate size)`` of the TRTLLM-Gen weights.

    BF16 weights are BlockMajorK ``[E, K / block, N, block]`` (4-D) or MajorK;
    the packed FP4 formats are MajorK with two values per byte along K.
    """
    from flashinfer.tllm_enums import WeightLayout

    if fmt == "bf16" and gemm2_weights.ndim == 4:
        return int(WeightLayout.BlockMajorK), gemm2_weights.shape[1] * gemm2_weights.shape[-1]
    if fmt == "bf16":
        return int(WeightLayout.MajorK), gemm2_weights.shape[-1]
    return int(WeightLayout.MajorK), gemm2_weights.shape[-1] * 2


def _token_capacity(num_tokens: int, top_k: int, local_experts: int, tile_n: int) -> int:
    """Rows the routed GEMMs may address for ``tile_n``.

    The same bound as the C++ ``getMaxPermutedPaddedCount`` that sizes the
    ``moe_sort`` tables: one tile per expert that receives a token, plus full
    tiles for the remaining expanded tokens. Unlike FlashInfer's Python helper
    it accepts ``top_k`` larger than the local expert count, which expert
    parallelism produces.
    """
    expanded = num_tokens * top_k
    filled = min(local_experts, expanded)
    return (filled + (expanded - filled) // tile_n) * tile_n


@lru_cache(maxsize=256)
def _gemm_plan(
    fmt: str,
    tactic: tuple[int, ...],
    sm_arch: int,
    activation_type: int,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    num_experts: int,
    weight_layout: int,
    has_gemm1_alpha: bool,
    has_gemm1_beta: bool,
    has_gemm1_clamp_limit: bool,
    enable_pdl: bool,
) -> _GemmPlan:
    """Resolve and build the FC1/FC2 configs of ``tactic`` (``(-1, -1)``: heuristic)."""
    from ..kernels.prims_ts.moe.config_mapper import apply_weight_l2_policy

    _, map_tactic = _tactic_functions(fmt)
    pair = apply_weight_l2_policy(
        map_tactic(
            list(tactic),
            sm_arch=sm_arch,
            activation_type=activation_type,
            num_tokens=num_tokens,
            top_k=top_k,
            num_local_experts=num_local_experts,
            num_experts=num_experts,
            weight_layout=weight_layout,
            has_gemm1_alpha=has_gemm1_alpha,
            has_gemm1_beta=has_gemm1_beta,
            has_gemm1_clamp_limit=has_gemm1_clamp_limit,
            enable_pdl=enable_pdl,
        )
    )
    return _GemmPlan(pair.tile_n, pair.fc1.cfg.build(), pair.fc2.cfg.build())


@lru_cache(maxsize=32)
def _tuning_config(fmt: str, ep_size: int, tune_max_num_tokens: int, use_dp: bool) -> TuningConfig:
    """The autotuner profile of the op, over the FP4 runner's input ordering.

    The op passes its tensors in ``FP4BlockScaleMoEInputs`` order. The NVFP4
    runner's profile applies as is; the others take the MXFP4 runner's, whose
    inputs lack the three FP4 global scales, so its routing tensors move from
    positions 13/14 to 16/17.
    """
    if fmt == "nvfp4":
        return FP4BlockScaleMoERunner.get_tuning_config(ep_size, tune_max_num_tokens, use_dp)
    config = MxE4m3MxE2m1BlockScaleMoERunner.get_tuning_config(ep_size, tune_max_num_tokens, use_dp)
    return replace(
        config,
        constraint_specs=tuple(
            replace(spec, input_idx=spec.input_idx + 3) if spec.input_idx in (13, 14) else spec
            for spec in config.constraint_specs
        ),
    )


def _with_routing_profile_hook(
    config: TuningConfig, weights: torch.Tensor, ids: torch.Tensor
) -> TuningConfig:
    """Regenerate routes when a profile changes the warmup token count."""
    hook = config.inputs_pre_hook
    if hook is None:
        return config

    def prepare(inputs: list) -> list:
        # Shape constraints repeat integer tensors before the pre-hook runs.
        # Restore the unexpanded dummy so the shared routing hook can detect
        # the token-count change and generate a representative distribution.
        inputs[-2:] = [weights, ids]
        return hook(inputs)

    return replace(config, inputs_pre_hook=prepare)


def _run_prims_ts_moe(
    args: FP4BlockScaleMoEInputs,
    *,
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    output: torch.Tensor,
    tactic: tuple[int, ...],
    enable_pdl: bool,
) -> None:
    """Sort the routes, run FC1 and FC2, and finalize into ``output``."""
    import cuda.bindings.driver as cuda_driver

    from ..kernels.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple
    from ..kernels.prims_ts.moe.compile_cache import get_compiled_gemm, stable_config_hash
    from ..kernels.prims_ts.moe.tensor_adapter import (
        build_bf16_launch_io,
        build_mxfp4_mxfp8_launch_io,
        build_nvfp4_launch_io,
    )

    hidden_states = args.hidden_states
    topk_ids, topk_weights = args.topk_ids, args.topk_weights
    num_tokens, top_k = topk_ids.shape
    hidden_size = output.shape[1]
    local_experts = args.gemm1_weights.shape[0]
    fmt = _input_format(hidden_states)
    weight_layout, intermediate_size = _weight_geometry(fmt, args.gemm2_weights)
    plan = _gemm_plan(
        fmt,
        tactic,
        get_sm_version(),
        activation_type,
        num_tokens,
        top_k,
        local_experts,
        num_experts,
        weight_layout,
        args.gemm1_alpha is not None,
        args.gemm1_beta is not None,
        args.gemm1_clamp_limit is not None,
        enable_pdl,
    )
    capacity = _token_capacity(num_tokens, top_k, local_experts, plan.tile_n)

    # The kernels gather activation rows through the permuted-row -> token map.
    tile_idx, mn_limit, expanded_to_permuted, route_map, total_padded, num_tiles = (
        torch.ops.trtllm.moe_sort(
            topk_ids,
            topk_weights,
            num_experts,
            top_k,
            local_expert_offset,
            local_experts,
            plan.tile_n,
            True,
        )
    )

    device = hidden_states.device
    if fmt == "bf16":
        fc1_out_bytes = intermediate_size * 2
        fc1_dtype, fc1_cols = torch.bfloat16, intermediate_size
    elif fmt == "nvfp4":
        fc1_out_bytes = intermediate_size // 2
        fc1_dtype, fc1_cols = torch.uint8, fc1_out_bytes
    else:
        fc1_out_bytes = intermediate_size
        fc1_dtype, fc1_cols = torch.float8_e4m3fn, intermediate_size
    fc1_rows = max(capacity, ceil_div(_MIN_TMA_BYTES, fc1_out_bytes))
    gemm1_output = torch.empty((fc1_rows, fc1_cols), dtype=fc1_dtype, device=device)
    fc2_rows = max(capacity, ceil_div(_MIN_TMA_BYTES, hidden_size * 2))
    gemm2_output = torch.empty((fc2_rows, hidden_size), dtype=torch.bfloat16, device=device)

    io_kwargs = dict(
        hidden_states=hidden_states,
        gemm1_weights=args.gemm1_weights,
        gemm2_weights=args.gemm2_weights,
        gemm1_alpha=args.gemm1_alpha,
        gemm1_beta=args.gemm1_beta,
        gemm1_clamp_limit=args.gemm1_clamp_limit,
        gemm1_output=gemm1_output,
        gemm2_output=gemm2_output,
        tile_idx=tile_idx,
        mn_limit=mn_limit,
        route_map=route_map,
        num_non_exiting_ctas=num_tiles,
        total_num_padded_tokens=total_padded,
        routed_token_capacity=capacity,
        activation_type=activation_type,
        num_experts=local_experts,
        num_tokens=num_tokens,
        # Bounds the rows a launch reserves: a token reaches at most this many
        # of the local experts.
        top_k=min(top_k, local_experts),
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )
    if fmt == "bf16":
        build_io = build_bf16_launch_io
    else:
        # FC1's quantized output and its block scales feed FC2.
        sf_group = 16 if fmt == "nvfp4" else 32
        sf_bytes = ceil_div(fc1_rows, 128) * 128 * ceil_div(intermediate_size // sf_group, 4) * 4
        gemm1_output_scale = torch.empty((sf_bytes,), dtype=torch.uint8, device=device)
        hidden_states_scale = args.hidden_states_scale
        output_scales = (
            args.output1_scale_scalar,
            args.output1_scale_gate_scalar,
            args.output2_scale_scalar,
        )
        if fmt == "nvfp4":
            build_io = build_nvfp4_launch_io
            hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn)
            gemm1_output_scale = gemm1_output_scale.view(torch.float8_e4m3fn)
            if any(scale is None for scale in output_scales):
                raise ValueError("PrimsTS NVFP4 MoE requires the FC1 and FC2 global scales")
        else:
            build_io = build_mxfp4_mxfp8_launch_io
            # MXFP4 has no global scales. The launcher still takes pointer
            # operands for them, which the MXFP4 configs never read; an empty
            # tensor provides a null pointer without a fill kernel.
            unused = hidden_states.new_empty((0,), dtype=torch.float32)
            output_scales = (unused, unused, unused)
        io_kwargs.update(
            hidden_states_scale=hidden_states_scale,
            gemm1_weights_scale=args.gemm1_weights_scale,
            gemm2_weights_scale=args.gemm2_weights_scale,
            gemm1_output_scale=gemm1_output_scale,
            output1_scale_scalar=output_scales[0],
            output1_scale_gate_scalar=output_scales[1],
            output2_scale_scalar=output_scales[2],
        )

    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    for fc, cfg in (("fc1", plan.fc1_cfg), ("fc2", plan.fc2_cfg)):
        io = build_io(fc=fc, cfg=cfg, **io_kwargs)
        kernel = get_compiled_gemm(stable_config_hash(io["cfg"]), f"{fmt}_{fc}", io, stream)
        kernel(*_launch_arg_tuple(io, stream))

    torch.ops.trtllm.moe_unpermute_inplace(gemm2_output, output, expanded_to_permuted, topk_weights)


class PrimsTSMoERunner(TunableRunner):
    """Tune the vendored FC1/FC2 GEMM pairs through the TRT-LLM autotuner.

    Args:
        num_experts (int): Global expert (or EPLB slot) count the routing ids
            index.
        local_expert_offset (int): First global expert id this rank owns.
        activation_type (int): ``flashinfer.tllm_enums.ActivationType`` value of
            the fused FC1 activation.
        enable_pdl (bool): Whether the GEMMs use programmatic dependent launch.
    """

    def __init__(
        self,
        num_experts: int,
        local_expert_offset: int,
        activation_type: int,
        enable_pdl: bool,
    ) -> None:
        self.num_experts = num_experts
        self.local_expert_offset = local_expert_offset
        self.activation_type = activation_type
        self.enable_pdl = enable_pdl

    def unique_id(self) -> tuple:
        return (
            _TACTIC_SPACE_VERSION,
            self.num_experts,
            self.local_expert_offset,
            self.activation_type,
            self.enable_pdl,
        )

    def get_valid_tactics(self, inputs: list, profile: object, **kwargs: object) -> list:
        args = FP4BlockScaleMoEInputs(*inputs)
        fmt = _input_format(args.hidden_states)
        weight_layout, _ = _weight_geometry(fmt, args.gemm2_weights)
        enumerate_tactics, _ = _tactic_functions(fmt)
        return enumerate_tactics(
            sm_arch=get_sm_version(),
            activation_type=self.activation_type,
            num_tokens=args.hidden_states.shape[0],
            top_k=args.topk_ids.shape[1],
            num_local_experts=args.gemm1_weights.shape[0],
            num_experts=self.num_experts,
            weight_layout=weight_layout,
            has_gemm1_alpha=args.gemm1_alpha is not None,
            has_gemm1_beta=args.gemm1_beta is not None,
            has_gemm1_clamp_limit=args.gemm1_clamp_limit is not None,
            enable_pdl=self.enable_pdl,
        )

    def forward(
        self,
        inputs: list,
        *,
        tactic: object = -1,
        do_preparation: bool = False,
        output: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        del do_preparation
        args = FP4BlockScaleMoEInputs(*inputs)
        if output is None:
            # Profiling calls own their output. The hidden size is FC2's N,
            # the third axis of 4-D BlockMajorK weights and the second otherwise.
            w2 = args.gemm2_weights
            hidden_size = w2.shape[2] if w2.ndim == 4 else w2.shape[1]
            output = args.hidden_states.new_empty(
                (args.hidden_states.shape[0], hidden_size), dtype=torch.bfloat16
            )
        _run_prims_ts_moe(
            args,
            num_experts=self.num_experts,
            local_expert_offset=self.local_expert_offset,
            activation_type=self.activation_type,
            output=output,
            tactic=(-1, -1) if tactic == -1 else tuple(tactic),
            enable_pdl=self.enable_pdl,
        )


@torch.library.custom_op("trtllm::prims_ts_moe", mutates_args=("output",))
def prims_ts_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    output: torch.Tensor,
    routing_method_type: int,
    n_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    routed_scaling_factor: Optional[float] = None,
    enable_pdl: bool = False,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
) -> None:
    """Run one PrimsTS MoE over precomputed routing and finalize into ``output``.

    Args:
        hidden_states (torch.Tensor): Activations ``[num_tokens, K]``: BF16, NVFP4
            packed in ``uint8`` (K = hidden / 2), or MXFP8 ``float8_e4m3fn``.
        hidden_states_scale (torch.Tensor | None): Linear block scales of the
            quantized activations; ``None`` for BF16.
        gemm1_weights (torch.Tensor): FC1 weights in the TRTLLM-Gen shuffled
            layout, ``[local_experts, ...]``.
        gemm1_weights_scale (torch.Tensor | None): FC1 block scales.
        gemm2_weights (torch.Tensor): FC2 weights in the TRTLLM-Gen layout.
        gemm2_weights_scale (torch.Tensor | None): FC2 block scales.
        gemm1_alpha (torch.Tensor | None): Per-expert FP32 SwiGLU alpha or SiTU
            gate soft cap.
        gemm1_beta (torch.Tensor | None): Per-expert FP32 SwiGLU beta or SiTU
            linear soft cap.
        gemm1_clamp_limit (torch.Tensor | None): Per-expert FP32 SwiGLU clamp.
        output1_scale_scalar (torch.Tensor | None): NVFP4 FC1 output scale.
        output1_scale_gate_scalar (torch.Tensor | None): NVFP4 FC1 gate scale.
        output2_scale_scalar (torch.Tensor | None): NVFP4 FC2 output scale.
        topk_ids (torch.Tensor): Selected expert ids ``[num_tokens, top_k]``, int32.
        topk_weights (torch.Tensor): Routing weights ``[num_tokens, top_k]``, BF16.
        num_experts (int): Global expert (or EPLB slot) count ``topk_ids`` index.
        local_expert_offset (int): First global expert id this rank owns.
        activation_type (int): ``flashinfer.tllm_enums.ActivationType`` value.
        output (torch.Tensor): Finalized output ``[num_tokens, hidden]``, BF16;
            written in place.
        routing_method_type (int): Routing the tuning profile imitates.
        n_group (int | None): Expert groups of grouped routing, for tuning.
        topk_group (int | None): Groups each token selects, for tuning.
        routed_scaling_factor (float | None): Routing scale, for tuning.
        enable_pdl (bool): Use programmatic dependent launch.
        tune_max_num_tokens (int): Largest token count the autotuner profiles.
        use_dp (bool): Attention data parallelism, for the tuning profile.
    """
    if hidden_states.shape[0] == 0:
        return
    if hidden_states_scale is not None:
        # Flat, the shape the tuning profile gives the scales when it varies
        # the token count.
        hidden_states_scale = hidden_states_scale.reshape(-1)

    local_experts = gemm1_weights.shape[0]
    top_k = topk_ids.shape[1]
    tuning_config = _tuning_config(
        _input_format(hidden_states), num_experts // local_experts, tune_max_num_tokens, use_dp
    )
    tune_logits, tune_weights, tune_ids, tuning_config = prepare_dummy_topk_and_hook(
        topk_weights,
        topk_ids,
        hidden_states,
        None,
        routing_method_type,
        tuning_config,
        top_k,
        num_experts,
        local_experts,
        n_group,
        topk_group,
        routed_scaling_factor,
        local_expert_offset=local_expert_offset,
        use_dp=use_dp,
    )
    tuning_config = _with_routing_profile_hook(tuning_config, tune_weights, tune_ids)
    inputs = [
        tune_logits,
        None,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        None,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        None,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        tune_weights,
        tune_ids,
    ]
    runner = PrimsTSMoERunner(num_experts, local_expert_offset, activation_type, enable_pdl)
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm::prims_ts_moe", [runner], tuning_config, inputs
    )
    inputs[0] = None
    inputs[-2:] = [topk_weights, topk_ids]
    runner(inputs, tactic=tactic, output=output)


@prims_ts_moe.register_fake
def _prims_ts_moe_fake(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    output: torch.Tensor,
    routing_method_type: int,
    n_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    routed_scaling_factor: Optional[float] = None,
    enable_pdl: bool = False,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
) -> None:
    return None
