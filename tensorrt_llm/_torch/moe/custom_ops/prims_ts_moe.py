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
"""PrimsTS MoE launches over TRTLLM routing and quantized weight layouts."""

from dataclasses import replace
from functools import lru_cache
from typing import Optional

import torch

from tensorrt_llm.math_utils import ceil_div

from ...autotuner import AutoTuner, TunableRunner
from ...custom_ops.trtllm_gen_custom_ops import (
    FP4BlockScaleMoEInputs,
    FP4BlockScaleMoERunner,
    MxE4m3MxE2m1BlockScaleMoERunner,
    prepare_dummy_topk_and_hook,
)


@lru_cache(maxsize=256)
def _gemm_pair(
    num_tokens,
    top_k,
    local_experts,
    activation_type,
    is_nvfp4,
    has_alpha,
    has_beta,
    has_clamp,
    tactic=(-1, -1),
    enable_pdl=False,
):
    from ..flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_mxfp4_mxfp8_moe_tactic,
        map_trtllm_nvfp4_moe_tactic,
    )

    mapper = map_trtllm_nvfp4_moe_tactic if is_nvfp4 else map_trtllm_mxfp4_mxfp8_moe_tactic
    pair = mapper(
        tactic,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=local_experts,
        activation_type=activation_type,
        has_gemm1_alpha=has_alpha,
        has_gemm1_beta=has_beta,
        has_gemm1_clamp_limit=has_clamp,
        enable_pdl=enable_pdl,
    )
    return pair


@lru_cache(maxsize=512)
def _materialize_gemm_config(options, in_hidden):
    """Cache static launch metadata while keeping tensor pointers dynamic."""
    from ..flashinfer.prims_ts.batched_gemm.batched_gemm_config import make_config
    from ..flashinfer.prims_ts.batched_gemm.batched_gemm_run import _runtime_config
    from ..flashinfer.prims_ts.moe.compile_cache import stable_config_hash

    config = _runtime_config(make_config(**dict(options)), in_hidden)
    # The adapter specializes its loop options for the concrete K. Reuse this
    # private normalized config and its hash without rebuilding either.
    return config, stable_config_hash(config)


def _workspace_rows(num_tokens, top_k, local_experts, tile_n, hidden_size):
    expanded = num_tokens * top_k
    filled = min(local_experts, expanded)
    max_tiles = filled + (expanded - filled) // tile_n
    capacity = max_tiles * tile_n
    # TMA descriptors require at least 128 KiB of backing allocation.
    return capacity, max(capacity, ceil_div(128 * 1024, hidden_size * 2))


@lru_cache(maxsize=32)
def _tuning_config(is_nvfp4, ep_size, tune_max_num_tokens, use_dp):
    config_cls = FP4BlockScaleMoERunner if is_nvfp4 else MxE4m3MxE2m1BlockScaleMoERunner
    config = config_cls.get_tuning_config(ep_size, tune_max_num_tokens, use_dp)
    if not is_nvfp4:
        # Use the FP4 runner's common input ordering, including its three
        # optional global scales. MXFP4's route tensors move from 13/14 to 16/17.
        config = replace(
            config,
            constraint_specs=tuple(
                replace(spec, input_idx=spec.input_idx + 3) if spec.input_idx in (13, 14) else spec
                for spec in config.constraint_specs
            ),
        )
    return config


def _with_routing_profile_hook(config, weights, ids):
    """Regenerate routes when a profile expands the warmup token count."""
    hook = config.inputs_pre_hook
    if hook is None:
        return config

    def prepare(inputs):
        # Shape constraints repeat integer tensors before the pre-hook runs.
        # Restore the unexpanded dummy so the shared routing hook can detect
        # the token-count change and generate a representative distribution.
        inputs[-2:] = [weights, ids]
        return hook(inputs)

    return replace(config, inputs_pre_hook=prepare)


def _run_prims_ts_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    do_finalize: bool,
    output: torch.Tensor,
    tactic=(-1, -1),
    enable_pdl=False,
    router_logits=None,
    routing_bias=None,
    top_k=0,
    routing_method_type=0,
    n_group=None,
    topk_group=None,
    routed_scaling_factor=None,
    use_hybrid_routing=False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    has_precomputed_routing = topk_ids is not None
    if topk_ids is not None:
        top_k = topk_ids.shape[1]
    hidden_size = gemm1_weights.shape[-1] * 2
    if num_tokens == 0:
        return (
            hidden_states.new_empty((0, hidden_size), dtype=torch.bfloat16),
            hidden_states.new_empty((0, top_k), dtype=torch.int32),
            hidden_states.new_empty((0, top_k) if topk_ids is None else (0,), dtype=torch.bfloat16),
        )

    from cuda.bindings import driver as cuda

    from ..flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple
    from ..flashinfer.prims_ts.moe.compile_cache import get_compiled_gemm
    from ..flashinfer.prims_ts.moe.tensor_adapter import (
        build_mxfp4_mxfp8_launch_io,
        build_nvfp4_launch_io,
    )

    local_experts = gemm1_weights.shape[0]
    intermediate_size = gemm2_weights.shape[-1] * 2
    is_nvfp4 = hidden_states.dtype == torch.uint8
    pair = _gemm_pair(
        num_tokens,
        top_k,
        local_experts,
        activation_type,
        is_nvfp4,
        gemm1_alpha is not None,
        gemm1_beta is not None,
        gemm1_clamp_limit is not None,
        tactic,
        enable_pdl,
    )
    tile_n = pair.tile_n
    capacity, _ = _workspace_rows(num_tokens, top_k, local_experts, tile_n, hidden_size)
    # The externally visible workspace shape must not depend on the autotuned
    # tactic. Every vendored FP4 tactic uses a token tile of at most 256.
    _, fc2_rows = _workspace_rows(num_tokens, top_k, local_experts, 256, hidden_size)
    gemm2_output = torch.empty(
        (fc2_rows, hidden_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    if use_hybrid_routing and not has_precomputed_routing and 1 < num_tokens < 1024:
        # K3's medium-batch native fused router is slower than noaux_tc plus
        # sorting. Select here so autotune profiles the same path as inference
        # without splitting its cache into logits and precomputed-route keys.
        topk_weights, topk_ids = torch.ops.trtllm.noaux_tc_op(
            router_logits, routing_bias, n_group, topk_group, top_k, routed_scaling_factor
        )
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.bfloat16)
    if topk_ids is None:
        (tile_idx, mn_limit, expanded_map, route_map, total_padded, num_tiles, fused_weights) = (
            torch.ops.trtllm.moe_topk_sort(
                router_logits,
                routing_bias,
                num_experts,
                top_k,
                n_group,
                topk_group,
                local_expert_offset,
                local_experts,
                routed_scaling_factor,
                tile_n,
                routing_method_type,
                True,
            )
        )
        topk_weights = fused_weights
    else:
        (tile_idx, mn_limit, expanded_map, route_map, total_padded, num_tiles) = (
            torch.ops.trtllm.moe_sort(
                topk_ids,
                topk_weights,
                num_experts,
                top_k,
                local_expert_offset,
                local_experts,
                tile_n,
                True,
            )
        )
        # Custom-op outputs must not alias the caller's precomputed weights.
        # The adapter already owns those weights and needs no copy here.
        fused_weights = (
            hidden_states.new_empty((0,), dtype=torch.bfloat16)
            if has_precomputed_routing
            else topk_weights
        )
    row_bytes = intermediate_size // 2 if is_nvfp4 else intermediate_size
    fc1_rows = max(capacity, ceil_div(128 * 1024, row_bytes))
    fc1_dtype = torch.uint8 if is_nvfp4 else torch.float8_e4m3fn
    gemm1_output = torch.empty((fc1_rows, row_bytes), dtype=fc1_dtype, device=hidden_states.device)
    sf_group = 16 if is_nvfp4 else 32
    sf_size = ceil_div(fc1_rows, 128) * 128 * ceil_div(intermediate_size // sf_group, 4) * 4
    gemm1_output_scale = torch.empty((sf_size,), dtype=torch.uint8, device=hidden_states.device)
    if is_nvfp4:
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn)
        gemm1_output_scale = gemm1_output_scale.view(torch.float8_e4m3fn)
        if any(
            scale is None
            for scale in (output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar)
        ):
            raise ValueError("NVFP4 requires FC1 and FC2 global scales.")
    else:
        # MXFP4 has no global scales. The vendored launcher still accepts
        # pointer operands for them, but the MXFP4 configs never read them.
        # An empty tensor provides a null pointer without a fill kernel.
        unused_scale = hidden_states.new_empty((0,), dtype=torch.float32)
        output1_scale_scalar = unused_scale
        output1_scale_gate_scalar = unused_scale
        output2_scale_scalar = unused_scale
    build_io = build_nvfp4_launch_io if is_nvfp4 else build_mxfp4_mxfp8_launch_io
    stream = cuda.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream)
    for fc, config in (("fc1", pair.fc1), ("fc2", pair.fc2)):
        prepared_config, config_hash = _materialize_gemm_config(
            tuple(config.cfg.kwargs.items()), hidden_size if fc == "fc1" else intermediate_size
        )
        io = build_io(
            fc=fc,
            cfg=prepared_config,
            cfg_is_normalized=True,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm1_output=gemm1_output,
            gemm1_output_scale=gemm1_output_scale,
            gemm2_output=gemm2_output,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=output2_scale_scalar,
            tile_idx=tile_idx,
            mn_limit=mn_limit,
            route_map=route_map,
            num_non_exiting_ctas=num_tiles,
            total_num_padded_tokens=total_padded,
            routed_token_capacity=capacity,
            activation_type=activation_type,
            num_experts=local_experts,
            num_tokens=num_tokens,
            top_k=top_k,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        compiled = get_compiled_gemm(config_hash, fc, io, stream)
        compiled(*_launch_arg_tuple(io, stream))
    if do_finalize:
        torch.ops.trtllm.moe_unpermute_inplace(gemm2_output, output, expanded_map, topk_weights)
    return gemm2_output, expanded_map, fused_weights


class PrimsTSMoERunner(TunableRunner):
    """Tune the vendored FC1/FC2 pairs through the shared TRTLLM autotuner."""

    def __init__(
        self,
        num_experts,
        local_expert_offset,
        activation_type,
        do_finalize,
        enable_pdl,
        use_dp,
        top_k=0,
        routing_method_type=0,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=None,
        use_hybrid_routing=False,
    ):
        self.num_experts = num_experts
        self.local_expert_offset = local_expert_offset
        self.activation_type = activation_type
        self.do_finalize = do_finalize
        self.enable_pdl = enable_pdl
        self.use_dp = use_dp
        self.top_k = top_k
        self.routing_method_type = routing_method_type
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.use_hybrid_routing = use_hybrid_routing

    def unique_id(self):
        return (
            "hybrid_routing_v5",
            self.num_experts,
            self.local_expert_offset,
            self.activation_type,
            self.do_finalize,
            self.enable_pdl,
            self.use_dp,
            self.top_k,
            self.routing_method_type,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.use_hybrid_routing,
        )

    def get_valid_tactics(self, inputs, profile, **kwargs):
        from ..flashinfer.prims_ts.moe.config_mapper import (
            valid_prims_ts_mxfp4_mxfp8_moe_tactics,
            valid_prims_ts_nvfp4_moe_tactics,
        )

        args = FP4BlockScaleMoEInputs(*inputs)
        enumerate_tactics = (
            valid_prims_ts_nvfp4_moe_tactics
            if args.hidden_states.dtype == torch.uint8
            else valid_prims_ts_mxfp4_mxfp8_moe_tactics
        )
        return enumerate_tactics(
            num_tokens=args.hidden_states.shape[0],
            top_k=self.top_k,
            num_local_experts=args.gemm1_weights.shape[0],
            activation_type=self.activation_type,
            has_gemm1_alpha=args.gemm1_alpha is not None,
            has_gemm1_beta=args.gemm1_beta is not None,
            has_gemm1_clamp_limit=args.gemm1_clamp_limit is not None,
            enable_pdl=self.enable_pdl,
        )

    def forward(self, inputs, *, tactic=-1, output=None):
        args = FP4BlockScaleMoEInputs(*inputs)
        if output is None:
            output = args.hidden_states.new_empty(
                (args.hidden_states.shape[0], args.gemm1_weights.shape[-1] * 2),
                dtype=torch.bfloat16,
            )
        return _run_prims_ts_moe(
            args.hidden_states,
            args.hidden_states_scale,
            args.gemm1_weights,
            args.gemm1_weights_scale,
            args.gemm2_weights,
            args.gemm2_weights_scale,
            args.gemm1_alpha,
            args.gemm1_beta,
            args.gemm1_clamp_limit,
            args.output1_scale_scalar,
            args.output1_scale_gate_scalar,
            args.output2_scale_scalar,
            args.topk_ids,
            args.topk_weights,
            self.num_experts,
            self.local_expert_offset,
            self.activation_type,
            self.do_finalize,
            output,
            (-1, -1) if tactic == -1 else tuple(tactic),
            self.enable_pdl,
            router_logits=args.routing_logits,
            routing_bias=args.routing_bias,
            top_k=self.top_k,
            routing_method_type=self.routing_method_type,
            n_group=self.n_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            use_hybrid_routing=self.use_hybrid_routing,
        )


@torch.library.custom_op("trtllm::prims_ts_moe", mutates_args=("output",))
def prims_ts_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    do_finalize: bool,
    output: torch.Tensor,
    enable_pdl: bool = False,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
    routing_method_type: int = 0,
    n_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    routed_scaling_factor: Optional[float] = None,
    routing_bias: Optional[torch.Tensor] = None,
    router_logits: Optional[torch.Tensor] = None,
    top_k: int = 0,
    use_hybrid_routing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (topk_ids is None) != (topk_weights is None):
        raise ValueError("Precomputed routing requires both indices and weights.")
    if topk_ids is not None:
        top_k = topk_ids.shape[1]
    elif router_logits is None or top_k <= 0:
        raise ValueError("Fused routing requires logits and a positive top_k.")
    if hidden_states.shape[0] == 0:
        return (
            hidden_states.new_empty((0, gemm1_weights.shape[-1] * 2), dtype=torch.bfloat16),
            hidden_states.new_empty((0, top_k), dtype=torch.int32),
            hidden_states.new_empty((0, top_k) if topk_ids is None else (0,), dtype=torch.bfloat16),
        )

    local_experts = gemm1_weights.shape[0]
    is_nvfp4 = hidden_states.dtype == torch.uint8
    tuning_config = _tuning_config(
        is_nvfp4, num_experts // local_experts, tune_max_num_tokens, use_dp
    )
    tune_logits, tune_weights, tune_ids, tuning_config = prepare_dummy_topk_and_hook(
        topk_weights,
        topk_ids,
        hidden_states,
        router_logits,
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
        routing_bias=routing_bias,
    )
    tuning_config = _with_routing_profile_hook(tuning_config, tune_weights, tune_ids)
    inputs = [
        tune_logits,
        routing_bias,
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
    runner = PrimsTSMoERunner(
        num_experts,
        local_expert_offset,
        activation_type,
        do_finalize,
        enable_pdl,
        use_dp,
        top_k,
        routing_method_type,
        n_group,
        topk_group,
        routed_scaling_factor,
        use_hybrid_routing,
    )
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm::prims_ts_moe", [runner], tuning_config, inputs
    )
    inputs[-2:] = [topk_weights, topk_ids]
    return runner(inputs, tactic=tactic, output=output)


@prims_ts_moe.register_fake
def _prims_ts_moe_fake(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    gemm1_alpha,
    gemm1_beta,
    gemm1_clamp_limit,
    output1_scale_scalar,
    output1_scale_gate_scalar,
    output2_scale_scalar,
    topk_ids,
    topk_weights,
    num_experts,
    local_expert_offset,
    activation_type,
    do_finalize,
    output,
    enable_pdl=False,
    tune_max_num_tokens=8192,
    use_dp=False,
    routing_method_type=0,
    n_group=None,
    topk_group=None,
    routed_scaling_factor=None,
    routing_bias=None,
    router_logits=None,
    top_k=0,
    use_hybrid_routing=False,
):
    num_tokens = hidden_states.shape[0]
    if topk_ids is not None:
        top_k = topk_ids.shape[1]
    hidden_size = gemm1_weights.shape[-1] * 2
    rows = 0
    if num_tokens:
        _, rows = _workspace_rows(num_tokens, top_k, gemm1_weights.shape[0], 256, hidden_size)
    return (
        hidden_states.new_empty((rows, hidden_size), dtype=torch.bfloat16),
        hidden_states.new_empty((num_tokens, top_k), dtype=torch.int32),
        hidden_states.new_empty(
            (num_tokens, top_k) if topk_ids is None else (0,), dtype=torch.bfloat16
        ),
    )
