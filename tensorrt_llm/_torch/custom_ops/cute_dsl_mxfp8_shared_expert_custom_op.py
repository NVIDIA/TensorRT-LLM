# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 MXFP8 fused FC12 custom op for Kimi K3 shared experts."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.math_utils import pad_up

from ..autotuner import AutoTuner
from ..cute_dsl_utils import sm_reserve
from ..flashinfer_utils import get_env_enable_pdl
from ..modules.multi_stream_utils import do_multi_stream
from ..moe.kernels.cutedsl_megamoe.kernel_src.rubin.inference.mxfp8_fc12.fwd_glu.glu_mxfp8_fc12_kernel import (  # noqa: E501
    Sm107Mxfp8GluFc12Kernel,
)
from .cute_dsl_custom_ops import _with_input_cuda_device

# FC1 weights interleave gate and up in blocks of this many rows; the fused
# kernel and the unfused SiTU both read that layout.
GATE_UP_INTERLEAVE = 32
_SF_VEC_SIZE = 32
_SF_PADDING_BLOCK = 128
_TOKEN_PADDING_BLOCK = 128
_WORKSPACE_BUFFER_NAME = "sm107_mxfp8_shared_expert_fc12_workspace"
# Rubin crossover of fused FC12 vs. two MXFP8 GEMMs: I=6144 wins from M=1024
# and I=3072 from M=2048, i.e. once tokens * intermediate reaches 1024 * 6144.
_FUSED_FC12_MIN_TOKENS_X_INTERMEDIATE = 1024 * 6144
# Largest M the two GEMMs are tuned for. It covers the eager range below the
# fused crossover and decode CUDA graphs; larger M reuses the 2048 tactic.
_TWO_GEMM_TUNE_MAX_NUM_TOKENS = 2048

_TOPK_SCORES_CACHE: dict[torch.device, torch.Tensor] = {}
# Byte alignment of each kernel operand: 9 data/scale tensors, then the
# int32/float32 top-k scores, done counters and row offsets.
_OPERAND_ALIGNMENTS = (16,) * 9 + (4,) * 3


@dataclasses.dataclass
class _CompiledPlan:
    kernel: Sm107Mxfp8GluFc12Kernel
    max_active_clusters: int
    compiled: Optional[Callable] = None


_COMPILED_PLAN_CACHE: dict[tuple, _CompiledPlan] = {}


def _to_cute(tensor: torch.Tensor, assumed_align: int = 16) -> cute.Tensor:
    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    return cute_tensor.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


def use_fused_fc12(num_tokens: int, intermediate_size: int) -> bool:
    """Return whether this call runs the fused FC12 kernel instead of two MXFP8 GEMMs.

    The fused kernel is persistent with one CTA per resident cluster, and its
    FC2 tiles wait for FC1 tiles of other CTAs. It stays out of multi-stream
    execution and CUDA graph capture, where the shared expert can run beside
    the routed MoE on another stream, and out of runs with expert or dense
    DWDP, which prefetch the next layers' weights on a copy stream, possibly
    with SM gather kernels. In all of these not every CTA is guaranteed to
    become resident.
    """
    from ..modules.dwdp import fabric_gather
    from ..modules.dwdp.dense.manager import get_global_dense_dwdp_manager

    return (
        num_tokens * intermediate_size >= _FUSED_FC12_MIN_TOKENS_X_INTERMEDIATE
        and not do_multi_stream()
        and not torch.cuda.is_current_stream_capturing()
        and not fabric_gather.is_enabled()
        and get_global_dense_dwdp_manager() is None
    )


def _get_aux_tensors(device: torch.device, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit top-k scores and the single-expert row offset for ``num_tokens``."""
    padded_tokens = pad_up(num_tokens, _TOKEN_PADDING_BLOCK)
    topk_scores = _TOPK_SCORES_CACHE.get(device)
    if topk_scores is None or topk_scores.numel() < padded_tokens:
        topk_scores = torch.ones((padded_tokens,), dtype=torch.float32, device=device)
        _TOPK_SCORES_CACHE[device] = topk_scores
    offsets = torch.full((1,), num_tokens, dtype=torch.int32, device=device)
    return topk_scores[:padded_tokens], offsets


def _expected_swizzled_scale_size(rows: int, columns: int) -> int:
    return pad_up(rows, _SF_PADDING_BLOCK) * pad_up(columns // _SF_VEC_SIZE, 4)


def _validate_inputs(
    input: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
) -> tuple[int, int, int]:
    if get_sm_version() != 107:
        raise ValueError(
            f"SM107 MXFP8 shared-expert FC12 requires SM107, got SM{get_sm_version()}."
        )
    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must be bfloat16, got {input.dtype}.")
    if input.dim() < 2 or input.shape[-1] <= 0:
        raise ValueError(
            f"input must have rank >= 2 and a nonzero hidden size, got {tuple(input.shape)}."
        )
    if fc1_weight.dtype != torch.float8_e4m3fn or fc2_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("FC12 weights must be MXFP8 E4M3 tensors.")
    if fc1_weight_scale.dtype != torch.uint8 or fc2_weight_scale.dtype != torch.uint8:
        raise ValueError("FC12 weight scales must be E8M0 uint8 tensors.")
    tensors = (fc1_weight, fc1_weight_scale, fc2_weight, fc2_weight_scale)
    if any(tensor.device != input.device for tensor in tensors):
        raise ValueError("input, weights, and weight scales must share a device.")
    if not input.is_cuda:
        raise ValueError("SM107 MXFP8 shared-expert FC12 requires CUDA tensors.")
    if fc1_weight.dim() != 2 or fc2_weight.dim() != 2:
        raise ValueError(
            f"FC12 weights must be 2-D, got {tuple(fc1_weight.shape)} and "
            f"{tuple(fc2_weight.shape)}."
        )

    hidden_size = input.shape[-1]
    gate_up_size, fc1_hidden_size = fc1_weight.shape
    if gate_up_size % 2 != 0:
        raise ValueError(f"FC1 gate/up dimension must be even, got {gate_up_size}.")
    intermediate_size = gate_up_size // 2
    if fc1_hidden_size != hidden_size or fc2_weight.shape != (hidden_size, intermediate_size):
        raise ValueError(
            "FC12 weight shapes must be [2I, H] and [H, I]; got "
            f"{tuple(fc1_weight.shape)} and {tuple(fc2_weight.shape)} for "
            f"H={hidden_size}."
        )
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(
            "SM107 fused FC12 requires hidden and intermediate dimensions "
            f"divisible by 128, got H={hidden_size}, I={intermediate_size}."
        )
    if not fc1_weight.is_contiguous() or not fc2_weight.is_contiguous():
        raise ValueError("FC12 physical weight buffers must be contiguous.")

    expected_fc1_scale_size = _expected_swizzled_scale_size(gate_up_size, hidden_size)
    expected_fc2_scale_size = _expected_swizzled_scale_size(hidden_size, intermediate_size)
    if fc1_weight_scale.numel() != expected_fc1_scale_size:
        raise ValueError(
            f"FC1 scale buffer has {fc1_weight_scale.numel()} elements; "
            f"expected {expected_fc1_scale_size}."
        )
    if fc2_weight_scale.numel() != expected_fc2_scale_size:
        raise ValueError(
            f"FC2 scale buffer has {fc2_weight_scale.numel()} elements; "
            f"expected {expected_fc2_scale_size}."
        )

    num_tokens = input.numel() // hidden_size
    return num_tokens, hidden_size, intermediate_size


def _build_plan(
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> _CompiledPlan:
    # Prefill-sized tactic: the op is dispatched only above the fused crossover.
    cluster_shape_mnk = (2, 1, 1)
    resident_clusters = int(
        cutlass.utils.HardwareInfo(device_id=device.index).get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        )
    )
    kernel = Sm107Mxfp8GluFc12Kernel(
        mma_tiler_mnk=(512, 256, 128),
        cluster_shape_mnk=cluster_shape_mnk,
        use_2cta_instrs=True,
        group_hint=resident_clusters,
        token_padding_block=_TOKEN_PADDING_BLOCK,
        sf_padding_block=_SF_PADDING_BLOCK,
        load_balance_mode="static",
        static_expert_shape=(1, 2 * intermediate_size, hidden_size),
        ab_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=_SF_VEC_SIZE,
        apply_topk_in_fc1=False,
        generate_c=False,
        use_stg_fc1=True,
        act_func="situ",
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        combine_format=None,
        fc2_sf_blocked=False,
        one_shot_sched=False,
        use_pdl=False,
    )
    return _CompiledPlan(kernel=kernel, max_active_clusters=resident_clusters)


def _get_plan(
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> _CompiledPlan:
    # The persistent grid needs every cluster resident; a reserve set after the
    # first call (DWDP) must size a new grid instead of reusing this one.
    key = (
        device.index,
        hidden_size,
        intermediate_size,
        float(situ_beta),
        float(situ_linear_beta),
        sm_reserve(),
    )
    plan = _COMPILED_PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_plan(device, hidden_size, intermediate_size, situ_beta, situ_linear_beta)
        _COMPILED_PLAN_CACHE[key] = plan
    return plan


def _workspace_views(
    plan: _CompiledPlan,
    padded_tokens: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sf_rows = padded_tokens + _SF_PADDING_BLOCK
    sf_columns = pad_up(intermediate_size // _SF_VEC_SIZE, 4)
    fc1_output_bytes = padded_tokens * intermediate_size
    fc1_output_sf_bytes = sf_rows * sf_columns
    done_slots = plan.kernel.fc1_done_counter_slots(padded_tokens, 1)
    required_bytes = pad_up(fc1_output_bytes + fc1_output_sf_bytes + done_slots * 4, 128)
    workspace = get_memory_buffers().get_buffer(
        [required_bytes],
        torch.uint8,
        f"{_WORKSPACE_BUFFER_NAME}:{device.index}",
        reserve_buffer=False,
    )
    if workspace.device != device:
        raise RuntimeError(f"FC12 workspace is on {workspace.device}, expected {device}.")

    offset = 0
    fc1_output = (
        workspace[offset : offset + fc1_output_bytes]
        .view(torch.float8_e4m3fn)
        .view(padded_tokens, intermediate_size)
    )
    offset += fc1_output_bytes
    fc1_output_sf = (
        workspace[offset : offset + fc1_output_sf_bytes]
        .view(torch.float8_e8m0fnu)
        .view(sf_rows, sf_columns)
    )
    offset += fc1_output_sf_bytes
    fc1_done_counter = workspace[offset : offset + done_slots * 4].view(torch.int32)
    fc1_done_counter.zero_()
    return fc1_output, fc1_output_sf, fc1_done_counter


@torch.library.custom_op(
    "trtllm::cute_dsl_mxfp8_shared_expert_rubin",
    mutates_args=(),
    device_types="cuda",
)
@_with_input_cuda_device
def cute_dsl_mxfp8_shared_expert_rubin(
    input: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """Run the SM107 MXFP8 fused FC1 + SiTU + FC2 kernel on one expert.

    Args:
        input: BF16 activations, ``[..., H]``.
        fc1_weight: E4M3 ``[2I, H]`` with gate and up interleaved in blocks of
            ``GATE_UP_INTERLEAVE`` rows, as ``Mxfp8SharedExpertMLP`` packs them.
            A plain ``[gate | up]`` layout passes validation but gives wrong
            results.
        fc1_weight_scale: UE8M0 (uint8) 1x32 block scales of ``fc1_weight`` in
            the swizzled R128c4 layout from ``mxfp8_quantize(..., True)``.
        fc2_weight: E4M3 ``[H, I]``.
        fc2_weight_scale: UE8M0 1x32 scales of ``fc2_weight``, same layout.
        situ_beta: SiTU ``beta``.
        situ_linear_beta: SiTU ``linear_beta``.

    Returns:
        BF16 ``[..., H]``; empty for zero tokens. Raises under CUDA graph
        capture, where ``cute_dsl_mxfp8_gated_mlp_rubin`` runs two GEMMs instead.
    """
    num_tokens, hidden_size, intermediate_size = _validate_inputs(
        input, fc1_weight, fc1_weight_scale, fc2_weight, fc2_weight_scale
    )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SM107 MXFP8 fused FC12 cannot run in CUDA graph capture; "
            "cute_dsl_mxfp8_gated_mlp_rubin runs the two-GEMM path there."
        )
    if num_tokens == 0:
        return input.new_empty(input.shape, dtype=torch.bfloat16)
    padded_tokens = pad_up(num_tokens, _TOKEN_PADDING_BLOCK)
    input_2d = input.reshape(num_tokens, hidden_size).contiguous()
    if padded_tokens != num_tokens:
        padded_input = input_2d.new_empty((padded_tokens, hidden_size))
        padded_input[:num_tokens].copy_(input_2d)
        padded_input[num_tokens:].zero_()
    else:
        padded_input = input_2d
    activation, activation_scale = torch.ops.trtllm.mxfp8_quantize(
        padded_input, True, alignment=_SF_VEC_SIZE
    )
    activation_scale = activation_scale.view(torch.float8_e8m0fnu).view(
        padded_tokens,
        pad_up(hidden_size // _SF_VEC_SIZE, 4),
    )

    plan = _get_plan(input.device, hidden_size, intermediate_size, situ_beta, situ_linear_beta)
    fc1_output, fc1_output_scale, fc1_done_counter = _workspace_views(
        plan, padded_tokens, intermediate_size, input.device
    )
    output = torch.empty((padded_tokens, hidden_size), dtype=torch.bfloat16, device=input.device)
    topk_scores, offsets = _get_aux_tensors(input.device, num_tokens)

    # Positional order of Sm107Mxfp8GluFc12Kernel.__call__.
    operands = (
        activation,
        fc1_weight.view(1, 2 * intermediate_size, hidden_size).transpose(1, 2),
        activation_scale,
        fc1_weight_scale.view(torch.float8_e8m0fnu).view(1, -1),
        fc1_output,
        fc1_output_scale,
        fc2_weight.view(1, hidden_size, intermediate_size).transpose(1, 2),
        fc2_weight_scale.view(torch.float8_e8m0fnu).view(1, -1),
        output,
        topk_scores,
        fc1_done_counter,
        offsets,
    )
    if plan.compiled is None:
        # TVM-FFI takes torch tensors and the current stream at launch, which
        # avoids per-call DLPack conversion on the host.
        plan.compiled = cute.compile(
            plan.kernel,
            *(
                _to_cute(tensor, assumed_align)
                for tensor, assumed_align in zip(operands, _OPERAND_ALIGNMENTS)
            ),
            max_active_clusters=plan.max_active_clusters,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    plan.compiled(*operands)
    return output[:num_tokens].reshape(input.shape)


@torch.library.register_fake("trtllm::cute_dsl_mxfp8_shared_expert_rubin")
def _(
    input: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    return input.new_empty(input.shape, dtype=torch.bfloat16)


@torch.library.custom_op(
    "trtllm::cute_dsl_mxfp8_gated_mlp_rubin", mutates_args=(), device_types="cuda"
)
@_with_input_cuda_device
def cute_dsl_mxfp8_gated_mlp_rubin(
    input: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    allow_fused: bool = True,
) -> torch.Tensor:
    """Run a SiTU MLP on MXFP8 weights through fused FC12 or two MXFP8 GEMMs.

    Takes the same tensors as ``cute_dsl_mxfp8_shared_expert_rubin``. The
    fused kernel runs when ``allow_fused`` is set and ``use_fused_fc12`` holds;
    otherwise two MXFP8 GEMMs run around a SiTU that reads the interleaved
    gate/up output, with activations quantized per 1x128 block into the
    weight-scale layout. Zero tokens return an empty tensor before any GEMM is
    chosen.
    """
    num_tokens, hidden_size, intermediate_size = _validate_inputs(
        input, fc1_weight, fc1_weight_scale, fc2_weight, fc2_weight_scale
    )
    if num_tokens == 0:
        return input.new_empty(input.shape)
    if allow_fused and use_fused_fc12(num_tokens, intermediate_size):
        if AutoTuner.get().is_tuning_mode:
            # Smaller steps and CUDA graphs run the two GEMMs; tune them for
            # every bucket up to this M as well.
            _unfused_mxfp8_gated_mlp(
                input,
                fc1_weight,
                fc1_weight_scale,
                fc2_weight,
                fc2_weight_scale,
                situ_beta,
                situ_linear_beta,
            )
        return cute_dsl_mxfp8_shared_expert_rubin(
            input,
            fc1_weight,
            fc1_weight_scale,
            fc2_weight,
            fc2_weight_scale,
            situ_beta,
            situ_linear_beta,
        )
    return _unfused_mxfp8_gated_mlp(
        input,
        fc1_weight,
        fc1_weight_scale,
        fc2_weight,
        fc2_weight_scale,
        situ_beta,
        situ_linear_beta,
    )


def _unfused_mxfp8_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    from .cute_dsl_custom_ops import CuteDSLKimiK3MXFP8RubinLinear, _get_mxfp8_gemm_alpha

    runner = CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    inputs = [input, weight, input_scale, weight_scale, _get_mxfp8_gemm_alpha(input.device)]
    tuning_config = dataclasses.replace(
        runner.tuning_config, tune_max_num_tokens=_TWO_GEMM_TUNE_MAX_NUM_TOKENS
    )
    _, tactic = AutoTuner.get().choose_one(
        "trtllm::mxfp8_shared_expert_gemm_rubin", [runner], tuning_config, inputs
    )
    return runner(inputs, tactic=tactic)


def _unfused_mxfp8_gated_mlp(
    input: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    from ..modules.situ import situ_and_mul_kernel

    hidden_size, intermediate_size = fc2_weight.shape
    num_tokens = input.numel() // hidden_size
    input_2d = input.reshape(num_tokens, hidden_size).contiguous()
    # Activations take one UE8M0 scale per 1x128 block, written into the same
    # K32 R128c4 layout as the 1x32 weight scales.
    activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(input_2d)
    gate_up = _unfused_mxfp8_gemm(activation, fc1_weight, activation_scale, fc1_weight_scale)
    hidden = input.new_empty((num_tokens, intermediate_size))
    launch_with_pdl = get_env_enable_pdl()
    situ_and_mul_kernel[(num_tokens, (intermediate_size + 1023) // 1024)](
        hidden,
        hidden.stride(0),
        gate_up,
        gate_up.stride(0),
        intermediate_size,
        situ_beta,
        situ_linear_beta,
        BLOCK_SIZE=1024,
        HAS_LINEAR_BETA=True,
        LAUNCH_WITH_PDL=launch_with_pdl,
        GATE_UP_INTERLEAVE=GATE_UP_INTERLEAVE,
        launch_pdl=launch_with_pdl,
    )
    activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(hidden)
    output = _unfused_mxfp8_gemm(activation, fc2_weight, activation_scale, fc2_weight_scale)
    return output.reshape(input.shape)


@torch.library.register_fake("trtllm::cute_dsl_mxfp8_gated_mlp_rubin")
def _(
    input: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    allow_fused: bool = True,
) -> torch.Tensor:
    return input.new_empty(input.shape, dtype=torch.bfloat16)
