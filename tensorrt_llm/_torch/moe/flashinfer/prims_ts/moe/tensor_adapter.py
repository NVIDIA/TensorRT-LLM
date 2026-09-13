# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapters from TRT-LLM MoE stage tensors to Prims-TS batched GEMM launch IO.
"""

from __future__ import annotations

from typing import Literal

import torch

from tensorrt_llm._torch.moe.flashinfer.tllm_enums import ActivationType, Fp8QuantizationType, WeightLayout


_expert_scale_ones: dict[tuple[torch.device, int], torch.Tensor] = {}


def _get_expert_scale_ones(num_experts: int, device: torch.device) -> torch.Tensor:
    """Return the immutable per-expert unit scales shared by MoE launches."""

    key = (torch.device(device), int(num_experts))
    scales = _expert_scale_ones.get(key)
    if scales is None:
        scales = torch.ones(key[1], dtype=torch.float32, device=key[0])
        _expert_scale_ones[key] = scales
    return scales


def _is_gated_activation(activation_type: int) -> bool:
    return ActivationType(int(activation_type)).is_gated


def _fc1_out_hidden(intermediate_size: int, activation_type: int) -> int:
    return (
        intermediate_size * 2
        if _is_gated_activation(activation_type)
        else intermediate_size
    )


def _check_fp8_scale_storage(
    name: str, tensor: torch.Tensor, *, min_numel: int | None = None
) -> None:
    valid_dtypes = [torch.float8_e4m3fn, torch.uint8]
    if hasattr(torch, "float8_e8m0fnu"):
        valid_dtypes.append(torch.float8_e8m0fnu)
    if tensor.dtype not in valid_dtypes:
        raise ValueError(f"{name} must be byte-sized FP8 scale storage")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if min_numel is not None and tensor.numel() < int(min_numel):
        raise ValueError(
            f"{name} scale storage is too small: "
            f"need at least {min_numel} elements, got {tensor.numel()}"
        )


def _validate_weight_storage(
    *,
    name: str,
    tensor: torch.Tensor,
    cfg,
    num_experts: int,
    out_hidden: int,
    in_hidden: int,
) -> None:
    if int(cfg.weight_layout) == int(WeightLayout.MajorK):
        weight_bits = int(in_hidden) * int(cfg.weight_dtype_tma_bits)
        element_bits = tensor.element_size() * 8
        if weight_bits % element_bits != 0:
            raise ValueError(
                f"{name} logical K storage is not integral for dtype "
                f"{tensor.dtype}: in_hidden={in_hidden}, "
                f"weight_bits={cfg.weight_dtype_tma_bits}"
            )
        expected = (
            int(num_experts),
            int(out_hidden),
            weight_bits // element_bits,
        )
        if tensor.ndim != 3 or tuple(int(dim) for dim in tensor.shape) != expected:
            raise ValueError(
                f"{name} has invalid MajorK shape {tuple(tensor.shape)}, "
                f"expected {expected}"
            )
        return
    if int(cfg.weight_layout) != int(WeightLayout.BlockMajorK):
        raise ValueError(f"Unsupported weight_layout={cfg.weight_layout}")
    if tensor.ndim != 4:
        block_bytes = int(cfg.block_major_k_bytes)
        raise ValueError(
            f"{name} must use BlockMajorK rank-4 storage "
            f"[expert, K_bytes / {block_bytes}, Mn, {block_bytes}-byte-block]"
        )

    weight_bits = int(in_hidden) * int(cfg.weight_dtype_tma_bits)
    if weight_bits % 8 != 0:
        raise ValueError(
            f"{name} logical K byte size is not integral: "
            f"in_hidden={in_hidden}, weight_bits={cfg.weight_dtype_tma_bits}"
        )
    weight_bytes = weight_bits // 8
    block_bytes = int(cfg.block_major_k_bytes)
    if weight_bytes % block_bytes != 0:
        raise ValueError(
            f"{name} BlockMajorK K storage must be divisible by {block_bytes} "
            f"bytes, got {weight_bytes}"
        )
    if block_bytes % tensor.element_size() != 0:
        raise ValueError(
            f"{name} dtype element size {tensor.element_size()} does not divide "
            f"the {block_bytes}-byte BlockMajorK block"
        )

    expected = (
        int(num_experts),
        weight_bytes // block_bytes,
        int(out_hidden),
        block_bytes // tensor.element_size(),
    )
    if tuple(int(dim) for dim in tensor.shape) != expected:
        raise ValueError(
            f"{name} has invalid BlockMajorK shape {tuple(tensor.shape)}, "
            f"expected {expected}"
        )


def _logical_token_capacity(
    *,
    fc: Literal["fc1", "fc2"],
    output_buf: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    routed_token_capacity: int | None = None,
) -> int:
    if routed_token_capacity is not None:
        logical_capacity = int(routed_token_capacity)
        if logical_capacity <= 0:
            raise ValueError(f"{fc} logical token capacity is empty")
        if int(output_buf.shape[0]) < logical_capacity:
            raise ValueError(
                f"{fc} output buffer token capacity too small: need "
                f"{logical_capacity}, got {output_buf.shape[0]}"
            )
        return logical_capacity
    if (
        output_buf.device.type == "cuda"
        and hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "routed_token_capacity is required during CUDA graph capture"
        )
    if total_num_padded_tokens.numel() < 1:
        raise ValueError("total_num_padded_tokens must contain the routed token count")
    logical_capacity = int(total_num_padded_tokens.reshape(-1)[0].item())
    if logical_capacity <= 0:
        raise ValueError(f"{fc} logical token capacity is empty")
    if int(output_buf.shape[0]) < logical_capacity:
        raise ValueError(
            f"{fc} output buffer token capacity too small: need "
            f"{logical_capacity}, got {output_buf.shape[0]}"
        )
    return logical_capacity


def _launch_early_exit_max_token_ctas(
    *,
    cfg,
    num_tokens: int,
    num_routed_experts: int,
    top_k: int,
    metadata_token_ctas: int,
) -> int:
    if not cfg.use_early_exit:
        return 0

    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_max_num_ctas_in_token_dim_for_moe,
    )

    token_tile_size = int(cfg.tile_n if cfg.is_swap_ab else cfg.tile_m)
    cluster_dim_in_token = 1 if cfg.is_swap_ab else int(cfg.cluster_m)
    launch_token_ctas = compute_max_num_ctas_in_token_dim_for_moe(
        num_tokens=int(num_tokens),
        num_experts=int(num_routed_experts),
        top_k=int(top_k),
        token_tile_size=token_tile_size,
        cluster_dim_in_token=cluster_dim_in_token,
    )
    if launch_token_ctas > int(metadata_token_ctas):
        raise ValueError(
            "routing metadata capacity is smaller than the required token CTA "
            f"launch bound: need {launch_token_ctas}, got {metadata_token_ctas}"
        )
    return launch_token_ctas


def _select_bias(
    *,
    fc: Literal["fc1", "fc2"],
    cfg,
    gemm1_bias: torch.Tensor | None,
    gemm2_bias: torch.Tensor | None,
    num_experts: int,
    out_hidden: int,
    device: torch.device,
) -> torch.Tensor | None:
    name = "gemm1_bias" if fc == "fc1" else "gemm2_bias"
    bias = gemm1_bias if fc == "fc1" else gemm2_bias
    if not cfg.has_bias_m:
        if bias is not None:
            raise ValueError(f"{name} supplied but {fc} config has bias disabled")
        return None
    if bias is None:
        raise ValueError(f"{fc} config has bias enabled but {name} is None")
    if not bias.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if bias.device != device:
        raise ValueError(f"{name} must be on {device}, got {bias.device}")
    if bias.dtype != torch.float32:
        raise ValueError(f"{name} must be float32 for Prims-TS BiasType.M")
    required = int(num_experts) * int(out_hidden)
    if bias.numel() < required:
        raise ValueError(
            f"{name} too small: need at least {required} values for "
            f"({num_experts}, {out_hidden}), got {bias.numel()}"
        )
    return bias


def _select_gemm1_oa_param(
    *,
    name: str,
    tensor: torch.Tensor | None,
    cfg,
    flag_name: str,
    fc: Literal["fc1", "fc2"],
    num_experts: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not bool(getattr(cfg, flag_name)):
        if fc == "fc1" and tensor is not None:
            raise ValueError(f"{name} supplied but {flag_name} is disabled")
        return None
    if fc != "fc1":
        return None
    if tensor is None:
        raise ValueError(f"{flag_name} is enabled but {name} is None")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    if tensor.numel() < int(num_experts):
        raise ValueError(
            f"{name} too small: need at least {num_experts} values, got {tensor.numel()}"
        )
    return tensor


def _make_gemm1_oa_ptrs(
    *,
    make_ptr,
    cutlass,
    cfg,
    fc: Literal["fc1", "fc2"],
    dummy_data_ptr: int,
    num_experts: int,
    device: torch.device,
    gemm1_alpha: torch.Tensor | None,
    gemm1_beta: torch.Tensor | None,
    gemm1_clamp_limit: torch.Tensor | None,
) -> tuple:
    alpha = _select_gemm1_oa_param(
        name="gemm1_alpha",
        tensor=gemm1_alpha,
        cfg=cfg,
        flag_name="has_gemm1_alpha",
        fc=fc,
        num_experts=num_experts,
        device=device,
    )
    beta = _select_gemm1_oa_param(
        name="gemm1_beta",
        tensor=gemm1_beta,
        cfg=cfg,
        flag_name="has_gemm1_beta",
        fc=fc,
        num_experts=num_experts,
        device=device,
    )
    clamp_limit = _select_gemm1_oa_param(
        name="gemm1_clamp_limit",
        tensor=gemm1_clamp_limit,
        cfg=cfg,
        flag_name="has_gemm1_clamp_limit",
        fc=fc,
        num_experts=num_experts,
        device=device,
    )
    alpha_dp = make_ptr(
        cutlass.Float32,
        alpha.data_ptr() if alpha is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    beta_dp = make_ptr(
        cutlass.Float32,
        beta.data_ptr() if beta is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    clamp_limit_dp = make_ptr(
        cutlass.Float32,
        clamp_limit.data_ptr() if clamp_limit is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    return alpha_dp, beta_dp, clamp_limit_dp, alpha, beta, clamp_limit


def build_bf16_launch_io(
    *,
    fc: Literal["fc1", "fc2"],
    cfg,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm1_bias: torch.Tensor | None = None,
    gemm2_bias: torch.Tensor | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm1_beta: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor | None = None,
    gemm1_output: torch.Tensor,
    gemm2_output: torch.Tensor,
    tile_idx: torch.Tensor,
    mn_limit: torch.Tensor,
    route_map: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    routed_token_capacity: int | None = None,
    activation_type: int,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
) -> dict:
    """Build Prims-TS launch IO from TRT-LLM routing/workspace tensors."""
    from cutlass.cute.runtime import make_ptr
    import cutlass

    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _runtime_config,
    )

    if fc == "fc1":
        out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
        in_hidden = hidden_size
        weights = gemm1_weights
        activation_compact = hidden_states
        output_buf = gemm1_output
    else:
        out_hidden = hidden_size
        in_hidden = intermediate_size
        weights = gemm2_weights
        activation_compact = gemm1_output
        output_buf = gemm2_output

    cfg = _runtime_config(cfg, in_hidden)
    _validate_weight_storage(
        name="gemm1_weights" if fc == "fc1" else "gemm2_weights",
        tensor=weights,
        cfg=cfg,
        num_experts=num_experts,
        out_hidden=out_hidden,
        in_hidden=in_hidden,
    )
    logical_token_capacity = _logical_token_capacity(
        fc=fc,
        output_buf=output_buf,
        total_num_padded_tokens=total_num_padded_tokens,
        routed_token_capacity=routed_token_capacity,
    )
    if tile_idx.numel() != mn_limit.numel():
        raise ValueError(
            f"tile_idx/mn_limit capacity mismatch: {tile_idx.numel()} vs {mn_limit.numel()}"
        )
    if not tile_idx.is_contiguous():
        raise ValueError("tile_idx must be contiguous")
    if not mn_limit.is_contiguous():
        raise ValueError("mn_limit must be contiguous")
    if not route_map.is_contiguous():
        raise ValueError("route_map must be contiguous")
    if not num_non_exiting_ctas.is_contiguous():
        raise ValueError("num_non_exiting_ctas must be contiguous")
    if not total_num_padded_tokens.is_contiguous():
        raise ValueError("total_num_padded_tokens must be contiguous")
    if num_non_exiting_ctas.numel() < 1:
        raise ValueError("num_non_exiting_ctas must contain the GPU active CTA count")

    launch_early_exit_max_token_ctas = _launch_early_exit_max_token_ctas(
        cfg=cfg,
        num_tokens=num_tokens,
        num_routed_experts=int(weights.shape[0]),
        top_k=top_k,
        metadata_token_ctas=tile_idx.numel(),
    )
    if cfg.is_swap_ab and out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )
    if output_buf.shape[1] < out_hidden and not cfg.has_gated_epilogue:
        raise ValueError(
            f"{fc} output buffer hidden dim too small: need {out_hidden}, "
            f"got {output_buf.shape[1]}"
        )

    # Shuffled weights are already gated-reordered and epilogue-shuffled
    # offline; Prims-TS consumes MajorK or BlockMajorK storage directly.
    preprocessed_weights = weights

    if cfg.is_swap_ab:
        m_val, n_val, k_val, l_val = (
            out_hidden,
            logical_token_capacity,
            in_hidden,
            num_experts,
        )
        kernel_a = preprocessed_weights
        kernel_b = activation_compact
    else:
        m_val, n_val, k_val, l_val = (
            logical_token_capacity,
            out_hidden,
            in_hidden,
            num_experts,
        )
        kernel_a = activation_compact
        kernel_b = preprocessed_weights

    logical_output_m = (
        m_val // 2 if cfg.is_swap_ab and cfg.has_gated_epilogue else m_val
    )
    if output_buf.numel() < logical_output_m * n_val:
        raise ValueError(
            f"{fc} output buffer too small: need {logical_output_m * n_val}, "
            f"got {output_buf.numel()}"
        )

    act_torch = activation_compact
    bias_torch = _select_bias(
        fc=fc,
        cfg=cfg,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        num_experts=num_experts,
        out_hidden=out_hidden,
        device=hidden_states.device,
    )

    a_dp = make_ptr(
        cutlass.BFloat16,
        kernel_a.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    b_dp = make_ptr(
        cutlass.BFloat16,
        kernel_b.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    c0_dp = make_ptr(
        cutlass.BFloat16,
        output_buf.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    dummy_data_ptr = output_buf.data_ptr()
    sf_c_dp = make_ptr(
        cutlass.BFloat16, dummy_data_ptr, cutlass.AddressSpace.gmem, assumed_align=16
    )
    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    route_map_dp = make_ptr(
        cutlass.Int32, route_map.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        cutlass.BFloat16,
        act_torch.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    bias_dp = make_ptr(
        cutlass.Float32,
        bias_torch.data_ptr() if bias_torch is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    (
        gemm1_alpha_dp,
        gemm1_beta_dp,
        gemm1_clamp_limit_dp,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
    ) = _make_gemm1_oa_ptrs(
        make_ptr=make_ptr,
        cutlass=cutlass,
        cfg=cfg,
        fc=fc,
        dummy_data_ptr=dummy_data_ptr,
        num_experts=num_experts,
        device=hidden_states.device,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    per_token_sf_a_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_sf_b_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    _keepalive = [
        kernel_a,
        kernel_b,
        output_buf,
        tile_idx,
        mn_limit,
        route_map,
        act_torch,
        bias_torch,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
        num_non_exiting_ctas,
        total_num_padded_tokens,
    ]

    return {
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
        "M": m_val,
        "N": n_val,
        "K": k_val,
        "L": l_val,
        "num_tokens": num_tokens,
        "total_padded_tokens": logical_token_capacity,
        "out_hidden": out_hidden,
        "in_hidden": in_hidden,
        "logical_output_m": logical_output_m,
        "shape": (m_val, n_val, k_val, l_val, num_tokens),
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": a_dp,
        "sfb_dp": b_dp,
        "c0_dp": c0_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "_keepalive": _keepalive,
    }


def build_nvfp4_launch_io(
    *,
    fc: Literal["fc1", "fc2"],
    cfg,
    cfg_is_normalized: bool = False,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: torch.Tensor | None = None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: torch.Tensor | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm1_beta: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor | None = None,
    gemm1_output: torch.Tensor,
    gemm1_output_scale: torch.Tensor,
    gemm2_output: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    tile_idx: torch.Tensor,
    mn_limit: torch.Tensor,
    route_map: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    routed_token_capacity: int | None = None,
    activation_type: int,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
    per_token_sf_a: torch.Tensor | None = None,
    per_token_sf_b: torch.Tensor | None = None,
) -> dict:
    """Build Prims-TS launch IO for the NVFP4xNVFP4 MoE path."""
    from cutlass.cute.runtime import make_ptr
    import cutlass

    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _runtime_config,
    )
    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    if fc == "fc1":
        out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
        in_hidden = hidden_size
        weights = gemm1_weights
        weights_scale = gemm1_weights_scale
        activation_compact = hidden_states
        activation_scale = hidden_states_scale
        output_buf = gemm1_output
        output_scale = gemm1_output_scale
        scale_c = output1_scale_scalar
        scale_gate = output1_scale_gate_scalar
    else:
        out_hidden = hidden_size
        in_hidden = intermediate_size
        weights = gemm2_weights
        weights_scale = gemm2_weights_scale
        activation_compact = gemm1_output
        activation_scale = gemm1_output_scale
        output_buf = gemm2_output
        output_scale = gemm1_output_scale
        scale_c = output2_scale_scalar
        scale_gate = output2_scale_scalar

    # The TRTLLM caller caches a private config normalized for this K.
    if not cfg_is_normalized:
        cfg = _runtime_config(cfg, in_hidden)
    _validate_weight_storage(
        name="gemm1_weights" if fc == "fc1" else "gemm2_weights",
        tensor=weights,
        cfg=cfg,
        num_experts=num_experts,
        out_hidden=out_hidden,
        in_hidden=in_hidden,
    )
    logical_token_capacity = _logical_token_capacity(
        fc=fc,
        output_buf=output_buf,
        total_num_padded_tokens=total_num_padded_tokens,
        routed_token_capacity=routed_token_capacity,
    )
    if not cfg.is_swap_ab:
        raise ValueError("NVFP4 MoE expects swapAB Prims-TS configs")
    if tile_idx.numel() != mn_limit.numel():
        raise ValueError(
            f"tile_idx/mn_limit capacity mismatch: {tile_idx.numel()} vs {mn_limit.numel()}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("gemm1_weights", gemm1_weights),
        ("gemm1_weights_scale", gemm1_weights_scale),
        ("gemm2_weights", gemm2_weights),
        ("gemm2_weights_scale", gemm2_weights_scale),
        ("gemm1_output", gemm1_output),
        ("gemm1_output_scale", gemm1_output_scale),
        ("gemm2_output", gemm2_output),
        ("tile_idx", tile_idx),
        ("mn_limit", mn_limit),
        ("route_map", route_map),
        ("num_non_exiting_ctas", num_non_exiting_ctas),
        ("total_num_padded_tokens", total_num_padded_tokens),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if hidden_states.dtype != torch.uint8:
        raise ValueError("NVFP4 hidden_states must be packed uint8")
    if hidden_states_scale.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 hidden_states_scale must be float8_e4m3fn")
    if cfg.has_per_token_sf_b:
        if per_token_sf_b is None:
            raise ValueError(f"per_token_sf_b is required by this NVFP4 {fc} config")
        if not per_token_sf_b.is_contiguous():
            raise ValueError("per_token_sf_b must be contiguous")
        required_tokens = num_tokens if fc == "fc1" else logical_token_capacity
        if per_token_sf_b.shape[0] < required_tokens:
            raise ValueError(
                f"per_token_sf_b must cover at least {required_tokens} tokens for {fc}, "
                f"got {per_token_sf_b.shape[0]}"
            )
        expected_dtype = {
            int(DType.BF16): torch.bfloat16,
            int(DType.FP16): torch.float16,
            int(DType.FP32): torch.float32,
        }.get(int(cfg.per_token_sf_dtype))
        if expected_dtype is None:
            raise ValueError(
                f"Unsupported NVFP4 per-token scale dtype={cfg.per_token_sf_dtype}"
            )
        if per_token_sf_b.dtype != expected_dtype:
            raise ValueError(
                f"per_token_sf_b must be {expected_dtype}, got {per_token_sf_b.dtype}"
            )

    launch_early_exit_max_token_ctas = _launch_early_exit_max_token_ctas(
        cfg=cfg,
        num_tokens=num_tokens,
        num_routed_experts=int(weights.shape[0]),
        top_k=top_k,
        metadata_token_ctas=tile_idx.numel(),
    )
    if out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )

    m_val, n_val, k_val, l_val = (
        out_hidden,
        logical_token_capacity,
        in_hidden,
        num_experts,
    )
    logical_output_m = m_val // 2 if cfg.has_gated_epilogue else m_val
    kernel_a = weights
    kernel_b = activation_compact
    sf_a = weights_scale
    sf_b = activation_scale
    bias_torch = _select_bias(
        fc=fc,
        cfg=cfg,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        num_experts=num_experts,
        out_hidden=out_hidden,
        device=hidden_states.device,
    )

    data_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dp = make_ptr(
        data_dtype, kernel_a.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    b_dp = make_ptr(
        data_dtype, kernel_b.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    sfa_dp = make_ptr(
        sf_dtype, sf_a.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
    )
    sfb_dp = make_ptr(
        sf_dtype, sf_b.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
    )
    if cfg.has_epilogue_quant:
        c0_dp = make_ptr(
            data_dtype,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
        sf_c_dp = make_ptr(
            sf_dtype,
            output_scale.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        c0_dp = make_ptr(
            cutlass.BFloat16,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
        sf_c_dp = make_ptr(
            sf_dtype,
            output_scale.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )

    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    route_map_dp = make_ptr(
        cutlass.Int32, route_map.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    total_num_padded_tokens_dp = make_ptr(
        cutlass.Int32,
        total_num_padded_tokens.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        data_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    dummy_data_ptr = output_buf.data_ptr()
    bias_dp = make_ptr(
        cutlass.Float32,
        bias_torch.data_ptr() if bias_torch is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        scale_c.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        scale_gate.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    (
        gemm1_alpha_dp,
        gemm1_beta_dp,
        gemm1_clamp_limit_dp,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
    ) = _make_gemm1_oa_ptrs(
        make_ptr=make_ptr,
        cutlass=cutlass,
        cfg=cfg,
        fc=fc,
        dummy_data_ptr=dummy_data_ptr,
        num_experts=num_experts,
        device=hidden_states.device,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    per_token_sf_a_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    if cfg.has_per_token_sf_b:
        per_token_cutlass_dtype = {
            int(DType.BF16): cutlass.BFloat16,
            int(DType.FP16): cutlass.Float16,
            int(DType.FP32): cutlass.Float32,
        }[int(cfg.per_token_sf_dtype)]
        per_token_sf_b_dp = make_ptr(
            per_token_cutlass_dtype,
            per_token_sf_b.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        per_token_sf_b_dp = make_ptr(
            cutlass.Float32,
            dummy_data_ptr,
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )

    _keepalive = [
        kernel_a,
        kernel_b,
        sf_a,
        sf_b,
        output_buf,
        output_scale,
        tile_idx,
        mn_limit,
        route_map,
        activation_compact,
        bias_torch,
        scale_c,
        scale_gate,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
        per_token_sf_b,
        num_non_exiting_ctas,
        total_num_padded_tokens,
    ]

    return {
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
        "M": m_val,
        "N": n_val,
        "K": k_val,
        "L": l_val,
        "num_tokens": num_tokens,
        "total_padded_tokens": logical_token_capacity,
        "out_hidden": out_hidden,
        "in_hidden": in_hidden,
        "logical_output_m": logical_output_m,
        "shape": (m_val, n_val, k_val, l_val, num_tokens),
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": sfa_dp,
        "sfb_dp": sfb_dp,
        "c0_dp": c0_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "total_num_padded_tokens_dp": total_num_padded_tokens_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "_keepalive": _keepalive,
    }


def build_mxfp4_mxfp8_launch_io(
    *,
    fc: Literal["fc1", "fc2"],
    cfg,
    cfg_is_normalized: bool = False,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: torch.Tensor | None = None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: torch.Tensor | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm1_beta: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor | None = None,
    gemm1_output: torch.Tensor,
    gemm1_output_scale: torch.Tensor,
    gemm2_output: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    tile_idx: torch.Tensor,
    mn_limit: torch.Tensor,
    route_map: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    routed_token_capacity: int | None = None,
    activation_type: int,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
) -> dict:
    """Build Prims-TS launch IO for the MXFP4xMXFP8 MoE path."""
    from cutlass.cute.runtime import make_ptr
    import cutlass

    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _runtime_config,
    )

    if fc == "fc1":
        out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
        in_hidden = hidden_size
        weights = gemm1_weights
        weights_scale = gemm1_weights_scale
        activation_compact = hidden_states
        activation_scale = hidden_states_scale
        output_buf = gemm1_output
        output_scale = gemm1_output_scale
        scale_c = output1_scale_scalar
        scale_gate = output1_scale_gate_scalar
    else:
        out_hidden = hidden_size
        in_hidden = intermediate_size
        weights = gemm2_weights
        weights_scale = gemm2_weights_scale
        activation_compact = gemm1_output
        activation_scale = gemm1_output_scale
        output_buf = gemm2_output
        output_scale = gemm1_output_scale
        scale_c = output2_scale_scalar
        scale_gate = output2_scale_scalar

    # The TRTLLM caller caches a private config normalized for this K.
    if not cfg_is_normalized:
        cfg = _runtime_config(cfg, in_hidden)
    _validate_weight_storage(
        name="gemm1_weights" if fc == "fc1" else "gemm2_weights",
        tensor=weights,
        cfg=cfg,
        num_experts=num_experts,
        out_hidden=out_hidden,
        in_hidden=in_hidden,
    )
    logical_token_capacity = _logical_token_capacity(
        fc=fc,
        output_buf=output_buf,
        total_num_padded_tokens=total_num_padded_tokens,
        routed_token_capacity=routed_token_capacity,
    )
    if not cfg.is_swap_ab:
        raise ValueError("MXFP4xMXFP8 MoE expects swapAB Prims-TS configs")
    if tile_idx.numel() != mn_limit.numel():
        raise ValueError(
            f"tile_idx/mn_limit capacity mismatch: {tile_idx.numel()} vs {mn_limit.numel()}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("gemm1_weights", gemm1_weights),
        ("gemm1_weights_scale", gemm1_weights_scale),
        ("gemm2_weights", gemm2_weights),
        ("gemm2_weights_scale", gemm2_weights_scale),
        ("gemm1_output", gemm1_output),
        ("gemm1_output_scale", gemm1_output_scale),
        ("gemm2_output", gemm2_output),
        ("tile_idx", tile_idx),
        ("mn_limit", mn_limit),
        ("route_map", route_map),
        ("num_non_exiting_ctas", num_non_exiting_ctas),
        ("total_num_padded_tokens", total_num_padded_tokens),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise ValueError("MXFP4xMXFP8 hidden_states must be float8_e4m3fn")
    if gemm1_weights.dtype != torch.uint8 or gemm2_weights.dtype != torch.uint8:
        raise ValueError("MXFP4 weights must be packed uint8")
    sf_k_fc1 = (int(hidden_size) + 31) // 32
    sf_k_fc2 = (int(intermediate_size) + 31) // 32
    fc1_out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
    _check_fp8_scale_storage(
        "hidden_states_scale",
        hidden_states_scale,
        min_numel=int(num_tokens) * sf_k_fc1,
    )
    _check_fp8_scale_storage(
        "gemm1_weights_scale",
        gemm1_weights_scale,
        min_numel=int(num_experts) * fc1_out_hidden * sf_k_fc1,
    )
    _check_fp8_scale_storage(
        "gemm2_weights_scale",
        gemm2_weights_scale,
        min_numel=int(num_experts) * int(hidden_size) * sf_k_fc2,
    )
    _check_fp8_scale_storage(
        "gemm1_output_scale",
        gemm1_output_scale,
        min_numel=logical_token_capacity * sf_k_fc2,
    )

    launch_early_exit_max_token_ctas = _launch_early_exit_max_token_ctas(
        cfg=cfg,
        num_tokens=num_tokens,
        num_routed_experts=int(weights.shape[0]),
        top_k=top_k,
        metadata_token_ctas=tile_idx.numel(),
    )
    if out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )

    m_val, n_val, k_val, l_val = (
        out_hidden,
        logical_token_capacity,
        in_hidden,
        num_experts,
    )
    logical_output_m = m_val // 2 if cfg.has_gated_epilogue else m_val
    data_a_dtype = cutlass.Float4E2M1FN
    data_b_dtype = cutlass.Float8E4M3FN
    sf_dtype = cutlass.Float8E8M0FNU
    bias_torch = _select_bias(
        fc=fc,
        cfg=cfg,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        num_experts=num_experts,
        out_hidden=out_hidden,
        device=hidden_states.device,
    )

    a_dp = make_ptr(
        data_a_dtype, weights.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    b_dp = make_ptr(
        data_b_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    sfa_dp = make_ptr(
        sf_dtype, weights_scale.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
    )
    sfb_dp = make_ptr(
        sf_dtype,
        activation_scale.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=32,
    )
    if cfg.has_epilogue_quant:
        c0_dp = make_ptr(
            cutlass.Float8E4M3FN,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
        sf_c_dp = make_ptr(
            sf_dtype,
            output_scale.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        c0_dp = make_ptr(
            cutlass.BFloat16,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
        sf_c_dp = make_ptr(
            sf_dtype,
            output_scale.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )

    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    route_map_dp = make_ptr(
        cutlass.Int32, route_map.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        data_b_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    dummy_data_ptr = output_buf.data_ptr()
    bias_dp = make_ptr(
        cutlass.Float32,
        bias_torch.data_ptr() if bias_torch is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        scale_c.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        scale_gate.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    (
        gemm1_alpha_dp,
        gemm1_beta_dp,
        gemm1_clamp_limit_dp,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
    ) = _make_gemm1_oa_ptrs(
        make_ptr=make_ptr,
        cutlass=cutlass,
        cfg=cfg,
        fc=fc,
        dummy_data_ptr=dummy_data_ptr,
        num_experts=num_experts,
        device=hidden_states.device,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    per_token_sf_a_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_sf_b_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    _keepalive = [
        weights,
        weights_scale,
        activation_compact,
        activation_scale,
        output_buf,
        output_scale,
        bias_torch,
        tile_idx,
        mn_limit,
        route_map,
        scale_c,
        scale_gate,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
        num_non_exiting_ctas,
        total_num_padded_tokens,
    ]

    return {
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
        "M": m_val,
        "N": n_val,
        "K": k_val,
        "L": l_val,
        "num_tokens": num_tokens,
        "total_padded_tokens": logical_token_capacity,
        "out_hidden": out_hidden,
        "in_hidden": in_hidden,
        "logical_output_m": logical_output_m,
        "shape": (m_val, n_val, k_val, l_val, num_tokens),
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": sfa_dp,
        "sfb_dp": sfb_dp,
        "c0_dp": c0_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "_keepalive": _keepalive,
    }


def build_fp8_block_scale_launch_io(
    *,
    fc: Literal["fc1", "fc2"],
    cfg,
    fp8_quantization_type: int | Fp8QuantizationType,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm1_bias: torch.Tensor | None = None,
    gemm2_bias: torch.Tensor | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm1_beta: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor | None = None,
    gemm1_output: torch.Tensor,
    gemm1_output_scale: torch.Tensor,
    activation_output: torch.Tensor,
    activation_output_scale: torch.Tensor,
    gemm2_output: torch.Tensor,
    tile_idx: torch.Tensor,
    mn_limit: torch.Tensor,
    route_map: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    routed_token_capacity: int | None = None,
    activation_type: int,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
) -> dict:
    """Build Prims-TS launch IO for FP8 block-scale DeepSeek/MXFP8 MoE."""
    from cutlass.cute.runtime import make_ptr
    import cutlass

    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _runtime_config,
    )

    quantization_type = Fp8QuantizationType(int(fp8_quantization_type))
    is_deepseek = quantization_type == Fp8QuantizationType.DeepSeekFp8
    if quantization_type not in (
        Fp8QuantizationType.DeepSeekFp8,
        Fp8QuantizationType.MxFp8,
    ):
        raise ValueError(
            f"Unsupported FP8 block-scale quantization: {quantization_type}"
        )

    if fc == "fc1":
        out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
        in_hidden = hidden_size
        weights = gemm1_weights
        weights_scale = gemm1_weights_scale
        activation_compact = hidden_states
        activation_scale = hidden_states_scale
        output_buf = gemm1_output
        output_scale = gemm1_output_scale
    else:
        out_hidden = hidden_size
        in_hidden = intermediate_size
        weights = gemm2_weights
        weights_scale = gemm2_weights_scale
        activation_compact = activation_output
        activation_scale = activation_output_scale
        output_buf = gemm2_output
        output_scale = activation_output_scale

    cfg = _runtime_config(cfg, in_hidden)
    _validate_weight_storage(
        name="gemm1_weights" if fc == "fc1" else "gemm2_weights",
        tensor=weights,
        cfg=cfg,
        num_experts=num_experts,
        out_hidden=out_hidden,
        in_hidden=in_hidden,
    )
    logical_token_capacity = _logical_token_capacity(
        fc=fc,
        output_buf=output_buf,
        total_num_padded_tokens=total_num_padded_tokens,
        routed_token_capacity=routed_token_capacity,
    )
    if is_deepseek:
        # DeepSeek staged buffers can have different storage capacities:
        # output tensors are rounded up for TRT-LLM workspace minima, while
        # scale tensors keep their own row stride. The TS kernel uses N as the
        # FP32 scale stride, so use the relevant scale-buffer width here.
        scale_stride_tensor = output_scale if fc == "fc1" else activation_scale
        if scale_stride_tensor.ndim >= 2:
            logical_token_capacity = int(scale_stride_tensor.shape[1])
    if logical_token_capacity <= 0:
        raise ValueError(f"{fc} logical token capacity is empty")
    if not cfg.is_swap_ab:
        raise ValueError("FP8 block-scale MoE expects swapAB Prims-TS configs")
    if tile_idx.numel() != mn_limit.numel():
        raise ValueError(
            f"tile_idx/mn_limit capacity mismatch: {tile_idx.numel()} vs {mn_limit.numel()}"
        )

    for name, tensor in (
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("gemm1_weights", gemm1_weights),
        ("gemm1_weights_scale", gemm1_weights_scale),
        ("gemm2_weights", gemm2_weights),
        ("gemm2_weights_scale", gemm2_weights_scale),
        ("gemm1_output", gemm1_output),
        ("gemm1_output_scale", gemm1_output_scale),
        ("activation_output", activation_output),
        ("activation_output_scale", activation_output_scale),
        ("gemm2_output", gemm2_output),
        ("tile_idx", tile_idx),
        ("mn_limit", mn_limit),
        ("route_map", route_map),
        ("num_non_exiting_ctas", num_non_exiting_ctas),
        ("total_num_padded_tokens", total_num_padded_tokens),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 block-scale hidden_states must be float8_e4m3fn")
    if (
        gemm1_weights.dtype != torch.float8_e4m3fn
        or gemm2_weights.dtype != torch.float8_e4m3fn
    ):
        raise ValueError("FP8 block-scale weights must be float8_e4m3fn")
    if gemm1_output.dtype != torch.uint8 or activation_output.dtype != torch.uint8:
        raise ValueError("FP8 block-scale intermediate outputs must use uint8 storage")
    if gemm2_output.dtype != torch.bfloat16:
        raise ValueError("FP8 block-scale FC2 output must be bfloat16")
    if is_deepseek:
        for name, tensor in (
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("gemm1_output_scale", gemm1_output_scale),
            ("activation_output_scale", activation_output_scale),
        ):
            if tensor.dtype != torch.float32:
                raise ValueError(f"DeepSeek FP8 {name} must be float32")
    else:
        sf_k_fc1 = (int(hidden_size) + 31) // 32
        sf_k_fc2 = (int(intermediate_size) + 31) // 32
        fc1_out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
        for name, tensor, min_numel in (
            (
                "hidden_states_scale",
                hidden_states_scale,
                int(num_tokens) * sf_k_fc1,
            ),
            (
                "gemm1_weights_scale",
                gemm1_weights_scale,
                int(num_experts) * fc1_out_hidden * sf_k_fc1,
            ),
            (
                "gemm2_weights_scale",
                gemm2_weights_scale,
                int(num_experts) * int(hidden_size) * sf_k_fc2,
            ),
            (
                "gemm1_output_scale",
                gemm1_output_scale,
                logical_token_capacity * sf_k_fc2,
            ),
            (
                "activation_output_scale",
                activation_output_scale,
                logical_token_capacity * sf_k_fc2,
            ),
        ):
            _check_fp8_scale_storage(name, tensor, min_numel=min_numel)

    launch_early_exit_max_token_ctas = _launch_early_exit_max_token_ctas(
        cfg=cfg,
        num_tokens=num_tokens,
        num_routed_experts=int(weights.shape[0]),
        top_k=top_k,
        metadata_token_ctas=tile_idx.numel(),
    )
    if out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )

    m_val, n_val, k_val, l_val = (
        out_hidden,
        logical_token_capacity,
        in_hidden,
        num_experts,
    )
    logical_output_m = m_val // 2 if cfg.has_gated_epilogue else m_val
    data_dtype = cutlass.Float8E4M3FN
    sf_dtype = cutlass.Float32 if is_deepseek else cutlass.Float8E8M0FNU
    dummy_data_ptr = output_buf.data_ptr()
    global_scale = (
        _get_expert_scale_ones(num_experts, hidden_states.device)
        if cfg.uses_global_scales
        else None
    )
    global_scale_ptr = (
        global_scale.data_ptr() if global_scale is not None else dummy_data_ptr
    )
    selected_bias = _select_bias(
        fc=fc,
        cfg=cfg,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        num_experts=num_experts,
        out_hidden=out_hidden,
        device=hidden_states.device,
    )

    a_dp = make_ptr(
        data_dtype, weights.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    b_dp = make_ptr(
        data_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    sfa_dp = make_ptr(
        sf_dtype, weights_scale.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
    )
    sfb_dp = make_ptr(
        sf_dtype,
        activation_scale.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=32,
    )
    if cfg.uses_fp8_output or cfg.has_epilogue_quant:
        c0_dp = make_ptr(
            data_dtype,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
        sf_c_dp = make_ptr(
            sf_dtype,
            output_scale.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        c0_dp = make_ptr(
            cutlass.BFloat16,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
        sf_c_dp = make_ptr(
            sf_dtype,
            output_scale.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )

    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    route_map_dp = make_ptr(
        cutlass.Int32, route_map.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    total_num_padded_tokens_dp = make_ptr(
        cutlass.Int32,
        total_num_padded_tokens.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        data_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    bias_dp = make_ptr(
        cutlass.Float32,
        selected_bias.data_ptr() if selected_bias is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        global_scale_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        global_scale_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    (
        gemm1_alpha_dp,
        gemm1_beta_dp,
        gemm1_clamp_limit_dp,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
    ) = _make_gemm1_oa_ptrs(
        make_ptr=make_ptr,
        cutlass=cutlass,
        cfg=cfg,
        fc=fc,
        dummy_data_ptr=dummy_data_ptr,
        num_experts=num_experts,
        device=hidden_states.device,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    per_token_sf_a_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_sf_b_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    _keepalive = [
        weights,
        weights_scale,
        activation_compact,
        activation_scale,
        output_buf,
        output_scale,
        tile_idx,
        mn_limit,
        route_map,
        *(() if global_scale is None else (global_scale,)),
        selected_bias,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
        num_non_exiting_ctas,
        total_num_padded_tokens,
    ]

    return {
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
        "M": m_val,
        "N": n_val,
        "K": k_val,
        "L": l_val,
        "num_tokens": num_tokens,
        "total_padded_tokens": logical_token_capacity,
        "out_hidden": out_hidden,
        "in_hidden": in_hidden,
        "logical_output_m": logical_output_m,
        "shape": (m_val, n_val, k_val, l_val, num_tokens),
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": sfa_dp,
        "sfb_dp": sfb_dp,
        "c0_dp": c0_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "total_num_padded_tokens_dp": total_num_padded_tokens_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "_keepalive": _keepalive,
    }


def build_mxfp4_bf16_launch_io(
    *,
    fc: Literal["fc1", "fc2"],
    cfg,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: torch.Tensor | None = None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: torch.Tensor | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm1_beta: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor | None = None,
    gemm1_output: torch.Tensor,
    gemm2_output: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    tile_idx: torch.Tensor,
    mn_limit: torch.Tensor,
    route_map: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    routed_token_capacity: int | None = None,
    activation_type: int,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
) -> dict:
    """Build Prims-TS launch IO for the MXFP4xBF16 MoE CastA path."""
    from cutlass.cute.runtime import make_ptr
    import cutlass

    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _runtime_config,
    )

    if fc == "fc1":
        out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
        in_hidden = hidden_size
        weights = gemm1_weights
        weights_scale = gemm1_weights_scale
        activation_compact = hidden_states
        output_buf = gemm1_output
        scale_c = output1_scale_scalar
        scale_gate = output1_scale_gate_scalar
    else:
        out_hidden = hidden_size
        in_hidden = intermediate_size
        weights = gemm2_weights
        weights_scale = gemm2_weights_scale
        activation_compact = gemm1_output
        output_buf = gemm2_output
        scale_c = output2_scale_scalar
        scale_gate = output2_scale_scalar

    cfg = _runtime_config(cfg, in_hidden)
    _validate_weight_storage(
        name="gemm1_weights" if fc == "fc1" else "gemm2_weights",
        tensor=weights,
        cfg=cfg,
        num_experts=num_experts,
        out_hidden=out_hidden,
        in_hidden=in_hidden,
    )
    logical_token_capacity = _logical_token_capacity(
        fc=fc,
        output_buf=output_buf,
        total_num_padded_tokens=total_num_padded_tokens,
        routed_token_capacity=routed_token_capacity,
    )
    if not cfg.is_swap_ab:
        raise ValueError("MXFP4xBF16 MoE expects swapAB Prims-TS configs")
    if tile_idx.numel() != mn_limit.numel():
        raise ValueError(
            f"tile_idx/mn_limit capacity mismatch: {tile_idx.numel()} vs {mn_limit.numel()}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("gemm1_weights", gemm1_weights),
        ("gemm1_weights_scale", gemm1_weights_scale),
        ("gemm2_weights", gemm2_weights),
        ("gemm2_weights_scale", gemm2_weights_scale),
        ("gemm1_output", gemm1_output),
        ("gemm2_output", gemm2_output),
        ("tile_idx", tile_idx),
        ("mn_limit", mn_limit),
        ("route_map", route_map),
        ("num_non_exiting_ctas", num_non_exiting_ctas),
        ("total_num_padded_tokens", total_num_padded_tokens),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("MXFP4xBF16 hidden_states must be bfloat16")
    if gemm1_output.dtype != torch.bfloat16 or gemm2_output.dtype != torch.bfloat16:
        raise ValueError("MXFP4xBF16 GEMM outputs must be bfloat16")
    if gemm1_weights.dtype != torch.uint8 or gemm2_weights.dtype != torch.uint8:
        raise ValueError("MXFP4 weights must be packed uint8")
    sf_k_fc1 = (int(hidden_size) + 31) // 32
    sf_k_fc2 = (int(intermediate_size) + 31) // 32
    _check_fp8_scale_storage(
        "gemm1_weights_scale",
        gemm1_weights_scale,
        min_numel=(
            int(num_experts)
            * _fc1_out_hidden(intermediate_size, activation_type)
            * sf_k_fc1
        ),
    )
    _check_fp8_scale_storage(
        "gemm2_weights_scale",
        gemm2_weights_scale,
        min_numel=int(num_experts) * int(hidden_size) * sf_k_fc2,
    )

    launch_early_exit_max_token_ctas = _launch_early_exit_max_token_ctas(
        cfg=cfg,
        num_tokens=num_tokens,
        num_routed_experts=int(weights.shape[0]),
        top_k=top_k,
        metadata_token_ctas=tile_idx.numel(),
    )
    if out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )

    m_val, n_val, k_val, l_val = (
        out_hidden,
        logical_token_capacity,
        in_hidden,
        num_experts,
    )
    logical_output_m = m_val // 2 if cfg.has_gated_epilogue else m_val
    data_a_dtype = cutlass.Float4E2M1FN
    data_b_dtype = cutlass.BFloat16
    sf_dtype = cutlass.Float8E8M0FNU
    dummy_data_ptr = output_buf.data_ptr()
    bias_torch = _select_bias(
        fc=fc,
        cfg=cfg,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        num_experts=num_experts,
        out_hidden=out_hidden,
        device=hidden_states.device,
    )

    a_dp = make_ptr(
        data_a_dtype, weights.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    b_dp = make_ptr(
        data_b_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    sfa_dp = make_ptr(
        sf_dtype, weights_scale.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=32
    )
    sfb_dp = make_ptr(
        sf_dtype,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    c0_dp = make_ptr(
        cutlass.BFloat16,
        output_buf.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    sf_c_dp = make_ptr(
        sf_dtype,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    route_map_dp = make_ptr(
        cutlass.Int32, route_map.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        data_b_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    bias_dp = make_ptr(
        cutlass.Float32,
        bias_torch.data_ptr() if bias_torch is not None else dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        scale_c.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        scale_gate.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    (
        gemm1_alpha_dp,
        gemm1_beta_dp,
        gemm1_clamp_limit_dp,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
    ) = _make_gemm1_oa_ptrs(
        make_ptr=make_ptr,
        cutlass=cutlass,
        cfg=cfg,
        fc=fc,
        dummy_data_ptr=dummy_data_ptr,
        num_experts=num_experts,
        device=hidden_states.device,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    per_token_sf_a_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    per_token_sf_b_dp = make_ptr(
        cutlass.Float32,
        dummy_data_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )

    _keepalive = [
        weights,
        weights_scale,
        activation_compact,
        output_buf,
        bias_torch,
        tile_idx,
        mn_limit,
        route_map,
        scale_c,
        scale_gate,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
        num_non_exiting_ctas,
        total_num_padded_tokens,
    ]

    return {
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
        "M": m_val,
        "N": n_val,
        "K": k_val,
        "L": l_val,
        "num_tokens": num_tokens,
        "total_padded_tokens": logical_token_capacity,
        "out_hidden": out_hidden,
        "in_hidden": in_hidden,
        "logical_output_m": logical_output_m,
        "shape": (m_val, n_val, k_val, l_val, num_tokens),
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": sfa_dp,
        "sfb_dp": sfb_dp,
        "c0_dp": c0_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "_keepalive": _keepalive,
    }


def build_fp8_per_tensor_launch_io(
    *,
    fc: Literal["fc1", "fc2"],
    cfg,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_bias: torch.Tensor | None = None,
    gemm2_weights: torch.Tensor,
    gemm2_bias: torch.Tensor | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm1_beta: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor | None = None,
    gemm1_output: torch.Tensor,
    gemm2_output: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    tile_idx: torch.Tensor,
    mn_limit: torch.Tensor,
    route_map: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    routed_token_capacity: int | None = None,
    activation_type: int,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    intermediate_size: int,
    hidden_size: int,
    per_token_sf_a: torch.Tensor | None = None,
    per_token_sf_b: torch.Tensor | None = None,
) -> dict:
    """Build Prims-TS launch IO for the FP8 per-tensor MoE path."""
    from cutlass.cute.runtime import make_ptr
    import cutlass

    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _runtime_config,
    )
    from tensorrt_llm._torch.moe.flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    if fc == "fc1":
        out_hidden = _fc1_out_hidden(intermediate_size, activation_type)
        in_hidden = hidden_size
        weights = gemm1_weights
        activation_compact = hidden_states
        output_buf = gemm1_output
        scale_c = output1_scale_scalar
        scale_gate = output1_scale_gate_scalar
    else:
        out_hidden = hidden_size
        in_hidden = intermediate_size
        weights = gemm2_weights
        activation_compact = gemm1_output
        output_buf = gemm2_output
        scale_c = output2_scale_scalar
        scale_gate = output2_scale_scalar

    cfg = _runtime_config(cfg, in_hidden)
    _validate_weight_storage(
        name="gemm1_weights" if fc == "fc1" else "gemm2_weights",
        tensor=weights,
        cfg=cfg,
        num_experts=num_experts,
        out_hidden=out_hidden,
        in_hidden=in_hidden,
    )
    logical_token_capacity = _logical_token_capacity(
        fc=fc,
        output_buf=output_buf,
        total_num_padded_tokens=total_num_padded_tokens,
        routed_token_capacity=routed_token_capacity,
    )
    if not cfg.is_swap_ab:
        raise ValueError("FP8 per-tensor MoE expects swapAB Prims-TS configs")
    if tile_idx.numel() != mn_limit.numel():
        raise ValueError(
            f"tile_idx/mn_limit capacity mismatch: {tile_idx.numel()} vs {mn_limit.numel()}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("gemm1_weights", gemm1_weights),
        ("gemm2_weights", gemm2_weights),
        ("gemm1_output", gemm1_output),
        ("gemm2_output", gemm2_output),
        ("tile_idx", tile_idx),
        ("mn_limit", mn_limit),
        ("route_map", route_map),
        ("num_non_exiting_ctas", num_non_exiting_ctas),
        ("total_num_padded_tokens", total_num_padded_tokens),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 per-tensor hidden_states must be float8_e4m3fn")
    if gemm1_weights.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 per-tensor gemm1_weights must be float8_e4m3fn")
    if gemm2_weights.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 per-tensor gemm2_weights must be float8_e4m3fn")
    expected_per_token_dtype = {
        int(DType.BF16): torch.bfloat16,
        int(DType.FP16): torch.float16,
        int(DType.FP32): torch.float32,
    }.get(int(cfg.per_token_sf_dtype))
    if cfg.has_per_token_sf_a:
        if expected_per_token_dtype is None:
            raise ValueError(
                f"Unsupported FP8 per-token scale dtype={cfg.per_token_sf_dtype}"
            )
        if per_token_sf_a is None:
            raise ValueError(f"per_token_sf_a is required by this FP8 {fc} config")
        if not per_token_sf_a.is_contiguous():
            raise ValueError("per_token_sf_a must be contiguous")
        if per_token_sf_a.dtype != expected_per_token_dtype:
            raise ValueError(
                f"per_token_sf_a must be {expected_per_token_dtype}, got {per_token_sf_a.dtype}"
            )
        required_rows = out_hidden if fc == "fc1" else hidden_size
        if per_token_sf_a.shape[0] < required_rows:
            raise ValueError(
                f"per_token_sf_a must cover at least {required_rows} rows for {fc}, "
                f"got {per_token_sf_a.shape[0]}"
            )
    if cfg.has_per_token_sf_b:
        if fc != "fc1":
            raise ValueError("FP8 per-token B scaling is only supported for FC1")
        if per_token_sf_b is None:
            raise ValueError("per_token_sf_b is required by this FP8 FC1 config")
        if not per_token_sf_b.is_contiguous():
            raise ValueError("per_token_sf_b must be contiguous")
        if expected_per_token_dtype is None:
            raise ValueError(
                f"Unsupported FP8 per-token scale dtype={cfg.per_token_sf_dtype}"
            )
        if per_token_sf_b.dtype != expected_per_token_dtype:
            raise ValueError(
                f"per_token_sf_b must be {expected_per_token_dtype}, got {per_token_sf_b.dtype}"
            )

    launch_early_exit_max_token_ctas = _launch_early_exit_max_token_ctas(
        cfg=cfg,
        num_tokens=num_tokens,
        num_routed_experts=int(weights.shape[0]),
        top_k=top_k,
        metadata_token_ctas=tile_idx.numel(),
    )
    if out_hidden % cfg.tile_m != 0:
        raise ValueError(
            f"swapAB requires out_hidden to be a multiple of tile_m={cfg.tile_m}, "
            f"got {out_hidden}"
        )

    m_val, n_val, k_val, l_val = (
        out_hidden,
        logical_token_capacity,
        in_hidden,
        num_experts,
    )
    logical_output_m = m_val // 2 if cfg.has_gated_epilogue else m_val
    data_dtype = cutlass.Float8E4M3FN
    dummy_ptr = output_buf.data_ptr()
    bias_torch = _select_bias(
        fc=fc,
        cfg=cfg,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        num_experts=num_experts,
        out_hidden=out_hidden,
        device=hidden_states.device,
    )

    a_dp = make_ptr(
        data_dtype, weights.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    b_dp = make_ptr(
        data_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    sfa_dp = make_ptr(
        cutlass.Float32, dummy_ptr, cutlass.AddressSpace.gmem, assumed_align=16
    )
    sfb_dp = make_ptr(
        cutlass.Float32, dummy_ptr, cutlass.AddressSpace.gmem, assumed_align=16
    )
    if cfg.uses_fp8_output:
        c0_dp = make_ptr(
            data_dtype,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        c0_dp = make_ptr(
            cutlass.BFloat16,
            output_buf.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    sf_c_dp = make_ptr(
        cutlass.Float32, dummy_ptr, cutlass.AddressSpace.gmem, assumed_align=16
    )
    tile_idx_dp = make_ptr(
        cutlass.Int32, tile_idx.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    route_map_dp = make_ptr(
        cutlass.Int32, route_map.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    mn_limit_dp = make_ptr(
        cutlass.Int32, mn_limit.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
    )
    num_non_exiting_ctas_dp = make_ptr(
        cutlass.Int32,
        num_non_exiting_ctas.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=4,
    )
    act_dp = make_ptr(
        data_dtype,
        activation_compact.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    if cfg.has_per_token_sf_a or cfg.has_per_token_sf_b:
        per_token_cutlass_dtype = {
            int(DType.BF16): cutlass.BFloat16,
            int(DType.FP16): cutlass.Float16,
            int(DType.FP32): cutlass.Float32,
        }[int(cfg.per_token_sf_dtype)]
    if cfg.has_per_token_sf_a:
        per_token_sf_a_dp = make_ptr(
            per_token_cutlass_dtype,
            per_token_sf_a.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        per_token_sf_a_dp = make_ptr(
            cutlass.Float32, dummy_ptr, cutlass.AddressSpace.gmem, assumed_align=16
        )
    if cfg.has_per_token_sf_b:
        per_token_sf_b_dp = make_ptr(
            per_token_cutlass_dtype,
            per_token_sf_b.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    else:
        per_token_sf_b_dp = make_ptr(
            cutlass.Float32, dummy_ptr, cutlass.AddressSpace.gmem, assumed_align=16
        )
    bias_dp = make_ptr(
        cutlass.Float32,
        bias_torch.data_ptr() if bias_torch is not None else dummy_ptr,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_c_dp = make_ptr(
        cutlass.Float32,
        scale_c.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_gate_dp = make_ptr(
        cutlass.Float32,
        scale_gate.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    (
        gemm1_alpha_dp,
        gemm1_beta_dp,
        gemm1_clamp_limit_dp,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
    ) = _make_gemm1_oa_ptrs(
        make_ptr=make_ptr,
        cutlass=cutlass,
        cfg=cfg,
        fc=fc,
        dummy_data_ptr=dummy_ptr,
        num_experts=num_experts,
        device=hidden_states.device,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )

    _keepalive = [
        weights,
        activation_compact,
        output_buf,
        bias_torch,
        tile_idx,
        mn_limit,
        route_map,
        scale_c,
        scale_gate,
        selected_gemm1_alpha,
        selected_gemm1_beta,
        selected_gemm1_clamp_limit,
        per_token_sf_a,
        per_token_sf_b,
        num_non_exiting_ctas,
        total_num_padded_tokens,
    ]

    return {
        "cfg": cfg,
        "launch_early_exit_max_token_ctas": launch_early_exit_max_token_ctas,
        "M": m_val,
        "N": n_val,
        "K": k_val,
        "L": l_val,
        "num_tokens": num_tokens,
        "total_padded_tokens": logical_token_capacity,
        "out_hidden": out_hidden,
        "in_hidden": in_hidden,
        "logical_output_m": logical_output_m,
        "shape": (m_val, n_val, k_val, l_val, num_tokens),
        "a_dp": a_dp,
        "b_dp": b_dp,
        "sfa_dp": sfa_dp,
        "sfb_dp": sfb_dp,
        "c0_dp": c0_dp,
        "sf_c_dp": sf_c_dp,
        "tile_idx_dp": tile_idx_dp,
        "route_map_dp": route_map_dp,
        "mn_limit_dp": mn_limit_dp,
        "num_non_exiting_ctas_dp": num_non_exiting_ctas_dp,
        "act_dp": act_dp,
        "per_token_sf_a_dp": per_token_sf_a_dp,
        "per_token_sf_b_dp": per_token_sf_b_dp,
        "bias_dp": bias_dp,
        "scale_c_dp": scale_c_dp,
        "scale_gate_dp": scale_gate_dp,
        "gemm1_alpha_dp": gemm1_alpha_dp,
        "gemm1_beta_dp": gemm1_beta_dp,
        "gemm1_clamp_limit_dp": gemm1_clamp_limit_dp,
        "_keepalive": _keepalive,
    }
