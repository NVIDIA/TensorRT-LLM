# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 checkpoint validation and exact native FP8/TP weight preparation."""

import torch
from torch import nn

_NATIVE_LINEARS = frozenset(("wq_a", "wq_b", "wkv", "index_wq_b", "o_b_proj"))


def expand_mxfp8_scales(scales: torch.Tensor, shape: tuple[int, int], block: int) -> torch.Tensor:
    """Expand square HF scale blocks into exact per-row UE8M0 1x32 scales.

    This transforms scale metadata only; checkpoint E4M3 weight bytes never
    undergo dequantization, resmoothing, or requantization.
    """
    if block not in (32, 128) or shape[1] % 32:
        raise ValueError("Native CSA2 FP8 requires square 32/128 blocks and K divisible by 32")
    expected = tuple((dim + block - 1) // block for dim in shape)
    if tuple(scales.shape) != expected:
        raise ValueError(f"CSA2 FP8 scale geometry {tuple(scales.shape)} != {expected}")
    values = torch.exp2(scales.float() - 127) if scales.dtype == torch.uint8 else scales.float()
    if not bool((torch.isfinite(values) & (values > 0)).all()):
        raise ValueError("CSA2 MXFP8 scales must be finite positive powers of two")
    exponents = torch.log2(values)
    if not bool(
        ((exponents == exponents.round()) & (exponents >= -127) & (exponents <= 127)).all()
    ):
        raise ValueError("CSA2 MXFP8 cannot exactly represent non-UE8M0 checkpoint scales")
    encoded = (exponents + 127).to(torch.uint8)
    return (
        encoded.repeat_interleave(block, 0)[: shape[0]]
        .repeat_interleave(block // 32, 1)[:, : shape[1] // 32]
        .contiguous()
    )


def _checkpoint_scales(weights, key, tensor, blocks):
    scales = weights[key.removesuffix("weight") + "weight_scale_inv"]
    block = next(
        (b for b in blocks if tuple(scales.shape) == tuple((d + b - 1) // b for d in tensor.shape)),
        None,
    )
    if block is None:
        raise ValueError(f"CSA2 checkpoint has unsupported FP8 scale geometry for {key}")
    return scales, block


def load_attention_weights(
    module: nn.Module,
    weights: dict[str, torch.Tensor],
    prefix: str,
    num_groups: int,
    o_lora_rank: int,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Prepare every value/scale before modifying any module parameter."""
    if tp_size <= 0 or not 0 <= tp_rank < tp_size or num_groups % tp_size:
        raise ValueError("Invalid CSA2 output-group tensor parallel mapping")
    aliases = {
        "o_a_proj": "wo_a.weight",
        "o_b_proj.weight": "wo_b.weight",
        "index_wq_b.weight": "indexer.wq_b.weight",
        "index_weights_proj.weight": "indexer.weights_proj.weight",
        "index_wk.weight": "indexer.wk.weight",
        "index_k_norm.weight": "indexer.k_norm.weight",
    }
    native = getattr(module, "projection_quantization", "bf16") == "mxfp8"
    prepared = {}
    native_scales = {}
    grouped_scales = None
    grouped_block128 = False
    for name, parameter in module.named_parameters():
        owner_name, _, leaf = name.rpartition(".")
        if native and owner_name in _NATIVE_LINEARS and leaf == "weight_scale":
            continue
        key = prefix + aliases.get(name, name)
        tensor = weights[key]
        expected_shape = tuple(parameter.shape)
        if name == "o_a_proj":
            expected_shape = (num_groups * o_lora_rank, parameter.shape[2])
        elif name in ("wq_b.weight", "attn_sink"):
            expected_shape = (parameter.shape[0] * tp_size, *parameter.shape[1:])
        elif name == "o_b_proj.weight":
            expected_shape = (parameter.shape[0], parameter.shape[1] * tp_size)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"CSA2 checkpoint shape mismatch for {key}: {tensor.shape} != {expected_shape}"
            )
        native_weight = native and (
            name == "o_a_proj" or owner_name in _NATIVE_LINEARS and leaf == "weight"
        )
        row_scales = None
        if native_weight and tensor.dtype != torch.float8_e4m3fn:
            raise ValueError(f"CSA2 native projection requires E4M3 checkpoint weight {key}")
        if tensor.dtype == torch.float8_e4m3fn:
            scales, block = _checkpoint_scales(weights, key, tensor, (32, 128))
            if native_weight:
                if name == "o_a_proj" and block == 128:
                    if o_lora_rank % 128 or tensor.shape[1] % 128:
                        raise ValueError(
                            "CSA2 fused grouped FP8 requires 128-aligned group dimensions"
                        )
                    values = (
                        torch.exp2(scales.float() - 127)
                        if scales.dtype == torch.uint8
                        else scales.float()
                    )
                    if not bool((torch.isfinite(values) & (values > 0)).all()):
                        raise ValueError("CSA2 grouped FP8 scales must be finite and positive")
                    grouped_scales = values.reshape(num_groups, o_lora_rank // 128, -1)
                    groups_local = num_groups // tp_size
                    grouped_scales = (
                        grouped_scales.narrow(0, tp_rank * groups_local, groups_local)
                        .to(parameter.device)
                        .contiguous()
                    )
                    grouped_block128 = True
                else:
                    row_scales = expand_mxfp8_scales(scales, tuple(tensor.shape), block)
            else:
                values = (
                    torch.exp2(scales.float() - 127)
                    if scales.dtype == torch.uint8
                    else scales.float()
                )
                tensor = (
                    tensor.float()
                    * values.repeat_interleave(block, 0).repeat_interleave(block, 1)[
                        : tensor.shape[0], : tensor.shape[1]
                    ]
                )
        if name == "o_a_proj":
            groups_local = num_groups // tp_size
            tensor = tensor.reshape(num_groups, *parameter.shape[1:]).narrow(
                0, tp_rank * groups_local, groups_local
            )
            if row_scales is not None:
                row_scales = row_scales.reshape(num_groups, o_lora_rank, -1).narrow(
                    0, tp_rank * groups_local, groups_local
                )
                grouped_scales = torch.stack(
                    [
                        torch.ops.trtllm.block_scale_interleave(
                            scale.to(parameter.device).contiguous()
                        )
                        for scale in row_scales
                    ]
                )
        elif name in ("wq_b.weight", "attn_sink"):
            tensor = tensor.narrow(0, tp_rank * parameter.shape[0], parameter.shape[0])
            if row_scales is not None:
                row_scales = row_scales.narrow(0, tp_rank * parameter.shape[0], parameter.shape[0])
        elif name == "o_b_proj.weight":
            tensor = tensor.narrow(1, tp_rank * parameter.shape[1], parameter.shape[1])
            if row_scales is not None:
                if parameter.shape[1] % 32:
                    raise ValueError(
                        "CSA2 native row-parallel weight shards must align to 32 columns"
                    )
                row_scales = row_scales.narrow(
                    1, tp_rank * (parameter.shape[1] // 32), parameter.shape[1] // 32
                )
        if tensor.shape != parameter.shape:
            raise ValueError(
                f"CSA2 checkpoint shape mismatch for {key}: {tensor.shape} != {parameter.shape}"
            )
        prepared[name] = tensor.to(device=parameter.device, dtype=parameter.dtype)
        if row_scales is not None and name != "o_a_proj":
            linear = module.get_submodule(owner_name)
            if not linear.quant_method.use_cutlass:
                raise RuntimeError(
                    "CSA2 native projections cannot use the dequantized MXFP8 fallback"
                )
            value = torch.ops.trtllm.block_scale_interleave(
                row_scales.to(parameter.device).contiguous()
            )
            if value.numel() != linear.weight_scale.numel():
                raise ValueError(
                    "CSA2 native weight-scale allocation does not match checkpoint geometry"
                )
            native_scales[owner_name] = value.reshape_as(linear.weight_scale)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name in prepared:
                parameter.copy_(prepared[name])
        for name, scales in native_scales.items():
            module.get_submodule(name).weight_scale.copy_(scales)
        if native:
            module.o_a_proj_scale = grouped_scales
            module._o_a_fp8_block128 = grouped_block128
