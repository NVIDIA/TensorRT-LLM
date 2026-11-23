# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise FFN fusions for the qualified EarlyTop8 SM107 Ulysses shards."""

import os
from typing import Literal

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod
from tensorrt_llm.logger import logger

from ..transformer_minimax_h3 import MiniMaxH3Transformer3DModel, MiniMaxH3TransformerBlock

__all__ = ["Fp8UlyssesMlpMixin", "initialize_block_ops"]

_HIDDEN_SIZE = 5376
_INTERMEDIATE_SIZE = 14336
_BF16_BLOCKS = frozenset((1, 16, 32, 34, 35, 36, 37, 42))
_QUALIFIED_SHARD_ROWS = (9455, 9543)

_Precision = Literal["fp8", "bf16"]


@torch.library.custom_op("h3_u4::aligned_token_indices", mutates_args=(), device_types="cuda")
def _aligned_token_indices(indices: torch.Tensor) -> torch.Tensor:
    # Keep this copy opaque to Inductor, which can elide a read-only clone.
    return indices.clone()


@_aligned_token_indices.register_fake
def _aligned_token_indices_fake(indices: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(indices)


def _supported_shard(
    hidden: torch.Tensor, normalized: torch.Tensor, mod: torch.Tensor, indices: torch.Tensor
) -> bool:
    return (
        hidden.is_cuda
        and hidden.dtype == normalized.dtype == mod.dtype == torch.bfloat16
        and indices.dtype == torch.int64
        and hidden.ndim == 3
        and hidden.shape == normalized.shape
        and hidden.shape[0] == 1
        and hidden.shape[1] in _QUALIFIED_SHARD_ROWS
        and hidden.shape[2] == _HIDDEN_SIZE
        and mod.ndim == 2
        and mod.shape[1] == 6 * _HIDDEN_SIZE
        and indices.shape == (hidden.shape[1],)
        and indices.device == hidden.device
        and indices.is_contiguous()
        and all(x.device == hidden.device and x.is_contiguous() for x in (hidden, normalized, mod))
    )


class Fp8UlyssesMlpMixin(MiniMaxH3TransformerBlock):
    """Replace the FFN epilogue on the compiled SM107 U4 path."""

    _u4_ffn_precision: _Precision | None = None
    _u4_packed_gate_up: torch.Tensor
    _u4_bf16_unit: torch.Tensor

    def _forward_mlp_residual(
        self,
        hidden_states: torch.Tensor,
        norm_hidden_states: torch.Tensor,
        mod: torch.Tensor,
        adaln_indices: torch.Tensor,
    ) -> torch.Tensor:
        precision = self._u4_ffn_precision
        if (
            precision is None
            or not torch.compiler.is_compiling()
            or torch.is_grad_enabled()
            or not _supported_shard(hidden_states, norm_hidden_states, mod, adaln_indices)
        ):
            return super()._forward_mlp_residual(
                hidden_states, norm_hidden_states, mod, adaln_indices
            )

        from ..minimax_h3_fp8__sm_107__u1 import ops

        ff = self.ff
        gate, down = ff.gate_up_proj, ff.down_proj
        # Fused AdaLN supplies newly allocated aligned hidden, norm and modulation tensors.
        ff_input = norm_hidden_states.view(-1, _HIDDEN_SIZE)
        # Ulysses narrows an int64 index tensor to odd row counts. Materialize the
        # local view so the Down kernel's 16-byte alignment holds on every rank.
        indices = _aligned_token_indices(adaln_indices)
        if precision == "bf16":
            unit = self._u4_bf16_unit
            activation = ops.gate_up_bf16(
                ff_input, gate.weight, unit, ff.swiglu_limit, ff.swiglu_alpha, ff.swiglu_beta
            )
            return ops.bf16_down(activation, down.weight, unit, hidden_states, mod, indices)

        codes, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            ff_input, gate.input_scale
        )
        activation = ops.gate_up(
            codes, self._u4_packed_gate_up, gate.input_scale, gate.weight_scale, down.input_scale
        )
        return ops.down(
            activation,
            down.weight,
            down.input_scale,
            down.weight_scale,
            hidden_states,
            mod,
            indices,
        )


def _eligible(block: Fp8UlyssesMlpMixin, dtype: torch.dtype, device: torch.device) -> bool:
    ff = block.ff
    if (
        not block._fused_adaln
        or ff.split_gate_up
        or ff.fuse_bf16_gate_up_swiglu
        or ff.activation != F.silu
        or not ff._is_plain_swiglu()
        or ff.swiglu_limit not in (None, float("inf"))
        or block.adaln_proj.linear.weight.dtype != torch.bfloat16
    ):
        return False

    for linear, shape in (
        (ff.gate_up_proj, (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE)),
        (ff.down_proj, (_HIDDEN_SIZE, _INTERMEDIATE_SIZE)),
    ):
        weight = linear.weight
        if (
            weight.shape != shape
            or weight.dtype != dtype
            or weight.device != device
            or not weight.is_contiguous()
            or weight.data_ptr() % 256 != 0
            or linear.tp_size != 1
            or linear.bias is not None
        ):
            return False
        if dtype == torch.bfloat16:
            if linear.has_any_quant or linear.use_cute_dsl_bf16_gemm or linear.use_custom_cublas_mm:
                return False
        elif (
            type(linear.quant_method) is not FP8QDQLinearMethod
            or linear.force_dynamic_quantization
            or any(
                scale is None
                or scale.device != device
                or scale.dtype != torch.float32
                or scale.numel() != 1
                or scale.data_ptr() % 4 != 0
                for scale in (linear.input_scale, linear.weight_scale)
            )
        ):
            return False
    return True


def initialize_block_ops(transformer: MiniMaxH3Transformer3DModel) -> None:
    """Prepare eligible EarlyTop8 block kernels once after the checkpoint is loaded."""
    device = transformer.proj_out.weight.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 7):
        return

    blocks = transformer.transformer_blocks
    if len(blocks) != 50 or any(
        block.ff.down_proj.weight.dtype
        != (torch.bfloat16 if index in _BF16_BLOCKS else torch.float8_e4m3fn)
        for index, block in enumerate(blocks)
    ):
        return

    try:
        # On other architectures the U4 pipeline can load without the optional SM107 kernels.
        from ..minimax_h3_fp8__sm_107__u1 import ops
    except ImportError as error:
        logger.warning(f"MiniMax-H3 U4 is using the stock FFN: {error}")
        return

    gate_up_swiglu, projection_residual = ops.gate_up_swiglu, ops.projection_residual
    device_index = device.index
    assert device_index is not None
    settings = gate_up_swiglu._CFG
    gate_smem = 3 * 128 * settings["mma_tile_k"] * settings["num_ab_stage"] + 4096
    if projection_residual.block_shared_memory_bytes(device_index) < gate_smem:
        return

    for precision, dtype in (("fp8", torch.float8_e4m3fn), ("bf16", torch.bfloat16)):
        if precision == "fp8" and os.getenv("USE_CUTE_DSL_FP8_PER_TENSOR_MM") == "1":
            continue
        eligible = [
            block
            for block in blocks
            if isinstance(block, Fp8UlyssesMlpMixin)
            and block._u4_ffn_precision is None
            and _eligible(block, dtype, device)
        ]
        if not eligible:
            continue
        bf16 = precision == "bf16"
        try:
            gate_up_swiglu._get_compiled(device_index, bf16=bf16)
            projection_residual.compile_kernel(
                device_index,
                _HIDDEN_SIZE,
                _INTERMEDIATE_SIZE,
                projection_residual.DOWN_GATE,
                torch.bfloat16,
                torch.int64,
                bf16=bf16,
            )
            if bf16:
                buffer_name = "_u4_bf16_unit"
                buffers = [torch.ones(1, dtype=torch.float32, device=device)] * len(eligible)
            else:
                # Keep stock checkpoint weights intact for the shape, layout and eager fallbacks.
                buffer_name = "_u4_packed_gate_up"
                buffers = [gate_up_swiglu.pack_weight(b.ff.gate_up_proj.weight) for b in eligible]
        except (RuntimeError, ValueError) as error:
            logger.warning(f"MiniMax-H3 U4 is using the stock {precision} FFN: {error}")
            continue

        for block, buffer in zip(eligible, buffers, strict=True):
            block.register_buffer(buffer_name, buffer, persistent=False)
            block._u4_ffn_precision = precision
