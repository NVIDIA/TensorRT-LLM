# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instance-scoped FP8 H3 attention, blocks and final media projections."""

from pathlib import Path

import cuda.bindings.driver as cuda
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod

from ..minimax_h3_nvfp4__sm_107__u1_u4.kernels import fused_fp8_qk
from ..minimax_h3_nvfp4__sm_107__u1_u4.modeling import ScenarioAttention, ScenarioTransformer
from ..transformer_minimax_h3 import MiniMaxH3TransformerBlock
from . import ops
from .conditioning import ConditioningCache
from .kernels import attention_output, gate_up_swiglu
from .kernels.conversion import prepare
from .kernels.rope import half_rope_tensor_cast
from .transformer_forward import shared_output_forward


@torch.library.custom_op("h3_fp8::qk_norm_rope", mutates_args=())
def qk_norm_rope(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    descale: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
    layer_idx: int,
    valid_len: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs, scales = fused_fp8_qk._qk_norm_rope_fp8_impl(
        qkv,
        weight_q,
        weight_k,
        cos,
        sin,
        descale,
        eps,
        num_heads,
        head_dim,
        layer_idx,
        valid_len,
        0,
        cast_g_kernel=half_rope_tensor_cast,
    )
    return outputs[0], outputs[1], outputs[2], scales


@qk_norm_rope.register_fake
def _qk_fake(
    qkv, weight_q, weight_k, cos, sin, descale, eps, num_heads, head_dim, layer_idx, valid_len
):
    shape = (*qkv.shape[:2], num_heads * head_dim)
    return (
        torch.empty(shape, device=qkv.device, dtype=torch.float8_e4m3fn),
        torch.empty(shape, device=qkv.device, dtype=torch.float8_e4m3fn),
        torch.empty(shape, device=qkv.device, dtype=torch.float8_e4m3fn),
        torch.empty((3,), device=qkv.device, dtype=torch.float32),
    )


class Fp8QkvMethod(FP8QDQLinearMethod):
    """Use the selected framework GEMM for qualified full-sequence QKV projections."""

    def apply(self, module, input: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if (
            module.weight.shape != (21504, 5376)
            or module.dtype != torch.bfloat16
            or self.get_static_input_scale(module) is None
            or getattr(module, "all_reduce", None) is not None
            or input.numel() // input.shape[-1] not in ops.FULL_SEQUENCE_LENGTHS
        ):
            return super().apply(module, input, bias)
        shape = input.shape
        activation = input.reshape(-1, shape[-1])
        if activation.dtype != torch.float8_e4m3fn:
            activation = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                activation, module.input_scale
            )[0]
        output = ops.qkv_gemm(activation, module.weight, module.input_scale, module.weight_scale)
        if len(shape) > 2:
            output = output.reshape(*shape[:-1], output.shape[-1])
        return output + bias if bias is not None else output


class Fp8Attention(ScenarioAttention):
    """Raw attention for FP8 producers; projected family attention for BF16 inputs."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        key_padding_mask=None,
        timestep=None,
        valid_seq_len=None,
    ) -> torch.Tensor:
        if hidden_states.dtype == torch.bfloat16:
            return super().forward(
                hidden_states,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len,
            )
        if rotary_emb is None or not self.on_fp8_path():
            raise ValueError("The FP8 scenario needs rotary embeddings and FP8 QKV attention")
        if self.use_async_ulysses or getattr(self.attn, "world_size", 1) != 1:
            raise ValueError("The FP8 fused attention supports Ulysses=1 only")
        qkv = self.qkv_proj(hidden_states)
        cos, sin = rotary_emb
        scales = fused_fp8_qk.qk_static_descale(self.head_dim, hidden_states.device)
        valid_len = None
        if key_padding_mask is not None:
            valid_len = key_padding_mask.sum(dim=1).to(torch.int32).contiguous()
        query, key, value, descales = qk_norm_rope(
            qkv,
            self.norm_q.weight,
            self.norm_k.weight,
            cos,
            sin,
            scales,
            float(self.norm_q.variance_epsilon),
            self.local_num_attention_heads,
            self.head_dim,
            int(self.layer_idx),
            valid_len,
        )
        return self._attn_impl(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            discard_padded_queries=True,
            timestep=timestep,
            fp8_descales=descales,
        )


class Fp8Block(MiniMaxH3TransformerBlock):
    """Static-FP8 producers and fused GateUp, attention-output and Down GEMMs."""

    attention_cls = Fp8Attention

    def _forward_fused_adaln(
        self,
        hidden_states,
        temb,
        adaln_indices,
        rotary_emb,
        key_padding_mask,
        timestep,
        valid_seq_len,
    ) -> torch.Tensor:
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[0] != 1
            or hidden_states.shape[2] != 5376
            or hidden_states.dtype != torch.bfloat16
        ):
            raise ValueError("The fused FP8 block needs a qualified BF16 [1,S,5376] packed layout")
        if hidden_states.shape[1] not in ops.FULL_SEQUENCE_LENGTHS:
            return super()._forward_fused_adaln(
                hidden_states,
                temb,
                adaln_indices,
                rotary_emb,
                key_padding_mask,
                timestep,
                valid_seq_len,
            )
        modulation = self.adaln_proj.table(temb)
        codes = ops.norm(
            hidden_states,
            self.norm1.weight,
            modulation,
            adaln_indices,
            self._fp8_qkv_reciprocal,
            float(self.norm1.variance_epsilon),
        )
        raw = self.attn(codes, rotary_emb, key_padding_mask, timestep, valid_seq_len)
        output = self.attn.to_out[0]
        hidden_states, codes = ops.attention_projection(
            raw,
            output.weight,
            output.input_scale,
            output.weight_scale,
            hidden_states,
            self.norm2.weight,
            modulation,
            adaln_indices,
            self._fp8_gate_up_reciprocal,
            self._fp8_output_quant_factor,
            float(self.norm2.variance_epsilon),
        )
        gate_up, down = self.ff.gate_up_proj, self.ff.down_proj
        activated = ops.gate_up(
            codes.reshape(-1, codes.shape[-1]),
            self._fp8_packed_gate_up,
            gate_up.input_scale,
            gate_up.weight_scale,
            down.input_scale,
        )
        return ops.down(
            activated,
            down.weight,
            down.input_scale,
            down.weight_scale,
            hidden_states,
            modulation,
            adaln_indices,
        )


@triton.jit
def _reciprocal(scale, inverse):
    tl.store(inverse, tl.div_rn(1.0, tl.load(scale).to(tl.float32)))


class Fp8Transformer(ScenarioTransformer):
    """The H3 transformer with FP8-only blocks and one shared final FP32 cast."""

    block_cls = Fp8Block
    forward = shared_output_forward

    def _projection_input(self, projection, hidden_states: torch.Tensor) -> torch.Tensor:
        if projection is self.proj_out:
            return ops.output_cast(hidden_states)
        return super()._projection_input(projection, hidden_states)

    def initialize_fp8(self, kernel_cache: Path) -> None:
        """Prepare fixed kernels and derived weights before torch.compile and warmup."""
        with torch.cuda.device(self.proj_out.weight.device):
            self._initialize_fp8(kernel_cache)

    def _initialize_fp8(self, kernel_cache: Path) -> None:
        properties = torch.cuda.get_device_properties(self.proj_out.weight.device)
        if (properties.major, properties.minor, properties.multi_processor_count) != (10, 7, 212):
            raise ValueError("The fused FP8 scenario requires an SM107 GPU with 212 SMs")
        blocks = self.transformer_blocks
        if len(blocks) != 50 or int(self.sharder.size) != 1:
            raise ValueError("The fused FP8 scenario requires all 50 blocks on one GPU")
        expected_shapes = ((21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336))
        for block in blocks:
            group = (
                block.attn.qkv_proj,
                block.attn.to_out[0],
                block.ff.gate_up_proj,
                block.ff.down_proj,
            )
            for linear, shape in zip(group, expected_shapes):
                if (
                    not linear.has_fp8_qdq
                    or linear.weight.dtype != torch.float8_e4m3fn
                    or tuple(linear.weight.shape) != shape
                    or not linear.weight.is_contiguous()
                    or linear.bias is not None
                    or linear.input_scale is None
                    or linear.input_scale.numel() != 1
                    or linear.input_scale.dtype != torch.float32
                    or not linear.input_scale.is_contiguous()
                ):
                    raise ValueError(
                        "The fused FP8 scenario requires all 200 main Linears to be static FP8"
                    )
                for scale in (linear.input_scale, linear.weight_scale):
                    if (
                        scale.dtype != torch.float32
                        or scale.numel() != 1
                        or not scale.is_contiguous()
                        or not bool(torch.isfinite(scale).all())
                        or not bool((scale > 0).all())
                    ):
                        raise ValueError(
                            "The fused FP8 scenario needs finite positive per-tensor scales"
                        )
            if (
                not block._fused_adaln
                or block.norm1.weight.dtype != torch.bfloat16
                or block.adaln_proj.linear.dtype != torch.bfloat16
            ):
                raise ValueError("The fused FP8 scenario requires BF16 fused AdaLN")
            attention = block.attn
            if (
                attention.head_dim != 128
                or attention.local_num_attention_heads != attention.local_num_key_value_heads
                or attention.local_num_attention_heads != 56
                or not attention.qk_norm
                or not attention.on_fp8_path()
            ):
                raise ValueError("The fused FP8 attention requires 56 equal Q/K heads of width 128")
            for norm in (attention.norm_q, attention.norm_k):
                if (
                    norm.weight.dtype != torch.bfloat16
                    or norm.weight.numel() != 128
                    or getattr(norm, "allreduce", None) is not None
                    or getattr(norm, "use_gemma", False)
                ):
                    raise ValueError("The fused FP8 attention needs the original BF16 Q/K RMSNorm")
        for projection in (self.proj_out, self.audio_proj_out):
            if projection.has_any_quant or projection.weight.dtype != torch.float32:
                raise ValueError("The shared output cast requires the original FP32 media heads")
        device = self.proj_out.weight.device
        status, capacity = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute(150), cuda.CUdevice(device.index)
        )
        settings = gate_up_swiglu._CFG
        required = 3 * 128 * settings["mma_tile_k"] * settings["num_ab_stage"] + 4096
        if int(status) != 0 or int(capacity) < required:
            raise RuntimeError("The fused GateUp oversized shared-memory capacity is unavailable")
        scales = torch.cat(
            [block.attn.to_out[0].input_scale.detach().reshape(1) for block in blocks]
        )
        _, _, factors, self.fp8_kernel_qualification = prepare(scales, kernel_cache)
        attention_output._CFG = dict(
            tile_n=256,
            cluster_shape_mn=(4, 1),
            num_ab_stage=4,
            swizzle_size=7,
            fallback_cluster_shape_mn=(2, 1),
            mma_tile_k=128,
        )
        for index, block in enumerate(blocks):
            block.attn.qkv_proj.quant_method = Fp8QkvMethod()
            packed = gate_up_swiglu.pack_weight(block.ff.gate_up_proj.weight)
            block.register_buffer("_fp8_packed_gate_up", packed, persistent=False)
            for name, linear in [("qkv", block.attn.qkv_proj), ("gate_up", block.ff.gate_up_proj)]:
                inverse = torch.empty_like(linear.input_scale)
                _reciprocal[(1,)](linear.input_scale, inverse, num_warps=4)
                block.register_buffer(f"_fp8_{name}_reciprocal", inverse, persistent=False)
            block.register_buffer(
                "_fp8_output_quant_factor", factors[index : index + 1], persistent=False
            )
        gate_up_swiglu._get_compiled(device.index)
        self.fp8_conditioning_cache = ConditioningCache(self)
        torch.cuda.synchronize(device)
