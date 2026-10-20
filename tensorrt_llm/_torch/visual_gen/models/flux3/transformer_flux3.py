# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 JointSingleSeq Transformer (Native TRT-LLM).

FLUX.3 image-small is a joint image/text DiT:

- Per-modality "early" blocks (Flux3ModeBlock) are self-attentive/siloed per
  modality; cross-modality fusion happens only in the joint
  Flux3SingleStreamBlock stage (image + reference images + text per sample).
- Modulation is stage-scoped: one Flux3Modulation per modality per stage,
  reused by every block in that stage.
- Timesteps are per-token/per-modality (conditioning tensors may sit at a
  different noise level than the target), not a single scalar.
- Ordinary checkpoints use CFG in the sampling loop. Flow-map checkpoints
  condition every modality on the target time via ref_time_in and run without CFG.
- 4-axis RoPE built as 2x2 rotation matrices (flux3_rope), bridged to the
  (freqs_cos, freqs_sin) the shared/fused attention path expects via
  flux3_pe_to_cos_sin.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tensorrt_llm._torch.distributed.ops import AllReduce
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import (
    FP8RowwiseLinearMethod,
    Fp8RowwiseQuantizedTensor,
    Linear,
    NVFP4LinearMethod,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode, apply_rotary_emb
from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.flux.joint_proj import FluxJointAttnMLPProj
from tensorrt_llm._torch.visual_gen.models.flux3.flux3_core.packed import Bounds, Layout, bounds
from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import (
    Flux3FP8RowwiseLinearMethod,
    quantize_fp8r,
)
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import (
    NVFP4Activation,
    nvfp4_mlp_forward,
    original_input,
    prepare_fp8_block,
    prepare_nvfp4_block,
    separate_qkv_forward,
    shared_nvfp4_input,
    verify_nvfp4_coverage,
)
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import (
    linear_forward as _primsts_linear_forward,
)
from tensorrt_llm._torch.visual_gen.models.flux3.primsts import qkv_forward as _primsts_qkv_forward
from tensorrt_llm._torch.visual_gen.models.flux3.utils_flux3 import (
    Flux3PerTokenAdaLN,
    Flux3PerTokenAdaLNRuntime,
)
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

DEFAULT_IN_CHANNELS: Dict[str, int] = {
    "image": 128,
    "image_cond": 128,
}

DEFAULT_SEQUENCE: Dict[str, str] = {
    "x_image": "image",
    "x_image_cond": "image_cond",
}


# =============================================================================
# RoPE (FLUX.3-local: 2x2 rotation-matrix convention)
# =============================================================================


def flux3_rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    cos, sin = torch.cos(out), torch.sin(out)
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def flux3_pe_to_cos_sin(pe: Tensor) -> Tuple[Tensor, Tensor]:
    """Bridge FLUX.3's 2x2 rotation-matrix pe (..., head_dim/2, 2, 2) into the
    (freqs_cos, freqs_sin) the shared RoPE expects: cos = pe[..., 0, 0],
    sin = pe[..., 1, 0], each repeat-interleaved to head_dim."""
    cos = pe[..., 0, 0]
    sin = pe[..., 1, 0]
    freqs_cos = cos.repeat_interleave(2, dim=-1)
    freqs_sin = sin.repeat_interleave(2, dim=-1)
    return freqs_cos, freqs_sin


def _pe_to_block_freqs(pe: Tensor) -> Tuple[Tensor, Tensor]:
    """pe (B, 1, S, head_dim/2, 2, 2) -> (cos, sin) each (B, S, 1, head_dim),
    broadcastable against the blocks' [B, S, H, head_dim] q/k for the shared
    apply_rotary_emb."""
    cos, sin = flux3_pe_to_cos_sin(pe)
    return cos.transpose(1, 2).contiguous(), sin.transpose(1, 2).contiguous()


def apply_flux3_pag(
    attn: Tensor,
    v: Tensor,
    pag_strength: float,
    sample_bounds: Bounds | None = None,
    pag_samples: tuple[int, ...] = (0,),
) -> Tensor:
    if pag_strength <= 0.0:
        return attn
    if sample_bounds is not None:
        attn = attn.clone()
        values = v.transpose(1, 2).flatten(2)
        for sample in pag_samples:
            start, end = sample_bounds[sample]
            attn[:, start:end] = (1 - pag_strength) * attn[:, start:end] + pag_strength * values[
                :, start:end
            ]
        return attn
    B = attn.shape[0]
    half = B // 2 if B > 1 else B  # B=1 in CFG-parallel uncond rank: treat whole batch as uncond
    b, h, n, d = v.shape
    v_passthrough = v[:half].transpose(1, 2).reshape(half, n, h * d)
    attn = attn.clone()
    if pag_strength >= 1.0:
        attn[:half] = v_passthrough
    else:
        attn[:half] = (1 - pag_strength) * attn[:half] + pag_strength * v_passthrough
    return attn


class Flux3PosEmbed(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [
                flux3_rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(len(self.axes_dim))
            ],
            dim=-3,
        )
        return emb.unsqueeze(1)


def flux3_timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(out_dtype if out_dtype is not None else t.dtype)
    return embedding


# =============================================================================
# Checkpoint key remap
# =============================================================================


# Shared with flux3_lora.py so both encode gate_up_proj's layout once.
GATE_UP_ROLES: Tuple[str, str] = ("gate", "up")


def _remap_checkpoint_keys(weights: dict, fuse_qkv: bool = True) -> dict:
    """Remap checkpoint block-attention keys onto our shared-Attention layout
    (q/k/v_proj -> attn.*, norm.{query,key}_norm.scale -> attn.norm_{q,k}.weight).
    fuse_qkv picks the target Linear name: qkv_proj's params_map sub-keys
    (fused) vs Attention's own to_q/to_k/to_v (SEPARATE_QKV)."""
    q_name, k_name, v_name = (
        ("q_proj", "k_proj", "v_proj") if fuse_qkv else ("to_q", "to_k", "to_v")
    )
    remapped = {}
    for key, value in weights.items():
        new_key = key
        new_key = new_key.replace(".norm.query_norm.scale", ".attn.norm_q.weight")
        new_key = new_key.replace(".norm.key_norm.scale", ".attn.norm_k.weight")
        new_key = new_key.replace(".q_proj.", f".attn.{q_name}.")
        new_key = new_key.replace(".k_proj.", f".attn.{k_name}.")
        new_key = new_key.replace(".v_proj.", f".attn.{v_name}.")
        new_key = new_key.replace(".mlp_out.", ".ff.down_proj.")
        remapped[new_key] = value
    return remapped


def _use_fused_qkv(config: DiffusionModelConfig) -> bool:
    if config.extra_attrs.get("force_unfused_lora_topology"):
        # LoRA into a TP-sharded fused qkv_proj isn't supported yet; TP==1 stays fused.
        tp_size = config.mapping.tp_size if config.mapping else 1
        if tp_size > 1:
            return False
    quant_algo = config.quant_config.quant_algo if config.quant_config else None
    if quant_algo is None or config.dynamic_weight_quant:
        return True  # unquantized, or fuse-then-quantize-once is safe
    return quant_algo != QuantAlgo.NVFP4  # static NVFP4 may have mismatched per-shard scales


def _use_fused_proj_out(config: DiffusionModelConfig) -> bool:
    if config.extra_attrs.get("force_unfused_lora_topology"):
        return False
    quant_algo = config.quant_config.quant_algo if config.quant_config else None
    return quant_algo is None or config.dynamic_weight_quant


def _gated_mlp_in_tp_kwargs(mlp_hidden_dim: int, mapping, tp_size: int) -> dict:
    """TP kwargs for mlp_in (fused gate+up, feeds proj_out); mirrors
    GatedMLP.gate_up_proj's COLUMN + FUSED_GATE_UP_LINEAR sharding."""
    if tp_size <= 1:
        return {}
    tp_rank = mapping.tp_rank
    local_start = Linear._calc_shard(mlp_hidden_dim, tp_size, tp_rank)
    local_end = Linear._calc_shard(mlp_hidden_dim, tp_size, tp_rank + 1)
    local_size = local_end - local_start
    return dict(
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
        fused_weight_shard_indices_mapping={
            "gate": (0, local_size),
            "up": (local_size, local_size),
        },
        override_tp_sharding={"gate": (local_start, local_end), "up": (local_start, local_end)},
        reduce_output=False,
    )


def _fuse_out_proj_weights(weights: dict) -> dict:
    """Concatenate attn_out + mlp_out into one proj_out.weight (bias-free, unquantized only)."""
    out = {}
    consumed = set()
    for key, value in weights.items():
        if key.endswith(".attn_out.weight"):
            prefix = key[: -len(".attn_out.weight")]
            mlp_key = f"{prefix}.ff.down_proj.weight"
            if mlp_key not in weights:
                raise KeyError(f"{key} has no matching {mlp_key} to fuse into proj_out")
            out[f"{prefix}.proj_out.weight"] = torch.cat([value, weights[mlp_key]], dim=1)
            consumed.update([key, mlp_key])
        elif key.endswith(".attn_out.bias") or key.endswith(".ff.down_proj.bias"):
            raise NotImplementedError(
                f"Fused proj_out does not handle out-proj bias ({key}); "
                "FLUX.3 out-projections are bias-free."
            )
    for key, value in weights.items():
        if key not in consumed and key not in out:
            out[key] = value
    return out


def _split_mlp_in_weights(weights: dict) -> dict:
    """Split each mlp_in.weight [gate|up] into ff.gate_up_proj.{gate,up}.weight
    so GatedMLP's FUSED_GATE_UP_LINEAR loader can shard gate and up
    independently and then interleave them per rank."""
    out = {}
    for key, value in weights.items():
        if key.endswith(".mlp_in.weight"):
            prefix = key[: -len(".mlp_in.weight")]
            half = value.shape[0] // 2
            out[f"{prefix}.ff.gate_up_proj.gate.weight"] = value[:half]
            out[f"{prefix}.ff.gate_up_proj.up.weight"] = value[half:]
        elif key.endswith(".mlp_in.weight_scale"):
            prefix = key[: -len(".mlp_in.weight_scale")]
            half = value.shape[0] // 2
            out[f"{prefix}.ff.gate_up_proj.gate.weight_scale"] = value[:half]
            out[f"{prefix}.ff.gate_up_proj.up.weight_scale"] = value[half:]
        elif key.endswith(".mlp_in.bias"):
            prefix = key[: -len(".mlp_in.bias")]
            half = value.shape[0] // 2
            out[f"{prefix}.ff.gate_up_proj.gate.bias"] = value[:half]
            out[f"{prefix}.ff.gate_up_proj.up.bias"] = value[half:]
        elif key.endswith(
            (".mlp_in.weight_scale_2", ".mlp_in.input_scale", ".mlp_in.pre_quant_scale")
        ):
            prefix, suffix = key.rsplit(".mlp_in.", 1)
            out[f"{prefix}.ff.gate_up_proj.gate.{suffix}"] = value
            out[f"{prefix}.ff.gate_up_proj.up.{suffix}"] = value
        else:
            out[key] = value
    return out


def _split_mlp_in_weights_for_fused_proj_out(weights: dict) -> dict:
    """Split mlp_in.weight [gate|up] into mlp_in.{gate,up}.weight for TP>1
    fused-proj_out configs (see _gated_mlp_in_tp_kwargs)."""
    out = {}
    for key, value in weights.items():
        if key.endswith(".mlp_in.weight"):
            prefix = key[: -len(".mlp_in.weight")]
            half = value.shape[0] // 2
            out[f"{prefix}.mlp_in.gate.weight"] = value[:half]
            out[f"{prefix}.mlp_in.up.weight"] = value[half:]
        else:
            out[key] = value
    return out


# =============================================================================
# Building blocks
# =============================================================================


class Flux3MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, config: DiffusionModelConfig):
        super().__init__()
        self.in_layer = Linear(
            in_dim,
            hidden_dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )
        self.silu = nn.SiLU()
        self.out_layer = Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Flux3Modulation(nn.Module):
    """Single modulation: projects a (possibly per-token) vector to shift/scale/gate."""

    def __init__(self, dim: int, config: DiffusionModelConfig):
        super().__init__()
        self.multiplier = 3
        self.lin = Linear(
            dim,
            self.multiplier * dim,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )
        self.act = nn.SiLU()

    def forward(self, vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        out = self.lin(self.act(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        shift, scale, gate = out.chunk(self.multiplier, dim=-1)
        return shift, scale, gate


class Flux3LastLayer(nn.Module):
    def __init__(
        self, hidden_size: int, out_channels: int, config: DiffusionModelConfig, eps: float = 1e-6
    ):
        super().__init__()
        dtype = config.torch_dtype
        self.norm_final = LayerNorm(
            hidden_size=hidden_size, eps=eps, has_weights=False, has_bias=False, dtype=dtype
        )
        self.linear = Linear(
            hidden_size,
            out_channels,
            bias=False,
            dtype=dtype,
            quant_config=config.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            force_dynamic_quantization=config.force_dynamic_quantization,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            Linear(
                hidden_size,
                2 * hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
            ),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        return self.linear(x)


class Flux3Attention(Attention):
    """FLUX.3 self-attention on the shared TRT-LLM Attention: fused QKV, per-head
    QK-norm, FLUX.3 RoPE via the flux3_pe_to_cos_sin bridge. attn_pre_out returns
    the pre-out-projection result (and optionally v) so the block can apply PAG
    and its own output projection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        config: DiffusionModelConfig,
        eps: float = 1e-6,
        bias: bool = False,
        module_name: Optional[str] = None,
    ):
        head_dim = hidden_size // num_heads
        tp_size = config.mapping.tp_size if config.mapping else 1
        fuse_qkv = _use_fused_qkv(config)
        qkv_mode = QKVMode.FUSE_QKV if fuse_qkv else QKVMode.SEPARATE_QKV
        # Fused QK-norm+RoPE kernel needs the packed QKV tensor, head_dim in
        # {64, 128}, and no TP; else eager path.
        fuse_qk_norm_rope = fuse_qkv and head_dim in (64, 128) and tp_size == 1
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            qkv_mode=qkv_mode,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=eps,
            bias=bias,
            interleave=True,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            config=config,
            module_name=module_name,
        )
        # The block applies its own output projection (fused proj_out, or
        # separate attn_out/mlp_out), so drop the base to_out.
        del self.to_out

    def apply_qk_norm(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        q, k = q.float(), k.float()
        q = (
            q
            * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_q.variance_epsilon)
            * self.norm_q.weight.float()
        )
        k = (
            k
            * torch.rsqrt(k.square().mean(-1, keepdim=True) + self.norm_k.variance_epsilon)
            * self.norm_k.weight.float()
        )
        return q, k

    def attn_pre_out(
        self,
        hidden_states: Tensor,
        freqs: Tuple[Tensor, Tensor],
        need_v: bool = False,
        quantized: Fp8RowwiseQuantizedTensor | NVFP4Activation | None = None,
        sample_bounds: Bounds | None = None,
    ):
        """QKV -> QK-norm -> RoPE -> attention, returning the pre-out-projection
        result [B, S, H*D] (and v [B, H, S, D] when need_v, for PAG). Fused path
        does QK-norm+RoPE in one kernel; unfused uses eager rms_norm + rotary."""
        b, s = hidden_states.shape[:2]
        cos, sin = freqs
        projection_input = hidden_states if quantized is None else quantized

        qkv = _primsts_qkv_forward(self, projection_input, freqs)
        if self.fuse_qk_norm_rope or qkv is not None:
            if qkv is None:
                qkv = _primsts_linear_forward(self.qkv_proj, projection_input)
                fc = cos.reshape(-1, self.head_dim).contiguous()
                fs = sin.reshape(-1, self.head_dim).contiguous()
                self.apply_packed_qk_norm_rope(qkv, fc, fs)
            q, k, v = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)
            attn = self._attend(q, k, v, sample_bounds)
            if need_v:
                v = v.view(b, s, self.local_num_key_value_heads, self.head_dim).transpose(1, 2)
                return attn, v  # [B, H, S, D] for apply_flux3_pag
            return attn

        separate = (
            separate_qkv_forward(self, projection_input)
            if self.qkv_mode == QKVMode.SEPARATE_QKV
            else None
        )
        native_input = (
            original_input(projection_input)
            if isinstance(projection_input, NVFP4Activation)
            else projection_input
        )
        q, k, v = separate if separate is not None else self.get_qkv(native_input)
        q = q.view(b, -1, self.local_num_attention_heads, self.head_dim)
        k = k.view(b, -1, self.local_num_attention_heads, self.head_dim)
        v = v.view(b, -1, self.local_num_attention_heads, self.head_dim)
        q, k = self.apply_qk_norm(q, k)
        q = apply_rotary_emb(q, cos, sin).to(v.dtype)
        k = apply_rotary_emb(k, cos, sin).to(v.dtype)
        attn = self._attend(q.flatten(2), k.flatten(2), v.flatten(2), sample_bounds)
        if need_v:
            return attn, v.transpose(1, 2)  # v as [B, H, S, D] for apply_flux3_pag
        return attn

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, sample_bounds: Bounds | None) -> Tensor:
        if sample_bounds is None:
            return self._attn_impl(q, k, v)
        if isinstance(self.attn, VanillaAttention) and q.shape[0] == 1 and len(sample_bounds) > 1:
            length = sample_bounds[0][1]
            if (
                length > 0
                and sample_bounds[-1][1] == q.shape[1]
                and all(
                    (start, end) == (i * length, (i + 1) * length)
                    for i, (start, end) in enumerate(sample_bounds)
                )
            ):
                # Equal packed segments are independent batch rows.
                return self._attn_impl(
                    *(x.view(len(sample_bounds), length, x.shape[-1]) for x in (q, k, v))
                ).reshape_as(q)
        output = torch.zeros_like(q)
        for start, end in sample_bounds:
            if end > start:
                q_sample, k_sample, v_sample = q[:, start:end], k[:, start:end], v[:, start:end]
                # SDPA accepts strided QKV views without copying.
                if not isinstance(self.attn, VanillaAttention):
                    q_sample, k_sample, v_sample = (
                        q_sample.contiguous(),
                        k_sample.contiguous(),
                        v_sample.contiguous(),
                    )
                output[:, start:end] = self._attn_impl(q_sample, k_sample, v_sample)
        return output


def _flux3_shared_fp8_input(
    attn: Flux3Attention, mlp_in: Linear, x: Tensor
) -> Fp8RowwiseQuantizedTensor | NVFP4Activation | None:
    """Share quantized QKV/MLP inputs only when their loaded quantization recipes match."""
    nvfp4 = shared_nvfp4_input(attn, mlp_in, x)
    if nvfp4 is not None:
        return nvfp4
    qkv = getattr(attn, "qkv_proj", None)
    if not all(
        isinstance(layer, Linear) and isinstance(layer.quant_method, FP8RowwiseLinearMethod)
        for layer in (qkv, mlp_in)
    ):
        return None
    flat = x.reshape(-1, x.shape[-1])
    if isinstance(qkv.quant_method, Flux3FP8RowwiseLinearMethod):
        data, scales = quantize_fp8r(flat)
    elif FP8RowwiseLinearMethod.use_tunable_quantize:
        data, scales = torch.ops.trtllm.tunable_fp8_per_token_quant(flat)
    else:
        data, scales = torch.ops.tensorrt_llm.quantize_e4m3_activation(flat)
    return Fp8RowwiseQuantizedTensor(x, data, scales)


def _flux3_gated_mlp_forward(
    ff: GatedMLP, x: torch.Tensor | Fp8RowwiseQuantizedTensor | NVFP4Activation
) -> torch.Tensor:
    """Dispatch PrimsTS or native MLP fusion with the downstream scale contract."""
    nvfp4_output = nvfp4_mlp_forward(ff, x)
    if nvfp4_output is not None:
        return nvfp4_output
    prepared = getattr(ff.gate_up_proj, "_flux3_primsts_fp8", None)
    if prepared is None:
        prepared = getattr(ff.gate_up_proj, "_flux3_primsts_nvfp4", None)
    original = original_input(x)
    if prepared is not None and prepared.swiglu and prepared.can_run(original):
        return _primsts_linear_forward(ff.down_proj, prepared(x))
    if isinstance(x, NVFP4Activation):
        x = original
    return ff(x)


class Flux3ModeBlock(nn.Module):
    """Per-modality siloed self-attention block ("early"/"late" stage)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        config: DiffusionModelConfig,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        eps: float = 1e-6,
        module_name: Optional[str] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        dtype = config.torch_dtype
        mapping = getattr(config, "mapping", None)
        tp_size = getattr(mapping, "tp_size", 1)

        self.attn = Flux3Attention(
            hidden_size,
            num_heads,
            config,
            eps=eps,
            bias=qkv_bias,
            module_name=None if module_name is None else f"{module_name}.attn",
        )
        self._fuse_proj_out = _use_fused_proj_out(config)
        if self._fuse_proj_out:
            self.mlp_in = Linear(
                hidden_size,
                self.mlp_hidden_dim * 2,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **_gated_mlp_in_tp_kwargs(self.mlp_hidden_dim, mapping, tp_size),
            )
            self.proj_out = FluxJointAttnMLPProj(
                attn_dim=hidden_size,
                mlp_dim=self.mlp_hidden_dim,
                out_dim=hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                config=config,
            )
        else:
            use_cute_dsl_swiglu = (
                torch.cuda.is_available()
                and get_sm_version() in (100, 103)
                and config.quant_config is not None
                and config.quant_config.quant_algo == QuantAlgo.NVFP4
            )
            self.ff = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=self.mlp_hidden_dim,
                bias=False,
                dtype=dtype,
                config=config,
                reduce_output=False,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_swiglu,
            )
            out_kwargs = {}
            self._allreduce = None
            if tp_size > 1:
                out_kwargs = dict(
                    mapping=mapping,
                    tensor_parallel_mode=TensorParallelMode.ROW,
                    reduce_output=False,
                )
                self._allreduce = AllReduce(mapping, strategy=config.allreduce_strategy)
            self.attn_out = Linear(
                hidden_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **out_kwargs,
            )
        self.pre_norm = LayerNorm(
            hidden_size=hidden_size, eps=eps, has_weights=False, has_bias=False, dtype=dtype
        )
        self._pertoken_adaln = Flux3PerTokenAdaLN(hidden_size, dtype)

    def forward(
        self,
        x: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Bounds | None = None,
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = self._pertoken_adaln.apply(
            x, mod_shift, mod_scale, self.pre_norm, self.pre_norm.variance_epsilon
        )

        mlp_in = self.mlp_in if self._fuse_proj_out else self.ff.gate_up_proj
        quantized = _flux3_shared_fp8_input(self.attn, mlp_in, x_mod)
        attn = self.attn.attn_pre_out(
            x_mod, freqs, quantized=quantized, sample_bounds=sample_bounds
        )

        if self._fuse_proj_out:
            mlp = self.mlp_in(x_mod if quantized is None else quantized)
            x1, x2 = mlp.chunk(2, dim=-1)
            mlp_act = F.silu(x1) * x2
            output = self.proj_out(attn, mlp_act)
        else:
            mlp_shape = x_mod.shape
            mlp_input = x_mod.reshape(-1, mlp_shape[-1])
            if isinstance(quantized, NVFP4Activation):
                mlp_input = quantized._replace(original=mlp_input)
            elif quantized is not None:
                mlp_input = Fp8RowwiseQuantizedTensor(
                    mlp_input, quantized.quantized, quantized.scales
                )
            mlp_out = _flux3_gated_mlp_forward(self.ff, mlp_input).reshape(*mlp_shape[:-1], -1)
            output = _primsts_linear_forward(self.attn_out, attn) + mlp_out
            if self._allreduce is not None:
                output = self._allreduce(output)
        return x + mod_gate * output


class Flux3SingleStreamBlock(nn.Module):
    """Joint image, reference-image, and text attention within each packed sample."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        config: DiffusionModelConfig,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        module_name: Optional[str] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        dtype = config.torch_dtype
        mapping = getattr(config, "mapping", None)
        tp_size = getattr(mapping, "tp_size", 1)

        self.attn = Flux3Attention(
            hidden_size,
            num_heads,
            config,
            eps=eps,
            bias=False,
            module_name=None if module_name is None else f"{module_name}.attn",
        )
        self._fuse_proj_out = _use_fused_proj_out(config)
        if self._fuse_proj_out:
            self.mlp_in = Linear(
                hidden_size,
                self.mlp_hidden_dim * 2,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **_gated_mlp_in_tp_kwargs(self.mlp_hidden_dim, mapping, tp_size),
            )
            self.proj_out = FluxJointAttnMLPProj(
                attn_dim=hidden_size,
                mlp_dim=self.mlp_hidden_dim,
                out_dim=hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                config=config,
            )
        else:
            use_cute_dsl_swiglu = (
                torch.cuda.is_available()
                and get_sm_version() in (100, 103)
                and config.quant_config is not None
                and config.quant_config.quant_algo == QuantAlgo.NVFP4
            )
            self.ff = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=self.mlp_hidden_dim,
                bias=False,
                dtype=dtype,
                config=config,
                reduce_output=False,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_swiglu,
            )
            out_kwargs = {}
            self._allreduce = None
            if tp_size > 1:
                out_kwargs = dict(
                    mapping=mapping,
                    tensor_parallel_mode=TensorParallelMode.ROW,
                    reduce_output=False,
                )
                self._allreduce = AllReduce(mapping, strategy=config.allreduce_strategy)
            self.attn_out = Linear(
                hidden_size,
                hidden_size,
                bias=False,
                dtype=dtype,
                quant_config=config.quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
                **out_kwargs,
            )
        self.pre_norm = LayerNorm(
            hidden_size=hidden_size, eps=eps, has_weights=False, has_bias=False, dtype=dtype
        )
        self._pertoken_adaln = Flux3PerTokenAdaLN(hidden_size, dtype)

    def forward(
        self,
        combined: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        pag_strength: float = 0.0,
        sample_bounds: Bounds | None = None,
        pag_samples: tuple[int, ...] = (0,),
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        residual = combined

        eps = self.pre_norm.variance_epsilon
        combined = self._pertoken_adaln.apply(combined, mod_shift, mod_scale, self.pre_norm, eps)

        mlp_in = self.mlp_in if self._fuse_proj_out else self.ff.gate_up_proj
        quantized = _flux3_shared_fp8_input(self.attn, mlp_in, combined)
        attn, v = self.attn.attn_pre_out(
            combined, freqs, need_v=True, quantized=quantized, sample_bounds=sample_bounds
        )
        attn = apply_flux3_pag(attn, v, pag_strength, sample_bounds, pag_samples)

        if self._fuse_proj_out:
            mlp = self.mlp_in(combined if quantized is None else quantized)
            x1, x2 = mlp.chunk(2, dim=-1)
            mlp_act = F.silu(x1) * x2
            output = self.proj_out(attn, mlp_act)
        else:
            combined_shape = combined.shape
            mlp_input = combined.reshape(-1, combined_shape[-1])
            if isinstance(quantized, NVFP4Activation):
                mlp_input = quantized._replace(original=mlp_input)
            elif quantized is not None:
                mlp_input = Fp8RowwiseQuantizedTensor(
                    mlp_input, quantized.quantized, quantized.scales
                )
            mlp_out = _flux3_gated_mlp_forward(self.ff, mlp_input).reshape(*combined_shape[:-1], -1)
            output = _primsts_linear_forward(self.attn_out, attn) + mlp_out
            if self._allreduce is not None:
                output = self._allreduce(output)
        return residual + mod_gate * output


# =============================================================================
# Main model
# =============================================================================


class Flux3Model(BaseDiffusionModel):
    """FLUX.3 image DiT with packed modality and joint attention."""

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)

        mapping = getattr(model_config, "mapping", None)
        self._tp_size = mapping.tp_size if mapping is not None else 1

        pretrained_config = model_config.pretrained_config
        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        in_channels = dict(getattr(pretrained_config, "in_channels", DEFAULT_IN_CHANNELS))
        sequence = dict(getattr(pretrained_config, "sequence", DEFAULT_SEQUENCE))
        context_in_dim = getattr(pretrained_config, "context_in_dim", 20480)
        hidden_size = getattr(pretrained_config, "hidden_size", 3072)
        num_heads = getattr(pretrained_config, "num_heads", 24)
        depth = getattr(pretrained_config, "depth", 5)
        depth_single_blocks = getattr(pretrained_config, "depth_single_blocks", 28)
        axes_dim = list(getattr(pretrained_config, "axes_dim", [32, 32, 32, 32]))
        theta = getattr(pretrained_config, "theta", 10000)
        mlp_ratio = getattr(pretrained_config, "mlp_ratio", 3.0)
        qkv_bias = getattr(pretrained_config, "qkv_bias", False)
        eps = getattr(pretrained_config, "eps", 1e-6)

        if getattr(pretrained_config, "vec_in_dim", None) is not None or getattr(
            pretrained_config, "depth_late_blocks", 0
        ):
            raise ValueError("FLUX.3 image-small requires no pooled vector or late blocks")

        for _, mod_key in sequence.items():
            assert mod_key in in_channels, (sequence, in_channels)

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")

        self.in_channels = in_channels
        self.sequence = sequence
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks

        block_kwargs = dict(
            config=model_config,
            mlp_ratio=mlp_ratio,
            eps=eps,
        )

        self.pe_embedder = Flux3PosEmbed(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.emb_in = nn.ModuleDict(
            {
                m: Linear(
                    c,
                    hidden_size,
                    bias=False,
                    dtype=dtype,
                    quant_config=quant_config,
                    skip_create_weights_in_init=skip_create_weights,
                    force_dynamic_quantization=force_dynamic_quant,
                )
                for m, c in in_channels.items()
            }
        )
        self.txt_in = Linear(
            context_in_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
        )

        self.time_in = Flux3MLPEmbedder(256, hidden_size, config=model_config)
        self.flow_map = getattr(pretrained_config, "flow_map", False)
        if self.flow_map:
            self.ref_time_in = Flux3MLPEmbedder(256, hidden_size, config=model_config)
        content_modalities = sorted(in_channels.keys())

        def _make_modulations() -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    **{
                        m: Flux3Modulation(hidden_size, config=model_config)
                        for m in content_modalities
                    },
                    "txt": Flux3Modulation(hidden_size, config=model_config),
                }
            )

        self.early_stream_modulations = _make_modulations()
        self.single_stream_modulations = _make_modulations()

        self.content_mode_blocks = nn.ModuleDict(
            {
                m: nn.ModuleList(
                    [
                        Flux3ModeBlock(hidden_size, num_heads, qkv_bias=qkv_bias, **block_kwargs)
                        for _ in range(depth)
                    ]
                )
                for m in content_modalities
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                Flux3ModeBlock(hidden_size, num_heads, qkv_bias=qkv_bias, **block_kwargs)
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                Flux3SingleStreamBlock(hidden_size, num_heads, **block_kwargs)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = nn.ModuleDict(
            {
                m: Flux3LastLayer(hidden_size, c, config=model_config, eps=eps)
                for m, c in in_channels.items()
                if m in getattr(pretrained_config, "out_modes", ["image"])
            }
        )

        vgm = getattr(model_config, "visual_gen_mapping", None)
        self.sharder = SequenceSharder.from_vgm(vgm, num_attention_heads=num_heads)

        self._early_adaln_runtime = {
            name: Flux3PerTokenAdaLNRuntime([block._pertoken_adaln for block in blocks])
            for name, blocks in {
                "text": self.txt_mode_blocks,
                **self.content_mode_blocks,
            }.items()
        }
        self._single_adaln_runtime = Flux3PerTokenAdaLNRuntime(
            [block._pertoken_adaln for block in self.single_blocks]
        )

        self.__post_init__()

    def __post_init__(self):
        quant = self.model_config.quant_config
        if quant is not None and quant.exclude_modules:
            excluded = _remap_checkpoint_keys(
                {name + ".weight": None for name in quant.exclude_modules},
                fuse_qkv=_use_fused_qkv(self.model_config),
            )
            quant.exclude_modules = sorted(
                {
                    re.sub(r"\.attn\.[qkv]_proj\.", ".attn.qkv_proj.", name)
                    .replace(".mlp_in.", ".ff.gate_up_proj.")
                    .removesuffix(".weight")
                    for name in excluded
                }
            )
        self.apply_quant_config_exclude_modules()
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def apply_quant_config_exclude_modules(self):
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        kv_cache_quant_algo = quant_config.kv_cache_quant_algo if quant_config else None
        no_quant_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                is_excluded = quant_config.is_module_excluded_from_quantization(name)
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = no_quant_config

    def register_cuda_graph_extra_key_fns(self, runner) -> None:
        """Include sample boundaries and PAG settings in graph keys."""
        super().register_cuda_graph_extra_key_fns(runner)

        def _pag_key(*args, **kwargs):
            layers = kwargs.get("pag_single_layers")
            strength = kwargs.get("pag_strength", 1.0)
            # None when PAG is inactive -> runner omits this key part.
            if not layers or not strength:
                return None
            return (tuple(layers), float(strength))

        runner.register_extra_key_fn("flux3_pag", _pag_key)
        runner.register_extra_key_fn(
            "flux3_layout",
            lambda *args, **kwargs: (kwargs.get("layout"), kwargs.get("pag_samples", (0,))),
        )

    def _packed_inputs(
        self, inputs: dict, layout: Layout | None
    ) -> tuple[dict, Layout, int | None]:
        if layout is not None:
            return inputs, layout, None
        # Dense component callers enter the same packed execution path.
        ctx = inputs["ctx"]
        batch = ctx.shape[0]
        packed = {key: value for key, value in inputs.items() if key.startswith("_flux3_")}
        lengths = {"text": (ctx.shape[1],) * batch}
        for field, value in (
            ("", ctx),
            ("_ids", inputs["ctx_ids"]),
            ("_timesteps", inputs["timesteps_ctx"]),
        ):
            packed["x_text" + field] = value.flatten(0, 1)
        for name in self.in_channels:
            key = "x_" + name
            length = inputs[key].shape[1] if key in inputs else 0
            lengths[name] = (length,) * batch
            if key in inputs:
                for suffix in ("", "_ids", "_timesteps"):
                    packed[key + suffix] = inputs[key + suffix].flatten(0, 1)
            else:
                packed[key] = ctx.new_empty(0, self.in_channels[name])
                packed[key + "_ids"] = inputs["ctx_ids"].new_empty(0, 4)
                packed[key + "_timesteps"] = ctx.new_empty(0)
        return packed, tuple(sorted(lengths.items())), batch

    def _local_layout(self, layout: Layout) -> Layout:
        size = self.sharder.size
        if any(length % size for _, lengths in layout for length in lengths):
            raise ValueError(
                "FLUX.3 sequence parallelism requires each sample's modality length "
                "to be divisible by the sequence-parallel size"
            )
        return tuple(
            (name, tuple(length // size for length in lengths)) for name, lengths in layout
        )

    def _shard_packed(self, value: Tensor, lengths: tuple[int, ...], dim: int) -> Tensor:
        if not self.sharder.is_active or not sum(lengths):
            return value
        return torch.cat(
            [
                self.sharder.shard(value.narrow(dim, start, end - start), dim=dim)
                for start, end in bounds(lengths)
                if end > start
            ],
            dim=dim,
        )

    def _packed_stream(
        self, name: str, inputs: dict, lengths: tuple[int, ...], ref: Tensor | None = None
    ) -> dict[str, Tensor]:
        tokens, ts = inputs["x_" + name], inputs["x_" + name + "_timesteps"]
        if not tokens.shape[0]:
            tokens, ts = tokens.new_zeros(1, tokens.shape[-1]), ts.new_zeros(1)
        # Compact only when the pipeline guarantees one timestep per sample/stream.
        compact = inputs.get("_flux3_uniform_timesteps", False) and any(lengths)
        segments = [
            (index, start, end) for index, (start, end) in enumerate(bounds(lengths)) if end > start
        ]
        timestep_rows = torch.stack([ts[start] for _, start, _ in segments]) if compact else ts
        vec = self.time_in(
            flux3_timestep_embedding(timestep_rows, 256, out_dtype=tokens.dtype)
        ).unsqueeze(0)
        if ref is not None and any(lengths):
            ref_rows = (
                torch.stack([ref[index] for index, _, _ in segments])
                if compact
                else torch.cat(
                    [
                        ref[index].expand(length, -1)
                        for index, length in enumerate(lengths)
                        if length
                    ]
                )
            )
            vec = vec + ref_rows.unsqueeze(0)
        key = "txt" if name == "text" else name
        embedded = (self.txt_in if name == "text" else self.emb_in[name])(tokens).unsqueeze(0)
        state = {"tokens": embedded, "vec": vec}
        for stage, modules in (
            ("early", self.early_stream_modulations),
            ("single", self.single_stream_modulations),
        ):
            for field, value in zip(("shift", "scale", "gate"), modules[key](vec)):
                state[stage + "_" + field] = value
        if compact:
            for field, value in state.items():
                if field != "tokens":
                    state[field] = torch.cat(
                        [
                            value[:, row : row + 1].expand(-1, end - start, -1)
                            for row, (_, start, end) in enumerate(segments)
                        ],
                        dim=1,
                    )
        return {
            field: value if field == "vec" else self._shard_packed(value, lengths, 1)
            for field, value in state.items()
        }

    @staticmethod
    def _joint_chunks(layout: Layout) -> tuple[list[tuple[str, int, int]], Bounds]:
        ranges = {name: bounds(lengths) for name, lengths in layout}
        chunks, lengths = [], []
        for sample in range(len(layout[0][1])):
            length = 0
            for name, _ in layout:
                start, end = ranges[name][sample]
                if end > start:
                    chunks.append((name, start, end))
                    length += end - start
            lengths.append(length)
        return chunks, bounds(tuple(lengths))

    def _packed_positions(self, inputs: dict, layout: Layout) -> dict[str, Tensor]:
        prepared, positions = {}, {}
        for name, lengths in layout:
            if name.endswith("_cond") and not any(lengths):
                continue
            ids = inputs["x_" + name + "_ids"]
            if not ids.shape[0]:
                ids = ids.new_zeros(1, 4)
            pe = self.pe_embedder(ids.unsqueeze(0))
            pe = self._shard_packed(pe, lengths, 2)
            positions[name] = pe
            prepared[f"_flux3_{name}_cos"], prepared[f"_flux3_{name}_sin"] = _pe_to_block_freqs(pe)
        chunks, _ = self._joint_chunks(self._local_layout(layout))
        pe = torch.cat([positions[name][:, :, start:end] for name, start, end in chunks], dim=2)
        prepared["_flux3_joint_cos"], prepared["_flux3_joint_sin"] = _pe_to_block_freqs(pe)
        return prepared

    def prepare_static_inputs(self, *, layout: Layout | None = None, **inputs) -> dict[str, Tensor]:
        if torch.is_grad_enabled():
            return {}
        inputs, layout, _ = self._packed_inputs(inputs, layout)
        local_lengths = dict(self._local_layout(layout))
        prepared = self._packed_positions(inputs, layout)
        if self.flow_map:
            return prepared
        for name, lengths in layout:
            if name != "text" and not name.endswith("_cond"):
                continue
            if name.endswith("_cond") and not any(lengths):
                continue
            state = self._packed_stream(name, inputs, lengths)
            blocks = self.txt_mode_blocks if name == "text" else self.content_mode_blocks[name]
            freqs = prepared[f"_flux3_{name}_cos"], prepared[f"_flux3_{name}_sin"]
            mod = tuple(state["early_" + field] for field in ("shift", "scale", "gate"))
            state["tokens"] = self._early_adaln_runtime[name].prepare(state["tokens"], *mod[:2])
            for block in blocks:
                state["tokens"] = block(
                    state["tokens"], freqs, mod, sample_bounds=bounds(local_lengths[name])
                )
            prepared.update(
                {
                    f"_flux3_{name}_{field}": value
                    for field, value in state.items()
                    if not field.startswith("early_")
                }
            )
        return prepared

    def forward(
        self,
        *,
        layout: Layout | None = None,
        timesteps_ref: Tensor | None = None,
        pag_single_layers: list[int] | None = None,
        pag_strength: float = 1.0,
        pag_samples: tuple[int, ...] = (0,),
        **inputs,
    ) -> dict[str, Tensor]:
        inputs, layout, batch = self._packed_inputs(inputs, layout)
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("timesteps_ref is required by, and only by, a flow-map DiT")
        ref = None
        if timesteps_ref is not None:
            num_samples = len(layout[0][1])
            if timesteps_ref.shape != (num_samples,):
                raise ValueError(f"timesteps_ref must have shape ({num_samples},)")
            target = timesteps_ref.to(inputs["x_text_timesteps"])
            ref = self.ref_time_in(flux3_timestep_embedding(target, 256, out_dtype=target.dtype))
        local_layout = self._local_layout(layout)
        local_lengths = dict(local_layout)
        prepared = (
            inputs if "_flux3_joint_cos" in inputs else self._packed_positions(inputs, layout)
        )
        streams = {}
        for name, lengths in layout:
            if name.endswith("_cond") and not any(lengths):
                continue
            prefix = f"_flux3_{name}_"
            if not self.flow_map and prefix + "tokens" in inputs:
                streams[name] = {
                    field: inputs[prefix + field]
                    for field in ("tokens", "vec", "single_shift", "single_scale", "single_gate")
                }
                continue
            state = self._packed_stream(name, inputs, lengths, ref)
            blocks = self.txt_mode_blocks if name == "text" else self.content_mode_blocks[name]
            freqs = prepared[prefix + "cos"], prepared[prefix + "sin"]
            mod = tuple(state["early_" + field] for field in ("shift", "scale", "gate"))
            state["tokens"] = self._early_adaln_runtime[name].prepare(state["tokens"], *mod[:2])
            for block in blocks:
                state["tokens"] = block(
                    state["tokens"], freqs, mod, sample_bounds=bounds(local_lengths[name])
                )
            streams[name] = state

        chunks, sample_bounds = self._joint_chunks(local_layout)

        def joint(field: str) -> Tensor:
            return torch.cat(
                [streams[name][field][:, start:end] for name, start, end in chunks], dim=1
            )

        combined = joint("tokens")
        mod = tuple(joint("single_" + field) for field in ("shift", "scale", "gate"))
        freqs = prepared["_flux3_joint_cos"], prepared["_flux3_joint_sin"]
        combined = self._single_adaln_runtime.prepare(combined, *mod[:2])
        for index, block in enumerate(self.single_blocks):
            combined = block(
                combined,
                freqs,
                mod,
                pag_strength=pag_strength
                if pag_single_layers and index in pag_single_layers
                else 0.0,
                sample_bounds=sample_bounds,
                pag_samples=pag_samples,
            )
        outputs = {}
        for name in self.final_layer:
            parts, offset = [], 0
            for mode, start, end in chunks:
                if mode == name:
                    part = combined[:, offset : offset + end - start].contiguous()
                    if self.sharder.is_active:
                        part = self.sharder.gather(part, dim=1)
                    parts.append(part)
                offset += end - start
            tokens = torch.cat(parts, dim=1)
            output = self.final_layer[name](tokens, streams[name]["vec"])[0]
            outputs["x_" + name] = (
                output.reshape(batch, -1, output.shape[-1]) if batch is not None else output
            )
        return outputs

    def load_weights(self, weights: dict) -> None:
        fuse_qkv = _use_fused_qkv(self.model_config)
        weights = _remap_checkpoint_keys(weights, fuse_qkv=fuse_qkv)
        if _use_fused_proj_out(self.model_config):
            weights = _fuse_out_proj_weights(weights)

        params_map = {}
        if fuse_qkv:
            # Checkpoint stores q/k/v separately; fuse into qkv_proj.
            params_map["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]
        tp_size = getattr(self.model_config.mapping, "tp_size", 1)
        if not _use_fused_proj_out(self.model_config):
            # GatedMLP.gate_up_proj always uses FUSED_GATE_UP_LINEAR mode, so
            # split mlp_in.weight [gate|up] into ff.gate_up_proj.{gate,up}.
            weights = _split_mlp_in_weights(weights)
            params_map["gate_up_proj"] = [f"gate_up_proj.{role}" for role in GATE_UP_ROLES]
        elif tp_size > 1:
            # mlp_in stays column-parallel FUSED_GATE_UP_LINEAR under TP even
            # when proj_out is fused (see _gated_mlp_in_tp_kwargs).
            weights = _split_mlp_in_weights_for_fused_proj_out(weights)
            params_map["mlp_in"] = [f"mlp_in.{role}" for role in GATE_UP_ROLES]

        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)
        managed_prefixes: set = set()

        for name, module in self.named_modules():
            if any(name.startswith(p) for p in managed_prefixes):
                continue

            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

            if isinstance(module, FluxJointAttnMLPProj):
                managed_prefixes.add(name + ".")
                module_weights = loader.filter_weights(name, weights)
                module.load_weights(module_weights, loader)
                continue

            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    if type(module.quant_method) is NVFP4LinearMethod and name.startswith(
                        ("content_mode_blocks.image.", "single_blocks.")
                    ):
                        module._flux3_nvfp4_missing_input_scale = all(
                            "input_scale" not in shard for shard in weight_dicts
                        )
                    loader.load_linear_weights(module, name, weight_dicts)
            else:
                module_weights = loader.filter_weights(name, weights)
                target_dtype = self.model_config.torch_dtype
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(module_weights[param_name].to(target_dtype))

    def post_load_weights(self) -> None:
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
                if type(module.quant_method) is FP8RowwiseLinearMethod:
                    module.quant_method = Flux3FP8RowwiseLinearMethod()
        self.prepare_primsts_weights()

    def prepare_primsts_weights(self) -> None:
        """Refresh image/joint PrimsTS layouts after loading or merging LoRA weights."""
        if "image" in self.content_mode_blocks:
            for block in self.content_mode_blocks["image"]:
                prepare_fp8_block(block)
                prepare_nvfp4_block(block)
        for block in self.single_blocks:
            prepare_fp8_block(block)
            prepare_nvfp4_block(block)
        blocks = (
            [
                (f"content_mode_blocks.image.{i}", block)
                for i, block in enumerate(self.content_mode_blocks["image"])
            ]
            if "image" in self.content_mode_blocks
            else []
        )
        blocks.extend((f"single_blocks.{i}", block) for i, block in enumerate(self.single_blocks))
        verify_nvfp4_coverage(blocks)
