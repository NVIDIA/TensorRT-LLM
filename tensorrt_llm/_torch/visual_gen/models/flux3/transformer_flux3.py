# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 transformer with Diffusers module names.

Per-mode blocks run within each modality; single blocks then attend jointly within each sample.
"""

import dataclasses
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tensorrt_llm._torch.distributed.ops import AllReduce
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode, WeightMode
from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode, apply_rotary_emb
from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import _split_linear_in_weights
from tensorrt_llm._torch.visual_gen.models.flux3.packed import (
    NUM_POSITION_AXES,
    Bounds,
    Layout,
    bounds,
)
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm._utils import is_blackwell_datacenter
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

IMAGE = "image"
IMAGE_COND = "image_cond"
TEXT = "text"
# Modalities of a packed layout, in the sorted order packing produces.
PACKED_MODALITIES = (IMAGE, IMAGE_COND, TEXT)
CONTENT_MODALITIES = (IMAGE, IMAGE_COND)
OPTIONAL_MODALITIES = frozenset({IMAGE_COND})
# Only image tokens are decoded; reference and text tokens produce no output.
OUTPUT_MODALITY = IMAGE
# Callers scale the h and w position ids of target and reference image tokens by this factor.
IMAGE_ROPE_SCALE = 0.5

# Checkpoint sources of the fused Linears; linear_in is pre-split into gate and up halves.
_PARAMS_MAP = {
    "qkv_proj": ["to_q", "to_k", "to_v"],
    "gate_up_proj": ["linear_in_gate", "linear_in_up"],
}
# Checkpoint module names of the Linears whose TRT-LLM names differ.
_CHECKPOINT_MODULES = {
    "qkv_proj": ("to_q", "to_k", "to_v"),
    "gate_up_proj": ("linear_in",),
    "down_proj": ("linear_out",),
}
# Modules that load-time quantization keeps in BF16, as BFL's FP8 rowwise export does.
LOAD_TIME_QUANT_EXCLUDED_MODULES = (
    "x_embedders",
    "context_embedder",
    "timestep_embedder",
    "target_timestep_embedder",
    "txt_mode_blocks",
    "norm_out",
    "proj_out",
)
# Config keys added by the loader rather than the checkpoint architecture.
_NON_ARCHITECTURE_CONFIG_KEYS = frozenset({"quantization_config"})
# Stream tensors the single blocks read; the prepared text and reference streams hold these.
_JOINT_FIELDS = ("tokens", "single_shift", "single_scale", "single_gate")


@dataclasses.dataclass(frozen=True)
class Flux3TransformerConfig:
    """FLUX.3 architecture, read strictly from ``transformer/config.json``."""

    attention_head_dim: int
    axes_dims_rope: Tuple[int, ...]
    cond_in_channels: int
    eps: float
    in_channels: int
    joint_attention_dim: int
    mlp_ratio: float
    num_attention_heads: int
    num_layers: int
    num_single_layers: int
    rope_theta: float
    timestep_channels: int
    flow_map: bool = False

    @classmethod
    def from_pretrained_config(cls, pretrained_config: Any) -> "Flux3TransformerConfig":
        """Build from a Diffusers config namespace; unknown or missing keys raise ValueError."""
        if pretrained_config is None:
            raise ValueError("FLUX.3 transformer needs the checkpoint's transformer/config.json")
        values = {
            key: value
            for key, value in vars(pretrained_config).items()
            if not key.startswith("_") and key not in _NON_ARCHITECTURE_CONFIG_KEYS
        }
        fields = dataclasses.fields(cls)
        missing = sorted(
            field.name
            for field in fields
            if field.default is dataclasses.MISSING and field.name not in values
        )
        unknown = sorted(values.keys() - {field.name for field in fields})
        if missing or unknown:
            raise ValueError(
                f"FLUX.3 transformer config: missing keys {missing}, unsupported keys {unknown}"
            )
        values["axes_dims_rope"] = tuple(values["axes_dims_rope"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        """Check the relations between config values that the model relies on."""
        axes = self.axes_dims_rope
        if len(axes) != NUM_POSITION_AXES or any(dim % 2 for dim in axes):
            raise ValueError(f"FLUX.3 needs {NUM_POSITION_AXES} even RoPE axis dims, got {axes}")
        if sum(axes) != self.attention_head_dim:
            raise ValueError(
                f"FLUX.3 RoPE axis dims {axes} must sum to attention_head_dim "
                f"{self.attention_head_dim}"
            )

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)


# =============================================================================
# RoPE and timestep embedding
# =============================================================================


def flux3_rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    """2x2 RoPE rotation matrices [..., dim / 2, 2, 2] (fp32) for positions ``pos``."""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    cos, sin = torch.cos(out), torch.sin(out)
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def flux3_pe_to_cos_sin(pe: Tensor) -> Tuple[Tensor, Tensor]:
    """Rotation matrices [..., head_dim / 2, 2, 2] to interleaved (cos, sin) [..., head_dim]."""
    cos = pe[..., 0, 0]
    sin = pe[..., 1, 0]
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)


def _pe_to_block_freqs(pe: Tensor) -> Tuple[Tensor, Tensor]:
    """pe [B, 1, S, head_dim / 2, 2, 2] to (cos, sin), each [B, S, 1, head_dim]."""
    cos, sin = flux3_pe_to_cos_sin(pe)
    return cos.transpose(1, 2).contiguous(), sin.transpose(1, 2).contiguous()


def flux3_timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Sinusoidal [cos, sin] embedding [N, dim] of timesteps ``t`` [N] in [0, 1]."""
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


def _select_rows(value: Tensor, rows: List[int]) -> Tensor:
    """Rows of a per-sample tensor, indexed on the host so no index tensor is copied."""
    if rows == list(range(value.shape[0])):
        return value
    return torch.stack([value[row] for row in rows])


def _expand_rows(value: Tensor, lengths: List[int]) -> Tensor:
    """Rows [n, hidden] repeated over each sample's tokens, as [1, sum(lengths), hidden]."""
    return torch.cat([row.expand(length, -1) for row, length in zip(value, lengths)]).unsqueeze(0)


class Flux3PosEmbed(nn.Module):
    """RoPE rotation matrices for (t, h, w, l) position ids."""

    def __init__(self, theta: float, axes_dim: Tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """ids [B, S, 4] to rotation matrices [B, 1, S, head_dim / 2, 2, 2]."""
        emb = torch.cat(
            [flux3_rope(ids[..., i], dim, self.theta) for i, dim in enumerate(self.axes_dim)],
            dim=-3,
        )
        return emb.unsqueeze(1)


# =============================================================================
# Building blocks
# =============================================================================


def _linear_kwargs(config: DiffusionModelConfig) -> Dict[str, Any]:
    return dict(
        bias=False,
        dtype=config.torch_dtype,
        quant_config=config.quant_config,
        skip_create_weights_in_init=config.skip_create_weights_in_init,
        force_dynamic_quantization=config.force_dynamic_quantization,
    )


def _use_fused_qkv(config: DiffusionModelConfig, attn_name: str) -> bool:
    """Fused QKV, except for static NVFP4 projections, whose q/k/v scales may differ."""
    quant_config = config.quant_config
    if quant_config is None or quant_config.quant_algo != QuantAlgo.NVFP4:
        return True
    if config.dynamic_weight_quant:
        return True
    return all(
        quant_config.is_module_excluded_from_quantization(f"{attn_name}.{name}")
        for name in ("to_q", "to_k", "to_v")
    )


class Flux3TimestepEmbedder(nn.Module):
    """Two-layer SiLU MLP over a sinusoidal timestep embedding."""

    def __init__(self, in_channels: int, hidden_size: int, config: DiffusionModelConfig):
        super().__init__()
        self.linear_1 = Linear(in_channels, hidden_size, **_linear_kwargs(config))
        self.linear_2 = Linear(hidden_size, hidden_size, **_linear_kwargs(config))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class Flux3Modulation(nn.Module):
    """Projects per-sample conditioning rows [n, hidden] to shift, scale and gate rows."""

    def __init__(self, hidden_size: int, config: DiffusionModelConfig):
        super().__init__()
        self.linear = Linear(hidden_size, 3 * hidden_size, **_linear_kwargs(config))

    def forward(self, vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        shift, scale, gate = self.linear(F.silu(vec)).chunk(3, dim=-1)
        return shift, scale, gate


class Flux3AdaLayerNorm(nn.Module):
    """Output adaptive LayerNorm; its projection rows are ordered scale, then shift."""

    def __init__(self, hidden_size: int, eps: float, config: DiffusionModelConfig):
        super().__init__()
        self.linear = Linear(hidden_size, 2 * hidden_size, **_linear_kwargs(config))
        self.norm = LayerNorm(
            hidden_size=hidden_size,
            eps=eps,
            has_weights=False,
            has_bias=False,
            dtype=config.torch_dtype,
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """Modulate x [B, S, hidden] with one conditioning row per batch entry, vec [B, hidden]."""
        scale, shift = self.linear(F.silu(vec)).unsqueeze(1).chunk(2, dim=-1)
        return (1 + scale) * self.norm(x) + shift


class Flux3Attention(Attention):
    """FLUX.3 self-attention with per-head QK-norm and RoPE, attending within packed samples."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float,
        config: DiffusionModelConfig,
        layer_idx: int,
        module_name: str,
    ):
        head_dim = hidden_size // num_heads
        fuse_qkv = _use_fused_qkv(config, module_name)
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            qkv_mode=QKVMode.FUSE_QKV if fuse_qkv else QKVMode.SEPARATE_QKV,
            # Separate Q/K/V is still self-attention, so the requested backend is kept.
            separate_qkv_is_self_attention=True,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=eps,
            bias=False,
            interleave=True,
            # The fused QK-norm + RoPE kernel reads packed QKV with head_dim 64 or 128.
            fuse_qk_norm_rope=fuse_qkv and head_dim in (64, 128),
            config=config,
            layer_idx=layer_idx,
            module_name=module_name,
        )
        if self.tp_size > 1:
            # The block all-reduces the attention and MLP partial sums together.
            self.to_out = nn.ModuleList(
                [
                    Linear(
                        self.q_dim,
                        self.hidden_size,
                        bias=False,
                        dtype=self.dtype,
                        mapping=self.mapping,
                        quant_config=self.quant_config,
                        skip_create_weights_in_init=self.skip_create_weights_in_init,
                        force_dynamic_quantization=self.force_dynamic_quantization,
                        tensor_parallel_mode=TensorParallelMode.ROW,
                        reduce_output=False,
                        override_tp_sharding=(self.local_q_dim_start, self.local_q_dim_end),
                    )
                ]
            )

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

    def attend(
        self,
        hidden_states: Tensor,
        freqs: Tuple[Tensor, Tensor],
        sample_bounds: Optional[Bounds] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """Attention before the output projection: [B, S, hidden] to [B, S, local heads * head_dim].

        ``freqs`` are RoPE (cos, sin), each [B, S, 1, head_dim]; ``sample_bounds`` limits attention.
        """
        b = hidden_states.shape[0]
        cos, sin = freqs
        if self.fuse_qk_norm_rope:
            qkv = self.qkv_proj(hidden_states)
            self.apply_packed_qk_norm_rope(
                qkv,
                cos.reshape(-1, self.head_dim).contiguous(),
                sin.reshape(-1, self.head_dim).contiguous(),
            )
            q, k, v = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)
            return self._attend(q, k, v, sample_bounds, timestep)

        q, k, v = self.get_qkv(hidden_states)
        q = q.view(b, -1, self.local_num_attention_heads, self.head_dim)
        k = k.view(b, -1, self.local_num_key_value_heads, self.head_dim)
        v = v.view(b, -1, self.local_num_key_value_heads, self.head_dim)
        q, k = self.apply_qk_norm(q, k)
        q = apply_rotary_emb(q, cos, sin).to(v.dtype)
        k = apply_rotary_emb(k, cos, sin).to(v.dtype)
        return self._attend(q.flatten(2), k.flatten(2), v.flatten(2), sample_bounds, timestep)

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sample_bounds: Optional[Bounds],
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_bounds is None:
            return self._attn_impl(q, k, v, timestep=timestep)
        if q.shape[0] == 1 and len(sample_bounds) > 1:
            length = sample_bounds[0][1]
            if (
                length > 0
                and sample_bounds[-1][1] == q.shape[1]
                and all(
                    (start, end) == (i * length, (i + 1) * length)
                    for i, (start, end) in enumerate(sample_bounds)
                )
            ):
                # Equal packed segments are independent batch rows for every backend.
                return self._attn_impl(
                    *(x.view(len(sample_bounds), length, x.shape[-1]) for x in (q, k, v)),
                    timestep=timestep,
                ).reshape_as(q)
        # HND backends take strided [B, H, S, D] views; the others need contiguous slices.
        layout = getattr(self.attn, "preferred_layout", AttentionTensorLayout.NHD)
        copy_slices = layout != AttentionTensorLayout.HND
        output = torch.zeros_like(q)
        for start, end in sample_bounds:
            if end > start:
                q_sample, k_sample, v_sample = q[:, start:end], k[:, start:end], v[:, start:end]
                if copy_slices:
                    q_sample, k_sample, v_sample = (
                        q_sample.contiguous(),
                        k_sample.contiguous(),
                        v_sample.contiguous(),
                    )
                output[:, start:end] = self._attn_impl(
                    q_sample, k_sample, v_sample, timestep=timestep
                )
        return output


class _Flux3Block(nn.Module):
    """Parallel attention and gated MLP on one modulated input, with a gated residual."""

    def __init__(
        self,
        arch: Flux3TransformerConfig,
        config: DiffusionModelConfig,
        module_name: str,
        layer_idx: int,
    ):
        super().__init__()
        hidden_size = arch.hidden_size
        mapping = config.mapping
        tp_size = mapping.tp_size if mapping is not None else 1
        quant_algo = config.quant_config.quant_algo if config.quant_config is not None else None

        self.norm = LayerNorm(
            hidden_size=hidden_size,
            eps=arch.eps,
            has_weights=False,
            has_bias=False,
            dtype=config.torch_dtype,
        )
        self.attn = Flux3Attention(
            hidden_size,
            arch.num_attention_heads,
            arch.eps,
            config,
            layer_idx=layer_idx,
            module_name=f"{module_name}.attn",
        )
        self.ff = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=arch.mlp_hidden_dim,
            bias=False,
            dtype=config.torch_dtype,
            config=config,
            reduce_output=False,
            layer_idx=layer_idx,
            use_cute_dsl_blockscaling_mm=(
                quant_algo == QuantAlgo.NVFP4
                and torch.cuda.is_available()
                and is_blackwell_datacenter()
            ),
        )
        self._allreduce = (
            AllReduce(mapping, strategy=config.allreduce_strategy) if tp_size > 1 else None
        )

    def _update(
        self,
        x: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Optional[Bounds],
        timestep: Optional[Tensor],
    ) -> Tensor:
        shift, scale, gate = mod
        x_mod = (1 + scale) * self.norm(x) + shift
        attn = self.attn.attend(x_mod, freqs, sample_bounds=sample_bounds, timestep=timestep)
        mlp_out = self.ff(x_mod.reshape(-1, x_mod.shape[-1])).reshape(*x_mod.shape[:-1], -1)
        output = self.attn.to_out[0](attn) + mlp_out
        if self._allreduce is not None:
            output = self._allreduce(output)
        return x + gate * output


class Flux3ModeTransformerBlock(_Flux3Block):
    """Per-mode block: tokens attend only within their own modality and sample.

    Per-mode and joint blocks are separate classes so each compiles its own forward.
    """

    def forward(
        self,
        x: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Optional[Bounds] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        return self._update(x, freqs, mod, sample_bounds, timestep)


class Flux3SingleTransformerBlock(_Flux3Block):
    """Joint block: image, reference-image and text tokens of each sample attend together."""

    def forward(
        self,
        x: Tensor,
        freqs: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Optional[Bounds] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        return self._update(x, freqs, mod, sample_bounds, timestep)


# =============================================================================
# Main model
# =============================================================================


def _modulation_key(name: str) -> str:
    return "txt" if name == TEXT else name


def _checkpoint_key(key: str) -> str:
    """Name of a loader-side tensor key in the Diffusers checkpoint."""
    for old, new in (
        (".ff.linear_in_gate.", ".ff.linear_in."),
        (".ff.linear_in_up.", ".ff.linear_in."),
        (".ff.down_proj.", ".ff.linear_out."),
    ):
        key = key.replace(old, new)
    return key


def _checkpoint_module_names(name: str) -> Tuple[str, ...]:
    """Diffusers checkpoint module names a TRT-LLM Linear is loaded from."""
    parent, _, leaf = name.rpartition(".")
    if leaf not in _CHECKPOINT_MODULES:
        return (name,)
    return tuple(f"{parent}.{source}" if parent else source for source in _CHECKPOINT_MODULES[leaf])


def _format_keys(keys: List[str], limit: int = 10) -> str:
    shown = ", ".join(keys[:limit])
    return f"{len(keys)} [{shown}{', ...' if len(keys) > limit else ''}]"


class Flux3Transformer2DModel(BaseDiffusionModel):
    """FLUX.3 image DiT over packed modalities with per-sample joint attention.

    ``model_config.pretrained_config`` holds the checkpoint's transformer/config.json.
    """

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)
        arch = Flux3TransformerConfig.from_pretrained_config(model_config.pretrained_config)
        quant_config = model_config.quant_config
        if model_config.dynamic_weight_quant and quant_config.quant_algo is not None:
            excluded = list(quant_config.exclude_modules or [])
            excluded += [name for name in LOAD_TIME_QUANT_EXCLUDED_MODULES if name not in excluded]
            model_config.quant_config = quant_config.model_copy(
                update={"exclude_modules": excluded}
            )
        hidden_size = arch.hidden_size

        self.arch_config = arch
        self.flow_map = arch.flow_map
        self.timestep_channels = arch.timestep_channels
        self.content_channels = {IMAGE: arch.in_channels, IMAGE_COND: arch.cond_in_channels}
        self.context_in_dim = arch.joint_attention_dim

        self.pos_embed = Flux3PosEmbed(arch.rope_theta, arch.axes_dims_rope)
        self.timestep_embedder = Flux3TimestepEmbedder(
            arch.timestep_channels, hidden_size, model_config
        )
        self.target_timestep_embedder = (
            Flux3TimestepEmbedder(arch.timestep_channels, hidden_size, model_config)
            if arch.flow_map
            else None
        )
        self.early_stream_modulations = nn.ModuleDict(
            {
                _modulation_key(name): Flux3Modulation(hidden_size, model_config)
                for name in (*CONTENT_MODALITIES, TEXT)
            }
        )
        self.single_stream_modulations = nn.ModuleDict(
            {
                _modulation_key(name): Flux3Modulation(hidden_size, model_config)
                for name in (*CONTENT_MODALITIES, TEXT)
            }
        )
        self.x_embedders = nn.ModuleDict(
            {
                name: Linear(channels, hidden_size, **_linear_kwargs(model_config))
                for name, channels in self.content_channels.items()
            }
        )
        self.context_embedder = Linear(
            arch.joint_attention_dim, hidden_size, **_linear_kwargs(model_config)
        )
        # Layer indices are unique across all block stacks, in module order.
        layer_idx = itertools.count()
        self.mode_blocks = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        Flux3ModeTransformerBlock(
                            arch, model_config, f"mode_blocks.{name}.{i}", next(layer_idx)
                        )
                        for i in range(arch.num_layers)
                    ]
                )
                for name in CONTENT_MODALITIES
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                Flux3ModeTransformerBlock(
                    arch, model_config, f"txt_mode_blocks.{i}", next(layer_idx)
                )
                for i in range(arch.num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux3SingleTransformerBlock(
                    arch, model_config, f"single_transformer_blocks.{i}", next(layer_idx)
                )
                for i in range(arch.num_single_layers)
            ]
        )
        self.norm_out = Flux3AdaLayerNorm(hidden_size, arch.eps, model_config)
        self.proj_out = Linear(hidden_size, arch.in_channels, **_linear_kwargs(model_config))

        self.sharder = SequenceSharder.from_vgm(
            model_config.visual_gen_mapping, num_attention_heads=arch.num_attention_heads
        )
        self.__post_init__()

    def __post_init__(self):
        self.apply_quant_config_exclude_modules()
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    @property
    def packed_channels(self) -> Dict[str, int]:
        """Token width of every packed modality."""
        return {**self.content_channels, TEXT: self.context_in_dim}

    def apply_quant_config_exclude_modules(self) -> None:
        """Unquantize Linears excluded by their own or their checkpoint module names."""
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return
        no_quant_config = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for name, module in self.named_modules():
            if not isinstance(module, Linear) or getattr(module, "quant_config", None) is None:
                continue
            names = (name, *_checkpoint_module_names(name))
            if any(quant_config.is_module_excluded_from_quantization(n) for n in names):
                module.quant_config = no_quant_config

    def checkpoint_quantization(self) -> Dict[str, bool]:
        """Whether each Linear's Diffusers checkpoint module holds quantized weights."""
        quantized = {}
        for name, module in self.named_modules():
            if isinstance(module, Linear):
                quant_config = getattr(module, "quant_config", None)
                for source in _checkpoint_module_names(name):
                    quantized[source] = (
                        quant_config is not None and quant_config.quant_algo is not None
                    )
        return quantized

    def register_cuda_graph_extra_key_fns(self, runner) -> None:
        """Key graphs on the packed layout, which sets attention bounds at a fixed shape."""
        super().register_cuda_graph_extra_key_fns(runner)
        runner.register_extra_key_fn("flux3_layout", lambda *args, **kwargs: kwargs.get("layout"))

    def _mode_blocks(self, name: str) -> nn.ModuleList:
        return self.txt_mode_blocks if name == TEXT else self.mode_blocks[name]

    def _check_inputs(self, inputs: dict, layout: Layout) -> None:
        """Validate the packed layout and every input tensor this call reads."""
        names = tuple(name for name, _ in layout)
        if names != PACKED_MODALITIES:
            raise ValueError(f"FLUX.3 packed layout must list {PACKED_MODALITIES}, got {names}")
        num_samples = len(layout[0][1])
        for name, lengths in layout:
            if len(lengths) != num_samples:
                raise ValueError(f"FLUX.3 layout lists {num_samples} samples, {name} has {lengths}")
            if name not in OPTIONAL_MODALITIES and not all(lengths):
                raise ValueError(f"Every FLUX.3 sample needs {name} tokens, got {lengths}")
            if not any(lengths):
                continue
            expected = {}
            if self.flow_map or f"_flux3_{name}_tokens" not in inputs:
                expected[f"x_{name}"] = (sum(lengths), self.packed_channels[name])
                expected[f"x_{name}_timesteps"] = (num_samples,)
            if "_flux3_joint_cos" not in inputs:
                expected[f"x_{name}_ids"] = (sum(lengths), NUM_POSITION_AXES)
            for key, shape in expected.items():
                if key not in inputs:
                    raise ValueError(f"FLUX.3 transformer inputs need {key}")
                if tuple(inputs[key].shape) != shape:
                    raise ValueError(
                        f"FLUX.3 {key} must have shape {shape}, got {tuple(inputs[key].shape)}"
                    )

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

    def _shard_packed(self, value: Tensor, lengths: Tuple[int, ...], dim: int) -> Tensor:
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

    def _mode_stream(
        self,
        name: str,
        inputs: dict,
        lengths: Tuple[int, ...],
        local_lengths: Tuple[int, ...],
        freqs: Tuple[Tensor, Tensor],
        target: Optional[Tensor],
        timestep: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Embed one modality and run its per-mode blocks; conditioning is one row per sample."""
        rows = [index for index, length in enumerate(lengths) if length]
        row_lengths = [local_lengths[index] for index in rows]
        tokens = inputs[f"x_{name}"]
        timesteps = _select_rows(inputs[f"x_{name}_timesteps"], rows).to(tokens.dtype)
        vec = self.timestep_embedder(
            flux3_timestep_embedding(timesteps, self.timestep_channels, out_dtype=tokens.dtype)
        )
        if target is not None:
            vec = vec + _select_rows(target, rows)
        key = _modulation_key(name)
        early = tuple(
            _expand_rows(value, row_lengths) for value in self.early_stream_modulations[key](vec)
        )
        embedder = self.context_embedder if name == TEXT else self.x_embedders[name]
        x = self._shard_packed(embedder(tokens).unsqueeze(0), lengths, 1)
        sample_bounds = bounds(local_lengths)
        for block in self._mode_blocks(name):
            x = block(x, freqs, early, sample_bounds=sample_bounds, timestep=timestep)
        state = {"tokens": x}
        for field, value in zip(
            ("shift", "scale", "gate"), self.single_stream_modulations[key](vec)
        ):
            state["single_" + field] = _expand_rows(value, row_lengths)
        if name == OUTPUT_MODALITY:
            state["vec"] = vec
        return state

    @staticmethod
    def _joint_chunks(layout: Layout) -> Tuple[List[Tuple[str, int, int]], Bounds]:
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

    def _packed_positions(self, inputs: dict, layout: Layout) -> Dict[str, Tensor]:
        prepared, positions = {}, {}
        for name, lengths in layout:
            if not any(lengths):
                continue
            pe = self.pos_embed(inputs["x_" + name + "_ids"].unsqueeze(0))
            pe = self._shard_packed(pe, lengths, 2)
            positions[name] = pe
            prepared[f"_flux3_{name}_cos"], prepared[f"_flux3_{name}_sin"] = _pe_to_block_freqs(pe)
        chunks, _ = self._joint_chunks(self._local_layout(layout))
        pe = torch.cat([positions[name][:, :, start:end] for name, start, end in chunks], dim=2)
        prepared["_flux3_joint_cos"], prepared["_flux3_joint_sin"] = _pe_to_block_freqs(pe)
        return prepared

    def prepare_static_inputs(self, *, layout: Layout, **inputs) -> Dict[str, Tensor]:
        """Precompute RoPE tables and, without flow map, the step-independent text and reference
        streams. Pass the result to every forward call of the request.
        """
        if torch.is_grad_enabled():
            return {}
        self._check_inputs(inputs, layout)
        local_lengths = dict(self._local_layout(layout))
        prepared = self._packed_positions(inputs, layout)
        if self.flow_map:
            return prepared
        for name, lengths in layout:
            if name == OUTPUT_MODALITY or not any(lengths):
                continue
            freqs = prepared[f"_flux3_{name}_cos"], prepared[f"_flux3_{name}_sin"]
            state = self._mode_stream(
                name, inputs, lengths, local_lengths[name], freqs, target=None, timestep=None
            )
            prepared.update({f"_flux3_{name}_{field}": state[field] for field in _JOINT_FIELDS})
        return prepared

    def forward(
        self,
        *,
        layout: Layout,
        target_timesteps: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        **inputs,
    ) -> Dict[str, Tensor]:
        """Image velocity ``{"x_image": [image tokens, in_channels]}`` from ``pack(...).kwargs()``.

        ``target_timesteps`` [num_samples] is for flow map only; ``timestep`` reaches the attention.
        """
        self._check_inputs(inputs, layout)
        if (target_timesteps is not None) != self.flow_map:
            raise ValueError("target_timesteps is required by, and only by, a flow-map FLUX.3 DiT")
        output_tokens = inputs[f"x_{OUTPUT_MODALITY}"]
        target = None
        if target_timesteps is not None:
            num_samples = len(layout[0][1])
            if target_timesteps.shape != (num_samples,):
                raise ValueError(f"target_timesteps must have shape ({num_samples},)")
            target_timesteps = target_timesteps.to(output_tokens)
            target = self.target_timestep_embedder(
                flux3_timestep_embedding(
                    target_timesteps, self.timestep_channels, out_dtype=output_tokens.dtype
                )
            )
        local_layout = self._local_layout(layout)
        local_lengths = dict(local_layout)
        prepared = (
            inputs if "_flux3_joint_cos" in inputs else self._packed_positions(inputs, layout)
        )
        streams = {}
        for name, lengths in layout:
            if not any(lengths):
                continue
            prefix = f"_flux3_{name}_"
            if not self.flow_map and prefix + "tokens" in inputs:
                streams[name] = {field: inputs[prefix + field] for field in _JOINT_FIELDS}
                continue
            # Without flow map, text and reference streams do not depend on the step, as cached.
            step_dependent = self.flow_map or name == OUTPUT_MODALITY
            streams[name] = self._mode_stream(
                name,
                inputs,
                lengths,
                local_lengths[name],
                (prepared[prefix + "cos"], prepared[prefix + "sin"]),
                target=target,
                timestep=timestep if step_dependent else None,
            )

        chunks, sample_bounds = self._joint_chunks(local_layout)

        def joint(field: str) -> Tensor:
            return torch.cat(
                [streams[name][field][:, start:end] for name, start, end in chunks], dim=1
            )

        combined = joint("tokens")
        mod = tuple(joint("single_" + field) for field in ("shift", "scale", "gate"))
        freqs = prepared["_flux3_joint_cos"], prepared["_flux3_joint_sin"]
        for block in self.single_transformer_blocks:
            combined = block(combined, freqs, mod, sample_bounds=sample_bounds, timestep=timestep)

        vec = streams[OUTPUT_MODALITY]["vec"]
        parts, offset = [], 0
        for name, start, end in chunks:
            if name == OUTPUT_MODALITY:
                # Every sample has one image chunk, in sample order.
                row = len(parts)
                part = combined[:, offset : offset + end - start].contiguous()
                if self.sharder.is_active:
                    part = self.sharder.gather(part, dim=1)
                parts.append(self.norm_out(part, vec[row : row + 1]))
            offset += end - start
        output = self.proj_out(torch.cat(parts, dim=1))[0]
        return {"x_" + OUTPUT_MODALITY: output}

    def load_weights(self, weights: Dict[str, Tensor]) -> None:
        """Load a Diffusers-format FLUX.3 transformer state dict.

        Raises ValueError, naming the tensors, if the checkpoint misses or adds any tensor.
        """
        weights = {
            key.replace(".ff.linear_out.", ".ff.down_proj."): v for key, v in weights.items()
        }
        weights = _split_linear_in_weights(weights)
        loader = DynamicLinearWeightLoader(self.model_config, params_map=_PARAMS_MAP)

        linear_loads, param_loads, consumed, missing = [], [], set(), []
        for name, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()
            if isinstance(module, Linear):
                if module.weights_loading_config.weight_mode == WeightMode.VANILLA:
                    sources = [name]
                else:
                    parent, _, leaf = name.rpartition(".")
                    sources = [f"{parent}.{source}" for source in _PARAMS_MAP[leaf]]
                weight_dicts = loader.get_linear_weights(module, name, weights) or [{}]
                required = ("weight", "bias") if module.bias is not None else ("weight",)
                for source, weight_dict in zip(sources, weight_dicts):
                    missing.extend(f"{source}.{t}" for t in required if t not in weight_dict)
                    consumed.update(f"{source}.{suffix}" for suffix in weight_dict)
                linear_loads.append((name, module, weight_dicts))
                continue
            for param_name, param in module.named_parameters(recurse=False):
                key = f"{name}.{param_name}"
                if key in weights:
                    param_loads.append((param, weights[key]))
                    consumed.add(key)
                else:
                    missing.append(key)

        unexpected = sorted({_checkpoint_key(key) for key in weights.keys() - consumed})
        missing = sorted({_checkpoint_key(key) for key in missing})
        if missing or unexpected:
            raise ValueError(
                "FLUX.3 checkpoint does not match the transformer: "
                f"missing {_format_keys(missing)}, unexpected {_format_keys(unexpected)}"
            )
        for name, module, weight_dicts in linear_loads:
            loader.load_linear_weights(module, name, weight_dicts)
        for param, value in param_loads:
            param.data.copy_(value.to(param.dtype))

    def post_load_weights(self) -> None:
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
        # Composite modules settle their decisions after the Linears they read are final.
        for _, module in self.named_modules():
            if isinstance(module, (GatedMLP, Attention)):
                module.post_load_weights()
