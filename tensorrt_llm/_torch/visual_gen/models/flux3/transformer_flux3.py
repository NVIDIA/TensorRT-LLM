# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 transformer, loaded from Diffusers-format checkpoints.

Per-mode blocks run within each modality; single blocks then attend jointly within each sample.
"""

import dataclasses
import itertools
import math
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, TypeVar

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
from tensorrt_llm._torch.visual_gen.models.flux.pos_embed_flux import FluxPosEmbed
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
# Top-level per-modality modules: per-mode block stacks, modulations and input embedders.
_MODE_BLOCKS = {
    IMAGE: "image_mode_blocks",
    IMAGE_COND: "image_cond_mode_blocks",
    TEXT: "txt_mode_blocks",
}
_EARLY_MODULATIONS = {
    IMAGE: "image_early_modulation",
    IMAGE_COND: "image_cond_early_modulation",
    TEXT: "txt_early_modulation",
}
_SINGLE_MODULATIONS = {
    IMAGE: "image_single_modulation",
    IMAGE_COND: "image_cond_single_modulation",
    TEXT: "txt_single_modulation",
}
_EMBEDDERS = {IMAGE: "image_embedder", IMAGE_COND: "image_cond_embedder", TEXT: "context_embedder"}
# Diffusers names of the modules it keys by modality, and their module names.
_CHECKPOINT_PREFIXES = (
    *((f"mode_blocks.{name}", _MODE_BLOCKS[name]) for name in CONTENT_MODALITIES),
    *((f"x_embedders.{name}", _EMBEDDERS[name]) for name in CONTENT_MODALITIES),
    *(
        (f"{stage}_stream_modulations.{'txt' if name == TEXT else name}", modules[name])
        for stage, modules in (("early", _EARLY_MODULATIONS), ("single", _SINGLE_MODULATIONS))
        for name in modules
    ),
)

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
# Algorithms that quantize BF16 checkpoint weights on load.
LOAD_TIME_QUANT_ALGOS = (
    QuantAlgo.FP8,
    QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
    QuantAlgo.NVFP4,
)
# Modules that load-time quantization keeps in BF16, as BFL's FP8 rowwise export does.
LOAD_TIME_QUANT_EXCLUDED_MODULES = (
    *_EMBEDDERS.values(),
    "timestep_embedder",
    "target_timestep_embedder",
    _MODE_BLOCKS[TEXT],
    "norm_out",
    "proj_out",
)
# Config keys added by the loader rather than the checkpoint architecture.
_NON_ARCHITECTURE_CONFIG_KEYS = frozenset({"quantization_config"})
# Stream tensors the single blocks read; the prepared text and reference streams hold these.
_JOINT_FIELDS = ("tokens", "single_shift", "single_scale", "single_gate")


@dataclasses.dataclass(frozen=True)
class Flux3TransformerConfig:
    """FLUX 3 architecture, read strictly from ``transformer/config.json``."""

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
            raise ValueError("FLUX 3 transformer needs the checkpoint's transformer/config.json")
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
                f"FLUX 3 transformer config: missing keys {missing}, unsupported keys {unknown}"
            )
        values["axes_dims_rope"] = tuple(values["axes_dims_rope"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        """Check the relations between config values that the model relies on."""
        axes = self.axes_dims_rope
        if len(axes) != NUM_POSITION_AXES or any(dim % 2 for dim in axes):
            raise ValueError(f"FLUX 3 needs {NUM_POSITION_AXES} even RoPE axis dims, got {axes}")
        if sum(axes) != self.attention_head_dim:
            raise ValueError(
                f"FLUX 3 RoPE axis dims {axes} must sum to attention_head_dim "
                f"{self.attention_head_dim}"
            )

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)


def load_time_quant_excluded_modules(
    arch: Flux3TransformerConfig, quant_algo: QuantAlgo
) -> Tuple[str, ...]:
    """Modules that load-time ``quant_algo`` keeps in BF16, as BFL's export at that precision does.

    NVFP4 also keeps the modulations, the first image block and the last seventh of single blocks.
    """
    if quant_algo != QuantAlgo.NVFP4:
        return LOAD_TIME_QUANT_EXCLUDED_MODULES
    num_single = arch.num_single_layers
    first_bf16_single = num_single - max(1, num_single // 7)
    return (
        *LOAD_TIME_QUANT_EXCLUDED_MODULES,
        *_EARLY_MODULATIONS.values(),
        *_SINGLE_MODULATIONS.values(),
        f"{_MODE_BLOCKS[IMAGE]}.0",
        *(f"single_transformer_blocks.{i}" for i in range(first_bf16_single, num_single)),
    )


def _replace_prefix(name: str, prefixes: Iterable[Tuple[str, str]]) -> str:
    """``name`` with its first matching dotted ``(old, new)`` prefix replaced."""
    for old, new in prefixes:
        if name == old or name.startswith(old + "."):
            return new + name[len(old) :]
    return name


def diffusers_to_module_name(name: str) -> str:
    """Module or tensor name of a Diffusers checkpoint name, which differs only for modules keyed
    by modality: ``x_embedders.image_cond.weight`` is ``image_cond_embedder.weight``.
    """
    return _replace_prefix(name, _CHECKPOINT_PREFIXES)


# =============================================================================
# BFL checkpoint names
# =============================================================================
# BFL-native and Diffusers tensor names map one to one; the output norm also swaps its
# shift and scale rows (BFL: shift, scale; Diffusers: scale, shift).

TensorT = TypeVar("TensorT", bound=torch.Tensor)

BFL_NORM_OUT = "final_layer.image.adaLN_modulation.1"
DIFFUSERS_NORM_OUT = "norm_out.linear"

_TOP_LEVEL_MODULES: tuple[tuple[str, str], ...] = (
    ("txt_in", "context_embedder"),
    ("emb_in.image", "x_embedders.image"),
    ("emb_in.image_cond", "x_embedders.image_cond"),
    ("time_in.in_layer", "timestep_embedder.linear_1"),
    ("time_in.out_layer", "timestep_embedder.linear_2"),
    ("ref_time_in.in_layer", "target_timestep_embedder.linear_1"),
    ("ref_time_in.out_layer", "target_timestep_embedder.linear_2"),
    *(
        (f"{stage}_stream_modulations.{mode}.lin", f"{stage}_stream_modulations.{mode}.linear")
        for stage in ("early", "single")
        for mode in ("image", "image_cond", "txt")
    ),
    (BFL_NORM_OUT, DIFFUSERS_NORM_OUT),
    ("final_layer.image.linear", "proj_out"),
)
_BLOCK_STACKS: tuple[tuple[str, str], ...] = (
    ("content_mode_blocks.image", "mode_blocks.image"),
    ("content_mode_blocks.image_cond", "mode_blocks.image_cond"),
    ("txt_mode_blocks", "txt_mode_blocks"),
    ("single_blocks", "single_transformer_blocks"),
)
_BLOCK_MODULES: tuple[tuple[str, str], ...] = (
    ("q_proj", "attn.to_q"),
    ("k_proj", "attn.to_k"),
    ("v_proj", "attn.to_v"),
    ("attn_out", "attn.to_out.0"),
    ("mlp_in", "ff.linear_in"),
    ("mlp_out", "ff.linear_out"),
)
_BLOCK_TENSORS: tuple[tuple[str, str], ...] = (
    ("norm.query_norm.scale", "attn.norm_q.weight"),
    ("norm.key_norm.scale", "attn.norm_k.weight"),
)
_SUFFIX = re.compile(r"[A-Za-z_]\w*(?:\.\w+)*")
# Output-norm tensors indexed by the shift and scale rows, and those that hold no such rows.
_NORM_OUT_SWAPPED_SUFFIXES = frozenset({"weight", "bias", "lora_B.weight", "lora_up.weight"})
_NORM_OUT_UNCHANGED_SUFFIXES = frozenset({"lora_A.weight", "lora_down.weight", "alpha"})


class _Direction:
    """Name tables for one direction of the mapping."""

    def __init__(self, source: str, pairs_index: int) -> None:
        def table(pairs: tuple[tuple[str, str], ...]) -> dict[str, str]:
            return {pair[pairs_index]: pair[1 - pairs_index] for pair in pairs}

        self.source = source
        self.top_level = table(_TOP_LEVEL_MODULES)
        self.stacks = table(_BLOCK_STACKS)
        self.block_modules = table(_BLOCK_MODULES)
        self.block_tensors = table(_BLOCK_TENSORS)
        stacks = "|".join(re.escape(name) for name in self.stacks)
        self.block = re.compile(rf"(?P<stack>{stacks})\.(?P<index>\d+)\.(?P<rest>.+)")

    def _split(self, name: str) -> tuple[str, str, dict[str, str]]:
        """Split ``name`` into a mapped block prefix, the name inside it, and its module table."""
        match = self.block.fullmatch(name)
        if match is None:
            return "", name, self.top_level
        prefix = f"{self.stacks[match['stack']]}.{match['index']}."
        return prefix, match["rest"], self.block_modules

    def module(self, name: str) -> str:
        prefix, rest, modules = self._split(name)
        if rest not in modules:
            raise ValueError(f"Unknown {self.source} FLUX 3 module name: {name!r}")
        return prefix + modules[rest]

    def key(self, key: str) -> str:
        prefix, rest, modules = self._split(key)
        if prefix and rest in self.block_tensors:
            return prefix + self.block_tensors[rest]
        # Longest module name first, so no shorter name can claim part of a longer one.
        for module_name in sorted(modules, key=len, reverse=True):
            suffix = rest[len(module_name) + 1 :]
            if rest.startswith(module_name + ".") and _SUFFIX.fullmatch(suffix):
                return f"{prefix}{modules[module_name]}.{suffix}"
        raise ValueError(f"Unknown {self.source} FLUX 3 tensor name: {key!r}")


_BFL_TO_DIFFUSERS = _Direction("BFL", 0)
_DIFFUSERS_TO_BFL = _Direction("Diffusers", 1)


def bfl_to_diffusers_module(name: str) -> str:
    """Map a BFL module name, e.g. ``single_blocks.3.q_proj``, to Diffusers.

    Raises ValueError if ``name`` is not a FLUX 3 BFL module.
    """
    return _BFL_TO_DIFFUSERS.module(name)


def diffusers_to_bfl_module(name: str) -> str:
    """Map a Diffusers module name, e.g. ``single_transformer_blocks.3.attn.to_q``, to BFL.

    Raises ValueError if ``name`` is not a FLUX 3 Diffusers module.
    """
    return _DIFFUSERS_TO_BFL.module(name)


def bfl_to_diffusers_key(key: str) -> str:
    """Map a BFL tensor name to Diffusers, keeping its suffix; ValueError if unknown."""
    return _BFL_TO_DIFFUSERS.key(key)


def diffusers_to_bfl_key(key: str) -> str:
    """Map a Diffusers tensor name to BFL, keeping its suffix; ValueError if unknown."""
    return _DIFFUSERS_TO_BFL.key(key)


def swap_norm_out_halves(value: TensorT) -> TensorT:
    """Swap the shift and scale row halves of an output-norm tensor [2 * hidden, ...].

    The swap is its own inverse.
    """
    rows = value.shape[0]
    if rows % 2:
        raise ValueError(f"FLUX 3 output-norm tensor needs an even row count, got {rows}")
    # Rolling dim 0 by half its length swaps the two halves.
    return value.roll(rows // 2, dims=0)


def _convert_state_dict(
    state_dict: Mapping[str, TensorT], convert_key: Callable[[str], str], norm_out: str
) -> dict[str, TensorT]:
    converted = {}
    for key, value in state_dict.items():
        if key.startswith(norm_out + "."):
            suffix = key[len(norm_out) + 1 :]
            if suffix in _NORM_OUT_SWAPPED_SUFFIXES:
                value = swap_norm_out_halves(value)
            elif suffix not in _NORM_OUT_UNCHANGED_SUFFIXES:
                raise ValueError(f"No shift/scale order transform for FLUX 3 tensor {key!r}")
        converted[convert_key(key)] = value
    return converted


def bfl_to_diffusers_state_dict(state_dict: Mapping[str, TensorT]) -> dict[str, TensorT]:
    """Rename a BFL state dict to Diffusers names and reorder the output norm to scale, shift.

    Raises ValueError for unknown names and for output-norm tensors without a known row order.
    """
    return _convert_state_dict(state_dict, bfl_to_diffusers_key, BFL_NORM_OUT)


def diffusers_to_bfl_state_dict(state_dict: Mapping[str, TensorT]) -> dict[str, TensorT]:
    """Rename a Diffusers state dict to BFL names and reorder the output norm to shift, scale.

    Raises ValueError for unknown names and for output-norm tensors without a known row order.
    """
    return _convert_state_dict(state_dict, diffusers_to_bfl_key, DIFFUSERS_NORM_OUT)


# =============================================================================
# Timestep embedding
# =============================================================================


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
    """Projects SiLU-activated conditioning rows [n, hidden] to shift, scale and gate rows."""

    def __init__(self, hidden_size: int, config: DiffusionModelConfig):
        super().__init__()
        self.linear = Linear(hidden_size, 3 * hidden_size, **_linear_kwargs(config))

    def forward(self, temb: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.linear(temb).chunk(3, dim=-1)


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
    """FLUX 3 self-attention with per-head QK-norm and RoPE, attending within packed samples."""

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
        image_rotary_emb: Tuple[Tensor, Tensor],
        sample_bounds: Optional[Bounds] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """Attention before the output projection: [B, S, hidden] to [B, S, local heads * head_dim].

        ``image_rotary_emb`` (cos, sin) are [B, S, 1, head_dim]; ``sample_bounds`` limits attention.
        """
        b = hidden_states.shape[0]
        cos, sin = image_rotary_emb
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
        hidden_states: Tensor,
        image_rotary_emb: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Optional[Bounds],
        timestep: Optional[Tensor],
    ) -> Tensor:
        shift, scale, gate = mod
        norm_hidden_states = (1 + scale) * self.norm(hidden_states) + shift
        attn = self.attn.attend(
            norm_hidden_states, image_rotary_emb, sample_bounds=sample_bounds, timestep=timestep
        )
        mlp_out = self.ff(norm_hidden_states.reshape(-1, norm_hidden_states.shape[-1])).reshape(
            *norm_hidden_states.shape[:-1], -1
        )
        output = self.attn.to_out[0](attn) + mlp_out
        if self._allreduce is not None:
            output = self._allreduce(output)
        return hidden_states + gate * output


class Flux3ModeTransformerBlock(_Flux3Block):
    """Per-mode block: tokens attend only within their own modality and sample.

    Per-mode and joint blocks are separate classes so each compiles its own forward.
    """

    def forward(
        self,
        hidden_states: Tensor,
        image_rotary_emb: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Optional[Bounds] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        return self._update(hidden_states, image_rotary_emb, mod, sample_bounds, timestep)


class Flux3SingleTransformerBlock(_Flux3Block):
    """Joint block: image, reference-image and text tokens of each sample attend together."""

    def forward(
        self,
        hidden_states: Tensor,
        image_rotary_emb: Tuple[Tensor, Tensor],
        mod: Tuple[Tensor, Tensor, Tensor],
        sample_bounds: Optional[Bounds] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        return self._update(hidden_states, image_rotary_emb, mod, sample_bounds, timestep)


# =============================================================================
# Main model
# =============================================================================


def _checkpoint_key(key: str) -> str:
    """Name of a loader-side tensor key in the Diffusers checkpoint."""
    key = _replace_prefix(key, ((new, old) for old, new in _CHECKPOINT_PREFIXES))
    for old, new in (
        (".ff.linear_in_gate.", ".ff.linear_in."),
        (".ff.linear_in_up.", ".ff.linear_in."),
        (".ff.down_proj.", ".ff.linear_out."),
    ):
        key = key.replace(old, new)
    return key


def _checkpoint_module_names(name: str) -> Tuple[str, ...]:
    """Checkpoint Linears that a TRT-LLM Linear loads from, under TRT-LLM block names."""
    parent, _, leaf = name.rpartition(".")
    if leaf not in _CHECKPOINT_MODULES:
        return (name,)
    return tuple(f"{parent}.{source}" if parent else source for source in _CHECKPOINT_MODULES[leaf])


def _mode_block_list(
    arch: Flux3TransformerConfig, config: DiffusionModelConfig, stack: str, layer_idx: Iterator[int]
) -> nn.ModuleList:
    """``arch.num_layers`` per-mode blocks named ``{stack}.{i}``, numbered from ``layer_idx``."""
    return nn.ModuleList(
        [
            Flux3ModeTransformerBlock(arch, config, f"{stack}.{i}", next(layer_idx))
            for i in range(arch.num_layers)
        ]
    )


def format_names(names: List[str], limit: int = 10) -> str:
    """``3 [a, b, c]``: the count and at most ``limit`` of ``names``."""
    return f"{len(names)} [{', '.join(names[:limit])}{', ...' if len(names) > limit else ''}]"


class Flux3Transformer2DModel(BaseDiffusionModel):
    """FLUX 3 image DiT over packed modalities with per-sample joint attention.

    ``model_config.pretrained_config`` holds the checkpoint's transformer/config.json.
    """

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)
        arch = Flux3TransformerConfig.from_pretrained_config(model_config.pretrained_config)
        quant_config = model_config.quant_config
        if model_config.dynamic_weight_quant and quant_config.quant_algo is not None:
            excluded = list(quant_config.exclude_modules or [])
            defaults = load_time_quant_excluded_modules(arch, quant_config.quant_algo)
            excluded += [name for name in defaults if name not in excluded]
            model_config.quant_config = quant_config.model_copy(
                update={"exclude_modules": excluded}
            )
        hidden_size = arch.hidden_size

        self.arch_config = arch
        self.flow_map = arch.flow_map
        self.timestep_channels = arch.timestep_channels
        self.content_channels = {IMAGE: arch.in_channels, IMAGE_COND: arch.cond_in_channels}
        self.context_in_dim = arch.joint_attention_dim

        self.pos_embed = FluxPosEmbed(theta=arch.rope_theta, axes_dim=list(arch.axes_dims_rope))
        self.timestep_embedder = Flux3TimestepEmbedder(
            arch.timestep_channels, hidden_size, model_config
        )
        self.target_timestep_embedder = (
            Flux3TimestepEmbedder(arch.timestep_channels, hidden_size, model_config)
            if arch.flow_map
            else None
        )
        self.image_early_modulation = Flux3Modulation(hidden_size, model_config)
        self.image_cond_early_modulation = Flux3Modulation(hidden_size, model_config)
        self.txt_early_modulation = Flux3Modulation(hidden_size, model_config)
        self.image_single_modulation = Flux3Modulation(hidden_size, model_config)
        self.image_cond_single_modulation = Flux3Modulation(hidden_size, model_config)
        self.txt_single_modulation = Flux3Modulation(hidden_size, model_config)
        self.image_embedder = Linear(arch.in_channels, hidden_size, **_linear_kwargs(model_config))
        self.image_cond_embedder = Linear(
            arch.cond_in_channels, hidden_size, **_linear_kwargs(model_config)
        )
        self.context_embedder = Linear(
            arch.joint_attention_dim, hidden_size, **_linear_kwargs(model_config)
        )
        # Layer indices are unique across all block stacks, in module order.
        layer_idx = itertools.count()
        self.image_mode_blocks = _mode_block_list(
            arch, model_config, _MODE_BLOCKS[IMAGE], layer_idx
        )
        self.image_cond_mode_blocks = _mode_block_list(
            arch, model_config, _MODE_BLOCKS[IMAGE_COND], layer_idx
        )
        self.txt_mode_blocks = _mode_block_list(arch, model_config, _MODE_BLOCKS[TEXT], layer_idx)
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
        """Whether each checkpoint Linear named by ``_checkpoint_module_names`` is quantized."""
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

    def _check_inputs(self, inputs: dict, layout: Layout, timestep: Optional[Tensor]) -> None:
        """Validate the packed layout, ``timestep`` and every input tensor this call reads."""
        names = tuple(name for name, _ in layout)
        if names != PACKED_MODALITIES:
            raise ValueError(f"FLUX 3 packed layout must list {PACKED_MODALITIES}, got {names}")
        num_samples = len(layout[0][1])
        if timestep is None:
            raise ValueError("FLUX 3 transformer inputs need timestep, the image flow time")
        if tuple(timestep.shape) not in ((), (1,), (num_samples,)):
            raise ValueError(
                f"FLUX 3 timestep must be a scalar or have shape ({num_samples},), "
                f"got {tuple(timestep.shape)}"
            )
        for name, lengths in layout:
            if len(lengths) != num_samples:
                raise ValueError(f"FLUX 3 layout lists {num_samples} samples, {name} has {lengths}")
            if name not in OPTIONAL_MODALITIES and not all(lengths):
                raise ValueError(f"Every FLUX 3 sample needs {name} tokens, got {lengths}")
            if not any(lengths):
                continue
            expected = {}
            if self.flow_map or f"_flux3_{name}_tokens" not in inputs:
                expected[f"x_{name}"] = (sum(lengths), self.packed_channels[name])
            if "_flux3_joint_cos" not in inputs:
                expected[f"x_{name}_ids"] = (sum(lengths), NUM_POSITION_AXES)
            for key, shape in expected.items():
                if key not in inputs:
                    raise ValueError(f"FLUX 3 transformer inputs need {key}")
                if tuple(inputs[key].shape) != shape:
                    raise ValueError(
                        f"FLUX 3 {key} must have shape {shape}, got {tuple(inputs[key].shape)}"
                    )

    def _local_layout(self, layout: Layout) -> Layout:
        size = self.sharder.size
        if any(length % size for _, lengths in layout for length in lengths):
            raise ValueError(
                "FLUX 3 sequence parallelism requires each sample's modality length "
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
        rotary_emb: Tuple[Tensor, Tensor],
        target: Optional[Tensor],
        timestep: Tensor,
    ) -> Dict[str, Tensor]:
        """Embed one modality and run its per-mode blocks; conditioning is one row per sample.

        Image tokens flow at ``timestep``; text and reference tokens sit at timestep 0.
        """
        rows = [index for index, length in enumerate(lengths) if length]
        row_lengths = [local_lengths[index] for index in rows]
        tokens = inputs[f"x_{name}"]
        if name == OUTPUT_MODALITY:
            timesteps = _select_rows(timestep.reshape(-1).expand(len(lengths)), rows)
        else:
            timesteps = tokens.new_zeros(len(rows))
        vec = self.timestep_embedder(
            flux3_timestep_embedding(
                timesteps.to(tokens.dtype), self.timestep_channels, out_dtype=tokens.dtype
            )
        )
        if target is not None:
            vec = vec + _select_rows(target, rows)
        act = F.silu(vec)
        early = tuple(
            _expand_rows(value, row_lengths)
            for value in getattr(self, _EARLY_MODULATIONS[name])(act)
        )
        embedder = getattr(self, _EMBEDDERS[name])
        x = self._shard_packed(embedder(tokens).unsqueeze(0), lengths, 1)
        sample_bounds = bounds(local_lengths)
        for block in getattr(self, _MODE_BLOCKS[name]):
            x = block(x, rotary_emb, early, sample_bounds=sample_bounds, timestep=timestep)
        state = {"tokens": x}
        single = getattr(self, _SINGLE_MODULATIONS[name])(act)
        for field, value in zip(("shift", "scale", "gate"), single):
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
        prepared = {}
        for name, lengths in layout:
            if not any(lengths):
                continue
            cos, sin = self.pos_embed(inputs["x_" + name + "_ids"])
            prepared[f"_flux3_{name}_cos"] = self._shard_packed(cos, lengths, 1)
            prepared[f"_flux3_{name}_sin"] = self._shard_packed(sin, lengths, 1)
        chunks, _ = self._joint_chunks(self._local_layout(layout))
        for field in ("cos", "sin"):
            prepared[f"_flux3_joint_{field}"] = torch.cat(
                [prepared[f"_flux3_{name}_{field}"][:, start:end] for name, start, end in chunks],
                dim=1,
            )
        return prepared

    def prepare_static_inputs(
        self, *, layout: Layout, timestep: Optional[Tensor] = None, **inputs
    ) -> Dict[str, Tensor]:
        """Precompute RoPE tables and, without flow map, the step-independent text and reference
        streams, which attend at the first step's normalized ``timestep``. Every forward call of
        the request takes the result.
        """
        if torch.is_grad_enabled():
            return {}
        self._check_inputs(inputs, layout, timestep)
        local_lengths = dict(self._local_layout(layout))
        prepared = self._packed_positions(inputs, layout)
        if self.flow_map:
            return prepared
        for name, lengths in layout:
            if name == OUTPUT_MODALITY or not any(lengths):
                continue
            rotary_emb = prepared[f"_flux3_{name}_cos"], prepared[f"_flux3_{name}_sin"]
            state = self._mode_stream(
                name, inputs, lengths, local_lengths[name], rotary_emb, None, timestep
            )
            prepared.update({f"_flux3_{name}_{field}": state[field] for field in _JOINT_FIELDS})
        return prepared

    def forward(
        self,
        *,
        layout: Layout,
        timestep: Optional[Tensor] = None,
        target_timestep: Optional[Tensor] = None,
        **inputs,
    ) -> Dict[str, Tensor]:
        """Image velocity ``{"x_image": [image tokens, in_channels]}`` from ``pack(...).kwargs()``.

        Image flow time ``timestep``: scalar or [num_samples]; ``target_timestep``: flow map only.
        """
        self._check_inputs(inputs, layout, timestep)
        if (target_timestep is not None) != self.flow_map:
            raise ValueError("target_timestep is required by, and only by, a flow-map FLUX 3 DiT")
        output_tokens = inputs[f"x_{OUTPUT_MODALITY}"]
        target = None
        if target_timestep is not None:
            num_samples = len(layout[0][1])
            if target_timestep.shape != (num_samples,):
                raise ValueError(f"target_timestep must have shape ({num_samples},)")
            target_timestep = target_timestep.to(output_tokens)
            target = self.target_timestep_embedder(
                flux3_timestep_embedding(
                    target_timestep, self.timestep_channels, out_dtype=output_tokens.dtype
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
            streams[name] = self._mode_stream(
                name,
                inputs,
                lengths,
                local_lengths[name],
                (prepared[prefix + "cos"], prepared[prefix + "sin"]),
                target=target,
                timestep=timestep,
            )

        chunks, sample_bounds = self._joint_chunks(local_layout)

        def joint(field: str) -> Tensor:
            return torch.cat(
                [streams[name][field][:, start:end] for name, start, end in chunks], dim=1
            )

        combined = joint("tokens")
        mod = tuple(joint("single_" + field) for field in ("shift", "scale", "gate"))
        rotary_emb = prepared["_flux3_joint_cos"], prepared["_flux3_joint_sin"]
        for block in self.single_transformer_blocks:
            combined = block(
                combined, rotary_emb, mod, sample_bounds=sample_bounds, timestep=timestep
            )

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
        """Load a Diffusers-format FLUX 3 transformer state dict.

        Raises ValueError, naming the checkpoint tensors, if it misses or adds any tensor.
        """
        weights = {
            diffusers_to_module_name(key).replace(".ff.linear_out.", ".ff.down_proj."): v
            for key, v in weights.items()
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
                "FLUX 3 checkpoint does not match the transformer: "
                f"missing {format_names(missing)}, unexpected {format_names(unexpected)}"
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
