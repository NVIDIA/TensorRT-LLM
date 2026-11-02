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
"""MuseSparkOpen NaViT perception tower for the PyTorch backend.

A 50-layer packed variable-resolution ViT. Images and video clips of different
sizes are concatenated into one flat `[total_patches, patch_dim]` sequence and
kept apart by per-clip attention segments, so one forward covers a whole batch
of media without padding.

Flow::

    pixel_values [total_patches, 1176]
      -> conv1_linear (Linear, NOT a Conv2d)
      -> + interpolated positional embedding (F.grid_sample, bilinear)
      -> ln_pre
      -> window permutation
      -> 50 x (ln_1 -> attn(2D RoPE) -> +res -> ln_2 -> mlp -> +res)
      -> inverse window permutation
      -> ln_post
      -> pixel shuffle (factor 2, channel-major)
      -> adapter (c_fc -> GELU -> c_proj -> GELU)
      -> vision_projection
      -> [total_rows, 8192]

Five details this port has to get exactly right, none of which has a behavioral
proxy short of running a whole benchmark:

1. `conv1_linear` is a **Linear** over pre-vectorized patches, not a Conv2d.
   The input row layout is `[t][c][py][px]` and a still image is **expanded**
   (frame duplicated) to `patch_temporal` frames, never zero-padded. The
   input processor produces that layout; see `PATCH_VECTORIZATION` below.
2. The positional embedding is a learned `[1024, 1536]` grid viewed as
   `[1, 1536, 32, 32]` and sampled with `F.grid_sample(align_corners=False)`
   at pixel-centre coordinates built by a meshgrid with `indexing="xy"`. That
   indexing transposes the coordinate list relative to patch order; the model
   was trained that way, so it is replicated verbatim and deliberately not
   "fixed". The result is added **before** `ln_pre`, in the activation dtype.
3. 2D RoPE uses `head_dim // 4` = 24 base frequencies per axis, theta 10000,
   **1-indexed** positions, **width first** (`cat([freq_w, freq_h])`), and
   **interleaved** adjacent pairs (not a half-split). The rotation runs in fp32
   and is cast back.
4. Only 13 of the 50 layers attend over a whole clip: those with
   `(i + 1) % sparse_attention_factor == 0` plus the last one. The other 37
   are restricted to 32x32-patch windows, implemented as a single permutation
   of both `x` and the RoPE table applied once before the stack and inverted
   with `index_copy_` before `ln_post` -- not as a per-layer gather.
5. Normalization is LayerNorm (weight + bias, eps 1e-5), not RMSNorm. Biases
   exist on `qkv_proj`/`out_proj`/`mlp.c_fc`/`mlp.c_proj` and NOT on
   `conv1_linear`/`adapter.*`/`vision_projection`. Row-parallel biases are
   added after the tensor-parallel all-reduce.

The tower returns **unnormed** projected rows. It deliberately does not apply
`perception_emb_norm`: `MuseSparkOpenModel.forward` already applies the same
weightless `RMSNorm(8192, eps=1e-5)` to whatever `inputs_embeds` it is
handed, and RMSNorm is per-row, so merging first and norming once reproduces the
reference's norm-then-merge row for row. Applying it here would norm vision rows
twice, which is not a no-op in bf16 at small row scale.
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..._utils import prefer_pinned
from ...logger import logger
from ..attention.attention import Attention
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PredefinedAttentionMask
from ..attention.backends.utils import get_attention_backend
from ..configs.musespark_open import (
    MuseSparkOpenConfig,
    MuseSparkOpenVisionConfig,
    layer_uses_full_vision_attention,
)
from ..model_config import ModelConfig
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mlp import MLP
from .modeling_multimodal_encoder import MultimodalEncoderMixin

__all__ = [
    "CKPT_TO_MODULE",
    "MuseSparkOpenNaViTVisionTower",
    "PATCH_VECTORIZATION",
    "VISION_PROJECTION_CKPT_KEY",
    "vision_rows_for_grid",
]

# How one patch row of `pixel_values` is laid out, slowest-varying axis first.
# The input processor owns the vectorization; the tower owns this statement of
# what it expects, and `conv1_linear`'s weight is only correct for this order.
PATCH_VECTORIZATION = ("patch_temporal", "num_channels", "patch_y", "patch_x")

# Checkpoint suffix -> module attribute, for the entries where the two differ.
# Everything else is named exactly as the checkpoint names it, so the weight
# mapper is a prefix rename. The four deviations all come from reusing TRT-LLM
# modules instead of hand-rolling: `Attention` fuses q/k/v into one
# column-parallel `qkv_proj` and calls the output projection `o_proj`, and
# `MLP` calls its two projections `up_proj`/`down_proj`. Importable so the
# weight mapper reads it rather than restating it.
CKPT_TO_MODULE: Dict[str, str] = {
    # q, k and v are three separate checkpoint tensors (weight + bias each) that
    # concatenate along dim 0 in q, k, v order into the one fused projection.
    "attn.q_proj": "attn.qkv_proj",
    "attn.k_proj": "attn.qkv_proj",
    "attn.v_proj": "attn.qkv_proj",
    "attn.out_proj": "attn.o_proj",
    "mlp.c_fc": "mlp.up_proj",
    "mlp.c_proj": "mlp.down_proj",
}

# `vision_projection` is the one tower weight that does NOT live under the
# `vision_model.` prefix: the checkpoint stores it as this single key,
# [8192, 4096], with no bias.
VISION_PROJECTION_CKPT_KEY = "modalities.vision_perception.encoder.vision_projection.weight"


def _gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """Exact-erf GELU.

    Deliberately not `approximate="tanh"`: the reference builds `nn.GELU()`
    and the checkpoint's `hidden_act` is `"gelu"`. Keeping it a distinct
    function object also keeps `MLP` off its fused GELU(tanh) epilogue, which
    would silently substitute the tanh approximation.
    """
    return F.gelu(x)


def precompute_rope_freqs_cis(
    dim: int,
    max_height: int,
    max_width: int,
    *,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the 2D RoPE table up to `max_height` x `max_width` patches.

    Returns `[max_height, max_width, dim // 2, 2]` where the last axis is
    `(cos, sin)`::

        angle[h, w, :dim // 4] = (w + 1) * theta ** (-4i / dim)   # width first
        angle[h, w, dim // 4:] = (h + 1) * theta ** (-4i / dim)

    Three things here are easy to get wrong, and parity with the reference
    depends on all three: positions are **1-indexed**, the width half comes **first**, and
    there are `dim // 4` base frequencies per axis (not `dim // 2`).
    """
    if dim % 4:
        raise ValueError(f"2D RoPE needs head_dim divisible by 4, got {dim}")
    half_dim = dim // 2
    freq_range = torch.arange(0, half_dim, 2, dtype=dtype, device=device)[: half_dim // 2]
    inv_freq = 1.0 / (theta ** (freq_range / half_dim))

    pos_h = torch.arange(1, max_height + 1, dtype=dtype, device=device)
    pos_w = torch.arange(1, max_width + 1, dtype=dtype, device=device)
    mesh_h, mesh_w = torch.meshgrid(pos_h, pos_w, indexing="ij")

    freq_h = torch.einsum("hw,f->hwf", mesh_h, inv_freq)
    freq_w = torch.einsum("hw,f->hwf", mesh_w, inv_freq)
    freq = torch.cat([freq_w, freq_h], dim=-1)
    return torch.stack([torch.cos(freq), torch.sin(freq)], dim=-1)


def apply_rope_2d(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate `q`/`k` `[seq, heads, head_dim]` by `freqs_cis` `[seq, head_dim // 2, 2]`.

    Adjacent elements are paired (`[0,1], [2,3], ...`); a half-split rotation
    would use `[0, head_dim // 2]` pairs and produce a plausible-looking wrong
    answer. Computed in fp32 and cast back, matching the reference.
    """
    cos = freqs_cis[..., 0].unsqueeze(1)
    sin = freqs_cis[..., 1].unsqueeze(1)

    q_pairs = q.float().reshape(*q.shape[:-1], -1, 2)
    k_pairs = k.float().reshape(*k.shape[:-1], -1, 2)

    q0, q1 = q_pairs[..., 0], q_pairs[..., 1]
    k0, k1 = k_pairs[..., 0], k_pairs[..., 1]
    q_out = torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1).flatten(-2)
    k_out = torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


@lru_cache(maxsize=256)
def _window_geometry(
    grid_h: int, grid_w: int, window_h: int, window_w: int
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Window permutation and per-window lengths for one clip.

    Returns a **host** int64 permutation of `range(grid_h * grid_w)` that
    reorders row-major patch order into window-major order, plus the token count
    of each window. Built on the host on purpose: it depends only on the
    geometry, so caching it here keeps `forward` free of the boolean-mask
    compaction (a device-to-host sync) the reference does on device.
    """
    padded_h = math.ceil(grid_h / window_h) * window_h
    padded_w = math.ceil(grid_w / window_w) * window_w
    n_h, n_w = padded_h // window_h, padded_w // window_w

    perm = torch.arange(grid_h * grid_w, dtype=torch.int64).view(grid_h, grid_w)
    perm = F.pad(perm, (0, padded_w - grid_w, 0, padded_h - grid_h), value=-1)
    # rearrange("(h r1 w r2) -> (h w r1 r2)")
    perm = perm.reshape(n_h, window_h, n_w, window_w).permute(0, 2, 1, 3).reshape(-1)
    perm = perm[perm != -1]

    lengths = tuple(
        min(window_h, grid_h - i * window_h) * min(window_w, grid_w - j * window_w)
        for i in range(n_h)
        for j in range(n_w)
    )
    assert perm.numel() == grid_h * grid_w == sum(lengths)
    return perm, lengths


@lru_cache(maxsize=256)
def _pixel_shuffle_permutation(grid_h: int, grid_w: int, downsample_factor: int) -> torch.Tensor:
    """Host permutation that groups each `r x r` patch block contiguously.

    `rearrange("(h r1 w r2) -> (h w r1 r2)")`; the pixel shuffle then folds
    each group of `r ** 2` rows into one row of `r ** 2 * dim` features.
    """
    if grid_h % downsample_factor or grid_w % downsample_factor:
        raise ValueError(
            f"grid {grid_h}x{grid_w} is not divisible by downsample_factor {downsample_factor}; "
            "the input processor must fit every canvas to a multiple of "
            f"{downsample_factor} patches per side"
        )
    r = downsample_factor
    perm = torch.arange(grid_h * grid_w, dtype=torch.int64)
    return perm.reshape(grid_h // r, r, grid_w // r, r).permute(0, 2, 1, 3).reshape(-1)


def vision_rows_for_grid(t: int, h: int, w: int, cfg: MuseSparkOpenVisionConfig) -> int:
    """Decoder rows one medium occupies -- the tower's output first dim for it.

    A `grid_thw` row is `(clips, grid_h, grid_w)`: `t` counts temporal
    clips (`frames // patch_temporal`, so 1 for a still image) and `h`/`w`
    are **patch** counts of the resized canvas, not pixels. Each clip is one
    attention segment of `grid_h * grid_w` patches. Must equal the input
    processor's placeholder count for the medium, or the encoder's output no
    longer lines up with the placeholders and embedding caching is skipped.
    """
    return _validate_clips(t) * (h // cfg.downsample_factor) * (w // cfg.downsample_factor)


def _validate_clips(t: int) -> int:
    if t < 1:
        raise ValueError(f"grid_thw clip count must be at least 1, got {t}")
    return t


def vision_attn_backend(decoder_attn_backend: str) -> str:
    """The tower's attention backend, given the decoder's.

    The TRTLLM backend has no fused context-FMHA kernel for this tower's
    head_dim (96), so every tower call takes the padded unfused-MHA fallback.
    That path sizes its workspace from the metadata MAXIMA
    (`max_num_requests * max_seq_len ** 2` per head, see
    `AttentionOp::getWorkspaceSizeForContext`), not from the call's actual
    segment lengths. The window metadata declares one request per budget token,
    so with an 8192-token encoder budget the first windowed layer asked for a
    232 GiB workspace at startup encoder profiling; the full-attention metadata
    (`max_num_tokens // 4` requests at a budget-length maximum) would ask for
    far more. No capacity declaration fixes this: the two maxima cannot
    co-occur under the packed token budget, but the workspace formula prices
    them as if they did.

    FLASHINFER runs the same segments ragged with no quadratic workspace and is
    the backend the tower was validated on, so a TRTLLM decoder gets a
    FLASHINFER tower. Other decoder backends keep their choice.
    """
    return "FLASHINFER" if decoder_attn_backend == "TRTLLM" else decoder_attn_backend


def _unquantized_vision_model_config(
    model_config: ModelConfig[MuseSparkOpenConfig],
) -> ModelConfig[MuseSparkOpenConfig]:
    """A copy of `model_config` with quantization and sparsity turned off.

    The perception tower is BF16 even in the NVFP4 checkpoint -- all 1589
    perception tensors are BF16 and no scale tensor exists under any perception
    prefix. That is because the tower was never quantized, **not** because
    `hf_quant_config.json` excludes it: its `exclude_modules` list is
    `["self_attn", "lm_head", "layers.0.", "layers.61."]` with no vision entry,
    so inheriting the parent's quant config would build an NVFP4 tower against
    BF16 weights. Every tower submodule is therefore built with
    `quant_config=None`, explicitly. The decoder's sparse-attention and
    KV-compression settings are cleared for the same reason: they describe the
    text stack, and the tower runs cacheless context-only attention.
    """
    import copy

    vision_config = copy.copy(model_config)
    vision_config.extra_attrs = copy.copy(model_config.extra_attrs)
    vision_config._frozen = False
    vision_config.quant_config = None
    vision_config.quant_config_dict = None
    vision_config.per_layer_quant_configs = None
    vision_config.sparse_attention_config = None
    vision_config.kv_cache_compression_config = None
    vision_config.pretrained_config = copy.copy(model_config.pretrained_config)
    return vision_config


class MuseSparkOpenVisionAttention(Attention):
    """Encoder self-attention: fused QKV, 2D RoPE, cacheless segment attention.

    Per-clip (block-diagonal) masking comes from the attention metadata, which
    presents each clip -- or each 32x32 window on the sparse layers -- as its own
    context sequence. There is no attention mask tensor.
    """

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        *,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=None,
            bias=True,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=model_config.torch_dtype,
            # out_proj carries a bias. Linear applies a row-parallel bias on
            # rank 0 only, before the all-reduce, which sums to exactly one bias
            # on the reduced result -- the post-reduce placement the reference
            # gets from skip_bias_add.
            dense_bias=True,
            config=model_config,
            reduce_output=(
                not model_config.mapping.enable_attention_dp and model_config.mapping.tp_size > 1
            ),
        )

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(seq_len, self.num_key_value_heads, self.head_dim)

        q, k = apply_rope_2d(q, k, freqs_cis)

        q = q.reshape(seq_len, -1)
        k = k.reshape(seq_len, -1)
        v = v.reshape(seq_len, -1)
        q, k, v = self.convert_qkv(q, k, v)
        output = self.forward_impl(
            q=q,
            k=k,
            v=v,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
            attention_window_size=None,
            attention_mask_data=None,
            mrope_config=None,
            attention_sinks=None,
        )
        return self.o_proj(output, layer_idx=self.layer_idx)


class MuseSparkOpenVisionMLP(MLP):
    """`c_fc -> exact GELU -> c_proj` with biases on both projections."""

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        *,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=True,
            activation=_gelu_exact,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
        )


class MuseSparkOpenVisionEncoderLayer(nn.Module):
    """Pre-norm block: `ln_1 -> attn -> +res -> ln_2 -> mlp -> +res`."""

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        vision_config: MuseSparkOpenVisionConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(
            hidden_size=vision_config.hidden_size,
            eps=vision_config.norm_eps,
            dtype=model_config.torch_dtype,
        )
        self.attn = MuseSparkOpenVisionAttention(
            model_config,
            layer_idx=layer_idx,
            hidden_size=vision_config.hidden_size,
            num_heads=vision_config.num_attention_heads,
        )
        self.ln_2 = LayerNorm(
            hidden_size=vision_config.hidden_size,
            eps=vision_config.norm_eps,
            dtype=model_config.torch_dtype,
        )
        self.mlp = MuseSparkOpenVisionMLP(
            model_config,
            layer_idx=layer_idx,
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_metadata, freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


class MuseSparkOpenVisionEncoder(nn.Module):
    """The 50-layer stack. Exists so the module path matches `encoder.layers.{i}`."""

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        vision_config: MuseSparkOpenVisionConfig,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MuseSparkOpenVisionEncoderLayer(model_config, vision_config, layer_idx)
                for layer_idx in range(vision_config.num_hidden_layers)
            ]
        )
        self.full_attention_layers = frozenset(vision_config.full_attention_layers)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        full_attn_metadata: AttentionMetadata,
        window_attn_metadata: Optional[AttentionMetadata],
    ) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.full_attention_layers or window_attn_metadata is None:
                x = layer(x, full_attn_metadata, freqs_cis)
            else:
                x = layer(x, window_attn_metadata, freqs_cis)
        return x


class MuseSparkOpenVisionAdapter(nn.Module):
    """`c_fc -> GELU -> c_proj -> GELU`, both projections unbiased.

    The trailing GELU is not a typo: the reference's `simple_mlp` activates the
    output projection as well.
    """

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        *,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        mapping = model_config.mapping
        # Under attention DP the tower runs per-rank on that rank's local
        # requests only, so no cross-rank collective may appear anywhere in
        # it: a rank whose batch has no image never enters the tower, and a
        # row-parallel c_proj all-reduce would then wait on peers that are
        # already inside the decoder's MoE dispatch (deadlock until the
        # dispatch completion-flag timeout). Replicate the adapter in that
        # case, matching the tower attention (tp_size=1 under attention DP)
        # and MLP (overridden_tp_size=1); the math of the full-width GEMMs
        # is identical to the sharded column+row pair.
        adapter_tp = not mapping.enable_attention_dp
        logger.info_once(
            "MuseSparkOpenVisionAdapter: "
            + ("TP-sharded" if adapter_tp else "replicated (attention DP)"),
            key="musespark_open_vision_adapter_tp",
        )
        self.c_fc = Linear(
            input_dim,
            output_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN if adapter_tp else None,
            quant_config=None,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.c_proj = Linear(
            output_dim,
            output_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW if adapter_tp else None,
            quant_config=None,
            allreduce_strategy=model_config.allreduce_strategy,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu_exact(self.c_proj(_gelu_exact(self.c_fc(x))))


class MuseSparkOpenNaViTVisionTower(nn.Module, MultimodalEncoderMixin):
    """Packed variable-resolution NaViT tower, projected into the decoder's space.

    `forward` returns **unnormed** rows; the decoder's `embed_norm` is the
    single application of the weightless RMSNorm for both text and vision.
    """

    def __init__(self, model_config: ModelConfig[MuseSparkOpenConfig]) -> None:
        super().__init__()
        vision_config = getattr(model_config.pretrained_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(
                "MuseSparkOpen checkpoint has no vision_config; cannot build the tower"
            )
        if not isinstance(vision_config, MuseSparkOpenVisionConfig):
            vision_config = MuseSparkOpenVisionConfig.from_hf(
                vision_config if isinstance(vision_config, dict) else vision_config.to_dict()
            )
        self.config = vision_config

        self.model_config = _unquantized_vision_model_config(model_config)
        self.model_config.pretrained_config.head_dim = vision_config.head_dim
        # Why the tower may not share the decoder's attention backend: see
        # vision_attn_backend. Set before _frozen so every tower submodule and
        # metadata_cls below read the same choice.
        self.model_config.attn_backend = vision_attn_backend(model_config.attn_backend)
        self.model_config._frozen = True
        mapping = self.model_config.mapping
        dtype = self.model_config.torch_dtype

        hidden_size = vision_config.hidden_size
        self.hidden_size = hidden_size
        self.patch_dim = (
            vision_config.num_channels * vision_config.patch_temporal * vision_config.patch_size**2
        )
        self.patch_temporal = vision_config.patch_temporal
        self.patch_size = vision_config.patch_size
        self.downsample_factor = vision_config.downsample_factor
        self.window_size = vision_config.pos_emb_grid_size
        self.sparse_attention_factor = vision_config.sparse_attention_factor
        self.output_dim = model_config.pretrained_config.hidden_size

        # Patch embedding. A Linear over pre-vectorized patches, not a Conv2d,
        # and replicated rather than column-parallel: the [1536, 1176] weight is
        # small enough that replication is cheaper than the all-gather a sharded
        # output would need before ln_pre, and the arithmetic is identical.
        self.conv1_linear = Linear(
            self.patch_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=None,
            quant_config=None,
        )

        # Learned positional embedding on a fixed 32x32 grid (one 448px training
        # image), bilinearly resampled to each clip's grid at run time.
        self.pos_embed = nn.Parameter(
            torch.empty(vision_config.pos_emb_grid_size**2, hidden_size, dtype=dtype)
        )

        self.ln_pre = LayerNorm(hidden_size=hidden_size, eps=vision_config.norm_eps, dtype=dtype)
        self.encoder = MuseSparkOpenVisionEncoder(self.model_config, vision_config)
        self.ln_post = LayerNorm(hidden_size=hidden_size, eps=vision_config.norm_eps, dtype=dtype)

        self.adapter = MuseSparkOpenVisionAdapter(
            self.model_config,
            input_dim=vision_config.vision_output_dim,
            output_dim=vision_config.projector_input_dim,
        )

        # Bridges the vision space into the decoder's embedding space. Kept
        # unquantized and replicated: `use_rowwise_vision_projection` selects
        # only the reference's TP layout (row-parallel over a replicated input
        # versus column-parallel plus a gather), and both reduce to this single
        # full-width GEMM. Replicating a [8192, 4096] weight costs 64 MiB per
        # rank and removes a collective from the encoder's tail.
        self.vision_projection = Linear(
            vision_config.projector_input_dim,
            self.output_dim,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=None,
            quant_config=None,
        )

        # 2D RoPE table, sliced per clip. After the pixel shuffle a medium may
        # hold at most `max_num_image_tokens` rows, so before it there are at
        # most that many times downsample_factor ** 2 patches; the square root
        # bounds either axis.
        max_pre_downsample = vision_config.max_num_image_tokens * self.downsample_factor**2
        self.max_rope_grid = max(
            int(math.sqrt(max_pre_downsample)) + 1, vision_config.pos_emb_grid_size
        )
        self.register_buffer(
            "precomputed_freqs_cis",
            precompute_rope_freqs_cis(
                vision_config.head_dim,
                self.max_rope_grid,
                self.max_rope_grid,
                theta=vision_config.rope_theta,
            ),
            persistent=False,
        )

        self.metadata_cls = get_attention_backend(self.model_config.attn_backend).Metadata
        self.full_attn_metadata: Optional[AttentionMetadata] = None
        self.window_attn_metadata: Optional[AttentionMetadata] = None
        self._attn_max_seq_len = max(1, self.model_config.max_num_tokens or 1)
        self._window_attn_max_seq_len = min(self._attn_max_seq_len, self.window_size**2)

    # ---- engine-driven attention metadata --------------------------------

    def get_encoder_attention_metadata_capacity(self, max_num_tokens: int) -> Dict[str, int]:
        """Bound the two metadata objects this encoder uses.

        Full attention gets one sequence per clip; the loosest bound without
        processor geometry is one clip per `downsample_factor ** 2` tokens,
        since a clip contributes at least that many patches. Window attention
        gets one sequence per 32x32 window, and a window holds at least one
        patch, so it needs the full token budget.
        """
        min_patches_per_clip = self.downsample_factor**2
        return {
            "full_attention": max(1, max_num_tokens // min_patches_per_clip),
            "window_attention": max(1, max_num_tokens),
        }

    def setup_attn_metadata(
        self,
        max_num_tokens: int,
        attention_metadata_capacity: Optional[Dict[str, int]] = None,
    ) -> None:
        # The engine's capacity override comes from the input processor, which
        # does not have to know both of this encoder's metadata objects. Take
        # what it gives for the keys it names and keep the derived bound for the
        # rest, rather than failing on a partial dict.
        capacities = self.get_encoder_attention_metadata_capacity(max_num_tokens)
        if attention_metadata_capacity:
            unknown = set(attention_metadata_capacity) - set(capacities)
            if unknown:
                raise ValueError(
                    f"unknown vision attention-metadata capacity keys {sorted(unknown)}; "
                    f"this encoder has {sorted(capacities)}"
                )
            capacities.update(attention_metadata_capacity)
        kwargs = dict(max_num_tokens=max_num_tokens, kv_cache_manager=None)
        self.full_attn_metadata = self.metadata_cls(
            max_num_requests=capacities["full_attention"], **kwargs
        )
        self.window_attn_metadata = self.metadata_cls(
            max_num_requests=capacities["window_attention"], **kwargs
        )
        self.set_attn_max_seq_len(max_num_tokens)

    def set_attn_max_seq_len(self, max_seq_len: int) -> None:
        """Set the longest full-attention segment; windows are bounded by shape.

        The two metadata objects need **separate** maxima. A full-attention
        segment is a whole clip and can be as long as the token budget, but a
        window is a `window_size` x `window_size` tile, so it can never hold
        more than `window_size ** 2` patches no matter how large the budget
        is. Handing the window path the budget-sized maximum is not merely
        loose: this head size has no fused MHA kernel on this architecture, and
        the unfused path sizes its workspace from
        `max_num_requests * max_seq_len ** 2`. With one request declared per
        token and a budget-sized maximum, a single 1564-patch image asked for
        405 GB and the tower raised instead of running.
        """
        if max_seq_len <= 0:
            raise ValueError(f"vision attention max_seq_len must be positive, got {max_seq_len}")
        self._attn_max_seq_len = max_seq_len
        self._window_attn_max_seq_len = min(max_seq_len, self.window_size**2)

    def _max_seq_len_for(self, attn_metadata: AttentionMetadata) -> int:
        """Which of the two maxima this metadata object is entitled to."""
        if attn_metadata is self.window_attn_metadata:
            return self._window_attn_max_seq_len
        return self._attn_max_seq_len

    @property
    def attn_max_seq_len(self) -> int:
        """Longest full-attention segment (one clip, `grid_h * grid_w`
        patches) this tower accepts. Exposed so request admission can reject
        an oversized item with a request-scoped error instead of the same
        check raising mid-forward and failing the whole batch."""
        return self._attn_max_seq_len

    def prepare_attn_metadata(
        self, seq_lens: Sequence[int], attn_metadata: AttentionMetadata
    ) -> AttentionMetadata:
        """Present each clip (or window) to the attention op as its own context.

        Everything here is host arithmetic plus one pinned host-to-device copy;
        nothing reads a device tensor back.
        """
        if attn_metadata is None:
            raise RuntimeError(
                "vision AttentionMetadata is not initialized; setup_attn_metadata must run "
                "before the tower's first forward"
            )
        seq_lens = [int(s) for s in seq_lens]
        if not seq_lens:
            raise ValueError("vision attention needs at least one segment")
        max_seq_len = self._max_seq_len_for(attn_metadata)
        longest = max(seq_lens)
        if longest > max_seq_len:
            raise ValueError(
                f"vision attention segment length {longest} exceeds the configured maximum "
                f"{max_seq_len}; raise the encoder token budget or lower max_pixels"
            )

        num_segments = len(seq_lens)
        seq_lens_host = torch.tensor(seq_lens, dtype=torch.int, pin_memory=prefer_pinned())
        cu_seqlens = torch.empty(num_segments + 1, dtype=torch.int32, pin_memory=prefer_pinned())
        cu_seqlens[0] = 0
        torch.cumsum(seq_lens_host, dim=0, out=cu_seqlens[1:])

        attn_metadata.num_contexts = num_segments
        attn_metadata.request_ids = list(range(1, num_segments + 1))
        attn_metadata.seq_lens = seq_lens_host
        cu_seqlens = cu_seqlens.to(device=attn_metadata.seq_lens_cuda.device, non_blocking=True)
        attn_metadata.cu_q_seqlens = cu_seqlens
        attn_metadata.cu_kv_seqlens = cu_seqlens
        # A stable max keeps the attention op's cache key from churning with
        # every new image resolution; the real lengths travel in cu_seqlens.
        attn_metadata.max_seq_len = max_seq_len
        attn_metadata.prepare_encoder_only()
        return attn_metadata

    # ---- geometry ---------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.pos_embed.device

    @property
    def dtype(self) -> torch.dtype:
        return self.pos_embed.dtype

    def _host_grid_rows(
        self, grid_thw: torch.Tensor, grid_thw_cpu: Optional[torch.Tensor]
    ) -> List[Tuple[int, int, int]]:
        """Per-medium `(t, h, w)` on the host.

        The permutations, segment lengths and RoPE slices are all data-dependent
        shapes, so this geometry has to exist as Python ints. Reading it back off
        a device tensor would be a device-to-host sync in the middle of the
        encoder, so a device tensor is refused here rather than quietly copied.
        The geometry is a handful of ints the input processor already has on
        the host.
        """
        source = grid_thw_cpu if grid_thw_cpu is not None else grid_thw
        if source.device.type != "cpu":
            raise ValueError(
                f"MuseSparkOpen vision tower needs the grid geometry on the host, got a "
                f"{source.device} tensor and no host copy. Reading it back would sync the "
                "device inside forward. Pass grid_thw_cpu, or keep *_grid_thw out of the "
                "model's multimodal_data_device_paths so it is never staged to the device."
            )
        return [(int(t), int(h), int(w)) for t, h, w in source.tolist()]

    def _positional_embedding(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Sample the 32x32 learned grid at the given normalized coordinates.

        `align_corners=False` puts the learned samples at cell centres, which
        is what the pixel-centre coordinate construction assumes.
        """
        grid_side = self.config.pos_emb_grid_size
        weight = self.pos_embed.reshape(grid_side, grid_side, -1).permute(2, 0, 1).unsqueeze(0)
        sampled = F.grid_sample(
            weight,
            pos_xy.to(weight.dtype)[None, None],
            mode=self.config.grid_sample_mode,
            align_corners=False,
        )
        return sampled[0, :, 0, :].permute(1, 0)

    def _grid_coordinates(
        self, grid_h: int, grid_w: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Pixel-centre sampling coordinates for one clip, reference layout.

        `indexing="xy"` makes `meshgrid(pos_h, pos_w)` return `[grid_w,
        grid_h]` tensors, so the flattened coordinate list runs width-outer and
        each entry is `(h_coord, w_coord)` in a slot `grid_sample` reads as
        `(x, y)`. That is a transposition relative to the row-major patch
        order, and it is what the model was trained with -- replicated verbatim,
        not corrected.
        """
        inv_h, inv_w = 1.0 / grid_h, 1.0 / grid_w
        pos_h = torch.linspace(-1 + inv_h, 1 - inv_h, grid_h, device=device, dtype=dtype)
        pos_w = torch.linspace(-1 + inv_w, 1 - inv_w, grid_w, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1).reshape(-1, 2)

    def _rope_slice(self, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        """RoPE rows for one clip in row-major patch order, `[grid_h * grid_w, head_dim // 2, 2]`."""
        half = self.config.head_dim // 2
        if grid_h > self.max_rope_grid or grid_w > self.max_rope_grid:
            table = precompute_rope_freqs_cis(
                self.config.head_dim,
                grid_h,
                grid_w,
                theta=self.config.rope_theta,
                device=device,
            )
            return table.reshape(-1, half, 2)
        return self.precomputed_freqs_cis[:grid_h, :grid_w].reshape(-1, half, 2)

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        grid_thw_cpu: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode a packed batch of images and video clips.

        Args:
            pixel_values: `[total_patches, patch_dim]`, patches of every medium
                concatenated in request order, vectorized `[t][c][py][px]`.
            grid_thw: `[num_media, 3]` int32 `(clips, grid_h, grid_w)`, the
                last two in **patch** units, per the input processor. A still image is one clip; video is
                `frames // patch_temporal` clips.
            grid_thw_cpu: host-resident copy of `grid_thw`. Supplying it keeps
                the forward free of device-to-host syncs; without it the tower
                reads `grid_thw` back and says so once.

        Returns:
            `[total_rows, hidden_size_of_decoder]`, unnormed.
        """
        device = pixel_values.device
        rows = self._host_grid_rows(grid_thw, grid_thw_cpu)
        if not rows:
            return pixel_values.new_empty((0, self.output_dim))
        if pixel_values.shape[-1] != self.patch_dim:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[-1]} features per patch, expected "
                f"{self.patch_dim} = num_channels * patch_temporal * patch_size ** 2 "
                f"vectorized {PATCH_VECTORIZATION}"
            )

        coords: List[torch.Tensor] = []
        freqs: List[torch.Tensor] = []
        clip_lens: List[int] = []
        window_perms: List[torch.Tensor] = []
        window_lens: List[int] = []
        shuffle_perms: List[torch.Tensor] = []
        offset = 0
        shuffled_offset = 0

        for t, grid_h, grid_w in rows:
            clip_tokens = grid_h * grid_w
            for _ in range(_validate_clips(t)):
                coords.append(
                    self._grid_coordinates(grid_h, grid_w, device=device, dtype=pixel_values.dtype)
                )
                freqs.append(self._rope_slice(grid_h, grid_w, device))
                clip_lens.append(clip_tokens)

                perm, lengths = _window_geometry(grid_h, grid_w, self.window_size, self.window_size)
                window_perms.append(perm + offset)
                window_lens.extend(lengths)

                shuffle_perms.append(
                    _pixel_shuffle_permutation(grid_h, grid_w, self.downsample_factor)
                    + shuffled_offset
                )
                offset += clip_tokens
                shuffled_offset += clip_tokens

        if offset != pixel_values.shape[0]:
            raise ValueError(
                f"grid_thw describes {offset} patches but pixel_values holds "
                f"{pixel_values.shape[0]}; the input processor and the tower disagree on geometry"
            )

        x = self.conv1_linear(pixel_values)
        x = x + self._positional_embedding(torch.cat(coords))
        x = self.ln_pre(x)

        freqs_cis = torch.cat(freqs).to(device=device, non_blocking=True)
        full_attn_metadata = self.prepare_attn_metadata(clip_lens, self.full_attn_metadata)

        use_windows = self.sparse_attention_factor > 1 and len(self.encoder.layers) > 1
        window_perm = None
        window_attn_metadata = None
        if use_windows:
            # One permutation for the whole stack instead of a gather per sparse
            # layer: reorder x and the RoPE table into window-major order here,
            # and undo it once after the last layer.
            window_perm = torch.cat(window_perms).to(device=device, non_blocking=True)
            x = x[window_perm]
            freqs_cis = freqs_cis[window_perm]
            window_attn_metadata = self.prepare_attn_metadata(
                window_lens, self.window_attn_metadata
            )

        x = self.encoder(x, freqs_cis, full_attn_metadata, window_attn_metadata)

        if window_perm is not None:
            unpermuted = torch.empty_like(x)
            unpermuted.index_copy_(0, window_perm, x)
            x = unpermuted

        x = self.ln_post(x)

        # Pixel shuffle: group each 2x2 patch block, then fold the block into the
        # feature axis channel-major / sub-position-minor -- `(b r) d -> b (d r)`,
        # so feature f of sub-position s lands at f * r ** 2 + s.
        shuffle_perm = torch.cat(shuffle_perms).to(device=device, non_blocking=True)
        block = self.downsample_factor**2
        x = x[shuffle_perm]
        x = (
            x.reshape(x.shape[0] // block, block, x.shape[-1])
            .transpose(1, 2)
            .reshape(x.shape[0] // block, x.shape[-1] * block)
        )

        x = self.adapter(x)
        return self.vision_projection(x)

    def full_attention_layer_indices(self) -> List[int]:
        """Layers that attend over a whole clip -- 13 of 50 in the shipped config."""
        return sorted(self.encoder.full_attention_layers)


def vision_layer_is_full_attention(layer_idx: int, cfg: MuseSparkOpenVisionConfig) -> bool:
    """Re-exported for the weight mapper; the schedule lives in the config."""
    return layer_uses_full_vision_attention(
        layer_idx, cfg.num_hidden_layers, cfg.sparse_attention_factor
    )
