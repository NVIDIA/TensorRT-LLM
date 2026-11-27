# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replicated Pixtral image tower and learned spatial projector."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.models.pixtral.modeling_pixtral import apply_rotary_pos_emb

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.inputs.evs import compute_retention_mask
from tensorrt_llm.inputs.multimodal import MultimodalParams

from .decoder import rms_norm
from .manifest import load_unquantized, row_window

_VISION_PREFIX = "model.vision_tower."
_PROJECTOR_PREFIX = "model.multi_modal_projector."
_PIXTRAL_EPS = 1e-5


def fuse_video_template(projected, grid, merge_size, rate, template, text_embedding):
    """Keep row ends and timestamp text in their true positions after spatial pruning."""
    mask = compute_retention_mask(projected, grid, merge_size, rate)
    ids = torch.tensor(template, device=projected.device)
    # The template begins at the first patch and ends at the last frame terminator.
    patches_in_text = ids == ids[0]
    keep = torch.ones_like(patches_in_text)
    keep[patches_in_text] = mask
    retained_ids = ids[keep]
    fused = text_embedding(retained_ids)
    fused[retained_ids == ids[0]] = projected[mask]
    return fused


def merge_order_positions(rows: int, cols: int, side: int, merge: int) -> np.ndarray:
    """Rotary position ids (`row * side + column`) of an image's patches in merge order."""
    i, j, kh, kw = np.meshgrid(
        np.arange(rows // merge),
        np.arange(cols // merge),
        np.arange(merge),
        np.arange(merge),
        indexing="ij",
    )
    return ((merge * i + kh) * side + (merge * j + kw)).reshape(-1)


def axial_cos_sin_table(side: int, head_dim: int, theta: float) -> torch.Tensor:
    """`[side * side, head_dim]` fp32: per position id, `cos(f) | sin(f)` of the
    `head_dim / 2` rotated pairs -- the first half turned by the row, the second
    by the column."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(side, dtype=torch.float32)
    rows = torch.outer(pos, inv[0::2])  # [side, head_dim / 4]
    cols = torch.outer(pos, inv[1::2])
    angles = torch.cat(
        [
            rows[:, None, :].expand(side, side, -1),
            cols[None, :, :].expand(side, side, -1),
        ],
        dim=-1,
    ).reshape(side * side, head_dim // 2)
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def _flatten_patch_conv(t: torch.Tensor) -> torch.Tensor:
    """Conv2d weight `[H, C, p, p]` -> GEMM weight `[H, C * p * p]`."""
    return t.reshape(t.shape[0], -1)


def _merge_patch_major(merge: int):
    """Merge weight `[H, H * s * s]` with inputs `(channel, patch)` -> `(patch, channel)`."""

    def transform(t: torch.Tensor) -> torch.Tensor:
        out, cols = t.shape
        return (
            t.reshape(out, cols // (merge * merge), merge * merge)
            .transpose(1, 2)
            .reshape(out, cols)
        )

    return transform


def _manifest(vision: "ShortcutVisionFrontend") -> dict:
    """param key -> list of (checkpoint key, source window | None, destination
    rows | None, transform | None), the form of the targets' weight tables."""
    hd, il = vision.num_heads * vision.head_dim, vision.intermediate_size
    out = {
        "patch": [(_VISION_PREFIX + "patch_conv.weight", None, None, _flatten_patch_conv)],
        "ln_pre": [(_VISION_PREFIX + "ln_pre.weight", None, None, None)],
        "proj_norm": [(_PROJECTOR_PREFIX + "norm.weight", None, None, None)],
        "merge": [
            (
                _PROJECTOR_PREFIX + "patch_merger.merging_layer.weight",
                None,
                None,
                _merge_patch_major(vision.merge_size),
            )
        ],
        "proj_1": [(_PROJECTOR_PREFIX + "linear_1.weight", None, None, None)],
        "proj_2": [(_PROJECTOR_PREFIX + "linear_2.weight", None, None, None)],
    }
    for i in range(vision.num_layers):
        p = f"{_VISION_PREFIX}transformer.layers.{i}."
        a, f = p + "attention.", p + "feed_forward."
        out[f"l{i}_attn_norm"] = [(p + "attention_norm.weight", None, None, None)]
        out[f"l{i}_ffn_norm"] = [(p + "ffn_norm.weight", None, None, None)]
        out[f"l{i}_qkv"] = [
            (a + f"{name}_proj.weight", None, row_window(k * hd, hd), None)
            for k, name in enumerate(("q", "k", "v"))
        ]
        out[f"l{i}_o"] = [(a + "o_proj.weight", None, None, None)]
        out[f"l{i}_gate_up"] = [
            (f + "gate_proj.weight", None, row_window(0, il), None),
            (f + "up_proj.weight", None, row_window(il, il), None),
        ]
        out[f"l{i}_down"] = [(f + "down_proj.weight", None, None, None)]
    return out


class ShortcutVisionFrontend(nn.Module):
    """Produce identical image embeddings on each participating decoder rank."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        cfg = model_config.pretrained_config
        vision = cfg.vision_config
        if (
            vision.model_type != "pixtral"
            or vision.hidden_act != "silu"
            or vision.rope_parameters["rope_type"] != "axial"
            or vision.num_channels != 3
            or cfg.vision_feature_layer != -1
            or cfg.multimodal_projector_bias
        ):
            raise ValueError(
                "Expected a bias-free Pixtral axial image tower and final-layer projector."
            )
        heads, hidden, inter = (
            vision.num_attention_heads,
            vision.hidden_size,
            vision.intermediate_size,
        )
        self.head_dim = vision.head_dim
        if heads * self.head_dim != hidden or self.head_dim % 4:
            raise ValueError("Axial vision heads must span the hidden dimension in groups of four.")
        # Reducing sharded BF16 image projections changes the checkpoint's feature rows.
        # Keep the vision tower whole; the text decoder owns its parallel mapping.
        self.num_heads = heads
        self.intermediate_size = inter
        self.num_layers = vision.num_hidden_layers
        self.patch = vision.patch_size
        self.merge_size = cfg.spatial_merge_size
        self.side = vision.image_size // self.patch
        self.theta = vision.rope_parameters["rope_theta"]
        self.proj_eps = cfg.projector_norm_eps
        self.act = ACT2FN[cfg.projector_hidden_act]
        text_width = cfg.text_config.hidden_size

        def parameter(*shape):
            return nn.Parameter(torch.empty(*shape, dtype=torch.bfloat16), requires_grad=False)

        self.w = nn.ParameterDict(
            {
                "patch": parameter(hidden, 3 * self.patch**2),
                "ln_pre": parameter(hidden),
                "proj_norm": parameter(hidden),
                "merge": parameter(hidden, hidden * self.merge_size**2),
                "proj_1": parameter(text_width, hidden),
                "proj_2": parameter(text_width, text_width),
            }
        )
        for i in range(self.num_layers):
            prefix = f"l{i}_"
            self.w.update(
                {
                    prefix + "attn_norm": parameter(hidden),
                    prefix + "qkv": parameter(3 * hidden, hidden),
                    prefix + "o": parameter(hidden, hidden),
                    prefix + "ffn_norm": parameter(hidden),
                    prefix + "gate_up": parameter(2 * self.intermediate_size, hidden),
                    prefix + "down": parameter(hidden, self.intermediate_size),
                }
            )
        self.register_buffer("rotary_table", None, persistent=False)

    @torch.no_grad()
    def load_weights(self, weights: Mapping) -> None:
        entries = _manifest(self)
        actual = {key for key in weights if key.startswith((_VISION_PREFIX, _PROJECTOR_PREFIX))}
        expected = {key for tensors in entries.values() for key, *_ in tensors}
        if actual != expected:
            raise ValueError(
                f"Vision weight mismatch: missing {sorted(expected - actual)[:4]}, "
                f"unexpected {sorted(actual - expected)[:4]}"
            )
        for name, tensors in entries.items():
            load_unquantized(self.w[name], tensors, weights)
        with torch.device(self.w["patch"].device):
            self.rotary_table = axial_cos_sin_table(self.side, self.head_dim, self.theta)

    def _attention(self, x, prefix, cosine, sine):
        packed = F.linear(x, self.w[prefix + "qkv"])
        q, k, v = (
            t.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)[None]
            for t in packed.chunk(3, dim=-1)
        )
        q, k = apply_rotary_pos_emb(q, k, cosine, sine, unsqueeze_dim=0)
        # Materialize the checkpoint's BF16 probabilities in bounded query windows.
        # All patches in the same image attend to one another.
        outputs = []
        for query in q.split(256, dim=-2):
            scores = (query @ k.transpose(-1, -2)) * (self.head_dim**-0.5)
            probs = scores.softmax(dim=-1, dtype=torch.float32).to(v.dtype)
            outputs.append(probs @ v)
        attention = torch.cat(outputs, dim=-2).squeeze(0).transpose(0, 1).reshape(x.shape[0], -1)
        return F.linear(attention, self.w[prefix + "o"])

    def patch_embeddings(self, patches: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Apply the checkpoint's strided convolution to its reconstructed RGB image."""
        height, width = size
        p, m = self.patch, self.merge_size
        tiles = patches.reshape(height // (p * m), width // (p * m), m, m, 3, p, p)
        image = tiles.permute(4, 0, 2, 5, 1, 3, 6).reshape(1, 3, height, width).contiguous()
        kernel = self.w["patch"].reshape(-1, 3, p, p)
        # Pixtral concatenates raster patch rows before its first RMS reduction.
        features = F.conv2d(image, kernel, stride=p).flatten(2).transpose(1, 2)
        return features.squeeze(0).contiguous()

    def _block(self, x, i, cosine, sine):
        prefix = f"l{i}_"
        normalized = rms_norm(x, self.w[prefix + "attn_norm"], _PIXTRAL_EPS)
        x = x + self._attention(normalized, prefix, cosine, sine)
        normalized = rms_norm(x, self.w[prefix + "ffn_norm"], _PIXTRAL_EPS)
        gate, up = F.linear(normalized, self.w[prefix + "gate_up"]).chunk(2, dim=-1)
        return x + F.linear(F.silu(gate) * up, self.w[prefix + "down"])

    def encode_tower(self, patches, size):
        """Return pre-projector features in merge order for one image."""
        h, w = (edge // self.patch for edge in size)
        if (
            len(patches) != h * w
            or h % self.merge_size
            or w % self.merge_size
            or max(h, w) > self.side
        ):
            raise ValueError(
                "Image patches and grid dimensions are inconsistent with the vision tower."
            )
        positions = torch.as_tensor(
            merge_order_positions(h, w, self.side, self.merge_size), device=patches.device
        )
        # Keep each attention reduction in raster order, as in the checkpoint's Pixtral tower.
        raster = positions.argsort()
        cosine, sine = self.rotary_table[positions[raster]].to(patches.dtype).chunk(2, dim=-1)
        cosine, sine = torch.cat([cosine, cosine], -1), torch.cat([sine, sine], -1)
        x = rms_norm(self.patch_embeddings(patches, size), self.w["ln_pre"], _PIXTRAL_EPS)
        for i in range(self.num_layers):
            x = self._block(x, i, cosine, sine)
        return x[raster.argsort()]

    def project(self, features):
        normalized = rms_norm(features, self.w["proj_norm"], self.proj_eps)
        merged = normalized.reshape(len(features) // self.merge_size**2, -1)
        hidden = F.linear(merged, self.w["merge"])
        return F.linear(self.act(F.linear(hidden, self.w["proj_1"])), self.w["proj_2"])

    def _video_rows(self, patches, size, template, rate, text_embedding):
        height, width = (edge // self.patch for edge in size)
        frames = patches.split(height * width)
        projected = torch.cat([self.project(self.encode_tower(frame, size)) for frame in frames])
        if not template:
            return projected
        return fuse_video_template(
            projected, (len(frames), height, width), self.merge_size, rate, template, text_embedding
        )

    def forward(self, params: Sequence[MultimodalParams], text_embedding=None) -> torch.Tensor:
        rows = []
        for request in params:
            metadata = request.item_metadata
            if metadata is None:
                raise ValueError("Visual inputs require prompt-ordered item metadata.")
            for modality, index in metadata.item_refs:
                if modality not in ("image", "video"):
                    raise ValueError(f"Unsupported Shortcut input {modality}.")
                data = request.multimodal_data[modality]
                patches = metadata.input_partitions[modality]["patches"].select(
                    data["patches"], [index]
                )
                if modality == "image":
                    rows.append(
                        self.project(self.encode_tower(patches, data["image_sizes"][index]))
                    )
                else:
                    rows.append(
                        self._video_rows(
                            patches,
                            data["frame_sizes"][index],
                            data["templates"][index],
                            data["pruning_rate"],
                            text_embedding,
                        )
                    )
        if not rows:
            raise ValueError("The vision frontend requires at least one image or video.")
        return torch.cat(rows)
