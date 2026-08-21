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
"""Inkling multimodal towers and the shared multimodal input processor.

Four sections:

* Vision -- :class:`InklingImagePreprocessor` turns raw images into the
  ``vision_patches_bthwc`` tensor and :class:`InklingVisionModel` is the
  hierarchical-MLP encoder that maps those patches to one text-hidden row each.
* Audio -- :class:`InklingAudioPreprocessor` turns waveforms into discrete "dMel"
  bins and :class:`InklingAudioModel` embeds them to one row per frame.
* Video -- there is no separate video encoder: sampled frames are fed as ordinary
  images through the vision tower.
* Input processor -- :class:`InklingInputProcessor` expands each placeholder token
  into one token per feature row, and fails loudly when the counts disagree.

The preprocessing is pure vectorized numpy + torch, so the production path
carries no third-party kernel-cache dependency.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    DefaultInputProcessor,
)


class InklingRMSNorm(nn.Module):
    """RMSNorm over the last dim (fp32 variance, ``eps=1e-6``), shared by the
    vision and audio towers."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        return F.rms_norm(x, (self.hidden_size,), self.weight, self.variance_epsilon)


@torch.no_grad()
def _load_tower_weights(
    tower: nn.Module,
    weights: Dict[str, torch.Tensor],
    prefix: str,
    target_dtype: torch.dtype,
) -> None:
    """Strict-load a tower's checkpoint tensors, stripping ``prefix`` if present.

    Strict on purpose: each tower's module tree mirrors the checkpoint naming
    exactly, so a missing or extra key fails loudly rather than silently leaving
    a layer at init.
    """
    mapped = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in weights.items()}
    if any(p.is_meta for p in tower.parameters()):
        # Built under a meta-device init context (the full-model load path):
        # copy_ into a meta tensor is illegal, so assign the checkpoint tensors
        # directly (they carry the checkpoint's bf16 dtype).
        tower.load_state_dict(mapped, strict=True, assign=True)
        return
    mapped = {k: (v.to(target_dtype) if v.dtype != target_dtype else v) for k, v in mapped.items()}
    tower.load_state_dict(mapped, strict=True)


# ===========================================================================
# Vision tower: image preprocessing + hMLP encoder
# ===========================================================================
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)
PAD_RAW_VALUE = np.float32(-1.0 / 255.0)
# Normalized pad value; equals normalize(PAD_RAW_VALUE) so a canvas pre-filled
# with PAD_RAW_VALUE and normalized wholesale reproduces the per-patch pad.
PAD_NORM = (np.full((3,), PAD_RAW_VALUE, dtype=np.float32) - IMAGE_MEAN) / IMAGE_STD

# Default vision geometry (checkpoint ``vision_config``).
DEFAULT_PATCH_SIZE = 40
DEFAULT_TEMPORAL_PATCH_SIZE = 2  # a static image is temporally duplicated (T=2)
DEFAULT_RESCALE_IMAGE_FRAC = 2.0
DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE = 2048

# Inkling image placeholder id (``<|unused_200054|>``), the token the chat
# template renders for an image content part. One sentinel appears per image;
# the input processor expands it into one token per patch and the vision fusion
# overwrites those positions. It must be in-vocab, since the executor rejects
# out-of-range request token ids. ``config.json`` may override via
# ``image_token_id``.
DEFAULT_IMAGE_TOKEN_ID = 200054


def scaled_image_dimensions(
    width: int,
    height: int,
    rescale_image_frac: Optional[float] = DEFAULT_RESCALE_IMAGE_FRAC,
    rescale_image_max_upscaled_long_edge: Optional[int] = DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE,
) -> Tuple[int, int]:
    """Long-edge scale ``(width, height)`` before patching.

    Scales the long edge by ``rescale_image_frac`` (aspect preserved), optionally
    capping only upscaling (the cap never shrinks an image already above it),
    with half-away-from-zero rounding ``floor(v * ratio + 0.5)``.
    """
    if rescale_image_frac is None:
        return width, height
    long_edge = max(width, height)
    if long_edge == 0:
        return width, height
    target_long_edge = float(long_edge) * rescale_image_frac
    if rescale_image_max_upscaled_long_edge is not None:
        effective_cap = max(rescale_image_max_upscaled_long_edge, long_edge)
        target_long_edge = min(target_long_edge, float(effective_cap))
    ratio = target_long_edge / float(long_edge)
    if ratio == 1.0:
        return width, height

    def scale(value: int) -> int:
        return max(1, math.floor(float(value) * ratio + 0.5))

    return scale(width), scale(height)


def patch_grid(
    height: int, width: int, patch_size: int = DEFAULT_PATCH_SIZE
) -> Tuple[int, int, int]:
    """Return ``(num_patches, nph, npw)``.

    ``nph = ceil(H / P)``, ``npw = W // P + 1`` (the asymmetric ``+1`` width
    padding). One text-hidden token is emitted per patch, so
    ``placeholder_count == num_patches == nph * npw``.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")
    nph = (height + patch_size - 1) // patch_size
    npw = width // patch_size + 1
    return nph * npw, nph, npw


def _to_pil_rgb(image: Any):
    """Coerce one image input into a PIL ``RGB`` image.

    Accepts raw PNG/JPEG ``bytes``, a local path / ``file://`` URI, a PIL image,
    a numpy ``uint8`` HWC array, or a ``torch.Tensor``. HTTP/data URIs must be
    resolved to bytes upstream, so a URL raises rather than silently reaching for
    the network here.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            raise ValueError(
                "InklingImagePreprocessor received a URL/data: image; resolve "
                "it to bytes upstream before preprocessing."
            )
        path = image[len("file://") :] if image.startswith("file://") else image
        with open(path, "rb") as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        arr = arr.astype("uint8") if arr.dtype != np.uint8 else arr
        return Image.fromarray(arr).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image.astype("uint8") if image.dtype != np.uint8 else image
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


class InklingImagePreprocessor:
    """Raw images -> ``vision_patches_bthwc`` for the Inkling hMLP tower."""

    def __init__(
        self,
        patch_size: int = DEFAULT_PATCH_SIZE,
        temporal_patch_size: int = DEFAULT_TEMPORAL_PATCH_SIZE,
        rescale_image_frac: Optional[float] = DEFAULT_RESCALE_IMAGE_FRAC,
        rescale_image_max_upscaled_long_edge: Optional[
            int
        ] = DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be greater than zero")
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.rescale_image_frac = rescale_image_frac
        self.rescale_image_max_upscaled_long_edge = rescale_image_max_upscaled_long_edge
        self.dtype = dtype

    def _patchify(self, arr: np.ndarray) -> np.ndarray:
        """``(H, W, 3)`` uint8 -> ``(num_patches, P, P, 3)`` float32.

        Builds a PAD_RAW_VALUE canvas of ``(nph*P, npw*P, 3)``, drops the image
        into its top-left, normalizes the whole canvas in float32
        (``(uint8/255 - mean) / std``; the padded border becomes ``PAD_NORM``),
        then folds ``(nph, P, npw, P, 3)`` into row-major patch order ``i*npw+j``.
        """
        h, w = int(arr.shape[0]), int(arr.shape[1])
        num_patches, nph, npw = patch_grid(h, w, self.patch_size)
        p = self.patch_size
        inv255 = np.float32(1.0) / np.float32(255.0)

        canvas = np.empty((nph * p, npw * p, 3), dtype=np.float32)
        canvas[:] = PAD_RAW_VALUE
        canvas[:h, :w, :] = arr[:h, :w, :].astype(np.float32) * inv255
        canvas = (canvas - IMAGE_MEAN) / IMAGE_STD  # float32; border -> PAD_NORM

        # (nph*P, npw*P, 3) -> (nph, P, npw, P, 3) -> (nph, npw, P, P, 3)
        # -> (num_patches, P, P, 3) with patch index i*npw + j (row-major).
        patches = (
            canvas.reshape(nph, p, npw, p, 3).transpose(0, 2, 1, 3, 4).reshape(num_patches, p, p, 3)
        )
        return np.ascontiguousarray(patches)

    def encode_one(self, image: Any) -> torch.Tensor:
        """One image -> ``(num_patches, T, P, P, 3)`` tensor in ``self.dtype``."""
        pil = _to_pil_rgb(image)
        sw, sh = scaled_image_dimensions(
            pil.width,
            pil.height,
            self.rescale_image_frac,
            self.rescale_image_max_upscaled_long_edge,
        )
        if (sw, sh) != pil.size:
            from PIL import Image

            pil = pil.resize((sw, sh), Image.Resampling.LANCZOS)
        arr = np.array(pil, dtype=np.uint8, copy=True)
        patches = self._patchify(arr)  # (num_patches, P, P, 3) float32
        n = int(patches.shape[0])
        t = torch.from_numpy(patches).to(self.dtype)
        # temporal duplication: (n, 1, P, P, 3) -> (n, T, P, P, 3)
        return (
            t.view(n, 1, self.patch_size, self.patch_size, 3)
            .expand(n, self.temporal_patch_size, self.patch_size, self.patch_size, 3)
            .contiguous()
        )

    def preprocess(self, images: Union[Any, Sequence[Any]]) -> Dict[str, Any]:
        """Raw images -> ``{vision_patches_bthwc, num_patches, num_tokens}``.

        ``vision_patches_bthwc`` concatenates all images' patches along dim 0
        (one contiguous slice per image, in encounter order); ``num_patches`` /
        ``num_tokens`` are per-image and equal (one token per patch).
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        per_image: List[torch.Tensor] = []
        num_patches: List[int] = []
        for img in images:
            vp = self.encode_one(img)
            per_image.append(vp)
            num_patches.append(int(vp.shape[0]))
        if len(per_image) == 1:
            vision_patches_bthwc = per_image[0]
        elif per_image:
            vision_patches_bthwc = torch.cat(per_image, dim=0)
        else:
            vision_patches_bthwc = torch.empty(0, dtype=self.dtype)
        return {
            "vision_patches_bthwc": vision_patches_bthwc,
            "num_patches": num_patches,
            "num_tokens": list(num_patches),  # hMLP: one token per patch
        }


def _resolve_vision_geometry(config: Any) -> Tuple[int, int]:
    """Return ``(patch_size, temporal_patch_size)`` from a top-level or vision
    config, defaulting to the checkpoint's ``40`` / ``2``."""
    vision_config = getattr(config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", None)
    temporal = getattr(vision_config, "temporal_patch_size", None)
    return (
        int(patch_size) if patch_size else DEFAULT_PATCH_SIZE,
        int(temporal) if temporal else DEFAULT_TEMPORAL_PATCH_SIZE,
    )


def _resolve_image_token_id(config: Any) -> int:
    """The single ``<image>`` placeholder id the token stream carries per image.

    Prefers an explicit ``image_token_id`` on the (top-level or vision) config."""
    for obj in (config, getattr(config, "vision_config", None)):
        tok = getattr(obj, "image_token_id", None)
        if tok is not None:
            return int(tok)
    return DEFAULT_IMAGE_TOKEN_ID


def _prime_factors(n: int) -> List[int]:
    """Prime factors of ``n`` in ascending order."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    factors: List[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def _assign_monotone(cost: np.ndarray) -> List[int]:
    """Minimum-cost assignment of every row to a distinct, increasing column.

    Both axes are strictly increasing, so the cost matrix is Monge and the optimal
    assignment is non-crossing -- which makes this O(rows*cols) DP exact and keeps
    the tower off a runtime ``scipy`` dependency. Non-crossing matters on its own
    too: the caller consumes the result as an ordered scale progression.
    """
    n_rows, n_cols = cost.shape
    inf = float("inf")
    # best[i][j]: cost of assigning the first i rows within the first j columns.
    best = [[inf] * (n_cols + 1) for _ in range(n_rows + 1)]
    for j in range(n_cols + 1):
        best[0][j] = 0.0
    for i in range(1, n_rows + 1):
        for j in range(i, n_cols + 1):
            skip = best[i][j - 1]
            take = best[i - 1][j - 1] + float(cost[i - 1][j - 1])
            best[i][j] = skip if skip <= take else take
    idxs = [0] * n_rows
    i, j = n_rows, n_cols
    while i > 0:
        if best[i][j] == best[i][j - 1]:
            j -= 1  # column j-1 unused
        else:
            idxs[i - 1] = j - 1
            i -= 1
            j -= 1
    return idxs


def plan_out_scales(
    temporal_patch_size: int,
    patch_size: int,
    n_layers: int,
    n_channels: int = 3,
) -> List[Tuple[int, int, int, int]]:
    """Plan the ``(time, height, width, channels)`` scale at each hMLP layer.

    Builds the candidate scale progression (spatial folds first, then temporal,
    channels rounded up to a multiple of 64), then assigns ``n_layers + 1`` scales
    to the ideal log-spaced size reductions, pinning the first scale to the raw
    patch and the last to the full patch.
    """
    if patch_size <= 1:
        raise ValueError("patch_size must be greater than 1")

    def _round_up(x: int) -> int:
        return int(np.ceil(x / 64)) * 64

    last_h_scale = 1
    scales: List[Tuple[int, int, int, int]] = [(1, 1, 1, n_channels)]
    for pscale in _prime_factors(patch_size)[::-1]:
        last_h_scale *= pscale
        scales.append((1, last_h_scale, last_h_scale, _round_up((last_h_scale**2) * n_channels)))
    last_t_scale = 1
    for tscale in _prime_factors(temporal_patch_size)[::-1]:
        last_t_scale *= tscale
        scales.append(
            (
                last_t_scale,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels * last_t_scale),
            )
        )

    size_reduction = np.prod(np.array(scales)[:, :-1], 1)
    log_ideal_scales = np.linspace(
        0, np.log(patch_size * patch_size * temporal_patch_size * n_channels), n_layers + 1
    )
    cost_matrix = np.abs(log_ideal_scales[:, None] - np.log(size_reduction)[None])

    if n_layers >= len(scales):
        idxs = np.argmin(cost_matrix, axis=1)
    else:
        idxs = _assign_monotone(cost_matrix)

    assert len(idxs) >= 2
    idxs[0] = 0
    idxs[-1] = len(scales) - 1
    return [scales[i] for i in idxs]


def fold_timespace_to_depth(
    vision_patches_bthwc: torch.Tensor, t_fold: int, hw_fold: int
) -> torch.Tensor:
    """Fold a ``t_fold x hw_fold x hw_fold`` neighborhood into channel depth.

    ``(B, T, H, W, C) -> (B, T//t, H//hw, W//hw, C * t * hw**2)``. The
    reshape/permute order determines the exact channel interleave the projection
    weights expect, so it must not be reordered."""
    B, T, H, W, C = vision_patches_bthwc.shape
    assert T % t_fold == 0, f"T {T} not divisible by {t_fold}"
    assert H % hw_fold == 0, f"H {H} not divisible by {hw_fold}"
    assert W % hw_fold == 0, f"W {W} not divisible by {hw_fold}"
    t_new, h_new, w_new = T // t_fold, H // hw_fold, W // hw_fold
    x = vision_patches_bthwc.reshape(B, t_new, t_fold, h_new, hw_fold, w_new, hw_fold, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(B, t_new, h_new, w_new, t_fold * hw_fold * hw_fold * C)


def _require_vc(vision_config: Any, name: str) -> Any:
    val = getattr(vision_config, name, None) if vision_config is not None else None
    if val is None:
        raise ValueError(
            f"InklingVisionModel: required vision_config field {name!r} is "
            f"missing; it must be set so the hMLP geometry matches the model."
        )
    return val


class InklingVisionModel(nn.Module):
    """Inkling hMLP vision tower: ``vision_patches_bthwc`` -> one text-hidden row
    per patch.

    The encoder repeatedly folds a temporal/spatial neighborhood into the channel
    depth, projects with a bias-free Linear, and (for all but the last layer)
    applies RMSNorm + exact GELU. The module tree mirrors the checkpoint
    ``model.visual.*`` layout exactly.
    """

    def __init__(self, vision_config: Any) -> None:
        super().__init__()
        self.decoder_dmodel = int(_require_vc(vision_config, "decoder_dmodel"))
        self.patch_size = int(getattr(vision_config, "patch_size", DEFAULT_PATCH_SIZE))
        self.temporal_patch_size = int(
            getattr(vision_config, "temporal_patch_size", DEFAULT_TEMPORAL_PATCH_SIZE)
        )
        self.n_channels = int(getattr(vision_config, "n_channels", 3))
        self.n_layers = int(getattr(vision_config, "n_layers", 4))
        self.use_vision_norm = bool(getattr(vision_config, "use_vision_norm", True))

        self.scales = plan_out_scales(
            self.temporal_patch_size, self.patch_size, self.n_layers, self.n_channels
        )
        self.layers = nn.ModuleDict()
        for i, (start, end) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            shuffle_mult = (end[0] // start[0]) * (end[1] // start[1]) * (end[2] // start[2])
            in_dim = start[3] * shuffle_mult
            if i == self.n_layers - 1:
                self.layers[f"linear_{i}"] = nn.Linear(in_dim, self.decoder_dmodel, bias=False)
            else:
                self.layers[f"linear_{i}"] = nn.Linear(in_dim, end[3], bias=False)
                self.layers[f"norm_{i}"] = InklingRMSNorm(end[3])
        self.final_norm = InklingRMSNorm(self.decoder_dmodel) if self.use_vision_norm else None

    def forward(self, vision_patches_bthwc: torch.Tensor) -> torch.Tensor:
        """``(num_patches, T, H, W, C)`` -> ``(num_patches, decoder_dmodel)``."""
        num_patches = vision_patches_bthwc.shape[0]
        x = vision_patches_bthwc
        for i, (start, end) in enumerate(zip(self.scales[:-1], self.scales[1:])):
            t_fold = end[0] // start[0]
            hw_fold = end[1] // start[1]
            if hw_fold > 1 or t_fold > 1:
                x = fold_timespace_to_depth(x, t_fold, hw_fold)
            x = self.layers[f"linear_{i}"](x)
            if i < self.n_layers - 1:
                x = self.layers[f"norm_{i}"](x)
                x = F.gelu(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x.reshape(num_patches, -1)

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load the ``model.visual.*`` checkpoint tensors into this tower.

        Accepts either full ``model.visual.layers.linear_0.weight`` keys or
        already-stripped ``layers.linear_0.weight`` keys."""
        _load_tower_weights(self, weights, "model.visual.", self.layers["linear_0"].weight.dtype)


# ===========================================================================
# Audio tower: dMel preprocessing + codebook encoder
# ===========================================================================
# Default Inkling dMel geometry (checkpoint ``audio_config`` +
# ``processor_config.json`` ``feature_extractor``).
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_WINDOW_SIZE_MULTIPLIER = 2.0
DEFAULT_AUDIO_N_MELS = 80
DEFAULT_AUDIO_NUM_DMEL_BINS = 16  # == audio_config.mel_vocab_size
DEFAULT_AUDIO_DMEL_MIN_VALUE = -7.0
DEFAULT_AUDIO_DMEL_MAX_VALUE = 2.0
DEFAULT_AUDIO_TOKEN_DURATION_S = 0.05  # 1 dMel frame == 1 audio token (hop/sr)

# Inkling audio placeholder id (``<|unused_200053|>``), the direct analogue of
# the image sentinel: one per audio clip, expanded to one token per dMel frame.
# The top-level config may override via ``audio_token_id``.
DEFAULT_AUDIO_TOKEN_ID = 200053


def _audio_hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa/torchaudio convention)."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = frequencies / f_sp
    log = min_log_mel + np.log(np.maximum(frequencies, min_log_hz) / min_log_hz) / logstep
    return np.where(frequencies >= min_log_hz, log, linear)


def _audio_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Inverse Slaney mel scale."""
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = mels * f_sp
    log = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log, linear)


def audio_mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Slaney area-normalized mel filterbank ``(n_mels, n_fft//2+1)`` (float32)."""
    fft_bins = n_fft // 2 + 1
    fft_freqs = np.arange(fft_bins, dtype=np.float64) * sample_rate / n_fft
    mel_edges = _audio_mel_to_hz(
        np.linspace(
            _audio_hz_to_mel(np.array([0.0]))[0],
            _audio_hz_to_mel(np.array([sample_rate / 2.0]))[0],
            n_mels + 2,
            dtype=np.float64,
        )
    )
    mel_widths = np.diff(mel_edges)
    lower = (fft_freqs[None, :] - mel_edges[:-2, None]) / mel_widths[:-1, None]
    upper = (mel_edges[2:, None] - fft_freqs[None, :]) / mel_widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney area normalization.
    weights *= (2.0 / (mel_edges[2:] - mel_edges[:-2]))[:, None]
    return np.ascontiguousarray(weights.astype(np.float32, copy=False))


def _audio_to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


class InklingAudioPreprocessor:
    """Raw audio -> Inkling dMel bin tensor for the audio tower.

    Raw waveform -> log-mel spectrogram (Slaney mel scale, area-normalized) ->
    per-bin nearest-center quantization into ``num_dmel_bins`` discrete levels.
    One STFT frame is one audio token, matching the tower's one row per frame.

    ``encode_one`` accepts a decoded mono waveform or raw file bytes / a path;
    ``soundfile`` and ``torchaudio`` are imported lazily, only when decoding.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        window_size_multiplier: float = DEFAULT_AUDIO_WINDOW_SIZE_MULTIPLIER,
        n_fft: Optional[int] = None,
        n_mels: int = DEFAULT_AUDIO_N_MELS,
        num_dmel_bins: int = DEFAULT_AUDIO_NUM_DMEL_BINS,
        dmel_min_value: float = DEFAULT_AUDIO_DMEL_MIN_VALUE,
        dmel_max_value: float = DEFAULT_AUDIO_DMEL_MAX_VALUE,
        audio_token_duration_s: float = DEFAULT_AUDIO_TOKEN_DURATION_S,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window_size_multiplier = float(window_size_multiplier)
        self.n_fft = int(n_fft) if n_fft else None
        self.n_mels = int(n_mels)
        self.num_dmel_bins = int(num_dmel_bins)
        self.dmel_min_value = float(dmel_min_value)
        self.dmel_max_value = float(dmel_max_value)
        self.audio_token_duration_s = float(audio_token_duration_s)
        self._mel_basis_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _mel_basis(self, n_fft: int) -> torch.Tensor:
        key = (self.sample_rate, n_fft, self.n_mels)
        cached = self._mel_basis_cache.get(key)
        if cached is None:
            cached = torch.from_numpy(audio_mel_basis(self.sample_rate, n_fft, self.n_mels))
            self._mel_basis_cache[key] = cached
        return cached

    def _to_waveform(self, audio: Any) -> torch.Tensor:
        """Coerce one audio input into a 1-D mono float32 waveform at ``sample_rate``."""
        if isinstance(audio, torch.Tensor):
            wav = audio.detach().to(torch.float32)
            return wav.mean(dim=-1) if wav.ndim > 1 else wav.reshape(-1)
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            return wav.mean(dim=-1) if wav.ndim > 1 else wav.reshape(-1)
        # soundfile is an optional dependency: only a raw file/bytes input needs
        # it, so a caller passing a decoded waveform never imports it.
        import soundfile as sf

        if isinstance(audio, (bytes, bytearray, memoryview)):
            raw = bytes(audio)
        elif hasattr(audio, "read"):
            raw = audio.read()
        elif isinstance(audio, str):
            path = audio[len("file://") :] if audio.startswith("file://") else audio
            with open(path, "rb") as f:
                raw = f.read()
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio)!r}")
        samples, src_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))
        if int(src_sr) != self.sample_rate:
            import torchaudio.functional as AF  # lazy: only resampling needs it

            wav = AF.resample(wav, orig_freq=int(src_sr), new_freq=self.sample_rate)
        return wav

    def _dmel_bins(self, audio: torch.Tensor) -> torch.Tensor:
        """1-D float32 waveform -> ``(n_frames, n_mels)`` int32 dMel bins."""
        hop_length = _audio_to_exact_int(
            self.audio_token_duration_s * self.sample_rate,
            "audio_token_duration_s * sample_rate",
        )
        window_size = _audio_to_exact_int(
            self.audio_token_duration_s * self.window_size_multiplier * self.sample_rate,
            "audio_token_duration_s * window_size_multiplier * sample_rate",
        )
        n_fft = self.n_fft or window_size
        if hop_length <= 0 or window_size <= 0 or n_fft <= 0:
            raise ValueError("audio hop length, window size, and n_fft must be positive")
        if audio.numel() == 0:
            return torch.empty((0, self.n_mels), dtype=torch.int32)

        right_pad = math.ceil(audio.numel() / hop_length) * hop_length - audio.numel()
        left_pad = max(n_fft - hop_length, 0)
        audio = F.pad(audio, (left_pad, right_pad))

        window = torch.hann_window(window_size, periodic=True, dtype=torch.float32)
        spec = torch.stft(
            audio.unsqueeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_size,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec_ri = torch.view_as_real(spec)
        magnitude = (
            (spec_ri[..., 0].square() + spec_ri[..., 1].square()).clamp_min(1e-10).sqrt().squeeze(0)
        )
        mel = self._mel_basis(n_fft).matmul(magnitude).clamp_min(1e-10).log10()
        mel = mel.to(torch.float64).clamp(min=self.dmel_min_value, max=self.dmel_max_value)
        bin_centers = torch.linspace(
            self.dmel_min_value, self.dmel_max_value, self.num_dmel_bins, dtype=torch.float64
        )
        dmel_bins = (mel.unsqueeze(-1) - bin_centers).abs().argmin(dim=-1)
        return dmel_bins.to(torch.int32).T.contiguous()

    def encode_one(self, audio: Any) -> torch.Tensor:
        """One audio clip -> ``(n_frames, n_mels)`` int32 dMel bins."""
        return self._dmel_bins(self._to_waveform(audio))

    def preprocess(self, audios: Union[Any, Sequence[Any]]) -> Dict[str, Any]:
        """Raw audio -> ``{dmel_bins, num_frames, num_tokens}``.

        ``dmel_bins`` concatenates all clips' frames along dim 0 (one contiguous
        slice per clip, in encounter order); ``num_frames`` / ``num_tokens`` are
        per-clip and equal (one audio token per dMel frame)."""
        if not isinstance(audios, (list, tuple)):
            audios = [audios]
        per_clip: List[torch.Tensor] = []
        num_frames: List[int] = []
        for a in audios:
            bins = self.encode_one(a)
            per_clip.append(bins)
            num_frames.append(int(bins.shape[0]))
        if len(per_clip) == 1:
            dmel_bins = per_clip[0]
        elif per_clip:
            dmel_bins = torch.cat(per_clip, dim=0)
        else:
            dmel_bins = torch.empty((0, self.n_mels), dtype=torch.int32)
        return {
            "dmel_bins": dmel_bins,
            "num_frames": num_frames,
            "num_tokens": list(num_frames),  # one audio token per dMel frame
        }


def _resolve_audio_geometry(config: Any) -> Dict[str, Any]:
    """Return the dMel preprocessing kwargs from a top-level or audio config,
    defaulting to the checkpoint values."""
    ac = getattr(config, "audio_config", None)

    def _get(name, default):
        val = getattr(ac, name, None)
        return default if val is None else val

    return {
        "n_mels": int(_get("n_mel_bins", DEFAULT_AUDIO_N_MELS)),
        "num_dmel_bins": int(_get("mel_vocab_size", DEFAULT_AUDIO_NUM_DMEL_BINS)),
        "dmel_min_value": float(_get("dmel_min_value", DEFAULT_AUDIO_DMEL_MIN_VALUE)),
        "dmel_max_value": float(_get("dmel_max_value", DEFAULT_AUDIO_DMEL_MAX_VALUE)),
    }


def _resolve_audio_token_id(config: Any) -> int:
    """The single ``<audio>`` placeholder id the token stream carries per clip.

    Prefers an explicit ``audio_token_id`` on the (top-level or audio) config."""
    for obj in (config, getattr(config, "audio_config", None)):
        tok = getattr(obj, "audio_token_id", None)
        if tok is not None:
            return int(tok)
    return DEFAULT_AUDIO_TOKEN_ID


def _require_ac(audio_config: Any, name: str) -> Any:
    val = getattr(audio_config, name, None) if audio_config is not None else None
    if val is None:
        raise ValueError(
            f"InklingAudioModel: required audio_config field {name!r} is missing; "
            f"it must be set so the dMel tower geometry matches the model."
        )
    return val


class InklingAudioModel(nn.Module):
    """Inkling dMel audio tower: dMel bins -> one text-hidden row per frame.

    Each of the ``n_mel_bins`` per-frame bin indices selects a row from a shared
    codebook (bin ``m`` occupies rows ``[m*mel_vocab_size, (m+1)*mel_vocab_size)``)
    and the per-bin embeddings are summed per frame. The module tree mirrors the
    checkpoint ``model.audio.*`` layout exactly.
    """

    def __init__(self, audio_config: Any) -> None:
        super().__init__()
        audio_mode = getattr(audio_config, "audio_mode", "dmel")
        if audio_mode != "dmel":
            raise ValueError(
                f"InklingAudioModel supports audio_mode='dmel' only, got {audio_mode!r}."
            )
        self.decoder_dmodel = int(_require_ac(audio_config, "decoder_dmodel"))
        self.n_mel_bins = int(getattr(audio_config, "n_mel_bins", DEFAULT_AUDIO_N_MELS))
        self.mel_vocab_size = int(
            getattr(audio_config, "mel_vocab_size", DEFAULT_AUDIO_NUM_DMEL_BINS)
        )
        self.use_audio_norm = bool(getattr(audio_config, "use_audio_norm", True))
        self.encoder = nn.Embedding(self.n_mel_bins * self.mel_vocab_size, self.decoder_dmodel)
        self.final_norm = InklingRMSNorm(self.decoder_dmodel) if self.use_audio_norm else None

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """``(n_frames, n_mel_bins)`` dMel bin indices -> ``(n_frames, decoder_dmodel)``."""
        dev = self.encoder.weight.device
        if audio_features.numel() == 0:
            return torch.zeros(
                (0, self.decoder_dmodel), dtype=self.encoder.weight.dtype, device=dev
            )
        if audio_features.shape[-1] != self.n_mel_bins:
            raise ValueError(
                f"InklingAudioModel: audio_features last dim {audio_features.shape[-1]} "
                f"!= n_mel_bins {self.n_mel_bins}."
            )
        af = audio_features.to(device=dev)
        n_frames = int(af.shape[0])
        # bin ``m`` -> codebook rows [m*mel_vocab_size, (m+1)*mel_vocab_size)
        bin_offsets = torch.arange(self.n_mel_bins, device=dev) * self.mel_vocab_size
        idx = bin_offsets.unsqueeze(0) + af.to(torch.long)  # (n_frames, n_mel_bins)
        hidden = (
            self.encoder(idx.reshape(-1)).reshape(n_frames, self.n_mel_bins, -1).sum(dim=1)
        )  # (n_frames, decoder_dmodel)
        if self.final_norm is not None:
            hidden = self.final_norm(hidden)
        return hidden

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load the ``model.audio.*`` checkpoint tensors into this tower.

        Accepts either full ``model.audio.encoder.weight`` keys or
        already-stripped ``encoder.weight`` keys."""
        _load_tower_weights(self, weights, "model.audio.", self.encoder.weight.dtype)


# ===========================================================================
# Video tower: frame sampling onto the image path
# ===========================================================================
# Inkling has no separate video tower: a video is decoded to frames and each
# sampled frame is fed as an ordinary image through the same hMLP vision tower,
# so the only video-specific logic is choosing which frames to keep.


class DecodedVideo:
    """A minimal decoded-video view over an ordered list of frame images.

    Wraps already-decoded frames plus the clip's average FPS. Decoding itself is
    deliberately upstream, which keeps this free of a video-decoder dependency.
    """

    def __init__(self, frames: Sequence[Any], avg_fps: float) -> None:
        self.frames = list(frames)
        if not self.frames:
            raise ValueError("DecodedVideo requires at least one frame")
        self.avg_fps = float(avg_fps)

    def __len__(self) -> int:
        return len(self.frames)


def sample_video_frames(video: Any, *, desired_fps: int, max_frames: int) -> List[int]:
    """Frame indices to keep from ``video``.

    ``video`` is any object exposing ``__len__`` (total decoded frames) and
    ``avg_fps`` (the clip's average FPS). The sampled count is bounded by the
    desired FPS, ``max_frames``, and the total frame count, with at least one
    frame always kept; the returned indices are strictly increasing (temporal
    order preserved).
    """
    total_frames = len(video)
    assert total_frames > 0, "Video must have at least one frame"

    avg_fps = video.avg_fps
    duration = total_frames / avg_fps if avg_fps > 0 else 0
    fps = min(desired_fps, avg_fps)

    num_frames = math.floor(duration * fps)
    num_frames = min(max_frames, num_frames, total_frames)
    num_frames = max(1, num_frames)  # At least one frame
    if num_frames == total_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def sample_video_as_images(video: DecodedVideo, *, desired_fps: int, max_frames: int) -> List[Any]:
    """Video -> the sampled frames as an ordered list of images, in temporal
    order. The caller renders one ``<image>`` content part per returned frame."""
    idxs = sample_video_frames(video, desired_fps=desired_fps, max_frames=max_frames)
    return [video.frames[i] for i in idxs]


# ===========================================================================
# Multimodal input processor (image + audio + video-as-images)
# ===========================================================================
class InklingInputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Inkling multimodal input processor.

    Registered on ``InklingForConditionalGeneration`` with placeholders
    ``{"image": "<image>", "audio": "<audio>"}``. Inherits
    :class:`BaseMultimodalDummyInputsBuilder` as well, the two-base pattern the
    in-tree VLM processors use, because ``trtllm-serve`` and the KV-cache encoder
    profiler call hooks that live on the builder.

    Text-only requests pass straight through; each ``<image>`` placeholder is
    expanded into one token per vision patch and each ``<audio>`` placeholder into
    one token per dMel frame. Any count mismatch fails loudly rather than dropping
    or padding media.
    """

    # Accept a pre-tokenized ``prompt_token_ids`` stream carrying one placeholder
    # per media item without detokenizing: the checkpoint tokenizer has no
    # ``<image>`` -> placeholder mapping, so a round trip would drop it.
    supports_token_id_mm_expansion = True

    def __init__(self, model_path, config, tokenizer, trust_remote_code: bool = True, **kwargs):
        super().__init__(model_path, config, tokenizer, trust_remote_code, **kwargs)
        patch_size, temporal = _resolve_vision_geometry(config)
        self.image_token_id = _resolve_image_token_id(config)
        self.audio_token_id = _resolve_audio_token_id(config)
        if self.image_token_id == self.audio_token_id:
            # assemble() dispatches on the token value, so a shared id would send
            # every audio placeholder down the image branch and report the
            # mismatch against the wrong modality.
            raise ValueError(
                f"InklingInputProcessor: image_token_id and audio_token_id must "
                f"differ; both resolved to {self.image_token_id}."
            )
        self._dtype = torch.bfloat16
        self._preprocessor = InklingImagePreprocessor(
            patch_size=patch_size,
            temporal_patch_size=temporal,
            dtype=self._dtype,
        )
        # Built unconditionally (geometry falls back to the checkpoint defaults)
        # so a checkpoint without an ``audio_config`` still serves audio; inert
        # unless the token stream carries an audio placeholder.
        self._audio_preprocessor = InklingAudioPreprocessor(**_resolve_audio_geometry(config))
        # Delegate text-only tokenization to the stock DefaultInputProcessor so
        # it cannot drift from the text tower; the media paths reuse it too.
        self._text_processor = DefaultInputProcessor(
            model_path, config, tokenizer, trust_remote_code
        )

    # ---- required BaseMultimodalInputProcessor properties -----------------
    @property
    def processor(self):
        # Inkling ships no HF AutoProcessor; expose the local image preprocessor
        # so base helpers that read ``self.processor`` work.
        return self._preprocessor

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def model_path(self):
        # Concrete impl of the ``BaseMultimodalDummyInputsBuilder`` abstract
        # property; the base processor only stores ``self._model_path``.
        return self._model_path

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        """The placeholder id(s) that mark media rows in the token stream.

        Returned as int32 so the engine's ``torch.isin(input_ids, mm_token_ids)``
        lookup matches both."""
        return torch.tensor([self.image_token_id, self.audio_token_id], dtype=torch.int32)

    # ---- multimodal-hash prefix caching: explicitly unsupported -----------
    # The per-item token count comes from the preprocessor geometry, not a static
    # HF formula, and the multimodal-hash prefix cache needs it up front. Refuse
    # it the same way for every modality rather than inheriting the base's opaque
    # errors; the caller falls back to the uncached path, so only cross-request
    # MM prefix reuse is lost, never a request itself.
    def _reject_mm_hash_cache(self, modality: str) -> None:
        raise NotImplementedError(
            f"InklingInputProcessor does not support multimodal-hash prefix "
            f"caching ({modality}): the per-item token count is derived from the "
            f"vision/audio preprocessor geometry and is not wired to "
            f"get_num_multimodal_tokens. Multimodal requests still run, uncached."
        )

    def get_num_tokens_per_image(self, *, image=None, **kwargs) -> int:
        self._reject_mm_hash_cache("image")

    def get_num_tokens_per_audio(self, *, audio=None, **kwargs) -> int:
        self._reject_mm_hash_cache("audio")

    def get_num_tokens_per_video(self, *, video=None, **kwargs) -> int:
        self._reject_mm_hash_cache("video")

    # ---- core (pure, testable) expansion + validation ---------------------
    def assemble(
        self,
        input_ids: Sequence[int],
        image_data: Optional[Sequence[Any]] = None,
        audio_data: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Expand ``<image>`` / ``<audio>`` placeholders and attach media features.

        Returns ``(expanded_ids, multimodal_data)``, the latter ``{}`` for a
        text-only request. Raises ``ValueError`` on any placeholder / media /
        feature-row count mismatch.
        """
        input_ids = list(input_ids)
        image_data = list(image_data) if image_data else []
        audio_data = list(audio_data) if audio_data else []

        self._check_placeholder_count(input_ids, self.image_token_id, len(image_data), "image")
        self._check_placeholder_count(input_ids, self.audio_token_id, len(audio_data), "audio")
        if not image_data and not audio_data:
            return input_ids, {}  # text-only passthrough

        img_feat = self._preprocessor.preprocess(image_data) if image_data else None
        aud_feat = self._audio_preprocessor.preprocess(audio_data) if audio_data else None

        out_ids: List[int] = []
        img_offsets: List[Tuple[int, int]] = []  # (start, end) inclusive per image
        aud_offsets: List[Tuple[int, int]] = []  # (start, end) inclusive per clip
        i_img = 0
        i_aud = 0
        for tok in input_ids:
            if image_data and tok == self.image_token_id:
                n_tok = int(img_feat["num_tokens"][i_img])
                img_offsets.append((len(out_ids), len(out_ids) + n_tok - 1))
                out_ids.extend([self.image_token_id] * n_tok)
                i_img += 1
            elif audio_data and tok == self.audio_token_id:
                n_tok = int(aud_feat["num_tokens"][i_aud])
                aud_offsets.append((len(out_ids), len(out_ids) + n_tok - 1))
                out_ids.extend([self.audio_token_id] * n_tok)
                i_aud += 1
            else:
                out_ids.append(int(tok))

        # Central fail-loud invariant, per modality: the number of expanded
        # placeholder tokens must equal the number of feature rows the tower emits.
        multimodal_data: Dict[str, Any] = {}
        if image_data:
            vision_patches = img_feat["vision_patches_bthwc"]
            self._check_feature_rows(
                out_ids, self.image_token_id, img_feat["num_patches"], vision_patches, "image"
            )
            multimodal_data["image"] = {
                "vision_patches_bthwc": vision_patches,
                "num_patches": img_feat["num_patches"],
                "offsets": img_offsets,
            }
        if audio_data:
            dmel_bins = aud_feat["dmel_bins"]
            self._check_feature_rows(
                out_ids, self.audio_token_id, aud_feat["num_frames"], dmel_bins, "audio"
            )
            multimodal_data["audio"] = {
                "dmel_bins": dmel_bins,
                "num_frames": aud_feat["num_frames"],
                "offsets": aud_offsets,
            }
        return out_ids, multimodal_data

    @staticmethod
    def _check_placeholder_count(
        input_ids: List[int], token_id: int, num_items: int, modality: str
    ) -> None:
        n_placeholders = sum(1 for t in input_ids if t == token_id)
        if n_placeholders != num_items:
            raise ValueError(
                f"InklingInputProcessor: {n_placeholders} {modality} placeholder "
                f"token(s) (id={token_id}) in input_ids but {num_items} "
                f"{modality} item(s) provided; counts must match (media is never "
                f"dropped or padded silently)."
            )

    @staticmethod
    def _check_feature_rows(
        out_ids: List[int],
        token_id: int,
        per_item_rows: List[int],
        features: torch.Tensor,
        modality: str,
    ) -> None:
        expanded = sum(1 for t in out_ids if t == token_id)
        expected = int(sum(per_item_rows))
        feat_rows = int(features.shape[0]) if features.ndim else 0
        if not (expanded == expected == feat_rows):
            raise ValueError(
                f"InklingInputProcessor: {modality} placeholder-token count "
                f"({expanded}) must equal feature-row count (expected {expected}, "
                f"features have {feat_rows} row(s))."
            )

    @staticmethod
    def _as_list(mm_data: Dict[str, Any], key: str) -> List[Any]:
        items = mm_data.get(key) or []
        return items if isinstance(items, list) else [items]

    # ---- SamplingParams/TextPrompt entrypoint -----------------------------
    def call_with_text_prompt(self, inputs, sampling_params):
        """Turn a text prompt (optionally with media) into ``(ids, extra)``.

        Text-only requests are delegated verbatim to ``DefaultInputProcessor``, so
        registering this processor does not change the text path.
        """
        mm_data = inputs.get("multi_modal_data") or {}
        images = self._as_list(mm_data, "image")
        audios = self._as_list(mm_data, "audio")
        if not images and not audios:
            # Exactly the pre-registration text path (tiktoken-safe, honors
            # add_special_tokens / truncation / query). Returns (ids, None).
            return self._text_processor(inputs, sampling_params)

        token_ids, _extra = self._text_processor(inputs, sampling_params)
        expanded_ids, multimodal_data = self.assemble(token_ids, images, audios)
        return expanded_ids, {"multimodal_data": multimodal_data}

    # ---- pre-tokenized + media fast path ----------------------------------
    def call_with_token_ids(self, inputs, sampling_params):
        """Pre-tokenized multimodal entry (``supports_token_id_mm_expansion``).

        ``prompt_token_ids`` must already contain exactly one placeholder per
        media item. The base implementation drives an HF processor over a
        synthetic dummy prompt, which does not apply: Inkling ships no HF
        AutoProcessor and the placeholder is already in the stream.
        """
        ids = list(inputs.get("prompt_token_ids") or [])
        mm_data = inputs.get("multi_modal_data") or {}
        images = self._as_list(mm_data, "image")
        audios = self._as_list(mm_data, "audio")
        if not images and not audios:
            # No media -> plain token passthrough (no multimodal payload).
            return ids, None
        expanded_ids, multimodal_data = self.assemble(ids, images, audios)
        return expanded_ids, {"multimodal_data": multimodal_data}
