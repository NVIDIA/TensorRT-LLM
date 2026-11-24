# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 Video VAE and chunk CUDA-graph configuration for MiniMax-H3 FP8 U4 (SM103 & SM107)."""

from __future__ import annotations

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "AudioDecodeGraph",
    "configure_u4_video_vae_pre_load",
    "configure_u4_video_vae_post_load",
    "install_audio_vae_cuda_graph",
]

_SUPPORTED_VAE_CAPABILITIES = frozenset(((10, 3), (10, 7)))
_DEFAULT_VAE_GRAPH_MAX_SHAPES = 2
_VAE_FP8_TILE = (16, 1024, 8)
_VAE_FP8_QTILE_U1 = (32, 8, 8)
_VAE_FP8_QTILE_U4 = (64, 4, 8)


def _is_supported_fp8_vae_cuda(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) in _SUPPORTED_VAE_CAPABILITIES
    )


def configure_u4_video_vae_pre_load(vae, device: torch.device, *, max_shapes: int | None = None) -> bool:
    """Enable VAE decoder chunk CUDA graphs on SM103/SM107 before ``super().post_load_weights()``."""
    if vae is None or not _is_supported_fp8_vae_cuda(device):
        return False
    shapes = int(_DEFAULT_VAE_GRAPH_MAX_SHAPES if max_shapes is None else max_shapes)
    vae.decoder_graph_max_shapes = shapes
    logger.info(f"MiniMax-H3 FP8 U4 VAE decoder_graph_max_shapes={shapes}")
    return True


def configure_u4_video_vae_post_load(
    vae,
    device: torch.device,
    *,
    ulysses_size: int = 4,
    enable_fp8: bool = True,
) -> bool:
    """Convert Video VAE decoder transformer Linears to tiled FP8 on SM103/SM107 after weights load."""
    if not enable_fp8 or vae is None or not _is_supported_fp8_vae_cuda(device):
        return False
    from ..minimax_h3_nvfp4__sm_107__u1_u4.kernels.vae_decoder_fp8 import (
        convert_decoder_linears_fp8,
        set_tiles,
    )

    qtile = _VAE_FP8_QTILE_U4 if int(ulysses_size) >= 4 else _VAE_FP8_QTILE_U1
    set_tiles(mlp=_VAE_FP8_TILE, q=qtile)
    convert_decoder_linears_fp8(vae.decoder)
    logger.info(f"MiniMax-H3 FP8 U4 Video VAE decoder converted to FP8 (ulysses={ulysses_size})")
    return True


class AudioDecodeGraph:
    """Bounded per-shape CUDA graph replay wrapper for Audio VAE decode (bitwise exact)."""

    def __init__(self, original, *, max_shapes: int = 2):
        self._original = original
        self._max_shapes = int(max_shapes)
        self._cache: dict[tuple, tuple[torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def __call__(self, latent: torch.Tensor, *, return_dict: bool = True):
        if (
            return_dict is not False
            or not latent.is_cuda
            or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
        ):
            return self._original(latent, return_dict=return_dict)
        stream = torch.cuda.current_stream(latent.device)
        if stream.cuda_stream == 0:
            return self._original(latent, return_dict=return_dict)
        key = (tuple(latent.shape), latent.dtype, latent.device, stream.cuda_stream)
        entry = self._cache.get(key)
        if entry is None:
            if len(self._cache) >= self._max_shapes:
                return self._original(latent, return_dict=return_dict)
            buf = latent.clone()
            self._original(buf, return_dict=False)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                (out,) = self._original(buf, return_dict=False)
            entry = (buf, graph, out)
            self._cache[key] = entry
        buf, graph, out = entry
        buf.copy_(latent)
        graph.replay()
        return (out.clone(),)


def install_audio_vae_cuda_graph(audio_vae, device: torch.device) -> bool:
    """Wrap ``audio_vae.decode`` with per-shape CUDA graph replay on SM103/SM107."""
    if audio_vae is None or not _is_supported_fp8_vae_cuda(device):
        return False
    if isinstance(getattr(audio_vae, "decode", None), AudioDecodeGraph):
        return True
    audio_vae.decode = AudioDecodeGraph(audio_vae.decode)
    logger.info("MiniMax-H3 FP8 Audio VAE CUDA graph replay enabled")
    return True
