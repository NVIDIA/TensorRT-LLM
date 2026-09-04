# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

"""Native utility classes for MiniMax-H3 VisualGen components."""

from __future__ import annotations

import functools
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as _load_safetensors_file
except ImportError:  # pragma: no cover - safetensors is a runtime dependency in containers.
    _load_safetensors_file = None


@dataclass
class MiniMaxH3DecoderOutput:
    """Decoder output compatible with the attributes used by the H3 pipeline."""

    sample: torch.Tensor

    def __iter__(self):
        yield self.sample


@dataclass
class MiniMaxH3AutoencoderKLOutput:
    """Autoencoder encode output containing a latent distribution."""

    latent_dist: Any

    def __iter__(self):
        yield self.latent_dist


def minimax_h3_randn_tensor(
    shape: tuple[int, ...] | torch.Size,
    *,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate random tensors with Diffusers-compatible CPU-generator behavior."""

    shape = tuple(shape)
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    generator_device = None
    if generator is not None:
        first_generator = generator[0] if isinstance(generator, list) else generator
        generator_device = first_generator.device
    random_device = target_device
    if generator_device is not None and generator_device.type != target_device.type:
        if generator_device.type != "cpu":
            raise ValueError(
                f"Cannot generate a {target_device} tensor from a {generator_device.type} generator."
            )
        random_device = torch.device("cpu")

    if isinstance(generator, list):
        if len(generator) != shape[0]:
            raise ValueError(
                "A generator list must have one entry per batch element; "
                f"got {len(generator)} for batch size {shape[0]}."
            )
        samples = [
            torch.randn(
                (1, *shape[1:]),
                generator=batch_generator,
                device=random_device,
                dtype=dtype,
            )
            for batch_generator in generator
        ]
        return torch.cat(samples, dim=0).to(target_device)
    return torch.randn(
        shape,
        generator=generator,
        device=random_device,
        dtype=dtype,
    ).to(target_device)


class MiniMaxH3DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior for video VAE moments.

    The released video VAE writes concatenated mean/log-variance tensors along
    the channel dimension.  This is the subset of Diffusers' distribution API
    used by MiniMax-H3: ``sample`` and ``mode``.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = torch.zeros_like(self.mean)
            self.std = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        noise = minimax_h3_randn_tensor(
            tuple(self.mean.shape),
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


def identity_forward_hook(function: Callable[..., Any]) -> Callable[..., Any]:
    """Native no-op replacement for Diffusers' offload hook decorator."""

    return function


def get_module_parameter_dtype(module: nn.Module) -> torch.dtype:
    """Return the first parameter/buffer dtype, or float32 for empty modules."""

    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.dtype
    return torch.float32


def build_config_namespace(config: dict[str, Any]) -> SimpleNamespace:
    """Create an attribute namespace from JSON config values."""

    return SimpleNamespace(**config)


def minimax_h3_register_to_config(init: Callable[..., None]) -> Callable[..., None]:
    """Decorator that records ``__init__`` arguments on ``self.config``.

    Diffusers' ``register_to_config`` is convenient but not acceptable in
    candidate runtime files.  MiniMax-H3 only needs attribute-style access to
    constructor parameters after local checkpoint loading, so a small native
    equivalent is sufficient.
    """

    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = {key: value for key, value in bound.arguments.items() if key != "self"}
        init(self, *args, **kwargs)
        self.config = build_config_namespace(config)

    return wrapper


class MiniMaxH3ModelMixin(nn.Module):
    """Minimal local pretrained-loader mixin for MiniMax-H3 VAE modules."""

    config_name = "config.json"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "MiniMaxH3ModelMixin":
        return load_native_pretrained_model(
            cls,
            pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            **kwargs,
        )


def _load_json_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {config_path}.")
    for metadata_key in ("_class_name", "_diffusers_version", "_name_or_path"):
        config.pop(metadata_key, None)
    return config


def _state_dict_files(model_dir: Path) -> list[Path]:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        with index_files[0].open(encoding="utf-8") as index_file:
            index = json.load(index_file)
        if not isinstance(index, dict):
            raise ValueError(f"Invalid safetensors index in {index_files[0]}.")
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index in {index_files[0]}.")
        model_root = model_dir.resolve()
        weight_files = set()
        for filename in weight_map.values():
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Invalid shard filename in {index_files[0]}: {filename!r}.")
            relative_path = Path(filename)
            if relative_path.is_absolute():
                raise ValueError(f"Shard filename must be relative to {model_root}: {filename!r}.")
            weight_path = (model_root / relative_path).resolve()
            try:
                weight_path.relative_to(model_root)
            except ValueError as error:
                raise ValueError(
                    f"Shard filename escapes component directory {model_root}: {filename!r}."
                ) from error
            if weight_path.suffix != ".safetensors":
                raise ValueError(f"Safetensors index references unsupported shard {filename!r}.")
            if not weight_path.is_file():
                raise FileNotFoundError(f"Checkpoint shard not found: {weight_path}")
            weight_files.add(weight_path)
        return sorted(weight_files)

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if safetensors_files:
        return safetensors_files

    bin_files = sorted(model_dir.glob("*.bin"))
    if bin_files:
        return bin_files

    raise FileNotFoundError(f"No model weight files found in {model_dir}.")


def load_native_pretrained_model(
    cls: type[nn.Module],
    pretrained_model_name_or_path: str | Path,
    *,
    subfolder: str | None = None,
    torch_dtype: torch.dtype | None = None,
    **override_config: Any,
) -> nn.Module:
    """Instantiate ``cls`` from a local HF-style config and state dict."""

    model_dir = Path(pretrained_model_name_or_path)
    if subfolder is not None:
        model_dir /= str(getattr(subfolder, "value", subfolder))
    if not model_dir.is_dir():
        raise FileNotFoundError(f"MiniMax-H3 component directory not found: {model_dir}")

    config = _load_json_config(model_dir)
    config.update({key: value for key, value in override_config.items() if value is not None})
    model = cls(**config)

    state_dict: dict[str, torch.Tensor] = {}
    for weight_path in _state_dict_files(model_dir):
        if weight_path.suffix == ".safetensors":
            if _load_safetensors_file is None:
                raise ImportError("safetensors is required to load MiniMax-H3 checkpoints.")
            state_dict.update(_load_safetensors_file(str(weight_path), device="cpu"))
        else:
            state_dict.update(torch.load(weight_path, map_location="cpu", weights_only=True))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        missing_preview = ", ".join(missing[:8])
        unexpected_preview = ", ".join(unexpected[:8])
        raise ValueError(
            "MiniMax-H3 checkpoint state dict mismatch for "
            f"{model_dir}: missing={len(missing)} [{missing_preview}], "
            f"unexpected={len(unexpected)} [{unexpected_preview}]"
        )
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    return model


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings matching Diffusers' tensor recipe."""

    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.flatten().float()
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class MiniMaxH3Timesteps(nn.Module):
    """Sinusoidal timestep projection used by the MiniMax-H3 transformer."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class MiniMaxH3TimestepEmbedding(nn.Module):
    """Two-layer SiLU timestep MLP with Diffusers-compatible parameter names."""

    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim or time_embed_dim
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class MiniMaxH3SwiGLU(nn.Module):
    """Diffusers-compatible SwiGLU projection with a ``proj`` child module."""

    def __init__(self, dim: int, inner_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return F.silu(gate) * hidden_states


class MiniMaxH3FeedForward(nn.Module):
    """Small native FFN matching Diffusers' ``FeedForward(..., swiglu)`` layout."""

    def __init__(
        self,
        dim: int,
        *,
        inner_dim: int | None = None,
        mult: int = 4,
        activation_fn: str = "swiglu",
        bias: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if activation_fn != "swiglu":
            raise ValueError(
                "MiniMax-H3 native FeedForward only implements activation_fn='swiglu', "
                f"got {activation_fn!r}."
            )
        inner_dim = inner_dim or dim * mult
        self.net = nn.ModuleList(
            [
                MiniMaxH3SwiGLU(dim, inner_dim, bias=bias),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


__all__ = [
    "MiniMaxH3AutoencoderKLOutput",
    "MiniMaxH3DecoderOutput",
    "MiniMaxH3DiagonalGaussianDistribution",
    "MiniMaxH3FeedForward",
    "MiniMaxH3ModelMixin",
    "MiniMaxH3SwiGLU",
    "MiniMaxH3Timesteps",
    "MiniMaxH3TimestepEmbedding",
    "build_config_namespace",
    "get_module_parameter_dtype",
    "get_timestep_embedding",
    "identity_forward_hook",
    "load_native_pretrained_model",
    "minimax_h3_randn_tensor",
    "minimax_h3_register_to_config",
]
