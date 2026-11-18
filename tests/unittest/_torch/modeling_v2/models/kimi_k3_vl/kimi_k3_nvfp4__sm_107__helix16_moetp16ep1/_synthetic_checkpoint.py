# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic Kimi K3 NVFP4 checkpoint with lazily computed slices (torch only).

Every tensor is a :class:`SynthTensor`: a safetensors-slice look-alike (``get_shape``,
``__getitem__`` with ``()``, a row slice or a ``(rows, cols)`` pair) whose elements are a hash of
their flat index and the key, so a reader that cuts the rank's rows / columns materializes only
those. The key names, shapes and dtypes are the checkpoint's (``_chain_weights`` module docstring).
"""

from __future__ import annotations

import zlib
from typing import Dict, Iterable

import torch

HIDDEN, HEADS, HEAD_DIM, EXPERTS, LATENT, SHARED, INTER = 7168, 96, 128, 896, 3584, 6144, 3072
_M = 0xFFFFFFFF


def _hash(flat: torch.Tensor, seed: int) -> torch.Tensor:
    x = (flat * 0x9E3779B1 + seed * 0x85EBCA77 + 0x165667B1) & _M
    x = x ^ (x >> 15)
    x = (x * 0x2C1B3C6D) & _M
    x = x ^ (x >> 12)
    x = (x * 0x297A2D39) & _M
    return x ^ (x >> 15)


class SynthTensor:
    """``kind``: ``u8`` (bytes), ``e4m3`` (finite FP8 codes, both signs), ``ue4m3`` (positive
    FP8 block scales), ``bf16`` / ``f32`` (values in [-scale, scale]), ``pos`` (FP32 in
    [scale, 2 scale])."""

    def __init__(self, key: str, shape, kind: str, scale: float = 1.0):
        self.key, self.shape, self.kind, self.scale = key, tuple(shape), kind, scale
        self.seed = zlib.crc32(key.encode())

    def get_shape(self):
        return list(self.shape)

    def _ranges(self, index):
        if index == ():
            index = ()
        elif not isinstance(index, tuple):
            index = (index,)
        out = []
        for d, n in enumerate(self.shape):
            s = index[d] if d < len(index) else slice(None)
            start, stop, step = s.indices(n)
            if step != 1:
                raise ValueError("strided synthetic slice")
            out.append(torch.arange(start, stop, dtype=torch.int64))
        return out

    def __getitem__(self, index) -> torch.Tensor:
        ranges = self._ranges(index)
        flat = torch.zeros((), dtype=torch.int64)
        stride = 1
        for d in reversed(range(len(self.shape))):
            shape = [1] * len(self.shape)
            shape[d] = -1
            flat = flat + ranges[d].view(shape) * stride
            stride *= self.shape[d]
        x = _hash(flat, self.seed)
        return self._values(x)

    def full(self) -> torch.Tensor:
        return self[()] if not self.shape else self[tuple(slice(None) for _ in self.shape)]

    def _values(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "u8":
            return (x & 0xFF).to(torch.uint8)
        if self.kind == "e4m3":
            byte = (x & 0x7F) % 0x7E | ((x >> 8) & 1) << 7
            return byte.to(torch.uint8).view(torch.float8_e4m3fn)
        if self.kind == "ue4m3":
            return (0x28 + (x % 0x20)).to(torch.uint8).view(torch.float8_e4m3fn)
        unit = ((x % 2001) - 1000).double() / 1000.0
        if self.kind == "bf16":
            return (unit * self.scale).to(torch.bfloat16)
        if self.kind == "f32":
            return (unit * self.scale).float()
        if self.kind == "pos":
            return ((unit.abs() + 1.0) * self.scale).float()
        raise ValueError(self.kind)


def _fp8(out: Dict[str, SynthTensor], key: str, shape, scale: float = 1e-3) -> None:
    out[key] = SynthTensor(key, shape, "e4m3")
    grid = (-(-shape[0] // 128), 1, -(-shape[1] // 128), 1)
    out[key + "_scale_inv"] = SynthTensor(key + "_scale_inv", grid, "pos", scale)


def _dense(out, key, shape, fp8: bool, scale: float = 0.02) -> None:
    if fp8:
        _fp8(out, key, shape)
    else:
        out[key] = SynthTensor(key, shape, "bf16", scale)


def synthetic_checkpoint(
    layers: Iterable[int],
    *,
    prefix: str = "language_model.",
    experts: int = EXPERTS,
    fp8_moe_dense: bool = False,
    mismatched_scale2: bool = True,
) -> Dict[str, SynthTensor]:
    """The keys of ``layers`` (KDA, MoE dense, NVFP4 routed experts of ``experts`` experts,
    AttnRes / norm vectors) and the output vectors. ``fp8_moe_dense``: the MoE dense weights
    FP8 + block scales (the reader's offset path); ``mismatched_scale2``: w1 / w3
    ``weight_scale_2`` differ on every other expert (the packer's requantization)."""
    out: Dict[str, SynthTensor] = {}
    d = HEADS * HEAD_DIM
    base = f"{prefix}model."
    for name in ("norm", "output_attn_res_norm"):
        out[f"{base}{name}.weight"] = SynthTensor(f"{base}{name}.weight", (HIDDEN,), "pos", 0.5)
    key = f"{base}output_attn_res_proj.weight"
    out[key] = SynthTensor(key, (1, HIDDEN), "bf16", 0.02)
    for i in layers:
        layer = f"{base}layers.{i}."
        for name in ("self_attention_res_norm", "input_layernorm", "post_attention_layernorm",
                     "mlp_res_norm"):  # fmt: skip
            out[layer + name + ".weight"] = SynthTensor(
                layer + name + ".weight", (HIDDEN,), "pos", 0.5
            )
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            key = layer + name + ".weight"
            out[key] = SynthTensor(key, (1, HIDDEN), "bf16", 0.02)
        if (i + 1) % 4 != 0 and i <= 90:  # KDA
            a = layer + "self_attn."
            for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                _fp8(out, a + name + ".weight", (d, HIDDEN))
            _fp8(out, a + "f_a_proj.weight", (HEAD_DIM, HIDDEN))
            _fp8(out, a + "b_proj.weight", (HEADS, HIDDEN))
            _fp8(out, a + "f_b_proj.weight", (d, HEAD_DIM))
            _fp8(out, a + "o_proj.weight", (HIDDEN, d))
            for x in ("q", "k", "v"):
                key = a + f"{x}_conv1d.weight"
                out[key] = SynthTensor(key, (d, 1, 4), "f32", 0.5)
            out[a + "A_log"] = SynthTensor(a + "A_log", (1, 1, 128, 1), "f32", 1.0)
            out[a + "dt_bias"] = SynthTensor(a + "dt_bias", (d,), "f32", 0.5)
            out[a + "o_norm.weight"] = SynthTensor(a + "o_norm.weight", (HEAD_DIM,), "pos", 0.5)
        if i == 0:
            continue
        m = layer + "block_sparse_moe."
        _dense(out, m + "gate.weight", (EXPERTS, HIDDEN), fp8_moe_dense)
        key = m + "gate.e_score_correction_bias"
        out[key] = SynthTensor(key, (EXPERTS,), "f32", 0.1)
        _dense(out, m + "routed_expert_down_proj.weight", (LATENT, HIDDEN), fp8_moe_dense)
        _dense(out, m + "routed_expert_up_proj.weight", (HIDDEN, LATENT), fp8_moe_dense)
        key = m + "routed_expert_norm.weight"
        out[key] = SynthTensor(key, (LATENT,), "pos", 0.5)
        for name in ("gate_proj", "up_proj"):
            _dense(out, m + f"shared_experts.{name}.weight", (SHARED, HIDDEN), fp8_moe_dense)
        _dense(out, m + "shared_experts.down_proj.weight", (HIDDEN, SHARED), fp8_moe_dense)
        for e in range(experts):
            x = f"{m}experts.{e}."
            for w in ("w1", "w3"):
                out[x + w + ".weight"] = SynthTensor(x + w + ".weight", (INTER, LATENT // 2), "u8")
                key = x + w + ".weight_scale"
                out[key] = SynthTensor(key, (INTER, LATENT // 16), "ue4m3")
            out[x + "w2.weight"] = SynthTensor(x + "w2.weight", (LATENT, INTER // 2), "u8")
            out[x + "w2.weight_scale"] = SynthTensor(
                x + "w2.weight_scale", (LATENT, INTER // 16), "ue4m3"
            )
            for w in ("w1", "w2", "w3"):
                scale = 1e-3 * (2.0 if (w == "w3" and mismatched_scale2 and e % 2) else 1.0)
                key = x + w + ".weight_scale_2"
                out[key] = SynthTensor(key + ("" if w != "w3" else ".w3"), (), "pos", scale)
                key = x + w + ".input_scale"
                out[key] = SynthTensor(key, (), "pos", 0.01)
    return out


def text_config(**overrides) -> dict:
    """A ``config.json``-like dict with the Kimi K3 values the chain reads."""
    cfg = dict(
        num_hidden_layers=93,
        rms_norm_eps=1e-5,
        routed_scaling_factor=1.0,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        linear_attn_config={"gate_lower_bound": -5.0},
    )
    cfg.update(overrides)
    return {"text_config": cfg}
