# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 ULL target: the weight table, the checkpoint loader and the weight containers it fills.

Sections: the native MXFP8 weights, the DFlash2 drafter's shape, the routed experts and the loader.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Dict, Optional, Protocol, Tuple

import torch


# ---- native MXFP8 weights ----------------------------------------
def dequantize_mxfp8(codes: torch.Tensor, raw_scales: torch.Tensor) -> torch.Tensor:
    """Exact BF16 [N,K] of E4M3 codes x 2^(UE8M0 - 127) per 1x32 block.

    An E4M3 value has at most four significant bits, so each product is a
    BF16 value unless its exponent leaves BF16's range; the product is formed
    exactly in FP32 (2^-136 is its smallest magnitude) and any value that BF16
    would round, or that is not finite, is rejected rather than rounded.
    """
    if codes.ndim != 2 or codes.dtype != torch.float8_e4m3fn:
        raise TypeError("MXFP8 codes must be E4M3 [N,K]")
    n, k = codes.shape
    if k % 32 or raw_scales.dtype != torch.uint8 or raw_scales.shape != (n, k // 32):
        raise ValueError("MXFP8 scales must be UE8M0 bytes [N,K/32]")
    if raw_scales.device != codes.device:
        raise ValueError("MXFP8 codes and scales must share one device")
    exponent = raw_scales.to(torch.int32)
    if bool((exponent == 255).any()):
        raise ValueError("UE8M0 scale 0xFF (NaN) in a checkpoint MXFP8 weight")
    # 2^(e-127) as FP32 bits: normal for e >= 1, the subnormal 2^-127 for e == 0.
    bits = torch.where(exponent > 0, exponent << 23, torch.full_like(exponent, 1 << 22))
    scale = bits.view(torch.float32)
    exact = codes.to(torch.float32).view(n, k // 32, 32) * scale.unsqueeze(-1)
    exact = exact.view(n, k)
    out = exact.to(torch.bfloat16)
    if not bool(torch.isfinite(exact).all()) or not torch.equal(out.to(torch.float32), exact):
        raise ValueError("A checkpoint MXFP8 element is not exactly representable in BF16")
    return out


@dataclass(frozen=True)
class DequantizedMxfp8:
    """BF16 [N,K] equal element for element to a checkpoint MXFP8 weight.

    Built by ``dequantize_mxfp8``; the checkpoint codes and scales are not
    retained, so this is the projection's only resident representation.
    """

    weight: torch.Tensor

    def __post_init__(self) -> None:
        w = self.weight
        if w.ndim != 2 or w.dtype != torch.bfloat16 or not w.is_contiguous():
            raise TypeError("Dequantized MXFP8 weights must be contiguous BF16 [N,K]")
        if w.data_ptr() % 16:
            raise ValueError("Dequantized MXFP8 weights must be 16-byte aligned for TMA")

    @classmethod
    def from_checkpoint(cls, codes: torch.Tensor, raw_scales: torch.Tensor) -> "DequantizedMxfp8":
        return cls(dequantize_mxfp8(codes, raw_scales))


def kda_pitch(local_heads: int) -> int:
    """MR8 in-projection row count/stride: [q,k,v,g,f_a,beta] padded to 16 BF16."""
    columns = 4 * local_heads * 128 + 128 + local_heads
    return (columns + 15) // 16 * 16


@dataclass(frozen=True)
class Bf16KdaInput:
    """MR8's packed BF16 in-projection [pitch, H], exactly dequantized from checkpoint MXFP8.

    Rows are [q,k,v,g (local heads), f_a (all 128), beta (local heads), zeros],
    the layout MR8's PreparedProjection -> PreparedKda reads (K3 ULL's
    ``in_proj_qkvgfab``). Only this rank's beta rows are kept: the BF16 GEMM has
    no N % 32 constraint. The zero rows only pad the stride and are never read.
    """

    packed: DequantizedMxfp8
    head_rank: int
    local_heads: int

    def __post_init__(self) -> None:
        if self.local_heads <= 0 or self.head_rank < 0:
            raise ValueError("Invalid KDA head partition")
        if (self.head_rank + 1) * self.local_heads > 96:
            raise ValueError("KDA head partition exceeds the checkpoint's 96 heads")
        if self.packed.weight.shape[0] != self.pitch:
            raise ValueError("Packed KDA projection rows must equal the MR8 pitch")

    @property
    def pitch(self) -> int:
        return kda_pitch(self.local_heads)

    @property
    def device(self) -> torch.device:
        return self.packed.weight.device

    @property
    def hidden(self) -> int:
        return self.packed.weight.shape[1]


@dataclass(frozen=True)
class Bf16SharedExpertWeights:
    """One rank's shared expert for MR8's BF16 SharedFront, exactly dequantized from MXFP8.

    ``gate_up`` is [gate; up] rows [2*I_local, H] (MR8's ``gate_up_proj`` layout:
    SiTU reads the first half as gate) and ``down`` is [H, I_local].
    """

    gate_up: DequantizedMxfp8
    down: DequantizedMxfp8

    def __post_init__(self) -> None:
        rows, hidden = self.gate_up.weight.shape
        if rows % 2 or self.down.weight.shape != (hidden, rows // 2):
            raise ValueError("Shared down must reverse the gate/up shard dimensions")
        if self.gate_up.weight.device != self.down.weight.device:
            raise ValueError("Shared expert weights must share one device")

    @property
    def device(self) -> torch.device:
        return self.gate_up.weight.device


# ---- the DFlash2 drafter ----------------------------------------
#: The DFlash2 drafter this target runs; modeling.py's ``draft_config_for`` asserts its config.
DRAFT_SHAPE = MappingProxyType(
    {
        "hidden_size": 7168,
        "num_hidden_layers": 5,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-5,
        "sliding_window": 1024,
    }
)
DRAFT_DFLASH = MappingProxyType(
    {
        "attention_mode": "mla",
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 32,
    }
)
#: Target layers whose captures ``fc`` projects (``dflash_config.target_layer_ids``).
DRAFT_CAPTURES = 6


# ---- routed experts ----------------------------------------
HIDDEN = 7168
LATENT = 3584
EXPERTS = 896
TP = 16
SLICE = 192
SF_VEC = 16


def _expert_shape(value) -> Tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return tuple(value.get_shape())


def _read(value, rows: Optional[slice] = None, cols: Optional[slice] = None) -> torch.Tensor:
    """``value[rows, cols]`` materialized (a lazy slice reads only that block)."""
    if rows is None and cols is None:
        if isinstance(value, torch.Tensor):
            return value
        if len(_expert_shape(value)) == 0:
            return value[()]
        return value[:]
    rows = rows if rows is not None else slice(None)
    return value[rows] if cols is None else value[rows, cols]


def checkpoint_prefix(weights: Mapping[str, Any]) -> str:
    """``language_model.`` when the checkpoint's text-model keys carry it, else ``""``."""
    for key in weights.keys():
        if key.startswith("language_model."):
            return "language_model."
    return ""


def w13_row_order() -> torch.Tensor:
    """Source of each of the 384 w13 rows the kernel reads, as (is_gate, channel) packed into
    ``channel + 192 * is_gate``: row ``128 c + 32 w + l`` of tile ``c`` feeds TMEM lane ``32 w + l``
    of epilogue warp ``w``; the up lanes (``l & 8 == 0``) read up channel
    ``64 c + 16 w + 2 (l % 8) + l // 16``, lane ``l | 8`` the gate row of the same channel (the
    epilogue pairs lanes ``l`` and ``l ^ 8``)."""
    order = torch.empty(3 * 128, dtype=torch.long)
    for c in range(3):
        for w in range(4):
            for lane in range(32):
                up = lane & ~8
                channel = 64 * c + 16 * w + 2 * (up % 8) + up // 16
                order[128 * c + 32 * w + lane] = channel + SLICE * int(lane & 8 != 0)
    return order


def w2_row_order(publish: bool = True) -> torch.Tensor:
    """Output column of each w2 row the kernel reads. The FC2 epilogue stages TMEM rows
    ``(l, l + 8)`` of a warp's 32 as one BF16 pair: with ``publish`` (the staged rows the
    finalize sums) row ``r`` of a 32-row group lands in column ``8 (j // 2) + 4 (j % 2) + k``,
    ``j = r % 8``, ``k = (r % 32) // 8``; without (direct rows) in ``4 j + k``."""
    r = torch.arange(LATENT)
    block, w = r // 32 * 32, r % 32
    j, k = w % 8, w // 8
    column = 8 * (j // 2) + 4 * (j % 2) + k if publish else 4 * j + k
    return block + column


def _atoms(scales: torch.Tensor, tiles: int, kblocks: int) -> torch.Tensor:
    """Scales [E, 128 * tiles, 4 * kblocks] (UE4M3 bytes) in the UMMA 128x4 atom order per
    (tile, K64 block): byte ``(m % 32) * 16 + (m // 32) * 4 + k`` of each 512 B atom."""
    experts = scales.shape[0]
    s = scales.reshape(experts, tiles, 4, 32, kblocks, 4)  # e, tile, m // 32, m % 32, kb, k
    return s.permute(0, 1, 4, 3, 2, 5).reshape(experts, -1).contiguous()


@torch.no_grad()
def pack_routed_experts_tp16(
    w1, w3, w2, w1_sf, w3_sf, w2_sf, w1_scale2, w3_scale2, w2_scale2, fc1_input_scale,
    fc2_input_scale, *, publish: bool = True,
):  # fmt: skip
    """One rank's TP16 routed experts in the layout of
    ``kernels/moe/rubin_ull_experts_nv_experts.RubinNvExpertsKernel``.

    Inputs (the checkpoint's NVFP4 tensors of every expert, sliced for this rank; any device):
      w1 / w3   gate / up rows of this rank's 192 intermediate channels, uint8 [E, 192, 1792]
                (packed E2M1, element 2 i in the low nibble)
      w2        this rank's 192 K columns of the down projection, uint8 [E, 3584, 96]
      *_sf      row-major UE4M3 block scales (one per 16 elements): [E, 192, 224], [E, 3584, 12]
                (float8_e4m3fn or uint8)
      *_scale2  per-expert global dequant scales (``weight_scale_2``), FP32 [E]
      fc1_input_scale / fc2_input_scale  the activations' global dequant scales (``input_scale``;
                every expert must use the checkpoint's unit activation scale)
      publish   the w2 row order of the published staged rows (True) or of direct rows

    Returns a dict: ``w13`` uint8 [E, 384, 1792], ``w13_sf`` uint8 [E, 86016], ``w2`` uint8
    [E, 3584, 96], ``w2_sf`` uint8 [E, 43008], ``alphas`` FP32 [E + 1, 2] ((alpha1, alpha2) per
    expert, then (h_gsf, 0)), ``x_gsf`` FP32 [1] (the latent image's quantization scale, which the
    routing kernel needs), ``h_gsf`` FP32 [1]. Gate/up global scales must match;
    unequal scales are rejected instead of rescaling checkpoint block scales.
    """
    experts = w1.shape[0]
    assert w1.shape == w3.shape == (experts, SLICE, LATENT // 2), w1.shape
    assert w2.shape == (experts, LATENT, SLICE // 2), w2.shape
    as_u8 = lambda t: t.view(torch.uint8) if t.dtype != torch.uint8 else t  # noqa: E731
    w1_sf, w3_sf, w2_sf = (as_u8(t) for t in (w1_sf, w3_sf, w2_sf))
    assert w1_sf.shape == w3_sf.shape == (experts, SLICE, LATENT // SF_VEC)
    assert w2_sf.shape == (experts, LATENT, SLICE // SF_VEC)
    globals_ = (w1_scale2, w3_scale2, w2_scale2)
    if any(t.dtype != torch.float32 for t in globals_):
        raise TypeError("Native NVFP4 global scales must remain FP32")
    s1, s3, s2 = (t.reshape(experts) for t in globals_)
    s13 = torch.maximum(s1, s3)
    if not torch.equal(s1, s3):
        raise ValueError("Native NVFP4 gate/up scales differ; lossy rescaling is forbidden")
    inputs = tuple(torch.as_tensor(t) for t in (fc1_input_scale, fc2_input_scale))
    if any(t.dtype != torch.float32 or not torch.all(t == 1.0) for t in inputs):
        raise ValueError("The native K3.1 checkpoint requires FP32 unit activation scales")
    if any(not torch.all(torch.isfinite(t) & (t > 0)) for t in globals_):
        raise ValueError("Native NVFP4 global scales must be finite and positive")
    in1, in2 = (t.reshape(-1)[0] for t in inputs)

    order = w13_row_order().to(w1.device)
    w13 = torch.cat([w3, w1], dim=1)[:, order]  # channel + 192 * is_gate
    w13_sf = torch.cat([w3_sf, w1_sf], dim=1)[:, order]
    columns = w2_row_order(publish).to(w2.device)
    w2p, w2p_sf = w2[:, columns], w2_sf[:, columns]

    alphas = torch.zeros(experts + 1, 2, dtype=torch.float32, device=w1.device)
    alphas[:experts, 0] = (in1 * s13).to(alphas.device)
    alphas[:experts, 1] = (in2 * s2).to(alphas.device)
    alphas[experts, 0] = 1.0 / in2
    return {
        "w13": w13.contiguous(),
        "w13_sf": _atoms(w13_sf, 3, LATENT // 64),
        "w2": w2p.contiguous(),
        "w2_sf": _atoms(w2p_sf, LATENT // 128, SLICE // 64),
        "alphas": alphas,
        "x_gsf": (1.0 / in1).reshape(1).to(alphas.device),
        "h_gsf": (1.0 / in2).reshape(1).to(alphas.device),
    }


def _unatoms(packed: torch.Tensor, tiles: int, kblocks: int) -> torch.Tensor:
    """Inverse of :func:`_atoms`: row-major scales [E, 128 * tiles, 4 * kblocks]."""
    experts = packed.shape[0]
    s = packed.reshape(experts, tiles, kblocks, 32, 4, 4)  # e, tile, kb, m % 32, m // 32, k
    return s.permute(0, 1, 4, 3, 2, 5).reshape(experts, 128 * tiles, 4 * kblocks).contiguous()


@torch.no_grad()
def repack_w2(packed: Mapping[str, torch.Tensor], columns: torch.Tensor, *, publish: bool = True):
    """``packed["w2"]`` / ``packed["w2_sf"]`` (rows in :func:`w2_row_order` ``(publish)``) with the
    rows in another order: row ``i`` of the result is the down projection's output column
    ``columns[i]``. Returns ``(w2, w2_sf)``."""
    source = w2_row_order(publish).to(packed["w2"].device)
    where = torch.empty_like(source)
    where[source] = torch.arange(source.numel(), device=source.device)
    index = where[columns.to(source.device)]
    w2 = packed["w2"][:, index].contiguous()
    sf = _unatoms(packed["w2_sf"], LATENT // 128, SLICE // 64)[:, index]
    return w2, _atoms(sf, LATENT // 128, SLICE // 64)


def _scalar(value) -> torch.Tensor:
    """A checkpoint scalar (a tensor or a lazy slice of shape ``[]`` / ``[1]``) as FP32 [1]."""
    if not isinstance(value, torch.Tensor):
        try:
            value = value[()]
        except Exception:  # noqa: BLE001 (slice types differ in what they accept for 0-d)
            value = value[:]
    value = torch.as_tensor(value)
    if value.dtype != torch.float32 or value.numel() != 1:
        raise TypeError("Native NVFP4 checkpoint scalar must be exactly one FP32 value")
    return value.reshape(1)


@torch.no_grad()
def read_routed_experts_tp16(
    weights: Mapping[str, Any], layer: int, rank: int, *, tp: int = TP, prefix: str = "", device=None,
) -> Dict[str, torch.Tensor]:  # fmt: skip
    """TP rank ``rank``'s slices of KDA + MoE ``layer``'s routed experts, read lazily from the
    checkpoint ``weights`` (``key -> tensor or safetensors slice``: only the rank's rows / columns
    are read), as the keyword arguments of :func:`pack_routed_experts_tp16`.

    Checkpoint (``hf-1ec10fa``, modelopt NVFP4, one tensor set per expert ``e``):
    ``{prefix}model.layers.{layer}.block_sparse_moe.experts.{e}.w{1,2,3}.{weight, weight_scale,
    weight_scale_2, input_scale}``; w1 (gate) / w3 (up) ``weight`` U8 [3072, 1792] and
    ``weight_scale`` E4M3 [3072, 224] are cut by rows (the rank's 192 intermediate channels), w2
    ``weight`` U8 [3584, 1536] / ``weight_scale`` E4M3 [3584, 192] by columns (its 96 bytes / 12
    scales); ``weight_scale_2`` / ``input_scale`` are FP32 scalars per expert (the activations'
    ``input_scale`` of every expert is passed; the packer uses their maximum).
    """
    if tp != TP or not 0 <= rank < tp:  # SLICE = 3072 / 16 channels per rank
        raise ValueError(f"Kimi K3 routed experts: the NVFP4 kernels take TP {TP}, not {tp}")
    rows = slice(rank * SLICE, (rank + 1) * SLICE)
    w2_cols = slice(rank * SLICE // 2, (rank + 1) * SLICE // 2)
    sf_cols = slice(rank * SLICE // SF_VEC, (rank + 1) * SLICE // SF_VEC)
    base = f"{prefix}model.layers.{layer}.block_sparse_moe.experts."
    u8, e4m3 = torch.uint8, torch.float8_e4m3fn
    out = {
        "w1": torch.empty(EXPERTS, SLICE, LATENT // 2, dtype=u8, device="cpu"),
        "w3": torch.empty(EXPERTS, SLICE, LATENT // 2, dtype=u8, device="cpu"),
        "w2": torch.empty(EXPERTS, LATENT, SLICE // 2, dtype=u8, device="cpu"),
        "w1_sf": torch.empty(EXPERTS, SLICE, LATENT // SF_VEC, dtype=u8, device="cpu"),
        "w3_sf": torch.empty(EXPERTS, SLICE, LATENT // SF_VEC, dtype=u8, device="cpu"),
        "w2_sf": torch.empty(EXPERTS, LATENT, SLICE // SF_VEC, dtype=u8, device="cpu"),
    }
    scalars = {
        k: torch.empty(EXPERTS, dtype=torch.float32, device="cpu")
        for k in ("w1_scale2", "w3_scale2", "w2_scale2", "in1", "in3", "in2")
    }

    def u8_of(t: torch.Tensor) -> torch.Tensor:
        if t.dtype not in (u8, e4m3):
            raise TypeError("Native NVFP4 weights/scales must contain U8/E4M3 checkpoint bytes")
        return t.view(u8)

    for e in range(EXPERTS):
        key = f"{base}{e}."
        for w in ("w1", "w3"):
            out[w][e] = u8_of(_read(weights[key + w + ".weight"], rows))
            out[w + "_sf"][e] = u8_of(_read(weights[key + w + ".weight_scale"], rows))
        out["w2"][e] = u8_of(_read(weights[key + "w2.weight"], slice(None), w2_cols))
        out["w2_sf"][e] = u8_of(_read(weights[key + "w2.weight_scale"], slice(None), sf_cols))
        for w, name in (("w1", "1"), ("w3", "3"), ("w2", "2")):
            scalars[f"w{name}_scale2"][e] = _scalar(weights[key + w + ".weight_scale_2"])[0]
            scalars[f"in{name}"][e] = _scalar(weights[key + w + ".input_scale"])[0]
    result = {k: v.to(device) for k, v in out.items()}
    result.update({k: scalars[k].to(device) for k in ("w1_scale2", "w3_scale2", "w2_scale2")})
    if not torch.equal(scalars["in1"], scalars["in3"]):
        raise ValueError(
            "Native gate/up activation scales differ; scale harmonization is forbidden"
        )
    result["fc1_input_scale"] = scalars["in1"].to(device)
    result["fc2_input_scale"] = scalars["in2"].to(device)
    return result


def load_routed_experts_tp16(
    weights: Mapping[str, Any], layer: int, rank: int, *, tp: int = TP, prefix: Optional[str] = None,
    device=None, publish: bool = True,
) -> Dict[str, torch.Tensor]:  # fmt: skip
    """:func:`pack_routed_experts_tp16` of :func:`read_routed_experts_tp16` (``prefix`` None: the
    checkpoint's, :func:`checkpoint_prefix`)."""
    prefix = checkpoint_prefix(weights) if prefix is None else prefix
    raw = read_routed_experts_tp16(weights, layer, rank, tp=tp, prefix=prefix, device=device)
    return pack_routed_experts_tp16(**raw, publish=publish)


# ---- loader ----------------------------------------
class TensorSlice(Protocol):
    """The safetensors slice interface used without reading a full tensor."""

    def get_shape(self) -> list[int]: ...

    def __getitem__(self, selection: tuple[slice, ...]) -> torch.Tensor: ...


TensorSource = torch.Tensor | TensorSlice


def _shape(source: TensorSource) -> tuple[int, ...]:
    return tuple(source.shape if isinstance(source, torch.Tensor) else source.get_shape())


@dataclass(frozen=True)
class KdaWeights:
    """KDA projections and MR8-layout constants; recurrent state is external.

    ``inputs``/``output`` are the exact BF16 dequantization of the checkpoint's MXFP8
    projections (``Bf16KdaInput`` [pitch,H] packed for MR8, ``DequantizedMxfp8``
    [H,D_local]); the codes and scales are not kept.
    ``conv`` is FP32 [3,W,D_local]; f_b is BF16 [D_local,128]; a_log is
    FP32 [H_local], dt_bias FP32 [D_local], and output_norm FP32 [128].
    """

    inputs: Bf16KdaInput
    output: DequantizedMxfp8
    f_b: torch.Tensor
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    output_norm: torch.Tensor


_SAFETENSORS_DTYPES = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}


def _safetensors_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    return 8 + size, header


def pread_routed_experts_tp16(
    path: str, layer: int, rank: int, *, prefix: str, device, threads: int = 16
) -> dict[str, torch.Tensor]:
    """:func:`read_routed_experts_tp16` of one shard file through ``pread``.

    Same bytes and result keys. Each rank-local range (w1/w3 rows, full w2
    rows cut by columns in memory, FP32 scalars) is one ``pread``, which
    releases the GIL, instead of a lazy-mmap slice that faults the file
    page by page on Lustre.
    """

    base_offset, header = _safetensors_header(path)
    base = f"{prefix}model.layers.{layer}.block_sparse_moe.experts."
    u8 = torch.uint8
    out = {
        "w1": torch.empty(EXPERTS, SLICE, LATENT // 2, dtype=u8),
        "w3": torch.empty(EXPERTS, SLICE, LATENT // 2, dtype=u8),
        "w2": torch.empty(EXPERTS, LATENT, SLICE // 2, dtype=u8),
        "w1_sf": torch.empty(EXPERTS, SLICE, LATENT // SF_VEC, dtype=u8),
        "w3_sf": torch.empty(EXPERTS, SLICE, LATENT // SF_VEC, dtype=u8),
        "w2_sf": torch.empty(EXPERTS, LATENT, SLICE // SF_VEC, dtype=u8),
    }
    scalars = {
        k: torch.empty(EXPERTS, dtype=torch.float32)
        for k in ("w1_scale2", "w3_scale2", "w2_scale2", "in1", "in3", "in2")
    }
    fd = os.open(path, os.O_RDONLY)

    def entry(name: str, dtypes: tuple[str, ...], shape: tuple[int, ...] | None):
        meta = header[name]
        if meta["dtype"] not in dtypes:
            raise TypeError(f"{name}: native NVFP4 bank holds {meta['dtype']}, not {dtypes}")
        if shape is not None and tuple(meta["shape"]) != shape:
            raise ValueError(f"{name}: shape {meta['shape']} != {shape}")
        start, end = meta["data_offsets"]
        return base_offset + start, end - start

    def read_into(dst: torch.Tensor, offset: int) -> None:
        view = memoryview(dst.numpy()).cast("B")
        done = 0
        while done < len(view):
            n = os.preadv(fd, [view[done:]], offset + done)
            if n <= 0:
                raise OSError(f"short read of {path} at {offset + done}")
            done += n

    def scalar(name: str) -> float:
        offset, length = entry(name, ("F32",), None)
        if length != 4:
            raise TypeError("Native NVFP4 checkpoint scalar must be exactly one FP32 value")
        value = torch.empty(1, dtype=torch.float32)
        read_into(value, offset)
        return float(value[0])

    w1_shape, sf_shape = (SLICE * 16, LATENT // 2), (SLICE * 16, LATENT // SF_VEC)
    w2_shape, w2sf_shape = (LATENT, SLICE * 16 // 2), (LATENT, SLICE * 16 // SF_VEC)
    w2_cols = slice(rank * SLICE // 2, (rank + 1) * SLICE // 2)
    sf_cols = slice(rank * SLICE // SF_VEC, (rank + 1) * SLICE // SF_VEC)

    def one(e: int) -> None:
        key = f"{base}{e}."
        for w in ("w1", "w3"):
            offset, _ = entry(key + w + ".weight", ("U8",), w1_shape)
            read_into(out[w][e], offset + rank * SLICE * (LATENT // 2))
            offset, _ = entry(key + w + ".weight_scale", ("U8", "F8_E4M3"), sf_shape)
            read_into(out[w + "_sf"][e], offset + rank * SLICE * (LATENT // SF_VEC))
        full = torch.empty(w2_shape, dtype=u8)
        read_into(full, entry(key + "w2.weight", ("U8",), w2_shape)[0])
        out["w2"][e] = full[:, w2_cols]
        full = torch.empty(w2sf_shape, dtype=u8)
        read_into(full, entry(key + "w2.weight_scale", ("U8", "F8_E4M3"), w2sf_shape)[0])
        out["w2_sf"][e] = full[:, sf_cols]
        for w, n in (("w1", "1"), ("w3", "3"), ("w2", "2")):
            scalars[f"w{n}_scale2"][e] = scalar(key + w + ".weight_scale_2")
            scalars[f"in{n}"][e] = scalar(key + w + ".input_scale")

    try:
        with ThreadPoolExecutor(threads, thread_name_prefix="k31-experts") as pool:
            list(pool.map(one, range(EXPERTS)))
    finally:
        os.close(fd)
    result = {k: v.to(device) for k, v in out.items()}
    result.update({k: scalars[k].to(device) for k in ("w1_scale2", "w3_scale2", "w2_scale2")})
    if not torch.equal(scalars["in1"], scalars["in3"]):
        raise ValueError(
            "Native gate/up activation scales differ; scale harmonization is forbidden"
        )
    result["fc1_input_scale"] = scalars["in1"].to(device)
    result["fc2_input_scale"] = scalars["in2"].to(device)
    return result


class ExpertReadAhead:
    """Sequential page-cache read-ahead of the shard files holding upcoming expert layers.

    Per-tensor slices through a lazy mmap fault the checkpoint in page-sized
    random reads (~250 MB/s per node on Lustre). Each layer's routed experts
    live in one shard file, so the node's ranks instead stream the files of the
    next layers with large sequential reads, striped across the node's ranks by
    chunk, and the slice reads then hit the shared page cache. Read bytes are
    discarded; nothing is copied into the weights from here.
    """

    CHUNK = 256 << 20

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        prefix: str,
        local_rank: int,
        local_size: int,
        depth: int = 2,
        threads: int = 8,
    ) -> None:
        index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        pattern = re.compile(
            rf"^{re.escape(prefix)}model\.layers\.(\d+)\.block_sparse_moe\.experts\."
        )
        self.files: dict[int, set[str]] = {}
        for key, name in weight_map.items():
            match = pattern.match(key)
            if match:
                self.files.setdefault(int(match.group(1)), set()).add(
                    os.path.join(checkpoint_dir, name)
                )
        self.local_rank, self.local_size, self.depth = local_rank, local_size, depth
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="k31-readahead")
        self.started: set[str] = set()

    def _read(self, path: str, start: int, length: int) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            buffer = bytearray(min(self.CHUNK, 64 << 20))
            view = memoryview(buffer)
            end, pos = start + length, start
            while pos < end:
                n = os.preadv(fd, [view[: min(len(buffer), end - pos)]], pos)
                if n <= 0:
                    break
                pos += n
        finally:
            os.close(fd)

    def ensure(self, layer: int) -> None:
        """Start reading ``layer`` .. ``layer + depth`` (each file once per process)."""
        for upcoming in range(layer, layer + self.depth + 1):
            for path in sorted(self.files.get(upcoming, ())):
                if path in self.started:
                    continue
                self.started.add(path)
                size = os.path.getsize(path)
                for i, start in enumerate(range(0, size, self.CHUNK)):
                    if i % self.local_size == self.local_rank:
                        self.pool.submit(self._read, path, start, min(self.CHUNK, size - start))

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


class FlatCheckpointLoader:
    """One ULL rank's loader over tensors or lazy safetensors slices.

    ``quantized_layers`` is the unmodified ModelOpt per-layer recipe dictionary.
    ``weights`` and its keys use checkpoint names including the language-model
    prefix. CPU devices are useful for inspecting shards with ``read_mxfp8``;
    final scale swizzling and ready-to-run containers require a GPU.

    The MXFP8 KDA in/o and shared gate/up/down projections load as their exact BF16
    dequantization for MR8's BF16 kernels, freeing the codes and scales; the KDA
    in-projection is packed in MR8's operand order (``Bf16KdaInput``).
    """

    def __init__(
        self,
        weights: Mapping[str, TensorSource],
        quantized_layers: Mapping[str, Mapping[str, object]],
        *,
        device: torch.device,
        rank: int,
        size: int = 16,
    ) -> None:
        if size != 16 or not 0 <= rank < size:
            raise ValueError("The flat ULL loader requires one rank of TP16")
        self.weights = weights
        self.quantized_layers = quantized_layers
        self.device = torch.device(device)
        self.rank, self.size = rank, size
        self.read_ahead: ExpertReadAhead | None = None
        self._pread_checked = False
        checkpoint_dir = getattr(weights, "checkpoint_dir", None)
        if checkpoint_dir and os.path.isfile(
            os.path.join(checkpoint_dir, "model.safetensors.index.json")
        ):
            # Ranks per host: the GPUs this process sees (every served host shows its 4).
            local = torch.cuda.device_count() or 1
            self.read_ahead = ExpertReadAhead(
                checkpoint_dir, prefix="language_model.", local_rank=rank % local, local_size=local
            )

    def read(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        shard_dim: int | None = None,
        partition: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Read the requested checkpoint dtype exactly, optionally one TP shard."""
        source = self.weights[name]
        if _shape(source) != shape:
            raise ValueError(f"{name}: checkpoint shape {_shape(source)} != {shape}")
        size, rank = (self.size, self.rank) if partition is None else partition
        if size <= 0 or not 0 <= rank < size:
            raise ValueError("Invalid checkpoint tensor partition")
        selection = [slice(None)] * len(shape)
        if shard_dim is not None:
            if not 0 <= shard_dim < len(shape):
                raise ValueError("Shard dimension is outside the checkpoint tensor")
            if shape[shard_dim] % size:
                raise ValueError(f"{name}: checkpoint dimension is not divisible by partition size")
            width = shape[shard_dim] // size
            start = rank * width
            selection[shard_dim] = slice(start, start + width)
        tensor = source[tuple(selection)]
        if tensor.dtype != dtype:
            raise TypeError(f"{name}: checkpoint dtype {tensor.dtype} != required {dtype}")
        return tensor.to(device=self.device).contiguous()

    def read_mxfp8(
        self, name: str, shape: tuple[int, int], *, shard_dim: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read native [N,K] E4M3 codes and [N,K/32] UE8M0 scale shards."""
        if self.quantized_layers.get(name, {}).get("quant_algo") != "MXFP8":
            raise ValueError(f"{name}: checkpoint recipe must declare MXFP8")
        n, k = shape
        if n % 32 or k % 32:
            raise ValueError(f"{name}: native MXFP8 shape must be group32-aligned")
        if shard_dim == 1 and k % (32 * self.size):
            raise ValueError(f"{name}: TP16 K shard splits an MXFP8 scale group")
        codes = self.read(name + ".weight", shape, torch.float8_e4m3fn, shard_dim=shard_dim)
        scales = self.read(
            name + ".weight_scale",
            (n, k // 32),
            torch.uint8,
            shard_dim=shard_dim,
        )
        if scales.shape != (codes.shape[0], codes.shape[1] // 32):
            raise ValueError(f"{name}: code and scale shards do not cover the same elements")
        return codes, scales

    def kda_a_log(self, name: str, *, heads: int = 96, padded: int = 128) -> torch.Tensor:
        """Select logical heads before sharding the checkpoint's zero-padded vector."""
        shape = _shape(self.weights[name])
        if shape not in ((heads,), (padded,)) or heads % self.size:
            raise ValueError(
                f"{name}: expected {heads} logical heads with optional {padded} padding"
            )
        values = self.read(name, shape, torch.float32)
        if shape[0] > heads and torch.count_nonzero(values[heads:]).item():
            raise ValueError(f"{name}: A_log padding must contain only zeros")
        local = heads // self.size
        start = self.rank * local
        # A copy, not a view: a view of the full vector starts at 4 * local * rank bytes, which the
        # 16-byte-aligned KDA kernel operands cannot take on every rank.
        return values[start : start + local].clone()

    def kda(
        self,
        name: str,
        *,
        hidden: int,
        heads: int = 96,
        head_dim: int = 128,
        conv_width: int = 4,
    ) -> KdaWeights:
        """Load one checkpoint ``...layers.i.self_attn`` directly for the MR8 core."""
        if heads != 96 or head_dim != 128:
            raise ValueError("The ULL MR8 adapter requires 96 heads with dimension 128")
        if conv_width <= 0:
            raise ValueError("KDA convolution width must be positive")
        width = heads * head_dim
        inputs = self.kda_inputs_bf16(name, hidden=hidden, heads=heads, head_dim=head_dim)
        output = self.dequantized(name + ".o_proj", (hidden, width), shard_dim=1)
        f_b = self.read(name + ".f_b_proj.weight", (width, head_dim), torch.bfloat16, shard_dim=0)
        conv = torch.stack(
            [
                self.read(
                    name + "." + p + "_conv1d.weight",
                    (width, 1, conv_width),
                    torch.float32,
                    shard_dim=0,
                )
                .squeeze(1)
                .transpose(0, 1)
                for p in ("q", "k", "v")
            ]
        ).contiguous()
        a_log = self.kda_a_log(name + ".A_log", heads=heads, padded=head_dim)
        dt_bias = self.read(name + ".dt_bias", (width,), torch.float32, shard_dim=0)
        norm = self.read(name + ".o_norm.weight", (head_dim,), torch.float32)
        return KdaWeights(inputs, output, f_b, conv, a_log, dt_bias, norm)

    def dequantized(
        self, name: str, shape: tuple[int, int], *, shard_dim: int | None = None
    ) -> DequantizedMxfp8:
        """A checkpoint MXFP8 projection shard as its exact BF16 values; codes/scales are freed."""
        codes, scales = self.read_mxfp8(name, shape, shard_dim=shard_dim)
        return DequantizedMxfp8(dequantize_mxfp8(codes, scales))

    def kda_inputs_bf16(
        self, name: str, *, hidden: int, heads: int = 96, head_dim: int = 128
    ) -> Bf16KdaInput:
        """MR8's packed [q,k,v,g,f_a,beta,0] BF16 in-projection, dequantized part by part.

        Each part's codes and scales are freed once its rows are written, so at
        most one part's MXFP8 staging coexists with the packed bank during load.
        """
        local = heads // self.size
        width = heads * head_dim
        packed = torch.zeros((kda_pitch(local), hidden), dtype=torch.bfloat16, device=self.device)
        row = 0
        for p in ("q", "k", "v", "g"):
            codes, scales = self.read_mxfp8(name + "." + p + "_proj", (width, hidden), shard_dim=0)
            packed[row : row + codes.shape[0]] = dequantize_mxfp8(codes, scales)
            row += codes.shape[0]
            del codes, scales
        codes, scales = self.read_mxfp8(name + ".f_a_proj", (head_dim, hidden))
        packed[row : row + head_dim] = dequantize_mxfp8(codes, scales)
        row += head_dim
        # The beta GEMM no longer needs N % 32: keep only this rank's whole rows.
        codes, scales = self.read_mxfp8(name + ".b_proj", (heads, hidden))
        start = self.rank * local
        packed[row : row + local] = dequantize_mxfp8(
            codes[start : start + local].contiguous(), scales[start : start + local].contiguous()
        )
        del codes, scales
        return Bf16KdaInput(DequantizedMxfp8(packed), self.rank, local)

    def shared(self, name: str, *, hidden: int, intermediate: int) -> Bf16SharedExpertWeights:
        """Load ``...shared_experts`` with column gate/up and row down shards."""
        gate_up = torch.cat(
            [
                self.dequantized(name + "." + p, (intermediate, hidden), shard_dim=0).weight
                for p in ("gate_proj", "up_proj")
            ]
        )
        down = self.dequantized(name + ".down_proj", (hidden, intermediate), shard_dim=1)
        return Bf16SharedExpertWeights(DequantizedMxfp8(gate_up), down)

    def experts(self, layer: int, *, prefix: str = "language_model.") -> dict[str, torch.Tensor]:
        """Load one native NVFP4 TP16 expert bank directly into the MR8 byte layout."""
        bank = f"{prefix}model.layers.{layer}.block_sparse_moe.experts"
        if self.quantized_layers.get(bank, {}).get("quant_algo") != "NVFP4":
            raise ValueError(f"{bank}: checkpoint recipe must declare NVFP4")

        files = sorted(self.read_ahead.files.get(layer, ())) if self.read_ahead else []
        if len(files) == 1:
            self.read_ahead.ensure(layer)
            raw = pread_routed_experts_tp16(
                files[0], layer, self.rank, prefix=prefix, device=self.device
            )
            if not self._pread_checked:
                self._check_pread(raw, layer, prefix)
            return pack_routed_experts_tp16(**raw, publish=True)
        return load_routed_experts_tp16(
            self.weights, layer, self.rank, prefix=prefix, device=self.device
        )

    def _check_pread(self, raw: dict[str, torch.Tensor], layer: int, prefix: str) -> None:
        """Compare expert 0 and 895 of the first pread bank with the lazy-slice reader."""

        base = f"{prefix}model.layers.{layer}.block_sparse_moe.experts."
        rows = slice(self.rank * 192, (self.rank + 1) * 192)
        for e in (0, 895):
            key = f"{base}{e}."
            pairs = (
                ("w1", _read(self.weights[key + "w1.weight"], rows)),
                ("w3_sf", _read(self.weights[key + "w3.weight_scale"], rows)),
                (
                    "w2",
                    _read(
                        self.weights[key + "w2.weight"],
                        slice(None),
                        slice(self.rank * 96, (self.rank + 1) * 96),
                    ),
                ),
                (
                    "w2_sf",
                    _read(
                        self.weights[key + "w2.weight_scale"],
                        slice(None),
                        slice(self.rank * 12, (self.rank + 1) * 12),
                    ),
                ),
            )
            for name, ref in (
                ("w2_scale2", "w2.weight_scale_2"),
                ("fc1_input_scale", "w1.input_scale"),
            ):
                if float(raw[name][e]) != float(_scalar(self.weights[key + ref])[0]):
                    raise RuntimeError(
                        f"pread expert reader differs from the checkpoint: {key}{ref}"
                    )
            for name, ref in pairs:
                if not torch.equal(raw[name][e].cpu(), ref.view(torch.uint8)):
                    raise RuntimeError(
                        f"pread expert reader differs from the checkpoint: {key}{name}"
                    )
        self._pread_checked = True

    def moe(
        self,
        layer: int,
        *,
        scale: float,
        renormalize: bool,
        eps: float,
        prefix: str = "language_model.",
    ):
        """Load directly into native TP16 front/shared/experts/tail representations."""
        from .modeling import MoeWeights

        name = f"{prefix}model.layers.{layer}.block_sparse_moe"
        router = self.bf16(name + ".gate", (896, 7168), shard_dim=0).weight
        down = self.bf16(name + ".routed_expert_down_proj", (3584, 7168), shard_dim=0).weight
        front = torch.cat((router, router.new_zeros((8, 7168)), down))
        return MoeWeights(
            front=front,
            bias=self.read(name + ".gate.e_score_correction_bias", (896,), torch.float32),
            norm=self.read(name + ".routed_expert_norm.weight", (3584,), torch.bfloat16),
            up=self.bf16(name + ".routed_expert_up_proj", (7168, 3584), shard_dim=0).weight,
            shared=self.shared(name + ".shared_experts", hidden=7168, intermediate=6144),
            routed=self.experts(layer, prefix=prefix),
            scale=scale,
            renormalize=renormalize,
            eps=eps,
        )

    def bf16(
        self,
        name: str,
        shape: tuple[int, int],
        *,
        shard_dim: int | None = None,
        partition: tuple[int, int] | None = None,
    ):
        """Load a declared unquantized projection without constructing a Linear."""
        if self.quantized_layers.get(name, {}).get("quant_algo") is not None:
            raise ValueError(f"{name}: quantized checkpoint projection cannot use BF16 GEMM")
        from .modeling import Bf16Weight

        return Bf16Weight(
            self.read(
                name + ".weight", shape, torch.bfloat16, shard_dim=shard_dim, partition=partition
            )
        )

    def norm(self, name: str, *, hidden: int, eps: float):
        """Load an excluded glue norm; KDA's FP32 output norm uses its separate loader."""
        from .modeling import NormWeight

        return NormWeight(self.read(name + ".weight", (hidden,), torch.bfloat16), eps)

    def attn_res(self, name: str, output_norm: str, *, hidden: int, eps: float):
        """Load a ``...self_attention_res`` / ``...mlp_res`` / ``...output_attn_res`` site."""
        from .modeling import AttnResWeights

        return AttnResWeights(
            self.bf16(name + "_proj", (1, hidden)),
            self.norm(name + "_norm", hidden=hidden, eps=eps),
            self.norm(output_norm, hidden=hidden, eps=eps),
        )

    def dense(self, name: str, *, hidden: int, intermediate: int, tp_size: int):
        """Load L0's dense MLP on a TP subgroup of ``tp_size`` ranks (all 16 on this deployment)."""
        from .modeling import Bf16Weight, DenseWeights

        if tp_size <= 0 or self.size % tp_size:
            raise ValueError("Dense TP subgroup must divide the ULL TP16 group")
        partition = (tp_size, self.rank % tp_size)
        gate = self.bf16(
            name + ".gate_proj", (intermediate, hidden), shard_dim=0, partition=partition
        )
        up = self.bf16(name + ".up_proj", (intermediate, hidden), shard_dim=0, partition=partition)
        gate_up = Bf16Weight(torch.cat((gate.weight, up.weight), dim=0))
        down = self.bf16(
            name + ".down_proj", (hidden, intermediate), shard_dim=1, partition=partition
        )
        return DenseWeights(gate_up, down)

    def model(self, config, *, prefix: str = "language_model."):
        """Load the flat model's sole BF16 glue/embedding/head/dense weight set.

        L0's dense MLP is split over all 16 ranks (every rank one sixteenth of the
        intermediate, ``dense0.FlatDense0``).

        Attention and MoE runtimes consume their native containers separately;
        this does not construct a shared model and then copy its parameters.
        """
        from .modeling import FlatLayerWeights, FlatModelWeights

        hidden, vocab = config.hidden_size, config.vocab_size
        layers, block = config.num_hidden_layers, config.attn_res_block_size
        if (hidden, vocab, layers, config.first_k_dense_replace) != (7168, 163840, 93, 1):
            raise ValueError("Flat ULL model loading requires the native K3.1 model geometry")
        if block <= 0 or config.intermediate_size != 33792:
            raise ValueError("Flat K3.1 needs its declared AttnRes block and native dense width")
        beta = float(getattr(config, "activation_situ_beta", None) or 1.0)
        linear = getattr(config, "activation_situ_linear_beta", None)
        if not isfinite(beta) or beta <= 0:
            raise ValueError("Dense SiTU beta must be finite and positive")
        if linear is not None:
            linear = float(linear)
            if not isfinite(linear) or linear <= 0:
                raise ValueError("Dense SiTU linear beta must be finite and positive")
        name, eps = prefix + "model", config.rms_norm_eps
        weights = []
        for layer in range(layers):
            local = f"{name}.layers.{layer}"
            weights.append(
                FlatLayerWeights(
                    attention_res=self.attn_res(
                        local + ".self_attention_res",
                        local + ".input_layernorm",
                        hidden=hidden,
                        eps=eps,
                    ),
                    mlp_res=self.attn_res(
                        local + ".mlp_res",
                        local + ".post_attention_layernorm",
                        hidden=hidden,
                        eps=eps,
                    ),
                    dense=self.dense(
                        local + ".mlp",
                        hidden=hidden,
                        intermediate=config.intermediate_size,
                        tp_size=self.size,
                    )
                    if layer == 0
                    else None,
                )
            )
        return FlatModelWeights(
            embedding=self.bf16(name + ".embed_tokens", (vocab, hidden)),
            layers=tuple(weights),
            output=self.attn_res(name + ".output_attn_res", name + ".norm", hidden=hidden, eps=eps),
            lm_head=self.bf16(prefix + "lm_head", (vocab, hidden), shard_dim=0),
            vocab_size=vocab,
            vocab_start=self.rank * (vocab // self.size),
            attn_res_block_size=block,
            dense_beta=beta,
            dense_linear_beta=linear,
        )

    def dsa(self, layer: int, config, *, prefix: str = "language_model."):
        """Load K3.1's packed, unquantized DSA projections for Helix CP16."""
        from .modeling import Bf16Weight, DsaWeights

        sparse = config.dsa_config
        if not isinstance(sparse, dict):
            sparse = vars(sparse)
        geometry = (
            config.hidden_size,
            config.num_attention_heads,
            config.q_lora_rank,
            config.kv_lora_rank,
            config.qk_nope_head_dim,
            config.qk_rope_head_dim,
            config.v_head_dim,
            sparse["index_num_heads"],
            sparse["index_head_dim"],
            sparse.get("index_rope_dim", 0),
            sparse.get("index_k_norm_type", "layernorm"),
        )
        if geometry != (7168, 96, 1536, 512, 128, 64, 128, 32, 128, 0, "layernorm"):
            raise ValueError(
                "Flat DSA requires the native K3.1 CP16/position-free indexer geometry"
            )
        if layer + 1 not in sparse["dsa_layers"]:
            raise ValueError("Requested layer is not a checkpoint DSA layer")
        name = f"{prefix}model.layers.{layer}.self_attn"
        index = name + ".indexer"
        # One [q_a | kv_a | wk | weights_proj | this rank's g_proj rows] bank: ``hidden`` and
        # ``gate`` are its row views, ``packed`` the whole (one GEMM for both).
        packed = torch.cat(
            [
                self.bf16(name + ".q_a_proj", (1536, 7168)).weight,
                self.bf16(name + ".kv_a_proj_with_mqa", (576, 7168)).weight,
                self.bf16(index + ".wk", (128, 7168)).weight,
                self.bf16(index + ".weights_proj", (32, 7168)).weight,
                self.bf16(name + ".g_proj", (96 * 128, 7168), shard_dim=0).weight,
            ]
        )
        hidden, gate = packed.split((2272, 768))
        query = torch.cat(
            [
                self.bf16(name + ".q_b_proj", (96 * 192, 1536)).weight,
                self.bf16(index + ".wq_b", (32 * 128, 1536)).weight,
            ]
        )
        # Checkpoint rows are per-head [K | V], unlike the shared MLA loader's
        # grouped runtime [all K | all V] representation. Retain only operands.
        kv_b = self.bf16(name + ".kv_b_proj", (96 * 256, 512)).weight.view(96, 256, 512)
        eps = config.rms_norm_eps
        index_eps = sparse.get("index_k_norm_eps")
        return DsaWeights(
            hidden=Bf16Weight(hidden),
            query=Bf16Weight(query),
            q_norm=self.norm(name + ".q_a_layernorm", hidden=1536, eps=eps),
            kv_norm=self.norm(name + ".kv_a_layernorm", hidden=512, eps=eps),
            index_norm=self.read(index + ".k_norm.weight", (128,), torch.bfloat16).float(),
            index_bias=self.read(index + ".k_norm.bias", (128,), torch.bfloat16).float(),
            index_eps=1e-6 if index_eps is None else float(index_eps),
            k_b=kv_b[:, :128].contiguous(),
            v_b=kv_b[:, 128:].contiguous(),
            gate=Bf16Weight(gate),
            output=self.bf16(name + ".o_proj", (7168, 96 * 128), shard_dim=1),
            packed=Bf16Weight(packed),
        )

    def dflash2(
        self, config, *, cp_rank: int, head_parallel: bool = False
    ) -> dict[str, torch.Tensor]:
        """The DFlash2 drafter's BF16 checkpoint as ``DraftCore.w``'s ``d_*`` tensors.

        Keys are the checkpoint's (``fc.weight``, ``hidden_norm.weight``, ``norm.weight``,
        ``layers.{j}.*``, ``candidate_selector.*``); every one is read. The helix layouts keep all
        96 query and KV-B heads on every rank (each rank attends over its own history pages), so
        ``q_b`` and ``kv_b`` are whole; ``head_parallel`` (the "headtp" layout: each rank attends
        with its own 6 heads over the whole window) keeps the rows of heads ``[6 cp_rank, 6
        cp_rank + 6)`` of both. ``o_proj`` keeps those 6 heads' input columns (the helix
        exchange's or the head-parallel attention's output), and the MLP a sixteenth of the
        intermediate (rows of gate/up, columns of down), row-parallel over the 16 ranks repurposed
        as TP; ``fc`` a sixteenth of its output rows (all-gathered after the projection).
        ``d_kv_a`` stacks the five layers' ``kv_a_proj_with_mqa`` (the history's projection);
        ``d_l{j}_q_kv_a`` is layer ``j``'s ``[q_a_proj ; kv_a_proj_with_mqa]`` bank (one block
        GEMM, kv_a's rows held twice); ``d_kv_b`` holds its heads' ``k_nope`` rows, then their
        ``v`` rows.
        """
        for key, want in DRAFT_SHAPE.items():
            if getattr(config, key, None) != want:
                raise ValueError(f"DFlash2 drafter {key} {getattr(config, key, None)} != {want}")
        if not 0 <= cp_rank < self.size:
            raise ValueError("The drafter's shares need this rank's Helix CP rank")
        hidden, layers = DRAFT_SHAPE["hidden_size"], DRAFT_SHAPE["num_hidden_layers"]
        heads, q_rank = DRAFT_SHAPE["num_attention_heads"], DRAFT_SHAPE["q_lora_rank"]
        latent, rope = DRAFT_SHAPE["kv_lora_rank"], DRAFT_SHAPE["qk_rope_head_dim"]
        nope, value = DRAFT_SHAPE["qk_nope_head_dim"], DRAFT_SHAPE["v_head_dim"]
        inter, vocab = DRAFT_SHAPE["intermediate_size"], DRAFT_SHAPE["vocab_size"]
        taps = DRAFT_DFLASH["conv_kernel_size"]
        groups = hidden // DRAFT_DFLASH["conv_group_size"]
        rank = DRAFT_DFLASH["selector_rank"]
        share = (self.size, cp_rank)
        heads_cut = dict(shard_dim=0, partition=share) if head_parallel else {}
        read: set[str] = set()

        def bf16(name: str, shape: tuple[int, ...], **cut) -> torch.Tensor:
            read.add(name)
            return self.read(name, shape, torch.bfloat16, **cut)

        def layer(j: int, name: str) -> str:
            return f"layers.{j}.{name}"

        kv_a = [
            bf16(layer(j, "self_attn.kv_a_proj_with_mqa.weight"), (latent + rope, hidden))
            for j in range(layers)
        ]
        w = {
            "d_fc": bf16(
                "fc.weight", (hidden, DRAFT_CAPTURES * hidden), shard_dim=0, partition=share
            ),
            "d_hidden_norm": bf16("hidden_norm.weight", (hidden,)),
            "d_kv_a": torch.cat(kv_a),
            "d_kv_a_norm": torch.stack(
                [
                    bf16(layer(j, "self_attn.kv_a_layernorm.weight"), (latent,))
                    for j in range(layers)
                ]
            ),
            "d_final_norm": bf16("norm.weight", (hidden,)),
            "d_sel_pred": bf16("candidate_selector.predecessor_codebook", (vocab, rank)),
            "d_sel_succ": bf16("candidate_selector.successor_codebook", (vocab, rank)),
            "d_sel_hidden_proj": bf16(
                "candidate_selector.hidden_projection.weight", (rank, hidden)
            ),
        }
        for j in range(layers):
            # Checkpoint rows are head-major (each head's nope + value rows): a head slice is a
            # row slice.
            kv_b = bf16(
                layer(j, "self_attn.kv_b_proj.weight"),
                (heads * (nope + value), latent),
                **heads_cut,
            )
            kv_b = kv_b.view(-1, nope + value, latent)
            gate = bf16(
                layer(j, "mlp.gate_proj.weight"), (inter, hidden), shard_dim=0, partition=share
            )
            up = bf16(layer(j, "mlp.up_proj.weight"), (inter, hidden), shard_dim=0, partition=share)
            w.update(
                {
                    f"d_l{j}_in_norm": bf16(layer(j, "input_layernorm.weight"), (hidden,)),
                    f"d_l{j}_post_norm": bf16(
                        layer(j, "post_attention_layernorm.weight"), (hidden,)
                    ),
                    f"d_l{j}_q_kv_a": torch.cat(
                        (bf16(layer(j, "self_attn.q_a_proj.weight"), (q_rank, hidden)), kv_a[j])
                    ),
                    f"d_l{j}_q_a_norm": bf16(layer(j, "self_attn.q_a_layernorm.weight"), (q_rank,)),
                    f"d_l{j}_q_b": bf16(
                        layer(j, "self_attn.q_b_proj.weight"),
                        (heads * (nope + rope), q_rank),
                        **heads_cut,
                    ),
                    f"d_l{j}_kv_b": torch.cat(
                        (kv_b[:, :nope].reshape(-1, latent), kv_b[:, nope:].reshape(-1, latent))
                    ),
                    f"d_l{j}_o": bf16(
                        layer(j, "self_attn.o_proj.weight"),
                        (hidden, heads * value),
                        shard_dim=1,
                        partition=share,
                    ),
                    f"d_l{j}_gate_up": torch.cat((gate, up)),
                    f"d_l{j}_down": bf16(
                        layer(j, "mlp.down_proj.weight"),
                        (hidden, inter),
                        shard_dim=1,
                        partition=share,
                    ),
                    f"d_l{j}_attn_conv_base": bf16(
                        layer(j, "attention_conv.base_kernel"), (2, taps, hidden)
                    ),
                    f"d_l{j}_attn_conv_proj": bf16(
                        layer(j, "attention_conv.kernel_projection.weight"),
                        (2 * taps * groups, hidden),
                    ),
                    f"d_l{j}_mlp_conv_base": bf16(
                        layer(j, "mlp_conv.base_kernel"), (2, taps, hidden)
                    ),
                    f"d_l{j}_mlp_conv_proj": bf16(
                        layer(j, "mlp_conv.kernel_projection.weight"), (2 * taps * groups, hidden)
                    ),
                }
            )
        unread = sorted(set(self.weights.keys()) - read)
        if unread:
            raise ValueError(
                f"DFlash2 drafter checkpoint keys this target does not read: {unread[:8]}"
            )
        return w
