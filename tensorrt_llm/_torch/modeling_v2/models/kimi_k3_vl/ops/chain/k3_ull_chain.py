# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Helix16 ULL MR8 chain and its rank-local checkpoint views."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Iterable
from collections.abc import Mapping as AbcMapping
from functools import partial
from types import SimpleNamespace as NS
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

CHAIN_HIDDEN = 7168
HEADS = 96
HEAD_DIM = 128
CONV = 4
EXPERTS = 896
TOP_K = 16
LATENT = 3584
SHARED_INTER = 6144  # moe_intermediate_size 3072 x 2 shared experts
NUM_LAYERS = 93
TP = 16
ATTN_RES_BLOCK = 12
# Each rank's in-projection rows are padded to a multiple of 16 (the tuned tables are keyed by the
# 3216-row pitch).
INPROJ_ALIGN = 16
# The front packs the rank's router rows padded to a multiple of 16 before its latent shard.
FRONT_ROW_ALIGN = 16
CHAIN_FP8_BLOCK = 128
SLICE = 192  # routed expert intermediate channels per rank (3072 / 16)
SF_VEC = 16

# Model scalars the chain's kernels take; the checkpoint config's values are used
# (:func:`chain_config`), these are the Kimi K3 values.
RMS_NORM_EPS = 1e-5
GATE_LOWER_BOUND = -5.0
ROUTED_SCALING_FACTOR = 1.0
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

_ROUTED_KEYS = ("w13", "w13_sf", "w2", "w2_sf", "alphas", "x_gsf", "h_gsf")


# ---------------------------------------------------------------------------------------------
# Layer schedule
# ---------------------------------------------------------------------------------------------


def is_kda_layer(layer: int) -> bool:
    return (layer + 1) % 4 != 0 and layer <= 90


def kda_moe_layers(num_layers: int = NUM_LAYERS) -> Tuple[int, ...]:
    """The KDA + MoE layers: 1, 2, 4, 5, 6, ..., 89, 90 (68 layers)."""
    return tuple(i for i in range(1, num_layers) if is_kda_layer(i))


def mla_moe_layers(num_layers: int = NUM_LAYERS) -> Tuple[int, ...]:
    """The MLA + MoE layers: 3, 7, ..., 87, 91, 92 (24 layers)."""
    return tuple(i for i in range(1, num_layers) if not is_kda_layer(i))


def inproj_rows(tp: int = TP) -> Tuple[int, int]:
    """(rows, padded pitch) of one rank's KDA in-projection."""
    d = HEADS // tp * HEAD_DIM
    rows = 4 * d + HEAD_DIM + HEADS // tp
    return rows, rows + (-rows % INPROJ_ALIGN)


# ---------------------------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------------------------


def _check(name: str, value: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Kimi K3 chain weights: {name} is not a tensor")
    if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError(
            f"Kimi K3 chain weights: {name} is {value.dtype} {tuple(value.shape)}, "
            f"expected {dtype} {tuple(shape)}"
        )
    if not value.is_contiguous():
        raise ValueError(f"Kimi K3 chain weights: {name} must be contiguous")


@dataclasses.dataclass(frozen=True)
class RmsNormWeights:
    """An RMSNorm's weight [n] BF16 and epsilon (``variance_epsilon`` is the name the chain's
    kernels read)."""

    weight: torch.Tensor
    eps: float

    @property
    def variance_epsilon(self) -> float:
        return self.eps


@dataclasses.dataclass(frozen=True)
class ProjWeights:
    """An AttnRes query projection: ``weight`` [1, 7168] BF16."""

    weight: torch.Tensor


@dataclasses.dataclass(frozen=True)
class AttnResWeights:
    """One layer's AttnRes and RMSNorm vectors, and its AttnRes block geometry (block size 12).

    The pre-attention site reads ``self_attention_res_*`` and ``input_layernorm``; the
    post-attention one ``mlp_res_*`` and ``post_attention_layernorm``."""

    layer: int
    self_attention_res_norm: RmsNormWeights
    self_attention_res_proj: ProjWeights
    input_layernorm: RmsNormWeights
    post_attention_layernorm: RmsNormWeights
    mlp_res_norm: RmsNormWeights
    mlp_res_proj: ProjWeights

    def __post_init__(self) -> None:
        for name in ("self_attention_res_norm", "input_layernorm", "post_attention_layernorm",
                     "mlp_res_norm"):  # fmt: skip
            _check(
                f"L{self.layer} {name}", getattr(self, name).weight, (CHAIN_HIDDEN,), torch.bfloat16
            )
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            _check(
                f"L{self.layer} {name}",
                getattr(self, name).weight,
                (1, CHAIN_HIDDEN),
                torch.bfloat16,
            )

    @property
    def layer_idx(self) -> int:
        return self.layer

    @property
    def is_block_write_layer(self) -> bool:
        return self.layer % ATTN_RES_BLOCK == 0

    @property
    def block_write_idx(self) -> int:
        return self.layer // ATTN_RES_BLOCK

    @property
    def prev_valid_blocks(self) -> int:
        return -(-self.layer // ATTN_RES_BLOCK)

    def tensors(self) -> Dict[str, torch.Tensor]:
        return {
            f.name: getattr(self, f.name).weight
            for f in dataclasses.fields(self)
            if f.name != "layer"
        }


@dataclasses.dataclass(frozen=True)
class OutputWeights:
    """The output AttnRes and the final norm (``num_attn_res_blocks``: the bank rows they mix)."""

    output_attn_res_norm: RmsNormWeights
    output_attn_res_proj: ProjWeights
    norm: RmsNormWeights
    num_attn_res_blocks: int

    def __post_init__(self) -> None:
        _check(
            "output_attn_res_norm",
            self.output_attn_res_norm.weight,
            (CHAIN_HIDDEN,),
            torch.bfloat16,
        )
        _check(
            "output_attn_res_proj",
            self.output_attn_res_proj.weight,
            (1, CHAIN_HIDDEN),
            torch.bfloat16,
        )
        _check("norm", self.norm.weight, (CHAIN_HIDDEN,), torch.bfloat16)

    def tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "output_attn_res_norm": self.output_attn_res_norm.weight,
            "output_attn_res_proj": self.output_attn_res_proj.weight,
            "norm": self.norm.weight,
        }


@dataclasses.dataclass(frozen=True)
class KdaWeights:
    """One KDA layer's attention weights on one TP16 rank (module docstring); ``o_proj`` None for
    a KDA + dense layer (its o-projection is not on the chain)."""

    layer: int
    tp: int
    rank: int
    in_proj: torch.Tensor
    f_b: torch.Tensor
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    o_norm: torch.Tensor
    o_proj: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        heads = HEADS // self.tp
        d = heads * HEAD_DIM
        _, pitch = inproj_rows(self.tp)
        where = f"L{self.layer} KDA"
        _check(f"{where} in_proj", self.in_proj, (pitch, CHAIN_HIDDEN), torch.bfloat16)
        _check(f"{where} f_b", self.f_b, (d, HEAD_DIM), torch.bfloat16)
        _check(f"{where} conv", self.conv, (3 * d, 1, CONV), torch.float32)
        _check(f"{where} a_log", self.a_log, (heads,), torch.float32)
        _check(f"{where} dt_bias", self.dt_bias, (d,), torch.float32)
        _check(f"{where} o_norm", self.o_norm, (HEAD_DIM,), torch.float32)
        if self.o_proj is not None:
            _check(f"{where} o_proj", self.o_proj, (CHAIN_HIDDEN, d), torch.bfloat16)

    def tensors(self) -> Dict[str, torch.Tensor]:
        names = ("in_proj", "f_b", "conv", "a_log", "dt_bias", "o_norm", "o_proj")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}


@dataclasses.dataclass(frozen=True)
class MoeWeights:
    """One MoE layer's weights on one TP16 rank (module docstring)."""

    layer: int
    tp: int
    rank: int
    front: torch.Tensor
    score_bias: torch.Tensor
    latent_up: torch.Tensor
    latent_norm: torch.Tensor
    shared_gate_up: torch.Tensor
    shared_down: torch.Tensor
    routed: Dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        valid, shard = EXPERTS // self.tp, LATENT // self.tp
        inter = SHARED_INTER // self.tp
        rows = valid + (-valid % FRONT_ROW_ALIGN) + shard
        where = f"L{self.layer} MoE"
        _check(f"{where} front", self.front, (rows, CHAIN_HIDDEN), torch.bfloat16)
        _check(f"{where} score_bias", self.score_bias, (EXPERTS,), torch.float32)
        _check(f"{where} latent_up", self.latent_up, (CHAIN_HIDDEN, LATENT), torch.bfloat16)
        _check(f"{where} latent_norm", self.latent_norm, (LATENT,), torch.bfloat16)
        _check(
            f"{where} shared_gate_up",
            self.shared_gate_up,
            (2 * inter, CHAIN_HIDDEN),
            torch.bfloat16,
        )
        _check(f"{where} shared_down", self.shared_down, (CHAIN_HIDDEN, inter), torch.bfloat16)
        check_routed(self.routed, where)

    @property
    def up_shard(self) -> torch.Tensor:
        """The rank's rows of the latent up projection [448, 3584] (a view)."""
        rows = CHAIN_HIDDEN // self.tp
        return self.latent_up[self.rank * rows : (self.rank + 1) * rows]

    def tensors(self) -> Dict[str, torch.Tensor]:
        names = ("front", "score_bias", "latent_up", "latent_norm", "shared_gate_up", "shared_down")
        out = {n: getattr(self, n) for n in names}
        out.update({f"routed.{k}": v for k, v in self.routed.items()})
        return out


def check_routed(routed: AbcMapping[str, torch.Tensor], where: str = "routed experts") -> None:
    """The keys, shapes and dtypes :func:`pack_routed_experts_tp16` returns."""
    missing = [k for k in _ROUTED_KEYS if k not in routed]
    if missing:
        raise ValueError(f"Kimi K3 chain weights: {where} lacks {missing}")
    e = routed["w13"].shape[0]
    _check(f"{where} w13", routed["w13"], (e, 2 * SLICE, LATENT // 2), torch.uint8)
    _check(f"{where} w13_sf", routed["w13_sf"], (e, 2 * SLICE * LATENT // SF_VEC), torch.uint8)
    _check(f"{where} w2", routed["w2"], (e, LATENT, SLICE // 2), torch.uint8)
    _check(f"{where} w2_sf", routed["w2_sf"], (e, LATENT * SLICE // SF_VEC), torch.uint8)
    _check(f"{where} alphas", routed["alphas"], (e + 1, 2), torch.float32)
    _check(f"{where} x_gsf", routed["x_gsf"], (1,), torch.float32)
    _check(f"{where} h_gsf", routed["h_gsf"], (1,), torch.float32)


@dataclasses.dataclass(frozen=True)
class ChainLayerWeights:
    """One model layer's weights the chain owns: AttnRes / norm vectors (every layer), the KDA
    weights (KDA layers), the MoE weights (MoE layers)."""

    layer: int
    attn_res: AttnResWeights
    kda: Optional[KdaWeights] = None
    moe: Optional[MoeWeights] = None


@dataclasses.dataclass(frozen=True)
class ChainConfig:
    """The model scalars the chain's kernels take."""

    rms_norm_eps: float = RMS_NORM_EPS
    gate_lower_bound: float = GATE_LOWER_BOUND
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR
    situ_beta: float = SITU_BETA
    situ_linear_beta: float = SITU_LINEAR_BETA
    num_layers: int = NUM_LAYERS


@dataclasses.dataclass(frozen=True)
class ChainWeights:
    """Everything the chain reads, for TP16 rank ``rank``: ``layers`` (model layer -> weights;
    a subset only in tests), ``output``, ``config``."""

    tp: int
    rank: int
    layers: Dict[int, ChainLayerWeights]
    output: OutputWeights
    config: ChainConfig

    def nbytes(self) -> int:
        """Bytes of distinct storages (views counted once)."""
        seen, total = set(), 0
        for t in self.tensors():
            key = t.untyped_storage().data_ptr()
            if key not in seen:
                seen.add(key)
                total += t.untyped_storage().nbytes()
        return total

    def tensors(self) -> Iterable[torch.Tensor]:
        yield from self.output.tensors().values()
        for w in self.layers.values():
            yield from w.attn_res.tensors().values()
            for part in (w.kda, w.moe):
                if part is not None:
                    yield from part.tensors().values()


# ---------------------------------------------------------------------------------------------
# Checkpoint reads (lazy safetensors slices or tensors)
# ---------------------------------------------------------------------------------------------


def _shape(value) -> Tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return tuple(value.get_shape())


def _read(value, rows: Optional[slice] = None, cols: Optional[slice] = None) -> torch.Tensor:
    """``value[rows, cols]`` materialized (a lazy slice reads only that block)."""
    if rows is None and cols is None:
        if isinstance(value, torch.Tensor):
            return value
        if len(_shape(value)) == 0:
            return value[()]
        return value[:]
    rows = rows if rows is not None else slice(None)
    return value[rows] if cols is None else value[rows, cols]


def _scale_key(weights: AbcMapping[str, Any], key: str) -> str:
    for name in (key + "_scale", key + "_scale_inv"):
        if name in weights:
            return name
    raise KeyError(f"Kimi K3 chain: {key} is FP8 E4M3 but has no {key}_scale(_inv)")


def _span(s: slice, n: int) -> Tuple[int, int]:
    start, stop, step = s.indices(n)
    if step != 1:
        raise ValueError("Kimi K3 chain: strided checkpoint slices are not supported")
    return start, stop


def dequant_block(
    codes: torch.Tensor, scale: torch.Tensor, row0: int = 0, col0: int = 0
) -> torch.Tensor:
    """BF16 of FP8 ``codes`` times their FP32 block ``scale`` (one value per 128 x 128 block).

    ``codes`` is the block ``[row0:row0 + R, col0:col0 + C]`` of the checkpoint matrix and
    ``scale`` the rows / columns of the checkpoint's scale grid from ``row0 // 128`` /
    ``col0 // 128`` on. The product is formed in FP32 and rounded once to BF16.
    """
    if codes.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Kimi K3 chain: expected FP8 E4M3 codes, got {codes.dtype}")
    r, c = codes.shape
    scale = scale.float()
    if scale.dim() == 4:  # [N / 128, 1, K / 128, 1] as stored
        scale = scale.reshape(scale.shape[0], scale.shape[2])
    if scale.numel() == 1:
        return (codes.float() * scale.reshape(())).to(torch.bfloat16)
    lr, lc = row0 % CHAIN_FP8_BLOCK, col0 % CHAIN_FP8_BLOCK
    br, bc = math.ceil((lr + r) / CHAIN_FP8_BLOCK), math.ceil((lc + c) / CHAIN_FP8_BLOCK)
    if scale.shape[0] < br or scale.shape[1] < bc:
        raise ValueError(
            f"Kimi K3 chain: scale {tuple(scale.shape)} does not cover codes {tuple(codes.shape)} "
            f"at ({row0}, {col0})"
        )
    grid = scale[:br, :bc].to(codes.device)
    grid = grid.repeat_interleave(CHAIN_FP8_BLOCK, 0).repeat_interleave(CHAIN_FP8_BLOCK, 1)
    return (codes.float() * grid[lr : lr + r, lc : lc + c]).to(torch.bfloat16)


def read_block(
    weights: AbcMapping[str, Any],
    key: str,
    rows: Optional[slice] = None,
    cols: Optional[slice] = None,
    device=None,
) -> torch.Tensor:
    """The BF16 block ``[rows, cols]`` of checkpoint matrix ``key`` (FP8 + scale dequantized, BF16
    as is), on ``device``."""
    source = weights[key]
    shape = _shape(source)
    codes = _read(source, rows, cols).to(device)
    if codes.dtype == torch.bfloat16:
        return codes
    if codes.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Kimi K3 chain: {key} is {codes.dtype}, expected BF16 or FP8 E4M3")
    r0, _ = _span(rows or slice(None), shape[0])
    c0, _ = _span(cols or slice(None), shape[1]) if len(shape) > 1 else (0, 1)
    grid = _read(weights[_scale_key(weights, key)]).float()
    if grid.dim() == 4:
        grid = grid.reshape(grid.shape[0], grid.shape[2])
    if grid.numel() > 1:
        if tuple(grid.shape) != tuple(math.ceil(d / CHAIN_FP8_BLOCK) for d in shape):
            raise ValueError(f"Kimi K3 chain: {key} scale {tuple(grid.shape)} vs codes {shape}")
        grid = grid[r0 // CHAIN_FP8_BLOCK :, c0 // CHAIN_FP8_BLOCK :]
    return dequant_block(codes, grid, r0, c0)


def checkpoint_prefix(weights: AbcMapping[str, Any]) -> str:
    """``language_model.`` when the checkpoint's text-model keys carry it, else ``""``."""
    for key in weights.keys():
        if key.startswith("language_model."):
            return "language_model."
    return ""


class LazyCheckpoint(AbcMapping):
    """A read-only ``key -> safetensors slice`` mapping over a sharded checkpoint directory (no
    tensor read up front)."""

    def __init__(self, root: str):
        from safetensors import safe_open

        self._open = safe_open
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self._handles: Dict[str, Any] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = self._open(os.path.join(self.root, shard), framework="pt")
        return self._handles[shard]

    def __getitem__(self, key: str):
        return self._handle(self.weight_map[key]).get_slice(key)

    def __iter__(self):
        return iter(self.weight_map)

    def __len__(self) -> int:
        return len(self.weight_map)

    def __contains__(self, key) -> bool:
        return key in self.weight_map

    def config(self) -> dict:
        with open(os.path.join(self.root, "config.json")) as f:
            cfg = json.load(f)
        return cfg.get("text_config", cfg)


# ---------------------------------------------------------------------------------------------
# One layer
# ---------------------------------------------------------------------------------------------


def read_kda(
    weights: AbcMapping[str, Any],
    layer: int,
    rank: int,
    *,
    tp: int = TP,
    prefix: str = "",
    device=None,
    o_proj: bool = True,
) -> KdaWeights:
    """KDA ``layer``'s weights on TP rank ``rank`` (module docstring); ``o_proj``: also the rank's
    o-projection columns."""
    if HEADS % tp or CHAIN_HIDDEN % tp:
        raise ValueError(f"Kimi K3 chain: TP {tp} does not divide the K3 shapes")
    attn = f"{prefix}model.layers.{layer}.self_attn."
    heads = HEADS // tp
    d = heads * HEAD_DIM
    hs = slice(rank * d, (rank + 1) * d)  # the rank's head-major rows / columns
    rows, pitch = inproj_rows(tp)

    def get(key, rows_=None, cols_=None):
        return read_block(weights, key, rows_, cols_, device)

    def get_plain(key, rows_=None) -> torch.Tensor:
        return _read(weights[key], rows_).to(device)

    in_proj = torch.zeros((pitch, CHAIN_HIDDEN), dtype=torch.bfloat16, device=device)
    offset = 0
    for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
        in_proj[offset : offset + d] = get(attn + name + ".weight", hs)
        offset += d
    in_proj[offset : offset + HEAD_DIM] = get(attn + "f_a_proj.weight")
    offset += HEAD_DIM
    beta = slice(rank * heads, (rank + 1) * heads)
    in_proj[offset : offset + heads] = get(attn + "b_proj.weight", beta)
    offset += heads
    if offset != rows:
        raise AssertionError(f"Kimi K3 chain: in-projection rows {offset} != {rows}")
    f_b = get(attn + "f_b_proj.weight", hs).contiguous()
    conv = torch.cat(
        [get_plain(attn + f"{x}_conv1d.weight", hs).float() for x in ("q", "k", "v")]
    ).reshape(3 * d, 1, CONV)
    a_log = get_plain(attn + "A_log").float().reshape(-1)[beta].clone()
    dt_bias = get_plain(attn + "dt_bias", hs).float().contiguous()
    o_norm = get_plain(attn + "o_norm.weight").float().contiguous()
    return KdaWeights(
        layer=layer,
        tp=tp,
        rank=rank,
        in_proj=in_proj,
        f_b=f_b,
        conv=conv,
        a_log=a_log,
        dt_bias=dt_bias,
        o_norm=o_norm,
        o_proj=get(attn + "o_proj.weight", None, hs).contiguous() if o_proj else None,
    )


def read_moe(
    weights: AbcMapping[str, Any],
    layer: int,
    rank: int,
    *,
    tp: int = TP,
    prefix: str = "",
    device=None,
) -> MoeWeights:
    """MoE ``layer``'s weights on TP rank ``rank`` (module docstring): only the rank's router and
    latent-down rows are read (into ``front``), the latent up projection whole."""
    if SHARED_INTER % tp or EXPERTS % tp or LATENT % tp or CHAIN_HIDDEN % tp:
        raise ValueError(f"Kimi K3 chain: TP {tp} does not divide the K3 shapes")
    moe = f"{prefix}model.layers.{layer}.block_sparse_moe."

    def get(key, rows_=None, cols_=None):
        return read_block(weights, key, rows_, cols_, device)

    def get_plain(key, rows_=None) -> torch.Tensor:
        return _read(weights[key], rows_).to(device)

    valid, shard = EXPERTS // tp, LATENT // tp
    router = get(moe + "gate.weight", slice(rank * valid, (rank + 1) * valid))
    down = get(moe + "routed_expert_down_proj.weight", slice(rank * shard, (rank + 1) * shard))
    pad = -valid % FRONT_ROW_ALIGN
    front = torch.cat((router, router.new_zeros((pad, CHAIN_HIDDEN)), down)).contiguous()
    del router, down
    score_bias = get_plain(moe + "gate.e_score_correction_bias").float().contiguous()
    latent_up = get(moe + "routed_expert_up_proj.weight").contiguous()
    latent_norm = get_plain(moe + "routed_expert_norm.weight").to(torch.bfloat16).contiguous()
    inter = SHARED_INTER // tp
    ss = slice(rank * inter, (rank + 1) * inter)
    shared_gate_up = torch.cat(
        (
            get(moe + "shared_experts.gate_proj.weight", ss),
            get(moe + "shared_experts.up_proj.weight", ss),
        )
    ).contiguous()
    shared_down = get(moe + "shared_experts.down_proj.weight", None, ss).contiguous()
    routed = load_routed_experts_tp16(weights, layer, rank, tp=tp, prefix=prefix, device=device)
    return MoeWeights(
        layer=layer,
        tp=tp,
        rank=rank,
        front=front,
        score_bias=score_bias,
        latent_up=latent_up,
        latent_norm=latent_norm,
        shared_gate_up=shared_gate_up,
        shared_down=shared_down,
        routed=routed,
    )


def _vector(weights: AbcMapping[str, Any], key: str, device) -> torch.Tensor:
    return _read(weights[key]).to(device, torch.bfloat16).contiguous()


def read_attn_res(
    weights: AbcMapping[str, Any], layer: int, *, prefix: str = "", device=None, eps: float
) -> AttnResWeights:
    """Layer ``layer``'s AttnRes and RMSNorm vectors (BF16)."""
    base = f"{prefix}model.layers.{layer}."

    def norm(name):
        return RmsNormWeights(_vector(weights, base + name + ".weight", device), float(eps))

    def proj(name):
        return ProjWeights(_vector(weights, base + name + ".weight", device).reshape(1, -1))

    return AttnResWeights(
        layer=layer,
        self_attention_res_norm=norm("self_attention_res_norm"),
        self_attention_res_proj=proj("self_attention_res_proj"),
        input_layernorm=norm("input_layernorm"),
        post_attention_layernorm=norm("post_attention_layernorm"),
        mlp_res_norm=norm("mlp_res_norm"),
        mlp_res_proj=proj("mlp_res_proj"),
    )


def read_output(
    weights: AbcMapping[str, Any],
    *,
    prefix: str = "",
    device=None,
    eps: float,
    num_layers: int = NUM_LAYERS,
) -> OutputWeights:
    """The output AttnRes vectors and the final norm (BF16)."""
    base = f"{prefix}model."

    def norm(name):
        return RmsNormWeights(_vector(weights, base + name + ".weight", device), float(eps))

    proj = _vector(weights, base + "output_attn_res_proj.weight", device).reshape(1, -1)
    return OutputWeights(
        output_attn_res_norm=norm("output_attn_res_norm"),
        output_attn_res_proj=ProjWeights(proj),
        norm=norm("norm"),
        num_attn_res_blocks=-(-num_layers // ATTN_RES_BLOCK),
    )


# ---------------------------------------------------------------------------------------------
# Routed experts (NVFP4, TP16 Rubin experts kernels)
# ---------------------------------------------------------------------------------------------


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


def _unatoms(packed: torch.Tensor, tiles: int, kblocks: int) -> torch.Tensor:
    """Inverse of :func:`_atoms`: row-major scales [E, 128 * tiles, 4 * kblocks]."""
    experts = packed.shape[0]
    s = packed.reshape(experts, tiles, kblocks, 32, 4, 4)  # e, tile, kb, m % 32, m // 32, k
    return s.permute(0, 1, 4, 3, 2, 5).reshape(experts, 128 * tiles, 4 * kblocks).contiguous()


def _requantize(scales: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """UE4M3 block scales times a per-expert ``ratio`` (<= 1), rounded back to UE4M3."""
    value = scales.view(torch.float8_e4m3fn).float() * ratio.view(-1, 1, 1)
    return value.to(torch.float8_e4m3fn).view(torch.uint8)


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


@torch.no_grad()
def pack_routed_experts_tp16(
    w1, w3, w2, w1_sf, w3_sf, w2_sf, w1_scale2, w3_scale2, w2_scale2, fc1_input_scale,
    fc2_input_scale, *, publish: bool = True,
):  # fmt: skip
    """One rank's TP16 routed experts in the layout of ``kernels/k3r_moe/ull/nv_experts``'
    ``RubinNvExpertsKernel`` / ``RubinNvGroupedExpertsKernel``.

    Inputs (the checkpoint's NVFP4 tensors of every expert, sliced for this rank; any device):
      w1 / w3   gate / up rows of this rank's 192 intermediate channels, uint8 [E, 192, 1792]
                (packed E2M1, element 2 i in the low nibble)
      w2        this rank's 192 K columns of the down projection, uint8 [E, 3584, 96]
      *_sf      row-major UE4M3 block scales (one per 16 elements): [E, 192, 224], [E, 3584, 12]
                (float8_e4m3fn or uint8)
      *_scale2  per-expert global dequant scales (``weight_scale_2``), FP32 [E]
      fc1_input_scale / fc2_input_scale  the activations' global dequant scales (``input_scale``;
                one for every expert: a tensor of E values uses its maximum)
      publish   the w2 row order of the published staged rows (True) or of direct rows

    Returns a dict: ``w13`` uint8 [E, 384, 1792], ``w13_sf`` uint8 [E, 86016], ``w2`` uint8
    [E, 3584, 96], ``w2_sf`` uint8 [E, 43008], ``alphas`` FP32 [E + 1, 2] ((alpha1, alpha2) per
    expert, then (h_gsf, 0)), ``x_gsf`` FP32 [1] (the latent image's quantization scale, which the
    routing kernel needs), ``h_gsf`` FP32 [1]. w1 / w3 with different global scales share the larger
    one; the smaller side's block scales are rescaled (UE4M3 rounding).
    """
    experts = w1.shape[0]
    if w1.shape != w3.shape or tuple(w1.shape) != (experts, SLICE, LATENT // 2):
        raise ValueError(f"Kimi K3 routed experts: w1 / w3 {tuple(w1.shape)} / {tuple(w3.shape)}")
    if tuple(w2.shape) != (experts, LATENT, SLICE // 2):
        raise ValueError(f"Kimi K3 routed experts: w2 {tuple(w2.shape)}")
    w1_sf, w3_sf, w2_sf = (_as_u8(t) for t in (w1_sf, w3_sf, w2_sf))
    if w1_sf.shape != w3_sf.shape or tuple(w1_sf.shape) != (experts, SLICE, LATENT // SF_VEC):
        raise ValueError(f"Kimi K3 routed experts: w1 / w3 scales {tuple(w1_sf.shape)}")
    if tuple(w2_sf.shape) != (experts, LATENT, SLICE // SF_VEC):
        raise ValueError(f"Kimi K3 routed experts: w2 scales {tuple(w2_sf.shape)}")
    s1, s3, s2 = (t.float().reshape(experts) for t in (w1_scale2, w3_scale2, w2_scale2))
    s13 = torch.maximum(s1, s3)
    if not torch.equal(s1, s3):
        w1_sf = _requantize(w1_sf, s1 / s13)
        w3_sf = _requantize(w3_sf, s3 / s13)
    in1 = torch.as_tensor(fc1_input_scale).float().max()
    in2 = torch.as_tensor(fc2_input_scale).float().max()

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


@torch.no_grad()
def repack_w2(
    packed: AbcMapping[str, torch.Tensor], columns: torch.Tensor, *, publish: bool = True
):
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


def direct_columns() -> torch.Tensor:
    """w2 row order under which the non-published rows (publish False) come out in column order:
    ``d[p^-1[d]]``, ``d`` / ``p`` = :func:`w2_row_order` ``(False / True)``."""
    d, p = w2_row_order(False), w2_row_order(True)
    inverse = torch.empty_like(p)
    inverse[p] = torch.arange(p.numel())
    return d[inverse[d]]


def _scalar(value) -> torch.Tensor:
    """A checkpoint scalar (a tensor or a lazy slice of shape ``[]`` / ``[1]``) as FP32 [1]."""
    if not isinstance(value, torch.Tensor):
        value = value[()] if len(_shape(value)) == 0 else value[:]
    return torch.as_tensor(value).float().reshape(-1)[:1]


@torch.no_grad()
def read_routed_experts_tp16(
    weights: AbcMapping[str, Any], layer: int, rank: int, *, tp: int = TP, prefix: str = "",
    device=None, experts: int = EXPERTS,
) -> Dict[str, torch.Tensor]:  # fmt: skip
    """TP rank ``rank``'s slices of MoE ``layer``'s routed experts, read lazily from the checkpoint
    ``weights`` (only the rank's rows / columns are read), as the keyword arguments of
    :func:`pack_routed_experts_tp16`.

    Checkpoint (modelopt NVFP4, one tensor set per expert ``e``):
    ``{prefix}model.layers.{layer}.block_sparse_moe.experts.{e}.w{1,2,3}.{weight, weight_scale,
    weight_scale_2, input_scale}``; w1 (gate) / w3 (up) ``weight`` U8 [3072, 1792] and
    ``weight_scale`` E4M3 [3072, 224] are cut by rows (the rank's 192 intermediate channels), w2
    ``weight`` U8 [3584, 1536] / ``weight_scale`` E4M3 [3584, 192] by columns (its 96 bytes / 12
    scales); ``weight_scale_2`` / ``input_scale`` are FP32 scalars per expert (the activations'
    ``input_scale`` of every expert is passed; the packer uses their maximum).
    """
    if tp != TP:  # SLICE = 3072 / 16 channels per rank
        raise ValueError(f"Kimi K3 routed experts: the NVFP4 kernels take TP {TP}, not {tp}")
    rows = slice(rank * SLICE, (rank + 1) * SLICE)
    w2_cols = slice(rank * SLICE // 2, (rank + 1) * SLICE // 2)
    sf_cols = slice(rank * SLICE // SF_VEC, (rank + 1) * SLICE // SF_VEC)
    base = f"{prefix}model.layers.{layer}.block_sparse_moe.experts."
    u8, e4m3 = torch.uint8, torch.float8_e4m3fn
    out = {
        "w1": torch.empty(experts, SLICE, LATENT // 2, dtype=u8),
        "w3": torch.empty(experts, SLICE, LATENT // 2, dtype=u8),
        "w2": torch.empty(experts, LATENT, SLICE // 2, dtype=u8),
        "w1_sf": torch.empty(experts, SLICE, LATENT // SF_VEC, dtype=u8),
        "w3_sf": torch.empty(experts, SLICE, LATENT // SF_VEC, dtype=u8),
        "w2_sf": torch.empty(experts, LATENT, SLICE // SF_VEC, dtype=u8),
    }
    names = ("w1_scale2", "w3_scale2", "w2_scale2", "in1", "in3", "in2")
    scalars = {k: torch.empty(experts) for k in names}

    def u8_of(t: torch.Tensor) -> torch.Tensor:
        return t.view(u8) if t.dtype in (u8, e4m3) else t.to(e4m3).view(u8)

    for e in range(experts):
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
    result["fc1_input_scale"] = torch.maximum(scalars["in1"], scalars["in3"]).to(device)
    result["fc2_input_scale"] = scalars["in2"].to(device)
    return result


def load_routed_experts_tp16(
    weights: AbcMapping[str, Any], layer: int, rank: int, *, tp: int = TP, prefix: Optional[str] = None,
    device=None, publish: bool = True,
) -> Dict[str, torch.Tensor]:  # fmt: skip
    """:func:`pack_routed_experts_tp16` of :func:`read_routed_experts_tp16` (``prefix`` None: the
    checkpoint's, :func:`checkpoint_prefix`)."""
    prefix = checkpoint_prefix(weights) if prefix is None else prefix
    raw = read_routed_experts_tp16(weights, layer, rank, tp=tp, prefix=prefix, device=device)
    return pack_routed_experts_tp16(**raw, publish=publish)


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------


def _text_config(config):
    """The text config of a model config / HF config / config dict (a VLM's ``text_config``)."""
    config = getattr(config, "pretrained_config", config)
    if isinstance(config, AbcMapping):
        return config.get("text_config", config)
    if not hasattr(config, "linear_attn_config") and hasattr(config, "text_config"):
        return config.text_config
    return config


def _field(config, name: str, default):
    value = config.get(name) if isinstance(config, AbcMapping) else getattr(config, name, None)
    return default if value is None else value


def chain_config(model_config) -> ChainConfig:
    """The chain's scalars from the model config (an engine ``ModelConfig``, an HF config or the
    checkpoint's ``config.json`` dict); a missing field takes the Kimi K3 value."""
    cfg = _text_config(model_config)
    lin = _field(cfg, "linear_attn_config", {}) or {}
    lin_get = lin.get if isinstance(lin, AbcMapping) else (lambda k, d=None: getattr(lin, k, d))
    lower = lin_get("gate_lower_bound", None)
    return ChainConfig(
        rms_norm_eps=float(_field(cfg, "rms_norm_eps", RMS_NORM_EPS)),
        gate_lower_bound=float(GATE_LOWER_BOUND if lower is None else lower),
        routed_scaling_factor=float(_field(cfg, "routed_scaling_factor", ROUTED_SCALING_FACTOR)),
        situ_beta=float(_field(cfg, "activation_situ_beta", SITU_BETA)),
        situ_linear_beta=float(_field(cfg, "activation_situ_linear_beta", SITU_LINEAR_BETA)),
        num_layers=int(_field(cfg, "num_hidden_layers", NUM_LAYERS)),
    )


def load_chain_weights(
    weights: AbcMapping[str, Any],
    model_config,
    rank: int,
    device,
    *,
    tp: int = TP,
    layers: Optional[Iterable[int]] = None,
) -> ChainWeights:
    """Every weight the chain reads, for TP16 rank ``rank``, from the checkpoint mapping
    ``weights`` (``key -> tensor`` or a lazy safetensors slice), on ``device``:

    * every layer's AttnRes / norm vectors and the output AttnRes + final norm;
    * the KDA weights of every KDA layer (the o-projection of the KDA + MoE layers only: layer 0's
      is not on the chain);
    * the MoE weights (dense and NVFP4 routed experts) of every MoE layer (1 ..).

    ``layers``: only those model layers (tests; default all ``num_hidden_layers``)."""
    config = chain_config(model_config)
    if config.num_layers != NUM_LAYERS:
        raise ValueError(
            f"Kimi K3 chain: {config.num_layers} layers; the chain is built for {NUM_LAYERS}"
        )
    if not 0 <= rank < tp:
        raise ValueError(f"Kimi K3 chain: rank {rank} of TP {tp}")
    prefix = checkpoint_prefix(weights)
    eps = config.rms_norm_eps
    chosen = range(config.num_layers) if layers is None else sorted(set(layers))
    out: Dict[int, ChainLayerWeights] = {}
    for i in chosen:
        if not 0 <= i < config.num_layers:
            raise ValueError(f"Kimi K3 chain: no layer {i}")
        kda = None
        if is_kda_layer(i):
            kda = read_kda(weights, i, rank, tp=tp, prefix=prefix, device=device, o_proj=i > 0)
        moe = read_moe(weights, i, rank, tp=tp, prefix=prefix, device=device) if i > 0 else None
        attn_res = read_attn_res(weights, i, prefix=prefix, device=device, eps=eps)
        out[i] = ChainLayerWeights(layer=i, attn_res=attn_res, kda=kda, moe=moe)
    output = read_output(
        weights, prefix=prefix, device=device, eps=eps, num_layers=config.num_layers
    )
    return ChainWeights(tp=tp, rank=rank, layers=out, output=output, config=config)


def _runtime():
    from ..moe import k3_ull_nv_experts_runtime as nv_experts_runtime

    return nv_experts_runtime


def image_rows(batch: int) -> bool:
    """Rows on ``RubinNvExpertsKernel`` (the latent image); above: the grouped kernel (plan)."""
    return batch <= 8


def polled_rows(batch: int) -> bool:
    """Rows whose experts poll the route mailbox: the image holds every token row only at 8 (the
    routing kernel's image count reaches the experts' ``2 * 8 - 1`` only then)."""
    return batch == 8


class NvFront:
    """``SharedFront(nv=...)`` of one MoE layer."""

    def __init__(self, gsf: float, *, wide_routing: bool):
        nv = _runtime().nv
        self.writers = (nv.image_chunk_nv, nv.plan_chunk_nv)
        self.gsf = float(gsf)
        self.wide_routing = wide_routing

    @staticmethod
    def image(batch: int) -> bool:
        return image_rows(batch)

    @staticmethod
    def polled(batch: int) -> bool:
        return polled_rows(batch)


class NvScratch:
    """The MoE layers' shared experts workspace, route mailbox and plan buffer (one device)."""

    def __init__(self, device):
        from ...kernels.k3r_moe.ull import experts as mr8_experts

        rt = _runtime()
        self.device = torch.device(device)
        self.slots = rt.slots_for(self.device, polled=True)
        buffers = rt.scratch(self.device, self.slots)
        self.workspace = buffers["workspace"]
        self.routes = rt.route_mailbox(self.workspace, self.slots)
        # SharedFront compiles its routing for MR8's plan size; the NVFP4 image / plan are smaller
        # and start at the same base.
        size = max(mr8_experts.PLAN_BYTES, rt.nv.PLAN_BYTES)
        self.plan = torch.zeros(size, dtype=torch.uint8, device=device)
        self.image = self.plan[: rt.nv.IMAGE_BYTES]
        self.rows = self.plan[: rt.nv.PLAN_BYTES]

    def rearm(self):
        """The workspace as built (0xFF, hidden images 0x88): after a failed or aborted call."""
        _runtime().arm(self.workspace, self.slots)


class NvExperts:
    """One layer's NVFP4 routed experts.

    ``packed``: ``_chain_weights.pack_routed_experts_tp16(..., publish=True)``; ``scratch``: the
    shared :class:`NvScratch`. ``prepare`` compiles outside CUDA graph captures; ``run`` only
    launches."""

    def __init__(
        self, packed: Dict[str, torch.Tensor], scratch: NvScratch, *, beta, linear_beta, q=1,
        rows: Optional[Sequence[int]] = None,
    ):  # fmt: skip
        import cutlass

        from ...kernels.k3r_moe.ull import experts_runtime

        self.rt = rt = _runtime()
        self.er = experts_runtime
        self.packed, self.scratch, self.q = packed, scratch, q
        self.device = packed["w13"].device
        self.tp, self.experts = 16, packed["w13"].shape[0]
        self.grid, self.grids = self.er.grids(self.tp, self.device)
        self.dies = ()
        self.slots = scratch.slots
        self.x_gsf = float(packed["x_gsf"].reshape(-1)[0])
        self.scalars = tuple(map(cutlass.Float32, (beta * linear_beta, 1 / beta, 1 / linear_beta)))
        self.views = tuple(rt.to_cute(packed[k]) for k in ("alphas", "w13_sf", "w2_sf"))
        self.weights = (packed["w13"], packed["w13_sf"], packed["w2"], packed["w2_sf"])
        self.direct = None  # (w2, w2_sf) in the direct rows' order, for publish False
        self.rubin: set = set()  # rows on RubinNvExpertsKernel ("image8")
        self.prepared: Dict[int, dict] = {}
        # ``rows``: the row counts compiled at ``prepare``; others compile at their first ``run``,
        # outside CUDA graph captures.
        self.rows = None if rows is None else set(rows)
        self.deferred: Dict[int, tuple] = {}

    def grid_for(self, batch):
        return self.grids.get(batch, self.grid)

    def _w2(self, mode):
        if mode is not False:
            return self.packed["w2"], self.views[2]
        if self.direct is None:
            self.direct = repack_w2(self.packed, direct_columns())
            self.direct_sf = self.rt.to_cute(self.direct[1])
        return self.direct[0], self.direct_sf

    def prepare(self, batch, out, publish=(), planned=False, twoshot=False):
        """Compile a row shape, or record it for compilation on the first run.

        With ``planned`` the routing kernel writes the shared :class:`NvScratch` image or plan.
        Otherwise ``run`` writes it from its input before launching the experts.
        """
        if self.rows is not None and batch not in self.rows:
            self.deferred[batch] = (batch, out, publish, planned, twoshot)
            return
        self._prepare(batch, out, publish, planned, twoshot)

    def _prepare(self, batch, out, publish, planned, twoshot):
        import cutlass

        from ..common import k3_ull_cute_host as host

        rt = self.rt
        if not 1 <= batch <= 64:
            raise ValueError(f"NVFP4 routed experts: rows {batch} (1-64 only)")
        if twoshot and image_rows(batch):
            raise NotImplementedError(f"NVFP4 routed experts, rows {batch}: no two-shot publish")
        mode = "twoshot" if twoshot else "push" if len(publish) > 3 else bool(publish)
        grid, dev = self.grid_for(batch), self.device.index
        if image_rows(batch):
            poll = bool(planned) and polled_rows(batch)
            _, code = rt.compile_experts(
                self.experts, grid, self.slots, dev, mode, poll, True, True
            )
            plan, writer = self.scratch.image, rt.write_image
            self.rubin.add(batch)
        else:
            _, code = rt.compile_grouped(self.experts, batch, grid, self.slots, dev, mode, True)
            plan, writer = self.scratch.rows, rt.write_plan
        host.check_residency(code, dev, grid, 384, "Kimi K3 NVFP4 experts")
        w2, w2_sf = self._w2(mode)
        addresses = (cutlass.Int64(self.packed["w13"].data_ptr()), cutlass.Int64(w2.data_ptr()))
        views = (
            rt.to_cute(plan),
            self.views[0],
            self.views[1],
            w2_sf,
            rt.to_cute(self.scratch.workspace),
        )
        self.prepared[batch] = dict(
            code=code, out=out, output=rt.to_cute(out), box=tuple(publish), addresses=addresses,
            views=views, writer=None if planned else (writer, plan), mode=mode,
        )  # fmt: skip

    def group(self, ids):
        """No pregrouped unit table (the NVFP4 engines group in the kernel)."""

    def run(self, x, ids, weights=None):
        import cutlass

        from ..common import k3_ull_cute_host as host

        b = x.shape[0]
        if b not in self.prepared and b in self.deferred:
            self._prepare(*self.deferred.pop(b))
        p = self.prepared[b]
        if p["writer"] is not None:
            write, plan = p["writer"]
            write(x, plan, self.x_gsf)
        output = self.rt.to_cute(weights) if p["box"] else p["output"]
        plan, alphas, w13_sf, w2_sf, workspace = p["views"]
        args = (plan, self.rt.to_cute(ids), alphas, w13_sf, w2_sf, workspace, output)
        args += (cutlass.Int32(b), *self.scalars, *p["box"], *p["addresses"])
        p["code"](*args, host.stream(self.device))
        return p["out"]


#: Layer kinds of ``K3UllChain(layers=...)``: not on the chain, a KDA + MoE layer the chain runs
#: whole, an MLA + MoE layer whose MoE the chain runs.
CHAIN_OURS, CHAIN_CHAINED, CHAIN_MOE = 0, 1, 2
#: The step rows (requests x q) and verify length the chain serves.
ROWS = (8, 16, 24, 32)
Q = 8
#: Largest step whose run-exit consumers also store the MLA layer's MXFP8 input rows.
MX_ROWS = 16
#: Preferred shared-memory carve-out of a run-exit consumer that stores the MXFP8 rows (the MLA
#: projection it releases stages its weights beside it).
MX_CARVEOUT = 100
H = CHAIN_HIDDEN


@runtime_checkable
class KdaCheckpointState(Protocol):
    """What the chain needs of the cache manager in checkpoint mode (one KDA state per verify
    position of every request slot); the target's adapter is ``cache.K3UllKdaState``.

    ``caches(layer_idx)`` -> ``[conv_flat, state_flat]``: the flat kernel views of model layer
    ``layer_idx``'s checkpoints. ``state_flat`` [rows, heads, 128, 128] (BF16 or FP32) and
    ``conv_flat`` [rows, 3 * heads * 128, conv_window + q - 1] address row ``r`` at byte
    ``r * unit`` of the layer's buffer (checkpoint ``t`` of slot ``s`` at ``header + s * slot_units
    + t * state_units``; ``slot <= 0`` is padding). The views must not move after the chain binds
    them (the kernels and the captured graphs hold their addresses).

    ``indices(state_indices, slots, accepted)``: for the first ``slots.shape[0]`` requests of
    ``state_indices`` (their manager slots), write ``slots`` [n, q] int32 (the rows of their
    ``q`` checkpoints) and ``accepted`` [n] int32 (the recorded accepted count + 1: the kernels
    read checkpoint ``accepted - 1`` and write all ``q``) in place. Device ops only (captured with
    the step)."""

    def caches(self, layer_idx: int) -> Sequence[torch.Tensor]: ...

    def indices(
        self, state_indices: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor
    ) -> object: ...


def model_layers(num_layers: int = NUM_LAYERS) -> Tuple[Tuple[int, int], ...]:
    """``(layer_idx, kind)`` of every Kimi K3 layer: layer 0 (KDA + dense) :data:`OURS`, the KDA +
    MoE layers :data:`CHAINED`, the MLA + MoE layers :data:`MOE`."""
    return tuple(
        (i, CHAIN_OURS if i == 0 else CHAIN_CHAINED if is_kda_layer(i) else CHAIN_MOE)
        for i in range(num_layers)
    )


@dataclasses.dataclass(frozen=True)
class ChainPlan:
    """The chain's layer order and seams for ``layers`` (``(layer_idx, kind)`` of every model
    layer, in order).

    ``indices``: the layers on the chain (kind != OURS), in order; the MoE tail ordinal of a layer
    is its position here. ``first``: chained KDA layers after a layer not on the chain (their
    input comes from the model's mix). ``last``: chain layers followed by a layer that is not a
    chained KDA layer (an MLA layer, a layer off the chain, or the end). ``next_of``: the model
    layer after each chain layer (None: the last layer)."""

    layers: Tuple[Tuple[int, int], ...]
    indices: Tuple[int, ...]
    chained: frozenset
    moe_only: frozenset
    first: frozenset
    last: frozenset
    next_of: Dict[int, Optional[int]]


def plan_chain(layers: Sequence[Tuple[int, int]]) -> ChainPlan:
    model = tuple((int(i), int(c)) for i, c in layers)
    if any(c not in (CHAIN_OURS, CHAIN_CHAINED, CHAIN_MOE) for _, c in model):
        raise ValueError(f"layer kinds: {CHAIN_OURS} / {CHAIN_CHAINED} / {CHAIN_MOE}")
    indices = tuple(i for i, c in model if c)
    chained = frozenset(i for i, c in model if c == CHAIN_CHAINED)
    moe_only = frozenset(i for i, c in model if c == CHAIN_MOE)
    if not chained:
        raise ValueError("no chained layers")
    if len(indices) % 3 == 1:
        # The MoE tails' three parities: the first tail re-arms parity (n - 1) % 3, not its own.
        raise ValueError(f"{len(indices)} chained MoE layers: 1 mod 3, the tail parities collide")
    position = {i: k for k, (i, _) in enumerate(model)}
    on_chain = set(indices)
    first, last, next_of = set(), set(), {}
    for i in indices:
        k = position[i]
        if i in chained and (k == 0 or model[k - 1][0] not in on_chain):
            first.add(i)
        if k + 1 >= len(model):
            if i not in moe_only:
                raise ValueError(f"chained KDA layer {i} is the model's last layer")
            next_of[i] = None
            last.add(i)
            continue
        next_of[i] = model[k + 1][0]
        if model[k + 1][0] not in chained:
            last.add(i)
    for idx, kind in model:
        # The input of a layer after a chain consumer is that consumer's output: no block
        # snapshot of its own may be due there.
        if kind == CHAIN_OURS and idx > 0 and idx % ATTN_RES_BLOCK == 0:
            prev = model[position[idx] - 1][1]
            if prev:
                raise ValueError(f"layer {idx} after a chain layer is a block boundary")
    return ChainPlan(
        layers=model,
        indices=indices,
        chained=chained,
        moe_only=moe_only,
        first=frozenset(first),
        last=frozenset(last),
        next_of=next_of,
    )


# ---------------------------------------------------------------------------------------------
# Attribute views of the containers (what MR8's Prepared* classes read; no tensor copied)
# ---------------------------------------------------------------------------------------------


def _attention_view(w: KdaWeights, config: ChainConfig):
    """``KimiK3DeltaAttention``'s attributes MR8's KDA providers read, over ``w``."""
    heads = HEADS // w.tp
    name = f"model.layers.{w.layer}.self_attn"
    return NS(
        layer_idx=w.layer,
        layer_name=name,
        prefix=name,
        local_num_heads=heads,
        head_dim=HEAD_DIM,
        local_projection_size=heads * HEAD_DIM,
        in_proj_qkvgfab=NS(weight=w.in_proj),
        f_b_proj=NS(weight=w.f_b),
        conv1d=NS(weight=w.conv, bias=None),
        A_log=w.a_log,
        dt_bias=w.dt_bias,
        o_norm=NS(weight=w.o_norm, eps=config.rms_norm_eps),
        gate_lower_bound=config.gate_lower_bound,
        o_proj=NS(weight=w.o_proj),
    )


def _moe_view(w: MoeWeights, config: ChainConfig):
    """``KimiMoE``'s attributes MR8's MoE providers read, over ``w``."""
    eps = config.rms_norm_eps
    return NS(
        prefix=f"model.layers.{w.layer}.mlp",
        gate=NS(e_score_correction_bias=w.score_bias),
        routed_expert_norm=NS(weight=w.latent_norm, variance_epsilon=eps, eps=eps),
        routed_expert_up_proj=NS(weight=w.latent_up),
        shared_experts=NS(
            gate_up_proj=NS(weight=w.shared_gate_up), down_proj=NS(weight=w.shared_down)
        ),
        routed_scaling_factor=config.routed_scaling_factor,
        moe_renormalize=True,
    )


def _decoder_view(layer: ChainLayerWeights, attention, mlp):
    """A decoder layer as MR8's providers read it: its AttnRes sites, attention and MoE."""
    a = layer.attn_res
    return NS(
        layer_idx=layer.layer,
        prefix=f"model.layers.{layer.layer}",
        is_moe_layer=True,
        self_attn=attention,
        mlp=mlp,
        self_attention_res_norm=a.self_attention_res_norm,
        self_attention_res_proj=a.self_attention_res_proj,
        input_layernorm=a.input_layernorm,
        post_attention_layernorm=a.post_attention_layernorm,
        mlp_res_norm=a.mlp_res_norm,
        mlp_res_proj=a.mlp_res_proj,
        prev_valid_blocks=a.prev_valid_blocks,
        is_block_write_layer=a.is_block_write_layer,
        block_write_idx=a.block_write_idx,
    )


class _Kernels:
    """The MR8 modules the chain builds from (imported on first use: they need CUTLASS)."""

    def __init__(self):
        # isort: off
        from ..common import k3_ull_compat as _compat
        from ...kernels.k3r_attn_res.ull import (
            prepared as attn_res,
        )
        from ...kernels.k3r_comm import ull_twoshot as twoshot
        from ...kernels.k3r_kda.ull import head as kda_head
        from ...kernels.k3r_kda.ull import (
            prepared as kda_prepared,
        )
        from ...kernels.k3r_moe.ull import front as moe_front
        from ...kernels.k3r_moe.ull import (
            prepared as moe_prepared,
        )
        from ...kernels.k3r_moe.ull import tail_runtime
        from ...kernels.k3r_moe.ull.small_batch import (
            chain as small_batch,
        )
        from ...kernels.k3r_oproj.ull import prepared as oproj
        from . import k3_ull_prefetch as prefetch, k3_ull_providers as providers
        # isort: on

        self.compat, self.attn_res, self.twoshot = _compat, attn_res, twoshot
        self.kda_head, self.kda_prepared = kda_head, kda_prepared
        self.moe_front, self.moe_prepared, self.tail_runtime = moe_front, moe_prepared, tail_runtime
        self.small_batch, self.oproj = small_batch, oproj
        self.prefetch, self.providers = prefetch, providers


class _DecodeState:
    """Shared KDA provider state for the prepared batches, side stream and scratch buffers."""

    def __init__(self, k: _Kernels, *, device, tp, rank, q, rows, bf16_state):
        p = k.providers
        rounded = {k.compat.round_up(s, q) for s in rows}
        verify = {m for m in rounded if m <= p.PREPARED_MAX}
        if verify != set(rows):
            raise ValueError(
                f"chain rows {tuple(rows)}: whole q{q} requests up to {p.PREPARED_MAX}"
            )
        # MR8's tuned providers serve every batch up to 8 and 16 / 32 / 64, plus the verify rows.
        self.tuned = tuple(sorted({*p.BATCHES, *verify}))
        self.q = q
        self.batches = self.tuned
        self.capacity = self.batches[-1]
        self.device, self.tp, self.rank = device, tp, rank
        self.bf16_state = bf16_state
        self.buffers: dict = {}
        self.kda_side = k.kda_head.KdaSide(device)
        self.kda_heads = 0
        empty = partial(torch.empty, device=device, dtype=torch.bfloat16)
        self.hidden = (empty((self.capacity, H)), empty((self.capacity, H)))


class SoloKda:
    """MR8's KDA in-projection + core of one KDA layer outside the chain, without its
    o-projection: ``run`` returns the core rows (``[rows, 768]`` BF16, after the gated norm) for
    the model's o-projection. The kernels a chained layer runs at the verify shapes ``rows``
    (``q`` tokens per request): on Rubin TP16 with a BF16 state the ``HEAD_BATCHES`` /
    ``HEAD_MULTI`` rows (8: ``KdaHeadVerify``, 16: ``KdaHeadMulti``) take the in-projection and
    the verify in one kernel (F_A / beta on ``side``, the step's KDA side stream: joined at the
    end of the step), the other rows the tuned split-K in-projection and ``PreparedKda`` (GB300
    and FP32 states: ``KdaSplit``). ``caches``: the layer's ``[conv, state]`` checkpoint views."""

    def __init__(self, k: _Kernels, attention, caches, *, rows: Sequence[int], q: int, tp: int,
                 side):  # fmt: skip
        providers = k.providers
        weight = attention.in_proj_qkvgfab.weight
        pitch, device = weight.shape[0], weight.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.layer_idx = attention.layer_idx
        self.q, self.rows, self.caches = q, tuple(sorted(rows)), list(caches)
        if q < 2 or any(r % q for r in self.rows):
            raise ValueError(f"solo KDA: verify rows {self.rows} at q {q}")
        policies = providers.projection_policies(tp, pitch, self.rows, sms, q)
        missing = [r for r in self.rows if r not in policies]
        if missing:
            raise ValueError(f"solo KDA: no tuned in-projection for rows {missing}")
        self.projection = providers.PreparedProjection(weight, policies).prepare()
        verify = {(r // q, q): (pitch, int(tp == 8)) for r in self.rows}
        dim_first = providers.is_conv_state_dim_first()
        self.core = k.kda_prepared.PreparedKda(attention, verify, dim_first)
        self.core.prepare_caches(self.caches)
        self.head = None
        if sms > 152 and tp == 16 and pitch == 3216 and self.caches[1].dtype == torch.bfloat16:
            multi = providers.batch_table(providers.HEAD_MULTI.get(q, ""))
            heads = [(b, q) for b in providers.HEAD_BATCHES if b in self.rows and b == q == 8]
            heads += [(b, q) for b in self.rows if multi.get(b)]
            if heads:
                # This layer's own [ready, done] counter pair and F_A rows (the chained layers'
                # are indexed by their KdaHead count, which this one must not shift).
                counter = torch.zeros(2, dtype=torch.int32, device=device)
                fa_out = torch.empty(
                    (64, pitch - 4 * self.core.dim), dtype=torch.bfloat16, device=device
                )
                self.head = k.kda_head.PreparedKdaHead(
                    weight, self.core, (), side, counter, fa_out, verify=heads
                )
                self.head.prepare_caches(self.caches)
        cap = self.rows[-1]
        self.projected = torch.empty((cap, pitch), dtype=torch.bfloat16, device=device)
        self.out = torch.empty((cap, self.core.dim), dtype=torch.bfloat16, device=device)

    def kernels(self) -> Dict[int, str]:
        """Rows -> the kernels that run them."""
        verify = () if self.head is None else self.head.verify_kernels
        return {
            r: "head_verify" if (r, self.q) in verify else "split_k+prepared_kda" for r in self.rows
        }

    def run(self, x: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
        """The core rows of the step's ``x`` (the normed attention input, ``[rows, 7168]`` BF16);
        ``slots`` [rows / q, q] / ``accepted`` [rows / q]: the step's checkpoint rows and
        accepted counts."""
        b = x.shape[0]
        out = self.out[:b]
        if self.head is not None and (b, self.q) in self.head.verify_kernels:
            return self.head.run_verify(x, out, caches=self.caches, slots=slots, accepted=accepted)
        projected = self.projection.run(x, self.projected[:b])
        return self.core.run(projected, out, caches=self.caches, slots=slots, accepted=accepted)


class K3UllChain:
    """One rank's chain over the model layers ``layers`` (all ranks build it together:
    symmetric-memory mailboxes, collective allocations; outside CUDA graph capture).

    ``layers``: ``(layer_idx, kind)`` of every model layer in order (:func:`model_layers`);
    ``group``: the NCCL process group of the 16 ranks (symmetric memory); ``weights``:
    ``_chain_weights.load_chain_weights``'s, this rank's (it must hold every layer of ``layers``
    on the chain, the next layer's AttnRes vectors of each and the solo layers' KDA weights).
    ``rows``: the step rows (requests x ``q``) served; ``solo``: KDA layers off the chain whose
    in-projection + core also run on MR8's kernels (:meth:`run_solo`); ``tap_mix``: the DSpark taps
    store the pre-norm AttnRes mixture (else the updated prefix); ``kda_state``: the cache
    manager's checkpoints (:class:`KdaCheckpointState`, bound here); ``nv_force``: run the NVFP4
    routed experts' Rubin engines on another device (GB300)."""

    def __init__(
        self,
        layers: Sequence[Tuple[int, int]],
        group,
        *,
        weights: ChainWeights,
        kda_state: KdaCheckpointState,
        nv_force: bool,
        rows: Sequence[int] = ROWS,
        q: int = Q,
        solo: Sequence[int] = (0,),
        tap_mix: bool = True,
    ):
        self.k = k = _Kernels()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.group = group
        self.tp = torch.distributed.get_world_size(group)
        self.rank = torch.distributed.get_rank(group)
        k.compat.set_tp_group(group, self.device)
        if self.tp != TP or weights.tp != self.tp or weights.rank != self.rank:
            raise ValueError(
                f"Kimi K3 chain: group rank {self.rank} of {self.tp}, weights rank {weights.rank} "
                f"of {weights.tp} (TP{TP} only)"
            )
        self.q, self.rows = q, tuple(sorted(rows))
        self.tap_mix = tap_mix
        self.weights, config = weights, weights.config
        self.plan = plan = plan_chain(layers)
        self.indices, self.first, self.last = plan.indices, plan.first, plan.last
        self.next_of, self.moe_only = plan.next_of, plan.moe_only
        props = torch.cuda.get_device_properties(self.device)
        rubin = (props.major, props.minor) == (10, 7)
        if not (rubin or nv_force):
            raise RuntimeError(
                "Kimi K3 chain: the NVFP4 routed experts run on the Rubin engines (SM107); "
                "build the chain with nv_force=True to run them on this device"
            )
        self.nv_force = bool(nv_force)
        sms = props.multi_processor_count
        missing = [
            i for i in plan.indices
            if i not in weights.layers or weights.layers[i].moe is None
            or (i in plan.chained and (weights.layers[i].kda is None
                                       or weights.layers[i].kda.o_proj is None))
        ]  # fmt: skip
        if missing:
            raise ValueError(f"Kimi K3 chain: no chain weights for layers {missing}")
        state_dtype = self._state_dtype(kda_state, min(plan.chained))
        ds = self.ds = _DecodeState(
            k, device=self.device, tp=self.tp, rank=self.rank, q=q, rows=self.rows,
            bf16_state=state_dtype == torch.bfloat16,
        )  # fmt: skip
        self.kda_side = ds.kda_side
        self.nv_scratch = NvScratch(self.device)

        # Decoder views in chain order (the MLA layers' attention is not on the chain: their
        # view only carries the o-projection the next-attention prefetch table reads).
        stub = NS(o_proj=NS(weight=torch.zeros(256, dtype=torch.bfloat16, device=self.device)))
        self.decoders: Dict[int, NS] = {}
        for i in plan.indices:
            lw = weights.layers[i]
            attention = _attention_view(lw.kda, config) if i in plan.chained else stub
            self.decoders[i] = _decoder_view(lw, attention, _moe_view(lw.moe, config))
        decoders = [self.decoders[i] for i in plan.indices]

        # Prepare the per-layer KDA and MoE providers.
        beta, linear_beta = config.situ_beta, config.situ_linear_beta
        owner = (group, self.device, self.tp, self.rank)
        fronts = {i: weights.layers[i].moe.front for i in plan.indices}
        self.prefetch_controller = k.prefetch.PrefetchController(ds, decoders, fronts)
        state = self.moe_state = k.moe_prepared.MoEScratch(self.tp, self.device, ds.tuned)
        mailbox = k.oproj.OprojMailbox(*owner)
        self.tail_box = k.tail_runtime.MoeTailBox(*owner)
        twoshot = k.twoshot.TwoShot(*owner)
        self.front_box = k.oproj.OprojMailbox(*owner, (1, 64, LATENT + 2 * EXPERTS))
        small_batch = k.small_batch.SmallBatchBox(self.device, self.tp)
        self.boxes = (mailbox, twoshot, small_batch)
        self.attention: Dict[int, object] = {}
        self.feed_forward: Dict[int, object] = {}
        tail_op = NS(contract=NS(tp_size=self.tp, rms_eps=config.rms_norm_eps), rank=self.rank)
        count = len(plan.indices)
        for ordinal, i in enumerate(plan.indices):
            decoder = self.decoders[i]
            if i in plan.chained:
                self.attention[i] = k.providers.Attention(
                    decoder, ds, self.prefetch_controller, mailbox, small_batch, twoshot
                )
            moe = weights.layers[i].moe
            gsf = float(moe.routed["x_gsf"].reshape(-1)[0])
            front = k.moe_front.SharedFront(
                decoder.mlp, moe.front, self.front_box, state.front, ds.tuned, beta, linear_beta,
                plan=self.nv_scratch.plan, counters=state.shared, q=q,
                routes=self.nv_scratch.routes, nv=NvFront(gsf, wide_routing=sms <= 152),
            )  # fmt: skip
            experts = NvExperts(
                moe.routed, self.nv_scratch, beta=beta, linear_beta=linear_beta, q=q,
                rows=self.rows,
            )  # fmt: skip
            tail = k.tail_runtime.PreparedTail(
                tail_op, moe.latent_norm, moe.latent_up, ds.tuned, (self.tail_box, ordinal, count)
            )
            self.feed_forward[i] = k.moe_prepared.PreparedMoE(
                front, experts, tail, state, prefetch=self.prefetch_controller, layer_idx=i
            )  # fmt: skip
        # A tail followed by a layer whose input projection preloads its weights before its PDL
        # wait (a KdaHead; at q > 1 the MLA layer's tensor-memory in-projection) releases that
        # layer's AttnRes early.
        for i, after in zip(plan.indices, plan.indices[1:]):
            heads = self._tail_heads(after)
            if heads:
                self.feed_forward[i].tail.tail_weights.use_heads(heads)

        # Tail consumers: the next layer's AttnRes, or the output AttnRes + final norm.
        attn_res = k.attn_res.PreparedAttnRes
        ring = self.tail_box.out.buffer
        self.final = None
        self.consumers: Dict[int, object] = {}
        self.sites: Dict[int, object] = {}
        self.exit_mx: Dict[int, object] = {}
        self.mx_rows: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        mx_cap = max((b for b in self.rows if b <= MX_ROWS), default=0)
        for i in plan.indices:
            nxt = self.next_of[i]
            site = None
            if nxt is None:
                consumer = self.final = attn_res.for_target(weights.output, final=True)
            else:
                if nxt not in weights.layers:
                    raise ValueError(f"Kimi K3 chain: no AttnRes vectors of layer {nxt}")
                target = weights.layers[nxt].attn_res
                carveout = self.attention[nxt].before_carveout if nxt in plan.chained else None
                consumer = attn_res.for_target(target, has_delta=True, carveout=carveout)
                if nxt not in plan.chained:
                    site = self.sites[nxt] = target
            consumer.bind_ring(ring)
            consumer.bind_aux(mix=tap_mix)
            self.consumers[i] = consumer
            if site is not None and mx_cap:
                mx = attn_res.for_target(site, has_delta=True, carveout=MX_CARVEOUT)
                mx.bind_ring(ring)
                mx.bind_aux(mix=tap_mix)
                mx.bind_mx()
                self.exit_mx[i] = mx
                self.mx_rows[nxt] = (
                    torch.empty((mx_cap, H), dtype=torch.float8_e4m3fn, device=self.device),
                    torch.empty((mx_cap, H // 32), dtype=torch.uint8, device=self.device),
                )
        # No next-attention prefetch before an MLA + MoE layer (its attention is not here).
        for i in plan.indices:
            if self.next_of[i] in plan.moe_only:
                for b in self.rows:
                    self.prefetch_controller.tables.pop(("next_attention", i, b), None)

        empty = partial(torch.empty, device=self.device, dtype=torch.bfloat16)
        cap = max(self.rows)
        self.hidden = (empty((cap, H)), empty((cap, H)))
        self.prefix = empty((cap, H))
        self.out = empty((cap, H)) if None in self.next_of.values() else None
        self.metadata = {}
        for b in self.rows:
            n = b // q
            self.metadata[b] = NS(
                spec_state_indices_tensor=torch.zeros(
                    (n, q), dtype=torch.int32, device=self.device
                ),
                num_accepted_tokens=torch.ones((n,), dtype=torch.int32, device=self.device),
                non_spec_state_indices_tensor=None,
                num_prefills=0,
                num_spec_decodes=n,
                num_decodes=0,
                num_spec_decode_tokens=n * q,
                num_decode_tokens=0,
            )
        self.solo_layers = tuple(solo)
        for i in self.solo_layers:
            if i in plan.indices:
                raise ValueError(f"layer {i} is on the chain, not solo")
            if i not in weights.layers or weights.layers[i].kda is None:
                raise ValueError(f"Kimi K3 chain: no KDA weights of solo layer {i}")
        self.kda_state = None
        self.caches: Dict[int, list] = {}
        self.solo: Dict[int, SoloKda] = {}
        self.bind_kda_state(kda_state)
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ build helpers
    def _tail_heads(self, after: int) -> Tuple[int, ...]:
        """The batches whose tail before layer ``after`` releases its AttnRes early."""
        if after in self.moe_only:  # the MLA layer's tensor-memory in-projection
            return tuple(self.ds.batches) if self.q > 1 else ()
        return self.k.providers.head_batches(self.attention[after], self.q)

    @staticmethod
    def _state_dtype(kda_state: KdaCheckpointState, layer_idx: int) -> torch.dtype:
        """The KDA checkpoint state dtype (the KdaHead batches, the AttnRes carve-outs and the
        tails' early releases are set up for it)."""
        if not isinstance(kda_state, KdaCheckpointState):
            raise TypeError("Kimi K3 chain: kda_state must provide the KdaCheckpointState methods")
        return kda_state.caches(layer_idx)[1].dtype

    def bind_kda_state(self, kda_state: KdaCheckpointState) -> None:
        """Bind the manager's checkpoint views: every chained layer's KDA cores (and KdaHeads) are
        compiled for their layout, the solo layers' providers built. Outside CUDA graph capture;
        the views must not move afterwards."""
        if kda_state is self.kda_state:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Kimi K3 chain: bind the KDA checkpoint views before graph capture")
        caches, dtypes = {}, set()
        for i in (*sorted(self.plan.chained), *self.solo_layers):
            caches[i] = list(kda_state.caches(i))
            if len(caches[i]) != 2:
                raise ValueError(f"Kimi K3 chain: layer {i} KDA caches are not [conv, state]")
            dtypes.add(caches[i][1].dtype)
        if len(dtypes) != 1 or not dtypes <= {torch.bfloat16, torch.float32}:
            raise ValueError(f"Kimi K3 chain: KDA checkpoint states {sorted(map(str, dtypes))}")
        dtype = dtypes.pop()
        if self.kda_state is not None and dtype != self._bound_dtype:
            raise ValueError("Kimi K3 chain: a rebound KDA state must keep its dtype")
        if (dtype == torch.bfloat16) != self.ds.bf16_state:
            raise ValueError(
                f"Kimi K3 chain: built for bf16_state {self.ds.bf16_state}, got {dtype}"
            )
        self._bound_dtype = dtype
        for i in self.plan.chained:
            self.attention[i].prepare_layout(caches[i])
        # KdaHead and the MLA layer's tensor-memory in-projection preload their own weights: no
        # next-attention prefetch before them.
        pf = self.k.prefetch
        for i, after in zip(self.plan.indices, self.plan.indices[1:]):
            if after in self.plan.chained:
                heads = self.k.providers.head_batches(self.attention[after], self.q)
                if heads:
                    self.prefetch_controller.skip_before(i, heads, pf.BEFORE_HEAD)
            elif self.q > 1:
                self.prefetch_controller.skip_before(i, self.ds.batches, pf.BEFORE_TMEM_MLA)
        self.solo = {}
        for i in self.solo_layers:
            attention = _attention_view(self.weights.layers[i].kda, self.weights.config)
            self.solo[i] = SoloKda(
                self.k, attention, caches[i], rows=self.rows, q=self.q, tp=self.tp,
                side=self.kda_side,
            )  # fmt: skip
        self.caches = {i: caches[i] for i in self.plan.chained}
        self.kda_state = kda_state

    # ------------------------------------------------------------------ queries
    def supports(self, rows: int, q: int) -> bool:
        return q == self.q and rows in self.rows

    def chained(self, layer_idx: int) -> bool:
        """A chained KDA + MoE layer (the whole layer on the chain)."""
        return layer_idx in self.plan.chained

    def moe(self, layer_idx: int) -> bool:
        """An MLA + MoE layer whose MoE is on the chain (:meth:`run_moe`)."""
        return layer_idx in self.moe_only

    def num_snapshots_after(self, layer_idx: int) -> int:
        """The bank rows valid after chain layer ``layer_idx`` (its block write included)."""
        d = self.decoders[layer_idx]
        return d.prev_valid_blocks + int(d.is_block_write_layer)

    # ------------------------------------------------------------------ the step
    @staticmethod
    def blocks_view(block_residual: torch.Tensor) -> torch.Tensor:
        """The kernels' ``[M, K, H]`` view of the model's ``[K, M, H]`` bank (no copy)."""
        return block_residual.permute(1, 0, 2)

    def prefix_buffer(self, rows: int) -> torch.Tensor:
        return self.prefix[:rows]

    def bind_slots(self, state_indices: torch.Tensor, rows: int) -> None:
        """The step's KDA checkpoint rows and accepted counts of the requests in
        ``state_indices`` (their manager slots, [rows / q] or more) into the ``rows`` metadata, in
        place (device ops: once per step, before the first KDA layer, captured with it)."""
        md = self.metadata[rows]
        n = md.num_spec_decodes
        if state_indices.shape[0] < n:
            raise RuntimeError(f"{state_indices.shape[0]} state indices for {n} requests")
        self.kda_state.indices(
            state_indices[:n], md.spec_state_indices_tensor, md.num_accepted_tokens
        )

    def prefetch(self, point: str, layer_idx: int, rows: int) -> bool:
        """The L2 weight prefetch at ``point`` (``"attention_moe"`` after an MLA layer's attention
        core: the MoE front's and shared expert's weights; ``"oproj_down"`` after its
        o-projection: the front rows) of layer ``layer_idx`` on ``rows``-row steps, on the
        prefetch stream (joined by :meth:`join`); False where no table is set up."""
        return self.prefetch_controller.fork(point, layer_idx, rows)

    def run_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        mx: bool = False,
    ) -> torch.Tensor:
        """Chained layer ``layer_idx`` on its attention input ``x``: in-projection + KDA core,
        o-projection with its fused AttnRes, the MoE (front, shared, routed experts, tail) and the
        next layer's AttnRes as the tail's ring consumer. Updates ``prefix`` / ``blocks`` in place;
        returns the next layer's normed attention input (``aux``: the tap value also stored there,
        the next layer's DSpark tap). ``mx`` (a run's last layer, :meth:`mx_input`): the consumer
        also stores the MLA layer's MXFP8 input rows and releases it at entry with the carve-out
        its staged projection needs."""
        b = x.shape[0]
        first, second = self.hidden[0][:b], self.hidden[1][:b]
        moe_in = self.attention[layer_idx].run(
            x, second, prefix, blocks, self.caches[layer_idx], self.metadata[b], self.q
        )
        rows = self.mx_input(self.next_of[layer_idx], b) if mx else None
        if rows is not None:
            consume = partial(self.exit_mx[layer_idx].run_ring, prefix, blocks, aux=aux, mx=rows)
        else:
            consume = partial(self.consumers[layer_idx].run_ring, prefix, blocks, aux=aux)
        self.feed_forward[layer_idx].run(moe_in, first, consume=consume)
        return first

    def run_moe(
        self,
        layer_idx: int,
        x: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        mx: bool = False,
    ) -> torch.Tensor:
        """MLA + MoE layer ``layer_idx``'s MoE on its normed MoE input ``x`` (the model's
        post-attention mix; ``prefix`` already holds the attention output): front, shared and
        routed experts, tail, and the next layer's AttnRes as the tail's ring consumer. Updates
        ``prefix`` / ``blocks`` in place; returns the next layer's normed attention input, or (the
        model's last layer) the output AttnRes + final norm rows (``aux``: this layer's DSpark
        tap). ``mx``: as :meth:`run_layer` (next layer MLA)."""
        b = x.shape[0]
        nxt = self.next_of[layer_idx]
        target = self.hidden[0][:b] if nxt is not None else self.out[:b]
        rows = self.mx_input(nxt, b) if mx and nxt is not None else None
        if rows is not None:
            consume = partial(self.exit_mx[layer_idx].run_ring, prefix, blocks, aux=aux, mx=rows)
        else:
            consume = partial(self.consumers[layer_idx].run_ring, prefix, blocks, aux=aux)
        self.feed_forward[layer_idx].run(x, target, consume=consume)
        return target

    def mx_input(
        self, site: Optional[int], rows: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The MXFP8 rows ``(xq, xs)`` of MLA layer ``site``'s input that the run-exit consumer
        before it stores (``run_layer(..., mx=True)``) on ``rows``-row steps, or ``None`` where it
        does not (above :data:`MX_ROWS`)."""
        held = self.mx_rows.get(site)
        if held is None or rows > held[0].shape[0]:
            return None
        return held[0][:rows], held[1][:rows]

    def run_solo(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Solo KDA layer ``layer_idx``'s core rows of the attention input ``x`` (its
        in-projection + verify on the step's slots: :meth:`bind_slots` first)."""
        md = self.metadata[x.shape[0]]
        return self.solo[layer_idx].run(x, md.spec_state_indices_tensor, md.num_accepted_tokens)

    def join(self) -> None:
        """Join the step's side streams (L2 prefetch, KDA F_A / beta) into the current stream."""
        self.prefetch_controller.join()
        self.kda_side.join()

    # ------------------------------------------------------------------ checks
    def describe(self) -> dict:
        i = next(iter(sorted(self.plan.chained)))
        head = self.attention[i].head
        experts = self.feed_forward[i].experts
        return dict(
            chained=len(self.plan.chained),
            moe_only=sorted(self.moe_only),
            first=sorted(self.first),
            last=sorted(self.last),
            tp=self.tp,
            sms=torch.cuda.get_device_properties(self.device).multi_processor_count,
            rows=self.rows,
            tuned=self.ds.tuned,
            kda_head=None
            if head is None
            else dict(verify=sorted(map(list, head.verify_kernels)), polled=list(head.polled)),
            oproj={b: len(self.attention[i].output.codes.get(b, ())) for b in self.rows},
            experts_rubin=sorted(experts.rubin),
            tap="mix" if self.tap_mix else "prefix",
            kda_state=str(getattr(self, "_bound_dtype", None)),
            solo={j: s.kernels() for j, s in self.solo.items()},
            mx_sites=sorted(self.mx_rows),
            weights_gib=round(self.weights.nbytes() / 2**30, 2),
            nv_force=self.nv_force,
        )

    def effect_ok(self) -> Tuple[bool, str]:
        """Rubin (10.7): MR8's TP16 B8 q8 deployment path (KdaHeadVerify, fused o-projection,
        NVFP4 image8 experts) at 8 rows, the solo layers' KdaHeadVerify too; elsewhere always true
        (GB300 runs its own tables)."""
        if torch.cuda.get_device_capability(self.device) != (10, 7):
            return True, "not Rubin: GB300 tables"
        d = self.describe()
        miss = []
        if d["kda_head"] is None or [8, self.q] not in d["kda_head"]["verify"]:
            miss.append("head_verify")
        if d["oproj"].get(8) != 1:
            miss.append("oproj fused")
        if 8 not in d["experts_rubin"]:
            miss.append("image8")
        miss += [f"solo{j}:head_verify" for j, k in d["solo"].items() if k.get(8) != "head_verify"]
        return not miss, f"missing {miss}; kda_head {d['kda_head']}; oproj {d['oproj']}"
