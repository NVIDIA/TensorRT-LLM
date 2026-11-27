# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weight-table machinery every mistral-large-4-nvfp4 target shares.

A target's `weights.py` states its table: one `W` row per weight role, saying which checkpoint
tensors land in which parameter and, through the row's `shard`, which slice of them this rank
reads under the target's parallel layout. It also states `dims`, the dimension bundle its rows
are written against. Everything else is the same for every target and lives here:

* the row type `W` and the source, slice and layer-set builders the rows are written with;
* the checkpoint's layouts and the operands built from them: FP8 128x128 block scales into
  DeepGEMM's packed UE8M0 operand, and each routed expert into the NVFP4 MoE runner's layout;
* the checks on the per-tensor NVFP4 scalars the forward relies on;
* the BF16 vision tower's rows, replicated on every rank in every layout;
* `MistralLarge4Weights`, which declares a table's parameters, maps them to checkpoint keys and
  loads them, streaming from the checkpoint's own shard files when it can find them.

The checkpoint is Mistral-native (`consolidated-*.safetensors`, `layers.N.attention.wq`, ...)
and ModelOpt `MIXED_PRECISION`: the routed experts NVFP4, the other big linears FP8 128x128
block (`FP8_PB_WO`), everything else bf16.

FP8 linears are e4m3 `[N, K]` as stored, and their fp32 block scales -- stored as
`.weight_scale` `[N/128, 1, K/128, 1]`, squeezed here to the `[N/128, K/128]` grid -- are
installed in DeepGEMM's packed UE8M0 operand, `transform_sf_into_required_layout(recipe=(1, 128,
128))`: what `FP8BlockScalesLinearMethod.transform_weights` installs as `weight_scale` on
sm_100f. That operand is an MN-major `[N, ceil(K/512)]` int32 view; the parameter holds the dense
`[ceil(K/512), N]` behind it and the GEMM takes `.t()`. Where K/128 is not a multiple of four
the last word of each row is partly padding. The method's `resmooth_to_fp8_e8m0` is not run:
every scale is asserted an exact power of two, which makes it a power-of-two re-exponent of the
same values.

The routed experts are stacked on an expert axis in the NVFP4 MoE runner's layout: e2m1 codes,
two per byte, `[E, 2I, H/2]` / `[E, H, I/2]` uint8 and their e4m3 block scales, one per 16
inputs, `[E, 2I, H/16]` / `[E, H, I/16]`. Each source lands as stored -- FC1's rows `[w3 ; w1]`,
up rows first, the order the runner's FC1 concatenates them in -- and once every source of an
expert stack has landed, its row's `layout` rearranges each expert in place (the runner reads
nothing else, and a wrong layout costs 300-3400 bf16 ulp and raises nothing):

* FC1, codes and block scales alike: the halves interleaved (row `2i` up row `i`, row `2i + 1`
  gate row `i`), then the 32-row block shuffle (within each 32 rows, row `4u + v` moves to
  `8v + u`);
* FC2: the 32-row block shuffle;
* both block scales: then the 128x4 swizzle (`torch.ops.trtllm.block_scale_interleave`'s order,
  written here in torch so the table also runs on meta tensors).

The layout is a function of the shape it is applied to, so it runs on whatever slice of an
expert the rank holds.

The runner's output scalars are not derived here: the four per-tensor scalars are handed over
as stored -- `isc1` and `isc2`, the one FC1 and FC2 activation `input_scale` of the layer, and
per local expert `ws2_13` (`w1`'s `weight_scale_2`, which `w3` shares) and `ws2_2`. ModelOpt
stores these as reciprocals of the global scales a quantizer divides by. Before any of them is
read, `load` reads the per-tensor scalars of all the layer's experts, the other ranks' included,
and asserts what the forward relies on: `w1` and `w3` store the same `input_scale` and
`weight_scale_2` on every expert, so one FC1 alpha serves the stacked `[w3 ; w1]`; and each
projection stores one `input_scale` for every expert, so one NVFP4 quantization of the
activations is right for every expert on every rank and the ranks' partial outputs sum.
"""

from __future__ import annotations

import collections
import functools
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn
from safetensors import safe_open

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

F8, F32, I32, U8 = torch.float8_e4m3fn, torch.float32, torch.int32, torch.uint8

#: The FP8 block edge on both weight axes (the checkpoint's `FP8_PB_WO` 128x128 blocks), and
#: the K granularity of the DeepGEMM scale recipe `(1, 128, 128)`.
BLOCK = 128
#: UE8M0 bytes per int32 word of the DeepGEMM scale operand: four K blocks.
SF_PACK = 4
#: NVFP4: one e4m3 block scale per 16 inputs (the checkpoint's `group_size`), two e2m1 codes
#: per byte.
NVFP4_BLOCK = 16
E2M1_PER_BYTE = 2

#: A routed expert's three projections: `w1` gate, `w3` up, `w2` down.
PROJECTIONS = ("w1", "w2", "w3")
#: The per-tensor NVFP4 scalars each of them carries beside its codes and block scales.
EXPERT_SCALARS = ("weight_scale_2", "input_scale")

#: The vision tower's checkpoint prefixes.
VISION_PREFIXES = (
    "vision_encoder.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "vision_language_adapter.",
)


#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape.

    `src` says where the bytes come from: a checkpoint key template with `{p}`
    for the layer prefix (`layers.N`), or a callable `(d, i) -> [(key, index), ...]`
    where `index` is where in the parameter that tensor lands (`None`: all of it).
    A `key` may be a tuple of per-tensor scalars, which `transform` is handed together.

    `layers` says which layers carry this weight: `ALL_LAYERS`, `None` for a
    weight that is not per-layer, or a callable `d -> iterable[int]`. Two rows
    may share a name over disjoint layer sets, with different shapes.

    `shard` takes the dimension bundle and returns a function of the checkpoint
    entry: this rank's slice, applied to every source of the row. It runs before
    the copy to the device and before `transform`, so a rank moves only its own
    bytes, and a `transform` that reads a per-rank count off its input (`wkv_b`'s
    regroup reads this rank's head count) sees the slice. `None`: the source is
    read whole.

    `transform` runs on the (sliced) source tensor, on the destination's device.

    `layout` runs once every source of the parameter has landed, on each slot of its leading
    (expert) axis, and the slot is overwritten with the result: a rearrangement that mixes rows
    several sources fill, which no single source's `transform` can make.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS
    shard: Callable | None = None
    layout: Callable | None = None
    when: Callable | None = None
    tag: str = "l"

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple[Any, Any]]:
        """`[(checkpoint key | key tuple, index into the parameter | None), ...]`."""
        if callable(self.src):
            return list(self.src(d, i))
        return [(self.src.format(p=prefix), None)]


def _keys(ckpt) -> tuple[str, ...]:
    """The checkpoint keys of one source: itself, or the members of a key tuple."""
    return ckpt if isinstance(ckpt, tuple) else (ckpt,)


# --- this rank's slice of a source ---


def shard_rows(start, width):
    """A `shard`: rows `[start(d), start(d) + width(d))` of the checkpoint entry."""
    return lambda d: (lambda e: e[start(d) : start(d) + width(d)])


def shard_cols(start, width):
    """A `shard`: columns `[start(d), start(d) + width(d))` of the checkpoint entry."""
    return lambda d: (lambda e: e[:, start(d) : start(d) + width(d)])


def shard_sf_cols(start, width):
    """`shard_cols` on an FP8 block scale as stored, `[N/128, 1, K/128, 1]`: its K blocks are
    axis 2."""
    return lambda d: (lambda e: e[:, :, start(d) : start(d) + width(d)])


# --- FP8 block scales ---


def sf_words(k: int) -> int:
    """int32 words per row of the DeepGEMM scale operand of a K-wide weight."""
    return -(-k // (BLOCK * SF_PACK))


def _assert_ue8m0(s: torch.Tensor) -> None:
    """Every block scale is a positive normal power of two, so its UE8M0 byte (the fp32
    exponent field) holds it exactly. A meta tensor carries no values to check."""
    if s.is_meta:
        return
    bits = s.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exact = ((bits & 0x807FFFFF) == 0) & (exp > 0) & (exp < 255)
    assert bool(exact.all()), (
        f"weight_scale: {int((~exact).sum())} of {s.numel()} block scales are not a positive "
        "normal power of two (UE8M0); the scale layouts below would change their values"
    )


def _deepgemm_sf(s: torch.Tensor, core) -> torch.Tensor:
    """fp32 `[N/128, K/128]` block scales -> the dense `[ceil(K/512), N]` int32 storage of
    DeepGEMM's MN-major packed UE8M0 operand, whose `.t()` is that operand. N is a multiple of
    128 for every role here, so the operand's row stride is N and its storage has no MN padding.
    A K of a block count that is not a multiple of four (the shared expert's `w2` at K 768 on 8
    ranks: 6 blocks) leaves the high bytes of each row's last word zero."""
    _assert_ue8m0(s)
    n_blk, k_blk = s.shape
    sf = transform_sf_into_required_layout(
        s, mn=n_blk * BLOCK, k=k_blk * BLOCK, recipe=(1, BLOCK, BLOCK), is_sfa=False
    )
    return sf.t()


def block_sf(s: torch.Tensor, core) -> torch.Tensor:
    """An FP8 linear's block scales as stored, fp32 `[N/128, 1, K/128, 1]`: squeezed to the
    `[N/128, K/128]` grid, then in DeepGEMM's packed UE8M0 operand."""
    assert s.dim() == 4 and s.shape[1] == 1 and s.shape[3] == 1, (
        f"FP8 block scale stored as {tuple(s.shape)}, not [N/128, 1, K/128, 1]"
    )
    return _deepgemm_sf(s.reshape(s.shape[0], s.shape[2]), core)


def shared_scalar(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """One per-tensor scalar the checkpoint stores several times (`w1`'s and `w3`'s, or every
    expert's of the window): asserted the same value everywhere, then that value."""
    first = t[0]
    if not first.is_meta:
        values = torch.stack(t)
        assert bool((values == first).all()), (
            f"{int((values != first).sum())} of {len(t)} copies of one shared scale differ from "
            f"the first, {first.item()!r}"
        )
    return first


def to_fp32(t: torch.Tensor, core) -> torch.Tensor:
    return t.to(torch.float32)


# --- the NVFP4 MoE runner's expert layout, one expert at a time ---

#: The runner's row shuffle works on blocks of this many rows.
SHUFFLE_ROWS = 32
#: The block-scale swizzle's tile: 128 rows by 4 scale columns.
SWIZZLE_ROWS, SWIZZLE_COLS = 128, 4


@functools.lru_cache(maxsize=None)
def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: within each block of 32 rows, source row
    `4u + v` lands at row `8v + u`."""
    assert rows % SHUFFLE_ROWS == 0, rows
    j = torch.arange(SHUFFLE_ROWS)
    one = torch.empty(SHUFFLE_ROWS, dtype=torch.long)
    one[(j % 4) * 8 + j // 4] = j
    blocks = torch.arange(rows // SHUFFLE_ROWS).repeat_interleave(SHUFFLE_ROWS) * SHUFFLE_ROWS
    return (one.repeat(rows // SHUFFLE_ROWS) + blocks).to(device)


@functools.lru_cache(maxsize=None)
def _fc1_perm(rows: int, device) -> torch.Tensor:
    """Gather index of FC1's `[up ; gate]` rows: interleaved (row `2i` up row `i`, row `2i + 1`
    gate row `i`), then block-shuffled."""
    interleave = torch.empty(rows, dtype=torch.long)
    interleave[0::2] = torch.arange(rows // 2)
    interleave[1::2] = torch.arange(rows // 2, rows)
    return interleave.to(device)[_block32_perm(rows, device)]


def _swizzle(s: torch.Tensor) -> torch.Tensor:
    """One expert's `[M, C]` e4m3 block scales in the 128x4-swizzled byte order, kept `[M, C]`:
    byte `(m, c)` lands at flat offset `(m // 128) * 128 * C + (c // 4) * 512 + (m % 32) * 16 +
    ((m % 128) // 32) * 4 + c % 4`. Every target's scale stacks tile it exactly (M a multiple of
    128, C of 4), so there is no padding."""
    m, c = s.shape
    assert m % SWIZZLE_ROWS == 0 and c % SWIZZLE_COLS == 0, (m, c)
    tiles = s.view(torch.uint8).reshape(m // SWIZZLE_ROWS, 4, 32, c // SWIZZLE_COLS, SWIZZLE_COLS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(m, c).view(s.dtype)


def fc1_codes(w: torch.Tensor) -> torch.Tensor:
    return torch.index_select(w, 0, _fc1_perm(w.shape[0], w.device))


def fc1_scales(s: torch.Tensor) -> torch.Tensor:
    rows = torch.index_select(s.view(torch.uint8), 0, _fc1_perm(s.shape[0], s.device))
    return _swizzle(rows.view(s.dtype))


def fc2_codes(w: torch.Tensor) -> torch.Tensor:
    return torch.index_select(w, 0, _block32_perm(w.shape[0], w.device))


def fc2_scales(s: torch.Tensor) -> torch.Tensor:
    rows = torch.index_select(s.view(torch.uint8), 0, _block32_perm(s.shape[0], s.device))
    return _swizzle(rows.view(s.dtype))


def _materialize(entry, shard=None) -> torch.Tensor:
    """This rank's bytes of one checkpoint entry: a tensor, or a lazy safetensors slice, which
    reads only what the index selects. `[...]`, not `[:]`: the per-tensor scalars are 0-dim."""
    return entry[...] if shard is None else shard(entry)


# --- the checkpoint's shard files ---

#: The checkpoint's shard index, beside its `consolidated-*.safetensors`.
SHARD_INDEX = "consolidated.safetensors.index.json"


def checkpoint_shards(checkpoint_dir) -> tuple[Path, dict[str, str]]:
    """`(directory, key -> shard file)` of a checkpoint directory: its index when it ships
    one, else the header of each `consolidated*.safetensors`."""
    directory = Path(checkpoint_dir)
    index = directory / SHARD_INDEX
    if index.is_file():
        return directory, json.loads(index.read_text())["weight_map"]
    weight_map: dict[str, str] = {}
    for path in sorted(directory.glob("consolidated*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            weight_map.update(dict.fromkeys(handle.keys(), path.name))
    if not weight_map:
        raise FileNotFoundError(f"no consolidated*.safetensors in {directory}")
    return directory, weight_map


def _shard_files(core, weights) -> tuple[Path, dict[str, str]] | None:
    """`(directory, key -> shard file)` of the checkpoint a dict of sources was read from, or
    None when the sources are not the shard files of one (the directory the config records
    lacks the index, or the index does not name every key `weights` holds)."""
    cfg = getattr(core.model_config, "pretrained_config", None)
    root = getattr(cfg, "_name_or_path", None)
    if not root:
        return None
    index = Path(root) / SHARD_INDEX
    if not index.is_file():
        return None
    weight_map = json.loads(index.read_text())["weight_map"]
    if not set(weights.keys()) <= set(weight_map):
        return None
    return Path(root), weight_map


def _drop_page_cache(path: Path) -> None:
    """Advise the kernel to drop a consumed shard file's cached pages. Advice only: a
    filesystem that ignores it costs memory pressure, not correctness. Every rank reads every
    shard file, a node's ranks through that node's page cache, so the advice can also drop pages
    a sibling rank has yet to read, which that rank then reads again from disk; the kernel keeps
    the pages a process has mapped."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


# --- which layers carry a weight ---


def owners(d):
    """Layers that own a latent KV. The others read `kv_source_layers[i]`'s and project none."""
    return [i for i in range(d.num_layers) if d.kv_source[i] == i]


def rope_owners(d):
    return [i for i in owners(d) if not d.nope_layer[i]]


def nope_owners(d):
    return [i for i in owners(d) if d.nope_layer[i]]


def rope_layers(d):
    return [i for i in range(d.num_layers) if not d.nope_layer[i]]


def nope_layers(d):
    return [i for i in range(d.num_layers) if d.nope_layer[i]]


def indexer_layers(d):
    """Layers that own an indexer. The 'shared' ones reuse the most recent one's selection."""
    return [i for i in range(d.num_layers) if d.full_indexer[i]]


def dense_layers(d):
    return range(d.dense_layers)


def moe_layers(d):
    return range(d.dense_layers, d.num_layers)


def draft_layers(d):
    return range(d.draft_layers)


def _vision_blocks(d):
    return list(range(d.v_layers))


def _has_vision(d):
    return d.has_vision


# --- where the bytes come from ---


def stack(*parts, prefix="layers"):
    """Several checkpoint tensors into one parameter: each `(key template, index(d))`, the
    index being where in the parameter that tensor lands."""
    return lambda d, i: [(t.format(p=f"{prefix}.{i}"), index(d)) for t, index in parts]


def _expert_key(i: int, e: int, suffix: str) -> str:
    return f"layers.{i}.experts.{e}.{suffix}"


def expert_window(*parts):
    """This rank's routed experts, local slot `s` <- global expert `expert_offset + s`: each
    `(checkpoint suffix | suffix tuple, index(d) after the expert axis)`."""

    def key(i, e, suffix):
        if isinstance(suffix, tuple):
            return tuple(_expert_key(i, e, s) for s in suffix)
        return _expert_key(i, e, suffix)

    def build(d, i):
        return [
            (key(i, d.expert_offset + slot, suffix), (slot, *index(d)))
            for slot in range(d.local_experts)
            for suffix, index in parts
        ]

    return build


def window_scalar(*suffixes):
    """One per-tensor scalar every expert of the window stores: all of the window's copies, one
    key tuple, into element 0."""
    return lambda d, i: [
        (
            tuple(
                _expert_key(i, d.expert_offset + slot, s)
                for slot in range(d.local_experts)
                for s in suffixes
            ),
            (0,),
        )
    ]


def draft_source(suffix):
    return lambda d, i: [(f"mtp_dspark.backbone.layers.{i}.{suffix}", None)]


def _flatten_patch_conv(tensor, core):
    return tensor.flatten(1)


def _expert_scalar_keys(d) -> list[str]:
    """Every routed expert's per-tensor scalars on every MoE layer, the other ranks' included:
    what `load` reads, and checks, before anything else."""
    return [
        _expert_key(i, e, f"{proj}.{s}")
        for i in moe_layers(d)
        for e in range(d.num_experts)
        for proj in PROJECTIONS
        for s in EXPERT_SCALARS
    ]


def _check_expert_scalars(scalars: dict[str, torch.Tensor], d) -> None:
    """Per MoE layer, over all of its experts: `w1` and `w3` store the same `input_scale` and
    `weight_scale_2` on every expert; `w1` and `w2` each store one `input_scale` for every expert;
    every scalar is a positive finite fp32. Meta tensors carry no values to check."""
    if scalars and next(iter(scalars.values())).is_meta:
        return
    experts = range(d.num_experts)
    for i in moe_layers(d):
        v = {
            (proj, s): torch.stack([scalars[_expert_key(i, e, f"{proj}.{s}")] for e in experts])
            for proj in PROJECTIONS
            for s in EXPERT_SCALARS
        }
        for (proj, s), t in v.items():
            bad = (~(torch.isfinite(t) & (t > 0))).nonzero().flatten()
            assert bad.numel() == 0, (
                f"layers.{i}: {proj}.{s} of expert {int(bad[0])} is {t[bad[0]].item()!r}, "
                "not a positive finite scale"
            )
        for s in EXPERT_SCALARS:
            bad = (v["w1", s] != v["w3", s]).nonzero().flatten()
            assert bad.numel() == 0, (
                f"layers.{i}: w1 and w3 {s} differ on {bad.numel()} of {len(experts)} experts, "
                f"first expert {int(bad[0])}: {v['w1', s][bad[0]].item()!r} vs "
                f"{v['w3', s][bad[0]].item()!r}"
            )
        for proj in ("w1", "w2"):
            isc = v[proj, "input_scale"]
            bad = (isc != isc[0]).nonzero().flatten()
            assert bad.numel() == 0, (
                f"layers.{i}: {proj}.input_scale is not one value over the experts: expert "
                f"{int(bad[0])} stores {isc[bad[0]].item()!r}, expert 0 {isc[0].item()!r}"
            )


def _read_scalars(keys: list[str], weights, shards) -> dict[str, torch.Tensor]:
    """The given per-tensor scalars, off the shard files one file at a time when they are
    found, otherwise off the engine's dict."""
    if shards is None:
        return {key: _materialize(weights[key]) for key in keys}
    directory, weight_map = shards
    by_file: dict[str, list] = collections.defaultdict(list)
    for key in keys:
        by_file[weight_map[key]].append(key)
    out = {}
    for name in sorted(by_file):
        with safe_open(str(directory / name), framework="pt", device="cpu") as handle:
            for key in by_file[name]:
                out[key] = handle.get_slice(key)[...].clone()
    return out


# --- rows ---


def fp8_linear(name, key, n, k, layers, w_shard=None, s_shard=None):
    """An FP8 128x128-block linear: e4m3 `[N, K]` and its DeepGEMM scale operand's storage."""
    return (
        W(name, shape=lambda d: (n(d), k(d)), dtype=F8, src=key, layers=layers, shard=w_shard),
        W(
            f"{name}_s",
            shape=lambda d: (sf_words(k(d)), n(d)),
            dtype=I32,
            src=key.replace(".weight", ".weight_scale"),
            transform=block_sf,
            layers=layers,
            shard=s_shard,
        ),
    )


#: The BF16 vision encoder, pre-projector norm, patch merger and language adapter, replicated on
#: every rank when the configuration declares a vision encoder.
VISION_WEIGHTS: tuple[W, ...] = (
    W(
        "vis_patch",
        shape=lambda d: (d.v_hidden, d.v_channels * d.v_patch**2),
        src="vision_encoder.patch_conv.weight",
        transform=_flatten_patch_conv,
        layers=None,
        when=_has_vision,
    ),
    W(
        "vis_ln_pre",
        shape=lambda d: (d.v_hidden,),
        src="vision_encoder.ln_pre.weight",
        layers=None,
        when=_has_vision,
    ),
    W(
        "attn_norm",
        shape=lambda d: (d.v_hidden,),
        src="{p}.attention_norm.weight",
        layers=_vision_blocks,
        tag="v",
    ),
    W(
        "qkv",
        shape=lambda d: (3 * d.v_hidden, d.v_hidden),
        src=stack(
            ("{p}.attention.wq.weight", lambda d: (slice(0, d.v_hidden),)),
            ("{p}.attention.wk.weight", lambda d: (slice(d.v_hidden, 2 * d.v_hidden),)),
            ("{p}.attention.wv.weight", lambda d: (slice(2 * d.v_hidden, 3 * d.v_hidden),)),
            prefix="vision_encoder.transformer.layers",
        ),
        layers=_vision_blocks,
        tag="v",
    ),
    W(
        "wo",
        shape=lambda d: (d.v_hidden, d.v_hidden),
        src="{p}.attention.wo.weight",
        layers=_vision_blocks,
        tag="v",
    ),
    W(
        "ffn_norm",
        shape=lambda d: (d.v_hidden,),
        src="{p}.ffn_norm.weight",
        layers=_vision_blocks,
        tag="v",
    ),
    W(
        "w13",
        shape=lambda d: (2 * d.v_inter, d.v_hidden),
        src=stack(
            ("{p}.feed_forward.w1.weight", lambda d: (slice(0, d.v_inter),)),
            ("{p}.feed_forward.w3.weight", lambda d: (slice(d.v_inter, 2 * d.v_inter),)),
            prefix="vision_encoder.transformer.layers",
        ),
        layers=_vision_blocks,
        tag="v",
    ),
    W(
        "w2",
        shape=lambda d: (d.v_hidden, d.v_inter),
        src="{p}.feed_forward.w2.weight",
        layers=_vision_blocks,
        tag="v",
    ),
    W(
        "vis_proj_norm",
        shape=lambda d: (d.v_hidden,),
        src="pre_mm_projector_norm.weight",
        layers=None,
        when=_has_vision,
    ),
    W(
        "vis_merge",
        shape=lambda d: (d.v_hidden, d.v_hidden * d.v_merge**2),
        src="patch_merger.merging_layer.weight",
        layers=None,
        when=_has_vision,
    ),
    W(
        "vis_w_in",
        shape=lambda d: (d.hidden, d.v_hidden),
        src="vision_language_adapter.w_in.weight",
        layers=None,
        when=_has_vision,
    ),
    W(
        "vis_w_out",
        shape=lambda d: (d.hidden, d.hidden),
        src="vision_language_adapter.w_out.weight",
        layers=None,
        when=_has_vision,
    ),
)


def checkpoint_dims(core) -> dict[str, Any]:
    """The dimensions every target reads the same way: the checkpoint's, the DSpark draft's and
    the vision tower's. A target's `dims` adds what its parallel layout fixes per rank."""
    cfg = core.model_config.pretrained_config
    vis = getattr(cfg, "vision_config", None)
    spec = getattr(core.model_config, "spec_config", None)
    draft = getattr(spec, "_draft_model_config", None) or {}
    return dict(
        draft_layers=int(draft.get("n_layers", 0)),
        draft_rank=int((draft.get("dspark") or {}).get("markov_rank", 0)),
        draft_depth=int(draft.get("depth", 0)),
        draft_capture_layers=len(spec.target_layer_ids) if draft else 0,
        num_layers=cfg.num_hidden_layers,
        hidden=cfg.hidden_size,
        nope=cfg.qk_nope_head_dim,
        rope_dim=cfg.qk_rope_head_dim,
        qk_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        v_dim=cfg.v_head_dim,
        kv_lora=cfg.kv_lora_rank,
        lat_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
        nope_layer=tuple(bool(f) for f in cfg.nope_layer_flags),
        kv_source=tuple(int(s) for s in cfg.kv_source_layers),
        full_indexer=tuple(t == "full" for t in cfg.index_sharing),
        idx_heads=cfg.index_n_heads,
        idx_dim=cfg.index_head_dim,
        dense_layers=cfg.first_k_dense_replace,
        num_experts=cfg.n_routed_experts,
        vocab=cfg.vocab_size,
        dtype=core.model_config.torch_dtype,
        # The vision tower (replicated; absent without a vision encoder).
        has_vision=vis is not None,
        v_layers=vis.num_hidden_layers if vis is not None else 0,
        v_hidden=vis.hidden_size if vis is not None else 0,
        v_heads=vis.num_attention_heads if vis is not None else 0,
        v_head_dim=vis.hidden_size // vis.num_attention_heads if vis is not None else 0,
        v_inter=vis.intermediate_size if vis is not None else 0,
        v_channels=vis.num_channels if vis is not None else 0,
        v_patch=vis.patch_size if vis is not None else 0,
        v_merge=vis.spatial_merge_size if vis is not None else 0,
        v_theta=float(vis.rope_theta) if vis is not None else 0.0,
        image_token_id=vis.image_token_id if vis is not None else None,
    )


class MistralLarge4Weights(ABC):
    """A target's weight table: its `WEIGHTS` and `DRAFT_WEIGHTS` rows, the dimension bundle
    `dims` they are written against, and the slice of the checkpoint's `output.weight` that
    `lm_head_shard` gives the shell's `lm_head`. This class declares, maps and loads them."""

    WEIGHTS: tuple[W, ...] = ()
    #: The DSpark draft's rows, declared when the speculative config carries the checkpoint's
    #: own draft.
    DRAFT_WEIGHTS: tuple[W, ...] = ()

    @abstractmethod
    def dims(self, core) -> SimpleNamespace:
        """The dimension bundle the rows are written against: `checkpoint_dims` and what the
        target's parallel layout fixes per rank."""

    @abstractmethod
    def lm_head_shard(self, d: SimpleNamespace) -> Callable | None:
        """This rank's slice of the checkpoint's `output.weight`, as the shell builds `lm_head`,
        or None for all of it."""

    def _entries(self, d):
        return self.WEIGHTS + (self.DRAFT_WEIGHTS if d.draft_layers else ())

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self._entries(d):
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                assert key not in w, f"two rows declare {key}"
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        """`[(param key, layer index | None, checkpoint layer prefix), ...]`.

        One place decides which parameters an entry declares, so `declare` and
        `manifest` cannot disagree about whether a weight exists.
        """
        if entry.when is not None and not entry.when(d):
            return []
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        prefix = "vision_encoder.transformer.layers" if entry.tag == "v" else "layers"
        return [(f"{entry.tag}{i}_{entry.name}", i, f"{prefix}.{i}") for i in layers]

    def manifest(self, core) -> dict:
        """param key -> [(checkpoint key | key tuple, index into the param | None, transform,
        shard)]."""
        d = self.dims(core)
        rows: dict = {}
        for entry in self._entries(d):
            shard = entry.shard(d) if entry.shard else None
            for key, i, prefix in self._targets(entry, d):
                assert key not in rows, f"two rows declare {key}"
                rows[key] = [
                    (ckpt, index, entry.transform, shard)
                    for ckpt, index in entry.sources(d, i, prefix)
                ]
        return rows

    def expected_unconsumed(self, core, weights) -> dict[str, str]:
        """Declared unused keys: off-rank experts and disabled draft weights. Off-rank expert
        scalars are still read to validate the layer's shared scales."""
        d = self.dims(core)
        lo, hi = d.expert_offset, d.expert_offset + d.local_experts
        keys = {
            k: "DSpark draft disabled"
            for k in weights.keys()
            if not d.draft_layers and k.startswith("mtp_dspark.")
        }
        window = f"routed expert outside this rank's window [{lo}, {hi})"
        never_read = f"{window}: never read"
        checked = f"{window}: read only to check the layer's shared scales"
        for i in moe_layers(d):
            for e in range(d.num_experts):
                if lo <= e < hi:
                    continue
                for proj in PROJECTIONS:
                    for s in ("weight", "weight_scale"):
                        key = _expert_key(i, e, f"{proj}.{s}")
                        if key in weights:
                            keys[key] = never_read
                    for s in EXPERT_SCALARS:
                        keys[_expert_key(i, e, f"{proj}.{s}")] = checked
        return keys

    def load_checkpoint(self, model, checkpoint_dir) -> None:
        """Fill every declared weight from a checkpoint directory, which is what the engine
        loads: `load` over the directory's keys, every source read from its shard file."""
        directory, weight_map = checkpoint_shards(checkpoint_dir)
        self.load(model, weight_map, shards=(directory, weight_map))

    def load(self, model, weights, shards=None) -> None:
        """Fill every declared weight from the checkpoint: the shard column, multi-source rows,
        the shell's `lm_head`, and coverage asserted in both directions.

        First every routed expert's per-tensor scalars are read and checked, the other ranks'
        included, so a checkpoint that breaks what the forward relies on fails before any
        weight moves. When the checkpoint's own shard files can be found, every source is read
        from them one file at a time, this rank's slices only, and each file's cached pages are
        dropped once its sources are copied: the engine hands every rank a dict that maps every
        shard file, and reading through those mappings would fault what all of a node's ranks
        read into that node's host memory. Last, each row's `layout` puts the expert stacks into
        the MoE runner's layout, so a second `load` starts again from the stored bytes.
        """
        core = model.model
        manifest = self.manifest(core)
        consumed: set = set()

        def fill(param, ckpt, entry, index, transform, shard) -> None:
            dst = param.data if index is None else param.data[index]
            if isinstance(ckpt, tuple):
                src = transform(
                    tuple(_materialize(e, shard).to(dst.device, non_blocking=True) for e in entry),
                    core,
                )
            else:
                src = _materialize(entry, shard)
                if transform is not None:
                    # A column shard is a strided view: make it dense before a transform reads it.
                    src = transform(src.to(dst.device, non_blocking=True).contiguous(), core)
            assert tuple(dst.shape) == tuple(src.shape), (ckpt, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(_keys(ckpt))

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            sorted(set(manifest.keys()) ^ set(core.w.keys()))[:12],
        )
        d = self.dims(core)
        plan = [
            (core.w[param_key], ckpt, index, transform, shard)
            for param_key, sources in manifest.items()
            for ckpt, index, transform, shard in sources
        ]
        plan.append((model.lm_head.weight, "output.weight", None, None, self.lm_head_shard(d)))
        scalar_keys = _expert_scalar_keys(d)
        for key in (*(k for row in plan for k in _keys(row[1])), *scalar_keys):
            assert key in weights, f"checkpoint key missing: {key}"
        if shards is None:
            shards = _shard_files(core, weights)
        scalars = _read_scalars(scalar_keys, weights, shards)
        _check_expert_scalars(scalars, d)

        streamed = []
        for param, ckpt, index, transform, shard in plan:
            if all(k in scalars for k in _keys(ckpt)):
                entry = tuple(scalars[k] for k in _keys(ckpt))
                entry = entry if isinstance(ckpt, tuple) else entry[0]
                fill(param, ckpt, entry, index, transform, shard)
            else:
                assert isinstance(ckpt, str), f"a key tuple of other than expert scalars: {ckpt}"
                streamed.append((param, ckpt, index, transform, shard))
        if shards is None:
            for param, ckpt, index, transform, shard in streamed:
                fill(param, ckpt, weights[ckpt], index, transform, shard)
        else:
            directory, weight_map = shards
            by_file: dict[str, list] = collections.defaultdict(list)
            for row in streamed:
                by_file[weight_map[row[1]]].append(row)
            on_device = model.lm_head.weight.device.type == "cuda"
            for name in sorted(by_file):
                with safe_open(str(directory / name), framework="pt", device="cpu") as handle:
                    for param, ckpt, index, transform, shard in by_file[name]:
                        fill(param, ckpt, handle.get_slice(ckpt), index, transform, shard)
                if on_device:
                    torch.cuda.synchronize()
                _drop_page_cache(directory / name)
        # Every source has landed: rearrange each expert in place, one at a time.
        for entry in self._entries(d):
            if entry.layout is None:
                continue
            for key, _, _ in self._targets(entry, d):
                param = core.w[key].data
                for slot in range(param.shape[0]):
                    param[slot].copy_(entry.layout(param[slot]))
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        if model.lm_head.weight.device.type == "cuda":
            torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == set(expected), (
            "checkpoint coverage: unexpected "
            f"{sorted(leftover - set(expected))[:8]}, missed {sorted(set(expected) - leftover)[:8]}"
        )
        not_loaded = collections.Counter(expected.values())
        logger.info(
            f"modeling_v2 weights: {len(consumed)} checkpoint tensors loaded "
            f"({'streamed per shard file' if shards else 'from the given dict'}), "
            f"{len(scalars)} expert scalars checked; not loaded: "
            + "; ".join(f"{n} x {reason}" for reason, n in sorted(not_loaded.items()))
        )
