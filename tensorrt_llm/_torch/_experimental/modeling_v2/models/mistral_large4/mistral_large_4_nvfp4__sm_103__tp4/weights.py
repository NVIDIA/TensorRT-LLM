# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table and loader: mistral-large-4-nvfp4 / sm_103 / tp4, the language model.

The checkpoint is Mistral-native (`consolidated-*.safetensors`, `layers.N.attention.wq`, ...)
and ModelOpt `MIXED_PRECISION`: the routed experts NVFP4, the other big linears FP8 128x128
block (`FP8_PB_WO`), everything else bf16. Every rank is handed the whole of it, so every split
is this table's, every width read off the mapping (`tp_size` and `moe_tp_size` 4 on one node, or
8 over two nodes of four, and `moe_ep_size` 1):

* **tensor-parallel slices** -- `wq` by output rows and `wo` by input columns (this rank's 16
  heads at 4 ranks, 8 heads -- 1024 rows or columns -- at 8); `wkv_b` by heads; the MLA SSMax
  `s`/`b` by heads; the dense layer's and the shared expert's `w1`/`w3` by rows and `w2` by
  columns (6144 and 1536 per rank at 4 ranks, 3072 and 768 at 8); the shell's `lm_head` by
  vocabulary rows (51200 or 25600 per rank). Every slice is whole 128-blocks, and a block scale
  is sliced on the same boundary. The slice is taken on the checkpoint entry before the copy to
  the device, so a rank moves only its own bytes. That is the `shard` column of `W`.
* **routed-expert width slices** -- every rank loads all 128 routed experts (the window
  `[expert_offset, expert_offset + local_experts)` that `moe_ep_size` 1 makes `[0, 128)`), each
  cut along its intermediate width to `moe_inter = moe_intermediate_size / moe_tp_size` (1536 at
  4 ranks, 768 at 8) at `moe_tp_rank * moe_inter`: `w1` and `w3`, codes and block scales alike,
  by output rows `[Ir, Ir + I)`; `w2`'s codes by input bytes `[Ir/2, Ir/2 + I/2)` (two e2m1 per
  byte: 768 or 384 bytes) and its block scales by columns `[Ir/16, Ir/16 + I/16)` (one per 16
  inputs, 96 or 48 columns, so the cut splits no block). These are `shard`s too: a row read, and
  for `w2` a strided column read, of the checkpoint entry.
* **replicated** -- `wkv_a_with_mqa` and `kv_a_norm`, the whole indexer (all 32 of its heads),
  the router, the embedding, every norm, `lerope.alpha`, and the routed experts' per-tensor
  scalars.

Layer sets come from the config: `wkv_a_with_mqa` and `kv_a_norm` exist on the KV-owning layers
only (`kv_source_layers[i] == i`; the sharers read their source's latent), the indexer on the
`index_sharing == 'full'` layers only, the dense MLP on the first `first_k_dense_replace`
layers and the router, shared expert and routed experts on the rest.

Load-time layouts, each the one its consumer reads:

* `wkv_b` regrouped per head `[k_nope | v]` -> `[all k_nope ; all v]` over this rank's heads
  (k_nope is 64 wide on RoPE layers and 128 on NoPE layers);
* the dense MLP and the shared expert `[w1 ; w3]`, gate rows first, the half order
  `flashinfer_silu_and_mul` reads;
* FP8 linears (`wq`, `wo`, the indexer's `wq_b`, the dense and shared `w1`/`w3`/`w2`): e4m3
  `[N, K]` as stored, and the fp32 block scales -- stored as `.weight_scale`
  `[N/128, 1, K/128, 1]`, squeezed here to the `[N/128, K/128]` grid -- in DeepGEMM's packed
  UE8M0 operand, `transform_sf_into_required_layout(recipe=(1, 128, 128))`: what
  `FP8BlockScalesLinearMethod.transform_weights` installs as `weight_scale` on sm_100f. That
  operand is an MN-major `[N, ceil(K/512)]` int32 view; the parameter holds the dense
  `[ceil(K/512), N]` behind it and the GEMM takes `.t()`. Where K/128 is not a multiple of four
  the last word of each row is partly padding: the shared expert's `w2` at K 768 (8 ranks) has 6
  blocks in 2 words, and bytes 2 and 3 of the second word are zero. The method's
  `resmooth_to_fp8_e8m0` is not run: every scale is asserted an exact power of two, which makes
  it a power-of-two re-exponent of the same values;
* the indexer's `wk` and `weights_proj` stacked into one `[128 + 32, hidden]` bf16 GEMM operand;
  its `k_norm` and `lerope.alpha` widened to fp32.

The routed experts are this rank's width slice of every expert, stacked on an expert axis, in
the NVFP4 MoE runner's layout: e2m1 codes, two per byte, `[E, 2I, H/2]` / `[E, H, I/2]` uint8
and their e4m3 block scales, one per 16 inputs, `[E, 2I, H/16]` / `[E, H, I/16]`, with `E` 128
and `I` this rank's 1536 (4 ranks) or 768 (8 ranks). Each sliced source lands as stored -- FC1's
rows `[w3 ; w1]`, up rows first, the order the runner's FC1 concatenates them in -- and once
every source of an expert stack has landed, its row's `layout` rearranges each expert's slice in
place (the runner reads nothing else, and a wrong layout costs 300-3400 bf16 ulp and raises
nothing):

* FC1, codes and block scales alike: the halves interleaved (row `2i` up row `i`, row `2i + 1`
  gate row `i`), then the 32-row block shuffle (within each 32 rows, row `4u + v` moves to
  `8v + u`);
* FC2: the 32-row block shuffle;
* both block scales: then the 128x4 swizzle (`torch.ops.trtllm.block_scale_interleave`'s order,
  written here in torch so the table also runs on meta tensors).

The layout is a function of the slice's shape, so it runs on the slice and never on a whole
expert: the swizzle's byte order depends on the scale column count (FC2's 96 or 48 here, not
the stored 384), and the interleave pairs this rank's up and gate rows. At either per-rank width
every rule is met without padding: FC1's 2 x 1536 = 3072 or 2 x 768 = 1536 rows and FC2's 6144
are multiples of the 32-row shuffle block and of the swizzle's 128 rows, and FC1's 384 and FC2's
96 or 48 scale columns of its 4.

The runner's output scalars are not derived here: the four per-tensor scalars are handed over
as stored, unsliced -- `isc1` and `isc2`, the one FC1 and FC2 activation `input_scale` of the
layer, and per local expert `ws2_13` (`w1`'s `weight_scale_2`, which `w3` shares) and `ws2_2`.
ModelOpt stores these as reciprocals of the global scales a quantizer divides by. A per-tensor
scale holds for every slice of its tensor, so each rank's slice dequantizes with the whole
expert's scalars.

Before any of that is read, `load` reads the per-tensor scalars of all the layer's experts and
asserts what the forward relies on: `w1` and `w3` store the same `input_scale` and
`weight_scale_2` on every expert, so one FC1 alpha serves the stacked `[w3 ; w1]`; and each
projection stores one `input_scale` for every expert, so one NVFP4 quantization of the
replicated activations is right for every expert on every rank, and the ranks' partial
outputs, each over its slice of every expert, sum.

The BF16 vision encoder, pre-projector norm, patch merger and language adapter are
replicated on every rank when the configuration declares a vision encoder.

`expected_unconsumed` accounts for off-rank routed experts and disabled DSpark draft weights.
Off-rank expert scalars are still read to check shared scales; their codes and block scales
are never read.
At `moe_ep_size` 1 every expert belongs to the rank and is read, sliced.
"""

from __future__ import annotations

import collections
import functools
import json
import os
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
    entry: this rank's tensor-parallel slice, applied to every source of the row.
    It runs before the copy to the device and before `transform`, which is what
    `wkv_b` relies on: the regroup reads this rank's head count off the slice.

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


def _rows(start, width):
    return lambda d: (lambda e: e[start(d) : start(d) + width(d)])


def _cols(start, width):
    return lambda d: (lambda e: e[:, start(d) : start(d) + width(d)])


def _sf_cols(start, width):
    """`_cols` on an FP8 block scale as stored, `[N/128, 1, K/128, 1]`: its K blocks are axis 2."""
    return lambda d: (lambda e: e[:, :, start(d) : start(d) + width(d)])


def _sf_words(k: int) -> int:
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


def _block_sf(s: torch.Tensor, core) -> torch.Tensor:
    """An FP8 linear's block scales as stored, fp32 `[N/128, 1, K/128, 1]`: squeezed to the
    `[N/128, K/128]` grid, then `_deepgemm_sf`."""
    assert s.dim() == 4 and s.shape[1] == 1 and s.shape[3] == 1, (
        f"FP8 block scale stored as {tuple(s.shape)}, not [N/128, 1, K/128, 1]"
    )
    return _deepgemm_sf(s.reshape(s.shape[0], s.shape[2]), core)


def _shared_scalar(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
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


def _regroup_kv_b(t: torch.Tensor, core) -> torch.Tensor:
    """This rank's `[heads * (k_nope + v), kv_lora]` rows, per-head `[k_nope | v]` ->
    `[heads * k_nope ; heads * v]`: the K-B operand, then the V-B one. k_nope is read off the
    slice: 64 on RoPE layers, 128 on NoPE layers."""
    cfg = core.model_config.pretrained_config
    heads = cfg.num_attention_heads // core.model_config.mapping.tp_size
    v = cfg.v_head_dim
    t3 = t.reshape(heads, t.shape[0] // heads, t.shape[1])
    nope = t3.shape[1] - v
    return torch.cat(
        [t3[:, :nope, :].reshape(heads * nope, -1), t3[:, nope:, :].reshape(heads * v, -1)], dim=0
    ).contiguous()


def _to_fp32(t: torch.Tensor, core) -> torch.Tensor:
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
    ((m % 128) // 32) * 4 + c % 4`. Both of this target's scale stacks tile it exactly (M a
    multiple of 128, C of 4), so there is no padding."""
    m, c = s.shape
    assert m % SWIZZLE_ROWS == 0 and c % SWIZZLE_COLS == 0, (m, c)
    tiles = s.view(torch.uint8).reshape(m // SWIZZLE_ROWS, 4, 32, c // SWIZZLE_COLS, SWIZZLE_COLS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(m, c).view(s.dtype)


def _fc1_codes(w: torch.Tensor) -> torch.Tensor:
    return torch.index_select(w, 0, _fc1_perm(w.shape[0], w.device))


def _fc1_scales(s: torch.Tensor) -> torch.Tensor:
    rows = torch.index_select(s.view(torch.uint8), 0, _fc1_perm(s.shape[0], s.device))
    return _swizzle(rows.view(s.dtype))


def _fc2_codes(w: torch.Tensor) -> torch.Tensor:
    return torch.index_select(w, 0, _block32_perm(w.shape[0], w.device))


def _fc2_scales(s: torch.Tensor) -> torch.Tensor:
    rows = torch.index_select(s.view(torch.uint8), 0, _block32_perm(s.shape[0], s.device))
    return _swizzle(rows.view(s.dtype))


def _materialize(entry, shard) -> torch.Tensor:
    """This rank's bytes of one checkpoint entry: a tensor, or a lazy safetensors slice, which
    reads only what the index selects. `[...]`, not `[:]`: the per-tensor scalars are 0-dim."""
    return entry[...] if shard is None else shard(entry)


#: The checkpoint's shard index, beside its `consolidated-*.safetensors`.
SHARD_INDEX = "consolidated.safetensors.index.json"


def _shard_files(core, weights) -> tuple[Path, dict[str, str]] | None:
    """`(directory, key -> shard file)` of the checkpoint the engine loaded, or None when the
    sources are not the shard files of one (the directory the engine records on the config
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


def _owners(d):
    """Layers that own a latent KV. The others read `kv_source_layers[i]`'s and project none."""
    return [i for i in range(d.num_layers) if d.kv_source[i] == i]


def _rope_owners(d):
    return [i for i in _owners(d) if not d.nope_layer[i]]


def _nope_owners(d):
    return [i for i in _owners(d) if d.nope_layer[i]]


def _rope_layers(d):
    return [i for i in range(d.num_layers) if not d.nope_layer[i]]


def _nope_layers(d):
    return [i for i in range(d.num_layers) if d.nope_layer[i]]


def _indexer_layers(d):
    """Layers that own an indexer. The 'shared' ones reuse the most recent one's selection."""
    return [i for i in range(d.num_layers) if d.full_indexer[i]]


def _dense_layers(d):
    return range(d.dense_layers)


def _moe_layers(d):
    return range(d.dense_layers, d.num_layers)


# --- where the bytes come from ---


def _stack(*parts, prefix="layers"):
    """Several checkpoint tensors into one parameter: each `(key template, index(d))`, the
    index being where in the parameter that tensor lands."""
    return lambda d, i: [(t.format(p=f"{prefix}.{i}"), index(d)) for t, index in parts]


def _expert_key(i: int, e: int, suffix: str) -> str:
    return f"layers.{i}.experts.{e}.{suffix}"


def _expert_window(*parts):
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


def _window_scalar(*suffixes):
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


def _expert_rows():
    """This rank's slice of an expert's intermediate output rows: `w1` and `w3`, codes and block
    scales alike, `[moe_tp_rank * moe_inter, + moe_inter)`."""
    return _rows(lambda d: d.moe_tp_rank * d.moe_inter, lambda d: d.moe_inter)


def _expert_cols(per_column: int):
    """This rank's slice of an expert's intermediate input axis, `w2`'s, as stored with
    `per_column` inputs to a column: 2 for the e2m1 codes, 16 for the block scales."""
    width = lambda d: d.moe_inter // per_column  # noqa: E731
    return _cols(lambda d: d.moe_tp_rank * width(d), width)


def _expert_scalar_keys(d) -> list[str]:
    """Every routed expert's per-tensor scalars on every MoE layer: what `load` reads, and
    checks, before anything else."""
    return [
        _expert_key(i, e, f"{proj}.{s}")
        for i in _moe_layers(d)
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
    for i in _moe_layers(d):
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
        return {key: _materialize(weights[key], None) for key in keys}
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


def _fp8_linear(name, key, n, k, layers, w_shard=None, s_shard=None):
    """An FP8 128x128-block linear: e4m3 `[N, K]` and its DeepGEMM scale operand's storage."""
    return (
        W(name, shape=lambda d: (n(d), k(d)), dtype=F8, src=key, layers=layers, shard=w_shard),
        W(
            f"{name}_s",
            shape=lambda d: (_sf_words(k(d)), n(d)),
            dtype=I32,
            src=key.replace(".weight", ".weight_scale"),
            transform=_block_sf,
            layers=layers,
            shard=s_shard,
        ),
    )


def _tp_mlp(module, inter, layers):
    """A SwiGLU MLP split over its intermediate dimension: `[w1 ; w3]` by output rows, gate
    first (the half order `flashinfer_silu_and_mul` reads), `w2` by input columns. The same
    row slice applies to `w1` and `w3` before they are stacked, and the scales follow in
    128-blocks; in the scale storage the stacked axis is the second one."""
    blocks = lambda d: inter(d) // BLOCK  # noqa: E731
    start = lambda d: d.tp_rank * inter(d)  # noqa: E731
    b_start = lambda d: d.tp_rank * blocks(d)  # noqa: E731
    p = "{p}." + module
    return (
        W(
            "mlp_gu",
            shape=lambda d: (2 * inter(d), d.hidden),
            dtype=F8,
            src=_stack(
                (f"{p}.w1.weight", lambda d: (slice(0, inter(d)),)),
                (f"{p}.w3.weight", lambda d: (slice(inter(d), 2 * inter(d)),)),
            ),
            layers=layers,
            shard=_rows(start, inter),
        ),
        W(
            "mlp_gu_s",
            shape=lambda d: (_sf_words(d.hidden), 2 * inter(d)),
            dtype=I32,
            src=_stack(
                (f"{p}.w1.weight_scale", lambda d: (slice(None), slice(0, inter(d)))),
                (f"{p}.w3.weight_scale", lambda d: (slice(None), slice(inter(d), 2 * inter(d)))),
            ),
            transform=_block_sf,
            layers=layers,
            shard=_rows(b_start, blocks),
        ),
        *_fp8_linear(
            "mlp_dn",
            f"{p}.w2.weight",
            lambda d: d.hidden,
            inter,
            layers,
            w_shard=_cols(start, inter),
            s_shard=_sf_cols(b_start, blocks),
        ),
    )


def _kv_b(nope, layers):
    """`wkv_b`, this rank's heads, regrouped. Per-head rows are `nope(d) + v` wide."""
    width = lambda d: d.heads * (nope(d) + d.v_dim)  # noqa: E731
    return W(
        "wkv_b",
        shape=lambda d: (width(d), d.kv_lora),
        src="{p}.attention.wkv_b.weight",
        transform=_regroup_kv_b,
        layers=layers,
        shard=_rows(lambda d: d.tp_rank * width(d), width),
    )


def _heads(width):
    """This rank's slice of a per-head axis of the given per-head width."""
    return _rows(lambda d: d.tp_rank * d.heads * width(d), lambda d: d.heads * width(d))


def _vision_blocks(d):
    return list(range(d.v_layers))


def _has_vision(d):
    return d.has_vision


def _flatten_patch_conv(tensor, core):
    return tensor.flatten(1)


def _draft_layers(d):
    return range(d.draft_layers)


def _draft_source(suffix):
    return lambda d, i: [(f"mtp_dspark.backbone.layers.{i}.{suffix}", None)]


class MistralLarge4Tp4Weights:
    """mistral-large-4-nvfp4 / sm_103 / tp4."""

    WEIGHTS: tuple[W, ...] = (
        # --- MLA: every layer, KV owners only where marked ---
        W("attn_norm", shape=lambda d: (d.hidden,), src="{p}.attention_norm.weight"),
        # Per head `[nope 64 | rope 64]` on RoPE layers, 128 nope on NoPE layers: 128 either way.
        *_fp8_linear(
            "wq",
            "{p}.attention.wq.weight",
            lambda d: d.heads * d.qk_dim,
            lambda d: d.hidden,
            ALL_LAYERS,
            w_shard=_heads(lambda d: d.qk_dim),
            s_shard=_heads(lambda d: d.qk_dim // BLOCK),
        ),
        W(
            "ssmax_s",
            shape=lambda d: (d.heads,),
            src="{p}.attention.ssmax_scaling.s",
            shard=_heads(lambda d: 1),
        ),
        W(
            "ssmax_b",
            shape=lambda d: (d.heads,),
            src="{p}.attention.ssmax_scaling.b",
            shard=_heads(lambda d: 1),
        ),
        # `[c_kv 512 | k_pe 64]` on RoPE owners; NoPE owners project the 512 latent alone.
        W(
            "wkv_a",
            shape=lambda d: (d.lat_dim, d.hidden),
            src="{p}.attention.wkv_a_with_mqa.weight",
            layers=_rope_owners,
        ),
        W(
            "wkv_a",
            shape=lambda d: (d.kv_lora, d.hidden),
            src="{p}.attention.wkv_a_with_mqa.weight",
            layers=_nope_owners,
        ),
        W(
            "kv_a_norm",
            shape=lambda d: (d.kv_lora,),
            src="{p}.attention.kv_a_norm.weight",
            layers=_owners,
        ),
        # Every layer absorbs through its own `wkv_b`, KV sharers included.
        _kv_b(lambda d: d.nope, _rope_layers),
        _kv_b(lambda d: d.qk_dim, _nope_layers),
        *_fp8_linear(
            "wo",
            "{p}.attention.wo.weight",
            lambda d: d.hidden,
            lambda d: d.heads * d.v_dim,
            ALL_LAYERS,
            w_shard=_cols(lambda d: d.tp_rank * d.heads * d.v_dim, lambda d: d.heads * d.v_dim),
            s_shard=_sf_cols(
                lambda d: d.tp_rank * d.heads * d.v_dim // BLOCK,
                lambda d: d.heads * d.v_dim // BLOCK,
            ),
        ),
        W("ffn_norm", shape=lambda d: (d.hidden,), src="{p}.ffn_norm.weight"),
        # --- the DSA indexer: replicated, on the 'full' layers ---
        W(
            "ix_q_norm_w",
            shape=lambda d: (d.hidden,),
            src="{p}.attention.indexer.q_input_norm.weight",
            layers=_indexer_layers,
        ),
        W(
            "ix_q_norm_b",
            shape=lambda d: (d.hidden,),
            src="{p}.attention.indexer.q_input_norm.bias",
            layers=_indexer_layers,
        ),
        # Per head `[rope 64 | nope 64]`, the reverse of MLA's.
        *_fp8_linear(
            "ix_wq_b",
            "{p}.attention.indexer.wq_b.weight",
            lambda d: d.idx_heads * d.idx_dim,
            lambda d: d.hidden,
            _indexer_layers,
        ),
        # `[wk ; weights_proj]`: the key and the head weights, one GEMM.
        W(
            "ix_wk_wp",
            shape=lambda d: (d.idx_dim + d.idx_heads, d.hidden),
            src=_stack(
                ("{p}.attention.indexer.wk.weight", lambda d: (slice(0, d.idx_dim),)),
                (
                    "{p}.attention.indexer.weights_proj.weight",
                    lambda d: (slice(d.idx_dim, d.idx_dim + d.idx_heads),),
                ),
            ),
            layers=_indexer_layers,
        ),
        # fp32, as the fused key entry (LayerNorm in fp32) takes them.
        W(
            "ix_k_norm_w",
            shape=lambda d: (d.idx_dim,),
            dtype=F32,
            src="{p}.attention.indexer.k_norm.weight",
            transform=_to_fp32,
            layers=_indexer_layers,
        ),
        W(
            "ix_k_norm_b",
            shape=lambda d: (d.idx_dim,),
            dtype=F32,
            src="{p}.attention.indexer.k_norm.bias",
            transform=_to_fp32,
            layers=_indexer_layers,
        ),
        W(
            "ix_ssmax_s",
            shape=lambda d: (d.idx_heads,),
            src="{p}.attention.indexer.ssmax_scaling.s",
            layers=_indexer_layers,
        ),
        W(
            "ix_ssmax_b",
            shape=lambda d: (d.idx_heads,),
            src="{p}.attention.indexer.ssmax_scaling.b",
            layers=_indexer_layers,
        ),
        # --- the TP-sliced SwiGLU MLP: the dense layer's own, then each MoE layer's shared
        # expert, one key ---
        *_tp_mlp("feed_forward", lambda d: d.dense_inter, _dense_layers),
        *_tp_mlp("shared_experts", lambda d: d.shared_inter, _moe_layers),
        # --- routing ---
        W(
            "router",
            shape=lambda d: (d.num_experts, d.hidden),
            src="{p}.gate.weight",
            layers=_moe_layers,
        ),
        # --- the routed experts, NVFP4: this rank's width slice of every expert, `[w3 ; w1]` as
        # stored, then each expert's slice in the MoE runner's layout ---
        W(
            "fc1_w",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // E2M1_PER_BYTE),
            dtype=U8,
            src=_expert_window(
                ("w3.weight", lambda d: (slice(0, d.moe_inter),)),
                ("w1.weight", lambda d: (slice(d.moe_inter, 2 * d.moe_inter),)),
            ),
            layers=_moe_layers,
            shard=_expert_rows(),
            layout=_fc1_codes,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // NVFP4_BLOCK),
            dtype=F8,
            src=_expert_window(
                ("w3.weight_scale", lambda d: (slice(0, d.moe_inter),)),
                ("w1.weight_scale", lambda d: (slice(d.moe_inter, 2 * d.moe_inter),)),
            ),
            layers=_moe_layers,
            shard=_expert_rows(),
            layout=_fc1_scales,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // E2M1_PER_BYTE),
            dtype=U8,
            src=_expert_window(("w2.weight", lambda d: ())),
            layers=_moe_layers,
            shard=_expert_cols(E2M1_PER_BYTE),
            layout=_fc2_codes,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // NVFP4_BLOCK),
            dtype=F8,
            src=_expert_window(("w2.weight_scale", lambda d: ())),
            layers=_moe_layers,
            shard=_expert_cols(NVFP4_BLOCK),
            layout=_fc2_scales,
        ),
        # The per-tensor scalars, whole: the layer's one FC1 and one FC2 activation scale, and
        # each local expert's global weight scales (`w3`'s are `w1`'s).
        W(
            "isc1",
            shape=lambda d: (1,),
            dtype=F32,
            src=_window_scalar("w1.input_scale", "w3.input_scale"),
            transform=_shared_scalar,
            layers=_moe_layers,
        ),
        W(
            "isc2",
            shape=lambda d: (1,),
            dtype=F32,
            src=_window_scalar("w2.input_scale"),
            transform=_shared_scalar,
            layers=_moe_layers,
        ),
        W(
            "ws2_13",
            shape=lambda d: (d.local_experts,),
            dtype=F32,
            src=_expert_window((("w1.weight_scale_2", "w3.weight_scale_2"), lambda d: ())),
            transform=_shared_scalar,
            layers=_moe_layers,
        ),
        W(
            "ws2_2",
            shape=lambda d: (d.local_experts,),
            dtype=F32,
            src=_expert_window(("w2.weight_scale_2", lambda d: ())),
            layers=_moe_layers,
        ),
        # --- not per-layer ---
        W("final_norm", shape=lambda d: (d.hidden,), src="norm.weight", layers=None),
        W("embed", shape=lambda d: (d.vocab, d.hidden), src="tok_embeddings.weight", layers=None),
        # The LeRoPE per-frequency log-scales the rope tables are built from, fp32 as the
        # reference widens them.
        W(
            "lerope_alpha",
            shape=lambda d: (d.rope_dim // 2,),
            dtype=F32,
            src="lerope.alpha",
            transform=_to_fp32,
            layers=None,
        ),
        # BF16 vision encoder and projector, replicated on every rank.
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
            src=_stack(
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
            src=_stack(
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

    # BF16 draft weights use the same attention and dense-MLP partition as
    # this target. The RNN and target-feature projection are replicated.
    DRAFT_WEIGHTS: tuple[W, ...] = (
        W(
            "draft_attn_norm",
            lambda d: (d.hidden,),
            _draft_source("attention_norm.weight"),
            layers=_draft_layers,
        ),
        W(
            "draft_wq",
            lambda d: (d.heads * d.qk_dim, d.hidden),
            _draft_source("attention.wq.weight"),
            layers=_draft_layers,
            shard=_rows(
                lambda d: d.draft_tp_rank * d.heads * d.qk_dim, lambda d: d.heads * d.qk_dim
            ),
        ),
        W(
            "draft_wkv_a",
            lambda d: (d.lat_dim, d.hidden),
            _draft_source("attention.wkv_a_with_mqa.weight"),
            layers=_draft_layers,
        ),
        W(
            "draft_kv_a_norm",
            lambda d: (d.kv_lora,),
            _draft_source("attention.kv_a_norm.weight"),
            layers=_draft_layers,
        ),
        W(
            "draft_wkv_b",
            lambda d: (d.heads * (d.nope + d.v_dim), d.kv_lora),
            _draft_source("attention.wkv_b.weight"),
            layers=_draft_layers,
            shard=_rows(
                lambda d: d.draft_tp_rank * d.heads * (d.nope + d.v_dim),
                lambda d: d.heads * (d.nope + d.v_dim),
            ),
        ),
        W(
            "draft_wo",
            lambda d: (d.hidden, d.heads * d.v_dim),
            _draft_source("attention.wo.weight"),
            layers=_draft_layers,
            shard=_cols(lambda d: d.draft_tp_rank * d.heads * d.v_dim, lambda d: d.heads * d.v_dim),
        ),
        W(
            "draft_ffn_norm",
            lambda d: (d.hidden,),
            _draft_source("ffn_norm.weight"),
            layers=_draft_layers,
        ),
        W(
            "draft_w1",
            lambda d: (d.dense_inter, d.hidden),
            _draft_source("feed_forward.w1.weight"),
            layers=_draft_layers,
            shard=_rows(lambda d: d.draft_tp_rank * d.dense_inter, lambda d: d.dense_inter),
        ),
        W(
            "draft_w3",
            lambda d: (d.dense_inter, d.hidden),
            _draft_source("feed_forward.w3.weight"),
            layers=_draft_layers,
            shard=_rows(lambda d: d.draft_tp_rank * d.dense_inter, lambda d: d.dense_inter),
        ),
        W(
            "draft_w2",
            lambda d: (d.hidden, d.dense_inter),
            _draft_source("feed_forward.w2.weight"),
            layers=_draft_layers,
            shard=_cols(lambda d: d.draft_tp_rank * d.dense_inter, lambda d: d.dense_inter),
        ),
        W(
            "draft_final_norm",
            lambda d: (d.hidden,),
            "mtp_dspark.backbone.final_norm.weight",
            layers=None,
        ),
        W(
            "draft_target_norm",
            lambda d: (d.hidden,),
            "mtp_dspark.backbone.target_norm.weight",
            layers=None,
        ),
        W(
            "draft_target_projection",
            lambda d: (d.hidden, d.hidden * d.draft_capture_layers),
            "mtp_dspark.backbone.target_projection.weight",
            layers=None,
        ),
        W(
            "draft_mask",
            lambda d: (d.draft_depth - 1, d.hidden),
            "mtp_dspark.backbone.mask_embedding",
            layers=None,
        ),
        W(
            "draft_markov_embedding",
            lambda d: (d.vocab, d.draft_rank),
            "mtp_dspark.autoregressive_head.embedding.weight",
            layers=None,
        ),
        W(
            "draft_markov_output",
            lambda d: (d.vocab, d.draft_rank),
            "mtp_dspark.autoregressive_head.output.weight",
            layers=None,
        ),
        W(
            "draft_joint",
            lambda d: (3 * d.draft_rank, d.hidden + 2 * d.draft_rank),
            "mtp_dspark.autoregressive_head.joint_proj.weight",
            layers=None,
        ),
        W(
            "draft_joint_bias",
            lambda d: (3 * d.draft_rank,),
            "mtp_dspark.autoregressive_head.joint_proj.bias",
            layers=None,
        ),
    )

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
        for i in _moe_layers(d):
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

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint: the shard column, multi-source rows,
        the vocab-parallel `lm_head`, and coverage asserted in both directions.

        First every routed expert's per-tensor scalars are read and checked, so a checkpoint
        that breaks what the forward relies on fails before any weight moves. When the
        checkpoint's own shard files can be found, every source is read from them one file at a
        time, this rank's slices only -- of the routed experts its width slice of each, never a
        whole expert -- and each file's cached pages are dropped once its sources are copied.
        The engine hands every rank a dict that maps every shard file; reading through those
        mappings would fault a node's four ranks' slices into that node's host memory: on one
        node of four, between them every expert, 539 GiB, the whole expert stack; on each of
        two nodes, half of every expert's `w1` and `w3` and every page of every `w2`, which a
        strided column read touches whole, 359 GiB. Last, each row's `layout` puts each
        expert's slice into the MoE runner's layout, so a second `load` starts again from the
        stored bytes.
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
        # The shell builds lm_head vocab-parallel under TP: this rank's block of rows.
        d = self.dims(core)
        rows = d.vocab // d.tp_size
        plan = [
            (core.w[param_key], ckpt, index, transform, shard)
            for param_key, sources in manifest.items()
            for ckpt, index, transform, shard in sources
        ]
        plan.append(
            (
                model.lm_head.weight,
                "output.weight",
                None,
                None,
                _rows(lambda _: d.tp_rank * rows, lambda _: rows)(d),
            )
        )
        scalar_keys = _expert_scalar_keys(d)
        for key in (*(k for row in plan for k in _keys(row[1])), *scalar_keys):
            assert key in weights, f"checkpoint key missing: {key}"
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
            f"({'streamed per shard file' if shards else 'from the engine dict'}), "
            f"{len(scalars)} expert scalars checked; not loaded: "
            + "; ".join(f"{n} x {reason}" for reason, n in sorted(not_loaded.items()))
        )

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        vis = getattr(cfg, "vision_config", None)
        mapping = core.model_config.mapping
        spec = getattr(core.model_config, "spec_config", None)
        draft = getattr(spec, "_draft_model_config", None) or {}
        tp = mapping.tp_size
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        return SimpleNamespace(
            draft_layers=int(draft.get("n_layers", 0)),
            draft_rank=int((draft.get("dspark") or {}).get("markov_rank", 0)),
            draft_depth=int(draft.get("depth", 0)),
            draft_capture_layers=len(spec.target_layer_ids) if draft else 0,
            draft_tp_rank=mapping.tp_rank,
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            heads=cfg.num_attention_heads // tp,
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
            moe_inter=cfg.moe_intermediate_size // mapping.moe_tp_size,
            moe_tp_rank=mapping.moe_tp_rank,
            local_experts=local_experts,
            expert_offset=local_experts * mapping.moe_ep_rank,
            shared_inter=cfg.moe_intermediate_size * cfg.n_shared_experts // tp,
            dense_inter=cfg.intermediate_size // tp,
            vocab=cfg.vocab_size,
            tp_rank=mapping.tp_rank,
            tp_size=tp,
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


MODEL_WEIGHTS = MistralLarge4Tp4Weights()
