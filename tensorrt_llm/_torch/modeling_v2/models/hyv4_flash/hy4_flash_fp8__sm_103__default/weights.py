# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loader: hf-003107f_orig / sm_103 / unified.

Every weight role of the checkpoint's text core, declared at this rank's share.
Every share is computed once, in `dims`, from the engine's `Mapping`:

* **Attention heads** are split over the attention TP group, which is the whole
  TP group without attention DP and a group of one with it. A rank holds its
  heads' rows of `q_b_proj`, `linear_gate` and its per-head sinks, the matching
  input columns of `o_proj`, and its heads' slice of `kv_b_proj`. `q_a_proj`,
  `kv_a_proj_with_mqa` and the indexer are replicated: their outputs are the
  per-token latents and keys every head reads, and the latent KV cache is the
  same on every rank.

* **`kv_b_proj`** is relaid out into the two absorbed operands the forward reads:
  `k_b_proj [heads, nope, kv_lora]` for the query absorption and
  `v_b_proj [heads, kv_lora, v_dim]` for the latent-to-value BMM. The
  checkpoint's per-head row block is `[nope | v]` over `kv_lora` columns, so
  each operand is one half of that block, transposed into the `bmm` orientation
  `[heads, k, n]`.

* **The dense MLP** (layer 0) **and the shared expert** are split over the same
  group along their intermediate: a rank holds a slice of the gate and up rows
  and the matching down columns, and its output is a partial sum.

* **Experts** are split over the MoE EP group: a rank holds the contiguous id
  window `[expert_offset, expert_offset + local_experts)` of the packed
  `[512, 2I, H]` / `[512, H, I]` tensors. The expert intermediate is split over
  the MoE TP group. The checkpoint stores FC1 gate-first; the runner reads
  `[up ; gate]`, so the halves swap and each rank's intermediate slice is taken
  from both. The two latent projections around the experts (`routed_expert_down_proj`
  and `routed_expert_up_proj`) are never cut by the MoE TP split: the latent is the
  experts' input width and `up_proj`'s K, whole on every rank, and it is their
  partial outputs the MoE's allreduce sums.

* **Block-FP8 scales** follow their weight's cut and are promoted to fp32, the
  dtype the runner's `gemm1_weights_scale` / `gemm2_weights_scale` take; the
  checkpoint stores bf16.

* **Engram's four tables** are 31.7 GB each and never device-resident: they are
  `EngramHostTable`s in pinned host memory, filled here from the checkpoint's
  own shards and read by the gather through a device-visible mapping. Everything
  else of an Engram layer is small and replicated.

Everything else -- embedding, norms, `q_a`/`kv_a`, the hyper-connection
functions, router and bias, the latent pre/post projections -- is replicated.
lm_head is the shell's: vocab-parallel under TP, whole under attention DP;
`load` reads the rows the shell built.

**A rank reads only its share.** Each split weight carries a `cut`: the rows or
columns of the checkpoint tensor this rank holds, applied to the lazy
safetensors entry at read time, so the bytes of every other rank's share are
never read. After the cut, the transforms are the kernels' relayouts.

Loading contract: `load(model, weights)` consumes the engine-provided checkpoint
dict, fills every declared parameter exactly once, and asserts bidirectional
coverage -- every target parameter written, and every checkpoint key either
consumed or in the predicted non-load set (the DFly2 drafter's layers, which the
`dflash` unit of this flow owns; a config without a vision tower's `vit.*`).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash import engram_host
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.dims import (  # noqa: E501
    draft_dims,
)

#: `layers=ALL_LAYERS` is the common case and the default.
ALL_LAYERS = object()

#: An index into a checkpoint tensor: a tuple of slices, one per leading axis.
Cut = tuple[slice, ...]


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and the layer index (or None) and returns
    the declared shape -- this
    rank's share, every width derived from the config and the mapping.

    `src` says where the bytes come from: a checkpoint key template, or a
    callable `(d, i) -> [(key, index), ...]` taking the dimension bundle and the
    layer index. `index` is the slice of the destination parameter this source
    fills (`None` for the whole parameter); a `key` may be a tuple, meaning the
    transform is handed those tensors together.

    `cut` takes `(d, i)` and returns this rank's rows or columns of the
    checkpoint tensor, read from the lazy entry so nothing else is read; the same
    cut applies to every key of a multi-key row. `None` reads the tensor whole.
    It takes the layer index because the per-head width differs between a sparse
    layer and a windowed one.

    `transform` runs on the cut tensor(s) before the copy, on the destination's
    device. `None` means the cut is already the declared shape.
    """

    name: str
    shape: Callable[..., tuple[int, ...]]
    src: Any
    dtype: torch.dtype | None = None
    cut: Callable[[Any, int | None], Cut] | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        """The layers this weight exists on, or None if it is not per-layer."""
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None) -> list[tuple[Any, Any]]:
        """`[(checkpoint key or key tuple, index into the param | None), ...]`.

        A string source is a key template: `{i}` becomes this layer's index. A
        callable source names its own keys, already formatted.
        """
        if callable(self.src):
            return list(self.src(d, i))
        key = self.src
        if "{i}" in key:
            assert i is not None, f"{self.name} is per-layer but has no layer index"
            key = key.format(i=i)
        return [(key, None)]


def _read(entry, cut: Cut | None) -> torch.Tensor:
    """Realize one checkpoint entry, reading only `cut` of it."""
    if cut is None:
        return entry if getattr(entry, "ndim", 1) == 0 else entry[:]
    return entry[cut[0]] if len(cut) == 1 else entry[cut]


# --- cuts: this rank's block of rows or columns, read from the lazy entry ---


def _rows(block: Callable[[Any, int | None], tuple[int, int]]) -> Callable[..., Cut]:
    """Rows `[rank * span, (rank + 1) * span)`."""

    def cut(d, i=None) -> Cut:
        rank, span = block(d, i)
        return (slice(rank * span, (rank + 1) * span),)

    return cut


def _cols(block: Callable[[Any, int | None], tuple[int, int]]) -> Callable[..., Cut]:
    """Columns `[rank * span, (rank + 1) * span)`, every row."""

    def cut(d, i=None) -> Cut:
        rank, span = block(d, i)
        return (slice(None), slice(rank * span, (rank + 1) * span))

    return cut


def _expert_rows(d, i=None) -> tuple[int, int]:
    """The expert axis: `[expert_offset, expert_offset + local_experts)`."""
    return d.moe_ep_rank, d.local_experts


def _kv_b_rows(d, i=None) -> tuple[int, int]:
    """The row block of `kv_b_proj`'s `[heads * (nope + v), kv_lora]`, this rank's heads.

    A head owns `nope + v` consecutive rows, so the cut is in head blocks, not
    in heads: 8 heads at tep4 are 8 * 448 rows.
    """
    return d.attn_tp_rank, d.layers[i].heads_local * (d.nope + d.v_dim)


def _q_b_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local * d.qk_dim


def _gate_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local * d.v_dim


def _o_cols(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local * d.v_dim


def _head_scales(d, i=None) -> tuple[int, int]:
    """Scale blocks are 128 wide: a rank's head block is `heads_local * width / 128` rows."""
    return d.attn_tp_rank, d.layers[i].heads_local * d.qk_dim // 128


def _o_scale_cols(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local * d.v_dim // 128


def _gate_scale_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local * d.v_dim // 128


def _sink_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local


# --- relayouts, on the rank's cut ---


def _kv_b_k(t: torch.Tensor, core) -> torch.Tensor:
    """`[heads_local * (nope + v), kv_lora]` -> `[heads_local, nope, kv_lora]`.

    The checkpoint's per-head row block is `[nope | v]`; the query absorption
    reads the no-PE half, already in the `bmm` operand orientation.
    """
    d = core.dims
    heads = d.layers[core._loading_layer].heads_local
    return t.view(heads, d.nope + d.v_dim, d.kv_lora)[:, : d.nope].contiguous()


def _kv_b_v(t: torch.Tensor, core) -> torch.Tensor:
    """`[heads_local * (nope + v), kv_lora]` -> `[heads_local, kv_lora, v_dim]`.

    The latent-to-value BMM's operand: the same per-head block's value half,
    transposed.
    """
    d = core.dims
    heads = d.layers[core._loading_layer].heads_local
    return t.view(heads, d.nope + d.v_dim, d.kv_lora)[:, d.nope :].transpose(1, 2).contiguous()


def _kv_b_context(t: torch.Tensor, core) -> torch.Tensor:
    """`[heads_local * (nope + v), kv_lora]` -> the same rows as `[nope | v]`.

    The unabsorbed context attention reads K and V as two blocks over the whole
    head axis, `[all heads' nope][all heads' v]`, and V as the `[.., H*nope:]`
    split view of that buffer (the context FMHA hard-codes V's row stride to the
    packed width). The checkpoint's own per-head `[nope | v]` blocks are the same
    rows in a different order, so this is a permutation, never a value change.
    """
    d = core.dims
    heads = d.layers[core._loading_layer].heads_local
    view = t.view(heads, d.nope + d.v_dim, d.kv_lora)
    return torch.cat(
        [
            view[:, : d.nope].reshape(heads * d.nope, d.kv_lora),
            view[:, d.nope :].reshape(heads * d.v_dim, d.kv_lora),
        ],
        dim=0,
    ).contiguous()


def _experts_fc1(t: torch.Tensor, core) -> torch.Tensor:
    """`[E_loc, 2I, H]` gate-first e4m3 -> `[E_loc, 2I/parts, H]` up-first, this rank's I slice."""
    d = core.dims
    e, two_i, h = t.shape
    i = two_i // 2
    view = t.view(e, 2, i, h)
    gate, up = view[:, 0], view[:, 1]
    span = i // d.moe_tp
    lo = d.moe_tp_rank * span
    return torch.cat([up[:, lo : lo + span], gate[:, lo : lo + span]], dim=1).contiguous()


def _experts_fc1_scale(t: torch.Tensor, core) -> torch.Tensor:
    """`[E_loc, 2I/128, H/128]` bf16 gate-first -> fp32 up-first, this rank's slice."""
    d = core.dims
    e, two_b, hb = t.shape
    b = two_b // 2
    view = t.view(e, 2, b, hb)
    gate, up = view[:, 0], view[:, 1]
    span = b // d.moe_tp
    lo = d.moe_tp_rank * span
    return torch.cat([up[:, lo : lo + span], gate[:, lo : lo + span]], dim=1).float().contiguous()


def _experts_fc2(t: torch.Tensor, core) -> torch.Tensor:
    """`[E_loc, H, I]` -> `[E_loc, H, I/parts]`, this rank's intermediate columns."""
    d = core.dims
    span = t.shape[2] // d.moe_tp
    lo = d.moe_tp_rank * span
    return t[:, :, lo : lo + span].contiguous()


def _experts_fc2_scale(t: torch.Tensor, core) -> torch.Tensor:
    """`[E_loc, H/128, I/128]` bf16 -> fp32, this rank's intermediate columns."""
    d = core.dims
    span = t.shape[2] // d.moe_tp
    lo = d.moe_tp_rank * span
    return t[:, :, lo : lo + span].float().contiguous()


def _fp32(t: torch.Tensor, core) -> torch.Tensor:
    return t.float().contiguous()


def _to_dtype(t: torch.Tensor, core) -> torch.Tensor:
    return t.to(core.dims.dtype).contiguous()


# --- the vision tower's fused q|k|v ---


def _vis_qkv(d, i) -> list[tuple[str, slice]]:
    """A vision block's three projection weights as the slices of one fused parameter.

    `thop_attention`'s `is_fused_qkv` reads one `[T, 3 * heads, head_dim]` block: q's rows
    first, then k's, then v's. The checkpoint stores the three projections apart, so each
    fills its own third of the fused rows.
    """
    hidden = d.vision.hidden
    prefix = f"vit.layers.{i}.self_attn"
    return [
        (f"{prefix}.q_proj.weight", slice(0, hidden)),
        (f"{prefix}.k_proj.weight", slice(hidden, 2 * hidden)),
        (f"{prefix}.v_proj.weight", slice(2 * hidden, 3 * hidden)),
    ]


def _vis_qkv_b(d, i) -> list[tuple[str, slice]]:
    """The same three projections' biases, packed into the fused bias."""
    hidden = d.vision.hidden
    prefix = f"vit.layers.{i}.self_attn"
    return [
        (f"{prefix}.q_proj.bias", slice(0, hidden)),
        (f"{prefix}.k_proj.bias", slice(hidden, 2 * hidden)),
        (f"{prefix}.v_proj.bias", slice(2 * hidden, 3 * hidden)),
    ]


# --- layer predicates ---


def _sparse_layers(d) -> list[int]:
    return list(d.sparse_layers)


def _dense_layers(d) -> list[int]:
    return list(range(d.first_dense))


def _moe_layers(d) -> list[int]:
    return list(range(d.first_dense, d.num_layers))


def _engram_layers(d) -> list[int]:
    return list(d.engram_layers)


def _vision_layers(d) -> list[int]:
    """The tower's 27 blocks; empty when the config publishes no vision tower."""
    return [] if d.vision is None else list(range(d.vision.depth))


def _is_fp8_layer(d, i: int) -> bool:
    """Layers past the first dense one are block-FP8 on `q_b`/`o`/`gate`; layer 0 is bf16."""
    return i >= d.first_dense


class _WeightTable:
    """The mechanics every weight table of this target shares.

    A table is a tuple of `W` roles plus the three things that read them: what a
    role allocates at this rank, which checkpoint keys fill it, and what those
    keys become on the way. All three are written against a *host* -- any object
    whose `dims` is the dimension bundle the table was declared over, plus
    whatever the table's transforms read (`_loading_layer` for the core's
    per-layer relayouts). The core's table is hosted by the model's core; the
    drafter's, by the bundle `draft_dims` returns.
    """

    WEIGHTS: tuple[W, ...] = ()

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares, at this rank's share.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives when
        the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, i in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*self._shape(entry, d, i), dtype=dtype), requires_grad=False
                )
        return w

    @staticmethod
    def _shape(entry: W, d: SimpleNamespace, i: int | None) -> tuple[int, ...]:
        """The declared shape at layer `i`, or without one for a global weight."""
        return entry.shape(d) if i is None else entry.shape(d, i)

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        """`[(param key, layer index | None), ...]`."""
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None)]
        return [(f"l{i}_{entry.name}", i) for i in layers]

    def manifest(self, core) -> dict[str, list[tuple[Any, Any, Cut | None, Callable | None]]]:
        """param key -> [(checkpoint key, index into the param | None, cut | None,
        transform | None)]."""
        d = core.dims
        rows: dict[str, list[tuple[Any, Any, Cut | None, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i in self._targets(entry, d):
                cut = entry.cut(d, i) if entry.cut is not None else None
                sources = []
                for ckpt, index in entry.sources(d, i):
                    sources.append((ckpt, index, cut, entry.transform))
                rows[key] = sources
        return rows

    def read(self, weights, ckpt_key, cut: Cut | None, transform, core, device=None):
        """One manifest source -> the tensor its destination receives."""
        keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
        for key in keys:
            assert key in weights, f"checkpoint key missing: {key}"
        src_all = tuple(_read(weights[key], cut) for key in keys)
        if device is not None:
            src_all = tuple(t.to(device, non_blocking=True) for t in src_all)
        if transform is not None:
            return transform(src_all if isinstance(ckpt_key, tuple) else src_all[0], core)
        assert len(src_all) == 1, "a multi-key row needs a transform"
        return src_all[0]


class Hyv4FlashUnifiedWeights(_WeightTable):
    """HY4-Flash's text-core weights at one rank's share of one deployment."""

    #: Release cached blocks after a parameter whose relayout allocates per-expert scratch.
    RELEASE_AFTER = "experts_down_scale"

    WEIGHTS: tuple[W, ...] = (
        # --- hyper-connections (fp32, never quantized) ---
        W(
            "hc_head_fn",
            lambda d, i=None: (d.hc_mult, d.hidden * d.hc_mult),
            "model.hc_head.hc_head_fn",
            dtype=torch.float32,
            layers=None,
        ),
        W(
            "hc_head_base",
            lambda d, i=None: (d.hc_mult,),
            "model.hc_head.hc_head_base",
            dtype=torch.float32,
            layers=None,
        ),
        W(
            "hc_head_scale",
            lambda d, i=None: (1,),
            "model.hc_head.hc_head_scale",
            dtype=torch.float32,
            layers=None,
        ),
        W(
            "hc_attn_fn",
            lambda d, i=None: (2 * d.hc_mult, d.hidden * d.hc_mult),
            "model.layers.{i}.hc_attn_layer.hc_pre.hc_fn",
            dtype=torch.float32,
        ),
        W(
            "hc_attn_base",
            lambda d, i=None: (2 * d.hc_mult,),
            "model.layers.{i}.hc_attn_layer.hc_pre.hc_base",
            dtype=torch.float32,
        ),
        W(
            "hc_attn_scale",
            lambda d, i=None: (2,),
            "model.layers.{i}.hc_attn_layer.hc_pre.hc_scale",
            dtype=torch.float32,
        ),
        W(
            "hc_mlp_fn",
            lambda d, i=None: (2 * d.hc_mult, d.hidden * d.hc_mult),
            "model.layers.{i}.hc_mlp_layer.hc_pre.hc_fn",
            dtype=torch.float32,
        ),
        W(
            "hc_mlp_base",
            lambda d, i=None: (2 * d.hc_mult,),
            "model.layers.{i}.hc_mlp_layer.hc_pre.hc_base",
            dtype=torch.float32,
        ),
        W(
            "hc_mlp_scale",
            lambda d, i=None: (2,),
            "model.layers.{i}.hc_mlp_layer.hc_pre.hc_scale",
            dtype=torch.float32,
        ),
        # --- the two sandwich norms of a layer ---
        W(
            "input_layernorm",
            lambda d, i=None: (d.hidden,),
            "model.layers.{i}.input_layernorm.weight",
        ),
        W(
            "post_attention_layernorm",
            lambda d, i=None: (d.hidden,),
            "model.layers.{i}.post_attention_layernorm.weight",
        ),
        # --- MLA: replicated latents, head-split expansions ---
        W(
            "q_a_proj",
            lambda d, i=None: (d.q_lora, d.hidden),
            "model.layers.{i}.self_attn.q_a_proj.weight",
        ),
        W(
            "q_a_layernorm",
            lambda d, i=None: (d.q_lora,),
            "model.layers.{i}.self_attn.q_a_layernorm.weight",
        ),
        W(
            "kv_a_proj_with_mqa",
            lambda d, i=None: (d.lat_dim, d.hidden),
            "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
        ),
        W(
            "kv_a_layernorm",
            lambda d, i=None: (d.kv_lora,),
            "model.layers.{i}.self_attn.kv_a_layernorm.weight",
        ),
        W(
            "k_b_proj",
            lambda d, i=None: (d.layers[i].heads_local, d.nope, d.kv_lora),
            "model.layers.{i}.self_attn.kv_b_proj.weight",
            cut=_rows(_kv_b_rows),
            transform=_kv_b_k,
        ),
        W(
            "v_b_proj",
            lambda d, i=None: (d.layers[i].heads_local, d.kv_lora, d.v_dim),
            "model.layers.{i}.self_attn.kv_b_proj.weight",
            cut=_rows(_kv_b_rows),
            transform=_kv_b_v,
        ),
        #: The same rows again, ordered for the unabsorbed context attention: one
        #: `[T, heads_local*(nope + v)]` GEMM, then K's no-PE half and V's split
        #: view are both slices of it. Read by the context phase only.
        W(
            "kv_b_proj",
            lambda d, i=None: (d.layers[i].heads_local * (d.nope + d.v_dim), d.kv_lora),
            "model.layers.{i}.self_attn.kv_b_proj.weight",
            cut=_rows(_kv_b_rows),
            transform=_kv_b_context,
        ),
        W(
            "post_proj_norm",
            lambda d, i=None: (d.hidden,),
            "model.layers.{i}.self_attn.post_proj_norm.weight",
        ),
        W(
            "sink",
            lambda d, i=None: (d.layers[i].heads_local,),
            "model.layers.{i}.self_attn.learnable_sink_param",
            dtype=torch.float32,
            cut=_rows(_sink_rows),
        ),
        # --- MLA: block-FP8 on every layer but the first dense one ---
        W(
            "q_b_proj",
            lambda d, i=None: (d.layers[i].heads_local * d.qk_dim, d.q_lora),
            "model.layers.{i}.self_attn.q_b_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_q_b_rows),
            layers=_moe_layers,
        ),
        W(
            "q_b_proj_scale",
            lambda d, i=None: (d.layers[i].heads_local * d.qk_dim // 128, d.q_lora // 128),
            "model.layers.{i}.self_attn.q_b_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_head_scales),
            transform=_fp32,
            layers=_moe_layers,
        ),
        W(
            "q_b_proj",
            lambda d, i=None: (d.layers[i].heads_local * d.qk_dim, d.q_lora),
            "model.layers.{i}.self_attn.q_b_proj.weight",
            cut=_rows(_q_b_rows),
            layers=_dense_layers,
        ),
        W(
            "linear_gate",
            lambda d, i=None: (d.layers[i].heads_local * d.v_dim, d.hidden),
            "model.layers.{i}.self_attn.linear_gate.weight",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_gate_rows),
            layers=_moe_layers,
        ),
        W(
            "linear_gate_scale",
            lambda d, i=None: (d.layers[i].heads_local * d.v_dim // 128, d.hidden // 128),
            "model.layers.{i}.self_attn.linear_gate.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_gate_scale_rows),
            transform=_fp32,
            layers=_moe_layers,
        ),
        W(
            "linear_gate",
            lambda d, i=None: (d.layers[i].heads_local * d.v_dim, d.hidden),
            "model.layers.{i}.self_attn.linear_gate.weight",
            cut=_rows(_gate_rows),
            layers=_dense_layers,
        ),
        W(
            "o_proj",
            lambda d, i=None: (d.hidden, d.layers[i].heads_local * d.v_dim),
            "model.layers.{i}.self_attn.o_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_cols(_o_cols),
            layers=_moe_layers,
        ),
        W(
            "o_proj_scale",
            lambda d, i=None: (d.hidden // 128, d.layers[i].heads_local * d.v_dim // 128),
            "model.layers.{i}.self_attn.o_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_cols(_o_scale_cols),
            transform=_fp32,
            layers=_moe_layers,
        ),
        W(
            "o_proj",
            lambda d, i=None: (d.hidden, d.layers[i].heads_local * d.v_dim),
            "model.layers.{i}.self_attn.o_proj.weight",
            cut=_cols(_o_cols),
            layers=_dense_layers,
        ),
        # --- compressed indexer (the twelve sparse layers) ---
        W(
            "idx_query",
            lambda d, i=None: (d.index_n_heads * d.index_head_dim, d.q_lora),
            "model.layers.{i}.self_attn.compressed_indexer.query_projection.weight",
            layers=_sparse_layers,
        ),
        W(
            "idx_head_weight",
            lambda d, i=None: (d.index_n_heads, d.hidden),
            "model.layers.{i}.self_attn.compressed_indexer.head_weight_projection.weight",
            layers=_sparse_layers,
        ),
        W(
            "idx_token_key",
            lambda d, i=None: (d.index_head_dim, d.hidden),
            "model.layers.{i}.self_attn.compressed_indexer.token_key_projection.weight",
            layers=_sparse_layers,
        ),
        W(
            "idx_token_key_norm",
            lambda d, i=None: (d.index_head_dim,),
            "model.layers.{i}.self_attn.compressed_indexer.token_key_norm.weight",
            layers=_sparse_layers,
        ),
        W(
            "idx_kv_proj",
            lambda d, i=None: (d.index_head_dim, d.index_head_dim),
            "model.layers.{i}.self_attn.compressed_indexer.key_value_projection.weight",
            dtype=torch.float32,
            layers=_sparse_layers,
        ),
        W(
            "idx_gate_proj",
            lambda d, i=None: (d.index_head_dim, d.index_head_dim),
            "model.layers.{i}.self_attn.compressed_indexer.key_gate_projection.weight",
            dtype=torch.float32,
            layers=_sparse_layers,
        ),
        W(
            "idx_key_norm",
            lambda d, i=None: (d.index_head_dim,),
            "model.layers.{i}.self_attn.compressed_indexer.key_norm.weight",
            dtype=torch.float32,
            layers=_sparse_layers,
        ),
        W(
            "idx_ape",
            lambda d, i=None: (d.index_kpool, d.index_head_dim),
            "model.layers.{i}.self_attn.compressed_indexer.intra_group_ape",
            dtype=torch.float32,
            layers=_sparse_layers,
        ),
        # --- layer-0 dense MLP ---
        W(
            "dense_gate",
            lambda d, i=None: (d.dense_inter, d.hidden),
            "model.layers.{i}.mlp.gate_proj.weight",
            cut=_rows(lambda d, i=None: (d.mlp_tp_rank, d.dense_inter)),
            layers=_dense_layers,
        ),
        W(
            "dense_up",
            lambda d, i=None: (d.dense_inter, d.hidden),
            "model.layers.{i}.mlp.up_proj.weight",
            cut=_rows(lambda d, i=None: (d.mlp_tp_rank, d.dense_inter)),
            layers=_dense_layers,
        ),
        W(
            "dense_down",
            lambda d, i=None: (d.hidden, d.dense_inter),
            "model.layers.{i}.mlp.down_proj.weight",
            cut=_cols(lambda d, i=None: (d.mlp_tp_rank, d.dense_inter)),
            layers=_dense_layers,
        ),
        W(
            "post_fc2_norm",
            lambda d, i=None: (d.hidden,),
            "model.layers.{i}.mlp.post_fc2_norm.weight",
            layers=_dense_layers,
        ),
        # --- MoE: router, latent pre/post projection, shared expert, packed experts ---
        W(
            "router",
            lambda d, i=None: (d.num_experts, d.hidden),
            "model.layers.{i}.mlp.gate.weight",
            layers=_moe_layers,
        ),
        W(
            "router_bias",
            lambda d, i=None: (d.num_experts,),
            "model.layers.{i}.mlp.gate.e_score_correction_bias",
            dtype=torch.float32,
            layers=_moe_layers,
        ),
        W(
            "pre_expert",
            lambda d, i=None: (d.latent, d.hidden),
            "model.layers.{i}.mlp.routed_expert_down_proj.weight",
            layers=_moe_layers,
        ),
        # `post_expert`'s K is the latent, whole on every rank: the MoE TP split
        # falls on the experts' intermediate, so the pair reads the checkpoint
        # tensor entire whatever `moe_tp` is, and the allreduce completes the
        # sum of the ranks' partial outputs.
        W(
            "post_expert",
            lambda d, i=None: (d.hidden, d.latent),
            "model.layers.{i}.mlp.routed_expert_up_proj.weight",
            layers=_moe_layers,
        ),
        W(
            "shared_gate",
            lambda d, i=None: (d.shared_inter, d.hidden),
            "model.layers.{i}.mlp.shared_experts.gate_proj.weight",
            cut=_rows(lambda d, i=None: (d.mlp_tp_rank, d.shared_inter)),
            layers=_moe_layers,
        ),
        W(
            "shared_up",
            lambda d, i=None: (d.shared_inter, d.hidden),
            "model.layers.{i}.mlp.shared_experts.up_proj.weight",
            cut=_rows(lambda d, i=None: (d.mlp_tp_rank, d.shared_inter)),
            layers=_moe_layers,
        ),
        W(
            "shared_down",
            lambda d, i=None: (d.hidden, d.shared_inter),
            "model.layers.{i}.mlp.shared_experts.down_proj.weight",
            cut=_cols(lambda d, i=None: (d.mlp_tp_rank, d.shared_inter)),
            layers=_moe_layers,
        ),
        W(
            "combine_output_norm",
            lambda d, i=None: (d.hidden,),
            "model.layers.{i}.mlp.combine_output_norm.weight",
            layers=_moe_layers,
        ),
        W(
            "experts_gate_up",
            lambda d, i=None: (d.local_experts, 2 * d.moe_inter, d.latent),
            "model.layers.{i}.mlp.experts.gate_up_proj",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_expert_rows),
            transform=_experts_fc1,
            layers=_moe_layers,
        ),
        W(
            "experts_gate_up_scale",
            lambda d, i=None: (d.local_experts, 2 * d.moe_inter // 128, d.latent // 128),
            "model.layers.{i}.mlp.experts.gate_up_proj_scale_inv",
            dtype=torch.float32,
            cut=_rows(_expert_rows),
            transform=_experts_fc1_scale,
            layers=_moe_layers,
        ),
        W(
            "experts_down",
            lambda d, i=None: (d.local_experts, d.latent, d.moe_inter),
            "model.layers.{i}.mlp.experts.down_proj",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_expert_rows),
            transform=_experts_fc2,
            layers=_moe_layers,
        ),
        W(
            "experts_down_scale",
            lambda d, i=None: (d.local_experts, d.latent // 128, d.moe_inter // 128),
            "model.layers.{i}.mlp.experts.down_proj_scale_inv",
            dtype=torch.float32,
            cut=_rows(_expert_rows),
            transform=_experts_fc2_scale,
            layers=_moe_layers,
        ),
        # --- Engram: everything but the four host tables (see `_engram_tables`) ---
        W(
            "engram_lookup",
            lambda d, i=None: (d.engram_table_size,),
            "model.layers.{i}.engram_layer.lookup_table",
            dtype=torch.int32,
            layers=_engram_layers,
        ),
        W(
            "engram_key_w",
            lambda d, i=None: (d.hc_mult, d.hidden, d.engram_size),
            lambda d, i: [
                (f"model.layers.{i}.engram_layer.key_projs.{k}.weight", k) for k in range(d.hc_mult)
            ],
            layers=_engram_layers,
        ),
        W(
            "engram_key_b",
            lambda d, i=None: (d.hc_mult, d.hidden),
            lambda d, i: [
                (f"model.layers.{i}.engram_layer.key_projs.{k}.bias", k) for k in range(d.hc_mult)
            ],
            layers=_engram_layers,
        ),
        W(
            "engram_norm_key",
            lambda d, i=None: (d.hc_mult, d.hidden),
            lambda d, i: [
                (f"model.layers.{i}.engram_layer.norm_key.{k}.weight", k) for k in range(d.hc_mult)
            ],
            layers=_engram_layers,
        ),
        W(
            "engram_norm_query",
            lambda d, i=None: (d.hc_mult, d.hidden),
            lambda d, i: [
                (f"model.layers.{i}.engram_layer.norm_query.{k}.weight", k)
                for k in range(d.hc_mult)
            ],
            layers=_engram_layers,
        ),
        W(
            "engram_head_norm",
            lambda d, i=None: (d.engram_num_heads, d.engram_head_dim),
            lambda d, i: [
                (f"model.layers.{i}.engram_layer.lookup_per_head_norms.{k}.weight", k)
                for k in range(d.engram_num_heads)
            ],
            layers=_engram_layers,
        ),
        W(
            "engram_value_w",
            lambda d, i=None: (d.hidden, d.engram_size),
            "model.layers.{i}.engram_layer.value_proj.weight",
            layers=_engram_layers,
        ),
        W(
            "engram_value_b",
            lambda d, i=None: (d.hidden,),
            "model.layers.{i}.engram_layer.value_proj.bias",
            layers=_engram_layers,
        ),
        W(
            "engram_pre_gamma",
            lambda d, i=None: (d.hidden,),
            "model.layers.{i}.engram_layer.unified_pre_gamma",
            layers=_engram_layers,
        ),
        W(
            "engram_hc_scale",
            lambda d, i=None: (d.hc_mult, d.hidden),
            lambda d, i: [
                (f"model.layers.{i}.engram_layer.unified_per_hc_scales.{k}", k)
                for k in range(d.hc_mult)
            ],
            layers=_engram_layers,
        ),
        # --- not per-layer ---
        W("final_norm", lambda d, i=None: (d.hidden,), "model.norm.weight", layers=None),
        W("embed", lambda d, i=None: (d.vocab, d.hidden), "model.embed_tokens.weight", layers=None),
        # --- the vision tower: `vit.*`, bf16 even in this FP8 checkpoint -------------
        # Whole on every rank (the tower is replicated), read by the shell's own
        # `vision.VisionTower`. Shapes come from `d.vision`, so a config without a
        # vision tower declares none of these and leaves `vit.*` unconsumed.
        W(
            "vis_patch_w",
            lambda d, i=None: (d.vision.hidden, d.vision.channels, d.vision.patch, d.vision.patch),
            "vit.embeddings.patch_embedding.weight",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_patch_b",
            lambda d, i=None: (d.vision.hidden,),
            "vit.embeddings.patch_embedding.bias",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_pos",
            lambda d, i=None: (d.vision.position_edge**2 + 1, d.vision.hidden),
            "vit.embeddings.position_embedding.weight",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_m_norm",
            lambda d, i=None: (d.vision.hidden,),
            "vit.perceive.norm.weight",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_m_conv_w",
            lambda d, i=None: (d.vision.merge_dim, d.vision.hidden, d.vision.merge, d.vision.merge),
            "vit.perceive.proj.0.weight",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_m_conv_b",
            lambda d, i=None: (d.vision.merge_dim,),
            "vit.perceive.proj.0.bias",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_m_out_w",
            lambda d, i=None: (d.vision.out_hidden, d.vision.merge_dim, 1, 1),
            "vit.perceive.proj.2.weight",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_m_out_b",
            lambda d, i=None: (d.vision.out_hidden,),
            "vit.perceive.proj.2.bias",
            dtype=torch.bfloat16,
            layers=None,
        ),
        W(
            "vis_m_newline",
            lambda d, i=None: (d.vision.out_hidden,),
            "vit.perceive.image_newline",
            dtype=torch.bfloat16,
            layers=None,
        ),
        # A block's two LayerNorms have biases (the merger's RMSNorm does not), and its
        # attention and FFN are `nn.Linear`s: one `[out, in]` weight and one bias each.
        W(
            "vis_n1_w",
            lambda d, i=None: (d.vision.hidden,),
            "vit.layers.{i}.input_layernorm.weight",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_n1_b",
            lambda d, i=None: (d.vision.hidden,),
            "vit.layers.{i}.input_layernorm.bias",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_n2_w",
            lambda d, i=None: (d.vision.hidden,),
            "vit.layers.{i}.post_attention_layernorm.weight",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_n2_b",
            lambda d, i=None: (d.vision.hidden,),
            "vit.layers.{i}.post_attention_layernorm.bias",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        # The checkpoint's separate q|k|v projections, packed into the one row block
        # `is_fused_qkv` reads: q's rows, then k's, then v's. Three sources fill the
        # one parameter's three slices.
        W(
            "vis_qkv",
            lambda d, i=None: (3 * d.vision.hidden, d.vision.hidden),
            _vis_qkv,
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_qkv_b",
            lambda d, i=None: (3 * d.vision.hidden,),
            _vis_qkv_b,
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_proj_w",
            lambda d, i=None: (d.vision.hidden, d.vision.hidden),
            "vit.layers.{i}.self_attn.o_proj.weight",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_proj_b",
            lambda d, i=None: (d.vision.hidden,),
            "vit.layers.{i}.self_attn.o_proj.bias",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_fc1_w",
            lambda d, i=None: (d.vision.inter, d.vision.hidden),
            "vit.layers.{i}.mlp.dense_h_to_4h.weight",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_fc1_b",
            lambda d, i=None: (d.vision.inter,),
            "vit.layers.{i}.mlp.dense_h_to_4h.bias",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_fc2_w",
            lambda d, i=None: (d.vision.hidden, d.vision.inter),
            "vit.layers.{i}.mlp.dense_4h_to_h.weight",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
        W(
            "vis_fc2_b",
            lambda d, i=None: (d.vision.hidden,),
            "vit.layers.{i}.mlp.dense_4h_to_h.bias",
            dtype=torch.bfloat16,
            layers=_vision_layers,
        ),
    )

    def engram_tables(self, core) -> dict[int, "engram_host.EngramHostTable"]:
        """One host-resident table per Engram layer, keyed by layer index.

        Built at declare time (it reserves this rank's chunks and the whole
        table's device VA), filled in `load`. The table is 31.7 GB, so it is not
        a parameter and is never device-resident: the forward gathers through it.
        """
        d = core.dims
        device = torch.device("cuda", torch.cuda.current_device())
        tables = {}
        for i in d.engram_layers:
            tables[i] = engram_host.EngramHostTable(
                rows=d.engram_table_rows[i],
                cols=d.engram_head_dim,
                dtype=d.dtype,
                device=device,
                rank=d.rank,
                world=d.world,
                name=f"model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight",
            )
        return tables

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint."""
        core = model.model
        d = core.dims
        manifest = self.manifest(core)
        consumed: set = set()

        def fill(param: torch.nn.Parameter, ckpt_key, index, cut, transform) -> None:
            dst = param.data if index is None else param.data[index]
            src = self.read(weights, ckpt_key, cut, transform, core, dst.device)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,))

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            layer = None
            if param_key.startswith("l"):
                layer = int(param_key[1 : param_key.index("_")])
            core._loading_layer = layer if layer is not None else 0
            for ckpt_key, index, cut, transform in sources:
                fill(core.w[param_key], ckpt_key, index, cut, transform)
            if param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # The four Engram tables: 126.7 GB of host memory, filled from the
        # checkpoint's own shards through the family's pinned-host mapping.
        for i, table in core.engram_tables.items():
            key = f"model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight"
            assert key in weights, f"checkpoint key missing: {key}"
            table.fill(
                *engram_host.shard_rows(
                    weights, [key], d.engram_table_rows[i], lambda entry, cut: _read(entry, cut)
                )
            )
            consumed.add(key)

        # The shell builds lm_head vocab-parallel under TP and whole under
        # attention DP; read the rows it built and nothing else.
        mapping = core.model_config.mapping
        cut, pad = lm_head_share(model.lm_head.weight.shape[0], d.vocab, mapping.tp_rank)
        fill(model.lm_head.weight, "lm_head.weight", None, cut, pad)
        # Every parameter the shell carries must be filled by one of the two
        # reads this unit has, and by nothing else: the text core's table
        # (`model.w.`, read here), the one-engine DFly2 drafter's
        # (`draft_model.w.`, read by `load_draft_weights` -- `DRAFT_WEIGHTS.load`
        # checks that table's own coverage), and this lm_head, read just above.
        # A name outside all three is a weight nobody reads, and a failure here.
        unfed = {
            n
            for n, _ in model.named_parameters()
            if n != "lm_head.weight" and not n.startswith(("model.w.", "draft_model.w."))
        }
        assert not unfed, f"unfed shell parameters: {sorted(unfed)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly what the other units own; "
            f"unexpected {sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )

    def expected_unconsumed(self, core, weights) -> set:
        """The keys this unit does not load: the DFly2 drafter's layers, and the vision
        tower's when the config publishes no tower to load them into.

        The drafter belongs to another unit of the same flow (``dflash``) and is read off
        the checkpoint by key prefix, so a key belonging to a text layer can never land
        here. Every ``vit.*`` key this table declares a ``vis_*`` weight for is consumed;
        the rest of the tower's keys can only be missing when ``dims.vision`` is None, and
        then the whole prefix is expected to stay behind. A ``core`` of None is a caller
        asking for the partition without a built model, and reads the same way: no tower
        was built, so the whole prefix stays behind.
        """
        tower_keys_expected = core is None or core.dims.vision is None
        return {
            key
            for key in weights.keys()
            if key.startswith("model.dfly_layers.")
            or (tower_keys_expected and key.startswith("vit."))
        }


# --- the DFly2 drafter's table -------------------------------------------------


def _draft_q_b_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.qk_dim


def _draft_gate_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.v_dim


def _draft_o_cols(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.v_dim


def _draft_kv_b_rows(d, i=None) -> tuple[int, int]:
    """`kv_b_proj`'s `[heads * (nope + v), kv_lora]` in this rank's head blocks."""
    return d.attn_tp_rank, d.heads_local * (d.nope + d.v_dim)


def _draft_sink_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local


def _draft_q_b_scale_rows(d, i=None) -> tuple[int, int]:
    """Block-FP8 scales are 128 wide, so the head cut is in 128-row blocks."""
    return d.attn_tp_rank, d.heads_local * d.qk_dim // 128


def _draft_gate_scale_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.v_dim // 128


def _draft_o_scale_cols(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.v_dim // 128


def _draft_mlp_rows(d, i=None) -> tuple[int, int]:
    """The draft FFN's intermediate, split over the same group as the core's MLPs."""
    return d.mlp_tp_rank, d.mlp_inter // d.mlp_tp


def _draft_mlp_scale_rows(d, i=None) -> tuple[int, int]:
    return d.mlp_tp_rank, d.mlp_inter // d.mlp_tp // 128


def _draft_mlp_down_cols(d, i=None) -> tuple[int, int]:
    return d.mlp_tp_rank, d.mlp_inter // d.mlp_tp


def _draft_mlp_down_scale_cols(d, i=None) -> tuple[int, int]:
    return d.mlp_tp_rank, d.mlp_inter // d.mlp_tp // 128


def _draft_kv_b_k(t: torch.Tensor, d) -> torch.Tensor:
    """`[heads_local * (nope + v), kv_lora]` -> `[heads_local, nope, kv_lora]`.

    The draft layer absorbs its queries exactly as the target's sparse layers do,
    so `kv_b_proj` is relaid out into the same two operands (`_kv_b_k`/`_kv_b_v`
    for the core, which read the layer index; a draft layer's geometry is uniform
    across its five layers, so these read the bundle alone).
    """
    return t.view(d.heads_local, d.nope + d.v_dim, d.kv_lora)[:, : d.nope].contiguous()


def _draft_kv_b_v(t: torch.Tensor, d) -> torch.Tensor:
    """`[heads_local * (nope + v), kv_lora]` -> `[heads_local, kv_lora, v_dim]`."""
    return (
        t.view(d.heads_local, d.nope + d.v_dim, d.kv_lora)[:, d.nope :].transpose(1, 2).contiguous()
    )


def draft_host(core) -> SimpleNamespace:
    """The drafter table's host: the drafter's own bundle over the model's config.

    `_WeightTable`'s methods were written against a host whose `dims` is the
    bundle the table was declared over; the drafter's `dims` is `draft_dims`'s,
    not the core's.
    """
    return SimpleNamespace(dims=draft_dims(core), model_config=core.model_config)


class Hyv4FlashDraftWeights(_WeightTable):
    """The DFly2 drafter's five layers and four heads at one rank's share.

    117 checkpoint tensors, all of them under `model.dfly_layers.`: five decoder
    layers of 21 (a sparse layer's MLA, minus the indexer it does not own, plus
    its own FFN) and twelve singletons -- the slot-0 fusion, the hidden
    correction, the confidence head and the final norm. The drafter has no
    embedding and no lm_head of its own: it reads the target's.

    Every per-layer tensor but `kv_b_proj` is stored once and read once. All five
    draft layers share one geometry (32 heads, FFN 9216), so a cut depends on the
    rank and not on the layer -- unlike the core's, where a sparse and a windowed
    layer differ.

    Declared shapes, cuts and dtypes are checked against the checkpoint's own
    safetensors headers by `test_draft.py`; the FFN width, which the checkpoint's
    config does not publish, is `draft_dims`'s derivation and that test holds it.
    """

    WEIGHTS: tuple[W, ...] = (
        # --- the sandwich norms ---
        W(
            "input_layernorm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.{i}.input_layernorm.weight",
        ),
        W(
            "post_attention_layernorm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.{i}.post_attention_layernorm.weight",
        ),
        # --- MLA: replicated latents, head-split expansions (as the core's) ---
        W(
            "q_a_proj",
            lambda d, i=None: (d.q_lora, d.hidden),
            "model.dfly_layers.{i}.self_attn.q_a_proj.weight",
        ),
        W(
            "q_a_layernorm",
            lambda d, i=None: (d.q_lora,),
            "model.dfly_layers.{i}.self_attn.q_a_layernorm.weight",
        ),
        W(
            "kv_a_proj_with_mqa",
            lambda d, i=None: (d.lat_dim, d.hidden),
            "model.dfly_layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
        ),
        W(
            "kv_a_layernorm",
            lambda d, i=None: (d.kv_lora,),
            "model.dfly_layers.{i}.self_attn.kv_a_layernorm.weight",
        ),
        W(
            "q_b_proj",
            lambda d, i=None: (d.heads_local * d.qk_dim, d.q_lora),
            "model.dfly_layers.{i}.self_attn.q_b_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_draft_q_b_rows),
        ),
        W(
            "q_b_proj_scale",
            lambda d, i=None: (d.heads_local * d.qk_dim // 128, d.q_lora // 128),
            "model.dfly_layers.{i}.self_attn.q_b_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_draft_q_b_scale_rows),
            transform=_fp32,
        ),
        W(
            "k_b_proj",
            lambda d, i=None: (d.heads_local, d.nope, d.kv_lora),
            "model.dfly_layers.{i}.self_attn.kv_b_proj.weight",
            cut=_rows(_draft_kv_b_rows),
            transform=_draft_kv_b_k,
        ),
        W(
            "v_b_proj",
            lambda d, i=None: (d.heads_local, d.kv_lora, d.v_dim),
            "model.dfly_layers.{i}.self_attn.kv_b_proj.weight",
            cut=_rows(_draft_kv_b_rows),
            transform=_draft_kv_b_v,
        ),
        W(
            "linear_gate",
            lambda d, i=None: (d.heads_local * d.v_dim, d.hidden),
            "model.dfly_layers.{i}.self_attn.linear_gate.weight",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_draft_gate_rows),
        ),
        W(
            "linear_gate_scale",
            lambda d, i=None: (d.heads_local * d.v_dim // 128, d.hidden // 128),
            "model.dfly_layers.{i}.self_attn.linear_gate.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_draft_gate_scale_rows),
            transform=_fp32,
        ),
        W(
            "o_proj",
            lambda d, i=None: (d.hidden, d.heads_local * d.v_dim),
            "model.dfly_layers.{i}.self_attn.o_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_cols(_draft_o_cols),
        ),
        W(
            "o_proj_scale",
            lambda d, i=None: (d.hidden // 128, d.heads_local * d.v_dim // 128),
            "model.dfly_layers.{i}.self_attn.o_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_cols(_draft_o_scale_cols),
            transform=_fp32,
        ),
        #: The checkpoint stores the draft layers' sinks bf16; the core's are fp32.
        #: The attention op reads the sink as a raw fp32 buffer, and refuses any
        #: other dtype (`prepareInputs`: "Expected attention_sinks to have float
        #: dtype"), so the parameter is fp32 and the read upcasts. bf16 -> fp32 is
        #: exact: the parameter holds the checkpoint's own values.
        W(
            "sink",
            lambda d, i=None: (d.heads_local,),
            "model.dfly_layers.{i}.self_attn.learnable_sink_param",
            dtype=torch.float32,
            cut=_rows(_draft_sink_rows),
            transform=_fp32,
        ),
        W(
            "post_proj_norm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.{i}.self_attn.post_proj_norm.weight",
        ),
        # --- the draft FFN: block-FP8 on all five layers ---
        W(
            "mlp_gate",
            lambda d, i=None: (d.mlp_inter // d.mlp_tp, d.hidden),
            "model.dfly_layers.{i}.mlp.gate_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_draft_mlp_rows),
        ),
        W(
            "mlp_gate_scale",
            lambda d, i=None: (d.mlp_inter // d.mlp_tp // 128, d.hidden // 128),
            "model.dfly_layers.{i}.mlp.gate_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_draft_mlp_scale_rows),
            transform=_fp32,
        ),
        W(
            "mlp_up",
            lambda d, i=None: (d.mlp_inter // d.mlp_tp, d.hidden),
            "model.dfly_layers.{i}.mlp.up_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_draft_mlp_rows),
        ),
        W(
            "mlp_up_scale",
            lambda d, i=None: (d.mlp_inter // d.mlp_tp // 128, d.hidden // 128),
            "model.dfly_layers.{i}.mlp.up_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_draft_mlp_scale_rows),
            transform=_fp32,
        ),
        W(
            "mlp_down",
            lambda d, i=None: (d.hidden, d.mlp_inter // d.mlp_tp),
            "model.dfly_layers.{i}.mlp.down_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_cols(_draft_mlp_down_cols),
        ),
        W(
            "mlp_down_scale",
            lambda d, i=None: (d.hidden // 128, d.mlp_inter // d.mlp_tp // 128),
            "model.dfly_layers.{i}.mlp.down_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_cols(_draft_mlp_down_scale_cols),
            transform=_fp32,
        ),
        # --- slot-0 fusion: cat(anchor, target hidden) -> eh_proj, gated residual ---
        W(
            "fusion_enorm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.mtp_fusion.enorm.weight",
            layers=None,
        ),
        W(
            "fusion_hnorm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.mtp_fusion.hnorm.weight",
            layers=None,
        ),
        W(
            "fusion_eh_proj",
            lambda d, i=None: (d.hidden, d.fusion_in),
            "model.dfly_layers.mtp_fusion.eh_proj.weight",
            layers=None,
        ),
        W(
            "fusion_gate",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.mtp_fusion.gate",
            layers=None,
        ),
        # --- hidden correction: cat(hidden, previous embedding), gated residual ---
        W(
            "corr_hidden_norm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.hidden_correction.hidden_norm.weight",
            layers=None,
        ),
        W(
            "corr_embed_norm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.hidden_correction.embed_norm.weight",
            layers=None,
        ),
        W(
            "corr_gate",
            lambda d, i=None: (d.correction_inter, d.fusion_in),
            "model.dfly_layers.hidden_correction.gate_proj.weight",
            layers=None,
        ),
        W(
            "corr_up",
            lambda d, i=None: (d.correction_inter, d.fusion_in),
            "model.dfly_layers.hidden_correction.up_proj.weight",
            layers=None,
        ),
        W(
            "corr_down",
            lambda d, i=None: (d.hidden, d.correction_inter),
            "model.dfly_layers.hidden_correction.down_proj.weight",
            layers=None,
        ),
        # --- the confidence head and the final norm ---
        W(
            "conf_proj",
            lambda d, i=None: (1, d.hidden),
            "model.dfly_layers.confidence_head.proj.weight",
            layers=None,
        ),
        W(
            "conf_bias",
            lambda d, i=None: (1,),
            "model.dfly_layers.confidence_head.proj.bias",
            layers=None,
        ),
        W(
            "final_norm",
            lambda d, i=None: (d.hidden,),
            "model.dfly_layers.final_norm.weight",
            layers=None,
        ),
    )

    def expected_unconsumed(self, core, weights) -> set:
        """The keys the drafter does not load: the text core's and the vision tower's.

        The complement of the core's set, read off the checkpoint by prefix, so
        neither unit can silently take a key that belongs to the other.
        """
        return {key for key in weights.keys() if not key.startswith("model.dfly_layers.")}

    def load(self, draft, weights) -> None:
        """Fill every declared draft weight from the checkpoint.

        `draft` is the drafter -- the object whose `w` this table declared and
        whose `dims` the table was built over -- and `weights` is the draft
        read's map, which the engine narrows to `model.dfly_layers.` through the
        drafter's `checkpoint_key_prefixes`. Unlike the core's, a draft layer's
        geometry does not change from layer to layer, so no transform here reads
        the layer it is filling and the loop needs no layer bookkeeping.
        """
        d = draft.dims
        manifest = self.manifest(draft)
        consumed: set = set()

        def fill(param: torch.nn.Parameter, ckpt_key, index, cut, transform) -> None:
            dst = param.data if index is None else param.data[index]
            src = self.read(weights, ckpt_key, cut, transform, d, dst.device)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,))

        assert set(manifest.keys()) == set(draft.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(draft.w.keys()),
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, cut, transform in sources:
                fill(draft.w[param_key], ckpt_key, index, cut, transform)

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(draft, weights)
        assert leftover == expected, (
            "draft checkpoint coverage: leftover keys are not exactly what the other units own; "
            f"unexpected {sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )


MODEL_WEIGHTS = Hyv4FlashUnifiedWeights()

#: The drafter's table. Its host is `draft_host(core)`, not the core.
DRAFT_WEIGHTS = Hyv4FlashDraftWeights()


def lm_head_share(rows: int, vocab: int, tp_rank: int) -> tuple[Cut, Callable]:
    """The cut and the pad that fill an lm_head the shell built with `rows` rows.

    Whole under attention DP (`rows == vocab`); a vocab-parallel shard under TP.
    A vocab that does not split evenly leaves the last shard short, and its
    missing rows are zero, as the built-in LMHead pads them.
    """
    start = 0 if rows == vocab else tp_rank * rows
    stop = min(start + rows, vocab)
    assert start < stop, (rows, vocab, tp_rank)

    def pad(t: torch.Tensor, core) -> torch.Tensor:
        if t.shape[0] == rows:
            return t.contiguous()
        out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
        out[: t.shape[0]] = t
        return out

    return (slice(start, stop),), pad
