# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loader: step5-preview-nvfp4 / sm_103 / unified.

Every weight role of the checkpoint's text decoder, declared at this rank's share.
Every share is computed once, in `dims`, from the engine's `Mapping`:

* **Attention heads** are split over the attention TP group: the whole TP group
  without attention DP, a group of one with it. A rank holds its query heads' rows of
  `q_proj` and `g_proj` (one sigmoid gate per head), the rows of the KV heads those
  query heads read (`k_proj`, `v_proj`; GQA groups are contiguous), and the matching
  input columns of `o_proj`. The four projections are one fused `[q | k | v | g]`
  GEMM operand.

* **The sparse indexer** of the full-attention layers follows the same split: its 16
  heads form 4 provider groups, one per KV head, so a rank holds the query rows and
  per-head weights `w` of its KV heads' groups. The single proxy-key head `k` and its
  logits `z` are replicated. The four are one fused `[q | k | z | w]` operand, `w`
  rounded from the checkpoint's fp32 to bf16 (the reference's own rounding) and the
  rows zero-padded to a multiple of 8.

* **The dense MLPs** (layers 0-2 and 91) **and the shared experts** are split over
  the attention TP group along their intermediate: a rank holds a slice of the gate
  and up rows (one fused operand, gate first) and the matching down columns.

* **Routed experts** are split over the MoE EP group, a contiguous id window
  `[expert_offset, expert_offset + local_experts)` of the stacked `[352, ...]`
  tensors, and over the MoE TP group along their intermediate. FC1 is `[up | gate]`,
  the half order the CUTLASS MoE reads. Layers 3-87 are ModelOpt NVFP4: the e2m1
  codes stay packed, the e4m3 block scales are 128x4-swizzled per expert, and the
  per-expert global scales are kept whole so every rank forms the same activation
  scales. Layers 88-90 are bf16.

Everything else -- embedding, norms, router and its bias -- is replicated. lm_head
is the shell's: vocab-parallel under TP, whole under attention DP.

**The vision tower and its projector** (`vision_model.*`, `vit_large_projector.*`, all
bf16) are replicated: every rank encodes its own requests' images. Their per-block
weights are keyed `v{block}_<name>`. The absolute position embedding is published for
the 52 x 52 overview grid; the 36 x 36 grid of a crop reads it resampled (bilinear,
`align_corners=False`, in bf16, as the reference resamples it per call), done once here.

**A rank reads only its share.** Each source carries a `cut`, the rows or columns of
the checkpoint tensor this rank holds, applied to the checkpoint entry before
anything is moved to the device. Transforms run after the cut: the expert scale
swizzle permutes within 128-row tiles of one expert, which a cut at whole experts and
128-row multiples never splits.

**Not loaded.** The MTP layers 92-94 (no draft path is built) and every `ssmax_s` (an
initialisation value the indexer's weighted-ReLU scoring has no softmax to apply to).
`expected_unconsumed` names them, and `load` asserts the leftover keys are exactly those.
"""

import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.pyexecutor.config_utils import get_step5_nvfp4_expert_layers

from .input_processor import CROP_SIDE

#: An index into a checkpoint tensor: a tuple of slices, one per leading axis.
Cut = tuple[slice, ...]

SF_BLOCK = 16  # NVFP4: one e4m3 block scale per 16 elements along K

#: Every per-rank expert, shared-expert and dense-MLP width is a multiple of this: the
#: 128-row tile of the swizzled scale layout (128 / 16 = 8 scale columns, a multiple of
#: the 4-column tile).
_ROW_ALIGN = 128

#: The fused indexer projection's rows are padded to this multiple: the bf16 GEMM
#: takes an operand whose leading dimension is a multiple of 8 elements.
_GEMM_ROW_ALIGN = 8

#: The proxy-key heads of the indexer: one, shared by every provider group.
_INDEX_K_HEADS = 1

#: The layer type whose layers run the sparse indexer, as `layer_types` names it.
FULL_ATTENTION = "full_attention"

#: The vision MLP's width: the reference encoder's `mlp_ratio` default, 8960 / 1536 of
#: `vision_config.width` (the config states no ratio).
_VISION_MLP_RATIO = 8960 / 1536

#: The vision tower's two downsamplers each double the channels: the projector reads
#: `4 * width`.
_VISION_DOWNSAMPLE_WIDTH = 4


@dataclass(frozen=True)
class Src:
    """One source of a parameter: checkpoint key(s), where they land, how they are cut
    and transformed.

    `key` is a checkpoint key, or a tuple of keys handed to `transform` together (each
    read through the same `cut`). `index` is a slice tuple into the destination
    parameter, None for the whole of it. `transform` runs on the device after the cut.
    """

    key: Any
    index: tuple | None = None
    cut: Cut | None = None
    transform: Callable | None = None


@dataclass(frozen=True)
class W:
    """One weight role: its declared shape at this rank's share, its sources, its dtype
    (None: the model's), the layers that carry it (`layers(d) -> iterable`, None for a
    weight that is not per-layer), `zero`: zero-fill before the sources land (the
    padding rows of the fused indexer operand), and `prefix`: a per-layer key is
    `{prefix}{i}_{name}` (`l` a decoder layer, `v` a vision block)."""

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    sources: Callable[[Any, int | None], list[Src]]
    dtype: torch.dtype | None = None
    layers: Callable[[Any], Any] | None = None
    zero: bool = False
    prefix: str = "l"


def _read(entry, cut: Cut | None) -> torch.Tensor:
    """Realize one checkpoint entry, reading only `cut` of it. A 0-dim entry rejects
    `[:]`, and no 0-dim entry has a cut."""
    if cut is None:
        return entry if getattr(entry, "ndim", 1) == 0 else entry[:]
    return entry[cut[0]] if len(cut) == 1 else entry[cut]


# --- transforms, on the rank's cut, on the device ---


def _to_bf16(t: torch.Tensor) -> torch.Tensor:
    """An fp32 checkpoint weight rounded to bf16, as the reference rounds it before use
    (the indexer's `w` projection, its q-norm weight and its k-norm affine)."""
    return t.to(torch.bfloat16)


def _cat_experts(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """(up, gate) expert stacks `[E, I, K]` -> FC1 `[E, 2I, K]`, up rows first."""
    return torch.cat(t, dim=1).contiguous()


def _swizzle(s: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 block-scale swizzle of `[E, M, C]` e4m3 bytes, same shape."""
    u8 = s.contiguous().view(torch.uint8)
    return torch.ops.trtllm.block_scale_interleave(u8).reshape(u8.shape)


def _cat_swizzle(t: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """(up, gate) block scales -> FC1's `[E, 2I, K/16]`, up first, swizzled per expert."""
    return _swizzle(torch.cat([x.view(torch.uint8) for x in t], dim=1))


def _contig(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous()


def _resample_grid(t: torch.Tensor, grid: int, side: int) -> torch.Tensor:
    """A `[grid * grid, C]` position embedding resampled to `[side * side, C]`: bilinear,
    `align_corners=False`, in its own dtype, as the reference encoder's
    `sample_abs_posemb` resamples it."""
    image = t.reshape(1, grid, grid, -1).permute(0, 3, 1, 2).contiguous()
    out = F.interpolate(image, size=(side, side), mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1).reshape(side * side, -1).contiguous()


# --- layer families ---


def _all_layers(d):
    return range(d.num_layers)


def _sparse_layers(d):
    return d.sparse_layers


def _dense_mlp_layers(d):
    return d.dense_mlp_layers


def _moe_layers(d):
    return d.moe_layers


def _nvfp4_expert_layers(d):
    return d.nvfp4_layers


def _bf16_expert_layers(d):
    return d.bf16_expert_layers


def _vision_blocks(d):
    return range(d.v_layers)


# --- sources ---


def _layer(i: int, name: str) -> str:
    return f"model.layers.{i}.{name}"


def _block(i: int, name: str) -> str:
    return f"vision_model.transformer.resblocks.{i}.{name}"


def _vblock(name: str):
    return lambda d, i: [Src(_block(i, name))]


def _vkey(name: str):
    return lambda d, i: [Src(name)]


def _crop_posemb(d, i) -> list[Src]:
    return [
        Src(
            "vision_model.positional_embedding",
            transform=functools.partial(_resample_grid, grid=d.v_grid, side=d.v_crop_grid),
        )
    ]


def _rows(rank: int, span: int) -> Cut:
    return (slice(rank * span, (rank + 1) * span),)


def _cols(rank: int, span: int) -> Cut:
    return (slice(None), slice(rank * span, (rank + 1) * span))


def _whole(name: str, transform: Callable | None = None):
    return lambda d, i: [Src(_layer(i, name), transform=transform)]


def _qkvg(d, i) -> list[Src]:
    """This rank's query heads of q and g, and the KV heads their groups read."""
    q, kv, g = d.heads * d.head_dim, d.kv_heads * d.head_dim, d.heads
    r = d.attn_tp_rank
    return [
        Src(_layer(i, "self_attn.q_proj.weight"), (slice(0, q),), _rows(r, q)),
        Src(_layer(i, "self_attn.k_proj.weight"), (slice(q, q + kv),), _rows(r, kv)),
        Src(_layer(i, "self_attn.v_proj.weight"), (slice(q + kv, q + 2 * kv),), _rows(r, kv)),
        Src(
            _layer(i, "self_attn.g_proj.weight"),
            (slice(q + 2 * kv, q + 2 * kv + g),),
            _rows(r, g),
        ),
    ]


def _o(d, i) -> list[Src]:
    return [
        Src(_layer(i, "self_attn.o_proj.weight"), cut=_cols(d.attn_tp_rank, d.heads * d.head_dim))
    ]


def _idx(d, i) -> list[Src]:
    """`[q | k | z | w]` of the indexer: this rank's provider groups' query heads and
    weights, the replicated proxy key and its logits."""
    q, p, h = d.idx_heads * d.proxy_dim, d.proxy_dim, d.idx_heads
    r = d.attn_tp_rank
    return [
        Src(_layer(i, "self_attn.sparse_indexer_q.weight"), (slice(0, q),), _rows(r, q)),
        Src(_layer(i, "self_attn.sparse_indexer_k.weight"), (slice(q, q + p),)),
        Src(_layer(i, "self_attn.sparse_indexer_z.weight"), (slice(q + p, q + 2 * p),)),
        Src(
            _layer(i, "self_attn.sparse_indexer_w.weight"),
            (slice(q + 2 * p, q + 2 * p + h),),
            _rows(r, h),
            _to_bf16,
        ),
    ]


def _gate_up(prefix: str, inter: Callable[[Any], int]):
    """A SwiGLU MLP's fused `[gate | up]` rows at this rank's intermediate slice."""

    def build(d, i) -> list[Src]:
        n, r = inter(d), d.mlp_tp_rank
        return [
            Src(_layer(i, f"{prefix}.gate_proj.weight"), (slice(0, n),), _rows(r, n)),
            Src(_layer(i, f"{prefix}.up_proj.weight"), (slice(n, 2 * n),), _rows(r, n)),
        ]

    return build


def _down(prefix: str, inter: Callable[[Any], int]):
    return lambda d, i: [
        Src(_layer(i, f"{prefix}.down_proj.weight"), cut=_cols(d.mlp_tp_rank, inter(d)))
    ]


def _expert_cut(d, span: int, axis: int) -> Cut:
    """This rank's EP window of the stacked experts, and its moe_tp slice (`span` rows,
    or columns when `axis` is 2) of each."""
    experts = slice(d.expert_offset, d.expert_offset + d.local_experts)
    part = slice(d.moe_tp_rank * span, (d.moe_tp_rank + 1) * span)
    return (experts, part) if axis == 1 else (experts, slice(None), part)


def _fc1(suffix: str, transform: Callable):
    return lambda d, i: [
        Src(
            (_layer(i, f"moe.up_proj.{suffix}"), _layer(i, f"moe.gate_proj.{suffix}")),
            cut=_expert_cut(d, d.moe_inter, 1),
            transform=transform,
        )
    ]


def _fc2(suffix: str, span: Callable[[Any], int], transform: Callable):
    return lambda d, i: [
        Src(
            _layer(i, f"moe.down_proj.{suffix}"),
            cut=_expert_cut(d, span(d), 2),
            transform=transform,
        )
    ]


#: The six per-expert NVFP4 scalars of a routed-expert layer, as (parameter name,
#: checkpoint name), each `[352]` and read whole on every rank. ModelOpt stores
#: reciprocals: `input_scale` = amax / (448 * 6), `weight_scale_2` = the weight's
#: global dequantization factor.
_EXPERT_SCALARS = (
    ("e_isc1", "moe.gate_proj.input_scale"),
    ("e_isc1_up", "moe.up_proj.input_scale"),
    ("e_ws2_1", "moe.gate_proj.weight_scale_2"),
    ("e_ws2_1_up", "moe.up_proj.weight_scale_2"),
    ("e_isc2", "moe.down_proj.input_scale"),
    ("e_ws2_2", "moe.down_proj.weight_scale_2"),
)


def _expert_scalar(name: str, src: str) -> W:
    return W(name, lambda d: (d.num_experts,), _whole(src), torch.float32, _nvfp4_expert_layers)


class Step5PreviewWeights:
    """step5-preview-nvfp4 / sm_103 / unified.

    The table and the passes over it: `declare` allocates every weight it names,
    `manifest` maps checkpoint keys onto them, `read` turns one source into the tensor
    to copy, `load` fills them. `declare` and `manifest` share one "which parameters
    exist" decision (`_targets`), so they cannot disagree about a weight.
    """

    WEIGHTS: tuple[W, ...] = (
        # --- every decoder layer: its two norms and attention ---
        W(
            "norm1",
            lambda d: (d.hidden,),
            _whole("input_layernorm.weight"),
            torch.float32,
            _all_layers,
        ),
        W(
            "norm2",
            lambda d: (d.hidden,),
            _whole("post_attention_layernorm.weight"),
            torch.float32,
            _all_layers,
        ),
        W(
            "qkvg",
            lambda d: ((d.heads + 2 * d.kv_heads) * d.head_dim + d.heads, d.hidden),
            _qkvg,
            None,
            _all_layers,
        ),
        W(
            "q_norm",
            lambda d: (d.head_dim,),
            _whole("self_attn.q_norm.weight"),
            torch.float32,
            _all_layers,
        ),
        W(
            "k_norm",
            lambda d: (d.head_dim,),
            _whole("self_attn.k_norm.weight"),
            torch.float32,
            _all_layers,
        ),
        W("o", lambda d: (d.hidden, d.heads * d.head_dim), _o, None, _all_layers),
        # --- the sparse indexer of the full-attention layers ---
        W("idx", lambda d: (d.idx_rows, d.hidden), _idx, None, _sparse_layers, zero=True),
        W(
            "idx_qn",
            lambda d: (d.proxy_dim,),
            _whole("self_attn.sparse_indexer_q_norm.weight", _to_bf16),
            torch.bfloat16,
            _sparse_layers,
        ),
        W(
            "idx_kn_w",
            lambda d: (d.proxy_dim,),
            _whole("self_attn.sparse_indexer_k_norm.weight", _to_bf16),
            torch.bfloat16,
            _sparse_layers,
        ),
        W(
            "idx_kn_b",
            lambda d: (d.proxy_dim,),
            _whole("self_attn.sparse_indexer_k_norm.bias", _to_bf16),
            torch.bfloat16,
            _sparse_layers,
        ),
        # --- the dense SwiGLU MLP of layers 0-2 and 91 ---
        W(
            "gu",
            lambda d: (2 * d.dense_inter, d.hidden),
            _gate_up("mlp", lambda d: d.dense_inter),
            None,
            _dense_mlp_layers,
        ),
        W(
            "dn",
            lambda d: (d.hidden, d.dense_inter),
            _down("mlp", lambda d: d.dense_inter),
            None,
            _dense_mlp_layers,
        ),
        # --- MoE layers 3-90: router, its bias and the shared expert ---
        W(
            "router",
            lambda d: (d.num_experts, d.hidden),
            _whole("moe.gate.weight"),
            None,
            _moe_layers,
        ),
        W(
            "router_bias",
            lambda d: (d.num_experts,),
            _whole("moe.router_bias"),
            torch.float32,
            _moe_layers,
        ),
        W(
            "sgu",
            lambda d: (2 * d.shared_inter, d.hidden),
            _gate_up("share_expert", lambda d: d.shared_inter),
            None,
            _moe_layers,
        ),
        W(
            "sdn",
            lambda d: (d.hidden, d.shared_inter),
            _down("share_expert", lambda d: d.shared_inter),
            None,
            _moe_layers,
        ),
        # --- NVFP4 routed experts (layers 3-87), this rank's window and slice ---
        W(
            "fc1_w",
            lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // 2),
            _fc1("weight", _cat_experts),
            torch.uint8,
            _nvfp4_expert_layers,
        ),
        W(
            "fc1_s",
            lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // SF_BLOCK),
            _fc1("weight_scale", _cat_swizzle),
            torch.uint8,
            _nvfp4_expert_layers,
        ),
        W(
            "fc2_w",
            lambda d: (d.local_experts, d.hidden, d.moe_inter // 2),
            _fc2("weight", lambda d: d.moe_inter // 2, _contig),
            torch.uint8,
            _nvfp4_expert_layers,
        ),
        W(
            "fc2_s",
            lambda d: (d.local_experts, d.hidden, d.moe_inter // SF_BLOCK),
            _fc2("weight_scale", lambda d: d.moe_inter // SF_BLOCK, _swizzle),
            torch.uint8,
            _nvfp4_expert_layers,
        ),
        *(_expert_scalar(name, src) for name, src in _EXPERT_SCALARS),
        # --- bf16 routed experts (layers 88-90) ---
        W(
            "fc1",
            lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden),
            _fc1("weight", _cat_experts),
            None,
            _bf16_expert_layers,
        ),
        W(
            "fc2",
            lambda d: (d.local_experts, d.hidden, d.moe_inter),
            _fc2("weight", lambda d: d.moe_inter, _contig),
            None,
            _bf16_expert_layers,
        ),
        # --- not per-layer ---
        W(
            "final_norm",
            lambda d: (d.hidden,),
            lambda d, i: [Src("model.norm.weight")],
            torch.float32,
        ),
        W("embed", lambda d: (d.vocab, d.hidden), lambda d, i: [Src("model.embed_tokens.weight")]),
        # --- the vision tower (replicated, bf16) ---
        W(
            "v_patch",
            lambda d: (d.v_width, d.v_channels, d.v_patch, d.v_patch),
            _vkey("vision_model.conv1.weight"),
        ),
        W(
            "v_pos",
            lambda d: (d.v_grid * d.v_grid, d.v_width),
            _vkey("vision_model.positional_embedding"),
        ),
        W("v_pos_crop", lambda d: (d.v_crop_grid * d.v_crop_grid, d.v_width), _crop_posemb),
        W("v_ln_pre_w", lambda d: (d.v_width,), _vkey("vision_model.ln_pre.weight")),
        W("v_ln_pre_b", lambda d: (d.v_width,), _vkey("vision_model.ln_pre.bias")),
        *(
            W(name, shape, _vblock(key), None, _vision_blocks, prefix="v")
            for name, shape, key in (
                ("ln1_w", lambda d: (d.v_width,), "ln_1.weight"),
                ("ln1_b", lambda d: (d.v_width,), "ln_1.bias"),
                ("qkv", lambda d: (3 * d.v_width, d.v_width), "attn.in_proj_weight"),
                ("qkv_b", lambda d: (3 * d.v_width,), "attn.in_proj_bias"),
                ("o", lambda d: (d.v_width, d.v_width), "attn.out_proj.weight"),
                ("o_b", lambda d: (d.v_width,), "attn.out_proj.bias"),
                ("ls1", lambda d: (d.v_width,), "ls_1.gamma"),
                ("ln2_w", lambda d: (d.v_width,), "ln_2.weight"),
                ("ln2_b", lambda d: (d.v_width,), "ln_2.bias"),
                ("fc1", lambda d: (d.v_inter, d.v_width), "mlp.c_fc.weight"),
                ("fc1_b", lambda d: (d.v_inter,), "mlp.c_fc.bias"),
                ("fc2", lambda d: (d.v_width, d.v_inter), "mlp.c_proj.weight"),
                ("fc2_b", lambda d: (d.v_width,), "mlp.c_proj.bias"),
                ("ls2", lambda d: (d.v_width,), "ls_2.gamma"),
            )
        ),
        W(
            "v_down1",
            lambda d: (2 * d.v_width, d.v_width, 3, 3),
            _vkey("vision_model.vit_downsampler1.weight"),
        ),
        W("v_down1_b", lambda d: (2 * d.v_width,), _vkey("vision_model.vit_downsampler1.bias")),
        W(
            "v_down2",
            lambda d: (d.v_out, 2 * d.v_width, 3, 3),
            _vkey("vision_model.vit_downsampler2.weight"),
        ),
        W("v_down2_b", lambda d: (d.v_out,), _vkey("vision_model.vit_downsampler2.bias")),
        W("v_proj", lambda d: (d.hidden, d.v_out), _vkey("vit_large_projector.weight")),
    )

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares, at this rank's share. Meta-init
        intercepts `torch.empty`; real storage arrives when the engine materializes the
        registry."""
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        """`[(param key, layer index | None), ...]`: the one place that decides which
        parameters an entry declares."""
        if entry.layers is None:
            return [(entry.name, None)]
        return [(f"{entry.prefix}{i}_{entry.name}", i) for i in entry.layers(d)]

    def manifest(self, core) -> dict[str, tuple[W, list[Src]]]:
        """param key -> (its table entry, its sources)."""
        d = core.dims
        rows: dict[str, tuple[W, list[Src]]] = {}
        for entry in self.WEIGHTS:
            for key, i in self._targets(entry, d):
                rows[key] = (entry, entry.sources(d, i))
        return rows

    def read(self, weights, src: Src, device=None) -> torch.Tensor:
        """One source -> the tensor its destination receives: every key read through the
        source's cut, moved to `device`, handed to its transform. The one path `load`
        takes, so the shard test drives it directly."""
        keys = src.key if isinstance(src.key, tuple) else (src.key,)
        for key in keys:
            assert key in weights, f"checkpoint key missing: {key}"
        parts = tuple(_read(weights[key], src.cut) for key in keys)
        if device is not None:
            parts = tuple(t.to(device, non_blocking=True) for t in parts)
        if src.transform is not None:
            return src.transform(parts if isinstance(src.key, tuple) else parts[0])
        assert len(parts) == 1, "a multi-key source needs a transform"
        return parts[0]

    def expected_unconsumed(self, core, weights) -> set:
        """The checkpoint keys `load` leaves, each family for a stated reason:

        * the MTP layers past `num_hidden_layers` (no speculative decoding is built);
        * every sparse layer's `ssmax_s` (unused: the indexer scores with a weighted
          ReLU, and there is no softmax for a scalable-softmax scale to act on).
        """
        n = core.dims.num_layers
        mtp = int(getattr(core.text_config, "num_nextn_predict_layers", 0) or 0)
        prefixes = tuple(f"model.layers.{i}." for i in range(n, n + mtp))
        keys = {k for k in weights.keys() if k.startswith(prefixes)}
        keys |= {_layer(i, "self_attn.ssmax_s") for i in core.dims.sparse_layers}
        return keys

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint, each source read through its
        cut, and assert coverage both ways: every target parameter written, every
        checkpoint key consumed or in `expected_unconsumed`."""
        core = model.model
        manifest = self.manifest(core)
        consumed: set = set()

        def fill(param: torch.nn.Parameter, src: Src) -> None:
            dst = param.data if src.index is None else param.data[src.index]
            value = self.read(weights, src, dst.device)
            assert dst.shape == value.shape, (src.key, tuple(dst.shape), tuple(value.shape))
            assert value.dtype == dst.dtype, (src.key, value.dtype, dst.dtype)
            dst.copy_(value, non_blocking=True)
            consumed.update(src.key if isinstance(src.key, tuple) else (src.key,))

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, (entry, sources) in manifest.items():
            param = core.w[param_key]
            if entry.zero:
                param.data.zero_()
            for src in sources:
                fill(param, src)
            # The expert relayouts allocate per-layer scratch; release it before the
            # next parameter so peak load memory stays one layer deep.
            if param_key.endswith(("_fc1_s", "_fc2_s", "_fc1", "_fc2")):
                torch.cuda.empty_cache()

        # The shell builds lm_head vocab-parallel under TP and whole under attention DP;
        # read the rows it built and nothing else.
        mapping = core.model_config.mapping
        cut, pad = lm_head_share(model.lm_head.weight.shape[0], core.dims.vocab, mapping.tp_rank)
        fill(model.lm_head.weight, Src("lm_head.weight", None, cut, pad))
        # Parameter-side coverage the other way: name every registered parameter outside
        # the target's own ParameterDict instead of assuming lm_head is the only one.
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly the MTP layers and the "
            f"ssmax_s scales; unexpected {sorted(leftover - expected)[:8]}, "
            f"missed {sorted(expected - leftover)[:8]}"
        )

    def dims(self, core) -> SimpleNamespace:
        """Every per-rank width, from the text config and the engine's Mapping, and the
        vision tower's, from the composite's `vision_config`."""
        cfg = core.text_config
        vis = core.composite_config.vision_config
        v_heads = int(vis.heads)
        v_width = int(vis.width)
        assert v_width % v_heads == 0, (v_width, v_heads)
        m = core.model_config.mapping
        attention_dp = bool(m.enable_attention_dp)
        attn_tp = 1 if attention_dp else m.tp_size
        attn_tp_rank = 0 if attention_dp else m.tp_rank
        sparse = cfg.sparse_config
        heads_total = cfg.num_attention_heads
        kv_total = cfg.num_attention_groups
        idx_total = int(sparse["sparse_indexer_num_heads"])
        for name, total in (("query", heads_total), ("kv", kv_total), ("indexer", idx_total)):
            assert total % attn_tp == 0, f"{total} {name} heads do not split {attn_tp} ways"
        num_experts = cfg.moe_num_experts
        assert num_experts % m.moe_ep_size == 0, (num_experts, m.moe_ep_size)
        local_experts = num_experts // m.moe_ep_size
        widths = {
            "dense": (cfg.intermediate_size, attn_tp),
            "shared": (cfg.share_expert_dim, attn_tp),
            "expert": (cfg.moe_intermediate_size, m.moe_tp_size),
        }
        for name, (full, parts) in widths.items():
            assert full % parts == 0 and (full // parts) % _ROW_ALIGN == 0, (
                f"{name} intermediate {full} does not split {parts} ways on {_ROW_ALIGN}-row tiles"
            )
        num_layers = cfg.num_hidden_layers
        layer_types = list(cfg.layer_types)
        sparse_layers = tuple(
            i
            for i in range(num_layers)
            if layer_types[i] == FULL_ATTENTION and cfg.is_sparse_layer(i)
        )
        moe_layers = tuple(i for i in range(num_layers) if cfg.is_moe_layer(i))
        nvfp4_layers = tuple(get_step5_nvfp4_expert_layers(cfg, core.model_config.quant_config))
        idx_heads = idx_total // attn_tp
        proxy_dim = int(sparse["proxy_dim"])
        idx_rows = idx_heads * proxy_dim + 2 * _INDEX_K_HEADS * proxy_dim + idx_heads
        idx_rows = -(-idx_rows // _GEMM_ROW_ALIGN) * _GEMM_ROW_ALIGN
        return SimpleNamespace(
            num_layers=num_layers,
            hidden=cfg.hidden_size,
            vocab=cfg.vocab_size,
            head_dim=cfg.head_dim,
            attention_dp=attention_dp,
            attn_tp=attn_tp,
            attn_tp_rank=attn_tp_rank,
            heads=heads_total // attn_tp,
            kv_heads=kv_total // attn_tp,
            idx_heads=idx_heads,
            proxy_dim=proxy_dim,
            idx_rows=idx_rows,
            sliding=tuple(layer_types[i] != FULL_ATTENTION for i in range(num_layers)),
            sparse_layers=sparse_layers,
            moe_layers=moe_layers,
            dense_mlp_layers=tuple(i for i in range(num_layers) if i not in moe_layers),
            nvfp4_layers=nvfp4_layers,
            bf16_expert_layers=tuple(i for i in moe_layers if i not in nvfp4_layers),
            num_experts=num_experts,
            mlp_tp_rank=attn_tp_rank,
            dense_inter=cfg.intermediate_size // attn_tp,
            shared_inter=cfg.share_expert_dim // attn_tp,
            moe_tp=m.moe_tp_size,
            moe_tp_rank=m.moe_tp_rank,
            moe_inter=cfg.moe_intermediate_size // m.moe_tp_size,
            moe_ep=m.moe_ep_size,
            moe_ep_rank=m.moe_ep_rank,
            local_experts=local_experts,
            expert_offset=local_experts * m.moe_ep_rank,
            dtype=core.model_config.torch_dtype,
            v_layers=int(vis.layers),
            v_width=v_width,
            v_heads=v_heads,
            v_head_dim=v_width // v_heads,
            v_inter=int(v_width * _VISION_MLP_RATIO),
            v_channels=int(getattr(vis, "num_channels", 3)),
            v_patch=int(vis.patch_size),
            v_grid=int(vis.image_size) // int(vis.patch_size),
            v_crop_grid=CROP_SIDE // int(vis.patch_size),
            v_out=_VISION_DOWNSAMPLE_WIDTH * v_width,
        )


MODEL_WEIGHTS = Step5PreviewWeights()


def lm_head_share(rows: int, vocab: int, tp_rank: int) -> tuple[Cut, Callable]:
    """The cut and the pad that fill an lm_head the shell built with `rows` rows: whole
    under attention DP (`rows == vocab`), a vocab-parallel shard under TP. A vocab that
    does not split evenly leaves the last shard short; its missing rows are zero, as the
    built-in LMHead pads them."""
    start = 0 if rows == vocab else tp_rank * rows
    stop = min(start + rows, vocab)
    assert start < stop, (rows, vocab, tp_rank)

    def pad(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] == rows:
            return t.contiguous()
        out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
        out[: t.shape[0]] = t
        return out

    return (slice(start, stop),), pad
