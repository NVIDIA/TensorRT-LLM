# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loader: deepseek-r1-0528-nvfp4 / sm_10x / default.

Every weight role of the checkpoint, declared at this rank's share. Every share
is computed once, in `dims`, from the engine's `Mapping`:

* **Attention heads** are split over the attention TP group, which is the
  whole TP group without attention DP and a group of one with it. A rank
  holds its heads' rows of `q_b_proj` and `kv_b_proj` and the matching input
  columns of `o_proj`, as the built-in MLA's column- and row-parallel Linears
  do. `q_a_proj` and `kv_a_proj_with_mqa` are replicated: their outputs are
  the per-token latents every head reads, and the latent KV cache is the same
  on every rank.

* **The dense MLPs** (layers 0-2) **and the shared expert** are split over the
  same group along their intermediate: a rank holds a slice of the gate and up
  rows and the matching down columns, and its output is a partial sum.

* **Experts** are split over the MoE EP group: a rank holds the contiguous id
  window `[expert_offset, expert_offset + local_experts)`.

* **The expert intermediate** is split over the MoE TP group, along the FC1
  output rows and the FC2 input columns. At this geometry (I=2048, one e4m3
  scale per 16 values) every split up to 8 lands on whole scale groups and
  on the kernel's 128-row alignment, so no padding is needed.

Under attention DP the attention and MLP groups are one rank, so those three
families are whole and only the experts are split.

Everything else -- embedding, norms, `q_a`/`kv_a`, router and bias, every
per-tensor NVFP4 scalar -- is replicated. lm_head is the shell's:
vocab-parallel under TP, whole under attention DP; `load` reads the rows the
shell built.

**A rank reads only its share.** Each split weight carries a `cut`: the rows
or columns of the checkpoint tensor this rank holds, applied to the lazy
safetensors entry at read time, so the bytes of every other rank's share are
never read. Materializing whole tensors and cutting after is what a 4-rank
tray cannot afford here: under moe_tp every rank would read all 256 experts
at full width, and under attention TP the whole dense MLP, shared expert and
attention projections, several times the rank's share of host memory.

After the cut, the transforms are the kernels' relayouts (`kv_b_proj` row
regroup, swizzled dense scales, the expert interleave / shuffle / swizzle),
run on the rank's share. At world size 1 the cut is the whole tensor, so a
rank's operands are the single-rank relayout of its slice. The order is not
free: the swizzle and the shuffle permute within blocks of rows, so cutting
after them would cut through blocks.

**The MTP module** is layer `num_hidden_layers` of the same checkpoint and of
the same KV pool, loaded only under a `speculative_config`. Its attention
block, norms, router and bias are a trunk layer's rows one layer further
along, cut the same way. What it has of its own is replicated -- `enorm`,
`hnorm`, `eh_proj` and `shared_head.norm` -- or bf16 where the trunk is
NVFP4: `hf_quant_config.json` excludes `model.layers.61*` from quantization,
so its shared expert is split over the attention TP group like the trunk's
and its routed experts are cut by moe_ep and moe_tp like the trunk's, but
need none of the interleave / shuffle / swizzle. They land in `fused_moe`'s
layout: `[E, 2I, H]` with the **up** rows first, `[E, H, I]` for FC2. Its
`embed_tokens` and `shared_head.head` are bitwise copies of the trunk's
embedding and `lm_head`, which the draft model reads instead.

Loading contract: `load(model, weights)` consumes the engine-provided
checkpoint dict, fills every declared parameter exactly once, and asserts
bidirectional coverage -- every target parameter written, and every checkpoint
key either consumed (a key read through a cut is consumed) or in the two
predicted non-load sets (this rank's off-window experts, and what the MTP
layer leaves: all of it without MTP, its two aliased keys with it).
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

#: `layers=ALL_LAYERS` is the common case and the default. A sentinel rather
#: than a lambda so the table reads as data, and rather than `True` so the
#: field has one type of answer: which layers.
ALL_LAYERS = object()

#: An index into a checkpoint tensor: a tuple of slices, one per leading axis.
Cut = tuple[slice, ...]


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    `shape` takes the dimension bundle and returns the declared shape -- this
    rank's share, every width derived from the config and the mapping.

    `src` says where the bytes come from, in one of two forms:

    * a checkpoint key template, for a weight that is not per-layer;
    * a callable `(d, i) -> [(key, index), ...]`, taking the dimension bundle
      and the layer index. A `key` may itself be a tuple, meaning the
      transform is handed those tensors together.

    `layers` says which layers carry this weight: `ALL_LAYERS` for every one,
    `None` for a weight that is not per-layer at all, or a callable
    `d -> iterable[int]` when only some layers have it.

    `cut` takes the dimension bundle and returns this rank's rows or columns of
    each checkpoint tensor, read from the lazy entry so nothing else is read;
    the same cut applies to every key of a multi-key row. `None` reads the
    tensor whole.

    `transform` runs on the cut tensor(s) before the copy, on the destination's
    device: the kernels' relayouts, with no slicing of their own. `None` means
    the cut is already the declared shape.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    cut: Callable[[Any], Cut] | None = None
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
        """`[(checkpoint key or key tuple, index into the param | None), ...]`."""
        if callable(self.src):
            return list(self.src(d, i))
        return [(self.src, None)]


SF_BLOCK = 16  # NVFP4: one e4m3 scale per 16 elements along K

# Every per-rank NVFP4 width is a multiple of this: the 128-row tile of the
# swizzled scale layout, which also covers the 32-row expert shuffle and the
# 4-column scale tile (128 / 16 = 8).
_ROW_ALIGN = 128


# --- cuts: this rank's block of rows or columns, read from the lazy entry ---


def _rows(block: Callable[[Any], tuple[int, int]]) -> Callable[[Any], Cut]:
    """Rows `[rank * span, (rank + 1) * span)`, with `block(d) -> (rank, span)`."""

    def cut(d) -> Cut:
        rank, span = block(d)
        return (slice(rank * span, (rank + 1) * span),)

    return cut


def _cols(block: Callable[[Any], tuple[int, int]]) -> Callable[[Any], Cut]:
    """Columns `[rank * span, (rank + 1) * span)`, every row."""

    def cut(d) -> Cut:
        rank, span = block(d)
        return (slice(None), slice(rank * span, (rank + 1) * span))

    return cut


def _read(entry, cut: Cut | None) -> torch.Tensor:
    """Realize one checkpoint entry, reading only `cut` of it.

    `entry[cut]` on a lazy safetensors slice reads those rows or columns and
    nothing else; on a plain tensor it is the same view. Uncut, `[:]` reads it
    whole -- but a 0-dim tensor (every NVFP4 per-tensor scalar and every KV
    scale) rejects that index, and no 0-dim entry has a cut."""
    if cut is None:
        return entry if getattr(entry, "ndim", 1) == 0 else entry[:]
    return entry[cut[0]] if len(cut) == 1 else entry[cut]


# --- relayouts, on the rank's cut ---


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each aligned block of
    32 rows, source row `4u + v` (0<=u<8, 0<=v<4) lands at destination row
    `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _interleave_perm(rows: int, device) -> torch.Tensor:
    """Gather index that interleaves the `[up | gate]` halves of a `2*I` row
    stack into up0, gate0, up1, gate1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


def _fc1_perm(rows: int, device) -> torch.Tensor:
    """Interleave then 32-row block shuffle, composed into one gather."""
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 block-scale swizzle of a `[M, C]` uint8 scale matrix, returned
    flat (`M % 128 == 0` and `C % 4 == 0` hold for every share `dims`
    admits, so the result is exactly `M * C` bytes)."""
    return torch.ops.trtllm.block_scale_interleave(x.contiguous())


def _regroup_kv_b(t: torch.Tensor, core) -> torch.Tensor:
    """This rank's `[Hr*(nope+v), C]` per-head-interleaved rows -> `[Hr*nope ; Hr*v]`,
    at this rank's head count: the MLA context FMHA reads V as the column block
    after every head's k_nope."""
    d = core.dims
    t3 = t.reshape(d.heads, d.nope + d.v_dim, d.kv_lora)
    return torch.cat(
        [
            t3[:, : d.nope, :].reshape(d.heads * d.nope, d.kv_lora),
            t3[:, d.nope :, :].reshape(d.heads * d.v_dim, d.kv_lora),
        ],
        dim=0,
    ).contiguous()


def _cat_rows(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """Fused `gate_up` weight bytes: gate rows first (the half order
    `flashinfer_silu_and_mul` reads)."""
    return torch.cat(t, dim=0).contiguous()


def _cat_rows_swizzle(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """Fused `gate_up` scale bytes: concatenate, then one swizzle. Swizzling
    the halves apart and concatenating after gives a different byte order."""
    return _swizzle(torch.cat([x.view(torch.uint8) for x in t], dim=0))


def _swizzle_only(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` scale bytes: swizzle in place of the row-major order."""
    return _swizzle(t.view(torch.uint8))


def _expert_fc1_w(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) packed nibbles -> one expert's FC1 operand: concat
    `[up ; gate]`, interleave, shuffle."""
    up, gate = t
    w = torch.cat([up, gate], dim=0)
    return torch.index_select(w, 0, _fc1_perm(w.shape[0], w.device)).contiguous()


def _expert_fc1_s(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """(up, gate) e4m3 scales -> one expert's FC1 scale operand."""
    up, gate = t
    s = torch.cat([up.view(torch.uint8), gate.view(torch.uint8)], dim=0)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _fc1_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


def _expert_fc2_w(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` packed nibbles -> one expert's FC2 operand (no interleave)."""
    return torch.index_select(t, 0, _block32_perm(t.shape[0], t.device)).contiguous()


def _expert_fc2_s(t: torch.Tensor, core) -> torch.Tensor:
    """`down_proj` e4m3 scales -> one expert's FC2 scale operand."""
    s = t.view(torch.uint8)
    rows, cols = s.shape
    s = torch.index_select(s, 0, _block32_perm(rows, s.device))
    return _swizzle(s).reshape(rows, cols)


def _mtp_fc1(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    """One bf16 MTP expert's FC1 operand: `[up ; gate]` rows, up first -- the
    half order `fused_moe` reads, the opposite of the dense gate_up linear's."""
    return torch.cat(t, dim=0).contiguous()


# --- the shares, as (rank, span) in checkpoint rows or columns ---


def _q_b_block(d):
    """Each head's `qk` rows are contiguous, so a rank's heads are one block."""
    return d.attn_tp_rank, d.heads * d.qk_dim


def _kv_b_block(d):
    """Per-head `[k_nope | v]` rows, so a rank's heads are one block."""
    return d.attn_tp_rank, d.heads * (d.nope + d.v_dim)


def _o_block(d):
    """The input columns of this rank's heads (row-parallel)."""
    return d.attn_tp_rank, d.heads * d.v_dim


def _expert_rows(d):
    """FC1 output rows of this rank's moe_tp slice, the same for up and gate."""
    return d.moe_tp_rank, d.moe_inter


def _expert_cols_w(d):
    """FC2 input columns of this rank's moe_tp slice, two values per byte."""
    return d.moe_tp_rank, d.moe_inter // 2


def _expert_cols_s(d):
    """FC2 scale columns of this rank's moe_tp slice, one per 16 values."""
    return d.moe_tp_rank, d.moe_inter // SF_BLOCK


def _expert_cols_bf16(d):
    """FC2 input columns of this rank's moe_tp slice, one value per element (the
    MTP layer's bf16 experts)."""
    return d.moe_tp_rank, d.moe_inter


def _shared_rows(d):
    """Gate and up rows of this rank's slice of a shared expert's intermediate."""
    return d.mlp_tp_rank, d.shared_inter


def _shared_cols(d):
    """Down input columns of the same slice, one value per element (bf16)."""
    return d.mlp_tp_rank, d.shared_inter


def _attn_layers(d):
    """Every trunk layer, plus the MTP module when it is live.

    The MTP module is layer `num_hidden_layers` of the same checkpoint and of the
    same KV pool, and its attention block and norms are a trunk layer's exactly --
    same shape, same key template, same cut and transform. They are those weights,
    one layer further along, not a second family that happens to look alike.
    """
    return range(d.num_layers + (1 if d.mtp_enabled else 0))


def _router_layers(d):
    """Layers with a router: the trunk's MoE layers, plus MTP when live."""
    return range(d.dense_layers, d.num_layers + (1 if d.mtp_enabled else 0))


def _moe_layers(d):
    """Trunk layers with NVFP4 routed experts and an NVFP4 shared expert. MTP is
    excluded: its experts are bf16, so they are different weights rather than
    these at another layer."""
    return range(d.dense_layers, d.num_layers)


def _mtp_only(d):
    """The MTP module alone, and nothing at all when it is not live."""
    return [d.num_layers] if d.mtp_enabled else []


def _dense_mlp_layers(d):
    """The leading dense layers, whose MLP is the checkpoint's own `mlp`."""
    return range(d.dense_layers)


def _one(template):
    """A single checkpoint key, no index into the destination."""
    return lambda d, i, t=template: [(t.format(i=i), None)]


def _scalar(template):
    """A per-tensor scalar, written to element 0 of a one-element parameter."""
    return lambda d, i, t=template: [(t.format(i=i), (0,))]


def _pair(a, b):
    """Two checkpoint tensors handed to the transform together."""
    return lambda d, i, x=a, y=b: [((x.format(i=i), y.format(i=i)), None)]


def _expert_window(suffix, *, pair_with=None):
    """This rank's EP window of the routed experts, one source per local expert.

    The checkpoint key depends on the rank -- `expert_offset` is this rank's first
    expert -- which is why the source is computed rather than templated.
    """

    def build(d, i):
        out = []
        for local in range(d.local_experts):
            q = f"model.layers.{i}.mlp.experts.{d.expert_offset + local}"
            if pair_with is None:
                out.append((f"{q}.{suffix}", (local,)))
            else:
                out.append(((f"{q}.{pair_with}", f"{q}.{suffix}"), (local,)))
        return out

    return build


def _expert_scalars(suffix):
    """A per-tensor scalar replicated over the whole routing space, not windowed.

    The routed FC1 activation scale is the shared expert's `input_scale`, which
    is the max over every routed expert, so all of them are present on every rank.
    """
    return lambda d, i: [
        (f"model.layers.{i}.mlp.experts.{e}.{suffix}", (e,)) for e in range(d.num_experts)
    ]


#: The six per-tensor NVFP4 scalars every MLP and every routed expert carries, as
#: (parameter suffix, checkpoint suffix). The checkpoint stores reciprocals.
_MLP_SCALARS = (
    ("isc1", "gate_proj.input_scale"),
    ("isc1_up", "up_proj.input_scale"),
    ("ws2_1", "gate_proj.weight_scale_2"),
    ("ws2_1_up", "up_proj.weight_scale_2"),
    ("isc2", "down_proj.input_scale"),
    ("ws2_2", "down_proj.weight_scale_2"),
)


def _nvfp4_mlp(key, prefix, inter, layers):
    """The ten rows of one NVFP4 SwiGLU MLP at this rank's intermediate slice:
    a fused gate_up, a down, and six scalars.

    Used twice over disjoint layer ranges -- the dense layers' own `mlp` and the MoE
    layers' shared expert -- because the two have the same shape in terms of their own
    intermediate width, and different widths. `inter(d)` is this rank's width; the
    slice is over the attention TP group.
    """
    return (
        W(
            f"{key}_gu_w",
            shape=lambda d: (2 * inter(d), d.hidden // 2),
            dtype=torch.uint8,
            src=_pair(f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"),
            cut=_rows(lambda d: (d.mlp_tp_rank, inter(d))),
            transform=_cat_rows,
            layers=layers,
        ),
        W(
            f"{key}_gu_s",
            shape=lambda d: (2 * inter(d) * (d.hidden // SF_BLOCK),),
            dtype=torch.uint8,
            src=_pair(f"{prefix}.gate_proj.weight_scale", f"{prefix}.up_proj.weight_scale"),
            cut=_rows(lambda d: (d.mlp_tp_rank, inter(d))),
            transform=_cat_rows_swizzle,
            layers=layers,
        ),
        W(
            f"{key}_dn_w",
            shape=lambda d: (d.hidden, inter(d) // 2),
            dtype=torch.uint8,
            src=_one(f"{prefix}.down_proj.weight"),
            cut=_cols(lambda d: (d.mlp_tp_rank, inter(d) // 2)),
            layers=layers,
        ),
        W(
            f"{key}_dn_s",
            shape=lambda d: (d.hidden * (inter(d) // SF_BLOCK),),
            dtype=torch.uint8,
            src=_one(f"{prefix}.down_proj.weight_scale"),
            cut=_cols(lambda d: (d.mlp_tp_rank, inter(d) // SF_BLOCK)),
            transform=_swizzle_only,
            layers=layers,
        ),
        *(
            W(
                f"{key}_{n}",
                shape=lambda d: (1,),
                dtype=torch.float32,
                src=_scalar(f"{prefix}.{s}"),
                layers=layers,
            )
            for n, s in _MLP_SCALARS
        ),
    )


class DeepseekV3DefaultWeights:
    """deepseek-r1-0528-nvfp4 / sm_10x / default.

    The table, and the passes over it: `declare` allocates every weight it names,
    `manifest` maps checkpoint keys onto them, `read` turns one manifest source into
    the tensor to copy, `load` fills them. `declare` and `manifest` share one "which
    parameters exist" decision (`_targets`), so they cannot disagree about a weight.
    """

    RELEASE_AFTER = "_fc2_s"

    WEIGHTS: tuple[W, ...] = (
        # --- attention and the two layer norms: every trunk layer, and MTP ---
        W(
            "norm1",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.input_layernorm.weight"),
            layers=_attn_layers,
        ),
        W(
            "qa",
            shape=lambda d: (d.q_lora, d.hidden),
            src=_one("model.layers.{i}.self_attn.q_a_proj.weight"),
            layers=_attn_layers,
        ),
        W(
            "q_norm",
            shape=lambda d: (d.q_lora,),
            src=_one("model.layers.{i}.self_attn.q_a_layernorm.weight"),
            layers=_attn_layers,
        ),
        W(
            "qb",
            shape=lambda d: (d.heads * d.qk_dim, d.q_lora),
            src=_one("model.layers.{i}.self_attn.q_b_proj.weight"),
            cut=_rows(_q_b_block),
            layers=_attn_layers,
        ),
        W(
            "kva",
            shape=lambda d: (d.lat_dim, d.hidden),
            src=_one("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
            layers=_attn_layers,
        ),
        W(
            "kv_norm",
            shape=lambda d: (d.kv_lora,),
            src=_one("model.layers.{i}.self_attn.kv_a_layernorm.weight"),
            layers=_attn_layers,
        ),
        W(
            "kvb",
            shape=lambda d: (d.heads * (d.nope + d.v_dim), d.kv_lora),
            src=_one("model.layers.{i}.self_attn.kv_b_proj.weight"),
            cut=_rows(_kv_b_block),
            transform=_regroup_kv_b,
            layers=_attn_layers,
        ),
        W(
            "o",
            shape=lambda d: (d.hidden, d.heads * d.v_dim),
            src=_one("model.layers.{i}.self_attn.o_proj.weight"),
            cut=_cols(_o_block),
            layers=_attn_layers,
        ),
        # The fp8 KV-cache scales sit under the k/v projections this MLA checkpoint does
        # not otherwise have.
        W(
            "k_scale",
            shape=lambda d: (1,),
            dtype=torch.float32,
            src=_scalar("model.layers.{i}.self_attn.k_proj.k_scale"),
            layers=_attn_layers,
        ),
        W(
            "v_scale",
            shape=lambda d: (1,),
            dtype=torch.float32,
            src=_scalar("model.layers.{i}.self_attn.v_proj.v_scale"),
            layers=_attn_layers,
        ),
        W(
            "norm2",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.post_attention_layernorm.weight"),
            layers=_attn_layers,
        ),
        # --- the NVFP4 MLP: the same parameter key over two layer families ---
        *_nvfp4_mlp("mlp", "model.layers.{i}.mlp", lambda d: d.dense_inter, _dense_mlp_layers),
        *_nvfp4_mlp(
            "mlp",
            "model.layers.{i}.mlp.shared_experts",
            lambda d: d.shared_inter,
            _moe_layers,
        ),
        # --- routing: the trunk's MoE layers, and MTP ---
        W(
            "router",
            shape=lambda d: (d.num_experts, d.hidden),
            src=_one("model.layers.{i}.mlp.gate.weight"),
            layers=_router_layers,
        ),
        W(
            "router_bias",
            shape=lambda d: (d.num_experts,),
            dtype=torch.float32,
            src=_one("model.layers.{i}.mlp.gate.e_score_correction_bias"),
            layers=_router_layers,
        ),
        # --- this rank's routed experts, NVFP4, at its moe_tp slice ---
        W(
            "fc1_w",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // 2),
            dtype=torch.uint8,
            src=_expert_window("gate_proj.weight", pair_with="up_proj.weight"),
            cut=_rows(_expert_rows),
            transform=_expert_fc1_w,
            layers=_moe_layers,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // SF_BLOCK),
            dtype=torch.uint8,
            src=_expert_window("gate_proj.weight_scale", pair_with="up_proj.weight_scale"),
            cut=_rows(_expert_rows),
            transform=_expert_fc1_s,
            layers=_moe_layers,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // 2),
            dtype=torch.uint8,
            src=_expert_window("down_proj.weight"),
            cut=_cols(_expert_cols_w),
            transform=_expert_fc2_w,
            layers=_moe_layers,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // SF_BLOCK),
            dtype=torch.uint8,
            src=_expert_window("down_proj.weight_scale"),
            cut=_cols(_expert_cols_s),
            transform=_expert_fc2_s,
            layers=_moe_layers,
        ),
        # --- the routed experts' per-tensor scalars, over the whole routing space ---
        *(
            W(
                f"e_{n}",
                shape=lambda d: (d.num_experts,),
                dtype=torch.float32,
                src=_expert_scalars(s),
                layers=_moe_layers,
            )
            for n, s in _MLP_SCALARS
        ),
        # --- the MTP module's own weights: the ones a trunk layer has no counterpart for ---
        W(
            "enorm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.enorm.weight"),
            layers=_mtp_only,
        ),
        W(
            "hnorm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.hnorm.weight"),
            layers=_mtp_only,
        ),
        # Replicated: every rank holds the whole residual stream it mixes into.
        W(
            "eh",
            shape=lambda d: (d.hidden, 2 * d.hidden),
            src=_one("model.layers.{i}.eh_proj.weight"),
            layers=_mtp_only,
        ),
        W(
            "head_norm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.shared_head.norm.weight"),
            layers=_mtp_only,
        ),
        # bf16, not NVFP4, so a weight of its own rather than the trunk's `mlp_*`
        # at another layer: gate rows first, the half order the dense gate_up reads.
        W(
            "sh_gu",
            shape=lambda d: (2 * d.shared_inter, d.hidden),
            src=_pair(
                "model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                "model.layers.{i}.mlp.shared_experts.up_proj.weight",
            ),
            cut=_rows(_shared_rows),
            transform=_cat_rows,
            layers=_mtp_only,
        ),
        W(
            "sh_dn",
            shape=lambda d: (d.hidden, d.shared_inter),
            src=_one("model.layers.{i}.mlp.shared_experts.down_proj.weight"),
            cut=_cols(_shared_cols),
            layers=_mtp_only,
        ),
        # This rank's EP window of the bf16 routed experts, at its moe_tp slice.
        W(
            "fc1",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden),
            src=_expert_window("gate_proj.weight", pair_with="up_proj.weight"),
            cut=_rows(_expert_rows),
            transform=_mtp_fc1,
            layers=_mtp_only,
        ),
        W(
            "fc2",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter),
            src=_expert_window("down_proj.weight"),
            cut=_cols(_expert_cols_bf16),
            layers=_mtp_only,
        ),
        # --- not per-layer ---
        W("final_norm", shape=lambda d: (d.hidden,), src="model.norm.weight", layers=None),
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src="model.embed_tokens.weight",
            layers=None,
        ),
    )

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares, at this rank's share.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        """`[(param key, layer index | None), ...]`.

        One place decides which parameters an entry declares, so `declare` and
        `manifest` cannot disagree about whether a weight exists.
        """
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
            cut = entry.cut(d) if entry.cut is not None else None
            for key, i in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, cut, entry.transform) for ckpt, index in entry.sources(d, i)
                ]
        return rows

    def read(self, weights, ckpt_key, cut: Cut | None, transform, core, device=None):
        """One manifest source -> the tensor its destination receives.

        Every key of the source is read through the same `cut` -- only this rank's
        rows or columns leave the checkpoint -- moved to `device`, and handed to
        `transform`. The one path `load` takes, so the shard test drives it directly.
        """
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

    def expected_unconsumed(self, core, weights) -> set:
        return _offwindow_expert_keys(core) | _mtp_keys(core, weights)

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint. Each source is read through
        its cut, multi-key rows feed their transform several tensors at once, and
        coverage is asserted in both directions, not only checkpoint-side."""
        core = model.model
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
            for ckpt_key, index, cut, transform in sources:
                fill(core.w[param_key], ckpt_key, index, cut, transform)
            # The expert relayouts allocate per-expert scratch once per local expert
            # per layer; release it before the next parameter so peak load memory
            # stays one layer deep.
            if param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

        # The shell builds lm_head vocab-parallel under TP and whole under
        # attention DP; read the rows it built and nothing else.
        mapping = core.model_config.mapping
        cut, pad = lm_head_share(model.lm_head.weight.shape[0], core.dims.vocab, mapping.tp_rank)
        fill(model.lm_head.weight, "lm_head.weight", None, cut, pad)
        # Parameter-side coverage, the other direction: the shell's construction
        # is topology-dependent, so name every registered parameter outside the
        # target's own ParameterDict instead of assuming lm_head is the only one.
        shell = {n for n, _ in model.named_parameters() if not n.startswith("model.w.")}
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"

        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(core, weights)
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly this rank's "
            f"off-window experts plus what the MTP layer leaves; unexpected "
            f"{sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )

    def dims(self, core) -> SimpleNamespace:
        """Every per-rank width, from the config and the engine's Mapping."""
        cfg = core.model_config.pretrained_config
        m = core.model_config.mapping
        mtp_enabled = getattr(core.model_config, "spec_config", None) is not None
        assert not mtp_enabled or int(getattr(cfg, "num_nextn_predict_layers", 0) or 0) >= 1, (
            "a speculative_config needs the checkpoint's MTP layer"
        )
        attention_dp = bool(m.enable_attention_dp)
        attn_tp = 1 if attention_dp else m.tp_size
        attn_tp_rank = 0 if attention_dp else m.tp_rank
        heads_total = cfg.num_attention_heads
        assert heads_total % attn_tp == 0, (heads_total, attn_tp)
        assert cfg.n_routed_experts % m.moe_ep_size == 0, (cfg.n_routed_experts, m.moe_ep_size)
        local_experts = cfg.n_routed_experts // m.moe_ep_size
        shared_full = cfg.moe_intermediate_size * cfg.n_shared_experts
        # The built-in model sizes the dense MLP's TP down when it would cross
        # nodes (`_compute_mlp_tp_size`); this target splits it over the whole
        # attention TP group, and the allreduce that follows spans the group
        # either way.
        widths = {
            "dense": (cfg.intermediate_size, attn_tp),
            "shared": (shared_full, attn_tp),
            "expert": (cfg.moe_intermediate_size, m.moe_tp_size),
        }
        for name, (full, parts) in widths.items():
            assert full % parts == 0 and (full // parts) % _ROW_ALIGN == 0, (
                f"{name} intermediate {full} does not split {parts} ways on "
                f"{_ROW_ALIGN}-row NVFP4 tiles"
            )
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            heads_total=heads_total,
            attention_dp=attention_dp,
            attn_tp=attn_tp,
            attn_tp_rank=attn_tp_rank,
            heads=heads_total // attn_tp,
            nope=cfg.qk_nope_head_dim,
            rope_dim=cfg.qk_rope_head_dim,
            v_dim=cfg.v_head_dim,
            kv_lora=cfg.kv_lora_rank,
            q_lora=cfg.q_lora_rank,
            qk_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
            lat_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            vocab=cfg.vocab_size,
            dense_layers=cfg.first_k_dense_replace,
            num_experts=cfg.n_routed_experts,
            mlp_tp=attn_tp,
            mlp_tp_rank=attn_tp_rank,
            dense_inter=cfg.intermediate_size // attn_tp,
            shared_inter=shared_full // attn_tp,
            moe_tp=m.moe_tp_size,
            moe_tp_rank=m.moe_tp_rank,
            moe_inter=cfg.moe_intermediate_size // m.moe_tp_size,
            moe_ep=m.moe_ep_size,
            moe_ep_rank=m.moe_ep_rank,
            local_experts=local_experts,
            expert_offset=local_experts * m.moe_ep_rank,
            mtp_enabled=mtp_enabled,
            dtype=core.model_config.torch_dtype,
        )


MODEL_WEIGHTS = DeepseekV3DefaultWeights()


def lm_head_share(rows: int, vocab: int, tp_rank: int) -> tuple[Cut, Callable]:
    """The cut and the pad that fill an lm_head the shell built with `rows` rows.

    Whole under attention DP (`rows == vocab`); a vocab-parallel shard under TP.
    A vocab that does not split evenly leaves the last shard short, and its
    missing rows are zero, as the built-in LMHead pads them -- so the cut stops
    at `vocab` and the pad fills the rest.
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


def _offwindow_expert_keys(core) -> set:
    """The first predicted non-load: the weight/scale tensors of every routed
    expert outside this rank's EP window. Their per-tensor scalars are consumed
    on every rank, so nothing else of theirs is left over.

    The trunk's MoE layers store six such tensors per expert (NVFP4 data plus
    block scales); the MTP layer's are bf16, so when it is live its off-window
    experts leave three each and no `weight_scale`."""
    d = core.dims
    lo, hi = d.expert_offset, d.expert_offset + d.local_experts
    keys = set()
    for i in _router_layers(d):
        quantized = i < d.num_layers
        for e in range(d.num_experts):
            if lo <= e < hi:
                continue
            q = f"model.layers.{i}.mlp.experts.{e}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys.add(f"{q}.{proj}.weight")
                if quantized:
                    keys.add(f"{q}.{proj}.weight_scale")
    return keys


def _mtp_keys(core, weights) -> set:
    """The second predicted non-load: what the MTP layers the checkpoint ships
    past `num_hidden_layers` leave behind. Read off the checkpoint by layer
    index rather than enumerated, so a key belonging to a real layer can never
    land here.

    Without MTP that is those layers whole -- on this checkpoint layer 61's 790
    keys. With MTP the table consumes all but two per rank (and this rank's
    off-window experts, which the family above names): `embed_tokens` and
    `shared_head.head` are bitwise copies of the trunk's embedding and
    `lm_head`, which the draft model reads instead of loading them twice."""
    cfg = core.model_config.pretrained_config
    n = cfg.num_hidden_layers
    extra = int(getattr(cfg, "num_nextn_predict_layers", 0) or 0)
    prefixes = tuple(f"model.layers.{i}." for i in range(n, n + extra))
    if not prefixes:
        return set()
    keys = {k for k in weights if k.startswith(prefixes)}
    if not core.dims.mtp_enabled:
        return keys
    aliased = (".embed_tokens.weight", ".shared_head.head.weight")
    return {k for k in keys if k.endswith(aliased)}
