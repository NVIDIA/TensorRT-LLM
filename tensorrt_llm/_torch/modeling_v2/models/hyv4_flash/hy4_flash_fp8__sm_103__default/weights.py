# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loader of the text core, the vision tower and the DFly2 drafter.

Each rank reads only its cut of a lazy checkpoint entry. Checkpoint quirks: `kv_b_proj` rows are
per-head `[nope | v]`; expert FC1 is stored gate-first and the runner reads `[up ; gate]`; block-FP8
scales are stored bf16 and the runner takes fp32. The four Engram tables live in pinned host memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash import engram_host

ALL_LAYERS = object()

#: An index into a checkpoint tensor: a tuple of slices, one per leading axis.
Cut = tuple[slice, ...]


@dataclass(frozen=True)
class W:
    """One weight role.

    `shape(d, i)` is this rank's declared shape. `src` is a key template (`{i}` = layer) or a
    callable `(d, i) -> [(key, index into the param | None), ...]`. `cut(d, i)` is this rank's
    rows/columns of the checkpoint tensor (None: whole). `transform(t, host)` relayouts the cut.
    """

    name: str
    shape: Callable[..., tuple[int, ...]]
    src: Any
    dtype: torch.dtype | None = None
    cut: Callable[[Any, int | None], Cut] | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))

    def sources(self, d: SimpleNamespace, i: int | None) -> list[tuple[Any, Any]]:
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
    return d.moe_ep_rank, d.local_experts


def _kv_b_rows(d, i=None) -> tuple[int, int]:
    """Whole `nope + v` row blocks of this rank's heads."""
    return d.attn_tp_rank, d.layers[i].heads_local * (d.nope + d.v_dim)


def _q_b_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local * d.qk_dim


def _q_b_scale_rows(d, i=None) -> tuple[int, int]:
    """Block-FP8 scales are 128 wide."""
    return d.attn_tp_rank, d.layers[i].heads_local * d.qk_dim // 128


def _v_rows(d, i=None) -> tuple[int, int]:
    """`linear_gate` rows / `o_proj` columns."""
    return d.attn_tp_rank, d.layers[i].heads_local * d.v_dim


def _v_scale_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local * d.v_dim // 128


def _sink_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.layers[i].heads_local


def _kv_b_k(t: torch.Tensor, host) -> torch.Tensor:
    """`[heads * (nope + v), kv_lora]` -> `[heads, nope, kv_lora]`."""
    d = host.dims
    return t.view(-1, d.nope + d.v_dim, d.kv_lora)[:, : d.nope].contiguous()


def _kv_b_v(t: torch.Tensor, host) -> torch.Tensor:
    """`[heads * (nope + v), kv_lora]` -> `[heads, kv_lora, v_dim]`."""
    d = host.dims
    return t.view(-1, d.nope + d.v_dim, d.kv_lora)[:, d.nope :].transpose(1, 2).contiguous()


def _kv_b_context(t: torch.Tensor, host) -> torch.Tensor:
    """Per-head `[nope | v]` rows -> `[all heads' nope][all heads' v]` (a row permutation)."""
    d = host.dims
    view = t.view(-1, d.nope + d.v_dim, d.kv_lora)
    return torch.cat(
        [
            view[:, : d.nope].reshape(-1, d.kv_lora),
            view[:, d.nope :].reshape(-1, d.kv_lora),
        ],
        dim=0,
    ).contiguous()


def _experts_fc1(t: torch.Tensor, host) -> torch.Tensor:
    """`[E_loc, 2I, H]` gate-first -> `[E_loc, 2I/moe_tp, H]` up-first, this rank's I slice."""
    d = host.dims
    e, two_i, h = t.shape
    i = two_i // 2
    view = t.view(e, 2, i, h)
    gate, up = view[:, 0], view[:, 1]
    span = i // d.moe_tp
    lo = d.moe_tp_rank * span
    return torch.cat([up[:, lo : lo + span], gate[:, lo : lo + span]], dim=1).contiguous()


def _experts_fc1_scale(t: torch.Tensor, host) -> torch.Tensor:
    """`[E_loc, 2I/128, H/128]` bf16 gate-first -> fp32 up-first, this rank's slice."""
    d = host.dims
    e, two_b, hb = t.shape
    b = two_b // 2
    view = t.view(e, 2, b, hb)
    gate, up = view[:, 0], view[:, 1]
    span = b // d.moe_tp
    lo = d.moe_tp_rank * span
    return torch.cat([up[:, lo : lo + span], gate[:, lo : lo + span]], dim=1).float().contiguous()


def _experts_fc2(t: torch.Tensor, host) -> torch.Tensor:
    """`[E_loc, H, I]` -> `[E_loc, H, I/moe_tp]`."""
    d = host.dims
    span = t.shape[2] // d.moe_tp
    lo = d.moe_tp_rank * span
    return t[:, :, lo : lo + span].contiguous()


def _experts_fc2_scale(t: torch.Tensor, host) -> torch.Tensor:
    """`[E_loc, H/128, I/128]` bf16 -> fp32, this rank's columns."""
    d = host.dims
    span = t.shape[2] // d.moe_tp
    lo = d.moe_tp_rank * span
    return t[:, :, lo : lo + span].float().contiguous()


def _fp32(t: torch.Tensor, host) -> torch.Tensor:
    return t.float().contiguous()


def _vis_qkv(d, i) -> list[tuple[str, slice]]:
    """The fused `is_fused_qkv` weight: q's rows, then k's, then v's."""
    hidden = d.vision.hidden
    prefix = f"vit.layers.{i}.self_attn"
    return [
        (f"{prefix}.q_proj.weight", slice(0, hidden)),
        (f"{prefix}.k_proj.weight", slice(hidden, 2 * hidden)),
        (f"{prefix}.v_proj.weight", slice(2 * hidden, 3 * hidden)),
    ]


def _vis_qkv_b(d, i) -> list[tuple[str, slice]]:
    hidden = d.vision.hidden
    prefix = f"vit.layers.{i}.self_attn"
    return [
        (f"{prefix}.q_proj.bias", slice(0, hidden)),
        (f"{prefix}.k_proj.bias", slice(hidden, 2 * hidden)),
        (f"{prefix}.v_proj.bias", slice(2 * hidden, 3 * hidden)),
    ]


def _sparse_layers(d) -> list[int]:
    return list(d.sparse_layers)


def _dense_layers(d) -> list[int]:
    return list(range(d.first_dense))


def _moe_layers(d) -> list[int]:
    return list(range(d.first_dense, d.num_layers))


def _engram_layers(d) -> list[int]:
    return list(d.engram_layers)


def _vision_layers(d) -> list[int]:
    return [] if d.vision is None else list(range(d.vision.depth))


class _WeightTable:
    """Declare, map and load a tuple of `W` roles against a host with `dims` (and `w` to load)."""

    WEIGHTS: tuple[W, ...] = ()
    #: Release cached blocks after a parameter whose relayout allocates per-expert scratch.
    RELEASE_AFTER: str | None = None

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
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
        return entry.shape(d) if i is None else entry.shape(d, i)

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        """`[(param key, layer index | None), ...]`."""
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None)]
        return [(f"l{i}_{entry.name}", i) for i in layers]

    def manifest(self, host) -> dict[str, list[tuple[Any, Any, Cut | None, Callable | None]]]:
        """param key -> [(checkpoint key, index into the param | None, cut, transform)]."""
        d = host.dims
        rows: dict[str, list[tuple[Any, Any, Cut | None, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i in self._targets(entry, d):
                cut = entry.cut(d, i) if entry.cut is not None else None
                rows[key] = [
                    (ckpt, index, cut, entry.transform) for ckpt, index in entry.sources(d, i)
                ]
        return rows

    def read(self, weights, ckpt_key, cut: Cut | None, transform, host, device=None):
        """One manifest source -> the tensor its destination receives."""
        keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
        for key in keys:
            assert key in weights, f"checkpoint key missing: {key}"
        src_all = tuple(_read(weights[key], cut) for key in keys)
        if device is not None:
            src_all = tuple(t.to(device, non_blocking=True) for t in src_all)
        if transform is not None:
            return transform(src_all if isinstance(ckpt_key, tuple) else src_all[0], host)
        assert len(src_all) == 1, "a multi-key row needs a transform"
        return src_all[0]

    def _fill(self, weights, param, ckpt_key, index, cut, transform, host, consumed: set) -> None:
        dst = param.data if index is None else param.data[index]
        src = self.read(weights, ckpt_key, cut, transform, host, dst.device)
        assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
        assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
        dst.copy_(src, non_blocking=True)
        consumed.update(ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,))

    def _fill_table(self, host, weights, consumed: set) -> None:
        manifest = self.manifest(host)
        assert set(manifest.keys()) == set(host.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(host.w.keys()),
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, cut, transform in sources:
                self._fill(
                    weights, host.w[param_key], ckpt_key, index, cut, transform, host, consumed
                )
            if self.RELEASE_AFTER is not None and param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()

    def _check_coverage(self, host, weights, consumed: set) -> None:
        torch.cuda.synchronize()
        leftover = set(weights.keys()) - consumed
        expected = self.expected_unconsumed(host, weights)
        assert leftover == expected, (
            f"{type(self).__name__} checkpoint coverage: "
            f"unexpected {sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )

    def load(self, host, weights) -> None:
        """Fill every declared weight of `host.w` and check the checkpoint's coverage."""
        consumed: set = set()
        self._fill_table(host, weights, consumed)
        self._check_coverage(host, weights, consumed)


class Hyv4FlashUnifiedWeights(_WeightTable):
    """The text core's and the vision tower's weights at one rank's share."""

    RELEASE_AFTER = "experts_down_scale"

    WEIGHTS: tuple[W, ...] = (
        # --- hyper-connections (fp32) ---
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
        # The same rows again, ordered for the unabsorbed context attention.
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
            cut=_rows(_q_b_scale_rows),
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
            cut=_rows(_v_rows),
            layers=_moe_layers,
        ),
        W(
            "linear_gate_scale",
            lambda d, i=None: (d.layers[i].heads_local * d.v_dim // 128, d.hidden // 128),
            "model.layers.{i}.self_attn.linear_gate.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_v_scale_rows),
            transform=_fp32,
            layers=_moe_layers,
        ),
        W(
            "linear_gate",
            lambda d, i=None: (d.layers[i].heads_local * d.v_dim, d.hidden),
            "model.layers.{i}.self_attn.linear_gate.weight",
            cut=_rows(_v_rows),
            layers=_dense_layers,
        ),
        W(
            "o_proj",
            lambda d, i=None: (d.hidden, d.layers[i].heads_local * d.v_dim),
            "model.layers.{i}.self_attn.o_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_cols(_v_rows),
            layers=_moe_layers,
        ),
        W(
            "o_proj_scale",
            lambda d, i=None: (d.hidden // 128, d.layers[i].heads_local * d.v_dim // 128),
            "model.layers.{i}.self_attn.o_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_cols(_v_scale_rows),
            transform=_fp32,
            layers=_moe_layers,
        ),
        W(
            "o_proj",
            lambda d, i=None: (d.hidden, d.layers[i].heads_local * d.v_dim),
            "model.layers.{i}.self_attn.o_proj.weight",
            cut=_cols(_v_rows),
            layers=_dense_layers,
        ),
        # --- compressed indexer (sparse layers) ---
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
        # --- dense MLP (layer 0), intermediate split over the attention group ---
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
        # --- MoE ---
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
        # The latent pre/post projections are whole on every rank: MoE TP cuts the experts'
        # intermediate, never the latent.
        W(
            "pre_expert",
            lambda d, i=None: (d.latent, d.hidden),
            "model.layers.{i}.mlp.routed_expert_down_proj.weight",
            layers=_moe_layers,
        ),
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
        # --- Engram, except the host tables (`engram_tables`) ---
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
        W("final_norm", lambda d, i=None: (d.hidden,), "model.norm.weight", layers=None),
        W("embed", lambda d, i=None: (d.vocab, d.hidden), "model.embed_tokens.weight", layers=None),
        # --- the vision tower (`vit.*`, bf16, replicated); none without `d.vision` ---
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
        """One pinned-host table per Engram layer; reserved here, filled in `load`."""
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
        core = model.model
        d = core.dims
        consumed: set = set()
        self._fill_table(core, weights, consumed)

        for i, table in core.engram_tables.items():
            key = f"model.layers.{i}.engram_layer.multi_head_embedding.embedding.weight"
            assert key in weights, f"checkpoint key missing: {key}"
            table.fill(*engram_host.shard_rows(weights, [key], d.engram_table_rows[i], _read))
            consumed.add(key)

        # lm_head is vocab-parallel under TP and whole under attention DP.
        mapping = core.model_config.mapping
        cut, pad = lm_head_share(model.lm_head.weight.shape[0], d.vocab, mapping.tp_rank)
        self._fill(weights, model.lm_head.weight, "lm_head.weight", None, cut, pad, core, consumed)
        # The drafter's parameters are filled by `load_draft_weights`.
        unfed = {
            n
            for n, _ in model.named_parameters()
            if n != "lm_head.weight" and not n.startswith(("model.w.", "draft_model.w."))
        }
        assert not unfed, f"unfed shell parameters: {sorted(unfed)}"

        self._check_coverage(core, weights, consumed)

    def expected_unconsumed(self, core, weights) -> set:
        """The drafter's keys, and `vit.*` when the config has no tower."""
        tower_keys_expected = core.dims.vision is None
        return {
            key
            for key in weights.keys()
            if key.startswith("model.dfly_layers.")
            or (tower_keys_expected and key.startswith("vit."))
        }


def _draft_q_b_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.qk_dim


def _draft_q_b_scale_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.qk_dim // 128


def _draft_v_rows(d, i=None) -> tuple[int, int]:
    """`linear_gate` rows / `o_proj` columns."""
    return d.attn_tp_rank, d.heads_local * d.v_dim


def _draft_v_scale_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * d.v_dim // 128


def _draft_kv_b_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local * (d.nope + d.v_dim)


def _draft_sink_rows(d, i=None) -> tuple[int, int]:
    return d.attn_tp_rank, d.heads_local


def _draft_mlp_rows(d, i=None) -> tuple[int, int]:
    """FFN gate/up rows and down columns."""
    return d.mlp_tp_rank, d.mlp_inter // d.mlp_tp


def _draft_mlp_scale_rows(d, i=None) -> tuple[int, int]:
    return d.mlp_tp_rank, d.mlp_inter // d.mlp_tp // 128


class Hyv4FlashDraftWeights(_WeightTable):
    """The DFly2 drafter's five layers and its fusion/correction/confidence heads (`model.dfly_layers.*`)."""

    WEIGHTS: tuple[W, ...] = (
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
            transform=_kv_b_k,
        ),
        W(
            "v_b_proj",
            lambda d, i=None: (d.heads_local, d.kv_lora, d.v_dim),
            "model.dfly_layers.{i}.self_attn.kv_b_proj.weight",
            cut=_rows(_draft_kv_b_rows),
            transform=_kv_b_v,
        ),
        W(
            "linear_gate",
            lambda d, i=None: (d.heads_local * d.v_dim, d.hidden),
            "model.dfly_layers.{i}.self_attn.linear_gate.weight",
            dtype=torch.float8_e4m3fn,
            cut=_rows(_draft_v_rows),
        ),
        W(
            "linear_gate_scale",
            lambda d, i=None: (d.heads_local * d.v_dim // 128, d.hidden // 128),
            "model.dfly_layers.{i}.self_attn.linear_gate.weight_scale_inv",
            dtype=torch.float32,
            cut=_rows(_draft_v_scale_rows),
            transform=_fp32,
        ),
        W(
            "o_proj",
            lambda d, i=None: (d.hidden, d.heads_local * d.v_dim),
            "model.dfly_layers.{i}.self_attn.o_proj.weight",
            dtype=torch.float8_e4m3fn,
            cut=_cols(_draft_v_rows),
        ),
        W(
            "o_proj_scale",
            lambda d, i=None: (d.hidden // 128, d.heads_local * d.v_dim // 128),
            "model.dfly_layers.{i}.self_attn.o_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_cols(_draft_v_scale_rows),
            transform=_fp32,
        ),
        # The checkpoint stores the draft sinks bf16; the attention op requires fp32.
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
            cut=_cols(_draft_mlp_rows),
        ),
        W(
            "mlp_down_scale",
            lambda d, i=None: (d.hidden // 128, d.mlp_inter // d.mlp_tp // 128),
            "model.dfly_layers.{i}.mlp.down_proj.weight_scale_inv",
            dtype=torch.float32,
            cut=_cols(_draft_mlp_scale_rows),
            transform=_fp32,
        ),
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
        """Every key outside `model.dfly_layers.`."""
        return {key for key in weights.keys() if not key.startswith("model.dfly_layers.")}


MODEL_WEIGHTS = Hyv4FlashUnifiedWeights()
DRAFT_WEIGHTS = Hyv4FlashDraftWeights()


def lm_head_share(rows: int, vocab: int, tp_rank: int) -> tuple[Cut, Callable]:
    """The cut and zero-pad that fill an lm_head of `rows` rows (the last TP shard may be short)."""
    start = 0 if rows == vocab else tp_rank * rows
    stop = min(start + rows, vocab)
    assert start < stop, (rows, vocab, tp_rank)

    def pad(t: torch.Tensor, host) -> torch.Tensor:
        if t.shape[0] == rows:
            return t.contiguous()
        out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
        out[: t.shape[0]] = t
        return out

    return (slice(start, stop),), pad
