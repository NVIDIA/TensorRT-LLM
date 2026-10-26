# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight tables and loaders: MiniMax-M3 NVFP4 / sm_103 / tp4.

Two tables, one per checkpoint the target reads:

* ``MODEL_WEIGHTS`` (``MiniMaxM3Tp4Weights``): the vision-language checkpoint's language model,
  ``language_model.*``. Every rank is handed the whole checkpoint, so every tensor-parallel split is this
  table's: the Q heads, the K/V/index-Q heads (one each per rank), the dense MLPs and the shared expert's
  intermediate, the o_proj / down-projection input columns, every routed expert's intermediate (all 128
  experts live on every rank), and the vocabulary of the shell's ``lm_head``. The index-K head is replicated.
* ``DRAFT_WEIGHTS`` (``MiniMaxM3Tp4DraftWeights``): the one-layer BF16 Eagle3 drafter's own checkpoint
  (``speculative_model``), loaded through the shell's ``load_draft_weights`` when speculation is on.

Storage is HF ``[out, in]`` row-major. Five families need a transform, each a layout a kernel demands:

* **MXFP8 block scales** (every attention projection, the dense MLPs and the shared expert): the checkpoint
  stores UE8M0 bytes ``[N, K / 32]``; the GEMMs read the 128x4-swizzled order
  (``trtllm::block_scale_interleave``). A fused projection concatenates its shards' scales first and swizzles
  once.
* **The five-way sparse-layer projection** ``[Q | K | V | index-Q | index-K]`` and the three-way dense one
  ``[Q | K | V]``: the per-rank shards concatenated along N.
* **Gated MLPs** (dense and shared expert): ``[gate ; up]`` rows.
* **NVFP4 routed experts** (layers 3..59): ``[up (w3) ; gate (w1)]`` rows, then the trtllm-gen row shuffle of
  the weight and of its E4M3 scales (gated-activation reorder + 128-row epilogue tile shuffle,
  ``trtllmgen_maybe_get_cached_*_permute_indices``) and the 128x4 swizzle of the scales.
* **Gemma boundary norms**: ``input_layernorm``, ``post_attention_layernorm`` and ``model.norm`` are stored as
  Gemma norms, ``(1 + w) * x``; the fused all-reduce + RMSNorm kernels apply ``w * x``, so ``1`` is added
  (in BF16) at load. The per-head Q/K/index norms stay Gemma: their kernels apply the offset themselves.

The routed experts' per-tensor scalars (input scales and second-level weight scales, every expert) load raw;
``modeling`` derives the kernels' global scales, alphas and activation constants from them once the weights
are real.

Loading contract: ``load(model, weights)`` consumes the engine-provided checkpoint dict, fills every declared
parameter exactly once and asserts coverage in both directions: every declared parameter written, and every
checkpoint key either consumed or in ``expected_unconsumed`` (the vision tower and its projector, which this
text-only target does not build, and the language model's multi-token-prediction head).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

from tensorrt_llm._torch.moe.fused_moe.quantization import (
    trtllmgen_maybe_get_cached_w2_permute_indices,
    trtllmgen_maybe_get_cached_w3_w1_permute_indices,
)

#: ``layers=ALL_LAYERS`` is the common case: a weight every decoder layer carries.
ALL_LAYERS = object()

_LM = "language_model."
#: Checkpoint key families this target does not build: the vision tower and its two projectors (text only), and the
#: language model's multi-token-prediction head (speculation, when on, is the separate Eagle3 drafter).
_UNBUILT_PREFIXES = (
    "vision_tower.",
    "multi_modal_projector.",
    "patch_merge_mlp.",
    _LM + "model.mtp.",
)

MX_BLOCK = 32  # MXFP8: one UE8M0 scale per 32 elements along K
NV_BLOCK = 16  # NVFP4: one E4M3 scale per 16 elements along K
EPILOGUE_TILE_M = (
    128  # the trtllm-gen MoE runner's row tile: fixes its weight and scale row shuffles
)


@dataclass(frozen=True)
class W:
    """One weight role: what it is, where its bytes come from, what happens on the way.

    ``shape`` takes the dimension bundle and returns the declared shape. ``src`` is a callable
    ``(d, i) -> [(key or key tuple, index into the param | None), ...]`` with keys relative to the checkpoint
    root; a key tuple hands the transform those tensors together. ``layers`` is ``ALL_LAYERS``, ``None`` (not
    per-layer) or ``d -> layer indices``. ``transform(src, d)`` runs on the destination's device before the copy.
    """

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        if self.layers is None:
            return None
        if self.layers is ALL_LAYERS:
            return list(range(d.num_layers))
        return list(self.layers(d))


# ------------------------------------------------------------------------------------------------ shards


def _col(
    t: torch.Tensor, d: SimpleNamespace, tp: int | None = None, rank: int | None = None
) -> torch.Tensor:
    """This rank's rows (column-parallel output split), even split, sliced on the host and moved to the
    destination device (only this rank's bytes are read)."""
    tp = d.tp if tp is None else tp
    rank = d.rank if rank is None else rank
    n = t.shape[0] // tp
    return _dev(t[rank * n : (rank + 1) * n], d)


def _row(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """This rank's input columns (row-parallel input split), even split, on the destination device."""
    n = t.shape[1] // d.tp
    return _dev(t[:, d.rank * n : (d.rank + 1) * n], d)


def _dev(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """``t`` on the destination device (``d.dev``, set per fill)."""
    return t.to(d.dev, non_blocking=True)


def _swizzle(scale: torch.Tensor) -> torch.Tensor:
    """128x4 swizzle of a row-major ``[N, K / 32]`` UE8M0 scale matrix, flat."""
    return torch.ops.trtllm.block_scale_interleave(scale.contiguous())


# --------------------------------------------------------------------------------------------- transforms


def _gemma_fold(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """A Gemma boundary norm as the plain norm the fused all-reduce epilogue applies: ``w + 1`` in BF16."""
    return t + 1.0


def _f32(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    return t.to(torch.float32)


def _qkv_sparse_w(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    """``[Q | K | V | index-Q | index-K]``: Q by attention heads, K/V/index-Q by KV heads, index-K replicated."""
    q, k, v, iq, ik = t
    return torch.cat(
        [_col(q, d), _col(k, d), _col(v, d), _col(iq, d), _dev(ik, d)], dim=0
    ).contiguous()


def _qkv_sparse_s(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    q, k, v, iq, ik = t
    return _swizzle(
        torch.cat([_col(q, d), _col(k, d), _col(v, d), _col(iq, d), _dev(ik, d)], dim=0)
    )


def _qkv_dense_w(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    q, k, v = t
    return torch.cat([_col(q, d), _col(k, d), _col(v, d)], dim=0).contiguous()


def _qkv_dense_s(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    q, k, v = t
    return _swizzle(torch.cat([_col(q, d), _col(k, d), _col(v, d)], dim=0))


def _index_qk_w(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    """``[index-Q | index-K]`` in BF16: this rank's index-Q heads and the replicated index-K, the E4M3 values cast
    (the scale tensors, the last two sources, are consumed only)."""
    iq, ik = t[0], t[1]
    return torch.cat([_col(iq, d), _dev(ik, d)], dim=0).to(d.dtype).contiguous()


def _gate_up_w(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    gate, up = t
    return torch.cat([_col(gate, d), _col(up, d)], dim=0).contiguous()


def _gate_up_s(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    gate, up = t
    return _swizzle(torch.cat([_col(gate, d), _col(up, d)], dim=0))


def _row_w(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    return _row(t, d).contiguous()


def _row_s(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    return _swizzle(_row(t, d))


def _col_w(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    return _col(t, d).contiguous()


#: Permute indices of the trtllm-gen expert relayouts, keyed by (shape, family, scale group): one expert's are
#: every expert's, so each is computed once per load.
_PERMUTE_CACHE: dict = {}


def _expert_w31(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    """One expert's FC1 operand: ``[up ; gate]`` packed-FP4 rows of this rank, then the gated-activation row
    reorder and the epilogue-tile shuffle."""
    up, gate = t
    w = torch.cat([_col(up, d), _col(gate, d)], dim=0).contiguous()
    perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
        w, _PERMUTE_CACHE, EPILOGUE_TILE_M, is_gated_act_gemm=True
    )
    return torch.ops.trtllm.shuffle_matrix(w, perm.to(w.device)).view(w.dtype)


def _expert_s31(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    """One expert's FC1 E4M3 scales: ``[up ; gate]`` rows of this rank, the same row reorder at scale
    granularity, then the 128x4 swizzle."""
    up, gate = t
    s = torch.cat([_col(up, d), _col(gate, d)], dim=0).contiguous()
    shape = s.shape
    sf = s.view(torch.uint8)
    perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
        sf, _PERMUTE_CACHE, EPILOGUE_TILE_M, num_elts_per_sf=NV_BLOCK, is_gated_act_gemm=True
    )
    shuffled = torch.ops.trtllm.shuffle_matrix(sf, perm)
    return (
        torch.ops.trtllm.block_scale_interleave(shuffled.view(torch.uint8).reshape(shape))
        .view(torch.float8_e4m3fn)
        .reshape(shape)
    )


def _expert_w2(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """One expert's FC2 operand: this rank's input columns, then the epilogue-tile row shuffle."""
    w = _row(t, d).contiguous()
    perm = trtllmgen_maybe_get_cached_w2_permute_indices(w, _PERMUTE_CACHE, EPILOGUE_TILE_M)
    return torch.ops.trtllm.shuffle_matrix(w, perm.to(w.device)).view(w.dtype)


def _expert_s2(t: torch.Tensor, d: SimpleNamespace) -> torch.Tensor:
    """One expert's FC2 E4M3 scales: this rank's columns, the row shuffle at scale granularity, the swizzle."""
    s = _row(t, d).contiguous()
    shape = s.shape
    sf = s.view(torch.uint8)
    perm = trtllmgen_maybe_get_cached_w2_permute_indices(
        sf, _PERMUTE_CACHE, EPILOGUE_TILE_M, num_elts_per_sf=NV_BLOCK
    )
    shuffled = torch.ops.trtllm.shuffle_matrix(sf, perm)
    return (
        torch.ops.trtllm.block_scale_interleave(shuffled).view(torch.float8_e4m3fn).reshape(shape)
    )


# ------------------------------------------------------------------------------------------------ sources


def _one(template: str):
    return lambda d, i, t=template: [(_LM + t.format(i=i), None)]


def _many(*templates: str):
    return lambda d, i, ts=templates: [(tuple(_LM + t.format(i=i) for t in ts), None)]


def _per_expert(*suffixes: str):
    """One source per routed expert, written at its expert index (all experts are local at tp4)."""

    def build(d, i, sx=suffixes):
        out = []
        for e in range(d.num_experts):
            keys = tuple(_LM + f"model.layers.{i}.block_sparse_moe.experts.{e}.{s}" for s in sx)
            out.append((keys if len(keys) > 1 else keys[0], (e,)))
        return out

    return build


def _dense_layers(d):
    return range(d.first_sparse)


def _sparse_layers(d):
    return range(d.first_sparse, d.num_layers)


def _three_way_qkv_layers(d):
    """Layers whose QKV is the three-way [Q | K | V]: the dense layers, and every layer without the five-way packed
    projection (``fuse_qkv_index_projection=False``)."""
    return range(d.first_sparse) if d.fused else range(d.num_layers)


def _five_way_qkv_layers(d):
    return range(d.first_sparse, d.num_layers) if d.fused else range(0)


def _unfused_sparse_layers(d):
    return range(0) if d.fused else range(d.first_sparse, d.num_layers)


_ATTN = "model.layers.{i}.self_attn"
_MOE = "model.layers.{i}.block_sparse_moe"


class MiniMaxM3Tp4Weights:
    """MiniMax-M3 NVFP4 / sm_103 / tp4: the language model's table and the passes over it.

    ``declare`` allocates every weight it names, ``manifest`` maps checkpoint keys onto them, ``load`` fills
    them; ``_targets`` is the one place that decides which parameters exist.
    """

    #: Free the per-expert relayout scratch after each expert stack.
    RELEASE_AFTER = "_e_s2"

    WEIGHTS: tuple[W, ...] = (
        # --- the two boundary norms of every layer (Gemma, folded) ---
        W(
            "in_norm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.input_layernorm.weight"),
            transform=_gemma_fold,
        ),
        W(
            "post_norm",
            shape=lambda d: (d.hidden,),
            src=_one("model.layers.{i}.post_attention_layernorm.weight"),
            transform=_gemma_fold,
        ),
        # --- attention: per-head Q/K norms (Gemma, unfolded), the packed QKV and the output projection ---
        W("q_norm", shape=lambda d: (d.head_dim,), src=_one(_ATTN + ".q_norm.weight")),
        W("k_norm", shape=lambda d: (d.head_dim,), src=_one(_ATTN + ".k_norm.weight")),
        W(
            "qkv_w",
            shape=lambda d: (d.qkv_dense_rows, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=_many(*(f"{_ATTN}.{p}_proj.weight" for p in "qkv")),
            transform=_qkv_dense_w,
            layers=_three_way_qkv_layers,
        ),
        W(
            "qkv_s",
            shape=lambda d: (d.qkv_dense_rows * (d.hidden // MX_BLOCK),),
            dtype=torch.uint8,
            src=_many(*(f"{_ATTN}.{p}_proj.weight_scale_inv" for p in "qkv")),
            transform=_qkv_dense_s,
            layers=_three_way_qkv_layers,
        ),
        W(
            "qkv_w",
            shape=lambda d: (d.qkv_sparse_rows, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=_many(*(f"{_ATTN}.{p}_proj.weight" for p in ("q", "k", "v", "index_q", "index_k"))),
            transform=_qkv_sparse_w,
            layers=_five_way_qkv_layers,
        ),
        W(
            "qkv_s",
            shape=lambda d: (d.qkv_sparse_rows * (d.hidden // MX_BLOCK),),
            dtype=torch.uint8,
            src=_many(
                *(
                    f"{_ATTN}.{p}_proj.weight_scale_inv"
                    for p in ("q", "k", "v", "index_q", "index_k")
                )
            ),
            transform=_qkv_sparse_s,
            layers=_five_way_qkv_layers,
        ),
        # The separate BF16 index projection [index-Q | index-K] of the unfused sparse layers. The built-in model
        # builds it as an unquantized Linear and fills it through the fused gate/up loader, which casts the
        # checkpoint's E4M3 values to BF16 and does not apply their block scales: so does this table (the scale
        # tensors are consumed, not applied).
        W(
            "idx_w",
            shape=lambda d: (d.index_heads * d.index_dim + d.index_dim, d.hidden),
            src=_many(
                _ATTN + ".index_q_proj.weight",
                _ATTN + ".index_k_proj.weight",
                _ATTN + ".index_q_proj.weight_scale_inv",
                _ATTN + ".index_k_proj.weight_scale_inv",
            ),
            transform=_index_qk_w,
            layers=_unfused_sparse_layers,
        ),
        W(
            "idx_q_norm",
            shape=lambda d: (d.index_dim,),
            src=_one(_ATTN + ".index_q_norm.weight"),
            layers=_sparse_layers,
        ),
        W(
            "idx_k_norm",
            shape=lambda d: (d.index_dim,),
            src=_one(_ATTN + ".index_k_norm.weight"),
            layers=_sparse_layers,
        ),
        W(
            "o_w",
            shape=lambda d: (d.hidden, d.q_width),
            dtype=torch.float8_e4m3fn,
            src=_one(_ATTN + ".o_proj.weight"),
            transform=_row_w,
        ),
        W(
            "o_s",
            shape=lambda d: (d.hidden * (d.q_width // MX_BLOCK),),
            dtype=torch.uint8,
            src=_one(_ATTN + ".o_proj.weight_scale_inv"),
            transform=_row_s,
        ),
        # --- dense SwiGLU-OAI MLP (layers 0..2) ---
        W(
            "mlp_gu_w",
            shape=lambda d: (2 * d.dense_inter, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=_many(
                "model.layers.{i}.mlp.gate_proj.weight", "model.layers.{i}.mlp.up_proj.weight"
            ),
            transform=_gate_up_w,
            layers=_dense_layers,
        ),
        W(
            "mlp_gu_s",
            shape=lambda d: (2 * d.dense_inter * (d.hidden // MX_BLOCK),),
            dtype=torch.uint8,
            src=_many(
                "model.layers.{i}.mlp.gate_proj.weight_scale_inv",
                "model.layers.{i}.mlp.up_proj.weight_scale_inv",
            ),
            transform=_gate_up_s,
            layers=_dense_layers,
        ),
        W(
            "mlp_dn_w",
            shape=lambda d: (d.hidden, d.dense_inter),
            dtype=torch.float8_e4m3fn,
            src=_one("model.layers.{i}.mlp.down_proj.weight"),
            transform=_row_w,
            layers=_dense_layers,
        ),
        W(
            "mlp_dn_s",
            shape=lambda d: (d.hidden * (d.dense_inter // MX_BLOCK),),
            dtype=torch.uint8,
            src=_one("model.layers.{i}.mlp.down_proj.weight_scale_inv"),
            transform=_row_s,
            layers=_dense_layers,
        ),
        # --- MoE (layers 3..59): the fp32 router and its bias, the shared expert ---
        W(
            "router",
            shape=lambda d: (d.num_experts, d.hidden),
            dtype=torch.float32,
            src=_one(_MOE + ".gate.weight"),
            transform=_f32,
            layers=_sparse_layers,
        ),
        W(
            "router_bias",
            shape=lambda d: (d.num_experts,),
            dtype=torch.float32,
            src=_one(_MOE + ".e_score_correction_bias"),
            transform=_f32,
            layers=_sparse_layers,
        ),
        W(
            "sh_gu_w",
            shape=lambda d: (2 * d.shared_inter, d.hidden),
            dtype=torch.float8_e4m3fn,
            src=_many(
                _MOE + ".shared_experts.gate_proj.weight", _MOE + ".shared_experts.up_proj.weight"
            ),
            transform=_gate_up_w,
            layers=_sparse_layers,
        ),
        W(
            "sh_gu_s",
            shape=lambda d: (2 * d.shared_inter * (d.hidden // MX_BLOCK),),
            dtype=torch.uint8,
            src=_many(
                _MOE + ".shared_experts.gate_proj.weight_scale_inv",
                _MOE + ".shared_experts.up_proj.weight_scale_inv",
            ),
            transform=_gate_up_s,
            layers=_sparse_layers,
        ),
        W(
            "sh_dn_w",
            shape=lambda d: (d.hidden, d.shared_inter),
            dtype=torch.float8_e4m3fn,
            src=_one(_MOE + ".shared_experts.down_proj.weight"),
            transform=_row_w,
            layers=_sparse_layers,
        ),
        W(
            "sh_dn_s",
            shape=lambda d: (d.hidden * (d.shared_inter // MX_BLOCK),),
            dtype=torch.uint8,
            src=_one(_MOE + ".shared_experts.down_proj.weight_scale_inv"),
            transform=_row_s,
            layers=_sparse_layers,
        ),
        # --- the routed experts, NVFP4 (trtllm-gen layouts), all 128 on every rank ---
        W(
            "e_w31",
            shape=lambda d: (d.num_experts, 2 * d.moe_inter, d.hidden // 2),
            dtype=torch.uint8,
            src=_per_expert("w3.weight", "w1.weight"),
            transform=_expert_w31,
            layers=_sparse_layers,
        ),
        W(
            "e_w2",
            shape=lambda d: (d.num_experts, d.hidden, d.moe_inter // 2),
            dtype=torch.uint8,
            src=_per_expert("w2.weight"),
            transform=_expert_w2,
            layers=_sparse_layers,
        ),
        W(
            "e_s31",
            shape=lambda d: (d.num_experts, 2 * d.moe_inter, d.hidden // NV_BLOCK),
            dtype=torch.float8_e4m3fn,
            src=_per_expert("w3.weight_scale", "w1.weight_scale"),
            transform=_expert_s31,
            layers=_sparse_layers,
        ),
        W(
            "e_s2",
            shape=lambda d: (d.num_experts, d.hidden, d.moe_inter // NV_BLOCK),
            dtype=torch.float8_e4m3fn,
            src=_per_expert("w2.weight_scale"),
            transform=_expert_s2,
            layers=_sparse_layers,
        ),
        # The per-tensor scalars of every expert, raw; modeling derives the kernels' scalars from them.
        *(
            W(
                f"e_{name}",
                shape=lambda d: (d.num_experts,),
                dtype=torch.float32,
                src=_per_expert(suffix),
                layers=_sparse_layers,
            )
            for name, suffix in (
                ("isc1", "w1.input_scale"),
                ("isc3", "w3.input_scale"),
                ("isc2", "w2.input_scale"),
                ("ws2_1", "w1.weight_scale_2"),
                ("ws2_3", "w3.weight_scale_2"),
                ("ws2_2", "w2.weight_scale_2"),
            )
        ),
        # --- not per-layer ---
        W(
            "embed",
            shape=lambda d: (d.vocab, d.hidden),
            src=_one("model.embed_tokens.weight"),
            layers=None,
        ),
        W(
            "final_norm",
            shape=lambda d: (d.hidden,),
            src=_one("model.norm.weight"),
            transform=_gemma_fold,
            layers=None,
        ),
    )

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        sparse = cfg.sparse_attention_config
        freq = [int(f) for f in sparse["sparse_attention_freq"]]
        tp = mapping.tp_size
        head_dim = int(cfg.head_dim)
        heads = cfg.num_attention_heads // tp
        kv_heads = max(cfg.num_key_value_heads // tp, 1)
        index_heads = max(int(sparse["sparse_num_index_heads"]) // tp, 1)
        index_dim = int(sparse["sparse_index_dim"])
        return SimpleNamespace(
            fused=bool(
                getattr(
                    core.model_config.sparse_attention_config, "fuse_qkv_index_projection", False
                )
            ),
            num_layers=cfg.num_hidden_layers,
            first_sparse=freq.index(1),
            hidden=cfg.hidden_size,
            head_dim=head_dim,
            heads=heads,
            kv_heads=kv_heads,
            index_heads=index_heads,
            index_dim=index_dim,
            q_width=heads * head_dim,
            qkv_dense_rows=(heads + 2 * kv_heads) * head_dim,
            qkv_sparse_rows=(heads + 2 * kv_heads) * head_dim + index_heads * index_dim + index_dim,
            dense_inter=int(cfg.dense_intermediate_size) // tp,
            shared_inter=int(cfg.shared_intermediate_size) * int(cfg.n_shared_experts) // tp,
            moe_inter=int(cfg.intermediate_size) // tp,
            num_experts=int(cfg.num_local_experts),
            vocab=cfg.vocab_size,
            tp=tp,
            rank=mapping.tp_rank,
            dtype=cfg.torch_dtype,
        )

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None)]
        return [(f"l{i}_{entry.name}", i) for i in layers]

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares (meta under the engine's meta init)."""
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def manifest(self, core) -> dict[str, list[tuple[Any, Any, Callable | None]]]:
        """param key -> [(checkpoint key or key tuple, index into the param | None, transform | None)]."""
        d = self.dims(core)
        rows: dict[str, list[tuple[Any, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i in self._targets(entry, d):
                rows[key] = [(ckpt, index, entry.transform) for ckpt, index in entry.src(d, i)]
        return rows

    def expected_unconsumed(self, core, weights) -> set:
        """The vision tower and its projectors, and the multi-token-prediction head: in the checkpoint, never built
        by this target."""
        return {k for k in weights.keys() if k.startswith(_UNBUILT_PREFIXES)}

    def load(self, model, weights) -> None:
        """Fill every declared weight and the shell's ``lm_head`` shard; assert coverage both ways."""
        core = model.model
        d = self.dims(core)
        manifest = self.manifest(core)
        assert set(manifest) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest) ^ set(core.w.keys()),
        )
        all_keys = set(weights.keys())
        expected = self.expected_unconsumed(core, weights)
        consumed: set = set()
        for param_key, sources in manifest.items():
            for ckpt, index, transform in sources:
                _fill(core.w[param_key], ckpt, index, transform, weights, d, consumed)
            if param_key.endswith(self.RELEASE_AFTER):
                torch.cuda.empty_cache()
        # The shell owns the vocab-parallel head: this rank's rows of the full table.
        _fill(model.lm_head.weight, _LM + "lm_head.weight", None, _col_w, weights, d, consumed)
        shell = {
            n for n, _ in model.named_parameters() if not n.startswith(("model.w.", "draft_model."))
        }
        assert shell == {"lm_head.weight"}, f"unfed shell parameters: {sorted(shell)}"
        torch.cuda.synchronize()
        leftover = all_keys - consumed
        assert leftover == expected, (
            "checkpoint coverage: leftover keys are not exactly the unbuilt families (vision, MTP head); "
            f"unexpected {sorted(leftover - expected)[:8]}, missed {sorted(expected - leftover)[:8]}"
        )


def _materialize(entry) -> torch.Tensor:
    """``[:]`` on a lazy safetensors slice; a 0-dim tensor (the per-tensor scalars) rejects that index."""
    return entry if getattr(entry, "ndim", 1) == 0 else entry[:]


def _fill(param, ckpt, index, transform, weights, d, consumed) -> None:
    keys = ckpt if isinstance(ckpt, tuple) else (ckpt,)
    for key in keys:
        assert key in weights, f"checkpoint key missing: {key}"
    dst = param.data if index is None else param.data[index]
    d.dev = dst.device
    # Host tensors: a transform slices this rank's part on the host and moves only that to the device.
    src_all = tuple(_materialize(weights[key]) for key in keys)
    if transform is not None:
        src = transform(src_all if isinstance(ckpt, tuple) else src_all[0], d)
    else:
        assert len(src_all) == 1, "a multi-key row needs a transform"
        src = src_all[0]
    if src.dim() == 0 and dst.dim() == 0:
        pass
    elif src.numel() == 1 and dst.numel() == 1:
        src = src.reshape(dst.shape)
    src = src.to(dst.device, non_blocking=True)
    assert tuple(dst.shape) == tuple(src.shape), (ckpt, tuple(dst.shape), tuple(src.shape))
    assert src.dtype == dst.dtype, (ckpt, src.dtype, dst.dtype)
    dst.copy_(src, non_blocking=True)
    consumed.update(keys)
    _release(weights, keys)


def _release(weights, keys) -> None:
    """Drop consumed checkpoint tensors from the engine's dict as the load goes (bounded host memory)."""
    if hasattr(weights, "mark_consumed_keys"):
        weights.mark_consumed_keys(keys)
    else:
        for key in keys:
            weights.pop(key, None)


MODEL_WEIGHTS = MiniMaxM3Tp4Weights()


# -------------------------------------------------------------------------------------------- the drafter


def _draft_one(template: str):
    return lambda d, i, t=template: [(t, None)]


def _draft_many(*templates: str):
    return lambda d, i, ts=templates: [(tuple(ts), None)]


def _draft_fc_norms(d):
    return range(d.capture_layers)


def _draft_qkv(t: tuple[torch.Tensor, ...], d: SimpleNamespace) -> torch.Tensor:
    q, k, v = t
    return torch.cat([_col(q, d), _col(k, d), _col(v, d)], dim=0).contiguous()


class MiniMaxM3Tp4DraftWeights:
    """The one-layer BF16 Eagle3 drafter (``speculative_model``), tp4.

    The fc over the three captured target layers and its per-layer fc_norms, the layer's three norms, its QKV
    (input ``[embeds | hidden]``, so twice the hidden width; Q by heads, K/V by KV heads) and o_proj, its gated
    MLP, the final norm and the full-vocabulary head (vocab-parallel). The checkpoint's ``embed_tokens`` is the
    target's table (the drafter reads the target's); it is a predicted non-load.
    """

    WEIGHTS: tuple[W, ...] = (
        W(
            "fc",
            shape=lambda d: (d.hidden, d.capture_layers * d.hidden),
            src=_draft_one("fc.weight"),
            layers=None,
        ),
        W(
            "fc_norm",
            shape=lambda d: (d.hidden,),
            src=lambda d, i: [(f"fc_norm.{i}.weight", None)],
            layers=_draft_fc_norms,
        ),
        W(
            "in_norm",
            shape=lambda d: (d.hidden,),
            src=_draft_one("layers.0.input_layernorm.weight"),
            layers=None,
        ),
        W(
            "hidden_norm",
            shape=lambda d: (d.hidden,),
            src=_draft_one("layers.0.hidden_norm.weight"),
            layers=None,
        ),
        W(
            "post_norm",
            shape=lambda d: (d.hidden,),
            src=_draft_one("layers.0.post_attention_layernorm.weight"),
            layers=None,
        ),
        W(
            "qkv",
            shape=lambda d: (d.qkv_rows, 2 * d.hidden),
            src=_draft_many(*(f"layers.0.self_attn.{p}_proj.weight" for p in "qkv")),
            transform=_draft_qkv,
            layers=None,
        ),
        W(
            "o",
            shape=lambda d: (d.hidden, d.q_width),
            src=_draft_one("layers.0.self_attn.o_proj.weight"),
            transform=_row_w,
            layers=None,
        ),
        W(
            "gu",
            shape=lambda d: (2 * d.inter, d.hidden),
            src=_draft_many("layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"),
            transform=_gate_up_w,
            layers=None,
        ),
        W(
            "dn",
            shape=lambda d: (d.hidden, d.inter),
            src=_draft_one("layers.0.mlp.down_proj.weight"),
            transform=_row_w,
            layers=None,
        ),
        W("final_norm", shape=lambda d: (d.hidden,), src=_draft_one("norm.weight"), layers=None),
    )

    def dims(self, draft) -> SimpleNamespace:
        cfg = draft.draft_config.pretrained_config
        mapping = draft.draft_config.mapping
        tp = mapping.tp_size
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        heads = cfg.num_attention_heads // tp
        kv_heads = max(cfg.num_key_value_heads // tp, 1)
        return SimpleNamespace(
            hidden=cfg.hidden_size,
            capture_layers=draft.capture_layers,
            head_dim=head_dim,
            q_width=heads * head_dim,
            qkv_rows=(heads + 2 * kv_heads) * head_dim,
            inter=cfg.intermediate_size // tp,
            vocab=cfg.vocab_size,
            tp=tp,
            rank=mapping.tp_rank,
            dtype=cfg.torch_dtype,
        )

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None]]:
        if entry.layers is None:
            return [(entry.name, None)]
        return [(f"{entry.name}{i}", i) for i in entry.layers(d)]

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def manifest(self, draft) -> dict[str, list[tuple[Any, Any, Callable | None]]]:
        d = self.dims(draft)
        rows: dict[str, list[tuple[Any, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i in self._targets(entry, d):
                rows[key] = [(ckpt, index, entry.transform) for ckpt, index in entry.src(d, i)]
        return rows

    def expected_unconsumed(self, draft, weights) -> set:
        """The drafter checkpoint's own embedding: the drafter reads the target's table."""
        return {"embed_tokens.weight"} & set(weights.keys())

    def load(self, draft, weights) -> None:
        d = self.dims(draft)
        manifest = self.manifest(draft)
        assert set(manifest) == set(draft.w.keys()), set(manifest) ^ set(draft.w.keys())
        all_keys = set(weights.keys())
        expected = self.expected_unconsumed(draft, weights)
        consumed: set = set()
        for param_key, sources in manifest.items():
            for ckpt, index, transform in sources:
                _fill(draft.w[param_key], ckpt, index, transform, weights, d, consumed)
        _fill(draft.lm_head.weight, "lm_head.weight", None, _col_w, weights, d, consumed)
        torch.cuda.synchronize()
        leftover = all_keys - consumed
        assert leftover == expected, (
            f"drafter checkpoint coverage: unexpected {sorted(leftover - expected)[:8]}, "
            f"missed {sorted(expected - leftover)[:8]}"
        )


DRAFT_WEIGHTS = MiniMaxM3Tp4DraftWeights()
