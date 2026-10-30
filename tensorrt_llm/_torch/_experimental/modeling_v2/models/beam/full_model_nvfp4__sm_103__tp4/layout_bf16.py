# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration, schedule and four-rank ownership of the Beam checkpoint layout component.

The layout is one fixed four-rank configuration: attention TP4 and MoE TP4 (EP1) on the
same four ranks, PP1, no attention data parallelism. Every logical checkpoint tensor is
either split along one axis into four contiguous, gap-free, non-overlapping intervals in
global checkpoint coordinates, or replicated identically on every rank::

    embed_tokens, lm_head        [151936, 5120]  vocabulary rows    [37984 r, 37984 (r + 1))
    q_proj, self_attn.gate_proj  [6144, 5120]    query-head rows    [1536 r, 1536 (r + 1))
    k_proj, v_proj               [1024, 5120]    kv-head rows       [256 r, 256 (r + 1))
    o_proj                       [5120, 6144]    query-head columns [1536 r, 1536 (r + 1))
    gate_bias                    [48]            query heads        [12 r, 12 (r + 1))
    dense gate_proj, up_proj     [20480, 5120]   intermediate rows  [5120 r, 5120 (r + 1))
    dense down_proj              [5120, 20480]   intermediate cols  [5120 r, 5120 (r + 1))
    expert gate_proj, up_proj    [2560, 5120]    intermediate rows  [640 r, 640 (r + 1))
    expert down_proj             [5120, 2560]    intermediate cols  [640 r, 640 (r + 1))
    norms, q_norm, k_norm        replicated BF16
    router weight, correction    replicated FP32; expert ids stay global

Rank r therefore owns query heads [12 r, 12 r + 12) with their two kv heads [2 r, 2 r + 2),
keeping the 6:1 grouped-query ratio, and an intermediate slice of every expert. A row-split
FP8 matrix takes the matching rows of its per-output-channel scale; a column-split one keeps
all of its output scales. Each sparse layer stores its experts' slices in three contiguous
banks, ``gate`` and ``up`` ``[256, 640, 5120]`` and ``down`` ``[256, 5120, 640]``; expert e
is slot e of each bank.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any, Mapping, Optional

WORLD_SIZE = 4

#: The configuration this component computes, as the checkpoint's config.json states it.
SUPPORTED_CONFIG = {
    "model_type": "beam",
    "hidden_size": 5120,
    "intermediate_size": 20480,
    "moe_intermediate_size": 2560,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "norm_topk_prob": True,
    "vocab_size": 151936,
    "num_hidden_layers": 52,
    "rms_norm_eps": 1e-06,
    "routed_scaling_factor": 2.5,
    "final_logit_softcapping": 30.0,
    "sliding_window": 512,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "attention_bias": False,
}
SLIDING_ROPE_THETA = 10000.0

#: Stored dtype names, as safetensors headers spell them.
FP8 = "F8_E4M3"
BF16 = "BF16"
F32 = "F32"

ATTENTION_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
EMBED = "model.embed_tokens.weight"
FINAL_NORM = "model.norm.weight"
HEAD = "lm_head.weight"


@dataclasses.dataclass(frozen=True)
class Schedule:
    """The per-layer attention and feed-forward kinds the checkpoint's configuration states."""

    attention: tuple  # "sliding_attention" | "full_attention", one per logical layer
    mlp: tuple  # "dense" | "sparse", one per logical layer

    @property
    def num_layers(self) -> int:
        return len(self.attention)

    def is_sparse(self, layer: int) -> bool:
        return self.mlp[layer] == "sparse"

    def is_sliding(self, layer: int) -> bool:
        return self.attention[layer] == "sliding_attention"


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """One logical checkpoint tensor: global geometry, stored dtype and four-rank rule."""

    name: str
    shape: tuple
    stored_dtype: str
    family: str
    layer: Optional[int]
    expert: Optional[int]
    split_dim: Optional[int]  # None: replicated on every rank

    @property
    def scale_name(self) -> Optional[str]:
        """The consumed per-output-channel scale of an FP8 matrix, ``[out, 1]`` BF16."""
        return self.name + "_scale" if self.stored_dtype == FP8 else None

    def interval(self, rank: int, world_size: int = WORLD_SIZE) -> Optional[tuple]:
        """``(dim, start, stop)`` of rank ``rank``'s slice, or None for a replicated tensor."""
        check_rank(rank, world_size)
        if self.split_dim is None:
            return None
        extent = self.shape[self.split_dim]
        if extent % world_size:
            raise ValueError(
                f"{self.name}: dim {self.split_dim} of {extent} does not split by {world_size}"
            )
        size = extent // world_size
        return self.split_dim, rank * size, (rank + 1) * size

    def local_shape(self, rank: int, world_size: int = WORLD_SIZE) -> tuple:
        interval = self.interval(rank, world_size)
        if interval is None:
            return tuple(self.shape)
        dim, start, stop = interval
        shape = list(self.shape)
        shape[dim] = stop - start
        return tuple(shape)

    @property
    def effective_dtype(self) -> str:
        """BF16 for FP8 matrices (decoded and scaled) and BF16 tensors; F32 router tensors."""
        return BF16 if self.stored_dtype in (FP8, BF16) else F32

    @property
    def bank(self) -> Optional[str]:
        """The sparse layer's bank holding this expert slice, or None for a parameter."""
        if self.expert is None:
            return None
        return f"layers.{self.layer}.experts.{self.family.removeprefix('expert_')}"


def check_rank(rank: int, world_size: int = WORLD_SIZE) -> None:
    """Raise ValueError on any rank/world other than one of the four ranks of this layout."""
    if type(world_size) is not int or world_size != WORLD_SIZE:
        raise ValueError(
            f"this layout runs on exactly {WORLD_SIZE} ranks, got world_size={world_size!r}"
        )
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError(f"rank must be an int in [0, {world_size}), got {rank!r}")


def _get(config: Any, field: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def check_config(config: Any) -> Schedule:
    """Raise ValueError on the first field outside what this layout computes; return the schedule.

    ``config`` is the checkpoint's config.json as a mapping (or an object with those
    attributes). The schedule comes from its ``layer_types``/``mlp_layer_types``: every layer
    is sliding or full attention and dense or sparse, the dense layers precede the sparse
    ones, and only sliding layers use RoPE (``rope_parameters``).
    """
    for field, expected in SUPPORTED_CONFIG.items():
        actual = _get(config, field)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported {field}={actual!r}; this layout computes {expected!r}")
    layers = SUPPORTED_CONFIG["num_hidden_layers"]
    attention = tuple(_get(config, "layer_types") or ())
    mlp = tuple(_get(config, "mlp_layer_types") or ())
    if len(attention) != layers or set(attention) - {"sliding_attention", "full_attention"}:
        raise ValueError(f"layer_types must name {layers} sliding/full layers, got {attention!r}")
    if len(mlp) != layers or set(mlp) - {"dense", "sparse"}:
        raise ValueError(f"mlp_layer_types must name {layers} dense/sparse layers, got {mlp!r}")
    if (
        "dense" in mlp
        and "sparse" in mlp
        and mlp.index("sparse") < len(mlp) - mlp[::-1].index("dense")
    ):
        raise ValueError("dense feed-forward layers must precede the sparse ones")
    rope = _get(config, "rope_parameters")
    sliding = (rope or {}).get("sliding_attention") if isinstance(rope, Mapping) else None
    if (
        not isinstance(rope, Mapping)
        or rope.get("full_attention") is not None
        or not isinstance(sliding, Mapping)
        or sliding.get("rope_type") != "default"
        or sliding.get("rope_theta") != SLIDING_ROPE_THETA
    ):
        raise ValueError(
            "rope_parameters must give sliding layers default RoPE with theta "
            f"{SLIDING_ROPE_THETA} and full layers none, got {rope!r}"
        )
    return Schedule(attention=attention, mlp=mlp)


def tensor_specs(config: Any, layers: Optional[tuple] = None) -> "OrderedDict[str, TensorSpec]":
    """Every logical tensor of the selected layers and the outer tensors, by checkpoint name.

    ``layers`` selects logical layers (default: all of them); the embedding, final norm and
    head are always included. Consumed FP8 scales are not logical tensors; see
    ``TensorSpec.scale_name``.
    """
    schedule = check_config(config)
    hidden = _get(config, "hidden_size")
    heads = _get(config, "num_attention_heads")
    kv_heads = _get(config, "num_key_value_heads")
    head_dim = _get(config, "head_dim")
    vocab = _get(config, "vocab_size")
    dense_inter = _get(config, "intermediate_size")
    expert_inter = _get(config, "moe_intermediate_size")
    experts = _get(config, "num_experts")
    selected = tuple(range(schedule.num_layers)) if layers is None else tuple(layers)
    for layer in selected:
        if type(layer) is not int or not 0 <= layer < schedule.num_layers:
            raise ValueError(f"layer {layer!r} is not a logical layer of the checkpoint")
    if len(set(selected)) != len(selected):
        raise ValueError(f"layers must be distinct, got {selected}")

    specs: "OrderedDict[str, TensorSpec]" = OrderedDict()

    def add(name, shape, dtype, family, layer=None, expert=None, split_dim=None):
        specs[name] = TensorSpec(name, tuple(shape), dtype, family, layer, expert, split_dim)

    add(EMBED, (vocab, hidden), BF16, "embed_tokens", split_dim=0)
    for layer in sorted(selected):
        p = f"model.layers.{layer}."
        for norm in ATTENTION_NORMS:
            add(p + norm + ".weight", (hidden,), BF16, "block_norm", layer)
        add(
            p + "self_attn.q_proj.weight",
            (heads * head_dim, hidden),
            FP8,
            "q_proj",
            layer,
            split_dim=0,
        )
        add(
            p + "self_attn.k_proj.weight",
            (kv_heads * head_dim, hidden),
            FP8,
            "k_proj",
            layer,
            split_dim=0,
        )
        add(
            p + "self_attn.v_proj.weight",
            (kv_heads * head_dim, hidden),
            FP8,
            "v_proj",
            layer,
            split_dim=0,
        )
        add(
            p + "self_attn.gate_proj.weight",
            (heads * head_dim, hidden),
            FP8,
            "attn_gate",
            layer,
            split_dim=0,
        )
        add(
            p + "self_attn.o_proj.weight",
            (hidden, heads * head_dim),
            FP8,
            "o_proj",
            layer,
            split_dim=1,
        )
        add(p + "self_attn.q_norm.weight", (head_dim,), BF16, "qk_norm", layer)
        add(p + "self_attn.k_norm.weight", (head_dim,), BF16, "qk_norm", layer)
        add(p + "self_attn.gate_bias", (heads,), BF16, "gate_bias", layer, split_dim=0)
        if schedule.is_sparse(layer):
            add(p + "mlp.gate.weight", (experts, hidden), F32, "router_weight", layer)
            add(p + "mlp.gate.e_score_correction_bias", (experts,), F32, "router_bias", layer)
            for e in range(experts):
                q = f"{p}mlp.experts.{e}."
                add(q + "gate_proj.weight", (expert_inter, hidden), FP8, "expert_gate", layer, e, 0)
                add(q + "up_proj.weight", (expert_inter, hidden), FP8, "expert_up", layer, e, 0)
                add(q + "down_proj.weight", (hidden, expert_inter), FP8, "expert_down", layer, e, 1)
        else:
            add(
                p + "mlp.gate_proj.weight",
                (dense_inter, hidden),
                FP8,
                "dense_gate",
                layer,
                split_dim=0,
            )
            add(
                p + "mlp.up_proj.weight", (dense_inter, hidden), FP8, "dense_up", layer, split_dim=0
            )
            add(
                p + "mlp.down_proj.weight",
                (hidden, dense_inter),
                FP8,
                "dense_down",
                layer,
                split_dim=1,
            )
    add(FINAL_NORM, (hidden,), BF16, "final_norm")
    add(HEAD, (vocab, hidden), BF16, "lm_head", split_dim=0)
    return specs


def ownership_ledger(specs: Mapping[str, TensorSpec], world_size: int = WORLD_SIZE) -> list:
    """One row per logical tensor: its rule and every rank's interval, replica flag and storage.

    Rows are plain data (the same fields an independent ledger records), so two ledgers can
    be compared without sharing code: ``dim``/``start``/``stop`` per split rank, ``replicated``
    otherwise, and ``bank``/``slot`` for expert slices.
    """
    rows = []
    for spec in specs.values():
        ranks = []
        for rank in range(world_size):
            interval = spec.interval(rank, world_size)
            entry = {"rank": rank, "local_shape": list(spec.local_shape(rank, world_size))}
            if interval is None:
                entry["replicated"] = True
            else:
                dim, start, stop = interval
                entry.update(replicated=False, dim=dim, start=start, stop=stop)
            if spec.bank is not None:
                entry.update(bank=spec.bank, slot=spec.expert)
            ranks.append(entry)
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "layer": spec.layer,
                "expert": spec.expert,
                "shape": list(spec.shape),
                "stored_dtype": spec.stored_dtype,
                "effective_dtype": spec.effective_dtype,
                "scale": spec.scale_name,
                "ranks": ranks,
            }
        )
    return rows


def check_partition(specs: Mapping[str, TensorSpec], world_size: int = WORLD_SIZE) -> None:
    """Raise unless every split tensor's rank intervals tile its axis exactly, in rank order."""
    for spec in specs.values():
        if spec.split_dim is None:
            continue
        cursor = 0
        for rank in range(world_size):
            dim, start, stop = spec.interval(rank, world_size)
            if start != cursor or stop <= start:
                raise ValueError(
                    f"{spec.name}: rank {rank} interval [{start}, {stop}) leaves a gap or overlap"
                )
            cursor = stop
        if cursor != spec.shape[spec.split_dim]:
            raise ValueError(
                f"{spec.name}: rank intervals cover {cursor} of {spec.shape[spec.split_dim]}"
            )
