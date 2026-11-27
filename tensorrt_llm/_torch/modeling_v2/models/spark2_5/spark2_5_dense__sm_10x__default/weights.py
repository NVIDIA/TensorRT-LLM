# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: spark2_5 dense / sm_10x / default (one rank).

Single-rank only, so every declared shape is the whole checkpoint tensor and
the only transform is the MLP's gate/up concatenation (gate rows first,
matching `gelu_erf_and_mul`).

Spark checkpoint naming: the embedding is ``model.embedding.weight`` (tied
with lm_head, which the checkpoint does not store); QKV ships pre-fused as
``self_attn.q_k_v_proj.weight`` in q|k|v row order -- the packed layout the
attention op reads, so it copies whole; the output projection is
``self_attn.out_proj``; the headwise gate is ``self_attn.g_proj``,
[num_heads, hidden].
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


@dataclass(frozen=True)
class W:
    """One weight role: declared `shape(d)`, checkpoint `src` (a `{p}`-template
    key, or a callable returning [(key-or-key-tuple, index)]), per-layer or
    not (`layers`), and an optional load-time `transform`."""

    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

    def layer_indices(self, d: SimpleNamespace) -> list[int] | None:
        if self.layers is None:
            return None
        return list(range(d.num_layers))

    def sources(self, d: SimpleNamespace, i: int | None, prefix: str) -> list[tuple[Any, Any]]:
        if callable(self.src):
            return list(self.src(d, i))
        return [(self.src.format(p=prefix), None)]


def _gate_up(t: tuple[torch.Tensor, torch.Tensor], core) -> torch.Tensor:
    """(gate_proj, up_proj) -> one packed [2I, hidden] operand, gate rows first."""
    gate, up = t
    return torch.cat([gate, up], dim=0).contiguous()


def _gate_up_src():
    def src(d, i):
        p = f"model.layers.{i}.mlp"
        return [((f"{p}.gate_proj.weight", f"{p}.up_proj.weight"), None)]

    return src


class Spark2_5DenseWeights:
    """The table and its passes: `dims` reads widths off the config, `declare`
    allocates, `manifest` maps checkpoint keys onto parameters, `load` fills
    them plus the tied embedding. `declare` and `manifest` share `_targets`,
    so they cannot disagree about which weights exist."""

    WEIGHTS: tuple[W, ...] = (
        W("norm1", shape=lambda d: (d.hidden,), src="{p}.input_layernorm.weight"),
        W(
            "qkv",
            shape=lambda d: (d.q_width + 2 * d.kv_width, d.hidden),
            src="{p}.self_attn.q_k_v_proj.weight",
        ),
        W("g", shape=lambda d: (d.heads_q, d.hidden), src="{p}.self_attn.g_proj.weight"),
        W("o", shape=lambda d: (d.hidden, d.q_width), src="{p}.self_attn.out_proj.weight"),
        W("norm2", shape=lambda d: (d.hidden,), src="{p}.post_attention_layernorm.weight"),
        W(
            "fc",
            shape=lambda d: (2 * d.inter, d.hidden),
            src=_gate_up_src(),
            transform=_gate_up,
        ),
        W("down", shape=lambda d: (d.hidden, d.inter), src="{p}.mlp.down_proj.weight"),
        W("final_norm", shape=lambda d: (d.hidden,), src="model.norm.weight", layers=None),
    )

    def dims(self, core) -> SimpleNamespace:
        """Every width, from the config. The target is single-rank: every
        share is the whole tensor."""
        cfg = core.model_config.pretrained_config
        return SimpleNamespace(
            num_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            head_dim=cfg.head_dim,
            heads_q=cfg.num_attention_heads,
            heads_kv=cfg.num_key_value_heads,
            q_width=cfg.num_attention_heads * cfg.head_dim,
            kv_width=cfg.num_key_value_heads * cfg.head_dim,
            inter=cfg.intermediate_size,
            vocab=cfg.vocab_size,
            dtype=core.model_config.torch_dtype,
        )

    def declare(self, d: SimpleNamespace) -> nn.ParameterDict:
        """Allocate every weight the table declares.

        Meta-init intercepts `torch.empty` here -- real CUDA storage arrives
        when the engine materializes the registry.
        """
        w = nn.ParameterDict()
        for entry in self.WEIGHTS:
            dtype = entry.dtype or d.dtype
            for key, _, _ in self._targets(entry, d):
                w[key] = nn.Parameter(
                    torch.empty(*entry.shape(d), dtype=dtype), requires_grad=False
                )
        return w

    def _targets(self, entry: W, d: SimpleNamespace) -> list[tuple[str, int | None, str]]:
        """`[(param key, layer index | None, checkpoint layer prefix), ...]`."""
        layers = entry.layer_indices(d)
        if layers is None:
            return [(entry.name, None, "")]
        return [(f"l{i}_{entry.name}", i, f"model.layers.{i}") for i in layers]

    def manifest(self, core) -> dict[str, list[tuple[Any, Any, Callable | None]]]:
        """param key -> [(checkpoint key or key tuple, index | None, transform | None)]."""
        d = core.dims
        rows: dict[str, list[tuple[Any, Any, Callable | None]]] = {}
        for entry in self.WEIGHTS:
            for key, i, prefix in self._targets(entry, d):
                rows[key] = [
                    (ckpt, index, entry.transform) for ckpt, index in entry.sources(d, i, prefix)
                ]
        return rows

    def load(self, model, weights) -> None:
        """Fill every declared weight from the checkpoint, plus the tied
        embedding. lm_head needs no fill: the shell ties it to the embedding
        weight, and the checkpoint stores no lm_head tensor."""
        core = model.model
        manifest = self.manifest(core)
        consumed: set[str] = set()

        def fill(param: nn.Parameter, ckpt_key, index, transform) -> None:
            keys = ckpt_key if isinstance(ckpt_key, tuple) else (ckpt_key,)
            for key in keys:
                assert key in weights, f"checkpoint key missing: {key}"
            dst = param.data if index is None else param.data[index]
            # [:] materializes the lazy slice.
            src_all = tuple(weights[key][:].to(dst.device, non_blocking=True) for key in keys)
            if transform is not None:
                src = transform(src_all if isinstance(ckpt_key, tuple) else src_all[0], core)
            else:
                assert len(src_all) == 1, "a multi-key row needs a transform"
                src = src_all[0]
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src, non_blocking=True)
            consumed.update(keys)

        assert set(manifest.keys()) == set(core.w.keys()), (
            "manifest/parameter drift",
            set(manifest.keys()) ^ set(core.w.keys()),
        )
        for param_key, sources in manifest.items():
            for ckpt_key, index, transform in sources:
                fill(core.w[param_key], ckpt_key, index, transform)

        # The embedding is declared by the core (the shell ties lm_head to it),
        # not by the table; fill it here so the consumed-key audit covers it.
        fill(core.embed_tokens.weight, "model.embedding.weight", None, None)

        torch.cuda.synchronize()
        # Every key in this checkpoint is one the table names; nothing is meant
        # to be left untouched.
        leftover = set(weights.keys()) - consumed
        assert not leftover, f"unconsumed checkpoint keys: {sorted(leftover)[:8]}"


MODEL_WEIGHTS = Spark2_5DenseWeights()
