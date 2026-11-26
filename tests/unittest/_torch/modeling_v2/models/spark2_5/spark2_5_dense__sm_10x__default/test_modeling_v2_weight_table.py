# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weight table declares what the Spark checkpoint contains, on CPU.

`declare` is compared against a transcription of the declaration at the 1.7B
checkpoint's real numbers (shapes/dtypes only, meta-free); `manifest` is
compared against the checkpoint's own key names (``model.embedding``, fused
``q_k_v_proj``, ``out_proj``, ``g_proj``, gate/up pair), so a rename or typo
fails here rather than mid-load.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

import pytest

from tensorrt_llm._torch.configs import Spark2_5Config
from tensorrt_llm._torch.modeling_v2.models.spark2_5.spark2_5_dense__sm_10x__default import (
    weights as W,
)

pytestmark = pytest.mark.cpu_only

# Spark-X2.5/X2.6 1.7B's own numbers, so the comparison is against the shapes
# this target actually ships rather than a toy configuration.
_DIMS = dict(
    num_layers=28,
    hidden=2048,
    head_dim=256,
    heads_q=8,
    heads_kv=2,
    q_width=8 * 256,
    kv_width=2 * 256,
    inter=6656,
    vocab=131072,
    dtype=torch.bfloat16,
)


def _core_stub(**cfg_overrides):
    fields = dict(
        vocab_size=_DIMS["vocab"],
        hidden_size=_DIMS["hidden"],
        intermediate_size=_DIMS["inter"],
        num_hidden_layers=_DIMS["num_layers"],
        num_attention_heads=_DIMS["heads_q"],
        num_key_value_heads=_DIMS["heads_kv"],
        head_dim=_DIMS["head_dim"],
        sliding_window=512,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        * (_DIMS["num_layers"] // 4),
    )
    fields.update(cfg_overrides)
    cfg = Spark2_5Config(**fields)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=cfg,
            torch_dtype=_DIMS["dtype"],
        )
    )


def _reference(d):
    """The declaration this table replaces, transcribed.

    Deliberately a copy rather than an import: the point is to compare the
    table against what the model says, and importing the model would make
    this test agree with itself.
    """
    import torch.nn as nn

    def P(*shape, dtype=d["dtype"]):
        return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

    hidden, q_width, kv_width = d["hidden"], d["q_width"], d["kv_width"]
    heads_q, inter = d["heads_q"], d["inter"]
    w = {}
    for i in range(d["num_layers"]):
        w[f"l{i}_norm1"] = P(hidden)
        w[f"l{i}_qkv"] = P(q_width + 2 * kv_width, hidden)
        w[f"l{i}_g"] = P(heads_q, hidden)
        w[f"l{i}_o"] = P(hidden, q_width)
        w[f"l{i}_norm2"] = P(hidden)
        w[f"l{i}_fc"] = P(2 * inter, hidden)
        w[f"l{i}_down"] = P(hidden, inter)
    w["final_norm"] = P(hidden)
    return w


def test_declare_matches_transcription():
    core = _core_stub()
    d = W.MODEL_WEIGHTS.dims(core)
    for key, expected in _DIMS.items():
        assert getattr(d, key) == expected, (key, getattr(d, key), expected)

    declared = W.MODEL_WEIGHTS.declare(d)
    reference = _reference(_DIMS)
    assert set(declared.keys()) == set(reference.keys()), (
        "parameter drift",
        set(declared.keys()) ^ set(reference.keys()),
    )
    for key, ref in reference.items():
        assert declared[key].shape == ref.shape, (key, declared[key].shape, ref.shape)
        assert declared[key].dtype == ref.dtype, (key, declared[key].dtype, ref.dtype)


def test_manifest_names_the_spark_checkpoint():
    core = _core_stub()
    manifest = W.MODEL_WEIGHTS.manifest(core)

    def sources(param_key):
        return [ckpt for ckpt, _, _ in manifest[param_key]]

    p = "model.layers.0"
    assert sources("l0_qkv") == [f"{p}.self_attn.q_k_v_proj.weight"]
    assert sources("l0_g") == [f"{p}.self_attn.g_proj.weight"]
    assert sources("l0_o") == [f"{p}.self_attn.out_proj.weight"]
    assert sources("l0_norm1") == [f"{p}.input_layernorm.weight"]
    assert sources("l0_norm2") == [f"{p}.post_attention_layernorm.weight"]
    assert sources("l0_down") == [f"{p}.mlp.down_proj.weight"]
    # gate and up feed one packed parameter through one transform, gate first.
    assert sources("l0_fc") == [(f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight")]
    assert manifest["l0_fc"][0][2] is not None, "the packed fc needs its concat transform"
    assert sources("final_norm") == ["model.norm.weight"]

    # Every checkpoint key the table consumes, for the whole model: the load's
    # unconsumed-key audit only closes when this is the checkpoint's full set
    # (plus model.embedding.weight, which load fills into the tied embedding).
    consumed = set()
    for rows in manifest.values():
        for ckpt, _, _ in rows:
            consumed.update(ckpt if isinstance(ckpt, tuple) else (ckpt,))
    last = _DIMS["num_layers"] - 1
    assert f"model.layers.{last}.self_attn.q_k_v_proj.weight" in consumed
    assert not any("embed" in k for k in consumed), "the embedding is filled by load, not the table"
