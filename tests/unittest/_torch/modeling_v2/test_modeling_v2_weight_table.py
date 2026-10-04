# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weight table reproduces the declaration it replaces.

One weight's shape, dtype, checkpoint source and load-time transform used to
live in three files. The table merges them, which is only safe if it declares
exactly what the hand-written block declared -- same keys, same shapes, same
dtypes. This test is that proof.

The block it compares against no longer exists in the model; `_reference`
below is a transcription of it, kept deliberately. That is the point of
having transcribed rather than imported it: the table is still held to what
the model used to say, and an import would now have the test agreeing with
itself. A shape that changes here is a real change to what this target
allocates, and should be made knowingly rather than discovered.

No GPU and no built extensions: the parameters are meta tensors, so only
shape and dtype are compared, which is all the declaration ever stated.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

__extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.gpt_oss_120b__sm_103__tp1 import (
    weights as W,
)

# gpt-oss-120b's own numbers, so the comparison is against the shapes this
# target actually ships rather than a toy configuration.
_DIMS = dict(
    num_layers=36,
    hidden=2880,
    q_width=64 * 64,
    kv_width=8 * 64,
    heads_q=64,
    num_experts=128,
    vocab=201088,
    fc1_rows=2 * 2944,
    # 2880 padded up to FC1_K_ALIGN (512), not to 128 like the other three.
    # This read 2944 until `dims` was compared against it: the declaration
    # tests feed both sides the same bundle, so a wrong number here agreed
    # with itself and proved the table at a shape the model never allocates.
    fc1_k_pad=3072,
    inter_pad=2944,
    fc2_rows_pad=2944,
    dtype=torch.bfloat16,
)


def _reference(d):
    """The declaration block from modeling.py, transcribed.

    Deliberately a copy rather than an import: the point is to compare the
    table against what the model *said*, and importing the model would make
    this test agree with itself once the block is deleted in Task 3.
    """
    import torch.nn as nn

    def P(*shape, dtype=d["dtype"]):
        return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

    fc1_rows, hidden, q_width = d["fc1_rows"], d["hidden"], d["q_width"]
    kv_width, heads_q, ne = d["kv_width"], d["heads_q"], d["num_experts"]
    w = {}
    for i in range(d["num_layers"]):
        w[f"l{i}_norm1"] = P(hidden)
        w[f"l{i}_qkv"] = P(q_width + 2 * kv_width, hidden)
        w[f"l{i}_qkv_bias"] = P(q_width + 2 * kv_width)
        w[f"l{i}_sinks"] = P(heads_q, dtype=torch.float32)
        w[f"l{i}_o"] = P(hidden, q_width)
        w[f"l{i}_o_bias"] = P(hidden)
        w[f"l{i}_norm2"] = P(hidden)
        w[f"l{i}_router"] = P(ne, hidden)
        w[f"l{i}_router_bias"] = P(ne)
        w[f"l{i}_fc1_w"] = P(ne, fc1_rows, d["fc1_k_pad"] // 2, dtype=torch.uint8)
        w[f"l{i}_fc1_s"] = P(ne, fc1_rows, d["fc1_k_pad"] // 32, dtype=torch.uint8)
        w[f"l{i}_fc1_b"] = P(ne, fc1_rows, dtype=torch.float32)
        w[f"l{i}_fc2_w"] = P(ne, d["fc2_rows_pad"], d["inter_pad"] // 2, dtype=torch.uint8)
        w[f"l{i}_fc2_s"] = P(ne, d["fc2_rows_pad"], d["inter_pad"] // 32, dtype=torch.uint8)
        w[f"l{i}_fc2_b"] = P(ne, d["fc2_rows_pad"], dtype=torch.float32)
    w["final_norm"] = P(hidden)
    w["embed"] = P(d["vocab"], hidden)
    return w


def _fake_core():
    """The configuration surface `dims` reads, carrying this target's numbers.

    `dims` reads `model_config` only, never the half-built core around it,
    which is what lets this run without constructing a model.
    """
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        num_hidden_layers=_DIMS["num_layers"],
        hidden_size=_DIMS["hidden"],
        num_attention_heads=_DIMS["heads_q"],
        num_key_value_heads=_DIMS["kv_width"] // 64,
        head_dim=64,
        num_local_experts=_DIMS["num_experts"],
        intermediate_size=2880,
        vocab_size=_DIMS["vocab"],
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=cfg, torch_dtype=_DIMS["dtype"])
    )


def test_the_table_declares_the_same_keys():
    assert set(W.MODEL_WEIGHTS.declare(SimpleNamespace(**_DIMS)).keys()) == set(
        _reference(_DIMS).keys()
    )


def test_the_table_declares_the_same_shapes_and_dtypes():
    """Shape and dtype are the whole content of a declaration.

    Checked per key rather than as one aggregate so a failure names the weight
    that drifted instead of only saying the two disagree.
    """
    got, want = W.MODEL_WEIGHTS.declare(SimpleNamespace(**_DIMS)), _reference(_DIMS)
    for key in sorted(want):
        assert tuple(got[key].shape) == tuple(want[key].shape), key
        assert got[key].dtype == want[key].dtype, key


def test_every_table_entry_states_a_source():
    """A declared weight with no checkpoint source would allocate memory that
    nothing ever fills -- silently zero rather than an error, which is the
    failure mode `load`'s coverage assert exists to catch from the other side.
    """
    for entry in W.MODEL_WEIGHTS.WEIGHTS:
        assert entry.src, entry.name


def test_dims_reproduces_the_declaration_numbers():
    """`dims` is now the only place a derived width or padded size is worked
    out; it has to land on the same numbers the declaration was written for.
    """
    d = W.MODEL_WEIGHTS.dims(_fake_core())
    for key, want in _DIMS.items():
        assert getattr(d, key) == want, key


def test_the_table_builds_the_hand_written_manifest():
    """The generated manifest against the hand-written one it replaces.

    `weights._manifest` is dead code kept for exactly this comparison: the
    checkpoint key, destination index and transform of all seventeen roles
    used to be written out a second time there, and the point of the table is
    that they are not. Compared against the real function rather than a
    transcription, which is only possible while both still exist -- once this
    is green, `_manifest` goes.
    """
    core = _fake_core()
    got, want = W.MODEL_WEIGHTS.manifest(core), W._manifest(core)
    assert set(got) == set(want)
    for key in sorted(want):
        assert got[key] == want[key], key
