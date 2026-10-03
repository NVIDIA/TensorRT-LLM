# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weight table reproduces the declaration it replaces.

One weight's shape, dtype, checkpoint source and load-time transform used to
live in three files. The table merges them, which is only safe if it declares
exactly what the hand-written block declared -- same keys, same shapes, same
dtypes. This test is that proof, and it runs while both forms still exist so
the switchover has something to stand on.

No GPU and no built extensions: the parameters are meta tensors, so only
shape and dtype are compared, which is all the declaration ever stated.
"""

from __future__ import annotations

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
    fc1_k_pad=2944,
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


def test_the_table_declares_the_same_keys():
    assert set(W.declare(**_DIMS).keys()) == set(_reference(_DIMS).keys())


def test_the_table_declares_the_same_shapes_and_dtypes():
    """Shape and dtype are the whole content of a declaration.

    Checked per key rather than as one aggregate so a failure names the weight
    that drifted instead of only saying the two disagree.
    """
    got, want = W.declare(**_DIMS), _reference(_DIMS)
    for key in sorted(want):
        assert tuple(got[key].shape) == tuple(want[key].shape), key
        assert got[key].dtype == want[key].dtype, key


def test_every_table_entry_states_a_source():
    """A declared weight with no checkpoint source would allocate memory that
    nothing ever fills -- silently zero rather than an error, which is the
    failure mode `load`'s coverage assert exists to catch from the other side.
    """
    for entry in W.WEIGHTS:
        assert entry.src, entry.name
