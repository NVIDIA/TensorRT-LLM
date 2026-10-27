# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The M3.1 sm_107 TP2 target packs the sparse layers' QKV + index projection by default (a measured
win); an explicit ``fuse_qkv_index_projection`` in the deployment's config wins, so False is the kill switch."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    weights as tp_weights,
)
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

_MSA = dict(implementation="msa", indexer_kv_dtype="fp8")


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        ({}, True),
        ({"fuse_qkv_index_projection": True}, True),
        ({"fuse_qkv_index_projection": False}, False),
    ],
)
def test_fused_projection_default_and_kill(explicit: dict, expected: bool) -> None:
    config = MiniMaxM3SparseAttentionConfig(**_MSA, **explicit)
    assert tp_weights.fused_projection(config) is expected


def test_no_sparse_config_is_unfused() -> None:
    assert tp_weights.fused_projection(None) is False


@pytest.mark.parametrize("value", [True, False])
def test_a_config_without_set_fields_is_taken_at_its_word(value: bool) -> None:
    from types import SimpleNamespace

    assert tp_weights.fused_projection(SimpleNamespace(fuse_qkv_index_projection=value)) is value
