# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha import registry
from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention_backend.fmha.prims_ts_block_sparse import PrimsTSBlockSparseFmha
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.block_sparse import (
    BlockSparseForwardInputs,
    BlockSparseParams,
    BlockSparseRoutes,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata

EXPECTED_DEFAULT_LIBS = (
    "msa_sparse_gqa",
    "prims_ts_block_sparse",
    "prims_ts",
    "cute_dsl_mla",
    "flashinfer_trtllm_gen",
    "fallback",
)


def _enabled_names() -> tuple[str, ...]:
    classes = registry.get_enabled_fmha_lib_classes()
    names_by_class = {cls: name for name, cls in registry.FMHA_LIBS.items()}
    return tuple(names_by_class[cls] for cls in classes)


class _BlockSparseAttentionOwner:
    def __init__(self, sparse_params: object) -> None:
        self.sparse_params = sparse_params


def test_default_fmha_lib_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)

    assert _enabled_names() == EXPECTED_DEFAULT_LIBS


def test_generic_block_sparse_fmha_is_registered_and_cannot_fall_back_dense() -> None:
    assert registry.FMHA_LIBS["prims_ts_block_sparse"] is PrimsTSBlockSparseFmha

    owner = _BlockSparseAttentionOwner(BlockSparseParams(q_block_size=64, kv_block_size=64))
    fmha = FallbackFmha(owner)
    q = torch.empty(0)

    assert not fmha.is_supported(q, q, q, object(), AttentionForwardArgs())
    routes = BlockSparseRoutes(
        block_indptr=torch.empty((1, 1, 1), dtype=torch.int32),
        block_indices=torch.empty(0, dtype=torch.int32),
        max_blocks_per_row=0,
    )
    permissive_owner = _BlockSparseAttentionOwner(None)
    permissive_fmha = FallbackFmha(permissive_owner)
    assert not permissive_fmha.is_supported(
        q,
        q,
        q,
        object(),
        AttentionForwardArgs(block_sparse_inputs=BlockSparseForwardInputs(routes=routes)),
    )


def test_trtllm_attention_rejects_live_block_sparse_inputs_without_static_params() -> None:
    attention = TrtllmAttention.__new__(TrtllmAttention)
    attention.sparse_params = None
    metadata = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
    routes = BlockSparseRoutes(
        block_indptr=torch.empty((1, 1, 1), dtype=torch.int32),
        block_indices=torch.empty(0, dtype=torch.int32),
        max_blocks_per_row=0,
    )

    with pytest.raises(ValueError, match="must be provided together"):
        attention.forward(
            torch.empty((0, 0)),
            None,
            None,
            metadata,
            AttentionForwardArgs(
                block_sparse_inputs=BlockSparseForwardInputs(routes=routes),
            ),
        )


@pytest.mark.parametrize("value", ["   ", ", ,"])
def test_empty_fmha_lib_env_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", value)

    assert _enabled_names() == EXPECTED_DEFAULT_LIBS


def test_exact_fmha_lib_env_preserves_order_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TLLM_FMHA_LIBS",
        " fallback, prims_ts, fallback, flashinfer_trtllm_gen ",
    )

    assert _enabled_names() == (
        "fallback",
        "prims_ts",
        "flashinfer_trtllm_gen",
    )


def test_delta_fmha_lib_env_applies_entries_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "TLLM_FMHA_LIBS",
        "-prims_ts, -fallback, +fallback, +prims_ts",
    )

    assert _enabled_names() == (
        "msa_sparse_gqa",
        "prims_ts_block_sparse",
        "cute_dsl_mla",
        "flashinfer_trtllm_gen",
        "fallback",
        "prims_ts",
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("prims_ts,-fallback", "either an exact comma-separated list"),
        ("prims_ts,unknown", "Unknown FMHA library 'unknown'"),
        ("+", "Invalid empty FMHA library entry"),
    ],
)
def test_invalid_fmha_lib_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", value)

    with pytest.raises(ValueError, match=message):
        registry.get_enabled_fmha_lib_classes()
