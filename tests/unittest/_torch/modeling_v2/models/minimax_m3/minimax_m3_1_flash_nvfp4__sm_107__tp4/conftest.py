# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures of the sm_107 TP4 MiniMax-M3.1 tests."""

import sys

import pytest

_CTX_STORE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_store"
)


@pytest.fixture(autouse=True)
def _restore_ctx_store_fold():
    """The DSpark worker's prebuild sets the catalog ctx store's process-wide k_norm fold (TAILBK_F); restore it after
    every test so a test that ran prebuild does not change what a later test in the same process stores or compares
    against. Nothing is imported here: a test that never loads the store leaves nothing to restore."""
    store = sys.modules.get(_CTX_STORE)
    saved = store._fold_w[0] if store is not None else False
    yield
    store = sys.modules.get(_CTX_STORE)
    if store is not None:
        store._fold_w[0] = saved
