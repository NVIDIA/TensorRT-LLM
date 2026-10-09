# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    _padding_dummy_draft_token_capacity,
    cuda_graph_dummy_request_id,
)


@pytest.mark.parametrize("mode", ["Eagle", "MTP"])
def test_tree_speculation_keeps_native_dummy_identity(mode):
    config = SimpleNamespace(decoding_type=mode, max_draft_len=3)
    capacity = _padding_dummy_draft_token_capacity(config, 12)
    assert capacity == 12
    assert cuda_graph_dummy_request_id(12, variant=0, max_draft_len=capacity) == (
        CUDA_GRAPH_DUMMY_REQUEST_ID - 12
    )


def test_dspark_does_not_relax_the_physical_k_bound():
    config = SimpleNamespace(decoding_type="DSpark", max_draft_len=3)
    capacity = _padding_dummy_draft_token_capacity(config, 12)
    assert capacity == 3
    with pytest.raises(ValueError, match="invalid CUDA padding dummy identity"):
        cuda_graph_dummy_request_id(12, variant=0, max_draft_len=capacity)


def test_secondary_dspark_dummy_ids_are_disjoint():
    native = {cuda_graph_dummy_request_id(k, variant=0, max_draft_len=3) for k in range(4)}
    secondary = {cuda_graph_dummy_request_id(k, variant=1, max_draft_len=3) for k in range(4)}
    assert native.isdisjoint(secondary)


def test_target_only_dummy_identity_is_unchanged():
    capacity = _padding_dummy_draft_token_capacity(None, 0)
    assert cuda_graph_dummy_request_id(0, variant=0, max_draft_len=capacity) == (
        CUDA_GRAPH_DUMMY_REQUEST_ID
    )
