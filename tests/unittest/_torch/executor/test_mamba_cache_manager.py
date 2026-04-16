# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MambaCacheManager padding-slot behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    CppMambaCacheManager,
    PythonMambaCacheManager,
)
from tensorrt_llm.mapping import Mapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_mgr(max_batch_size=4, max_draft_len=2):
    return PythonMambaCacheManager(
        d_state=8,
        d_conv=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        num_layers=2,
        max_batch_size=max_batch_size,
        spec_state_size=max_batch_size,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
        speculative_num_draft_tokens=max_draft_len,
    )


@skip_no_cuda
def test_padding_never_borrows_other_live_slot():
    """get_state_indices must raise instead of picking a slot owned by
    a live request outside the current batch."""
    mgr = _make_mgr()
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])  # pool full
    dummy = CUDA_GRAPH_DUMMY_REQUEST_ID - 2
    with pytest.raises(RuntimeError, match="Run out of free Mamba slots"):
        mgr.get_state_indices([100, 101, dummy], [False, False, True])


@skip_no_cuda
def test_update_mamba_states_skips_padding_slots():
    """update_mamba_states must not write into slots owned by live
    requests outside the current batch (via stale state_indices)."""
    mgr = _make_mgr()
    # All four slots live; slot 0 = request 103 (not in current batch).
    mgr._prepare_mamba_cache_blocks([100, 101, 102, 103])
    ssm, conv = mgr.mamba_cache.temporal, mgr.mamba_cache.conv
    ssm[:, 0].fill_(42.0)
    conv[:, 0].fill_(42.0)

    # Prepare for 3-real batch, then force state_indices[3] to alias
    # slot 0 (= R103's). The buggy slice would overwrite it.
    mgr._prepare_mamba_cache_blocks([100, 101, 102])
    mgr.state_indices[3] = 0
    mgr.mamba_cache.intermediate_ssm.fill_(7.0)
    mgr.mamba_cache.intermediate_conv_window.fill_(7.0)

    dummy = CUDA_GRAPH_DUMMY_REQUEST_ID - 2
    attn = SimpleNamespace(num_seqs=4, num_contexts=0, request_ids=[100, 101, 102, dummy])
    mgr.update_mamba_states(attn, torch.tensor([1, 1, 1, 1], dtype=torch.int32, device="cuda"))

    assert torch.all(ssm[:, 0] == 42.0), "corrupted R103's slot 0"
    assert torch.all(conv[:, 0] == 42.0)


def test_cpp_add_dummy_requests_filters_sentinel():
    """CppMambaCacheManager.add_dummy_requests must drop the sentinel
    dummy before calling the C++ allocator. (Cpp doesn't support spec
    decoding, so only the exact sentinel is reserved.)"""
    stub = SimpleNamespace(mamba_impl=MagicMock(), speculative_num_draft_tokens=None)
    CppMambaCacheManager.add_dummy_requests(stub, [100, CUDA_GRAPH_DUMMY_REQUEST_ID, 101])
    stub.mamba_impl.allocate_cache_blocks.assert_called_once_with([100, 101])


def test_cpp_add_dummy_requests_noop_when_only_sentinel():
    stub = SimpleNamespace(mamba_impl=MagicMock(), speculative_num_draft_tokens=None)
    CppMambaCacheManager.add_dummy_requests(stub, [CUDA_GRAPH_DUMMY_REQUEST_ID])
    stub.mamba_impl.allocate_cache_blocks.assert_not_called()
