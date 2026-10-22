# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target on a unified target/draft KV cache (KVCacheManagerV2 layer specs).

* The verify fold's and the context-K/V fork's written-out worker steps leave the DFlash
  dummy slot empty, as the built-in ``_advance_ctx_len`` does.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.verify_fold import (  # noqa: E501
    fold as verify_fold,
)

_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# --------------------------------------------------------------------------
# The written-out worker steps keep the dummy slot empty
# --------------------------------------------------------------------------
class _Drafter:
    """``project_target_hidden`` / ``precompute_context_kv`` of a drafter with 2 layers of 1 x 4 heads."""

    def project_target_hidden(self, hidden):
        return hidden

    def precompute_context_kv(self, projected, positions):
        k = (positions + 1).view(-1, 1, 1, 1).expand(-1, 2, 1, 4).to(projected.dtype).clone()
        return k, k + 0.5


def test_the_fold_store_empties_the_dummy_slot():
    worker = object.__new__(verify_fold.SpecLoopFoldDFlashWorker)
    # Slot 3 is the dummy slot, with a length a stale step left there.
    worker._ctx_len = torch.tensor([5, 0, 0, 6])
    worker._dummy_slot, worker._max_ctx, worker._ctx_base, worker._ctx_keep = 3, 32, None, 0
    worker._ctx_block_tables, worker._ctx_paged = None, False
    worker._ctx_k_buf = torch.zeros(4, 2, 40, 1, 4, dtype=torch.bfloat16)
    worker._ctx_v_buf = torch.zeros_like(worker._ctx_k_buf)
    slots, accepted = torch.tensor([0, 3]), torch.tensor([3, 8], dtype=torch.int32)
    offsets = torch.arange(8)
    worker._fold_store_accepted_context(
        _Drafter(),
        torch.randn(16, 4, dtype=torch.bfloat16),
        SimpleNamespace(num_ctx_tokens=0),
        2,
        8,
        7,
        slots,
        torch.arange(2),
        offsets,
        worker._ctx_len[slots].unsqueeze(1) + offsets,
        accepted,
        accepted.long(),
    )
    assert worker._ctx_len.tolist() == [8, 0, 0, 0]


@_needs_cuda
@pytest.mark.parametrize("windowed", [False, True])
def test_the_fork_commit_empties_the_dummy_slot(windowed):
    k_arena = torch.randn(4, 5, 64, 1, 128, device="cuda").bfloat16()
    worker = SimpleNamespace(
        _ctx_k_buf=k_arena,
        _ctx_v_buf=torch.randn_like(k_arena),
        _ctx_len=torch.tensor([5, 0, 0, 6], device="cuda"),
        # A windowed arena's slots start at a base; the dummy slot's is stale too.
        _ctx_base=torch.tensor([2, 0, 0, 3], device="cuda") if windowed else None,
        _ctx_keep=48 if windowed else 0,
        _max_ctx=64,
        _dummy_slot=3,
    )
    fork = context_kv_fork.ContextKvFork(worker, None, context_kv_fork.POINT)
    launched = context_kv_fork.Launched(
        torch.cuda.current_stream(), None, 2, 7, context_kv_fork.POINT, None
    )
    accepted = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
    num_ctx = fork.commit(launched, torch.tensor([0, 3], device="cuda"), accepted)
    fork.join(launched)
    torch.cuda.synchronize()
    assert worker._ctx_len.tolist() == [8, 0, 0, 0]
    assert num_ctx.tolist() == [8, 0]
    if windowed:
        assert worker._ctx_base.tolist() == [2, 0, 0, 0]
