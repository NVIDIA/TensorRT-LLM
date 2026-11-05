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
"""The k2q CSR build's reused buffers belong to one CUDA stream (TRTLLM_MSA_K2Q_WS): any other stream takes the
stock builder, so a build there cannot overwrite buffers an attention on the first stream still reads."""

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    require_msa_module,
)


class _BuilderTaken(Exception):
    """Raised where the reused-buffer build starts, so a test sees that it was taken."""


class _Utils:
    @property
    def _K2Q_CSR_BUILDER(self):
        raise _BuilderTaken


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm, current, owner, taken",
    [(107, 7, 7, True), (107, 8, 7, False), (107, 8, None, True), (100, 8, None, False)],
)
def test_the_reused_buffers_serve_one_stream(monkeypatch, sm, current, owner, taken) -> None:
    # SM107 only: other GPUs keep the stock builder.
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: sm)
    monkeypatch.setattr(msa_prefill, "_K2Q_WS", True)
    monkeypatch.setattr(msa_prefill, "_k2q_ws_stream", owner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=current)
    )
    monkeypatch.setitem(sys.modules, "sparse_index_utils", _Utils())
    monkeypatch.setitem(sys.modules, "src.sm100.prepare_scheduler", SimpleNamespace())
    kv_block_indexes = torch.zeros((4, 2, 8), dtype=torch.int32)
    cu = torch.zeros(2, dtype=torch.int32)
    kwargs = dict(max_seqlen_k=512, max_seqlen_q=4, total_rows=16, qhead_per_kv=8)
    try:
        got = msa_prefill._k2q_csr_ws(kv_block_indexes, cu, cu, 128, **kwargs)
    except _BuilderTaken:
        got = "taken"
    assert (got == "taken") if taken else got is None


def _active_csr_schedule(result: tuple[torch.Tensor, torch.Tensor, Any]) -> tuple:
    row_ptr, q_indices, schedule = result
    pointers = row_ptr.cpu().tolist()
    active_queries = tuple(
        tuple(q_indices[head, : bounds[-1]].cpu().tolist()) for head, bounds in enumerate(pointers)
    )
    active_splits = tuple(
        tuple(schedule.qsplit_indices[head, : bounds[-1]].cpu().tolist())
        for head, bounds in enumerate(pointers)
    )
    work_count = int(schedule.work_count.item())
    assert schedule.enabled
    assert 0 < work_count <= schedule.work_capacity
    # Metadata work items use atomic allocation: their order is unspecified,
    # while each active six-field item and its multiplicity must be preserved.
    active_work = tuple(
        sorted(tuple(row) for row in schedule.scheduler_metadata[:work_count].cpu().tolist())
    )
    return (
        tuple(tuple(row) for row in pointers),
        active_queries,
        active_splits,
        active_work,
        tuple(tuple(row) for row in schedule.split_counts.cpu().tolist()),
        schedule.target_q_per_cta,
    )


def _assert_csr_reference(
    result: tuple[torch.Tensor, torch.Tensor, Any],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    row_ptr, q_indices, _ = result
    expected_ptr, expected_indices = reference
    assert torch.equal(row_ptr, expected_ptr)
    # The independent Torch reference sorts each row. Preserve every query
    # membership and multiplicity without imposing that order on the kernel.
    pointers = row_ptr.cpu().tolist()
    for head, bounds in enumerate(pointers):
        for begin, end in zip(bounds, bounds[1:]):
            assert torch.equal(
                q_indices[head, begin:end].sort().values,
                expected_indices[head, begin:end],
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("topk", [4, 8, 16, 32])
@pytest.mark.parametrize("strided", [False, True], ids=["head-major", "mixed-strided"])
def test_k2q_workspace_matches_stock_contents(
    monkeypatch: pytest.MonkeyPatch, topk: int, strided: bool
) -> None:
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Reused CSR workspace dispatch requires SM107")
    # Load the real packaged builder and the aliases used by the serving path.
    require_msa_module()
    sparse = importlib.import_module("fmha_sm100.sparse")
    utils = importlib.import_module("sparse_index_utils")
    monkeypatch.setattr(msa_prefill, "_K2Q_WS", True)
    monkeypatch.setattr(msa_prefill, "_k2q_ws", None)
    monkeypatch.setattr(msa_prefill, "_k2q_ws_stream", None)
    previous_key = None
    for q_lengths, kv_lengths, heads in [((129, 17), (513, 385), 2), ((33, 9), (257, 129), 1)]:
        total_q = sum(q_lengths)
        cu_q = torch.tensor([0, q_lengths[0], total_q], dtype=torch.int32, device="cuda")
        cu_kv = torch.tensor([0, kv_lengths[0], sum(kv_lengths)], dtype=torch.int32, device="cuda")
        kwargs = dict(
            max_seqlen_k=max(kv_lengths),
            max_seqlen_q=max(q_lengths),
            total_rows=sum((length + 127) // 128 for length in kv_lengths),
            qhead_per_kv=8,
        )
        reused_pointers = None
        for revision in range(2):
            # Construct valid unique block selections with trailing -1s on the
            # host; revision changes content without changing the workspace key.
            selection = torch.full((heads, total_q, topk), -1, dtype=torch.int32)
            offset = 0
            for q_length, kv_length in zip(q_lengths, kv_lengths):
                blocks = (kv_length + 127) // 128
                for head in range(heads):
                    for row in range(q_length):
                        count = 1 + (row + head + revision) % min(topk, blocks)
                        selection[head, offset + row, :count] = (
                            torch.arange(count, dtype=torch.int32) + row + head + revision
                        ) % blocks
                offset += q_length
            head_major = selection.cuda()
            indexes = head_major.permute(1, 0, 2)
            if strided:
                backing = torch.full(
                    (total_q, heads, topk * 2), -1, dtype=torch.int32, device="cuda"
                )
                backing[:, :, ::2].copy_(indexes)
                indexes = backing[:, :, ::2]
                assert not indexes.permute(1, 0, 2).is_contiguous()
            else:
                assert indexes.permute(1, 0, 2).is_contiguous()
            q2k = indexes.permute(1, 0, 2).contiguous()
            reference = utils.build_k2q_csr_torch_reference(q2k, cu_q, cu_kv, 128)
            stock = sparse.build_k2q_csr(
                q2k, cu_q, cu_kv, 128, total_k=sum(kv_lengths), return_schedule=True, **kwargs
            )
            stock_snapshot = _active_csr_schedule(stock)
            _assert_csr_reference(stock, reference)
            # Stock-versus-stock is the control for unspecified allocation order.
            stock_again = sparse.build_k2q_csr(
                q2k, cu_q, cu_kv, 128, total_k=sum(kv_lengths), return_schedule=True, **kwargs
            )
            assert _active_csr_schedule(stock_again) == stock_snapshot
            reused = msa_prefill._k2q_csr_ws(indexes, cu_q, cu_kv, 128, **kwargs)
            assert reused is not None, "Expected the actual reused-buffer dispatch"
            assert _active_csr_schedule(reused) == stock_snapshot
            _assert_csr_reference(reused, reference)
            pointers = (reused[0].data_ptr(), reused[1].data_ptr())
            if reused_pointers is not None:
                assert pointers == reused_pointers
            reused_pointers = pointers
        key = msa_prefill._k2q_ws[0]
        if previous_key is not None:
            assert key != previous_key
        previous_key = key
