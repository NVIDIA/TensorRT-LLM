# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the target's MTP IndexShare plumbing.

The draft layer's selection sort, the capture of each request's draft-extend row
into the shared buffers (decode and mixed-batch forms), and the reuse entry's
order preservation over sorted captures. The targets are built without weights:
the capture methods read only their arguments, the metadata and the core's
draft binding.
"""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as m,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_mtp_reuse_topk import (  # noqa: E501
    qsa_mtp_reuse_topk as reuse,
)

assert torch.cuda.is_available(), "the IndexShare plumbing runs on a CUDA device"

DEV = "cuda"
K = 512
SLOT = 1


def _target(cls):
    target = object.__new__(cls)
    target.core = SimpleNamespace(_draft_bound=SimpleNamespace(share_slot=SLOT))
    return target


def _metadata(batch: int, accepted, seq_lens=None, num_contexts: int = 0):
    return SimpleNamespace(
        num_seqs=batch,
        num_contexts=num_contexts,
        seq_lens_cuda=None
        if seq_lens is None
        else torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        qsa_mtp_num_accepted=torch.tensor(accepted, dtype=torch.int32, device=DEV),
        qsa_shared_topk_indices=torch.full((3, 8, K), -7, dtype=torch.int32, device=DEV),
        qsa_shared_topk_visible_blocks=torch.full((3, 8), -7, dtype=torch.int32, device=DEV),
        qsa_shared_topk_captured_slots=set(),
    )


def _selection_rows(rows: int, gen: torch.Generator) -> torch.Tensor:
    """Rows as `qsa_topk_select` leaves them: distinct ids in any order, then -1."""
    out = torch.full((rows, K), -1, dtype=torch.int32)
    for r in range(rows):
        visible = int(torch.randint(0, 3 * K, (1,), generator=gen))
        take = min(visible, K)
        out[r, :take] = torch.randperm(max(visible, 1), generator=gen)[:take].to(torch.int32)
    return out.to(DEV)


def test_sort_selection_orders_ids_and_keeps_padding_last():
    gen = torch.Generator().manual_seed(3)
    rows = _selection_rows(64, gen)
    # Holes as well: the sort must pack them, not just order a packed prefix.
    rows[5, 3] = -1
    rows[9, 0] = -1
    expected = []
    for row in rows.cpu().tolist():
        ids = sorted(i for i in row if i >= 0)
        expected.append(ids + [-1] * (K - len(ids)))
    out = rows.clone()

    m._sort_selection(out)

    assert out.dtype == torch.int32
    assert out.cpu().tolist() == expected


def test_decode_capture_takes_each_request_last_accepted_row():
    p, batch = 4, 3
    accepted = [1, 4, 2]
    topk = torch.arange(batch * p * K, dtype=torch.int32, device=DEV).reshape(batch * p, K)
    visible = torch.arange(100, 100 + batch * p, dtype=torch.int32, device=DEV)
    md = _metadata(batch, accepted)

    _target(m.DecodeTarget)._capture_selection(md, topk, visible, batch, p)

    rows = [s * p + a - 1 for s, a in enumerate(accepted)]
    torch.testing.assert_close(md.qsa_shared_topk_indices[SLOT, :batch], topk[rows])
    torch.testing.assert_close(md.qsa_shared_topk_visible_blocks[SLOT, :batch], visible[rows])
    # Other slots and rows past the batch are untouched.
    assert torch.all(md.qsa_shared_topk_indices[SLOT, batch:] == -7)
    assert torch.all(md.qsa_shared_topk_indices[0] == -7)
    assert md.qsa_shared_topk_captured_slots == {SLOT}


def test_prefill_capture_takes_chunk_last_row_and_last_accepted_row():
    """Two context requests (chunks of 5 and 3 rows) then two generation
    requests (4 rows each, 2 and 4 accepted). The context requests' accepted
    count is whatever the worker left there and must not matter."""
    seq_lens = [5, 3, 4, 4]
    accepted = [3, 1, 2, 4]
    rows_total = sum(seq_lens)
    topk = torch.arange(rows_total * K, dtype=torch.int32, device=DEV).reshape(rows_total, K)
    visible = torch.arange(200, 200 + rows_total, dtype=torch.int32, device=DEV)
    md = _metadata(4, accepted, seq_lens=seq_lens, num_contexts=2)

    _target(m.PrefillTarget)._capture_selection(md, topk, visible)

    rows = [4, 7, 8 + 1, 12 + 3]
    torch.testing.assert_close(md.qsa_shared_topk_indices[SLOT, :4], topk[rows])
    torch.testing.assert_close(md.qsa_shared_topk_visible_blocks[SLOT, :4], visible[rows])
    assert md.qsa_shared_topk_captured_slots == {SLOT}


def _reuse_case(fresh: int):
    """A sorted capture of 16 requests and the reuse call's arguments for it."""
    gen = torch.Generator().manual_seed(11 + fresh)
    batch = 16
    captured_visible = torch.randint(0, 3 * K, (batch,), generator=gen).to(torch.int32)
    captured = torch.full((batch, K), -1, dtype=torch.int32)
    for r, vis in enumerate(captured_visible.tolist()):
        take = min(vis, K)
        captured[r, :take] = torch.randperm(max(vis, 1), generator=gen)[:take].to(torch.int32)
    captured = captured.to(DEV)
    m._sort_selection(captured)
    shared = torch.full((8, 32, K), -1, dtype=torch.int32, device=DEV)
    shared[SLOT, :batch] = captured
    shared_visible = torch.zeros((8, 32), dtype=torch.int32, device=DEV)
    shared_visible[SLOT, :batch] = captured_visible.to(DEV)
    args = dict(
        shared_indices=shared[SLOT],
        shared_visible_blocks=shared_visible[SLOT],
        request_indices=torch.arange(batch, dtype=torch.int32, device=DEV),
        visible_blocks=(captured_visible + fresh).to(DEV),
        output=torch.empty((batch, K), dtype=torch.int32, device=DEV),
    )
    return args, captured_visible


def test_reuse_of_a_sorted_capture_stays_sorted():
    """A sorted capture plus the groups completed since is still ascending:
    appended ids are newer than every captured one, and a full row loses its
    largest ids to them."""
    for fresh in (0, 1, 2):
        args, captured_visible = _reuse_case(fresh)

        out = reuse(**args)

        assert torch.equal(out, reuse.reference(**args))
        for row, vis in zip(out.cpu().tolist(), captured_visible.tolist()):
            ids = [i for i in row if i >= 0]
            assert row == ids + [-1] * (K - len(ids)), "padding must stay last"
            assert ids == sorted(ids) and len(set(ids)) == len(ids)
            assert len(ids) == min(vis + fresh, K)
            assert all(i < vis + fresh for i in ids)
