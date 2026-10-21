# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the target's MTP IndexShare plumbing.

The target's canonical selection, the capture of each request's draft-extend row
into the shared buffers (decode and mixed-batch forms), and the reuse entry's
order preservation over canonical captures. The targets are built without weights:
the capture methods read only their arguments, the metadata and the core's
draft binding.
"""

from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as m,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_canonical_selection import (  # noqa: E501
    qsa_canonical_selection as canonical,
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


COLS = 2080


def _scores_and_windows(rows: int, gen: torch.Generator):
    """Relu-shaped scores (half zeros: ties at the budget's last score whenever fewer than
    K are positive) over windows of 0 to 3K groups; the columns past a window are NaN."""
    ends = torch.randint(0, COLS + 1, (rows,), generator=gen).to(torch.int32).to(DEV)
    scores = torch.relu(torch.randn(rows, COLS, generator=gen)).to(DEV)
    scores = scores * (torch.rand(rows, COLS, generator=gen).to(DEV) < 0.3)
    outside = torch.arange(COLS, device=DEV)[None, :] >= ends[:, None]
    scores.masked_fill_(outside, float("nan"))
    return scores, torch.zeros(rows, dtype=torch.int32, device=DEV), ends


def _select(scores, starts, ends) -> torch.Tensor:
    out = torch.empty((scores.shape[0], K), dtype=torch.int32, device=DEV)
    m._select(SimpleNamespace(block_topk=K), scores, starts, ends, out)
    return out


def test_select_is_the_canonical_row_ascending_with_padding_last():
    gen = torch.Generator().manual_seed(3)
    scores, starts, ends = _scores_and_windows(64, gen)

    out = _select(scores, starts, ends)

    assert out.dtype == torch.int32
    assert torch.equal(out, canonical.reference(scores, starts, ends, out))
    for row, end in zip(out.cpu().tolist(), ends.tolist()):
        ids = [i for i in row if i >= 0]
        assert ids == sorted(ids) and len(set(ids)) == len(ids) == min(end, K)
        assert row == ids + [-1] * (K - len(ids)), "padding must stay last"


def test_decode_capture_at_draft_seven_takes_row_s_times_8_plus_a_minus_1():
    """max_draft_len 7: eight rows per request, accepted 1..8, so the captured row is
    `s * 8 + a - 1` -- the first and the last row of a block included."""
    p, batch = 8, 4
    accepted = [1, 8, 5, 3]
    topk = torch.arange(batch * p * K, dtype=torch.int32, device=DEV).reshape(batch * p, K)
    visible = torch.arange(300, 300 + batch * p, dtype=torch.int32, device=DEV)
    md = _metadata(batch, accepted)

    _target(m.DecodeTarget)._capture_selection(md, topk, visible, batch, p)

    rows = [s * p + a - 1 for s, a in enumerate(accepted)]
    assert rows == [0, 15, 20, 26]
    torch.testing.assert_close(md.qsa_shared_topk_indices[SLOT, :batch], topk[rows])
    torch.testing.assert_close(md.qsa_shared_topk_visible_blocks[SLOT, :batch], visible[rows])
    # Other slots and rows past the batch are untouched.
    assert torch.all(md.qsa_shared_topk_indices[SLOT, batch:] == -7)
    assert torch.all(md.qsa_shared_topk_indices[0] == -7)
    assert md.qsa_shared_topk_captured_slots == {SLOT}


def test_prefill_capture_at_draft_seven():
    """A mixed batch at max_draft_len 7: two context chunks of 5 and 3 rows, then three
    generation requests of 8 rows with 8, 1 and 6 accepted. A context request captures its
    chunk's last row whatever accepted count the worker left there."""
    seq_lens = [5, 3, 8, 8, 8]
    accepted = [2, 3, 8, 1, 6]
    rows_total = sum(seq_lens)
    topk = torch.arange(rows_total * K, dtype=torch.int32, device=DEV).reshape(rows_total, K)
    visible = torch.arange(400, 400 + rows_total, dtype=torch.int32, device=DEV)
    md = _metadata(5, accepted, seq_lens=seq_lens, num_contexts=2)

    _target(m.PrefillTarget)._capture_selection(md, topk, visible)

    rows = [4, 7, 8 + 7, 16 + 0, 24 + 5]
    torch.testing.assert_close(md.qsa_shared_topk_indices[SLOT, :5], topk[rows])
    torch.testing.assert_close(md.qsa_shared_topk_visible_blocks[SLOT, :5], visible[rows])
    assert md.qsa_shared_topk_captured_slots == {SLOT}


def _reuse_case(fresh: int):
    """A canonical capture of 16 requests and the reuse call's arguments for it."""
    gen = torch.Generator().manual_seed(11 + fresh)
    batch = 16
    scores, starts, ends = _scores_and_windows(batch, gen)
    captured_visible = ends.cpu()
    captured = _select(scores, starts, ends)
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


def test_reuse_of_a_canonical_capture_stays_ascending():
    """A canonical capture plus the groups completed since is still ascending: appended ids
    are newer than every captured one, and a full row gives up its oldest ids to them, so
    the newest captured groups -- the ones next to the new ones -- are kept."""
    for fresh in (0, 2):
        args, captured_visible = _reuse_case(fresh)

        out = reuse(**args)

        assert torch.equal(out, reuse.reference(**args))
        for row, shared_row, vis in zip(
            out.cpu().tolist(), args["shared_indices"].cpu().tolist(), captured_visible.tolist()
        ):
            ids = [i for i in row if i >= 0]
            assert row == ids + [-1] * (K - len(ids)), "padding must stay last"
            assert ids == sorted(ids) and len(set(ids)) == len(ids)
            assert len(ids) == min(vis + fresh, K)
            assert all(i < vis + fresh for i in ids)
            captured = [i for i in shared_row if i >= 0]
            kept = captured[max(0, len(captured) + fresh - K) :]
            assert ids == kept + list(range(vis, vis + fresh))
