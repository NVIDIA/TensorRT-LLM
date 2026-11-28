# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host's live rows of a forward (``forward_word`` / ``live_row_count``, which the route staging
computes on the device too): only the padding requests behind the last real request are dropped."""

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from tensorrt_llm._torch.moe import live_rows as lr
from tensorrt_llm._torch.utils import model_extra_attrs

pytestmark = pytest.mark.cpu_only


def _requests(*kinds: str) -> List[SimpleNamespace]:
    """Requests in row order: ``r`` real, ``d`` padding."""
    return [SimpleNamespace(is_dummy=kind == "d") for kind in kinds]


def _word(
    requests: List[SimpleNamespace],
    rows_per_request: int = 1,
    *,
    num_contexts: int = 0,
    seq_lens: Optional[torch.Tensor] = None,
) -> int:
    if seq_lens is None:
        seq_lens = torch.full((len(requests),), rows_per_request, dtype=torch.int32)
    return lr.forward_word(requests, num_contexts=num_contexts, seq_lens=seq_lens)


@pytest.mark.parametrize("rows_per_request", [1, 4, 3])
@pytest.mark.parametrize("kinds", ["rrrrrrrr", "rrrrrddd", "rddddddd", "dddddddd", "d"])
def test_padding_rows_dropped_and_real_rows_kept(kinds: str, rows_per_request: int) -> None:
    # 1 row per request: no speculation; 4: an MTP target forward with three draft tokens;
    # 3: an MTP draft layer.
    requests = _requests(*kinds)
    kept = lr.live_row_count(_word(requests, rows_per_request), len(requests) * rows_per_request)
    assert kept == kinds.count("r") * rows_per_request, (kinds, rows_per_request)


def test_a_forward_with_contexts_or_unequal_rows_keeps_every_row() -> None:
    requests = _requests(*"rrdd")
    assert _word(requests, 4) == lr.encode(2, 4)
    # A context request in the forward (a mixed batch), whatever the rows.
    assert _word(requests, 4, num_contexts=1) == lr.ALL_ROWS
    # Sequences of unequal rows (ragged verification, a one-token context among MTP requests).
    ragged = torch.tensor([4, 2, 4, 4], dtype=torch.int32)
    assert _word(requests, seq_lens=ragged) == lr.ALL_ROWS
    # A forward whose metadata holds no rows.
    assert lr.forward_word(requests, num_contexts=0, seq_lens=None) == lr.ALL_ROWS
    assert _word(requests, seq_lens=torch.zeros(0, dtype=torch.int32)) == lr.ALL_ROWS
    # Padding only drops every row, and no requests keep every row.
    assert _word(_requests(*"dd"), num_contexts=1, seq_lens=ragged) == lr.NO_ROWS
    assert _word([]) == lr.ALL_ROWS
    # Only padding behind the last real request is dropped.
    assert _word(_requests(*"rdrdd")) == lr.encode(3, 5)
    assert lr.live_row_count(_word(_requests(*"rdrdd")), 10) == 6
    assert _word(_requests(*"rrdr")) == lr.ALL_ROWS


def test_word_layout() -> None:
    assert lr.ALL_ROWS == lr.encode(1, 1)
    assert lr.NO_ROWS == 0
    # One word serves launches of different row counts in one forward.
    assert [lr.live_row_count(lr.encode(3, 5), rows) for rows in (5, 10, 20)] == [3, 6, 12]
    assert lr.live_row_count(lr.encode(3, 5), 7) == 7  # rows not split evenly: all kept
    assert lr.live_row_count(lr.encode(3, 5), 0) == 0
    assert lr.live_row_count(lr.ALL_ROWS, 96) == 96
    assert lr.live_row_count(lr.NO_ROWS, 96) == 0
    assert lr.live_row_count(lr.encode(0, 4), 96) == 0
    assert lr.live_row_count(lr.encode(4, 4), 96) == 96
    top = lr.MAX_REQUESTS
    assert lr.live_row_count(lr.encode(top - 1, top), 2 * top) == 2 * (top - 1)
    assert lr.encode(top, top) >> lr.REQUEST_BITS == top
    for real, requests in ((1, 0), (2, 1), (-1, 1), (0, top + 1)):
        with pytest.raises(ValueError):
            lr.encode(real, requests)


@pytest.mark.parametrize("keep", ["0", "1"])
def test_the_debug_switch_keeps_every_row(monkeypatch: pytest.MonkeyPatch, keep: str) -> None:
    """With ``TRTLLM_MEGAMOE_KEEP_PADDING_ROWS=1`` the engine registers no word, so kernels read
    the word that keeps every row."""
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setenv(lr.KEEP_PADDING_ROWS_ENV, keep)
    monkeypatch.setattr(lr.LiveRowWord, "tensor", lambda word, device: word)
    model = SimpleNamespace(extra_attrs={})
    engine = lr.engine_word(model, Mapping())
    with model_extra_attrs(model.extra_attrs):
        read = lr.current_word("cuda:0")
    assert read is (lr._ALL_ROWS_WORDS[0] if keep == "1" else engine)


# 512 requests of 4 rows, the first 400 real: 2048 rows, 1600 real. Read with the forward's word,
# the first of two 1024-row chunks would keep only 800 rows and drop real ones.
_CHUNKED_FORWARD = dict(real=400, requests=512, rows=2048, chunk=1024)


@pytest.mark.parametrize("num_chunks", [1, 2])
def test_a_moe_layer_reads_the_forward_word_only_when_it_runs_in_one_chunk(
    monkeypatch: pytest.MonkeyPatch, num_chunks: int
) -> None:
    from tensorrt_llm._torch.moe.fused_moe import moe_scheduler

    class Stop(Exception):
        pass

    engine = lr.LiveRowWord()
    engine.write(lr.encode(_CHUNKED_FORWARD["real"], _CHUNKED_FORWARD["requests"]))
    monkeypatch.setattr(lr.LiveRowWord, "tensor", lambda word, device: word)
    scheduler = moe_scheduler.FusedCommMoEScheduler.__new__(moe_scheduler.FusedCommMoEScheduler)
    scheduler.moe = SimpleNamespace(
        backend=SimpleNamespace(input_requirement=SimpleNamespace(run_moe_reuses_buffers=False))
    )
    rows = _CHUNKED_FORWARD["rows"]
    x, logits = torch.zeros(rows, 8), torch.zeros(rows, 4)
    chunk = rows // num_chunks
    monkeypatch.setattr(
        scheduler,
        "_strip_adp_padding",
        lambda x, rl, ids, all_rank: (x, rl, None, [rows], 0, False),
    )
    monkeypatch.setattr(
        scheduler,
        "_compute_chunk_layout",
        lambda x, rl, all_rank, ep_rank: (
            num_chunks,
            list(x.split(chunk)),
            list(rl.split(chunk)),
            [[chunk] * num_chunks],
        ),
    )
    seen = []

    def run_chunks(x_chunks: List[torch.Tensor], *args: object, **kwargs: object) -> None:
        seen.extend(
            lr.live_row_count(lr.current_word("cuda:0")._value, c.shape[0]) for c in x_chunks
        )
        raise Stop

    monkeypatch.setattr(scheduler, "_run_chunks", run_chunks)
    with model_extra_attrs({lr.LIVE_ROW_WORD_ATTR: engine}):
        with pytest.raises(Stop):
            scheduler.forward(
                x,
                logits,
                do_finalize=True,
                output_dtype=None,
                all_rank_num_tokens=None,
                use_dp_padding=None,
                input_ids=None,
            )
        assert lr.current_word("cuda:0") is engine  # restored after the chunks
    # One launch over the whole forward keeps the real rows; chunks keep every row.
    assert seen == ([1600] if num_chunks == 1 else [chunk] * num_chunks)
