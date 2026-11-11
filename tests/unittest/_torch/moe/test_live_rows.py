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
"""The live-row word's host reference (``live_row_count``, which the kernels compute too) and the engine's words."""

import ast
from pathlib import Path
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


def _row_owner(requests: List[SimpleNamespace], rows_per_request: int) -> List[SimpleNamespace]:
    """The request of every row: each request owns ``rows_per_request`` consecutive rows."""
    return [request for request in requests for _ in range(rows_per_request)]


def _word(
    requests: List[SimpleNamespace],
    rows_per_request: int = 1,
    *,
    num_contexts: int = 0,
    seq_lens: Optional[torch.Tensor] = None,
) -> int:
    """The word of a forward whose sequences own ``rows_per_request`` rows each, unless given."""
    if seq_lens is None:
        seq_lens = torch.full((len(requests),), rows_per_request, dtype=torch.int32)
    return lr.forward_word(requests, num_contexts=num_contexts, seq_lens=seq_lens)


def _kept_rows(requests: List[SimpleNamespace], rows_per_request: int) -> int:
    return lr.live_row_count(_word(requests, rows_per_request), len(requests) * rows_per_request)


@pytest.mark.parametrize("rows_per_request", [1, 4, 3])
@pytest.mark.parametrize(
    "kinds",
    [
        "rrrrrrrr",  # a full batch
        "rrrrrddd",  # graph padding behind five real requests
        "rddddddd",  # one real request left
        "dddddddd",  # an attention-DP placeholder rank padded to the graph size
        "d",  # an attention-DP placeholder rank running eagerly
    ],
)
def test_padding_rows_dropped_and_real_rows_kept(kinds: str, rows_per_request: int) -> None:
    # 1 row per request: no speculation; 4: the target forward of one-model MTP with
    # three draft tokens; 3: an MTP draft layer over the same requests.
    requests = _requests(*kinds)
    kept = _kept_rows(requests, rows_per_request)
    for row, owner in enumerate(_row_owner(requests, rows_per_request)):
        assert (row < kept) == (not owner.is_dummy), (kinds, rows_per_request, row)


def test_every_launch_of_one_forward_keeps_its_own_rows() -> None:
    # One word serves launches of different row counts in one forward.
    word = _word(_requests(*"rrrdd"))
    assert word == lr.encode(3, 5)
    assert [lr.live_row_count(word, rows) for rows in (5, 10, 15, 20)] == [3, 6, 9, 12]


def test_rows_not_split_evenly_into_requests_are_all_kept() -> None:
    word = lr.encode(3, 5)
    assert lr.live_row_count(word, 7) == 7
    assert lr.live_row_count(word, 0) == 0


def test_padding_before_a_real_request_is_kept() -> None:
    # Only padding behind the last real request is dropped.
    requests = _requests(*"rdrdd")
    assert _word(requests) == lr.encode(3, 5)
    assert lr.live_row_count(_word(requests), 10) == 6
    assert _word(_requests(*"rrdr")) == lr.ALL_ROWS


def test_a_forward_with_contexts_or_unequal_rows_keeps_every_row() -> None:
    requests = _requests(*"rrdd")
    assert _word(requests, 4) == lr.encode(2, 4)
    # A context request in the forward (a mixed batch), whatever the rows.
    assert _word(requests, 4, num_contexts=1) == lr.ALL_ROWS
    # Sequences of unequal rows (ragged verification, a one-token context among MTP requests).
    ragged = torch.tensor([4, 2, 4, 4], dtype=torch.int32)
    assert _word(requests, seq_lens=ragged) == lr.ALL_ROWS
    # A forward whose metadata holds no rows.
    empty = torch.zeros(0, dtype=torch.int32)
    assert lr.forward_word(requests, num_contexts=0, seq_lens=None) == lr.ALL_ROWS
    assert _word(requests, seq_lens=empty) == lr.ALL_ROWS
    # Padding only drops every row, and no requests keep every row.
    assert _word(_requests(*"dd"), num_contexts=1, seq_lens=ragged) == lr.NO_ROWS
    assert _word([]) == lr.ALL_ROWS


def test_word_layout() -> None:
    assert lr.ALL_ROWS == lr.encode(1, 1)
    assert lr.NO_ROWS == 0
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


def test_kernels_read_the_word_of_the_running_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    # Writes before a kernel allocates the word only record the value: no device.
    engine = lr.LiveRowWord()
    engine.write(lr.encode(2, 4))
    assert engine._value == lr.encode(2, 4)
    monkeypatch.setattr(lr.LiveRowWord, "tensor", lambda word, device: word)
    with model_extra_attrs({lr.LIVE_ROW_WORD_ATTR: engine}):
        assert lr.current_word("cuda:0") is engine
    for attrs in ({}, None):
        with model_extra_attrs(attrs):
            assert lr.current_word("cuda:0") is lr._ALL_ROWS_WORDS[0]
    assert lr._ALL_ROWS_WORDS[0]._value == lr.ALL_ROWS


# A forward of 512 requests of 4 rows each (one-model MTP), the first 400 real: 2048 rows, the
# first 1600 real. A MoE layer that splits it into two chunks of 1024 rows launches over each
# chunk; read with the forward's word, the first chunk would keep 1024 / 512 * 400 = 800 rows and
# drop rows 800-1023, which belong to the real requests 200-255.
_CHUNKED_FORWARD = dict(real=400, requests=512, rows=2048, chunk=1024)


def test_launches_over_part_of_a_forward_keep_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = lr.LiveRowWord()
    engine.write(lr.encode(_CHUNKED_FORWARD["real"], _CHUNKED_FORWARD["requests"]))
    monkeypatch.setattr(lr.LiveRowWord, "tensor", lambda word, device: word)
    with model_extra_attrs({lr.LIVE_ROW_WORD_ATTR: engine}):
        assert lr.live_row_count(lr.current_word("cuda:0")._value, _CHUNKED_FORWARD["rows"]) == 1600
        with lr.part_of_the_forward():
            chunk_word = lr.current_word("cuda:0")
        assert lr.current_word("cuda:0") is engine
    assert chunk_word is lr._ALL_ROWS_WORDS[0]
    assert (
        lr.live_row_count(chunk_word._value, _CHUNKED_FORWARD["chunk"]) == _CHUNKED_FORWARD["chunk"]
    )


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
    with model_extra_attrs({lr.LIVE_ROW_WORD_ATTR: engine}), pytest.raises(Stop):
        scheduler.forward(
            x,
            logits,
            do_finalize=True,
            output_dtype=None,
            all_rank_num_tokens=None,
            use_dp_padding=None,
            input_ids=None,
        )
    # One launch over the whole forward keeps the real rows; chunks keep every row.
    assert seen == ([1600] if num_chunks == 1 else [chunk] * num_chunks)


def test_the_sm100_megamoe_kernel_still_reads_the_word() -> None:
    """The kernel tree is refreshed by copying upstream sources in; a copy that drops the
    live-row changes would keep every padding row with nothing else failing on the CPU."""
    import tensorrt_llm

    kernel_file = (
        Path(tensorrt_llm.__file__).parent
        / "_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega"
        / "block_scaled_swap_ab_mega_moe_kernel.py"
    )
    tree = ast.parse(kernel_file.read_text())
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BlockScaledSwapAbMegaMoeKernel"
    )
    declares = [
        node.value.value
        for node in kernel.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "reads_live_row_word"
    ]
    call = next(
        node
        for node in kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    assert declares == [True]
    assert "live_row_word" in [argument.arg for argument in call.args.args]
