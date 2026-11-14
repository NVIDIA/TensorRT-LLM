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
"""CPU tests for DSpark under a draft length schedule."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.utils import update_draft_len
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

pytestmark = pytest.mark.cpu_only

BLOCK = 3
WINDOW = 4


def _worker(num_rows=2):
    worker = DSv4DSparkWorker.__new__(DSv4DSparkWorker)
    worker.spec_config = SimpleNamespace(max_draft_len=BLOCK)
    worker.return_confidence = False
    worker.guided_decoder = None
    worker._scratch_slot = num_rows
    worker._batch_to_slot = torch.arange(num_rows)
    worker._ctx_len = torch.zeros(num_rows + 1, dtype=torch.long)
    worker._valid_len = torch.zeros(num_rows + 1, dtype=torch.long)
    worker._position_initialized = torch.zeros(num_rows + 1, dtype=torch.bool)
    worker._kv_windows = torch.zeros(num_rows + 1, 1, WINDOW, 2)
    worker._win = WINDOW
    worker._position_cap = None
    worker._confidence_logits = None
    worker._confidence_stamp = None
    worker._output_verify_lens = None
    worker._lazy_init = Mock()
    # Any use of the draft-verifying entry on a step without drafts is a bug.
    worker.sample_and_accept_draft_tokens = Mock(side_effect=AssertionError)
    return worker


def _draft_model():
    def forward(main_hidden, bonus, start_pos, **kwargs):
        return None, None, torch.zeros(main_hidden.shape[0], BLOCK, 4)

    return SimpleNamespace(
        block_size=BLOCK,
        forward_batched=Mock(side_effect=forward),
        write_context_windows_batched=Mock(),
        run_moe_lockstep_noop=Mock(),
    )


def _skip_outputs(batch_size):
    """What ``SpecWorkerBase.skip_drafting`` returns for a batch."""
    tokens = torch.arange(100, 100 + batch_size, dtype=torch.int).unsqueeze(1)
    return {
        "logits": torch.zeros(batch_size, 4),
        "new_tokens": tokens,
        "new_tokens_lens": torch.ones(batch_size, dtype=torch.int),
        "next_draft_tokens": torch.zeros((batch_size, 0), dtype=torch.int),
        "next_new_tokens": tokens.clone(),
    }


def _metadata(captured, **overrides):
    values = dict(
        runtime_draft_len=0,
        is_cuda_graph=False,
        qo_indptr=None,
        verify_lens=None,
        get_hidden_states=lambda num_tokens: captured[:num_tokens],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _forward(worker, model, metadata, batch_size, positions, num_contexts=0, num_ctx_tokens=0):
    worker.skip_drafting = Mock(return_value=_skip_outputs(batch_size))
    num_tokens = num_ctx_tokens + batch_size - num_contexts
    return worker._forward_impl(
        torch.zeros(num_tokens, dtype=torch.long),
        positions,
        None,
        torch.zeros(batch_size, 4),
        SimpleNamespace(
            num_seqs=batch_size, num_contexts=num_contexts, num_ctx_tokens=num_ctx_tokens
        ),
        metadata,
        model,
    )


@pytest.fixture(autouse=True)
def _not_capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_dspark_mode_supports_a_draft_len_schedule():
    assert SpeculativeDecodingMode.DSPARK.support_dynamic_draft_len()


@pytest.mark.parametrize("batch_size, expected", [(2, [7, 8, 9]), (4, [7, 8]), (5, [])])
def test_schedule_trims_the_verified_part_of_the_block(batch_size, expected):
    """The drafter always proposes a full block; the schedule keeps its prefix."""
    config = DSparkDecodingConfig(max_draft_len=BLOCK, draft_len_schedule={2: 3, 4: 2})
    engine = SimpleNamespace(spec_config=config, max_draft_len=BLOCK)
    requests = [
        SimpleNamespace(py_draft_tokens=[7, 8, 9], py_needs_onehot_draft_probs=False)
        for _ in range(batch_size)
    ]
    batch = SimpleNamespace(batch_size=batch_size, generation_requests=requests)

    assert update_draft_len(engine, batch) == len(expected)
    assert all(request.py_draft_tokens == expected for request in requests)


def test_max_concurrency_becomes_a_speculation_cutoff():
    config = DSparkDecodingConfig(max_draft_len=BLOCK, max_concurrency=8)
    assert config.draft_len_schedule == {8: BLOCK}


def test_step_without_speculation_follows_the_target_in_the_windows():
    worker = _worker()
    worker._ctx_len[:2] = torch.tensor([10, 20])
    worker._valid_len[:2] = torch.tensor([2, WINDOW])
    worker._position_initialized[:2] = True
    model = _draft_model()
    captured = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    outputs = _forward(worker, model, _metadata(captured), 2, torch.tensor([[10, 20]]))

    # The step proposes nothing and runs no draft block.
    model.forward_batched.assert_not_called()
    assert outputs["next_draft_tokens"].shape == (2, 0)
    assert outputs["new_tokens_lens"].tolist() == [1, 1]
    assert outputs["native_uniform_verify"] is True
    # Each request's window gains the frame of the one token the target ran,
    # one position past the request's previous extent.
    hidden, positions, slots, mask, windows = model.write_context_windows_batched.call_args.args
    assert torch.equal(hidden, captured.unsqueeze(1))
    assert positions.tolist() == [[11], [21]]
    assert slots.tolist() == [0, 1]
    assert mask.tolist() == [[True], [True]]
    assert windows is worker._kv_windows
    assert worker._ctx_len[:2].tolist() == [11, 21]
    assert worker._valid_len[:2].tolist() == [3, WINDOW]


def test_step_without_speculation_reads_generation_rows_past_the_context_tokens():
    """In a mixed batch the generation rows start after every context token."""
    worker = _worker(num_rows=3)
    worker._seed_context_windows = Mock()
    model = _draft_model()
    # One context request of three tokens, then one row per generation request.
    captured = torch.arange(5, dtype=torch.float).unsqueeze(1).expand(5, 2)
    positions = torch.tensor([[0, 1, 2, 10, 20]])

    outputs = _forward(
        worker, model, _metadata(captured), 3, positions, num_contexts=1, num_ctx_tokens=3
    )

    worker._seed_context_windows.assert_called_once()
    model.forward_batched.assert_not_called()
    assert outputs["next_draft_tokens"].shape == (3, 0)
    hidden, window_positions, slots, mask, _windows = (
        model.write_context_windows_batched.call_args.args
    )
    assert torch.equal(hidden, captured[3:5].unsqueeze(1))
    assert window_positions.tolist() == [[11], [21]]
    assert slots.tolist() == [1, 2]
    assert mask.tolist() == [[True], [True]]
    assert worker._ctx_len[1:3].tolist() == [11, 21]


def test_step_without_speculation_bootstraps_a_new_slot_from_its_input_position():
    worker = _worker()
    model = _draft_model()

    _forward(worker, model, _metadata(torch.zeros(2, 2)), 2, torch.tensor([[40, 7]]))

    assert worker._ctx_len[:2].tolist() == [41, 8]
    assert worker._position_initialized[:2].tolist() == [True, True]


def test_step_without_speculation_keeps_padding_rows_out_of_live_windows():
    worker = _worker()
    worker._batch_to_slot = torch.tensor([0, worker._scratch_slot])
    model = _draft_model()
    # The padding row lies past the captured extent.
    captured = torch.tensor([[1.0, 1.0]])

    _forward(worker, model, _metadata(captured), 2, torch.tensor([[5]]))

    hidden, _positions, slots, mask, _windows = model.write_context_windows_batched.call_args.args
    assert torch.equal(hidden[:, 0], captured[[0, 0]])
    assert slots.tolist() == [0, worker._scratch_slot]
    assert mask.tolist() == [[True], [False]]


def test_warmup_step_without_speculation_restores_the_window_state():
    worker = _worker()
    worker._ctx_len[:2] = torch.tensor([10, 20])
    worker._position_initialized[:2] = True
    saved = (worker._ctx_len.clone(), worker._valid_len.clone())
    model = _draft_model()

    _forward(
        worker,
        model,
        _metadata(torch.zeros(2, 2), is_cuda_graph=True),
        2,
        torch.tensor([[10, 20]]),
    )

    model.write_context_windows_batched.assert_called_once()
    assert torch.equal(worker._ctx_len, saved[0])
    assert torch.equal(worker._valid_len, saved[1])


def test_step_without_speculation_keeps_the_draft_moe_in_lockstep():
    """Ranks sharing the draft MoE barrier make the same calls as a drafting peer."""
    worker = _worker()
    model = _draft_model()
    metadata = _metadata(
        torch.zeros(2, 2), all_rank_num_gens=[2, 1], all_rank_draft_lens=[0, BLOCK]
    )

    outputs = _forward(worker, model, metadata, 2, torch.tensor([[10, 20]]))

    model.forward_batched.assert_called_once()
    assert model.forward_batched.call_args.kwargs["all_rank_num_tokens"] == [2 * BLOCK, BLOCK]
    # The block ran for the barrier only; its proposal is dropped.
    assert outputs["next_draft_tokens"].shape == (2, 0)
    assert worker._ctx_len[:2].tolist() == [11, 21]


def test_idle_rank_without_speculation_still_crosses_the_draft_moe_barrier():
    worker = _worker()
    model = _draft_model()
    metadata = _metadata(
        torch.zeros(0, 2), all_rank_num_gens=[0, 2], all_rank_draft_lens=[0, BLOCK]
    )

    _forward(worker, model, metadata, 0, torch.zeros((1, 0), dtype=torch.long))

    model.forward_batched.assert_not_called()
    model.run_moe_lockstep_noop.assert_called_once()
    assert model.run_moe_lockstep_noop.call_args.args[0] == [1, 2 * BLOCK]


@pytest.mark.parametrize("num_gens", [2, 0])
def test_step_without_speculation_skips_the_draft_moe_when_no_rank_drafts(num_gens):
    """With every rank at draft length zero, none of them crosses the barrier."""
    worker = _worker()
    worker._ctx_len[:2] = torch.tensor([10, 20])
    worker._position_initialized[:2] = True
    model = _draft_model()
    metadata = _metadata(
        torch.zeros(num_gens, 2), all_rank_num_gens=[num_gens, 2], all_rank_draft_lens=[0, 0]
    )
    positions = torch.tensor([[10, 20]])[:, :num_gens]

    outputs = _forward(worker, model, metadata, num_gens, positions)

    model.forward_batched.assert_not_called()
    model.run_moe_lockstep_noop.assert_not_called()
    assert outputs["next_draft_tokens"].shape == (num_gens, 0)
    # The windows still follow the target, as they do without a barrier.
    assert model.write_context_windows_batched.call_count == (1 if num_gens else 0)
    assert worker._ctx_len[:2].tolist() == ([11, 21] if num_gens else [10, 20])
