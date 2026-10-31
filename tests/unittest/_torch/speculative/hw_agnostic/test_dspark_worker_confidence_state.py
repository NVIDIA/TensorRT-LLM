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
"""CPU regression tests for the embedded worker's confidence-state contract."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_speculative import DSparkConfidenceHead
from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker

pytestmark = pytest.mark.cpu_only


def _worker(num_rows=2):
    worker = DSv4DSparkWorker.__new__(DSv4DSparkWorker)
    worker.spec_config = SimpleNamespace(max_draft_len=3)
    worker.return_confidence = True
    worker._scratch_slot = num_rows
    worker._batch_to_slot = torch.arange(num_rows)
    worker._ctx_len = torch.zeros(num_rows + 1, dtype=torch.long)
    worker._valid_len = torch.zeros(num_rows + 1, dtype=torch.long)
    worker._position_initialized = torch.zeros(num_rows + 1, dtype=torch.bool)
    worker._kv_windows = torch.zeros(num_rows + 1, 1, 4, 2)
    worker._win = 4
    worker._position_cap = None
    worker._confidence_logits = torch.full((num_rows + 2, 3), 30.0)
    worker._confidence_stamp = torch.zeros(num_rows + 2, dtype=torch.int32)
    worker._draft_seq_host = 0
    worker._draft_seq_cuda = torch.zeros((), dtype=torch.int32)
    worker._output_verify_lens = None
    return worker


def _draft_model():
    def forward(main_hidden, bonus, start_pos, **kwargs):
        batch = main_hidden.shape[0]
        confidence = torch.full((batch, 3), -3.0) if kwargs["return_confidence"] else None
        return None, confidence, torch.zeros(batch, 3, 4)

    return SimpleNamespace(
        block_size=3,
        forward_batched=Mock(side_effect=forward),
        write_context_windows_batched=Mock(),
    )


def _draft(
    worker,
    model,
    captured,
    *,
    num_contexts=0,
    gen_start=0,
    lengths=None,
    offsets=None,
    accepted_counts=None,
    positions=None,
):
    num_gens = len(accepted_counts) if accepted_counts is not None else 2
    accepted_counts = [1] * num_gens if accepted_counts is None else accepted_counts
    metadata = SimpleNamespace(
        qo_indptr=None if offsets is None else torch.tensor(offsets, dtype=torch.int32),
        verify_lens=None if lengths is None else torch.tensor(lengths, dtype=torch.int32),
        get_hidden_states=lambda _num_tokens: captured,
    )
    accepted = torch.zeros(num_contexts + num_gens, 4, dtype=torch.int32)
    counts = torch.tensor([1] * num_contexts + accepted_counts, dtype=torch.int32)
    positions = torch.arange(len(captured)) if positions is None else positions
    return worker._draft_gen_block_batched(
        model,
        metadata,
        SimpleNamespace(num_ctx_tokens=gen_start),
        accepted,
        counts,
        num_contexts,
        num_contexts + num_gens,
        len(captured),
        positions,
    )


@pytest.mark.parametrize("return_confidence", [False, True])
def test_mixed_ragged_draft_uses_generation_offsets_for_hidden_and_positions(return_confidence):
    worker = _worker(3)
    worker.return_confidence = return_confidence
    model = _draft_model()
    captured = torch.arange(9).float().unsqueeze(1).repeat(1, 2)
    positions = torch.tensor([0, 1, 2, 100, 101, 102, 103, 200, 201])

    _draft(
        worker,
        model,
        captured,
        num_contexts=1,
        gen_start=3,
        lengths=[4, 2],
        offsets=[0, 4, 6],
        accepted_counts=[3, 2],
        positions=positions,
    )

    args = model.forward_batched.call_args.args
    assert args[0][:, 0].tolist() == [5.0, 8.0]
    assert args[2].tolist() == [103, 202]
    backfill = model.write_context_windows_batched.call_args.args
    assert backfill[0][:, :, 0][backfill[3]].tolist() == [3.0, 4.0, 7.0]
    assert backfill[3].tolist() == [[True, True, False], [True, False, False]]


@pytest.mark.parametrize("offset, accepted_count", [(0, 3), (-1, 1)])
def test_real_bonus_row_out_of_range_is_not_clamped_to_another_request(offset, accepted_count):
    worker = _worker(1)
    with pytest.raises(IndexError):
        _draft(
            worker,
            _draft_model(),
            torch.zeros(2, 2),
            lengths=[3],
            offsets=[offset, offset + 3],
            accepted_counts=[accepted_count],
        )


def test_padding_offsets_do_not_index_past_captured_hidden_or_positions():
    worker = _worker(2)
    worker._batch_to_slot[1] = worker._scratch_slot
    model = _draft_model()

    _draft(
        worker,
        model,
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        lengths=[2, 2],
        offsets=[0, 3, 5],
        accepted_counts=[2, 2],
    )

    assert model.forward_batched.call_args.args[0][0].tolist() == [2.0, 2.0]
    assert worker._confidence_logits[0].tolist() == [-3.0, -3.0, -3.0]
    assert worker._confidence_logits[worker._scratch_slot + 1].tolist() == [30.0, 30.0, 30.0]


def test_draft_stamp_matches_previous_sequence_after_next_step_bump():
    worker = _worker(2)
    assert worker.bump_draft_seq() == 0
    _draft(worker, _draft_model(), torch.zeros(8, 2))
    assert worker.confidence_stamp_buffer()[:2].tolist() == [1, 1]
    assert worker.bump_draft_seq() == 1
    assert worker.verified_draft_seq_cuda().item() == 1
    assert torch.equal(
        worker.confidence_stamp_buffer()[:2], worker.verified_draft_seq_cuda().expand(2)
    )
    assert worker._confidence_stamp[worker._scratch_slot + 1].item() == 0


def test_warmup_restores_confidence_logits_and_their_matching_stamps(monkeypatch):
    worker = _worker(1)
    worker.guided_decoder = None
    worker._lazy_init = Mock()
    worker._execute_guided_decoder_if_present = Mock()
    accepted = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    worker.sample_and_accept_draft_tokens = Mock(return_value=(accepted, counts))
    worker.sample_draft_tokens = Mock(return_value=torch.zeros(1, 3, dtype=torch.int32))
    worker.write_context_onehot_draft_probs = Mock()
    worker._prepare_next_new_tokens = Mock(return_value=accepted)
    worker._confidence_logits[0] = torch.tensor([4.0, 5.0, 6.0])
    worker._confidence_stamp[0] = 7
    worker._draft_seq_cuda.fill_(8)
    saved_logits = worker._confidence_logits.clone()
    saved_stamps = worker._confidence_stamp.clone()
    saved_addresses = (worker._confidence_logits.data_ptr(), worker._confidence_stamp.data_ptr())
    metadata = SimpleNamespace(
        is_cuda_graph=True,
        qo_indptr=None,
        verify_lens=None,
        get_hidden_states=lambda _num_tokens: torch.zeros(4, 2),
        draft_probs_last_dim=4,
        batch_indices_cuda=torch.tensor([0]),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    worker._forward_impl(
        torch.tensor([1, 2, 3, 4]),
        torch.arange(4),
        None,
        torch.zeros(4, 4),
        SimpleNamespace(num_seqs=1, num_contexts=0, num_ctx_tokens=0),
        metadata,
        _draft_model(),
    )

    assert torch.equal(worker._confidence_logits, saved_logits)
    assert torch.equal(worker._confidence_stamp, saved_stamps)
    assert saved_addresses == (
        worker._confidence_logits.data_ptr(),
        worker._confidence_stamp.data_ptr(),
    )


def test_configured_sts_is_loaded_on_the_resolved_head_and_planner(tmp_path):
    worker = _worker(1)
    path = tmp_path / "sts.json"
    path.write_text(json.dumps({"sts_temperatures": [0.5, 1.0, 2.0]}), encoding="utf-8")
    worker.spec_config.confidence_sts_path = str(path)
    worker.spec_config.confidence_sps_table_path = None
    worker.spec_config.enable_fused_confidence_scheduler = False
    head = DSparkConfidenceHead(hidden_size=2, block_size=3)
    model = SimpleNamespace(
        dspark_model=SimpleNamespace(mtp_layers=[SimpleNamespace(confidence_head=head)])
    )
    temperatures = head.sts_temperatures
    raw = torch.tensor([[1.0, 2.0, 3.0]])

    worker._build_verify_planner(model, block_size=3)

    assert head.sts_temperatures is temperatures
    assert torch.equal(temperatures, torch.tensor([0.5, 1.0, 2.0]))
    assert worker.verify_planner.apply_calibration.__self__ is head
    assert torch.allclose(
        worker.verify_planner.apply_calibration(raw), torch.sigmoid(raw / temperatures)
    )
    assert worker.verify_planner.max_verify_len == 3
    assert worker.verify_planner.cost_table is None


def test_configured_sts_without_a_confidence_head_is_rejected_before_loading(tmp_path):
    worker = _worker(1)
    worker.spec_config.confidence_sts_path = str(tmp_path / "must-not-be-opened.json")
    worker.spec_config.confidence_sps_table_path = None
    worker.spec_config.enable_fused_confidence_scheduler = False

    with pytest.raises(
        ValueError, match="confidence_sts_path is set but no confidence head was found"
    ):
        worker._build_verify_planner(SimpleNamespace(), block_size=3)

    assert not hasattr(worker, "verify_planner")


@pytest.mark.parametrize("bias", [False, True])
def test_embedded_checkpoint_load_preserves_optional_confidence_bias(monkeypatch, bias):
    draft = modeling_dspark.DSv4DSparkDraftModel.__new__(modeling_dspark.DSv4DSparkDraftModel)
    nn.Module.__init__(draft)
    stage = nn.Module()
    stage.confidence_head = DSparkConfidenceHead(hidden_size=2, block_size=2)
    draft.mtp_layers = nn.ModuleList([stage])
    draft.num_stages = 1
    draft.post_load_weights = Mock()
    draft._use_bf16_linear = False
    wrapper = modeling_dspark.DSv4DSparkForCausalLM.__new__(modeling_dspark.DSv4DSparkForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.dspark_model = draft
    wrapper.num_stages = 1
    checkpoint = {"proj.weight": torch.tensor([[2.0, 3.0]])}
    if bias:
        checkpoint["proj.bias"] = torch.tensor([4.0])
    prefix = "mtp_layers.0.confidence_head."
    remapped = {prefix + name: value for name, value in checkpoint.items()}
    expected_weights = {name: value.clone() for name, value in remapped.items()}
    monkeypatch.setattr(modeling_dspark, "remap_dspark_draft_keys", Mock(return_value=remapped))

    def load_received(weights: dict[str, torch.Tensor]) -> None:
        assert set(weights) == set(expected_weights)
        for name, value in expected_weights.items():
            assert torch.equal(weights[name], value)
        head_weights = {name.removeprefix(prefix): value for name, value in weights.items()}
        stage.confidence_head.load_weights([head_weights])

    loader = SimpleNamespace(load_weights=Mock(side_effect=load_received))
    monkeypatch.setattr(modeling_dspark, "DeepseekV4WeightLoader", lambda _draft: loader)
    weight_address = stage.confidence_head.proj.weight.data_ptr()
    rng = torch.get_rng_state()

    wrapper.load_weights({})

    loader.load_weights.assert_called_once()
    assert stage.confidence_head.proj.weight.data_ptr() == weight_address
    assert torch.equal(torch.get_rng_state(), rng)
    assert (stage.confidence_head.proj.bias is not None) is bias
    expected = 5.0 + (4.0 if bias else 0.0)
    assert stage.confidence_head(torch.ones(1, 2)).item() == expected
