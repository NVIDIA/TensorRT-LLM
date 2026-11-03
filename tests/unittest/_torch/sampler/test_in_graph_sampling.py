# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tier selection for in-graph sampling (``enable_in_graph_sampling``).

A batch may only be sampled inside the model's forward CUDA graph when nothing
it depends on happens outside that graph. Two classes of request must therefore
stay on the eager FULL tier, and both regress silently -- the request still
produces tokens, just ones that ignore what the user asked for:

* Features that rewrite the logits *before* sampling. ``min_length`` is the
  subtle one: it is enforced by banning ``end_id`` out of the logits until the
  request is long enough (``TokenBanHandler._add_min_length_bans``), not by a
  finish-reason check, so sampling in-graph would skip the ban and let EOS
  through early.
* Features that read or rewrite the logits *after* the forward returns --
  guided decoding's grammar bitmask, and the user's own logits processors run
  by ``_execute_logit_post_processors``. The in-graph hook runs at the tail of
  ``_forward_step``, before both, so their output would be discarded.

Tier selection touches no device state: it only inspects requests and the
sampling modes resolved for their slots at admission.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LogprobMode, SamplingConfig
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.sampler.sampling_resolution import SlotSamplingParams
from tensorrt_llm._torch.pyexecutor.sampler.seed_manager import _SeedManager
from tensorrt_llm.sampling_params import SamplingParams

# Params that are expressible as per-row (temperature, top_k, top_p) and so
# qualify for the in-graph tier on their own.
FAST_PARAMS = dict(temperature=0.8, top_k=50, top_p=0.9)


@dataclass
class _Request:
    """Exposes exactly what the tier predicates and admission read."""

    sampling_config: SamplingConfig
    py_request_id: int = 0
    py_seq_slot: int | None = 0
    is_dummy: bool = False
    py_beam_width: int = 1
    is_context_init_state: bool = False
    py_return_log_probs: bool = False
    py_logprobs_mode: LogprobMode = LogprobMode.RAW
    py_min_length: int = 0
    py_stop_words_list: Any = None
    py_bad_words: Any = None
    py_no_repeat_ngram_size: Any = None
    py_logits_post_processors: Any = None
    py_guided_decoding_params: Any = None
    py_embedding_bias: Any = None
    py_draft_tokens: Any = None

    def get_beam_width_by_iter(self, for_next_iteration: bool = False) -> int:
        return self.sampling_config.beam_width


def make_request(sampling: dict[str, Any] | None = None, **attrs: Any) -> LlmRequest:
    params = SamplingParams(**(sampling if sampling is not None else FAST_PARAMS))
    request = _Request(sampling_config=SamplingConfig(params._get_sampling_config()), **attrs)
    return cast(LlmRequest, request)


class _Sampler(TorchSampler):
    """TorchSampler with __init__ bypassed.

    Tier selection reads only the attributes set below, so the real
    constructor -- which allocates device buffers -- is not needed.
    """

    NUM_SLOTS = 16

    def __init__(self, device: str = "cpu") -> None:  # noqa: D107  (skips super().__init__)
        self.max_beam_width = 1
        self._slot_params = SlotSamplingParams(self.NUM_SLOTS, device=device)
        # No request in these tests uses top-p decay.
        self._top_p_decay = SimpleNamespace(uses_decay=lambda slot: False)
        # Declared by the real __init__, which also allocates device buffers.
        self._in_graph_dest_indices = None
        self._in_graph_staged = False
        self._in_graph_rows = 0
        self._in_graph_live_rows = 0
        self._in_graph_request_ids = []
        self._current_sample_type = SampleType.FULL
        self._fast_params = None
        self._staged_batch = None

    def admit(self, *requests: LlmRequest) -> None:
        """Resolve the live requests' params into their slots, as admission does."""
        live = [r for r in requests if not r.is_dummy and r.py_seq_slot is not None]
        if live:
            slots = torch.tensor([r.py_seq_slot for r in live], dtype=torch.int64)
            self._slot_params.admit(live, slots.to(self._slot_params.floats.device))


def sample_type(sampler: _Sampler, requests: list[LlmRequest]) -> SampleType:
    sampler.admit(*requests)
    return sampler.get_sample_type(requests)


@pytest.fixture
def sampler() -> TorchSampler:
    return _Sampler()


class TestTierSelection:
    def test_plain_temperature_top_k_top_p_is_fast(self, sampler):
        assert sample_type(sampler, [make_request()]) is SampleType.FAST

    def test_min_p_is_fast(self, sampler):
        request = make_request(dict(temperature=0.8, min_p=0.05))
        assert sample_type(sampler, [request]) is SampleType.FAST

    def test_greedy_is_full(self, sampler):
        request = make_request(dict(temperature=0.0))
        assert sample_type(sampler, [request]) is SampleType.FULL

    @pytest.mark.parametrize(
        "attrs",
        [
            # Rewrite the logits before sampling; the in-graph path reads raw ones.
            pytest.param(dict(py_min_length=50), id="min_length"),
            pytest.param(dict(py_bad_words=[[7]]), id="bad_words"),
            pytest.param(dict(py_no_repeat_ngram_size=3), id="no_repeat_ngram"),
            pytest.param(dict(py_embedding_bias=object()), id="embedding_bias"),
            # Consume the logits after the forward returns.
            pytest.param(dict(py_guided_decoding_params=object()), id="guided_decoding"),
            pytest.param(dict(py_logits_post_processors=[object()]), id="logits_processor"),
            # Needs outputs the in-graph path does not produce.
            pytest.param(dict(py_return_log_probs=True), id="logprobs"),
        ],
    )
    def test_features_the_graph_cannot_serve_are_full(self, sampler, attrs):
        assert sample_type(sampler, [make_request(**attrs)]) is SampleType.FULL

    def test_one_disqualifying_request_demotes_the_whole_batch(self, sampler):
        # A graph is replayed for the batch as a whole, so the tier has to be
        # the least capable request's.
        batch = [make_request(), make_request(py_min_length=50), make_request()]
        assert sample_type(sampler, batch) is SampleType.FULL

    def test_per_request_verdict_is_computed_once(self, sampler):
        # Whether a request fits the fast tier depends on parameters fixed at
        # admission, so the batch loop must not redo the checks every step.
        request = make_request()
        assert sample_type(sampler, [request]) is SampleType.FAST
        assert request.py_fast_path_ok is True

        def fail(req):
            raise AssertionError("verdict recomputed")

        sampler._request_fast_path_ok = fail
        assert sampler.get_sample_type([request]) is SampleType.FAST

    def test_disqualified_request_keeps_its_verdict(self, sampler):
        request = make_request(py_min_length=50)
        assert sample_type(sampler, [request]) is SampleType.FULL
        assert request.py_fast_path_ok is False

    def test_live_request_after_padding_dummy_is_full(self, sampler):
        # Dummies must be a trailing run: the live rows are read back as the
        # leading prefix of the batch.
        live = make_request(py_request_id=1, py_seq_slot=1)
        dummy = make_request(py_request_id=2, py_seq_slot=None, is_dummy=True)
        sampler.admit(live)
        scheduled = SimpleNamespace(generation_requests=[dummy, live], num_context_requests=0)
        assert sampler.resolve_in_graph_sample_type(scheduled) is SampleType.FULL
        scheduled = SimpleNamespace(generation_requests=[live, dummy], num_context_requests=0)
        assert sampler.resolve_in_graph_sample_type(scheduled) is SampleType.FAST

    def test_stop_words_stay_fast(self, sampler):
        # Stop words only decide when a request finishes, which still runs after
        # the forward, so they do not disqualify the tier.
        assert (
            sample_type(sampler, [make_request(py_stop_words_list=([[13]], [1]))])
            is SampleType.FAST
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="staging writes CUDA buffers")
class TestStaging:
    """``stage_in_graph_sampling`` fills the buffers the in-graph step reads.

    A captured graph bakes in the addresses it scatters through, so the buffers
    must be allocated once and updated in place; and because staging and the
    in-graph step are separated by the whole forward, the staged state has to
    identify the batch it belongs to.
    """

    MAX_SEQUENCES = 8

    @pytest.fixture
    def sampler(self) -> TorchSampler:
        sampler = _Sampler(device="cuda")
        sampler.max_num_sequences = self.MAX_SEQUENCES
        sampler.dummy_slot_row = self.MAX_SEQUENCES
        sampler.max_tokens = 1
        sampler._seed_manager = _SeedManager(max_num_sequences=self.MAX_SEQUENCES, global_seed=0)
        sampler.store = SimpleNamespace(
            new_tokens=torch.zeros((1, self.MAX_SEQUENCES + 1, 1), dtype=torch.int32, device="cuda")
        )
        return sampler

    @staticmethod
    def _batch(*requests: LlmRequest) -> Any:
        return SimpleNamespace(generation_requests=list(requests), num_context_requests=0)

    @staticmethod
    def _stage(sampler: _Sampler, batch: Any, sample_type: SampleType) -> None:
        sampler.admit(*batch.generation_requests)
        sampler.stage_in_graph_sampling(batch, sample_type)

    def test_stages_destinations_for_the_batch(self, sampler):
        batch = self._batch(
            make_request(py_request_id=1, py_seq_slot=3),
            make_request(py_request_id=2, py_seq_slot=5),
        )
        self._stage(sampler, batch, SampleType.FAST)

        assert sampler._in_graph_staged
        assert sampler._in_graph_request_ids == [1, 2]
        assert sampler._in_graph_rows == 2
        assert sampler._in_graph_live_rows == 2
        assert sampler._in_graph_dest_indices[:2].tolist() == [3, 5]

    def test_stages_per_row_filters(self, sampler):
        batch = self._batch(
            make_request(py_request_id=1, py_seq_slot=3),
            make_request(
                dict(temperature=0.7, top_p=0.8, min_p=0.05), py_request_id=2, py_seq_slot=5
            ),
        )
        self._stage(sampler, batch, SampleType.FAST)

        params = sampler._fast_params
        assert params.temperature[:2].tolist() == pytest.approx([0.8, 0.7])
        assert params.top_k[:2].tolist() == [50, 0]
        assert params.top_p[:2].tolist() == pytest.approx([0.9, 0.8])
        assert params.min_p[:2].tolist() == pytest.approx([0.0, 0.05])

    def test_full_stages_nothing(self, sampler):
        batch = self._batch(make_request(py_request_id=1, py_seq_slot=0))
        self._stage(sampler, batch, SampleType.FULL)
        assert not sampler._in_graph_staged

    def test_restaging_reuses_the_same_buffer(self, sampler):
        # The captured graph scatters through a fixed address, so a second
        # staging must write through the first allocation rather than replace it.
        first = self._batch(make_request(py_request_id=1, py_seq_slot=2))
        self._stage(sampler, first, SampleType.FAST)
        buffer = sampler._in_graph_dest_indices

        second = self._batch(
            make_request(py_request_id=7, py_seq_slot=6),
            make_request(py_request_id=8, py_seq_slot=1),
        )
        self._stage(sampler, second, SampleType.FAST)

        assert sampler._in_graph_dest_indices is buffer
        assert sampler._in_graph_dest_indices[:2].tolist() == [6, 1]
        assert sampler._in_graph_request_ids == [7, 8]

    def test_repeated_batch_is_not_restaged_but_seeds_advance(self, sampler):
        # A steady decode batch restages the same requests every step; only the
        # Philox offsets move.
        batch = self._batch(
            make_request(py_request_id=1, py_seq_slot=3),
            make_request(py_request_id=2, py_seq_slot=5),
        )
        self._stage(sampler, batch, SampleType.FAST)
        first_offsets = sampler._fast_params.offset[:2].tolist()
        calls = []
        original = sampler.stage_fast_sampling_params
        sampler.stage_fast_sampling_params = lambda *a, **k: (calls.append(1), original(*a, **k))

        self._stage(sampler, batch, SampleType.FAST)

        assert calls == []
        assert sampler._in_graph_staged
        assert sampler._in_graph_request_ids == [1, 2]
        assert sampler._in_graph_dest_indices[:2].tolist() == [3, 5]
        assert sampler._fast_params.offset[:2].tolist() == [o + 2 for o in first_offsets]

    def test_staged_batch_records_what_the_readback_needs(self, sampler):
        live = [
            make_request(py_request_id=1, py_seq_slot=3),
            make_request(py_request_id=2, py_seq_slot=5),
        ]
        dummy = make_request(py_request_id=99, py_seq_slot=None, is_dummy=True)
        self._stage(sampler, self._batch(*live, dummy), SampleType.FAST)

        staged = sampler._staged_batch
        assert staged.requests == [*live, dummy]
        assert staged.live_requests == live
        assert staged.live_request_ids == [1, 2]
        assert staged.live_seq_slots_host.tolist() == [3, 5]
        assert staged.live_seq_slots_host.dtype == torch.int32
        assert staged.single_step_greedy

    def test_multi_token_stop_words_need_the_finish_reason_kernel(self, sampler):
        batch = self._batch(
            make_request(py_request_id=1, py_seq_slot=3),
            make_request(py_request_id=2, py_seq_slot=5, py_stop_words_list=[[13, 14]]),
        )
        self._stage(sampler, batch, SampleType.FAST)
        assert not sampler._staged_batch.single_step_greedy

    @pytest.mark.parametrize(
        "second",
        [
            pytest.param([(1, 3), (9, 5)], id="new_request_same_slot"),
            pytest.param([(1, 3), (2, 6)], id="request_moved_slot"),
            pytest.param([(1, 3)], id="request_left"),
        ],
    )
    def test_changed_batch_is_restaged(self, sampler, second):
        first = self._batch(
            make_request(py_request_id=1, py_seq_slot=3),
            make_request(py_request_id=2, py_seq_slot=5),
        )
        self._stage(sampler, first, SampleType.FAST)
        batch = self._batch(*[make_request(py_request_id=i, py_seq_slot=s) for i, s in second])
        self._stage(sampler, batch, SampleType.FAST)

        assert sampler._in_graph_rows == len(second)
        assert sampler._in_graph_request_ids == [i for i, _ in second]
        assert sampler._in_graph_dest_indices[: len(second)].tolist() == [s for _, s in second]

    def test_direct_param_staging_invalidates_the_repeat_shortcut(self, sampler):
        # stage_fast_sampling_params overwrites the buffers, so a later identical
        # batch has to write them again.
        batch = self._batch(make_request(py_request_id=1, py_seq_slot=3))
        self._stage(sampler, batch, SampleType.FAST)
        other = make_request(dict(temperature=0.5, top_p=0.7), py_request_id=4, py_seq_slot=6)
        sampler.admit(other)
        sampler.stage_fast_sampling_params([other], slots_per_row=[6], live_rows=1)
        assert sampler._fast_params.temperature[:1].tolist() == pytest.approx([0.5])

        self._stage(sampler, batch, SampleType.FAST)
        assert sampler._fast_params.temperature[:1].tolist() == pytest.approx([0.8])

    def test_staging_a_full_batch_clears_previous_state(self, sampler):
        # Otherwise a FULL step would be sampled against the previous batch's
        # destinations, writing its tokens into another request's slot.
        self._stage(
            sampler, self._batch(make_request(py_request_id=1, py_seq_slot=2)), SampleType.FAST
        )
        self._stage(
            sampler, self._batch(make_request(py_request_id=9, py_seq_slot=4)), SampleType.FULL
        )
        assert not sampler._in_graph_staged
        assert sampler._in_graph_request_ids == []

    def test_padding_dummies_do_not_take_a_live_slot(self, sampler):
        # pad_batch appends one shared request object, so every padded row
        # carries the same (unset) slot; they must not land on a live request's.
        live = make_request(py_request_id=1, py_seq_slot=2)
        dummy = make_request(py_request_id=99, py_seq_slot=None, is_dummy=True)
        self._stage(sampler, self._batch(live, dummy, dummy), SampleType.FAST)

        staged = sampler._in_graph_dest_indices[: sampler._in_graph_rows].tolist()
        assert sampler._in_graph_rows == 3
        assert sampler._in_graph_live_rows == 1
        assert staged[0] == 2
        # Padding rows go to the scratch row, which is past every real slot, so
        # they cannot be read back as another request's next input token.
        assert staged[1:] == [self.MAX_SEQUENCES, self.MAX_SEQUENCES]
        # Only the live prefix is reported back.
        assert sampler._in_graph_request_ids == [1]

    def test_paramless_padding_dummies_do_not_raise(self, sampler):
        # Runtime padding dummies are built without capture_sampling_params, so
        # they resolve to greedy -- which the fast-tier kernels cannot express.
        # Staging gives them neutral filters rather than raising.
        live = make_request(py_request_id=1, py_seq_slot=2)
        dummy = make_request(
            sampling=dict(temperature=0.0), py_request_id=99, py_seq_slot=None, is_dummy=True
        )
        self._stage(sampler, self._batch(live, dummy), SampleType.FAST)
        assert sampler._in_graph_staged
        assert sampler._in_graph_live_rows == 1

    def test_padding_rows_read_neutral_params(self, sampler):
        # A padding dummy may carry the slot of a request the batch did not
        # schedule; its row must not sample with that request's filters.
        sampler.admit(make_request(dict(temperature=0.5, top_k=3), py_request_id=2, py_seq_slot=4))
        live = make_request(py_request_id=1, py_seq_slot=2)
        dummy = make_request(py_request_id=99, py_seq_slot=4, is_dummy=True)
        self._stage(sampler, self._batch(live, dummy), SampleType.FAST)

        params = sampler._fast_params
        assert params.temperature[:2].tolist() == pytest.approx([0.8, 1.0])
        assert params.top_k[:2].tolist() == [50, 0]
        assert params.top_p[:2].tolist() == pytest.approx([0.9, 1.0])
        assert params.min_p[:2].tolist() == pytest.approx([0.0, 0.0])

    def test_padding_rows_do_not_advance_another_slot_rng(self, sampler):
        # Padding dummies borrow a slot that may belong to an active request the
        # batch simply did not schedule. Advancing its Philox offset would make
        # that request's stream depend on concurrent load, which is exactly what
        # _SeedManager exists to prevent.
        live = make_request(sampling={**FAST_PARAMS, "seed": 7}, py_request_id=1, py_seq_slot=0)
        dummy = make_request(py_request_id=99, py_seq_slot=None, is_dummy=True)
        before = sampler._seed_manager._offsets.clone()

        self._stage(sampler, self._batch(live, dummy, dummy), SampleType.FAST)

        after = sampler._seed_manager._offsets
        moved = [s for s in range(self.MAX_SEQUENCES) if after[s] != before[s]]
        assert moved == [0], f"only the live slot may advance, but {moved} did"
