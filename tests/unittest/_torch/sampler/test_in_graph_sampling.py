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

"""Tier selection for in-graph sampling (``enable_fast_sampler``).

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

These tests touch no device state: tier selection only inspects requests.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.sampler.seed_manager import _SeedManager
from tensorrt_llm.sampling_params import SamplingParams

# Params that are expressible as per-row (temperature, top_k, top_p) and so
# qualify for the in-graph tier on their own.
FAST_PARAMS = dict(temperature=0.8, top_k=50, top_p=0.9)


@dataclass
class _Request:
    """Exposes exactly what the tier predicates and strategy resolution read."""

    sampling_config: SamplingConfig
    py_request_id: int = 0
    py_seq_slot: int | None = 0
    is_dummy: bool = False
    py_beam_width: int = 1
    py_sampling_strategy: Any = None
    is_context_init_state: bool = False
    py_return_log_probs: bool = False
    py_min_length: int = 0
    py_stop_words_list: Any = None
    py_bad_words: Any = None
    py_no_repeat_ngram_size: Any = None
    py_is_draft: bool = False
    py_logits_post_processors: Any = None
    guided_decoding_params: Any = None
    _py_embedding_bias_1d: Any = None

    def get_beam_width_by_iter(self, for_next_iteration: bool = False) -> int:
        return self.sampling_config.beam_width


def make_request(sampling: dict[str, Any] | None = None, **attrs: Any) -> LlmRequest:
    params = SamplingParams(**(sampling if sampling is not None else FAST_PARAMS))
    request = _Request(sampling_config=SamplingConfig(params._get_sampling_config()), **attrs)
    return cast(LlmRequest, request)


class _Sampler(TorchSampler):
    """TorchSampler with __init__ bypassed.

    Tier selection reads only the two attributes set below, so the real
    constructor -- which allocates device buffers -- is not needed.
    """

    def __init__(self) -> None:  # noqa: D107  (deliberately skips super().__init__)
        self.max_beam_width = 1
        # Declared by the real __init__, which also allocates device buffers.
        self._in_graph_dest_indices = None
        self._in_graph_staged = False
        self._in_graph_rows = 0
        self._in_graph_live_rows = 0
        self._in_graph_request_ids = []
        self._current_sample_type = SampleType.FULL
        self._fast_temperatures = None
        self._fast_top_ks = None
        self._fast_top_ps = None
        self._fast_seeds = None
        self._fast_offsets = None
        self._fast_num_rows = 0


@pytest.fixture
def sampler() -> TorchSampler:
    return _Sampler()


class TestTierSelection:
    def test_plain_temperature_top_k_top_p_is_fast(self, sampler):
        assert sampler.get_sample_type([make_request()]) is SampleType.FAST

    @pytest.mark.parametrize(
        "sampling",
        [
            pytest.param(dict(temperature=0.0), id="greedy"),
            pytest.param(dict(temperature=0.8, min_p=0.05), id="min_p"),
        ],
    )
    def test_sampling_params_outside_the_tier_are_full(self, sampler, sampling):
        assert sampler.get_sample_type([make_request(sampling)]) is SampleType.FULL

    @pytest.mark.parametrize(
        "attrs",
        [
            # Rewrite the logits before sampling; the in-graph path reads raw ones.
            pytest.param(dict(py_min_length=50), id="min_length"),
            pytest.param(dict(py_bad_words=[[7]]), id="bad_words"),
            pytest.param(dict(py_no_repeat_ngram_size=3), id="no_repeat_ngram"),
            pytest.param(dict(_py_embedding_bias_1d=object()), id="embedding_bias"),
            # Consume the logits after the forward returns.
            pytest.param(dict(guided_decoding_params=object()), id="guided_decoding"),
            pytest.param(dict(py_logits_post_processors=[object()]), id="logits_processor"),
            # Shares this sampler but registers no hooks.
            pytest.param(dict(py_is_draft=True), id="draft"),
            # Needs outputs the in-graph path does not produce.
            pytest.param(dict(py_return_log_probs=True), id="logprobs"),
        ],
    )
    def test_features_the_graph_cannot_serve_are_full(self, sampler, attrs):
        assert sampler.get_sample_type([make_request(**attrs)]) is SampleType.FULL

    def test_one_disqualifying_request_demotes_the_whole_batch(self, sampler):
        # A graph is replayed for the batch as a whole, so the tier has to be
        # the least capable request's.
        batch = [make_request(), make_request(py_min_length=50), make_request()]
        assert sampler.get_sample_type(batch) is SampleType.FULL

    def test_stop_words_stay_fast(self, sampler):
        # Stop words only decide when a request finishes, which still runs after
        # the forward, so they do not disqualify the tier.
        assert (
            sampler.get_sample_type([make_request(py_stop_words_list=([[13]], [1]))])
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
        sampler = _Sampler()
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

    def test_stages_destinations_for_the_batch(self, sampler):
        batch = self._batch(
            make_request(py_request_id=1, py_seq_slot=3),
            make_request(py_request_id=2, py_seq_slot=5),
        )
        sampler.stage_in_graph_sampling(batch, SampleType.FAST)

        assert sampler._in_graph_staged
        assert sampler._in_graph_request_ids == [1, 2]
        assert sampler._in_graph_rows == 2
        assert sampler._in_graph_live_rows == 2
        assert sampler._in_graph_dest_indices[:2].tolist() == [3, 5]

    def test_full_stages_nothing(self, sampler):
        batch = self._batch(make_request(py_request_id=1, py_seq_slot=0))
        sampler.stage_in_graph_sampling(batch, SampleType.FULL)
        assert not sampler._in_graph_staged

    def test_restaging_reuses_the_same_buffer(self, sampler):
        # The captured graph scatters through a fixed address, so a second
        # staging must write through the first allocation rather than replace it.
        first = self._batch(make_request(py_request_id=1, py_seq_slot=2))
        sampler.stage_in_graph_sampling(first, SampleType.FAST)
        buffer = sampler._in_graph_dest_indices

        second = self._batch(
            make_request(py_request_id=7, py_seq_slot=6),
            make_request(py_request_id=8, py_seq_slot=1),
        )
        sampler.stage_in_graph_sampling(second, SampleType.FAST)

        assert sampler._in_graph_dest_indices is buffer
        assert sampler._in_graph_dest_indices[:2].tolist() == [6, 1]
        assert sampler._in_graph_request_ids == [7, 8]

    def test_staging_a_full_batch_clears_previous_state(self, sampler):
        # Otherwise a FULL step would be sampled against the previous batch's
        # destinations, writing its tokens into another request's slot.
        sampler.stage_in_graph_sampling(
            self._batch(make_request(py_request_id=1, py_seq_slot=2)), SampleType.FAST
        )
        sampler.stage_in_graph_sampling(
            self._batch(make_request(py_request_id=9, py_seq_slot=4)), SampleType.FULL
        )
        assert not sampler._in_graph_staged
        assert sampler._in_graph_request_ids == []

    def test_padding_dummies_do_not_take_a_live_slot(self, sampler):
        # pad_batch appends one shared request object, so every padded row
        # carries the same (unset) slot; they must not land on a live request's.
        live = make_request(py_request_id=1, py_seq_slot=2)
        dummy = make_request(py_request_id=99, py_seq_slot=None, is_dummy=True)
        sampler.stage_in_graph_sampling(self._batch(live, dummy, dummy), SampleType.FAST)

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
        # Staging must fall back to neutral filters rather than raising.
        live = make_request(py_request_id=1, py_seq_slot=2)
        dummy = make_request(
            sampling=dict(temperature=0.0), py_request_id=99, py_seq_slot=None, is_dummy=True
        )
        sampler.stage_in_graph_sampling(self._batch(live, dummy), SampleType.FAST)
        assert sampler._in_graph_staged
        assert sampler._in_graph_live_rows == 1

    def test_padding_rows_do_not_advance_another_slot_rng(self, sampler):
        # Padding dummies borrow a slot that may belong to an active request the
        # batch simply did not schedule. Advancing its Philox offset would make
        # that request's stream depend on concurrent load, which is exactly what
        # _SeedManager exists to prevent.
        live = make_request(py_request_id=1, py_seq_slot=0)
        dummy = make_request(py_request_id=99, py_seq_slot=None, is_dummy=True)
        before = sampler._seed_manager._offsets.clone()

        sampler.stage_in_graph_sampling(self._batch(live, dummy, dummy), SampleType.FAST)

        after = sampler._seed_manager._offsets
        moved = [s for s in range(self.MAX_SEQUENCES) if after[s] != before[s]]
        assert moved == [0], f"only the live slot may advance, but {moved} did"
