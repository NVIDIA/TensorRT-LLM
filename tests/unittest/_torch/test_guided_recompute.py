# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Guided output must continue across native request recomputation."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.guided_decoder import (
    CapturableGuidedDecoder,
    GuidedDecoder,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    SamplingConfig,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig


def _make_request(request_id: int, params: GuidedDecodingParams | None = None) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=12,
        input_tokens=[4],
        sampling_config=SamplingConfig(1),
        is_streaming=True,
        end_id=5,
        pad_id=5,
        exclude_input_from_output=True,
        guided_decoding_params=params,
    )


def _schedule(request: LlmRequest, slots: SeqSlotManager) -> ScheduledRequests:
    batch = ScheduledRequests()
    if request.is_context_init_state:
        batch.append_context_request(request)
    else:
        batch.append_generation_request(request)
    slots.prepare_resources(batch)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("mode", ["eager", "capture", "drafts"])
@pytest.mark.parametrize("move_slot", [False, True])
@pytest.mark.parametrize(
    "transition,pause_points",
    [
        ("none", ()),
        ("v1", (0,)),
        ("v2", (0,)),
        ("v1", (1,)),
        ("v2", (1,)),
        ("v1", (2,)),
        ("v2", (2,)),
        ("v2", (1, 2)),
    ],
)
def test_guided_output_continues_across_recompute(
    transition: str, pause_points: tuple[int, ...], move_slot: bool, mode: str
) -> None:
    vocabulary = list("abcdx") + ["<eos>"] + [f"Z{i}" for i in range(26)]
    config = GuidedDecodingConfig(encoded_vocab=vocabulary, stop_token_ids=[5])
    params = GuidedDecodingParams(GuidedDecodingParams.GuideType.REGEX, "abcd")
    decoder_type = CapturableGuidedDecoder if mode == "capture" else GuidedDecoder
    decoder = decoder_type(
        config,
        max_num_sequences=2,
        vocab_size_padded=32,
        max_num_draft_tokens=2 if mode == "drafts" else 0,
    )
    logits = torch.zeros((3 if mode == "drafts" else 1, 32), device="cuda")
    graph = None
    slots = SeqSlotManager(2 if move_slot else 1)
    request = _make_request(41, params)
    _schedule(request, slots)
    blocker = None
    if move_slot:
        blocker = _make_request(42)
        _schedule(blocker, slots)

    streamed = []
    for generated in range(4):
        old_slot = request.py_seq_slot
        if generated in pause_points:
            slots.free_resources(request)
            if transition == "v1":
                request.pause(128)
            else:
                request.reset_for_recompute(128)
            assert request.request_id == 41 and request.orig_prompt_len == 1
            assert request.prompt_len == 1 + generated
            assert request.get_tokens(0) == [4] + streamed
            assert request.seq_slot is None and request.is_context_init_state
            if move_slot:
                # The new request occupies the only free slot (the victim's old
                # slot); release the other slot so replay must move there.
                replacement = _make_request(100 + generated)
                _schedule(replacement, slots)
                assert replacement.py_seq_slot == old_slot
                slots.free_resources(blocker)
                blocker = replacement

        batch = _schedule(request, slots)
        if generated in pause_points:
            assert (request.py_seq_slot != old_slot) == move_slot
        request.py_num_accepted_draft_tokens = 0
        request.py_draft_tokens = (
            list(range(generated, min(generated + 2, 4)))
            if mode == "drafts" and request.is_generation_in_progress_state
            else []
        )
        decoder.add_batch(batch)
        logits.zero_()
        logits[0, :4] = torch.tensor([10.0, 9.0, 8.0, 7.0], device="cuda")
        if mode == "capture":
            if graph is None:
                decoder.token_event.record()
                decoder.execute(logits)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    decoder.token_event.record()
                    decoder.execute(logits)
            graph.replay()
        else:
            assert decoder.execute(logits) == [], "Guidance failed before output validation"
            if mode == "drafts":
                decoder.rollback_rejected_tokens()
        token = int(logits[0].argmax().item())
        request.add_new_token(token, 0)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        request.py_decoding_iter += 1
        request.py_batch_idx = 0
        result = request.create_result()
        assert result is not None, "Native streaming result was lost"
        streamed.extend(result.output_token_ids[0])
        assert len(streamed) == generated + 1, "Pause duplicated or suppressed output"
        actual = "".join(vocabulary[tid] for tid in streamed)
        # Independent singleton-language oracle: it does not use xgrammar or
        # the production mask. Every streamed prefix must extend the same word.
        assert actual == "abcd"[: generated + 1], (
            f"transition={transition}, pause_points={pause_points}, "
            f"old_slot={old_slot}, slot={request.py_seq_slot}, output={actual!r}, "
            f"orig_prompt_len={request.orig_prompt_len}, prompt_len={request.prompt_len}, "
            f"py_orig_prompt_len={request.py_orig_prompt_len}"
        )
    slots.free_resources(request)
    if blocker is not None:
        slots.free_resources(blocker)
    slots.slot_manager.shutdown()
