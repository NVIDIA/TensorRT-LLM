# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""First-token EOS suppression for linear EAGLE3, including graph replay."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import (
    _make_single_token_context_graph_batch,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelSpecMetadata, Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm._torch.speculative.utils import get_spec_decoder

def _request(
    min_tokens: int | None,
    *,
    eos_id: int | None = 2,
    generated: int = 0,
    generation: bool = False,
    slot: int | None = 0,
    greedy: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        state=(
            LlmRequestState.GENERATION_IN_PROGRESS if generation else LlmRequestState.CONTEXT_INIT
        ),
        py_min_length=min_tokens,
        py_original_end_id=eos_id,
        py_end_id=-1,  # Disaggregated context may disable its local EOS termination.
        orig_prompt_len=8,
        py_orig_prompt_len=8 + generated,  # Recompute folds generated tokens into this field.
        get_num_tokens=Mock(return_value=8 + generated),
        py_seq_slot=slot,
        is_context_init_state=not generation,
        py_beam_width=1,
        get_beam_width_by_iter=Mock(return_value=1),
        py_return_context_logits=False,
        py_return_generation_logits=False,
        py_return_log_probs=False,
        sampling_config=SimpleNamespace(
            temperature=0.0 if greedy else 1.0,
            top_k=None if greedy else 2,
            top_p=1.0,
            min_p=None,
            top_p_decay=None,
            top_p_min=None,
            top_p_reset_ids=None,
            beam_width=1,
            beam_width_array=None,
            length_penalty=None,
            beam_search_diversity_rate=None,
            early_stopping=None,
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            seed=17,
        ),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("min_tokens", [None, 0, 1, 2])
def test_min_length_admission_is_explicit_and_bounded(
    supported: bool, min_tokens: int | None
) -> None:
    sampler = object.__new__(SpecSampler)
    sampler._min_length_one_supported = supported
    sampler._enable_penalty = False
    sampler._penalty_supported = True
    sampler._fused_sampling = False
    request = _request(min_tokens)
    if min_tokens and (min_tokens != 1 or not supported):
        with pytest.raises(ValueError, match="min_length"):
            sampler.validate_request(request)
    else:
        sampler.validate_request(request)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mode,tree,expected",
    [
        (SpeculativeDecodingMode.EAGLE3_ONE_MODEL, "linear", True),
        (SpeculativeDecodingMode.EAGLE3_ONE_MODEL, "dynamic", False),
        (SpeculativeDecodingMode.EAGLE3_ONE_MODEL, "topk_one", True),
        (SpeculativeDecodingMode.MTP, "linear", False),
        (SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL, "linear", False),
        (SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL, "linear", False),
    ],
)
def test_only_linear_eagle3_opts_into_min_length_one(
    mode: SpeculativeDecodingMode, tree: str, expected: bool
) -> None:
    config = SimpleNamespace(
        spec_dec_mode=mode,
        use_dynamic_tree=tree in ("dynamic", "topk_one"),
        dynamic_tree_max_topK=1 if tree == "topk_one" else 2,
        enable_penalty=False,
        advanced_sampling_mode=SimpleNamespace(is_fused=True),
    )
    with patch("tensorrt_llm._torch.speculative.utils.SpecSampler") as constructor:
        get_spec_decoder(SimpleNamespace(max_total_draft_tokens=3), config)
    assert constructor.call_args.kwargs["min_length_one_supported"] is expected


@pytest.mark.cpu_only
def test_metadata_uses_original_eos_and_true_generated_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only bypass the unrelated CUDA sampling buffers; run the real EOS population.
    metadata = object.__new__(Eagle3OneModelSpecMetadata)
    metadata.first_token_eos_ids = torch.full((8,), 99, dtype=torch.long)
    monkeypatch.setattr(SpecMetadata, "populate_sampling_params_for_one_model", Mock())
    monkeypatch.setattr("tensorrt_llm._torch.speculative.eagle3.prefer_pinned", lambda: False)
    requests = [
        _request(1, eos_id=2),
        _request(0, eos_id=3),
        _request(1, eos_id=0),
        _request(1, eos_id=None),
        _request(1, generated=1),  # Paused/recomputed request, not a fresh first token.
        _request(0, slot=None),  # CUDA-graph padding/dummy request.
        _request(1, generated=1, generation=True),  # Token already imported from prefill.
    ]
    metadata.populate_sampling_params_for_one_model(requests)
    assert metadata.first_token_eos_ids.tolist() == [2, -1, 0, -1, -1, -1, 99, 99]
    requests[-1].get_num_tokens.assert_not_called()

    # A batch position/slot previously carrying a ban must not leak to a new request.
    metadata.populate_sampling_params_for_one_model([_request(0), _request(1, eos_id=4)])
    assert metadata.first_token_eos_ids[:2].tolist() == [-1, 4]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "min_tokens,generated,preserve,can_promote",
    [(1, 0, True, False), (0, 0, True, True), (1, 1, True, True), (1, 0, False, True)],
)
def test_first_token_context_cannot_bypass_mask_through_decode_graph_promotion(
    min_tokens: int, generated: int, preserve: bool, can_promote: bool
) -> None:
    request = _request(min_tokens, generated=generated)
    request.py_request_id = 17
    request.context_chunk_size = 1
    request.context_remaining_length = 1
    request.context_current_position = request.py_orig_prompt_len - 1
    request.py_prompt_len = request.py_orig_prompt_len
    request.py_draft_tokens = []
    request.py_is_first_draft = False
    request.is_context_only_request = False
    request.is_generation_only_request = False
    request.py_disaggregated_params = None
    request.py_mm_encoder_event = None
    request.py_multimodal_data = None
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    if preserve:
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch, preserve_eagle3_min_length_context=True
        )
    else:
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)
    if can_promote:
        assert graph_batch.generation_requests == [request]
        assert promoted_ids == frozenset({17})
    else:
        assert graph_batch is batch
        assert not promoted_ids
    assert batch.context_requests_last_chunk == [request]


@pytest.mark.cpu_only
def test_mask_changes_only_eligible_context_eos_without_mutating_model_logits() -> None:
    logits = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    original = logits.clone()
    metadata = SimpleNamespace(first_token_eos_ids=torch.tensor([2, -1, 0, 9]))
    masked = Eagle3OneModelWorker._apply_min_length_one(logits, 4, metadata)
    expected = original.clone()
    expected[0, 2] = -torch.inf
    expected[2, 0] = -torch.inf
    torch.testing.assert_close(masked, expected)
    torch.testing.assert_close(logits, original)
    assert masked.data_ptr() != logits.data_ptr()
    assert Eagle3OneModelWorker._apply_min_length_one(logits, 0, metadata) is logits
    metadata.first_token_eos_ids = None
    assert Eagle3OneModelWorker._apply_min_length_one(logits, 4, metadata) is logits
