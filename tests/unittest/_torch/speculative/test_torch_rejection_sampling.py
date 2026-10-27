import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from scipy.stats import entropy

from tensorrt_llm._torch.pyexecutor.handle_logits import HandleLogits
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, PyResult
from tensorrt_llm._torch.pyexecutor.sampler import (TorchSampler,
                                                    get_rejected_indices,
                                                    sample_rejected)
from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter


def test_get_rejected_indices():
    vocab_size = 500
    num_iter = 50000
    draft_probs = torch.rand(1, vocab_size)
    drop_idx = torch.topk(draft_probs[0], k=400, largest=False)[1]
    draft_probs[0, drop_idx] = 0.0
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    target_probs = torch.rand(2, vocab_size)
    drop_idx = torch.topk(target_probs[0], k=400, largest=False)[1]
    target_probs[0, drop_idx] = 0.0
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    generator = torch.Generator()
    sampled_tokens = []
    sampled_regular = []
    for _ in range(num_iter):
        draft_tokens = [
            cast(
                int,
                torch.multinomial(draft_probs,
                                  num_samples=1,
                                  generator=generator).item())
        ]
        rejected_indices = get_rejected_indices(draft_probs, target_probs,
                                                generator, draft_tokens)
        if rejected_indices.shape[0] == 0:
            sampled_tokens.append(draft_tokens[0])
        else:
            sampled_tokens.append(
                sample_rejected(draft_probs, target_probs, generator, 0))
        sampled_regular.append(
            torch.multinomial(target_probs[0],
                              num_samples=1,
                              generator=generator).item())
    bins = np.arange(vocab_size + 1) - 0.5  # Bins for histogram
    sampled_tokens, _ = np.histogram(sampled_tokens, bins=bins, density=True)
    sampled_regular, _ = np.histogram(sampled_regular, bins=bins, density=True)
    expected_prob = target_probs[0].squeeze().numpy()

    # KL Divergence check
    kl_divergence = entropy(expected_prob, sampled_tokens)
    kl_divergence_regular = entropy(expected_prob, sampled_regular)
    assert abs(kl_divergence - kl_divergence_regular) < 0.01


def test_get_rejected_indices_zero_target_prob_rejected():
    """Reject tokens outside target support even when draft probability is zero."""
    # Token 1 has p == 0 and q == 0 (0 / 0 = NaN in unguarded q / p);
    # Token 2 has p > 0 and q == 0. Both must always be rejected.
    draft_probs = torch.tensor([
        [0.5, 0.0, 0.5, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    target_probs = torch.tensor([
        [0.5, 0.0, 0.5, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    generator = torch.Generator()
    generator.manual_seed(42)

    # Draft token 0 at pos 0 (p=0.5, q=0.5 -> accepted),
    # Draft token 1 at pos 1 (p=0.0, q=0.0 -> must be rejected),
    # Draft token 2 at pos 2 (p=1.0, q=0.0 -> must be rejected).
    draft_tokens = [0, 1, 2]
    rejected_indices = get_rejected_indices(draft_probs, target_probs,
                                            generator, draft_tokens)
    rejected_list = rejected_indices.squeeze(-1).tolist()
    if isinstance(rejected_list, int):
        rejected_list = [rejected_list]
    assert 1 in rejected_list
    assert 2 in rejected_list
    assert rejected_list[0] == 1


def test_get_rejected_indices_mismatched_row_count_raises():
    """Reject missing or extra draft rows instead of silently accepting tokens."""
    vocab_size = 8
    draft_probs = torch.full((2, vocab_size), 1.0 / vocab_size)
    target_probs = torch.full((4, vocab_size), 1.0 / vocab_size)
    generator = torch.Generator()

    with pytest.raises(ValueError, match="draft_probs row count"):
        get_rejected_indices(draft_probs, target_probs, generator, [1, 2, 3])

    with pytest.raises(ValueError, match="draft_probs row count"):
        get_rejected_indices(draft_probs, target_probs, generator, [1])


@pytest.mark.parametrize("use_chunked_generation_logits", [False, True])
def test_process_decoded_tokens_accumulates_draft_logits_in_order(
        monkeypatch, use_chunked_generation_logits: bool):
    """Pair every draft token with its sampling row in either storage mode."""
    if not torch.cuda.is_available():
        orig_empty = torch.empty
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *args, **kwargs: orig_empty(
                *args, **{
                    **kwargs, "pin_memory": False
                }),
        )

    vocab_size = 8
    max_draft_len = 3
    target_prompt_len = 5

    py_result = PyResult(
        prompt_len=target_prompt_len + 1,
        max_new_tokens=max_draft_len,
        use_device_memory=False,
        streaming=False,
        return_log_probs=False,
        return_context_logits=False,
        return_generation_logits=True,
        exclude_last_generation_logits=False,
        use_chunked_generation_logits=use_chunked_generation_logits,
    )

    target_req = SimpleNamespace(
        py_request_id=7,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_draft_pages_allocated=max_draft_len,
        py_draft_tokens=[],
        py_draft_logits=None,
    )
    draft_req = SimpleNamespace(
        py_request_id=7,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_prompt_len=target_prompt_len + 1,
        context_current_position=0,
        context_chunk_size=3,
        py_return_context_logits=False,
        py_return_generation_logits=True,
        py_use_chunked_generation_logits=use_chunked_generation_logits,
        is_last_context_chunk=False,
        py_is_draft=True,
        py_result=py_result,
        _last_token=0,
        get_last_tokens=lambda beam: draft_req._last_token,
    )

    drafter = ModelDrafter.__new__(ModelDrafter)
    drafter.max_draft_len = max_draft_len
    drafter.max_total_draft_tokens = max_draft_len
    drafter.draft_tokens_accumulator = {}
    drafter.draft_logits_accumulator = {}
    drafter.req_id_to_old_request = {7: target_req}
    drafter.draft_seq_slot_manager = SimpleNamespace(
        free_resources=lambda req: None)
    drafter.update_requests = lambda state, resource_manager=None: None

    with torch.inference_mode():
        step_logits = [
            torch.arange(vocab_size, dtype=torch.float32) + 10.0 * step
            for step in range(max_draft_len)
        ]
    drafted_tokens = [3, 5, 2]

    # Non-final context chunk (end_compute=3 < py_prompt_len - 1=5) must not append logits.
    HandleLogits()(
        context_requests=[draft_req],
        generation_requests=[],
        logits=step_logits[0].unsqueeze(0),
        beam_width=1,
        num_context_logits_prefix_sum=[0, 1],
        is_generation_model=True,
    )
    assert draft_req.py_result.generation_logits is None

    # Final draft context chunk (end_compute=5 == py_prompt_len - 1) appends L_0.
    draft_req.context_chunk_size = target_prompt_len
    HandleLogits()(
        context_requests=[draft_req],
        generation_requests=[],
        logits=step_logits[0].unsqueeze(0),
        beam_width=1,
        num_context_logits_prefix_sum=[0, 1],
        is_generation_model=True,
    )

    draft_batch = SimpleNamespace(all_requests=lambda: [draft_req])

    # In _execute_draft_loop, step i+1's HandleLogits runs before
    # process_decoded_tokens(draft_position=i+1).
    for i in range(max_draft_len - 1):
        draft_req._last_token = drafted_tokens[i]
        HandleLogits()(
            context_requests=[],
            generation_requests=[draft_req],
            logits=step_logits[i + 1].unsqueeze(0),
            beam_width=1,
            num_context_logits_prefix_sum=[0],
            is_generation_model=True,
        )
        drafter.process_decoded_tokens(draft_batch, draft_position=i + 1)
        assert target_req.py_draft_logits is None

    # Final step in process_dynamic_draft_outputs (draft_position=max_draft_len)
    draft_req._last_token = drafted_tokens[max_draft_len - 1]
    final_state = SimpleNamespace(scheduled_requests=draft_batch)
    drafter.process_dynamic_draft_outputs(final_state)

    assert target_req.py_draft_tokens == drafted_tokens
    assert target_req.py_draft_logits is not None
    assert target_req.py_draft_logits.shape == (max_draft_len, vocab_size)
    for step in range(max_draft_len):
        assert torch.equal(target_req.py_draft_logits[step].cpu(),
                           step_logits[step])


@pytest.mark.parametrize("has_final_state", [False, True])
def test_prepare_draft_tokens_stacks_logits_once(monkeypatch, has_final_state):
    """Retain one stacked tensor per request across both draft-loop exits."""
    target_req = SimpleNamespace(py_draft_tokens=[], py_draft_logits=None)
    draft_batch = SimpleNamespace(all_requests=lambda: [])
    final_state = SimpleNamespace(scheduled_requests=draft_batch)
    rows = [torch.arange(8, dtype=torch.float32) + step for step in range(3)]

    drafter = ModelDrafter.__new__(ModelDrafter)
    drafter.max_draft_len = len(rows)
    drafter.draft_model_engine = object()
    drafter.use_static_draft_loop = False
    drafter.guided_decoder = None
    drafter.draft_tokens_accumulator = {7: [1, 2, 3]}
    drafter.draft_logits_accumulator = {7: rows}
    drafter.req_id_to_old_request = {7: target_req}
    drafter._setup_draft_batch_and_resources = lambda requests: draft_batch
    drafter.update_cur_draft_layer_idx = lambda *args: None
    drafter.forward_draft_model = lambda *args, **kwargs: {}
    drafter.sample_async = lambda *args: None
    drafter.update_request_states = lambda *args: None
    drafter._execute_draft_loop = lambda *args: (
        final_state if has_final_state else None)
    drafter.update_requests = lambda *args: None
    drafter.process_decoded_tokens = lambda *args: None

    stacked = []
    original_stack = torch.stack

    def track_stack(*args, **kwargs):
        """Record allocations while preserving the real tensor operation."""
        result = original_stack(*args, **kwargs)
        stacked.append(result)
        return result

    monkeypatch.setattr(torch, "stack", track_stack)
    drafter.prepare_draft_tokens(draft_batch, resource_manager=object())

    assert len(stacked) == 1
    assert target_req.py_draft_logits is stacked[0]
    assert torch.equal(target_req.py_draft_logits, original_stack(rows))
    assert target_req.py_draft_tokens == [1, 2, 3]
    assert drafter.draft_logits_accumulator == {}


def test_early_stop_draft_rejection_sampling(monkeypatch):
    """Verify only real drafts and emit the bonus after the last real draft."""
    if not torch.cuda.is_available():
        orig_empty = torch.empty
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *args, **kwargs: orig_empty(
                *args, **{
                    **kwargs, "pin_memory": False
                }),
        )

    vocab_size = 4
    max_draft_len = 3
    target_prompt_len = 5

    py_result = PyResult(
        prompt_len=target_prompt_len + 1,
        max_new_tokens=max_draft_len,
        use_device_memory=False,
        streaming=False,
        return_log_probs=False,
        return_context_logits=False,
        return_generation_logits=True,
        exclude_last_generation_logits=False,
        use_chunked_generation_logits=True,
    )

    added_tokens: list[int] = []
    target_req = SimpleNamespace(
        py_request_id=11,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_draft_pages_allocated=max_draft_len,
        py_draft_tokens=[0] * max_draft_len,
        py_draft_logits=None,
        py_target_probs=None,
        py_draft_use_greedy_sampling=False,
        py_sampling_strategy=("temperature", 1.0),
        seq_slot=0,
        py_seq_slot=0,
        py_end_id=-1,
        py_orig_prompt_len=target_prompt_len,
        py_max_new_tokens=32,
        py_stop_words_list=None,
        add_new_token=lambda token, beam_idx: added_tokens.append(token),
        get_num_tokens=lambda beam_idx: target_prompt_len + len(added_tokens),
        get_tokens=lambda beam_idx: [1] * target_prompt_len + added_tokens,
        finish_by=lambda reason, beam_idx: None,
    )
    draft_req = SimpleNamespace(
        py_request_id=11,
        state=LlmRequestState.GENERATION_COMPLETE,
        py_prompt_len=target_prompt_len + 1,
        context_current_position=0,
        context_chunk_size=target_prompt_len,
        py_return_context_logits=False,
        py_return_generation_logits=True,
        py_use_chunked_generation_logits=True,
        is_last_context_chunk=False,
        py_is_draft=True,
        py_result=py_result,
        _last_token=2,
        get_last_tokens=lambda beam: draft_req._last_token,
    )

    drafter = ModelDrafter.__new__(ModelDrafter)
    drafter.max_draft_len = max_draft_len
    drafter.max_total_draft_tokens = max_draft_len
    drafter._static_max_total_draft_tokens = max_draft_len
    drafter.draft_tokens_accumulator = {}
    drafter.draft_logits_accumulator = {}
    drafter.req_id_to_old_request = {11: target_req}
    drafter.draft_seq_slot_manager = SimpleNamespace(
        free_resources=lambda req: None)
    drafter.update_requests = lambda state, resource_manager=None: None

    with torch.inference_mode():
        l0 = torch.tensor([0.0, 0.0, 10.0, 0.0])

    # Step 0 produces L_0; draft_req then finishes after m=1 (< K=3) token.
    HandleLogits()(
        context_requests=[draft_req],
        generation_requests=[],
        logits=l0.unsqueeze(0),
        beam_width=1,
        num_context_logits_prefix_sum=[0, 1],
        is_generation_model=True,
    )
    draft_batch = SimpleNamespace(all_requests=lambda: [draft_req])
    remaining = drafter.process_decoded_tokens(draft_batch, draft_position=1)
    assert remaining == []

    empty_batch = SimpleNamespace(all_requests=lambda: [])
    drafter.process_dynamic_draft_outputs(
        SimpleNamespace(scheduled_requests=empty_batch))

    # Padded draft tokens have length K=3 ([2, 0, 0]), while py_draft_logits has m=1 row.
    assert target_req.py_draft_tokens == [2, 0, 0]
    assert target_req.py_draft_logits is not None
    assert target_req.py_draft_logits.shape == (1, vocab_size)

    target_req.py_draft_logits = target_req.py_draft_logits.cpu()
    target_req.py_target_probs = torch.tensor([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])

    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = TorchSampler.__new__(TorchSampler)
    sampler.max_seq_len = 128
    sampler.get_generator = lambda device: generator

    new_tokens_tensor = torch.zeros((max_draft_len + 1, 1, 1),
                                    dtype=torch.int32)
    # Distinct bonus tokens at each step: step m=1 has token 3, step K=3 has token 1.
    new_tokens_list = [[[2]], [[3]], [[0]], [[1]]]

    num_accepted = sampler._process_draft_tokens_rejection_sampling(
        target_req, new_tokens_list, new_tokens_tensor)
    assert num_accepted == 1
    assert added_tokens == [2, 3]

    # Out-of-bounds draft_probs row count (> K) raises a clear ValueError.
    target_req.py_draft_logits = torch.zeros((max_draft_len + 1, vocab_size))
    with pytest.raises(ValueError, match="draft_probs row count"):
        sampler._process_draft_tokens_rejection_sampling(
            target_req, new_tokens_list, new_tokens_tensor)


if __name__ == "__main__":
    unittest.main()
