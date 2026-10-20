# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused end-ID and maximum-length checks must match the tensor path."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
from tensorrt_llm.bindings.executor import FinishReason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("max_tokens,max_beam_width", [(1, 1), (3, 1), (1, 2), (2, 3)])
def test_fused_matches_tensor_ops(max_tokens: int, max_beam_width: int) -> None:
    torch.manual_seed(0)
    max_num_sequences, end_id = 5, 99
    handler = FinishReasonsHandler(
        max_num_sequences=max_num_sequences,
        max_beam_width=max_beam_width,
        max_tokens=max_tokens,
        max_seq_len=64,
    )
    store = handler.store
    store.max_lengths_cuda.fill_(14)
    store.end_ids_cuda.fill_(end_id)

    # The sampler forces int64 slots; the tensor path's index_fill_ rejects int32.
    seq_slots = torch.tensor([3, 0, 4], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([7, 13, 15], dtype=torch.int32, device="cuda")
    new_tokens = torch.randint(
        0,
        50,
        (max_tokens, max_num_sequences + 1, max_beam_width),
        dtype=torch.int32,
        device="cuda",
    )
    new_tokens[0, 4, 0] = end_id  # one slot finishes on the end ID, one on max length

    def run() -> torch.Tensor:
        # Seeding with END_ID also shows that untouched slots keep their value.
        store.finish_reasons_cuda.fill_(FinishReason.END_ID.value)
        handler._write_finish_reasons(seq_slots=seq_slots, seq_lens=seq_lens, new_tokens=new_tokens)
        return store.finish_reasons_cuda.clone()

    # Without this the comparison below would pass with both arms on the fallback.
    assert handler._can_fuse_finish_reasons(
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        new_tokens=new_tokens,
        stop_word_indices=None,
        first_finish_reasons=None,
    )
    fused = run()

    handler._can_fuse_finish_reasons = lambda **_: False
    torch.testing.assert_close(fused, run(), rtol=0, atol=0)
    assert fused[0, 4, 0].item() == FinishReason.END_ID.value
    assert fused[0, 0, 0].item() == FinishReason.LENGTH.value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_stop_words_history_and_slot_reuse() -> None:
    """Packed matching agrees with a token-list oracle across accepted prefixes."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.sampler.ops.triton import reorder_compact_stop_word_history

    for steps, beams in ((1, 1), (3, 1), (1, 3)):
        handler = FinishReasonsHandler(
            max_num_sequences=64,
            max_beam_width=beams,
            max_tokens=steps,
            max_seq_len=512,
        )
        # Non-contiguous slots and different rule counts / lengths in the same batch.
        words = [[[7], [2, 3, 4], [0, 1, 2, 3, 4]], [[9], [8]], [[4, 5]]]
        prompts = [[0, 1, 2, 3], [8, 2], [2, 3, 4]]
        slots_host = torch.tensor([5, 1, 8], dtype=torch.int32)
        slots = slots_host.cuda().long()
        requests = [
            SimpleNamespace(
                py_seq_slot=slot,
                py_stop_words_list=rules,
                py_orig_prompt_len=len(prompt),
                py_max_new_tokens=64,
                py_end_id=99,
                is_context_only_request=False,
                py_beam_width=beams,
                is_finished=False,
                get_tokens=lambda prompt=prompt: [prompt[:] for _ in range(beams)],
            )
            for slot, rules, prompt in zip(slots_host.tolist(), words, prompts)
        ]
        handler.setup_new_request_handling()
        for request in requests:
            handler.prepare_for_new_request(request)
        handler.update_for_new_request(
            seq_slots_cuda_long=slots,
            max_lengths_cuda=torch.full_like(slots, 100, dtype=torch.int32),
            end_ids_cuda=torch.full_like(slots, 99, dtype=torch.int32),
            seq_slots_host=slots_host,
        )
        histories = [[prompt[:] for _ in range(beams)] for prompt in prompts]
        previous = None
        for iteration in range(4):
            accepted_host = [1, steps, max(1, steps - 1)]
            if steps > 1 and iteration == 2:
                accepted_host[0] = steps
            if previous is not None:
                for request_idx in range(3):
                    for beam in range(beams):
                        histories[request_idx][beam].extend(
                            previous[: accepted_host[request_idx], request_idx, beam].tolist()
                        )
            if steps > 1 and iteration == 2:
                retained = [handler._stop_word_allocations[slot] for slot in (5, 1)]
                retained_data = [entry.tensor.clone() for entry in retained]
                retained_metadata = handler.store.stop_words_metadata_cuda[slots[:2]].clone()
                handler.release_request(requests[2])
                words[2] = [[6, 7], [5, 6, 7]]
                requests[2] = SimpleNamespace(
                    **{
                        **vars(requests[2]),
                        "py_stop_words_list": words[2],
                        "py_orig_prompt_len": 2,
                        "get_tokens": lambda: [[5, 6] for _ in range(beams)],
                    }
                )
                handler.setup_new_request_handling()
                handler.prepare_for_new_request(requests[2])
                handler.update_for_new_request(
                    seq_slots_cuda_long=slots[2:],
                    max_lengths_cuda=torch.full_like(slots[2:], 100, dtype=torch.int32),
                    end_ids_cuda=torch.full_like(slots[2:], 99, dtype=torch.int32),
                    seq_slots_host=slots_host[2:],
                )
                for slot, entry, before in zip((5, 1), retained, retained_data):
                    assert handler._stop_word_allocations[slot] is entry
                    torch.testing.assert_close(entry.tensor, before)
                torch.testing.assert_close(
                    handler.store.stop_words_metadata_cuda[slots[:2]], retained_metadata
                )
                replacement = handler._stop_word_allocations[8]
                assert replacement.request is requests[2]
                assert handler.store.stop_words_metadata_cuda[8].tolist() == [
                    replacement.tensor.data_ptr(),
                    2,
                    8,
                    steps + 2,
                    beams,
                ]
                assert handler.store.max_stop_word_lengths_host[8].item() == 3
                histories[2] = [[5, 6] for _ in range(beams)]
                accepted_host[2] = 1
            if beams > 1:
                parents_host = torch.tensor([[2, 0, 0], [1, 1, 0], [0, 2, 1]])
                reorder_compact_stop_word_history(
                    handler.store.stop_words_metadata_cuda, slots, parents_host.cuda()
                )
                histories = [
                    [history[parent][:] for parent in parents]
                    for history, parents in zip(histories, parents_host.tolist())
                ]
            tokens_host = (
                torch.arange(steps * 3 * beams).reshape(steps, 3, beams) + 4 + iteration
            ) % 10
            if beams > 1 and iteration < 2:
                tokens_host[0, 2] = torch.tensor([5, 7, 4] if iteration == 0 else [5, 5, 5])
            if steps > 1 and iteration == 2:
                tokens_host[0, 2] = 7
            tokens = tokens_host.cuda().int()
            accepted = torch.tensor(accepted_host, dtype=torch.int32, device="cuda")
            if iteration == 0:
                accepted.fill_(1)
            actual = handler._are_stop_words(slots, tokens, accepted).cpu()
            expected = torch.zeros_like(tokens_host, dtype=torch.bool)
            for step in range(steps):
                for request_idx in range(3):
                    for beam in range(beams):
                        prefix = (
                            histories[request_idx][beam]
                            + tokens_host[: step + 1, request_idx, beam].tolist()
                        )
                        expected[step, request_idx, beam] = any(
                            prefix[-len(word) :] == word for word in words[request_idx]
                        )
            torch.testing.assert_close(actual, expected)
            previous = tokens_host

        # Reusing a slot without stop words must clear its metadata and ownership.
        requests[0].py_stop_words_list = None
        handler.setup_new_request_handling()
        handler.prepare_for_new_request(requests[0])
        handler.update_for_new_request(
            seq_slots_cuda_long=slots[:1],
            max_lengths_cuda=torch.tensor([100], device="cuda", dtype=torch.int32),
            end_ids_cuda=torch.tensor([99], device="cuda", dtype=torch.int32),
            seq_slots_host=slots_host[:1],
        )
        assert 5 not in handler._stop_word_allocations
        assert handler.store.max_stop_word_lengths_host[5].item() == 0
        assert handler.store.stop_words_metadata_cuda[5, :4].count_nonzero().item() == 0
        requests[1].is_finished = True
        handler.release_request(requests[1])
        assert 1 not in handler._stop_word_allocations
        assert 8 in handler._stop_word_allocations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_stop_words_storage_and_finish_precedence() -> None:
    """An outlier does not expand other slots or reset their device histories."""
    from types import SimpleNamespace

    handler = FinishReasonsHandler(
        max_num_sequences=1024,
        max_beam_width=1,
        max_tokens=1,
        max_seq_len=4096,
    )

    def admit(slot: int, words: list[list[int]], prompt: list[int]) -> SimpleNamespace:
        request = SimpleNamespace(
            py_seq_slot=slot,
            py_stop_words_list=words,
            py_orig_prompt_len=len(prompt),
            py_max_new_tokens=1000,
            py_end_id=99,
            is_context_only_request=False,
            py_beam_width=1,
            is_finished=False,
            get_tokens=lambda: [prompt],
        )
        handler.setup_new_request_handling()
        handler.prepare_for_new_request(request)
        handler.update_for_new_request(
            seq_slots_cuda_long=torch.tensor([slot], device="cuda", dtype=torch.int64),
            max_lengths_cuda=torch.tensor([10], device="cuda", dtype=torch.int32),
            end_ids_cuda=torch.tensor([99], device="cuda", dtype=torch.int32),
            seq_slots_host=torch.tensor([slot]),
        )
        return request

    assert not handler.has_stop_word_history
    many = admit(2, [[i] for i in range(1024)], [1, 2, 3])
    assert not handler.has_stop_word_history
    allocation = handler._stop_word_allocations[2].tensor
    snapshot = allocation.clone()
    long = admit(3, [list(range(256))], list(range(255)))
    assert handler.has_stop_word_history
    assert handler._stop_word_allocations[2].tensor is allocation
    torch.testing.assert_close(allocation, snapshot)
    total_bytes = handler.store.stop_words_metadata_cuda.nbytes + sum(
        entry.tensor.nbytes for entry in handler._stop_word_allocations.values()
    )
    # Metadata plus actual rules and per-request windows; no 1024 x 1024 x 256 tensor.
    assert total_bytes < 64 * 1024
    assert handler._stop_word_allocations[3].tensor.numel() == 2 + 256 + 2 * 256

    slots = torch.tensor([2], device="cuda", dtype=torch.int64)
    tokens = torch.zeros((1, 1025, 1), device="cuda", dtype=torch.int32)
    for token, seq_len, expected in (
        (7, 1, FinishReason.STOP_WORDS),
        (7, 9, FinishReason.LENGTH),
        (99, 9, FinishReason.END_ID),
    ):
        tokens[0, 2, 0] = token
        handler.write_finish_reasons(
            seq_slots_host=slots.cpu(),
            is_draft_batch=False,
            seq_slots_cuda=slots,
            seq_lens_cuda=torch.tensor([seq_len], device="cuda"),
            new_tokens_cuda=tokens,
        )
        assert handler.store.finish_reasons_cuda[0, 2, 0].item() == expected.value

    long_match = handler._are_stop_words(
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        torch.tensor([[[255]]], dtype=torch.int32, device="cuda"),
        1,
    )
    assert long_match.item()

    primary_stream = torch.cuda.current_stream()
    alternate_stream = torch.cuda.Stream()
    for stream in (primary_stream, alternate_stream, primary_stream):
        transient = admit(4, [[7, 8]], [7])
        stream.wait_stream(primary_stream)
        with torch.cuda.stream(stream):
            match_slots = torch.tensor([4], dtype=torch.int64, device="cuda")
            match_tokens = torch.tensor([[[8]]], dtype=torch.int32, device="cuda")
            first_match = handler._are_stop_words(match_slots, match_tokens, 1)
            second_match = handler._are_stop_words(match_slots, match_tokens, 1)
        primary_stream.wait_stream(stream)
        assert first_match.item()
        assert not second_match.item()
        transient.is_finished = True
        handler.release_request(transient)
        assert 4 not in handler._stop_word_allocations

    old = admit(4, [[7, 8]], [7])
    replacement = admit(4, [[8, 9]], [8])
    replacement_allocation = handler._stop_word_allocations[4].tensor
    handler.release_request(old)
    assert handler._stop_word_allocations[4].tensor is replacement_allocation
    handler.release_request(replacement)

    # Releasing one request preserves other live requests and is idempotent.
    many.is_finished = True
    handler.release_request(many)
    handler.release_request(many)
    assert 2 not in handler._stop_word_allocations
    assert 3 in handler._stop_word_allocations
    handler.release_request(long)
    long.py_seq_slot = None
    handler.release_request(long)
    assert not handler._stop_word_allocations
    assert not handler.has_stop_word_history
