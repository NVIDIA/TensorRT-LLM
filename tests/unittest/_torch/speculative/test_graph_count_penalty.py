import pytest
import torch

import tensorrt_llm  # noqa: F401


def _require_spec_count_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        torch.ops.trtllm.speculative_apply_count_frequency_penalty
        torch.ops.trtllm.speculative_append_accepted_token_counts
    except AttributeError:
        pytest.skip("speculative count penalty ops are not available")


def test_count_frequency_penalty_uses_replay_time_counts_in_cuda_graph():
    _require_spec_count_ops()

    vocab_size = 8
    row_slots = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    frequency_penalties = torch.tensor([0.5, 0.25],
                                       dtype=torch.float32,
                                       device="cuda")
    accepted_tokens = torch.empty((2, 3), dtype=torch.int32, device="cuda")
    accepted_lens = torch.empty((2, ), dtype=torch.int32, device="cuda")
    token_counts = torch.zeros((2, vocab_size), dtype=torch.int32, device="cuda")
    logits = torch.empty((2, vocab_size), dtype=torch.float32, device="cuda")

    def run_count_penalty():
        token_counts.zero_()
        logits.zero_()
        torch.ops.trtllm.speculative_append_accepted_token_counts(
            token_counts, row_slots, accepted_tokens, accepted_lens)
        torch.ops.trtllm.speculative_apply_count_frequency_penalty(
            logits, token_counts, row_slots, frequency_penalties)

    warmup_tokens = torch.tensor([[1, 2, 2], [3, 3, 7]],
                                 dtype=torch.int32,
                                 device="cuda")
    warmup_lens = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    accepted_tokens.copy_(warmup_tokens)
    accepted_lens.copy_(warmup_lens)
    for _ in range(3):
        run_count_penalty()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_count_penalty()

    graph.replay()
    expected = torch.zeros_like(logits)
    expected[0, 1] = -0.5
    expected[0, 2] = -1.0
    expected[1, 3] = -0.5
    torch.testing.assert_close(logits, expected)

    replay_tokens = torch.tensor([[4, 4, 4], [0, 5, 5]],
                                 dtype=torch.int32,
                                 device="cuda")
    replay_lens = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    accepted_tokens.copy_(replay_tokens)
    accepted_lens.copy_(replay_lens)
    graph.replay()

    expected.zero_()
    expected[0, 4] = -0.5
    expected[1, 0] = -0.25
    expected[1, 5] = -0.5
    torch.testing.assert_close(logits, expected)
