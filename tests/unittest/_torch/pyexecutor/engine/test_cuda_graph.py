# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.engine.cuda_graph import (
    cuda_graph_capture,
    cuda_graph_disabled,
    decode_tokens_per_forward,
    filter_cuda_graph_batch_sizes,
    filter_cuda_graph_num_tokens,
    filter_cuda_graph_seq_lens,
    plan_cuda_graphs,
    resolve_cuda_graph_batch_sizes,
)
from tensorrt_llm.llmapi.llm_args import BaseLlmArgs, CudaGraphConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    (
        "batch_sizes",
        "max_batch_size",
        "max_num_tokens",
        "tokens_per_request",
        "padding",
        "expected",
    ),
    [
        ([1, 2, 4, 8], 8, 3_000, 1_500, False, [1, 2]),
        ([1, 2, 4, 8], 8, 1_499, 1_500, True, []),
        ([1, 2, 4, 8], 6, 10_000, 1, True, [1, 2, 4, 6]),
        ([1, 2, 4, 8], 4, 10_000, 1, True, [1, 2, 4]),
    ],
)
def test_filter_cuda_graph_batch_sizes_respects_batch_and_token_budgets(
    batch_sizes: list[int],
    max_batch_size: int,
    max_num_tokens: int,
    tokens_per_request: int,
    padding: bool,
    expected: list[int],
) -> None:
    assert (
        filter_cuda_graph_batch_sizes(
            batch_sizes,
            max_batch_size,
            max_num_tokens,
            tokens_per_request,
            padding,
        )
        == expected
    )


def test_decode_tokens_stay_within_the_token_budget_at_the_default_sizes() -> None:
    """The default request and token budgets with four draft tokens per request: the largest
    decode forward holds max_num_tokens // 5 requests, not max_batch_size (10240 tokens)."""
    max_batch_size = BaseLlmArgs.model_fields["max_batch_size"].default
    max_num_tokens = BaseLlmArgs.model_fields["max_num_tokens"].default
    plan = plan_cuda_graphs(
        CudaGraphConfig(),
        None,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        max_seq_len=4096,
        decode_tokens_per_request=5,
    )
    tokens = decode_tokens_per_forward(plan.batch_sizes, max_batch_size, max_num_tokens, 5)
    assert tokens == [size * 5 for size in plan.batch_sizes] + [max_num_tokens // 5 * 5]
    assert decode_tokens_per_forward([], 64, 4096, 4) == [256]


@pytest.mark.parametrize(
    ("values", "limit", "padding", "expected"),
    [
        ([64, 128, 256], 128, False, [64, 128]),
        ([64, 128, 256], 192, True, [64, 128, 192]),
        ([256], 192, True, [192]),
        ([64, 128, 256], 128, True, [64, 128]),
    ],
)
def test_filter_cuda_graph_token_and_sequence_buckets(
    values: list[int], limit: int, padding: bool, expected: list[int]
) -> None:
    assert filter_cuda_graph_num_tokens(values, limit, padding) == expected
    assert filter_cuda_graph_seq_lens(values, limit, padding) == expected


@pytest.mark.parametrize(
    ("batch_sizes", "max_batch_size", "pad_to_limit", "expected"),
    [
        ((1, 2, 4, 8), 6, True, (1, 2, 4, 6)),
        ((1, 2, 4, 8), 6, False, (1, 2, 4)),
        ((1, 2, 4, 8), 4, True, (1, 2, 4)),
        ((1, 2, 4, 8), 0, True, ()),
    ],
)
def test_resolve_cuda_graph_batch_sizes(
    batch_sizes: tuple[int, ...],
    max_batch_size: int,
    pad_to_limit: bool,
    expected: tuple[int, ...],
) -> None:
    assert (
        resolve_cuda_graph_batch_sizes(
            batch_sizes,
            max_batch_size,
            pad_to_limit=pad_to_limit,
        )
        == expected
    )


@pytest.mark.parametrize("raises", [False, True])
def test_cuda_graph_disabled_restores_state(raises: bool) -> None:
    runner = SimpleNamespace(enabled=True)

    def use_context() -> None:
        with cuda_graph_disabled(runner):
            assert not runner.enabled
            if raises:
                raise RuntimeError("failure")

    if raises:
        with pytest.raises(RuntimeError, match="failure"):
            use_context()
    else:
        use_context()
    assert runner.enabled


@pytest.mark.parametrize("raises", [False, True])
def test_cuda_graph_capture_restores_phase_and_exits_backend_context(raises: bool) -> None:
    events: list[str] = []

    @contextmanager
    def allow_capture():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    runner = SimpleNamespace(is_warmup_only=True, allow_capture=allow_capture)

    def use_context() -> None:
        with cuda_graph_capture(runner):
            runner.is_warmup_only = False
            if raises:
                raise RuntimeError("failure")

    if raises:
        with pytest.raises(RuntimeError, match="failure"):
            use_context()
    else:
        use_context()
    assert runner.is_warmup_only
    assert events == ["enter", "exit"]
