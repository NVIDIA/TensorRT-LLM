# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch._dynamo.backends.registry import get_default_backend, set_default_backend

from tensorrt_llm._torch.attention import attention

pytestmark = pytest.mark.cpu_only


class CountingBackend:
    def __init__(self) -> None:
        self.graphs = 0

    def __call__(
        self, graph: torch.fx.GraphModule, inputs: list[torch.Tensor]
    ) -> Callable[..., object]:
        self.graphs += 1
        return graph.forward


@pytest.fixture
def backend() -> Generator[CountingBackend, None, None]:
    torch._dynamo.reset()
    original_backend = get_default_backend()
    counter = CountingBackend()
    set_default_backend(counter)
    with torch._dynamo.config.patch(recompile_limit=8):
        try:
            yield counter
        finally:
            set_default_backend(original_backend)
            torch._dynamo.reset()


def make_helpers() -> tuple[attention._HelixNcclCompile, attention._HelixNcclCompile]:
    # Exercise the real decorator and tensor bodies with a CPU counting backend.
    return (
        attention._HelixNcclCompile(attention._helix_nccl_pre_alltoall._fn),
        attention._HelixNcclCompile(attention._helix_nccl_post_alltoall._fn),
    )


def inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    partial = torch.arange(batch_size * 32, dtype=torch.float32).reshape(batch_size, 32)
    stats = torch.ones(batch_size, 4, 2)
    mask = torch.arange(batch_size) % 2 == 0
    return partial, stats, mask


def run_helpers(
    pre: attention._HelixNcclCompile, post: attention._HelixNcclCompile, batch_size: int
) -> None:
    partial, stats, mask = inputs(batch_size)
    expected = pre._fn(partial, stats, mask, 2)
    actual = pre(partial, stats, mask, 2)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    gathered = [tensor.unsqueeze(0) for tensor in actual]
    for actual_tensor, expected_tensor in zip(post(gathered), post._fn(gathered)):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("configured,variants", [(8, 16), (32, 24)])
def test_helpers_compile_all_variants_and_reuse_cache(
    backend: CountingBackend, configured: int, variants: int
) -> None:
    pre, post = make_helpers()
    # User settings applied after importing/constructing the helpers still apply.
    with torch._dynamo.config.patch(recompile_limit=configured):
        for batch_size in range(1, variants + 1):
            run_helpers(pre, post, batch_size)
        assert backend.graphs == 2 * variants
        for batch_size in range(1, variants + 1):
            run_helpers(pre, post, batch_size)
        assert backend.graphs == 2 * variants
        assert torch._dynamo.config.recompile_limit == configured


def test_warmup_user_limit_survives_executor_thread(backend: CountingBackend) -> None:
    pre, post = make_helpers()
    with torch._dynamo.config.patch(recompile_limit=32):
        run_helpers(pre, post, 1)

        def worker() -> None:
            assert torch._dynamo.config.recompile_limit == 8
            for batch_size in range(2, 25):
                run_helpers(pre, post, batch_size)
            assert torch._dynamo.config.recompile_limit == 8

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(worker).result()
        assert torch._dynamo.config.recompile_limit == 32
    assert backend.graphs == 48


def test_helpers_compile_inside_opaque_op(backend: CountingBackend) -> None:
    pre, post = make_helpers()

    @torch.library.custom_op("helix_compile_budget::reformat", mutates_args=())
    def reformat(partial: torch.Tensor, stats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        chunks = pre(partial, stats, mask, 2)
        return post([tensor.unsqueeze(0) for tensor in chunks])[0].clone()

    @reformat.register_fake
    def fake_reformat(
        partial: torch.Tensor, stats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return partial[:, : partial.shape[1] // 2].unsqueeze(0).clone()

    compiled = torch.compile(
        reformat, backend="eager", fullgraph=True, dynamic=True, recompile_limit=16
    )
    for batch_size in range(1, 17):
        partial, stats, mask = inputs(batch_size)
        chunks = pre._fn(partial, stats, mask, 2)
        expected = post._fn([tensor.unsqueeze(0) for tensor in chunks])[0]
        torch.testing.assert_close(compiled(partial, stats, mask), expected)
    assert backend.graphs == 32
    assert torch._dynamo.config.recompile_limit == 8


def test_helpers_inline_inside_nonopaque_fullgraph(backend: CountingBackend) -> None:
    pre, post = make_helpers()

    def reformat(partial: torch.Tensor, stats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        chunks = pre(partial, stats, mask, 2)
        return post([tensor.unsqueeze(0) for tensor in chunks])[0]

    outer_backend = CountingBackend()
    compiled = torch.compile(
        reformat, backend=outer_backend, fullgraph=True, dynamic=False, recompile_limit=16
    )
    for batch_size in range(1, 13):
        partial, stats, mask = inputs(batch_size)
        chunks = pre._fn(partial, stats, mask, 2)
        expected = post._fn([tensor.unsqueeze(0) for tensor in chunks])[0]
        torch.testing.assert_close(compiled(partial, stats, mask), expected)
    assert outer_backend.graphs == 12
    assert backend.graphs == 0
    assert torch._dynamo.config.recompile_limit == 8


def test_helper_exception_preserves_context_limit(backend: CountingBackend) -> None:
    _, post = make_helpers()
    with torch._dynamo.config.patch(recompile_limit=32):
        with pytest.raises(IndexError, match="Dimension out of range"):
            post([torch.ones(2)])
        assert torch._dynamo.config.recompile_limit == 32
    assert torch._dynamo.config.recompile_limit == 8


def test_specialization_warning_uses_helper_budget(
    backend: CountingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    pre, _ = make_helpers()
    monkeypatch.setattr(attention, "_helix_nccl_pre_alltoall", pre)
    monkeypatch.setattr(attention, "_HELIX_NCCL_SHAPES", set())
    monkeypatch.setattr(attention, "_HELIX_NCCL_LIMIT_REPORTED", False)
    warnings = []
    monkeypatch.setattr(attention.logger, "warning", lambda *args: warnings.append(args))
    for batch_size in range(1, 17):
        partial, stats, _ = inputs(batch_size)
        attention._helix_note_nccl_specialization(partial, stats, 2)
    assert not warnings
    # note() precedes __call__, so use a newly raised limit before rebinding.
    with torch._dynamo.config.patch(recompile_limit=32):
        partial, stats, mask = inputs(17)
        attention._helix_note_nccl_specialization(partial, stats, 2)
        pre(partial, stats, mask, 2)
    for batch_size in range(18, 33):
        partial, stats, _ = inputs(batch_size)
        attention._helix_note_nccl_specialization(partial, stats, 2)
    assert not warnings
    partial, stats, _ = inputs(33)
    attention._helix_note_nccl_specialization(partial, stats, 2)
    assert len(warnings) == 1
    assert warnings[0][-1] == 32
