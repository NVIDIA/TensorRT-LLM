# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family ``maybe_execute_in_parallel`` forks, joins and records as the engine's does, on the engine's switch."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.common import multi_stream
from tensorrt_llm._torch.modules import multi_stream_utils as engine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_reads_the_engine_switch() -> None:
    assert multi_stream.do_multi_stream is engine.do_multi_stream
    with engine.with_multi_stream(True):
        assert multi_stream.do_multi_stream()
    assert not multi_stream.do_multi_stream()


def _run(fn, multi: bool, aux):
    a = torch.randn(64, 7168, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1536, 7168, device="cuda", dtype=torch.bfloat16)
    streams = []

    def fn0():
        streams.append(torch.cuda.current_stream())
        return a @ w.t()

    def fn1():
        streams.append(torch.cuda.current_stream())
        return (a * 2) @ w.t(), {"x": a + 1}

    with engine.with_multi_stream(multi):
        out = fn(fn0, fn1, torch.cuda.Event(), torch.cuda.Event(), aux)
    torch.cuda.synchronize()
    return out, streams


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("with_aux", [False, True])
def test_matches_the_engine(multi, with_aux) -> None:
    aux = torch.cuda.Stream() if with_aux else None
    torch.manual_seed(0)
    ours, our_streams = _run(multi_stream.maybe_execute_in_parallel, multi, aux)
    torch.manual_seed(0)
    theirs, their_streams = _run(engine.maybe_execute_in_parallel, multi, aux)
    assert torch.equal(ours[0], theirs[0])
    assert torch.equal(ours[1][0], theirs[1][0]) and torch.equal(ours[1][1]["x"], theirs[1][1]["x"])
    assert our_streams == their_streams
    assert (our_streams[1] == aux) == (multi and with_aux)
