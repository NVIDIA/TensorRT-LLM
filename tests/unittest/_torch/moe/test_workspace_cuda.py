# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CUTLASS regression tests; requires native libraries matching this checkout."""

from contextlib import nullcontext

import pytest
import torch

from tensorrt_llm._torch.custom_ops import torch_custom_ops as ops
from tensorrt_llm._torch.moe.workspace import CutlassWorkspaceReclaimer


class CutlassWorkload:
    """Fixed inputs shared by enabled/disabled arms, three calls per forward."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        generator = torch.Generator(device="cuda").manual_seed(19051)
        self.x = torch.randn(1024, 256, device="cuda", dtype=dtype, generator=generator)
        self.w1 = torch.randn(8, 512, 256, device="cuda", dtype=dtype, generator=generator) / 16
        self.w2 = torch.randn(8, 256, 256, device="cuda", dtype=dtype, generator=generator) / 16
        self.experts = torch.stack(
            (torch.arange(1024, device="cuda") % 8, (torch.arange(1024, device="cuda") + 1) % 8),
            dim=1,
        ).int()
        self.scales = torch.full((1024, 2), 0.5, device="cuda")

    def forward(
        self, tokens: int, scope: CutlassWorkspaceReclaimer | None, *, warmup: bool = False
    ) -> list[torch.Tensor]:
        with (
            scope.forward(warmup=warmup, num_tokens=tokens) if scope is not None else nullcontext()
        ):
            return [
                ops.fused_moe(
                    self.x[:n],
                    self.experts[:n],
                    self.scales[:n],
                    self.w1,
                    None,
                    self.w2,
                    None,
                    self.x.dtype,
                    [],
                    tune_max_num_tokens=1024,
                )[0]
                for n in (tokens, max(1, tokens // 2), max(1, tokens // 4))
            ]


def test_cutlass_growth_reclamation_and_regrowth(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = CutlassWorkload()
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    references = {
        n: [output.cpu() for output in workload.forward(n, None)] for n in (1024, 16, 8, 4)
    }
    # Use a separate native owner, preserving identical inputs/tactics. This
    # also prevents sharing with other tests from silently disabling reclaim.
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    scope = CutlassWorkspaceReclaimer()
    workload.forward(32, scope, warmup=True)
    baseline = torch.cuda.memory_allocated()
    for _ in range(3):
        peak = 0
        for step, n in enumerate((1024, 16, 8, 4)):
            outputs = workload.forward(n, scope)
            for output, expected in zip(outputs, references[n]):
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            del outputs, output
            retained = torch.cuda.memory_allocated()
            if step == 0:
                peak = retained
                assert peak > baseline, "Real kernel sizing must grow beyond the initial allocation"
            elif step < 3:
                assert retained == peak, "Count complete forwards, not the three layer calls"
            else:
                assert retained == baseline, (
                    "Third underfilled forward must return to the warmup floor"
                )
                assert retained < peak


def test_cutlass_graph_replay_after_eager_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = CutlassWorkload()
    monkeypatch.setattr(ops.MoERunner, "runner_dict", {})
    scope = CutlassWorkspaceReclaimer()
    workload.forward(32, scope, warmup=True)
    expected = [x.cpu() for x in workload.forward(1024, scope)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        workload.forward(1024, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = workload.forward(1024, scope)
    for _ in range(5):
        workload.forward(4, scope)
    graph.replay()
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
