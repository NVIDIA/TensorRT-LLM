# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lazily created exchange workspaces refuse to be created inside a CUDA-graph capture (K3 audit #150, check #10).

The draft-step and sampled-accept workspaces are created collectively (both ranks, a multicast buffer) on the first
eager call. Creating one inside a capture would allocate and synchronize mid-capture, so a capture that finds it missing
reports a miss (``kernels/m3_report.capture_missed``) and gets None (the caller then takes its stock path) instead.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (
    m3_draft_step,
    m3_report,
    m3_sampled_accept,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU"
)


def _refuse(*args, **kwargs):
    raise AssertionError("a workspace must not be created inside a CUDA-graph capture")


@pytest.mark.parametrize(
    "module, creator",
    [
        (m3_draft_step, "create_workspace"),
        (m3_sampled_accept, "SampledAcceptWorkspace.create"),
    ],
    ids=["draft_step", "sampled_accept"],
)
def test_a_capture_without_the_workspace_misses_instead_of_creating_it(
    monkeypatch, module, creator
):
    monkeypatch.setattr(module, "_WS", None)
    owner, _, attr = creator.rpartition(".")
    monkeypatch.setattr(getattr(module, owner) if owner else module, attr, _refuse)
    misses = []
    monkeypatch.setattr(m3_report, "capture_missed", lambda *args: misses.append(args))

    device = torch.device("cuda", torch.cuda.current_device())
    found = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        found.append(module.workspace(None, device))
    torch.cuda.current_stream().wait_stream(stream)

    assert found == [None]
    assert module._WS is None
    assert len(misses) == 1
