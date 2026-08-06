# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 shared/routed expert multi-stream tests."""

from types import SimpleNamespace

import torch


def test_kimi_k3_moe_records_shared_output_on_main_stream(monkeypatch):
    import tensorrt_llm._torch.models.modeling_kimi_linear as modeling_kimi_linear
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoERuntime

    class _Output:
        def __init__(self):
            self.recorded_streams = []

        def record_stream(self, stream):
            self.recorded_streams.append(stream)

        def __add__(self, other):
            return other

    runtime = KimiK3MoERuntime.__new__(KimiK3MoERuntime)
    torch.nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(compute_logits=lambda hidden_states: None)
    runtime.shared_experts = lambda hidden_states: hidden_states
    runtime.moe_main_event = object()
    runtime.moe_shared_event = object()
    runtime.aux_stream = object()

    routed_out = _Output()
    shared_out = _Output()
    main_stream = object()
    monkeypatch.setattr(
        modeling_kimi_linear,
        "maybe_execute_in_parallel",
        lambda *args, **kwargs: (routed_out, shared_out),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main_stream)

    assert runtime.forward(object()) is shared_out
    assert shared_out.recorded_streams == [main_stream]
