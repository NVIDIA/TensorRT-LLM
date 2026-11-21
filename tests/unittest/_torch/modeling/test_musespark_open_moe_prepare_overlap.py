# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small batches pool on the auxiliary stream while the router runs, eagerly and in graphs."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _module(*, stream=True, comm=None):
    mod = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(mod)
    mod._prepare_stream = torch.cuda.Stream() if stream else None
    mod._prepare_fork_event = torch.cuda.Event()
    mod._prepare_ready_event = torch.cuda.Event()
    mod.nvtx = lambda name: contextlib.nullcontext()
    mod.experts = SimpleNamespace(comm=comm)
    router_weight = torch.randn(256, 256, device="cuda")
    mod.router_logits = lambda x: x.float() @ router_weight
    mod.pooler = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    mod.prepare_expert_input = lambda x: x * 0.5
    return mod


def _serial(mod, x):
    return mod.router_logits(x), mod.prepare_expert_input(mod.pooler(x))


def _prepare(mod, x):
    router_logits, expert_input, ready = mod.prepare_moe_inputs(x)
    if ready is not None:
        # The MoE joins the event before its first read of the expert input.
        ready.wait()
    return router_logits, expert_input, ready


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 80, 128, 129])
def test_prepare_overlap_matches_serial_and_graph_replay(tokens):
    torch.manual_seed(923)
    with torch.inference_mode():
        mod = _module()
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
        expected = _serial(mod, x)
        router_logits, expert_input, ready = _prepare(mod, x)
        assert (ready is not None) == (tokens <= 128)
        torch.cuda.synchronize()
        for got, want in zip((router_logits, expert_input), expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _prepare(mod, x)[:2]
        for _ in range(3):
            x.normal_()
            expected = _serial(mod, x)
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(captured, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["no_stream", "moe_comm", "no_tokens"])
def test_the_overlap_stays_off_where_it_does_not_apply(reason):
    with torch.inference_mode():
        mod = _module(stream=reason != "no_stream", comm=object() if reason == "moe_comm" else None)
        x = torch.randn(0 if reason == "no_tokens" else 4, 256, device="cuda", dtype=torch.bfloat16)
        router_logits, expert_input, ready = mod.prepare_moe_inputs(x)
        assert ready is None
        for got, want in zip((router_logits, expert_input), _serial(mod, x)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
