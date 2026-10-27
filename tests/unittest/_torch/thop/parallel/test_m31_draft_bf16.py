# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B1 native BF16 consumer vs stock FP32 bytes; D8 is diagnostic, not served engagement."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import m31_draft_bf16 as b1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Requires SM107",
)


@pytest.fixture(autouse=True)
def native_operator(monkeypatch):
    monkeypatch.setattr(b1, "_DISABLED", False)
    assert b1._load(), "SM107 CI must ship the BF16 Markov epilogue operator"


@pytest.mark.parametrize(
    "batch", [1, *range(13, 29)], ids=lambda n: "D8_diagnostic" if n == 1 else f"D{n * 8}"
)
@pytest.mark.parametrize("ties", [False, True], ids=["random", "ties"])
def test_bf16_consumer_exact_eager_and_changing_graph(batch, ties):
    generator = torch.Generator(device="cuda").manual_seed(870884 + batch)
    base = torch.randn(batch, 7, 200064, device="cuda", dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(batch, 200064, device="cuda", dtype=torch.bfloat16, generator=generator)
    w1 = torch.randn(200064, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    if ties:
        base.zero_()
        bias.zero_()
    fp32 = base.float()
    splits = int(torch.ops.trtllm.minimax_m31_markov_epilogue_splits(batch, 200064))
    partials = torch.empty(batch * splits * 8, device="cuda", dtype=torch.uint8)
    counters = torch.zeros(batch, device="cuda", dtype=torch.int32)
    expected = torch.empty_like(fp32)
    actual = torch.empty_like(fp32)
    prev_expected = torch.empty(7, batch, device="cuda", dtype=torch.int64)
    prev_actual = torch.empty_like(prev_expected)
    emb_expected = torch.empty(7, batch, 256, device="cuda", dtype=torch.bfloat16)
    emb_actual = torch.empty_like(emb_expected)

    def reference():
        for pos in range(7):
            torch.ops.trtllm.minimax_m31_markov_epilogue(
                fp32,
                pos,
                bias,
                expected,
                prev_expected[pos],
                w1,
                emb_expected[pos],
                partials,
                counters,
                splits,
            )

    def candidate():
        for pos in range(7):
            torch.ops.trtllm.minimax_m31_markov_epilogue_bf16(
                base,
                pos,
                bias,
                actual,
                prev_actual[pos],
                w1,
                emb_actual[pos],
                partials,
                counters,
                splits,
            )

    def check():
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        assert torch.equal(prev_actual, prev_expected)
        assert torch.equal(emb_actual.view(torch.int16), emb_expected.view(torch.int16))

    reference()
    candidate()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for replay in range(3):
        if ties:
            base.fill_(replay + 1)
            bias.fill_(0.25 * replay)
        else:
            base.neg_()
            bias.add_(0.125)
        w1.add_(0.125)
        fp32.copy_(base)
        # Recompute stock outside the graph: two stale captured outputs cannot pass.
        reference()
        graph.replay()
        check()


def test_bf16_adapter_kill_and_d8_bypass(monkeypatch):
    worker = type("DSparkWorker", (), {"__module__": b1._TARGET + ".dspark"})()
    processor = type("_DraftA1Logits", (), {"__module__": b1._TARGET + ".dspark"})()
    model = SimpleNamespace(logits_processor=processor)
    assert b1.try_head_and_refine(worker, model, None, None, None, None, 1, 7) is None
    monkeypatch.setattr(b1, "_DISABLED", True)
    assert b1.try_head_and_refine(worker, model, None, None, None, None, 22, 7) is None
