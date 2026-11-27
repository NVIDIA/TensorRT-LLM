# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
import torch
from test_common.serve.app_factory import AggConfig

from tensorrt_llm.serve.decision import scoring
from tensorrt_llm.serve.decision.scoring import ReadQuestion, _distribution
from tensorrt_llm.serve.render import decisions
from tensorrt_llm.serve.render.decisions import (
    RenderedDecision,
    _generic_body,
    _pplx_body,
    _read_labels,
)
from tensorrt_llm.serve.render.resources import RenderResources

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
async def test_text_renderer_passes_parsed_messages_to_template(serve_env, monkeypatch):
    server = serve_env.build_openai_server(AggConfig())
    resources = RenderResources.from_server(server)

    async def render(res, *, conversation, mm_placeholder_counts, **kwargs):
        assert [msg["role"] for msg in conversation] == ["system", "user"]
        assert conversation[-1]["content"] == "Pick one."
        assert mm_placeholder_counts == [{}, {}]
        return SimpleNamespace(text="Pick one. Answer:")

    monkeypatch.setattr(decisions, "render_conversation", render)
    assert await decisions._render_text(resources, "Pick one.", "Classify.") == "Pick one. Answer:"


def test_selected_logits_include_candidates_below_top_k():
    logits = torch.tensor([50.0, -3.0, 0.0, -2.0])
    assert _distribution(logits, [3, 1], 2) == pytest.approx([0.62245933, 0.37754067])
    assert _distribution(logits, [1, 3], 1e-250) == [0, 1]
    with pytest.raises(RuntimeError, match="non-finite"):
        _distribution(torch.tensor([0.0, torch.nan]), [0, 1], 1)


def test_pplx_noul_uses_checkpoint_order_and_json_encoding():
    question = ReadQuestion("noul", {"q": ["α", "β"]}, ["yes", "no"], [None, None])
    body, indices = _pplx_body({"a": 1}, question, ["A", "B"])
    assert indices == [1, 0]
    assert body == (
        'State:\n{"a": 1}\n\nQuestion:\n{"q": ["α", "β"]}\n\nOptions:\n'
        "A: No / false\nB: Yes / true\n\nReturn only the letter code of the best option."
    )


def test_score_renders_ordered_structured_levels():
    q = ReadQuestion("score", "Urgency?", ["0", "1"], ["low", {"high": 1}])
    assert _generic_body(["x", "y"], q, q.names) == (
        '["x","y"]\n\nQuestion: Urgency?\n0: low\n1: {"high":1}\nAnswer with the number of one level only.'
    )


def test_literal_chat_and_vision_controls_have_distinct_text_spellings():
    from tensorrt_llm.serve.render.decisions import _escape_text

    tokens = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|im_end|>", "<think>"]
    for token in tokens:
        quoted = _escape_text(token)
        assert quoted == "<\\" + token[1:]
        assert _escape_text(quoted) == "<\\" + quoted[1:]
    assert _escape_text("mug < 5 > 3") == "mug < 5 > 3"


class MergingTokenizer:
    def get_added_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return {"p": [1], "pa": [1, 2], "pb": [3], "pc": [1, 2]}[text]


def test_labels_must_extend_the_actual_prompt_and_remain_distinct():
    assert _read_labels(MergingTokenizer(), "p", ["a"]) == ([1], [2])
    with pytest.raises(ValueError, match="distinct"):
        _read_labels(MergingTokenizer(), "p", ["a", "b"])
    with pytest.raises(ValueError, match="distinct"):
        _read_labels(MergingTokenizer(), "p", ["a", "c"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_batch", "expected_window"), [(None, 1), (2, 2), (16, 16), (2048, 20)]
)
async def test_decision_rendering_and_reads_are_bounded_in_request_order(
    monkeypatch, engine_batch, expected_window
):
    args = SimpleNamespace(max_batch_size=engine_batch, max_seq_len=4096, context_parallel_size=1)
    server = SimpleNamespace(
        generator=SimpleNamespace(_hf_model_dir=None, args=args), encode_batcher=None
    )
    completed = 0
    max_outstanding = 0
    count = 20

    async def render(*args):
        nonlocal max_outstanding
        for i in range(count):
            max_outstanding = max(max_outstanding, i + 1 - completed)
            yield RenderedDecision([i], [1], None)

    async def read(server, row, *args):
        nonlocal completed
        await asyncio.sleep(0)
        completed += 1
        return scoring.ReadResult([1.0], input_tokens=row.token_ids[0])

    monkeypatch.setattr(scoring, "render_decisions", render)
    monkeypatch.setattr(scoring, "_generate_read", read)
    results = await scoring.score_decision_questions(server, state="shared", questions=[])
    assert [r.input_tokens for r in results] == list(range(count))
    assert max_outstanding == expected_window


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_context", [4, None])
async def test_first_oversize_prompt_stops_rendering_later_questions(
    monkeypatch, configured_context
):
    args = SimpleNamespace(
        max_batch_size=16, max_seq_len=configured_context, context_parallel_size=2
    )
    server = SimpleNamespace(
        generator=SimpleNamespace(_hf_model_dir=None, args=args), encode_batcher=None
    )
    rendered = []
    submitted = []

    async def render(*args):
        for i in range(200):
            rendered.append(i)
            yield RenderedDecision(list(range(8)), [1], None)

    async def read(server, row, *args):
        submitted.append(row)
        raise ValueError("maximum context length")

    monkeypatch.setattr(scoring, "render_decisions", render)
    monkeypatch.setattr(scoring, "_generate_read", read)
    with pytest.raises(ValueError, match="maximum context length"):
        await scoring.score_decision_questions(server, state="shared", questions=[])
    assert rendered == [0]
    assert len(submitted) == int(configured_context is None)
