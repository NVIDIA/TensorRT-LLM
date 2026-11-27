# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the HTTP contracts independently of the checkpoint's readout."""

import asyncio
import gzip
import json
import sys
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request
from test_common.serve.app_factory import AggConfig

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.decision import routes
from tensorrt_llm.serve.decision.openai import OpenAIDecisionRequest
from tensorrt_llm.serve.decision.scoring import ReadResult
from tensorrt_llm.serve.encode_batcher import QueueFullError

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


@pytest.fixture
def scoring(monkeypatch):
    score = AsyncMock()
    monkeypatch.setattr(routes, "score_decision_questions", score)
    return score


@pytest.mark.parametrize("role", [None, ServerRole.EMBEDDING])
def test_systemone_named_answers(serve_env, scoring, role):
    server = serve_env.build_openai_server(AggConfig(role=role))
    scoring.return_value = [
        ReadResult([0.75, 0.25], 11),
        ReadResult([0.1, 0.9], 5),
        ReadResult([0.0, 0.25, 0.75], 3),
    ]
    levels = [{"tier": "low"}, {"tier": "medium"}, "high"]
    state = {"article": "Decision APIs"}
    body = {
        "model": "jev-latest",
        "state": state,
        "client_revision": "v7",
        "questions": {
            "has_topic": {"type": "noul", "criteria": {"true": {"topic": "AI"}}},
            "topic": {"type": "choice", "criteria": {"science": None, "computing": "Computers"}},
            "strength": {"type": "score", "criteria": levels},
        },
    }
    response = serve_env.client(server.app).post(
        "/v1/systemone", json=body, headers={"X-Request-Id": "abc"}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) == {"answers", "usage", "model"}
    assert data["model"] == server.model
    assert data["usage"] == {"input_tokens": 19, "output_tokens": 0}
    answers = data["answers"]
    assert list(answers) == ["has_topic", "topic", "strength"]
    assert answers["has_topic"] == {"type": "noul", "noul": 0.75}
    assert set(answers["topic"]) == {"type", "choice", "probabilities", "confidence"}
    assert answers["topic"]["choice"] == "computing"
    assert answers["topic"]["confidence"] == pytest.approx(0.8)
    assert answers["strength"]["legend"] == dict(zip(["0", "1", "2"], levels))
    assert answers["strength"]["score"] == 1.75
    assert answers["strength"]["confidence"] == pytest.approx(0.625)


def test_openai_order_optional_name_typed_options_and_usage(serve_env, scoring):
    server = serve_env.build_openai_server(AggConfig())
    scoring.return_value = [
        ReadResult([0.75, 0.25], 8, 1, label_mass=0.5, cached_tokens=2),
        ReadResult([0.6, 0.4], 10, 1, label_mass=0.9),
        ReadResult([0.75, 0.25], 9, 1, label_mass=0.8),
    ]
    response = serve_env.client(server.app).post(
        "/v1/decisions",
        json={
            "model": server.model,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "One."}]},
                {"type": "message", "role": "user", "content": "Two."},
            ],
            "questions": [
                {"type": "predicate", "instructions": "True?"},
                {
                    "type": "choice",
                    "name": "x",
                    "instructions": "Choose.",
                    "choices": [{"value": True}, {"value": "true"}],
                },
                {
                    "type": "score",
                    "name": "x",
                    "instructions": "Rate.",
                    "levels": [{"label": "low"}, {"label": "high"}],
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert response.headers["x-request-id"]
    predicate, choice, score = data["answers"]
    assert predicate == {"name": None, "type": "predicate", "probability": 0.75}
    assert choice["choice"] is True
    assert [item["value"] for item in choice["probabilities"]] == [True, "true"]
    assert choice["confidence"] == pytest.approx(0.54)
    assert score["score"] == 0.25
    assert score["probabilities"][1] == {"value": 1, "label": "high", "probability": 0.25}
    assert data["usage"] == {
        "input_tokens": 27,
        "output_tokens": 3,
        "total_tokens": 30,
        "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    _, state, questions = scoring.await_args.args
    assert state == [
        {"role": "user", "content": "One."},
        {"role": "user", "content": "Two."},
    ]
    assert questions[0].names == ["yes", "no"]
    assert questions[1].names == ["true", '"true"']


def test_user_message_boundaries_survive_rendering():
    def evidence(value):
        return OpenAIDecisionRequest(
            model="m", input=value, questions=[{"type": "predicate", "instructions": "True?"}]
        ).evidence()

    two = [{"role": "user", "content": "One"}, {"role": "user", "content": "Two"}]
    assert evidence("One") == evidence(two[:1])
    assert evidence(two) != evidence(json.dumps(two))
    assert evidence(two) != evidence("One\n\nTwo")


@pytest.mark.parametrize(
    "rows",
    [[], [ReadResult([1.0], 2)], [ReadResult([float("nan"), 1.0], 2)], [ReadResult([0.2, 0.3], 2)]],
)
def test_invalid_scorer_result_reports_server_fault(serve_env, scoring, rows):
    server = serve_env.build_openai_server(AggConfig())
    scoring.return_value = rows
    response = serve_env.client(server.app).post(
        "/v1/systemone",
        json={
            "model": server.model,
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Decision scoring failed."


def test_excludes_other_wire_formats_and_unsupported_standard_features(serve_env, scoring):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app)
    state = {"model": server.model, "state": "x", "questions": {"q": {"type": "noul"}}}
    oa = {
        "model": server.model,
        "input": "x",
        "questions": [{"type": "predicate", "instructions": "True?"}],
    }
    for route, body, status, envelope in [
        ("systemone", oa, 422, "detail"),
        ("decisions", state, 400, "error"),
        ("systemone", {**state, "questions": {"q": {"type": "box"}}}, 422, "detail"),
        ("systemone", {k: v for k, v in state.items() if k != "model"}, 422, "detail"),
        ("systemone", {**state, "state": False}, 422, "detail"),
        (
            "decisions",
            {**oa, "questions": [{"type": "yes_no", "id": "q", "question": "True?"}]},
            400,
            "error",
        ),
        ("decisions", {**oa, "questions": [{**oa["questions"][0], "id": "q"}]}, 400, "error"),
        (
            "decisions",
            {**oa, "input": [{"role": "user", "content": [{"type": "input_json", "json": {}}]}]},
            400,
            "error",
        ),
        ("decisions", {**oa, "input": [{"role": "developer", "content": "Test"}]}, 400, "error"),
        ("decisions", {**oa, "kv_hints": {}}, 400, "error"),
        ("decisions", {**oa, "questions": oa["questions"] * 65}, 400, "error"),
        (
            "decisions",
            {
                **oa,
                "questions": [
                    {
                        "type": "choice",
                        "instructions": "pick",
                        "choices": [{"value": str(i)} for i in range(27)],
                    }
                ],
            },
            400,
            "error",
        ),
        (
            "systemone",
            {
                **state,
                "questions": {
                    "q": {"type": "choice", "criteria": dict.fromkeys(str(i) for i in range(256))}
                },
            },
            422,
            "detail",
        ),
    ]:
        res = client.post(f"/v1/{route}", json=body)
        assert res.status_code == status, res.text
        assert envelope in res.json()
    for field in (
        "assets",
        "images",
        "audio",
        "videos",
        "instructions",
        "chat_template_kwargs",
        "joint",
        "priority",
        "request_id",
        "cache_salt",
    ):
        assert client.post("/v1/systemone", json={**state, field: None}).status_code == 422
    for field in (
        "prompt_cache_key",
        "prompt_cache_options",
        "joint",
        "cache_salt",
        "temperature",
        "return_prompt_token_ids",
    ):
        assert client.post("/v1/decisions", json={**oa, field: None}).status_code == 400
    for route, body in [("systemone", state), ("decisions", oa)]:
        malformed = client.post(
            f"/v1/{route}", content=b'{"questions":', headers={"content-type": "application/json"}
        )
        assert malformed.status_code == 400
        assert ("detail" if route == "systemone" else "error") in malformed.json()
        wire = json.dumps(body).encode()
        for coding, encoded in [
            ("gzip", gzip.compress(wire)),
            ("zstd", zstd.compress(wire)),
        ]:
            res = client.post(
                f"/v1/{route}",
                content=encoded,
                headers={"content-type": "application/json", "content-encoding": coding},
            )
            assert res.status_code == 415, res.text
        res = client.post(
            f"/v1/{route}",
            files=[
                ("request", (None, wire, "application/json")),
                ("photo", ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")),
            ],
        )
        assert res.status_code == 415, res.text
        res = client.post(f"/v1/{route}", json=body, headers={"x-trtllm-msgpack": "1"})
        assert res.status_code == 415, res.text
    scoring.assert_not_awaited()


def test_context_and_model_errors_keep_protocol_shape(serve_env, scoring):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app)
    state = {"model": "unserved", "state": "text", "questions": {"q": {"type": "noul"}}}
    res = client.post("/v1/systemone", json=state)
    assert res.status_code == 404 and "detail" in res.json()
    oa = {
        "model": "unserved",
        "input": "",
        "questions": [{"type": "predicate", "instructions": "True?"}],
    }
    res = client.post("/v1/decisions", json=oa)
    assert res.status_code == 404 and res.json()["error"]["code"] == "model_not_found"
    assert res.headers["x-request-id"]
    scoring.side_effect = ValueError("maximum context length")
    res = client.post("/v1/decisions", json={**oa, "model": server.model})
    assert res.status_code == 400 and "maximum context" in res.json()["error"]["message"]
    scoring.side_effect = QueueFullError("full")
    res = client.post("/v1/systemone", json={**state, "model": server.model})
    assert res.status_code == 429 and res.json()["detail"] == "full"


@pytest.mark.asyncio
async def test_disconnect_cancels_inference(monkeypatch):
    started, finished = asyncio.Event(), asyncio.Event()

    async def score(*args):
        started.set()
        try:
            await asyncio.Future()
        finally:
            finished.set()

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    monkeypatch.setattr(routes, "score_decision_questions", score)
    assert await routes._read(Request({"type": "http"}, receive), None, "", []) is None
    assert finished.is_set()
