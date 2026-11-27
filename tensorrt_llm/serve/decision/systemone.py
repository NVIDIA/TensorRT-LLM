# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named questions and answers for the TypeSafe System One text endpoint.

See https://docs.typesafe.ai/api/system-one
For choice and score confidence, see https://docs.typesafe.ai/confidence
"""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .scoring import ReadQuestion, ReadResult

Text = str | dict[str, JsonValue] | list[JsonValue]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _NoulCriteria(_Model):
    true: Text | None = None
    false: Text | None = None


class _SystemQuestion(_Model):
    instructions: Text | None = None


class NoulQuestion(_SystemQuestion):
    type: Literal["noul"]
    criteria: _NoulCriteria | None = None


class SystemChoiceQuestion(_SystemQuestion):
    type: Literal["choice"]
    criteria: dict[str, Text | None] = Field(min_length=1, max_length=255)


class SystemScoreQuestion(_SystemQuestion):
    type: Literal["score"]
    criteria: list[Text] = Field(min_length=1, max_length=10)


SystemQuestion = Annotated[
    NoulQuestion | SystemChoiceQuestion | SystemScoreQuestion, Field(discriminator="type")
]


class SystemOneRequest(_Model):
    model_config = ConfigDict(extra="ignore")
    model: str
    state: Text
    questions: dict[str, SystemQuestion] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_fields(cls, data):
        # TypeSafe permits client metadata. Fields that change the evidence or
        # execution must not be silently treated as metadata.
        if isinstance(data, dict):
            unsupported = data.keys() & {
                "assets",
                "images",
                "audio",
                "videos",
                "instructions",
                "chat_template_kwargs",
                "joint",
                "cache_salt",
                "priority",
                "request_id",
                "temperature",
                "prompt_format_version",
                "return_prompt_token_ids",
            }
            if unsupported:
                raise ValueError(f"Unsupported System One fields: {sorted(unsupported)}")
        return data


class NoulAnswer(_Model):
    type: Literal["noul"]
    noul: float


class _DistributionAnswer(_Model):
    probabilities: dict[str, float]
    confidence: float


class SystemChoiceAnswer(_DistributionAnswer):
    type: Literal["choice"]
    choice: str


class SystemScoreAnswer(_DistributionAnswer):
    type: Literal["score"]
    score: float
    legend: dict[str, Text]


class DecisionUsage(_Model):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(_Model):
    model: str
    answers: dict[str, NoulAnswer | SystemChoiceAnswer | SystemScoreAnswer]
    usage: DecisionUsage


def read_question(question: SystemQuestion) -> ReadQuestion:
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        descriptions = [criteria.true, criteria.false] if criteria else [None, None]
        return ReadQuestion("noul", question.instructions, ["yes", "no"], descriptions)
    if isinstance(question, SystemChoiceQuestion):
        return ReadQuestion(
            "choice",
            question.instructions,
            list(question.criteria),
            list(question.criteria.values()),
        )
    names = [str(i) for i in range(len(question.criteria))]
    return ReadQuestion("score", question.instructions, names, question.criteria)


def _system_confidence(kind: str, p: list[float], winner: int) -> float:
    size = len(p)
    if size == 1:
        return 1.0
    if kind == "score":
        centre = (size - 1) / 2
        spread = math.fsum(abs(i - winner) * value for i, value in enumerate(p))
        uniform_spread = sum(abs(i - centre) for i in range(size)) / size
        return max(0.0, 1.0 - spread / uniform_spread)
    return min(1.0, max(0.0, (size * p[winner] - 1) / (size - 1)))


def answer(q: ReadQuestion, result: ReadResult) -> dict:
    p = result.probabilities
    if q.kind == "noul":
        return {"type": q.kind, "noul": p[0]}
    winner = p.index(max(p))
    values = {
        "type": q.kind,
        "probabilities": dict(zip(q.names, p, strict=True)),
        "confidence": _system_confidence(q.kind, p, winner),
    }
    if q.kind == "choice":
        return {**values, "choice": q.names[winner]}
    return {
        **values,
        "score": math.fsum(i * prob for i, prob in enumerate(p)),
        "legend": dict(zip(q.names, q.descriptions, strict=True)),
    }
