# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text request, question and answer types for OpenAI Decisions.

See https://developers.openai.com/api/reference/resources/decisions/methods/create
"""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator
from pydantic.json_schema import SkipJsonSchema

from ..openai_protocol import ResponseUsage
from .scoring import ReadQuestion, ReadResult

# The Decisions schema expresses these maxLength limits in characters.
ShortText = Annotated[StrictStr, Field(max_length=1_048_576)]
InputText = Annotated[StrictStr, Field(max_length=10_485_760)]
ChoiceValue = StrictStr | StrictBool


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextPart(_Model):
    type: Literal["input_text"]
    text: InputText


class UserMessage(_Model):
    type: Literal["message"] = "message"
    role: Literal["user"]
    content: InputText | Annotated[list[TextPart], Field(max_length=16384)]


class _OpenAIQuestion(_Model):
    name: ShortText | SkipJsonSchema[None] = None
    instructions: ShortText

    @field_validator("name")
    @classmethod
    def _name_if_provided(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("name must be a string when provided")
        return value


class PredicateQuestion(_OpenAIQuestion):
    type: Literal["predicate"]


class ChoiceOption(_Model):
    value: ShortText | StrictBool
    description: ShortText = ""


class OpenAIChoiceQuestion(_OpenAIQuestion):
    type: Literal["choice"]
    choices: list[ChoiceOption] = Field(min_length=2, max_length=255)

    @field_validator("choices")
    @classmethod
    def _distinct_values(cls, choices: list[ChoiceOption]) -> list[ChoiceOption]:
        if len({item.value for item in choices}) != len(choices):
            raise ValueError("choice values must be distinct")
        return choices


class ScoreLevel(_Model):
    label: ShortText
    description: ShortText = ""


class OpenAIScoreQuestion(_OpenAIQuestion):
    type: Literal["score"]
    levels: list[ScoreLevel] = Field(min_length=2, max_length=10)


OpenAIQuestion = Annotated[
    PredicateQuestion | OpenAIChoiceQuestion | OpenAIScoreQuestion, Field(discriminator="type")
]


class OpenAIDecisionRequest(_Model):
    model: ShortText
    input: InputText | Annotated[list[UserMessage], Field(max_length=131072)]
    questions: list[OpenAIQuestion] = Field(min_length=1, max_length=64)
    safety_identifier: Annotated[StrictStr, Field(max_length=128)] | None = None

    def evidence(self) -> list[dict[str, str]]:
        if isinstance(self.input, str):
            return [{"role": "user", "content": self.input}]
        return [
            {
                "role": m.role,
                "content": m.content
                if isinstance(m.content, str)
                else "\n".join(p.text for p in m.content),
            }
            for m in self.input
        ]


class PredicateAnswer(_Model):
    type: Literal["predicate"]
    name: str | None
    probability: float


class ChoiceProbability(_Model):
    value: ChoiceValue
    probability: float


class OpenAIChoiceAnswer(_Model):
    type: Literal["choice"]
    name: str | None
    choice: ChoiceValue
    probabilities: list[ChoiceProbability]
    confidence: float


class ScoreProbability(_Model):
    value: int
    label: str
    probability: float


class OpenAIScoreAnswer(_Model):
    type: Literal["score"]
    name: str | None
    score: float
    probabilities: list[ScoreProbability]
    confidence: float


class OpenAIDecisionResponse(_Model):
    model: str
    answers: list[PredicateAnswer | OpenAIChoiceAnswer | OpenAIScoreAnswer]
    usage: ResponseUsage


def read_question(question: OpenAIQuestion, max_choices: int) -> ReadQuestion:
    if isinstance(question, PredicateQuestion):
        return ReadQuestion("noul", question.instructions, ["yes", "no"], [None, None])
    if isinstance(question, OpenAIChoiceQuestion):
        if len(question.choices) > max_choices:
            raise ValueError(f"This model supports at most {max_choices} OpenAI choice candidates.")
        names = [json.dumps(c.value, ensure_ascii=False) for c in question.choices]
        descriptions = [c.description for c in question.choices]
    else:
        names = [level.label for level in question.levels]
        descriptions = [level.description for level in question.levels]
    return ReadQuestion("choice", question.instructions, names, descriptions)


def answer(q: OpenAIQuestion, result: ReadResult) -> dict:
    p = result.probabilities
    answer = dict(type=q.type, name=q.name)
    if isinstance(q, PredicateQuestion):
        return {**answer, "probability": p[0]}
    winner = p.index(max(p))
    answer["confidence"] = p[winner] * (result.label_mass if result.label_mass is not None else 1.0)
    if isinstance(q, OpenAIChoiceQuestion):
        answer.update(
            choice=q.choices[winner].value,
            probabilities=[
                {"value": option.value, "probability": prob}
                for option, prob in zip(q.choices, p, strict=True)
            ],
        )
    else:
        answer.update(
            score=sum(i * prob for i, prob in enumerate(p)),
            probabilities=[
                {"value": i, "label": level.label, "probability": prob}
                for i, (level, prob) in enumerate(zip(q.levels, p, strict=True))
            ],
        )
    return answer
