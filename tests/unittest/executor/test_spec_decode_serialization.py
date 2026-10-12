# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serialization tests for the per-request speculative_decoding field.

Per-request spec-decode stats are off by default, so the overwhelmingly common
case is a response that carries none. The field must then be *absent* from the
wire, not present as null: the serving layer dumps responses several different
ways -- plain ``model_dump()`` for non-streaming chat and completions,
``model_dump_json(exclude_unset=False)`` for the completions stream, and
``model_dump_json(exclude_none=True)`` for the chat stream -- and only the last
would have dropped a null on its own. Without a field-scoped serializer every
user would gain a ``"speculative_decoding": null`` key on responses whether or
not they enabled the feature.

The omission is deliberately scoped to this one field: unrelated optional
fields that clients may rely on being present must still serialize as null.

GPU-free logic, but importing the serving protocol is unproven on the CPU-only
CI stage, so this file is wired into a GPU list (l0_a10.yml) like its siblings.
"""

import pytest
from pytest import param

from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    DeltaMessage,
    SpeculativeDecodingStats,
)

STATS = SpeculativeDecodingStats(
    acceptance_rate=0.5,
    total_accepted_draft_tokens=30,
    total_draft_tokens=60,
    num_spec_steps=20,
    acceptance_histogram=[8, 0, 6, 6],
    num_spec_tokens=3,
)


def _choices(stats):
    """One instance of each choice model that can carry the field."""
    return {
        "completion": CompletionResponseChoice(index=0, text="hi", speculative_decoding=stats),
        "completion_stream": CompletionResponseStreamChoice(
            index=0, text="hi", speculative_decoding=stats
        ),
        "chat": ChatCompletionResponseChoice(
            index=0, message=ChatMessage(role="assistant", content="hi"), speculative_decoding=stats
        ),
        "chat_stream": ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content="hi"), speculative_decoding=stats
        ),
    }


ABSENT = _choices(None)
PRESENT = _choices(STATS)


@pytest.mark.parametrize("name", sorted(ABSENT))  # fmt: skip
class TestAbsentStatsAreOmitted:
    """Every dump style the serving layer uses must omit the key."""

    def test_model_dump(self, name):
        assert "speculative_decoding" not in ABSENT[name].model_dump()

    def test_model_dump_json(self, name):
        assert "speculative_decoding" not in ABSENT[name].model_dump_json()

    def test_model_dump_json_exclude_unset_false(self, name):
        # The completions stream serializes this way.
        assert "speculative_decoding" not in ABSENT[name].model_dump_json(exclude_unset=False)

    def test_model_dump_json_exclude_none(self, name):
        # The chat stream serializes this way; it would have dropped the null
        # anyway, but the field-scoped serializer must not conflict with it.
        assert "speculative_decoding" not in ABSENT[name].model_dump_json(exclude_none=True)


@pytest.mark.parametrize("name", sorted(PRESENT))  # fmt: skip
class TestPresentStatsSurvive:
    """Omission must not swallow real statistics."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            param({}, id="default"),
            param({"exclude_unset": False}, id="exclude_unset_false"),
        ],
    )  # fmt: skip
    def test_round_trips(self, name, kwargs):
        dumped = PRESENT[name].model_dump_json(**kwargs)
        assert '"speculative_decoding"' in dumped
        assert '"num_spec_steps":20' in dumped

    def test_values_intact(self, name):
        data = PRESENT[name].model_dump()["speculative_decoding"]
        assert data["acceptance_histogram"] == [8, 0, 6, 6]
        assert data["total_accepted_draft_tokens"] == 30


@pytest.mark.parametrize("name", sorted(ABSENT))  # fmt: skip
def test_unrelated_optional_fields_still_serialize_as_null(name):
    """The exclusion is scoped to one field, not blanket exclude_none.

    ``avg_decoded_tokens_per_iter`` is the neighbouring optional field on these
    same models; clients parsing it must keep seeing it.
    """
    assert ABSENT[name].model_dump()["avg_decoded_tokens_per_iter"] is None
    assert '"avg_decoded_tokens_per_iter":null' in ABSENT[name].model_dump_json()
