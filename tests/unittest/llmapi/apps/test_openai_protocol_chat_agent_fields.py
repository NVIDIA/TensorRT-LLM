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
"""ChatCompletionRequest accepts `service_tier` and `client_metadata`.

The request schema rejects unknown fields (extra="forbid"), so a field that agent clients send
must be declared for their requests to validate. Neither field changes generation.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

pytestmark = pytest.mark.cpu_only

# Shaped like the map Codex CLI sends: string keys, string values (one holding a JSON document).
_CLIENT_METADATA = {
    "session_id": "session-1",
    "x-codex-turn-metadata": '{"turn_id": "turn-1"}',
}


def _request(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **fields)


def _error_kinds(caught: pytest.ExceptionInfo) -> set[tuple[str, str]]:
    """(field, error type) of each validation error."""
    return {(error["loc"][0], error["type"]) for error in caught.value.errors()}


@pytest.mark.parametrize("tier", ["auto", "default", "flex", "scale", "priority", "fast", None])
def test_service_tier_accepted(tier):
    assert _request(service_tier=tier).service_tier == tier, (
        "every service_tier the OpenAI API defines for Chat Completions must validate"
    )


def test_service_tier_rejects_an_unknown_tier():
    with pytest.raises(ValidationError) as caught:
        _request(service_tier="turbo")
    assert _error_kinds(caught) == {("service_tier", "literal_error")}, (
        "an unknown tier must be rejected for its value, not as an undeclared field"
    )


def test_client_metadata_accepted():
    assert _request(client_metadata=_CLIENT_METADATA).client_metadata == _CLIENT_METADATA, (
        "a client_metadata map of strings must validate and be kept as sent"
    )


@pytest.mark.parametrize(
    "metadata",
    [{"turn": 1}, {"turn": {"id": "turn-1"}}, ["turn-1"]],
    ids=["number_value", "object_value", "list"],
)
def test_client_metadata_is_a_map_of_strings(metadata):
    with pytest.raises(ValidationError) as caught:
        _request(client_metadata=metadata)
    kinds = _error_kinds(caught)
    assert kinds and kinds <= {
        ("client_metadata", "string_type"),
        ("client_metadata", "dict_type"),
    }, "client_metadata must be rejected for its shape, not as an undeclared field"


def test_both_fields_validate_from_a_json_body():
    body = (
        '{"model": "m", "messages": [{"role": "user", "content": "hi"}],'
        ' "service_tier": "flex", "client_metadata": {"session_id": "s-1"}}'
    )
    request = ChatCompletionRequest.model_validate_json(body)
    assert request.service_tier == "flex"
    assert request.client_metadata == {"session_id": "s-1"}


def test_accepted_fields_do_not_change_sampling_params():
    plain = _request().to_sampling_params()
    tagged = _request(service_tier="flex", client_metadata=_CLIENT_METADATA).to_sampling_params()
    assert tagged == plain, "service_tier and client_metadata are accepted and ignored"


@pytest.mark.parametrize("field", ["service_tiers", "client_meta_data"])
def test_unknown_fields_remain_rejected(field):
    with pytest.raises(ValidationError) as caught:
        _request(**{field: "x"})
    assert _error_kinds(caught) == {(field, "extra_forbidden")}
