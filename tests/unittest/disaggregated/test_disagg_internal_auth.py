# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from tensorrt_llm.serve.disagg_auth import (
    INTERNAL_DISAGG_AUTH_HEADER,
    build_internal_disagg_auth_headers,
    validate_internal_disagg_request,
)
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams


def _make_request(encoded_opaque_state: str | None) -> CompletionRequest:
    return CompletionRequest(
        model="test-model",
        prompt="hello",
        stream=False,
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only",
            encoded_opaque_state=encoded_opaque_state,
        ),
    )


def test_no_opaque_state_does_not_require_internal_auth():
    request = _make_request(encoded_opaque_state=None)

    assert build_internal_disagg_auth_headers(None, request) == {}
    validate_internal_disagg_request(None, request, {})


def test_opaque_state_requires_internal_auth_key():
    request = _make_request(encoded_opaque_state="b3BhcXVl")

    with pytest.raises(ValueError, match="requires authenticated"):
        build_internal_disagg_auth_headers(None, request)
    with pytest.raises(ValueError, match="requires authenticated"):
        validate_internal_disagg_request(None, request, {})


def test_opaque_state_accepts_valid_internal_auth_header():
    request = _make_request(encoded_opaque_state="b3BhcXVl")
    headers = build_internal_disagg_auth_headers("secret", request)

    assert INTERNAL_DISAGG_AUTH_HEADER in headers
    validate_internal_disagg_request("secret", request, headers)


def test_opaque_state_rejects_tampered_payload():
    request = _make_request(encoded_opaque_state="b3BhcXVl")
    headers = build_internal_disagg_auth_headers("secret", request)
    request.disaggregated_params.encoded_opaque_state = "dGFtcGVyZWQ="

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", request, headers)
