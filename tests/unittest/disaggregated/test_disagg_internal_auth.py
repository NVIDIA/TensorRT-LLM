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
    build_internal_disagg_auth_headers,
    request_requires_internal_disagg_auth,
    validate_internal_disagg_request,
)
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams


def _generation_request(endpoint: str = "tcp://10.0.0.1:5000"):
    return CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only",
            ctx_request_id=1,
            disagg_request_id=2,
            ctx_info_endpoint=endpoint,
        ),
    )


def test_ctx_info_endpoint_rejects_missing_auth_header():
    request = _generation_request()

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", request, {})


def test_ctx_info_endpoint_accepts_signed_proxy_request():
    request = _generation_request()
    headers = build_internal_disagg_auth_headers("secret", request)

    validate_internal_disagg_request("secret", request, headers)


def test_ctx_info_endpoint_rejects_tampered_request():
    request = _generation_request()
    headers = build_internal_disagg_auth_headers("secret", request)
    tampered_request = _generation_request("tcp://10.0.0.2:5000")

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", tampered_request, headers)


def test_ctx_info_endpoint_rejects_missing_server_key():
    request = _generation_request()
    headers = build_internal_disagg_auth_headers("secret", request)

    with pytest.raises(ValueError, match="requires authenticated"):
        validate_internal_disagg_request(None, request, headers)


def test_disagg_auth_not_required_without_ctx_info_endpoint():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only",
            ctx_request_id=1,
            disagg_request_id=2,
        ),
    )

    assert not request_requires_internal_disagg_auth(request)
    validate_internal_disagg_request(None, request, {})
