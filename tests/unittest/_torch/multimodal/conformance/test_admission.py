# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the executor's multimodal admission rule at the request boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.multimodal.conformance.models import _V2_BASE
from _torch.pyexecutor.executor_test_utils import make_partial_executor

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

pytestmark = pytest.mark.cpu_only


def _request(data: dict) -> LlmRequest:
    request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=[1],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        draft_tokens=None,
    )
    request.py_multimodal_data = data
    return request


@pytest.mark.skipif(
    _V2_BASE is None, reason="Central V2 admission lands with the composite in #22450"
)
def test_v2_request_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _V2_BASE is not None
    # An encoder-free prefill worker needs only the decoder vocabulary for admission.
    model = _V2_BASE.__new__(_V2_BASE)
    torch.nn.Module.__init__(model)
    model.llm = SimpleNamespace(lm_head=SimpleNamespace(num_embeddings=8))
    executor = make_partial_executor(
        model_engine=SimpleNamespace(model=model), _validate_request_budget=Mock()
    )
    image = {"image": {"pixel_values": torch.zeros(1)}}
    embeds = {"multimodal_embedding": torch.zeros(1, 4)}
    mrope = {"mrope_config": {"mrope_position_deltas": torch.zeros(1, 1)}}
    monkeypatch.delenv("TLLM_MULTIMODAL_DISAGGREGATED", raising=False)
    with pytest.raises(ValueError, match="no local multimodal encoder"):
        executor._validate_request(_request(image))
    with pytest.raises(ValueError, match="Precomputed multimodal embeddings"):
        executor._validate_request(_request(embeds))
    executor._validate_request(_request(mrope))

    monkeypatch.setenv("TLLM_MULTIMODAL_DISAGGREGATED", "1")
    executor._validate_request(_request(embeds))
    with pytest.raises(ValueError, match="no local multimodal encoder"):
        executor._validate_request(_request({**image, **embeds}))


def test_v1_request_uses_the_model_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_MULTIMODAL_DISAGGREGATED", raising=False)
    validate = Mock()
    model = SimpleNamespace(validate_multimodal_request_data=validate)
    executor = make_partial_executor(
        model_engine=SimpleNamespace(model=model), _validate_request_budget=Mock()
    )
    embeds = {"multimodal_embedding": torch.zeros(1, 4)}
    executor._validate_request(_request(embeds))
    validate.assert_called_once_with(embeds)
    validate.side_effect = ValueError("unsupported image")
    with pytest.raises(ValueError, match="unsupported image"):
        executor._validate_request(_request({"image": {"pixel_values": torch.zeros(1)}}))
