# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the executor's multimodal admission rule at the request boundary."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.multimodal.conformance.models import _V2_BASE, Key, inventory
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


_KIMI = ("KimiK3ForConditionalGeneration", "kimi_k3_vl/kimi_k3_nvfp4__sm_107__tp1")
_GLM = ("Glm5NextForConditionalGeneration", "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4")
_MINIMAX = (
    "MiniMaxM3SparseForConditionalGeneration",
    "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx",
)
_DEFERRED = pytest.mark.xfail(
    _V2_BASE is None,
    reason="Text-only target admits this payload before the central rule in #22450",
    raises=pytest.fail.Exception,
    strict=True,
)


@pytest.mark.parametrize(
    ("key", "payload"),
    [
        pytest.param(_KIMI, "image", marks=_DEFERRED),
        pytest.param(_GLM, "image"),
        pytest.param(_MINIMAX, "image", marks=_DEFERRED),
        *(
            pytest.param(key, "multimodal_embedding", marks=_DEFERRED)
            for key in (_KIMI, _GLM, _MINIMAX)
        ),
    ],
)
@pytest.mark.parametrize("split", [False, True], ids=["aggregated", "split"])
def test_registered_text_only_admission(
    key: Key, payload: str, split: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = inventory()[key]
    assert not entry.media and key[1] and entry.processor_cls
    module = import_module(
        f"tensorrt_llm._torch.modeling_v2.models.{key[1].replace('/', '.')}.modeling"
    )
    cls = getattr(module, entry.class_name)
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    engine = SimpleNamespace(model=model, input_processor=object.__new__(entry.processor_cls))
    executor = make_partial_executor(model_engine=engine, _validate_request_budget=Mock())
    monkeypatch.setenv("TLLM_MULTIMODAL_DISAGGREGATED", str(int(split)))
    executor._validate_request(
        _request({"mrope_config": {"mrope_position_deltas": torch.zeros(1, 1)}})
    )
    data = {"image": {"pixel_values": torch.zeros(1)}, "multimodal_embedding": torch.zeros(1, 4)}
    with pytest.raises(
        ValueError, match="text only|no local multimodal encoder|Precomputed multimodal embeddings"
    ):
        executor._validate_request(_request({payload: data[payload]}))
