# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from _torch.pyexecutor.engine_test_utils import _AttentionMetadata

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs


@pytest.mark.cpu_only
def test_model_caller_uses_current_forward_and_restores_outer_attribute_context() -> None:
    metadata = _AttentionMetadata()
    spec_metadata = object()
    attrs = {}
    outer_attrs = {"outer": True}
    model = SimpleNamespace(model_config=SimpleNamespace(extra_attrs={"model_flag": True}))

    def first_forward(**kwargs: object) -> str:
        assert get_model_extra_attrs() is attrs
        assert attrs["attention_metadata"]() is metadata
        assert attrs["spec_metadata"] is spec_metadata
        assert attrs["model_flag"] is True
        return "first"

    model.forward = Mock(side_effect=first_forward)
    caller = ModelCaller(model)
    with model_extra_attrs(outer_attrs):
        with model_extra_attrs(attrs):
            assert caller(attn_metadata=metadata, spec_metadata=spec_metadata) == "first"
            model.forward.assert_called_once_with(
                attn_metadata=metadata, spec_metadata=spec_metadata
            )
            model.forward = Mock(return_value="replacement")
            assert caller(attn_metadata=metadata) == "replacement"
            model.forward.assert_called_once_with(attn_metadata=metadata)
            assert attrs["spec_metadata"] is None
            model.forward.side_effect = RuntimeError("model failure")
            with pytest.raises(RuntimeError, match="model failure"):
                caller(attn_metadata=metadata)
        assert get_model_extra_attrs() is outer_attrs


@pytest.mark.cpu_only
def test_model_caller_borrows_live_compile_streams_and_events() -> None:
    streams = Backend.Streams()
    backend = SimpleNamespace(events=Backend.Events())
    metadata = _AttentionMetadata()
    current_stream = object()
    attrs = {}

    def forward(**kwargs: object) -> tuple[int, int]:
        assert get_model_extra_attrs() is attrs
        assert attrs["aux_streams"]() is streams
        assert attrs["events"]() is backend.events
        assert attrs["global_stream"] is current_stream
        return len(attrs["aux_streams"]()), len(attrs["events"]())

    model = SimpleNamespace(model_config=SimpleNamespace(extra_attrs={}), forward=forward)
    caller = ModelCaller(model, compile_backend=backend, aux_streams=streams)
    with patch("torch.cuda.current_stream", return_value=current_stream), model_extra_attrs(attrs):
        assert caller(attn_metadata=metadata) == (0, 0)
        streams.append(object())
        backend.events = Backend.Events([object(), object()])
        assert caller(attn_metadata=metadata) == (1, 2)


@pytest.mark.cpu_only
def test_model_caller_requires_streams_with_compile_backend() -> None:
    with pytest.raises(ValueError, match="requires its auxiliary stream container"):
        ModelCaller(SimpleNamespace(), compile_backend=SimpleNamespace(events=Backend.Events()))
