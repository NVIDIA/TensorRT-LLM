# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.encoder_executor import EncoderExecutor

pytestmark = pytest.mark.cpu_only


def test_encoder_executor_warms_up_through_model_engine() -> None:
    model_engine = Mock()

    executor = EncoderExecutor(model_engine, SimpleNamespace())

    assert executor.model_engine is model_engine
    model_engine.warmup.assert_called_once_with(None)


def test_batch_forward_adapts_inputs_to_scheduled_requests_once() -> None:
    expected = {"logits": torch.tensor([1.0, 2.0])}
    model_engine = Mock()
    model_engine.forward.return_value = expected
    executor = object.__new__(EncoderExecutor)
    executor.model_engine = model_engine
    inputs = {
        "input_ids": torch.tensor([11, 12, 21, 22, 23]),
        "seq_lens": torch.tensor([2, 3]),
        "multi_item_part_lens": [[1, 1], [2, 1]],
        "token_type_ids": torch.tensor([0, 0, 1, 1, 1]),
    }

    actual = executor.batch_forward(inputs, gather_context_logits=True)

    assert actual is expected
    assert set(inputs) == {
        "input_ids",
        "seq_lens",
        "multi_item_part_lens",
        "token_type_ids",
    }
    scheduled_requests = model_engine.forward.call_args.args[0]
    assert [request.get_tokens(0) for request in scheduled_requests.context_requests] == [
        [11, 12],
        [21, 22, 23],
    ]
    assert [request.py_multi_item_part_lens for request in scheduled_requests.context_requests] == [
        [1, 1],
        [2, 1],
    ]
    model_engine.forward.assert_called_once_with(
        scheduled_requests,
        resource_manager=None,
        token_type_ids=inputs["token_type_ids"],
        gather_context_logits=True,
    )


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"input_ids": [1, 2], "seq_lens": [1]}, "sum of seq_lens"),
        (
            {
                "input_ids": [1, 2],
                "seq_lens": [1, 1],
                "multi_item_part_lens": [[1]],
            },
            "provided for all requests or for none",
        ),
    ],
)
def test_encoder_executor_rejects_inconsistent_request_boundaries(
    inputs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EncoderExecutor._build_scheduled_requests(inputs)
