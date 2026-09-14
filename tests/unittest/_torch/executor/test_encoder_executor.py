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


def test_batch_forward_hands_the_packed_batch_to_the_engine() -> None:
    expected = {"logits": torch.tensor([1.0, 2.0])}
    model_engine = Mock()
    model_engine.forward_encode_batch.return_value = expected
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
    model_engine.forward_encode_batch.assert_called_once_with(
        [11, 12, 21, 22, 23],
        [2, 3],
        multi_item_part_lens=[[1, 1], [2, 1]],
        token_type_ids=inputs["token_type_ids"],
        gather_context_logits=True,
    )
    # The tokens must not be split into requests only to be repacked downstream.
    model_engine.forward.assert_not_called()
