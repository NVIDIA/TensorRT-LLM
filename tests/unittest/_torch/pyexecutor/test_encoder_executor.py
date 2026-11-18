# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.encoder_executor import EncoderExecutor
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import PackedInputs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def test_create_rejects_more_than_one_rank() -> None:
    with pytest.raises(ValueError, match="single GPU"):
        EncoderExecutor.create(
            Mock(),
            model_path="model",
            checkpoint_loader=Mock(),
            mapping=Mapping(world_size=2, tp_size=2),
            dist=Mock(),
        )


def test_batch_forward_hands_the_packed_batch_to_the_runner() -> None:
    expected = {"logits": torch.tensor([1.0, 2.0])}
    runner = Mock()
    runner.forward_packed.return_value = expected
    executor = object.__new__(EncoderExecutor)
    executor._runner = runner
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
    runner.forward_packed.assert_called_once_with(
        PackedInputs(
            input_ids=[11, 12, 21, 22, 23],
            sequence_lengths=[2, 3],
            multi_item_part_lens=[[1, 1], [2, 1]],
            model_inputs={"token_type_ids": inputs["token_type_ids"]},
            gather_context_logits=True,
        ),
    )
