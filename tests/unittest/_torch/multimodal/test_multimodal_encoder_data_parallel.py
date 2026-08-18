# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.models.multimodal_encoder_data_parallel import (
    EncoderDpItem,
    execute_encoder_dp_items,
    plan_encoder_dp_items,
)

pytestmark = pytest.mark.cpu_only


def test_plan_encoder_dp_items_accepts_an_empty_batch():
    plan = plan_encoder_dp_items([], num_ranks=2)

    assert plan.placements == ()
    assert plan.rank_input_token_counts == (0, 0)
    assert plan.total_output_rows == 0


def test_plan_encoder_dp_items_uses_input_cost_and_preserves_output_order():
    items = [
        EncoderDpItem(ordinal=0, input_token_count=100, output_row_count=1),
        EncoderDpItem(ordinal=1, input_token_count=10, output_row_count=100),
        EncoderDpItem(ordinal=2, input_token_count=10, output_row_count=100),
    ]

    plan = plan_encoder_dp_items(items, num_ranks=2)

    assert plan.rank_input_token_counts == (100, 20)
    assert [placement.rank for placement in plan.placements] == [0, 1, 1]
    assert [placement.global_row_start for placement in plan.placements] == [0, 1, 101]
    assert plan.total_output_rows == 201


@pytest.mark.parametrize(
    "items,error_match",
    [
        (
            [
                EncoderDpItem(ordinal=0, input_token_count=1, output_row_count=1),
                EncoderDpItem(ordinal=0, input_token_count=2, output_row_count=2),
            ],
            "ordinals must be unique",
        ),
        (
            [EncoderDpItem(ordinal=0, input_token_count=0, output_row_count=1)],
            "input_token_count must be positive",
        ),
        (
            [EncoderDpItem(ordinal=0, input_token_count=1, output_row_count=0)],
            "output_row_count must be positive",
        ),
    ],
)
def test_plan_encoder_dp_items_rejects_invalid_descriptors(items, error_match):
    with pytest.raises(ValueError, match=error_match):
        plan_encoder_dp_items(items, num_ranks=2)


def test_execute_encoder_dp_items_reconstructs_global_output():
    items = [
        EncoderDpItem(ordinal=0, input_token_count=4, output_row_count=2),
        EncoderDpItem(ordinal=1, input_token_count=3, output_row_count=1),
        EncoderDpItem(ordinal=2, input_token_count=2, output_row_count=2),
    ]
    peer_contribution = torch.tensor([[0.0], [0.0], [20.0], [30.0], [30.0]])
    prepared_ordinals: list[int] = []

    def prepare_local_inputs(local_items):
        prepared_ordinals.extend(item.ordinal for item in local_items)
        return local_items

    def encode_local_inputs(local_items):
        rows = []
        for item in local_items:
            rows.append(torch.full((item.output_row_count, 1), float((item.ordinal + 1) * 10)))
        return torch.cat(rows)

    def allreduce(tensor):
        if tensor.dtype == torch.int32:
            return tensor
        return tensor + peer_contribution

    output = execute_encoder_dp_items(
        items,
        rank=0,
        num_ranks=2,
        prepare_local_inputs=prepare_local_inputs,
        encode_local_inputs=encode_local_inputs,
        allreduce=allreduce,
        output_dim=1,
        output_dtype=torch.float32,
        output_device=torch.device("cpu"),
    )

    assert prepared_ordinals == [0]
    torch.testing.assert_close(
        output,
        torch.tensor([[10.0], [10.0], [20.0], [30.0], [30.0]]),
    )


def test_execute_encoder_dp_items_synchronizes_local_shape_failure():
    items = [EncoderDpItem(ordinal=0, input_token_count=1, output_row_count=2)]
    collective_dtypes: list[torch.dtype] = []

    def allreduce(tensor):
        collective_dtypes.append(tensor.dtype)
        return tensor

    with pytest.raises(RuntimeError, match="rank failed") as error:
        execute_encoder_dp_items(
            items,
            rank=0,
            num_ranks=1,
            prepare_local_inputs=lambda local_items: local_items,
            encode_local_inputs=lambda _: torch.zeros((1, 1)),
            allreduce=allreduce,
            output_dim=1,
            output_dtype=torch.float32,
            output_device=torch.device("cpu"),
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert collective_dtypes == [torch.int32]


def test_execute_encoder_dp_items_keeps_empty_rank_in_collectives():
    items = [EncoderDpItem(ordinal=0, input_token_count=1, output_row_count=2)]
    peer_contribution = torch.full((2, 1), 7.0)
    collective_dtypes: list[torch.dtype] = []

    def allreduce(tensor):
        collective_dtypes.append(tensor.dtype)
        if tensor.dtype == torch.int32:
            return tensor
        return tensor + peer_contribution

    output = execute_encoder_dp_items(
        items,
        rank=1,
        num_ranks=2,
        prepare_local_inputs=lambda _: pytest.fail(
            "An empty encoder-DP rank must not prepare inputs."
        ),
        encode_local_inputs=lambda _: pytest.fail(
            "An empty encoder-DP rank must not run the encoder."
        ),
        allreduce=allreduce,
        output_dim=1,
        output_dtype=torch.float32,
        output_device=torch.device("cpu"),
    )

    torch.testing.assert_close(output, peer_contribution)
    assert collective_dtypes == [torch.int32, torch.float32]


def test_execute_encoder_dp_items_stops_before_output_collective_on_peer_failure():
    items = [EncoderDpItem(ordinal=0, input_token_count=1, output_row_count=1)]
    collective_dtypes: list[torch.dtype] = []

    def allreduce(tensor):
        collective_dtypes.append(tensor.dtype)
        if tensor.dtype == torch.int32:
            return torch.ones_like(tensor)
        pytest.fail("Output collective must not run after a peer failure.")

    with pytest.raises(RuntimeError, match="peer rank failed"):
        execute_encoder_dp_items(
            items,
            rank=1,
            num_ranks=2,
            prepare_local_inputs=lambda _: [],
            encode_local_inputs=lambda _: torch.empty((0, 1)),
            allreduce=allreduce,
            output_dim=1,
            output_dtype=torch.float32,
            output_device=torch.device("cpu"),
        )

    assert collective_dtypes == [torch.int32]
