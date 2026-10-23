# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "_modeling_v2_lockstep_test", Path(__file__).with_name("_lockstep.py")
)
lockstep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lockstep)
pytestmark = pytest.mark.cpu_only


class _TwoRankExchange:
    def __init__(self) -> None:
        self.first: object | None = None

    def allgather(self, payload: object) -> list[object]:
        if self.first is None:
            self.first = payload
        return [self.first, payload]


def _agree(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> bool:
    rank = lockstep.Rank.__new__(lockstep.Rank)
    rank.comm = _TwoRankExchange()
    assert rank.same_on_ranks(*first)
    return rank.same_on_ranks(*second)


@pytest.mark.parametrize(
    "first,second",
    [
        (torch.tensor([[1, 2]], dtype=torch.int16), torch.tensor([[2, 1]], dtype=torch.int16)),
        (torch.tensor([0.0]), torch.tensor([-0.0])),
        (torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.float32)),
        (torch.tensor([[1, 2]], dtype=torch.int16), torch.tensor([1, 2], dtype=torch.int16)),
        (
            torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32),
            torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32),
        ),
    ],
    ids=["permuted-words", "signed-zero", "dtype", "shape", "nan-payload"],
)
def test_rank_equality_rejects_different_bits_or_metadata(
    first: torch.Tensor, second: torch.Tensor
) -> None:
    assert not _agree((first,), (second,))


def test_rank_equality_preserves_tensor_boundaries_and_order() -> None:
    first = torch.tensor([[1, 2]], dtype=torch.int16)
    second = torch.tensor([[3, 4]], dtype=torch.int16)
    assert not _agree((first, second), (second, first))
    assert not _agree((first, second), (first,))
    assert not _agree((), (first,))


def test_rank_equality_accepts_identical_values_in_noncontiguous_tensors() -> None:
    words = torch.arange(12, dtype=torch.int16).reshape(3, 4).t()
    nan = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    assert _agree((words, nan, torch.tensor(7)), (words.contiguous(), nan.clone(), torch.tensor(7)))
    assert _agree((), ())
