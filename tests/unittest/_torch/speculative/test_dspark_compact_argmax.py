# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL compact TP argmax against a full-vocabulary FP32 argmax."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dspark import tp_argmax as compact

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _fixture(rows: int, local_vocab: int, ranks: int, bias_dtype: torch.dtype) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(100392)
    parts = -(-local_vocab // compact.WORDS_PER_PART)

    def strided(dtype):
        # Rows are strided like the [B, K, V] projection's per-step slices.
        storage = torch.randn((rows, local_vocab + 8), device="cuda", generator=generator)
        return storage.to(dtype)[:, :local_vocab]

    return dict(
        base=[strided(torch.bfloat16) for _ in range(ranks)],
        bias=[strided(bias_dtype) for _ in range(ranks)],
        packets=[
            torch.empty((rows, parts, 2), dtype=torch.float32, device="cuda") for _ in range(ranks)
        ],
        output=torch.empty((rows,), dtype=torch.int64, device="cuda"),
        local_vocab=local_vocab,
        global_vocab=local_vocab * ranks,
    )


def _reference(data: dict) -> torch.Tensor:
    logits = torch.cat(
        [base.float() + bias.float() for base, bias in zip(data["base"], data["bias"])], dim=1
    )
    return logits[:, : data["global_vocab"]].argmax(dim=-1)


def _run(data: dict) -> None:
    for rank, (base, bias, packet) in enumerate(zip(data["base"], data["bias"], data["packets"])):
        compact.local_candidates(
            base, bias, rank * data["local_vocab"], data["global_vocab"], packet
        )
    compact.global_argmax(torch.cat(data["packets"], dim=0), data["output"], data["global_vocab"])


@pytest.mark.parametrize(
    "rows,local_vocab,ranks", [(1, 8, 1), (5, 4104, 4), (2, 520, 8), (128, 32320, 4)]
)
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_matches_full_vocab_argmax(rows, local_vocab, ranks, bias_dtype) -> None:
    data = _fixture(rows, local_vocab, ranks, bias_dtype)
    _run(data)
    assert torch.equal(data["output"], _reference(data))


@pytest.mark.parametrize("case", ["ties", "nan", "promotion"])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_global_order(case: str, bias_dtype: torch.dtype) -> None:
    data = _fixture(2, 8200, 4, bias_dtype)
    for base, bias in zip(data["base"], data["bias"]):
        base.fill_(-8)
        bias.zero_()
    if case == "ties":
        # Ties across ranks and parts choose the smallest global id.
        data["base"][0][:, [4096, 8192]] = 12
        data["base"][2][:, 0] = 12
    elif case == "nan":
        # NaN wins, as in torch.argmax.
        data["base"][0][:, 4096] = float("nan")
        data["bias"][3][:, 7] = float("nan")
    else:
        # A BF16 add would round the small second bias away; the FP32 sum keeps it.
        for base in data["base"]:
            base.fill_(1)
        data["bias"][1][:, 4096] = 2.0**-10
    _run(data)
    assert torch.equal(data["output"], _reference(data))


@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_vocab_padding_is_excluded(bias_dtype: torch.dtype) -> None:
    data = _fixture(3, 8200, 4, bias_dtype)
    data["global_vocab"] = 2 * data["local_vocab"] + 7
    data["base"][2][:, 7:] = float("nan")
    data["base"][3].fill_(float("nan"))
    _run(data)
    assert torch.equal(data["output"], _reference(data))
