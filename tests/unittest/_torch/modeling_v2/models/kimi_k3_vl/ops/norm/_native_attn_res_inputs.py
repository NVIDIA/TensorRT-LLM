# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and comparison helpers shared by the attn_res op tests in this directory."""

import zlib

import torch

HIDDEN_SIZE = 7168


def make_inputs(cell) -> dict:
    """bf16 inputs at the scales of the built-in op tests, seeded by the cell: layer_residual [T, 1, H], block_residual
    [N - 1, T, 1, H], res_weight, rms_weight and output_rms_weight [H], and an addend shaped like layer_residual."""
    spec = cell.spec
    assert spec["hidden"] == HIDDEN_SIZE
    tokens, candidates = spec["tokens"], spec["candidates"]
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    return dict(
        layer_residual=randn(tokens, 1, HIDDEN_SIZE) * 0.05,
        block_residual=randn(candidates - 1, tokens, 1, HIDDEN_SIZE) * 0.05,
        res_weight=randn(HIDDEN_SIZE) * 0.02,
        rms_weight=1 + randn(HIDDEN_SIZE) * 0.02,
        output_rms_weight=1 + randn(HIDDEN_SIZE) * 0.02,
        addend=randn(tokens, 1, HIDDEN_SIZE) * 0.05,
    )


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers, so that equality means identical bits."""
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def similarity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Cosine similarity and relative L2 error, the metric of the built-in op tests."""
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    return cosine, relative_l2
