# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention.sol_predictor import (
    SOL_BLOCK_SIZE,
    sol_predictor,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SOL predictor requires CUDA")
@pytest.mark.parametrize("cell", sol_predictor.CELLS, ids=lambda cell: cell.why)
def test_sol_predictor(cell):
    length = cell.spec["tokens"]
    rng = torch.Generator(device="cuda").manual_seed(123)
    shape = (2, length, 2, 128)
    noise = torch.randint(-4, 5, shape, device="cuda", generator=rng) / 32
    block = torch.arange(length, device="cuda") // SOL_BLOCK_SIZE
    # The weakest positive keys fall between the mean and the diagonal threshold.
    amplitudes = torch.tensor([1, -1, 0.5, -0.5, 0.0625, -0.0625], device="cuda")
    key_mean = amplitudes[block % amplitudes.numel()].view(1, length, 1, 1)
    q = (1 + noise).bfloat16()
    k = (key_mean + noise).bfloat16()
    v = (torch.randint(-16, 17, shape, device="cuda", generator=rng) / 16).bfloat16()
    kwargs = dict(tau=1.25, sm_scale=0.05, thresh_type=cell.spec["thresh_type"])
    predict = (
        torch.compile(sol_predictor, backend="eager", fullgraph=True)
        if cell.spec["compiled"]
        else sol_predictor
    )
    actual = predict(q, k, v, **kwargs)
    expected = sol_predictor.reference(q, k, v, **kwargs)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    alternative = "exact" if kwargs["thresh_type"] == "diag" else "diag"
    other_bits, _, _ = sol_predictor.reference(q, k, v, **{**kwargs, "thresh_type": alternative})
    assert not torch.equal(expected[0], other_bits)
    if kwargs["thresh_type"] == "diag":
        mean_bits, _, _ = sol_predictor.reference(q, k, v, **{**kwargs, "tau": 0.0})
        assert not torch.equal(expected[0], mean_bits)
