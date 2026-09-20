# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.timestep_phase import (
    graph_phase_for_timestep,
    timestep_to_float,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("timestep", "expected"),
    [
        (None, None),
        (torch.empty(0), None),
        (0.25, 0.25),
        (torch.tensor(0.5), 0.5),
        # Per-token timesteps reduce to the largest live value.
        (torch.tensor([0.0, 0.8]), 0.8),
    ],
)
def test_timestep_to_float_reduces_to_the_largest_live_value(timestep, expected) -> None:
    assert timestep_to_float(timestep) == (
        expected if expected is None else pytest.approx(expected)
    )


def test_timestep_to_float_rejects_non_real_values() -> None:
    with pytest.raises(TypeError, match="real scalar or tensor"):
        timestep_to_float(True)


@pytest.mark.parametrize(
    "timestep",
    [
        float("nan"),
        float("inf"),
        torch.tensor([float("nan")]),
        # The reduction keeps the non-finite value.
        torch.tensor([0.5, float("inf")]),
    ],
    ids=["nan", "inf", "tensor_nan", "tensor_inf"],
)
def test_timestep_to_float_rejects_non_finite_values(timestep) -> None:
    with pytest.raises(ValueError, match="finite"):
        timestep_to_float(timestep)


@pytest.mark.parametrize(
    ("timestep", "cutoff", "expected"),
    [
        (0.8, 0.6, 0),
        (0.6, 0.6, 0),
        (0.2, 0.6, 1),
        (None, 0.6, None),
        (0.2, None, None),
        (torch.tensor([0.0, 0.8]), 0.6, 0),
        (torch.tensor([0.0, 0.2]), 0.6, 1),
    ],
)
def test_graph_phase_for_timestep_marks_dense_prefix_and_sparse_suffix(
    timestep, cutoff, expected
) -> None:
    assert graph_phase_for_timestep(timestep, disabled_until_timestep=cutoff) == expected
