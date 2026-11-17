# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import partial
from threading import Lock
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3 import reporting

pytestmark = pytest.mark.cpu_only


class _FastPathUnavailable(RuntimeError):
    pass


class _Target:
    FastPathUnavailable = _FastPathUnavailable

    def __init__(self) -> None:
        self._require = [False]
        self._lock = Lock()
        self._logged: set[tuple[str, str]] = set()
        self.logger = Mock(spec_set=logging.Logger)
        self.mpi_rank = Mock(return_value=3)
        self._log_once = partial(reporting._log_once, self)
        self.missed = partial(reporting.missed, self)


def test_fullgraph_reporting_leaves_host_state_untouched() -> None:
    state = _Target()

    def forward(x: torch.Tensor) -> torch.Tensor:
        reporting.taken(state, "projection", x.shape[0])
        reporting.missed(state, "fallback", x.shape[0], "unsupported shape")
        reporting.uncovered(state, "decode", "context step")
        reporting.capture_missed(state, "capture", x.shape[0], "engine")
        return x + 1

    compiled = torch.compile(forward, fullgraph=True, backend="eager")
    x = torch.ones(2)
    torch.testing.assert_close(compiled(x), x + 1)
    state.mpi_rank.assert_not_called()
    state.logger.info.assert_not_called()
    state.logger.warning.assert_not_called()
    assert not state._logged

    forward(x)
    forward(x)
    assert state.logger.info.call_count == 2
    assert state.logger.warning.call_count == 2
    assert "rank 3" in state.logger.info.call_args_list[0].args[0]


def test_compilation_preserves_required_path_failure() -> None:
    state = _Target()
    reporting.set_require(state, True)

    def forward(x: torch.Tensor) -> torch.Tensor:
        reporting.missed(state, "required_projection", x.shape[0], "unavailable")
        return x + 1

    # Catch inside the trace so the test distinguishes the required error from
    # an unrelated Dynamo compilation error.
    def catches_required_failure(x: torch.Tensor) -> torch.Tensor:
        try:
            return forward(x)
        except _FastPathUnavailable:
            return x - 1

    x = torch.ones(2)
    compiled = torch.compile(catches_required_failure, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(x), x - 1)
    with pytest.raises(_FastPathUnavailable, match="required_projection.*unavailable"):
        forward(x)
    compiled_forward = torch.compile(forward, backend="eager")
    with pytest.raises(_FastPathUnavailable, match="required_projection.*unavailable"):
        compiled_forward(x)
    state.logger.warning.assert_not_called()
