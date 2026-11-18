# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup checks for the executor loops a non-generation model can run in."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor_creator import _check_no_kv_execution
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_PP2 = dict(world_size=2, tp_size=1, pp_size=2)
_ADP = dict(world_size=2, tp_size=2, enable_attention_dp=True)


def _args():
    return SimpleNamespace(enable_chunked_prefill=False)


@pytest.mark.parametrize(
    ("mapping", "error"),
    [(_PP2, "Pipeline parallelism"), (_ADP, "attention DP")],
    ids=["pp", "adp"],
)
def test_non_generation_models_reject_unsupported_loops(mapping, error):
    with pytest.raises(ValueError, match=error):
        _check_no_kv_execution(_args(), Mapping(**mapping), is_generation=False)


@pytest.mark.parametrize("mapping", [_PP2, _ADP], ids=["pp", "adp"])
def test_generation_models_are_not_checked(mapping):
    _check_no_kv_execution(_args(), Mapping(**mapping), is_generation=True)
