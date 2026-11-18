# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregated Helix startup checks that run before the model weights load."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.py_executor_creator import _validate_agg_helix_args

pytestmark = pytest.mark.cpu_only


def _args(**overrides: object) -> SimpleNamespace:
    args = dict(
        sparse_attention_config=None,
        enable_chunked_prefill=True,
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True),
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize(
    ("sm_version", "overrides", "error"),
    [
        (90, {}, "SM100 or newer"),
        (100, {"sparse_attention_config": SimpleNamespace(algorithm="dsa")}, "dense MLA only"),
        (
            100,
            {"sparse_attention_config": SimpleNamespace(algorithm="skip_softmax")},
            "including skip_softmax",
        ),
        (100, {"enable_chunked_prefill": False}, "requires chunked prefill"),
        (
            100,
            {"kv_cache_config": SimpleNamespace(use_kv_cache_manager_v2=False)},
            "requires KVCacheManagerV2",
        ),
    ],
    ids=["pre_sm100", "sparse", "sparse_skip_softmax", "no_chunked_prefill", "kv_cache_manager_v1"],
)
def test_agg_helix_args_rejected_before_model_load(
    sm_version: int, overrides: dict[str, object], error: str
) -> None:
    with patch.object(py_executor_creator, "get_sm_version", return_value=sm_version):
        with pytest.raises((NotImplementedError, ValueError), match=error):
            _validate_agg_helix_args(_args(**overrides))


@pytest.mark.parametrize("use_kv_cache_manager_v2", [True, "auto"])
def test_agg_helix_args_accepted(use_kv_cache_manager_v2: bool | str) -> None:
    kv_cache_config = SimpleNamespace(use_kv_cache_manager_v2=use_kv_cache_manager_v2)
    with patch.object(py_executor_creator, "get_sm_version", return_value=100):
        _validate_agg_helix_args(_args(kv_cache_config=kv_cache_config))
