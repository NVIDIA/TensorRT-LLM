# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default context coalescing for one-model EAGLE3 (https://nvbugs/6917001)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import (
    SPEC_DEC_CTX_COALESCE_ITERS,
    SPEC_DEC_CTX_COALESCE_MAX_TOKENS_RATIO,
    PyExecutor,
    resolve_spec_dec_batch_wait_defaults,
)
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend, TorchLlmArgs

_DEFAULT = (SPEC_DEC_CTX_COALESCE_ITERS, SPEC_DEC_CTX_COALESCE_MAX_TOKENS_RATIO)


def _engine(mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL, cuda_graph=True):
    spec_config = None if mode is None else SimpleNamespace(spec_dec_mode=mode)
    return SimpleNamespace(
        spec_config=spec_config,
        cuda_graph_config=object() if cuda_graph else None,
    )


def test_eagle3_one_model_gets_coalescing_default():
    llm_args = TorchLlmArgs(model="dummy")
    assert resolve_spec_dec_batch_wait_defaults(llm_args, _engine(), False) == _DEFAULT
    assert SPEC_DEC_CTX_COALESCE_ITERS > 0 and SPEC_DEC_CTX_COALESCE_MAX_TOKENS_RATIO > 0


@pytest.mark.parametrize(
    "user_kwargs",
    [
        {"batch_wait_timeout_iters": 0},
        {"batch_wait_timeout_iters": 4},
        {"batch_wait_max_tokens_ratio": 0.0},
        {"batch_wait_max_tokens_ratio": 0.3},
        {"batch_wait_timeout_ms": 0.0},
        {"batch_wait_timeout_ms": 5.0},
    ],
)
def test_explicit_user_batch_wait_config_is_respected(user_kwargs):
    llm_args = TorchLlmArgs(model="dummy", **user_kwargs)
    assert resolve_spec_dec_batch_wait_defaults(llm_args, _engine(), False) is None


@pytest.mark.parametrize(
    "engine",
    [
        _engine(mode=None),
        _engine(mode=SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL),
        _engine(mode=SpeculativeDecodingMode.MTP),
        _engine(mode=SpeculativeDecodingMode.NGRAM),
        _engine(mode=SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL),
        _engine(cuda_graph=False),
    ],
    ids=[
        "no_spec",
        "mtp_eagle",
        "mtp",
        "ngram",
        "draft_target",
        "no_cuda_graph",
    ],
)
def test_not_applied_outside_one_model_eagle3_with_eager_prefill(engine):
    llm_args = TorchLlmArgs(model="dummy")
    assert resolve_spec_dec_batch_wait_defaults(llm_args, engine, False) is None


def test_not_applied_with_prefill_cuda_graph():
    llm_args = TorchLlmArgs(
        model="dummy", prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE
    )
    assert resolve_spec_dec_batch_wait_defaults(llm_args, _engine(), False) is None


def test_not_applied_with_attention_dp():
    llm_args = TorchLlmArgs(model="dummy")
    assert resolve_spec_dec_batch_wait_defaults(llm_args, _engine(), True) is None


def test_not_applied_in_disaggregated_serving():
    llm_args = TorchLlmArgs(model="dummy")
    assert (
        resolve_spec_dec_batch_wait_defaults(llm_args, _engine(), False, is_disaggregated=True)
        is None
    )


def test_non_pydantic_llm_args_are_left_untouched():
    assert resolve_spec_dec_batch_wait_defaults(MagicMock(), _engine(), False) is None


def _bare_executor(llm_args, engine=None):
    executor = PyExecutor.__new__(PyExecutor)
    executor.llm_args = llm_args
    executor.model_engine = engine if engine is not None else _engine()
    executor.enable_attention_dp = False
    return executor


def test_executor_init_enables_batch_waiting_for_eagle3():
    executor = _bare_executor(TorchLlmArgs(model="dummy"))
    executor._init_batch_waiting(is_disaggregated=False)
    assert (
        executor.batch_wait_timeout_iters,
        executor.batch_wait_max_tokens_ratio,
    ) == _DEFAULT
    assert executor.batch_wait_timeout_ms == 0
    assert executor.enable_batch_waiting


def test_executor_init_respects_explicit_opt_out():
    executor = _bare_executor(TorchLlmArgs(model="dummy", batch_wait_timeout_iters=0))
    executor._init_batch_waiting(is_disaggregated=False)
    assert executor.batch_wait_timeout_iters == 0
    assert executor.batch_wait_max_tokens_ratio == 0
    assert not executor.enable_batch_waiting


@pytest.mark.parametrize(
    "engine, is_disaggregated",
    [(_engine(mode=None), False), (_engine(), True)],
    ids=["no_spec", "disaggregated"],
)
def test_executor_init_leaves_batch_waiting_off_elsewhere(engine, is_disaggregated):
    executor = _bare_executor(TorchLlmArgs(model="dummy"), engine)
    executor._init_batch_waiting(is_disaggregated=is_disaggregated)
    assert executor.batch_wait_timeout_iters == 0
    assert executor.batch_wait_max_tokens_ratio == 0
    assert not executor.enable_batch_waiting
