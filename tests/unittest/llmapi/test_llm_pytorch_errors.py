# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from contextlib import contextmanager, nullcontext

import pytest
from utils.util import force_ampere, skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.executor import GenerationExecutorWorker, RequestError
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import _test_llm_capture_request_error, global_kvcache_config
from .model_paths import llama_model_path


def test_llm_capture_request_error():
    _test_llm_capture_request_error(pytorch_backend=True, tp_size=1)


@contextmanager
def _validate_invalid_token_error_scope():
    with pytest.raises(RuntimeError) as exc_info:
        yield
    assert "Token ID out of range" in str(exc_info.value)


@force_ampere
def test_llm_invalid_input_token():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    prompts = [
        [-1],
    ]
    # NB: exc_info in _validate_invalid_token_error_scope creates a reference
    #     to a traceback which outlives the scope of 'exc_info' and prevents
    #     deletion of 'llm'. However, using the context manager protocol is
    #     anyways more robust than delegating cleanup to __del__.
    with llm:
        with _validate_invalid_token_error_scope():
            llm.generate(
                prompts,
                sampling_params=SamplingParams(max_tokens=5),
            )


@force_ampere
def test_llm_invalid_input_token_async():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    # NB: exc_info in _validate_invalid_token_error_scope creates a reference
    #     to a traceback which outlives the scope of 'exc_info' and prevents
    #     deletion of 'llm'. However, using the context manager protocol is
    #     anyways more robust than delegating cleanup to __del__.
    with llm:
        prompts = [
            [-1],
            [42],
        ]
        fail_idx = [0]
        for submit_order in [[0, 1], [1, 0]]:
            for collect_order in [[0, 1], [1, 0]]:
                print(f"submitting {submit_order}")
                futures = [
                    llm.generate_async(
                        prompts[submit_idx],
                        sampling_params=SamplingParams(max_tokens=5),
                    )
                    for submit_idx in submit_order
                ]
                for collect_idx in collect_order:
                    with (
                        _validate_invalid_token_error_scope()
                        if submit_order[collect_idx] in fail_idx
                        else nullcontext()
                    ):
                        print(f"collect order {collect_order}, collecting {collect_idx}")
                        futures[collect_idx].result()


class TestLlmError:
    def test_max_num_token_check(self):
        """LLM should raise error when got prompt length exceed the valid range."""
        llm = LLM(llama_model_path, kv_cache_config=global_kvcache_config, max_num_tokens=100)

        try:
            with pytest.raises(RequestError, match="should not exceed max_num_tokens"):
                ids = [random.randint(10, 100) for _ in range(101)]
                llm.generate([ids])
        finally:
            llm.shutdown()


class FailingExecutorWorker(GenerationExecutorWorker):
    """Mock worker that fails during initialization to test error handling."""

    def __init__(self, *args, **kwargs):
        # Simulate a constructor failure
        raise RuntimeError("Mock GenerationExecutorWorker initialization failed")


FailingExecutor = type(
    "FailingExecutor",
    (),
    {"create": classmethod(lambda cls, *args, **kwargs: FailingExecutorWorker(*args, **kwargs))},
)


@skip_ray
def test_llm_with_proxy_error():
    """Test that LLM properly handles GenerationExecutorWorker constructor failures.

    This test mocks the GenerationExecutorWorker to fail during __init__ and
    verifies that the LLM class properly catches and re-raises the error.
    """
    from unittest.mock import patch

    # Test that the error is properly caught and re-raised by LLM
    # We patch GenerationExecutor.create directly to return our failing worker
    with patch(
        "tensorrt_llm.executor.executor.GenerationExecutor.create",
        side_effect=lambda *args, **kwargs: FailingExecutorWorker(*args, **kwargs),
    ):
        with pytest.raises(
            RuntimeError, match="Mock GenerationExecutorWorker initialization failed"
        ):
            LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
