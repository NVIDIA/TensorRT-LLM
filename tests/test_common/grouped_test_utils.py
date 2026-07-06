# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for tests that reuse one MPI worker pool across LLM instances."""

import os
from contextlib import contextmanager
from functools import wraps


class SharedMpiSessionRegistry:
    """Own lazily-created MPI pools keyed by worker count and layout."""

    def __init__(self):
        self._sessions = {}

    def get(self, n_workers: int, key=()):
        from tensorrt_llm._utils import mpi_disabled

        if mpi_disabled():
            return None

        session_key = (n_workers, key)
        if session_key not in self._sessions:
            from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

            self._sessions[session_key] = MpiPoolSession(n_workers=n_workers)
        return self._sessions[session_key]

    def shutdown(self):
        sessions = list(self._sessions.values())
        self._sessions.clear()
        shutdown_error = None
        for session in reversed(sessions):
            try:
                session.shutdown()
            except Exception as error:
                shutdown_error = shutdown_error or error
        if shutdown_error is not None:
            raise shutdown_error


@contextmanager
def share_torch_llm_mpi_sessions(registry):
    """Inject borrowed registry pools into eligible PyTorch ``LLM`` objects."""
    from tensorrt_llm.llmapi.llm import BaseLLM, _TorchLLM
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession, external_mpi_comm_available

    original_init = BaseLLM.__init__

    def environment_key():
        prefixes = ("CUDA_", "NCCL_", "PYTORCH_", "RAY_", "TLLM_", "TRTLLM_", "UCX_")
        return tuple(
            sorted((name, value) for name, value in os.environ.items() if name.startswith(prefixes))
        )

    default_environment = environment_key()
    default_pool_launcher = MpiPoolSession._start_mpi_pool

    def get_pool_config(llm, tensor_parallel_size, kwargs):
        if not isinstance(llm, _TorchLLM):
            return None
        if kwargs.get("backend", "pytorch") != "pytorch":
            return None
        if kwargs.get("orchestrator_type") is not None:
            return None
        if kwargs.get("_mpi_session") is not None:
            return None
        if kwargs.get("executor_cls") is not None:
            return None
        if kwargs.get("env_overrides") is not None:
            return None
        if kwargs.get("encode_only") or kwargs.get("mm_encoder_only"):
            return None

        # These modes do not use the MPI worker pool.
        if os.environ.get("TLLM_DISABLE_MPI") == "1":
            return None
        if os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS") == "1":
            return None
        if os.environ.get("TLLM_SPAWN_PROXY_PROCESS") == "1":
            return None
        if os.environ.get("RAY_LOCAL_WORLD_SIZE") is not None:
            return None
        if environment_key() != default_environment:
            return None
        if MpiPoolSession._start_mpi_pool is not default_pool_launcher:
            return None

        pipeline_parallel_size = kwargs.get("pipeline_parallel_size", 1)
        context_parallel_size = kwargs.get("context_parallel_size", 1)
        world_size = tensor_parallel_size * pipeline_parallel_size * context_parallel_size
        if world_size <= 0 or external_mpi_comm_available(world_size):
            return None

        # These paths may select the single-process executor.
        build_config = kwargs.get("build_config")
        if kwargs.get("gather_generation_logits") or getattr(
            build_config, "gather_context_logits", False
        ):
            return None

        # The worker process keeps its PP communicator globally. Reuse a pool
        # only when the TP/PP/CP layout is unchanged.
        layout = (tensor_parallel_size, pipeline_parallel_size, context_parallel_size)
        return world_size, layout

    @wraps(original_init)
    def patched_init(
        self,
        model,
        tokenizer=None,
        tokenizer_mode="auto",
        skip_tokenizer_init=False,
        trust_remote_code=False,
        tensor_parallel_size=1,
        dtype="auto",
        revision=None,
        tokenizer_revision=None,
        **kwargs,
    ):
        pool_config = get_pool_config(self, tensor_parallel_size, kwargs)
        if pool_config is not None:
            world_size, layout = pool_config
            kwargs = dict(kwargs)
            kwargs["_mpi_session"] = registry.get(world_size, key=layout)
        return original_init(
            self,
            model,
            tokenizer,
            tokenizer_mode,
            skip_tokenizer_init,
            trust_remote_code,
            tensor_parallel_size,
            dtype,
            revision,
            tokenizer_revision,
            **kwargs,
        )

    BaseLLM.__init__ = patched_init
    try:
        yield
    finally:
        BaseLLM.__init__ = original_init
