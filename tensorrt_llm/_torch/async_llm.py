#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, List, Optional

from ..llmapi.llm import LLM
from ..llmapi.llm_args import ExecutorMemoryType, RayPlacementConfig, RuntimeMemoryStatus


class AsyncLLM(LLM):
    """LLM runtime with asynchronous setup, generation, and memory control.

    AsyncLLM uses the Ray orchestrator. Weight-update methods remain available
    for compatibility with existing reinforcement-learning integrations.
    """

    def __init__(
        self,
        placement_groups: Optional[List[Any]] = None,
        placement_bundle_indices: Optional[List[List[int]]] = None,
        per_worker_gpu_share: Optional[float] = None,
        *args,
        **kwargs,
    ):
        kwargs["orchestrator_type"] = "ray"
        kwargs["ray_placement_config"] = RayPlacementConfig(
            defer_workers_init=True,
            placement_groups=placement_groups,
            placement_bundle_indices=placement_bundle_indices,
            per_worker_gpu_share=per_worker_gpu_share,
        )

        # WAR: RL integration needs to use NCCL AllReduce for TP>1 due to a bug in TRTLLM's AllReduce
        # which will cause convergence issue when using multiple rollout instances.
        kwargs["allreduce_strategy"] = "NCCL"

        if "ray_worker_extension_cls" not in kwargs:
            kwargs["ray_worker_extension_cls"] = "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension"

        super().__init__(*args, **kwargs)
        self._async_initialized = False
        self._paused = False

    async def setup_async(self):
        """Setup the LLM asynchronously."""
        if not self._async_initialized:
            await self._executor.init_workers_async()
            await self._executor.setup_engine_remote_async()
            self._async_initialized = True
        return self

    async def get_memory_status(self) -> RuntimeMemoryStatus:
        """Return the authoritative runtime-memory checkpoint status."""
        self._check_runtime_memory_enabled()
        replies = await self.collective_rpc("get_memory_status")
        statuses = [RuntimeMemoryStatus.model_validate(reply) for reply in replies]
        if not statuses:
            raise RuntimeError("No worker returned runtime-memory status.")
        expected = statuses[0]
        if any(status != expected for status in statuses[1:]):
            raise RuntimeError(
                "Runtime-memory state diverged across workers: "
                f"{[status.model_dump(mode='json') for status in statuses]}"
            )
        return expected

    async def release(
        self,
        tags: Sequence[ExecutorMemoryType | str] | None = None,
    ) -> None:
        """Release the GPU memory used by the LLM asynchronously.

        Args:
            tags: Memory types to release. When omitted, release every
                sleep-managed runtime memory type.
        """
        self._check_runtime_memory_enabled()
        default_tags = [
            memory_type
            for memory_type in ExecutorMemoryType
            if memory_type
            not in (ExecutorMemoryType.INIT_KV_CACHE, ExecutorMemoryType.INIT_EXTRA_RESOURCES)
        ]
        normalized = self._normalize_runtime_memory_tags(tags, default_tags=default_tags)
        await self.collective_rpc("sleep", args=([tag.value for tag in normalized],))

    async def resume(
        self,
        tags: Sequence[ExecutorMemoryType | str] | None = None,
    ) -> None:
        """Resume the GPU memory used by the LLM asynchronously.

        Args:
            tags: Memory types to restore. When omitted, restore every
                currently parked memory type.
        """
        self._check_runtime_memory_enabled()
        default_tags = (await self.get_memory_status()).parked_tags
        normalized = self._normalize_runtime_memory_tags(tags, default_tags=default_tags)
        if normalized:
            await self.collective_rpc("wakeup", args=([tag.value for tag in normalized],))

    async def update_weights(self, weights: dict[str, str]):
        """Update the weights of the LLM asynchronously.


        Args:
            weights: Dictionary mapping device UUIDs to IPC handles for weight tensors.
        """
        await self.collective_rpc("update_weights", args=(weights,))

    async def collective_rpc(
        self,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict] = None,
        unique_reply_rank: Optional[int] = None,  # TODO: deprecate this in the future
        target_ranks: int | list[int] | None = None,
    ) -> list[Any]:
        """Execute an asynchronous RPC call on all GPU workers. Currently, this is only supported for RayExecutor.

        Args:
            method (str): The name of the worker method to execute.
            args (tuple[Any, ...]): Positional arguments to pass to the worker method. Defaults to ().
            kwargs (dict, optional): Keyword arguments to pass to the worker method. Defaults to None.
            unique_reply_rank (int, optional): The rank of the worker that will be used to send the reply.
            target_ranks (int | list[int] | None): The ranks of the workers that will be used to send the reply.

        Returns:
            list[Any]: A list of results from each worker.
        """
        return await self._executor.collective_rpc_async(
            method, args, kwargs, unique_reply_rank=unique_reply_rank, target_ranks=target_ranks
        )

    def generate_async(self, *args, **kwargs):
        if self._paused:
            raise RuntimeError(
                "AsyncLLM is paused. Call resume_generation() before submitting new requests."
            )
        return super().generate_async(*args, **kwargs)

    async def pause_generation(self) -> None:
        """Abort all in-flight requests and block new ones until resume_generation() is called.

        Sends abort signals then drains the executor. It only returns once the engine
        has no active or queued requests.
        """
        self._paused = True
        self._executor.abort_all_requests()
        await self.collective_rpc("wait_for_engine_idle")

    async def resume_generation(self) -> None:
        """Allow new generation requests after a pause_generation() call."""
        self._paused = False

    def __await__(self):
        return self.setup_async().__await__()

    def __enter__(self):
        raise RuntimeError("Please use 'async with AsyncLLM' instead")

    async def __aenter__(self):
        await self.setup_async()
        return super().__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)
