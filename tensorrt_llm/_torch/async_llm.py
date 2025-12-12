from typing import Any, List, Optional

from ..llmapi.llm import LLM
from ..llmapi.llm_args import RayPlacementConfig


class AsyncLLM(LLM):
    """AsyncLLM is a subclass of LLM that supports asynchronous setup, release and
    resume operations that are necessary for RL or agentic scenarios.

    Currently, RL APIs are only supported with Ray orchestrator.
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

    async def setup_async(self):
        """Setup the LLM asynchronously."""
        if not self._async_initialized:
            await self._executor.init_workers_async()
            await self._executor.setup_engine_remote_async()
            self._async_initialized = True
        return self

    async def release(self, tags: list[str]):
        """Release the GPU memory used by the LLM asynchronously.

        Args:
            tags: List of memory tag strings to release (e.g., ["model", "kv_cache"]).
        """
        await self.collective_rpc("sleep", args=(tags,))

    async def resume(self, tags: list[str]):
        """Resume the GPU memory used by the LLM asynchronously.

        Args:
            tags: List of memory tag strings to resume (e.g., ["model", "kv_cache"]).
        """
        await self.collective_rpc("wakeup", args=(tags,))

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
        unique_reply_rank: Optional[int] = None,
    ) -> list[Any]:
        """Execute an asynchronous RPC call on all GPU workers. Currently, this is only supported for RayExecutor.

        Args:
            method (str): The name of the worker method to execute.
            args (tuple[Any, ...]): Positional arguments to pass to the worker method. Defaults to ().
            kwargs (dict, optional): Keyword arguments to pass to the worker method. Defaults to None.
            unique_reply_rank (int, optional): The rank of the worker that will be used to send the reply.

        Returns:
            list[Any]: A list of results from each worker.
        """
        return await self._executor.collective_rpc_async(
            method, args, kwargs, unique_reply_rank=unique_reply_rank
        )

    def __await__(self):
        return self.setup_async().__await__()

    def __enter__(self):
        raise RuntimeError("Please use 'async with AsyncLLM' instead")

    async def __aenter__(self):
        await self.setup_async()
        return super().__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)
