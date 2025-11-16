from typing import Any, Optional

from ..llmapi.llm import LLM
from .virtual_memory import ExecutorMemoryType


class AsyncLLM(LLM):
    """AsyncLLM is a subclass of LLM that supports asynchronous setup, release and
    resume operations that are necessary for RL or agentic scenarios.
    """

    def __init__(self, *args, **kwargs):
        # AsyncLLM is only supported with Ray orchestrator now.
        kwargs["orchestrator_type"] = "ray"
        if 'ray_worker_extension_cls' not in kwargs:
            kwargs['ray_worker_extension_cls'] = 'tensorrt_llm.llmapi.rlhf_utils.WorkerExtension'
        super().__init__(*args, **kwargs)

    async def setup_async(self):
        """Setup the LLM asynchronously."""
        await self._executor.init_workers_async()

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
