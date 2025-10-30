import importlib
import os
from pathlib import Path
from queue import Queue
from typing import Any, Optional, Type, Union

import ray
import torch

from ..bindings import executor as tllm
from ..builder import Engine
from ..llmapi.llm_args import BaseLlmArgs
from ..llmapi.tokenizer import TokenizerBase
from ..sampling_params import BatchedLogitsProcessor
from .base_worker import BaseWorker
from .postproc_worker import PostprocWorkerConfig
from .request import GenerationRequest
from .result import GenerationResult

__all__ = [
    "RayGPUWorker",
    "RayWorkerWrapper",
]


def resolve_obj_by_qualname(qualname: str) -> Any:
    """Resolve an object by its fully qualified name."""
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


@ray.remote
class RayWorkerWrapper:

    def __init__(self, worker_cls, worker_kwargs, world_size, rank):
        self.master_address = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]

        # Ray can't pickle TensorRT logger
        global logger
        from tensorrt_llm.logger import logger

        # Expect to see global counts w/ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1,
        # unless CUDA_VISIBLE_DEVICES is set.
        logger.debug(
            f"CUDA device count visible to Ray: {torch.cuda.device_count()}")

        # Physical gpu id
        self.gpu = int(ray.get_gpu_ids()[0])
        local_gpu = self.physical_to_local_id(self.gpu)

        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=world_size,
            rank=rank)

        logger.info(
            f"[Rank {rank}] Finished PG init. Global GPU ID: {self.gpu}, local GPU ID: {local_gpu}"
        )

        torch.cuda.set_device(local_gpu)

        worker_cls = RayWorkerWrapper._inject_worker_extension(
            worker_cls, worker_kwargs.pop("ray_worker_extension_cls", None))
        self.worker = worker_cls(device_id=local_gpu, **worker_kwargs)

    def submit(self, request: GenerationRequest) -> GenerationResult:
        return self.worker.submit(request)

    def enqueue_request(self,
                        request: GenerationRequest,
                        result_wait_queue: Queue | None = None) -> int:
        return self.worker.enqueue_request(request, result_wait_queue)

    def abort_request(self, request_id: int) -> None:
        self.worker.abort_request(request_id)

    def report_device_id(self) -> str:
        from tensorrt_llm._torch.utils import get_device_uuid
        local_id = self.physical_to_local_id(self.gpu)
        return get_device_uuid(local_id)

    def call_worker_method(self, method_name: str, *args, **kwargs):
        """Generic method to call any method on the underlying worker."""
        if hasattr(self.worker, method_name):
            method = getattr(self.worker, method_name)
            if callable(method):
                return method(*args, **kwargs)
            else:
                raise AttributeError(
                    f"'{method_name}' is not a callable method of RayGPUWorker."
                )
        else:
            raise AttributeError(
                f"The RayGPUWorker has no method called '{method_name}'.")

    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        Refer to https://github.com/NVIDIA-NeMo/RL/blob/faad02113c3c502437ccb339cb848796334aedd9/nemo_rl/models/policy/dtensor_policy_worker_v2.py#L95
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    @staticmethod
    def physical_to_local_id(phys_id: int) -> int:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible_devices:
            return phys_id
        id_mapping = list(map(int, visible_devices.split(",")))
        return id_mapping.index(phys_id)

    @staticmethod
    def _inject_worker_extension(
            worker_class: Type[BaseWorker],
            extension_cls_name: Optional[str]) -> Type[BaseWorker]:
        """Inject worker extension into the worker class if specified."""
        if not extension_cls_name:
            return worker_class

        try:
            extension_cls = resolve_obj_by_qualname(extension_cls_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load worker extension '{extension_cls_name}'"
            ) from e

        # Check for conflicts
        for attr in dir(extension_cls):
            if attr.startswith("__"):
                continue
            if hasattr(worker_class, attr):
                raise ValueError(
                    f"Worker class {worker_class.__name__} already defines '{attr}', "
                    f"which conflicts with extension {extension_cls.__name__}.")

        derived_name = f"{worker_class.__name__}With{extension_cls.__name__}"
        ExtendedWorker = type(derived_name, (worker_class, extension_cls),
                              {'__module__': worker_class.__module__})
        return ExtendedWorker


class RayGPUWorker(BaseWorker):

    def __init__(
        self,
        device_id: int,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
    ) -> None:
        global logger
        from tensorrt_llm.logger import logger

        super().__init__(
            engine=engine,
            executor_config=executor_config,
            batched_logits_processor=batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            llm_args=llm_args,
        )

        if not self._is_pytorch_backend:
            raise ValueError(f"Ray GPU worker only supports PyTorch backend.")

        self.device_id = device_id

        # Override rank attributes using torch
        self.global_rank = torch.distributed.get_rank()
        if self.global_rank > 1:
            logger.set_rank(self.global_rank)

        self.setup_engine()

    def _get_comm_ranks_device_id(self):
        # Make sure C++ executor would use same devices/ranks as py_executor
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        comm_ranks = [None] * world_size
        device_ids = [None] * world_size

        torch.distributed.all_gather_object(comm_ranks, global_rank)
        torch.distributed.all_gather_object(device_ids, self.device_id)
        return comm_ranks, device_ids

    def enqueue_request(self,
                        request: GenerationRequest,
                        result_wait_queue: Queue | None = None) -> int:
        return self._enqueue_request(request, result_wait_queue)

    def submit(self, request: GenerationRequest):
        raise NotImplementedError(
            "Ray GPU worker does not support submit() yet.")

    def shutdown(self):

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        logger.debug(f'Worker {self.rank} shutting down...')

        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

            assert self._executor_config is None, "An empty executor_config is expected in shutdown when LLM arguments are defined."
            if (self.llm_args.backend == "pytorch"
                    and hasattr(self, "checkpoint_loader")
                    and self.checkpoint_loader is not None):
                self.checkpoint_loader.cleanup()
                self.checkpoint_loader = None

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

        logger.debug(f"Worker {self.rank} shutdown done.")

    def __enter__(self):
        return self

    def __del__(self):
        self.shutdown()
