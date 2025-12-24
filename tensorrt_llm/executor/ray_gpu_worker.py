import gc
import importlib
import os
from functools import wraps
from pathlib import Path
from queue import Queue
from typing import Any, List, Optional, Type, Union

import ray
import torch

from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm._torch.virtual_memory import (materialize_with_tag,
                                                release_with_tag,
                                                verify_sleep_wakeup_tags)

from ..bindings import executor as tllm
from ..builder import Engine
from ..llmapi.llm_args import BaseLlmArgs
from ..llmapi.tokenizer import TokenizerBase
from ..sampling_params import BatchedLogitsProcessor
from .base_worker import BaseWorker
from .postproc_worker import PostprocWorkerConfig
from .request import GenerationRequest
from .result import GenerationResult
from .rpc_worker_mixin import RpcWorkerMixin

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
        self.world_size = world_size
        self.rank = rank
        # Ray can't pickle TensorRT logger
        global logger
        from tensorrt_llm.logger import logger

        # Expect to see global counts w/ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1,
        # unless CUDA_VISIBLE_DEVICES is set.
        logger.debug(
            f"CUDA device count visible to Ray: {torch.cuda.device_count()}")

        # Physical gpu id
        self.gpu = int(ray.get_gpu_ids()[0])
        self.local_gpu = self.physical_to_local_id(self.gpu)

        torch.cuda.set_device(self.local_gpu)

        self.worker_cls = RayWorkerWrapper._inject_worker_extension(
            worker_cls, worker_kwargs.pop("ray_worker_extension_cls", None))
        self.worker_kwargs = worker_kwargs

    def _create_tcp_store(self,
                          port: Optional[int] = None
                          ) -> torch.distributed.TCPStore:
        # port=0 means let the OS pick an available port (only valid for master)
        # For non-master, port must be specified to connect to master's port
        actual_port = port if port is not None else 0
        return torch.distributed.TCPStore(host_name=self.master_address,
                                          port=actual_port,
                                          world_size=self.world_size,
                                          is_master=(self.rank == 0),
                                          wait_for_workers=False)

    def setup_tcp_store(self):
        if self.rank != 0:
            raise RuntimeError("Only the master worker can setup TCP store")
        self.store = self._create_tcp_store()
        return self.store.port

    def setup_distributed_env_and_worker(self, port: int):
        if self.rank != 0:
            self.store = self._create_tcp_store(port)

        torch.distributed.init_process_group(backend="cuda:nccl,cpu:gloo",
                                             store=self.store,
                                             world_size=self.world_size,
                                             rank=self.rank)
        assert torch.distributed.get_world_size(
        ) == self.world_size, "Process group world size must match the expected world size"
        logger.info(
            f"[Rank {self.rank}] Finished PG init. Global GPU ID: {self.gpu}, local GPU ID: {self.local_gpu}"
        )

        self.worker = self.worker_cls(device_id=self.local_gpu,
                                      **self.worker_kwargs)
        self._has_setup_distributed_env_and_worker = True

    @property
    def has_setup_distributed_env_and_worker(self) -> bool:
        return getattr(self, '_has_setup_distributed_env_and_worker', False)

    def ensure_distributed_setup(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.has_setup_distributed_env_and_worker:
                raise RuntimeError(
                    "Have not setup distributed environment and worker yet")
            return func(self, *args, **kwargs)

        return wrapper

    @ensure_distributed_setup
    def submit(self, request: GenerationRequest) -> GenerationResult:
        return self.worker.submit(request)

    @ensure_distributed_setup
    def enqueue_request(self,
                        request: GenerationRequest,
                        result_wait_queue: Queue | None = None) -> int:
        return self.worker.enqueue_request(request, result_wait_queue)

    @ensure_distributed_setup
    def abort_request(self, request_id: int) -> None:
        self.worker.abort_request(request_id)

    @ensure_distributed_setup
    def report_device_id(self) -> str:
        local_id = self.physical_to_local_id(self.gpu)
        return get_device_uuid(local_id)

    @ensure_distributed_setup
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

    def shutdown(self):
        if hasattr(self, 'worker'):
            self.worker.shutdown()

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


class RayGPUWorker(RpcWorkerMixin, BaseWorker):

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
        rpc_addr: Optional[str] = None,
        hmac_key: Optional[bytes] = None,
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

        self.device_id = device_id
        self.global_rank = torch.distributed.get_rank()
        if self.global_rank > 1:
            logger.set_rank(self.global_rank)

        if rpc_addr is None:
            raise RuntimeError(
                "RPC mode enabled but no rpc_addr provided to RayGPUWorker")
        self.init_rpc_worker(self.global_rank, rpc_addr, hmac_key)
        self.start_rpc_server()

    def setup_engine(self):
        if torch.distributed.is_initialized(
        ) and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        super().setup_engine()

    def enqueue_request(self,
                        request: GenerationRequest,
                        result_wait_queue: Queue | None = None) -> int:
        return self._enqueue_request(request, result_wait_queue)

    @control_action_decorator
    def sleep(self, sleep_tags: List[str]):
        if not self.llm_args.enable_sleep:
            raise ValueError(
                "Sleep feature is not enabled, please set enable_sleep=True in the LLM arguments."
            )
        try:
            tags = verify_sleep_wakeup_tags(sleep_tags)
            logger.info(f"Sleep: {tags}")
            torch.cuda.synchronize()
            release_with_tag(*tags)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Encountered an error in sleep: {e}")
            raise e

    @control_action_decorator
    def wakeup(self, wakeup_tags: List[str]):
        if not self.llm_args.enable_sleep:
            raise ValueError(
                "Sleep feature is not enabled, please set enable_sleep=True in the LLM arguments."
            )
        try:
            tags = verify_sleep_wakeup_tags(wakeup_tags)
            logger.info(f"Wakeup: {tags}")
            torch.cuda.synchronize()
            materialize_with_tag(*tags)
            torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"Encountered an error in wakeup")
            raise e

    def start(self):
        pass

    def shutdown(self):

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        logger.debug(f'Worker {self.rank} shutting down...')

        if hasattr(self, 'shutdown_event'):
            self.shutdown_event.set()

        if hasattr(self, 'rpc_server') and self.rpc_server is not None:
            logger.info(f"[Rank {self.global_rank}] Shutting down RPC server")
            try:
                self.rpc_server.shutdown()
            except Exception as e:
                # Suppress errors during RPC server shutdown
                # These can occur if the server is already closed or during cleanup
                logger.debug(
                    f"[Rank {self.global_rank}] Suppressed error during RPC server shutdown: {e}"
                )
            self.rpc_server = None

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

    def _get_comm_ranks_device_id(self):
        # Make sure C++ executor would use same devices/ranks as py_executor
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        comm_ranks = [None] * world_size
        device_ids = [None] * world_size

        torch.distributed.all_gather_object(comm_ranks, global_rank)
        torch.distributed.all_gather_object(device_ids, self.device_id)

        self._configure_affinity(self.device_id)

        return comm_ranks, device_ids

    def __enter__(self):
        return self

    def __del__(self):
        self.shutdown()
