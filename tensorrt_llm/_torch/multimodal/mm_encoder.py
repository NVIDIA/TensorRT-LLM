from pathlib import Path
from typing import Any, Optional, Union, List

from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.llmapi.llm import TorchLlmArgs, TrtLlmArgs
from tensorrt_llm.llmapi.utils import exception_handler, get_device_count, print_colored_debug
from tensorrt_llm.llmapi.llm_utils import LlmBuildStats, CachedModelLoader, _ModelRuntimeContext
from tensorrt_llm.logger import logger
from tensorrt_llm.executor.utils import get_spawn_proxy_process_env, create_mpi_comm_session
from tensorrt_llm.llmapi.mpi_session import external_mpi_comm_available, MpiPoolSession
import tempfile
import atexit
import weakref
from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.executor.multimodal import MultimodalRequest
import asyncio
from tensorrt_llm.bindings import executor as tllm

class MultimodalEncoder:
    def __init__(self,
                 model: Union[str, Path],
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,  # TP should never be used for mm-encoder
                 data_parallel_size: int = 1,  # placeholder for future use
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 **kwargs: Any) -> None:

        self._executor_cls = kwargs.pop("executor_cls", GenerationExecutor)

        kwargs_dict = dict(kwargs)
        kwargs_dict['backend'] = 'pytorch'
        try:
            # Reuse the LLM arg parser for mm-encoder for now as some configs/args can be shared
            # e.g., max_batch_size, parallel_config, mpi_session, etc.
            self.args = TorchLlmArgs.from_kwargs(
                model=model,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                revision=revision,
                **kwargs_dict)

        except Exception as e:
            logger.error(
                f"Failed to parse the arguments for the mm encoder constructor: {e}")
            raise e


        print_colored_debug(f"Encoder.args.mpi_session: {self.args.mpi_session}\n",
                            "yellow")
        self.mpi_session = self.args.mpi_session

        if self.args.parallel_config.is_multi_gpu:
            if get_device_count(
            ) < self.args.parallel_config.world_size_per_node:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.args.parallel_config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.args.parallel_config.world_size} workers'
            )
            if not self.mpi_session:
                mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()
                if not mpi_process_pre_spawned:
                    print_colored_debug(f"Encoder create MpiPoolSession\n",
                                        "yellow")
                    self.mpi_session = MpiPoolSession(
                        n_workers=self.args.parallel_config.world_size)
                else:
                    print_colored_debug(f"Encoder create MpiCommSession\n",
                                        "yellow")
                    self.mpi_session = create_mpi_comm_session(
                        self.args.parallel_config.world_size)

        try:
            # Due to the Executor can only accept a engine path, we need to save the engine to a directory
            self._engine_dir: Optional[Path] = None
            self._executor: Optional[GenerationExecutor] = None
            if self._on_trt_backend:
                self._workspace = tempfile.TemporaryDirectory(
                    suffix="-mm-encoder-workspace", dir=self.args.workspace)
            else:
                self._workspace = None

            self._hf_model_dir: Optional[Path] = None

            self.runtime_context: Optional[_ModelRuntimeContext] = None
            self.llm_build_stats = LlmBuildStats()

            self._build_model()

        except Exception as e:
            if self.mpi_session is not None:
                self.mpi_session.shutdown()
            raise e

        exception_handler.register(self, 'shutdown')
        atexit.register(MultimodalEncoder._shutdown_wrapper, weakref.ref(self))

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name) if self._on_trt_backend else None

    def generate_from_mm_request(
        self,
        mm_requests: List[MultimodalRequest],
    ):
        """Generate embeddings for multiple multimodal requests in parallel.

        Args:
            mm_requests: List of multimodal requests to process

        Returns:
            List of generation results
        """
        async def _process_requests():
            # Submit all requests first
            futures = []
            for request in mm_requests:
                future = await self.generate_async(request)
                futures.append(future)

            # Then wait for all results
            results = []
            for future in futures:
                result = await future.aresult()
                results.append(result)
            return results

        # Run the async operations in an event loop
        return asyncio.run(_process_requests())

    @nvtx_range_debug("Encoder.generate_async", color="green", category="Encoder")
    async def generate_async(
        self,
        mm_request: MultimodalRequest,
    ):
        """Generate embeddings for a multimodal request asynchronously.

        Args:
            mm_request: The multimodal request containing items to process

        Returns:
            A promise that will be resolved with the generation results
        """
        # First fetch and load all the data
        await mm_request.fetch()
        # Then generate the embeddings asynchronously
        result = self._executor.generate_multimodal_async(
            mm_request,
        )
        return result


    def _build_model(self):
        model_loader = CachedModelLoader(self.args,
                                         mpi_session=self.mpi_session,
                                         workspace=self.workspace,
                                         llm_build_stats=weakref.proxy(
                                             self.llm_build_stats))
        self._engine_dir, self._hf_model_dir = model_loader()
        # update the model_dir to a local dir for the runtime, such as tokenizer loading.
        if self._engine_dir is not None:
            self.args.model = self._engine_dir

        max_batch_size = self.args.max_batch_size or self.args.build_config.max_batch_size
        # In _build_model method:
        executor_config = tllm.ExecutorConfig(1)
        executor_config.backend = "pytorch"
        executor_config.mm_encoder_only = True
        executor_config.mapping = self.args.parallel_config.to_mapping()
        executor_config.build_config = self.args.build_config
        executor_config.hf_model_dir = self._hf_model_dir
        executor_config.trt_engine_dir = self._engine_dir
        executor_config.max_batch_size = max_batch_size
        executor_config.max_num_active_requests = 2048


        self._executor = self._executor_cls.create(
            self._engine_dir,
            executor_config=executor_config,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size),
            is_llm_executor=False)

    @property
    def _on_trt_backend(self) -> bool:
        return isinstance(self.args, TrtLlmArgs)

    def shutdown(self) -> None:
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        if hasattr(self, 'mpi_session') and self.mpi_session is not None:
            self.mpi_session.shutdown()
            self.mpi_session = None

    @staticmethod
    def _shutdown_wrapper(self_ref):
        # Retrieve the instance if it still exists
        instance = self_ref()
        if instance is not None:
            instance.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self.shutdown()
        return False  # propagate exceptions

    def __getstate__(self):
        raise RuntimeError("Encoder object can not be pickled.")

    def __del__(self):
        self.shutdown()