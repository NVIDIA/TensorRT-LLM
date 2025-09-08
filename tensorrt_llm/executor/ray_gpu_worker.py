import copy
import os
from pathlib import Path
from queue import Queue
from typing import Dict, Optional, Union

import ray
import torch

from tensorrt_llm.logger import logger

from .._utils import mpi_rank
from ..bindings import executor as tllm
from ..builder import Engine, EngineConfig
from ..llmapi.llm_args import BaseLlmArgs, KvCacheConnectorConfig, PybindMirror
from ..llmapi.tokenizer import TokenizerBase
from ..lora_manager import LoraConfig, LoraManager
from ..prompt_adapter_manager import PromptAdapterManager
from ..runtime import ModelConfig
from ..runtime.model_runner import _engine_config_to_model_config
from ..sampling_params import BatchedLogitsProcessor
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .request import GenerationRequest, LoRARequest, PromptAdapterRequest
from .result import GenerationResult
from .utils import RequestError

__all__ = [
    "RayGPUWorker",
    "RayWorkerWrapper",
]


@ray.remote
class RayWorkerWrapper:
    # Refer to https://github.com/NVIDIA-NeMo/RL/blob/faad02113c3c502437ccb339cb848796334aedd9/nemo_rl/models/policy/dtensor_policy_worker_v2.py#L95
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(self, worker_cls, worker_kwargs, world_size, rank):
        self.master_address = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]

        # Ray can't pickle TensorRT logger; import/use it inside methods only.
        from tensorrt_llm.logger import logger

        # Expect to see global counts w/ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1,
        # unless CUDA_VISIBLE_DEVICES is set to a subset of the global devices
        logger.debug(
            f"CUDA device count visible to Ray: {torch.cuda.device_count()}")

        torch.cuda.is_available()
        assert len(ray.get_gpu_ids()) == 1
        # Physical gpu id. Ray might return str and this would cause issues in cuda.set_device() w/o int
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

    @staticmethod
    def physical_to_local_id(phys_id: int) -> int:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible_devices:
            return phys_id
        id_mapping = list(map(int, visible_devices.split(",")))
        return id_mapping.index(phys_id)

    def call_worker_method(self, method_name: str, *args, **kwargs):
        """Generic method to call any method on the underlying worker."""
        if hasattr(self.worker, method_name):
            method = getattr(self.worker, method_name)
            if callable(method):
                return method(*args, **kwargs)
            else:
                raise AttributeError(
                    f"'{method_name}' is not callable on the underlying worker")
        else:
            raise AttributeError(
                f"Underlying worker has no method '{method_name}'")


class RayGPUWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        device_id: int,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        lora_config: Optional[LoraConfig] = None,
        kv_connector_config: Optional[KvCacheConnectorConfig] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
    ) -> None:
        postproc_config = postproc_worker_config or PostprocWorkerConfig()
        super().__init__(
            num_postprocess_workers=postproc_config.num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_config.postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self.engine = None
        self.device_id = device_id
        self.rank = torch.distributed.get_rank()
        self.global_rank = torch.distributed.get_world_size()

        # mapping: client_id from Proxy -> request_id returned from runtime backend
        self._client_id_to_request_id: Dict[int, int] = {}

        self._executor_config = executor_config
        self._is_pytorch_backend = llm_args is not None and llm_args.backend == "pytorch"
        self.llm_args = llm_args

        if not self._is_pytorch_backend:
            raise ValueError(f"Ray GPU worker only supports pytorch backend")

        if self.global_rank > 1:
            from tensorrt_llm.logger import logger
            logger.set_rank(self.global_rank)

        if isinstance(engine, list):
            engine = engine[self.rank]

        def _get_comm_ranks_device_id():
            # Make sure C++ executor would use same devices/ranks as py_executor
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            comm_ranks = [None] * world_size
            device_ids = [None] * world_size

            torch.distributed.all_gather_object(comm_ranks, global_rank)
            torch.distributed.all_gather_object(device_ids, self.device_id)
            return comm_ranks, device_ids

        def _create_py_executor():
            # Largely adapted from GenerationExecutorWorker. WAR: Keep in sync manually.

            args = {}
            assert hasattr(
                self.llm_args, "backend"
            ), "llm_args should be with backend in _create_py_executor"
            _ = _get_comm_ranks_device_id()
            if self.llm_args.backend == "pytorch":
                from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
                    create_py_executor
                create_executor = create_py_executor
                args["llm_args"] = self.llm_args
                args["checkpoint_dir"] = hf_model_dir
                args["tokenizer"] = tokenizer
                args["lora_config"] = lora_config
                args["kv_connector_config"] = kv_connector_config
            else:
                raise ValueError(
                    f"Unsupported backend config: {self.llm_args.backend}")

            # Define additional attributes that can be used later, such as in _deduce_max_tokens
            self.mapping = self.llm_args.parallel_config.to_mapping()
            self.checkpoint_loader = None
            if self.llm_args.backend == "pytorch":
                from tensorrt_llm._torch.pyexecutor.config import \
                    _construct_checkpoint_loader
                self.checkpoint_loader = _construct_checkpoint_loader(
                    self.llm_args.backend, self.llm_args.checkpoint_loader,
                    self.llm_args.checkpoint_format)

            _executor = create_executor(**args)
            self.max_seq_len = self.llm_args.max_seq_len
            if _executor.max_seq_len is not None:
                # max_seq_len might be updated by model engine as in create_py_executor
                self.max_seq_len = _executor.max_seq_len
            return _executor

        self.engine = _create_py_executor()

        self._lora_manager: Optional[LoraManager] = None
        self._prompt_adapter_manager: Optional[PromptAdapterManager] = None
        self._runtime_model_config: Optional[ModelConfig] = None
        if self.rank == 0 and isinstance(self.engine, tllm.Executor):
            if isinstance(engine, Engine):
                engine_config = engine.config
            else:
                engine_config = EngineConfig.from_json_file(
                    f"{engine}/config.json")
            self._runtime_model_config = _engine_config_to_model_config(
                engine_config)
            if engine_config.build_config.plugin_config.lora_plugin:
                # TODO(azuker): Passing peft cache manager to LoraManager is used for LoRA optimization
                # (see LoraManager constructor docstring). Getting the peft cache manager from this
                # point in the TRT flow is currently not supported (it's at the CPP
                # Executor->ExecutorImpl->TrtGptModel->mPeftCacheManager) therefore for now this LoRA
                # optimization is not available in TRT-python flow.
                self._lora_manager = LoraManager(cpp_peft_cache_manager=None)
            if engine_config.build_config.max_prompt_embedding_table_size > 0:
                self._prompt_adapter_manager = PromptAdapterManager()

        if self.llm_args and getattr(
                self.llm_args, "backend",
                "") == "pytorch" and lora_config is not None:
            from tensorrt_llm._torch.pyexecutor.resource_manager import \
                ResourceManagerType
            peft_cache_manager = self.engine.resource_manager.resource_managers.get(
                ResourceManagerType.PEFT_CACHE_MANAGER)
            self._lora_manager = LoraManager(
                cpp_peft_cache_manager=peft_cache_manager.impl)
            lora_model_config = self.engine.model_engine.lora_model_config
            assert lora_model_config is not None
            self._lora_model_config = lora_model_config

    def abort_request(self, client_id: int) -> None:
        # NOTE: the request_id is the request_id generated by cpp runtime, not the client_id
        if self.engine.can_enqueue_requests():
            request_id = self._client_id_to_request_id.get(client_id, None)
            if request_id is None:
                from tensorrt_llm.logger import logger
                logger.warning(
                    f"Request of client_id {client_id} is finished, cannot abort it."
                )
                return
            self.engine.cancel_request(request_id)

    def _load_lora_adapter(self, lora_request: LoRARequest) -> bool:
        """Returns True if the adapter was loaded by this call, False if it was already loaded"""
        # WAR: Copied from GenerationExecutorWorker. Keep in sync manually.
        adapter_id = str(lora_request.adapter_id)
        newly_loaded_uids = self._lora_manager.load_from_ckpt(
            [lora_request.path],
            model_config=self._runtime_model_config if
            self._runtime_model_config is not None else self._lora_model_config,
            runtime_mapping=None,
            uids=[adapter_id],
            ckpt_source=lora_request.ckpt_source)
        return adapter_id in newly_loaded_uids

    def _load_prompt_adapter(self,
                             prompt_adapter_request: PromptAdapterRequest):
        # WAR: Copied from GenerationExecutorWorker. Keep in sync manually.
        self._prompt_adapter_manager.load_from_ckpt(
            [prompt_adapter_request.local_path],
            model_config=self._runtime_model_config,
            uids=[str(prompt_adapter_request.adapter_id)])

    def enqueue_request(self,
                        request: GenerationRequest,
                        result_wait_queue: Queue | None = None) -> int:
        # Largely adapted from GenerationExecutorWorker. WAR: Keep in sync manually.
        assert request.id is not None
        py_lora_path = None
        if self._lora_manager is not None and request.lora_request is not None:
            adapter_in_cache = self._lora_manager.is_adapter_in_cpu_cache(
                request.lora_request.adapter_id)
            self._load_lora_adapter(request.lora_request)
            uid = str(request.lora_request.adapter_id)
            lora_config = tllm.LoraConfig(
                task_id=request.lora_request.adapter_id,
                weights=self._lora_manager.cpp_lora_weights[uid]
                if not adapter_in_cache else None,
                config=self._lora_manager.cpp_lora_config[uid])
            py_lora_path = request.lora_request.lora_path
        else:
            lora_config = None

        prompt_token_ids = copy.deepcopy(request.prompt_token_ids)
        prompt_tuning_config = None
        if request.prompt_adapter_request is not None:
            self._load_prompt_adapter(request.prompt_adapter_request)
            uid = str(request.prompt_adapter_request.adapter_id)
            prompt_tuning_config = tllm.PromptTuningConfig(
                self._prompt_adapter_manager.uid_to_weights[uid])
            vocab_size = self._runtime_model_config.vocab_size
            pa_length = prompt_tuning_config.embedding_table.size(0)
            prompt_token_ids = list(range(
                vocab_size, vocab_size + pa_length)) + prompt_token_ids

        # MULTIMODAL
        # NOTE: Since, we only support PyTorch backend for multimodal, we will send multimodal_data through the 'py_multimodal_data' field
        # except `multimodal_input` as it needs to go through the C++ runtime.
        multimodal_input = None
        if request.multimodal_params is not None and request.multimodal_params.has_content(
        ):
            if request.multimodal_params.multimodal_input is not None:
                multimodal_input = tllm.MultimodalInput(
                    multimodal_hashes=request.multimodal_params.
                    multimodal_input.multimodal_hashes,
                    multimodal_positions=request.multimodal_params.
                    multimodal_input.multimodal_positions,
                    multimodal_lengths=request.multimodal_params.
                    multimodal_input.multimodal_lengths)
            # NOTE: Setting to None here to avoid sending multimodal_input again through the 'py_multimodal_data' field
            request.multimodal_params.multimodal_input = None

        context_phase_params = None
        request_type = tllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        if request.disaggregated_params is not None:
            assert (
                not self._is_pytorch_backend
                or self.engine.kv_cache_transceiver is not None
            ), "kv_cache_transceiver is disabled, please set 'cache_transceiver_config: backend:<backend_type>` in config file for disaggregated serving"
            request_type = request.disaggregated_params.get_request_type()
            if request_type == tllm.RequestType.REQUEST_TYPE_GENERATION_ONLY:
                context_phase_params = request.disaggregated_params.get_context_phase_params(
                )

        if self._is_pytorch_backend:
            if not self.llm_args.disable_overlap_scheduler:
                is_disaggregated = self.engine.kv_cache_transceiver is not None
                if is_disaggregated and (
                        request_type
                        == tllm.RequestType.REQUEST_TYPE_CONTEXT_ONLY):
                    raise ValueError(
                        "Context only requests are not supported in pytorch backend when overlap is enabled."
                    )

        assert request.id is not None

        def _deduce_max_tokens(request: GenerationRequest,
                               executor_config: tllm.ExecutorConfig,
                               llm_args: Optional[BaseLlmArgs] = None) -> int:
            # deduce max_tokens when it's not set by user
            max_tokens = request.sampling_params.max_tokens
            query_token_len = len(
                request.query_token_ids) if request.query_token_ids else 0

            cp_size = 1
            max_seq_len = None
            if llm_args is not None:
                # deduce max_tokens by llm args
                assert executor_config is None, "An empty executor_config in _deduce_max_tokens is expected when LLM arguments are defined."
                if hasattr(self,
                           "mapping") and self.mapping.cp_size is not None:
                    cp_size = self.mapping.cp_size
                max_seq_len = getattr(self, "max_seq_len", None)
            else:
                # deduce max_tokens by executor config
                if hasattr(executor_config, "mapping"
                           ) and executor_config.mapping.cp_size is not None:
                    cp_size = executor_config.mapping.cp_size
                max_seq_len = getattr(executor_config, "max_seq_len", None)
            if max_seq_len is None:
                logger.warning("`default_max_tokens` cannot be deduced")
                if max_tokens is None:
                    raise ValueError(
                        "`max_tokens` must be set when `default_max_tokens` cannot be deduced"
                    )
                else:
                    # use max_tokens if can't deduce default_max_tokens
                    return max_tokens
            splited_prompt_len = int(len(prompt_token_ids) / cp_size)
            default_max_tokens = max_seq_len - splited_prompt_len - query_token_len
            if default_max_tokens <= 0:
                logger.warning(
                    f"`default_max_tokens` ({default_max_tokens}) should be greater than 0, "
                    f"`default_max_tokens` ({default_max_tokens}) = max_seq_len ({max_seq_len})"
                    f" - `splited_prompt_len` ({splited_prompt_len}) - `query_token_len` ({query_token_len})"
                )
                if max_tokens is None:
                    raise ValueError(
                        "`max_tokens` must be set when `default_max_tokens` is illegal"
                    )
            # default_max_tokens is the biggest available value
            if max_tokens is None:
                return default_max_tokens
            elif max_tokens > default_max_tokens:
                logger.warning(
                    f"User-specified `max_tokens` ({max_tokens}) is greater than deduced "
                    f"`default_max_tokens` ({default_max_tokens}), using default_max_tokens instead."
                )
                return default_max_tokens
            return max_tokens

        try:
            executor_request = tllm.Request(
                client_id=request.id,
                input_token_ids=prompt_token_ids,
                max_tokens=_deduce_max_tokens(request, self._executor_config,
                                              self.llm_args),
                streaming=request.streaming,
                sampling_config=request.sampling_params._get_sampling_config(),
                end_id=-1 if request.sampling_params.ignore_eos else
                request.sampling_params.end_id,
                pad_id=request.sampling_params.pad_id,
                output_config=request.sampling_params._get_output_config(
                    is_pytorch_backend=self._is_pytorch_backend),
                # Beam search enforces return_all_generated_tokens=True regardless of the passed value
                return_all_generated_tokens=False,
                # convert python config into pybind config
                lookahead_config=PybindMirror.maybe_to_pybind(
                    request.sampling_params.lookahead_config),
                guided_decoding_params=request.sampling_params.
                _get_guided_decoding_params(),
                bad_words=request.sampling_params._get_bad_words(),
                stop_words=request.sampling_params._get_stop_words(),
                embedding_bias=request.sampling_params.embedding_bias,
                lora_config=lora_config,
                prompt_tuning_config=prompt_tuning_config,
                multimodal_input=multimodal_input,
                # NOTE: `multimodal_embedding` and `mrope_config` will be in MultimodalParams.multimodal_data. And this will be handled below by `py_multimodal_data`.
                multimodal_embedding=None,
                mrope_config=None,
                logits_post_processor_name=(
                    tllm.Request.BATCHED_POST_PROCESSOR_NAME
                    if request.sampling_params.apply_batched_logits_processor
                    else None),
                logits_post_processor=None if self._is_pytorch_backend else
                request.sampling_params.logits_processor,
                kv_cache_retention_config=request.kv_cache_retention_config,
                context_phase_params=context_phase_params,
                type=request_type,
                cache_salt_id=request.cache_salt_id)
            executor_request.py_lora_path = py_lora_path

            if self._is_pytorch_backend and request.multimodal_params is not None:
                if request.multimodal_params.multimodal_data is not None:
                    # NOTE: Deserialize SharedTensor handle to actual tensor
                    request.multimodal_params.to_tensor("multimodal_data")
                    executor_request.py_multimodal_data = request.multimodal_params.multimodal_data

            if self._is_pytorch_backend and request.sampling_params.logits_processor:
                # For PyTorch backend, we attach logits processors as a dynamic Python attribute
                # instead of using the C++ binding, since the latter will cause PyCapsule pickling issues.
                lp = request.sampling_params.logits_processor
                executor_request.py_logits_post_processors = lp if isinstance(
                    lp, list) else [lp]

            executor_request.py_scheduling_params = None
            if self._is_pytorch_backend and request.scheduling_params is not None:
                executor_request.py_scheduling_params = request.scheduling_params

            if request.query_token_ids is not None:
                # pytorch star attention workflow
                # a workaround to avoid public interface update
                req_id = self.engine.enqueue_request(
                    executor_request,
                    request.query_token_ids,
                    result_wait_queue=result_wait_queue)
            else:
                req_id = self.engine.enqueue_request(
                    executor_request, result_wait_queue=result_wait_queue)
            self._client_id_to_request_id[request.id] = req_id
            return req_id
        except Exception as e:
            raise RequestError(str(e)) from e

    def submit(self, request: GenerationRequest):
        raise NotImplementedError(
            "Ray GPU worker does not support submit() yet.")

    def shutdown(self):

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        from tensorrt_llm.logger import logger
        logger.debug(f'Worker {mpi_rank()} shutting down...')

        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

            if self.llm_args is not None:
                assert self._executor_config is None, "An empty executor_config is expected in shutdown when LLM arguments are defined."
                if (self.llm_args.backend == "pytorch"
                        and hasattr(self, "checkpoint_loader")
                        and self.checkpoint_loader is not None):
                    self.checkpoint_loader.cleanup()
                    self.checkpoint_loader = None
            else:
                if hasattr(
                        self._executor_config, "checkpoint_loader"
                ) and self._executor_config.checkpoint_loader is not None:
                    self._executor_config.checkpoint_loader.cleanup()
                    self._executor_config.checkpoint_loader = None

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

        logger.debug(f"Worker {mpi_rank()} shutdown done.")

    def block_subordinates(self):
        # WAR: Copied from GenerationExecutorWorker. Keep in sync manually.
        if self.rank != 0:
            if isinstance(self.engine, tllm.Executor):
                self.shutdown()
                raise self.WorkerExit(
                    "block_subordinates() should be used in a `with GenerationExecutorWorker() as ...:` block"
                )
            from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
            if isinstance(self.engine, PyExecutor):
                self.engine.wait_shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown()
        return exc_type is None or exc_type == RayGPUWorker.WorkerExit

    def __del__(self):
        self.shutdown()
