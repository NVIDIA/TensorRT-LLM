# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import enum
import gc
import json
import weakref
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm.logger import logger

from .._torch.pyexecutor.kv_cache_stats import append_kv_cache_iteration_stats
from .._torch.pyexecutor.llm_request import LlmResponse
from .._utils import (global_mpi_rank, global_mpi_size, mpi_comm, mpi_rank,
                      nvtx_range_debug)
from ..bindings import executor as tllm
from ..builder import ConfigEncoder, Engine, EngineConfig
from ..llmapi.llm_args import BaseLlmArgs, ExecutorMemoryType, PybindMirror
from ..llmapi.tokenizer import TokenizerBase
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import _SyncQueue, configure_cpu_affinity, logger_debug
from ..lora_manager import LoraManager
from ..prompt_adapter_manager import PromptAdapterManager
from ..runtime import ModelConfig
from ..runtime.model_runner import _engine_config_to_model_config
from ..sampling_params import BatchedLogitsProcessor, SamplingParams
from .executor import GenerationExecutor, IterationResultQueue
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import (PostprocParams, PostprocWorker,
                              PostprocWorkerConfig)
from .request import GenerationRequest, LoRARequest, PromptAdapterRequest
from .result import (GenerationResult, LogProbsResult, ResponseWrapper,
                     compute_logprobs, get_metrics_dict)
from .utils import (ErrorResponse, IntraProcessQueue, RequestError,
                    is_llm_response)

if TYPE_CHECKING:
    from ..disaggregated_params import DisaggregatedParams

__all__ = [
    "BaseWorker",
    "_init_hf_modules",
]


def _init_hf_modules():
    """Initialize cached HuggingFace modules for models with trust_remote_code=True.

    This is safe to call multiple times (idempotent) and should be called:
    1. At module import time (for main process and spawned subprocesses)
    2. At worker_main entry (for forked processes or external MPI ranks)

    References: https://github.com/vllm-project/vllm/pull/871
    """
    try:
        from transformers.dynamic_module_utils import init_hf_modules
        init_hf_modules()
        logger.debug("HF modules initialized")
    except ImportError as e:
        logger.warning(f"ImportError initializing HF modules: {e}")
    except Exception as e:
        logger.error(f"Exception initializing HF modules: {e}")


_init_hf_modules()


class BaseWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
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

        # inputs
        self._engine = engine
        self._executor_config = executor_config
        self._batched_logits_processor = batched_logits_processor
        self._postproc_worker_config = postproc_worker_config
        self._is_llm_executor = is_llm_executor
        self._hf_model_dir = hf_model_dir
        self._tokenizer = tokenizer
        self.llm_args = llm_args

        self.engine = None
        self.result_queue: Optional[IpcQueue] = None
        self.postproc_queues: Optional[List[IpcQueue]] = None
        self.rank = mpi_rank()
        self.global_rank = global_mpi_rank()
        # mapping: client_id -> GenerationResult
        self._results: Dict[int, GenerationResult] = {}
        # mapping: client_id from Proxy -> request_id returned from runtime backend
        self._client_id_to_request_id: Dict[int, int] = {}
        self._await_response_helper = AwaitResponseHelper(weakref.proxy(self))
        self._backend = None if llm_args is None else llm_args.backend
        self._is_pytorch_backend = self._backend in ["pytorch", "_autodeploy"]
        self._lora_config = llm_args.lora_config if self._is_pytorch_backend else None
        self._resource_governor_queue = None

        if global_mpi_size() > 1:
            logger.set_rank(self.global_rank)

    @property
    def resource_governor_queue(self):
        return self._resource_governor_queue

    def _get_comm_ranks_device_id(self):
        device_id = self.global_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        # Make sure C++ executor would use same devices/ranks as py_executor
        global_rank = global_mpi_rank()
        comm_ranks = mpi_comm().allgather(global_rank)
        device_ids = mpi_comm().allgather(device_id)

        configure_cpu_affinity(device_id)

        return comm_ranks, device_ids

    def setup_engine(self):
        """
        Setup the engine for the worker.
        """

        if isinstance(self._engine, list):
            self._engine = self._engine[self.rank]

        def _create_py_executor():
            args = {}
            assert hasattr(
                self.llm_args, "backend"
            ), "llm_args should be with backend in _create_py_executor"
            _ = self._get_comm_ranks_device_id()
            if self._backend == "pytorch":
                from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
                    create_py_executor
                create_executor = create_py_executor
                args["llm_args"] = self.llm_args
                args["checkpoint_dir"] = self._hf_model_dir
                args["tokenizer"] = self._tokenizer
            elif self._backend == "_autodeploy":
                from tensorrt_llm._torch.auto_deploy.llm_args import \
                    LlmArgs as ADLlmArgs
                from tensorrt_llm._torch.auto_deploy.shim.ad_executor import \
                    create_autodeploy_executor
                create_executor = create_autodeploy_executor
                assert isinstance(self.llm_args, ADLlmArgs)
                args["ad_config"] = self.llm_args
                args["tokenizer"] = self._tokenizer
            else:
                raise ValueError(f"Unsupported backend config: {self._backend}")

            if self._resource_governor_queue is not None:
                args["resource_governor_queue"] = self._resource_governor_queue

            # Define additional attributes that can be used later, such as in _deduce_max_tokens
            self.mapping = self.llm_args.parallel_config.to_mapping()
            self.checkpoint_loader = None
            if self._backend == "pytorch":
                from tensorrt_llm._torch.pyexecutor.model_loader import \
                    _construct_checkpoint_loader
                self.checkpoint_loader = _construct_checkpoint_loader(
                    self.llm_args.backend,
                    self.llm_args.checkpoint_loader,
                    self.llm_args.checkpoint_format,
                    mx_config=self.llm_args.mx_config,
                    mx_model_name=self.llm_args.model,
                )

            self.max_seq_len = self.llm_args.max_seq_len
            # creare_py_executor may change some fields of llm_args
            _executor = create_executor(**args)
            if _executor.max_seq_len is not None:
                # max_seq_len might be updated by model engine as in create_py_executor
                self.max_seq_len = _executor.max_seq_len
            return _executor

        def _create_engine(executor_config):
            engine = self._engine
            if executor_config is None:
                executor_config = tllm.ExecutorConfig(1)
            executor_config.logits_post_processor_config = tllm.LogitsPostProcessorConfig(
                processor_batched=self._batched_logits_processor,
                replicate=False)
            comm_ranks, device_ids = self._get_comm_ranks_device_id()
            executor_config.parallel_config = tllm.ParallelConfig(
                participant_ids=comm_ranks, device_ids=device_ids)

            if isinstance(engine, Engine):
                return tllm.Executor(engine.engine,
                                     json.dumps(engine.config.to_dict(),
                                                cls=ConfigEncoder),
                                     tllm.ModelType.DECODER_ONLY,
                                     executor_config=executor_config,
                                     managed_weights=engine.managed_weights)

            assert not hasattr(executor_config, "backend")
            return tllm.Executor(engine, tllm.ModelType.DECODER_ONLY,
                                 executor_config)

        self.engine = _create_py_executor(
        ) if self.llm_args is not None else _create_engine(
            self._executor_config)

        self._lora_manager: Optional[LoraManager] = None
        self._prompt_adapter_manager: Optional[PromptAdapterManager] = None
        self._runtime_model_config: Optional[ModelConfig] = None
        if self.rank == 0 and isinstance(self.engine, tllm.Executor):
            if isinstance(self.engine, Engine):
                engine_config = self.engine.config
            else:
                engine_config = EngineConfig.from_json_file(
                    f"{self._engine}/config.json")
            self._runtime_model_config = _engine_config_to_model_config(
                engine_config)
            if engine_config.build_config.plugin_config.lora_plugin:
                # TODO(azuker): Passing peft cache manager to LoraManager is used for LoRA optimization
                # (see LoraManager constructor docstring). Getting the peft cache manager from this
                # point in the TRT flow is currently not supported (it's at the CPP
                # Executor->ExecutorImpl->TrtGptModel->mPeftCacheManager) therefore for now this LoRA
                # optimization is not available in TRT-python flow.
                self._lora_manager = LoraManager(
                    mapping=engine_config.pretrained_config.mapping,
                    model_config=self._runtime_model_config,
                    cpp_peft_cache_manager=None)
            if engine_config.build_config.max_prompt_embedding_table_size > 0:
                self._prompt_adapter_manager = PromptAdapterManager()

        if self._backend == "pytorch" and self._lora_config is not None:
            from tensorrt_llm._torch.pyexecutor.resource_manager import \
                ResourceManagerType
            peft_cache_manager = self.engine.resource_manager.resource_managers.get(
                ResourceManagerType.PEFT_CACHE_MANAGER)
            self._lora_manager = peft_cache_manager.get_lora_manager()
            lora_model_config = self.engine.model_engine.lora_model_config
            assert lora_model_config is not None
            self._lora_model_config = lora_model_config

    def await_responses(self, timeout: Optional[float] = None) -> list:
        return self.engine.await_responses(timeout=datetime.timedelta(
            seconds=timeout) if timeout is not None else None)

    def fetch_stats(self) -> list:
        if isinstance(self.engine, tllm.Executor):
            iter_stats = self.engine.get_latest_iteration_stats()
            #TODO: Support req stats with TRT engine
            #      This would require ensuring iter and req stats have same size
            return [(iter_stat, None, None) for iter_stat in iter_stats]
        else:
            return self.engine.get_latest_iteration_stats()

    def fetch_kv_cache_capacity(self) -> dict:
        if self.engine is None or isinstance(self.engine, tllm.Executor):
            return {}

        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        if isinstance(self.engine, PyExecutor):
            return self.engine.get_kv_cache_capacity()

        return {}

    def fetch_kv_cache_events(self) -> list:
        if isinstance(self.engine, tllm.Executor):
            return self.engine.get_latest_kv_cache_events()
        else:
            return self.engine.get_latest_kv_cache_events()

    def set_result_queue(self, queue):
        """In multi-gpu mode, result_queue will be set here to communicate between the proxy and the worker 0 process."""
        assert self.postproc_queues is None
        self.result_queue = queue

    def set_postproc_queues(self, queues: List["IpcQueue"]):
        """ Set the IPC queues for feeding post-processing processes. """
        assert self.result_queue is None
        self.postproc_queues = queues

    def _set_iteration_result_queue(self, it_result_queue: IterationResultQueue,
                                    queue: Union[Queue, FusedIpcQueue,
                                                 IntraProcessQueue]):
        assert not it_result_queue.is_initialized, "Iteration result queue should not already be initialized."
        it_result_queue.is_initialized = True
        it_result_queue.queue = queue
        it_result_queue.aqueue = None

    def return_queue(self, client_id: int):
        """Return the queue used to deliver responses for ``client_id``.

        If a centralized result queue is registered (used for communication
        with the proxy) send the message there. Otherwise, push the result
        directly in the GenerationResult queue.
        """
        if self.result_queue is not None:
            return self.result_queue
        return self._results[client_id].queue

    def abort_request(self, client_id: int) -> None:
        # NOTE: the request_id is the request_id generated by cpp runtime, not the client_id
        if self.engine.can_enqueue_requests():
            request_id = self._client_id_to_request_id.get(client_id, None)
            if request_id is None:
                logger.warning(
                    f"Request of client_id {client_id} is finished, cannot abort it."
                )
                return
            self.engine.cancel_request(request_id)

    def _engine_response_callback(self, response: tllm.Response):
        return response

    def _has_background_error(self) -> bool:
        return not self._error_queue.empty()

    def _create_error_response(self, response: tllm.Response) -> ErrorResponse:
        bck_error = self._error_queue.get_nowait()
        assert isinstance(bck_error, Exception)
        return ErrorResponse(response.client_id, str(bck_error),
                             response.request_id)

    def start(self):
        raise NotImplementedError(
            "start method is not implemented in BaseWorker")

    def _load_lora_adapter(self, lora_request: LoRARequest) -> bool:
        """Returns True if the adapter was loaded by this call, False if it was already loaded"""
        adapter_id = str(lora_request.adapter_id)
        newly_loaded_uids = self._lora_manager.load_from_ckpt(
            [lora_request.path],
            model_config=self._runtime_model_config if
            self._runtime_model_config is not None else self._lora_model_config,
            uids=[adapter_id],
            ckpt_source=lora_request.ckpt_source)
        return adapter_id in newly_loaded_uids

    def _load_prompt_adapter(self,
                             prompt_adapter_request: PromptAdapterRequest):
        self._prompt_adapter_manager.load_from_ckpt(
            [prompt_adapter_request.local_path],
            model_config=self._runtime_model_config,
            uids=[str(prompt_adapter_request.adapter_id)])

    def _enqueue_request(self,
                         request: GenerationRequest,
                         result_wait_queue=None) -> int:
        assert request.id is not None
        py_lora_path = None
        if self._lora_manager is not None and request.lora_request is not None:
            try:
                if self._is_pytorch_backend:
                    # PyTorch backend: don't embed weights in the request.
                    # Each rank loads independently from disk via py_lora_path
                    # in PeftCacheManager.add_request_peft().
                    # Pre-load on rank 0 to warm the LoRA manager cache so that
                    # add_request_peft finds the adapter already loaded.
                    self._load_lora_adapter(request.lora_request)
                    uid = str(request.lora_request.adapter_id)
                    lora_config = tllm.LoraConfig(
                        task_id=request.lora_request.adapter_id,
                        weights=None,
                        config=self._lora_manager.cpp_lora_config[uid])
                else:
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
            except Exception as e:
                raise RequestError(f"Failed to load LoRA adapter: {e}") from e
        else:
            lora_config = None

        # prompt_token_ids stays list[int] for all consumers. If an int32 buffer
        # rode along on the wire (GenerationRequest._prompt_token_ids_i32), hand
        # THAT to the C++ Request ctor (memcpy) instead of the list -- this avoids
        # the O(ISL) list copy + element-wise nanobind cast on the GIL-held submit
        # thread. No buffer (in-process, or prompt-adapter prepend) -> list path.
        # If both forms exist, they are assumed to describe the same token ids;
        # in-place mutations of request.prompt_token_ids must clear the i32 buffer.
        i32_buf = getattr(request, "_prompt_token_ids_i32", None)
        if i32_buf is not None and request.prompt_adapter_request is None:
            prompt_token_ids = i32_buf
        else:
            prompt_token_ids = list(request.prompt_token_ids)
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
                multimodal_input = request.multimodal_params.multimodal_input.to_binding(
                    tllm)
            # NOTE: Setting to None here to avoid sending multimodal_input again through the 'py_multimodal_data' field
            request.multimodal_params.multimodal_input = None

        context_phase_params = None
        request_type = tllm.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        disagg_request_id = 0

        if request.disaggregated_params is not None:
            assert (
                not self._is_pytorch_backend
                or self.engine.kv_cache_transceiver is not None
                or request.disaggregated_params.request_type
                == "context_and_generation"
            ), "kv_cache_transceiver is disabled, please set 'cache_transceiver_config: backend:<backend_type>` in config file for disaggregated serving"
            request_type = request.disaggregated_params.get_request_type()
            disagg_request_id = request.disaggregated_params.disagg_request_id
            if request_type == tllm.RequestType.REQUEST_TYPE_GENERATION_ONLY:
                context_phase_params = request.disaggregated_params.get_context_phase_params(
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
            if executor_config is not None:
                assert (
                    len(prompt_token_ids) <= executor_config.max_seq_len
                ), f"`prompt_token_ids` length ({len(prompt_token_ids)}) is greater than `max_seq_len` ({executor_config.max_seq_len})"
            splited_prompt_len = int(len(prompt_token_ids) / cp_size)
            default_max_tokens = max_seq_len - splited_prompt_len - query_token_len
            if default_max_tokens <= 0:
                # Raise error on `default_max_tokens` not enough, since max_tokens should be less than `default_max_tokens``
                raise ValueError(
                    f"`default_max_tokens` ({default_max_tokens}) must be greater than 0, "
                    f"`default_max_tokens` ({default_max_tokens}) = max_seq_len ({max_seq_len})"
                    f" - `splited_prompt_len` ({splited_prompt_len}) - `query_token_len` ({query_token_len})"
                )

            # default_max_tokens is the biggest available value
            if max_tokens is None:
                return default_max_tokens
            elif max_tokens > default_max_tokens and default_max_tokens > 0:
                logger.warning(
                    f"User-specified `max_tokens` ({max_tokens}) is greater than deduced "
                    f"`default_max_tokens` ({default_max_tokens}), using default_max_tokens instead."
                )
                return default_max_tokens
            elif max_tokens <= 0:
                raise ValueError(
                    f"`max_tokens` ({max_tokens}) must be greater than 0")
            else:
                return max_tokens

        try:
            executor_request = tllm.Request(
                client_id=request.id,
                input_token_ids=prompt_token_ids,
                max_tokens=_deduce_max_tokens(
                    request,
                    self._executor_config if not self.llm_args else None,
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
                stop_words=[] if request.sampling_params.ignore_eos else
                request.sampling_params._get_stop_words(),
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
                encoder_input_token_ids=request.encoder_input_token_ids,
                type=request_type,
                disagg_request_id=disagg_request_id,
                cache_salt=request.cache_salt,
                priority=request.priority)
            executor_request.py_original_end_id = request.sampling_params.end_id
            executor_request.py_num_logprobs = request.sampling_params.logprobs
            executor_request.py_lora_path = py_lora_path
            executor_request.py_logprobs_mode = request.sampling_params.logprobs_mode
            executor_request.py_logprobs_simple_format = (
                request.sampling_params.logprobs_simple_format)

            # here we add executor_request.py_disaggregated_params= request.disaggregated_params for python cache transceiver
            if self._is_pytorch_backend and request.disaggregated_params is not None:
                executor_request.py_disaggregated_params = request.disaggregated_params
            if self._is_pytorch_backend and request.multimodal_params is not None:
                if request.multimodal_params.multimodal_data is not None:
                    # Resolve SharedTensorContainer dicts inside multimodal_data, including
                    # E/P handoff embedding handles parked under "multimodal_embedding".
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

            executor_request.py_conversation_params = None
            if self._is_pytorch_backend and request.conversation_params is not None:
                executor_request.py_conversation_params = request.conversation_params

            if request.arrival_time is not None:
                executor_request.py_arrival_time = request.arrival_time

            if request.query_token_ids is not None:
                # pytorch star attention workflow
                # a workaround to avoid public interface update
                if self._is_pytorch_backend and result_wait_queue is not None:
                    req_id = self.engine.enqueue_request(
                        executor_request,
                        request.query_token_ids,
                        result_wait_queue=result_wait_queue)
                else:
                    req_id = self.engine.enqueue_request(
                        executor_request, request.query_token_ids)
            else:
                if self._is_pytorch_backend and result_wait_queue is not None:
                    req_id = self.engine.enqueue_request(
                        executor_request, result_wait_queue=result_wait_queue)
                else:
                    req_id = self.engine.enqueue_request(executor_request)
            return req_id
        except Exception as e:
            raise RequestError(str(e)) from e

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """ Low-level API to the executor. Return a "future" GenerationResult which can be waited. """
        self.start()

        if self.rank != 0:
            raise RuntimeError(
                "Only rank 0 can submit requests.\n"
                "To fix this, ensure that the llm.generate(...) method is "
                "guarded with the `if __name__ == '__main__':` block.")

        client_id = request.id if request.id is not None else self._get_next_client_id(
        )
        if request.id is None:
            request.set_id(client_id)

        logprob_params = self._get_logprob_params(request)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)

        self._results[client_id] = result

        request_id = self._enqueue_request(request)
        # request_id returned from backend is necessary for the abort_request method.
        self._client_id_to_request_id[client_id] = request_id

        self._handle_background_error()

        return result

    def _check_sleep_wakeup_preconditions(self, method: str) -> None:
        """Validate preconditions shared by sleep() and wakeup().

        Args:
            method: Name of the calling method (``"sleep"`` or ``"wakeup"``)
                used in error messages.

        Raises:
            ValueError: If the backend is not ``"pytorch"`` or
                ``sleep_config`` is not set.
            NotImplementedError: If ``parallel_config.world_size > 1``.
        """
        # _autodeploy is intentionally excluded: its allocations are not tagged
        # under sleep_config VMM scopes, so release_with_tag would silently
        # no-op instead of actually freeing GPU memory.  Use _backend directly
        # rather than _is_pytorch_backend, which also covers _autodeploy.
        if self._backend != "pytorch":
            raise ValueError(
                f"{method}() is only available for the PyTorch (TorchLLM) "
                "backend.")
        if self.llm_args is None or self.llm_args.sleep_config is None:
            raise ValueError(
                "Sleep feature is not enabled, please set sleep_config in "
                "the LLM arguments.")
        # Non-rank-0 processes block on their local control_action_done
        # threading.Event with no Python caller to release it — deadlock.
        if self.llm_args.parallel_config.world_size > 1:
            raise NotImplementedError(
                f"{method}() requires parallel_config.world_size == 1; "
                "use the Ray executor for multi-rank deployments.")

    def sleep(self, sleep_tags: List[str]) -> None:
        """Release GPU virtual memory for the specified memory type tags.

        Single-rank (``world_size == 1``) only.  Uses
        ``PyExecutor.control_action()`` to drain in-flight requests and pause
        the event loop before calling ``release_with_tag()``, matching the
        ``@control_action_decorator`` behaviour used in Ray.

        Only allocations backed by virtual memory (VMM) and registered under
        the active :class:`~tensorrt_llm.llmapi.llm_args.SleepConfig` are
        released.  Components using alternative memory management (e.g.
        ``LoadFormat.GMS``-managed weights) are not VMM-tagged and will be
        silently skipped by ``release_with_tag``.

        Args:
            sleep_tags: List of
                :class:`~tensorrt_llm.llmapi.llm_args.ExecutorMemoryType`
                value strings (e.g. ``["kv_cache"]``).

        Returns:
            None.  The call is synchronous; when it returns all requested
            VMM-tagged allocations have been released and the event loop
            has been resumed.

        Raises:
            ValueError: If the backend is not ``"pytorch"`` or
                ``sleep_config`` is not set.
            NotImplementedError: If ``parallel_config.world_size > 1``.
        """
        self._check_sleep_wakeup_preconditions("sleep")

        from tensorrt_llm._torch.virtual_memory import release_with_tag

        tags = [ExecutorMemoryType(tag) for tag in sleep_tags]
        logger.info(f"Sleep: {tags}")
        with self.engine.control_action():
            torch.cuda.synchronize()
            release_with_tag(*tags)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def wakeup(self, wakeup_tags: List[str]) -> None:
        """Materialize GPU virtual memory for the specified memory type tags.

        Single-rank (``world_size == 1``) only.  See :meth:`sleep` for
        details on VMM scope restrictions and backend prerequisites.

        Args:
            wakeup_tags: List of
                :class:`~tensorrt_llm.llmapi.llm_args.ExecutorMemoryType`
                value strings (e.g. ``["kv_cache"]``).

        Returns:
            None.  The call is synchronous; when it returns all requested
            VMM-tagged allocations have been materialized and the event loop
            has been resumed.

        Raises:
            ValueError: If the backend is not ``"pytorch"`` or
                ``sleep_config`` is not set.
            NotImplementedError: If ``parallel_config.world_size > 1``.
        """
        self._check_sleep_wakeup_preconditions("wakeup")

        from tensorrt_llm._torch.virtual_memory import materialize_with_tag

        tags = [ExecutorMemoryType(tag) for tag in wakeup_tags]
        logger.info(f"Wakeup: {tags}")
        with self.engine.control_action():
            torch.cuda.synchronize()
            materialize_with_tag(*tags)
            torch.cuda.synchronize()

    def shutdown(self):
        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        if self.engine is not None and self.engine.can_enqueue_requests():
            self.engine.shutdown()
            self.engine = None

    def get_disaggregated_params(self) -> dict:
        if self.engine is None or self.engine.kv_cache_transceiver is None:
            logger.warning("Engine or kv cache transceiver is not initialized")
            return {}
        return self.engine.kv_cache_transceiver.get_disaggregated_params()

    @staticmethod
    def _stats_serializer(stats) -> str:
        # Per-rank path: stats is ("per_rank_dict", {..., "rank": N}).
        # Already serialized on the producing rank via allgather — just emit.
        if (isinstance(stats, tuple) and len(stats) == 2
                and stats[0] == "per_rank_dict"):
            return json.dumps(stats[1])

        iteration_stats, req_stats = stats[0], stats[1]
        kv_iter_stats = stats[2] if len(stats) > 2 else None
        attention_dp_rank = stats[3] if len(stats) > 3 else None
        # Newer slots — guarded with len() checks so historical 4-tuples and
        # any external code still appending the legacy shape keep working.
        host_step_time_ms = stats[4] if len(stats) > 4 else None
        prev_device_step_time_ms = stats[5] if len(stats) > 5 else None
        scheduler_mode = stats[6] if len(stats) > 6 else None
        gpu_forward_time_ms = stats[7] if len(stats) > 7 else None

        stats_dict = json.loads(iteration_stats.to_json_str())
        # Always tag the row so Dynamo's adapter can read
        # stat["attentionDpRank"] without a missing-key branch. Non-ADP stats
        # default to rank 0; ADP stats carry the rank supplied by PyExecutor.
        stats_dict["attentionDpRank"] = (0 if attention_dp_rank is None else
                                         attention_dp_rank)

        if req_stats is not None and len(req_stats) > 0:
            stats_dict["requestStats"] = []
            for req_stat in req_stats:
                stats_dict["requestStats"].append(
                    json.loads(req_stat.to_json_str()))

        append_kv_cache_iteration_stats(stats_dict, kv_iter_stats)

        # Per-loop CPU wall captured by profile_step() — always a clean
        # single-loop measurement, matching the log line's `host_step_time`.
        # Prefer this over iterLatencyMS when you need absolute per-loop
        # CPU cost, especially under the overlap scheduler where
        # iterLatencyMS measures the batch's full lifecycle (~2 loops).
        if host_step_time_ms is not None:
            stats_dict["hostStepTimeMS"] = host_step_time_ms
        # GPU forward time read via the ping-pong CUDA event pair in
        # profile_step(). Note the "prev" in the name: under steady state
        # the value lags its sibling host_step_time on the same record by
        # one loop (the event-pair being read corresponds to the loop
        # before the one host_step_time describes). See the ping-pong
        # comment in PyExecutor._profiler for the design rationale.
        if prev_device_step_time_ms is not None:
            stats_dict["prevDeviceStepTimeMS"] = prev_device_step_time_ms
        # Batch-matched GPU forward time measured from the CUDA events around
        # this record's _forward_step. This is the preferred field for
        # ForwardPassMetrics wall_time.
        if gpu_forward_time_ms is not None:
            stats_dict["gpuForwardTimeMS"] = gpu_forward_time_ms
        # Scheduler mode for this record. "overlap" means iterLatencyMS
        # spans ~2 loops (use hostStepTimeMS for clean per-loop cost);
        # "non_overlap" means iterLatencyMS is itself the clean per-loop
        # CPU wall. Set per-record so consumers do not need server config.
        if scheduler_mode is not None:
            stats_dict["schedulerMode"] = scheduler_mode

        # Convert back to JSON string
        return json.dumps(stats_dict)

    @staticmethod
    def _kv_cache_capacity_serializer(capacity) -> str:
        return json.dumps(capacity)

    # Define a Callable to serialize KV cache events
    @staticmethod
    def _kv_cache_events_serializer(events) -> str:
        from .._utils import KVCacheEventSerializer
        return json.dumps(KVCacheEventSerializer.serialize(events))

    def _pop_result(self, client_id: int):
        self._results.pop(client_id, None)
        self._client_id_to_request_id.pop(client_id, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown()
        return exc_type is None or exc_type == self.WorkerExit

    def __del__(self):
        self.shutdown()


class AwaitResponseHelper:
    ''' Multiple-implementations for await_response for performance. '''

    class HandlerKind(enum.Enum):
        unknown = 0
        single_process_worker = 1
        ipc_batched = 2

    def __init__(self, worker: "BaseWorker"):
        # TODO: make worker weakref
        self.worker = worker
        self.handler_kind: AwaitResponseHelper.HandlerKind = AwaitResponseHelper.HandlerKind.unknown
        self.enable_postprocprocess_parallel = self.worker.enable_postprocess_parallel
        # The error responses when submit request failed will be put here
        self.temp_error_responses = Queue()

    def responses_handler(self, responses: List[tllm.Response]):
        HandlerKind = AwaitResponseHelper.HandlerKind

        if self.handler_kind is HandlerKind.unknown:
            if not (self.worker.result_queue is not None
                    or self.worker.postproc_queues is not None):
                logger_debug(f"creating await_response helper for Worker\n",
                             color="yellow")
                # When ExecutorBindingWorker is used in the main process
                # aka the single process mode
                self.handler_kind = HandlerKind.single_process_worker
            elif self.worker.result_queue is not None or self.worker.postproc_queues is not None:
                # The ExecutorBindingProxy is used
                logger_debug(f"creating await_response helper for IPC\n",
                             color="yellow")
                self.handler_kind = HandlerKind.ipc_batched
            else:
                raise NotImplementedError

        match self.handler_kind:
            case HandlerKind.single_process_worker:
                return self.handle_for_worker(responses)
            case HandlerKind.ipc_batched:
                return self.handle_for_ipc_batched(responses)
            case _:
                raise NotImplementedError

    def __call__(self, timeout: Optional[float] = None) -> bool:
        ''' This method should be called by a ManagedThread. '''
        timeout = timeout or 0.1
        try:
            responses = self.worker.engine.await_responses(
                timeout=datetime.timedelta(seconds=timeout))
        except Exception as e:
            # Defensive: with id=None, PyExecutor.await_responses routes
            # to _await_any_response, which does not raise on event-loop
            # crash — it returns [] silently and we detect the crash
            # via engine._event_loop_error after this block. But any
            # unexpected exception out of await_responses (e.g. from a
            # different engine implementation, or a future change to
            # _await_any_response) is also a clear signal to broadcast
            # and stop the thread.
            return self._broadcast_event_loop_error(e)
        # filter since The _engine_response_callback may return None
        responses = list(
            filter(
                lambda _: _,
                [self.worker._engine_response_callback(r) for r in responses]))

        # append the error responses to the temp_error_responses
        while not self.temp_error_responses.empty():
            responses.append(self.temp_error_responses.get())

        with nvtx_range_debug(f"await_response-{len(responses)}",
                              color="red",
                              category="Worker"):
            self.responses_handler(responses)

        # Even when await_responses returned normally (e.g. via
        # _await_any_response, whose predicate already includes
        # is_shutdown but does not raise), an event-loop crash leaves
        # _event_loop_error stashed on the engine. Broadcast and stop the
        # thread in that case too — see nvbug 6038228.
        error = getattr(self.worker.engine, "_event_loop_error", None)
        if error is not None:
            return self._broadcast_event_loop_error(error)
        return True

    def _broadcast_event_loop_error(self, error: BaseException) -> bool:
        """Wake every pending ``GenerationResult`` after an event-loop crash.

        Inject an ``ErrorResponse`` into every pending ``GenerationResult``
        queue so callers parked in ``queue.get()`` / ``aqueue.get()``
        (``LLM.generate``, ``generate_async`` + ``aresult``,
        ``trtllm-bench``) wake up with a meaningful error instead of
        hanging when the PyExecutor event loop dies.

        Returns ``False`` so the calling ``ManagedThread`` exits — there
        is no point polling a dead engine.

        Scope: single-process worker (the path that backs ``LLM.generate``
        and the bench async client). The IPC / proxy path tracks pending
        results on a different side of the boundary and would need a
        separate poison-pill on ``self.worker.result_queue``; that is left
        as a follow-up consistent with the PyExecutor-side fix.
        """
        error_msg = f"Event loop terminated with error: {error}"
        pending_client_ids = list(self.worker._results.keys())
        if not pending_client_ids:
            logger.error(
                f"Event-loop error with no pending results to wake: {error}")
            return False

        logger.error(
            f"Broadcasting event-loop error to {len(pending_client_ids)} "
            f"pending request(s): {error}")

        event_loop = None
        async_queues: List[_SyncQueue] = []
        for client_id in pending_client_ids:
            try:
                queue = self.worker.return_queue(client_id)
            except KeyError:
                continue
            err_resp = ErrorResponse(
                client_id=client_id,
                error_msg=error_msg,
                request_id=client_id,
            )
            try:
                if isinstance(queue, _SyncQueue):
                    queue.put_nowait(err_resp)
                    async_queues.append(queue)
                    event_loop = event_loop or queue.loop
                else:
                    queue.put(err_resp)
            except Exception as put_error:
                logger.error(f"Failed to push ErrorResponse for client_id="
                             f"{client_id}: {put_error}")
                continue
            self.worker._pop_result(client_id)

        if async_queues:
            try:
                _SyncQueue.notify_many(event_loop, async_queues)
            except Exception as notify_error:
                logger.error(
                    f"Failed to notify async queues on event-loop error: "
                    f"{notify_error}")

        return False

    def handle_for_worker(self, responses: List[tllm.Response]) -> None:
        ''' Return the responses to asyncio.event_loop. '''
        event_loop = None
        async_queues = []
        for response in responses:
            assert response is not None
            queue = self.worker.return_queue(response.client_id)

            if not response.has_error():
                response = _maybe_wrap_response(self.worker, response,
                                                self.worker._is_pytorch_backend)

            # For AsyncQueue.sync_q, we will batch the events to avoid too many
            # event notifications, thus put without wait here.
            if isinstance(queue, _SyncQueue):
                global_tracer().log_instant("worker-rsp.put")
                queue.put_nowait(response)
                async_queues.append(queue)
                # all the loops are identical
                event_loop = event_loop or queue.loop
            else:
                queue.put(response)

            if response.has_error() or response.result.is_final:
                self.worker._pop_result(response.client_id)

        # Notify the events in bulk for performance.
        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    def handle_for_ipc_batched(self, responses: List[tllm.Response]) -> None:
        ''' Perform the IPC in batch explicitly. '''
        postproc_batches = [
            []
            for _ in range(self.worker.postproc_config.num_postprocess_workers)
        ] if self.enable_postprocprocess_parallel else None
        rsp_batch = [] if not self.enable_postprocprocess_parallel else None

        for response in responses:

            if isinstance(response, ErrorResponse):
                pass  # send ErrorResponse directly
            elif self.worker._has_background_error():
                response = self.worker._create_error_response(response)
            elif response.has_error():
                # Convert to ErrorResponse, because tllm.Response cannot be
                # serialized when it has error.
                response = ErrorResponse(response.client_id, response.error_msg,
                                         response.request_id)
            else:
                response = _maybe_wrap_response(self.worker, response,
                                                self.worker._is_pytorch_backend)

            _send_rsp(self.worker,
                      response,
                      postproc_batches=postproc_batches,
                      rsp_batch=rsp_batch)

        if postproc_batches:
            for wid, batch in enumerate(postproc_batches):
                if batch:
                    self.worker.postproc_queues[wid].put(batch)

        if rsp_batch:
            self.worker.result_queue.put(rsp_batch)


def _get_params_for_first_rsp(
    worker, client_id
) -> Tuple[Optional[SamplingParams], Optional[PostprocParams],
           Optional["DisaggregatedParams"]]:
    res = worker._results.get(client_id, None)
    assert res is not None
    if not res._params_transmitted:
        res._params_transmitted = True
        return res.sampling_params, res.postproc_params, res.disaggregated_params
    return None, None, None


def _compute_pytorch_prompt_logprobs(
        generation_result: GenerationResult,
        response: LlmResponse) -> Optional[LogProbsResult]:
    """Compute prompt logprobs for PyTorch backend (cached when streaming) """
    logprob_params = generation_result._logprob_params  # should be present and non None
    assert logprob_params is not None
    if generation_result._streaming:
        cached = getattr(generation_result, '_cached_prompt_logprobs', None)
        if cached is not None:
            return LogProbsResult(
                prompt=cached, generation=None
            )  # generation logprobs, if requested, is provided directly in response.result.log_probs from the sampler.
    context_logits = response.result.context_logits
    assert context_logits is not None, "context_logits must not be None when prompt_logprobs is requested."
    result = response.result.get_result()
    assert result is not None, "result must not be None when prompt_logprobs is requested."
    # Single element list
    first_generation_token = result.output_token_ids[0][:1]
    assert first_generation_token, "first generation token must not be empty when prompt_logprobs is requested."
    # Pass prompt_token_ids with an offset of 1 for correct mapping to the context logits
    prompt_token_ids = generation_result._generation_request.prompt_token_ids[
        1:] + first_generation_token

    logprobs_result = compute_logprobs(
        logprob_params.prompt_logprobs,
        None,
        context_logits,
        None,
        None,
        prompt_token_ids,
        simple_prompt_logprobs=logprob_params.prompt_logprobs_simple_format,
    )
    if generation_result._streaming:
        generation_result._cached_prompt_logprobs = logprobs_result.prompt

    return logprobs_result


def _get_logprobs(worker,
                  response: Union[tllm.Response, LlmResponse],
                  is_pytorch_backend=False) -> Optional[LogProbsResult]:
    """Compute logprobs from response logits when needed.

    Logprobs provenance varies by backend:
    - PyTorch: Generation logprobs computed in sampler, only prompt logprobs computed here
    - TRT: Both prompt and generation logprobs computed here from logits
    """

    logprobs_result = None
    generation_result = worker._results.get(response.client_id, None)

    if not generation_result:
        return None

    logprob_params = getattr(generation_result, "_logprob_params", None)
    if logprob_params:
        if is_pytorch_backend:
            if logprob_params.prompt_logprobs is None:
                # PyTorch: generation logprobs computed in sampler, no post-processing needed
                return None
            else:
                logprobs_result = _compute_pytorch_prompt_logprobs(
                    generation_result, response)

                if logprob_params.drop_context_logits:
                    response.clear_context_logits()

                return logprobs_result

        # TRT backend: compute both prompt and generation logprobs from logits
        logprobs_result = compute_logprobs(
            logprob_params.prompt_logprobs,
            logprob_params.logprobs,
            response.result.context_logits,
            response.result.generation_logits,
            response.result.output_token_ids[0],
            simple_prompt_logprobs=logprob_params.prompt_logprobs_simple_format,
            simple_logprobs=logprob_params.logprobs_simple_format,
        )

        if logprob_params.drop_context_logits:
            response.clear_context_logits()

        if logprob_params.drop_generation_logits:
            response.clear_generation_logits()

    if response.result.is_final:
        generation_result.clear_logprob_params()

    return logprobs_result


def _send_rsp(
        worker,
        response: Union[tllm.Response, ResponseWrapper, ErrorResponse],
        postproc_batches: Optional[List[List["PostprocWorker.Input"]]] = None,
        rsp_batch: Optional[List[tllm.Response]] = None):
    # if postproc_batches is set, append to batch instead of putting to IpcQueue

    if worker.result_queue is not None:
        if rsp_batch is not None:
            rsp_batch.append(response)
        else:
            worker.result_queue.put(response)
    else:
        sampling_params, postproc_params, disaggregated_params = (
            _get_params_for_first_rsp(worker, response.client_id))
        inp = PostprocWorker.Input(
            response,
            # sampling_params is necessary for creating fake GenerationResult
            # instances in the postproc processes. They are for incremental
            # detokenize. They should be transmitted only once for each
            # Request.
            sampling_params=sampling_params,
            postproc_params=postproc_params,
            disaggregated_params=disaggregated_params,
            streaming=worker._results.get(response.client_id, None)._streaming)

        pid = response.client_id % worker.postproc_config.num_postprocess_workers

        if not postproc_batches:
            # Group the responses into buckets for the postprocessing steps.
            # Bucketing is used instead of random dispatching because the
            # incremental detokenization during postprocessing relies on the
            # prior CompletionOutput of a given request.
            worker.postproc_queues[pid].put(inp)
        else:
            postproc_batches[pid].append(inp)

    # Eliminate the finished GenerationRequest instances timely, which may
    # take considerable memory.
    if is_llm_response(response):
        if response.has_error() or response.result.is_final:
            worker._pop_result(response.client_id)
    elif isinstance(response, ErrorResponse):
        worker._pop_result(response.client_id)
    else:
        raise ValueError(f"Unknown response type: {response}")


def _maybe_wrap_response(
        worker,
        response: tllm.Response,
        is_pytorch_backend=False) -> Union[tllm.Response, ResponseWrapper]:

    logprobs_result = _get_logprobs(worker, response, is_pytorch_backend)
    req_perf_metrics = get_metrics_dict(response)
    if logprobs_result or req_perf_metrics:
        response = ResponseWrapper(response, logprobs_result, req_perf_metrics)
    return response
