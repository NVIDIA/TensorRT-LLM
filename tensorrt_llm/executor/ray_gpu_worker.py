import copy
import importlib
import json
import os
import socket
from pathlib import Path
from queue import Queue
from typing import Any, Optional, Type, Union

import ray
import torch

from tensorrt_llm.logger import logger

from .._torch.virtual_memory import materialize_with_tag, release_with_tag
from .._utils import mpi_rank
from ..bindings import executor as tllm
from ..builder import ConfigEncoder, Engine, EngineConfig
from ..llmapi.llm_args import PybindMirror
from ..llmapi.utils import print_colored_debug
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


def resolve_obj_by_qualname(qualname: str) -> Any:
    """Resolve an object by its fully qualified name."""
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


@ray.remote
class RayWorkerWrapper:

    def __init__(self,
                 worker_cls,
                 worker_kwargs,
                 world_size,
                 rank,
                 worker_extension_cls: Optional[str] = None):
        self.master_address = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]

        # expect to see global counts w/ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1,
        # unless CUDA_VISIBLE_DEVICES is set to a subset of the global devices
        print("Cuda devices: ", torch.cuda.device_count())

        torch.cuda.is_available()
        assert len(ray.get_gpu_ids()) == 1
        # Physical gpu id. Ray might return str and this would cause issues in cuda.set_device() w/o int
        self.gpu = int(ray.get_gpu_ids()[0])
        local_gpu = self.physical_to_local_id(self.gpu)

        # for debug
        ip = ray.util.get_node_ip_address()
        hostname = socket.gethostname()
        print(
            f"[Worker rank={rank}] Running on node: {hostname} (IP: {ip}), GPU ID Physical: {self.gpu}, GPU ID Local: {local_gpu} master_addr: {self.master_address} port: {self.master_port}",
            flush=True)

        torch.distributed.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=world_size,
            rank=rank)

        print(
            f"[FINISHED PG INIT rank={rank}] Running on node: {hostname} (IP: {ip}), GPU ID: {self.gpu}",
            flush=True)

        torch.cuda.set_device(local_gpu)
        device = torch.cuda.get_device_properties(local_gpu)
        print(
            f"pid: {os.getpid()}, device {self.gpu}: {device.name}, UUID: {device.uuid}, Memory: {device.total_memory / 1024**2:.0f}MB"
        )

        worker_cls = self._inject_worker_extension(worker_cls,
                                                   worker_extension_cls)
        self.worker = worker_cls(**worker_kwargs)

    def submit(self, request: GenerationRequest) -> GenerationResult:
        return self.worker.submit(request)

    def enqueue_request(self,
                        request: GenerationRequest,
                        result_wait_queue: Queue | None = None) -> int:
        return self.worker.enqueue_request(request, result_wait_queue)

    def abort_request(self, request_id: int) -> None:
        self.worker.abort_request(request_id)

    def update_weights(self, weights: dict):
        return self.worker.update_weights(weights)

    def update_weights_from_ipc_handles(self, ipc_handles: dict):
        return self.worker.update_weights_from_ipc_handles(ipc_handles)

    def report_device_id(self) -> str:
        from tensorrt_llm._torch.utils import get_device_uuid
        local_id = self.physical_to_local_id(self.gpu)
        return get_device_uuid(local_id)

    def reset_prefix_cache(self):
        self.worker.reset_prefix_cache()

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

    def _inject_worker_extension(
            self, worker_class: Type['GenerationExecutor'],
            extension_cls_name: Optional[str]) -> Type['GenerationExecutor']:
        """Inject worker extension into the worker class if specified."""
        if not extension_cls_name:
            return worker_class

        try:
            extension_cls = resolve_obj_by_qualname(extension_cls_name)
            # Check for conflicts
            for attr in dir(extension_cls):
                if attr.startswith("__"):
                    continue
                if hasattr(worker_class, attr):
                    # TODO: change to serializable loggings
                    print(
                        f"Worker class {worker_class.__name__} already has attribute '{attr}', "
                        f"which conflicts with extension {extension_cls.__name__}. "
                        f"Extension method will override.")

            if extension_cls not in worker_class.__bases__:
                worker_class.__bases__ = worker_class.__bases__ + (
                    extension_cls, )
                print(
                    f"Finished injection of {extension_cls.__name__} into {worker_class.__name__}."
                )

            return worker_class

        except Exception as e:
            raise RuntimeError(
                f"Failed to load worker extension '{extension_cls_name}': {e}")


class RayGPUWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        lora_config: Optional[LoraConfig] = None,
    ) -> None:
        postproc_config = postproc_worker_config or PostprocWorkerConfig()
        super().__init__(
            num_postprocess_workers=postproc_config.num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_config.postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self.engine = None
        self.rank = torch.distributed.get_rank()
        self.global_rank = torch.distributed.get_world_size()

        self._executor_config = executor_config
        self._is_pytorch_backend = getattr(self._executor_config, "backend",
                                           None) == "pytorch"
        assert self._is_pytorch_backend

        if isinstance(engine, list):
            engine = engine[self.rank]

        if executor_config is None:
            executor_config = tllm.ExecutorConfig(1)

        executor_config.logits_post_processor_config = tllm.LogitsPostProcessorConfig(
            processor_batched=batched_logits_processor, replicate=False)

        def _create_engine():
            if isinstance(engine, Engine):
                return tllm.Executor(engine.engine,
                                     json.dumps(engine.config.to_dict(),
                                                cls=ConfigEncoder),
                                     tllm.ModelType.DECODER_ONLY,
                                     executor_config=executor_config,
                                     managed_weights=engine.managed_weights)

            if not hasattr(executor_config, "backend"):
                return tllm.Executor(engine, tllm.ModelType.DECODER_ONLY,
                                     executor_config)
            args = {
                "executor_config": executor_config,
                "checkpoint_dir": executor_config.hf_model_dir,
            }
            if executor_config.backend == "pytorch":
                from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
                    create_py_executor
                create_executor = create_py_executor
                args["lora_config"] = lora_config
            else:
                raise ValueError(
                    f"Unsupported backend config: {executor_config.backend}")

            return create_executor(**args)

        self.engine = _create_engine()

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
                self._lora_manager = LoraManager()
            if engine_config.build_config.max_prompt_embedding_table_size > 0:
                self._prompt_adapter_manager = PromptAdapterManager()

        if getattr(executor_config, "backend",
                   "") == "pytorch" and lora_config is not None:
            self._lora_manager = LoraManager()
            lora_model_config = self.engine.model_engine.lora_model_config
            assert lora_model_config is not None
            self._lora_model_config = lora_model_config

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

    def _load_lora_adapter(self, lora_request: LoRARequest):
        self._lora_manager.load_from_ckpt(
            [lora_request.path],
            model_config=self._runtime_model_config if
            self._runtime_model_config is not None else self._lora_model_config,
            runtime_mapping=None,
            uids=[str(lora_request.adapter_id)])

    def _load_prompt_adapter(self,
                             prompt_adapter_request: PromptAdapterRequest):
        self._prompt_adapter_manager.load_from_ckpt(
            [prompt_adapter_request.local_path],
            model_config=self._runtime_model_config,
            uids=[str(prompt_adapter_request.adapter_id)])

    def enqueue_request(self,
                        request: GenerationRequest,
                        result_wait_queue: Queue | None = None) -> int:
        assert request.id is not None

        if self._lora_manager is not None and request.lora_request is not None:
            self._load_lora_adapter(request.lora_request)
            uid = str(request.lora_request.adapter_id)
            lora_config = tllm.LoraConfig(
                task_id=request.lora_request.adapter_id,
                weights=self._lora_manager.cpp_lora_weights[uid],
                config=self._lora_manager.cpp_lora_config[uid])
        else:
            lora_config = None

        prompt_token_ids = copy.deepcopy(request.prompt_token_ids)
        prompt_tuning_config = None
        multimodal_embedding = None
        mrope_config = None
        # TODO: Request class has refactored for v1.0, need update.
        # if request.multimodal_embedding is not None:
        #     multimodal_embedding = request.multimodal_embedding
        if request.prompt_adapter_request is not None:
            self._load_prompt_adapter(request.prompt_adapter_request)
            uid = str(request.prompt_adapter_request.adapter_id)
            prompt_tuning_config = tllm.PromptTuningConfig(
                self._prompt_adapter_manager.uid_to_weights[uid])
            vocab_size = self._runtime_model_config.vocab_size
            pa_length = prompt_tuning_config.embedding_table.size(0)
            prompt_token_ids = list(range(
                vocab_size, vocab_size + pa_length)) + prompt_token_ids

        # TODO: Request class has refactored for v1.0, need update.
        # if request.mrope_config is not None:
        #     mrope_config = tllm.MropeConfig(**request.mrope_config)

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

        is_overlap_enabled = self._is_pytorch_backend and not self._executor_config.pytorch_backend_config.disable_overlap_scheduler
        if is_overlap_enabled:
            is_disaggregated = self.engine.kv_cache_transceiver is not None
            if is_disaggregated and (
                    request_type == tllm.RequestType.REQUEST_TYPE_CONTEXT_ONLY):
                raise ValueError(
                    "Context only requests are not supported in pytorch backend when overlap is enabled."
                )

        assert request.id is not None

        def _deduce_max_tokens(request: GenerationRequest,
                               executor_config: tllm.ExecutorConfig) -> int:
            if request.sampling_params.max_tokens:
                return request.sampling_params.max_tokens
            # deduce max_tokens when it's not set by user
            query_token_len = len(
                request.query_token_ids) if request.query_token_ids else 0
            cp_size = 1 if (not hasattr(executor_config, "mapping")
                            or executor_config.mapping.cp_size
                            is None) else executor_config.mapping.cp_size
            if not hasattr(executor_config, "max_seq_len"):
                raise RuntimeError(
                    "max_tokens for sampling is not set and cannot be deduced")
            splited_prompt_len = int(len(prompt_token_ids) / cp_size)
            default_max_tokens = executor_config.max_seq_len - splited_prompt_len - query_token_len
            if default_max_tokens < 0:
                raise ValueError(
                    f"Deduced max_tokens {default_max_tokens} is less than 0, because"
                    f"prompt length {splited_prompt_len} plus query length {query_token_len} "
                    f"is larger than max_seq_len {executor_config.max_seq_len}")
            return default_max_tokens

        try:
            executor_request = tllm.Request(
                client_id=request.id,
                input_token_ids=prompt_token_ids,
                max_tokens=_deduce_max_tokens(request, self._executor_config),
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
                multimodal_embedding=multimodal_embedding,
                mrope_config=mrope_config,
                logits_post_processor_name=(
                    tllm.Request.BATCHED_POST_PROCESSOR_NAME
                    if request.sampling_params.apply_batched_logits_processor
                    else None),
                logits_post_processor=None if self._is_pytorch_backend else
                request.sampling_params.logits_processor,
                kv_cache_retention_config=request.kv_cache_retention_config,
                context_phase_params=context_phase_params,
                type=request_type)

            if self._is_pytorch_backend and request.sampling_params.logits_processor:
                # For PyTorch backend, we attach logits processors as a dynamic Python attribute
                # instead of using the C++ binding, since the latter will cause PyCapsule pickling issues.
                lp = request.sampling_params.logits_processor
                executor_request.py_logits_post_processors = lp if isinstance(
                    lp, list) else [lp]

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
            return req_id
        except Exception as e:
            raise RequestError(str(e)) from e

    def update_weights(self, weights: dict):
        try:
            self.engine.update_weights(weights)
        except Exception as e:
            logger.error(
                f"Worker rank {self.rank} failed to update weights: {e}")
            raise

    def update_weights_from_ipc_handles(self, ipc_handles: dict):
        try:
            self.engine.update_weight_from_ipc_handles(ipc_handles)
        except Exception as e:
            logger.error(
                f"Worker rank {self.rank} failed to update weights from ipc handles: {e}"
            )
            return False

    def reset_prefix_cache(self):
        self.engine.reset_prefix_cache()

    @staticmethod
    def sleep(*tags: str):
        torch.cuda.synchronize()
        release_with_tag(*tags)
        torch.cuda.synchronize()

    @staticmethod
    def wakeup(*tags: str):
        torch.cuda.synchronize()
        materialize_with_tag(*tags)
        torch.cuda.synchronize()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError("Ray GPU worker does not support submit")

    def shutdown(self):

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        print_colored_debug(f'Worker {mpi_rank()} shutdown...\n', "yellow")

        if self.engine is not None:
            if self.engine.can_enqueue_requests():

                if self.await_response_thread.is_alive():
                    self.await_response_thread.stop()
                    self.await_response_thread.join()
                if self.dispatch_stats_thread.is_alive():
                    self.dispatch_stats_thread.stop()
                    self.dispatch_stats_thread.join()
                if self.dispatch_kv_cache_events_thread.is_alive():
                    self.dispatch_kv_cache_events_thread.stop()
                    self.dispatch_kv_cache_events_thread.join()

            self.engine.shutdown()
            self.engine = None

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

        print_colored_debug(f"Worker {mpi_rank()} shutdown done.\n", "yellow")

    def block_subordinates(self):
        if self.rank != 0:
            if isinstance(self.engine, tllm.Executor):
                self.shutdown()
                raise self.WorkerExit(
                    "block_subordinates() should be used in a `with ExecutorBindingsWorker() as ...:` block"
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
