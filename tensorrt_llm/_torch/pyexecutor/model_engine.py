from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path

import tensorrt as trt
import torch
from cuda import cudart
from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM

import tensorrt_llm
import tensorrt_llm.bindings

from ...builder import Engine, get_engine_version
from ...logger import logger
from ...mapping import Mapping
from ...runtime.generation import CUASSERT, RuntimeTensor, _Profiler, _Runtime
from ...runtime.model_runner import get_engine_name
from ...runtime.session import _scoped_stream
from .llm_request import *
from .resource_manager import ResourceManager
from .runtime_buffer import RuntimeBuffer, TRTLLMBuffer, TRTTransformerBuffer
from .scheduler import ScheduledRequests

LlmRequest.past_key_values = None
ModelConfig = tensorrt_llm.bindings.ModelConfig
GptJsonConfig = tensorrt_llm.bindings.GptJsonConfig


class ModelEngine(ABC):

    @abstractmethod
    def get_max_num_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, scheduled_requests: ScheduledRequests,
                new_tokens_device: Optional[torch.Tensor],
                resource_manager: ResourceManager):
        raise NotImplementedError

    def warmup(self, resource_manager: ResourceManager) -> None:
        """
        This method is called after the KV cache manager is initialized
        inside the given resource manager. Override to perform any
        warmup actions: instantiating CUDA graphs, running torch.compile, etc.
        """
        return


class DummyModelEngine(ModelEngine):

    def __init__(self):
        super(DummyModelEngine, self).__init__()

    def get_max_num_sequences(self) -> int:
        return 16

    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tokens_device: Optional[torch.Tensor] = None):
        total_requests = len(scheduled_requests.context_requests) + len(
            scheduled_requests.generation_requests)
        return {"logits": torch.randn(total_requests)}


class TorchUnbatchedModelEngine(ModelEngine):

    def __init__(self, model_name_or_path: str):
        super(TorchUnbatchedModelEngine, self).__init__()
        self.model = HFAutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.cuda()

    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tokens_device: Optional[torch.Tensor] = None):
        logits_tensors = []
        with torch.no_grad():
            for request in scheduled_requests.context_requests:
                input_ids = torch.IntTensor(
                    request.get_tokens(0)).cuda().unsqueeze(
                        0)  # shape [1, seq_len]
                outputs = self.model(input_ids, use_cache=True)
                logits = outputs['logits'].detach().squeeze(0)[
                    -1]  # logits from outputs shape [1, seq_len, vocab_size]
                past_key_values = outputs['past_key_values']
                request.past_key_values = past_key_values
                logits_tensors.append(logits)

            for request in scheduled_requests.generation_requests:
                input_ids = torch.IntTensor([
                    request.get_token(0,
                                      request.get_num_tokens(0) - 1)
                ]).cuda().unsqueeze(0)
                outputs = self.model(input_ids,
                                     past_key_values=request.past_key_values,
                                     use_cache=True)
                logits = outputs['logits'].detach().squeeze(0)[
                    -1]  # logits from outputs shape [1, seq_len, vocab_size]
                past_key_values = outputs['past_key_values']
                request.past_key_values = past_key_values
                logits_tensors.append(logits)
        all_logits = torch.stack(logits_tensors)
        return {'logits': all_logits}

    def get_max_num_sequences(self) -> int:
        return 16


class _RuntimeIFB(_Runtime):
    contexts: list[trt.IExecutionContext]
    OPT_PROFILES_SPLIT_POINTS = [0, 64, 128, 256, 512, 1024]

    def __init__(self, engine_buffer, mapping: Mapping):
        self.address = None
        self.device_memory_size = 0
        self.__prepare(mapping, engine_buffer)

    def __create_and_setup_context(self, address, size, profile_idx,
                                   stream) -> trt.IExecutionContext:
        context = self.engine.create_execution_context_without_device_memory()
        assert context is not None, "Failed to create an execution context with the provided device memory!"
        context.set_device_memory(address, size)
        context.set_optimization_profile_async(profile_idx, stream)
        # If nvtx verbosity is DETAILED, change it to LAYER_NAMES_ONLY for inference performance
        if context.nvtx_verbosity == trt.ProfilingVerbosity.DETAILED:
            context.nvtx_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        return context

    def __prepare(self, mapping: Mapping, engine_buffer):
        self.runtime_rank = mapping.rank
        local_rank = self.runtime_rank % mapping.gpus_per_node
        torch.cuda.set_device(local_rank)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.runtime = trt.Runtime(logger.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        assert self.engine is not None

        self.input_tensor_names = set()
        self.output_tensor_names = set()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_tensor_names.add(name)
            else:
                self.input_tensor_names.add(name)

        self.profiler = None
        self.engine_inspector = self.engine.create_engine_inspector()
        # cuda graph ping-pong instances
        self.cuda_graph_instances = [None for _ in range(2)]
        if not self.engine.streamable_weights_size:
            # engine does not have weight streaming enabled
            self.__prepare_execution_contexts()

    def __prepare_execution_contexts(self):
        self.contexts = []
        # The device_memory_size_v2 stores the memory required by the largest profile.
        # When weight streaming is enable, it must be queried after the weight streaming budget set.
        if self.address:
            if self.device_memory_size != self.engine.device_memory_size_v2:
                self.device_memory_size = self.engine.device_memory_size_v2
                CUASSERT(cudart.cudaFree(self.address))
                address = CUASSERT(cudart.cudaMalloc(
                    self.device_memory_size))[0]
                self.address = address
        else:
            self.device_memory_size = self.engine.device_memory_size_v2
            address = CUASSERT(cudart.cudaMalloc(self.device_memory_size))[0]
            self.address = address
        logger.info(
            f'[MemUsageChange] Allocated {self.device_memory_size/(1<<20):.2f} MiB for execution context memory.'
        )

        with _scoped_stream() as stream:
            num_opt_profiles = self.engine.num_optimization_profiles
            for i in range(num_opt_profiles):
                context = self.__create_and_setup_context(
                    self.address, self.device_memory_size, i, stream)
                self.contexts.append(context)

    def get_opt_profile_id(self, num_tokens):
        if self.engine.num_optimization_profiles == 1:
            return 0
        split_points = self.OPT_PROFILES_SPLIT_POINTS
        for i in range(len(split_points) - 1):
            if split_points[i] <= num_tokens < split_points[i + 1]:
                return i
        return len(split_points) - 1

    def _set_profiler(self):
        if self.profiler is not None:
            return
        self.profiler = _Profiler()
        for context in self.contexts:
            context.profiler = self.profiler
            context.enqueue_emits_profile = False

    def _set_weight_streaming(self, gpu_weights_percent):
        assert False, "Cannot support weight steaming now."


class TRTModel(ModelEngine):
    model_config: ModelConfig
    runtime: _RuntimeIFB
    mapping: Mapping
    runtime_buffers: dict[str, RuntimeBuffer]
    inputs_dict: dict
    outputs_dict: dict
    context_index: int

    def __init__(self, model_path: Path, meta_config: dict = {}):
        # load model and model config
        self.load_model(model_path)

        free_mem1, _ = torch.cuda.mem_get_info()
        # runtime buffers
        self.runtime_buffers = {
            'llm_buffer':
            TRTLLMBuffer(self.runtime, self.model_config, self.mapping,
                         meta_config),
            'transformer_buffer':
            TRTTransformerBuffer(self.runtime, self.model_config, self.mapping,
                                 meta_config),
        }
        free_mem2, _ = torch.cuda.mem_get_info()
        logger.info(
            f'[MemUsageChange] Allocated {((free_mem1 - free_mem2) / (1<<30)):.2f} GB GPU memory for runtime buffers.'
        )

        # set device and stream
        self.device = torch.device(
            f'cuda:{self.mapping.rank % self.mapping.gpus_per_node}')
        self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        # expected tensors
        self.input_tensor_names = []
        self.input_preallocated_tensor_names = []
        self.output_tensor_names = []
        self.output_preallocated_tensor_names = []
        for _, buffer in self.runtime_buffers.items():
            self.input_tensor_names.extend(buffer.input_tensor_names)
            self.input_preallocated_tensor_names.extend(
                buffer.input_preallocated_tensor_names)
            self.output_tensor_names.extend(buffer.output_tensor_names)
            self.output_preallocated_tensor_names.extend(
                buffer.output_preallocated_tensor_names)

        # check tensors
        input_tensor_names = self.input_tensor_names + self.input_preallocated_tensor_names
        output_tensor_names = self.output_tensor_names + self.output_preallocated_tensor_names
        expected_tensor_names = input_tensor_names + output_tensor_names
        found_tensor_names = [
            self.runtime.engine.get_tensor_name(i)
            for i in range(self.runtime.engine.num_io_tensors)
        ]
        if set(expected_tensor_names) != set(found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected, to use this GenerationSession, "
                "you need to use PretrainedModel.prepare_inputs to create TRT Network inputs."
            )

    def load_model(self, engine_dir):
        engine_version = get_engine_version(engine_dir)
        gpt_json_config = GptJsonConfig.parse_file(
            Path(engine_dir) / 'config.json')
        self.model_config = gpt_json_config.model_config
        rank = tensorrt_llm.mpi_rank()
        if engine_version is None:
            engine_dir = Path(engine_dir)
            # load mapping config
            self.mapping = Mapping(world_size=gpt_json_config.world_size,
                                   rank=rank,
                                   tp_size=gpt_json_config.tensor_parallelism,
                                   pp_size=gpt_json_config.pipeline_parallelism)
            # get engine buffer
            engine_name = get_engine_name(gpt_json_config.name,
                                          gpt_json_config.precision,
                                          gpt_json_config.tensor_parallelism,
                                          gpt_json_config.pipeline_parallelism,
                                          rank)
            serialize_path = engine_dir / engine_name
            with open(serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            engine = Engine.from_dir(engine_dir, rank)
            # load mapping config
            self.mapping = engine.config.pretrained_config.mapping
            # get engine buffer
            engine_buffer = engine.engine
        # load engine
        self.runtime = _RuntimeIFB(engine_buffer, self.mapping)

    def get_max_num_sequences(self):
        num_batches = self.mapping.pp_size if self.mapping.has_pp() else 1
        return num_batches * self.model_config.max_batch_size

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    def prepare_batch_inputs(self, scheduled_requests: ScheduledRequests,
                             resource_manager: ResourceManager):
        self.inputs_dict = {}
        self.outputs_dict = {}
        # prepare buffers
        for _, buffer in self.runtime_buffers.items():
            buffer.prepare_batch_inputs(scheduled_requests, resource_manager)
            buffer.prepare_batch_outputs(scheduled_requests, resource_manager)
            self.inputs_dict.update(buffer.input_buffers)
            self.outputs_dict.update(buffer.output_buffers)

        # get opt profile id and context
        num_tokens = self.runtime_buffers['llm_buffer'].num_tokens
        opt_profile_id = self.runtime.get_opt_profile_id(num_tokens)
        self.context_index = opt_profile_id
        return self.inputs_dict

    @cuda_stream_guard
    def forward_impl(self, inputs_dict):
        # get input trt tensor
        trt_dict = {}
        for name in self.input_tensor_names:
            trt_dict.update(
                {name: RuntimeTensor.from_torch(name, inputs_dict[name])})
        for name in self.input_preallocated_tensor_names:
            trt_dict.update({
                name:
                RuntimeTensor.from_torch(name,
                                         inputs_dict[name],
                                         override_shape=list(
                                             inputs_dict[name].shape))
            })
        # get output trt tensor
        for name in self.output_tensor_names:
            trt_dict.update(
                {name: RuntimeTensor.from_torch(name, self.outputs_dict[name])})
        for name in self.output_preallocated_tensor_names:
            trt_dict.update({
                name:
                RuntimeTensor.from_torch(name,
                                         inputs_dict[name],
                                         override_shape=list(
                                             self.outputs_dict[name].shape))
            })

        # set tensor
        context = self.runtime.contexts[self.context_index]
        self.runtime._set_tensors(context, trt_dict)

        # execute
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.runtime._run(context, stream)
        if not ok:
            raise RuntimeError(f"Executing TRT engine failed!")
        # TODO: add async support
        torch.cuda.synchronize()
        return self.outputs_dict

    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tokens_device: Optional[torch.Tensor] = None):
        batch_inputs = self.prepare_batch_inputs(scheduled_requests,
                                                 resource_manager)
        batch_outputs = self.forward_impl(batch_inputs)
        return batch_outputs
