import io
# pickle is not secure, but but this whole file is a wrapper to make it
# possible to mitigate the primary risk of code injection via pickle.
import pickle  # nosec B403
from functools import partial

# This is an example class (white list) to showcase how to guard serialization with approved classes.
# If a class is needed routinely it should be added into the whitelist. If it is only needed in a single instance
# the class can be added at runtime using register_approved_class.
BASE_EXAMPLE_CLASSES = {
    "builtins": [
        "Exception", "ValueError", "NotImplementedError", "AttributeError",
        "AssertionError", "RuntimeError"
    ],  # each Exception Error class needs to be added explicitly
    "collections": ["OrderedDict"],
    "datetime": ["timedelta"],
    "pathlib": ["PosixPath"],
    "llmapi.run_llm_with_postproc": ["perform_faked_oai_postprocess"
                                     ],  # only used in tests
    ### starting import of torch models classes. They are used in test_llm_multi_gpu.py.
    "tensorrt_llm._torch.model_config": ["MoeLoadBalancerConfig"],
    "tensorrt_llm._torch.models.modeling_bert":
    ["BertForSequenceClassification"],
    "tensorrt_llm._torch.models.modeling_clip": ["CLIPVisionModel"],
    "tensorrt_llm._torch.models.modeling_deepseekv3": ["DeepseekV3ForCausalLM"],
    "tensorrt_llm._torch.models.modeling_gemma3": ["Gemma3ForCausalLM"],
    "tensorrt_llm._torch.models.modeling_hyperclovax": ["HCXVisionForCausalLM"],
    "tensorrt_llm._torch.models.modeling_llama": [
        "Eagle3LlamaForCausalLM",
        "LlamaForCausalLM",
        "Llama4ForCausalLM",
        "Llama4ForConditionalGeneration",
    ],
    "tensorrt_llm._torch.models.modeling_llava_next": ["LlavaNextModel"],
    "tensorrt_llm._torch.models.modeling_mistral": ["MistralForCausalLM"],
    "tensorrt_llm._torch.models.modeling_mixtral": ["MixtralForCausalLM"],
    "tensorrt_llm._torch.models.modeling_mllama":
    ["MllamaForConditionalGeneration"],
    "tensorrt_llm._torch.models.modeling_nemotron": ["NemotronForCausalLM"],
    "tensorrt_llm._torch.models.modeling_nemotron_h": ["NemotronHForCausalLM"],
    "tensorrt_llm._torch.models.modeling_nemotron_nas":
    ["NemotronNASForCausalLM"],
    "tensorrt_llm._torch.models.modeling_qwen":
    ["Qwen2ForCausalLM", "Qwen2ForProcessRewardModel", "Qwen2ForRewardModel"],
    "tensorrt_llm._torch.models.modeling_qwen2vl":
    ["Qwen2VLModel", "Qwen2_5_VLModel"],
    "tensorrt_llm._torch.models.modeling_qwen3": ["Qwen3ForCausalLM"],
    "tensorrt_llm._torch.models.modeling_qwen3_moe": ["Qwen3MoeForCausalLM"],
    "tensorrt_llm._torch.models.modeling_qwen_moe": ["Qwen2MoeForCausalLM"],
    "tensorrt_llm._torch.models.modeling_siglip": ["SiglipVisionModel"],
    "tensorrt_llm._torch.models.modeling_vila": ["VilaModel"],
    "tensorrt_llm._torch.models.modeling_gpt_oss": ["GptOssForCausalLM"],
    "tensorrt_llm._torch.pyexecutor.config": ["PyTorchConfig", "LoadFormat"],
    "tensorrt_llm._torch.pyexecutor.llm_request":
    ["LogitsStorage", "PyResult", "LlmResult", "LlmResponse", "LogProbStorage"],
    "tensorrt_llm._torch.speculative.mtp": ["MTPConfig"],
    "tensorrt_llm._torch.speculative.interface": ["SpeculativeDecodingMode"],
    ### ending import of torch models classes
    "tensorrt_llm.bindings.executor": [
        "BatchingType", "CacheTransceiverConfig", "CapacitySchedulerPolicy",
        "ContextPhaseParams", "ContextChunkingPolicy", "DynamicBatchConfig",
        "ExecutorConfig", "ExtendedRuntimePerfKnobConfig", "Response", "Result",
        "FinishReason", "KvCacheConfig", "KvCacheTransferMode",
        "KvCacheRetentionConfig",
        "KvCacheRetentionConfig.TokenRangeRetentionConfig", "PeftCacheConfig",
        "SchedulerConfig"
    ],
    "tensorrt_llm.builder": ["BuildConfig"],
    "tensorrt_llm.disaggregated_params": ["DisaggregatedParams"],
    "tensorrt_llm.inputs.multimodal": ["MultimodalInput"],
    "tensorrt_llm.executor.postproc_worker": [
        "PostprocArgs", "PostprocParams", "PostprocWorkerConfig",
        "PostprocWorker.Input", "PostprocWorker.Output"
    ],
    "tensorrt_llm.executor.request": [
        "CancellingRequest", "GenerationRequest", "LoRARequest",
        "PromptAdapterRequest"
    ],
    "tensorrt_llm.executor.result": [
        "CompletionOutput", "DetokenizedGenerationResultBase",
        "GenerationResult", "GenerationResultBase", "IterationResult",
        "Logprob", "LogProbsResult", "ResponseWrapper"
    ],
    "tensorrt_llm.executor.utils": ["ErrorResponse", "WorkerCommIpcAddrs"],
    "tensorrt_llm.executor.worker": ["GenerationExecutorWorker", "worker_main"],
    "tensorrt_llm.llmapi.llm_args": [
        "_ModelFormatKind", "_ParallelConfig", "CalibConfig",
        "CapacitySchedulerPolicy", "KvCacheConfig", "LookaheadDecodingConfig",
        "TrtLlmArgs", "SchedulerConfig", "LoadFormat", "DynamicBatchConfig"
    ],
    "tensorrt_llm.llmapi.mpi_session": ["RemoteTask"],
    "tensorrt_llm.llmapi.llm_utils":
    ["CachedModelLoader._node_build_task", "LlmBuildStats"],
    "tensorrt_llm.llmapi.tokenizer": ["TransformersTokenizer"],
    "tensorrt_llm.lora_manager": ["LoraConfig"],
    "tensorrt_llm.mapping": ["Mapping"],
    "tensorrt_llm.models.modeling_utils":
    ["QuantConfig", "SpeculativeDecodingMode"],
    "tensorrt_llm.plugin.plugin": ["PluginConfig"],
    "tensorrt_llm.sampling_params":
    ["SamplingParams", "GuidedDecodingParams", "GreedyDecodingParams"],
    "tensorrt_llm.serve.postprocess_handlers": [
        "chat_response_post_processor", "chat_stream_post_processor",
        "completion_stream_post_processor",
        "completion_response_post_processor", "CompletionPostprocArgs",
        "ChatPostprocArgs"
    ],
    "torch._utils": ["_rebuild_tensor_v2"],
    "torch.storage": ["_load_from_bytes"],
}


def _register_class(dict, obj):
    name = getattr(obj, '__qualname__', None)
    if name is None:
        name = obj.__name__
    module = pickle.whichmodule(obj, name)
    if module not in dict.keys():
        dict[module] = []
    dict[module].append(name)


def register_approved_class(obj):
    _register_class(BASE_EXAMPLE_CLASSES, obj)


class Unpickler(pickle.Unpickler):

    def __init__(self, *args, approved_imports={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.approved_imports = approved_imports

    # only import approved classes, this is the security boundary.
    def find_class(self, module, name):
        if name not in self.approved_imports.get(module, []):
            # If this is triggered when it shouldn't be, then the module
            # and class should be added to the approved_imports. If the class
            # is being used as part of a routine scenario, then it should be added
            # to the appropriate base classes above.
            raise ValueError(f"Import {module} | {name} is not allowed")
        return super().find_class(module, name)


# these are taken from the pickle module to allow for this to be a drop in replacement
# source: https://github.com/python/cpython/blob/3.13/Lib/pickle.py
# dump and dumps are just aliases because the serucity controls are on the deserialization
# side. However they are included here so that in the future if a more secure serialization
# soliton is identified, it can be added with less impact to the rest of the application.
dump = partial(pickle.dump, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301
dumps = partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)  # nosec B301


def load(file,
         *,
         fix_imports=True,
         encoding="ASCII",
         errors="strict",
         buffers=None,
         approved_imports={}):
    return Unpickler(file,
                     fix_imports=fix_imports,
                     buffers=buffers,
                     encoding=encoding,
                     errors=errors,
                     approved_imports=approved_imports).load()


def loads(s,
          /,
          *,
          fix_imports=True,
          encoding="ASCII",
          errors="strict",
          buffers=None,
          approved_imports={}):
    if isinstance(s, str):
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(s)
    return Unpickler(file,
                     fix_imports=fix_imports,
                     buffers=buffers,
                     encoding=encoding,
                     errors=errors,
                     approved_imports=approved_imports).load()
