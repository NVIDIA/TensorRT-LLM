import io
# pickle is not secure, but but this whole file is a wrapper to make it
# possible to mitigate the primary risk of code injection via pickle.
import pickle  # nosec B403

# These are the base classes that are generally serialized by the ZeroMQ IPC.
# If a class is needed by ZMQ routinely it should be added here. If
# it is only needed in a single instance the class can be added at runtime
# using register_approved_ipc_class.
BASE_ZMQ_CLASSES = {
    "builtins": ["Exception"],
    "tensorrt_llm.executor.request":
    ["CancellingRequest", "GenerationRequest", "LoRARequest"],
    "tensorrt_llm.sampling_params":
    ["SamplingParams", "GuidedDecodingParams", "GreedyDecodingParams"],
    "tensorrt_llm.executor.postproc_worker":
    ["PostprocParams", "PostprocWorker.Input", "PostprocWorker.Output"],
    "tensorrt_llm.serve.postprocess_handlers":
    ["CompletionPostprocArgs", "completion_response_post_processor"],
    "tensorrt_llm.bindings.executor": [
        "Response", "Result", "FinishReason", "KvCacheRetentionConfig",
        "KvCacheRetentionConfig.TokenRangeRetentionConfig"
    ],
    "tensorrt_llm.serve.openai_protocol":
    ["CompletionResponse", "CompletionResponseChoice", "UsageInfo"],
    "torch._utils": ["_rebuild_tensor_v2"],
    "torch.storage": ["_load_from_bytes"],
    "collections": ["OrderedDict"],
    "datetime": ["timedelta"],
    "tensorrt_llm.llmapi.llm_args": ["LookaheadDecodingConfig"],
    "tensorrt_llm._torch.pyexecutor.llm_request":
    ["LogitsStorage", "PyResult", "LlmResult", "LlmResponse", "LogProbStorage"],
    "tensorrt_llm.executor.utils": ["ErrorResponse"],
}


def _register_class(dict, obj):
    name = getattr(obj, '__qualname__', None)
    if name is None:
        name = obj.__name__
    module = pickle.whichmodule(obj, name)
    if module not in BASE_ZMQ_CLASSES.keys():
        BASE_ZMQ_CLASSES[module] = []
    BASE_ZMQ_CLASSES[module].append(name)


def register_approved_ipc_class(obj):
    _register_class(BASE_ZMQ_CLASSES, obj)


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
dump = pickle.dump  # nosec B301
dumps = pickle.dumps  # nosec B301


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
