import io
# pickle is not secure, but but this whole file is a wrapper to make it
# possible to mitigate the primary risk of code injection via pickle.
import pickle  # nosec B403

# These are the base classes that are generally serialized by the ZeroMQ IPC.
# If a class is needed by ZMQ routinely it should be added here. If
# it is only needed in a single instance the class can be added at runtime
# using register_approved_ipc_class.
BASE_ZMQ_CLASSES = {
    "builtins":
    ["*"],  # each Exception Error class needs to be added explicitly
    "collections": ["*"],
    "datetime": ["*"],
    "pathlib": ["*"],
    "llmapi.run_llm_with_postproc": ["*"],  # only used in tests
    ### starting import of torch models classes. They are used in test_llm_multi_gpu.py.
    "tensorrt_llm.*": ["*"],
    "torch.*": ["*"],
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
        # First check for exact module match
        if module in self.approved_imports:
            if name in self.approved_imports[
                    module] or "*" in self.approved_imports[module]:
                return super().find_class(module, name)

        # Then check for wildcard module patterns
        for approved_module, approved_names in self.approved_imports.items():
            if approved_module.endswith(".*"):
                # Convert pattern to prefix for matching
                module_prefix = approved_module[:-2]  # Remove ".*"
                if module.startswith(module_prefix):
                    if "*" in approved_names or name in approved_names:
                        return super().find_class(module, name)

        # If we get here, the import was not approved
        raise ValueError(f"Import {module} | {name} is not allowed")


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
