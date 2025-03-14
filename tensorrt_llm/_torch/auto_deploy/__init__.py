# import submodules that require registration process
from . import compile, custom_ops, models, shim  # noqa: F401

try:
    # This will overwrite the AutoModelForCausalLM.from_config to support modelopt quantization
    import modelopt
except ImportError:
    pass
