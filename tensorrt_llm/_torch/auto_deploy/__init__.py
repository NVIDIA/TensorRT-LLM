# import submodules that require registration process
from . import compile, custom_ops, export, models, shim  # noqa: F401

# import AutoDeploy LLM and LlmArgs
from .llm import *
from .llm_args import *

try:
    # This will overwrite the AutoModelForCausalLM.from_config to support modelopt quantization
    import modelopt
except ImportError:
    pass
