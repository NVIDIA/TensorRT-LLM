# import submodules that require registration process
from . import compile, custom_ops, export, models  # noqa: F401
from ._compat import TRTLLM_AVAILABLE

if TRTLLM_AVAILABLE:
    from . import shim  # noqa: F401

    # import AutoDeploy LLM and LlmArgs (require TRT-LLM base classes)
    from .llm import *
    from .llm_args import *

try:
    # This will overwrite the AutoModelForCausalLM.from_config to support modelopt quantization
    import modelopt
except ImportError:
    pass
