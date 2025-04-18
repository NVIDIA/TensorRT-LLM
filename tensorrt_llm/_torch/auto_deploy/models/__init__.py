from . import hf
from .deepseek import *
from .factory import *

try:
    from .llama4 import *
except ImportError:
    from ..utils.logger import ad_logger

    ad_logger.warning(
        "Failed to import Llama-4 models. Please install `transformers[hf_xet]>=4.51.0`."
    )
