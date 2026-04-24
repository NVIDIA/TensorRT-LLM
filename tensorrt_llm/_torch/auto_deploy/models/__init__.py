import importlib
import logging

_logger = logging.getLogger(__name__)

from .factory import *  # noqa: E402, F401, F403

# Import model submodules individually so that modules with transitive TRT-LLM
# dependencies (e.g., eagle needing MTPDecodingConfig) don't prevent others
# from loading in standalone mode.
for _name in ("custom", "eagle", "hf", "nemotron_flash", "patches"):
    try:
        importlib.import_module(f".{_name}", __name__)
    except (ImportError, ModuleNotFoundError) as _exc:
        _logger.debug("Skipping models.%s: %s", _name, _exc)
