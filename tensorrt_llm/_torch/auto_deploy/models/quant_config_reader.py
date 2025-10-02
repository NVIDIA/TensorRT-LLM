"""
Quantization Config Reader Registry.

This module defines a registry system for parsing quantization configurations
from various sources (e.g., 'modelopt'). It enables extensible support for different
quantization producers by delegating parsing logic to dedicated subclasses.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type


class QuantConfigReader(ABC):
    """Base class for reading and parsing quantization config."""

    def __init__(self):
        self._quant_config: Optional[Dict] = {}

    def get_config(self) -> Dict:
        """Return the parsed quantization config."""
        return self._quant_config

    @abstractmethod
    def read_config(self, config: Dict) -> Dict:
        """
        Parse and normalize a quantization config dictionary.

        Args:
            config: The raw parsed JSON object.

        Returns:
            A dictionary of extra model kwargs derived from the quantization config.
            Implementations must also populate self._quant_config with the normalized
            quantization config.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_path: str) -> Optional[Tuple["QuantConfigReader", Dict[str, Any]]]:
        """
        Load and parse a quantization config file from disk.

        This method is implemented by each reader to handle loading and parsing logic.

        Args:
            file_path: Path to the quant config JSON file.

        Returns:
            A (reader, extra_model_kwargs) tuple, or None if the file doesn't exist.
        """
        pass

    # optional hook; default is no-op
    def post_process_model(self, model, model_config):
        """Optional hook to mutate model after construction (e.g., dtype moves)."""
        return model


class QuantConfigReaderRegistry:
    _registry: Dict[str, Type[QuantConfigReader]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[QuantConfigReader]], Type[QuantConfigReader]]:
        def inner(reader_cls: Type[QuantConfigReader]) -> Type[QuantConfigReader]:
            cls._registry[name] = reader_cls
            return reader_cls

        return inner

    @classmethod
    def get(cls, name: str) -> Type[QuantConfigReader]:
        if name not in cls._registry:
            raise ValueError(f"QuantConfigReader for '{name}' not registered.")
        return cls._registry[name]

    @classmethod
    def has(cls, reader_cls: str) -> bool:
        return reader_cls in cls._registry


@QuantConfigReaderRegistry.register("modelopt")
class ModelOPTQuantConfigReader(QuantConfigReader):
    _ALWAYS_EXCLUDE = ("lm_head", "model.embed_tokens")

    def read_config(self, config: Dict) -> Dict:
        producer = config.get("producer", {}).get("name")
        # sanity check
        if producer != "modelopt":
            raise ValueError(f"Expected producer 'modelopt', got '{producer}'")

        quant_config = config.get("quantization", {})
        # Inject default exclusion, add "model.embed_tokens" for "tie_word_embedding:true" case
        excludes = quant_config.get("exclude_modules", [])
        quant_config["exclude_modules"] = excludes + [
            n for n in self._ALWAYS_EXCLUDE if n not in excludes
        ]
        # Update dtype
        if quant_config.get("quant_algo") == "NVFP4":
            quant_config["torch_dtype"] = "float16"

        # Handle kv cache
        kv_algo = quant_config.get("kv_cache_quant_algo")
        if kv_algo:
            if kv_algo != "FP8":
                raise ValueError(f"KV cache quantization format {kv_algo} not supported.")
            quant_config["kv_cache_dtype"] = "float8_e4m3fn"

        self._quant_config = quant_config

        extra_model_kwargs: Dict[str, Any] = {}
        if quant_config.get("quant_algo", None) == "NVFP4":
            extra_model_kwargs["torch_dtype"] = "float16"

        return extra_model_kwargs

    @classmethod
    def from_file(
        cls, ckpt_dir: str
    ) -> Optional[Tuple["ModelOPTQuantConfigReader", Dict[str, Any]]]:
        """
        Load and parse a modelopt-style quantization config from a checkpoint directory.

        Args:
            ckpt_dir: Path to the root directory containing the checkpoint.

        Returns:
            An initialized ModelOPTQuantConfigReader instance, or None if the file doesn't exist.
        """
        quant_file = os.path.join(ckpt_dir, "hf_quant_config.json")
        if not os.path.exists(quant_file):
            return None

        with open(quant_file, "r") as f:
            raw = json.load(f)
        reader = cls()
        extra_model_kwargs = reader.read_config(raw)
        return reader, extra_model_kwargs


@QuantConfigReaderRegistry.register("hf")
class HFQuantConfigReader(QuantConfigReader):
    """
    Quantization reader that process transformers.quantizers.HFQuantizer functionality
    """

    def __init__(self):
        super().__init__()
        self._hf_quantizer = None

    def read_config(self, config: Dict) -> Dict:
        # 'config' here is the full HF config.json object
        qconf = config.get("quantization_config")
        if not qconf:
            raise ValueError("HF quantization_config not found.")

        self._quant_config = qconf
        from transformers.quantizers import AutoHfQuantizer

        try:
            self._hf_quantizer = AutoHfQuantizer.from_config(qconf, pre_quantized=True)
        except Exception:
            self._hf_quantizer = None

        return {}

    @classmethod
    def from_file(cls, ckpt_dir: str) -> Optional[Tuple["HFQuantConfigReader", Dict[str, Any]]]:
        config_file = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(config_file):
            return None
        with open(config_file, "r") as f:
            raw = json.load(f)
        qconf = raw.get("quantization_config")
        if not isinstance(qconf, dict):
            return None

        # TODO(Fridah-nv):this class is only verified with GPT-OSS MXFP4, other hf quantizers
        # should have similar workflow and will be added to the pipeline
        quant_method = str(qconf.get("quant_method", "")).lower()
        if quant_method != "mxfp4":
            return None

        reader = cls()
        extra_model_kwargs = reader.read_config(raw)
        return reader, extra_model_kwargs

    # Apply dtype using HF quantizer, this is needed for GPT-OSS MXFP4 integration into AD pipeline,
    # more features to be added
    def post_process_model(self, model, model_config):
        if self._hf_quantizer is None:
            return
        dtype = getattr(model_config, "dtype", None)
        new_dtype = self._hf_quantizer.update_dtype(dtype)
        if new_dtype is None:
            return
        model.to(new_dtype)
        return model


def autodetect_quant_config_reader(
    fetched_dir: str,
) -> Optional[Tuple["QuantConfigReader", Dict[str, Any]]]:
    """Try ModelOPT first; if not found, fall back to HF. Returns (reader, extra_kwargs) or None."""
    reader_cls = QuantConfigReaderRegistry.get("modelopt")
    result = reader_cls.from_file(fetched_dir)
    # Fallback to HF reader if ModelOPT not present
    if result is None:
        hf_cls = QuantConfigReaderRegistry.get("hf")
        try:
            result = hf_cls.from_file(fetched_dir)
        except Exception:
            # Skip HF reader if it errors out during probing
            result = None
    return result
