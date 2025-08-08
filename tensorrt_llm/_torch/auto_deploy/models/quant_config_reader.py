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

import torch


class QuantConfigReader(ABC):
    """Base class for reading and parsing quantization config."""

    def __init__(self):
        self._quant_config: Optional[Dict] = None

    def get_config(self) -> Dict:
        """Return the parsed quantization config."""
        return self._quant_config or {}

    @abstractmethod
    def read_config(self, config: Dict) -> Dict:
        """
        Parse and normalize a quantization config dictionary.

        Args:
            config: The raw "quantization" field from the JSON file.

        Returns:
            A processed and normalized config dictionary.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_path: str) -> Optional["QuantConfigReader"]:
        """
        Load and parse a quantization config file from disk.

        This method is implemented by each reader to handle loading and parsing logic.

        Args:
            file_path: Path to the quant config JSON file.

        Returns:
            An initialized QuantConfigReader instance, or None if the file doesn't exist.
        """
        pass


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
    def read_config(self, config: Dict) -> Dict:
        # Inject default exclusion
        config.setdefault("exclude_modules", ["lm_head"])

        # Update dtype
        if config.get("quant_algo") == "NVFP4":
            config["torch_dtype"] = "float16"

        # Handle kv cache
        kv_algo = config.get("kv_cache_quant_algo")
        if kv_algo:
            if kv_algo != "FP8":
                raise ValueError(f"KV cache quantization format {kv_algo} not supported.")
            config["kv_cache_dtype"] = "float8_e4m3fn"

        self._quant_config = config
        return self._quant_config

    @classmethod
    def from_file(
        cls, ckpt_dir: str
    ) -> Optional[Tuple["ModelOPTQuantConfigReader", Optional[torch.dtype]]]:
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

        producer = raw.get("producer", {}).get("name")
        # sanity check
        if producer != "modelopt":
            raise ValueError(f"Expected producer 'modelopt', got '{producer}'")

        quant_config = raw.get("quantization", {})
        reader = cls()
        reader.read_config(quant_config)
        extra_model_kwargs: Dict[str, Any] = {}
        if quant_config and quant_config.get("quant_algo", None) == "NVFP4":
            extra_model_kwargs["torch_dtype"] = "float16"
        return reader, extra_model_kwargs
