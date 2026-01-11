"""Transformation to apply torch.compile, CUDAGraph, and other torch.compile-like optimizations.

This is useful as final optimization step for in-framework deployment of our inference models.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type

import torch.nn as nn

ArgsKwargs = Tuple[List[Any], Dict[str, Any]]
GetArgsKwargsForBatchSize = Callable[[int], ArgsKwargs]


class CompileBackendRegistry:
    _backend_registry: Dict[str, Type["CompilerBackend"]] = {}

    @classmethod
    def register(cls, backend: str) -> Type["CompilerBackend"]:
        def decorator(compiler_cls: Type["CompilerBackend"]):
            assert backend not in cls._backend_registry, f"Backend {backend} already registered."
            cls._backend_registry[backend] = compiler_cls
            return compiler_cls

        return decorator

    @classmethod
    def get(cls, backend: str) -> Type["CompilerBackend"]:
        assert cls.has(backend), f"Backend {backend} not registered."
        return cls._backend_registry[backend]

    @classmethod
    def has(cls, backend: str) -> bool:
        return backend in cls._backend_registry


class CompilerBackend(ABC):
    def __init__(self, model: nn.Module, **compiler_kwargs):
        self.model = model

    @abstractmethod
    def compile(self) -> nn.Module:
        raise NotImplementedError
