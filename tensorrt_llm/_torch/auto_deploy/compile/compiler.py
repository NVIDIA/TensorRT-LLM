"""Transformation to apply torch.compile, CUDAGraph, and other torch.compile-like optimizations.

This is useful as final optimization step for in-framework deployment of our inference models.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx._pytree import tree_flatten_spec
from torch.utils._pytree import PyTree

from ..utils.logger import ad_logger


def _flatten_args(in_spec, *args, **kwargs) -> Tuple[torch.Tensor, List[Any]]:
    """Flatten inputs from in_spec where we assume the first input is the main input tensor."""
    all_args: PyTree = (args, kwargs)
    input_t, *flat_args = tree_flatten_spec(all_args, in_spec)
    assert input_t.ndim > 1, "Expecting at least a 2D input tensor."
    return input_t, flat_args


class BackendRegistry:
    _backend_registry: Dict[str, Type["BackendCompiler"]] = {}

    @classmethod
    def register(cls, backend: str) -> Type["BackendCompiler"]:
        def decorator(compiler_cls: Type["BackendCompiler"]):
            assert backend not in cls._backend_registry, f"Backend {backend} already registered."
            cls._backend_registry[backend] = compiler_cls
            return compiler_cls

        return decorator

    @classmethod
    def get(cls, backend: str) -> Type["BackendCompiler"]:
        assert cls.has(backend), f"Backend {backend} not registered."
        return cls._backend_registry[backend]

    @classmethod
    def has(cls, backend: str) -> bool:
        return backend in cls._backend_registry


class BackendCompiler(ABC):
    max_batch_size: int

    def __init__(
        self,
        gm: GraphModule,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes=None,
    ):
        self.gm = gm
        self.args = args
        self.kwargs = kwargs or {}
        self.dynamic_shapes = dynamic_shapes

        # identify max_batch_size
        if self.dynamic_shapes is not None and 0 in self.dynamic_shapes[0]:
            self.max_batch_size = self.dynamic_shapes[0][0].max
        else:
            idxs, *_ = _flatten_args(self.gm._in_spec, *self.args, **self.kwargs)
            self.max_batch_size = idxs.shape[0]

    @abstractmethod
    def compile(self) -> nn.Module:
        raise NotImplementedError


def compile_and_capture(
    gm: GraphModule,
    backend: str,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes=None,
) -> nn.Module:
    """Compile or capture graph for single-token generation."""
    elapsed_time = -time.time()
    ad_logger.info("Fusion before compiling...")

    ad_logger.info(f"Compiling for {backend} backend...")

    compiler_cls = BackendRegistry.get(backend)
    compiled_module = compiler_cls(gm, args, kwargs, dynamic_shapes).compile()

    elapsed_time += time.time()
    ad_logger.info(f"Compile time with backend {backend}: {elapsed_time:.6f} seconds")

    return compiled_module
