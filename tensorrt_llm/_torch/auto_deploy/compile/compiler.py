# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
