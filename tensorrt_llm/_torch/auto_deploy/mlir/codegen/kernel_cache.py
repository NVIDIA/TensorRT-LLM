# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Hash-based caching for generated Triton kernels."""

import hashlib
from typing import Callable, Optional


class KernelCache:
    """Cache generated kernels by subgraph structure hash."""

    def __init__(self):
        self._cache: dict[str, Callable] = {}

    def get(self, key: str) -> Optional[Callable]:
        """Return cached kernel for the given key, or None."""
        return self._cache.get(key)

    def put(self, key: str, kernel: Callable) -> None:
        """Store a kernel under the given key."""
        self._cache[key] = kernel

    @staticmethod
    def hash_subgraph(subgraph) -> str:
        """Compute a stable hash from op types + operand connectivity + dtypes."""
        op_index = {id(op): i for i, op in enumerate(subgraph.ops)}
        input_index = {id(inp): i for i, inp in enumerate(subgraph.inputs)}
        parts = []
        for op in subgraph.ops:
            parts.append(op.name)
            for operand in op.operands:
                owner = operand.owner
                if id(owner) in op_index:
                    parts.append(f"op{op_index[id(owner)]}")
                elif id(operand) in input_index:
                    parts.append(f"in{input_index[id(operand)]}")
                else:
                    parts.append("ext")
            for r in op.results:
                parts.append(str(r.type))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
