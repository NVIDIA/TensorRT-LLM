# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
# rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""FLUX.3 fused per-token AdaLN modulate."""

from typing import Sequence

import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.pertoken_adaln import (
        fused_pertoken_adaln as _fused_pertoken_adaln,
    )

    _PERTOKEN_ADALN_IMPORT_OK = True
except (ImportError, OSError):
    _fused_pertoken_adaln = None
    _PERTOKEN_ADALN_IMPORT_OK = False

_PERTOKEN_ADALN_ALIGNMENT = 32


def _pertoken_adaln_arch_ok(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(device) == (10, 0)
    except RuntimeError:
        return False


def _outer_strides_ok(t: torch.Tensor, hidden_size: int) -> bool:
    return all(t.stride(i) % hidden_size == 0 for i in range(t.ndim - 1))


class Flux3PerTokenAdaLN:
    """Adapt FLUX.3's bf16 per-token AdaLN modulate sites to the fused SM100 kernel."""

    def __init__(self, hidden_size: int, dtype: torch.dtype) -> None:
        self._hidden_size = hidden_size
        self._eligible = (
            _PERTOKEN_ADALN_IMPORT_OK
            and dtype == torch.bfloat16
            and hidden_size % 256 == 0
            and hidden_size <= 8192
        )
        self._enabled = False

    def set_runtime_enabled(self, enabled: bool) -> None:
        self._enabled = self._eligible and enabled

    def apply(
        self,
        x: torch.Tensor,
        mod_shift: torch.Tensor,
        mod_scale: torch.Tensor,
        pre_norm: LayerNorm,
        eps: float,
    ) -> torch.Tensor:
        if self._enabled and self._metadata_ok(x, mod_shift, mod_scale):
            assert _fused_pertoken_adaln is not None
            return _fused_pertoken_adaln(x, mod_shift, mod_scale, None, None, eps)
        return (1 + mod_scale) * pre_norm(x) + mod_shift

    def _metadata_ok(
        self,
        x: torch.Tensor,
        mod_shift: torch.Tensor,
        mod_scale: torch.Tensor,
    ) -> bool:
        hidden_size = self._hidden_size
        return (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim == 3
            and x.shape[-1] == hidden_size
            and x.stride(-1) == 1
            and _outer_strides_ok(x, hidden_size)
            and mod_shift.device == x.device
            and mod_shift.dtype == x.dtype
            and mod_shift.shape == x.shape
            and mod_shift.stride(-1) == 1
            and _outer_strides_ok(mod_shift, hidden_size)
            and mod_scale.device == x.device
            and mod_scale.dtype == x.dtype
            and mod_scale.shape == x.shape
            and mod_scale.stride(-1) == 1
            and _outer_strides_ok(mod_scale, hidden_size)
        )


class Flux3PerTokenAdaLNRuntime:
    """Resolve shared runtime eligibility before individually compiled FLUX.3 blocks."""

    def __init__(self, adapters: Sequence[Flux3PerTokenAdaLN]) -> None:
        self._adapters = tuple(adapters)
        self._runtime_key: tuple[torch.device, torch.dtype, int, bool] | None = None
        self._enabled = False

    def prepare(
        self, x: torch.Tensor, mod_shift: torch.Tensor, mod_scale: torch.Tensor
    ) -> torch.Tensor:
        """Validate [B, S, D] modulation views and return an aligned activation.

        Call outside compiled blocks whenever a stream's modulation changes.
        """
        hidden_size = x.shape[-1]
        modulation_ok = x.ndim == 3 and all(
            value.device == x.device
            and value.dtype == x.dtype
            and value.shape == x.shape
            and value.stride(-1) == 1
            and _outer_strides_ok(value, hidden_size)
            and value.data_ptr() % _PERTOKEN_ADALN_ALIGNMENT == 0
            for value in (mod_shift, mod_scale)
        )
        runtime_key = (x.device, x.dtype, hidden_size, modulation_ok)
        if runtime_key != self._runtime_key:
            self._enabled = (
                modulation_ok
                and x.dtype == torch.bfloat16
                and _pertoken_adaln_arch_ok(x.device)
                and any(adapter._eligible for adapter in self._adapters)
            )
            for adapter in self._adapters:
                adapter.set_runtime_enabled(self._enabled)
            self._runtime_key = runtime_key

        if self._enabled and (
            x.stride(-1) != 1
            or not _outer_strides_ok(x, hidden_size)
            or x.data_ptr() % _PERTOKEN_ADALN_ALIGNMENT != 0
        ):
            return x.clone(memory_format=torch.contiguous_format)
        return x
