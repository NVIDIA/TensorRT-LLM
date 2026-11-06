# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import platform
import sys
import types
from typing import Optional

from ..logger import logger


def _skip_legacy_cutlass_mlir_helpers() -> None:
    """Keep Cutlass version discovery from importing its legacy helper tree."""
    legacy_name = "cutlass.base_dsl._mlir_helpers"
    if legacy_name in sys.modules:
        return

    # Cutlass uses pkgutil.walk_packages to hash its sources. The internal
    # package also ships this unused legacy helper tree alongside the canonical
    # cutlass._mlir_helpers package. Prevent pkgutil from descending into the
    # legacy tree, which would register the same MLIR value casters twice.
    legacy_module = types.ModuleType(legacy_name)
    legacy_module.__path__ = []
    sys.modules[legacy_name] = legacy_module


IS_CUTLASS_DSL_AVAILABLE = False

# Whether the public CuTe DSL package provides the SM107/Rubin helper module.
# Rubin kernels stay disabled when it is absent and callers retain their
# existing fallback paths.
# TODO: flips to True once a Rubin-capable CuTe DSL package ships in the image.
IS_CUTLASS_DSL_RUBIN_AVAILABLE = False

if platform.system() != "Windows":
    try:
        from cutlass import cute  # noqa
        _skip_legacy_cutlass_mlir_helpers()
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True

        try:
            import cutlass.utils.rubin_helpers  # noqa
        except ImportError:
            pass
        else:
            logger.info("cutlass dsl Rubin helpers are available")
            IS_CUTLASS_DSL_RUBIN_AVAILABLE = True
    except ImportError:
        pass


class _GraphSafeDlpack:
    """DLPack export without a stream synchronization (safe under CUDA-graph capture)."""

    def __init__(self, tensor):
        self._tensor = tensor.detach()

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def graph_safe_cute_tensor(tensor,
                           assumed_align: int = 16,
                           dynamic_leading_dim: Optional[int] = None):
    """CuTe DSL view of a torch tensor whose export does not synchronize streams.

    ``dynamic_leading_dim``: the shape and the strides other than that (unit-stride) dimension's are runtime
    values, so one compiled kernel serves every size.
    """
    from cutlass.cute.runtime import from_dlpack

    view = from_dlpack(_GraphSafeDlpack(tensor), assumed_align=assumed_align)
    if dynamic_leading_dim is not None:
        view = view.mark_layout_dynamic(leading_dim=dynamic_leading_dim)
    return view


def gmem_ptr(tensor, dtype, assumed_align: int):
    """CuTe global-memory pointer of ``dtype`` to ``tensor``'s data (keeps no reference to the tensor)."""
    import cutlass.cute as cute

    return cute.runtime.make_ptr(dtype,
                                 tensor.data_ptr(),
                                 cute.AddressSpace.gmem,
                                 assumed_align=assumed_align)


def install_cutlass_dsl_compatibility() -> None:
    """Restore CuTe aliases required by pinned third-party FA4 and QuACK."""
    if not IS_CUTLASS_DSL_AVAILABLE:
        return

    import cutlass.cute as cute

    for name in ("ThrCopy", "ThrMma"):
        if not hasattr(cute.core, name) and hasattr(cute, name):
            setattr(cute.core, name, getattr(cute, name))
    if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
        cute.make_fragment = cute.make_rmem_tensor


# SMs that a kernel resident for the whole step holds (the DWDP fabric gather;
# ``DwdpManager`` sets it once, before the model is built). Persistent grids
# leave them out, so CTAs that would land on those SMs do not run their static
# tile share as a second wave. The hook below wraps
# ``HardwareInfo.get_max_active_clusters`` on the class (installed when this
# module is imported, before any kernel queries occupancy) so every CuTeDSL
# persistent grid honors the reserve. Kernels that size their own persistent
# grids take the reserve as an argument from their module, which reads
# ``sm_reserve()`` (the KDA mixer reads it at construction). Callers that
# memoize the occupancy key it on ``sm_reserve()`` (the dense GEMM custom op
# does).
_sm_reserve = 0


def set_sm_reserve(reserve: int) -> None:
    """Keep ``reserve`` SMs out of persistent grids from now on (0 = none).

    A nonzero ``reserve`` must leave at least one SM of the current device.
    """
    global _sm_reserve
    if reserve < 0:
        raise ValueError(f"SM reserve must be non-negative, got {reserve}")
    if reserve:
        import torch

        if torch.cuda.is_available():
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
            if reserve >= sm_count:
                raise ValueError(
                    f"SM reserve {reserve} must leave at least one of "
                    f"{sm_count} SMs")
    if reserve and reserve != _sm_reserve:
        logger.info(
            f"[cute_dsl_utils] persistent grids launch with {reserve} SMs reserved"
        )
    _sm_reserve = reserve


def sm_reserve() -> int:
    """SMs kept out of persistent grids (see :func:`set_sm_reserve`)."""
    return _sm_reserve


def reserved_max_active_clusters(full: int, cluster_size: int,
                                 reserve: int) -> int:
    """``full`` clusters less enough to free ``reserve`` SMs, never below one.

    A non-positive ``full`` means the device cannot host the cluster shape and
    is returned unchanged, so callers' "unsupported" checks still fire.
    """
    if reserve <= 0 or full <= 0:
        return full
    # One cluster occupies cluster_size SMs.
    return max(1, full - math.ceil(reserve / max(1, cluster_size)))


def _install_sm_reserve() -> None:
    if not IS_CUTLASS_DSL_AVAILABLE:
        return
    try:
        from cutlass.utils import HardwareInfo
    except ImportError:
        return
    if getattr(HardwareInfo.get_max_active_clusters, "_trtllm_sm_reserve_hook",
               False):
        return
    original = HardwareInfo.get_max_active_clusters

    def get_max_active_clusters(self, cluster_size: int, *args,
                                **kwargs) -> int:
        full = original(self, cluster_size, *args, **kwargs)
        return reserved_max_active_clusters(full, cluster_size, sm_reserve())

    get_max_active_clusters._trtllm_sm_reserve_hook = True
    HardwareInfo.get_max_active_clusters = get_max_active_clusters


_install_sm_reserve()
