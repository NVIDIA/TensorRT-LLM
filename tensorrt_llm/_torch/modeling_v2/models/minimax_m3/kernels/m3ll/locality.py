# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-process locality-map cache of the MiniMax-M3.1 code, over the family's shared placement implementation
(``ops/gemm/locality``)."""

import threading
from typing import Dict

import torch

from ...ops.gemm.locality import LocalityMap as _LocalityMap
from ...ops.gemm.locality import _Mapping  # noqa: F401 (re-exported for the die-local placements)

_LOCK = threading.Lock()
_MAPS: Dict[int, "LocalityMap"] = {}


class LocalityMap(_LocalityMap):
    @classmethod
    def get(cls, device) -> "LocalityMap":
        idx = torch.device(device).index if not isinstance(device, int) else device
        idx = torch.cuda.current_device() if idx is None else idx
        with _LOCK:
            m = _MAPS.get(idx)
            if m is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "LocalityMap probes the device: build it before CUDA-graph capture"
                    )
                m = _MAPS[idx] = cls(idx)
        return m
