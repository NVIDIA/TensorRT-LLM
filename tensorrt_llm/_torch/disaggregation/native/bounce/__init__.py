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
"""Opt-in (TransferWorkerConfig.bounce) VRAM d2d KV bounce buffering: coalesce a
transfer's scattered per-block KV into ONE contiguous fabric-VMM WRITE (reliable
cuda_ipc/MNNVL) through the BounceTransport interface; default per-block path is
unchanged when no Config is given. config_from_size() is the on/off switch."""

from .buffer import Buffer, SlotAllocator
from .config import Config, FixedSizing, Sizing, SizingContext, config_from_size
from .core import BounceTransport, Disposition, ScatterState, TransferContext, TransferState
from .gather_scatter import Plan
from .impl import (
    NoBounceTransport,
    VmmBounceTransport,
    build_send_request,
    create_bounce,
    decode_result_tail,
    encode_result_tail,
    scatter_write_result,
)

__all__ = [
    "BounceTransport",
    "Buffer",
    "Config",
    "Disposition",
    "FixedSizing",
    "NoBounceTransport",
    "Plan",
    "ScatterState",
    "Sizing",
    "SizingContext",
    "SlotAllocator",
    "TransferContext",
    "TransferState",
    "VmmBounceTransport",
    "build_send_request",
    "config_from_size",
    "create_bounce",
    "decode_result_tail",
    "encode_result_tail",
    "scatter_write_result",
]
