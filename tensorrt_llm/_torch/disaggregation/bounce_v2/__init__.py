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
"""bounce_v2: hybrid Python/C++ bounce transport for disaggregated KV transfer.

This package holds the PURE-LOGIC layer (config, buddy arena allocator,
credit scheduler, chunk planner, wire codec): stdlib + numpy only, importable
standalone, no CUDA / NIXL / torch / tensorrt_llm imports. The mechanism
layer (fabric arena, batched copy op, completion poller) binds in separately.
"""

from .buddy import BuddyAllocator
from .codec import (
    BOUNCE_MAGIC,
    BOUNCE_VERSION,
    AckEntry,
    BounceMsgHeader,
    BounceMsgType,
    CreditEntry,
    decode_ack,
    decode_credits,
    decode_header,
    decode_scatter,
    decode_want,
    encode_ack,
    encode_cancel,
    encode_data,
    encode_grant,
    encode_want,
    has_bounce_magic,
    is_cancel_want,
)
from .config import BounceV2Config
from .plan import ALIGNMENT, SCATTER_RUN_DTYPE, BounceChunk, Plan, build_plan
from .scheduler import CreditScheduler, Grant

__all__ = [
    "ALIGNMENT",
    "BOUNCE_MAGIC",
    "BOUNCE_VERSION",
    "SCATTER_RUN_DTYPE",
    "AckEntry",
    "BounceChunk",
    "BounceMsgHeader",
    "BounceMsgType",
    "BounceV2Config",
    "BuddyAllocator",
    "CreditEntry",
    "CreditScheduler",
    "Grant",
    "Plan",
    "build_plan",
    "decode_ack",
    "decode_credits",
    "decode_header",
    "decode_scatter",
    "decode_want",
    "encode_ack",
    "encode_cancel",
    "encode_data",
    "encode_grant",
    "encode_want",
    "has_bounce_magic",
    "is_cancel_want",
]
