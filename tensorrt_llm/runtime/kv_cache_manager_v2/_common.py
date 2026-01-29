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

import enum
import os
from dataclasses import dataclass
from typing import Final, NewType

NDEBUG: Final[int] = int(os.environ.get("TLLM_KV_CACHE_MANAGER_V2_DEBUG", "0")) == 0


class PageStatus(enum.IntEnum):
    LOCKED = 0  # Required in GPU. Eviction/dropping not allowed
    HELD = 1  # Allow eviction but not dropping
    DROPPABLE = 2  # Allow eviction and dropping


# Can extend to more tiers in the future, e.g. object storage like AWS S3.
class CacheTier(enum.IntEnum):
    GPU_MEM = 0
    HOST_MEM = 1
    DISK = 2


CacheLevel = NewType("CacheLevel", int)

GPU_LEVEL: Final[CacheLevel] = CacheLevel(0)

# Normal token id that falls in the tokenizer vocabulary.
TokenId = NewType("TokenId", int)

# For multi-modal tokens, we can handle it in either of the following ways:
#   1. Hash combine image digest and local_token_id, then use digest for every multi-modal token.
#   2. Use digest only for the first multi-modal token, and use int(vocab_size + local_token_id) for the rest.
#   3. Hash the multi-modal token embedding data and use the digest as TokenIdExt for every multi-modal token.
#      If we do this, we can't skip the encoder.
TokenIdExt = TokenId | bytes

BlockOrdinal = NewType("BlockOrdinal", int)
BlockOrdinalT = type(BlockOrdinal(0))

LayerId = NewType("LayerId", int)

CudaStream = NewType("CudaStream", int)

BeamIndex = NewType("BeamIndex", int)
DEFAULT_BEAM_INDEX: Final[BeamIndex] = BeamIndex(0)

UserId = NewType("UserId", int)

MemAddress = NewType("MemAddress", int)

FileDescriptor = NewType("FileDescriptor", int)

BAD_FILE_DESCRIPTOR: Final[FileDescriptor] = FileDescriptor(-1)

PageIndex = NewType("PageIndex", int)
BAD_PAGE_INDEX: Final[PageIndex] = PageIndex(-1)


@dataclass(slots=True, frozen=True)
class DiskAddress:
    fd: FileDescriptor
    pos: int


Address = MemAddress | DiskAddress

SlidingWindowSize = int | None

Priority = NewType("Priority", int)
PRIORITY_MIN: Final[Priority] = Priority(0)
PRIORITY_MAX: Final[Priority] = Priority(100)
PRIORITY_DEFAULT: Final[Priority] = Priority(35)
