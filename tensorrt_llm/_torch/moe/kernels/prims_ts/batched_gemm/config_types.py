# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-safe batched GEMM option types; no CUDA or DSL imports."""

from enum import IntEnum


class BatchMode(IntEnum):
    BATCH_N = 0
    BATCH_M = 1


class RouteImpl(IntEnum):
    AUTO = -1
    NONE = 0
    TMA = 1
    LDGSTS = 2


class TileScheduler(IntEnum):
    STATIC = 0
    PERSISTENT = 1


class CtaRasterOrder(IntEnum):
    ALONG_M = 0
    ALONG_N = 1


class ActKind(IntEnum):
    NONE = 0
    SWIGLU = 1
    GEGLU = 2
    RELU2 = 3
    SILU = 4
    SITU = 5


class BiasType(IntEnum):
    NONE = 0
    M = 1


class DType(IntEnum):
    FP32 = 1
    FP16 = 2
    BF16 = 3
    E2M1 = 4
    MXE2M1 = 5
    MXE4M3 = 6
    E4M3 = 7


class SfLayout(IntEnum):
    R8c4 = 1
    LINEAR = 2
    R128c4 = 3


class SfSmemToTmemCopy(IntEnum):
    NONE = 0
    UTCCP = 2
    LDS_STTM = 3
