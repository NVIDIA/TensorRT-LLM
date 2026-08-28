# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Local integer helpers used by the vendored PrimTS package."""


def ceil_div(x: int, y: int) -> int:
    """Return the ceiling of ``x / y`` for positive integral extents."""

    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    """Round ``x`` up to the nearest multiple of ``y``."""

    return ceil_div(x, y) * y
