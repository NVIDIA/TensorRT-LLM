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

import cuda.bindings.driver as drv

from ._common import CacheTier


class OutOfMemoryError(Exception):
    pass


class HostOOMError(OutOfMemoryError):
    pass


class DiskOOMError(OutOfMemoryError):
    pass


class CuOOMError(OutOfMemoryError):
    pass


class LogicError(Exception):
    """
    This exception indicates a bug in the code.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InsufficientQuotaError(ValueError):
    """A configured cache-tier quota cannot satisfy its minimum layout."""

    def __init__(self, cache_tier: CacheTier, quota: int, min_quota: int) -> None:
        self.cache_tier = cache_tier
        self.quota = quota
        self.min_quota = min_quota
        tier_name = {
            CacheTier.GPU_MEM: "GPU",
            CacheTier.HOST_MEM: "host",
            CacheTier.DISK: "disk",
        }[cache_tier]
        super().__init__(
            f"{tier_name} cache tier quota {quota} is insufficient for the minimum "
            f"storage layout (requires at least {min_quota})"
        )

    def __reduce__(self) -> tuple[type["InsufficientQuotaError"], tuple[CacheTier, int, int]]:
        return (self.__class__, (self.cache_tier, self.quota, self.min_quota))


class CuError(Exception):
    error_code: drv.CUresult

    def __init__(self, error_code: drv.CUresult) -> None:
        self.error_code = error_code
        err, err_str = drv.cuGetErrorString(error_code)
        if err != drv.CUresult.CUDA_SUCCESS:
            err_str = "<Failed to get error string with cuGetErrorString>"
        super().__init__(f"CUDA driver error: {error_code} ({err_str})")

    def __reduce__(self) -> tuple[type["CuError"], tuple[drv.CUresult]]:
        return (self.__class__, (self.error_code,))


class ResourceBusyError(Exception):
    pass


class OutOfPagesError(Exception):
    pass
