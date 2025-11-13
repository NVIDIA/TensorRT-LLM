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

from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

if IS_CUDA_TILE_AVAILABLE:
    from .rms_norm import rms_norm_kernel, rms_norm_kernel_gather, rms_norm_kernel_static_persistent
    from .rms_norm_fuse_residual import (
        rms_norm_fuse_residual_kernel,
        rms_norm_fuse_residual_kernel_gather,
        rms_norm_fuse_residual_kernel_static_persistent,
    )

    __all__ = [
        "rms_norm_kernel",
        "rms_norm_kernel_gather",
        "rms_norm_kernel_static_persistent",
        "rms_norm_fuse_residual_kernel",
        "rms_norm_fuse_residual_kernel_gather",
        "rms_norm_fuse_residual_kernel_static_persistent",
    ]
