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
"""DeCo-Attention kernels in CuTe DSL: NVFP4 QK with FP8 PV on B200, GB300 and Rubin, NVFP4 QK with NVFP4 PV or FP8
PV on RTX PRO 6000 (README.md)."""

from .api import attention, prepare, pv_formats, ranks, workspace_size
from .inputs import sample_rows

__all__ = ["attention", "prepare", "pv_formats", "ranks", "sample_rows", "workspace_size"]
