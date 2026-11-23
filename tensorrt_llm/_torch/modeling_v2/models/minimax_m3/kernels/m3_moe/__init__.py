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
"""MiniMax-M3.1-Flash MoE kernels of the sm_107 TP2 and TP4 targets for low-batch decode (CuTe DSL): ``m3_moe`` for
1..8 tokens and ``m3_moe16`` for 9..16 tokens. A target's ``kernels/m3_moe/op.py`` / ``op16.py`` loads a kernel file
once per trace-time configuration (``M3_CONFIG``)."""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
# The kernel files a target's op loads by path (one module per trace-time configuration).
M3_MOE_KERNEL_PATH = os.path.join(_HERE, "m3_moe_kernel.py")
M3_MOE16_KERNEL_PATH = os.path.join(_HERE, "m3_moe16_kernel.py")
