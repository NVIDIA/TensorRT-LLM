# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# torch.ops.trtllm MoE ops, grouped like kernels/moe/ (marlin/ is thop-only).
# The parent defines th_common_moe; moeOp.cpp holds the primary
# TORCH_LIBRARY(trtllm) block.
file(GLOB_RECURSE SRC_CPP CONFIGURE_DEPENDS *.cpp)
target_sources(th_common_moe PRIVATE ${SRC_CPP})
