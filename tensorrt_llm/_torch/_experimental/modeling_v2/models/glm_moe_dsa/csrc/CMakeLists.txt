#
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
#

# glm_moe_dsa native code. Each target keeps its own in <target>/csrc/; see the
# modeling_v2 README.
add_modeling_v2_family(glm_moe_dsa)

# The kernels are written for the SM100 (datacenter Blackwell) family, for
# example its 32-byte global loads.
add_modeling_v2_target(
  glm_moe_dsa glm_5_2_nvfp4__sm_103__tp8
  ARCHS 100f
  SOURCES *.cu *.cpp)
