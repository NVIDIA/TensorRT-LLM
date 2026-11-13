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

add_modeling_v2_family(kimi_k3_vl)

add_modeling_v2_shared(
  kimi_k3_vl
  ARCHS 100f
  SOURCES *.cu *.cpp
  DEPS internal_cutlass)

set_source_files_properties(k3AttnResDecode.cu PROPERTIES COMPILE_OPTIONS
                                                          "--use_fast_math")

add_modeling_v2_target(kimi_k3_vl kimi_k3_mxfp4__sm_100__tp16_moetp4ep4
                       ARCHS 100f)
add_modeling_v2_target(kimi_k3_vl kimi_k3_mxfp4__sm_100__tp16_moetp16ep1
                       ARCHS 100f)
