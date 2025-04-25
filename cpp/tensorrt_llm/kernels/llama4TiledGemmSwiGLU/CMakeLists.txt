#
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)

add_library(llama4_fc_swiglu_tiled_fp8 OBJECT ${SRC_CPP} ${SRC_CU})
set_property(TARGET llama4_fc_swiglu_tiled_fp8 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llama4_fc_swiglu_tiled_fp8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
