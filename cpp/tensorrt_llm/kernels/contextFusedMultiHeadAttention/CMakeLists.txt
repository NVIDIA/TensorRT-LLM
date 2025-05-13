#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

filter_cuda_archs("80" SRC_CPP)
filter_cuda_archs("86" SRC_CPP)
filter_cuda_archs("89" SRC_CPP)
filter_cuda_archs("90" SRC_CPP)
filter_cuda_archs("100" SRC_CPP)
filter_cuda_archs("120" SRC_CPP)

filter_cuda_archs("80" SRC_CU)
filter_cuda_archs("86" SRC_CU)
filter_cuda_archs("89" SRC_CU)
filter_cuda_archs("90" SRC_CU)
filter_cuda_archs("100" SRC_CU)
filter_cuda_archs("120" SRC_CU)

add_library(context_attention_src OBJECT ${SRC_CPP} ${SRC_CU})
target_include_directories(
  context_attention_src
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/fmha_v2/src/
          ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/fmha_v2/generated/)
target_compile_definitions(context_attention_src PRIVATE USE_DEMO_BERT_PARAMS=1
                                                         GENERATE_CUBIN=1)
set_property(TARGET context_attention_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET context_attention_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                   ON)
