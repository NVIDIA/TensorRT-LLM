# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
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

if(NOT ENABLE_MULTI_DEVICE EQUAL 0)
  add_gtest(allReduceKernelTest allReduce/allReduceKernelTest.cu)
  add_gtest(allReduceFusionTest allReduce/allReduceFusionTest.cu)
  add_gtest(gemmAllReduceTest allReduce/gemmAllReduceTest.cu)
  if(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
    target_link_libraries(gemmAllReduceTest PRIVATE ar_gemm_src)
    target_compile_definitions(gemmAllReduceTest
                               PRIVATE USING_OSS_CUTLASS_ALLREDUCE_GEMM)
  endif()
endif()
