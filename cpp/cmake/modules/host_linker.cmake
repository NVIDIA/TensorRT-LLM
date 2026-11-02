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

option(USE_LLD "Use LLVM lld for host linking (Linux, GCC/Clang)" ON)
if(USE_LLD)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_CXX_COMPILER_ID
                                               MATCHES "^(GNU|Clang)$")
    message(FATAL_ERROR "USE_LLD requires Linux and a GCC or Clang compiler")
  endif()

  include(CheckLinkerFlag)
  check_linker_flag(CXX "-fuse-ld=lld" TRTLLM_HAS_LLD_LINKER)
  if(NOT TRTLLM_HAS_LLD_LINKER)
    message(
      FATAL_ERROR
        "USE_LLD requires ld.lld on PATH and a compiler supporting -fuse-ld=lld"
    )
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      message(
        FATAL_ERROR
          "USE_LLD with GCC does not support LTO; disable CMAKE_INTERPROCEDURAL_OPTIMIZATION or USE_LLD"
      )
    endif()
    # lld cannot consume GCC LTO objects. Defining this variable also prevents
    # pybind11_add_module from enabling its own automatic LTO flags.
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
  endif()

  # CMake uses the host compiler for the final link of CUDA targets. Device
  # linking and static archives use their configured tools.
  add_link_options("$<HOST_LINK:-fuse-ld=lld>")
  message(STATUS "Using LLVM lld for host linking")
endif()
