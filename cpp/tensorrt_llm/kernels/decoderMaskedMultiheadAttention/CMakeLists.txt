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
# Initialize empty list of src cuda files.
set(SRC_CU)
set(SRC_CU_EXTRA)

# Exclude files in nvrtcWrapper folder; add it as a subdirectory instead.
#
# nvrtcWrapper directory will produce nvrtc_wrapper_src, which is linked to
# decoder_attention_src.
list(FILTER SRC_CPP EXCLUDE REGEX ".*nvrtcWrapper/.*")
add_subdirectory(decoderXQAImplJIT/nvrtcWrapper)

filter_cuda_archs("80" SRC_CPP)
filter_cuda_archs("86" SRC_CPP)
filter_cuda_archs("89" SRC_CPP)
filter_cuda_archs("90" SRC_CPP)
filter_cuda_archs("120" SRC_CPP)

set(basic_heads 32 64 128)
foreach(HEAD ${basic_heads})
  file(GLOB_RECURSE HEAD_SRCS
       instantiation/decoderMaskedMultiheadAttention${HEAD}*.cu)
  list(APPEND SRC_CU ${HEAD_SRCS})
endforeach()

# skip mmha 48, 80, 96, 104, 112, 144, 160, 192 and 224 for fast build
if(FAST_BUILD)
  set(extra_heads 256)
else()
  set(extra_heads
      48
      80
      96
      104
      112
      144
      160
      192
      224
      256)
endif()

foreach(HEAD ${extra_heads})
  file(GLOB_RECURSE HEAD_SRCS
       instantiation/decoderMaskedMultiheadAttention${HEAD}*.cu)
  list(APPEND SRC_CU_EXTRA ${HEAD_SRCS})
endforeach()

if(NOT WIN32)
  add_library(decoder_attention_src OBJECT ${SRC_CPP})

  # Split some sources to shared library for Linux
  add_library(${DECODER_SHARED_TARGET_0} SHARED ${SRC_CU})
  set_property(TARGET ${DECODER_SHARED_TARGET_0}
               PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${DECODER_SHARED_TARGET_0}
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_library(${DECODER_SHARED_TARGET_1} SHARED ${SRC_CU_EXTRA})
  set_property(TARGET ${DECODER_SHARED_TARGET_1}
               PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${DECODER_SHARED_TARGET_1}
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
  add_library(decoder_attention_src OBJECT ${SRC_CPP} ${SRC_CU} ${SRC_CU_EXTRA})
endif()

target_link_libraries(decoder_attention_src PUBLIC nvrtc_wrapper_src)
set_property(TARGET decoder_attention_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET decoder_attention_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                   ON)
