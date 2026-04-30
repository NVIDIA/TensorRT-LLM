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

set(SRC_CU compressorKernels.cu)

add_library(compressorKernels_src OBJECT ${SRC_CU})
set_property(TARGET compressorKernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET compressorKernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                   ON)
target_compile_options(compressorKernels_src
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
