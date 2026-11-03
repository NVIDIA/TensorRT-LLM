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

add_library(csa2_sparse_decode_src OBJECT csa2SparseDecode.cu)

# The kernel uses tcgen05/TMEM, TMA gather4 and the SM100-family FP4/FP8
# conversions, so it is built only for the sm_100 family (sm_100/sm_103), and
# for sm_103 on its own, where the MXFP8 epilogue reads its block maxima with
# tcgen05.ld.red. Without that family the target compiles a trapping placeholder
# and isSupported() reports false.
set_cuda_architectures(csa2_sparse_decode_src 100f 103)

set_property(TARGET csa2_sparse_decode_src PROPERTY POSITION_INDEPENDENT_CODE
                                                    ON)
set_property(TARGET csa2_sparse_decode_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                    ON)
target_compile_options(
  csa2_sparse_decode_src PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math
                                 --expt-relaxed-constexpr>)
