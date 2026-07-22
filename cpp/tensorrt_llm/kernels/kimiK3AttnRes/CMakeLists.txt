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

add_library(kimi_k3_attn_res_src OBJECT attnResFwd.cu)

set_property(TARGET kimi_k3_attn_res_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kimi_k3_attn_res_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS
                                                  ON)
target_compile_options(kimi_k3_attn_res_src
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
