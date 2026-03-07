#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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

set(AETHER_SPARSE_ATTN_PLUGIN_SOURCES aetherSparseAttentionPlugin.cpp)

set(AETHER_SPARSE_ATTN_PLUGIN_CU_SOURCES 
    aether_generate_indices.cu
    aether_sparse_flash_fwd_kernel.cu)

list(APPEND PLUGIN_SOURCES ${AETHER_SPARSE_ATTN_PLUGIN_SOURCES})


list(APPEND PLUGIN_CU_SOURCES ${AETHER_SPARSE_ATTN_PLUGIN_CU_SOURCES})

set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)
set(PLUGIN_CU_SOURCES ${PLUGIN_CU_SOURCES} PARENT_SCOPE)
