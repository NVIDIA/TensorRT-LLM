/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "unfusedAttentionKernels_2_template.h"

namespace tensorrt_llm
{
namespace kernels
{

#ifdef ENABLE_BF16
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVBlockArray, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVLinearBuffer, false);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVBlockArray, true);
INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(__nv_bfloat16, KVLinearBuffer, true);
#endif

} // namespace kernels
} // namespace tensorrt_llm
