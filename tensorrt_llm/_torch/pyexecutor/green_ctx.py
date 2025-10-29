# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from cuda.bindings import driver

from tensorrt_llm.runtime.generation import CUASSERT


def green_ctx_create_streams(res_list, device):
    streams = []
    for res in res_list:
        desc = CUASSERT(driver.cuDevResourceGenerateDesc([res], 1))[0]
        green_ctx = CUASSERT(
            driver.cuGreenCtxCreate(
                desc, device, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )[0]
        stream = CUASSERT(
            driver.cuGreenCtxStreamCreate(
                green_ctx, driver.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
            )
        )[0]
        stream = torch.cuda.get_stream_from_external(stream, device)
        streams.append(stream)
    return streams


def green_ctx_split_percent(sm_percent: float, device_id: int = 0):
    device = CUASSERT(driver.cuDeviceGet(device_id))[0]

    res = CUASSERT(
        driver.cuDeviceGetDevResource(device, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
    )[0]
    sm_count = res.sm.smCount

    major = CUASSERT(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
        )
    )[0]
    if major >= 9:
        sm_min = 8
        sm_align = 8
    else:
        sm_min = 4 if major == 8 else 2
        sm_align = 2

    def green_ctx_split_aligned(sm_g1):
        sm_g1 = round(sm_g1 / sm_align) * sm_align
        sm_g1 = min(max(sm_g1, sm_min), sm_count - sm_min)
        result = CUASSERT(
            driver.cuDevSmResourceSplitByCount(
                1,  # nbGroups
                res,
                0,  # useFlags
                sm_g1,
            )
        )
        res_split = (result[0][0], result[2])
        streams = green_ctx_create_streams(res_split, device)
        return streams, res_split

    sm_g1 = round(sm_count * sm_percent)
    sm_g2 = sm_count - sm_g1
    # Choose the split closer to sm_percent when sm_count is not divisible by sm_align
    sm_g1_dist = min(sm_g1 % sm_align, sm_align - (sm_g1 % sm_align))
    sm_g2_dist = min(sm_g2 % sm_align, sm_align - (sm_g2 % sm_align))
    if sm_g1_dist <= sm_g2_dist:
        (stream_g1, stream_g2), (res_g1, res_g2) = green_ctx_split_aligned(sm_g1)
    else:
        (stream_g2, stream_g1), (res_g2, res_g1) = green_ctx_split_aligned(sm_g2)
    return (stream_g1, stream_g2), (res_g1, res_g2)
