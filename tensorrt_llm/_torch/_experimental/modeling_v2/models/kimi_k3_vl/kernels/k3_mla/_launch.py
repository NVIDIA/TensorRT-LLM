# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cuda.bindings.driver as cuda_driver


def _attn_no_cluster(stream, clusters: int, cluster_size: int, cluster_wave: int) -> bool:
    if clusters > cluster_wave:
        sm_type = cuda_driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        status, context = cuda_driver.cuStreamGetCtx(stream)
        if status != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"k3_mla_attn: cuStreamGetCtx failed: {status.name}")
        status, resource = cuda_driver.cuCtxGetDevResource(context, sm_type)
        if status != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"k3_mla_attn: stream SM resource query failed: {status.name}")
        if resource.type != sm_type or resource.sm.smCount <= 0:
            raise RuntimeError("k3_mla_attn: stream has no valid SM resource count")
        if clusters * cluster_size <= resource.sm.smCount:
            return True
    status, green_context = cuda_driver.cuStreamGetGreenCtx(stream)
    if status != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"k3_mla_attn: cuStreamGetGreenCtx failed: {status.name}")
    if int(green_context) != 0:
        raise RuntimeError(
            "k3_mla_attn: the no-cluster grid must fit the green-context stream's SM resources; "
            "a 16-CTA cluster fallback is not qualified for this stream"
        )
    return False
