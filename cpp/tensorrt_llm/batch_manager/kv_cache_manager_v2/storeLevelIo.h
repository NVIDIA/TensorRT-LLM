/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
/* The IO table of a KVCMv2 store level: plain C, so the KV cache manager and the store module (two separate binaries)
 * exchange no C++ object. The store module owns the table and its ctx; KVCMv2 calls it. */
#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif
#define TLLM_KV_STORE_LEVEL_IO_ABI_VERSION 1u
/* Set in a done word, together with the op's doneValue, by an IO in degrade fault mode when a row of the op failed
 * (a put row's slot then holds no valid bytes; a get row's destination is not valid). An IO in the default fault mode
 * never sets it: a failed row is fatal there. KVCMv2 waits for a done word on the GPU with CU_STREAM_WAIT_VALUE_GEQ,
 * which is a cyclic compare, (int32_t)(*word - doneValue) >= 0: the flag must therefore stay below bit 31 (bit 31
 * would make the difference negative and the wait never pass), and done values must stay below the flag. */
#define TLLM_KV_STORE_DONE_FAILED 0x40000000u

    enum
    {
        TLLM_KV_STORE_OP_PUT = 1,
        TLLM_KV_STORE_OP_GET = 2
    };

    typedef struct TllmKvStoreRow
    {
        uint32_t poolGroup; /* store-level pool group */
        uint32_t reserved;  /* 0 */
        uint64_t slot;      /* slot id in the pool group */
        uint64_t address;   /* registered host memory: PUT reads, GET writes slotBytes[poolGroup] bytes */
    } TllmKvStoreRow;

    typedef struct TllmKvStoreOp
    {
        uint64_t seq;               /* unique per attach, increasing in enqueue order */
        uint32_t kind;              /* TLLM_KV_STORE_OP_PUT or _GET */
        uint32_t numRows;           /* 1..maxRowsPerOp */
        TllmKvStoreRow const* rows; /* valid only during submit() */
        uint32_t* doneWord;         /* once every row completed: atomic store of doneValue, release order; a degrade
                                       IO ORs TLLM_KV_STORE_DONE_FAILED into it when a row failed */
        uint32_t doneValue;
        uint32_t reserved;
    } TllmKvStoreOp;

    typedef struct TllmKvStoreLevelIoStats
    {
        uint64_t putOps, putRows, putBytes, getOps, getRows, getBytes, retries, queuedOps, maxQueuedOps;
        uint64_t putP50Us, putP99Us, getP50Us, getP99Us; /* per store call, since configure */
    } TllmKvStoreLevelIoStats;

    typedef struct TllmKvStoreLevelIoV1
    {
        uint32_t abiVersion;   /* TLLM_KV_STORE_LEVEL_IO_ABI_VERSION */
        uint32_t structBytes;  /* sizeof(TllmKvStoreLevelIoV1) */
        uint32_t maxRowsPerOp; /* KVCMv2 never submits more rows in one op */
        uint32_t reserved;
        void* ctx;
        /* 0 or a negative code. Ranges stay registered until unregisterMemory. */
        int32_t (*registerMemory)(void* ctx, void* base, uint64_t bytes);
        int32_t (*unregisterMemory)(void* ctx, void* base);
        /* Once, after the ranges are registered: the store level's layout. Allocates every slot object and self-tests
         * (blocking, bring-up only). 0 or a negative code. */
        int32_t (*configure)(void* ctx, uint32_t numPoolGroups, uint64_t const* slotBytes, uint64_t const* numSlots);
        /* Copies op and rows and returns at once. May run on a CUDA host-function thread: no CUDA call, no GIL, no I/O
         * wait. */
        void (*submit)(void* ctx, TllmKvStoreOp const* op);
        /* Waits until every submitted op completed or timeoutS passed; returns the ops still pending. */
        uint64_t (*drain)(void* ctx, double timeoutS);
        void (*stats)(void* ctx, TllmKvStoreLevelIoStats* out);
    } TllmKvStoreLevelIoV1;
#ifdef __cplusplus
}
#endif
