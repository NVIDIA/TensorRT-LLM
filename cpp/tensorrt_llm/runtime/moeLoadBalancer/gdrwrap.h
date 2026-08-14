/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#ifdef _WIN32

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include <cstdint>
#include <cstdlib>

// Dummy types for Windows to allow compilation.
struct gdr;
typedef struct gdr* gdr_t;

typedef struct gdr_mh_s
{
    unsigned long h;
} gdr_mh_t;

struct gdr_info
{
};
typedef struct gdr_info gdr_info_t;

namespace tensorrt_llm
{
namespace runtime
{
namespace gdrcopy
{

struct GdrMemDesc
{
};

// On Windows, GDRCopy is not supported. These are stub implementations.
inline bool initialize()
{
    TLLM_LOG_INFO("GDRCopy is not supported on Windows.");
    return false;
}

inline bool isInitialized()
{
    return false;
}

#define GDRCOPY_UNSUPPORTED() TLLM_THROW("GDRCopy is not supported on Windows")

inline gdr_t open()
{
    GDRCOPY_UNSUPPORTED();
    return nullptr;
}

inline int close(gdr_t /*g*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

inline int pin_buffer(gdr_t /*g*/, unsigned long /*addr*/, size_t /*size*/, uint64_t /*p2p_token*/,
    uint32_t /*va_space*/, gdr_mh_t* /*handle*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

inline int unpin_buffer(gdr_t /*g*/, gdr_mh_t /*handle*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

inline int get_info(gdr_t /*g*/, gdr_mh_t /*handle*/, gdr_info_t* /*info*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

inline int map(gdr_t /*g*/, gdr_mh_t /*handle*/, void** /*va*/, size_t /*size*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

inline int unmap(gdr_t /*g*/, gdr_mh_t /*handle*/, void* /*va*/, size_t /*size*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

inline void runtime_get_version(int* /*major*/, int* /*minor*/)
{
    GDRCOPY_UNSUPPORTED();
}

inline void driver_get_version(gdr_t /*g*/, int* /*major*/, int* /*minor*/)
{
    GDRCOPY_UNSUPPORTED();
}

inline int copy_to_mapping(gdr_mh_t /*handle*/, void* /*map_d_ptr*/, void const* /*h_ptr*/, size_t /*size*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

inline int copy_from_mapping(gdr_mh_t /*handle*/, void* /*h_ptr*/, void const* /*map_d_ptr*/, size_t /*size*/)
{
    GDRCOPY_UNSUPPORTED();
    return -1;
}

template <typename T>
void gdrCudaMalloc(T** /*ptr*/, T** /*devPtr*/, size_t /*nelem*/, GdrMemDesc** /*memDesc*/, gdr_t /*handle*/)
{
    GDRCOPY_UNSUPPORTED();
}

inline void gdrCudaFree(GdrMemDesc* /*memDesc*/, gdr_t /*handle*/)
{
    GDRCOPY_UNSUPPORTED();
}

} // namespace gdrcopy
} // namespace runtime
} // namespace tensorrt_llm

#else // NOT _WIN32

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

// These definitions are from gdrapi.h to avoid a direct dependency on the header.
#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)

struct gdr;
typedef struct gdr* gdr_t;

typedef struct gdr_mh_s
{
    unsigned long h;
} gdr_mh_t;

struct gdr_info
{
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped : 1;
    unsigned wc_mapping : 1;
};
typedef struct gdr_info gdr_info_t;

namespace tensorrt_llm
{
namespace runtime
{
namespace gdrcopy
{

// This is required as the GDR memory is mapped WC
#if !defined(__NVCC__)
#if defined(__PPC__)
static inline void wc_store_fence(void)
{
    asm volatile("sync");
}
#elif defined(__x86_64__)
#include <immintrin.h>

static inline void wc_store_fence(void)
{
    _mm_sfence();
}
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>

static inline void wc_store_fence(void)
{
    std::atomic_thread_fence(std::memory_order_release);
}
#else
#include <stdatomic.h>

static inline void wc_store_fence(void)
{
    atomic_thread_fence(memory_order_release);
}
#endif
#endif
#endif

// Initializes the GDRCopy library by dynamically loading it.
// This function is thread-safe.
// Returns true on success, false on failure.
bool initialize();

// Returns true if the GDRCopy library has been successfully initialized.
bool isInitialized();

// All functions below are wrappers around the GDRCopy library functions.
// They are thread-safe.
// Before calling any of these functions, ensure the library is initialized.

gdr_t open();
int close(gdr_t g);
int pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t* handle);
int unpin_buffer(gdr_t g, gdr_mh_t handle);
int get_info(gdr_t g, gdr_mh_t handle, gdr_info_t* info);
int map(gdr_t g, gdr_mh_t handle, void** va, size_t size);
int unmap(gdr_t g, gdr_mh_t handle, void* va, size_t size);
void runtime_get_version(int* major, int* minor);
void driver_get_version(gdr_t g, int* major, int* minor);
int copy_to_mapping(gdr_mh_t handle, void* map_d_ptr, void const* h_ptr, size_t size);
int copy_from_mapping(gdr_mh_t handle, void* h_ptr, void const* map_d_ptr, size_t size);

// --- GDRCopy Memory Management Helpers ---

struct GdrMemDesc
{
    void* gdrDeviceMem;
    void* gdrMap;
    size_t gdrOffset;
    size_t gdrMapSize;
    gdr_mh_t gdrMh;
};

// Allocates memory that can be used with GDRCopy.
void gdrCudaMalloc(void** ptr, void** devPtr, size_t mapSize, GdrMemDesc** memDesc, gdr_t handle);

// Frees memory allocated with gdrCudaMalloc.
void gdrCudaFree(GdrMemDesc* memDesc, gdr_t handle);

} // namespace gdrcopy
} // namespace runtime
} // namespace tensorrt_llm

#endif // _WIN32
