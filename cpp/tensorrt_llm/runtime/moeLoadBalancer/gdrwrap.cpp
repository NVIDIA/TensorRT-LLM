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

#ifndef _WIN32

#include "gdrwrap.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <dlfcn.h>
#include <pthread.h>

namespace tensorrt_llm
{
namespace runtime
{
namespace gdrcopy
{

// Function pointers assigned from dlopen()
static gdr_t (*gdr_internal_open)(void);
static int (*gdr_internal_close)(gdr_t g);
static int (*gdr_internal_pin_buffer)(
    gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t* handle);
static int (*gdr_internal_unpin_buffer)(gdr_t g, gdr_mh_t handle);
static int (*gdr_internal_get_info)(gdr_t g, gdr_mh_t handle, gdr_info_t* info);
static int (*gdr_internal_map)(gdr_t g, gdr_mh_t handle, void** va, size_t size);
static int (*gdr_internal_unmap)(gdr_t g, gdr_mh_t handle, void* va, size_t size);
static void (*gdr_internal_runtime_get_version)(int* major, int* minor);
static void (*gdr_internal_driver_get_version)(gdr_t g, int* major, int* minor);
static int (*gdr_internal_copy_to_mapping)(gdr_mh_t handle, void* map_d_ptr, void const* h_ptr, size_t size);
static int (*gdr_internal_copy_from_mapping)(gdr_mh_t handle, void* h_ptr, void const* map_d_ptr, size_t size);

static pthread_mutex_t gGdrLock = PTHREAD_MUTEX_INITIALIZER;
static bool gGdrInitialized = false;
static void* gGdrApiHandle = nullptr;

#define GDRAPI_LIBNAME "libgdrapi.so"

#define LOAD_SYM(handle, symbol, funcptr)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        *(void**) (&(funcptr)) = dlsym(handle, symbol);                                                                \
        if ((funcptr) == NULL)                                                                                         \
        {                                                                                                              \
            TLLM_LOG_WARNING("dlsym failed on %s - %s", symbol, dlerror());                                            \
            dlclose(handle);                                                                                           \
            gGdrApiHandle = nullptr;                                                                                   \
            gInitStatus = false;                                                                                       \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

static pthread_once_t gInitOnceControl = PTHREAD_ONCE_INIT;
static bool gInitStatus = false;

static void initialize_internal()
{
    gGdrApiHandle = dlopen(GDRAPI_LIBNAME, RTLD_NOW);
    if (!gGdrApiHandle)
    {
        TLLM_LOG_INFO("Failed to open %s. GDRCopy support is disabled.", GDRAPI_LIBNAME);
        gInitStatus = false;
        return;
    }

    LOAD_SYM(gGdrApiHandle, "gdr_open", gdr_internal_open);
    LOAD_SYM(gGdrApiHandle, "gdr_close", gdr_internal_close);
    LOAD_SYM(gGdrApiHandle, "gdr_pin_buffer", gdr_internal_pin_buffer);
    LOAD_SYM(gGdrApiHandle, "gdr_unpin_buffer", gdr_internal_unpin_buffer);
    LOAD_SYM(gGdrApiHandle, "gdr_get_info", gdr_internal_get_info);
    LOAD_SYM(gGdrApiHandle, "gdr_map", gdr_internal_map);
    LOAD_SYM(gGdrApiHandle, "gdr_unmap", gdr_internal_unmap);
    LOAD_SYM(gGdrApiHandle, "gdr_runtime_get_version", gdr_internal_runtime_get_version);
    LOAD_SYM(gGdrApiHandle, "gdr_driver_get_version", gdr_internal_driver_get_version);
    LOAD_SYM(gGdrApiHandle, "gdr_copy_to_mapping", gdr_internal_copy_to_mapping);
    LOAD_SYM(gGdrApiHandle, "gdr_copy_from_mapping", gdr_internal_copy_from_mapping);

    gdr_t g = gdr_internal_open();
    if (g == nullptr)
    {
        TLLM_LOG_WARNING("gdr_open failed. GDRCopy support is disabled.");
        dlclose(gGdrApiHandle);
        gGdrApiHandle = nullptr;
        gInitStatus = false;
        return;
    }

    int libMajor, libMinor, drvMajor, drvMinor;
    gdr_internal_runtime_get_version(&libMajor, &libMinor);
    gdr_internal_driver_get_version(g, &drvMajor, &drvMinor);
    gdr_internal_close(g);

    if (libMajor < 2 || (libMajor == 2 && libMinor < 1) || drvMajor < 2 || (drvMajor == 2 && drvMinor < 1))
    {
        TLLM_LOG_WARNING(
            "GDRCopy library version (%d.%d) or driver version (%d.%d) is too old. Required >= 2.1. GDRCopy support "
            "is disabled.",
            libMajor, libMinor, drvMajor, drvMinor);
        dlclose(gGdrApiHandle);
        gGdrApiHandle = nullptr;
        gInitStatus = false;
        return;
    }

    TLLM_LOG_INFO("GDRCopy enabled library %d.%d driver %d.%d", libMajor, libMinor, drvMajor, drvMinor);
    gInitStatus = true;
    gGdrInitialized = true;
}

bool initialize()
{
    pthread_once(&gInitOnceControl, initialize_internal);
    return gInitStatus;
}

bool isInitialized()
{
    return gGdrInitialized;
}

#define CHECK_INITIALIZED()                                                                                            \
    TLLM_CHECK_WITH_INFO(gGdrInitialized, "GDRCopy library is not initialized. Call gdrcopy::initialize() first.")

#define GDRLOCKCALL(cmd)                                                                                               \
    [&]                                                                                                                \
    {                                                                                                                  \
        pthread_mutex_lock(&gGdrLock);                                                                                 \
        auto ret = (cmd);                                                                                              \
        pthread_mutex_unlock(&gGdrLock);                                                                               \
        return ret;                                                                                                    \
    }()

gdr_t open()
{
    CHECK_INITIALIZED();
    return gdr_internal_open();
}

int close(gdr_t g)
{
    CHECK_INITIALIZED();
    return gdr_internal_close(g);
}

int pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t* handle)
{
    CHECK_INITIALIZED();
    return GDRLOCKCALL(gdr_internal_pin_buffer(g, addr, size, p2p_token, va_space, handle));
}

int unpin_buffer(gdr_t g, gdr_mh_t handle)
{
    CHECK_INITIALIZED();
    return GDRLOCKCALL(gdr_internal_unpin_buffer(g, handle));
}

int get_info(gdr_t g, gdr_mh_t handle, gdr_info_t* info)
{
    CHECK_INITIALIZED();
    return GDRLOCKCALL(gdr_internal_get_info(g, handle, info));
}

int map(gdr_t g, gdr_mh_t handle, void** va, size_t size)
{
    CHECK_INITIALIZED();
    return GDRLOCKCALL(gdr_internal_map(g, handle, va, size));
}

int unmap(gdr_t g, gdr_mh_t handle, void* va, size_t size)
{
    CHECK_INITIALIZED();
    return GDRLOCKCALL(gdr_internal_unmap(g, handle, va, size));
}

void runtime_get_version(int* major, int* minor)
{
    CHECK_INITIALIZED();
    gdr_internal_runtime_get_version(major, minor);
}

void driver_get_version(gdr_t g, int* major, int* minor)
{
    CHECK_INITIALIZED();
    gdr_internal_driver_get_version(g, major, minor);
}

int copy_to_mapping(gdr_mh_t handle, void* map_d_ptr, void const* h_ptr, size_t size)
{
    CHECK_INITIALIZED();
    return gdr_internal_copy_to_mapping(handle, map_d_ptr, h_ptr, size);
}

int copy_from_mapping(gdr_mh_t handle, void* h_ptr, void const* map_d_ptr, size_t size)
{
    CHECK_INITIALIZED();
    return gdr_internal_copy_from_mapping(handle, h_ptr, map_d_ptr, size);
}

void gdrCudaMalloc(void** ptr, void** devPtr, size_t mapSize, GdrMemDesc** memDesc, gdr_t handle)
{
    TLLM_CHECK_WITH_INFO(isInitialized(), "GDRCopy library is not initialized.");
    gdr_info_t info;
    gdr_mh_t mh;
    char* devMem;
    void* gdrMap;

    // GDRCOPY Pinned buffer has to be a minimum of a GPU_PAGE_SIZE
    size_t alignedMapSize = (mapSize + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    if (alignedMapSize == 0 && mapSize > 0)
    {
        alignedMapSize = GPU_PAGE_SIZE;
    }
    TLLM_CUDA_CHECK(cudaMalloc(&devMem, alignedMapSize + GPU_PAGE_SIZE - 1));
    uint64_t alignedAddr = ((uint64_t) devMem + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
    size_t align = alignedAddr - (uint64_t) devMem;

    TLLM_CHECK_WITH_INFO(pin_buffer(handle, alignedAddr, alignedMapSize, 0, 0, &mh) == 0, "GDR pin_buffer failed");
    TLLM_CHECK_WITH_INFO(map(handle, mh, &gdrMap, alignedMapSize) == 0, "GDR map failed");
    TLLM_CHECK_WITH_INFO(get_info(handle, mh, &info) == 0, "GDR get_info failed");

    ssize_t off = info.va - alignedAddr;

    *memDesc = new GdrMemDesc();
    (*memDesc)->gdrDeviceMem = devMem;
    (*memDesc)->gdrMap = gdrMap;
    (*memDesc)->gdrMapSize = alignedMapSize;
    (*memDesc)->gdrOffset = off + align;
    (*memDesc)->gdrMh = mh;

    *ptr = (void*) ((char*) gdrMap + off);
    if (devPtr)
        *devPtr = (void*) (devMem + off + align);

    TLLM_LOG_DEBUG("GDRCOPY : allocated devMem %p gdrMap %p offset %lx mh %lx mapSize %zu at %p",
        (*memDesc)->gdrDeviceMem, (*memDesc)->gdrMap, (*memDesc)->gdrOffset, (*memDesc)->gdrMh.h,
        (*memDesc)->gdrMapSize, *ptr);
}

void gdrCudaFree(GdrMemDesc* memDesc, gdr_t handle)
{
    CHECK_INITIALIZED();
    if (memDesc)
    {
        unmap(handle, memDesc->gdrMh, memDesc->gdrMap, memDesc->gdrMapSize);
        unpin_buffer(handle, memDesc->gdrMh);
        TLLM_CUDA_CHECK(cudaFree(memDesc->gdrDeviceMem));
        delete memDesc;
    }
}

} // namespace gdrcopy
} // namespace runtime
} // namespace tensorrt_llm

#endif // _WIN32
