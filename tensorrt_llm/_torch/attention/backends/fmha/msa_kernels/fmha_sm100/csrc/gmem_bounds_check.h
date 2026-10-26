// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once
// GMEM bounds-checking helpers for FMHA kernel debugging.
// Activated only when compiled with -DFMHA_GMEM_BOUNDS_CHECK.
// Usage: set env FMHA_GMEM_CHECK=1 before running Python → JIT adds the flag.

#ifdef FMHA_GMEM_BOUNDS_CHECK

#include <cstdio>
#include <cstdint>

struct GmemBounds {
  int packed_work_range_size;
  int packed_work_info_size;
  int qo_segment_offsets_size;
  int kv_segment_offsets_size;
  int segment_lens_size;
  int kv_page_indptr_size;
  int kv_indices_size;
  int max_score_numel;
  int64_t kv_block_indexes_span;
  int split_kv_size;
  int qo_offsets_size;
};

template <typename T>
__device__ __forceinline__ T gmem_load_checked(
    const T* base, int64_t idx, int64_t size, const char* name) {
  if (__builtin_expect(idx < 0 || idx >= size, 0)) {
    printf("GMEM-OOB-LOAD [%s] idx=%lld size=%lld blk=(%d,%d,%d) thr=%d\n",
           name, static_cast<long long>(idx), static_cast<long long>(size), blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
    return T(0);
  }
  return base[idx];
}

__device__ __forceinline__ void gmem_stwt_checked(
    float* addr, float val, const float* base, int numel, const char* name) {
  long long offset = (long long)(addr - base);
  if (__builtin_expect(offset < 0 || offset >= numel, 0)) {
    printf("GMEM-OOB-STWT [%s] offset=%lld numel=%d blk=(%d,%d,%d) thr=%d val=%f\n",
           name, offset, numel, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, val);
    return;
  }
  __stwt(addr, val);
}

__device__ __forceinline__ void gmem_store_f_checked(
    float* base, int idx, float val, int size, const char* name) {
  if (__builtin_expect(idx < 0 || idx >= size, 0)) {
    printf("GMEM-OOB-STORE [%s] idx=%d size=%d blk=(%d,%d,%d) thr=%d val=%f\n",
           name, idx, size, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, val);
    return;
  }
  base[idx] = val;
}

// Pointer-range check for cp.async / vec stores where we have (addr, base, numel)
__device__ __forceinline__ bool gmem_range_ok(
    const void* addr, int bytes, const void* base, int numel_bytes, const char* name) {
  long long off = (long long)((const char*)addr - (const char*)base);
  if (__builtin_expect(off < 0 || off + bytes > numel_bytes, 0)) {
    printf("GMEM-OOB-RANGE [%s] off=%lld bytes=%d buf_bytes=%d blk=(%d,%d,%d) thr=%d\n",
           name, off, bytes, numel_bytes, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
    return false;
  }
  return true;
}

#define GMEM_BOUNDS_FIELD GmemBounds gmem_bounds;

// Macros for split_kv buffer reads (used at 7+ sites in the kernel)
#define SPLIT_KV_BEGIN(p, wp) gmem_load_checked((p).kv_tile_begin_indices, (wp), (p).split_kv_buf_size, "kv_tile_begin")
#define SPLIT_KV_END(p, wp)   gmem_load_checked((p).kv_tile_end_indices,   (wp), (p).split_kv_buf_size, "kv_tile_end")
#define SPLIT_KV_SPLIT(p, wp) gmem_load_checked((p).kv_split_indices,      (wp), (p).split_kv_buf_size, "kv_split")

// Macros for segment_offsets reads
#define SEG_OFF_LOAD(vl, idx, sz) gmem_load_checked((vl).segment_offsets, (idx), (sz), "seg_offsets")
#define SEG_LEN_LOAD(vl, idx, sz) gmem_load_checked((vl).segment_lens,   (idx), (sz), "seg_lens")

// Macros for paged KV reads
#define KV_INDPTR_LOAD(ptr, idx, sz)  gmem_load_checked((ptr), (idx), (sz), "kv_page_indptr")
#define KV_INDICES_LOAD(ptr, idx, sz) gmem_load_checked((ptr), (idx), (sz), "kv_indices")
#define KV_BLOCK_INDEX_LOAD(ptr, idx, sz) gmem_load_checked((ptr), (idx), (sz), "kv_block_indexes")

#else  // FMHA_GMEM_BOUNDS_CHECK not defined

#define GMEM_BOUNDS_FIELD
#define SPLIT_KV_BEGIN(p, wp)  (p).kv_tile_begin_indices[(wp)]
#define SPLIT_KV_END(p, wp)    (p).kv_tile_end_indices[(wp)]
#define SPLIT_KV_SPLIT(p, wp)  (p).kv_split_indices[(wp)]
#define SEG_OFF_LOAD(vl, idx, sz)  (vl).segment_offsets[(idx)]
#define SEG_LEN_LOAD(vl, idx, sz)  (vl).segment_lens[(idx)]
#define KV_INDPTR_LOAD(ptr, idx, sz)  __ldg(&(ptr)[(idx)])
#define KV_INDICES_LOAD(ptr, idx, sz) __ldg(&(ptr)[(idx)])
#define KV_BLOCK_INDEX_LOAD(ptr, idx, sz) __ldg(&(ptr)[(idx)])

#endif  // FMHA_GMEM_BOUNDS_CHECK
