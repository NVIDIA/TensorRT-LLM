/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ALLOCATOR_H_
#define FLASHINFER_ALLOCATOR_H_

#include <memory>
#include <sstream>

#include "exception.h"

namespace flashinfer {

// create a function that returns T* from base pointer and offset
template <typename T>
T* GetPtrFromBaseOffset(void* base_ptr, int64_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(base_ptr) + offset);
}

struct AlignedAllocator {
  void* base_ptr;
  void* cur_ptr;
  size_t remaining_space;
  AlignedAllocator(void* buf, size_t space) : base_ptr(buf), cur_ptr(buf), remaining_space(space) {}
  template <typename T>
  T* aligned_alloc(size_t size, size_t alignment, std::string name) {
    if (std::align(alignment, size, cur_ptr, remaining_space)) {
      T* result = reinterpret_cast<T*>(cur_ptr);
      cur_ptr = (char*)cur_ptr + size;
      remaining_space -= size;
      return result;
    } else {
      std::ostringstream oss;
      oss << "Buffer overflow when allocating memory for " << name << " with size " << size
          << " and alignment " << alignment << ", but only " << remaining_space
          << " bytes available in AlignedAllocator. Increase the workspace buffer size.";
      FLASHINFER_ERROR(oss.str());
    }
    return nullptr;
  }

  size_t aligned_alloc_offset(size_t size, size_t alignment, std::string name) {
    return (char*)aligned_alloc<char>(size, alignment, name) - (char*)base_ptr;
  }

  size_t num_allocated_bytes() { return (char*)cur_ptr - (char*)base_ptr; }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ALLOCATOR_H_
