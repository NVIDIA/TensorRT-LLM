// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

// Memory locality domains (Rubin uGPU, Blackwell's dies): the domain of every SM, and
// partial-output buffers whose pages alternate between the domains so a CTA can read its tile
// from the domain of its own SM. Devices with one domain report a count of 1 and the callers keep
// ordinary allocations.

#include "prefill_attention_api.hpp"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

// The locality-domain driver API ships with CUDA 13.4.
#if CUDA_VERSION >= 13040
#define FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS 1
#else
#define FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS 0
#endif

namespace fmha_sm100::prefill_bf16x {
namespace {

int checked_device(int64_t device) {
  int count = 0;
  C10_CUDA_CHECK(cudaGetDeviceCount(&count));
  TORCH_CHECK(device >= 0 && device < count, "device ordinal ", device, " out of range");
  return static_cast<int>(device);
}

#if FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS

void check_driver(CUresult result, char const *what) {
  if (result != CUDA_SUCCESS) {
    char const *message = nullptr;
    cuGetErrorString(result, &message);
    TORCH_CHECK(false, what, " failed: ", message ? message : "unknown CUDA driver error");
  }
}

CUdevice driver_device(int device) {
  check_driver(cuInit(0), "cuInit");
  CUdevice cu_device;
  check_driver(cuDeviceGet(&cu_device, device), "cuDeviceGet");
  return cu_device;
}

// cuDevSmResourceSplit is a CUDA 13.4 driver entry point. Resolving it at run time keeps this
// extension loadable on older drivers (CUDA 13.0 to 13.3), where locality-domain striping is simply
// unavailable; the green-context and virtual-memory calls below predate it.
decltype(&cuDevSmResourceSplit) sm_resource_split() {
  static decltype(&cuDevSmResourceSplit) const function = [] {
    void *address = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
    CUresult const result = cuGetProcAddress("cuDevSmResourceSplit", &address, 13040,
                                             CU_GET_PROC_ADDRESS_DEFAULT, &status);
    return result == CUDA_SUCCESS && status == CU_GET_PROC_ADDRESS_SUCCESS
               ? reinterpret_cast<decltype(&cuDevSmResourceSplit)>(address)
               : nullptr;
  }();
  return function;
}

int domain_count(int device) {
  if (sm_resource_split() == nullptr) {
    return 1;
  }
  int count = 0;
  CUresult const result = cuDeviceGetAttribute(
      &count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, driver_device(device));
  // Drivers without the attribute expose one domain.
  return result == CUDA_SUCCESS && count > 1 ? count : 1;
}

CUmemAllocationProp localized_prop(int device, int domain) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  prop.location.localized.deviceId = static_cast<unsigned char>(device);
  prop.location.localized.localityDomainId = static_cast<unsigned char>(domain);
  return prop;
}

size_t page_bytes(int device) {
  CUmemAllocationProp const prop = localized_prop(device, 0);
  size_t granularity = 0;
  check_driver(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
               "cuMemGetAllocationGranularity");
  return granularity;
}

// Every CTA records the SM it runs on. The shared-memory request admits one CTA per SM and the
// spin outlasts the dispatch of the whole grid, so the first wave covers every SM of the stream's
// green context.
__global__ void record_sm_ids(int32_t *sm_ids, int32_t *sm_id_count, long long spin_cycles) {
  if (threadIdx.x == 0) {
    unsigned int sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    sm_ids[blockIdx.x] = static_cast<int32_t>(sm_id);
    if (blockIdx.x == 0) {
      unsigned int count;
      asm volatile("mov.u32 %0, %%nsmid;" : "=r"(count));
      *sm_id_count = static_cast<int32_t>(count);
    }
    long long const start = clock64();
    while (clock64() - start < spin_cycles) {
    }
  }
}

// A green context over one locality domain's SMs and a stream on it.
struct DomainContext {
  CUgreenCtx context = nullptr;
  CUstream stream = nullptr;

  DomainContext(CUdevice device, CUdevResource const &resource) {
    CUdevResourceDesc descriptor;
    check_driver(cuDevResourceGenerateDesc(&descriptor, const_cast<CUdevResource *>(&resource), 1),
                 "cuDevResourceGenerateDesc");
    check_driver(cuGreenCtxCreate(&context, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM),
                 "cuGreenCtxCreate");
    check_driver(cuGreenCtxStreamCreate(&stream, context, CU_STREAM_NON_BLOCKING, 0),
                 "cuGreenCtxStreamCreate");
  }
  DomainContext(DomainContext const &) = delete;
  DomainContext &operator=(DomainContext const &) = delete;
  ~DomainContext() {
    if (stream != nullptr) {
      cuStreamDestroy(stream);
    }
    if (context != nullptr) {
      cuGreenCtxDestroy(context);
    }
  }
};

std::vector<CUdevResource> split_sms_by_domain(CUdevice cu_device, int num_domains) {
  CUdevResource all_sms{};
  check_driver(cuDeviceGetDevResource(cu_device, &all_sms, CU_DEV_RESOURCE_TYPE_SM),
               "cuDeviceGetDevResource");
  std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> group_params(num_domains);
  for (int domain = 0; domain < num_domains; ++domain) {
    group_params[domain] = CU_DEV_SM_RESOURCE_GROUP_PARAMS{};
    group_params[domain].coscheduledSmCount = 2;
    group_params[domain].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
    group_params[domain].localityDomainId = static_cast<unsigned int>(domain);
  }
  std::vector<CUdevResource> groups(num_domains);
  CUdevResource remainder{};
  check_driver(sm_resource_split()(groups.data(), static_cast<unsigned int>(num_domains), &all_sms,
                                   &remainder, 0, group_params.data()),
               "cuDevSmResourceSplit");
  return groups;
}

#endif // FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS

} // namespace

int64_t locality_domain_count(int64_t device) {
  int const index = checked_device(device);
#if FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS
  return domain_count(index);
#else
  (void)index;
  return 1;
#endif
}

int64_t locality_page_bytes(int64_t device) {
  int const index = checked_device(device);
#if FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS
  return static_cast<int64_t>(page_bytes(index));
#else
  (void)index;
  TORCH_CHECK(false, "locality domains need a CUDA 13.4 or newer toolkit");
#endif
}

torch::Tensor sm_locality_domains(int64_t device) {
  int const index = checked_device(device);
#if FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS
  c10::cuda::CUDAGuard const device_guard(index);
  CUdevice const cu_device = driver_device(index);
  int const num_domains = domain_count(index);

  std::vector<CUdevResource> const groups = split_sms_by_domain(cu_device, num_domains);

  int clock_khz = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, index));
  int smem_per_block = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, index));
  C10_CUDA_CHECK(cudaFuncSetAttribute(record_sm_ids, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_per_block));
  // Two waves of one CTA per SM; the spin lasts longer than dispatching them.
  long long const spin_cycles = static_cast<long long>(clock_khz) * 2;
  auto const options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, index);
  torch::Tensor const sm_id_count = torch::zeros({1}, options);

  torch::Tensor table;
  for (int domain = 0; domain < num_domains; ++domain) {
    unsigned int const sm_count = groups[domain].sm.smCount;
    if (sm_count == 0) {
      continue;
    }
    DomainContext const context(cu_device, groups[domain]);
    int const ctas = static_cast<int>(sm_count) * 2;
    torch::Tensor const sm_ids = torch::full({ctas}, -1, options);
    record_sm_ids<<<ctas, 32, smem_per_block, static_cast<cudaStream_t>(context.stream)>>>(
        sm_ids.data_ptr<int32_t>(), sm_id_count.data_ptr<int32_t>(), spin_cycles);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)));
    if (!table.defined()) {
      table = torch::full({sm_id_count.item<int32_t>()}, -1, torch::dtype(torch::kInt32));
    }
    auto const ids = sm_ids.cpu();
    auto const ids_accessor = ids.accessor<int32_t, 1>();
    auto table_accessor = table.accessor<int32_t, 1>();
    for (int cta = 0; cta < ctas; ++cta) {
      int32_t const sm_id = ids_accessor[cta];
      if (sm_id >= 0 && sm_id < table.size(0)) {
        table_accessor[sm_id] = domain;
      }
    }
  }
  TORCH_CHECK(table.defined(), "no locality domain exposes any SM");
  return table;
#else
  (void)index;
  TORCH_CHECK(false, "locality domains need a CUDA 13.4 or newer toolkit");
#endif
}

torch::Tensor alloc_locality_striped(int64_t bytes, int64_t device) {
  int const index = checked_device(device);
#if FMHA_SM100_PREFILL_BF16X_HAS_LOCALITY_DOMAINS
  c10::cuda::CUDAGuard const device_guard(index);
  int const num_domains = domain_count(index);
  size_t const page = page_bytes(index);
  TORCH_CHECK(bytes > 0 && static_cast<size_t>(bytes) % (page * num_domains) == 0,
              "a striped buffer must hold a multiple of ", num_domains, " x ", page, " bytes");
  struct Mapping {
    CUdeviceptr base = 0;
    size_t bytes = 0;
    std::vector<CUmemGenericAllocationHandle> handles;
  };
  auto release = [](Mapping const &mapping) {
    if (mapping.base != 0) {
      cuMemUnmap(mapping.base, mapping.bytes);
      cuMemAddressFree(mapping.base, mapping.bytes);
    }
    for (CUmemGenericAllocationHandle const handle : mapping.handles) {
      cuMemRelease(handle);
    }
  };

  Mapping mapping;
  mapping.bytes = static_cast<size_t>(bytes);
  try {
    check_driver(cuMemAddressReserve(&mapping.base, mapping.bytes, page, 0, 0),
                 "cuMemAddressReserve");
    // cuMemMap maps a handle from its start, so every page is its own allocation.
    size_t const pages = mapping.bytes / page;
    mapping.handles.reserve(pages);
    for (size_t p = 0; p < pages; ++p) {
      CUmemAllocationProp const prop = localized_prop(index, static_cast<int>(p % num_domains));
      CUmemGenericAllocationHandle handle;
      check_driver(cuMemCreate(&handle, page, &prop, 0), "cuMemCreate");
      mapping.handles.push_back(handle);
      check_driver(cuMemMap(mapping.base + p * page, page, 0, handle, 0), "cuMemMap");
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = index;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check_driver(cuMemSetAccess(mapping.base, mapping.bytes, &access, 1), "cuMemSetAccess");
  } catch (...) {
    release(mapping);
    throw;
  }

  // The last reference may drop while a kernel still reads the buffer (a workspace replaced by
  // a larger one, or its owner destroyed): finish the device's work before unmapping.
  auto deleter = [mapping, release](void *) {
    cudaDeviceSynchronize();
    release(mapping);
  };
  auto const options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, index);
  return torch::from_blob(reinterpret_cast<void *>(mapping.base), {bytes}, deleter, options);
#else
  (void)index;
  (void)bytes;
  TORCH_CHECK(false, "locality domains need a CUDA 13.4 or newer toolkit");
#endif
}

} // namespace fmha_sm100::prefill_bf16x
