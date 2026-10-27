// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#include "prefill_attention_api.hpp"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "sm100/device/prefill_attention.hpp"

namespace fmha_sm100::prefill_bf16x {
namespace {

// FMHA_SM100_PREFILL_BF16X_DTYPE (defaulted in prefill_attention.hpp): 1 builds the BF16 data
// path, whose forward folds the K global scale into the K dequant and applies neither the V
// global scale nor an output scale (the combine applies them).
constexpr bool kBf16DataPath = FMHA_SM100_PREFILL_BF16X_DTYPE == 1;
static_assert(!kBf16DataPath || FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT == 0,
              "the BF16 data path has no block-scale staging");

constexpr int kQHeadsPerKv = 16;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
// The split index lives in the top 8 bits of each packed qsplit entry.
constexpr int64_t kMaxTopK = 255;
constexpr int64_t kCacheAlignment = 16;

void check_cuda_contiguous(torch::Tensor const &tensor, char const *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(torch::Tensor const &reference, torch::Tensor const &tensor,
                       char const *name) {
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as q");
}

int checked_int_dimension(int64_t value, char const *name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), name, " must fit int32");
  return static_cast<int>(value);
}

void check_cuda(torch::Tensor const &tensor, char const *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

float const *global_scale_ptr(c10::optional<torch::Tensor> const &scale, torch::Tensor const &q,
                              char const *name) {
  if (!scale.has_value()) {
    return nullptr;
  }
  TORCH_CHECK(scale->is_cuda() && scale->device() == q.device(), name,
              " must be a CUDA tensor on the device of q");
  TORCH_CHECK(scale->scalar_type() == at::kFloat && scale->numel() == 1, name,
              " must be a one-element float32 tensor");
  return scale->data_ptr<float>();
}

// A [pages, num_kv_heads, 128, row_bytes] view with contiguous token rows inside each head and
// free head/page strides that keep TMA's 16-byte alignment.
CacheView make_cache_view(torch::Tensor const &tensor, char const *name, int64_t pages,
                          int num_kv_heads, int row_bytes) {
  check_cuda(tensor, name);
  TORCH_CHECK(tensor.dim() == 4 && tensor.size(0) == pages && tensor.size(1) == num_kv_heads &&
                  tensor.size(2) == kPageSize && tensor.size(3) == row_bytes,
              name, " must have shape [", pages, ", ", num_kv_heads, ", 128, ", row_bytes, "]");
  TORCH_CHECK(tensor.element_size() == 1, name, " must have a one-byte dtype");
  TORCH_CHECK(tensor.stride(3) == 1 && tensor.stride(2) == row_bytes, name,
              " must keep the 128 token rows of each head contiguous");
  TORCH_CHECK(tensor.stride(1) % kCacheAlignment == 0 && tensor.stride(0) % kCacheAlignment == 0,
              name, " head and page strides must be multiples of 16 bytes");
  TORCH_CHECK(tensor.stride(1) >= kPageSize * row_bytes && tensor.stride(0) > 0, name,
              " head blocks must not overlap");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % kCacheAlignment == 0, name,
              " must be 16-byte aligned");
  return CacheView{reinterpret_cast<uint8_t const *>(tensor.data_ptr()), tensor.stride(1),
                   tensor.stride(0)};
}

} // namespace

void prefill_run(torch::Tensor q, torch::Tensor packed_k, torch::Tensor packed_v,
                 torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor kv_indices,
                 c10::optional<torch::Tensor> kv_indptr, torch::Tensor cu_seqlens_q,
                 torch::Tensor cu_seqlens_k,
                 c10::optional<torch::Tensor> seqused_k, torch::Tensor k2q_row_ptr, torch::Tensor qsplit_indices,
                 torch::Tensor scheduler_metadata, torch::Tensor work_count,
                 torch::Tensor o_partial, torch::Tensor lse_partial,
                 c10::optional<torch::Tensor> k_global_scale,
                 c10::optional<torch::Tensor> v_global_scale, double softmax_scale,
                 double output_scale, c10::optional<torch::Tensor> combine_tile_counters) {
  torch::Tensor tensors[] = {
      q,          cu_seqlens_q, cu_seqlens_k, k2q_row_ptr, qsplit_indices, scheduler_metadata,
      work_count, lse_partial,
  };
  char const *names[] = {
      "q",          "cu_seqlens_q", "cu_seqlens_k", "k2q_row_ptr", "qsplit_indices",
      "scheduler_metadata", "work_count", "lse_partial",
  };
  constexpr std::size_t kTensorCount = sizeof(tensors) / sizeof(tensors[0]);
  static_assert(kTensorCount == sizeof(names) / sizeof(names[0]));
  for (std::size_t i = 0; i < kTensorCount; ++i) {
    check_cuda_contiguous(tensors[i], names[i]);
    check_same_device(q, tensors[i], names[i]);
  }
  for (auto const &[tensor, name] : {std::pair{packed_k, "packed_k"}, std::pair{packed_v, "packed_v"},
                                     std::pair{k_scale, "k_scale"}, std::pair{v_scale, "v_scale"},
                                     std::pair{o_partial, "o_partial"}}) {
    check_cuda(tensor, name);
    check_same_device(q, tensor, name);
  }

  if constexpr (kBf16DataPath) {
    TORCH_CHECK(q.scalar_type() == at::kBFloat16,
                "q must have dtype torch.bfloat16 (BF16 data-path extension)");
    TORCH_CHECK(!v_global_scale.has_value(),
                "the BF16 data path does not apply v_global_scale in the forward; the combine "
                "applies it");
    TORCH_CHECK(output_scale == 1.0, "the BF16 data path takes no forward output_scale");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(q.data_ptr()) % kCacheAlignment == 0,
                "q must be 16-byte aligned (TMA)");
  } else {
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "q must have dtype torch.float8_e4m3fn");
  }
  TORCH_CHECK(packed_k.scalar_type() == at::kByte && packed_v.scalar_type() == at::kByte,
              "packed K/V must have dtype torch.uint8");
  for (auto const &scale : {k_scale, v_scale}) {
    TORCH_CHECK(scale.scalar_type() == at::kFloat8_e4m3fn || scale.scalar_type() == at::kByte,
                "K/V scales must have dtype torch.float8_e4m3fn or torch.uint8 (E4M3 bits)");
  }
  for (auto const &tensor : {kv_indices, cu_seqlens_q, cu_seqlens_k, k2q_row_ptr,
                             qsplit_indices, scheduler_metadata, work_count}) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
                "scheduler and page metadata must have dtype torch.int32");
  }
  TORCH_CHECK(o_partial.scalar_type() == at::kBFloat16, "o_partial must have dtype torch.bfloat16");
  TORCH_CHECK(lse_partial.scalar_type() == at::kFloat, "lse_partial must have dtype torch.float32");

  TORCH_CHECK(q.dim() == 3 && q.size(1) > 0 && q.size(2) == kHeadDim,
              "q must have shape [total_q, num_q_heads, 128]");
  int const num_q_heads = checked_int_dimension(q.size(1), "num_q_heads");
  int64_t const total_q_64 = q.size(0);
  int const total_q = checked_int_dimension(total_q_64, "total_q");
  TORCH_CHECK(packed_k.dim() == 4 && packed_k.size(1) > 0,
              "packed_k must have shape [pages, num_kv_heads, 128, 64]");
  int const num_kv_heads = checked_int_dimension(packed_k.size(1), "num_kv_heads");
  TORCH_CHECK(num_q_heads == static_cast<int64_t>(num_kv_heads) * kQHeadsPerKv,
              "BF16X prefill requires exactly 16 Q heads per KV head");
  int64_t const pages = packed_k.size(0);
  PrefillArguments arguments{};
  arguments.packed_k = make_cache_view(packed_k, "packed_k", pages, num_kv_heads, kHeadDim / 2);
  arguments.packed_v = make_cache_view(packed_v, "packed_v", pages, num_kv_heads, kHeadDim / 2);
  arguments.k_scale = make_cache_view(k_scale, "k_scale", pages, num_kv_heads, kHeadDim / 16);
  arguments.v_scale = make_cache_view(v_scale, "v_scale", pages, num_kv_heads, kHeadDim / 16);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2,
              "cu_seqlens_q must have shape [batch + 1]");
  TORCH_CHECK(cu_seqlens_k.sizes() == cu_seqlens_q.sizes(), "cu_seqlens_k must match cu_seqlens_q");
  check_cuda(kv_indices, "kv_indices");
  check_same_device(q, kv_indices, "kv_indices");
  int64_t const batch = cu_seqlens_q.numel() - 1;
  if (kv_indptr.has_value()) {
    check_cuda_contiguous(*kv_indptr, "kv_indptr");
    check_same_device(q, *kv_indptr, "kv_indptr");
    TORCH_CHECK(kv_indptr->scalar_type() == at::kInt && kv_indptr->sizes() == cu_seqlens_q.sizes(),
                "kv_indptr must be int32 with shape [batch + 1] like cu_seqlens_q");
    TORCH_CHECK(kv_indices.dim() == 1 && kv_indices.numel() > 0 && kv_indices.is_contiguous(),
                "with kv_indptr, kv_indices must be a contiguous flat [total_pages] list");
  } else {
    // A [batch, max_pages] table; rows may be views into a wider buffer (vLLM's block table).
    TORCH_CHECK(kv_indices.dim() == 2 && kv_indices.size(0) == batch && kv_indices.size(1) > 0,
                "without kv_indptr, kv_indices must be a [batch, max_pages] page table");
    TORCH_CHECK(kv_indices.stride(1) == 1 &&
                    (batch == 1 || kv_indices.stride(0) >= kv_indices.size(1)),
                "the page table must have contiguous rows");
  }
  TORCH_CHECK(k2q_row_ptr.dim() == 2 && k2q_row_ptr.size(0) == num_kv_heads &&
                  k2q_row_ptr.size(1) >= 1,
              "k2q_row_ptr must have shape [num_kv_heads, total_rows + 1]");
  TORCH_CHECK(qsplit_indices.dim() == 2 && qsplit_indices.size(0) == num_kv_heads,
              "qsplit_indices must have shape [num_kv_heads, nnz_capacity]");
  TORCH_CHECK(scheduler_metadata.dim() == 2 && scheduler_metadata.size(1) == 6,
              "scheduler_metadata must have shape [work_capacity, 6]");
  TORCH_CHECK(work_count.dim() == 1 && work_count.numel() == 1, "work_count must have shape [1]");
  // One split per TopK slot; the schedule's split indices stay below the list width.
  TORCH_CHECK(o_partial.dim() == 4 && o_partial.size(0) >= 1 && o_partial.size(0) <= kMaxTopK &&
                  o_partial.size(1) == total_q_64 && o_partial.size(2) == num_q_heads &&
                  o_partial.size(3) == kHeadDim,
              "o_partial must have shape [topk, total_q, num_q_heads, 128] with topk <= 255");
  // The rows of a split slab are contiguous; the slabs themselves may be padded apart, which
  // the locality-domain striping of the workspace relies on.
  int64_t const slab_elements = total_q_64 * num_q_heads * kHeadDim;
  TORCH_CHECK(o_partial.stride(3) == 1 && o_partial.stride(2) == kHeadDim &&
                  o_partial.stride(1) == static_cast<int64_t>(num_q_heads) * kHeadDim,
              "o_partial rows must be contiguous inside each split slab");
  TORCH_CHECK(o_partial.stride(0) >= slab_elements && o_partial.stride(0) % 8 == 0,
              "o_partial split stride must cover a slab and keep 16-byte alignment");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(o_partial.data_ptr()) % kCacheAlignment == 0,
              "o_partial must be 16-byte aligned");
  TORCH_CHECK(lse_partial.dim() == 3 && lse_partial.size(0) == o_partial.size(0) &&
                  lse_partial.size(1) == total_q_64 && lse_partial.size(2) == num_q_heads,
              "lse_partial must have shape [topk, total_q, num_q_heads]");
  if (seqused_k.has_value()) {
    check_cuda_contiguous(*seqused_k, "seqused_k");
    check_same_device(q, *seqused_k, "seqused_k");
    TORCH_CHECK(seqused_k->scalar_type() == at::kInt && seqused_k->dim() == 1 &&
                    seqused_k->numel() == cu_seqlens_q.numel() - 1,
                "seqused_k must be int32 with shape [batch]");
  }
  TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
              "softmax_scale must be finite and positive");
  TORCH_CHECK(std::isfinite(output_scale), "output_scale must be finite");

  c10::cuda::CUDAGuard const device_guard(q.device());
  int major = 0;
  int minor = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device()));
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.get_device()));
  TORCH_CHECK(major == 10 && (minor == 0 || minor == 3 || minor == 7),
              "Q8KV4 sparse prefill requires an SM100, SM103 or SM107 GPU");

  arguments.q_ptr = reinterpret_cast<uint8_t const *>(q.data_ptr());
  arguments.kv_indices_ptr = kv_indices.data_ptr<int32_t>();
  if (kv_indptr.has_value()) {
    arguments.kv_indptr_ptr = kv_indptr->data_ptr<int32_t>();
  } else {
    arguments.page_table_stride = checked_int_dimension(kv_indices.stride(0), "page table stride");
    arguments.page_table_width = checked_int_dimension(kv_indices.size(1), "max_pages");
  }
  arguments.cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int32_t>();
  arguments.cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int32_t>();
  arguments.seqused_k_ptr = seqused_k.has_value() ? seqused_k->data_ptr<int32_t>() : nullptr;
  arguments.k2q_row_ptr = k2q_row_ptr.data_ptr<int32_t>();
  arguments.qsplit_indices_ptr = qsplit_indices.data_ptr<int32_t>();
  arguments.scheduler_metadata_ptr = scheduler_metadata.data_ptr<int32_t>();
  arguments.work_count_ptr = work_count.data_ptr<int32_t>();
  arguments.o_partial_ptr =
      reinterpret_cast<cutlass::bfloat16_t *>(o_partial.data_ptr<at::BFloat16>());
  arguments.o_partial_split_stride = o_partial.stride(0);
  arguments.lse_partial_ptr = lse_partial.data_ptr<float>();
  arguments.total_q = total_q;
  arguments.num_q_heads = num_q_heads;
  arguments.num_kv_heads = num_kv_heads;
  arguments.physical_pages = checked_int_dimension(pages, "physical_pages");
  arguments.total_rows = checked_int_dimension(k2q_row_ptr.size(1) - 1, "total_rows");
  arguments.qsplit_stride = checked_int_dimension(qsplit_indices.size(1), "qsplit_stride");
  arguments.work_capacity = checked_int_dimension(scheduler_metadata.size(0), "work_capacity");
  arguments.k_global_scale_ptr = global_scale_ptr(k_global_scale, q, "k_global_scale");
  arguments.v_global_scale_ptr = global_scale_ptr(v_global_scale, q, "v_global_scale");
  if (combine_tile_counters.has_value()) {
    check_cuda_contiguous(*combine_tile_counters, "combine_tile_counters");
    check_same_device(q, *combine_tile_counters, "combine_tile_counters");
    TORCH_CHECK(combine_tile_counters->scalar_type() == at::kInt &&
                    combine_tile_counters->dim() == 1 && combine_tile_counters->numel() >= 1,
                "combine_tile_counters must be a one-dimensional int32 tensor");
    arguments.combine_tile_counters_ptr = combine_tile_counters->data_ptr<int32_t>();
    arguments.combine_tile_counter_count =
        checked_int_dimension(combine_tile_counters->numel(), "combine_tile_counters");
  }
  // The dequant stages every block scale by 2^-shift; scores and outputs get 2^shift back.
  double const stage_gain = static_cast<double>(1 << FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT);
#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1 || FMHA_SM100_PREFILL_BF16X_E4M3X == 1
  // BF16X M3 / E4M3X: the reference forms Float32(softmax_scale) * Float32(log2 e) (an fp32 product; shift is 0 on this path).
  {
    volatile float const scale32 = static_cast<float>(softmax_scale);
    volatile float const log2e32 = static_cast<float>(1.4426950408889634074);
    arguments.softmax_scale_log2 = scale32 * log2e32;
  }
#else
  arguments.softmax_scale_log2 =
      static_cast<float>(softmax_scale * stage_gain * 1.4426950408889634074);
#endif
  arguments.output_scale = static_cast<float>(output_scale * stage_gain);

  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaError_t const status = launch_prefill_attention(arguments, stream);
  TORCH_CHECK(status == cudaSuccess,
              "Q8KV4 sparse prefill launch failed: ", cudaGetErrorString(status));
}

int64_t block_scale_shift() { return FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT; }

int64_t data_path() { return FMHA_SM100_PREFILL_BF16X_DTYPE; }

} // namespace fmha_sm100::prefill_bf16x
