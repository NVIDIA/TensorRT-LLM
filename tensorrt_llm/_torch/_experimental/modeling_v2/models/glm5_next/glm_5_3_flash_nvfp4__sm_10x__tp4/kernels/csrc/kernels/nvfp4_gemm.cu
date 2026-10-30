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
// NVFP4 GEMMs (dense and grouped over experts) and the BF16 grouped GEMM on SM100 via CUTLASS
// (see nvfp4.h). Grouped arguments are filled on device from the group offsets, so a launch does
// not depend on the host and is CUDA-graph capturable.
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "nvfp4.h"

namespace glm {
namespace k {

using namespace cute;

namespace {

using Fp4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// ------------------------------------------------------------------ dense NVFP4
template <int TileM, int TileN>
struct Fp4DenseCfg {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;  // B^T [K, N] col-major == B [N, K] row-major
  using LayoutD = cutlass::layout::RowMajor;
  using ElementD = cutlass::bfloat16_t;
  static constexpr int AlignA = 32, AlignB = 32, AlignD = 8;
  using Acc = float;
  using MmaTile = Shape<Int<TileM>, Int<TileN>, _256>;
  using Cluster = Shape<_1, _1, _1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc, void, LayoutD, AlignD, ElementD,
      LayoutD, AlignD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, Fp4, LayoutA, AlignA, Fp4,
      LayoutB, AlignB, Acc, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename Cfg>
void run_fp4_dense(const uint8_t* A, const uint8_t* sfa, const uint8_t* B, const uint8_t* sfb,
                   bf16* D, int ldd, int M, int N, int K, float alpha, void* workspace,
                   size_t ws_bytes, cudaStream_t st) {
  using Gemm = typename Cfg::Gemm;
  using Kern = typename Gemm::GemmKernel;
  using BlkCfg = typename Kern::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using StrideA = typename Kern::StrideA;
  using StrideB = typename Kern::StrideB;
  using StrideD = typename Kern::StrideD;
  using EA = typename Gemm::ElementA;
  using EB = typename Gemm::ElementB;
  using SF = typename Kern::ElementSF;
  auto sA = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
  cute::get<0>(sD) = ldd;
  auto lSFA = BlkCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lSFB = BlkCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const EA*>(A), sA, reinterpret_cast<const EB*>(B), sB,
       reinterpret_cast<const SF*>(sfa), lSFA, reinterpret_cast<const SF*>(sfb), lSFB},
      {{}, nullptr, sD, reinterpret_cast<typename Cfg::ElementD*>(D), sD}};
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.beta = 0.f;
  Gemm gemm;
  auto status = gemm.can_implement(args);
  CHECK(status == cutlass::Status::kSuccess, "nvfp4 gemm cannot implement M=" +
                                                 std::to_string(M) + " N=" + std::to_string(N) +
                                                 " K=" + std::to_string(K));
  CHECK(Gemm::get_workspace_size(args) <= ws_bytes, "nvfp4 gemm workspace too small");
  status = gemm.initialize(args, workspace, st);
  CHECK(status == cutlass::Status::kSuccess, "nvfp4 gemm init failed");
  status = gemm.run(st);
  CHECK(status == cutlass::Status::kSuccess, "nvfp4 gemm run failed");
}

// ------------------------------------------------------------------ grouped NVFP4
// TwoSm: 256 x 256 tiles over an SM pair (the M = 256 token rows of an expert's tile split over the
// two SMs, the weight tile shared), so a prefill expert's ~230 rows take one M tile and every weight
// tile is streamed once instead of once per 128-row tile
template <bool TwoSm, int TileN = 256>
struct Fp4GroupedCfgT {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ElementD = cutlass::bfloat16_t;
  static constexpr int AlignA = 32, AlignB = 32, AlignD = 8;
  using Acc = float;
  using MmaTile = std::conditional_t<TwoSm, Shape<_256, Int<TileN>, _256>, Shape<_128, Int<TileN>, _256>>;
  using Cluster = std::conditional_t<TwoSm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;
  using ESched = std::conditional_t<TwoSm, cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
                                    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>;
  using KSched = std::conditional_t<TwoSm, cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100,
                                    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc, void, LayoutD*, AlignD,
      ElementD, LayoutD*, AlignD, ESched>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, Fp4, LayoutA*, AlignA, Fp4,
      LayoutB*, AlignB, Acc, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epi::SharedStorage))>,
      KSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using EA = typename Gemm::ElementA;
  using EB = typename Gemm::ElementB;
  using SF = typename Kernel::ElementSF;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
  using LayoutSFA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
  using BlkCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using UShape = typename ProblemShape::UnderlyingProblemShape;
};

template <typename C>
__global__ void fp4_grouped_setup(typename C::UShape* shapes, const typename C::EA** pA,
                                  const typename C::EB** pB, const typename C::SF** pSFA,
                                  const typename C::SF** pSFB, typename C::ElementD** pD,
                                  typename C::LayoutSFA* lSFA, const float** pAlpha,
                                  const int* offsets, const int* pad, const uint8_t* A,
                                  const uint8_t* sfa, const uint8_t* W, const uint8_t* sfw,
                                  const float* alpha, bf16* D, int G, int N, int K) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) return;
  const int off = offsets[g];
  const int m = offsets[g + 1] - off;
  shapes[g] = typename C::UShape{m, N, K};
  pA[g] = reinterpret_cast<const typename C::EA*>(A + (size_t)off * (K / 2));
  pSFA[g] = reinterpret_cast<const typename C::SF*>(sfa + (size_t)pad[g] * (K / 16));
  pB[g] = reinterpret_cast<const typename C::EB*>(W + (size_t)g * N * (K / 2));
  pSFB[g] = reinterpret_cast<const typename C::SF*>(sfw + (size_t)g * nvfp4_sf_bytes(N, K));
  pD[g] = reinterpret_cast<typename C::ElementD*>(D + (size_t)off * N);
  lSFA[g] = C::BlkCfg::tile_atom_to_shape_SFA(make_shape(m, N, K, 1));
  pAlpha[g] = alpha + g;
}

template <typename T>
void dmalloc(T*& p, size_t n) {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(n, 1) * sizeof(T)));
}

}  // namespace

void nvfp4_gemm(const uint8_t* A, const uint8_t* sfa, const uint8_t* B, const uint8_t* sfb,
                bf16* D, int ldd, int M, int N, int K, float alpha, void* workspace,
                size_t ws_bytes, cudaStream_t st) {
  if (M == 0) return;
  CHECK(K % 256 == 0 && N % 128 == 0, "nvfp4_gemm: K % 256, N % 128");
  run_fp4_dense<Fp4DenseCfg<128, 128>>(A, sfa, B, sfb, D, ldd, M, N, K, alpha, workspace,
                                       ws_bytes, st);
}

template <typename C>
struct Fp4GroupedImpl {
  int G, N, K;
  typename C::UShape* shapes = nullptr;
  const typename C::EA** pA = nullptr;
  const typename C::EB** pB = nullptr;
  const typename C::SF** pSFA = nullptr;
  const typename C::SF** pSFB = nullptr;
  typename C::ElementD** pD = nullptr;
  typename C::StrideA* sA = nullptr;
  typename C::StrideB* sB = nullptr;
  typename C::StrideD* sD = nullptr;
  typename C::LayoutSFA* lSFA = nullptr;
  typename C::LayoutSFB* lSFB = nullptr;
  const float** pAlpha = nullptr;
  void* workspace = nullptr;
  typename C::Gemm gemm;

  Fp4GroupedImpl(int G_, int N_, int K_, int device) : G(G_), N(N_), K(K_) {
    dmalloc(shapes, G);
    dmalloc(pA, G);
    dmalloc(pB, G);
    dmalloc(pSFA, G);
    dmalloc(pSFB, G);
    dmalloc(pD, G);
    dmalloc(sA, G);
    dmalloc(sB, G);
    dmalloc(sD, G);
    dmalloc(lSFA, G);
    dmalloc(lSFB, G);
    dmalloc(pAlpha, G);
    std::vector<typename C::StrideA> hA(G, cutlass::make_cute_packed_stride(typename C::StrideA{}, make_shape(1, K, 1)));
    std::vector<typename C::StrideB> hB(G, cutlass::make_cute_packed_stride(typename C::StrideB{}, make_shape(N, K, 1)));
    std::vector<typename C::StrideD> hD(G, cutlass::make_cute_packed_stride(typename C::StrideD{}, make_shape(1, N, 1)));
    std::vector<typename C::LayoutSFB> hSFB(G, C::BlkCfg::tile_atom_to_shape_SFB(make_shape(1, N, K, 1)));
    CUDA_CHECK(cudaMemcpy(sA, hA.data(), G * sizeof(hA[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sB, hB.data(), G * sizeof(hB[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sD, hD.data(), G * sizeof(hD[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lSFB, hSFB.data(), G * sizeof(hSFB[0]), cudaMemcpyHostToDevice));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = device;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    typename C::Gemm::Arguments args;
    decltype(args.epilogue.thread) fusion;
    fusion.alpha_ptr = nullptr;
    fusion.beta_ptr = nullptr;
    fusion.alpha = 0.f;
    fusion.alpha_ptr_array = pAlpha;
    fusion.dAlpha = {_0{}, _0{}, 1};
    fusion.beta = 0.f;
    fusion.beta_ptr_array = nullptr;
    fusion.dBeta = {_0{}, _0{}, 0};
    args = typename C::Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                       {G, shapes, nullptr},
                                       {pA, sA, pB, sB, pSFA, lSFA, pSFB, lSFB},
                                       {fusion, nullptr, sD, pD, sD},
                                       hw};
    const size_t ws = C::Gemm::get_workspace_size(args);
    CUDA_CHECK(cudaMalloc(&workspace, std::max<size_t>(ws, 256)));
    auto status = gemm.initialize(args, workspace);
    CHECK(status == cutlass::Status::kSuccess, "nvfp4 grouped gemm init failed");
  }

  ~Fp4GroupedImpl() {
    for (void* p : {(void*)shapes, (void*)pA, (void*)pB, (void*)pSFA, (void*)pSFB, (void*)pD,
                    (void*)sA, (void*)sB, (void*)sD, (void*)lSFA, (void*)lSFB, (void*)pAlpha,
                    workspace})
      cudaFree(p);
  }

  void run(const uint8_t* A, const uint8_t* sfa, const uint8_t* W, const uint8_t* sfw,
           const float* alpha, bf16* D, const int* offsets, const int* pad, cudaStream_t st) {
    fp4_grouped_setup<C><<<cdiv(G, 128), 128, 0, st>>>(shapes, pA, pB, pSFA, pSFB, pD, lSFA, pAlpha,
                                                      offsets, pad, A, sfa, W, sfw, alpha, D, G, N, K);
    auto status = gemm.run(st);
    CHECK(status == cutlass::Status::kSuccess, "nvfp4 grouped gemm run failed");
  }
};

// Grouped NVFP4 GEMM tiles, by the expected rows per expert (rows_hint): prefill-sized steps (~ T x 8 / 288 rows per
// expert) take 2-SM 256 x 256 tiles (an expert's rows in one M tile: every weight tile streamed once); steps with at
// most kFp4SmallTileMaxRows expected rows per expert take 1-SM 128 x 128 tiles, which a 256-row tile would leave half
// empty.
constexpr int kFp4SmallTileMaxRows = 128;

struct Nvfp4GroupedGemm::Impl {
  std::unique_ptr<Fp4GroupedImpl<Fp4GroupedCfgT<true, 256>>> t256;   // 2-SM 256 x 256
  std::unique_ptr<Fp4GroupedImpl<Fp4GroupedCfgT<false, 128>>> t128;  // 1-SM 128 x 128
};

Nvfp4GroupedGemm::Nvfp4GroupedGemm(int groups, int N, int K, int device) : impl_(new Impl) {
  CHECK(N % 128 == 0 && K % 256 == 0, "nvfp4 grouped gemm: N % 128, K % 256");
  impl_->t256 = std::make_unique<Fp4GroupedImpl<Fp4GroupedCfgT<true, 256>>>(groups, N, K, device);
  impl_->t128 = std::make_unique<Fp4GroupedImpl<Fp4GroupedCfgT<false, 128>>>(groups, N, K, device);
}

Nvfp4GroupedGemm::~Nvfp4GroupedGemm() = default;

void Nvfp4GroupedGemm::run(const uint8_t* A, const uint8_t* sfa, const uint8_t* W,
                           const uint8_t* sfw, const float* alpha, bf16* D, const int* offsets,
                           const int* pad, cudaStream_t st, int rows_hint) {
  Impl& m = *impl_;
  if (rows_hint >= 0 && rows_hint <= kFp4SmallTileMaxRows)
    m.t128->run(A, sfa, W, sfw, alpha, D, offsets, pad, st);
  else
    m.t256->run(A, sfa, W, sfw, alpha, D, offsets, pad, st);
}

// ------------------------------------------------------------------ grouped BF16
namespace {
struct Bf16GroupedCfg {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignA = 8, AlignB = 8, AlignD = 8;
  using Acc = float;
  using MmaTile = Shape<_128, _128, _64>;
  using Cluster = Shape<_1, _1, _1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc, void, LayoutD*, AlignD,
      ElementD, LayoutD*, AlignD, cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA, LayoutA*, AlignA, ElementB,
      LayoutB*, AlignB, Acc, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
  using UShape = typename ProblemShape::UnderlyingProblemShape;
};

__global__ void bf16_grouped_setup(Bf16GroupedCfg::UShape* shapes,
                                   const Bf16GroupedCfg::ElementA** pA,
                                   const Bf16GroupedCfg::ElementB** pB,
                                   Bf16GroupedCfg::ElementD** pD, const int* offsets,
                                   const bf16* A, const bf16* W, bf16* D, int G, int N, int K) {
  using C = Bf16GroupedCfg;
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) return;
  const int off = offsets[g];
  shapes[g] = C::UShape{offsets[g + 1] - off, N, K};
  pA[g] = reinterpret_cast<const C::ElementA*>(A + (size_t)off * K);
  pB[g] = reinterpret_cast<const C::ElementB*>(W + (size_t)g * N * K);
  pD[g] = reinterpret_cast<C::ElementD*>(D + (size_t)off * N);
}
}  // namespace

struct Bf16GroupedGemm::Impl {
  using C = Bf16GroupedCfg;
  int G, N, K;
  C::UShape* shapes = nullptr;
  const C::ElementA** pA = nullptr;
  const C::ElementB** pB = nullptr;
  C::ElementD** pD = nullptr;
  C::StrideA* sA = nullptr;
  C::StrideB* sB = nullptr;
  C::StrideD* sD = nullptr;
  void* workspace = nullptr;
  C::Gemm gemm;

  Impl(int G_, int N_, int K_, int device) : G(G_), N(N_), K(K_) {
    dmalloc(shapes, G);
    dmalloc(pA, G);
    dmalloc(pB, G);
    dmalloc(pD, G);
    dmalloc(sA, G);
    dmalloc(sB, G);
    dmalloc(sD, G);
    std::vector<C::StrideA> hA(G, cutlass::make_cute_packed_stride(C::StrideA{}, make_shape(1, K, 1)));
    std::vector<C::StrideB> hB(G, cutlass::make_cute_packed_stride(C::StrideB{}, make_shape(N, K, 1)));
    std::vector<C::StrideD> hD(G, cutlass::make_cute_packed_stride(C::StrideD{}, make_shape(1, N, 1)));
    CUDA_CHECK(cudaMemcpy(sA, hA.data(), G * sizeof(hA[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sB, hB.data(), G * sizeof(hB[0]), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sD, hD.data(), G * sizeof(hD[0]), cudaMemcpyHostToDevice));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = device;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    typename C::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {G, shapes, nullptr},
                                     {pA, sA, pB, sB},
                                     {{}, nullptr, sD, pD, sD},
                                     hw};
    args.epilogue.thread.alpha = 1.f;
    args.epilogue.thread.beta = 0.f;
    const size_t ws = C::Gemm::get_workspace_size(args);
    CUDA_CHECK(cudaMalloc(&workspace, std::max<size_t>(ws, 256)));
    auto status = gemm.initialize(args, workspace);
    CHECK(status == cutlass::Status::kSuccess, "bf16 grouped gemm init failed");
  }

  ~Impl() {
    for (void* p : {(void*)shapes, (void*)pA, (void*)pB, (void*)pD, (void*)sA, (void*)sB,
                    (void*)sD, workspace})
      cudaFree(p);
  }
};

Bf16GroupedGemm::Bf16GroupedGemm(int groups, int N, int K, int device)
    : impl_(new Impl(groups, N, K, device)) {
  CHECK(N % 128 == 0 && K % 64 == 0, "bf16 grouped gemm: N % 128, K % 64");
}

Bf16GroupedGemm::~Bf16GroupedGemm() = default;

void Bf16GroupedGemm::run(const bf16* A, const bf16* W, bf16* D, const int* offsets,
                          cudaStream_t st) {
  Impl& m = *impl_;
  bf16_grouped_setup<<<cdiv(m.G, 128), 128, 0, st>>>(m.shapes, m.pA, m.pB, m.pD, offsets, A, W,
                                                      D, m.G, m.N, m.K);
  auto status = m.gemm.run(st);
  CHECK(status == cutlass::Status::kSuccess, "bf16 grouped gemm run failed");
}

}  // namespace k
}  // namespace glm
