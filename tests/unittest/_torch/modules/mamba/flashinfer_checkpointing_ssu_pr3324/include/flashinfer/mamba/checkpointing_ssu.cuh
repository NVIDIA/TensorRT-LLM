/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_MAMBA_CHECKPOINTING_SSU_CUH_
#define FLASHINFER_MAMBA_CHECKPOINTING_SSU_CUH_

#include <cstdint>

namespace flashinfer::mamba::checkpointing {

struct CheckpointingSsuParams {
  uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{};
  uint32_t state_cache_size{};
  uint32_t npredicted{};
  uint32_t max_window{};
  int32_t pad_slot_id{-1};

  // v12 §59: per-head DIM split factor.  Must be one of {1, 2, 4}.  The host
  // launcher dispatches to a kernel template specialized on this value; the
  // kernel cross-checks via assert(params.d_split == D_SPLIT).
  int32_t d_split{1};

  bool dt_softplus{false};

  // Note: Programmatic Dependent Launch is JIT-stamped via the `ENABLE_PDL`
  // constexpr (see checkpointing_ssu_customize_config.jinja).  Each .so has
  // its PDL mode baked in; no runtime field needed.

  // ── Tensor pointers ──
  void* __restrict__ state{nullptr};    // (state_cache_size, nheads, dim, dstate)
  void* __restrict__ x{nullptr};        // (batch, NPREDICTED, nheads, dim)
  void* __restrict__ dt{nullptr};       // (batch, NPREDICTED, nheads, dim) tie_hdim
  void* __restrict__ A{nullptr};        // (nheads, dim, dstate) tie_hdim
  void* __restrict__ B{nullptr};        // (batch, NPREDICTED, ngroups, dstate)
  void* __restrict__ C{nullptr};        // (batch, NPREDICTED, ngroups, dstate)
  void* __restrict__ D{nullptr};        // (nheads, dim), optional
  void* __restrict__ z{nullptr};        // (batch, NPREDICTED, nheads, dim), optional
  void* __restrict__ dt_bias{nullptr};  // (nheads, dim) tie_hdim, optional
  void* __restrict__ output{nullptr};   // (batch, NPREDICTED, nheads, dim)

  // ── Cache tensors for incremental replay ──
  void* __restrict__ old_x{nullptr};  // (state_cache_size, MAX_WINDOW, nheads, dim) single-buffered
  void* __restrict__ old_B{
      nullptr};  // (state_cache_size, 2, MAX_WINDOW, ngroups, dstate) double-buffered
  void* __restrict__ old_dt{
      nullptr};  // (state_cache_size, 2, nheads, MAX_WINDOW) double-buffered, f32
  void* __restrict__ old_cumAdt{
      nullptr};  // (state_cache_size, 2, nheads, MAX_WINDOW) double-buffered, f32
  void* __restrict__ cache_buf_idx{nullptr};      // (state_cache_size,) int32
  void* __restrict__ prev_num_accepted{nullptr};  // (state_cache_size,) int32

  // ── Index tensors ──
  void* __restrict__ state_batch_indices{nullptr};  // (batch,) optional

  // ── Varlen (v20): packed inputs ──
  // When non-null, `x/dt/B/C/z/out` are laid out as
  // `(1, total_tokens, nheads, dim)` / `(1, total_tokens, ngroups, dstate)`
  // and `cu_seqlens[i]` gives the token-axis base of sequence i.
  // `seq_len_i = cu_seqlens[i+1] - cu_seqlens[i]`.  Kernel dispatch on
  // `cu_seqlens != nullptr` selects a `VARLEN=true` template.
  //
  // The `*_stride_seq` fields below already encode the outer iteration
  // stride for both modes — the wrapper sets them to:
  //   non-varlen:  `tensor.stride(0)`  (per-batch)
  //   varlen   :  `tensor.stride(1)`  (per-token, since sequences are packed
  //                                    into a single batch of total_tokens)
  // so the kernel uses one formula `seq * *_stride_seq` regardless of mode.
  void* __restrict__ cu_seqlens{nullptr};  // (batch+1,) int32, optional

  // ── Block-scale decode factors for quantized state ──
  void* __restrict__ state_scale{nullptr};  // float32: (state_cache_size, nheads, dim)

  // ── Philox PRNG seed for stochastic rounding ──
  const int64_t* rand_seed{nullptr};

  // ── Strides ──
  // state: (state_cache_size, nheads, dim, dstate) — inner 3 dims contiguous
  int64_t state_stride_seq{};

  // For the six batch-side tensors (x, dt, B, C, out, z), `*_stride_seq`
  // is the outer iteration stride — per-batch in non-varlen, per-token in
  // varlen.  `*_stride_token` is the inner per-row (T-axis) stride, same
  // in both modes.

  // x: (batch, NPREDICTED, nheads, dim) [non-varlen] / (1, total_tokens, nheads, dim) [varlen]
  int64_t x_stride_seq{};
  int64_t x_stride_token{};

  // dt: (batch, NPREDICTED, nheads, dim) — tie_hdim (stride_dim=0)
  int64_t dt_stride_seq{};
  int64_t dt_stride_token{};

  // B: (batch, NPREDICTED, ngroups, dstate)
  int64_t B_stride_seq{};
  int64_t B_stride_token{};

  // C: (batch, NPREDICTED, ngroups, dstate)
  int64_t C_stride_seq{};
  int64_t C_stride_token{};

  // output: (batch, NPREDICTED, nheads, dim)
  int64_t out_stride_seq{};
  int64_t out_stride_token{};

  // z: (batch, NPREDICTED, nheads, dim)
  int64_t z_stride_seq{};
  int64_t z_stride_token{};

  // old_x: (state_cache_size, MAX_WINDOW, nheads, dim) — single-buffered
  int64_t old_x_stride_seq{};
  int64_t old_x_stride_token{};

  // old_B: (state_cache_size, 2, MAX_WINDOW, ngroups, dstate) — double-buffered
  int64_t old_B_stride_seq{};
  int64_t old_B_stride_dbuf{};
  int64_t old_B_stride_token{};

  // old_dt: (state_cache_size, 2, nheads, MAX_WINDOW) — double-buffered, MAX_WINDOW contiguous
  int64_t old_dt_stride_seq{};
  int64_t old_dt_stride_dbuf{};
  int64_t old_dt_stride_head{};

  // old_cumAdt: (state_cache_size, 2, nheads, MAX_WINDOW) — double-buffered, MAX_WINDOW contiguous
  int64_t old_cumAdt_stride_seq{};
  int64_t old_cumAdt_stride_dbuf{};
  int64_t old_cumAdt_stride_head{};

  // state_scale: (state_cache_size, nheads, dim)
  int64_t state_scale_stride_seq{};
};

// Forward declaration — defined in kernel_checkpointing_ssu.cuh.
// `launchCheckpointingSsu` is the public dispatcher: it reads
// `params.d_split` and routes to the matching `launchCheckpointingSsuImpl`
// specialization (v12 §59).  Caller side stays single-entry.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void launchCheckpointingSsu(CheckpointingSsuParams& params, cudaStream_t stream);

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_CHECKPOINTING_SSU_CUH_
