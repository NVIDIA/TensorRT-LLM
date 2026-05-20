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
// clang-format off
// config.inc MUST come before the header: it defines DIM, DSTATE, NPREDICTED,
// MAX_WINDOW constexprs that the header's function templates rely on.
#include "checkpointing_ssu_config.inc"
#include <flashinfer/mamba/checkpointing_ssu.cuh>
#include <flashinfer/mamba/launch_checkpointing_ssu.cuh>
// clang-format on
#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

namespace flashinfer::mamba::checkpointing {

void checkpointing_ssu(
    TensorView state,  // (state_cache_size, nheads, dim, dstate)
    TensorView x,  // (batch, NPREDICTED, nheads, dim) / (1, total_tokens, nheads, dim) under varlen
    TensorView dt,  // (batch, NPREDICTED, nheads, dim) tie_hdim / (1, total_tokens, nheads, dim)
    TensorView A,   // (nheads, dim, dstate) tie_hdim
    TensorView B,   // (batch, NPREDICTED, ngroups, dstate) / (1, total_tokens, ngroups, dstate)
    TensorView C,   // same as B
    TensorView output,  // same layout as x
    // Cache tensors
    TensorView old_x,              // (state_cache_size, MAX_WINDOW, nheads, dim)
    TensorView old_B,              // (state_cache_size, 2, MAX_WINDOW, ngroups, dstate)
    TensorView old_dt,             // (state_cache_size, 2, nheads, MAX_WINDOW) f32
    TensorView old_cumAdt,         // (state_cache_size, 2, nheads, MAX_WINDOW) f32
    TensorView cache_buf_idx,      // (state_cache_size,) int32
    TensorView prev_num_accepted,  // (state_cache_size,) int32
    // Optional tensors
    Optional<TensorView> D,        // (nheads, dim)
    Optional<TensorView> z,        // same layout as x
    Optional<TensorView> dt_bias,  // (nheads, dim) tie_hdim
    bool dt_softplus,
    Optional<TensorView> state_batch_indices,  // (batch,) int32
    int64_t pad_slot_id,
    Optional<TensorView> state_scale,   // (state_cache_size, nheads, dim) f32
    Optional<TensorView> rand_seed,     // single int64
    int64_t d_split,                    // v12 §59: per-head DIM split factor (1, 2, or 4)
    Optional<TensorView> cu_seqlens) {  // (batch+1,) int32, varlen mode

  bool const is_varlen = cu_seqlens.has_value();

  // ── Extract dimensions ──
  auto const state_cache_size = state.size(0);
  auto const nheads = state.size(1);
  auto const dim = state.size(2);
  auto const dstate = state.size(3);
  auto const max_window = old_x.size(1);
  auto const ngroups = B.size(2);

  // In non-varlen mode, batch = x.size(0) and npredicted = x.size(1) (the
  // 4D batched layout).  In varlen, the JIT compile-time NPREDICTED is the
  // max seq_len the caller commits to; the wrapper stamped it into the JIT
  // URI from `max_seqlen` and we read it back as the constexpr `NPREDICTED`
  // (validated against runtime cu_seqlens on the host side below).  `batch`
  // = number of sequences = `cu_seqlens.size(0) - 1`.
  int64_t batch;
  int64_t npredicted;
  if (is_varlen) {
    auto const& cs = cu_seqlens.value();
    CHECK_CUDA(cs);
    CHECK_DIM(1, cs);
    FLASHINFER_CHECK(
        cs.size(0) >= 2,
        "cu_seqlens must have shape (batch+1,) with batch >= 1, got size(0)=", cs.size(0));
    FLASHINFER_CHECK(cs.dtype().code == kDLInt && cs.dtype().bits == 32,
                     "cu_seqlens must be int32");
    CHECK_CONTIGUOUS(cs);
    batch = cs.size(0) - 1;
    npredicted = NPREDICTED;  // JIT-stamped — wrapper ensures max(seq_lens) <= NPREDICTED.
  } else {
    batch = x.size(0);
    npredicted = x.size(1);
  }

  // ── JIT compile-time / runtime cross-check ──
  // NPREDICTED and MAX_WINDOW are JIT compile-time constants stamped by the
  // wrapper.  In non-varlen NPREDICTED = x.shape[1]; in varlen NPREDICTED =
  // user-supplied `max_seqlen` (upper bound on every cu_seqlens diff).
  FLASHINFER_CHECK(npredicted == NPREDICTED, is_varlen ? "max_seqlen=" : "x.size(1)=", npredicted,
                   " must equal JIT NPREDICTED=", NPREDICTED);
  FLASHINFER_CHECK(max_window == MAX_WINDOW, "old_x.size(1)=", max_window,
                   " must equal JIT MAX_WINDOW=", MAX_WINDOW);
  FLASHINFER_CHECK(npredicted <= max_window, "npredicted=", npredicted,
                   " must be <= max_window=", max_window);

  // ── Validate state ──
  CHECK_CUDA(state);
  CHECK_DIM(4, state);
  {
    auto s = state.strides();
    auto sz = state.sizes();
    FLASHINFER_CHECK(s[3] == 1, "state dim 3 (dstate) must have stride 1, got ", s[3]);
    FLASHINFER_CHECK(s[2] == sz[3], "state dim 2 (dim) must be contiguous with dim 3, got stride ",
                     s[2], " expected ", sz[3]);
    FLASHINFER_CHECK(s[1] == sz[2] * sz[3],
                     "state dim 1 (nheads) must be contiguous with dim 2, got stride ", s[1],
                     " expected ", sz[2] * sz[3]);
  }

  // ── Validate x ──
  // Non-varlen: shape (batch, NPREDICTED, nheads, dim).
  // Varlen    : shape (1, total_tokens, nheads, dim) — batch axis collapsed,
  //             token axis is the outer iteration.  The kernel reads x via
  //             `bos * x_stride_token + …` so x_stride_token is the per-token
  //             stride in either layout (= nheads*dim for contig).
  CHECK_CUDA(x);
  CHECK_DIM(4, x);
  if (is_varlen) {
    FLASHINFER_CHECK(x.size(0) == 1, "varlen: x.size(0)=", x.size(0), " must be 1");
  } else {
    FLASHINFER_CHECK(x.size(0) == batch, "x.size(0)=", x.size(0), " must equal batch=", batch);
    FLASHINFER_CHECK(x.size(1) == npredicted, "x.size(1)=", x.size(1),
                     " must equal npredicted=", npredicted);
  }
  FLASHINFER_CHECK(x.size(2) == nheads, "x.size(2)=", x.size(2), " must equal nheads=", nheads);
  FLASHINFER_CHECK(x.size(3) == dim, "x.size(3)=", x.size(3), " must equal dim=", dim);
  CHECK_LAST_DIM_CONTIGUOUS(x);
  FLASHINFER_CHECK(x.stride(2) == dim, "x.stride(2)=", x.stride(2), " must equal dim=", dim,
                   " ((nheads, dim) must be contiguous)");

  // In varlen, all per-token tensors share the flattened token axis — use
  // x.size(1) as the canonical total_tokens and cross-check the others below.
  int64_t const total_tokens = is_varlen ? x.size(1) : 0;

  // ── Validate dt ──
  CHECK_CUDA(dt);
  CHECK_DIM(4, dt);
  if (is_varlen) {
    FLASHINFER_CHECK(dt.size(0) == 1, "varlen: dt.size(0)=", dt.size(0), " must be 1");
    FLASHINFER_CHECK(dt.size(1) == total_tokens, "varlen: dt.size(1)=", dt.size(1),
                     " must equal x.size(1)=", total_tokens);
  } else {
    FLASHINFER_CHECK(dt.size(0) == batch, "dt.size(0)=", dt.size(0), " must equal batch=", batch);
    FLASHINFER_CHECK(dt.size(1) == npredicted, "dt.size(1)=", dt.size(1),
                     " must equal npredicted=", npredicted);
  }
  FLASHINFER_CHECK(dt.size(2) == nheads, "dt.size(2)=", dt.size(2), " must equal nheads=", nheads);
  FLASHINFER_CHECK(dt.size(3) == dim, "dt.size(3)=", dt.size(3), " must equal dim=", dim);
  FLASHINFER_CHECK(dt.stride(2) == 1, "dt.stride(2) must be 1 (tie_hdim), got ", dt.stride(2));
  FLASHINFER_CHECK(dt.stride(3) == 0, "dt.stride(3) must be 0 (tie_hdim), got ", dt.stride(3));

  // ── Validate A: (nheads, dim, dstate) tie_hdim ──
  CHECK_CUDA(A);
  CHECK_DIM(3, A);
  FLASHINFER_CHECK(A.size(0) == nheads, "A.size(0)=", A.size(0), " must equal nheads=", nheads);
  FLASHINFER_CHECK(A.size(1) == dim, "A.size(1)=", A.size(1), " must equal dim=", dim);
  FLASHINFER_CHECK(A.size(2) == dstate, "A.size(2)=", A.size(2), " must equal dstate=", dstate);
  FLASHINFER_CHECK(A.stride(0) == 1, "A.stride(0) must be 1, got ", A.stride(0));
  FLASHINFER_CHECK(A.stride(1) == 0, "A.stride(1) must be 0 (tie_hdim), got ", A.stride(1));
  FLASHINFER_CHECK(A.stride(2) == 0, "A.stride(2) must be 0 (tie_hdim), got ", A.stride(2));

  // ── Validate B ──
  CHECK_CUDA(B);
  CHECK_DIM(4, B);
  if (is_varlen) {
    FLASHINFER_CHECK(B.size(0) == 1, "varlen: B.size(0)=", B.size(0), " must be 1");
    FLASHINFER_CHECK(B.size(1) == total_tokens, "varlen: B.size(1)=", B.size(1),
                     " must equal x.size(1)=", total_tokens);
  } else {
    FLASHINFER_CHECK(B.size(0) == batch, "B.size(0)=", B.size(0), " must equal batch=", batch);
    FLASHINFER_CHECK(B.size(1) == npredicted, "B.size(1)=", B.size(1),
                     " must equal npredicted=", npredicted);
  }
  FLASHINFER_CHECK(B.size(3) == dstate, "B.size(3)=", B.size(3), " must equal dstate=", dstate);
  CHECK_LAST_DIM_CONTIGUOUS(B);
  FLASHINFER_CHECK(B.stride(2) == dstate, "B.stride(2)=", B.stride(2),
                   " must equal dstate=", dstate, " ((ngroups, dstate) must be contiguous)");
  FLASHINFER_CHECK(nheads % ngroups == 0, "nheads=", nheads,
                   " must be divisible by ngroups=", ngroups);

  // ── Validate C ──
  CHECK_CUDA(C);
  CHECK_DIM(4, C);
  if (is_varlen) {
    FLASHINFER_CHECK(C.size(0) == 1, "varlen: C.size(0)=", C.size(0), " must be 1");
    FLASHINFER_CHECK(C.size(1) == total_tokens, "varlen: C.size(1)=", C.size(1),
                     " must equal x.size(1)=", total_tokens);
  } else {
    FLASHINFER_CHECK(C.size(0) == batch, "C.size(0)=", C.size(0), " must equal batch=", batch);
    FLASHINFER_CHECK(C.size(1) == npredicted, "C.size(1)=", C.size(1),
                     " must equal npredicted=", npredicted);
  }
  FLASHINFER_CHECK(C.size(2) == ngroups, "C.size(2)=", C.size(2), " must equal ngroups=", ngroups);
  FLASHINFER_CHECK(C.size(3) == dstate, "C.size(3)=", C.size(3), " must equal dstate=", dstate);
  CHECK_LAST_DIM_CONTIGUOUS(C);
  FLASHINFER_CHECK(C.stride(2) == dstate, "C.stride(2)=", C.stride(2),
                   " must equal dstate=", dstate, " ((ngroups, dstate) must be contiguous)");

  // ── Validate output ──
  CHECK_CUDA(output);
  CHECK_DIM(4, output);
  if (is_varlen) {
    FLASHINFER_CHECK(output.size(0) == 1, "varlen: output.size(0)=", output.size(0), " must be 1");
    FLASHINFER_CHECK(output.size(1) == total_tokens, "varlen: output.size(1)=", output.size(1),
                     " must equal x.size(1)=", total_tokens);
  } else {
    FLASHINFER_CHECK(output.size(0) == batch, "output.size(0)=", output.size(0),
                     " must equal batch=", batch);
    FLASHINFER_CHECK(output.size(1) == npredicted, "output.size(1)=", output.size(1),
                     " must equal npredicted=", npredicted);
  }
  FLASHINFER_CHECK(output.size(2) == nheads, "output.size(2)=", output.size(2),
                   " must equal nheads=", nheads);
  FLASHINFER_CHECK(output.size(3) == dim, "output.size(3)=", output.size(3),
                   " must equal dim=", dim);
  CHECK_LAST_DIM_CONTIGUOUS(output);
  FLASHINFER_CHECK(output.stride(2) == dim, "output.stride(2)=", output.stride(2),
                   " must equal dim=", dim, " ((nheads, dim) must be contiguous)");

  // ── Validate cache tensors ──
  // old_x: kernel uses `head * DIM + d_tile_off` → (nheads, dim) contig.
  CHECK_CUDA(old_x);
  CHECK_DIM(4, old_x);  // (state_cache_size, MAX_WINDOW, nheads, dim)
  FLASHINFER_CHECK(old_x.size(0) == state_cache_size, "old_x.size(0)=", old_x.size(0),
                   " must equal state_cache_size=", state_cache_size);
  FLASHINFER_CHECK(old_x.size(1) == max_window, "old_x.size(1)=", old_x.size(1),
                   " must equal max_window=", max_window);
  FLASHINFER_CHECK(old_x.size(2) == nheads, "old_x.size(2)=", old_x.size(2),
                   " must equal nheads=", nheads);
  FLASHINFER_CHECK(old_x.size(3) == dim, "old_x.size(3)=", old_x.size(3), " must equal dim=", dim);
  CHECK_LAST_DIM_CONTIGUOUS(old_x);
  FLASHINFER_CHECK(old_x.stride(2) == dim, "old_x.stride(2)=", old_x.stride(2),
                   " must equal dim=", dim, " ((nheads, dim) must be contiguous)");

  // old_B: kernel uses `group_idx * DSTATE` → (ngroups, dstate) contig.
  CHECK_CUDA(old_B);
  CHECK_DIM(5, old_B);  // (state_cache_size, 2, MAX_WINDOW, ngroups, dstate)
  FLASHINFER_CHECK(old_B.size(0) == state_cache_size, "old_B.size(0)=", old_B.size(0),
                   " must equal state_cache_size=", state_cache_size);
  FLASHINFER_CHECK(old_B.size(1) == 2, "old_B.size(1) must be 2 (double-buffered), got ",
                   old_B.size(1));
  FLASHINFER_CHECK(old_B.size(2) == max_window, "old_B.size(2)=", old_B.size(2),
                   " must equal max_window=", max_window);
  FLASHINFER_CHECK(old_B.size(3) == ngroups, "old_B.size(3)=", old_B.size(3),
                   " must equal ngroups=", ngroups);
  FLASHINFER_CHECK(old_B.size(4) == dstate, "old_B.size(4)=", old_B.size(4),
                   " must equal dstate=", dstate);
  CHECK_LAST_DIM_CONTIGUOUS(old_B);
  FLASHINFER_CHECK(old_B.stride(3) == dstate, "old_B.stride(3)=", old_B.stride(3),
                   " must equal dstate=", dstate, " ((ngroups, dstate) must be contiguous)");

  // old_dt: kernel only assumes last-dim contig (head row).
  CHECK_CUDA(old_dt);
  CHECK_DIM(4, old_dt);  // (state_cache_size, 2, nheads, MAX_WINDOW)
  FLASHINFER_CHECK(old_dt.size(0) == state_cache_size, "old_dt.size(0)=", old_dt.size(0),
                   " must equal state_cache_size=", state_cache_size);
  FLASHINFER_CHECK(old_dt.size(1) == 2, "old_dt.size(1) must be 2, got ", old_dt.size(1));
  FLASHINFER_CHECK(old_dt.size(2) == nheads, "old_dt.size(2)=", old_dt.size(2),
                   " must equal nheads=", nheads);
  FLASHINFER_CHECK(old_dt.size(3) == max_window, "old_dt.size(3)=", old_dt.size(3),
                   " must equal max_window=", max_window);
  CHECK_LAST_DIM_CONTIGUOUS(old_dt);

  // old_cumAdt: same as old_dt.
  CHECK_CUDA(old_cumAdt);
  CHECK_DIM(4, old_cumAdt);  // (state_cache_size, 2, nheads, MAX_WINDOW)
  FLASHINFER_CHECK(old_cumAdt.size(0) == state_cache_size,
                   "old_cumAdt.size(0)=", old_cumAdt.size(0),
                   " must equal state_cache_size=", state_cache_size);
  FLASHINFER_CHECK(old_cumAdt.size(1) == 2, "old_cumAdt.size(1) must be 2, got ",
                   old_cumAdt.size(1));
  FLASHINFER_CHECK(old_cumAdt.size(2) == nheads, "old_cumAdt.size(2)=", old_cumAdt.size(2),
                   " must equal nheads=", nheads);
  FLASHINFER_CHECK(old_cumAdt.size(3) == max_window, "old_cumAdt.size(3)=", old_cumAdt.size(3),
                   " must equal max_window=", max_window);
  CHECK_LAST_DIM_CONTIGUOUS(old_cumAdt);

  CHECK_CUDA(cache_buf_idx);
  CHECK_DIM(1, cache_buf_idx);
  FLASHINFER_CHECK(cache_buf_idx.size(0) == state_cache_size,
                   "cache_buf_idx.size(0)=", cache_buf_idx.size(0),
                   " must equal state_cache_size=", state_cache_size);
  CHECK_CONTIGUOUS(cache_buf_idx);

  CHECK_CUDA(prev_num_accepted);
  CHECK_DIM(1, prev_num_accepted);
  FLASHINFER_CHECK(prev_num_accepted.size(0) == state_cache_size,
                   "prev_num_accepted.size(0)=", prev_num_accepted.size(0),
                   " must equal state_cache_size=", state_cache_size);
  CHECK_CONTIGUOUS(prev_num_accepted);

  // ── Validate optional D ──
  if (D.has_value()) {
    auto& Dv = D.value();
    CHECK_CUDA(Dv);
    CHECK_DIM(2, Dv);
    FLASHINFER_CHECK(Dv.size(0) == nheads, "D.size(0)=", Dv.size(0), " must equal nheads=", nheads);
    FLASHINFER_CHECK(Dv.size(1) == dim, "D.size(1)=", Dv.size(1), " must equal dim=", dim);
    FLASHINFER_CHECK(Dv.stride(0) == 1, "D.stride(0) must be 1 (tie_hdim), got ", Dv.stride(0));
    FLASHINFER_CHECK(Dv.stride(1) == 0, "D.stride(1) must be 0 (tie_hdim), got ", Dv.stride(1));
  }

  // ── Validate optional dt_bias ──
  if (dt_bias.has_value()) {
    auto& db = dt_bias.value();
    CHECK_CUDA(db);
    CHECK_DIM(2, db);
    FLASHINFER_CHECK(db.size(0) == nheads, "dt_bias.size(0)=", db.size(0),
                     " must equal nheads=", nheads);
    FLASHINFER_CHECK(db.size(1) == dim, "dt_bias.size(1)=", db.size(1), " must equal dim=", dim);
    FLASHINFER_CHECK(db.stride(0) == 1, "dt_bias.stride(0) must be 1 (tie_hdim), got ",
                     db.stride(0));
    FLASHINFER_CHECK(db.stride(1) == 0, "dt_bias.stride(1) must be 0 (tie_hdim), got ",
                     db.stride(1));
  }

  // ── Validate optional z: same layout/contig rules as x ──
  if (z.has_value()) {
    auto& zv = z.value();
    CHECK_CUDA(zv);
    CHECK_DIM(4, zv);
    if (is_varlen) {
      FLASHINFER_CHECK(zv.size(0) == 1, "varlen: z.size(0)=", zv.size(0), " must be 1");
      FLASHINFER_CHECK(zv.size(1) == total_tokens, "varlen: z.size(1)=", zv.size(1),
                       " must equal x.size(1)=", total_tokens);
    } else {
      FLASHINFER_CHECK(zv.size(0) == batch, "z.size(0)=", zv.size(0), " must equal batch=", batch);
      FLASHINFER_CHECK(zv.size(1) == npredicted, "z.size(1)=", zv.size(1),
                       " must equal npredicted=", npredicted);
    }
    FLASHINFER_CHECK(zv.size(2) == nheads, "z.size(2)=", zv.size(2), " must equal nheads=", nheads);
    FLASHINFER_CHECK(zv.size(3) == dim, "z.size(3)=", zv.size(3), " must equal dim=", dim);
    CHECK_LAST_DIM_CONTIGUOUS(zv);
    FLASHINFER_CHECK(zv.stride(2) == dim, "z.stride(2)=", zv.stride(2), " must equal dim=", dim,
                     " ((nheads, dim) must be contiguous)");
  }

  // ── Validate optional state_batch_indices ──
  if (state_batch_indices.has_value()) {
    auto& sbi = state_batch_indices.value();
    CHECK_CUDA(sbi);
    CHECK_DIM(1, sbi);
    FLASHINFER_CHECK(sbi.size(0) == batch, "state_batch_indices.size(0)=", sbi.size(0),
                     " must equal batch=", batch);
    CHECK_CONTIGUOUS(sbi);
  }

  // ── Validate optional state_scale: (state_cache_size, nheads, dim) ──
  // Inner two dims (nheads, dim) must be contiguous; only batch stride is
  // parameterized in the params struct.
  if (state_scale.has_value()) {
    auto const& ss = state_scale.value();
    CHECK_CUDA(ss);
    CHECK_DIM(3, ss);
    FLASHINFER_CHECK(ss.size(0) == state_cache_size, "state_scale.size(0)=", ss.size(0),
                     " must equal state_cache_size=", state_cache_size);
    FLASHINFER_CHECK(ss.size(1) == nheads, "state_scale.size(1)=", ss.size(1),
                     " must equal nheads=", nheads);
    FLASHINFER_CHECK(ss.size(2) == dim, "state_scale.size(2)=", ss.size(2),
                     " must equal dim=", dim);
    FLASHINFER_CHECK(ss.stride(2) == 1, "state_scale.stride(2) must be 1, got ", ss.stride(2));
    FLASHINFER_CHECK(ss.stride(1) == dim, "state_scale.stride(1)=", ss.stride(1),
                     " must equal dim=", dim, " ((nheads, dim) must be contiguous)");
  }

  // ── Dtype consistency ──
  // input_dtype = x.dtype; all activation tensors (B, C, output, z, old_x,
  // old_B) and the state cache's "input-side" mirrors must match it.
  // weight_dtype = D.dtype = dt_bias.dtype (kernel template sees one
  // weight_t for both).
  // Cache scalar tensors have fixed dtypes hardcoded in the kernel.
  {
    auto input_dtype = x.dtype();
    FLASHINFER_CHECK(B.dtype() == input_dtype, "B.dtype must match x.dtype");
    FLASHINFER_CHECK(C.dtype() == input_dtype, "C.dtype must match x.dtype");
    FLASHINFER_CHECK(output.dtype() == input_dtype, "output.dtype must match x.dtype");
    FLASHINFER_CHECK(old_x.dtype() == input_dtype, "old_x.dtype must match x.dtype");
    FLASHINFER_CHECK(old_B.dtype() == input_dtype, "old_B.dtype must match x.dtype");
    if (z.has_value()) {
      FLASHINFER_CHECK(z.value().dtype() == input_dtype, "z.dtype must match x.dtype");
    }
    if (D.has_value() && dt_bias.has_value()) {
      FLASHINFER_CHECK(D.value().dtype() == dt_bias.value().dtype(),
                       "D.dtype must equal dt_bias.dtype (kernel uses a single weight_t)");
    }
    // old_dt / old_cumAdt are produced by this same kernel in f32 and
    // consumed back in f32 on the next call.
    FLASHINFER_CHECK(old_dt.dtype().code == kDLFloat && old_dt.dtype().bits == 32,
                     "old_dt must be float32");
    FLASHINFER_CHECK(old_cumAdt.dtype().code == kDLFloat && old_cumAdt.dtype().bits == 32,
                     "old_cumAdt must be float32");
    // Index tensors used by the kernel as int32 scalars.
    FLASHINFER_CHECK(cache_buf_idx.dtype().code == kDLInt && cache_buf_idx.dtype().bits == 32,
                     "cache_buf_idx must be int32");
    FLASHINFER_CHECK(
        prev_num_accepted.dtype().code == kDLInt && prev_num_accepted.dtype().bits == 32,
        "prev_num_accepted must be int32");
    if (state_batch_indices.has_value()) {
      auto sbi_dt = state_batch_indices.value().dtype();
      FLASHINFER_CHECK(sbi_dt.code == kDLInt && (sbi_dt.bits == 32 || sbi_dt.bits == 64),
                       "state_batch_indices must be int32 or int64");
    }
    if (state_scale.has_value()) {
      auto ss_dt = state_scale.value().dtype();
      FLASHINFER_CHECK(ss_dt.code == kDLFloat && ss_dt.bits == 32, "state_scale must be float32");
    }
    // Quantized state dtypes (int8, fp8_e4m3fn, ...) require a state_scale
    // tensor; non-quantized dtypes must not pass one.  Mirrors the Python
    // wrapper assertion and matches the kernel's compile-time
    // `state_scale_t == void` gating.
    {
      auto sd = state.dtype();
      bool const is_int8 = (sd.code == kDLInt && sd.bits == 8);
      bool const is_fp8 = (sd.code == kDLFloat8_e4m3fn && sd.bits == 8);
      bool const is_quantized_state = is_int8 || is_fp8;
      if (is_quantized_state) {
        FLASHINFER_CHECK(state_scale.has_value(),
                         "Quantized state.dtype (int8/fp8_e4m3fn) requires a state_scale tensor "
                         "of shape (state_cache_size, nheads, dim) and dtype float32");
        // The 8-bit replay path uses Layout<_4, _1> (M-shard per warp) which
        // needs per-warp M = D_PER_CTA / 4 >= 16 (m16n8 atom M).  This forces
        // D_PER_CTA >= 64, i.e. d_split == 1.
        FLASHINFER_CHECK(
            d_split == 1,
            "Quantized state.dtype (int8/fp8_e4m3fn) requires d_split=1 (got d_split=", d_split,
            "); the M-shard-per-warp replay layout needs D_PER_CTA / 4 >= 16.");
      } else {
        FLASHINFER_CHECK(!state_scale.has_value(),
                         "state_scale must be None for non-quantized state.dtype "
                         "(allowed quantized dtypes: {int8, fp8_e4m3fn})");
      }
    }
  }

  // ── Populate params ──
  CheckpointingSsuParams p;

  // ── Validate d_split (v12 §59) ──
  // Allowed for v12: {1, 2}.  d_split=4 deferred to v12.x (needs warp-count
  // restructure — output MMA `_1×4` layout requires D_PER_CTA ≥ 32).
  FLASHINFER_CHECK(d_split == 1 || d_split == 2, "d_split=", d_split,
                   " must be one of {1, 2} (d_split=4 is deferred to v12.x)");
  FLASHINFER_CHECK(dim % d_split == 0, "dim=", dim, " must be divisible by d_split=", d_split);
  FLASHINFER_CHECK(dim / d_split >= 32, "d_split=", d_split, " gives D_PER_CTA=", dim / d_split,
                   " < 32 (output MMA m16n8 atom floor with _1×4 warp layout)");

  p.batch = batch;
  p.nheads = nheads;
  p.dim = dim;
  p.dstate = dstate;
  p.ngroups = ngroups;
  p.state_cache_size = state_cache_size;
  p.npredicted = npredicted;
  p.max_window = max_window;
  p.pad_slot_id = pad_slot_id;
  p.d_split = static_cast<int32_t>(d_split);
  p.dt_softplus = dt_softplus;

  // Pointers
  p.state = state.data_ptr();
  p.x = const_cast<void*>(x.data_ptr());
  p.dt = const_cast<void*>(dt.data_ptr());
  p.A = const_cast<void*>(A.data_ptr());
  p.B = const_cast<void*>(B.data_ptr());
  p.C = const_cast<void*>(C.data_ptr());
  p.output = output.data_ptr();

  p.old_x = old_x.data_ptr();
  p.old_B = const_cast<void*>(old_B.data_ptr());
  p.old_dt = const_cast<void*>(old_dt.data_ptr());
  p.old_cumAdt = const_cast<void*>(old_cumAdt.data_ptr());
  p.cache_buf_idx = const_cast<void*>(cache_buf_idx.data_ptr());
  p.prev_num_accepted = const_cast<void*>(prev_num_accepted.data_ptr());

  if (D.has_value()) p.D = const_cast<void*>(D.value().data_ptr());
  if (z.has_value()) {
    p.z = const_cast<void*>(z.value().data_ptr());
    // Same seq-dim selection as the rest of the batch-side tensors below.
    p.z_stride_seq = z.value().stride(is_varlen ? 1 : 0);
    p.z_stride_token = z.value().stride(1);
  }
  if (dt_bias.has_value()) p.dt_bias = const_cast<void*>(dt_bias.value().data_ptr());
  if (state_batch_indices.has_value())
    p.state_batch_indices = const_cast<void*>(state_batch_indices.value().data_ptr());
  if (is_varlen) {
    p.cu_seqlens = const_cast<void*>(cu_seqlens.value().data_ptr());
  }
  if (state_scale.has_value()) {
    p.state_scale = state_scale.value().data_ptr();
    p.state_scale_stride_seq = state_scale.value().stride(0);
  }
  if (rand_seed.has_value()) {
    auto const& rs = rand_seed.value();
    CHECK_CUDA(rs);
    FLASHINFER_CHECK(rs.numel() == 1, "rand_seed must be single-element, got numel=", rs.numel());
    FLASHINFER_CHECK(rs.dtype().code == kDLInt && rs.dtype().bits == 64, "rand_seed must be int64");
    p.rand_seed = static_cast<const int64_t*>(rs.data_ptr());
  }

  // Strides
  p.state_stride_seq = state.stride(0);

  // `*_stride_seq` is the outer iteration stride.  Non-varlen iterates over
  // dim 0 (per-batch), varlen iterates over dim 1 (per-token) — sequences
  // are packed into a single batch in the (1, total_tokens, ...) layout.
  // The kernel uses one formula `seq * *_stride_seq` for both modes.
  int const seq_dim = is_varlen ? 1 : 0;
  p.x_stride_seq = x.stride(seq_dim);
  p.x_stride_token = x.stride(1);
  p.dt_stride_seq = dt.stride(seq_dim);
  p.dt_stride_token = dt.stride(1);
  p.B_stride_seq = B.stride(seq_dim);
  p.B_stride_token = B.stride(1);
  p.C_stride_seq = C.stride(seq_dim);
  p.C_stride_token = C.stride(1);
  p.out_stride_seq = output.stride(seq_dim);
  p.out_stride_token = output.stride(1);

  p.old_x_stride_seq = old_x.stride(0);
  p.old_x_stride_token = old_x.stride(1);
  p.old_B_stride_seq = old_B.stride(0);
  p.old_B_stride_dbuf = old_B.stride(1);
  p.old_B_stride_token = old_B.stride(2);
  p.old_dt_stride_seq = old_dt.stride(0);
  p.old_dt_stride_dbuf = old_dt.stride(1);
  p.old_dt_stride_head = old_dt.stride(2);
  p.old_cumAdt_stride_seq = old_cumAdt.stride(0);
  p.old_cumAdt_stride_dbuf = old_cumAdt.stride(1);
  p.old_cumAdt_stride_head = old_cumAdt.stride(2);

  // Launch
  ffi::CUDADeviceGuard device_guard(state.device().device_id);
  const cudaStream_t stream = get_stream(state.device());

  launchCheckpointingSsu<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t, state_scale_t>(
      p, stream);
}

}  // namespace flashinfer::mamba::checkpointing
