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
// Sparse MLA (DSA with a pool-compressed indexer) kernels over a paged latent cache.
#pragma once

#include "../common.h"

namespace glm {
namespace k {

// Paged cache geometry shared by all MLA layers: the TensorRT-LLM V2 pools of the glm5_next
// cache manager (tokens_per_block 128), one latent pool and one INDEX_KEY pool per sparse layer,
// addressed through the same page (V2 slot) ids.
constexpr int kMlaBlock = 128;   // tokens per cache page (the runtime's tokens_per_block)
constexpr int kLatent = 512;     // kv_lora_rank
constexpr int kIdxDim = 128;     // indexer head dim
constexpr int kKpool = 4;
constexpr int kPoolsPerPage = kMlaBlock / kKpool;  // pool keys per cache page
constexpr int kKgRow = 2 * kIdxDim;               // indexer [k | gate] of one token
// INDEX_KEY page (kMlaBlock tokens x 3 x kIdxDim bf16 in the runtime's pool), in rows of kIdxDim:
// rows [0, kPoolsPerPage) the page's pool keys, then the kMlaBlock [k | gate] rows (two rows of
// kIdxDim each); the rest of the page is unused.
constexpr int kIdxPageRows = kMlaBlock * 3;        // rows of kIdxDim per page (dense pool)
constexpr int kIdxKgRow0 = kPoolsPerPage;          // first [k | gate] row of a page
static_assert(kIdxKgRow0 + 2 * kMlaBlock <= kIdxPageRows, "INDEX_KEY page layout overflows");
constexpr int kSelPools = 512;   // index_topk / kpool
constexpr int kIdxHeads = 32;    // indexer heads
// selected rows incl. the tail of the incomplete pool; + 4: the chained MTP passes attend a
// selection shared from an earlier position plus every row since (sparse_mla_attn vis)
constexpr int kMaxRows = kSelPools * kKpool + kKpool - 1 + 4;
constexpr int kMaxAttnSplits = 16;

// Per-token metadata of a packed batch (device pointers, T tokens).
struct MlaTokMeta {
  const int* token_seq;         // [T] batch row of each token (its row of block_tables)
  const int* positions;         // [T] cache position of each token
  const int64_t* block_tables;  // [rows, max_blocks] page ids (the runtime's raw V2 slot ids)
  int max_blocks;               // row stride of block_tables
  // nonzero: token_seq / positions / block_tables were written before the step (host uploads),
  // so decode kernels may read them (and earlier steps' cache rows) before their PDL wait
  int pre = 0;
  // Page strides of this layer's pools (a coalesced V2 pool interleaves other buffers between
  // consecutive pages): latent rows of kLatent per page, INDEX_KEY rows of kIdxDim per page.
  int64_t lat_page_rows = kMlaBlock;
  int64_t idx_page_rows = kIdxPageRows;
};

// Cache layout per MLA layer: latent rows [pages * lat_page_rows, 512] and the INDEX_KEY page
// layout above (idx_cache: the layer's INDEX_KEY pool viewed as rows of 128). The pool key of
// pool j (tokens [4j, 4j + 4)) is computed once the pool is complete; every token's indexer
// [LayerNorm(k) | gate] row stays in its page for the pool completed by a later step.
// Per step and layer: mla_append (latent and [k | gate] rows), then kpool_update.
// latent: [T, 512] bf16 (normalized); kgw: [T, ld_kgw] fp32 indexer projection whose columns
// [0, 128) are k (LayerNorm applied here) and [128, 256) the pool-compression gate.
void mla_append(bf16* latent_cache, bf16* idx_cache, const bf16* latent, const float* kgw,
                int ld_kgw, const bf16* kn_w, const bf16* kn_b, const MlaTokMeta& meta, int T,
                cudaStream_t st);

// Compute the pool key of every pool completed by a token of this step (members' [k | gate] rows
// from the pages, written by mla_append of this or an earlier step).
// mla_append + kpool_update in one single-CTA launch for steps of at most 32 tokens (bitwise the
// two kernels; returns false, nothing launched, for more tokens); latent nullptr: only the
// indexer's part (the norms wrote the latent rows into the cache)
constexpr int kCacheFuseMaxT = 32;  // (decode with MTP 3: up to 8 sequences)
bool mla_cache_update(bf16* latent_cache, bf16* idx_cache, const bf16* latent, const float* kgw,
                      int ld_kgw, const bf16* kn_w, const bf16* kn_b, const bf16* ape,
                      const MlaTokMeta& meta, int T, cudaStream_t st);
void kpool_update(bf16* idx_cache, const bf16* ape, const MlaTokMeta& meta, int T,
                  cudaStream_t st);

// For tokens with more than 512 complete pools, score the pools and select the top 512.
// q_idx: [T, 32, 128] bf16; w: head weights (fp32, row stride ld_w); scores: [T, max_pools]
// scratch; sel: [T, 512] selected pool ids. Tokens with <= 512 pools are skipped (dense).
void kpool_select(int* sel, float* scores, const bf16* q_idx, const float* w, int ld_w,
                  const bf16* idx_cache, const MlaTokMeta& meta, int T, int max_pools,
                  int idx_heads, cudaStream_t st);
// The top-k selection alone over precomputed score rows ([T, stride] floats, token t's first
// (positions[t] + 1) / kKpool entries valid): what kpool_select / kpool_select_tiled launch after scoring.
void kpool_topk(int* sel, const float* scores, const MlaTokMeta& meta, int T, int stride,
                cudaStream_t st);
// The same with query tiles (tiles[i] = {first token, count}: <= 8 consecutive tokens of one
// sequence, e.g. a decode sequence's token and MTP drafts) so that pool keys are streamed once per
// tile, for the tokens [row0, row0 + ntok) that the tiles cover. meta, q_idx, w, sel and tiles use
// absolute token indices; score rows are relative to row0 with row length stride (>= the pools of
// every covered token), ntok * stride floats of scratch.
// q_rows / idx_rows: rows (of 128 values) of the q_idx and idx_cache buffers; with both known the
// scoring runs on the tcgen05 tensor cores (kpool_score_umma)
void kpool_select_tiled(int* sel, float* scores, const bf16* q_idx, const float* w, int ld_w,
                        const bf16* idx_cache, const MlaTokMeta& meta, const int2* tiles,
                        int ntiles, int row0, int ntok, int stride, int idx_heads,
                        cudaStream_t st, int64_t q_rows = 0, int64_t idx_rows = 0,
                        cudaEvent_t score_done = nullptr,  // (recorded between scoring and top-k)
                        bool score_only = false);          // (no top-k: sparse_mla_attn_sel selects)
// The decode query path in one launch (mla_qba.cu): q_nope = q_resid x W_qb^T (bf16, [M, H x nope]) and
// q_lat[t, h] = q_nope[t, h] x w_kc[h] over w_kcT ([H][kl][nope]) into q_lat ([M, ldo], head h at h x kl),
// bitwise gemv_pre(q_b) + the head-batched gemv_pre(absorb); M <= 8, q_lora 1536, nope 256, kv_lora 512.
// false (nothing launched) for other shapes.
bool mla_qba_fits(int M, int H, int ql, int nope, int kl);
// NQ (the chained MTP passes with the shared selection, whose q_resid feeds nothing else): mla_qba takes
// qkv_a's raw rows (q_resid = the rows' q_a part at stride ldq) and normalizes them itself, and one of its CTAs writes
// the latent rows into the cache: both as mla_qkv_norm computes them (its thread mapping at 256 threads, its block
// sums and arithmetic: bitwise), so that mla_qkv_norm does not run (one dependent hop fewer)
struct QbaNorm {
  const bf16* wq = nullptr;    // [q_lora] q_a_norm
  const bf16* x_kv = nullptr;  // the rows' kv_a part (stride ldq)
  const bf16* wkv = nullptr;   // [kv_lora] kv_a_norm
  bf16* kv_cache = nullptr;    // the latent cache: each row's position
  MlaTokMeta cm{};
  float eps = 0.f;
};
bool mla_qba(const bf16* q_resid, int ldq, const bf16* w_qb, const bf16* w_kcT, bf16* q_lat, int ldo,
             int M, int H, cudaStream_t st, unsigned* done = nullptr, bool early = false,
             const QbaNorm* nq = nullptr);
// early: the next kernel launched at this one's start (else after this one's PDL wait)
// done (the SIG instance, for sparse_mla_attn_sel): every CTA adds 1 to *done once its q_lat stores are visible (4 x H
// CTAs)
void mla_qba_preload();  // (every instance configured and loaded: before a launch that waits for one)
// Query-tiled scoring on SM100 (tcgen05.mma, TMA): false if the buffer sizes are unknown.
bool kpool_score_umma(float* scores, const bf16* q_idx, int64_t q_rows, const float* w, int ld_w,
                      const bf16* idx_cache, int64_t idx_rows, const MlaTokMeta& meta,
                      const int2* tiles, int ntiles, int row0, int stride, cudaStream_t st);

// Sparse attention over the selected latent rows (or all rows when the token sees <= 512
// complete pools). q: [T, H, 512] absorbed queries; out: [T, H, 512] bf16. H <= 64 (CTAs of 16
// heads). nsplit > 1: split-KV over each token's rows (flash-decoding) with a combine pass; ws
// holds sparse_mla_ws_floats(T, nsplit, H) floats. sparse_mla_splits picks nsplit for T tokens
// (x head groups) with at most max_rows visible rows.
int sparse_mla_splits(int T, int sm_count, int max_rows, int ws_ctas = 0, int budget_dflt = 0);
size_t sparse_mla_ws_floats(int T, int nsplit, int H);
// vis (optional, [T]): the complete pools a shared selection was taken over (index-share of the MTP
// draft loop): rows sel x 4 + every row from vis x 4 to the token's position (no selection:
// vis <= kSelPools, all rows); null: vis = (position + 1) / kKpool
void sparse_mla_attn(bf16* out, const bf16* q, const bf16* latent_cache, const int* sel,
                     const MlaTokMeta& meta, int T, int H, float scale, cudaStream_t st,
                     int nsplit = 1, float* ws = nullptr, const int* vis = nullptr);
// The same split-KV attention in thread-block clusters (the CLW instance): n CTAs per (token, head group), n the
// largest split count in [9, nsplit] whose T x head-group clusters fit on the GPU at once; the partials are combined
// inside the cluster (no workspace, no combine launch; the same math as sparse_mla_attn's split + combine in another
// fp32 order). False (nothing launched) unless 2 <= nsplit <= 16 and such an n exists.
// pre (with vis): the chained MTP passes' PRE instance (the row table and the gathers before the PDL wait, the own row
// after it: see mla.cu; the caller guarantees that the pass's earlier kernels' inputs of the row table are final when
// the grid starts); bitwise the CLW instance.
bool sparse_mla_attn_cl(bf16* out, const bf16* q, const bf16* latent_cache, const int* sel,
                        const MlaTokMeta& meta, int T, int H, float scale, cudaStream_t st, int nsplit,
                        const int* vis = nullptr, bool pre = false);
// The decode top-k and the CLW attention (16 splits) in one launch of T 16-CTA clusters (decode steps of <= 7 tokens
// of 16 local heads, max_pools - 1 <= 32768): launched after kpool_select_tiled(score_only) on the scoring's stream;
// the cluster selects the top 512 of each token's score row (scores [T, stride]; the set of every select, ties by
// pool index, ascending pool order into sel [T, 512]) and gathers the rows, then waits until cnt[0] reaches
// cnt_target (mla_qba's SIG instance: q of every head in q) and attends: bitwise sparse_mla_attn_cl (16 splits) on that
// sel. cnt [2] zeroed before the first launch (each launch leaves it zeroed). *pev: see mla.cu. False: nothing launched.
// sparse_mla_attn_sel_ok: whether it launches at (T, H, stride, max_pools) (the shapes, score rows of at least
// max_pools - 1 pools, T clusters co-resident), for a caller whose scoring leaves the top-k to it.
bool sparse_mla_sel_fits(int T, int H, int max_pools);
bool sparse_mla_attn_sel_ok(int T, int H, int stride, int max_pools);
bool sparse_mla_attn_sel(bf16* out, const bf16* q, const bf16* latent_cache, int* sel, const MlaTokMeta& meta,
                         int T, int H, float scale, cudaStream_t st, const float* scores, int stride, int max_pools,
                         unsigned* cnt, int cnt_target, cudaEvent_t* pev = nullptr);

// Query-tiled variant for steps with several consecutive tokens per sequence (prefill chunks, MTP
// verification): tiles[i] = {first token, count} with count <= sparse_mla_tile_tokens(H)
// consecutive tokens of one sequence; the union of their visible rows is streamed once.
// max_pools: upper bound of pool ids (max_seq_len / kKpool + 1).
int sparse_mla_tile_tokens(int H);
// q_rows / cache_rows: rows (of 512 values) of the q and latent cache buffers, ws: scratch of
// ws_bytes for the tiles' pool unions; with them (and H dividing 64, H >= 8) the attention runs
// on the tcgen05 tensor cores (sparse_mla_attn_umma).
void sparse_mla_attn_tiled(bf16* out, const bf16* q, const bf16* latent_cache, const int* sel,
                           const MlaTokMeta& meta, const int2* tiles, int ntiles, int H,
                           int max_pools, float scale, cudaStream_t st, int64_t q_rows = 0,
                           int64_t cache_rows = 0, void* ws = nullptr, size_t ws_bytes = 0);
bool sparse_mla_attn_umma(bf16* out, const bf16* q, int64_t q_rows, const bf16* cache,
                          int64_t cache_rows, const int* sel, const MlaTokMeta& meta,
                          const int2* tiles, int ntiles, int H, int max_pools, float scale,
                          void* ws, size_t ws_bytes, cudaStream_t st);
// The same over FlashMLA's SM100 sparse prefill kernel (sparse_mla_fmla.cu; 64 local heads, one CTA
// per token): each token's visible rows are expanded to a list of kFmlaRows cache row ids (the 512
// selected pools' rows + <= 3 rows of its incomplete pool, or the dense prefix), -1 padded, into
// ws (sparse_mla_fmla_ws_bytes). False (nothing launched) when H != 64 or ws is too small.
constexpr int kFmlaRows = 2112;  // >= kSelPools * kKpool + kKpool - 1, a multiple of 64
void mla_expand_rows(int* idx, int* len, const int* sel, const MlaTokMeta& meta, int T, int width,
                     cudaStream_t st);
size_t sparse_mla_fmla_ws_bytes(int T, int H);
bool sparse_mla_fmla_on(int H);
// out[t, h] = the splits' partials (sparse_mla_attn's split-KV layout) combined
void mla_attn_combine(bf16* out, const float* ws, int T, int H, int nsplit, cudaStream_t st);
// Decode (tensor parallel, 16 local heads, the step's sequences of 4 tokens): FlashMLA's union kernel
// (fmla_union.cu) on one query tile per sequence (its tokens x heads = 64 rows, the union of their
// pools streamed once), split-KV: CTA (tile, s) takes the s-th of nsplit ranges of the union's 64-row
// blocks, partials into partial_ws (T x nsplit blocks of 16 x 514 floats, partial_blocks of them),
// then mla_attn_combine. The union in ws (ntiles x sparse_mla_union_ws_bytes(16)) by
// sparse_mla_union_dec (sparse_mla_union's layout for tiles of <= 4 tokens: short dependency chains; it triggers
// its dependents after its own PDL wait). False (nothing launched) when H != 16, nsplit < 2 or a buffer is short.
void sparse_mla_union_dec(void* ws, const int* sel, const MlaTokMeta& meta, const int2* tiles,
                          int ntiles, int H, int max_pools, cudaStream_t st);
int sparse_mla_fmla_split_splits(int ntiles, int T, int sm_count, int partial_blocks);
bool sparse_mla_attn_fmla_split(bf16* out, const bf16* q, int64_t q_rows, const bf16* cache,
                                int64_t cache_rows, const int* sel, const MlaTokMeta& meta,
                                const int2* tiles, int ntiles, int T, int H, int max_pools,
                                float scale, void* ws, size_t ws_bytes, float* partial_ws,
                                size_t partial_blocks, int nsplit, cudaStream_t st);
// sparse_mla_attn_umma's union pre-pass alone (its workspace layout is read by the FlashMLA tiled path)
size_t sparse_mla_union_ws_bytes(int H);
int sparse_mla_union_capacity(int H);
void sparse_mla_union(void* ws, const int* sel, const MlaTokMeta& meta, const int2* tiles,
                      int ntiles, int H, int max_pools, cudaStream_t st);
// FlashMLA's kernel on the query tiles (64 / H tokens x H heads per CTA, the tiles' pool unions):
// the tensor-parallel case (H = 16); false (nothing launched) when ws is short
bool sparse_mla_attn_fmla_tiled(bf16* out, const bf16* q, int64_t q_rows, const bf16* cache,
                                int64_t cache_rows, const int* sel, const MlaTokMeta& meta,
                                const int2* tiles, int ntiles, int H, int max_pools, float scale,
                                void* ws, size_t ws_bytes, cudaStream_t st);
bool sparse_mla_attn_fmla(bf16* out, const bf16* q, const bf16* cache, int64_t cache_rows,
                          const int* sel, const MlaTokMeta& meta, int T, int H, float scale,
                          void* ws, size_t ws_bytes, int num_sm, cudaStream_t st);

}  // namespace k
}  // namespace glm
