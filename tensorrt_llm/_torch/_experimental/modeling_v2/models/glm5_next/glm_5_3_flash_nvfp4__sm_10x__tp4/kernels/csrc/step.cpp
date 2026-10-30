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
// The step program (step.h): one forward runs the decoder layers, the vocabulary-parallel LM head and the MTP draft
// chain of one tensor-parallel rank over TensorRT-LLM's pools and metadata:
// - pools: the latent / INDEX_KEY pages of the V2 cache manager (128-token pages through int64 page ids, per-layer page
//   strides, a token's indexer [k | gate] row kept in its page) and its KDA state slots (slot strides; the drafts'
//   replay records in the runtime's per-slot replay buffers), addressed through MlaTokMeta / KdaSeqMeta;
// - metadata: the caller's device arrays (StepIn), which the step-metadata kernel builds from the runtime's buffers, so
//   that a captured decode step rebuilds them on replay;
// - collectives: the custom all-reduce and the peer-memory exchanges over IPC-mapped buffers of the ranks' processes,
//   NCCL on a communicator of the tensor-parallel ranks for large messages.
#include "step.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <utility>

#include "kernels/kernels.h"
#include "kernels/moe_fp4_prefill.h"

namespace glm {

#define NCCL_CHECK(expr)                                                                         \
  do {                                                                                           \
    ncclResult_t _r = (expr);                                                                    \
    if (_r != ncclSuccess) {                                                                     \
      fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(_r), __FILE__, __LINE__); \
      throw std::runtime_error("NCCL error");                                                    \
    }                                                                                            \
  } while (0)

int ModelConfig::num_kda_layers() const {
  int n = 0;
  for (auto t : attn_types) n += t == AttnType::kKDA;
  return n;
}
int ModelConfig::num_mla_layers() const {
  int n = 0;
  for (auto t : attn_types) n += t == AttnType::kMLA;
  return n;
}

namespace {
constexpr size_t kScoreBudget = 256ull << 20;  // bytes for the pool-score scratch
constexpr int kFusedHcMaxTokens = 256;          // steps up to this size use the fused (cluster / tiled) mHC boundary
constexpr size_t kFp4GemmWorkspace = 32ull << 20;
// MLA passes of up to this many tokens fork the indexer branch onto the side stream
constexpr int kMlaForkMaxTokens = 64;
constexpr int kKdaPreGatesMin = 256;  // decode tokens from which the KDA gate projections run as GEMMs
// BF16 GEMVs of up to 8 rows whose weights fit the shared memory of the SMs minus 52 CTAs take gemv_pre: the weights
// are preloaded during the previous kernel; the 52 SMs are left to the previous kernel and to the next one's early
// start.
int gemv_pre_ctas(int sm_count) { return sm_count - 52; }
// BF16 GEMVs of up to kGemvStreamMaxRows rows too large for gemv_pre (KDA in_proj, 52.6 MB per GPU) take gemv_stream
// with up to the SMs minus 32 CTAs
constexpr int kGemvStreamMaxRows = 4;
int gemv_stream_ctas(int sm_count) { return sm_count - 32; }
// BF16 GEMVs of kGemvPipeMinRows .. 16 rows that gemv_pre does not take: the pipelined GEMV, one wave over all SMs
constexpr int kGemvPipeMinRows = 5;
// The fused NVFP4 prefill MoE (moe_fp4_prefill) instead of grouped GEMM + SwiGLU + grouped GEMM while the expected
// rows per expert (T x top_k / E) are at most kMoePfMaxRows.
constexpr int kMoePfMaxRows = 240;
// expert-side tokens of the largest step the fused prefill MoE takes (T x top_k / E <= kMoePfMaxRows)
size_t moe_pf_max_tokens(int E, int top_k) { return ((size_t)(kMoePfMaxRows + 1) * E - 1) / top_k; }
// The pool-score scratch is also FlashMLA's prefill workspace (sparse_mla_attn_fmla*): per-token row lists with 64
// heads (the token-split attention), per-tile pool unions of 64 / Hm tokens with fewer, for every token of a
// max_num_tokens step.
size_t fmla_ws_bytes(size_t T, int Hm, int max_batch) {
  if (Hm == 64) return k::sparse_mla_fmla_ws_bytes((int)T, Hm);
  if (Hm >= 8 && 64 % Hm == 0)  // (a tile never spans two sequences)
    return ((size_t)cdiv(T, 64 / Hm) + max_batch + 1) * k::sparse_mla_union_ws_bytes(Hm);
  return 0;
}
// a and b swapped for the guard's lifetime and back when it ends, also when an exception ends it: a step that throws
// leaves the execution stream, its cuBLASLt and the step's buffers as they were
template <typename V>
class ScopedSwap {
 public:
  ScopedSwap(V& a, V& b) : a_(a), b_(b) { std::swap(a_, b_); }
  ~ScopedSwap() { std::swap(a_, b_); }
  ScopedSwap(const ScopedSwap&) = delete;
  ScopedSwap& operator=(const ScopedSwap&) = delete;

 private:
  V& a_;
  V& b_;
};
}  // namespace

// ------------------------------------------------------------------ buffers
struct GlmStep::Buffers {
  // residual streams and sublayer IO
  bf16* streams;
  float *mixes, *post, *comb;
  float* hc_rs;
  float* hc_partial;
  int* hc_counter;
  bf16 *x, *out;  // (the caller's: peer memory for the sequence-parallel boundaries)
  // KDA
  bf16 *proj, *g1, *g2, *conv_out, *core, *core_n;
  float* kda_scratch;
  int kda_max_chunks;
  float* attn_ws;      // split-KV partials of the per-token sparse attention
  int attn_ws_blocks;  // its 16 x 514-float blocks
  int2* unit_tiles;    // {i, 1}: one-token tiles (the chained MTP passes' tokens)
  int max_unit_tiles;
  int2* dec_tiles;     // {4 i, 4}: a decode step's 4-token sequences
  unsigned* selk_cnt;  // (sparse_mla_attn_sel) [MLA layers + MTP][2]
  unsigned* head_rdy;  // the MTP head's rows-ready count (mtp_pass)
  // MLA
  bf16 *qkva, *q_resid, *latent, *q_idx, *q, *q_lat, *o_lat, *o;
  float* kgw;
  int* sel;
  float* scores;
  size_t scores_bytes;
  int score_rows, max_pools;
  // MLP / MoE
  bf16 *gu, *h13, *y;
  float *router_logits, *topk_w;
  int *topk_ids, *pair_pos, *counts, *offsets, *active, *num_active, *row_src;
  int* mf4_ctr;
  int* moe_hist;  // [layers][k::kMoeFp4HistStride]: the fused kernel's routing history per layer
  uint8_t *x4, *x4s, *x4sw;
  uint8_t *h4, *h4s;
  uint8_t *xp4, *sfp4, *hq4, *hs4;
  uint8_t* hsa4;
  uint8_t* hsa_pf;
  int hsa_pf_items;
  int* pf_ctr;
  int xp4_rows;
  int* pad;
  int x4_rows, h4_rows;
  bf16 *xp16, *h16;
  bf16 *sh_gu, *sh_h;
  float* sh_part;
  int h16_rows;
  uint8_t* fp4_ws;
  // MTP
  bf16 *hfin, *eh_in, *mtp_h;
  unsigned long long *draft_keys, *draft_keys2;
  unsigned long long* sample_keys;
  int *accepted, *drafts;
  int *next_ids, *next_pos, *next_slot;
  int *mtp_sel, *mtp_vis;
  bf16 *next_hidden, *mtp_g;
  bf16 *mtp_xp, *mtp_hp;
  // head
  bf16 *last_streams, *head;
  float* logits_local;
  void* head_scratch;
  int* out_tokens;
};

template <typename T>
T* GlmStep::dalloc(size_t n) {
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
  allocs_.push_back(p);
  return static_cast<T*>(p);
}

GlmStep::GlmStep(const ModelConfig& cfg, const ParallelConfig& pc, const RunnerConfig& rc, int device)
    : cfg_(cfg), pc_(pc), rc_(rc), device_(device) {
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device_));
  CHECK(pc_.moe_tp * pc_.moe_ep == pc_.tp, "moe_tp x moe_ep must be the tensor-parallel size");
  mtp_n_ = rc_.mtp_drafts;
  mtp_ = mtp_n_ > 0;
  Hl_kda_ = cfg_.kda_heads / pc_.tp;
  Hl_mla_ = cfg_.num_heads / pc_.tp;
  N_in_ = 3 * Hl_kda_ * cfg_.kda_head_dim + Hl_kda_ + 2 * cfg_.kda_head_dim;
  int nk = 0, nm = 0;
  for (int li = 0; li < cfg_.num_layers; ++li) {
    const bool kda = cfg_.attn_types[li] == AttnType::kKDA;
    kda_index_.push_back(kda ? nk++ : -1);
    mla_index_.push_back(kda ? -1 : nm++);
  }
  w_.layers.resize(cfg_.num_layers);
  // split-KV partial buffers of the per-token sparse attention: 2 x SMs CTAs
  attn_ws_ctas_ = 2 * sm_count_;
  gemm_ = std::make_unique<Gemm>(64ull << 20);
  gemm2_ = std::make_unique<Gemm>(64ull << 20);
  CUDA_CHECK(cudaStreamCreateWithFlags(&side_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&pf_stream_, cudaStreamNonBlocking));
  for (cudaEvent_t* e : {&ev_fork_, &ev_q_, &ev_sel_, &ev_qi_, &ev_score_, &ev_pf_, &ev_pf_join_, &ev_sh_fork_,
                         &ev_sh_join_})
    CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
}

GlmStep::~GlmStep() {
  cudaDeviceSynchronize();
  for (void* p : allocs_) cudaFree(p);
  if (comm_) ncclCommDestroy(comm_);
}

void GlmStep::set_tune(bool on) {
  gemm_->set_tune(on);
  gemm2_->set_tune(on);
}

void GlmStep::allocate() {
  CUDA_CHECK(cudaSetDevice(device_));
  b_ = std::make_unique<Buffers>();
  Buffers& b = *b_;
  // the MTP layer's experts tensor-parallel while the main layers' are expert parallel. Its three passes per step route
  // few tokens each, so by expert the rank with the most experts would set the pass's time; sliced, every rank streams
  // its quarter of each active expert: the mean, whatever the routing. (The weights are loaded that way: weights.py's
  // Layout.)
  mtp_moe_tp_ = mtp_ && pc_.moe_ep > 1 && w_.mtp.moe.local.ep == 1;
  auto note_moe = [&](const MoeW& m) {
    moe_fp4_ |= m.fmt == WFmt::kNvfp4;
    moe_bf16_ |= m.fmt == WFmt::kBf16;
  };
  for (const auto& L : w_.layers) {
    if (L.moe)
      note_moe(L.moe_w);
    else
      mlp_fp4_ |= L.mlp.fmt == WFmt::kNvfp4;
  }
  if (w_.mtp.present) note_moe(w_.mtp.moe);
  const size_t T = rc_.max_num_tokens, Bm = rc_.max_batch, H = cfg_.hidden;
  const size_t HDk = (size_t)Hl_kda_ * cfg_.kda_head_dim;
  const size_t Bs = Bm + 1;
  const size_t Smax = Bs * (mtp_ ? mtp_n_ + 1 : 1);  // sampled rows per step
  b.streams = dalloc<bf16>(T * cfg_.hc_mult * H);
  b.mixes = dalloc<float>(T * (2 + cfg_.hc_mult) * cfg_.hc_mult);
  b.hc_rs = dalloc<float>(T);
  b.post = dalloc<float>(T * cfg_.hc_mult);
  const size_t hc_rows = kFusedHcMaxTokens;
  b.hc_partial = dalloc<float>(hc_rows * 16 * 32);
  b.hc_counter = dalloc<int>(hc_rows);
  CUDA_CHECK(cudaMemset(b.hc_counter, 0, hc_rows * sizeof(int)));
  b.comb = dalloc<float>(T * cfg_.hc_mult * cfg_.hc_mult);
  b.x = b.out = nullptr;  // (set_io)
  // KDA
  b.proj = dalloc<bf16>(T * N_in_);
  b.g1 = dalloc<bf16>(T * HDk);
  b.g2 = dalloc<bf16>(T * HDk);
  b.conv_out = dalloc<bf16>(T * 3 * HDk);
  b.core = dalloc<bf16>(T * HDk);
  b.core_n = dalloc<bf16>(T * HDk);
  b.kda_max_chunks = (int)(T / 64 + Bs + 64);
  b.kda_scratch = dalloc<float>(k::kda_chunk_scratch_floats(b.kda_max_chunks, Hl_kda_));
  {
    b.max_unit_tiles = (int)Bm + 1;
    std::vector<int2> unit(b.max_unit_tiles);
    for (int i = 0; i < b.max_unit_tiles; ++i) unit[i] = make_int2(i, 1);
    b.unit_tiles = dalloc<int2>(unit.size());
    CUDA_CHECK(cudaMemcpy(b.unit_tiles, unit.data(), unit.size() * sizeof(int2), cudaMemcpyHostToDevice));
    for (int i = 0; i < b.max_unit_tiles; ++i) unit[i] = make_int2(4 * i, 4);
    b.dec_tiles = dalloc<int2>(unit.size());
    CUDA_CHECK(cudaMemcpy(b.dec_tiles, unit.data(), unit.size() * sizeof(int2), cudaMemcpyHostToDevice));
  }
  // split-KV partials: sparse_mla_splits keeps T * nsplit <= 2 * SMs (4 x SMs for the query-tiled decode)
  b.attn_ws_blocks = std::max(attn_ws_ctas_, 4 * sm_count_);
  b.attn_ws = dalloc<float>(k::sparse_mla_ws_floats(b.attn_ws_blocks, 2, 16) / 2);
  // MLA
  const size_t Hm = Hl_mla_;
  b.qkva = dalloc<bf16>(T * (cfg_.q_lora + cfg_.kv_lora));
  b.q_resid = dalloc<bf16>(T * cfg_.q_lora);
  b.latent = dalloc<bf16>(T * cfg_.kv_lora);
  b.kgw = dalloc<float>(T * (2 * cfg_.idx_head_dim + cfg_.idx_heads));
  b.q_idx = dalloc<bf16>(T * cfg_.idx_heads * cfg_.idx_head_dim);
  b.sel = dalloc<int>((T + 8) * k::kSelPools);  // (split indexer: all-gather of 4 ceil(T / 4) rows)
  b.selk_cnt = dalloc<unsigned>(2 * (size_t)(cfg_.num_mla_layers() + 2));
  CUDA_CHECK(cudaMemset(b.selk_cnt, 0, 2 * (size_t)(cfg_.num_mla_layers() + 2) * sizeof(unsigned)));
  b.head_rdy = dalloc<unsigned>(1);
  CUDA_CHECK(cudaMemset(b.head_rdy, 0, sizeof(unsigned)));
  // (sized for any max_seq_len: set_max_seq_len sets the pools per sequence and the score rows per batch later)
  b.scores_bytes = std::max(kScoreBudget, fmla_ws_bytes(T, (int)Hm, rc_.max_batch));
  b.scores = dalloc<float>(cdiv(b.scores_bytes, sizeof(float)));
  set_max_seq_len(rc_.max_seq_len);
  b.q = dalloc<bf16>(T * Hm * cfg_.qk_nope);
  b.q_lat = dalloc<bf16>(T * Hm * cfg_.kv_lora);
  b.o_lat = dalloc<bf16>(T * Hm * cfg_.kv_lora);
  b.o = dalloc<bf16>(T * Hm * cfg_.v_head);
  // MLP / MoE
  const size_t Ild = cfg_.inter / pc_.tp, Il = cfg_.moe_inter / pc_.moe_tp;
  const size_t R = T * (cfg_.top_k + 1);  // routed slots + the shared expert
  b.gu = dalloc<bf16>(T * 2 * Ild);
  b.router_logits = dalloc<float>(T * cfg_.n_experts);
  b.topk_ids = dalloc<int>(R);
  b.topk_w = dalloc<float>(R);
  b.pair_pos = dalloc<int>(R);
  b.counts = dalloc<int>(cfg_.n_experts + 1);
  b.offsets = dalloc<int>(cfg_.n_experts + 2);
  b.active = dalloc<int>(cfg_.n_experts + 1 + R);
  b.row_src = dalloc<int>(R);
  b.num_active = dalloc<int>(1);
  b.mf4_ctr = dalloc<int>(k::kMoeFp4CtrAlloc);
  CUDA_CHECK(cudaMemset(b.mf4_ctr, 0, k::kMoeFp4CtrAlloc * sizeof(int)));
  {
    const size_t nh = (size_t)cfg_.num_layers * k::kMoeFp4HistStride;  // (rows by layer index)
    b.moe_hist = dalloc<int>(nh);
    CUDA_CHECK(cudaMemset(b.moe_hist, 0, nh * sizeof(int)));
  }
  b.fp4_ws = dalloc<uint8_t>(kFp4GemmWorkspace);
  // (the MTP layer's experts: all of them tensor-parallel, Il of their own)
  const size_t Ilm = mtp_moe_tp_ ? (size_t)w_.mtp.moe.inter_local : Il;
  const size_t Ilx = std::max(Il, Ilm);
  b.h13 = dalloc<bf16>(R * 2 * Ilx);
  b.y = dalloc<bf16>(R * H);
  {
    const size_t Gp = 128 * (size_t)(cfg_.n_experts + 1);  // per-group padding of scale rows
    const bool f4 = moe_fp4_ || mlp_fp4_;
    b.x4 = f4 ? dalloc<uint8_t>(T * H / 2) : nullptr;
    b.x4s = f4 ? dalloc<uint8_t>(k::nvfp4_sf_bytes((int)T, (int)H)) : nullptr;
    b.x4sw = f4 ? dalloc<uint8_t>(k::nvfp4_sf_bytes((int)T, (int)H)) : nullptr;
    b.h4 = mlp_fp4_ ? dalloc<uint8_t>(T * Ild / 2) : nullptr;
    b.h4s = mlp_fp4_ ? dalloc<uint8_t>(k::nvfp4_sf_bytes((int)T, (int)Ild)) : nullptr;
    b.xp4 = moe_fp4_ ? dalloc<uint8_t>(R * H / 2) : nullptr;
    b.sfp4 = moe_fp4_ ? dalloc<uint8_t>((R + Gp) * H / 16) : nullptr;
    b.hq4 = moe_fp4_ ? dalloc<uint8_t>((R + 128) * Il / 2) : nullptr;
    b.hs4 = moe_fp4_ ? dalloc<uint8_t>((R + Gp) * Il / 16) : nullptr;
    b.hsa4 = moe_fp4_ ? dalloc<uint8_t>((size_t)(cfg_.n_experts + 1) * k::nvfp4_sf_bytes(128, (int)Il)) : nullptr;
    b.pad = dalloc<int>(cfg_.n_experts + 2);
    b.x4_rows = (int)T;
    b.h4_rows = (int)R + 128;
    b.xp4_rows = (int)R;
    const size_t Rpf = std::min(R, moe_pf_max_tokens(cfg_.n_experts, cfg_.top_k) * (cfg_.top_k + 1));
    b.hsa_pf_items = moe_fp4_ && Rpf > 0
                         ? std::min(k::kMoePfMaxItems, k::moe_fp4_prefill_items((int)Rpf, cfg_.n_experts + 1))
                         : 0;
    b.hsa_pf = b.hsa_pf_items > 0
                   ? dalloc<uint8_t>((size_t)b.hsa_pf_items * k::nvfp4_sf_bytes(k::kMoePfHsRows, (int)Il))
                   : nullptr;
    b.pf_ctr = moe_fp4_ ? dalloc<int>(k::kMoePfCtrSize) : nullptr;
    if (b.pf_ctr) CUDA_CHECK(cudaMemset(b.pf_ctr, 0, k::kMoePfCtrSize * sizeof(int)));
    b.xp16 = moe_bf16_ ? dalloc<bf16>(R * H) : nullptr;
    b.h16 = moe_bf16_ ? dalloc<bf16>(R * Ilx) : nullptr;
    b.h16_rows = (int)R;
    const bool sep = moe_fp4_ || moe_bf16_;
    const size_t Ils = cfg_.moe_inter / pc_.tp;
    b.sh_gu = sep ? dalloc<bf16>(T * 2 * Ils) : nullptr;
    b.sh_h = sep ? dalloc<bf16>(T * Ils) : nullptr;
    b.sh_part = sep ? dalloc<float>(std::max(Ils / 64 * k::kMoeFp4ShSplit * 128, Ils / 128 * k::kMoeFp4WideSplit * 256) *
                                    k::kMoeFp4MaxTokens)
                    : nullptr;
  }
  // MTP
  if (mtp_) {
    b.hfin = dalloc<bf16>(T * H);
    b.eh_in = dalloc<bf16>(T * 2 * H);
    b.mtp_h = dalloc<bf16>(T * H);
    b.draft_keys = dalloc<unsigned long long>(Smax);
    b.draft_keys2 = dalloc<unsigned long long>(Bs);
    b.accepted = dalloc<int>(Bs);
    b.drafts = dalloc<int>(Bs * mtp_n_);
    b.next_ids = dalloc<int>(Bs);
    b.next_pos = dalloc<int>(Bs * std::max(1, mtp_n_ - 1));  // (mtp_forward's hop: every chained pass's positions)
    b.next_slot = dalloc<int>(Bs);
    b.mtp_sel = dalloc<int>((size_t)Bs * k::kSelPools);
    b.mtp_vis = dalloc<int>(Bs);
    b.next_hidden = dalloc<bf16>(Bs * H);
    b.mtp_g = dalloc<bf16>(Bs * H);
    b.mtp_xp = dalloc<bf16>(Bs * H);
    b.mtp_hp = dalloc<bf16>(Bs * H);
  } else {
    b.hfin = b.eh_in = b.mtp_h = b.next_hidden = b.mtp_g = b.mtp_xp = b.mtp_hp = nullptr;
    b.accepted = b.drafts = b.next_ids = b.next_pos = b.next_slot = b.mtp_sel = b.mtp_vis = nullptr;
    b.draft_keys = b.draft_keys2 = nullptr;
  }
  // head
  const size_t Vl = cfg_.vocab / pc_.tp;
  b.last_streams = mtp_ ? nullptr : dalloc<bf16>(Smax * cfg_.hc_mult * H);
  b.head = dalloc<bf16>(Smax * H);
  b.logits_local = dalloc<float>(Smax * Vl);
  b.out_tokens = dalloc<int>(Smax);
  b.sample_keys = dalloc<unsigned long long>(Smax);
  b.head_scratch = dalloc<uint8_t>(k::head_keys_scratch_bytes());
  CUDA_CHECK(cudaMemset(b.head_scratch, 0, k::head_keys_scratch_bytes()));
  const int G = cfg_.n_experts / pc_.moe_ep;  // local routed experts (the shared expert is separate)
  if (moe_fp4_) {
    fp4_g1_ = std::make_unique<k::Nvfp4GroupedGemm>(G, 2 * Il, H, device_);
    fp4_g2_ = std::make_unique<k::Nvfp4GroupedGemm>(G, H, Il, device_);
  }
  if (moe_bf16_ && !mtp_moe_tp_) {
    bf_g1_ = std::make_unique<k::Bf16GroupedGemm>(G, 2 * Il, H, device_);
    bf_g2_ = std::make_unique<k::Bf16GroupedGemm>(G, H, Il, device_);
  }
  if (mtp_moe_tp_ && w_.mtp.present) {  // (the tensor-parallel MTP experts: all of them, a slice each)
    const MoeW& M = w_.mtp.moe;
    if (M.fmt == WFmt::kNvfp4) {
      fp4_g1m_ = std::make_unique<k::Nvfp4GroupedGemm>(M.local.n, 2 * M.inter_local, H, device_);
      fp4_g2m_ = std::make_unique<k::Nvfp4GroupedGemm>(M.local.n, H, M.inter_local, device_);
    } else {
      bf_g1m_ = std::make_unique<k::Bf16GroupedGemm>(M.local.n, 2 * M.inter_local, H, device_);
      bf_g2m_ = std::make_unique<k::Bf16GroupedGemm>(M.local.n, H, M.inter_local, device_);
    }
  }
  // (LP: the two Lamport copies, all sentinel; an even number of KDA layers, so that the copies alternate across steps
  // too)
  if (cfg_.num_kda_layers() > 0 && cfg_.num_kda_layers() % 2 == 0) {
    kda_lp_ = dalloc<bf16>(2 * 4 * (size_t)N_in_);
    CUDA_CHECK(cudaMemset(kda_lp_, 0xff, 2 * 4 * (size_t)N_in_ * sizeof(bf16)));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void GlmStep::set_max_seq_len(int max_seq_len) {
  Buffers& b = *b_;
  rc_.max_seq_len = max_seq_len;
  b.max_pools = rc_.max_seq_len / k::kKpool + 1;
  b.score_rows = (int)std::max<size_t>(
      1, std::min<size_t>(rc_.max_num_tokens, kScoreBudget / ((size_t)b.max_pools * sizeof(float))));
  CHECK((size_t)b.score_rows * b.max_pools * sizeof(float) <= b.scores_bytes, "GlmStep: max_seq_len too large");
}

void GlmStep::set_allreduce(const k::ArPeers& peers, size_t max_elems, int max_blocks, uint32_t* epochs) {
  CUDA_CHECK(cudaSetDevice(device_));
  ar_peers_ = peers;
  ar_max_elems_ = max_elems;
  ar_max_blocks_ = max_blocks;
  ar_epochs_ = epochs;
  ar_enabled_ = pc_.tp > 1;
  // RF: a 4-token MoE layer's tensor-core router (moe_router_mma_ks_rf) takes its x rows from the attention-side
  // boundary's copies by the Lamport protocol (k::ar_rf_x) instead of after that boundary's grid has completed: the
  // copies, written by that boundary (rf_layer_ok), the sentinel in between
  if (pc_.tp == 4 && cfg_.n_experts == k::kRouterRfGroups * 16 && !rf_x_) {
    const size_t n = (size_t)2 * k::kRouterRfGroups * k::kRouterRfT * cfg_.hidden;
    rf_x_ = dalloc<bf16>(n);
    k::moe_router_rf_init(rf_x_, n, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void GlmStep::set_sp(const k::SpPeers& peers, uint32_t* state) {
  sp_peers_ = peers;
  sp_state_ = state;
  sp_p2p_ = true;
}

void GlmStep::set_ts(const k::TsPeers& peers, uint32_t* state) {
  ts_peers_ = peers;
  ts_state_ = state;
  ts_p2p_ = true;
}

void GlmStep::set_io(bf16* x, bf16* out) {
  b_->x = x;
  b_->out = out;
}

std::vector<char> GlmStep::nccl_unique_id() {
  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));
  return std::vector<char>(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

void GlmStep::init_nccl(const std::vector<char>& id) {
  CUDA_CHECK(cudaSetDevice(device_));
  CHECK(id.size() == NCCL_UNIQUE_ID_BYTES, "init_nccl: an NCCL unique id");
  ncclUniqueId uid;
  memcpy(uid.internal, id.data(), NCCL_UNIQUE_ID_BYTES);
  NCCL_CHECK(ncclCommInitRank(&comm_, pc_.tp, uid, pc_.rank));
}

StepOut GlmStep::outputs() const {
  StepOut o;
  o.hidden = b_->hfin;
  o.tokens = b_->out_tokens;
  o.drafts = b_->drafts;
  o.accepted = b_->accepted;
  return o;
}

bf16* GlmStep::head_buf() const { return b_->head; }

k::MlaTokMeta GlmStep::mla_meta(const int* tok_slot, const int* positions) const {
  k::MlaTokMeta m{tok_slot, positions, in_->block_tables, in_->bt_stride};
  m.lat_page_rows = pools_.lat_page_rows;
  m.idx_page_rows = pools_.idx_page_rows;
  return m;
}

k::KdaSeqMeta GlmStep::kda_meta(int b0) const {
  const StepIn& in = *in_;
  k::KdaSeqMeta m{in.cu_seqlens + b0, in.kda_slots + b0, in.has_state + b0,
                  mtp_ && in.kda_replay ? in.kda_replay + b0 : nullptr};
  m.state_ss = pools_.state_ss;
  m.conv_ss = pools_.conv_ss;
  m.rec_ss = pools_.rec_ss;
  m.crec_ss = pools_.crec_ss;
  m.crec_sec = pools_.crec_sec;
  m.crec_rs = pools_.crec_rs;
  return m;
}

void GlmStep::allreduce(bf16* x, size_t n) {
  if (pc_.tp == 1) return;
  if (ar_enabled_ && k::custom_allreduce(x, n, ar_peers_, ar_epochs_, pc_.rank, pc_.tp, ar_max_elems_,
                                         ar_max_blocks_, stream_))
    return;
  NCCL_CHECK(ncclAllReduce(x, x, n, ncclBfloat16, ncclSum, comm_, stream_));
}

// ------------------------------------------------------------------ the step
void GlmStep::forward(const StepIn& in, cudaStream_t st) {
  CHECK(b_ && b_->x && b_->out, "GlmStep: allocate() and set_io() before the first forward");
  CHECK(in.T > 0 && in.T <= rc_.max_num_tokens, "GlmStep: tokens per step");
  in_ = &in;
  stream_ = st;
  forward_impl(in);
  in_ = nullptr;
}

void GlmStep::mtp(const StepIn& in, cudaStream_t st) {
  CHECK(mtp_ && in.head < 0 && in.S > 0, "GlmStep::mtp: the draft passes after a forward without the head");
  in_ = &in;
  stream_ = st;
  mtp_forward(in.T, in.S, in.Bs, in.need_sel);
  in_ = nullptr;
}

void GlmStep::forward_impl(const StepIn& in) {
  Buffers& b = *b_;
  const int T = in.T, B = in.B, nd = in.nd, dt = in.dt, S = in.S;
  const int H = cfg_.hidden, hc = cfg_.hc_mult, mix = (2 + hc) * hc;
  // decode-only steps of 4-token sequences: their count (the decode attention's query tiles)
  dec_seqs_ = nd == B && dt == 4 && nd * dt == T ? nd : 0;
  k::embed_streams(b.streams, w_.embed, in.ids, T, H, hc, stream_);
  const bool fused_hc = T <= kFusedHcMaxTokens;
  // Tensor parallelism: the all-reduce of the last sublayer's partial output (b.out, or the MoE
  // combine of the expert outputs) is owed until the next boundary, which does it itself when
  // it runs the cluster kernel (one fewer kernel and no round trip of the sums through memory)
  // (the attention's o_proj GEMV before such a boundary pushes its sums into every rank's buffer itself; the pushed
  // boundary's form, ArFuse.pushed 2: it loads its inputs and polls before the PDL wait, which moves to its first
  // store)
  constexpr int kArPushLevel = 2;
  const bool fuse_ar = pc_.tp > 1 && ar_enabled_ && fused_hc && k::mhc_fused_cluster(T, H) &&
                       k::allreduce_fusable((size_t)T * H, pc_.tp, ar_max_elems_);
  bool pend_ar = false, pend_moe = false;
  // Deferred Sinkhorn: a fused-all-reduce cluster boundary's Sinkhorn (comb for
  // the next boundary's post-mapping) runs in the next boundary's spare warp while that one's all-reduce is
  // in flight; comb carries the logits in between. Never across the last boundary (mhc_post reads comb).
  const bool sk_on = fuse_ar && k::mhc_sk_ok(T, H);
  bool sk_pending = false, hc_last = false;
  auto flush_ar = [&] {
    if (!pend_ar) return;
    if (pend_moe)
      moe_combine_allreduce(T);
    else
      allreduce(b.out, (size_t)T * H);
    pend_ar = false;
  };
  // Sequence-parallel boundaries (tensor parallelism, steps of >= kSpMinTokens tokens): the sublayer's partial output
  // is reduce-scattered instead of all-reduced, the boundary runs on this rank's quarter of the rows and its output
  // rows are all-gathered. The residual streams of the other quarters are left stale: every boundary and the final
  // norm read only this rank's rows.
  constexpr int kSpMinTokens = 2048;
  const int tpn = pc_.tp;
  const bool sp = !fused_hc && tpn > 1 && T >= kSpMinTokens && rc_.max_num_tokens % tpn == 0;
  const int Tq = sp ? (T + tpn - 1) / tpn : T;  // rows per rank (the last may hold fewer)
  const int q0 = sp ? pc_.rank * Tq : 0;        // this rank's first row
  const int Tr = sp ? std::max(0, std::min(Tq, T - q0)) : T;
  // (over peer memory, set_sp: the reduce-scatter pulls the peers' rows and the all-gather stores into theirs; else
  // NCCL ring collectives)
  const bool sp_pull = sp && sp_p2p_;
  // token-split MLA prefill attention: the step's MLA layers attend this
  // rank's rows with every head (layer_mla)
  ts_on_ = ts_step_ = sp_pull && ts_p2p_ && in.idx_split == Tq;
  // (sp_fuse: the boundary in the three launches below; else as separate kernels: reduce-scatter, post-mapping, mixes
  // GEMM, pre-mapping, all-gather. The layer input is never FP8-quantized: the peers' FP8 buffers are not needed.)
  const bool sp_fuse = sp_pull && H == 4096 && hc == 4;
  // the owed sublayer sum (the MoE combine first) fused with the post-mapping of this rank's rows
  auto sp_reduce_post = [&](bf16* sq) {
    if (pend_moe) k::moe_combine(b.out, b.y, b.topk_w, b.pair_pos, nullptr, T, cfg_.top_k + 1, H, stream_);
    k::sp_rs_post(sp_peers_, tpn, pc_.rank, q0, Tr, H, sq, b.post + (size_t)q0 * hc, b.comb + (size_t)q0 * hc * hc,
                  sp_state_, stream_);
    pend_ar = false;
  };
  auto sp_reduce = [&] {  // the owed sublayer sum, this rank's rows only
    if (!pend_ar) return;
    if (pend_moe) k::moe_combine(b.out, b.y, b.topk_w, b.pair_pos, nullptr, T, cfg_.top_k + 1, H, stream_);
    if (sp_pull)
      k::sp_rs_pull(sp_peers_, tpn, pc_.rank, q0, Tr, H, sp_state_, stream_);
    else
      NCCL_CHECK(ncclReduceScatter(b.out, b.out + (size_t)q0 * H, (size_t)Tq * H, ncclBfloat16, ncclSum, comm_,
                                   stream_));
    pend_ar = false;
  };
  // Sublayer boundary: post-mapping of the previous sublayer output (b.out) unless first, then the pre-mapping of the
  // next sublayer into b.x. q4_gs > 0: also the NVFP4 input of an NVFP4 MoE (b.x4 / b.x4s, its global scale q4_gs)
  // when the cluster kernel runs (x4_ready_).
  auto boundary = [&](const HcW& hcw, const bf16* norm_w, bool has_prev, float q4_gs = 0.f) {
    if (fused_hc) {
      k::ArFuse ar;
      const k::ArFuse* arp = nullptr;
      if (has_prev && pend_ar && fuse_ar) {
        ar.pushed = ar_pushed_ && !pend_moe ? kArPushLevel : 0;
        ar_pushed_ = false;
        ar.peers = ar_peers_;
        ar.state = ar_epochs_ + ar_max_blocks_;
        ar.rank = pc_.rank;
        ar.tp = pc_.tp;
        ar.max_elems = ar_max_elems_;
        if (pend_moe) {
          ar.cm.y = b.y;
          ar.cm.w = b.topk_w;
          ar.cm.pair_pos = b.pair_pos;
          ar.cm.K1 = cfg_.top_k + 1;
          ar.cm.hidden = H;
        }
        ar.sk = (sk_pending ? 2 : 0) | (sk_on && !hc_last ? 1 : 0);
        sk_pending = (ar.sk & 1) != 0;
        if (rf_next_) {  // (RF: the router's copies, where the layer loop asked for them; ar_rf_x)
          CHECK(!pend_moe, "RF: an attention-side boundary");
          ar.cm.w = reinterpret_cast<const float*>(rf_next_);
        }
        arp = &ar;
        pend_ar = false;
      } else {
        CHECK(!ar_pushed_, "producer-pushed all-reduce without a fused boundary");
        CHECK(!sk_pending, "deferred Sinkhorn: the next boundary must take it");
        flush_ar();
      }
      const bool q4 = q4_gs > 0.f;
      // (the MoE combine's rows token-major: moe_tm_out_ of the MoE launch before)
      const bool tm = arp && arp->cm.y && moe_tm_out_;
      moe_tm_out_ = false;
      bool rf_done = false;
      x4_ready_ = k::mhc_fused(b.streams, has_prev ? b.out : nullptr, b.post, b.comb, b.x, nullptr, nullptr, hcw.fn,
                               hcw.base, hcw.scale, norm_w, b.hc_partial, b.hc_counter, T, H, hc, cfg_.rms_eps,
                               cfg_.hc_eps, cfg_.hc_iters, cfg_.hc_post_mult, cfg_.rms_eps, stream_, arp,
                               q4 ? b.x4 : nullptr, q4 ? b.x4s : nullptr, q4_gs, tm, &rf_done);
      rf_ready_ = rf_done;
      rf_set_ = rf_done ? rf_next_ : nullptr;
      return;
    }
    x4_ready_ = false;
    if (sp && sp_fuse) {
      // three separable parts: (1) the owed sum of the previous sublayer's partial outputs fused with the
      // post-mapping of this rank's rows (sp_reduce_post); a sublayer whose output is already this rank's rows in
      // b.out rows [q0, q0 + Tr) leaves pend_ar false and gets the post-mapping alone; (2) the mixes GEMM; (3) the
      // pre-mapping fused with the all-gather (x as stores into the peers, row by row: sp_pre_push)
      bf16* const sq = b.streams + (size_t)q0 * hc * H;
      if (pend_ar) {
        sp_reduce_post(sq);
      } else if (has_prev && Tr > 0) {
        k::mhc_post(sq, b.out + (size_t)q0 * H, b.post + (size_t)q0 * hc, b.comb + (size_t)q0 * hc * hc, Tr, H, hc,
                    stream_);
      }
      if (Tr > 0)
        gemm_->mm_bf16(sq, hc * H, hcw.fn, hc * H, b.mixes + (size_t)q0 * mix, mix, true, Tr, mix, hc * H, stream_);
      k::sp_pre_push(sp_peers_, tpn, pc_.rank, q0, Tr, H, b.post + (size_t)q0 * hc, b.comb + (size_t)q0 * hc * hc,
                     b.x, nullptr, nullptr, sq, b.mixes + (size_t)q0 * mix, hcw.base, hcw.scale, norm_w,
                     cfg_.rms_eps, cfg_.hc_eps, cfg_.hc_iters, cfg_.hc_post_mult, cfg_.rms_eps, b.hc_rs + q0,
                     sp_state_, stream_);
      return;
    }
    if (sp) {
      sp_reduce();
      bf16* const sq = b.streams + (size_t)q0 * hc * H;
      if (Tr > 0) {
        if (has_prev)
          k::mhc_post(sq, b.out + (size_t)q0 * H, b.post + (size_t)q0 * hc, b.comb + (size_t)q0 * hc * hc, Tr, H, hc,
                      stream_);
        gemm_->mm_bf16(sq, hc * H, hcw.fn, hc * H, b.mixes + (size_t)q0 * mix, mix, true, Tr, mix, hc * H, stream_);
        k::mhc_pre_apply(b.post + (size_t)q0 * hc, b.comb + (size_t)q0 * hc * hc, b.x + (size_t)q0 * H, nullptr,
                         nullptr, sq, b.mixes + (size_t)q0 * mix, hcw.base, hcw.scale, norm_w, Tr, H, hc,
                         cfg_.rms_eps, cfg_.hc_eps, cfg_.hc_iters, cfg_.hc_post_mult, cfg_.rms_eps, stream_,
                         b.hc_rs + q0);
      }
      if (sp_pull)
        k::sp_ag_push(sp_peers_, tpn, pc_.rank, Tq, T, H, sp_state_, stream_);
      else
        NCCL_CHECK(ncclAllGather(b.x + (size_t)q0 * H, b.x, (size_t)Tq * H, ncclBfloat16, comm_, stream_));
      return;
    }
    flush_ar();
    if (has_prev) k::mhc_post(b.streams, b.out, b.post, b.comb, T, H, hc, stream_);
    gemm_->mm_bf16(b.streams, hc * H, hcw.fn, hc * H, b.mixes, mix, true, T, mix, hc * H, stream_);
    k::mhc_pre_apply(b.post, b.comb, b.x, nullptr, nullptr, b.streams, b.mixes, hcw.base, hcw.scale, norm_w, T, H,
                     hc, cfg_.rms_eps, cfg_.hc_eps, cfg_.hc_iters, cfg_.hc_post_mult, cfg_.rms_eps, stream_,
                     b.hc_rs);
  };
  // KDA in_proj L2 prefetch (l2pf_async) at kL2pfKdaMinTokens .. kL2pfKdaMaxTokens tokens
  constexpr int kL2pfKdaMinTokens = 13, kL2pfKdaMaxTokens = 32;
  pf_ok_ = T >= kL2pfKdaMinTokens && T <= kL2pfKdaMaxTokens;
  pf_forked_ = false;
  // decode MoE layers: the fused kernel records its local experts per layer from kMoeHistMinTokens tokens, and the
  // next step prefetches them into L2 on pf_stream_ during the boundary before the layer at kMoePfMinTokens ..
  // kMoePfMaxTokens tokens (moe_pf_async)
  constexpr int kMoeHistMinTokens = 17, kMoePfMinTokens = 17, kMoePfMaxTokens = 64;
  const bool moe_hist_on = T >= kMoeHistMinTokens;
  const bool moe_pf_on = moe_hist_on && T >= kMoePfMinTokens && T <= kMoePfMaxTokens;
  // (the MoE-side boundary before an attention layer at <= kEarlyTriggerMaxTokens tokens launches its dependents early,
  // after its all-reduce push (k::mhc_set_et mode 3): before a KDA layer the in_proj's CTAs stage their weights during
  // the boundary's push / poll (layer_kda), before an MLA layer its first GEMVs (qkv_a / idx_wq on gemv_pre) stage
  // their slices during the boundary)
  constexpr int kEarlyTriggerMode = 3, kEarlyTriggerMaxTokens = 4;
  for (int li = 0; li < cfg_.num_layers; ++li) {
    const LayerW& L = w_.layers[li];
    if (L.attn == AttnType::kKDA)  // during the boundary below
      l2pf_async(L.kda.in_proj.w, (size_t)N_in_ * H * sizeof(bf16));
    k::mhc_set_et(li > 0 && T <= kEarlyTriggerMaxTokens ? kEarlyTriggerMode : 0);
    boundary(L.hc_attn, L.in_norm, li > 0);
    k::mhc_set_et(0);
    ar_push_req_ = fuse_ar;
    ar_push_T_ = T;
    if (L.attn == AttnType::kKDA)
      layer_kda(li, L, T, B, nd, dt);
    else
      layer_mla(li, mla_index_[li], L.mla, T, in.need_sel);
    ar_push_req_ = false;
    pend_ar = !ts_summed_;  // (token-split with every head's o_proj: this rank's rows are summed)
    pend_moe = false;
    ts_summed_ = false;
    int* const hist = b.moe_hist + (size_t)li * k::kMoeFp4HistStride;
    if (L.moe && moe_pf_on) moe_pf_async(L.moe_w, hist);  // (during the boundary)
    hc_last = li == cfg_.num_layers - 1;  // (the last boundary: mhc_post reads its comb)
    // (RF: the router's copies of this layer parity's set, where layer_moe takes moe_router_mma_ks at its RF size)
    rf_next_ = L.moe && rf_layer_ok(L.moe_w, T) ? rf_x_ + (size_t)(li & 1) * k::kRouterRfGroups * k::kRouterRfT * H
                                                 : nullptr;
    boundary(L.hc_ffn, L.post_norm, true,
             L.moe ? (L.moe_w.fmt == WFmt::kNvfp4 ? L.moe_w.gs13 : 0.f)
                   : (dense_fused(L.mlp, T) ? L.mlp.gate_up4.gs : 0.f));
    rf_next_ = nullptr;
    if (L.moe) {
      moe_hist_cur_ = moe_hist_on ? hist : nullptr;
      // (its consumer: the next layer's attention-side boundary, a fused one with the deferred Sinkhorn's input)
      moe_tm_req_ = fuse_ar && sk_on && li + 1 < cfg_.num_layers && k::mhc_moe_tm_ok(T, H, pc_.tp, cfg_.top_k + 1);
      layer_moe(L.moe_w, T);
      moe_tm_req_ = false;
      moe_hist_cur_ = nullptr;
      pend_ar = true;
      pend_moe = true;
    } else {
      layer_mlp(L.mlp, T);
      pend_ar = true;
      pend_moe = false;
    }
  }
  ts_on_ = false;  // (the MTP layer's pass over the step's tokens: ts_step_)
  if (pf_forked_) {  // (the prefetches ended long before; only the capture needs the join)
    CUDA_CHECK(cudaEventRecord(ev_pf_join_, pf_stream_));
    CUDA_CHECK(cudaStreamWaitEvent(stream_, ev_pf_join_, 0));
  }
  pf_ok_ = pf_forked_ = false;
  if (sp) {
    // sequence-parallel: the last post-mapping and the final norm on this rank's rows, then every rank's normalized
    // rows (the MTP layer's input, or b.x as scratch for the sampled rows)
    bf16* const sq = b.streams + (size_t)q0 * hc * H;
    bf16* const fin = mtp_ ? b.hfin : b.x;
    if (sp_fuse && pend_ar) {
      sp_reduce_post(sq);
    } else {
      sp_reduce();
      if (Tr > 0)
        k::mhc_post(sq, b.out + (size_t)q0 * H, b.post + (size_t)q0 * hc, b.comb + (size_t)q0 * hc * hc, Tr, H, hc,
                    stream_);
    }
    if (Tr > 0) k::hc_head_norm(fin + (size_t)q0 * H, sq, w_.final_norm, Tr, H, hc, cfg_.rms_eps, stream_);
    NCCL_CHECK(ncclAllGather(fin + (size_t)q0 * H, fin, (size_t)Tq * H, ncclBfloat16, comm_, stream_));
    if (S > 0) k::gather_rows(b.head, fin, in.sample_idx, S, H, stream_);
  } else {
    flush_ar();
    k::mhc_post(b.streams, b.out, b.post, b.comb, T, H, hc, stream_);
    if (mtp_) {
      // the MTP layer consumes the final hidden state of every token
      k::hc_head_norm(b.hfin, b.streams, w_.final_norm, T, H, hc, cfg_.rms_eps, stream_);
      if (S > 0) k::gather_rows(b.head, b.hfin, in.sample_idx, S, H, stream_);
    } else if (S > 0) {
      k::gather_rows(b.last_streams, b.streams, in.sample_idx, S, hc * H, stream_);
      k::hc_head_norm(b.head, b.last_streams, w_.final_norm, S, H, hc, cfg_.rms_eps, stream_);
    }
  }
  // LM head, vocabulary-parallel over the attention ranks: greedy / plain temperature rows without logprobs as
  // vocab-parallel argmax (sampling rows: Gumbel-max with noise keyed by the row's seed and step), no logits
  // all-gather; every rank gets the tokens
  if (S > 0 && in.head >= 0) {
    const int vtp = pc_.tp;
    CHECK(in.head <= 1 && (vtp == 1 || ar_enabled_) && S <= k::kMaxKeyRows && w_.lm_head.n % 4 == 0,
          "GlmStep: the vocab-parallel head");
    const int Vl = w_.lm_head.n;
    if (!head_keys(b.head, S, in.head == 1 ? in.sample_params : nullptr, b.sample_keys, b.out_tokens)) {
      gemm_->mm_bf16(b.head, H, w_.lm_head.w, H, b.logits_local, Vl, true, S, Vl, H, stream_);
      k::argmax_keys(b.sample_keys, b.logits_local, S, Vl, pc_.rank * Vl, stream_,
                     in.head == 1 ? in.sample_params : nullptr);
      if (vtp > 1)
        k::custom_max_u64(b.sample_keys, S, ar_peers_, ar_epochs_ + ar_max_blocks_ + 4, pc_.rank, pc_.tp, stream_,
                          b.out_tokens);
      else
        k::keys_to_tokens(b.out_tokens, b.sample_keys, S, stream_);
    }
  }
  if (mtp_ && in.head >= 0) mtp_forward(T, S, in.Bs, in.need_sel);
}

// ------------------------------------------------------------------ KDA
void GlmStep::layer_kda(int li, const LayerW& L, int T, int B, int nd, int dt) {
  Buffers& b = *b_;
  const KdaW& W = L.kda;
  const int D = cfg_.kda_head_dim, HD = Hl_kda_ * D, H = cfg_.hidden;
  const int ki = kda_index_[li];
  const KdaPool& P = pools_.kda[ki];
  const int Tin = T;
  // steps of up to kInprojWideMaxTokens tokens: the streamed in_proj GEMV on up to kInprojStreamCtas CTAs (48 rows each
  // instead of 64 on 101) instead of gemv_stream's one-wave default of 116
  constexpr int kInprojStreamCtas = 148, kInprojWideMaxTokens = 4;
  // tensor parallel (52.6 MB per rank): gemv_tma up to kInprojTmaMaxTokens rows
  constexpr int kInprojTmaMaxTokens = 4;
  // tensor parallel, 5-8 rows: the in_proj on gemv_pipe8 (the 48-row gemv_pipe with 8-row activation stages: 4 ring
  // stages instead of 3; bitwise gemv_pipe)
  // (LP: at one decode sequence of <= 4 tokens the in_proj GEMV (gemv_tma) writes its rows into the Lamport copy
  // kda_lp_ of this layer's parity instead of b.proj, and the KDA decode (its late co instance's G8 form) polls the
  // words it reads there instead of waiting for the in_proj grid's completion (it waits for it only before its first
  // global store) and resets them; the f_a / g_a columns of the other copy -- read by every CTA of the previous KDA
  // layer -- are reset by this layer's decode. The sentinel 0xFFFF is a NaN the GEMV's bf16 conversions never write.
  // Bitwise.)
  const bool lp_tma = Tin > 0 && Tin <= kInprojTmaMaxTokens && k::gemv_tma_fits(Tin, N_in_, H);
  bf16* const lp_copy = kda_lp_ ? kda_lp_ + (size_t)(ki & 1) * 4 * N_in_ : nullptr;
  const bool lp = lp_copy && lp_tma && nd == 1 && B == 1 && Tin == dt && dt <= 4 && nd * dt < kKdaPreGatesMin &&
                  k::kda_decode_lp_path(1, Hl_kda_, D, dt, N_in_, lp_copy, false, false);
  if (Tin > 0 && Tin <= kInprojTmaMaxTokens && k::gemv_tma_fits(Tin, N_in_, H)) {
    k::GemvPreArgs a;
    a.A = b.x;
    a.lda = H;
    a.W = W.in_proj.w;
    a.ldw = H;
    a.D = lp ? lp_copy : b.proj;
    a.ldd = N_in_;
    a.M = Tin;
    a.N = N_in_;
    a.K = H;
    a.late_trigger = true;
    // the last kInprojLateCtas CTAs (the MoE-side boundary's CTAs at <= 4 tokens, whose SMs they get only after it)
    // take kInprojLateRows rows each (two 16-row tiles instead of three)
    constexpr int kInprojLateCtas = 32, kInprojLateRows = 32;
    // (behind a boundary that launched this in_proj early, k::mhc_et_used(): its weights stage while the boundary runs,
    // in the register-staged form with its first kStagedRw stages in registers and its ring paced at kStagedPd)
    constexpr int kStagedPd = 3, kStagedRw = 4;
    const bool staged = Tin <= 4 && k::mhc_et_used() != 0;
    if (kInprojLateCtas < sm_count_)
      k::gemv_tma_uneven(a, sm_count_, kInprojLateCtas, kInprojLateRows, stream_, staged ? kStagedPd : 0,
                         staged ? kStagedRw : 0);
    else
      k::gemv_tma(a, sm_count_, stream_);
  } else if (Tin > 0 && Tin <= 8 && Tin >= kGemvPipeMinRows && k::gemv_pipe8_fits(Tin, N_in_, H, sm_count_)) {
    // (tensor parallel, 5-8 rows: what bf16_dense picks -- gemv_pipe -- with 8-row activation stages and 4 ring stages)
    k::GemvPreArgs a;
    a.A = b.x;
    a.lda = H;
    a.W = W.in_proj.w;
    a.ldw = H;
    a.D = b.proj;
    a.ldd = N_in_;
    a.M = Tin;
    a.N = N_in_;
    a.K = H;
    a.late_trigger = true;  // (as gemv_pipe in bf16_dense)
    k::gemv_pipe8(a, sm_count_, stream_);
  } else if (Tin > 0) {
    bf16_dense(b.x, H, W.in_proj.w, H, b.proj, N_in_, false, Tin, N_in_, H, true, 0,
               Tin <= kInprojWideMaxTokens ? kInprojStreamCtas : 0);
  }
  // the runtime's batch order: the context sequences [0, nc) on tokens [0, Tc), then the nd decode sequences of dt
  // tokens each (the kernels address a sequence's rows by its cu_seqlens entry)
  const int nc = B - nd, Tc = T - nd * dt;
  // decode sequences: one fused kernel; for many decode tokens the gate projections run as two GEMMs first (tensor
  // cores instead of per-CTA dot products)
  if (nd > 0) {
    const k::KdaSeqMeta meta = kda_meta(nc);
    const int Td = nd * dt;
    const bool pre = Td >= kKdaPreGatesMin;
    if (pre) {
      const bf16* fa = b.proj + (size_t)Tc * N_in_ + 3 * HD + Hl_kda_;
      gemm_->mm_bf16(fa, N_in_, W.f_b.w, D, b.g1 + (size_t)Tc * HD, HD, false, Td, HD, D, stream_);
      gemm_->mm_bf16(fa + D, N_in_, W.g_b.w, D, b.g2 + (size_t)Tc * HD, HD, false, Td, HD, D, stream_);
    }
    CHECK(!lp || !pre, "KDA LP: the precomputed gates read b.proj");
    k::kda_decode_fused(b.core_n, lp ? lp_copy : b.proj, N_in_, W.conv, P.conv, W.f_b.w, W.g_b.w, W.a_log, W.dt_bias,
                        W.o_norm, P.state, meta, nd, Hl_kda_, D, cfg_.kda_lower_bound, cfg_.rms_eps, stream_, dt,
                        P.rec, P.crec, mtp_n_, pre ? b.g1 : nullptr, pre ? b.g2 : nullptr, false,
                        lp ? kda_lp_ + (size_t)((ki + 1) & 1) * 4 * N_in_ + 3 * HD + Hl_kda_ : nullptr);
  }
  // context sequences: tokens [0, Tc)
  if (nc > 0) {
    const bf16* fa = b.proj + 3 * HD + Hl_kda_;
    gemm_->mm_bf16(fa, N_in_, W.f_b.w, D, b.g1, HD, false, Tc, HD, D, stream_);
    gemm_->mm_bf16(fa + D, N_in_, W.g_b.w, D, b.g2, HD, false, Tc, HD, D, stream_);
    k::KdaSeqMeta meta = kda_meta(0);
    meta.replay = nullptr;
    k::kda_conv(b.conv_out, b.proj, N_in_, W.conv, P.conv, meta, nc, 3 * HD, cfg_.kda_conv, stream_, Tc, 0);
    if (in_->num_chunks > 0)
      k::kda_chunked(b.core, b.conv_out, b.proj + 3 * HD, N_in_, b.g1, W.a_log, W.dt_bias, P.state, meta,
                     in_->kda_chunks, in_->kda_seq_chunks, in_->num_chunks, b.kda_scratch, nc, Hl_kda_, D,
                     cfg_.kda_lower_bound, stream_);
    else
      k::kda_recurrent(b.core, b.conv_out, b.proj + 3 * HD, N_in_, b.g1, W.a_log, W.dt_bias, P.state, meta, nc,
                       Hl_kda_, D, cfg_.kda_lower_bound, stream_);
    k::kda_out_norm(b.core_n, b.core, b.g2, W.o_norm, Tc, Hl_kda_, D, cfg_.rms_eps, stream_);
  }
  // tensor parallel: the o_proj at up to kOprojLeanMaxTokens rows on gemv_pre_lean: 128 CTAs of 32 rows (TMA staged)
  // instead of gemv_pre's 86 CTAs of 48 rows (above that, on gemv_pipe: its push epoch loaded before the PDL wait,
  // gemv_pipe_ep)
  constexpr int kOprojLeanMaxTokens = 4;
  if (Tin > 0)
    bf16_dense(b.core_n, HD, W.o_proj.w, HD, b.out, H, false, Tin, H, HD, true, 0, 0, Tin <= kOprojLeanMaxTokens,
               /*ep=*/true);
}

// ------------------------------------------------------------------ dense layers
// Decode-sized per-head projection of the MLA's absorb (q -> q_lat over w_kc^T, kept transposed as w_kcT): one gemv_pre
// launch over the Hm heads (grid rows = heads), its weight rows staged before the PDL wait, instead of cuBLASLt's
// batched GEMM. Not bitwise cuBLASLt (the 8 warps' k-slice partials are summed in warp order). Takes up to 16 tokens of
// 16 local heads.
bool GlmStep::head_gemv(const bf16* A, int lda, int sA, const bf16* W, int ldw, long long sW, bf16* D, int ldd,
                        int sD, int M, int N, int K, int batch, bool late, int ctas_call, int carveout) {
  const int ctas = ctas_call > 0 ? std::min(ctas_call, sm_count_) : sm_count_;
  if (!W || !k::gemv_pre_fits(M, N, K, ctas, batch)) return false;
  k::GemvPreArgs a;
  a.A = A;
  a.lda = lda;
  a.W = W;
  a.ldw = ldw;
  a.D = D;
  a.ldd = ldd;
  a.d_f32 = false;
  a.M = M;
  a.N = N;
  a.K = K;
  a.batch = batch;
  a.sA = sA;
  a.sW = sW;
  a.sD = sD;
  a.late_trigger = late;
  k::gemv_pre(a, ctas, stream_, carveout);
  return true;
}

void GlmStep::bf16_dense(const bf16* A, int lda, const bf16* W, int ldw, void* D, int ldd, bool d_f32, int M, int N,
                         int K, bool pipe_ok, int pre_ctas_call, int stream_cap, bool lean, bool ep) {
  // the attention's o_proj into b.out ahead of a fused all-reduce boundary: the GEMV kernels push their outputs into
  // every rank's buffer themselves (ar_push_req_)
  const bool push = ar_push_req_ && !d_f32 && D == b_->out && ldd == cfg_.hidden && N == ldd && M == ar_push_T_ &&
                    pc_.tp == 4;
  // (the GEMV must trigger the boundary only after its own wait: the boundary may then read the outputs of every
  // earlier kernel before its wait, ArFuse.pushed >= 2)
  auto set_push = [&](k::GemvPreArgs& a) {
    if (!push || !a.late_trigger) return;
    for (int q = 0; q < 4; ++q) a.push.buf[q] = ar_peers_.buf[q];
    a.push.state = ar_epochs_ + ar_max_blocks_;
    a.push.rank = pc_.rank;
    a.push.max_elems = ar_max_elems_;
    ar_pushed_ = true;
  };
  if (lean && !d_f32 && k::gemv_pre_lean_fits(M, N, K)) {
    k::GemvPreArgs a;
    a.A = A;
    a.lda = lda;
    a.W = W;
    a.ldw = ldw;
    a.D = D;
    a.ldd = ldd;
    a.M = M;
    a.N = N;
    a.K = K;
    a.late_trigger = true;  // (as gemv_pre)
    set_push(a);
    k::gemv_pre_lean(a, stream_);
    return;
  }
  const int pre_ctas =
      M > 8 ? 0 : pre_ctas_call > 0 ? std::min(pre_ctas_call, sm_count_) : gemv_pre_ctas(sm_count_);
  if (pre_ctas > 0 && k::gemv_pre_fits(M, N, K, pre_ctas)) {
    k::GemvPreArgs a;
    a.A = A;
    a.lda = lda;
    a.W = W;
    a.ldw = ldw;
    a.D = D;
    a.ldd = ldd;
    a.d_f32 = d_f32;
    a.M = M;
    a.N = N;
    a.K = K;
    // the next kernel launches once this one's wait is over: launched at its start, it would stay resident through the
    // whole previous kernel; launched after the wait it still stages its weights before this GEMV ends
    a.late_trigger = true;
    set_push(a);
    k::gemv_pre(a, pre_ctas, stream_);
    return;
  }
  // the attention's pushed o_proj too large for gemv_pre (MLA: 33.5 MB per rank) of up to kOprojTmaMaxTokens rows:
  // gemv_tma's ring on up to kOprojTmaCtas CTAs (whole 16-row tiles each) instead of gemv_stream
  constexpr int kOprojTmaMaxTokens = 4, kOprojTmaCtas = 128;
  if (push && M <= kOprojTmaMaxTokens && k::gemv_tma_fits(M, N, K)) {
    k::GemvPreArgs a;
    a.A = A;
    a.lda = lda;
    a.W = W;
    a.ldw = ldw;
    a.D = D;
    a.ldd = ldd;
    a.M = M;
    a.N = N;
    a.K = K;
    a.late_trigger = true;
    set_push(a);
    // (paced: before its PDL wait each CTA keeps at most kOprojPace ring stages in flight, the rest issued once the
    // wait has passed, the activation rows at the wait)
    constexpr int kOprojPace = 3;
    k::gemv_tma(a, std::min(kOprojTmaCtas, sm_count_), stream_, 4, nullptr, 16, kOprojPace);
    return;
  }
  const int stream_ctas = stream_cap > 0 ? std::min(stream_cap, sm_count_) : gemv_stream_ctas(sm_count_);
  if (stream_ctas > 0 && M <= kGemvStreamMaxRows && k::gemv_stream_fits(M, N, K, stream_ctas, stream_cap)) {
    k::GemvPreArgs a;
    a.A = A;
    a.lda = lda;
    a.W = W;
    a.ldw = ldw;
    a.D = D;
    a.ldd = ldd;
    a.d_f32 = d_f32;
    a.M = M;
    a.N = N;
    a.K = K;
    a.late_trigger = true;  // (as gemv_pre)
    set_push(a);
    k::gemv_stream(a, stream_ctas, stream_, stream_cap);
    return;
  }
  // rows the pipelined GEMV takes (kGemvPipeMinRows .. 16; one wave over all SMs); pipe_ok = false: the call site opts
  // out
  if (pipe_ok && M >= kGemvPipeMinRows && M <= k::kGemvPreMaxRows && k::gemv_pipe_fits(M, N, K, sm_count_)) {
    k::GemvPreArgs a;
    a.A = A;
    a.lda = lda;
    a.W = W;
    a.ldw = ldw;
    a.D = D;
    a.ldd = ldd;
    a.d_f32 = d_f32;
    a.M = M;
    a.N = N;
    a.K = K;
    a.late_trigger = true;  // (as gemv_pre)
    set_push(a);
    if (ep && a.push.state)
      k::gemv_pipe_ep(a, sm_count_, stream_);
    else
      k::gemv_pipe(a, sm_count_, stream_);
    return;
  }
  gemm_->mm_bf16(A, lda, W, ldw, D, ldd, d_f32, M, N, K, stream_);
}

// L2 prefetch of a KDA layer's in_proj weights (52.6 MB per GPU at TP4) during the attention boundary before it
// (latency-bound: the fused all-reduce and the mHC mixing leave HBM idle), so that the in_proj GEMM streams them from
// L2 (steps of 13 .. 32 tokens, pf_ok_): 64 CTAs of 32 threads, 32 KB per bulk prefetch.
void GlmStep::l2pf_async(const void* p, size_t bytes) {
  if (!pf_ok_) return;
  constexpr int kCtas = 64;
  constexpr size_t kChunk = 32 * 1024;
  CUDA_CHECK(cudaEventRecord(ev_pf_, stream_));
  CUDA_CHECK(cudaStreamWaitEvent(pf_stream_, ev_pf_, 0));
  k::l2_prefetch(p, bytes, kCtas, kChunk, pf_stream_);
  pf_forked_ = true;
}

// ------------------------------------------------------------------ sparse MLA
void GlmStep::layer_mla(int li, int mi, const MlaW& W, int T, bool need_sel, const int* tok_slot,
                        const int* positions, bool share) {
  Buffers& b = *b_;
  const StepIn& in = *in_;
  const int H = cfg_.hidden, ql = cfg_.q_lora, kl = cfg_.kv_lora, Hm = Hl_mla_;
  const int nope = cfg_.qk_nope, vh = cfg_.v_head;
  const int kgw_ld = 2 * cfg_.idx_head_dim + cfg_.idx_heads;
  const MlaPool& P = pools_.mla[mi];
  // decode-sized passes (the fork): the indexer branch (idx_kgw, cache update, idx_wq, selection) on the side
  // stream, overlapping qkv_a / norms and q_b / absorb; joined before the attention. Same kernels and inputs: bitwise
  // the serial order's results.
  const bool fork = T <= kMlaForkMaxTokens && need_sel && in.idx_split == 0;
  // run f on the side stream (with its own cuBLASLt workspace)
  auto on_side = [&](auto&& f) {
    ScopedSwap<cudaStream_t> side_st(stream_, side_stream_);
    ScopedSwap<std::unique_ptr<Gemm>> side_gemm(gemm_, gemm2_);
    f();
  };
  if (!tok_slot) tok_slot = in.token_seq;
  if (!positions) positions = in.positions;
  const bool own = positions == in.positions && tok_slot == in.token_seq;  // (the step's own tokens)
  k::MlaTokMeta meta = mla_meta(tok_slot, positions);
  // the step's positions / batch rows, and the chained MTP passes' device-built ones when final at the pass's start
  // (mtp_meta_pre_): decode kernels may read them before their PDL wait (the step-metadata kernel, the forward's first
  // launch and a plain one, writes the step's, so every early-launched kernel after it sees them complete)
  meta.pre = own || mtp_meta_pre_;
  // early (with the fork): the fused cache update right after the indexer key projection on the side stream, for the
  // indexer's part only (pool keys: they need only that projection), the norms writing the latent rows into the cache
  // themselves
  const bool early = fork && T <= k::kCacheFuseMaxT;
  // token-split attention (ts_on_: the decoder's MLA layers of a sequence-parallel step): this rank attends its rows
  // [ts_q0, ts_q0 + ts_tr) with every head, one CTA per token; the selection split's rows are the same rows (no
  // all-gather)
  const int ts_tq = cdiv(T, pc_.tp), ts_q0 = pc_.rank * ts_tq;
  const int ts_tr = std::max(0, std::min(ts_tq, T - ts_q0));
  const bool ts = (li >= 0 ? ts_on_ : ts_step_) && !share && own && in.num_attn_tiles > 0 &&
                  W.w_kc_all != nullptr && W.w_vc_all != nullptr && in.idx_split == ts_tq &&
                  k::sparse_mla_fmla_on(cfg_.num_heads) &&
                  k::sparse_mla_fmla_ws_bytes(ts_tq, cfg_.num_heads) <= b.scores_bytes;
  auto fork_side = [&] {
    CUDA_CHECK(cudaEventRecord(ev_fork_, stream_));
    CUDA_CHECK(cudaStreamWaitEvent(side_stream_, ev_fork_, 0));
    on_side([&] {
      bf16_dense(b.x, H, W.idx_kgw.w, H, b.kgw, kgw_ld, true, T, kgw_ld, H);
      if (early)
        CHECK(k::mla_cache_update(P.lat, P.idx, nullptr, b.kgw, kgw_ld, W.idx_kn_w, W.idx_kn_b, W.idx_ape, meta, T,
                                  stream_),
              "fused cache update");
    });
  };
  // qkv_a's weights all in shared memory (gemv_pre on up to kQkvaPreCtas CTAs, staged during the boundary before the
  // PDL wait) instead of the streamed GEMV, for steps of up to kQkvaPreMaxTokens tokens
  constexpr int kQkvaPreCtas = 136, kQkvaPreMaxTokens = 4;
  if (fork) fork_side();
  bf16_dense(b.x, H, W.qkv_a16.w, H, b.qkva, ql + kl, false, T, ql + kl, H, true,
             T <= kQkvaPreMaxTokens ? kQkvaPreCtas : 0);
  // q / latent norms (q_resid also feeds the indexer); early: the latent rows straight into the cache
  // (qn: a pass with the shared selection -- q_resid then feeds only mla_qba -- runs both norms in mla_qba's NQ
  // instance instead (k::QbaNorm: bitwise mla_qkv_norm), one dependent hop fewer)
  const bool qn = share && early && !ts && W.w_kcT != nullptr && Hm == 16 &&
                  k::mla_qba_fits(T, Hm, ql, nope, kl);
  if (!qn)
    k::mla_qkv_norm(b.q_resid, ql, nullptr, nullptr, b.latent, kl, b.qkva, ql + kl, W.q_a_norm, W.kv_a_norm, T, ql,
                    kl, cfg_.rms_eps, stream_, early ? P.lat : nullptr, early ? &meta : nullptr);
  if (!fork) bf16_dense(b.x, H, W.idx_kgw.w, H, b.kgw, kgw_ld, true, T, kgw_ld, H);
  // share (chained MTP passes): the selection of the draft loop's first pass is reused, so neither the indexer query
  // nor the selection runs (the cache update still does)
  const bool sel_here = need_sel && !share;
  // with the fork: the indexer query projection (idx_wq) on the main stream right after the norms, concurrent with the
  // side stream's cache update (both only need the norms' outputs); the scoring waits for both
  const bool idxq_main = fork && sel_here;
  // (with the fork: the side stream and its cuBLASLt from here to the join below)
  std::optional<ScopedSwap<cudaStream_t>> side_st;
  std::optional<ScopedSwap<std::unique_ptr<Gemm>>> side_gemm;
  if (fork) {
    CUDA_CHECK(cudaEventRecord(ev_q_, stream_));
    CUDA_CHECK(cudaStreamWaitEvent(side_stream_, ev_q_, 0));
    if (idxq_main) {
      bf16_dense(b.q_resid, ql, W.idx_wq.w, ql, b.q_idx, W.idx_wq.n, false, T, W.idx_wq.n, ql);
      CUDA_CHECK(cudaEventRecord(ev_qi_, stream_));
    }
    side_st.emplace(stream_, side_stream_);
    side_gemm.emplace(gemm_, gemm2_);
  }
  // the step's latent rows and indexer keys into the caches (one launch for decode-sized steps)
  if (early) {
  } else if (!k::mla_cache_update(P.lat, P.idx, b.latent, b.kgw, kgw_ld, W.idx_kn_w, W.idx_kn_b, W.idx_ape, meta, T,
                                  stream_)) {
    k::mla_append(P.lat, P.idx, b.latent, b.kgw, kgw_ld, W.idx_kn_w, W.idx_kn_b, meta, T, stream_);
    k::kpool_update(P.idx, W.idx_ape, meta, T, stream_);
  }
  // (with the fork the main stream's q_b and absorb start after the indexer's scoring kernel: its completion as an
  // event, recorded by the tiled selection)
  const bool score_ev_on = fork;
  bool score_ev = false;
  // the score rows' length: the step's (a context step's: its longest context's pools), else every pool
  const int score_stride = in.score_stride > 0 ? in.score_stride : b.max_pools;
  // the decode top-k folded into the attention's 16-CTA clusters (k::sparse_mla_attn_sel) for the fork's decode passes
  // of up to kSelkMaxTokens tokens with mla_qba (whose SIG instance it waits for), launched on the side stream right
  // after the scoring in the top-k's place. Only where the launch is certain (k::sparse_mla_attn_sel_ok: the score rows
  // span every pool, the clusters fit): its scoring leaves the top-k to it.
  constexpr int kSelkMaxTokens = 4;
  const bool selk_ok = fork && sel_here && T <= kSelkMaxTokens && !ts && Hm == 16 && W.w_kcT != nullptr &&
                       k::mla_qba_fits(T, Hm, ql, nope, kl) && in.idx_split == 0 && own &&
                       k::sparse_mla_attn_sel_ok(T, Hm, score_stride, b.max_pools);
  bool selk = false;               // (launched)
  cudaEvent_t selk_pev = ev_sel_;  // (nullptr once recorded by the launch)
  unsigned* selk_cnt = b.selk_cnt + 2 * (size_t)mi;
  auto selk_launch = [&](int stride, bool one_run) {
    if (!one_run) return false;
    return k::sparse_mla_attn_sel(b.o_lat, b.q_lat, P.lat, b.sel, meta, T, Hm, 1.f / sqrtf((float)nope), stream_,
                                  b.scores, stride, b.max_pools, selk_cnt, 4 * Hm, &selk_pev);
  };
  // indexer selection is only needed once some token sees more than 512 complete pools
  const auto& tiles = in.h_idx_tiles;
  if (sel_here) {
    // (tensor parallelism, split selection: only this rank's rows are scored here)
    const int Cs = in.idx_split, rs = pc_.rank;
    const bool split_rows = Cs > 0 && own && (int)tiles.size() > 0;
    if (idxq_main) {
      CUDA_CHECK(cudaStreamWaitEvent(stream_, ev_qi_, 0));
    } else if (split_rows) {
      const int r0 = std::min(T, rs * Cs), n = std::min(T, r0 + Cs) - r0;
      if (n > 0)
        bf16_dense(b.q_resid + (size_t)r0 * ql, ql, W.idx_wq.w, ql, b.q_idx + (size_t)r0 * W.idx_wq.n, W.idx_wq.n,
                   false, n, W.idx_wq.n, ql);
    } else {
      bf16_dense(b.q_resid, ql, W.idx_wq.w, ql, b.q_idx, W.idx_wq.n, false, T, W.idx_wq.n, ql);
    }
  }
  if (sel_here) {
    const int ihd = cfg_.idx_heads * cfg_.idx_head_dim;
    if (own && !tiles.empty()) {
      // the step's own tokens: query-tiled scoring (pool keys streamed once per tile), in runs of whole tiles whose
      // score rows fit the scratch
      const int rows_cap = (int)(((size_t)b.score_rows * b.max_pools) / score_stride);
      size_t ia = 0, ib = tiles.size();
      const int C = in.idx_split, r = pc_.rank;
      if (C > 0) {  // this rank's tiles: those starting in [r C, (r + 1) C)
        auto first_at = [&](int row) {
          return (size_t)(std::lower_bound(tiles.begin(), tiles.end(), row,
                                           [](const int2& a, int v) { return a.x < v; }) -
                          tiles.begin());
        };
        ia = first_at(r * C);
        ib = first_at((r + 1) * C);
      }
      for (size_t i0 = ia; i0 < ib;) {
        const int row0 = tiles[i0].x;
        size_t i1 = i0 + 1;
        while (i1 < ib && tiles[i1].x + tiles[i1].y - row0 <= rows_cap) ++i1;
        const int ntok = tiles[i1 - 1].x + tiles[i1 - 1].y - row0;
        // (SEL: one run of every token)
        const bool one_run = selk_ok && C == 0 && i0 == ia && i1 == ib && row0 == 0 && ntok == T;
        k::kpool_select_tiled(b.sel, b.scores, b.q_idx, b.kgw + 2 * cfg_.idx_head_dim, kgw_ld, P.idx, meta,
                              in.idx_tiles + i0, (int)(i1 - i0), row0, ntok, score_stride, cfg_.idx_heads, stream_,
                              (int64_t)rc_.max_num_tokens * cfg_.idx_heads, P.idx_rows,
                              score_ev_on ? ev_score_ : nullptr, one_run);
        if (one_run) {
          selk = selk_launch(score_stride, true);
          CHECK(selk, "sparse_mla_attn_sel: not launched after a score-only selection");
        }
        score_ev = score_ev_on;
        i0 = i1;
      }
      // every rank's rows to every rank (in place: rank r's block at r C); with the token-split attention each rank
      // reads only its own rows (the MTP layer's selections also seed the chained passes' shared selection: gathered)
      if (C > 0 && (!ts || li < 0))
        NCCL_CHECK(ncclAllGather(b.sel + (size_t)r * C * k::kSelPools, b.sel, (size_t)C * k::kSelPools, ncclInt32,
                                 comm_, stream_));
    } else if (T <= b.max_unit_tiles && T <= b.score_rows) {
      // other tokens (the chained MTP passes: one per sequence): one-token tiles
      k::kpool_select_tiled(b.sel, b.scores, b.q_idx, b.kgw + 2 * cfg_.idx_head_dim, kgw_ld, P.idx, meta,
                            b.unit_tiles, T, 0, T, score_stride, cfg_.idx_heads, stream_,
                            (int64_t)rc_.max_num_tokens * cfg_.idx_heads, P.idx_rows,
                            score_ev_on ? ev_score_ : nullptr);
      score_ev = score_ev_on;
    } else {
      for (int t0 = 0; t0 < T; t0 += b.score_rows) {
        const int n = std::min(b.score_rows, T - t0);
        k::MlaTokMeta m2 = mla_meta(tok_slot + t0, positions + t0);
        k::kpool_select(b.sel + (size_t)t0 * k::kSelPools, b.scores, b.q_idx + (size_t)t0 * ihd,
                        b.kgw + (size_t)t0 * kgw_ld + 2 * cfg_.idx_head_dim, kgw_ld, P.idx, m2, n, b.max_pools,
                        cfg_.idx_heads, stream_);
      }
    }
  }
  if (fork) {  // back on the main stream; the side stream's work joins before the attention
    side_gemm.reset();
    side_st.reset();
    // (SEL: ev_sel_ recorded by the attention's launch as its programmatic event)
    if (!selk || selk_pev) CUDA_CHECK(cudaEventRecord(ev_sel_, side_stream_));
  }
  if (score_ev) CUDA_CHECK(cudaStreamWaitEvent(stream_, ev_score_, 0));
  // (token-split with every head's q_b: q of every head for this rank's rows only, below)
  const bool ts_qb = ts && W.q_b16_all != nullptr;
  // (q_b and the absorb in one launch, below: k::mla_qba, one 4-CTA cluster per local head, bitwise the two gemv_pre
  // launches, for decode-sized tensor-parallel passes of up to 8 tokens)
  const bool qba = !ts && W.w_kcT != nullptr && Hm == 16 && k::mla_qba_fits(T, Hm, ql, nope, kl);
  CHECK(!qn || qba, "MLA: the NQ norms without mla_qba");
  if (!ts_qb && !qba) bf16_dense(b.q_resid, ql, W.q_b16.w, ql, b.q, Hm * nope, false, T, Hm * nope, ql);
  if (ts) {
    const int Hf = cfg_.num_heads;
    if (ts_qb) {
      // this rank's x is consumed (the peers' output rows may land in it), then q of every head
      k::mla_ts_ready(ts_peers_, pc_.tp, pc_.rank, ts_state_, stream_);
      if (ts_tr > 0)
        bf16_dense(b.q_resid + (size_t)ts_q0 * ql, ql, W.q_b16_all, ql, b.q, Hf * nope, false, ts_tr, Hf * nope, ql);
    } else {
      // every rank's heads of this rank's rows (b.out: [ts_tq][Hf nope], free until o_proj)
      k::mla_ts_q_push(ts_peers_, pc_.tp, pc_.rank, b.q, T, ts_tq, Hm * nope, ts_state_, stream_);
    }
    if (ts_tr > 0) {
      gemm_->mm_bf16_batched(ts_qb ? b.q : b.out, Hf * nope, nope, W.w_kc_all, kl, (long long)nope * kl, false,
                             b.q_lat, Hf * kl, kl, ts_tr, kl, nope, Hf, stream_);
      k::MlaTokMeta mq = meta;
      mq.token_seq += ts_q0;
      mq.positions += ts_q0;
      CHECK(k::sparse_mla_attn_fmla(b.o_lat, b.q_lat, P.lat, P.lat_rows, b.sel + (size_t)ts_q0 * k::kSelPools, mq,
                                    ts_tr, Hf, 1.f / sqrtf((float)nope), b.scores, b.scores_bytes, sm_count_,
                                    stream_),
            "token-split MLA attention");
      gemm_->mm_bf16_batched(b.o_lat, Hf * kl, kl, W.w_vc_all, kl, (long long)vh * kl, true, b.o, Hf * vh, vh, ts_tr,
                             vh, kl, Hf, stream_);
    }
    if (ts_qb && li >= 0 && W.o_proj16_all != nullptr) {
      // every head's o_proj on this rank's rows: their b.out rows hold the sublayer's sum (no reduce-scatter at the
      // boundary); every rank's projections must be done before the boundary's all-gather stores into its x
      if (ts_tr > 0)
        bf16_dense(b.o, Hf * vh, W.o_proj16_all, Hf * vh, b.out + (size_t)ts_q0 * H, H, false, ts_tr, H, Hf * vh);
      k::mla_ts_wait_ready(ts_peers_, pc_.tp, pc_.rank, ts_state_, stream_);
      ts_summed_ = true;
      return;
    }
    // each rank's head columns of these rows back: b.x = this rank's heads' output, every row
    k::mla_ts_o_push(ts_peers_, pc_.tp, pc_.rank, b.o, ts_tr, ts_tq, Hm * vh, ts_state_, stream_, ts_qb);
    bf16_dense(b.x, Hm * vh, W.o_proj16.w, Hm * vh, b.out, H, false, T, H, Hm * vh);
    return;
  }
  // absorb: q_lat[t, h] = q[t, h] @ w_kc[h]  (w_kc[h] is [nope, kv_lora])
  CHECK(!selk || qba, "sparse_mla_attn_sel without mla_qba");
  if (qba) {
    k::QbaNorm nqa;
    if (qn) {
      nqa.wq = W.q_a_norm;
      nqa.x_kv = b.qkva + ql;
      nqa.wkv = W.kv_a_norm;
      nqa.kv_cache = P.lat;
      nqa.cm = meta;
      nqa.eps = cfg_.rms_eps;
    }
    CHECK(k::mla_qba(qn ? b.qkva : b.q_resid, qn ? ql + kl : ql, W.q_b16.w, W.w_kcT, b.q_lat, Hm * kl, T, Hm, stream_,
                     selk ? selk_cnt : nullptr, /*early=*/false, qn ? &nqa : nullptr),
          "mla_qba");
  } else if (!head_gemv(b.q, Hm * nope, nope, W.w_kcT, nope, (long long)kl * nope, b.q_lat, Hm * kl, kl, T, kl, nope,
                        Hm)) {
    gemm_->mm_bf16_batched(b.q, Hm * nope, nope, W.w_kc, kl, (long long)nope * kl, false, b.q_lat, Hm * kl, kl, T, kl,
                           nope, Hm, stream_);
  }
  if (fork) CUDA_CHECK(cudaStreamWaitEvent(stream_, ev_sel_, 0));
  if (selk) {  // (the attention ran in the selection's launch)
  } else if (in.num_attn_tiles > 0 && own) {  // the step's own tokens
    // FlashMLA's kernel on the query tiles (64 / Hm tokens per CTA, the union of their pools); else the tiled kernels
    const size_t wsb = b.scores_bytes;
    const float sc = 1.f / sqrtf((float)nope);
    const int64_t qrows = (int64_t)rc_.max_num_tokens * Hm;
    if (!k::sparse_mla_attn_fmla(b.o_lat, b.q_lat, P.lat, P.lat_rows, b.sel, meta, T, Hm, sc, b.scores, wsb,
                                 sm_count_, stream_) &&
        !k::sparse_mla_attn_fmla_tiled(b.o_lat, b.q_lat, qrows, P.lat, P.lat_rows, b.sel, meta, in.attn_tiles,
                                       in.num_attn_tiles, Hm, b.max_pools + 8, sc, b.scores, wsb, stream_))
      k::sparse_mla_attn_tiled(b.o_lat, b.q_lat, P.lat, b.sel, meta, in.attn_tiles, in.num_attn_tiles, Hm,
                               b.max_pools + 8, sc, stream_, qrows, P.lat_rows, b.scores, wsb);
  } else {
    // decode steps of >= kFmlaSplitMinSeqs sequences of 4 tokens (MTP 3 verification, 16 local heads): FlashMLA's
    // kernel on a 64-row query tile per sequence over the union of its tokens' pools, split-KV (below that, the
    // per-token split-KV attention)
    constexpr int kFmlaSplitMinSeqs = 8;
    const int ut = dec_seqs_ >= kFmlaSplitMinSeqs && T == 4 * dec_seqs_ && !share && own && Hm == 16 &&
                           dec_seqs_ <= b.max_unit_tiles
                       ? dec_seqs_
                       : 0;
    if (!(ut > 0 && k::sparse_mla_attn_fmla_split(
                        b.o_lat, b.q_lat, (int64_t)rc_.max_num_tokens * Hm, P.lat, P.lat_rows, b.sel, meta,
                        b.dec_tiles, ut, T, Hm, b.max_pools + 8, 1.f / sqrtf((float)nope), b.scores, b.scores_bytes,
                        b.attn_ws, (size_t)b.attn_ws_blocks,
                        k::sparse_mla_fmla_split_splits(ut, T, sm_count_, b.attn_ws_blocks), stream_))) {
      // split-KV for few tokens (decode, MTP); depends only on (T, need_sel) for CUDA graphs
      const int hgroups = (int)cdiv(Hm, 16);  // CTAs of 16 heads
      const int nsplit =
          k::sparse_mla_splits(T * hgroups, sm_count_, need_sel ? k::kMaxRows : 2048, attn_ws_ctas_, 0);
      // up to kAttnClusterMaxTokens tokens: the splits of each token in one thread-block cluster that combines its
      // partials itself (one launch fewer), when the T clusters fit at once
      constexpr int kAttnClusterMaxTokens = 12;
      if (!(T <= kAttnClusterMaxTokens && Hm == 16 &&
            k::sparse_mla_attn_cl(b.o_lat, b.q_lat, P.lat, share ? b.mtp_sel : b.sel, meta, T, Hm,
                                  1.f / sqrtf((float)nope), stream_, nsplit, share ? b.mtp_vis : nullptr,
                                  share && att_pre_)))
        k::sparse_mla_attn(b.o_lat, b.q_lat, P.lat, share ? b.mtp_sel : b.sel, meta, T, Hm,
                           1.f / sqrtf((float)nope), stream_, nsplit, b.attn_ws, share ? b.mtp_vis : nullptr);
    }
  }
  // o[t, h] = o_lat[t, h] @ w_vc[h]^T  (w_vc[h] is [v, kv_lora])
  // w_vc as one head-batched gemv_pre (weights staged before its PDL wait) up to kVcHeadGemvMaxTokens tokens, the
  // o_proj launched at its start so that the o_proj GEMV's CTAs settle and preload beside the (cluster) attention
  constexpr int kVcHeadGemvMaxTokens = 7;
  // (in kVcCtas CTAs of 48 rows (53 KB) at the largest shared-memory carveout, so that each SM the attention leaves
  // free hosts at most one w_vc CTA and an o_proj CTA (165 KB) lands beside it at the w_vc's start: its ring then
  // streams before the o_proj's PDL wait. Bitwise.)
  constexpr int kVcCtas = 96;
  if (!(T <= kVcHeadGemvMaxTokens && Hm == 16 &&
        head_gemv(b.o_lat, Hm * kl, kl, W.w_vc, kl, (long long)vh * kl, b.o, Hm * vh, vh, T, vh, kl, Hm,
                  /*late=*/false, kVcCtas, /*carveout=*/100)))
    gemm_->mm_bf16_batched(b.o_lat, Hm * kl, kl, W.w_vc, kl, (long long)vh * kl, true, b.o, Hm * vh, vh, T, vh, kl,
                           Hm, stream_);
  bf16_dense(b.o, Hm * vh, W.o_proj16.w, Hm * vh, b.out, H, false, T, H, Hm * vh);
}

// ------------------------------------------------------------------ dense MLP (NVFP4)
bool GlmStep::dense_fused(const MlpW& W, int T) const {
  constexpr int kDenseFusedMaxTokens = 16;
  const int H = cfg_.hidden, Il = W.inter_local;
  return W.fmt == WFmt::kNvfp4 && W.alpha4 && T >= 1 && T <= kDenseFusedMaxTokens && H % 512 == 0 &&
         Il % 512 == 0 && Il / 64 + H / 128 <= sm_count_ && b_->x4_rows >= (T <= 8 ? 8 : 16);
}

void GlmStep::layer_mlp(const MlpW& W, int T) {
  Buffers& b = *b_;
  const int H = cfg_.hidden, Il = W.inter_local;
  CHECK(W.fmt == WFmt::kNvfp4, "dense MLP: NVFP4 weights");
  const Fp4W &gu = W.gate_up4, &dn = W.down4;
  if (dense_fused(W, T)) {
    // (the cluster boundary before this layer may already have quantized b.x: x4_ready_)
    if (!x4_ready_) k::nvfp4_quant(b.x4, b.x4s, b.x, H, T, H, gu.gs, /*swz=*/true, stream_);
    x4_ready_ = false;
    k::MoeFp4Args fa;
    fa.xq = b.x4;
    fa.xs = b.x4s;
    fa.x_rows = b.x4_rows;
    fa.w13 = gu.w;
    fa.sf13 = gu.sf;
    fa.alpha13 = W.alpha4;
    fa.w2 = dn.w;
    fa.sf2 = dn.sf;
    fa.alpha2 = W.alpha4 + 1;
    fa.hq = b.h4;
    fa.hsa = b.h4s;
    fa.h_rows = b.x4_rows;  // (b.h4: as many rows as b.x4)
    fa.y = b.out;
    fa.ctr = b.mf4_ctr;
    fa.H = H;
    fa.Il = Il;
    fa.G = 1;
    fa.limit = cfg_.swiglu_limit;
    fa.gs2 = dn.gs;
    k::moe_fp4_dense(fa, T, stream_);
    return;
  }
  x4_ready_ = false;
  k::nvfp4_quant(b.x4, b.x4sw, b.x, H, T, H, gu.gs, /*swz=*/true, stream_);
  k::nvfp4_gemm(b.x4, b.x4sw, gu.w, gu.sf, b.gu, 2 * Il, T, 2 * Il, H, gu.alpha, b.fp4_ws, kFp4GemmWorkspace,
                stream_);
  // (gate / up columns interleaved by 64, as the weights' rows)
  k::swiglu_quant_nvfp4(b.h4, b.h4s, b.gu, T, Il, cfg_.swiglu_limit, dn.gs, stream_, nullptr, nullptr, 0, nullptr,
                        /*gu64=*/true);
  k::nvfp4_gemm(b.h4, b.h4s, dn.w, dn.sf, b.out, H, T, H, Il, dn.alpha, b.fp4_ws, kFp4GemmWorkspace, stream_);
}

// ------------------------------------------------------------------ MoE
namespace {
// Steps of up to 4 tokens: router logits by moe_router_logits (weights read before the PDL wait, deterministic sums)
// instead of the cuBLASLt GEMM.
bool router_fast(int T) { return T <= 4; }
// Router logits of 9 .. 16 tokens by the tensor-core kernel moe_router_mma (weights in registers before the PDL wait,
// small enough that the fused MoE kernel it launches early fits beside it) instead of cuBLASLt.
bool router_mma(int T) { return T >= 9 && T <= 16; }
// layer_moe's in-kernel-routing path (the fused MoE's route_k) also takes moe_router_mma at 4 .. 8 tokens: in place of
// the cluster FMA router at 4 tokens, whose CTAs leave the MoE kernel room to launch early as the tensor-core router's
// do. Not bitwise: the 4- and 5-8-token logits' fp32 sums run in the tensor-core router's order (as at 9-16 tokens).
bool router_mma_in(int T) { return router_mma(T) || (T >= 4 && T <= 8); }
constexpr int kFp4FusedMaxTokens = 128;
// the fused decode kernel for this MoE's routed experts at T tokens: NVFP4, or BF16 (the MTP layer's experts)
bool fused_moe(const MoeW& W, int T) {
  return T <= std::min(kFp4FusedMaxTokens, k::kMoeFp4MaxTokens) && (W.fmt == WFmt::kNvfp4 || W.fmt == WFmt::kBf16);
}
// Decode MoE steps of up to kMoeRouteInMaxTokens tokens (at most 64): the routing and the dispatch tables inside the
// fused kernel (its epilogue warps route while the shared expert's first units stream; bitwise the dispatch kernel's
// ids, weights and expert outputs) instead of moe_dispatch_small (and moe_route_warp) before it.
constexpr int kMoeRouteInMaxTokens = 32;
// Prefill-sized MoE layers (from kMoeShForkMinTokens tokens): the shared expert (cuBLASLt GEMMs + SwiGLU,
// compute-bound) runs on the side stream while the routing, the NVFP4 quantization, the row permutation and the routed
// experts run on stream_; bitwise the serial order's outputs.
constexpr int kMoeShForkMinTokens = 512;
}  // namespace

void GlmStep::moe_route(const MoeW& W, int T) {
  Buffers& b = *b_;
  const int H = cfg_.hidden, E = cfg_.n_experts;
  if (T == 0) return;
  if ((router_mma(T) && k::moe_router_mma(b.router_logits, b.x, W.router, T, E, H, stream_)) ||
      (router_fast(T) && k::moe_router_logits(b.router_logits, b.x, W.router, T, E, H, stream_))) {
    k::moe_route_warp(b.topk_ids, b.topk_w, b.router_logits, W.bias, T, E, cfg_.top_k, cfg_.routed_scale, stream_,
                      /*shared_id=*/E, nullptr, -1);
    return;
  }
  bf16_dense(b.x, H, W.router, H, b.router_logits, E, true, T, E, H, /*pipe_ok=*/false);
  // one warp per token (bitwise the rank kernel's selection and weights)
  k::moe_route_warp(b.topk_ids, b.topk_w, b.router_logits, W.bias, T, E, cfg_.top_k, cfg_.routed_scale, stream_,
                    /*shared_id=*/E, nullptr, -1);
}

bool GlmStep::moe_experts_q(const MoeW& W, int T, const void* xq, const void* xs, int x_rows, const int* ids, int t0,
                            int shared_mode, const k::MoeRouteArgs* ra, int sh_T, bf16* sh_y, bool route_k) {
  Buffers& b = *b_;
  if (T == 0) return false;
  const int H = cfg_.hidden, K1 = cfg_.top_k + 1, Il = W.inter_local, G = W.local.n;
  const int R = T * K1;
  k::MoeGroupMap map = W.local;
  map.t0 = t0;
  map.shared_mode = shared_mode;
  map.shared_rows = T * cfg_.top_k;  // (routed pairs take at most T * top_k rows)
  const bool fp4 = W.fmt == WFmt::kNvfp4;
  if (fused_moe(W, T)) {
    // NVFP4 from kWideMinTokens tokens: the wide kernel (256 weight rows per MMA, half the tensor-pipe
    // instructions per weight byte, but units twice the size and one accumulator)
    constexpr int kWideMinTokens = 96;
    const bool wide = fp4 && T >= kWideMinTokens;
    // route_k: the routing (ra: its logits) inside the tall kernel
    const bool rtk = route_k && ra && !wide;
    // one work item per expert (all of an expert's rows: the kernel takes every token as N)
    if (!rtk)
      k::moe_dispatch_small(b.pair_pos, b.row_src, b.offsets, b.active, b.num_active, ids, T, K1, map, 1 << 20,
                            stream_, ra);
    k::MoeFp4Args fa;
    CHECK(!(ra && ra->ks) || rtk, "moe_experts_q: the router's {key, score} words need the in-kernel routing");
    if (rtk) {
      fa.rt_logits = ra->logits;
      fa.rt_ks = ra->ks;
      fa.rt_bias = ra->bias;
      fa.rt_T = T;
      fa.rt_E = cfg_.n_experts;
      fa.rt_topk = ra->top_k;
      fa.rt_lo = W.local.lo;
      fa.rt_scale = ra->scale;
      fa.rt_ids = ra->ids_out;
      fa.rt_w = ra->w_out;
      fa.rt_pos = b.pair_pos;
    }
    fa.rt_hist = moe_hist_cur_;  // (in-kernel routing or the dispatch's tables)
    fa.active = b.active;
    fa.num_active = b.num_active;
    fa.offsets = b.offsets;
    fa.a_rows = b.row_src;
    fa.routed_bf16 = !fp4;
    fa.xq = xq;
    fa.xs = static_cast<const uint8_t*>(xs);
    fa.x_rows = x_rows;
    fa.w13 = fp4 ? static_cast<const void*>(W.w13_4) : W.w13_16;
    fa.sf13 = W.sf13;
    fa.alpha13 = W.alpha13;
    fa.w2 = fp4 ? static_cast<const void*>(W.w2_4) : W.w2_16;
    fa.sf2 = W.sf2;
    fa.alpha2 = W.alpha2;
    fa.hq = fp4 ? static_cast<void*>(b.hq4) : b.h16;
    fa.hsa = b.hsa4;
    fa.h_rows = fp4 ? b.h4_rows : b.h16_rows;
    fa.y = b.y;
    // the BF16 shared expert in the same launch
    const bool sh = sh_y && sh_T > 0 && W.sh_inter > 0;
    if (sh) {
      fa.sh_inter = W.sh_inter;
      fa.sh_x = b.x;
      fa.sh_x_rows = b.x4_rows;
      fa.sh_T = sh_T;
      fa.sh_gu = W.sh_gate_up.w;
      fa.sh_dn = W.sh_down.w;
      fa.sh_h = b.sh_h;
      fa.sh_h_rows = b.x4_rows;
      fa.sh_part = b.sh_part;
      fa.sh_y = sh_y;
      fa.sh_ldy = H;
      // its weights (gate / up and down: sh_pf 2) into L2 before the kernel's dependency wait up to kShPfMaxTokens
      // tokens, requested by the last kShPfCtas CTAs
      constexpr int kShPfMaxTokens = 8, kShPfCtas = 64;
      fa.sh_pf = T <= kShPfMaxTokens ? 2 : 0;
      fa.sh_pf_ctas = kShPfCtas;
    }
    // token-major outputs for the boundary after this layer (moe_tm_req_: its kTm instances), under MoE tensor
    // parallelism (every pair on this rank) with the in-kernel routing and the fused shared expert
    moe_tm_out_ = moe_tm_req_ && rtk && !wide && sh && W.local.ep == 1 && G == cfg_.n_experts && W.local.lo == 0;
    if (moe_tm_out_) {
      fa.y_tm = true;
      fa.sh_y = b.y + (size_t)cfg_.top_k * H;
      fa.sh_ldy = K1 * H;
    }
    // the kernel's first claim and the tables at once
    fa.tab_early = true;
    fa.late_trigger = false;
    fa.ctr = b.mf4_ctr;
    fa.H = H;
    fa.Il = Il;
    fa.G = G;
    fa.limit = cfg_.swiglu_limit;
    fa.gs2 = W.gs2;
    if (wide)
      k::moe_fp4_wide(fa, T, sm_count_, stream_);
    else
      k::moe_fp4_fused(fa, T, sm_count_, stream_);
    return sh;
  }
  CHECK(ra == nullptr, "moe_experts_q: routing in the dispatch only on the fused path");
  moe_tm_out_ = false;
  const bool mtp_grp = mtp_moe_tp_ && &W == &w_.mtp.moe;  // (the tensor-parallel MTP experts' grouped GEMMs)
  k::moe_permute_rows(fp4 ? b.xp4 : reinterpret_cast<uint8_t*>(b.xp16), b.sfp4, b.pair_pos, b.counts, b.offsets,
                      b.pad, static_cast<const uint8_t*>(xq), static_cast<const uint8_t*>(xs), ids, T, K1, map, G, H,
                      fp4, stream_);
  if (fp4) {
    // (the expected rows per expert pick the grouped GEMMs' tile shape)
    const int rows_hint = (int)((int64_t)T * cfg_.top_k / cfg_.n_experts);
    // (its work items, at most ceil(R / 128) + G, within the hidden-row scale blocks and counters)
    if (rows_hint <= kMoePfMaxRows && G <= k::kMoePfMaxGroups && k::moe_fp4_prefill_items(R, G) <= b.hsa_pf_items) {
      // gate / up GEMM, SwiGLU + NVFP4 and down GEMM in one launch
      k::MoePrefillArgs pa;
      pa.offsets = b.offsets;
      pa.pad = b.pad;
      pa.xq = b.xp4;
      pa.xs = b.sfp4;
      pa.x_rows = b.xp4_rows;
      pa.w13 = W.w13_4;
      pa.sf13 = W.sf13;
      pa.alpha13 = W.alpha13;
      pa.w2 = W.w2_4;
      pa.sf2 = W.sf2;
      pa.alpha2 = W.alpha2;
      pa.hq = b.hq4;
      pa.hsa = b.hsa_pf;
      pa.h_rows = b.h4_rows;
      pa.rows = R;
      pa.hs_items = b.hsa_pf_items;
      pa.y = b.y;
      pa.ctr = b.pf_ctr;
      pa.H = H;
      pa.Il = Il;
      pa.G = G;
      pa.limit = cfg_.swiglu_limit;
      pa.gs2 = W.gs2;
      k::moe_fp4_prefill(pa, sm_count_, stream_);
      return false;
    }
    (mtp_grp ? fp4_g1m_ : fp4_g1_)->run(b.xp4, b.sfp4, W.w13_4, W.sf13, W.alpha13, b.h13, b.offsets, b.pad, stream_,
                                       rows_hint);
    k::swiglu_quant_nvfp4(b.hq4, b.hs4, b.h13, R, Il, cfg_.swiglu_limit, W.gs2, stream_, b.offsets, b.pad, G,
                          b.offsets + G, /*gu64=*/true);
    (mtp_grp ? fp4_g2m_ : fp4_g2_)->run(b.hq4, b.hs4, W.w2_4, W.sf2, W.alpha2, b.y, b.offsets, b.pad, stream_,
                                       rows_hint);
  } else {
    (mtp_grp ? bf_g1m_ : bf_g1_)->run(b.xp16, W.w13_16, b.h13, b.offsets, stream_);
    k::swiglu_bf16(b.h16, b.h13, R, Il, cfg_.swiglu_limit, stream_, b.offsets + G, /*gu64=*/true);
    (mtp_grp ? bf_g2m_ : bf_g2_)->run(b.h16, W.w2_16, b.y, b.offsets, stream_);
  }
  return false;
}

void GlmStep::moe_shared(const MoeW& W, int T, bf16* out) {
  Buffers& b = *b_;
  if (T == 0) return;
  const int H = cfg_.hidden, Is = W.sh_inter;
  bf16_dense(b.x, H, W.sh_gate_up.w, H, b.sh_gu, 2 * Is, false, T, 2 * Is, H);
  k::swiglu_bf16(b.sh_h, b.sh_gu, T, Is, cfg_.swiglu_limit, stream_, nullptr, /*gu64=*/true);
  bf16_dense(b.sh_h, Is, W.sh_down.w, Is, out, H, false, T, H, Is);
}

bool GlmStep::moe_shared_fork(const MoeW& W, int T, bf16* out) {
  if (T < kMoeShForkMinTokens || W.sh_inter <= 0) return false;
  CUDA_CHECK(cudaEventRecord(ev_sh_fork_, stream_));
  CUDA_CHECK(cudaStreamWaitEvent(side_stream_, ev_sh_fork_, 0));
  {
    ScopedSwap<cudaStream_t> side_st(stream_, side_stream_);
    ScopedSwap<std::unique_ptr<Gemm>> side_gemm(gemm_, gemm2_);
    moe_shared(W, T, out);
  }
  CUDA_CHECK(cudaEventRecord(ev_sh_join_, side_stream_));
  return true;
}

void GlmStep::moe_shared_join() { CUDA_CHECK(cudaStreamWaitEvent(stream_, ev_sh_join_, 0)); }

// RF: whether a main layer's MoE of T tokens takes moe_router_mma_ks (layer_moe's route_ks below) at the RF form's
// size, so that its attention-side boundary may write the router's copies (the layer loop asks the boundary for them)
bool GlmStep::rf_layer_ok(const MoeW& W, int T) const {
  const int E = cfg_.n_experts, H = cfg_.hidden;
  return rf_x_ && T == k::kRouterRfT && pc_.tp == 4 && W.fmt == WFmt::kNvfp4 && E == k::kRouterRfGroups * 16 &&
         H == 4096 && fused_moe(W, T) && T <= kMoeRouteInMaxTokens && W.local.shared_dyn == 0 &&
         k::moe_fp4_route_in_ok(T, E, cfg_.top_k, W.local.n) && router_mma_in(T) &&
         k::moe_fp4_route_ks_ok(T, H, W.inter_local, W.sh_inter, false);
}

void GlmStep::layer_moe(const MoeW& W, int T) {
  // (RF: the attention-side boundary just wrote this router's x copies: rf_set_)
  const bool rf = rf_ready_;
  rf_ready_ = false;
  // (SR pending: a chained MTP pass's summing form, folded into the tensor-core router below when that router runs)
  bool sr = false;
  if (sr_pend_) {
    sr_pend_ = false;
    const int E = cfg_.n_experts;
    sr = &W == &w_.mtp.moe && fused_moe(W, T) && T <= kMoeRouteInMaxTokens && W.local.shared_dyn == 0 &&
         k::moe_fp4_route_in_ok(T, E, cfg_.top_k, W.local.n) && (router_mma_in(T) || T <= 3) &&
         !(router_mma_in(T) && W.fmt == WFmt::kNvfp4 &&
           k::moe_fp4_route_ks_ok(T, cfg_.hidden, W.inter_local, W.sh_inter, false)) &&
         k::mtp_sr_ok(T, cfg_.hidden, pc_.tp, sr_args_.max_elems, E);
    if (!sr)
      k::mtp_lamport_norm(T, cfg_.hidden, false, sr_args_.peers, ar_epochs_, pc_.rank, pc_.tp, ar_max_elems_,
                          ar_max_blocks_, sr_args_.resid, true, sr_args_.w, sr_args_.out, nullptr, nullptr,
                          sr_args_.eps, /*late=*/false, stream_, nullptr);
  }
  // NVFP4 / BF16 routed experts; the shared expert's partial (this rank's share of its features) in the rows its
  // routing slot points at, summed by the all-reduce
  Buffers& b = *b_;
  const int H = cfg_.hidden;
  const void* xq = b.x;
  const void* xs = nullptr;
  // decode: the router logits here, the routing inside the fused kernel (kMoeRouteInMaxTokens) or in the fused path's
  // dispatch kernel
  const int E = cfg_.n_experts;
  const bool route_k = fused_moe(W, T) && T <= kMoeRouteInMaxTokens && W.local.shared_dyn == 0 &&
                       k::moe_fp4_route_in_ok(T, E, cfg_.top_k, W.local.n);
  CHECK(!sr || route_k, "SR: the in-kernel routing's conditions");
  CHECK(!rf || route_k, "RF: the in-kernel routing's conditions (rf_layer_ok)");
  bf16* sh_y = b.y + (size_t)T * cfg_.top_k * H;
  // prefill-sized steps: the shared expert concurrently on the side stream
  const bool sh_fork = !fused_moe(W, T) && moe_shared_fork(W, T, sh_y);
  bool route_in = false, route_ks = false;
  if (route_k) {
    route_in = true;
    // (the MTP layer's router at 1-3 tokens on the tensor cores as at 4)
    const bool rmma = router_mma_in(T) || (&W == &w_.mtp.moe && T <= 3);
    // the tensor-core router hands the in-kernel routing each pair's routing key and score instead of its logit
    // (moe_router_mma_ks; bitwise the same ids and weights)
    // (SR: the summing form's x and these logits from one kernel, bitwise the norm + moe_router_mma)
    if (sr) {
      CHECK(rmma, "SR: the tensor-core router's conditions");
      k::mtp_sr_router(b.router_logits, W.router, T, E, H, sr_args_, stream_);
    }
    // (RF: the same words from the boundary's copies, polled instead of after its grid)
    unsigned long long* const ksw = reinterpret_cast<unsigned long long*>(b.router_logits);
    if (rf) {
      CHECK(!sr && router_mma_in(T) && W.fmt == WFmt::kNvfp4 &&
                k::moe_fp4_route_ks_ok(T, H, W.inter_local, W.sh_inter, false) &&
                k::moe_router_mma_ks_rf(ksw, rf_set_, W.router, W.bias, T, E, H, stream_),
            "RF: the boundary wrote the router's copies, the RF router must read them");
      route_ks = true;
    } else {
      route_ks = !sr && router_mma_in(T) && W.fmt == WFmt::kNvfp4 &&
                 k::moe_fp4_route_ks_ok(T, H, W.inter_local, W.sh_inter, false) &&
                 k::moe_router_mma_ks(ksw, b.x, W.router, W.bias, T, E, H, stream_);
    }
    if (!sr && !route_ks && !(rmma && k::moe_router_mma(b.router_logits, b.x, W.router, T, E, H, stream_)) &&
        !(router_fast(T) && T <= k::kRouterMaxT &&
          k::moe_router_logits(b.router_logits, b.x, W.router, T, E, H, stream_)))
      bf16_dense(b.x, H, W.router, H, b.router_logits, E, true, T, E, H, /*pipe_ok=*/false);
  } else {
    route_in = fused_moe(W, T) && router_fast(T) && T <= k::kRouterMaxT &&
               k::moe_router_logits(b.router_logits, b.x, W.router, T, E, H, stream_);
    if (!route_in) moe_route(W, T);
  }
  if (W.fmt == WFmt::kNvfp4) {
    // (the cluster boundary before this layer may already have quantized b.x: x4_ready_)
    if (!x4_ready_) k::nvfp4_quant(b.x4, b.x4s, b.x, H, T, H, W.gs13, /*swz=*/true, stream_);
    xq = b.x4;
    xs = b.x4s;
  }
  x4_ready_ = false;
  const k::MoeRouteArgs ra{b.router_logits, W.bias, cfg_.top_k, cfg_.routed_scale, b.topk_ids, b.topk_w, route_ks};
  if (!moe_experts_q(W, T, xq, xs, b.x4_rows, b.topk_ids, 0, /*shared_mode=*/1, route_in ? &ra : nullptr, T, sh_y,
                     route_k) &&
      !sh_fork)
    moe_shared(W, T, sh_y);
  if (sh_fork) moe_shared_join();
}

void GlmStep::moe_combine_allreduce(int T) {
  Buffers& b = *b_;
  const int H = cfg_.hidden, K1 = cfg_.top_k + 1;
  CHECK(!moe_tm_out_, "moe_combine_allreduce: token-major MoE outputs (only the kTm boundary reads them)");
  if (pc_.tp > 1 && ar_enabled_) {
    k::MoeCombineArgs cm;
    cm.y = b.y;
    cm.w = b.topk_w;
    cm.pair_pos = b.pair_pos;
    cm.K1 = K1;
    cm.hidden = H;
    if (k::custom_allreduce_moe(b.out, cm, (size_t)T * H, ar_peers_, ar_epochs_, pc_.rank, pc_.tp, ar_max_elems_,
                                ar_max_blocks_, stream_))
      return;
  }
  k::moe_combine(b.out, b.y, b.topk_w, b.pair_pos, nullptr, T, K1, H, stream_);
  allreduce(b.out, (size_t)T * H);
}

// L2 prefetch of a decode MoE layer's experts of the previous step (their weights' four segments, by descending token
// count) on pf_stream_ during the boundary before the layer, while HBM idles (steps of 17 .. 64 tokens, forward_impl):
// at most 48 MB from 32 CTAs of 128 threads, with expert parallelism 96 MB from 96 CTAs, 32 KB per bulk prefetch,
// every expert's gate / up segments before the downs. Prefetches change no arithmetic.
void GlmStep::moe_pf_async(const MoeW& W, const int* hist) {
  if (W.fmt != WFmt::kNvfp4 || !W.w13_4 || !W.w2_4) return;
  constexpr int kCtas = 32, kCtasEp = 96;
  constexpr size_t kCap = 48ull << 20, kCapEp = 96ull << 20;
  constexpr size_t kChunk = 32 * 1024;
  const int H = cfg_.hidden, Il = W.inter_local;
  k::MoeL2pf a;
  a.hist = hist;
  a.hist_max = k::kMoeFp4HistMax;
  a.n_groups = W.local.n;
  a.seg[0] = W.w13_4;
  a.stride[0] = (size_t)2 * Il * (H / 2);
  a.seg[1] = W.sf13;
  a.stride[1] = k::nvfp4_sf_bytes(2 * Il, H);
  a.seg[2] = W.w2_4;
  a.stride[2] = (size_t)H * (Il / 2);
  a.seg[3] = W.sf2;
  a.stride[3] = k::nvfp4_sf_bytes(H, Il);
  // (expert parallelism: whole experts, a share of them per rank)
  const bool ep = W.local.ep > 1;
  a.cap = ep ? kCapEp : kCap;
  a.two_phase = true;
  a.by_count = true;
  CUDA_CHECK(cudaEventRecord(ev_pf_, stream_));
  CUDA_CHECK(cudaStreamWaitEvent(pf_stream_, ev_pf_, 0));
  k::moe_l2_prefetch(a, ep ? kCtasEp : kCtas, kChunk, pf_stream_);
  pf_forked_ = true;
}

// ------------------------------------------------------------------ head
// The vocab-parallel head of few rows as one kernel (k::head_keys: the weight slice streamed with TMA bulk copies, the
// argmax keys taken in its epilogue, the cross-rank max in its last CTA) instead of cuBLAS (split-K + its reduce
// kernel), argmax_keys and custom_max_u64 -- for rows <= kHeadKeysMaxRows. Callers check that the cross-rank max may
// use the custom all-reduce's key slots (tensor parallel 1, or ar_enabled_). Returns whether it ran.
bool GlmStep::head_keys_ok(int S) const {
  const int H = cfg_.hidden, Vl = w_.lm_head.n, vtp = pc_.tp;
  return S >= 1 && S <= k::kHeadKeysMaxRows && k::head_keys_fits(S, Vl, H) && Vl % 4 == 0 && vtp <= k::kMaxTp;
}

bool GlmStep::head_keys(const bf16* rows, int S, const k::SampleParams* params, unsigned long long* keys, int* tokens,
                        int tokens_ld, unsigned* ready) {
  const int H = cfg_.hidden, Vl = w_.lm_head.n, vtp = pc_.tp;
  if (!head_keys_ok(S)) return false;
  Buffers& b = *b_;
  k::HeadKeysArgs a;
  a.A = rows;
  a.lda = H;
  a.W = w_.lm_head.w;
  a.ldw = H;
  a.M = S;
  a.N = Vl;
  a.K = H;
  a.vocab_offset = pc_.rank * Vl;
  a.params = params;
  a.keys = keys;
  a.tokens = tokens;
  a.tokens_ld = tokens_ld;
  a.part = static_cast<unsigned long long*>(b.head_scratch);
  a.count = reinterpret_cast<unsigned int*>(static_cast<uint8_t*>(b.head_scratch) +
                                            (size_t)k::kHeadKeysMaxCtas * k::kHeadKeysMaxRows * 8);
  if (vtp > 1) {
    a.peers = ar_peers_;
    a.state = ar_epochs_ + ar_max_blocks_ + 4;
    a.rank = pc_.rank;
    a.tp = pc_.tp;
  }
  a.late_trigger = true;
  a.ready = ready;
  a.ready_target = ready ? (unsigned)S : 0u;
  k::head_keys(a, sm_count_, stream_);
  return true;
}

// ------------------------------------------------------------------ MTP
namespace {
// the MTP input projection streams its 64 MiB (replicated) with TMA bulk copies at up to this many rows
constexpr int kMtpEhTmaMaxRows = 8;
}  // namespace

bool GlmStep::mtp_eh_cp(int T) const {
  const int H = cfg_.hidden;
  return pc_.tp == 4 && ar_enabled_ && k::mtp_lamport_norm_ok(T, H, pc_.tp, ar_max_elems_) &&
         T <= kMtpEhTmaMaxRows && k::gemv_tma_fits(T, H / 4, 2 * H);
}

bool GlmStep::mtp_pass(const MtpPass& p, bool need_sel) {
  Buffers& b = *b_;
  const StepIn& in = *in_;
  const MtpW& M = w_.mtp;
  const int H = cfg_.hidden, T = p.T;
  const float eps = cfg_.rms_eps;
  bool picked = false;  // (pass 1: the attention's consumer did mtp_pick's copies)
  bool booked = false;  // (and mtp_accept's bookkeeping)
  CHECK(!p.ehf || (T > 0 && mtp_eh_cp(T)), "MTP: built input rows without the column-parallel input projection");
  if (T > 0) {
    if (!p.eh_ready && !p.ehf)
      k::mtp_eh_norm(b.eh_in, w_.embed, p.ids, p.src, b.out_tokens, p.positions, p.hidden, M.enorm, M.hnorm, T, H,
                     eps, stream_);
    // Tensor parallel (TP 4, one-shot Lamport sizes): the input projection column-parallel (mtp_eh_cp): rank q
    // computes output columns [q H / 4, (q + 1) H / 4) and pushes them into every rank's Lamport buffer (GemvArPush);
    // k::mtp_lamport_norm polls the gathered rows and runs the input norm. (These norms launch the next kernel at the
    // consumer's start: late = false.)
    const bool lam = pc_.tp == 4 && ar_enabled_ && k::mtp_lamport_norm_ok(T, H, pc_.tp, ar_max_elems_);
    const int Hc = H / 4;
    if (mtp_eh_cp(T)) {
      k::GemvPreArgs a;
      a.A = p.ehf ? nullptr : b.eh_in;
      a.lda = 2 * H;
      a.W = M.eh_proj.w + (size_t)pc_.rank * Hc * 2 * H;
      a.ldw = 2 * H;
      a.D = b.mtp_h + (size_t)pc_.rank * Hc;
      a.ldd = H;
      a.M = T;
      a.N = Hc;
      a.K = 2 * H;
      a.late_trigger = true;  // (the consumer polls before its wait)
      for (int q = 0; q < 4; ++q) a.push.buf[q] = ar_peers_.buf[q] + (size_t)pc_.rank * Hc;
      a.push.state = ar_epochs_ + ar_max_blocks_;
      a.push.rank = pc_.rank;
      a.push.max_elems = ar_max_elems_;
      // (the ring's stages at most kEhStages; tiles of 8 weight rows from kEhRt8MinRows rows on for loaded rows (pass
      // 1), and for the chained passes' built rows up to kEhBuiltRt8MaxRows rows)
      constexpr int kEhStages = 6, kEhRt8MinRows = 4, kEhBuiltRt8MaxRows = 4;
      const bool built = p.ehf && p.ehf->mode == 1;
      const int rt = (T >= kEhRt8MinRows && !built) || (built && T <= kEhBuiltRt8MaxRows) ? 8 : 16;
      k::gemv_tma(a, sm_count_, stream_, kEhStages, p.ehf, rt);
      k::mtp_lamport_norm(T, H, true, ar_peers_, ar_epochs_, pc_.rank, pc_.tp, ar_max_elems_, ar_max_blocks_, b.mtp_h,
                          true, M.in_norm, b.x, nullptr, nullptr, eps, /*late=*/false, stream_);
    } else if (T <= kMtpEhTmaMaxRows && k::gemv_tma_fits(T, H, 2 * H)) {
      k::GemvPreArgs a;
      a.A = b.eh_in;
      a.lda = 2 * H;
      a.W = M.eh_proj.w;
      a.ldw = 2 * H;
      a.D = b.mtp_h;
      a.ldd = H;
      a.M = T;
      a.N = H;
      a.K = 2 * H;
      a.late_trigger = true;
      k::gemv_tma(a, sm_count_, stream_);
      k::add_rmsnorm(b.mtp_h, nullptr, nullptr, M.in_norm, b.x, nullptr, nullptr, T, H, eps, false, stream_);
    } else {
      bf16_dense(b.eh_in, 2 * H, M.eh_proj.w, 2 * H, b.mtp_h, H, false, T, H, 2 * H);
      k::add_rmsnorm(b.mtp_h, nullptr, nullptr, M.in_norm, b.x, nullptr, nullptr, T, H, eps, false, stream_);
    }
    // attention sub-block (plain residual; MLA cache index after the decoder's MLA layers). Tensor parallel (as above):
    // the o_proj GEMV pushes its partial sums into every rank's Lamport buffer itself and k::mtp_lamport_norm sums them
    // in rank order with the residual add + post-attention norm (bitwise the all-reduce + add_rmsnorm kernels it
    // replaces); the pushed o_proj on gemv_tma at as many rows as the decoder's (bf16_dense)
    const bool op_push = lam;
    ar_push_req_ = op_push;
    ar_push_T_ = T;
    // (a chained pass with built input rows runs the shared selection's attention on the PRE instance: its row table
    // and gathers before its PDL wait, the own row after it; bitwise)
    att_pre_ = p.ehf != nullptr && p.share;
    // (the same passes' positions / rows -- the first pass's consumer's outputs, final by the same argument -- read
    // before the PDL waits like the step's own; bitwise)
    mtp_meta_pre_ = p.ehf != nullptr;
    layer_mla(-1, cfg_.num_mla_layers(), M.mla, T, need_sel, p.token_slot, p.positions, p.share);
    att_pre_ = false;
    mtp_meta_pre_ = false;
    ar_push_req_ = false;
    if (op_push && ar_pushed_) {
      ar_pushed_ = false;
      // (pass 1: the consumer also copies each sequence's last accepted row, mtp_pick's job)
      k::MtpPickArgs pk;
      if (p.pick > 0) {
        pk.xp = b.mtp_xp;
        pk.hp = b.mtp_hp;
        pk.xqp = nullptr;
        pk.xsp = nullptr;
        pk.force_accept = in.force_accept;
        pk.row0 = in.spec_row0;
        pk.nrows = in.spec_nrows;
        pk.sampled = b.out_tokens;
        pk.ids = in.ids;
        pk.sample_idx = in.sample_idx;
        pk.positions = in.positions;
        pk.token_slot = in.token_seq;
        pk.B = p.pick;
        picked = true;
        if (p.book) {
          pk.accepted = b.accepted;
          pk.next_pos = b.next_pos;
          pk.npos = mtp_n_ - 1;
          pk.next_slot = b.next_slot;
          if (p.book_sel) {
            pk.sel_src = b.sel;
            pk.sel_dst = b.mtp_sel;
            pk.vis_dst = b.mtp_vis;
            pk.sel_n = k::kSelPools;
            pk.kpool = k::kKpool;
          }
          booked = true;
        }
      }
      // (SR: a chained pass's summing form folded into the MoE router -- layer_moe launches k::mtp_sr_router in place
      // of the tensor-core router, or this norm when another router runs; bitwise)
      if (p.pick < 0 && !picked && p.ehf != nullptr && p.share &&
          k::mtp_sr_ok(T, H, pc_.tp, ar_max_elems_, cfg_.n_experts)) {
        sr_args_ = k::MtpGnArgs{};
        sr_args_.peers = ar_peers_;
        sr_args_.state = ar_epochs_ + ar_max_blocks_;
        sr_args_.rank = pc_.rank;
        sr_args_.max_elems = ar_max_elems_;
        sr_args_.resid = b.mtp_h;
        sr_args_.w = M.post_norm;
        sr_args_.out = b.x;
        sr_args_.eps = eps;
        sr_pend_ = true;
      } else {
        k::mtp_lamport_norm(T, H, false, ar_peers_, ar_epochs_, pc_.rank, pc_.tp, ar_max_elems_, ar_max_blocks_,
                            b.mtp_h, true, M.post_norm, b.x, nullptr, nullptr, eps, /*late=*/false, stream_,
                            picked ? &pk : nullptr);
      }
    } else {
      CHECK(!ar_pushed_, "MTP: pushed o_proj without its consumer");
      allreduce(b.out, (size_t)T * H);
      k::add_rmsnorm(b.mtp_h, b.out, nullptr, M.post_norm, b.x, nullptr, nullptr, T, H, eps, true, stream_);
    }
  }
  // pick >= 0 (pass 1): the MoE, the head norm and the draft head at each sequence's last accepted row only (mtp_pick
  // copies those rows; the other rows' MoE outputs feed nothing). b.x points at the picked rows while the MoE runs.
  const bool pick = p.pick >= 0;
  // Tensor parallel (TP 4, one-shot Lamport sizes): the MoE combine's all-reduce and the head norm of the same rows as
  // one kernel (k::mtp_lamport_combine_norm: bitwise the two kernels)
  const int moe_rows = pick ? p.pick : T;
  const bool moe_ln = pc_.tp == 4 && ar_enabled_ && moe_rows > 0 && (pick || p.rows == nullptr) &&
                      (pick ? p.pick : p.S) == moe_rows && cfg_.top_k + 1 <= k::kArMaxK1 &&
                      k::mtp_lamport_norm_ok(moe_rows, H, pc_.tp, ar_max_elems_);
  bool moe_normed = false;
  // (the head after the combine norm polls the norm's rows-ready count instead of its PDL wait; bitwise)
  const bool head_rdy = moe_ln && ar_enabled_ && head_keys_ok(moe_rows);
  auto moe_out = [&](int rows) {
    if (moe_ln) {
      k::MoeCombineArgs cm;
      cm.y = b.y;
      cm.w = b.topk_w;
      cm.pair_pos = b.pair_pos;
      cm.K1 = cfg_.top_k + 1;
      cm.hidden = H;
      // (late 2: the next kernel launches once the rows have arrived, before their arithmetic)
      k::mtp_lamport_combine_norm(rows, H, cm, ar_peers_, ar_epochs_, pc_.rank, pc_.tp, ar_max_elems_, ar_max_blocks_,
                                  pick ? b.mtp_hp : b.mtp_h, M.head_norm, p.g, eps, /*late=*/2, stream_,
                                  head_rdy ? b.head_rdy : nullptr);
      moe_normed = true;
    } else {
      moe_combine_allreduce(rows);
    }
  };
  if (pick) {
    if (p.pick > 0 && !picked)
      k::mtp_pick(b.mtp_xp, b.mtp_hp, nullptr, nullptr, nullptr, b.x, b.mtp_h, nullptr, nullptr, in.force_accept,
                  in.spec_row0, in.spec_nrows, b.out_tokens, in.ids, in.sample_idx, in.positions, in.token_seq,
                  p.pick, H, stream_);
    if (p.pick > 0) {
      ScopedSwap<bf16*> picked_x(b.x, b.mtp_xp);
      layer_moe(M.moe, p.pick);
      moe_out(p.pick);
    }
  } else {
    layer_moe(M.moe, T);
    moe_out(T);
  }
  CHECK(!sr_pend_, "MTP: the summing norm left pending (SR) without the MoE");
  // drafts: greedy over the shared head at the draft rows
  const int S = pick ? p.pick : p.S;
  if (S == 0) return booked;
  if (!moe_normed)
    k::add_rmsnorm(pick ? b.mtp_hp : b.mtp_h, b.out, pick ? nullptr : p.rows, M.head_norm, p.g, nullptr, nullptr, S,
                   H, eps, false, stream_);
  if (ar_enabled_ &&
      head_keys(p.g, S, nullptr, p.keys, p.tok_out, p.tok_ld, moe_normed && head_rdy ? b.head_rdy : nullptr))
    return booked;
  CHECK(!(moe_normed && head_rdy), "MTP: the combine norm signalled rows-ready for a head that did not run");
  CHECK(!p.tok_out, "MTP: the drafts' tokens need the fused head");
  const int Vl = w_.lm_head.n;
  gemm_->mm_bf16(p.g, H, w_.lm_head.w, H, b.logits_local, Vl, true, S, Vl, H, stream_);
  k::argmax_keys(p.keys, b.logits_local, S, Vl, pc_.rank * Vl, stream_);
  if (pc_.tp > 1) {
    if (ar_enabled_ && S <= k::kMaxKeyRows)
      k::custom_max_u64(p.keys, S, ar_peers_, ar_epochs_ + ar_max_blocks_ + 4, pc_.rank, pc_.tp, stream_);
    else
      NCCL_CHECK(ncclAllReduce(p.keys, p.keys, S, ncclUint64, ncclMax, comm_, stream_));
  }
  return booked;
}

void GlmStep::mtp_forward(int T, int S, int Bs, bool need_sel) {
  Buffers& b = *b_;
  const StepIn& in = *in_;
  const int H = cfg_.hidden;
  // pass 1 over the step's tokens: position p pairs the target hidden state at p with the token at p + 1 (known
  // context or sampled now); drafts at every sampled row
  MtpPass p1;
  p1.T = T;
  p1.ids = in.mtp_ids;
  p1.src = in.mtp_src;
  p1.positions = in.positions;
  p1.token_slot = in.token_seq;
  p1.hidden = b.hfin;
  p1.S = S;
  p1.rows = in.sample_idx;
  p1.g = b.head;
  p1.keys = b.draft_keys;
  // (pass 1's MoE and draft head at each sequence's last accepted row only: mtp_pass)
  p1.pick = Bs;
  // index share (the checkpoint's index_share_for_mtp_iteration, as TRT-LLM's draft loop): the chained passes reuse
  // the MTP layer's selection of each sequence's last accepted token in pass 1 and attend every row since
  const bool share = need_sel && mtp_n_ > 1;
  // the next pass's input rows (mtp_eh_norm) written by mtp_accept / mtp_next themselves
  const bool eh_fuse = H % 512 == 0 && H / 16 <= 1024;
  // No kernels between the passes (the hop; tensor parallel 4, the fused heads, the column-parallel input
  // projection): pass 1's attention consumer also does mtp_accept's bookkeeping (accepted drafts, the chained passes'
  // positions and rows, the shared selection), every head writes its drafts' tokens itself (head_keys), and each
  // chained pass's input projection builds its input rows from the previous pass's keys and head-norm rows (gemv_tma's
  // built rows: mtp_next's rows, bitwise) -- mtp_accept and both mtp_next go. At most kHopMaxSeqs draft sequences.
  constexpr int kHopMaxSeqs = 8;
  const bool hop = Bs <= kHopMaxSeqs && eh_fuse && H == 4096 && pc_.tp == 4 && ar_enabled_ && Bs >= 1 &&
                   head_keys_ok(Bs) && (mtp_n_ == 1 || mtp_eh_cp(Bs));
  if (hop) {
    p1.book = true;
    p1.book_sel = share;
    p1.tok_out = b.drafts;
    p1.tok_ld = mtp_n_;
  }
  const bool booked = mtp_pass(p1, need_sel);
  if (Bs == 0) return;
  if (hop && booked) {
    k::GemvEhArgs en;
    en.mode = 1;
    en.embed = w_.embed;
    en.enorm = w_.mtp.enorm;
    en.hnorm = w_.mtp.hnorm;
    en.eps = cfg_.rms_eps;
    for (int j = 2; j <= mtp_n_; ++j) {
      // (the previous pass's keys and head-norm rows: pass 1's per sequence, compact)
      en.keys = j == 2 ? b.draft_keys : b.draft_keys2;
      en.hid = j == 2 ? b.head : b.mtp_g;
      MtpPass pj;
      pj.T = Bs;
      pj.positions = b.next_pos + (size_t)(j - 2) * Bs;
      pj.token_slot = b.next_slot;
      pj.S = Bs;
      pj.g = b.mtp_g;
      pj.keys = b.draft_keys2;
      pj.share = share;
      pj.ehf = &en;
      pj.tok_out = b.drafts + (j - 1);
      pj.tok_ld = mtp_n_;
      mtp_pass(pj, need_sel);
    }
    return;
  }
  k::MtpEhArgs eh;
  eh.out = b.eh_in;
  eh.embed = w_.embed;
  eh.enorm = w_.mtp.enorm;
  eh.hnorm = w_.mtp.hnorm;
  eh.eps = cfg_.rms_eps;
  const bool fuse2 = eh_fuse && mtp_n_ > 1;
  // accepted drafts and the first new draft of every sampled sequence
  k::mtp_accept(b.accepted, b.drafts, mtp_n_, b.next_ids, b.next_pos, b.next_slot,
                mtp_n_ > 1 && !fuse2 ? b.next_hidden : nullptr, b.draft_keys, b.head, in.spec_row0, in.spec_nrows,
                b.out_tokens, in.ids, in.sample_idx, in.positions, in.token_seq, Bs, H, stream_, in.force_accept,
                share ? b.sel : nullptr, share ? b.mtp_sel : nullptr, share ? b.mtp_vis : nullptr, k::kSelPools,
                k::kKpool, fuse2 ? &eh : nullptr, p1.pick >= 0);
  // chained passes: the MTP layer re-applied to its own hidden state and draft
  for (int j = 2; j <= mtp_n_; ++j) {
    MtpPass pj;
    pj.T = Bs;
    pj.ids = b.next_ids;
    pj.positions = b.next_pos;
    pj.token_slot = b.next_slot;
    pj.hidden = b.next_hidden;
    pj.S = Bs;
    pj.g = b.mtp_g;
    pj.keys = b.draft_keys2;
    pj.share = share;
    pj.eh_ready = fuse2;  // (written by mtp_accept / the previous mtp_next)
    mtp_pass(pj, need_sel);
    const bool fuse_next = eh_fuse && j < mtp_n_;
    k::mtp_next(b.drafts, mtp_n_, j, b.next_ids, b.next_pos, j < mtp_n_ && !fuse_next ? b.next_hidden : nullptr,
                b.draft_keys2, b.mtp_g, Bs, H, stream_, fuse_next ? &eh : nullptr);
  }
}

}  // namespace glm
