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
// The step program of one tensor-parallel rank (4 ranks: tensor-parallel attention; routed experts expert parallel or
// tensor parallel): a forward runs the decoder layers, the vocabulary-parallel LM head and the MTP draft passes over
// the TensorRT-LLM runtime's pools and metadata. Scheduling, the caches and CUDA graphs are the runtime's: a forward
// launches its kernels on the caller's stream (and side streams forked from and joined to it), so that the runtime
// captures a decode step's forward into its graph.
#pragma once

#include <nccl.h>

#include <memory>
#include <vector>

#include "common.h"
#include "gemm.h"
#include "kernels/allreduce.h"
#include "kernels/gemv_pre.h"
#include "kernels/kda.h"
#include "kernels/mla.h"
#include "kernels/mla_ts.h"
#include "kernels/moe.h"
#include "kernels/moe_fp4.h"
#include "kernels/nvfp4.h"
#include "kernels/sampling.h"
#include "kernels/sp_comm.h"

namespace glm {

enum class AttnType { kKDA, kMLA };

// The checkpoint's text config (filled by the Python side).
struct ModelConfig {
  int hidden = 4096;
  int vocab = 154880;
  int num_layers = 45;
  std::vector<AttnType> attn_types;
  std::vector<bool> moe_layer;  // true: sparse MoE, false: dense MLP
  float rms_eps = 1e-5f;
  int hc_mult = 4;
  float hc_eps = 1e-6f;
  int hc_iters = 20;
  float hc_post_mult = 2.0f;
  int kda_heads = 64;
  int kda_head_dim = 128;
  int kda_conv = 4;
  float kda_lower_bound = -5.0f;
  int num_heads = 64;
  int q_lora = 1536;
  int kv_lora = 512;
  int qk_nope = 256;
  int v_head = 256;
  int idx_heads = 32;
  int idx_head_dim = 128;
  int inter = 12288;
  int moe_inter = 2048;
  int n_experts = 288;
  int top_k = 8;
  float routed_scale = 2.5f;
  float swiglu_limit = 10.0f;
  int num_kda_layers() const;
  int num_mla_layers() const;  // decoder MLA layers (excludes MTP)
};

// Rank layout (TensorRT-LLM Mapping semantics): attention, dense MLPs and the LM head tensor parallel over the tp
// ranks; MoE layers moe_tp-way tensor parallel x moe_ep-way expert parallel (moe_tp * moe_ep = tp).
struct ParallelConfig {
  int tp = 1;
  int rank = 0;
  int moe_tp = 1, moe_ep = 1;
  int moe_tp_rank() const { return rank / moe_ep; }
  int moe_ep_rank() const { return rank % moe_ep; }
};

struct RunnerConfig {
  int max_num_tokens = 8192;  // tokens per step
  int max_batch = 256;        // sequences per step
  int max_seq_len = 32768;
  int mtp_drafts = 0;         // MTP draft tokens per step (0: no MTP)
};

// ---- weights (device pointers owned by the Python side)
struct Bf16W {
  bf16* w = nullptr;  // [n, k] row-major
  int n = 0, k = 0;
};
enum class WFmt { kNvfp4, kBf16 };
// NVFP4 linear [n, k]: packed e2m1 rows, scales in the CUTLASS SM100 layout (k::nvfp4_sf_off)
struct Fp4W {
  uint8_t* w = nullptr;   // [n, k/2]
  uint8_t* sf = nullptr;  // nvfp4_sf_bytes(n, k)
  float alpha = 0.f;      // input_scale * weight_scale_2: the GEMM's output scale
  float gs = 0.f;         // input_scale: global scale of the activation's quantization
  int n = 0, k = 0;
};
struct HcW {
  bf16* fn = nullptr;     // [mix(24), hc*hidden] bf16
  float* base = nullptr;  // [24]
  float* scale = nullptr; // [3]
};
struct KdaW {
  Bf16W in_proj;            // rows: q | k | v (local heads) | beta (local heads) | f_a | g_a
  Bf16W f_b, g_b;           // [Hl*D, D]
  bf16* conv = nullptr;     // [3*Hl*D, W] (q|k|v channels)
  float* a_log = nullptr;   // [Hl]
  float* dt_bias = nullptr; // [Hl*D]
  bf16* o_norm = nullptr;   // [D]
  Bf16W o_proj;             // [hidden, Hl*D]
};
struct MlaW {
  WFmt fmt = WFmt::kBf16;   // BF16 projections (the NVFP4 export leaves attention unquantized)
  Bf16W qkv_a16, q_b16, o_proj16;
  bf16* q_a_norm = nullptr;
  bf16* kv_a_norm = nullptr;
  bf16* w_kc = nullptr;     // [Hl, nope, kv_lora]
  bf16* w_vc = nullptr;     // [Hl, v, kv_lora]
  // all heads' (the token-split prefill attention); w_kc / w_vc then point at this rank's heads in them
  bf16* w_kc_all = nullptr;     // [H, nope, kv_lora]
  bf16* w_vc_all = nullptr;     // [H, v, kv_lora]
  bf16* q_b16_all = nullptr;    // [H nope, q_lora]
  bf16* o_proj16_all = nullptr; // [hidden, H v]
  bf16* w_kcT = nullptr;    // [Hl, kv_lora, nope]: w_kc transposed (decode absorb, head-batched GEMV)
  Bf16W idx_wq;             // [idx_heads*idx_hd, q_lora]
  Bf16W idx_kgw;            // [idx_hd (k) + idx_hd (gate) + idx_heads (w), hidden]
  bf16* idx_kn_w = nullptr;
  bf16* idx_kn_b = nullptr;
  bf16* idx_ape = nullptr;  // [kpool, idx_hd]
};
struct MlpW {
  WFmt fmt = WFmt::kNvfp4;
  // gate_up4 rows interleaved by 64 (k::kGu64); alpha4 (device): the two GEMMs' output scales
  // [gate_up4.alpha, down4.alpha]
  Fp4W gate_up4, down4;
  float* alpha4 = nullptr;
  int inter_local = 0;
};
// The rank's routed experts (NVFP4 main layers, BF16 MTP layer) and the separate BF16 shared expert, split over the
// ranks (each rank's partial output is a row of the MoE combine, summed by the all-reduce).
struct MoeW {
  bf16* router = nullptr;  // [E, hidden]
  float* bias = nullptr;   // [E]
  WFmt fmt = WFmt::kNvfp4;
  uint8_t* w13_4 = nullptr;  // [n, 2*Il, hidden/2], gate / up rows interleaved by 64 (kGu64)
  uint8_t* sf13 = nullptr;   // [n] x nvfp4_sf_bytes(2*Il, hidden)
  float* alpha13 = nullptr;  // [n] device
  uint8_t* w2_4 = nullptr;   // [n, hidden, Il/2]
  uint8_t* sf2 = nullptr;    // [n] x nvfp4_sf_bytes(hidden, Il)
  float* alpha2 = nullptr;   // [n] device
  float gs13 = 0.f, gs2 = 0.f;
  bf16* w13_16 = nullptr;  // fmt kBf16: [n, 2*Il, hidden] (kGu64)
  bf16* w2_16 = nullptr;   // [n, hidden, Il]
  Bf16W sh_gate_up;        // [2*Ils, hidden] (kGu64)
  Bf16W sh_down;           // [hidden, Ils]
  int sh_inter = 0;
  int inter_local = 0;
  int num_experts = 0;
  k::MoeGroupMap local;    // experts held by this rank
};
struct LayerW {
  AttnType attn = AttnType::kKDA;
  bool moe = false;
  HcW hc_attn, hc_ffn;
  bf16* in_norm = nullptr;
  bf16* post_norm = nullptr;
  KdaW kda;
  MlaW mla;
  MlpW mlp;
  MoeW moe_w;
};
struct MtpW {
  bool present = false;
  bf16* enorm = nullptr;
  bf16* hnorm = nullptr;
  Bf16W eh_proj;  // [hidden, 2*hidden] (replicated)
  bf16* in_norm = nullptr;
  bf16* post_norm = nullptr;
  MlaW mla;
  MoeW moe;
  bf16* head_norm = nullptr;
};
struct ModelW {
  bf16* embed = nullptr;  // [V, hidden] (replicated)
  bf16* final_norm = nullptr;
  Bf16W lm_head;          // [V / tp, hidden] vocab shard
  std::vector<LayerW> layers;
  MtpW mtp;
};

// ---- the runtime's pools (per KV cache manager; addresses and strides stable for its lifetime)
struct MlaPool {
  bf16* lat = nullptr;  // latent rows of kLatent, pages lat_page_rows rows apart
  bf16* idx = nullptr;  // INDEX_KEY rows of kIdxDim, pages idx_page_rows rows apart (k::kIdxKgRow0 layout)
  int64_t lat_rows = 0, idx_rows = 0;  // rows of the two buffers (tensor maps)
};
struct KdaPool {
  float* state = nullptr;  // the layer's recurrent states [slots, Hl, D, D] at state_ss per slot
  bf16* conv = nullptr;    // conv windows [slots, 3*Hl*D, W-1] at conv_ss per slot
  float* rec = nullptr;    // MTP: the drafts' replay records (KdaSeqMeta::rec_ss)
  bf16* crec = nullptr;    // MTP: the drafts' raw conv inputs (kda_crec_off)
};
struct Pools {
  std::vector<MlaPool> mla;  // decoder MLA layers, then the MTP layer
  std::vector<KdaPool> kda;
  int64_t lat_page_rows = 0, idx_page_rows = 0;
  int64_t state_ss = 0, conv_ss = 0, rec_ss = 0, crec_ss = 0, crec_sec = 0, crec_rs = 0;
};

// ---- one step's inputs (device arrays owned by the caller: the target's step metadata and the runtime's buffers)
struct StepIn {
  int T = 0;       // tokens
  int B = 0;       // sequences: the contexts first (the runtime's order), then the decode sequences
  int nd = 0;      // the last nd sequences decode dt tokens each (the fused KDA decode kernel)
  int dt = 1;
  int S = 0;       // sampled rows (the head's)
  int Bs = 0;      // MTP: sequences that draft (their sampled rows: spec_row0 / spec_nrows)
  bool need_sel = true;  // some token sees more than 512 complete pools
  bool graph = false;    // a decode step the caller captures (lists depend on B and dt only)
  int head = 0;          // 0 greedy, 1 vocab-parallel sampling (sample_params), -1 none (the caller's)
  // per token
  const int* ids = nullptr;
  const int* positions = nullptr;
  const int* token_seq = nullptr;  // batch row of each token (its block-table row)
  // per sequence
  const int* cu_seqlens = nullptr;  // [B + 1]
  const int* kda_slots = nullptr;   // [B] the runtime's state slots
  const int* has_state = nullptr;   // [B]
  const int* kda_replay = nullptr;  // [B] accepted drafts of the previous step to re-apply (MTP decode)
  const int64_t* block_tables = nullptr;  // [rows, bt_stride] V2 page ids
  int bt_stride = 0;
  const int2* seq_tiles = nullptr;  // [B] (first token, tokens) of every sequence (decode selection tiles)
  // eager steps: host-built work lists (device copies) of the step's tokens
  const int2* kda_chunks = nullptr;      // KDA 64-token chunks of the context sequences
  const int* kda_seq_chunks = nullptr;   // [contexts + 1]
  int num_chunks = 0;
  const int2* idx_tiles = nullptr;       // indexer query tiles (<= 8 consecutive tokens of a sequence)
  std::vector<int2> h_idx_tiles;
  int idx_split = 0;                     // > 0: the selection split over the ranks (rows [r C, (r + 1) C))
  const int2* attn_tiles = nullptr;      // query tiles of the tiled sparse attention
  int num_attn_tiles = 0;
  int score_stride = 0;                  // score-row length (eager: the longest context's pools; 0: max_pools)
  // head / MTP
  const int* sample_idx = nullptr;       // [S] rows sampled
  const k::SampleParams* sample_params = nullptr;  // [S] (head 1)
  const int* mtp_ids = nullptr;          // [T] MTP pass 1's next token per row (mtp_src < 0)
  const int* mtp_src = nullptr;          // [T] ... or the token sampled at that row
  const int* spec_row0 = nullptr;        // [Bs]
  const int* spec_nrows = nullptr;       // [Bs]
  // forced acceptance (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS): MTP accepts floor(v) drafts per sequence plus one
  // with probability frac(v), whatever the drafts; < 0: real acceptance
  float force_accept = -1.f;
};

// The step's outputs (device buffers of the step program)
struct StepOut {
  bf16* hidden = nullptr;   // [T, hidden] the final norm of every token (MTP), or nullptr
  int* tokens = nullptr;    // [S] the head's tokens
  int* drafts = nullptr;    // [Bs, mtp_drafts] the drafts for the next step
  int* accepted = nullptr;  // [Bs] drafts accepted in this step
};

class GlmStep {
 public:
  GlmStep(const ModelConfig& cfg, const ParallelConfig& pc, const RunnerConfig& rc, int device);
  ~GlmStep();
  ModelW& weights() { return w_; }
  // after the weights are set: the formats present, the buffers, the grouped GEMMs
  void allocate();
  // the longest sequence (pools per sequence of the selection and the attention); allocate() takes the runner config's
  int max_seq_len() const { return rc_.max_seq_len; }
  void set_max_seq_len(int max_seq_len);
  // the tensor-parallel all-reduce (Lamport, peer memory): every rank's receive buffers, flags and key slots
  // (IPC-mapped), and this rank's epochs; the sequence-parallel boundaries' and token-split attention's peers
  void set_allreduce(const k::ArPeers& peers, size_t max_elems, int max_blocks, uint32_t* epochs);
  void set_sp(const k::SpPeers& peers, uint32_t* state);
  void set_ts(const k::TsPeers& peers, uint32_t* state);
  // NCCL over the tensor-parallel ranks (large collectives): the id from rank 0
  static std::vector<char> nccl_unique_id();
  void init_nccl(const std::vector<char>& id);
  // the layer input / sublayer output rows ([max_num_tokens, hidden]; peer memory for the sequence-parallel
  // boundaries)
  void set_io(bf16* x, bf16* out);
  void set_pools(const Pools& p) { pools_ = p; }
  // cuBLASLt autotune (times candidates with host synchronization: eager decode steps only)
  void set_tune(bool on);
  // one step: the decoder layers, the head and (MTP) the draft passes, on stream st. in.head < 0: no head (the
  // sampled rows' final norm in head_buf()), and the draft passes only by mtp() once the caller has written the
  // sampled tokens to outputs().tokens
  void forward(const StepIn& in, cudaStream_t st);
  void mtp(const StepIn& in, cudaStream_t st);
  StepOut outputs() const;
  int sm_count() const { return sm_count_; }
  // the head's rows (the final norm of the sampled rows, [S, hidden])
  bf16* head_buf() const;

 private:
  struct Buffers;
  struct MtpPass {
    int T = 0;
    const int* ids = nullptr;
    const int* src = nullptr;
    const int* positions = nullptr;
    const int* token_slot = nullptr;
    const bf16* hidden = nullptr;
    int S = 0;
    const int* rows = nullptr;
    bf16* g = nullptr;
    unsigned long long* keys = nullptr;
    bool share = false;
    bool eh_ready = false;
    const k::GemvEhArgs* ehf = nullptr;
    int* tok_out = nullptr;
    int tok_ld = 1;
    bool book = false, book_sel = false;
    int pick = -1;
  };
  void forward_impl(const StepIn& in);
  void layer_kda(int li, const LayerW& L, int T, int B, int nd, int dt);
  void layer_mla(int li, int mi, const MlaW& W, int T, bool need_sel, const int* tok_slot = nullptr,
                 const int* positions = nullptr, bool share = false);
  void layer_mlp(const MlpW& W, int T);
  bool dense_fused(const MlpW& W, int T) const;
  void layer_moe(const MoeW& W, int T);
  void moe_route(const MoeW& W, int T);
  bool moe_experts_q(const MoeW& W, int T, const void* xq, const void* xs, int x_rows, const int* ids, int t0,
                     int shared_mode, const k::MoeRouteArgs* ra, int sh_T, bf16* sh_y, bool route_k);
  void moe_shared(const MoeW& W, int T, bf16* out);
  bool moe_shared_fork(const MoeW& W, int T, bf16* out);
  void moe_shared_join();
  void moe_combine_allreduce(int T);
  void allreduce(bf16* x, size_t n);
  bool rf_layer_ok(const MoeW& W, int T) const;
  void moe_pf_async(const MoeW& W, const int* hist);
  void l2pf_async(const void* p, size_t bytes);
  bool head_gemv(const bf16* A, int lda, int sA, const bf16* W, int ldw, long long sW, bf16* D, int ldd, int sD,
                 int M, int N, int K, int batch, bool late = true, int ctas_call = 0, int carveout = -1);
  void bf16_dense(const bf16* A, int lda, const bf16* W, int ldw, void* D, int ldd, bool d_f32, int M, int N, int K,
                  bool pipe_ok = true, int pre_ctas = 0, int stream_ctas = 0, bool lean = false, bool ep = false);
  bool head_keys(const bf16* rows, int S, const k::SampleParams* params, unsigned long long* keys, int* tokens,
                 int tokens_ld = 1, unsigned* ready = nullptr);
  bool head_keys_ok(int S) const;
  bool mtp_eh_cp(int T) const;
  bool mtp_pass(const MtpPass& p, bool need_sel);
  void mtp_forward(int T, int S, int Bs, bool need_sel);
  k::MlaTokMeta mla_meta(const int* tok_slot, const int* positions) const;
  k::KdaSeqMeta kda_meta(int b0) const;

  ModelConfig cfg_;
  ParallelConfig pc_;
  RunnerConfig rc_;
  int device_ = 0;
  int sm_count_ = 0;
  bool mtp_ = false;
  int mtp_n_ = 0;
  ModelW w_;
  Pools pools_;
  std::unique_ptr<Buffers> b_;
  const StepIn* in_ = nullptr;  // (the step in flight)
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<Gemm> gemm_;
  // MLA decode passes: the indexer branch on a side stream with its own cuBLASLt
  cudaStream_t side_stream_ = nullptr;
  std::unique_ptr<Gemm> gemm2_;
  cudaEvent_t ev_fork_ = nullptr, ev_q_ = nullptr, ev_sel_ = nullptr, ev_qi_ = nullptr, ev_score_ = nullptr;
  // L2 prefetches of upcoming weights (l2pf_async, moe_pf_async): a third stream
  cudaStream_t pf_stream_ = nullptr;
  cudaEvent_t ev_pf_ = nullptr, ev_pf_join_ = nullptr;
  cudaEvent_t ev_sh_fork_ = nullptr, ev_sh_join_ = nullptr;
  bool pf_ok_ = false, pf_forked_ = false;
  int* moe_hist_cur_ = nullptr;
  ncclComm_t comm_ = nullptr;
  // weight formats present
  bool moe_fp4_ = false, moe_bf16_ = false, mlp_fp4_ = false;
  bool mtp_moe_tp_ = false;  // the MTP layer's experts tensor-parallel while the main layers' are expert parallel
  std::unique_ptr<k::Nvfp4GroupedGemm> fp4_g1_, fp4_g2_;
  std::unique_ptr<k::Bf16GroupedGemm> bf_g1_, bf_g2_;
  std::unique_ptr<k::Nvfp4GroupedGemm> fp4_g1m_, fp4_g2m_;
  std::unique_ptr<k::Bf16GroupedGemm> bf_g1m_, bf_g2m_;
  std::vector<int> mla_index_, kda_index_;  // per layer: index among the MLA / KDA layers (-1: other)
  int Hl_kda_ = 0, Hl_mla_ = 0, N_in_ = 0;
  int attn_ws_ctas_ = 0;
  int dec_seqs_ = 0;  // the step's sequences when it is decode-only with 4 tokens each, else 0
  bool x4_ready_ = false;
  // RF: the tensor-core router's x copies (two sets by layer parity)
  bf16* rf_x_ = nullptr;
  bf16* rf_next_ = nullptr;
  bf16* rf_set_ = nullptr;
  bool rf_ready_ = false;
  // KDA LP: the TP in_proj's Lamport copies of the projections at one sequence: [2][4][N_in_]
  bf16* kda_lp_ = nullptr;
  bool moe_tm_req_ = false, moe_tm_out_ = false;
  // custom all-reduce
  k::ArPeers ar_peers_{};
  bool ar_enabled_ = false;
  size_t ar_max_elems_ = 0;
  int ar_max_blocks_ = 0;
  uint32_t* ar_epochs_ = nullptr;
  bool ar_push_req_ = false;
  int ar_push_T_ = 0;
  bool ar_pushed_ = false;
  bool att_pre_ = false;
  bool mtp_meta_pre_ = false;
  bool sr_pend_ = false;
  k::MtpGnArgs sr_args_{};
  // sequence-parallel boundaries and token-split MLA prefill attention (peer memory)
  k::SpPeers sp_peers_{};
  bool sp_p2p_ = false;
  uint32_t* sp_state_ = nullptr;
  k::TsPeers ts_peers_{};
  bool ts_p2p_ = false;
  uint32_t* ts_state_ = nullptr;
  bool ts_on_ = false, ts_step_ = false, ts_summed_ = false;
  std::vector<void*> allocs_;
  template <typename T>
  T* dalloc(size_t n);
};

}  // namespace glm
