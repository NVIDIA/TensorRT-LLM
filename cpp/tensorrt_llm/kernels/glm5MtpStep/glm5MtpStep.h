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
#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace glm5_mtp_step
{

//! \brief The GLM-5.3-Flash MTP draft step for few rows, as nine weight-streaming kernels.
//!
//! A draft step of the MTP layer, per tensor-parallel rank (TP4 attention, EP4 experts), is a chain of GEMVs over
//! at most kMaxRows rows separated by three all-reduces. Every kernel here is one weight-bandwidth-bound GEMV with
//! its producer and consumer fused in: a prologue builds the input rows in shared memory (an all-reduce of the
//! peers' partial sums plus residual and RMSNorm, a norm of the previous kernel's output, or a plain load), one warp
//! streams the weight rows through a ring of shared-memory stages, eight warps run the tile MMAs (BF16 m16n8k16 or
//! FP8 e4m3 m16n8k32 with 128 x 128 block scales), and an epilogue consumes each 16-row tile of outputs (store,
//! SwiGLU, push to the all-reduce, argmax). Consecutive kernels overlap through programmatic dependent launch: the
//! weights of a kernel whose rows do not depend on its predecessor are prefetched before griddepcontrol.wait.
//!
//!   pre      embedding / hidden RMSNorm (this rank's eh_proj input columns) -> eh_proj partial -> all-reduce #1 push
//!   qkvA     all-reduce #1 -> input_layernorm -> [q_a | kv_a] and indexer [wk | gate] projections
//!   qB       q_a / kv_a RMSNorms, indexer k LayerNorm (latent and packed index state out) -> q_b projection
//!   absorb   per-head query absorption into the latent space (q_latent for the attention backend)
//!   (the attention backend: cache append, pool keys, selection expansion, sparse MLA decode)
//!   vUp      per-head latent attention output -> V space
//!   oProj    o_proj partial -> all-reduce #2 push
//!   router   all-reduce #2 + residual -> post_attention_layernorm -> router logits, shared expert gate/up + SwiGLU
//!   fc1      routing (sigmoid, top-k by score + bias, normalized and scaled); this rank's experts' FC1 + SwiGLU,
//!            requantized to FP8 1 x 128; the shared expert's down projection
//!   fc2      this rank's experts' FC2, weighted sum, + shared expert -> all-reduce #3 push
//!   head     all-reduce #3 + residual -> shared_head.norm (the step's hidden out) -> LM head argmax + key exchange
//!
//! The all-reduces are Lamport exchanges through peer-mapped buffers: every rank pushes its bf16 partial into
//! every rank's buffer, and each reader polls its own buffer until no element holds the empty marker (-0.0; pushes
//! write -0.0 as +0.0). The three all-reduces of a step use buffers 0, 1 and 2 in that order, and the reader of
//! buffer e re-marks buffer (e + 2) % 3, whose readers have all finished. Sums run in fp32 in rank order.

//! Rows one call takes (the token columns of one MMA).
constexpr int kMaxRows = 8;
//! Ranks of the tensor-parallel group.
constexpr int kMaxRanks = 8;
//! Model hidden size.
constexpr int kHidden = 4096;
//! Rotating all-reduce buffers.
constexpr int kArBuffers = 3;
//! bf16 elements of one rank's all-reduce buffer: kArBuffers x kMaxRanks x kMaxRows x kHidden.
constexpr int64_t kArBufferElems = static_cast<int64_t>(kArBuffers) * kMaxRanks * kMaxRows * kHidden;
//! Experts one rank may run per call (the distinct local experts of kMaxRows tokens' top-k).
constexpr int kMaxSlots = 64;
//! Router fan-out.
constexpr int kMaxExperts = 512;
constexpr int kMaxTopK = 8;
//! Routed expert intermediate size.
constexpr int kMoeInter = 2048;
//! fp32 elements of the routed experts' intermediate (fc1 -> fc2): [kMaxSlots, kMaxRows, kMoeInter].
constexpr int64_t kInterElems = static_cast<int64_t>(kMaxSlots) * kMaxRows * kMoeInter;
//! int32 words of the route table (the router's routing for fc1 / fc2, and its tile counter; zero at rest).
constexpr int kRouteWords = 4096;

//! The all-reduce peers: every rank's buffer (kArBufferElems bf16, filled with -0.0 once), by rank.
struct AllReduce
{
    void* peers[kMaxRanks]{};
    int rank{0};
    int ranks{1};
};

//! pre: eh_proj partial of this rank's input columns, pushed to all-reduce buffer 0.
struct PreParams
{
    int m{0};
    int32_t const* tokens{nullptr}; //!< [m] (tokens[t * tokensStride]); embedding ranks
    int tokensStride{1};
    void const* hidden{nullptr};    //!< bf16 [m, kHidden] previous hidden (row stride hiddenStride); hidden ranks
    int64_t hiddenStride{0};
    void const* embed{nullptr};     //!< bf16 [vocab, kHidden]
    int vocab{0};
    bool useEmbed{false};           //!< this rank's columns come from the embedding (else from the hidden)
    void const* gamma{nullptr};     //!< bf16 [kHidden]: enorm (embedding ranks) or hnorm (hidden ranks)
    float eps{1e-5f};
    int colStart{0};                //!< first of the rank's kHidden / 2 normalized columns
    void const* weight{nullptr};    //!< bf16 [kHidden, kHidden / 2] eh_proj shard
    AllReduce ar;
};

//! qkvA: all-reduce #1 -> h (residual out) -> input_layernorm -> [q_a | kv_a] (2048) and [wk | gate] (256).
struct QkvAParams
{
    int m{0};
    AllReduce ar;
    void const* gamma{nullptr};     //!< bf16 [kHidden] input_layernorm
    float eps{1e-5f};
    void* residualOut{nullptr};     //!< bf16 [m, kHidden]
    void const* qkvWeight{nullptr}; //!< bf16 [qaRows + kvaRows, kHidden] kv_a_proj_with_mqa
    int qkvRows{0};
    void const* idxWeight{nullptr}; //!< bf16 [>= idxRows, kHidden] indexer wk_gate_wp (first idxRows rows)
    int idxRows{0};
    void* out{nullptr};             //!< bf16 [m, qkvRows + idxRows]
};

//! qB: q_a RMSNorm -> q_b projection; kv_a RMSNorm (latent) and the indexer's [LayerNorm(wk) | gate] (packed).
struct QBParams
{
    int m{0};
    void const* qkv{nullptr}; //!< bf16 [m, >= qaRank + kvRank + 2 * idxDim] (qkvA's output), row stride qkvStride
    int qkvStride{0};
    int qaRank{0};
    int kvRank{0};
    int idxDim{0};
    void const* qaGamma{nullptr};    //!< bf16 [qaRank]
    void const* kvGamma{nullptr};    //!< bf16 [kvRank]
    float eps{1e-5f};
    void const* lnWeight{nullptr};   //!< bf16 [idxDim] indexer k_norm
    void const* lnBias{nullptr};     //!< bf16 [idxDim]
    float lnEps{1e-6f};
    void* latentOut{nullptr};        //!< bf16 [m, kvRank]
    void* packedOut{nullptr};        //!< bf16 [m, 2 * idxDim]
    void const* weight{nullptr};     //!< bf16 [qbRows, qaRank] q_b_proj shard
    int qbRows{0};
    void* queryOut{nullptr};         //!< bf16 [m, qbRows]
    void const* idxQWeight{nullptr}; //!< bf16 [idxQRows, qaRank] the indexer's wq_b (step 0), or none (0 rows)
    int idxQRows{0};
    void* idxQOut{nullptr};          //!< bf16 [m, idxQRows] the indexer's query
};

//! absorb: q_latent[t, h, c] = sum_i query[t, h, i] * wkT[h, c, i].
struct AbsorbParams
{
    int m{0};
    void const* query{nullptr}; //!< bf16 [m, heads * nope]
    void const* wkT{nullptr};   //!< bf16 [heads, kvRank, nope]
    int heads{0};
    int nope{0};
    int kvRank{0};
    void* out{nullptr}; //!< bf16 [m, heads * kvRank]
};

//! vUp: v[t, h, j] = sum_c latent[t, h, c] * kvB[h * (nope + vDim) + nope + j, c].
struct VUpParams
{
    int m{0};
    void const* latent{nullptr}; //!< bf16 [m, heads * kvRank] (row stride latentStride)
    int64_t latentStride{0};
    void const* kvB{nullptr};    //!< bf16 [heads * (nope + vDim), kvRank] kv_b_proj shard
    int heads{0};
    int nope{0};
    int vDim{0};
    int kvRank{0};
    void* out{nullptr}; //!< bf16 [m, heads * vDim]
};

//! oProj: o_proj partial pushed to all-reduce buffer 1.
struct OProjParams
{
    int m{0};
    void const* input{nullptr};  //!< bf16 [m, k]
    int k{0};
    void const* weight{nullptr}; //!< bf16 [kHidden, k]
    AllReduce ar;
};

//! The routing: sigmoid scores, top-k by score + bias, weights normalized over the top-k x scale.
struct Routing
{
    float const* logits{nullptr}; //!< fp32 [m, experts]
    float const* bias{nullptr};   //!< fp32 [experts] e_score_correction_bias
    int experts{0};
    int topK{0};
    float scale{1.f};  //!< routed_scaling_factor
    int localStart{0}; //!< first expert of this rank
    int localExperts{0};
};

//! router: all-reduce #2 + residual -> post_attention_layernorm -> router logits and the shared expert's SwiGLU;
//! the last CTA of the logits tiles computes the routing (top-k, this rank's slots, each slot's tokens) into route.
struct RouterParams
{
    int m{0};
    AllReduce ar;
    void const* residualIn{nullptr};   //!< bf16 [m, kHidden] (qkvA's h)
    void* residualOut{nullptr};        //!< bf16 [m, kHidden]
    void const* gamma{nullptr};        //!< bf16 [kHidden]
    float eps{1e-5f};
    void* normOut{nullptr};            //!< bf16 [m, kHidden] (the experts' input)
    void const* gateWeight{nullptr};   //!< bf16 [experts, kHidden]
    int experts{0};
    float* logitsOut{nullptr};         //!< fp32 [m, experts]
    void const* sharedGateUp{nullptr}; //!< bf16 [2 * sharedInter, kHidden] ([gate; up])
    int sharedInter{0};
    float swigluLimit{10.f};
    void* sharedOut{nullptr};   //!< bf16 [m, sharedInter]
    Routing routing;            //!< logits = logitsOut
    int const* rowIdx{nullptr}; //!< [m] the all-reduce / residualIn rows to take (step 0's drafting rows); null: 0..m-1
    int* route{nullptr};        //!< [kRouteWords]: the last CTA of the logits tiles writes the routing here
};

//! fc1: this rank's experts' FC1 + clamped SwiGLU, requantized to FP8 1 x 128 per token; the shared expert's down.
struct Fc1Params
{
    int m{0};
    Routing routing;                 //!< topK, localStart, localExperts (the rest is the router's)
    int const* route{nullptr};       //!< the router's route table
    void const* input{nullptr};      //!< bf16 [m, kHidden] (router's normOut)
    void const* w13{nullptr};        //!< e4m3 [localExperts, 2 * kMoeInter, kHidden] ([up; gate])
    float const* w13Scale{nullptr};  //!< fp32 [localExperts, 2 * kMoeInter / 128, kHidden / 128]
    float swigluLimit{10.f};
    float* inter{nullptr};           //!< fp32 [kMaxSlots, kMaxRows, kMoeInter] the clamped SwiGLU per slot and token
    void const* sharedIn{nullptr};   //!< bf16 [m, sharedInter] (router's sharedOut)
    int sharedInter{0};
    void const* sharedDown{nullptr}; //!< bf16 [kHidden, sharedInter]
    void* sharedDownOut{nullptr};    //!< bf16 [m, kHidden]
};

//! fc2: this rank's experts' FC2, weighted by the routing, + the shared expert, pushed to all-reduce buffer 2.
struct Fc2Params
{
    int m{0};
    Routing routing;               //!< topK, localStart, localExperts (the rest is the router's)
    int const* route{nullptr};     //!< the router's route table
    void const* w2{nullptr};       //!< e4m3 [localExperts, kHidden, kMoeInter]
    float const* w2Scale{nullptr}; //!< fp32 [localExperts, kHidden / 128, kMoeInter / 128]
    float const* inter{nullptr};   //!< fc1's inter (requantized to e4m3 per token and 128 columns here)
    void const* shared{nullptr};   //!< bf16 [m, kHidden] fc1's sharedDownOut
    AllReduce ar;
};

//! head: all-reduce #3 + residual -> shared_head.norm (hidden out) -> vocabulary-parallel greedy token. The
//! scratch and key-exchange buffers are those of lm_head_argmax (the same protocol; calls on one stream may share
//! them).
struct HeadParams
{
    int m{0};
    AllReduce ar;
    void const* residualIn{nullptr}; //!< bf16 [m, kHidden] (router's residualOut)
    void const* gamma{nullptr};      //!< bf16 [kHidden]
    float eps{1e-5f};
    void* hiddenOut{nullptr};        //!< bf16 [m, kHidden] (row stride hiddenStride)
    int64_t hiddenStride{0};
    void const* weight{nullptr};     //!< bf16 [n, kHidden] this rank's vocabulary shard
    int n{0};
    int vocabOffset{0};
    int32_t* tokens{nullptr};        //!< [m] at tokens[t * tokensStride]
    int tokensStride{1};
    uint64_t* scratch{nullptr};      //!< lm_head_argmax::kScratchWords words
    uint64_t* keyPeers[kMaxRanks]{}; //!< every rank's lm_head_argmax exchange buffer
    int keyRank{0};
    int keyRanks{1};
};

//! Launches on stream; ctas is the grid size (typically the SM count; a kernel uses fewer when it has fewer tiles).
void launchPre(PreParams const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchQkvA(QkvAParams const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchQB(QBParams const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchAbsorb(AbsorbParams const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchVUp(VUpParams const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchOProj(OProjParams const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchRouter(RouterParams const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchFc1(Fc1Params const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchFc2(Fc2Params const& p, int ctas, cudaStream_t stream, bool enablePdl);
void launchHead(HeadParams const& p, int ctas, cudaStream_t stream, bool enablePdl);

} // namespace glm5_mtp_step
} // namespace kernels

TRTLLM_NAMESPACE_END
