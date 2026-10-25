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
// Token-split MLA prefill attention (tensor parallelism, sequence-parallel steps): the exchanges
// around the attention. Each rank computes q for its local heads and all T tokens; rank r then
// attends its quarter of the tokens [r Tq, r Tq + Tr) with all heads (one CTA per token, every
// cache row it reads shared by all heads: no union of neighbouring tokens' pools) and returns each
// rank's head columns of the output rows. Both moves are stores into the peers' buffers (the
// sequence-parallel boundary buffers, free during the attention sublayer) with per-source arrival
// counters, as sp_ag_push.
#pragma once

#include "../common.h"
#include "sp_comm.h"

namespace glm {
namespace k {

struct TsPeers {
  bf16* out[kSpMaxTp];  // every rank's b.out: the q staging ([Tq][tp hn])
  bf16* x[kSpMaxTp];    // every rank's b.x: the returned output rows ([T][hv])
  uint32_t* flags[kSpMaxTp];
};
constexpr int kTsFlagWords = 3 * kSpMaxTp;  // [kind (q, o, ready)][source rank] arrival counters
constexpr int kTsStateWords = 5;  // [q epoch, o epoch, q block counter, o block counter, ready epoch]

// q rows [T][hn] (this rank's hn = local heads x qk_nope columns) -> rank r's out rows i < Tr_r
// (Tr_r = min(Tq, T - r Tq)), columns [rank hn, rank hn + hn) of its [Tq][tp hn] staging; returns
// when every rank's rows arrived here.
void mla_ts_q_push(const TsPeers& peers, int tp, int rank, const bf16* q, int T, int Tq, int hn,
                   uint32_t* state, cudaStream_t st);
// this rank's attention output rows o [Tr][tp hv] -> rank p's x rows rank Tq + i, columns
// [0, hv) from o's columns [p hv, p hv + hv); returns when every rank's rows arrived here.
// wait_ready: the stores into rank p wait for p's mla_ts_ready of this call (without a q push
// nothing else orders them after p's last read of its x).
void mla_ts_o_push(const TsPeers& peers, int tp, int rank, const bf16* o, int Tr, int Tq, int hv,
                   uint32_t* state, cudaStream_t st, bool wait_ready = false);
// this rank's x is consumed (its projections are done): counted in at every rank
void mla_ts_ready(const TsPeers& peers, int tp, int rank, uint32_t* state, cudaStream_t st);
// returns when every rank's mla_ts_ready of this call arrived here
void mla_ts_wait_ready(const TsPeers& peers, int tp, int rank, uint32_t* state, cudaStream_t st);

}  // namespace k
}  // namespace glm
