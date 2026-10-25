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
// The step program (step.h) and the few launchers the Python side calls itself (the step-metadata kernel, the
// all-reduce buffers' initialization). Device pointers cross as integers; the Python side owns the shape / dtype
// checks and passes torch tensors' data_ptr() and the current CUDA stream.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/allreduce.h"
#include "kernels/mla.h"
#include "kernels/mla_ts.h"
#include "kernels/sp_comm.h"
#include "kernels/step_meta.h"
#include "step.h"

namespace py = pybind11;
using namespace glm;

using ptr = uintptr_t;
template <typename T>
static T* P(ptr p) {
  return reinterpret_cast<T*>(p);
}
static cudaStream_t S(ptr p) { return reinterpret_cast<cudaStream_t>(p); }

// a device-pointer field as an integer property
#define PTR(cls, T, f) \
  def_property(#f, [](const cls& s) { return (ptr)s.f; }, [](cls& s, ptr v) { s.f = reinterpret_cast<T*>(v); })
#define CPTR(cls, T, f) \
  def_property(#f, [](const cls& s) { return (ptr)s.f; }, [](cls& s, ptr v) { s.f = reinterpret_cast<const T*>(v); })

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "GLM-5.3-Flash step program";
  auto o = m.def_submodule("ops", "the step program and its helpers");
  // ---- constants of the kernels' layouts
  o.attr("mla_block") = k::kMlaBlock;
  o.attr("idx_page_rows") = k::kIdxPageRows;
  o.attr("idx_kg_row0") = k::kIdxKgRow0;
  o.attr("kpool") = k::kKpool;
  o.attr("sel_pools") = k::kSelPools;
  o.attr("kda_rec") = k::kKdaRec;
  o.attr("max_tp") = k::kMaxTp;
  o.attr("max_key_rows") = k::kMaxKeyRows;
  o.attr("sp_flag_words") = k::kSpFlagWords;
  o.attr("sp_state_words") = k::kSpStateWords;
  o.attr("ts_flag_words") = k::kTsFlagWords;
  o.attr("ts_state_words") = k::kTsStateWords;
  o.attr("sample_params_bytes") = sizeof(k::SampleParams);
  // ---- per-forward metadata from the runtime's buffers
  o.def("step_meta", [](ptr cu, ptr token_seq, ptr has_state, ptr replay, ptr tiles, ptr seq_lens, ptr has_initial,
                        ptr slots, ptr prev_accepted, int64_t prev_stride, int B, int num_contexts, int T, ptr st) {
    k::step_meta(P<int>(cu), P<int>(token_seq), P<int>(has_state), P<int>(replay), P<int2>(tiles), P<int>(seq_lens),
                 P<bool>(has_initial), P<int>(slots), P<int>(prev_accepted), prev_stride, B, num_contexts, T, S(st));
  });
  o.def("spec_sample_params", [](ptr out, ptr temps, ptr seeds, ptr offsets, int rows, float greedy_below, ptr st) {
    k::spec_sample_params(P<k::SampleParams>(out), P<float>(temps), P<int64_t>(seeds), P<int64_t>(offsets), rows,
                          greedy_below, S(st));
  });
  o.def("spec_accept", [](ptr new_tokens, ptr lens, ptr next, ptr tokens, ptr drafts, ptr accepted, ptr row0,
                          ptr batch_idx, int rows, ptr slots, ptr dummy, ptr prev_accepted, int64_t prev_stride,
                          int B, int num_contexts, int D, ptr st) {
    k::spec_accept(P<int>(new_tokens), P<int>(lens), P<int>(next), P<int>(tokens), P<int>(drafts), P<int>(accepted),
                   P<int>(row0), P<int>(batch_idx), rows, P<int>(slots), P<bool>(dummy), P<int>(prev_accepted),
                   prev_stride, B, num_contexts, D, S(st));
  });
  // ---- collectives' buffers (IPC-mapped across the ranks' processes)
  o.def("allreduce_buf_elems", &k::allreduce_buf_elems);
  o.def("attn_tile_tokens", &k::sparse_mla_tile_tokens);
  o.def("allreduce_lamport_init",
        [](ptr buf, int tp, size_t max_elems, ptr st) { k::allreduce_lamport_init(P<bf16>(buf), tp, max_elems, S(st)); });
  py::class_<k::ArPeers>(o, "ArPeers")
      .def(py::init([](std::vector<ptr> buf, std::vector<ptr> flags, std::vector<ptr> keys) {
        CHECK(buf.size() == flags.size() && buf.size() == keys.size() && buf.size() <= k::kMaxTp,
              "ArPeers: one buffer of each kind per rank");
        k::ArPeers p{};
        for (size_t r = 0; r < buf.size(); ++r) {
          p.buf[r] = P<bf16>(buf[r]);
          p.flags[r] = P<uint32_t>(flags[r]);
          p.keys[r] = P<unsigned long long>(keys[r]);
        }
        return p;
      }));
  py::class_<k::SpPeers>(o, "SpPeers")
      .def(py::init([](std::vector<ptr> out, std::vector<ptr> x, std::vector<ptr> flags) {
        CHECK(out.size() == x.size() && out.size() == flags.size() && out.size() <= k::kSpMaxTp,
              "SpPeers: one buffer of each kind per rank");
        k::SpPeers p{};
        for (size_t r = 0; r < out.size(); ++r) {
          p.out[r] = P<bf16>(out[r]);
          p.x[r] = P<bf16>(x[r]);
          p.flags[r] = P<uint32_t>(flags[r]);
        }
        return p;
      }));
  py::class_<k::TsPeers>(o, "TsPeers")
      .def(py::init([](std::vector<ptr> out, std::vector<ptr> x, std::vector<ptr> flags) {
        CHECK(out.size() == x.size() && out.size() == flags.size() && out.size() <= k::kSpMaxTp,
              "TsPeers: one buffer of each kind per rank");
        k::TsPeers p{};
        for (size_t r = 0; r < out.size(); ++r) {
          p.out[r] = P<bf16>(out[r]);
          p.x[r] = P<bf16>(x[r]);
          p.flags[r] = P<uint32_t>(flags[r]);
        }
        return p;
      }));
  // ---- configuration
  py::class_<ModelConfig>(o, "ModelConfig")
      .def(py::init<>())
      .def_readwrite("hidden", &ModelConfig::hidden)
      .def_readwrite("vocab", &ModelConfig::vocab)
      .def_readwrite("rms_eps", &ModelConfig::rms_eps)
      .def_readwrite("hc_mult", &ModelConfig::hc_mult)
      .def_readwrite("hc_eps", &ModelConfig::hc_eps)
      .def_readwrite("hc_iters", &ModelConfig::hc_iters)
      .def_readwrite("hc_post_mult", &ModelConfig::hc_post_mult)
      .def_readwrite("kda_heads", &ModelConfig::kda_heads)
      .def_readwrite("kda_head_dim", &ModelConfig::kda_head_dim)
      .def_readwrite("kda_conv", &ModelConfig::kda_conv)
      .def_readwrite("kda_lower_bound", &ModelConfig::kda_lower_bound)
      .def_readwrite("num_heads", &ModelConfig::num_heads)
      .def_readwrite("q_lora", &ModelConfig::q_lora)
      .def_readwrite("kv_lora", &ModelConfig::kv_lora)
      .def_readwrite("qk_nope", &ModelConfig::qk_nope)
      .def_readwrite("v_head", &ModelConfig::v_head)
      .def_readwrite("idx_heads", &ModelConfig::idx_heads)
      .def_readwrite("idx_head_dim", &ModelConfig::idx_head_dim)
      .def_readwrite("inter", &ModelConfig::inter)
      .def_readwrite("moe_inter", &ModelConfig::moe_inter)
      .def_readwrite("n_experts", &ModelConfig::n_experts)
      .def_readwrite("top_k", &ModelConfig::top_k)
      .def_readwrite("routed_scale", &ModelConfig::routed_scale)
      .def_readwrite("swiglu_limit", &ModelConfig::swiglu_limit)
      .def("set_layers", [](ModelConfig& c, std::vector<bool> kda, std::vector<bool> moe) {
        CHECK(kda.size() == moe.size(), "ModelConfig: one entry per layer");
        c.attn_types.clear();
        for (bool x : kda) c.attn_types.push_back(x ? AttnType::kKDA : AttnType::kMLA);
        c.moe_layer = moe;
        c.num_layers = (int)kda.size();
      });
  py::class_<ParallelConfig>(o, "ParallelConfig")
      .def(py::init<>())
      .def_readwrite("tp", &ParallelConfig::tp)
      .def_readwrite("rank", &ParallelConfig::rank)
      .def_readwrite("moe_tp", &ParallelConfig::moe_tp)
      .def_readwrite("moe_ep", &ParallelConfig::moe_ep);
  py::class_<RunnerConfig>(o, "RunnerConfig")
      .def(py::init<>())
      .def_readwrite("max_num_tokens", &RunnerConfig::max_num_tokens)
      .def_readwrite("max_batch", &RunnerConfig::max_batch)
      .def_readwrite("max_seq_len", &RunnerConfig::max_seq_len)
      .def_readwrite("mtp_drafts", &RunnerConfig::mtp_drafts);
  // ---- weights (step.h's structs; device pointers owned by the Python side)
  py::enum_<WFmt>(o, "WFmt").value("NVFP4", WFmt::kNvfp4).value("BF16", WFmt::kBf16);
  py::class_<Bf16W>(o, "Bf16W").def(py::init<>()).PTR(Bf16W, bf16, w).def_readwrite("n", &Bf16W::n).def_readwrite(
      "k", &Bf16W::k);
  py::class_<Fp4W>(o, "Fp4W")
      .def(py::init<>())
      .PTR(Fp4W, uint8_t, w)
      .PTR(Fp4W, uint8_t, sf)
      .def_readwrite("alpha", &Fp4W::alpha)
      .def_readwrite("gs", &Fp4W::gs)
      .def_readwrite("n", &Fp4W::n)
      .def_readwrite("k", &Fp4W::k);
  py::class_<HcW>(o, "HcW").def(py::init<>()).PTR(HcW, bf16, fn).PTR(HcW, float, base).PTR(HcW, float, scale);
  py::class_<KdaW>(o, "KdaW")
      .def(py::init<>())
      .def_readwrite("in_proj", &KdaW::in_proj)
      .def_readwrite("f_b", &KdaW::f_b)
      .def_readwrite("g_b", &KdaW::g_b)
      .PTR(KdaW, bf16, conv)
      .PTR(KdaW, float, a_log)
      .PTR(KdaW, float, dt_bias)
      .PTR(KdaW, bf16, o_norm)
      .def_readwrite("o_proj", &KdaW::o_proj);
  py::class_<MlaW>(o, "MlaW")
      .def(py::init<>())
      .def_readwrite("qkv_a16", &MlaW::qkv_a16)
      .def_readwrite("q_b16", &MlaW::q_b16)
      .def_readwrite("o_proj16", &MlaW::o_proj16)
      .PTR(MlaW, bf16, q_a_norm)
      .PTR(MlaW, bf16, kv_a_norm)
      .PTR(MlaW, bf16, w_kc)
      .PTR(MlaW, bf16, w_vc)
      .PTR(MlaW, bf16, w_kc_all)
      .PTR(MlaW, bf16, w_vc_all)
      .PTR(MlaW, bf16, q_b16_all)
      .PTR(MlaW, bf16, o_proj16_all)
      .PTR(MlaW, bf16, w_kcT)
      .def_readwrite("idx_wq", &MlaW::idx_wq)
      .def_readwrite("idx_kgw", &MlaW::idx_kgw)
      .PTR(MlaW, bf16, idx_kn_w)
      .PTR(MlaW, bf16, idx_kn_b)
      .PTR(MlaW, bf16, idx_ape);
  py::class_<MlpW>(o, "MlpW")
      .def(py::init<>())
      .def_readwrite("fmt", &MlpW::fmt)
      .def_readwrite("gate_up4", &MlpW::gate_up4)
      .def_readwrite("down4", &MlpW::down4)
      .PTR(MlpW, float, alpha4)
      .def_readwrite("inter_local", &MlpW::inter_local);
  py::class_<k::MoeGroupMap>(o, "MoeGroupMap")
      .def(py::init<>())
      .def_readwrite("E", &k::MoeGroupMap::E)
      .def_readwrite("lo", &k::MoeGroupMap::lo)
      .def_readwrite("n", &k::MoeGroupMap::n)
      .def_readwrite("ep", &k::MoeGroupMap::ep)
      .def_readwrite("ep_rank", &k::MoeGroupMap::ep_rank);
  py::class_<MoeW>(o, "MoeWeights")
      .def(py::init<>())
      .PTR(MoeW, bf16, router)
      .PTR(MoeW, float, bias)
      .def_readwrite("fmt", &MoeW::fmt)
      .PTR(MoeW, uint8_t, w13_4)
      .PTR(MoeW, uint8_t, sf13)
      .PTR(MoeW, float, alpha13)
      .PTR(MoeW, uint8_t, w2_4)
      .PTR(MoeW, uint8_t, sf2)
      .PTR(MoeW, float, alpha2)
      .def_readwrite("gs13", &MoeW::gs13)
      .def_readwrite("gs2", &MoeW::gs2)
      .PTR(MoeW, bf16, w13_16)
      .PTR(MoeW, bf16, w2_16)
      .def_readwrite("sh_gate_up", &MoeW::sh_gate_up)
      .def_readwrite("sh_down", &MoeW::sh_down)
      .def_readwrite("sh_inter", &MoeW::sh_inter)
      .def_readwrite("inter_local", &MoeW::inter_local)
      .def_readwrite("num_experts", &MoeW::num_experts)
      .def_readwrite("local", &MoeW::local);
  py::class_<LayerW>(o, "LayerW")
      .def(py::init<>())
      .def_property(
          "kda_layer", [](const LayerW& l) { return l.attn == AttnType::kKDA; },
          [](LayerW& l, bool v) { l.attn = v ? AttnType::kKDA : AttnType::kMLA; })
      .def_readwrite("moe", &LayerW::moe)
      .def_readwrite("hc_attn", &LayerW::hc_attn)
      .def_readwrite("hc_ffn", &LayerW::hc_ffn)
      .PTR(LayerW, bf16, in_norm)
      .PTR(LayerW, bf16, post_norm)
      .def_readwrite("kda", &LayerW::kda)
      .def_readwrite("mla", &LayerW::mla)
      .def_readwrite("mlp", &LayerW::mlp)
      .def_readwrite("moe_w", &LayerW::moe_w);
  py::class_<MtpW>(o, "MtpW")
      .def(py::init<>())
      .def_readwrite("present", &MtpW::present)
      .PTR(MtpW, bf16, enorm)
      .PTR(MtpW, bf16, hnorm)
      .def_readwrite("eh_proj", &MtpW::eh_proj)
      .PTR(MtpW, bf16, in_norm)
      .PTR(MtpW, bf16, post_norm)
      .def_readwrite("mla", &MtpW::mla)
      .def_readwrite("moe", &MtpW::moe)
      .PTR(MtpW, bf16, head_norm);
  py::class_<ModelW>(o, "ModelW")
      .PTR(ModelW, bf16, embed)
      .PTR(ModelW, bf16, final_norm)
      .def_readwrite("lm_head", &ModelW::lm_head)
      .def_readwrite("mtp", &ModelW::mtp)
      .def(
          "layer", [](ModelW& w, int i) -> LayerW& { return w.layers.at(i); },
          py::return_value_policy::reference_internal);
  // ---- pools
  py::class_<Pools>(o, "Pools").def(
      py::init([](std::vector<ptr> lat, std::vector<ptr> idx, std::vector<int64_t> lat_rows, std::vector<int64_t> idx_rows,
                  std::vector<ptr> state, std::vector<ptr> conv, std::vector<ptr> rec, std::vector<ptr> crec,
                  int64_t lat_page_rows, int64_t idx_page_rows, int64_t state_ss, int64_t conv_ss, int64_t rec_ss,
                  int64_t crec_ss, int64_t crec_sec, int64_t crec_rs) {
        CHECK(lat.size() == idx.size() && lat.size() == lat_rows.size() && lat.size() == idx_rows.size(),
              "Pools: one of each per sparse layer");
        CHECK(state.size() == conv.size() && state.size() == rec.size() && state.size() == crec.size(),
              "Pools: one of each per KDA layer");
        Pools p;
        for (size_t i = 0; i < lat.size(); ++i)
          p.mla.push_back(MlaPool{P<bf16>(lat[i]), P<bf16>(idx[i]), lat_rows[i], idx_rows[i]});
        for (size_t i = 0; i < state.size(); ++i)
          p.kda.push_back(KdaPool{P<float>(state[i]), P<bf16>(conv[i]), P<float>(rec[i]), P<bf16>(crec[i])});
        p.lat_page_rows = lat_page_rows;
        p.idx_page_rows = idx_page_rows;
        p.state_ss = state_ss;
        p.conv_ss = conv_ss;
        p.rec_ss = rec_ss;
        p.crec_ss = crec_ss;
        p.crec_sec = crec_sec;
        p.crec_rs = crec_rs;
        return p;
      }));
  // ---- a step
  py::class_<StepIn>(o, "StepIn")
      .def(py::init<>())
      .def_readwrite("T", &StepIn::T)
      .def_readwrite("B", &StepIn::B)
      .def_readwrite("nd", &StepIn::nd)
      .def_readwrite("dt", &StepIn::dt)
      .def_readwrite("S", &StepIn::S)
      .def_readwrite("Bs", &StepIn::Bs)
      .def_readwrite("need_sel", &StepIn::need_sel)
      .def_readwrite("graph", &StepIn::graph)
      .def_readwrite("head", &StepIn::head)
      .CPTR(StepIn, int, ids)
      .CPTR(StepIn, int, positions)
      .CPTR(StepIn, int, token_seq)
      .CPTR(StepIn, int, cu_seqlens)
      .CPTR(StepIn, int, kda_slots)
      .CPTR(StepIn, int, has_state)
      .CPTR(StepIn, int, kda_replay)
      .CPTR(StepIn, int64_t, block_tables)
      .def_readwrite("bt_stride", &StepIn::bt_stride)
      .CPTR(StepIn, int2, seq_tiles)
      .CPTR(StepIn, int2, kda_chunks)
      .CPTR(StepIn, int, kda_seq_chunks)
      .def_readwrite("num_chunks", &StepIn::num_chunks)
      .CPTR(StepIn, int2, idx_tiles)
      .def("set_h_idx_tiles",
           [](StepIn& s, const std::vector<std::pair<int, int>>& t) {
             s.h_idx_tiles.clear();
             for (const auto& [a, n] : t) s.h_idx_tiles.push_back(make_int2(a, n));
           })
      .def_readwrite("idx_split", &StepIn::idx_split)
      .CPTR(StepIn, int2, attn_tiles)
      .def_readwrite("num_attn_tiles", &StepIn::num_attn_tiles)
      .def_readwrite("score_stride", &StepIn::score_stride)
      .CPTR(StepIn, int, sample_idx)
      .CPTR(StepIn, k::SampleParams, sample_params)
      .CPTR(StepIn, int, mtp_ids)
      .CPTR(StepIn, int, mtp_src)
      .CPTR(StepIn, int, spec_row0)
      .CPTR(StepIn, int, spec_nrows)
      .def_readwrite("force_accept", &StepIn::force_accept);
  py::class_<GlmStep>(o, "GlmStep")
      .def(py::init<const ModelConfig&, const ParallelConfig&, const RunnerConfig&, int>())
      .def("weights", &GlmStep::weights, py::return_value_policy::reference_internal)
      .def("allocate", &GlmStep::allocate)
      .def("set_max_seq_len", &GlmStep::set_max_seq_len)
      .def("max_seq_len", &GlmStep::max_seq_len)
      .def("set_allreduce",
           [](GlmStep& s, const k::ArPeers& peers, size_t max_elems, int max_blocks, ptr epochs) {
             s.set_allreduce(peers, max_elems, max_blocks, P<uint32_t>(epochs));
           })
      .def("set_sp", [](GlmStep& s, const k::SpPeers& peers, ptr state) { s.set_sp(peers, P<uint32_t>(state)); })
      .def("set_ts", [](GlmStep& s, const k::TsPeers& peers, ptr state) { s.set_ts(peers, P<uint32_t>(state)); })
      .def_static("nccl_unique_id",
                  [] {
                    const std::vector<char> id = GlmStep::nccl_unique_id();
                    return py::bytes(id.data(), id.size());
                  })
      .def("init_nccl",
           [](GlmStep& s, py::bytes id) {
             const std::string v = id;
             s.init_nccl(std::vector<char>(v.begin(), v.end()));
           })
      .def("set_io", [](GlmStep& s, ptr x, ptr out) { s.set_io(P<bf16>(x), P<bf16>(out)); })
      .def("set_pools", &GlmStep::set_pools)
      .def("set_tune", &GlmStep::set_tune)
      .def("forward", [](GlmStep& s, const StepIn& in, ptr st) { s.forward(in, S(st)); })
      .def("mtp", [](GlmStep& s, const StepIn& in, ptr st) { s.mtp(in, S(st)); })
      .def("outputs",
           [](const GlmStep& s) {
             const StepOut out = s.outputs();
             return py::make_tuple((ptr)out.hidden, (ptr)out.tokens, (ptr)out.drafts, (ptr)out.accepted);
           })
      .def("head_buf", [](const GlmStep& s) { return (ptr)s.head_buf(); })
      .def("sm_count", &GlmStep::sm_count);
}
