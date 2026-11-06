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
// Torch bindings of the step program (step.h) and of the few launchers the Python side calls itself (the
// step-metadata kernel, the sampled rows' parameters, the MTP acceptance and the all-reduce buffers' initialization).
// The step program's structs are torch classes of this target's namespace, the launchers its ops. Device pointers
// cross as integers: the Python side owns the shape / dtype checks and passes torch tensors' data_ptr(). Everything
// runs on the current CUDA stream.
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/kernels/allreduce.h"
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/kernels/mla.h"
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/kernels/mla_ts.h"
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/kernels/sp_comm.h"
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/kernels/step_meta.h"
#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/step.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

using namespace glm;

template <typename T>
T* ptr(int64_t address)
{
    return reinterpret_cast<T*>(static_cast<uintptr_t>(address));
}

int64_t address(void const* p)
{
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(p));
}

cudaStream_t currentStream()
{
    return at::cuda::getCurrentCUDAStream().stream();
}

//! A struct of step.h as a torch class. An object owns its value, or refers into the object that owns it and keeps
//! that alive: a struct member reads as such a reference, so that `layer.kda.in_proj = w` sets the layer's weights.
template <typename T>
struct Obj : torch::CustomClassHolder
{
    std::shared_ptr<T> owned;
    T* p;
    c10::intrusive_ptr<torch::CustomClassHolder> owner;

    explicit Obj(std::shared_ptr<T> value)
        : owned(std::move(value))
        , p(owned.get())
    {
    }

    Obj(T* member, c10::intrusive_ptr<torch::CustomClassHolder> memberOwner)
        : p(member)
        , owner(std::move(memberOwner))
    {
    }
};

template <typename T>
using Ref = c10::intrusive_ptr<Obj<T>>;

template <typename T, typename... Args>
Ref<T> make(Args&&... args)
{
    return c10::make_intrusive<Obj<T>>(std::make_shared<T>(std::forward<Args>(args)...));
}

//! The torch type a field crosses as: bool, float for floating point, int for integers, enums and device pointers.
template <typename F>
using Scalar = std::conditional_t<std::is_same_v<F, bool>, bool,
    std::conditional_t<std::is_floating_point_v<F>, double, int64_t>>;

template <typename F>
Scalar<F> toTorch(F v)
{
    if constexpr (std::is_pointer_v<F>)
    {
        return address(v);
    }
    else
    {
        return static_cast<Scalar<F>>(v);
    }
}

template <typename F>
F fromTorch(Scalar<F> v)
{
    if constexpr (std::is_pointer_v<F>)
    {
        return ptr<std::remove_pointer_t<F>>(v);
    }
    else
    {
        if constexpr (std::is_integral_v<F> && !std::is_same_v<F, bool>)
        {
            TORCH_CHECK(static_cast<Scalar<F>>(static_cast<F>(v)) == v, v, " is out of the field's range");
        }
        return static_cast<F>(v);
    }
}

template <typename T, typename F>
void field(torch::class_<Obj<T>>& cls, std::string const& name, F T::*member)
{
    if constexpr (std::is_class_v<F>)
    {
        cls.def_property(
            name, [member](Ref<T> const& self) { return c10::make_intrusive<Obj<F>>(&(self->p->*member), self); },
            [member](Ref<T> const& self, Ref<F> const& value) { self->p->*member = *value->p; });
    }
    else
    {
        cls.def_property(
            name, [member](Ref<T> const& self) { return toTorch(self->p->*member); },
            [member](Ref<T> const& self, Scalar<F> value) { self->p->*member = fromTorch<F>(value); });
    }
}

template <typename T, typename F>
std::pair<char const*, F T::*> f(char const* name, F T::*member)
{
    return {name, member};
}

//! A struct with a default constructor and the given fields.
template <typename T, typename... Fields>
torch::class_<Obj<T>> plain(torch::Library& m, char const* name, Fields... fields)
{
    auto cls = m.class_<Obj<T>>(name);
    cls.def(torch::init([] { return make<T>(); }));
    (field(cls, fields.first, fields.second), ...);
    return cls;
}

//! The peers of one collective: one buffer of each kind per rank.
template <typename Peers, typename A, typename B, typename C>
void peers(torch::Library& m, char const* name, int maxTp, A Peers::*a, B Peers::*b, C Peers::*c)
{
    m.class_<Obj<Peers>>(name).def(torch::init(
        [=](c10::List<int64_t> const& as, c10::List<int64_t> const& bs, c10::List<int64_t> const& cs)
        {
            TORCH_CHECK(as.size() == bs.size() && as.size() == cs.size() && as.size() <= static_cast<size_t>(maxTp),
                "one buffer of each kind per rank, at most ", maxTp, " ranks");
            auto p = make<Peers>();
            for (size_t r = 0; r < as.size(); ++r)
            {
                (p->p->*a)[r] = ptr<std::remove_pointer_t<std::remove_extent_t<A>>>(as.get(r));
                (p->p->*b)[r] = ptr<std::remove_pointer_t<std::remove_extent_t<B>>>(bs.get(r));
                (p->p->*c)[r] = ptr<std::remove_pointer_t<std::remove_extent_t<C>>>(cs.get(r));
            }
            return p;
        }));
}

//! Constants of the kernels' layouts, and the sizes the Python side allocates for them.
struct Layout : torch::CustomClassHolder
{
};

void registerLayout(torch::Library& m)
{
    auto cls = m.class_<Layout>("Layout");
    cls.def(torch::init<>());
    for (auto const& [name, value] : std::vector<std::pair<char const*, int64_t>>{
             {"mla_block", k::kMlaBlock},
             {"idx_page_rows", k::kIdxPageRows},
             {"idx_kg_row0", k::kIdxKgRow0},
             {"kpool", k::kKpool},
             {"sel_pools", k::kSelPools},
             {"kda_rec", k::kKdaRec},
             {"max_tp", k::kMaxTp},
             {"max_key_rows", k::kMaxKeyRows},
             {"sp_flag_words", k::kSpFlagWords},
             {"sp_state_words", k::kSpStateWords},
             {"ts_flag_words", k::kTsFlagWords},
             {"ts_state_words", k::kTsStateWords},
             {"sample_params_bytes", static_cast<int64_t>(sizeof(k::SampleParams))},
             {"wfmt_nvfp4", static_cast<int64_t>(WFmt::kNvfp4)},
             {"wfmt_bf16", static_cast<int64_t>(WFmt::kBf16)},
         })
    {
        cls.def_property(name, [value = value](c10::intrusive_ptr<Layout> const&) { return value; });
    }
    cls.def("allreduce_buf_elems",
        [](c10::intrusive_ptr<Layout> const&, int64_t tp, int64_t maxElems)
        { return static_cast<int64_t>(k::allreduce_buf_elems(static_cast<int>(tp), static_cast<size_t>(maxElems))); });
    cls.def("attn_tile_tokens",
        [](c10::intrusive_ptr<Layout> const&, int64_t heads)
        { return static_cast<int64_t>(k::sparse_mla_tile_tokens(static_cast<int>(heads))); });
}

void registerConfigs(torch::Library& m)
{
    plain<ModelConfig>(m, "ModelConfig", f("hidden", &ModelConfig::hidden), f("vocab", &ModelConfig::vocab),
        f("rms_eps", &ModelConfig::rms_eps), f("hc_mult", &ModelConfig::hc_mult), f("hc_eps", &ModelConfig::hc_eps),
        f("hc_iters", &ModelConfig::hc_iters), f("hc_post_mult", &ModelConfig::hc_post_mult),
        f("kda_heads", &ModelConfig::kda_heads), f("kda_head_dim", &ModelConfig::kda_head_dim),
        f("kda_conv", &ModelConfig::kda_conv), f("kda_lower_bound", &ModelConfig::kda_lower_bound),
        f("num_heads", &ModelConfig::num_heads), f("q_lora", &ModelConfig::q_lora), f("kv_lora", &ModelConfig::kv_lora),
        f("qk_nope", &ModelConfig::qk_nope), f("v_head", &ModelConfig::v_head), f("idx_heads", &ModelConfig::idx_heads),
        f("idx_head_dim", &ModelConfig::idx_head_dim), f("inter", &ModelConfig::inter),
        f("moe_inter", &ModelConfig::moe_inter), f("n_experts", &ModelConfig::n_experts),
        f("top_k", &ModelConfig::top_k), f("routed_scale", &ModelConfig::routed_scale),
        f("swiglu_limit", &ModelConfig::swiglu_limit))
        .def("set_layers",
            [](Ref<ModelConfig> const& self, c10::List<bool> const& kda, c10::List<bool> const& moe)
            {
                TORCH_CHECK(kda.size() == moe.size(), "ModelConfig: one entry per layer");
                ModelConfig& c = *self->p;
                c.attn_types.clear();
                c.moe_layer.clear();
                for (size_t i = 0; i < kda.size(); ++i)
                {
                    c.attn_types.push_back(kda.get(i) ? AttnType::kKDA : AttnType::kMLA);
                    c.moe_layer.push_back(moe.get(i));
                }
                c.num_layers = static_cast<int>(kda.size());
            });
    plain<ParallelConfig>(m, "ParallelConfig", f("tp", &ParallelConfig::tp), f("rank", &ParallelConfig::rank),
        f("moe_tp", &ParallelConfig::moe_tp), f("moe_ep", &ParallelConfig::moe_ep));
    plain<RunnerConfig>(m, "RunnerConfig", f("max_num_tokens", &RunnerConfig::max_num_tokens),
        f("max_batch", &RunnerConfig::max_batch), f("max_seq_len", &RunnerConfig::max_seq_len),
        f("mtp_drafts", &RunnerConfig::mtp_drafts));
}

//! The weights: step.h's structs over device pointers the Python side owns.
void registerWeights(torch::Library& m)
{
    plain<Bf16W>(m, "Bf16W", f("w", &Bf16W::w), f("n", &Bf16W::n), f("k", &Bf16W::k));
    plain<Fp4W>(m, "Fp4W", f("w", &Fp4W::w), f("sf", &Fp4W::sf), f("alpha", &Fp4W::alpha), f("gs", &Fp4W::gs),
        f("n", &Fp4W::n), f("k", &Fp4W::k));
    plain<HcW>(m, "HcW", f("fn", &HcW::fn), f("base", &HcW::base), f("scale", &HcW::scale));
    plain<KdaW>(m, "KdaW", f("in_proj", &KdaW::in_proj), f("f_b", &KdaW::f_b), f("g_b", &KdaW::g_b),
        f("conv", &KdaW::conv), f("a_log", &KdaW::a_log), f("dt_bias", &KdaW::dt_bias), f("o_norm", &KdaW::o_norm),
        f("o_proj", &KdaW::o_proj));
    plain<MlaW>(m, "MlaW", f("qkv_a16", &MlaW::qkv_a16), f("q_b16", &MlaW::q_b16), f("o_proj16", &MlaW::o_proj16),
        f("q_a_norm", &MlaW::q_a_norm), f("kv_a_norm", &MlaW::kv_a_norm), f("w_kc", &MlaW::w_kc),
        f("w_vc", &MlaW::w_vc), f("w_kc_all", &MlaW::w_kc_all), f("w_vc_all", &MlaW::w_vc_all),
        f("q_b16_all", &MlaW::q_b16_all), f("o_proj16_all", &MlaW::o_proj16_all), f("w_kcT", &MlaW::w_kcT),
        f("idx_wq", &MlaW::idx_wq), f("idx_kgw", &MlaW::idx_kgw), f("idx_kn_w", &MlaW::idx_kn_w),
        f("idx_kn_b", &MlaW::idx_kn_b), f("idx_ape", &MlaW::idx_ape));
    plain<MlpW>(m, "MlpW", f("fmt", &MlpW::fmt), f("gate_up4", &MlpW::gate_up4), f("down4", &MlpW::down4),
        f("alpha4", &MlpW::alpha4), f("inter_local", &MlpW::inter_local));
    plain<k::MoeGroupMap>(m, "MoeGroupMap", f("E", &k::MoeGroupMap::E), f("lo", &k::MoeGroupMap::lo),
        f("n", &k::MoeGroupMap::n), f("ep", &k::MoeGroupMap::ep), f("ep_rank", &k::MoeGroupMap::ep_rank));
    plain<MoeWeights>(m, "MoeWeights", f("router", &MoeWeights::router), f("bias", &MoeWeights::bias),
        f("fmt", &MoeWeights::fmt), f("w13_4", &MoeWeights::w13_4), f("sf13", &MoeWeights::sf13),
        f("alpha13", &MoeWeights::alpha13), f("w2_4", &MoeWeights::w2_4), f("sf2", &MoeWeights::sf2),
        f("alpha2", &MoeWeights::alpha2), f("gs13", &MoeWeights::gs13), f("gs2", &MoeWeights::gs2),
        f("w13_16", &MoeWeights::w13_16), f("w2_16", &MoeWeights::w2_16), f("sh_gate_up", &MoeWeights::sh_gate_up),
        f("sh_down", &MoeWeights::sh_down), f("sh_inter", &MoeWeights::sh_inter),
        f("inter_local", &MoeWeights::inter_local), f("num_experts", &MoeWeights::num_experts),
        f("local", &MoeWeights::local));
    plain<LayerW>(m, "LayerW", f("moe", &LayerW::moe), f("hc_attn", &LayerW::hc_attn), f("hc_ffn", &LayerW::hc_ffn),
        f("in_norm", &LayerW::in_norm), f("post_norm", &LayerW::post_norm), f("kda", &LayerW::kda),
        f("mla", &LayerW::mla), f("mlp", &LayerW::mlp), f("moe_w", &LayerW::moe_w))
        .def_property(
            "kda_layer", [](Ref<LayerW> const& self) { return self->p->attn == AttnType::kKDA; },
            [](Ref<LayerW> const& self, bool kda) { self->p->attn = kda ? AttnType::kKDA : AttnType::kMLA; });
    plain<MtpW>(m, "MtpW", f("present", &MtpW::present), f("enorm", &MtpW::enorm), f("hnorm", &MtpW::hnorm),
        f("eh_proj", &MtpW::eh_proj), f("in_norm", &MtpW::in_norm), f("post_norm", &MtpW::post_norm),
        f("mla", &MtpW::mla), f("moe", &MtpW::moe), f("head_norm", &MtpW::head_norm));
    // a step program's weights (GlmStep.weights()), not constructed on their own
    auto model = m.class_<Obj<ModelW>>("ModelW");
    field(model, "embed", &ModelW::embed);
    field(model, "final_norm", &ModelW::final_norm);
    field(model, "lm_head", &ModelW::lm_head);
    field(model, "mtp", &ModelW::mtp);
    model.def("layer",
        [](Ref<ModelW> const& self, int64_t i)
        {
            TORCH_CHECK(i >= 0 && i < static_cast<int64_t>(self->p->layers.size()), "no layer ", i);
            return c10::make_intrusive<Obj<LayerW>>(&self->p->layers[i], self);
        });
}

void registerPoolsAndStep(torch::Library& m)
{
    peers<k::ArPeers>(m, "ArPeers", k::kMaxTp, &k::ArPeers::buf, &k::ArPeers::flags, &k::ArPeers::keys);
    peers<k::SpPeers>(m, "SpPeers", k::kSpMaxTp, &k::SpPeers::out, &k::SpPeers::x, &k::SpPeers::flags);
    peers<k::TsPeers>(m, "TsPeers", k::kSpMaxTp, &k::TsPeers::out, &k::TsPeers::x, &k::TsPeers::flags);
    // the runtime's pools: per sparse layer the latent and INDEX_KEY pools, per KDA layer the states, the conv states
    // and (MTP) the drafts' replay records, with their strides
    m.class_<Obj<Pools>>("Pools").def(torch::init(
        [](c10::List<int64_t> const& lat, c10::List<int64_t> const& idx, c10::List<int64_t> const& latRows,
            c10::List<int64_t> const& idxRows, c10::List<int64_t> const& state, c10::List<int64_t> const& conv,
            c10::List<int64_t> const& rec, c10::List<int64_t> const& crec, int64_t latPageRows, int64_t idxPageRows,
            int64_t stateSs, int64_t convSs, int64_t recSs, int64_t crecSs, int64_t crecSec, int64_t crecRs)
        {
            TORCH_CHECK(lat.size() == idx.size() && lat.size() == latRows.size() && lat.size() == idxRows.size(),
                "Pools: one of each per sparse layer");
            TORCH_CHECK(state.size() == conv.size() && state.size() == rec.size() && state.size() == crec.size(),
                "Pools: one of each per KDA layer");
            auto pools = make<Pools>();
            Pools& p = *pools->p;
            for (size_t i = 0; i < lat.size(); ++i)
            {
                p.mla.push_back(MlaPool{ptr<bf16>(lat.get(i)), ptr<bf16>(idx.get(i)), latRows.get(i), idxRows.get(i)});
            }
            for (size_t i = 0; i < state.size(); ++i)
            {
                p.kda.push_back(KdaPool{
                    ptr<float>(state.get(i)), ptr<bf16>(conv.get(i)), ptr<float>(rec.get(i)), ptr<bf16>(crec.get(i))});
            }
            p.lat_page_rows = latPageRows;
            p.idx_page_rows = idxPageRows;
            p.state_ss = stateSs;
            p.conv_ss = convSs;
            p.rec_ss = recSs;
            p.crec_ss = crecSs;
            p.crec_sec = crecSec;
            p.crec_rs = crecRs;
            return pools;
        }));
    plain<StepIn>(m, "StepIn", f("T", &StepIn::T), f("B", &StepIn::B), f("nd", &StepIn::nd), f("dt", &StepIn::dt),
        f("S", &StepIn::S), f("Bs", &StepIn::Bs), f("need_sel", &StepIn::need_sel), f("graph", &StepIn::graph),
        f("head", &StepIn::head), f("ids", &StepIn::ids), f("positions", &StepIn::positions),
        f("token_seq", &StepIn::token_seq), f("cu_seqlens", &StepIn::cu_seqlens), f("kda_slots", &StepIn::kda_slots),
        f("has_state", &StepIn::has_state), f("kda_replay", &StepIn::kda_replay),
        f("block_tables", &StepIn::block_tables), f("bt_stride", &StepIn::bt_stride),
        f("seq_tiles", &StepIn::seq_tiles), f("kda_chunks", &StepIn::kda_chunks),
        f("kda_seq_chunks", &StepIn::kda_seq_chunks), f("num_chunks", &StepIn::num_chunks),
        f("idx_tiles", &StepIn::idx_tiles), f("idx_split", &StepIn::idx_split), f("attn_tiles", &StepIn::attn_tiles),
        f("num_attn_tiles", &StepIn::num_attn_tiles), f("score_stride", &StepIn::score_stride),
        f("sample_idx", &StepIn::sample_idx), f("sample_params", &StepIn::sample_params),
        f("mtp_ids", &StepIn::mtp_ids), f("mtp_src", &StepIn::mtp_src), f("spec_row0", &StepIn::spec_row0),
        f("spec_nrows", &StepIn::spec_nrows), f("force_accept", &StepIn::force_accept))
        .def("set_h_idx_tiles",
            [](Ref<StepIn> const& self, c10::List<std::tuple<int64_t, int64_t>> const& tiles)
            {
                StepIn& s = *self->p;
                s.h_idx_tiles.clear();
                for (size_t i = 0; i < tiles.size(); ++i)
                {
                    auto const [first, count] = static_cast<std::tuple<int64_t, int64_t>>(tiles.get(i));
                    s.h_idx_tiles.push_back(make_int2(static_cast<int>(first), static_cast<int>(count)));
                }
            });

    m.class_<Obj<GlmStep>>("GlmStep")
        .def(torch::init(
            [](Ref<ModelConfig> const& mc, Ref<ParallelConfig> const& pc, Ref<RunnerConfig> const& rc, int64_t device)
            { return make<GlmStep>(*mc->p, *pc->p, *rc->p, static_cast<int>(device)); }))
        .def("weights",
            [](Ref<GlmStep> const& self) { return c10::make_intrusive<Obj<ModelW>>(&self->p->weights(), self); })
        .def("allocate", [](Ref<GlmStep> const& self) { self->p->allocate(); })
        .def("set_max_seq_len",
            [](Ref<GlmStep> const& self, int64_t maxSeqLen) { self->p->set_max_seq_len(static_cast<int>(maxSeqLen)); })
        .def("max_seq_len", [](Ref<GlmStep> const& self) { return static_cast<int64_t>(self->p->max_seq_len()); })
        .def("set_allreduce",
            [](Ref<GlmStep> const& self, Ref<k::ArPeers> const& peers, int64_t maxElems, int64_t maxBlocks,
                int64_t epochs)
            {
                self->p->set_allreduce(
                    *peers->p, static_cast<size_t>(maxElems), static_cast<int>(maxBlocks), ptr<uint32_t>(epochs));
            })
        .def("set_sp",
            [](Ref<GlmStep> const& self, Ref<k::SpPeers> const& peers, int64_t state)
            { self->p->set_sp(*peers->p, ptr<uint32_t>(state)); })
        .def("set_ts",
            [](Ref<GlmStep> const& self, Ref<k::TsPeers> const& peers, int64_t state)
            { self->p->set_ts(*peers->p, ptr<uint32_t>(state)); })
        // NCCL over the tensor-parallel ranks: rank 0's id, as bytes, for every rank's init_nccl
        .def("nccl_unique_id",
            [](Ref<GlmStep> const&)
            {
                c10::List<int64_t> id;
                for (char const c : GlmStep::nccl_unique_id())
                {
                    id.push_back(static_cast<unsigned char>(c));
                }
                return id;
            })
        .def("init_nccl",
            [](Ref<GlmStep> const& self, c10::List<int64_t> const& id)
            {
                std::vector<char> bytes;
                for (size_t i = 0; i < id.size(); ++i)
                {
                    bytes.push_back(static_cast<char>(id.get(i)));
                }
                self->p->init_nccl(bytes);
            })
        .def("set_io",
            [](Ref<GlmStep> const& self, int64_t x, int64_t out) { self->p->set_io(ptr<bf16>(x), ptr<bf16>(out)); })
        .def("set_pools", [](Ref<GlmStep> const& self, Ref<Pools> const& pools) { self->p->set_pools(*pools->p); })
        .def("set_tune", [](Ref<GlmStep> const& self, bool on) { self->p->set_tune(on); })
        .def("forward",
            [](Ref<GlmStep> const& self, Ref<StepIn> const& in) { self->p->forward(*in->p, currentStream()); })
        .def("mtp", [](Ref<GlmStep> const& self, Ref<StepIn> const& in) { self->p->mtp(*in->p, currentStream()); })
        .def("outputs",
            [](Ref<GlmStep> const& self)
            {
                StepOut const out = self->p->outputs();
                return std::make_tuple(
                    address(out.hidden), address(out.tokens), address(out.drafts), address(out.accepted));
            })
        .def("head_buf", [](Ref<GlmStep> const& self) { return address(self->p->head_buf()); })
        .def("sm_count", [](Ref<GlmStep> const& self) { return static_cast<int64_t>(self->p->sm_count()); });
}

// ---- the launchers the Python side calls itself

//! The per-forward metadata from the runtime's buffers.
void stepMeta(int64_t cu, int64_t tokenSeq, int64_t hasState, int64_t replay, int64_t tiles, int64_t seqLens,
    int64_t hasInitial, int64_t slots, int64_t prevAccepted, int64_t prevStride, int64_t numSeqs, int64_t numContexts,
    int64_t numTokens)
{
    k::step_meta(ptr<int>(cu), ptr<int>(tokenSeq), ptr<int>(hasState), ptr<int>(replay), ptr<int2>(tiles),
        ptr<int>(seqLens), ptr<bool>(hasInitial), ptr<int>(slots), ptr<int>(prevAccepted), prevStride,
        static_cast<int>(numSeqs), static_cast<int>(numContexts), static_cast<int>(numTokens), currentStream());
}

//! The sampled rows' k::SampleParams from the runtime's temperatures and Philox seeds and offsets.
void specSampleParams(int64_t out, int64_t temps, int64_t seeds, int64_t offsets, int64_t rows, double greedyBelow)
{
    k::spec_sample_params(ptr<k::SampleParams>(out), ptr<float>(temps), ptr<int64_t>(seeds), ptr<int64_t>(offsets),
        static_cast<int>(rows), static_cast<float>(greedyBelow), currentStream());
}

//! What the MTP worker hands the runtime after a step: the accepted tokens, their counts, the next step's input
//! tokens, and the generation rows' KDA replay record.
void specAccept(int64_t newTokens, int64_t lens, int64_t next, int64_t tokens, int64_t drafts, int64_t accepted,
    int64_t row0, int64_t batchIdx, int64_t rows, int64_t slots, int64_t dummy, int64_t prevAccepted,
    int64_t prevStride, int64_t numSeqs, int64_t numContexts, int64_t numDrafts)
{
    k::spec_accept(ptr<int>(newTokens), ptr<int>(lens), ptr<int>(next), ptr<int>(tokens), ptr<int>(drafts),
        ptr<int>(accepted), ptr<int>(row0), ptr<int>(batchIdx), static_cast<int>(rows), ptr<int>(slots),
        ptr<bool>(dummy), ptr<int>(prevAccepted), prevStride, static_cast<int>(numSeqs), static_cast<int>(numContexts),
        static_cast<int>(numDrafts), currentStream());
}

//! Fills this rank's all-reduce receive buffers with the Lamport sentinel.
void allreduceLamportInit(int64_t buf, int64_t tp, int64_t maxElems)
{
    k::allreduce_lamport_init(ptr<bf16>(buf), static_cast<int>(tp), static_cast<size_t>(maxElems), currentStream());
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    TRTLLM_MV2_NS::registerLayout(m);
    TRTLLM_MV2_NS::registerConfigs(m);
    TRTLLM_MV2_NS::registerWeights(m);
    TRTLLM_MV2_NS::registerPoolsAndStep(m);
    m.def(
        "step_meta(int cu, int token_seq, int has_state, int replay, int tiles, int seq_lens, int has_initial, "
        "int slots, int prev_accepted, int prev_stride, int num_seqs, int num_contexts, int num_tokens) -> ()");
    m.def("spec_sample_params(int out, int temps, int seeds, int offsets, int rows, float greedy_below) -> ()");
    m.def(
        "spec_accept(int new_tokens, int lens, int next, int tokens, int drafts, int accepted, int row0, "
        "int batch_idx, int rows, int slots, int dummy, int prev_accepted, int prev_stride, int num_seqs, "
        "int num_contexts, int num_drafts) -> ()");
    m.def("allreduce_lamport_init(int buf, int tp, int max_elems) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("step_meta", &TRTLLM_MV2_NS::stepMeta);
    m.impl("spec_sample_params", &TRTLLM_MV2_NS::specSampleParams);
    m.impl("spec_accept", &TRTLLM_MV2_NS::specAccept);
    m.impl("allreduce_lamport_init", &TRTLLM_MV2_NS::allreduceLamportInit);
}
