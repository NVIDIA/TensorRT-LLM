# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: glm-5.3-flash-nvfp4 / sm_10x / tp4 — self-contained modeling code.

GLM-5.3-Flash (`glm5_next`), the ModelOpt NVFP4 export: 45 decoder layers — 34 KDA linear
attention (64 heads of 128, short conv 4, gated delta rule, gated output norm) and 11 DSA sparse
MLA layers (64 heads, q_lora 1536, NoPE latent 512, absorbed queries of 512, a pool-compressed
indexer of 32 heads selecting 512 pools of 4 tokens) — with mHC residual streams (4 streams,
Sinkhorn-normalized mixing at every sublayer boundary), dense MLPs in layers 0-2 (intermediate
12288) and 288-expert MoE layers 3-44 (top 8 sigmoid noaux_tc routing, scale 2.5, one shared
expert of 2048), clamped SwiGLU (10), and one MTP layer (index 45). Routed experts and the dense
MLPs are W4A4 NVFP4; attention, router, shared experts and the whole MTP layer are BF16. The
checkpoint is the text half of `Glm5NextForConditionalGeneration`; the vision tower is not built.

A forward is one call into this target's step program (`kernels/`: the decoder layers, the
vocabulary-parallel LM head and the MTP draft chain as CUDA kernels, compiled at first use for the
device's sm_100a / sm_103a) over the TensorRT-LLM runtime's buffers: scheduling, the glm5_next V2
cache manager (latent and INDEX_KEY pages of the 12 sparse layers, KDA recurrent / conv state
slots and the drafts' replay buffers, prefix reuse), CUDA graphs and the sampler are the
runtime's. The step program launches its kernels on the current stream (and its own side
streams, forked from and joined to it), so a decode step is captured whole into the runtime's
CUDA graph.

Topology (tp4; the mapping's MoE split selects one of two layouts): attention, the dense MLPs,
the shared expert and the LM head are tensor parallel over 4 ranks (16 KDA heads, 16 MLA heads,
3072 dense features, 512 shared-expert features and 38720 vocabulary rows per rank); routed
experts are expert parallel (moe_expert_parallel_size 4: 72 whole experts per rank) or tensor
parallel (moe_tensor_parallel_size 4: 512 features of every expert per rank, the faster layout
for small batches), tokens replicated. Every sublayer's partial output is summed by a Lamport
all-reduce over NVLink peer memory fused into the next mHC boundary (decode-sized steps), or
reduce-scattered with the boundary run on each rank's quarter of the rows (sequence-parallel
prefill steps, whose MLA layers attend their quarter with all 64 heads).

MTP (one-model MTP-Eagle, 3 drafts): the draft loop belongs to the target (`MTPStepWorker`).
One step-program call verifies the step's drafts (the head's token at every sampled row, strict
acceptance, or the forced acceptance of TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS), runs the
MTP layer over the step's tokens and its two chained passes from each sequence's last accepted
row, and leaves the accepted tokens and the next drafts in its buffers, which the worker hands
the sampler. The head samples greedy batches by argmax and temperature-only batches
(advanced_sampling_mode no_topk_no_topp) by Gumbel-max over the vocabulary shards; other
sampling modes take the runtime's sampler over gathered logits before the draft passes.

What this target sets in the runtime (read by the KV cache creator off the built model):
- the latent pool is BF16, the layout the kernels read; the checkpoint declares an FP8 KV cache
  without scales;
- a sparse layer's INDEX_KEY row holds the 288 values per token the kernels' page layout uses
  (the pool keys, then two [k | gate] rows per token) instead of the runtime's 3 x 128;
- `kda_draft_records` / `kda_commit_pending_drafts`: the KDA verify keeps each draft's replay
  record (k, the delta rule's update u, decay) in the slot's `kda_state_tok` buffer and its raw
  conv inputs in the tail rows of the `kda_conv_*` buffers; a suspended request's accepted drafts
  are committed from those by this target's commit.
"""

import copy
import functools
from typing import cast

import numpy as np
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import (
    Glm5NextCacheManager,
    Glm5NextMamba2Metadata,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config
from tensorrt_llm._torch.speculative import MTPEagleWorker
from tensorrt_llm._torch.speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, prefer_pinned
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode
from tensorrt_llm.logger import logger

from . import weights as _weights
from .kernels._lib import ops, stream
from .kernels.comm import AR_MAX_BLOCKS, Collectives

H = _weights.H
V = _weights.V
TP = _weights.TP
E = _weights.E
KDA_HL, KDA_D, KDA_HD, KDA_NIN = _weights.KDA_HL, _weights.KDA_D, _weights.KDA_HD, _weights.KDA_NIN
MLA_HEADS, MLA_HL = _weights.MLA_HEADS, _weights.MLA_HL
NOPE, VH, LORA, QLORA = _weights.NOPE, _weights.VH, _weights.LORA, _weights.QLORA
IDX_HEADS, IDX_D = _weights.IDX_HEADS, _weights.IDX_D
INTER_L, SH_IL = _weights.INTER_L, _weights.SH_IL
# Tokens per page of the latent and INDEX_KEY pools the kernels address (ops.mla_block), tokens
# per indexer pool (ops.kpool) and the pools a token's selection keeps (ops.sel_pools).
MLA_BLOCK = 128
KPOOL = 4
SEL_POOLS = 512
# INDEX_KEY rows of 128 a page's layout uses: the page's pool keys, then two [k | gate] rows per
# token; a sparse layer's INDEX_KEY row is set to the 288 values per token this makes.
IDX_PAGE_USED = MLA_BLOCK // KPOOL + 2 * MLA_BLOCK
IDX_ROW_VALUES = IDX_PAGE_USED * IDX_D // MLA_BLOCK
SWIGLU_LIMIT = 10.0
ROUTED_SCALE = 2.5
HC_EPS = 1e-6
HC_ITERS = 20
KDA_LOWER_BOUND = -5.0
MTP_DRAFTS = 3
# Sequences per step the step program takes: its head keys up to 1024 sampled rows (4 per
# sequence with MTP 3), its per-sequence buffers 256 + 1.
MAX_BATCH = 256
# Custom all-reduce capacity in tokens (larger messages: NCCL), eager steps from which the indexer
# selection is split over the ranks, query tiles from which the sparse attention runs tiled.
AR_MAX_TOKENS = 1024
IDX_SPLIT_MIN_T = 1024
ATTN_TILED_MIN = 64
# temperatures below this are greedy (the runtime's sampler threshold)
GREEDY_TEMPERATURE = 1e-4


def _runtime_model_config(model_config: ModelConfig) -> ModelConfig:
    """The model config the runtime builds this target's caches from (see the module docstring).

    The KV cache creator reads the cache layout off the built model's own config, so the latent
    dtype and the INDEX_KEY row width set here are what it allocates.
    """
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3)), (
        "glm-5.3-flash-nvfp4 / sm_10x / tp4 runs sm_100a / sm_103a kernels (tcgen05, tensor memory)"
    )
    # A shallow copy keeps every runtime-set field (``extra_attrs`` included, shared); only the
    # quantization the cache is built with changes.
    model_config = copy.copy(model_config)
    quant = model_config.quant_config
    if quant is not None and quant.kv_cache_quant_algo is not None:
        logger.info(
            f"glm-5.3-flash-nvfp4 / sm_10x / tp4: BF16 latent cache (the checkpoint declares "
            f"{quant.kv_cache_quant_algo} without scales; the kernels read BF16 latent rows)"
        )
        quant = copy.deepcopy(quant)
        quant.kv_cache_quant_algo = None
        # the quant modes derived from it are cached on the instance: drop them with it
        for klass in type(quant).__mro__:
            for name, attr in vars(klass).items():
                if isinstance(attr, functools.cached_property):
                    quant.__dict__.pop(name, None)
        model_config.quant_config = quant
    model_config.extra_attrs["glm_kpool_index_state_dim"] = IDX_ROW_VALUES
    return model_config


def _ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def _view(addr: int, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    """A tensor over memory the step program owns."""
    return convert_to_torch_tensor(TensorWrapper(addr, dtype, shape))


class _Pools:
    """The cache manager's per-layer pool views (stable for the manager's lifetime) as the step
    program's `Pools`, their strides checked against the layouts the kernels address."""

    def __init__(self, mgr, mla_layers: list[int], kda_layers: list[int], spec: bool):
        bf16, f32 = torch.bfloat16, torch.float32
        lat_l, idx_l, lat_rows, idx_rows = [], [], [], []
        lat_pr = idx_pr = None
        self.zero: list[torch.Tensor] = []
        for li in mla_layers:
            lat = mgr.get_latent_state_buffer(li)
            idx = mgr.get_index_state_buffer(li)
            assert lat is not None and idx is not None, f"layer {li}: no sparse-layer pools"
            assert lat.dtype == bf16 and idx.dtype == bf16, (
                f"layer {li}: the kernels read a BF16 latent pool, got {lat.dtype}"
            )
            slots, tpb = lat.shape[0], lat.shape[1]
            assert tpb == MLA_BLOCK, f"tokens_per_block must be {MLA_BLOCK}, got {tpb}"
            assert tuple(lat.shape[1:]) == (tpb, 1, LORA) and lat.stride()[1:] == (LORA, LORA, 1)
            w = idx.shape[-1]  # values per token: IDX_ROW_VALUES, or a wider default row
            assert tuple(idx.shape[1:]) == (tpb, 1, w) and idx.stride()[1:] == (w, w, 1)
            assert lat.stride(0) % LORA == 0 and idx.stride(0) % IDX_D == 0
            assert tpb * w // IDX_D >= IDX_PAGE_USED, "INDEX_KEY page smaller than its layout"
            lp, ip = lat.stride(0) // LORA, idx.stride(0) // IDX_D
            assert lat_pr in (None, lp) and idx_pr in (None, ip), (
                "page strides differ across layers"
            )
            lat_pr, idx_pr = lp, ip
            lat_l.append(lat.data_ptr())
            idx_l.append(idx.data_ptr())
            lat_rows.append((slots - 1) * lp + tpb)
            idx_rows.append((idx.shape[0] - 1) * ip + IDX_PAGE_USED)
            self.zero += [lat, idx]
        state_l, conv_l, rec_l, crec_l = [], [], [], []
        strides = None
        for li in kda_layers:
            st = mgr.mamba_layer_cache(li)
            conv, ssm = st.conv, st.temporal
            assert conv.dtype == bf16 and ssm.dtype == f32
            assert tuple(conv.shape[1:]) == (3 * KDA_HD, 3) and conv.stride()[1:] == (3, 1), (
                conv.shape
            )
            assert tuple(ssm.shape[1:]) == (KDA_HL, KDA_D, KDA_D)
            assert ssm.stride()[1:] == (KDA_D * KDA_D, KDA_D, 1)
            rec = crec = 0
            rs = (ssm.stride(0), conv.stride(0), 0, 0, 0, 0)
            if spec:
                assert st.kda_state_tok is not None, "MTP needs the runtime's KDA draft records"
                tok = st.kda_state_tok
                # a slot's [3, drafts, heads, 128] fp32 region holds the kernels' records:
                # [draft][head][k | u | decay]
                assert tok.dtype == f32 and tuple(tok.shape[1:]) == (3, MTP_DRAFTS, KDA_HL, KDA_D)
                assert tok.stride()[1:] == (MTP_DRAFTS * KDA_HL * KDA_D, KDA_HL * KDA_D, KDA_D, 1)
                assert ops.kda_rec == 3 * KDA_D
                # the drafts' raw conv inputs: columns [3, 6) of the dim-contiguous
                # [slots, 2048, 3 + drafts] replay conv caches, one per q | k | v section
                q, k, v = st.kda_conv_q, st.kda_conv_k, st.kda_conv_v
                for c in (q, k, v):
                    assert c.dtype == bf16 and tuple(c.shape[1:]) == (KDA_HD, 3 + MTP_DRAFTS)
                    assert c.stride()[1:] == (1, KDA_HD) and c.stride(0) == q.stride(0)
                sec = k.data_ptr() - q.data_ptr()
                assert sec % 2 == 0 and v.data_ptr() - k.data_ptr() == sec, (
                    "the KDA replay conv caches of a layer are not equally spaced"
                )
                rec = tok.data_ptr()
                crec = q.data_ptr() + 3 * KDA_HD * 2
                rs = (ssm.stride(0), conv.stride(0), tok.stride(0), q.stride(0), sec // 2, KDA_HD)
            assert strides in (None, rs), "state strides differ across layers"
            strides = rs
            state_l.append(ssm.data_ptr())
            conv_l.append(conv.data_ptr())
            rec_l.append(rec)
            crec_l.append(crec)
            self.zero += [conv, ssm]
        assert strides is not None and lat_pr is not None
        self.pools = ops.Pools(
            lat_l, idx_l, lat_rows, idx_rows, state_l, conv_l, rec_l, crec_l, lat_pr, idx_pr,
            *strides,
        )  # fmt: skip
        # (the runtime's accepted-draft count of each slot: a strided view into its state buffers;
        # the worker writes it after every step as the runtime's KDA-replay update_mamba_states
        # would, leaving the slots of the batch rows it flags as CUDA-graph padding as they are)
        self.prev_accepted = (
            mgr.mamba_layer_cache(kda_layers[0]).prev_num_accepted_tokens if spec else None
        )
        assert self.prev_accepted is None or self.prev_accepted.dtype == torch.int32
        self.prev_stride = self.prev_accepted.stride(0) if spec else 0
        self.dummy = mgr._dummy_request_mask if spec else None
        assert not spec or (mgr.use_kda_replay_update and self.dummy.dtype == torch.bool)
        # fresh pools may hold garbage, which warmup steps (forced has_initial_states, dummy
        # requests' never-written rows) would otherwise read
        with torch.no_grad():
            for t in self.zero:
                t.zero_()


def _commit_pending_drafts(
    ssm_states,
    conv_states,
    conv_q,
    conv_k,
    conv_v,
    k_cache,
    g_cache,
    v_cache,
    beta_cache,
    slots,
    num_accepted,
    state_tok=None,
) -> None:
    """The runtime's commit of a suspended request's pending drafts (commit_kda_pending_drafts'
    signature) for this target's KDA verify: the accepted drafts' records (each head's k | u |
    decay in the slot's state_tok region) re-applied to the slot's [head][k][v] state as the
    decode kernel replays them, S = decay * S + k u^T, and the conv window advanced over their raw
    conv inputs (columns [3, 3 + accepted) of the replay conv caches)."""
    assert state_tok is not None, "this target's drafts' records live in kda_state_tok"
    for slot, n in zip(slots.tolist(), num_accepted.tolist()):  # (a suspend: host values)
        if n <= 0:
            continue
        n = min(n, MTP_DRAFTS)
        rec = state_tok[slot].reshape(-1)[: MTP_DRAFTS * KDA_HL * 3 * KDA_D]
        rec = rec.view(MTP_DRAFTS, KDA_HL, 3, KDA_D)
        state = ssm_states[slot]
        for i in range(n):
            k, u, decay = rec[i, :, 0], rec[i, :, 1], rec[i, :, 2]
            state.mul_(decay.unsqueeze(-1)).add_(k.unsqueeze(-1) * u.unsqueeze(-2))
        window = conv_states[slot]
        drafts = torch.cat([c[slot, :, 3 : 3 + n] for c in (conv_q, conv_k, conv_v)])
        window.copy_(torch.cat([window, drafts], dim=1)[:, -window.shape[1] :])


class _StepLists:
    """Device copies of an eager step's host-built work lists. The core keeps them until the
    next step's: the step program holds only their addresses, and a later allocation of their
    memory is ordered after the step on the stream (its side streams join it before it ends)."""


class Glm5NextTp4ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = unwrap_glm5_next_text_config(model_config.pretrained_config)
        self.cfg = cfg
        mapping = model_config.mapping
        assert mapping.world_size == TP and mapping.tp_size == TP and mapping.pp_size == 1
        assert not mapping.enable_attention_dp, "tp4: tensor-parallel attention"
        assert mapping.moe_tp_size * mapping.moe_ep_size == TP and mapping.moe_ep_size in (1, TP), (
            "tp4: routed experts expert parallel (moe_ep 4) or tensor parallel (moe_tp 4)"
        )
        self.mapping = mapping
        assert model_config.torch_dtype == torch.bfloat16
        # identity of the checkpoint: assert, never adapt
        assert getattr(cfg, "model_type", None) == "glm5_next_text"
        assert cfg.hidden_size == H and cfg.vocab_size == V and not cfg.tie_word_embeddings
        assert cfg.num_hidden_layers == _weights.L and cfg.hc_mult == 4
        assert cfg.hc_sinkhorn_iters == HC_ITERS and cfg.hc_eps == HC_EPS
        assert cfg.linear_num_heads == 64 and cfg.linear_head_dim == KDA_D
        assert cfg.linear_conv_kernel_dim == 4 and cfg.linear_lower_bound == KDA_LOWER_BOUND
        assert cfg.num_attention_heads == MLA_HEADS and cfg.q_lora_rank == QLORA
        assert cfg.kv_lora_rank == LORA and cfg.qk_rope_head_dim == 0
        assert cfg.qk_nope_head_dim == NOPE and cfg.v_head_dim == VH
        assert cfg.index_n_heads == IDX_HEADS and cfg.index_head_dim == IDX_D
        assert (
            cfg.index_topk == SEL_POOLS * KPOOL
            and cfg.index_kpool == KPOOL
            and cfg.index_kpool_always_select_tail
        )
        assert all(t == "full" for t in cfg.indexer_types), "every DSA layer runs its own indexer"
        assert cfg.n_routed_experts == E and cfg.num_experts_per_tok == 8
        assert cfg.n_shared_experts == 1 and cfg.moe_intermediate_size == _weights.MOE_I
        assert cfg.n_group == 1 and cfg.topk_group == 1 and cfg.scoring_func == "sigmoid"
        assert cfg.norm_topk_prob and cfg.routed_scaling_factor == ROUTED_SCALE
        assert cfg.intermediate_size == _weights.INTER and cfg.swiglu_limit == SWIGLU_LIMIT
        assert cfg.hidden_act == "silu" and not cfg.attention_bias
        assert list(cfg.mlp_layer_types[:3]) == ["dense"] * 3
        assert all(t == "sparse" for t in cfg.mlp_layer_types[3:])
        self.kda_layers = _weights.kda_layers(cfg)
        self.mla_layers = _weights.mla_layers(cfg)
        assert len(self.kda_layers) == 34 and len(self.mla_layers) == 11
        self.spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = self.spec_config is not None
        if self.mtp_enabled:
            assert cfg.num_nextn_predict_layers == 1
            assert not self.spec_config._allow_separate_draft_kv_cache, (
                "the MTP layer must live in the target's cache manager"
            )
            assert self.spec_config.max_draft_len == MTP_DRAFTS, (
                f"the MTP draft chain takes {MTP_DRAFTS} drafts"
            )
        self.layout = _weights.Layout.of(
            mapping.tp_rank, mapping.moe_tp_size, mapping.moe_ep_size, self.mtp_enabled
        )
        self.mtp_layer = _weights.L  # the MTP layer's pool index (after the 45)
        self.max_num_tokens = int(model_config.max_num_tokens)
        assert self.max_num_tokens % TP == 0, "max_num_tokens must be a multiple of 4"
        # max_seq_len is resolved after the model is built (a speculative config also raises
        # it): the step program's pools per sequence are set from the metadata at the first
        # forward
        self.max_seq_len = int(model_config.max_seq_len or 32768)
        self._max_seq_len_set = False
        self.scalars: dict[str, float] = {}
        specs = _weights.param_specs(cfg, self.layout)
        self.w = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.empty(shape, dtype=dt), requires_grad=False)
                for k, (shape, dt) in specs.items()
            }
        )
        self.step = None
        self._lists: _StepLists | None = None
        self._pools: _Pools | None = None
        self._pools_mgr = None
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    # the runtime's KDA replay hooks (read off the built model by the KV cache creator)
    @property
    def kda_draft_records(self) -> bool:
        return self.mtp_enabled

    kda_commit_pending_drafts = staticmethod(_commit_pending_drafts)

    # ------------------------------------------------------------------ post-load
    def derive_after_load(self, lm_head: torch.Tensor) -> None:
        """The step program over the loaded weights, its buffers and its collectives' peer
        memory (all ranks together: IPC handles and the NCCL id are exchanged)."""
        from tensorrt_llm._torch.distributed.communicator import Distributed

        assert ops.mla_block == MLA_BLOCK and ops.kpool == KPOOL and ops.sel_pools == SEL_POOLS
        assert ops.idx_kg_row0 == MLA_BLOCK // KPOOL
        assert ops.sample_params_bytes == 40
        dev = torch.device("cuda", torch.cuda.current_device())
        cfg, lay = self.cfg, self.layout
        mc = ops.ModelConfig()
        mc.set_layers(
            [t == "linear_attention" for t in cfg.layer_types],
            [t == "sparse" for t in cfg.mlp_layer_types],
        )
        mc.rms_eps = cfg.rms_norm_eps
        mc.hc_eps = HC_EPS
        mc.hc_iters = HC_ITERS
        mc.kda_lower_bound = KDA_LOWER_BOUND
        mc.routed_scale = ROUTED_SCALE
        mc.swiglu_limit = SWIGLU_LIMIT
        pc = ops.ParallelConfig()
        pc.tp, pc.rank, pc.moe_tp, pc.moe_ep = TP, lay.rank, lay.moe_tp, lay.moe_ep
        rc = ops.RunnerConfig()
        rc.max_num_tokens = self.max_num_tokens
        rc.max_batch = MAX_BATCH
        rc.max_seq_len = self.max_seq_len
        rc.mtp_drafts = MTP_DRAFTS if self.mtp_enabled else 0
        step = ops.GlmStep(mc, pc, rc, dev.index)
        self._bind_weights(step.weights(), lm_head)
        step.allocate()
        # the collectives: Lamport all-reduce, sequence-parallel boundaries and token-split
        # attention over peer memory (x / out: the step program's layer input and sublayer
        # output rows), NCCL for the large ones
        self.comm = Collectives(self.mapping, self.max_num_tokens, H, AR_MAX_TOKENS)
        c = self.comm
        step.set_io(_ptr(c.x), _ptr(c.out))
        step.set_allreduce(c.ar_peers, c.ar_max_elems, AR_MAX_BLOCKS, _ptr(c.ar_epochs))
        step.set_sp(c.sp_peers, _ptr(c.sp_state))
        step.set_ts(c.ts_peers, _ptr(c.ts_state))
        dist = Distributed.get(self.mapping)
        uid = ops.GlmStep.nccl_unique_id() if lay.rank == 0 else None
        step.init_nccl(dist.tp_broadcast(uid, root=0))
        self.step = step
        self.sms = step.sm_count()
        # per-step metadata (step_meta), the sampled rows' tables and the head's parameters
        i32 = torch.int32
        T = self.max_num_tokens
        smax = (MAX_BATCH + 1) * (MTP_DRAFTS + 1)
        self.cu = torch.zeros(MAX_BATCH + 2, dtype=i32, device=dev)
        self.tok_seq = torch.zeros(T, dtype=i32, device=dev)
        self.has_st = torch.zeros(MAX_BATCH + 1, dtype=i32, device=dev)
        self.replay = torch.zeros(MAX_BATCH + 1, dtype=i32, device=dev)
        self.seq_tiles = torch.zeros(MAX_BATCH + 1, 2, dtype=i32, device=dev)
        self.iota = torch.arange(max(T, smax), dtype=i32, device=dev)
        self.mtp_ids = torch.zeros(T, dtype=i32, device=dev)
        self.mtp_src = torch.zeros(T, dtype=i32, device=dev)
        self.last_rows = torch.zeros(MAX_BATCH + 1, dtype=i32, device=dev)
        # a decode step's sequences: rows [4 b, 4 b + 4)
        self.dec_row0 = self.iota[: MAX_BATCH + 1] * (MTP_DRAFTS + 1)
        self.dec_nrows = torch.full_like(self.dec_row0, MTP_DRAFTS + 1)
        # k::SampleParams rows: temperature, top_p (1: off), top_k (0: off), min_p (0: off),
        # seed, step, num_logprobs (-1: none)
        self.sample_params = torch.zeros(smax, 40, dtype=torch.uint8, device=dev)
        f32v, i32v = self.sample_params.view(torch.float32), self.sample_params.view(i32)
        f32v[:, 1] = 1.0
        i32v[:, 8] = -1
        hfin, tokens, drafts, accepted = step.outputs()
        self.out_tokens = _view(tokens, i32, [smax])
        if self.mtp_enabled:
            self.out_drafts = _view(drafts, i32, [MAX_BATCH + 1, MTP_DRAFTS])
            self.out_accepted = _view(accepted, i32, [MAX_BATCH + 1])
            # what the worker hands the runtime (spec_accept): each sequence's head tokens, their
            # count to keep (accepted + 1) and the next step's input tokens
            self.spec_new = torch.zeros(MAX_BATCH + 1, MTP_DRAFTS + 1, dtype=i32, device=dev)
            self.spec_lens = torch.zeros(MAX_BATCH + 1, dtype=i32, device=dev)
            self.spec_next = torch.zeros(MAX_BATCH + 1, MTP_DRAFTS + 1, dtype=i32, device=dev)
        self.head_rows = _view(step.head_buf(), torch.bfloat16, [smax, H])
        self.attn_tile_tokens = ops.attn_tile_tokens(MLA_HL)
        self._decode_tiles: dict[tuple[int, int], tuple[torch.Tensor, int]] = {}
        torch.cuda.synchronize()

    def _bind_weights(self, mw, lm_head: torch.Tensor) -> None:
        """The step program's weight structs over this core's parameters (and the LM head's
        vocabulary shard)."""
        w, s, lay = self.w, self.scalars, self.layout
        r = lay.rank

        def p(name: str) -> int:
            return w[name].data_ptr()

        def bf(name: str, n: int, k: int, ptr: int | None = None) -> "ops.Bf16W":
            x = ops.Bf16W()
            x.w, x.n, x.k = p(name) if ptr is None else ptr, n, k
            return x

        def hc(n: str, which: str) -> "ops.HcW":
            x = ops.HcW()
            x.fn, x.base, x.scale = (
                p(f"{n}_hc_{which}_fn"),
                p(f"{n}_hc_{which}_base"),
                p(f"{n}_hc_{which}_scale"),
            )
            return x

        def mla(n: str) -> "ops.MlaW":
            x = ops.MlaW()
            x.qkv_a16 = bf(f"{n}_qkv_a", QLORA + LORA, H)
            # (token-split: every head's q_b / absorption weights, this rank's heads a slice)
            h0 = r * MLA_HL if lay.ts else 0
            qb = w[f"{n}_q_b"]
            x.q_b16 = bf("", MLA_HL * NOPE, QLORA, qb[h0 * NOPE :].data_ptr())
            x.o_proj16 = bf(f"{n}_o_proj", H, MLA_HL * VH)
            x.q_a_norm, x.kv_a_norm = p(f"{n}_q_a_norm"), p(f"{n}_kv_a_norm")
            x.w_kc = w[f"{n}_w_kc"][h0:].data_ptr()
            x.w_vc = w[f"{n}_w_vc"][h0:].data_ptr()
            if lay.ts:
                x.w_kc_all, x.w_vc_all, x.q_b16_all = p(f"{n}_w_kc"), p(f"{n}_w_vc"), p(f"{n}_q_b")
                if f"{n}_o_proj_all" in w:
                    x.o_proj16_all = p(f"{n}_o_proj_all")
            x.w_kcT = p(f"{n}_w_kcT")
            x.idx_wq = bf(f"{n}_idx_wq", IDX_HEADS * IDX_D, QLORA)
            x.idx_kgw = bf(f"{n}_idx_kgw", 2 * IDX_D + IDX_HEADS, H)
            x.idx_kn_w, x.idx_kn_b = p(f"{n}_idx_kn_w"), p(f"{n}_idx_kn_b")
            x.idx_ape = p(f"{n}_idx_ape")
            return x

        def group(mtp: bool) -> "ops.MoeGroupMap":
            e0, ne, _, _ = lay.experts(mtp)
            g = ops.MoeGroupMap()
            g.E, g.lo, g.n = E, e0, ne
            g.ep = TP if ne != E else 1
            g.ep_rank = r if ne != E else 0
            return g

        def shared(x: "ops.MoeWeights", n: str) -> None:
            x.router, x.bias = p(f"{n}_router"), p(f"{n}_bias")
            x.sh_gate_up = bf(f"{n}_sh_gu", 2 * SH_IL, H)
            x.sh_down = bf(f"{n}_sh_dn", H, SH_IL)
            x.sh_inter = SH_IL
            x.num_experts = E

        mw.embed = p("embed")
        mw.final_norm = p("final_norm")
        assert tuple(lm_head.shape) == (V // TP, H) and lm_head.dtype == torch.bfloat16
        mw.lm_head = bf("", V // TP, H, lm_head.data_ptr())
        _, _, il, _ = lay.experts()
        for i in range(_weights.L):
            n = f"l{i}"
            L = mw.layer(i)
            L.kda_layer = self.cfg.layer_types[i] == "linear_attention"
            L.moe = self.cfg.mlp_layer_types[i] == "sparse"
            L.hc_attn, L.hc_ffn = hc(n, "attn"), hc(n, "ffn")
            L.in_norm, L.post_norm = p(f"{n}_in_norm"), p(f"{n}_post_norm")
            if L.kda_layer:
                k = ops.KdaW()
                k.in_proj = bf(f"{n}_in_proj", KDA_NIN, H)
                k.f_b = bf(f"{n}_f_b", KDA_HD, KDA_D)
                k.g_b = bf(f"{n}_g_b", KDA_HD, KDA_D)
                k.conv, k.a_log, k.dt_bias = p(f"{n}_conv"), p(f"{n}_a_log"), p(f"{n}_dt_bias")
                k.o_norm = p(f"{n}_o_norm")
                k.o_proj = bf(f"{n}_kda_o_proj", H, KDA_HD)
                L.kda = k
            else:
                L.mla = mla(n)
            if L.moe:
                m = ops.MoeWeights()
                shared(m, n)
                m.fmt = ops.WFmt.NVFP4
                m.w13_4, m.sf13, m.alpha13 = p(f"{n}_w13"), p(f"{n}_sf13"), p(f"{n}_alpha13")
                m.w2_4, m.sf2, m.alpha2 = p(f"{n}_w2"), p(f"{n}_sf2"), p(f"{n}_alpha2")
                m.gs13, m.gs2 = s[f"{n}_gs13"], s[f"{n}_gs2"]
                m.inter_local = il
                m.local = group(False)
                L.moe_w = m
            else:
                d = ops.MlpW()
                d.fmt = ops.WFmt.NVFP4
                gu, dn = ops.Fp4W(), ops.Fp4W()
                gu.w, gu.sf, gu.n, gu.k = p(f"{n}_gu_w"), p(f"{n}_gu_sf"), 2 * INTER_L, H
                gu.alpha, gu.gs = s[f"{n}_gu_alpha"], s[f"{n}_gu_gs"]
                dn.w, dn.sf, dn.n, dn.k = p(f"{n}_dn_w"), p(f"{n}_dn_sf"), H, INTER_L
                dn.alpha, dn.gs = s[f"{n}_dn_alpha"], s[f"{n}_dn_gs"]
                d.gate_up4, d.down4 = gu, dn
                d.alpha4 = p(f"{n}_alpha4")
                d.inter_local = INTER_L
                L.mlp = d
        if lay.mtp:
            n = "mtp"
            t = ops.MtpW()
            t.present = True
            t.enorm, t.hnorm = p(f"{n}_enorm"), p(f"{n}_hnorm")
            t.eh_proj = bf(f"{n}_eh_proj", H, 2 * H)
            t.in_norm, t.post_norm = p(f"{n}_in_norm"), p(f"{n}_post_norm")
            t.head_norm = p(f"{n}_head_norm")
            t.mla = mla(n)
            m = ops.MoeWeights()
            shared(m, n)
            m.fmt = ops.WFmt.BF16
            m.w13_16, m.w2_16 = p(f"{n}_w13_16"), p(f"{n}_w2_16")
            m.inter_local = lay.experts(mtp=True)[2]
            m.local = group(True)
            t.moe = m
            mw.mtp = t

    # ------------------------------------------------------------------ step metadata
    def _pools_of(self, mgr) -> _Pools:
        if self._pools_mgr is not mgr:
            mla_layers = self.mla_layers + ([self.mtp_layer] if self.mtp_enabled else [])
            self._pools = _Pools(mgr, mla_layers, self.kda_layers, self.mtp_enabled)
            self._pools_mgr = mgr
            self.step.set_pools(self._pools.pools)
        return cast(_Pools, self._pools)

    def _select_target(self, md) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step: one host-side integer read. A mixed batch (context and
        generation rows in one step) goes to prefill, so prefill is the general case and decode
        the specialization that may assume it has no context rows."""
        return self._decode if md.num_contexts == 0 else self._prefill

    def _step_in(self, md, input_ids, position_ids, nctx: int, Tc: int) -> "ops.StepIn":
        """The step program's input for a forward of nctx context rows on the first Tc tokens
        (stated by the target): the per-forward metadata (step_meta, on the device, so a
        captured decode step rebuilds it on replay) and the pools; the target adds its work
        lists."""
        assert self.step is not None, "post_load_weights must run before the first forward"
        if not self._max_seq_len_set:
            self.max_seq_len = int(md.max_seq_len)
            self.step.set_max_seq_len(self.max_seq_len)
            self._max_seq_len_set = True
        assert int(md.max_seq_len) == self.max_seq_len, (
            "max_seq_len changed after the first forward"
        )
        assert getattr(md, "padded_num_tokens", None) is None, "token-padded steps"
        ids = input_ids.reshape(-1)
        pos = position_ids.reshape(-1)
        T = ids.shape[0]
        assert ids.dtype == torch.int32 and pos.dtype == torch.int32 and pos.numel() == T
        assert T <= self.max_num_tokens, (T, self.max_num_tokens)
        B = int(md.num_seqs)
        assert 1 <= B <= MAX_BATCH, (
            f"{B} sequences in a step: this target takes up to {MAX_BATCH} (max_batch_size)"
        )
        ngen, Tg = B - nctx, T - Tc
        dt = Tg // ngen if ngen else 1
        assert ngen == 0 or Tg == ngen * dt, "generation rows carry equal token counts"
        mm = md.mamba_metadata
        assert isinstance(mm, Glm5NextMamba2Metadata), type(mm)
        pools = self._pools_of(md.kv_cache_manager)
        slots = mm.state_indices
        assert slots.dtype == torch.int32 and slots.is_contiguous()
        assert md.seq_lens_cuda.dtype == torch.int32 and ids.is_contiguous() and pos.is_contiguous()
        st = stream()
        ops.step_meta(
            _ptr(self.cu),
            _ptr(self.tok_seq),
            _ptr(self.has_st),
            _ptr(self.replay),
            _ptr(self.seq_tiles),
            _ptr(md.seq_lens_cuda),
            _ptr(mm.has_initial_states) if nctx else 0,
            _ptr(slots),
            _ptr(pools.prev_accepted),
            pools.prev_stride,
            B,
            nctx,
            T,
            st,
        )
        bt = mm.glm_block_tables
        assert bt.dtype == torch.int64 and bt.stride(1) == 1
        s = ops.StepIn()
        s.T, s.B, s.nd, s.dt = T, B, ngen, dt
        s.graph = nctx == 0
        s.ids, s.positions, s.token_seq = _ptr(ids), _ptr(pos), _ptr(self.tok_seq)
        s.cu_seqlens, s.kda_slots, s.has_state = _ptr(self.cu), _ptr(slots), _ptr(self.has_st)
        s.kda_replay = _ptr(self.replay)
        s.block_tables, s.bt_stride = _ptr(bt), bt.stride(0)
        s.seq_tiles = _ptr(self.seq_tiles)
        self._lists = _StepLists()
        return s

    def _decode_lists(self, s) -> None:
        """A decode step's lists depend on its batch and tokens per sequence only (a captured
        step's are constants): every sequence one indexer tile (step_meta's seq_tiles), score
        rows of every position the graph may see, the selection always (some sequence may see
        more than 512 complete pools), and with enough query tiles the tiled attention."""
        B, dt = s.B, s.dt
        assert dt <= 8
        s.need_sel = True
        s.idx_tiles = s.seq_tiles
        s.set_h_idx_tiles([(b * dt, dt) for b in range(B)])
        s.idx_split = 0
        s.score_stride = 0
        s.num_chunks = 0
        qt = self.attn_tile_tokens
        if dt > 1 and B * -(-dt // qt) >= ATTN_TILED_MIN:
            key = (B, dt)
            if key not in self._decode_tiles:
                assert not torch.cuda.is_current_stream_capturing(), (
                    "decode attention tiles are built by the eager step before a capture"
                )
                at = [(b * dt + t, min(qt, dt - t)) for b in range(B) for t in range(0, dt, qt)]
                tiles = torch.tensor(at, dtype=torch.int32, device=self.cu.device)
                self._decode_tiles[key] = (tiles, len(at))
            tiles, n = self._decode_tiles[key]
            s.attn_tiles, s.num_attn_tiles = _ptr(tiles), n
        else:
            s.num_attn_tiles = 0

    def _host_lists(self, s, md, nctx: int, spec: bool) -> None:
        """Work lists of an eager step with context rows: KDA 64-token chunks of
        the contexts, indexer tiles (<= 8 consecutive tokens of one sequence), query tiles of
        the tiled sparse attention, and the score-row length. From IDX_SPLIT_MIN_T tokens the
        selection is split over the TP ranks: tiles also end at multiples of C = ceil(T / 4),
        rank r selects the rows [r C, (r + 1) C) and the layer all-gathers them."""
        B, T = s.B, s.T
        lens = md.seq_lens[:B].tolist()
        cached = md.kv_cache_params.num_cached_tokens_per_seq
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        assert cu[-1] == T, (cu[-1], T)
        chunks, seq_chunks = [], [0]
        for b in range(nctx):
            for t in range(cu[b], cu[b + 1], 64):
                chunks.append((t, min(64, cu[b + 1] - t)))
            seq_chunks.append(len(chunks))
        split = -(-T // TP) if T >= IDX_SPLIT_MIN_T else 0
        tiles = []
        for b in range(B):
            t = cu[b]
            while t < cu[b + 1]:
                n = min(8, cu[b + 1] - t)
                if split:
                    n = min(n, (t // split + 1) * split - t)
                tiles.append((t, n))
                t += n
        qt = self.attn_tile_tokens
        at = []
        if T > B:  # (some sequence has several tokens)
            at = [
                (t, min(qt, cu[b + 1] - t)) for b in range(B) for t in range(cu[b], cu[b + 1], qt)
            ]
            if len(at) < ATTN_TILED_MIN:
                at = []
        # generation rows' host lengths are provisional under the overlap scheduler with MTP:
        # bounded by every draft accepted
        margin = 2 * (MTP_DRAFTS + 1) if spec else 0
        kv_max = max(int(cached[b]) + lens[b] for b in range(B)) + margin
        s.need_sel = kv_max // KPOOL > SEL_POOLS
        max_pools = self.max_seq_len // KPOOL + 1
        s.score_stride = min(max_pools, -(-kv_max // KPOOL) + 8)
        dev = self.cu.device

        def up(x):
            h = torch.tensor(x, dtype=torch.int32, pin_memory=prefer_pinned())
            return h.to(dev, non_blocking=True)

        lists = self._lists
        lists.chunks, lists.seq_chunks, lists.tiles = up(chunks), up(seq_chunks), up(tiles)
        s.kda_chunks, s.kda_seq_chunks, s.num_chunks = (
            _ptr(lists.chunks),
            _ptr(lists.seq_chunks),
            len(chunks),
        )
        s.idx_tiles = _ptr(lists.tiles)
        s.set_h_idx_tiles(tiles)
        s.idx_split = split
        if at:
            lists.attn = up(at)
            s.attn_tiles, s.num_attn_tiles = _ptr(lists.attn), len(at)
        else:
            s.num_attn_tiles = 0

    # ------------------------------------------------------------------ forwards
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """Without MTP: the decoder layers and the final norm of each sequence's last token
        ([sequences, hidden], the LM head's rows)."""
        assert inputs_embeds is None, "token ids only (text)"
        assert lora_params is None, "LoRA is not implemented"
        assert spec_metadata is None and not self.mtp_enabled
        md = attn_metadata
        s = self._select_target(md).step_in(md, input_ids, position_ids, False)
        B = s.B
        torch.sub(self.cu[1 : B + 1], 1, out=self.last_rows[:B])
        s.S, s.Bs, s.head = B, 0, -1
        s.sample_idx = _ptr(self.last_rows)
        self.step.set_tune(s.graph and not torch.cuda.is_current_stream_capturing())
        self.step.forward(s, stream())
        return self.head_rows[:B]

    def spec_step(
        self, md, input_ids, position_ids, spec_metadata, head: int, force: float, sample
    ):
        """With MTP: one step of verification and drafting. head 0 / 1: the step program's
        greedy / temperature head; -1: `sample(head rows) -> tokens` (the runtime's sampler over
        gathered logits) between the decoder layers and the draft passes. Returns (the head's
        tokens [rows], the drafts [sequences, 3], the accepted drafts [sequences], the first
        sampled row of each sequence [sequences] or None when it is 4 b)."""
        s, S, row0 = self._select_target(md).spec_in(md, input_ids, position_ids, spec_metadata)
        B = s.B
        s.S, s.Bs, s.head, s.force_accept = S, B, head, force
        st = stream()
        if head == 1:
            # k::SampleParams of the sampled rows: the runtime's temperatures (greedy rows 0)
            # and Philox (seed, offset), one stream per row and step
            temps, seeds, offsets = (
                spec_metadata.temperatures,
                spec_metadata.seeds,
                spec_metadata.offsets,
            )
            assert temps.dtype == torch.float32 and seeds.dtype == offsets.dtype == torch.int64
            assert temps.is_contiguous() and seeds.is_contiguous() and offsets.is_contiguous()
            ops.spec_sample_params(
                _ptr(self.sample_params), _ptr(temps), _ptr(seeds), _ptr(offsets), S,
                GREEDY_TEMPERATURE, st,
            )  # fmt: skip
            s.sample_params = _ptr(self.sample_params)
        self.step.set_tune(s.graph and not torch.cuda.is_current_stream_capturing())
        self.step.forward(s, st)
        if head < 0:
            self.out_tokens[:S].copy_(sample(self.head_rows[:S]))
            self.step.mtp(s, st)
        # the worker's outputs, and the generation rows' KDA replay record (update_mamba_states)
        pools = self._pools_of(md.kv_cache_manager)
        bi = spec_metadata.batch_indices_cuda
        assert bi.dtype == torch.int32 and pools.dummy is not None
        ops.spec_accept(
            _ptr(self.spec_new), _ptr(self.spec_lens), _ptr(self.spec_next), _ptr(self.out_tokens),
            _ptr(self.out_drafts), _ptr(self.out_accepted), _ptr(row0), _ptr(bi), S, s.kda_slots,
            _ptr(pools.dummy), _ptr(pools.prev_accepted), pools.prev_stride, B, B - s.nd,
            MTP_DRAFTS, st,
        )  # fmt: skip
        return self.spec_new[:B], self.spec_lens[:B], self.spec_next[:B], self.out_drafts[:B]


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them. Eager: the
    step program's work lists (KDA chunks, indexer and attention tiles, the score-row length)
    are built on the host per step."""

    def __init__(self, core: Glm5NextTp4ModelingV2) -> None:
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: the metadata fields this target consumes must exist
        (they are private trtllm surface). Calling `step_args` is the check."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md) -> dict:
        mm = md.mamba_metadata
        return dict(
            num_seqs=md.num_seqs,
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            seq_lens=md.seq_lens,
            seq_lens_cuda=md.seq_lens_cuda,
            cached_tokens=md.kv_cache_params.num_cached_tokens_per_seq,
            max_seq_len=md.max_seq_len,
            kv_cache_manager=md.kv_cache_manager,
            state_indices=mm.state_indices,
            has_initial_states=mm.has_initial_states,
            block_tables=mm.glm_block_tables,
        )

    def step_in(self, md, input_ids, position_ids, spec: bool) -> "ops.StepIn":
        self._check_step_contract(md)
        core = self.core
        nctx = int(md.num_contexts)
        s = core._step_in(md, input_ids, position_ids, nctx, int(md.num_ctx_tokens))
        core._host_lists(s, md, nctx, spec)
        return s

    def spec_in(self, md, input_ids, position_ids, spec_metadata):
        """The step input with MTP, the sampled rows and each sequence's first sampled row: a
        context's last row, a generation row's 4 rows (the runtime's gather_ids)."""
        core = self.core
        s = self.step_in(md, input_ids, position_ids, True)
        nctx, Tc = int(md.num_contexts), int(md.num_ctx_tokens)
        T, ngen, dt = s.T, s.nd, s.dt
        assert ngen == 0 or dt == MTP_DRAFTS + 1, f"generation rows of {dt} tokens"
        S = nctx + ngen * dt
        gather = spec_metadata.gather_ids[:S]
        assert gather.dtype == torch.int32
        s.sample_idx = _ptr(gather)
        # MTP pass 1 pairs the hidden state at each row with the next token: the prompt's next
        # token, at each context's last row its lookahead (a chunk of a longer prompt) or the
        # token sampled there, at generation rows the token sampled there
        ids = input_ids.reshape(-1)
        core.mtp_ids[: T - 1].copy_(ids[1:T])
        core.mtp_src[:Tc].fill_(-1)
        last = gather[:nctx].long()
        rows = core.iota[:nctx]
        lookahead = spec_metadata.context_prompt_lookahead_tokens
        if lookahead is not None:
            la = lookahead[:nctx].to(torch.int32)
            valid = la != INVALID_PROMPT_LOOKAHEAD_TOKEN
            core.mtp_ids[last] = torch.where(valid, la, 0)
            core.mtp_src[last] = torch.where(valid, -1, rows)
        else:
            core.mtp_src[last] = rows
        if ngen:
            core.mtp_src[Tc:T].copy_(core.iota[nctx : nctx + T - Tc])
        s.mtp_ids, s.mtp_src = _ptr(core.mtp_ids), _ptr(core.mtp_src)
        r0 = list(range(nctx)) + [nctx + dt * g for g in range(ngen)]
        nr = [1] * nctx + [dt] * ngen
        h = torch.tensor([r0, nr], dtype=torch.int32, pin_memory=prefer_pinned())
        spec_rows = h.to(core.cu.device, non_blocking=True)
        core._lists.spec_rows = spec_rows
        s.spec_row0, s.spec_nrows = _ptr(spec_rows[0]), _ptr(spec_rows[1])
        return s, S, spec_rows[0]


class DecodeTarget:
    """The specialization: routed to only when there are no context rows. Captured in a CUDA
    graph per batch size; every per-step value it reads is a device buffer the runtime refreshes
    before a replay, and every host value is a per-capture constant."""

    def __init__(self, core: Glm5NextTp4ModelingV2) -> None:
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: the metadata fields this target consumes must exist
        (they are private trtllm surface). Calling `step_args` is the check."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md) -> dict:
        """States the two context counts instead of reading them back: this target runs only
        when there are no context rows."""
        mm = md.mamba_metadata
        return dict(
            num_seqs=md.num_seqs,
            num_contexts=0,
            num_ctx_tokens=0,
            seq_lens_cuda=md.seq_lens_cuda,
            max_seq_len=md.max_seq_len,
            kv_cache_manager=md.kv_cache_manager,
            state_indices=mm.state_indices,
            block_tables=mm.glm_block_tables,
        )

    def step_in(self, md, input_ids, position_ids, spec: bool) -> "ops.StepIn":
        self._check_step_contract(md)
        core = self.core
        s = core._step_in(md, input_ids, position_ids, 0, 0)
        core._decode_lists(s)
        return s

    def spec_in(self, md, input_ids, position_ids, spec_metadata):
        """The step input with MTP: every row sampled (4 per sequence), each paired with the
        token sampled at it by MTP pass 1."""
        core = self.core
        s = self.step_in(md, input_ids, position_ids, True)
        assert s.dt == MTP_DRAFTS + 1, f"generation rows of {s.dt} tokens"
        s.sample_idx = s.mtp_src = _ptr(core.iota)
        s.mtp_ids = _ptr(core.mtp_ids)
        s.spec_row0, s.spec_nrows = _ptr(core.dec_row0), _ptr(core.dec_nrows)
        return s, s.T, None


class MTPStepWorker(MTPEagleWorker):
    """The one-model MTP-Eagle worker of this target: verification, acceptance and the draft
    chain are one call of the step program (`Glm5NextTp4ModelingV2.spec_step`), whose buffers
    give the four tensors the sampler reads. The forced acceptance follows the built-in worker's
    (`_apply_force_accepted_tokens`): none during eager CUDA-graph warm-up."""

    def __init__(self, spec_config, model_config: ModelConfig, mapping, *, step, lm_head):
        super().__init__(spec_config, model_config, False, mapping=mapping)
        # The core's step and the LM head with its logits processor (bound methods / a tuple
        # rather than the modules, which belong to the shell and are no submodules of this
        # worker).
        self._step = step
        self._lm_head = lm_head
        mode = getattr(spec_config, "advanced_sampling_mode", None)
        self._temperature_only = mode == AdvancedSamplingMode.NO_TOPK_NO_TOPP

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        assert spec_metadata.runtime_draft_len == MTP_DRAFTS, (
            f"the MTP draft chain drafts {MTP_DRAFTS} tokens every step"
        )
        assert self.guided_decoder is None and self.sa_enhancer is None
        assert not spec_metadata.is_ragged_verify
        B = attn_metadata.num_seqs
        nctx = attn_metadata.num_contexts
        warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        force = self.force_num_accepted_tokens
        force = force if force != 0.0 and not warmup else -1.0
        if spec_metadata.is_all_greedy_sample:
            head = 0
        elif self._temperature_only:
            head = 1
        else:
            head = -1

        def sample(rows: torch.Tensor) -> torch.Tensor:
            lm_head, logits_processor = self._lm_head
            gathered = logits_processor.forward(rows, lm_head, attn_metadata, True)
            return self._sample_tokens_for_batch(gathered, spec_metadata, nctx, B).to(torch.int32)

        # (one step-program call and one kernel after it: the runtime's update_mamba_states (the
        # KDA replay record) and _prepare_next_new_tokens are part of spec_accept)
        new_tokens, new_tokens_lens, next_new_tokens, drafts = self._step(
            attn_metadata, input_ids, position_ids, spec_metadata, head, force, sample
        )
        attn_metadata.use_spec_decoding = True
        return {
            # (no logits: the head samples on the vocabulary shards; the runtime reads them only
            # for requests that return logits)
            "logits": torch.empty(max(int(input_ids.numel()), 1), 0, device=new_tokens.device),
            "new_tokens": new_tokens,
            "new_tokens_lens": new_tokens_lens,
            "next_draft_tokens": drafts,
            "next_new_tokens": next_new_tokens,
        }


class _SlotTablesMetadata(Glm5NextMamba2Metadata):
    """The glm5_next metadata with the sparse layers' slot tables built per sequence from the
    cache's int32 page-index buffer, one vectorized copy per row. The runtime's prepare() lists
    the same raw slot IDs one Python int per page (get_batch_slot_tables), host time on every
    step that grows with the sequences' lengths.
    Two pinned staging tables alternate, so a step never rewrites the source of the previous
    step's pending upload, and a staging row clears only the columns it held before. The device
    table holds what the runtime's prepare() writes, row for row; a step whose rows all hold what
    the device table holds (a decode step that takes no new page: most of them) uploads nothing."""

    def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) -> None:
        super().__init__(max_batch_size, chunk_size, max_num_tokens)
        self._stagings: list[torch.Tensor] = []
        self._held: list[list[int]] = []  # per staging table: the columns each row holds
        self._flip = 0
        self._slot_group: int | None = None
        # what the device table's rows hold: request id and page ids per row (None: unknown)
        self._dev_rows: list[tuple[int, np.ndarray] | None] = []

    def _slot_group_of(self, manager: Glm5NextCacheManager) -> int | None:
        """The V2 layer group of the sparse layers' latent / INDEX_KEY slots (None: this rank holds
        no sparse layer), as get_batch_slot_tables resolves it."""
        if self._slot_group is None:
            layers = [i for i in manager.sparse_layer_ids if i in manager.layer_offsets]
            if not layers:
                return None
            groups = {manager.layer_to_pool_mapping_dict[manager.layer_offsets[i]] for i in layers}
            assert len(groups) == 1, f"sparse layers span V2 layer groups {sorted(groups)}"
            self._slot_group = groups.pop()
        return self._slot_group

    def prepare(self, attn_metadata) -> None:
        manager = attn_metadata.kv_cache_manager
        assert manager is None or isinstance(manager, Glm5NextCacheManager), (
            "glm5_next metadata requires Glm5NextCacheManager"
        )
        KimiK3MambaMetadata.prepare(self, attn_metadata)
        kv_params = attn_metadata.kv_cache_params
        request_ids = attn_metadata.request_ids
        if (
            manager is None
            or kv_params is None
            or kv_params.num_cached_tokens_per_seq is None
            or request_ids is None
        ):
            return
        batch = attn_metadata.seq_lens.shape[0]
        num_contexts = int(attn_metadata.num_contexts)
        lens = attn_metadata.seq_lens[:batch].tolist()
        cached_src = kv_params.num_cached_tokens_per_seq
        if isinstance(cached_src, torch.Tensor):
            self.glm_cached_lens_host = cached_src[:batch].tolist()
        else:
            self.glm_cached_lens_host = [int(cached_src[i]) for i in range(batch)]
        cu = [0]
        for length in lens[:num_contexts]:
            cu.append(cu[-1] + length)
        self.glm_ctx_cu_seqlens = cu
        self._glm_ensure_tables(manager.max_blocks_per_seq)
        if not self._stagings:
            first = self._glm_block_tables_cpu
            assert first is not None
            self._stagings = [
                first,
                torch.zeros(first.shape, dtype=first.dtype, pin_memory=prefer_pinned()),
            ]
            self._held = [[0] * first.shape[0] for _ in self._stagings]
            self._dev_rows = [None] * first.shape[0]
        group = self._slot_group_of(manager)
        pages = []
        changed = False
        for row, rid in enumerate(list(request_ids)[:batch]):
            src = None
            if group is not None:
                cache = manager.kv_cache_map[rid]
                n = cache.num_blocks
                if n:
                    src = np.frombuffer(cache.get_base_page_indices(group), dtype=np.int32, count=n)
            dev = self._dev_rows[row]
            changed |= (
                dev is None
                or dev[0] != rid
                or len(dev[1]) != (0 if src is None else len(src))
                or (src is not None and not np.array_equal(dev[1], src))
            )
            pages.append((rid, src))
        if not changed:
            return
        staging, held = self._stagings[self._flip], self._held[self._flip]
        self._flip ^= 1
        rows = staging.numpy()
        for row, (rid, src) in enumerate(pages):
            n = 0 if src is None else len(src)
            if n:
                rows[row, :n] = src
            if held[row] > n:
                rows[row, n : held[row]] = 0
            held[row] = n
            # (a copy: the page-index buffer belongs to the cache and changes as it grows)
            self._dev_rows[row] = (rid, np.empty(0, np.int32) if src is None else src.copy())
        assert self.glm_block_tables is not None
        self.glm_block_tables[:batch].copy_(staging[:batch], non_blocking=True)


class ModelingV2Glm53FlashNvfp4Sm10xTp4(
    DecoderModelForCausalLM[Glm5NextTp4ModelingV2, PretrainedConfig]
):
    # the glm5_next cache manager's metadata (KDA slots, sparse-layer block tables), with the
    # slot tables built from the page-index buffers
    mamba_metadata_cls = _SlotTablesMetadata

    def __init__(self, model_config: ModelConfig):
        model_config = _runtime_model_config(model_config)
        cfg = unwrap_glm5_next_text_config(model_config.pretrained_config)
        super().__init__(
            Glm5NextTp4ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            mode = self.spec_config.spec_dec_mode
            assert mode.is_mtp_eagle_one_model(), f"one-model MTP-Eagle expected, got {mode!r}"
            self.spec_worker = MTPStepWorker(
                self.spec_config,
                model_config,
                model_config.mapping,
                step=self.model.spec_step,
                lm_head=(self.lm_head, self.logits_processor),
            )

    def validate_request(self, request) -> None:
        """Admission check, run by the executor on every new request: what the step cannot serve
        fails here as that request's own error. Raised inside the step instead, it would fail every
        active request.

        Refused:
        - with MTP, logits post-processors (`thinking_token_budget`, a request's `logits_processor`):
          the worker's head samples on the vocabulary shards and returns no logits for them to edit;
        - without MTP, context logits (`return_context_logits`, `prompt_logprobs`): the forward
          computes the logits of each sequence's last token only. With MTP the sampler refuses them.
        """
        if self.spec_worker is not None:
            if getattr(request, "py_logits_post_processors", None):
                raise ValueError(
                    "Logits post-processors (thinking_token_budget, logits_processor) are not "
                    "supported by the GLM-5.3-Flash ModelingV2 target with MTP: it samples without "
                    "materializing the logits. Drop them from the request, or serve without "
                    "speculative decoding."
                )
        elif getattr(request, "py_return_context_logits", False):
            raise ValueError(
                "return_context_logits / prompt_logprobs are not supported by the GLM-5.3-Flash "
                "ModelingV2 target: it computes the logits of each sequence's last token only. "
                "Drop them from the request."
            )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """One runtime step: the step program, then without MTP the LM head over each sequence's
        last token (the sampler's logits), with MTP the worker's outputs."""
        if self.spec_worker is None:
            assert spec_metadata is None and not return_context_logits
            rows = self.model(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                lora_params=lora_params,
            )
            return self.logits_processor.forward(rows, self.lm_head, attn_metadata, True)
        assert spec_metadata is not None and inputs_embeds is None and lora_params is None
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=None,
            logits=None,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=None,
            resource_manager=kwargs.get("resource_manager"),
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.load(self.model, weights)
        _weights.load_lm_head(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.derive_after_load(self.lm_head.weight)
