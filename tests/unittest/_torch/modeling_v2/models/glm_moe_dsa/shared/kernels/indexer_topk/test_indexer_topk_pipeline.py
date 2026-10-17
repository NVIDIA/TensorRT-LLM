# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""indexer_topk inside the real DSA decode pipeline.

Every object on the path to and from indexer_topk is the native one:

- ``DSACacheManagerV2`` holding three GLM-5.2-shaped MLA layers: layer 0 owns
  a full indexer, layer 1 is a shared-indexer layer, layer 2 is the MTP layer
  (full indexer). FP8 MLA cache, FP8 indexer K, 64-token pages and the served
  ``max_seq_len`` (2**20 + 14).
- ``MLA`` modules with the per-rank GLM-5.2 TP8 attention shapes (8 heads,
  q-nope 192, q-rope 64, v 256, kv-LoRA 512, q-LoRA 2048; indexer 32 x 128,
  top-k 2048), adopted as the target's ``GlmMLA`` (with its DSA backend and
  indexer), running the target's ``_forward_dsa_attn`` as a decode step:
  indexer K append, the indexer
  (native or indexer_topk), the backend's shared-layer copy, the local-to-global remap,
  RoPE/FP8-query/KV append and sparse MLA (native, or sparse_mla_decode/rope_kv_append when
  selected).
- ``DSAtrtllmAttentionMetadata`` with ``update_spec_dec_param(max_draft_len=5)``
  and ``prepare()`` per decode step, followed by the decode runner's
  ``_preprocess_inputs`` sequence: ``on_update_kv_lens()``, the overlap
  scheduler's accepted-token correction of ``kv_lens_cuda``, and
  ``on_update_kv_lens()`` again.
- The one-model MTP-Eagle draft loop of the target's worker,
  ``GlmMtpWorker._forward_linear_draft_loop``, itself (index share on, accepted-row
  stash, per-step metadata updates), with a stub draft model whose MTP layer runs the
  real layer-2 attention. With
  ``draft_loop`` selected the loop takes draft_loop's per-step metadata update.

The context is written by the native producers (indexer K quantize + scatter,
MLA RoPE-append of random latents). A decode sequence is four target verify
steps (next_n = 6) separated by three draft loops with accepted counts
``ACCEPTED``. Every arm replays the same sequence on the same caches; arms
differ only in whether indexer_topk runs. ``GLM_MOE_DSA_TEST_FAST_PATHS``
(``plan_testlib.FAST_PATHS_ENV``) is the base set of fast paths (for example
``sparse_mla_decode,rope_kv_append`` or ``draft_loop``), whose plan is installed before any layer
is built; the tests add ``indexer_topk`` on top of it for the indexer_topk arms.

``test_rank_agreement`` runs the same sequence on every rank of an MPI job
(8 ranks over 2 hosts) and compares digests across ranks.
``test_indexer_topk_region_timing`` (``GLM_MOE_DSA_INDEXER_TOPK_PIPELINE_TIMING=1``) times the attention region of a
full + shared layer pair in CUDA graphs with indexer_topk off and on.
"""

import hashlib
import json
import os
import socket
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

if not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("CUDA and the CUTLASS DSL are required", allow_module_level=True)
if get_sm_version() != 103:
    pytest.skip("indexer_topk is verified on GB300 (sm103) only", allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plan_testlib  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
    GlmMLA,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention_forward import (  # noqa: E501
    _forward_dsa_attn,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
    indexer_decode as indexer_topk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.spec_worker import (  # noqa: E501
    GlmMtpWorker,
)
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as indexer_mod
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# DSACacheManagerV2 owns background threads.
pytestmark = pytest.mark.threadleak(enabled=False)

LOCAL_HEADS = 8
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256
KV_LORA = 512
Q_LORA = 2048
HIDDEN = 6144
INDEX_HEADS = 32
INDEX_DIM = 128
TOP_K = 2048
DRAFT_LEN = 5
NEXT_N = DRAFT_LEN + 1
TOKENS_PER_BLOCK = 64
SCHEDULE_BLOCK_KV = indexer_mod._DG_SCHEDULE_BLOCK_KV
# GLM-5.2: max_position_embeddings 2**20, rope theta 8e6, no rope scaling.
MAX_POSITIONS = 1 << 20
ROPE_THETA = 8.0e6
# Served GEN: max_seq_len 2**20 plus the K = 5 one-model MTP slack.
SERVED_MAX_SEQ_LEN = (1 << 20) + 14
NUM_EXTRA_KV_TOKENS = DRAFT_LEN - 1
LAYER_FULL, LAYER_SHARED, LAYER_MTP = 0, 1, 2
DRAFT_VOCAB = 64
# Accepted tokens (bonus included, 1..K+1) of the three draft loops.
ACCEPTED = (3, 1, 6)
_DEV = torch.device("cuda")
BASE_FEATURES = os.environ.get(plan_testlib.FAST_PATHS_ENV, "")


def _base_feature_set():
    return set(plan_testlib.names(BASE_FEATURES))


@pytest.fixture(scope="module", autouse=True)
def base_plan():
    """The plan of the base fast paths, installed before any layer is adopted."""
    saved = (dispatch._INSTALLED[0], dispatch._VALIDATE[0], dispatch._require[0])
    plan_testlib.select(dispatch, BASE_FEATURES)
    yield
    dispatch._INSTALLED[0], dispatch._VALIDATE[0], dispatch._require[0] = saved


@pytest.fixture(autouse=True)
def decode_step():
    """Every test runs the target's decode steps."""
    with dispatch.step(dispatch.Phase.DECODE):
        yield


@dataclass
class _RopeConfig:
    hidden_size: int
    num_attention_heads: int
    rope_scaling: object
    max_position_embeddings: int
    rope_theta: float
    qk_rope_head_dim: int
    model_type: str


def _pos_embd_params():
    rope_config = _RopeConfig(
        hidden_size=HIDDEN,
        num_attention_heads=LOCAL_HEADS,
        rope_scaling=None,
        max_position_embeddings=MAX_POSITIONS,
        rope_theta=ROPE_THETA,
        qk_rope_head_dim=QK_ROPE,
        model_type="glm_moe_dsa",
    )
    # Same construction as the DeepSeek-V3-family attention layers.
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn, rope=RopeParams.from_config(rope_config), is_neox=False
    )


def _expected_target_kv_lens(context_len: int) -> List[int]:
    lens = [context_len]
    for accepted in ACCEPTED:
        lens.append(lens[-1] + accepted)
    return lens


@dataclass
class StepInputs:
    """Token-wise MLA/indexer inputs of one layer call."""

    q: torch.Tensor
    compressed_kv: torch.Tensor
    k_pe: torch.Tensor
    hidden: torch.Tensor
    qr: torch.Tensor
    position_ids: torch.Tensor


class Recorder:
    """Ordered record of one arm: indexer calls, shared rows and layer outputs."""

    def __init__(self, name: str, canonical: bool):
        self.name = name
        self.canonical = canonical
        self.events: List[dict] = []
        self.capability: List[tuple] = []
        self.sparse_mla_calls: List[tuple] = []
        self.tag = None

    def add(self, kind: str, **fields):
        fields.update(kind=kind, tag=self.tag)
        self.events.append(fields)


def _canonicalize_(rows: torch.Tensor):
    """Sort each top-k row in place (ascending local token ids)."""
    assert bool((rows >= 0).all()), "tested lengths fill every top-k slot"
    rows.copy_(torch.sort(rows, dim=1).values)


class IndexerTap:
    """Wraps ``Indexer.sparse_attn_indexer`` to record each call of an arm.

    With ``canonical`` set, the rows every consumer reads (the returned rows,
    and on a draft-loop reuse step the stashed row the reuse copies) are
    sorted in place, so the attention input order does not depend on the
    native top-k engine's in-row order, which varies run to run.
    """

    def __init__(self, recorder: Recorder):
        self.rec = recorder

    def wrap(self, original):
        tap = self

        def sparse_attn_indexer(indexer, metadata, *args, **kwargs):
            in_draft = bool(metadata.in_mtp_draft_loop)
            reuse = in_draft and bool(metadata.indexer_skip_topk)
            pre = {}
            if not in_draft:
                # The per-request schedule and lengths indexer_topk consumes, as they are
                # when the indexer runs.
                pre["sched"] = metadata.scheduler_metadata_buffer.clone()
                pre["kv"] = metadata.gen_indexer_kv_lens_cuda_runtime.clone()
                pre["kv_lens_cuda"] = metadata.kv_lens_cuda[: metadata.num_seqs].clone()
                pre["num_sms"] = metadata.num_sms
                pre["block_table_cols"] = metadata.indexer_k_cache_block_offsets.shape[1]
            if reuse:
                stash = metadata.shared_topk_indices[: metadata.num_generations]
                pre["stash"] = stash.clone()
                if tap.rec.canonical:
                    _canonicalize_(stash)
            engaged = indexer_topk.DECISION_COUNTS.get("engaged", 0)
            decided = sum(indexer_topk.DECISION_COUNTS.values())
            out = original(indexer, metadata, *args, **kwargs)
            rows = out[: metadata.num_tokens - metadata.num_ctx_tokens]
            if tap.rec.canonical and not reuse:
                _canonicalize_(rows)
            tap.rec.add(
                "indexer",
                layer=indexer.layer_idx,
                in_draft=in_draft,
                reuse=reuse,
                # On a reuse step rope_kv_append may defer the copy into its bridge, so the
                # returned buffer is only meaningful when the indexer selected.
                rows=None if reuse else torch.sort(rows, dim=1).values.clone(),
                indexer_topk_engaged=indexer_topk.DECISION_COUNTS.get("engaged", 0) - engaged,
                indexer_topk_decisions=sum(indexer_topk.DECISION_COUNTS.values()) - decided,
                **pre,
            )
            return out

        return sparse_attn_indexer


class _StubMtpLayer:
    """MTP layer of the stub draft model: runs the real layer-2 attention."""

    def __init__(self, pipe: "GlmDsaPipeline", rec: Recorder, loop_idx: int):
        self.pipe = pipe
        self.rec = rec
        self.loop_idx = loop_idx
        self.step = 0

    def __call__(
        self,
        *,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        position_ids,
        hidden_states,
        attn_metadata,
        spec_metadata,
        **kwargs,
    ):
        step = self.step
        self.step += 1
        self.rec.tag = ("draft", self.loop_idx, step)
        n = attn_metadata.num_tokens
        positions = position_ids.reshape(-1)[:n].to(torch.int32).contiguous()
        seed = 1000 + 37 * self.loop_idx + step
        inputs = self.pipe.make_inputs(seed=seed, n=n, positions=positions)
        out = self.pipe.layer_forward(LAYER_MTP, attn_metadata, inputs)
        self.rec.add("output", layer=LAYER_MTP, out=out.clone())
        # Hidden states that depend on the attention output, so the drafted
        # tokens do as well.
        hidden = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=_DEV)
        hidden[:, : out.shape[1]] = out
        return hidden

    @staticmethod
    def shared_head(hidden_states, lm_head, attn_metadata, return_context_logits):
        return hidden_states[:, :DRAFT_VOCAB].float()


def _sample_draft_tokens(logits, spec_metadata, batch_size, draft_step=0, mapping_lm_head_tp=None):
    return torch.argmax(logits, dim=-1).to(torch.int32)


def _make_worker():
    """The attributes of an MTP-Eagle one-model worker the draft loop reads."""
    worker = SimpleNamespace(
        is_mtp_eagle=True,
        guided_decoder=None,
        sa_enhancer=None,
        model_config=None,
        spec_config=SimpleNamespace(use_relaxed_acceptance_for_thinking=False),
        _auxiliary_state_handlers=None,
        _uses_external_shared_target_kv=False,
        use_dynamic_tree=False,
        _draft_loop_ops=None,
    )
    worker.draft_kv_cache_context = types.MethodType(SpecWorkerBase.draft_kv_cache_context, worker)
    for name in (
        "_run_draft_forward",
        "_get_step_all_rank_num_tokens",
        "_prepare_flash_mla_generation_layout",
    ):
        setattr(worker, name, types.MethodType(getattr(GlmMtpWorker, name), worker))
    worker.sample_draft_tokens = _sample_draft_tokens
    worker._can_use_rejection_sampling = lambda spec_metadata: False
    return worker


def _make_spec_metadata():
    return SimpleNamespace(
        runtime_draft_len=DRAFT_LEN,
        batch_indices_cuda=torch.zeros(1, dtype=torch.int32, device=_DEV),
        all_rank_num_tokens=None,
        subseq_all_rank_num_tokens=None,
        wants_advanced_draft_sampling=False,
        max_num_requests=1,
        draft_tokens=torch.zeros(DRAFT_LEN, dtype=torch.int32, device=_DEV),
        enable_penalty=False,
    )


class GlmDsaPipeline:
    """Three GLM-5.2-shaped DSA MLA layers over one ``DSACacheManagerV2``."""

    def __init__(self, context_len: int, seed: int = 0):
        # ``context_len`` is the KV length of the first target verify step
        # (its six verify tokens included).
        self.context_len = context_len
        self.prompt_len = context_len - NEXT_N
        self.capacity = context_len + sum(ACCEPTED) + 2 * NEXT_N + NUM_EXTRA_KV_TOKENS
        assert self.capacity + NUM_EXTRA_KV_TOKENS <= SERVED_MAX_SEQ_LEN
        self.mapping = Mapping(world_size=1, tp_size=1, rank=0)
        self.pretrained = SimpleNamespace(
            rms_norm_eps=1e-5,
            num_hidden_layers=2,
            index_topk_pattern=["F", "S"],
            qk_rope_head_dim=QK_ROPE,
        )
        self.sparse_config = DeepSeekSparseAttentionConfig(
            index_n_heads=INDEX_HEADS,
            index_head_dim=INDEX_DIM,
            index_topk=TOP_K,
            use_cute_dsl_topk=True,
            use_cute_dsl_paged_mqa_logits=True,
            enable_heuristic_topk=True,
            index_share_for_mtp_iteration=True,
        )
        quant_config = QuantConfig()
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8
        self.model_config = ModelConfig(
            mapping=self.mapping,
            sparse_attention_config=self.sparse_config,
            pretrained_config=self.pretrained,
            quant_config=quant_config,
        )
        self.model_config.extra_attrs["kv_cache_dtype"] = "fp8"
        pos = _pos_embd_params()
        self.layers = []
        for layer_idx in range(3):
            mla = MLA(
                hidden_size=HIDDEN,
                num_attention_heads=LOCAL_HEADS,
                num_key_value_heads=1,
                qk_nope_head_dim=QK_NOPE,
                qk_rope_head_dim=QK_ROPE,
                v_head_dim=V_HEAD,
                q_lora_rank=Q_LORA,
                kv_lora_rank=KV_LORA,
                predicted_tokens_per_seq=NEXT_N,
                max_position_embeddings=MAX_POSITIONS,
                bias=False,
                pos_embd_params=pos,
                layer_idx=layer_idx,
                dtype=torch.bfloat16,
                config=self.model_config,
            )
            # The model's DeepseekV32Attention adds only its constructor to MLA; the pipeline
            # builds the MLA at GLM shapes directly and adopts it as the model's attention. The
            # model builds the FMHA libraries in ``create_weights`` after adoption; a module built
            # alone rebuilds them for the library adoption declared.
            mla.__class__ = DeepseekV32Attention
            GlmMLA.adopt(mla)
            for backend in (mla.mha, mla.mqa):
                if backend is not None and backend.prepended_fmha_lib_classes:
                    backend.update_quant_config(backend.quant_config)
            mla = mla.to(_DEV)
            if mla.mqa.indexer is not None:
                mla.indexer = mla.mqa.indexer.to(_DEV)
            self.layers.append(mla)
        assert self.layers[LAYER_FULL].mqa.indexer is not None
        assert self.layers[LAYER_SHARED].mqa.indexer is None, "layer 1 must be shared"
        assert self.layers[LAYER_MTP].mqa.indexer is not None
        self._init_weights(seed)

        self.kv_cache_manager = DSACacheManagerV2(
            KvCacheConfig(
                dtype="fp8",
                # At least max_seq_len, or the manager clamps max_seq_len.
                max_tokens=SERVED_MAX_SEQ_LEN + 64 * TOKENS_PER_BLOCK,
                enable_block_reuse=False,
                host_cache_size=0,
            ),
            CacheType.SELFKONLY,
            num_layers=3,
            num_kv_heads=1,
            head_dim=KV_LORA + QK_ROPE,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=SERVED_MAX_SEQ_LEN,
            max_batch_size=1,
            mapping=self.mapping,
            dtype=DataType.FP8,
            sparse_attention_config=self.sparse_config,
            pretrained_config=self.pretrained,
        )
        assert self.kv_cache_manager.max_seq_len == SERVED_MAX_SEQ_LEN
        assert self.kv_cache_manager.indexer_k_cache_local_layer_mask == [True, False, True]
        added = self.kv_cache_manager.add_dummy_requests(
            request_ids=[0], token_nums=[self.capacity], is_gen=False, prepare_resource=True
        )
        assert added is not None, "KV cache allocation failed"
        sparse_params = self.sparse_config.to_sparse_params(pretrained_config=self.pretrained)
        self.attn_cls = get_attention_backend("TRTLLM", sparse_params=sparse_params)
        self.metadata_params = self.sparse_config.to_sparse_metadata_params(
            pretrained_config=self.pretrained
        )
        assert self.metadata_params.has_shared_indexer_layers
        assert self.metadata_params.mtp_index_share
        self._populate_context(seed)
        self.md = self._new_metadata(
            num_contexts=0,
            seq_len=NEXT_N,
            cached=self.prompt_len,
            max_num_tokens=64,
            extra_kv=NUM_EXTRA_KV_TOKENS,
        )
        with torch.no_grad():
            self.embed_tokens = torch.nn.Embedding(DRAFT_VOCAB, HIDDEN).to(_DEV, torch.bfloat16)

    # ------------------------------------------------------------------ setup
    def _init_weights(self, seed: int):
        gen = torch.Generator(device=_DEV).manual_seed(seed)

        def randn_like(t):
            return (torch.randn(t.shape, device=_DEV, generator=gen) * 0.02).to(t.dtype)

        with torch.no_grad():
            for mla in self.layers:
                mla.kv_b_proj.weight.copy_(randn_like(mla.kv_b_proj.weight))
                kv_b = mla.kv_b_proj.weight.data.view(LOCAL_HEADS, QK_NOPE + V_HEAD, KV_LORA)
                mla.v_b_proj.data = kv_b[:, QK_NOPE:, :].contiguous()
                mla.k_b_proj_trans.data = kv_b[:, :QK_NOPE, :].transpose(1, 2).contiguous()
                indexer = mla.mqa.indexer
                if indexer is not None:
                    for linear in (indexer.wq_b, indexer.wk, indexer.weights_proj):
                        linear.weight.copy_(randn_like(linear.weight))
                    indexer.post_load_weights()

    def _new_metadata(self, *, num_contexts, seq_len, cached, max_num_tokens, extra_kv):
        return self.attn_cls.Metadata(
            seq_lens=torch.tensor([seq_len], dtype=torch.int),
            request_ids=[0],
            max_num_requests=1,
            num_contexts=num_contexts,
            prompt_lens=[self.prompt_len],
            max_num_tokens=max_num_tokens,
            kv_cache_manager=self.kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[cached], num_extra_kv_tokens=extra_kv
            ),
            mapping=self.mapping,
            sparse_metadata_params=self.metadata_params,
        )

    @torch.inference_mode()
    def _populate_context(self, seed: int):
        """Write the prompt with the native producers (one prefill-shaped call)."""
        n = self.prompt_len
        md = self._new_metadata(num_contexts=1, seq_len=n, cached=0, max_num_tokens=n, extra_kv=0)
        md.prepare()
        gen = torch.Generator(device=_DEV).manual_seed(seed + 1)
        for layer_idx in (LAYER_FULL, LAYER_MTP):
            indexer = self.layers[layer_idx].mqa.indexer
            k = torch.randn(n, INDEX_DIM, device=_DEV, generator=gen).to(torch.bfloat16)
            rope = indexer.rope_dim
            # The indexer's own K quantizer and K-cache scatter.
            k_fp8, k_scale = indexer._prep_q_or_k(
                k[:, :rope].contiguous(), k[:, rope:].contiguous()
            )
            indexer._update_k_cache(k_fp8, k_scale, md)
            del k, k_fp8, k_scale
        for mla in self.layers:
            latent = torch.randn(n, KV_LORA + QK_ROPE, device=_DEV, generator=gen)
            latent = latent.to(torch.bfloat16)
            dummy_q = torch.zeros(
                n, LOCAL_HEADS * (QK_NOPE + QK_ROPE), device=_DEV, dtype=torch.bfloat16
            )
            mla.mqa.mla_rope_append_paged_kv_assign_q(dummy_q, latent, md)
            del latent, dummy_q
        torch.cuda.synchronize()
        del md
        torch.cuda.empty_cache()

    def shutdown(self):
        self.kv_cache_manager.shutdown()

    # ------------------------------------------------------------- execution
    def make_inputs(self, seed: int, n: int, positions: torch.Tensor) -> StepInputs:
        gen = torch.Generator(device=_DEV).manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, device=_DEV, generator=gen).to(torch.bfloat16)

        return StepInputs(
            q=rnd(n, LOCAL_HEADS * (QK_NOPE + QK_ROPE)),
            compressed_kv=rnd(n, KV_LORA),
            k_pe=rnd(n, QK_ROPE),
            hidden=rnd(n, HIDDEN),
            qr=rnd(n, Q_LORA),
            position_ids=positions,
        )

    def project(self, layer_idx: int, inputs: StepInputs):
        """Graph-capturable indexer projections of a full-indexer layer (Op 1)."""
        indexer = self.layers[layer_idx].mqa.indexer
        if indexer is None:
            return []
        return list(indexer.pre_indexer_proj(inputs.qr, inputs.hidden, inputs.position_ids))

    def attend(self, layer_idx: int, md, inputs: StepInputs, intermediates) -> torch.Tensor:
        """``_forward_dsa_attn`` of one layer (Op 2) on copies of the inputs."""
        mla = self.layers[layer_idx]
        n = md.num_tokens
        latent = torch.cat([inputs.compressed_kv, inputs.k_pe], dim=-1)
        output = torch.empty(n, LOCAL_HEADS * V_HEAD, dtype=torch.bfloat16, device=_DEV)
        _forward_dsa_attn(
            mla,
            inputs.q.clone(),
            inputs.compressed_kv.clone(),
            inputs.k_pe.clone(),
            latent,
            intermediates,
            inputs.position_ids,
            md,
            output,
        )
        return output

    def layer_forward(self, layer_idx: int, md, inputs: StepInputs) -> torch.Tensor:
        return self.attend(layer_idx, md, inputs, self.project(layer_idx, inputs))

    def prepare_target_step(self, host_cached: int, kv_offset: int):
        """Runner order: host metadata + ``prepare()``, then ``_preprocess_inputs``."""
        md = self.md
        md.seq_lens = torch.tensor([NEXT_N], dtype=torch.int)
        md.num_contexts = 0
        md.request_ids = [0]
        md.prompt_lens = [self.prompt_len]
        md.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[host_cached],
            num_extra_kv_tokens=NUM_EXTRA_KV_TOKENS,
        )
        md.update_spec_dec_param(
            batch_size=1,
            is_spec_decoding_enabled=True,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=DRAFT_LEN,
            max_total_draft_tokens=DRAFT_LEN,
            spec_metadata=None,
            spec_tree_manager=None,
            num_contexts=0,
        )
        md.prepare()
        # DecoderRunner._preprocess_inputs (overlap scheduler + spec decode).
        md.on_update_kv_lens()
        md.kv_lens_cuda[0:1] += kv_offset
        md.on_update_kv_lens()
        return md

    @torch.inference_mode()
    def run_sequence(self, rec: Recorder, draft_loop: bool):
        """Four target steps separated by three draft loops."""
        md = self.md
        kv = self.context_len
        cached = kv - NEXT_N
        host_cached = cached
        kv_offset = 0
        for step in range(len(ACCEPTED) + 1):
            rec.tag = ("target", step)
            self.prepare_target_step(host_cached, kv_offset)
            positions = torch.arange(cached, cached + NEXT_N, dtype=torch.int32, device=_DEV)
            rec.add("target_kv", kv=kv)
            for layer_idx in (LAYER_FULL, LAYER_SHARED):
                if layer_idx == LAYER_SHARED:
                    rec.add(
                        "shared_rows",
                        rows=torch.sort(md.shared_topk_indices[:NEXT_N], dim=1).values.clone(),
                    )
                inputs = self.make_inputs(seed=10 * step + layer_idx, n=NEXT_N, positions=positions)
                out = self.layer_forward(layer_idx, md, inputs)
                rec.add("output", layer=layer_idx, out=out.clone())
            if step == len(ACCEPTED):
                break
            accepted = ACCEPTED[step]
            self.run_draft_loop(rec, step, accepted, positions, draft_loop)
            rec.tag = ("after_loop", step)
            rec.add("stale_sched", sched=md.scheduler_metadata_buffer.clone())
            # Overlap scheduler: the host assumes every draft was accepted and
            # the next forward corrects kv_lens_cuda by (accepted - next_n).
            host_cached = cached + NEXT_N
            kv_offset = accepted - NEXT_N
            cached += accepted
            kv += accepted
        torch.cuda.synchronize()
        return rec

    def run_draft_loop(
        self, rec: Recorder, loop_idx: int, accepted: int, positions, draft_loop: bool
    ):
        md = self.md
        worker = _make_worker()
        spec_md = _make_spec_metadata()
        num_accepted = torch.tensor([accepted], dtype=torch.int32, device=_DEV)
        draft_model = SimpleNamespace(
            embed_tokens=self.embed_tokens,
            lm_head=None,
            mtp_layers=[_StubMtpLayer(self, rec, loop_idx)],
        )
        if draft_loop:
            ops = dispatch.get_impl("draft_loop", "draft_loop_ops")(worker, md, spec_md)
            assert ops is not None, "draft_loop declined the draft loop (see the capability log)"
            # Acceptance is not run here (there are no target logits); give draft_loop
            # the step-0 rows the native loop computes for one generation request.
            ops.gather_ids = (num_accepted.to(torch.int64) - 1).contiguous()
            worker._draft_loop_ops = ops
        gen = torch.Generator(device=_DEV).manual_seed(7000 + loop_idx)
        inputs = {
            "input_ids": torch.randint(0, DRAFT_VOCAB, (NEXT_N,), device=_DEV, generator=gen),
            "position_ids": positions.clone(),
            "hidden_states": torch.randn(NEXT_N, HIDDEN, device=_DEV, generator=gen).to(
                torch.bfloat16
            ),
            "attn_metadata": md,
            "spec_metadata": spec_md,
        }
        # The worker's _forward_impl around the loop.
        md.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
        try:
            tokens = GlmMtpWorker._forward_linear_draft_loop(
                worker,
                inputs,
                md,
                spec_md,
                draft_model,
                None,
                0,
                1,
                num_accepted,
                None,
            )
        finally:
            md.restore_from_spec_dec()
            md.on_update()
            md.use_spec_decoding = True
        rec.tag = ("draft_tokens", loop_idx)
        rec.add("draft_tokens", tokens=tokens.clone())


# --------------------------------------------------------------------- arms
def _install_recorders(mp, rec: Recorder):
    original_log = dispatch.log_capability

    def log_capability(feature_id, key, eligible, reason=""):
        rec.capability.append((feature_id, key, bool(eligible), reason, rec.tag))
        return original_log(feature_id, key, eligible, reason)

    mp.setattr(dispatch, "log_capability", log_capability)
    tap = IndexerTap(rec)
    original_indexer = indexer_mod.Indexer.sparse_attn_indexer
    mp.setattr(indexer_mod.Indexer, "sparse_attn_indexer", tap.wrap(original_indexer))
    if "sparse_mla_decode" in _base_feature_set():
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
            sparse_mla as sparse_mla_sparse_mla,
        )

        sparse_mla_cls = sparse_mla_sparse_mla.SparseMlaDecodeFmha
        original_forward = sparse_mla_cls.forward

        def forward(self, *args, **kwargs):
            rec.sparse_mla_calls.append(rec.tag)
            return original_forward(self, *args, **kwargs)

        mp.setattr(sparse_mla_cls, "forward", forward)


def _select(mp, with_indexer_topk: bool):
    features = _base_feature_set()
    if with_indexer_topk:
        features.add("indexer_topk")
    else:
        features.discard("indexer_topk")
    plan_testlib.select(dispatch, features, mp)
    assert dispatch.is_enabled("indexer_topk") == with_indexer_topk


def run_arm(pipe: GlmDsaPipeline, monkeypatch, name: str, with_indexer_topk: bool, canonical: bool):
    rec = Recorder(name, canonical)
    with monkeypatch.context() as mp:
        _select(mp, with_indexer_topk)
        _install_recorders(mp, rec)
        pipe.run_sequence(rec, draft_loop="draft_loop" in _base_feature_set())
    return rec


# -------------------------------------------------------------- comparisons
def _fresh_schedule(kv: torch.Tensor, num_sms: int) -> torch.Tensor:
    return get_paged_mqa_logits_metadata(kv.view(-1, 1), SCHEDULE_BLOCK_KV, num_sms)


def _indexer_calls(rec: Recorder, phase: str):
    return [e for e in rec.events if e["kind"] == "indexer" and e["tag"][0] == phase]


def check_arm_invariants(rec: Recorder, pipe: GlmDsaPipeline, with_indexer_topk: bool):
    """Per-arm checks that do not need another arm."""
    expected_kv = _expected_target_kv_lens(pipe.context_len)
    target_calls = _indexer_calls(rec, "target")
    assert len(target_calls) == len(expected_kv), "one full-layer indexer call per target step"
    for call, kv in zip(target_calls, expected_kv):
        assert call["layer"] == LAYER_FULL and not call["in_draft"]
        # The lengths and the per-request schedule indexer_topk reads are this step's,
        # not a draft step's leftovers.
        assert call["kv"].tolist() == [kv], (rec.name, call["tag"], call["kv"].tolist(), kv)
        assert call["kv_lens_cuda"].tolist() == [kv]
        fresh = _fresh_schedule(call["kv"], call["num_sms"])
        assert torch.equal(call["sched"], fresh), (rec.name, call["tag"], "stale schedule")
        assert call["block_table_cols"] % 2 == 0
        assert call["indexer_topk_decisions"] == (1 if with_indexer_topk else 0)
        assert call["indexer_topk_engaged"] == (1 if with_indexer_topk else 0), (
            rec.name,
            call["tag"],
        )
    draft_calls = _indexer_calls(rec, "draft")
    assert len(draft_calls) == DRAFT_LEN * len(ACCEPTED)
    for call in draft_calls:
        assert call["layer"] == LAYER_MTP and call["in_draft"]
        assert call["reuse"] == (call["tag"][2] > 0)
        # indexer_topk is target-only: it declines every draft selection before device
        # work, and the reuse steps never reach it.
        assert call["indexer_topk_engaged"] == 0
        assert call["indexer_topk_decisions"] == (
            1 if with_indexer_topk and not call["reuse"] else 0
        )
    # The shared layer reads exactly the full layer's rows of the same step.
    for step in range(len(expected_kv)):
        full = next(e for e in target_calls if e["tag"] == ("target", step))
        shared = next(
            e for e in rec.events if e["kind"] == "shared_rows" and e["tag"] == ("target", step)
        )
        assert torch.equal(full["rows"], shared["rows"]), (rec.name, step)
    features = _base_feature_set()
    caps = rec.capability
    num_attention_calls = 2 * len(expected_kv) + DRAFT_LEN * len(ACCEPTED)
    # The consumers of indexer_topk's rows are the target layers; draft-step decisions
    # of the neighbour packages are reported, not asserted.
    if "sparse_mla_decode" in features:
        declined = [c for c in caps if c[0] == "sparse_mla_decode" and not c[2]]
        assert not [c for c in declined if c[4] and c[4][0] == "target"], declined
        target_sparse_mla = [t for t in rec.sparse_mla_calls if t[0] == "target"]
        # Both target layers (full and shared) of every step attend through sparse_mla_decode.
        assert len(target_sparse_mla) == 2 * len(expected_kv), rec.sparse_mla_calls
        if len(rec.sparse_mla_calls) != num_attention_calls or declined:
            print(
                f"[indexer_topk] {rec.name}: sparse_mla_decode calls {len(rec.sparse_mla_calls)}, declines {declined}"
            )
    if "rope_kv_append" in features:
        bridge = [c for c in caps if c[0] == "rope_kv_append" and c[1] == "bridge"]
        target_bridge = [c for c in bridge if c[4][0] == "target"]
        assert len(target_bridge) == 2 * len(expected_kv), bridge
        assert all(c[2] for c in target_bridge), bridge
        if len(bridge) != num_attention_calls or not all(c[2] for c in bridge):
            print(
                f"[indexer_topk] {rec.name}: rope_kv_append bridge decisions {[c[2] for c in bridge]}"
            )
    if "draft_loop" in features:
        steps = [c for c in caps if c[0] == "draft_loop" and c[1].startswith("draft_step:")]
        assert len(steps) == DRAFT_LEN * len(ACCEPTED), steps
        assert all(c[1] == "draft_step:engaged" for c in steps), steps


def _pairs(rec_a: Recorder, rec_b: Recorder):
    assert [(e["kind"], e["tag"]) for e in rec_a.events] == [
        (e["kind"], e["tag"]) for e in rec_b.events
    ]
    return zip(rec_a.events, rec_b.events)


def compare_selection(rec_a: Recorder, rec_b: Recorder, draft_tokens: bool = True):
    """Discrete invariants: identical selections, shared rows and stashes."""
    for a, b in _pairs(rec_a, rec_b):
        if a["kind"] == "indexer":
            if a["rows"] is not None:
                assert torch.equal(a["rows"], b["rows"]), (rec_a.name, rec_b.name, a["tag"])
            if "stash" in a:
                sa = torch.sort(a["stash"], dim=1).values
                sb = torch.sort(b["stash"], dim=1).values
                assert torch.equal(sa, sb), (rec_a.name, rec_b.name, a["tag"], "stash")
        elif a["kind"] == "shared_rows":
            assert torch.equal(a["rows"], b["rows"]), (rec_a.name, rec_b.name, a["tag"])
        elif a["kind"] == "draft_tokens" and draft_tokens:
            assert torch.equal(a["tokens"], b["tokens"]), (rec_a.name, rec_b.name, a["tag"])


def output_diffs(rec_a: Recorder, rec_b: Recorder) -> Dict[tuple, float]:
    diffs = {}
    for a, b in _pairs(rec_a, rec_b):
        if a["kind"] == "output":
            fa, fb = a["out"].float(), b["out"].float()
            assert torch.isfinite(fa).all() and torch.isfinite(fb).all()
            diffs[(a["tag"], a["layer"])] = (fa - fb).abs().max().item()
    return diffs


def _report(title: str, diffs: Dict[tuple, float]):
    worst = max(diffs.values()) if diffs else 0.0
    nonzero = sum(1 for v in diffs.values() if v != 0.0)
    print(
        f"[indexer_topk] {title}: {len(diffs)} layer outputs, {nonzero} differ, max |d| = {worst:.3e}"
    )


# --------------------------------------------------------------------- tests
# DeepGEMM's decode schedule balances ceil(kv / SPLIT_KV) work units per
# request over the SMs (SPLIT_KV is one of 64..512 tokens), so it changes only
# where kv crosses a multiple of SPLIT_KV. 66,046 and 1,048,062 put the KV
# length the first draft loop leaves behind (L + a1 - 1) on a multiple of 512
# and the next target step's (L + a1) one past it, so a leftover draft-step
# schedule differs from the target schedule for every SPLIT_KV candidate.
SCHEDULE_SPLIT_LCM = 512
CONTEXT_LENS = [66046, 491061, 1048062]


def _is_schedule_boundary(context_len: int) -> bool:
    return (context_len + ACCEPTED[0] - 1) % SCHEDULE_SPLIT_LCM == 0


@pytest.fixture(scope="module", params=CONTEXT_LENS, ids=lambda n: f"kv{n}")
def pipeline(request):
    pipe = GlmDsaPipeline(request.param)
    yield pipe
    pipe.shutdown()


def _served_order_report(title: str, rec_a: Recorder, rec_b: Recorder) -> float:
    diffs = output_diffs(rec_a, rec_b)
    outputs_a = {(e["tag"], e["layer"]): e["out"] for e in rec_a.events if e["kind"] == "output"}
    outputs_b = {(e["tag"], e["layer"]): e["out"] for e in rec_b.events if e["kind"] == "output"}
    worst_cos = 1.0
    for key in diffs:
        cos = torch.nn.functional.cosine_similarity(
            outputs_a[key].float().flatten(), outputs_b[key].float().flatten(), dim=0
        ).item()
        worst_cos = min(worst_cos, cos)
    _report(title, diffs)
    print(f"[indexer_topk] {title}: min cosine {worst_cos:.7f}")
    return worst_cos


def test_indexer_topk_in_real_decode_pipeline(pipeline, monkeypatch):
    """indexer_topk on vs off in the real metadata/draft-loop pipeline (sets and outputs)."""
    print(f"[indexer_topk] base features '{BASE_FEATURES}', kv {pipeline.context_len}")
    native_a = run_arm(pipeline, monkeypatch, "native_a", with_indexer_topk=False, canonical=True)
    native_b = run_arm(pipeline, monkeypatch, "native_b", with_indexer_topk=False, canonical=True)
    with_indexer_topk = run_arm(
        pipeline, monkeypatch, "indexer_topk", with_indexer_topk=True, canonical=True
    )
    for rec, on in ((native_a, False), (native_b, False), (with_indexer_topk, True)):
        check_arm_invariants(rec, pipeline, on)
    compare_selection(native_a, native_b)
    compare_selection(native_a, with_indexer_topk)
    native_env = output_diffs(native_a, native_b)
    indexer_topk_diff = output_diffs(native_a, with_indexer_topk)
    _report("native vs native (canonical order)", native_env)
    _report("indexer_topk vs native (canonical order)", indexer_topk_diff)
    for key, value in indexer_topk_diff.items():
        # Identical index sets in the same order: the attention is the same
        # computation, so allow only what native shows run to run.
        assert value <= native_env[key], (key, value, native_env[key])

    # Served order (no canonicalization): the sets still match. The outputs
    # then depend on the native top-k's run-to-run in-row order (the FP8
    # attention tiles the selected tokens in that order), for native and indexer_topk
    # alike; this is characterized, not bounded by a tolerance.
    native_raw = run_arm(
        pipeline, monkeypatch, "native_raw", with_indexer_topk=False, canonical=False
    )
    native_raw_b = run_arm(
        pipeline, monkeypatch, "native_raw_b", with_indexer_topk=False, canonical=False
    )
    indexer_topk_raw = run_arm(
        pipeline, monkeypatch, "indexer_topk_raw", with_indexer_topk=True, canonical=False
    )
    check_arm_invariants(native_raw, pipeline, False)
    check_arm_invariants(native_raw_b, pipeline, False)
    check_arm_invariants(indexer_topk_raw, pipeline, True)
    compare_selection(native_raw, native_raw_b, draft_tokens=False)
    compare_selection(native_raw, indexer_topk_raw, draft_tokens=False)
    compare_selection(native_a, indexer_topk_raw, draft_tokens=False)
    native_cos = _served_order_report("native vs native (served order)", native_raw, native_raw_b)
    indexer_topk_cos = _served_order_report(
        "indexer_topk vs native (served order)", native_raw, indexer_topk_raw
    )
    assert native_cos > 0.999 and indexer_topk_cos > 0.999, (native_cos, indexer_topk_cos)

    # The freshness check is not vacuous at the boundary lengths: the schedule
    # the first draft loop leaves behind is that of its last step and differs
    # from the next target step's.
    stale = [e for e in native_a.events if e["kind"] == "stale_sched"]
    targets = _indexer_calls(native_a, "target")
    differs = [not torch.equal(s["sched"], t["sched"]) for s, t in zip(stale, targets[1:])]
    print(
        f"[indexer_topk] leftover draft schedule differs from the next target schedule: {differs}"
    )
    if _is_schedule_boundary(pipeline.context_len):
        leftover_kv = torch.tensor(
            [pipeline.context_len + ACCEPTED[0] - 1], dtype=torch.int32, device=_DEV
        )
        leftover = _fresh_schedule(leftover_kv, targets[1]["num_sms"])
        assert torch.equal(stale[0]["sched"], leftover), "leftover is the last draft step's"
        assert differs[0], "boundary case must exercise a schedule change"
    caps = sorted({c[:3] for c in with_indexer_topk.capability}, key=str)
    print(f"[indexer_topk] capability decisions in the indexer_topk arm: {caps}")
    print(
        f"[indexer_topk] sparse_mla_decode sparse-MLA calls per arm: {len(with_indexer_topk.sparse_mla_calls)}"
    )
    print(f"[indexer_topk] indexer block-table width: {targets[0]['block_table_cols']}")
    print(f"[indexer_topk] indexer_topk decisions: {dict(indexer_topk.DECISION_COUNTS)}")


def _mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


def _digest(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.view(torch.int16)
    elif t.element_size() == 1:
        t = t.view(torch.uint8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("context_len", [491061])
def test_rank_agreement(context_len, monkeypatch):
    """Every rank of the MPI job selects identical sets with indexer_topk on and off."""
    comm = _mpi_comm()
    if comm is None:
        pytest.skip("needs an MPI job with more than one rank")
    rank, size = comm.Get_rank(), comm.Get_size()
    pipe = GlmDsaPipeline(context_len)
    try:
        native = run_arm(pipe, monkeypatch, "native", with_indexer_topk=False, canonical=True)
        with_indexer_topk = run_arm(
            pipe, monkeypatch, "indexer_topk", with_indexer_topk=True, canonical=True
        )
        check_arm_invariants(native, pipe, False)
        check_arm_invariants(with_indexer_topk, pipe, True)
        compare_selection(native, with_indexer_topk)
        kvm = pipe.kv_cache_manager
        first_page = (
            int(kvm.get_pool_block_indices(1, request_ids=[0])[0, 0])
            * kvm.indexer_k_cache_page_scale
        )
        props = torch.cuda.get_device_properties(torch.cuda.current_device())

        def selections(rec):
            return [
                (str(e["tag"]), e["layer"], _digest(e["rows"]))
                for e in rec.events
                if e["kind"] == "indexer" and e["rows"] is not None
            ]

        digest = {
            "host": socket.gethostname(),
            "gpu": str(getattr(props, "uuid", props.name)),
            "indexer_k_first_page": _digest(
                kvm.get_indexer_k_cache_buffers(LAYER_FULL)[first_page]
            ),
            "selections": selections(with_indexer_topk),
            "native_selections": selections(native),
            "outputs": [
                (str(e["tag"]), e["layer"], _digest(e["out"]))
                for e in with_indexer_topk.events
                if e["kind"] == "output"
            ],
            "indexer_topk_decisions": dict(indexer_topk.DECISION_COUNTS),
        }
    finally:
        pipe.shutdown()
    everyone = comm.allgather(digest)
    if rank == 0:
        hosts = sorted({d["host"] for d in everyone})
        print(f"[indexer_topk] {size} ranks on {len(hosts)} hosts: {hosts}")
        for r, d in enumerate(everyone):
            print(
                f"[indexer_topk] rank {r} {d['host']} {d['gpu']} indexer_topk decisions {d['indexer_topk_decisions']}"
            )
        print(f"[indexer_topk] rank 0 selection digests: {everyone[0]['selections']}")
    ref = everyone[0]
    for r, d in enumerate(everyone):
        assert d["indexer_k_first_page"] == ref["indexer_k_first_page"], r
        assert d["selections"] == ref["selections"], r
        assert d["native_selections"] == ref["native_selections"], r
        assert d["selections"] == d["native_selections"], r
        assert d["indexer_topk_decisions"] == ref["indexer_topk_decisions"], r
        # The attention is replicated per rank on identical inputs as well.
        assert d["outputs"] == ref["outputs"], r


# ------------------------------------------------------------------- timing
TIMING_LENS = [4224, 16384, 32768, 65536, 131072, 314721, 391000, 491061]
_TIMING_REPS = 16
_TIMING_ROUNDS = 15
_TIMING_REPLAYS = 5


def _replay_us(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


def _capture_region(pipe, monkeypatch, kv: int, with_indexer_topk: bool, flush: torch.Tensor):
    """CUDA graph of ``_TIMING_REPS`` x (L2 flush, full layer + shared layer attention)."""
    with monkeypatch.context() as mp, torch.inference_mode():
        _select(mp, with_indexer_topk)
        md = pipe.prepare_target_step(host_cached=kv - NEXT_N, kv_offset=0)
        positions = torch.arange(kv - NEXT_N, kv, dtype=torch.int32, device=_DEV)
        work = []
        for layer_idx in (LAYER_FULL, LAYER_SHARED):
            inputs = pipe.make_inputs(seed=500 + layer_idx, n=NEXT_N, positions=positions)
            work.append((layer_idx, inputs, pipe.project(layer_idx, inputs)))

        def region():
            for layer_idx, inputs, intermediates in work:
                pipe.attend(layer_idx, md, inputs, intermediates)

        engaged = indexer_topk.DECISION_COUNTS.get("engaged", 0)
        region()  # compiles the kernels of this shape outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            for _ in range(_TIMING_REPS):
                flush.zero_()
                region()
        torch.cuda.synchronize()
        engaged = indexer_topk.DECISION_COUNTS.get("engaged", 0) - engaged
    assert engaged == ((_TIMING_REPS + 1) if with_indexer_topk else 0), engaged
    return graph, work


def _time_arm(pipe, monkeypatch, kv: int, with_indexer_topk: bool, flush, flush_graph):
    """Per-region microseconds (median, spread) of one arm at one KV length.

    The graph is replayed before the next ``prepare()``: the non-graph metadata
    re-derives per-step tensors (such as the cached pool view) there, which a
    graph captured earlier would still point at.
    """
    graph, work = _capture_region(pipe, monkeypatch, kv, with_indexer_topk, flush)
    flush_samples, samples = [], []
    for _ in range(_TIMING_ROUNDS):
        flush_samples.append(_replay_us(flush_graph, _TIMING_REPLAYS))
        samples.append(_replay_us(graph, _TIMING_REPLAYS))
    region_us = (_median(samples) - _median(flush_samples)) / _TIMING_REPS
    spread_us = (max(samples) - min(samples)) / _TIMING_REPS
    del graph, work
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return region_us, spread_us


@pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_INDEXER_TOPK_PIPELINE_TIMING") != "1",
    reason="set GLM_MOE_DSA_INDEXER_TOPK_PIPELINE_TIMING=1 to time",
)
def test_indexer_topk_region_timing(monkeypatch):
    """Attention region of a full + shared layer pair, indexer_topk off vs on, in CUDA graphs.

    The region is ``_forward_dsa_attn`` of both layers with the served
    neighbours of the indexer (indexer K scatter, shared-layer copy, remap,
    RoPE/FP8-query/KV append, sparse MLA). Each capture flushes L2 before
    every region (the served K caches of 21 full layers are not L2 resident);
    the flush alone is timed in its own graph and subtracted. Native is timed
    before and after indexer_topk at each length; their difference is the drift.
    """
    pipe = GlmDsaPipeline(max(TIMING_LENS) + TOKENS_PER_BLOCK)
    flush = torch.empty(1 << 29, dtype=torch.uint8, device=_DEV)
    results = []
    try:
        flush_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(flush_graph, capture_error_mode="thread_local"):
            for _ in range(_TIMING_REPS):
                flush.zero_()
        for kv in TIMING_LENS:
            native_1, spread_1 = _time_arm(pipe, monkeypatch, kv, False, flush, flush_graph)
            indexer_topk_us, indexer_topk_spread = _time_arm(
                pipe, monkeypatch, kv, True, flush, flush_graph
            )
            native_2, spread_2 = _time_arm(pipe, monkeypatch, kv, False, flush, flush_graph)
            native_us = (native_1 + native_2) / 2
            row = {
                "kv": kv,
                "base_features": BASE_FEATURES,
                "native_region_us": round(native_us, 3),
                "indexer_topk_region_us": round(indexer_topk_us, 3),
                "delta_us": round(indexer_topk_us - native_us, 3),
                "native_drift_us": round(native_2 - native_1, 3),
                "native_spread_us": round(max(spread_1, spread_2), 3),
                "indexer_topk_spread_us": round(indexer_topk_spread, 3),
            }
            results.append(row)
            print(f"[indexer_topk-timing] {json.dumps(row)}")
    finally:
        pipe.shutdown()
    assert len(results) == len(TIMING_LENS)
