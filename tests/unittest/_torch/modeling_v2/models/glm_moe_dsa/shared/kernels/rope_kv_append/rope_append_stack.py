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
"""Real DSA layer stack for the rope_kv_append call-site, composition and region tests.

``Stack`` builds real ``MLA`` modules at GLM-5.2 TP8 attention geometry (8
local heads, q-LoRA 2048, kv-LoRA 512, nope 192, rope 64, v 256, 32 x 128
indexer, top-k 2048, FP8 KV cache, block 64, heuristic GVR top-k, CuTe DSL
logits) in a full/shared indexer pattern plus the MTP layer, adopted as the target's
``GlmMLA``, a real ``DSACacheManager`` filled with FP8 context, and real generation metadata.
On decode steps, ``run_seg`` drives the target's ``_forward_dsa_attn`` with given projections;
``run_proj`` drives the target's ``forward_sparse_attn`` from hidden states, so the
projection producer (native, or absorbed_query/shared_indexer_absorbed_query/draft_absorbed_query
when they run) is the real one. ``Engagement`` counts native, rope_kv_append and neighbour-package
entry points and can record each layer's FP8 query/scheduler/scale buffers and global indices.
"""

import os
import sys

import torch
from rope_append_env import load_dev_library  # noqa: F401  (re-exported for the tests)
from transformers import PretrainedConfig

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
    GlmMLA,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention_forward import (  # noqa: E501
    _forward_dsa_attn,
    forward_sparse_attn,
)
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plan_testlib  # noqa: E402

GLM_HIDDEN = 6144

HIDDEN = 1024  # run_seg does not exercise the projections
HEADS = 8
NOPE, ROPE, V, Q_LORA, KV_LORA = 192, 64, 256, 2048, 512
PATTERN = ["F", "S", "S", "S"]
MTP_LAYER = len(PATTERN)
TOKENS_PER_BLOCK = 64
CTX_REQ_ID = 9  # a request in its context phase (fallback cases)
CTX_REQ_TOKENS = 256


def set_features(ids: str) -> None:
    """Install, for this process, the plan that runs exactly the fast paths ``ids`` (comma-separated)."""
    plan_testlib.select(dispatch, ids)


class Engagement:
    """Counts native and rope_kv_append entry points so a silent native fallback cannot pass a comparison."""

    def __init__(self, capture: bool = False):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (  # noqa: E501
            rope_kv_append as rope_append_pkg,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.rope_kv_append import (  # noqa: E501
            ops as rope_ops,
        )
        from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
        from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
        from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

        self.counts = {}
        self._undo = []
        self.capture = capture
        # Per rope call: (quant_q, cu_q, cu_kv, scheduler counter, bmm1, bmm2); per remap: indices.
        self.rope_outputs = []
        self.global_indices = []
        self._wrap(Indexer, "_update_k_cache", "native_scatter")
        self._wrap(TrtllmAttention, "mla_rope_generation", "native_rope")
        self._wrap(DSATrtllmAttention, "_remap_topk_to_global", "native_remap")
        self._wrap_neighbours()
        self.bridge_checks = 0
        orig_bridge = rope_ops.topk_bridge
        counted_bridge = self._counted(orig_bridge, "rope_append_bridge_kernel")

        def checked_bridge(topk, metadata, num_rows, **kwargs):
            out = counted_bridge(topk, metadata, num_rows, **kwargs)
            self._check_bridge(topk, metadata, num_rows, out, **kwargs)
            return out

        rope_ops.topk_bridge = checked_bridge
        self._undo.append(lambda: setattr(rope_ops, "topk_bridge", orig_bridge))
        del rope_append_pkg
        registry = dispatch._REGISTRY["rope_kv_append"]
        for name in (
            "mla_rope_generation",
            "remap_topk_to_global",
            "run_indexer_k_scatter_native",
            "check_indexer_k_scatter_consumed",
        ):
            orig = registry[name]
            registry[name] = self._counted(orig, "rope_append_" + name)
            self._undo.append(lambda n=name, o=orig: registry.__setitem__(n, o))
        orig_defer = registry["defer_indexer_k_scatter"]

        def counted_defer(*args, **kwargs):
            pkt = orig_defer(*args, **kwargs)
            key = (
                "rope_append_deferred_scatter"
                if pkt is not None
                else "rope_append_scatter_not_deferred"
            )
            self.counts[key] = self.counts.get(key, 0) + 1
            return pkt

        registry["defer_indexer_k_scatter"] = counted_defer
        self._undo.append(lambda: registry.__setitem__("defer_indexer_k_scatter", orig_defer))

    def _counted(self, fn, key):
        def wrapper(*args, **kwargs):
            self.counts[key] = self.counts.get(key, 0) + 1
            if key == "rope_append_mla_rope_generation" and kwargs.get("indexer_k") is not None:
                self.counts["rope_append_fused_scatter"] = (
                    self.counts.get("rope_append_fused_scatter", 0) + 1
                )
            out = fn(*args, **kwargs)
            if self.capture and key in ("native_rope", "rope_append_mla_rope_generation"):
                # (mqa, fused_q, q_pe, latent, metadata, cu_q, cu_kv, counter, bmm1, bmm2, quant_q)
                md = args[4]
                n = md.num_seqs + 1
                cu_q, cu_kv, counter, bmm1, bmm2, quant_q = args[5:11]
                self.rope_outputs.append(
                    tuple(t.clone() for t in (quant_q, cu_q[:n], cu_kv[:n], counter, bmm1, bmm2))
                )
            if self.capture and key in ("native_remap", "rope_append_remap_topk_to_global"):
                self.global_indices.append(out.clone())
            return out

        return wrapper

    def _wrap_neighbours(self):
        """Count engagement of the neighbour packages present in this tree."""

        def wrap_registry(pkg, name, key, engaged):
            try:
                orig = dispatch.get_impl(pkg, name)
            except (ImportError, KeyError):
                return
            registry = dispatch._REGISTRY[pkg]

            def wrapper(*args, **kwargs):
                out = orig(*args, **kwargs)
                if engaged(out):
                    self.counts[key] = self.counts.get(key, 0) + 1
                return out

            registry[name] = wrapper
            self._undo.append(lambda: registry.__setitem__(name, orig))

        wrap_registry("absorbed_query", "dsa_proj", "aq_packet", lambda o: o is not None)
        wrap_registry(
            "absorbed_query", "shared_dsa_proj", "shared_query_packet", lambda o: o is not None
        )
        wrap_registry(
            "absorbed_query", "mtp_dsa_proj", "draft_query_packet", lambda o: o is not None
        )
        wrap_registry("indexer_topk", "indexer_decode", "indexer_topk_decode", lambda o: bool(o))
        try:
            fmha = dispatch.get_impl("sparse_mla_decode", "fmha_lib")
        except (ImportError, KeyError):
            fmha = None
        if fmha is not None:
            self._wrap(fmha, "forward", "sparse_mla_fmha")

    def _check_bridge(
        self,
        topk,
        metadata,
        num_rows,
        out,
        *,
        layer_offset=None,
        layer_ids=None,
        page_index_scale,
        copy=None,
    ):
        """Per-call oracle at the real call site: native remap and native copy of the same rows.

        Independent of which rows the (run-to-run nondeterministic) native
        top-k selected in this run.
        """
        torch.cuda.synchronize()
        tpb = metadata.kv_cache_manager.tokens_per_block
        nc = metadata.num_contexts
        block_table = metadata.block_table[nc : metadata.num_seqs]
        req = (metadata.req_idx_per_token[metadata.num_ctx_tokens : metadata.num_tokens] - nc)[
            :num_rows
        ]
        src = topk[:num_rows]
        stride = page_index_scale * tpb
        if layer_ids is not None:
            want = torch.ops.trtllm.convert_req_index_to_global_grouped(
                req.to(torch.int32).contiguous(),
                block_table,
                src,
                tpb,
                src.shape[1],
                stride,
                layer_ids,
            )
        else:
            want = torch.ops.trtllm.convert_req_index_to_global(
                req.to(torch.int32).contiguous(),
                block_table,
                src,
                tpb,
                src.shape[1],
                stride,
                layer_offset,
            )
        assert torch.equal(out, want), (
            "rope_kv_append remap differs from the native remap of the same rows"
        )
        if copy is not None:
            k = topk.shape[1]
            got = copy.dst[copy.row_start : copy.row_start + copy.rows, :k]
            if copy.mode == 0:
                exp = topk[: copy.rows]
            else:
                base = torch.arange(copy.rows, device=topk.device, dtype=torch.long) * copy.next_n
                if copy.num_accepted is None:
                    off = torch.full_like(base, copy.next_n - 1)
                else:
                    off = (copy.num_accepted[: copy.rows].long() - 1).clamp(0, copy.next_n - 1)
                exp = topk[base + off]
            assert torch.equal(got, exp), (
                f"rope_kv_append top-k copy (mode {copy.mode}) differs from native"
            )
        self.bridge_checks += 1

    def _wrap(self, owner, attr, key):
        orig = getattr(owner, attr)
        setattr(owner, attr, self._counted(orig, key))
        self._undo.append(lambda: setattr(owner, attr, orig))

    def close(self):
        for undo in reversed(self._undo):
            undo()


class Stack:
    def __init__(self, ctx_len: int, pattern=None, real_proj: bool = False):
        """``real_proj``: GLM hidden size 6144 and random projection weights for ``run_proj``."""
        pattern = list(PATTERN if pattern is None else pattern)
        self.pattern = pattern
        self.mtp_layer = len(pattern)
        self.hidden = GLM_HIDDEN if real_proj else HIDDEN
        dev = torch.device("cuda")
        self.dev = dev
        self.mapping = Mapping(world_size=1, tp_size=1, rank=0)
        self.pretrained = PretrainedConfig(
            hidden_size=self.hidden,
            num_attention_heads=HEADS,
            num_key_value_heads=HEADS,
            max_position_embeddings=1 << 20,
            rope_parameters={"rope_theta": 8_000_000, "rope_type": "default"},
            qk_rope_head_dim=ROPE,
            qk_nope_head_dim=NOPE,
            kv_lora_rank=KV_LORA,
            q_lora_rank=Q_LORA,
            v_head_dim=V,
            model_type="glm_moe_dsa",
            rms_norm_eps=1e-5,
            num_hidden_layers=len(pattern),
            index_topk_pattern=pattern,
            index_share_for_mtp_iteration=True,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
        )
        self.sparse = DeepSeekSparseAttentionConfig(
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
            enable_heuristic_topk=True,
            use_cute_dsl_paged_mqa_logits=True,
            use_cute_dsl_topk=True,
            skip_indexer_for_short_seqs=False,
        )
        self.model_config = ModelConfig(
            mapping=self.mapping,
            sparse_attention_config=self.sparse,
            pretrained_config=self.pretrained,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        )
        g = torch.Generator(device=dev).manual_seed(1234)
        self.layers = []
        for idx in range(len(pattern) + 1):
            mla = MLA(
                hidden_size=self.hidden,
                num_attention_heads=HEADS,
                num_key_value_heads=1,
                qk_nope_head_dim=NOPE,
                qk_rope_head_dim=ROPE,
                v_head_dim=V,
                q_lora_rank=Q_LORA,
                kv_lora_rank=KV_LORA,
                predicted_tokens_per_seq=6,
                max_position_embeddings=1 << 20,
                bias=False,
                pos_embd_params=PositionalEmbeddingParams(
                    type=PositionEmbeddingType.yarn,
                    rope=RopeParams.from_config(self.pretrained),
                    is_neox=False,
                ),
                layer_idx=idx,
                dtype=torch.bfloat16,
                config=self.model_config,
            )
            # The model's DeepseekV32Attention adds only its constructor to MLA; the stack builds
            # the MLA at GLM geometry directly and adopts it as the model's attention.
            mla.__class__ = DeepseekV32Attention
            GlmMLA.adopt(mla)
            for backend in (mla.mha, mla.mqa):
                if backend is not None and backend.prepended_fmha_lib_classes:
                    backend.update_quant_config(backend.quant_config)
            mla = mla.to(dev)
            with torch.no_grad():
                if real_proj:
                    for name, prm in mla.named_parameters():
                        if not prm.dtype.is_floating_point:
                            continue
                        if name.endswith("bias"):
                            prm.zero_()
                        elif "norm" in name:
                            prm.fill_(1.0)
                        else:
                            prm.normal_(0.0, 0.02, generator=g)
                k_nope = torch.randn(HEADS, NOPE, KV_LORA, generator=g, device=dev) * 0.05
                v = torch.randn(HEADS, V, KV_LORA, generator=g, device=dev) * 0.05
                mla.k_b_proj_trans.data = k_nope.transpose(1, 2).contiguous().to(torch.bfloat16)
                mla.v_b_proj.data = v.contiguous().to(torch.bfloat16)
                if mla.mqa.indexer is not None:
                    mla.mqa.indexer.to(dev)
                    for w in (
                        mla.mqa.indexer.wq_b.weight,
                        mla.mqa.indexer.wk.weight,
                        mla.mqa.indexer.weights_proj.weight,
                    ):
                        w.normal_(0.0, 0.02, generator=g)
                    mla.mqa.indexer.post_load_weights()
            self.layers.append(mla)
        assert [m.mqa.indexer is not None for m in self.layers] == [p == "F" for p in pattern] + [
            True
        ]

        # A second request with a different cache length for the two-request cases.
        self.ctx_b = ctx_len // 3 + 4099
        # Block-table width = max_seq_len / 64 pages; keep it even so indexer_topk (whose logits
        # tiles span two pages) stays inside its envelope at every context length.
        pages_per_tile = 2
        max_seq_len = -(-(ctx_len + 1024) // (pages_per_tile * TOKENS_PER_BLOCK)) * (
            pages_per_tile * TOKENS_PER_BLOCK
        )
        self.kvm = DSACacheManager(
            KvCacheConfig(
                max_tokens=ctx_len + self.ctx_b + 8192 + CTX_REQ_TOKENS + 256,
                enable_block_reuse=False,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=len(pattern) + 1,
            num_kv_heads=1,
            head_dim=KV_LORA + ROPE,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=3,
            mapping=self.mapping,
            dtype=tensorrt_llm.bindings.DataType.FP8,
            sparse_attention_config=self.sparse,
            model_config=self.model_config,
            pretrained_config=self.pretrained,
        )
        self.ctx_len = ctx_len
        self.rid = 7
        self.rids = [7, 8]
        self.ctxs = [ctx_len, self.ctx_b]
        self.reqs = self.kvm.add_dummy_requests(
            self.rids, token_nums=[c + 1 for c in self.ctxs], is_gen=True, max_num_draft_tokens=5
        )
        # FP8 context: valid E4M3 latents and indexer keys, positive indexer scales.
        pool = self.kvm.get_unique_primary_pool()
        flat = pool.view(torch.uint8).view(-1)
        chunk = flat.numel() if flat.numel() <= (1 << 31) else (1 << 30)
        for start in range(0, flat.numel(), chunk):
            n = min(chunk, flat.numel() - start)
            flat[start : start + n].copy_(
                (torch.randn(n, generator=g, device=dev) * 0.5)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        for ipool in self.kvm.indexer_k_cache_pool_per_layer:
            if ipool is None:
                continue
            blocks = ipool.view(torch.uint8).view(self.kvm.num_blocks, -1)
            data = TOKENS_PER_BLOCK * 128
            blocks[:, :data].copy_(
                (torch.randn(blocks.shape[0], data, generator=g, device=dev))
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            blocks[:, data:].view(torch.float32).copy_(
                torch.rand(blocks.shape[0], TOKENS_PER_BLOCK, generator=g, device=dev) + 0.05
            )
        self.attn_cls = get_attention_backend(
            "TRTLLM", sparse_params=self.sparse.to_sparse_params(pretrained_config=self.pretrained)
        )

    def cached(self, cached: int, nreq: int):
        """Cached lengths of the first ``nreq`` requests; ``cached`` is request 0's."""
        return [c + (cached - self.ctx_len) for c in self.ctxs[:nreq]]

    def metadata(
        self, rows: int, cached: int, nreq: int = 1, ctx_tokens: int = 0, ragged_lens=None
    ):
        """Generation metadata for ``nreq`` requests (``rows`` query tokens each).

        ``ctx_tokens`` > 0 prepends the context request ``CTX_REQ_ID`` with that
        many new tokens (mixed batch; ``nreq`` = 0 gives a pure context batch).
        ``ragged_lens`` installs ragged verification with these per-request
        query lengths.
        """
        nctx = 1 if ctx_tokens else 0
        gen_lens = list(ragged_lens) if ragged_lens is not None else [rows] * nreq
        kwargs = {}
        if ragged_lens is not None:
            kwargs.update(enable_ragged_verification=True)
        md = self.attn_cls.Metadata(
            seq_lens=torch.tensor([ctx_tokens] * nctx + gen_lens, dtype=torch.int),
            request_ids=[CTX_REQ_ID] * nctx + self.rids[:nreq],
            max_num_requests=nctx + nreq,
            num_contexts=nctx,
            prompt_lens=[ctx_tokens] * nctx + self.ctxs[:nreq],
            max_num_tokens=128 + ctx_tokens,
            kv_cache_manager=self.kvm,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[0] * nctx + self.cached(cached, nreq)
            ),
            mapping=self.mapping,
            sparse_metadata_params=self.sparse.to_sparse_metadata_params(
                pretrained_config=self.pretrained
            ),
            **kwargs,
        )
        if ragged_lens is not None:
            md.ragged_verify_lens = list(ragged_lens)
        md.update_spec_dec_param(
            batch_size=nctx + nreq,
            is_spec_decoding_enabled=True,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=5,
            max_total_draft_tokens=5,
        )
        md.prepare()
        return md

    def add_context_request(self):
        """Allocate the context-phase request used by the mixed / context fallback cases."""
        if not hasattr(self, "ctx_req"):
            self.ctx_req = self.kvm.add_dummy_requests(
                [CTX_REQ_ID], token_nums=[CTX_REQ_TOKENS], is_gen=False
            )
            self.reqs = list(self.reqs) + list(self.ctx_req)

    def inputs(self, rows: int, cached: int, seed: int, nreq: int = 1):
        return self.inputs_seg([(rows, c) for c in self.cached(cached, nreq)], seed)

    def inputs_seg(self, segs, seed: int):
        """Random layer inputs for request-major segments ``(rows, first_position)``."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        n = sum(r for r, _ in segs)
        q = (torch.randn(n, HEADS * (NOPE + ROPE), generator=g, device=self.dev)).to(torch.bfloat16)
        ckv = torch.randn(n, KV_LORA, generator=g, device=self.dev).to(torch.bfloat16)
        k_pe = torch.randn(n, ROPE, generator=g, device=self.dev).to(torch.bfloat16)
        qr = torch.randn(n, Q_LORA, generator=g, device=self.dev).to(torch.bfloat16)
        hidden = torch.randn(n, self.hidden, generator=g, device=self.dev).to(torch.bfloat16)
        pos = torch.cat(
            [torch.arange(p, p + r, dtype=torch.int32, device=self.dev) for r, p in segs]
        ).unsqueeze(0)
        return q, ckv, k_pe, qr, hidden, pos

    def state(self, md):
        out = [p.view(torch.uint8).clone() for p in [self.kvm.get_unique_primary_pool()]]
        out += [
            p.view(torch.uint8).clone()
            for p in self.kvm.indexer_k_cache_pool_per_layer
            if p is not None
        ]
        if md.shared_topk_indices is not None:
            out.append(md.shared_topk_indices.clone())
        return out

    def restore(self, md, state):
        pools = [self.kvm.get_unique_primary_pool()] + [
            p for p in self.kvm.indexer_k_cache_pool_per_layer if p is not None
        ]
        for p, s in zip(pools, state):
            p.view(torch.uint8).copy_(s)
        if md.shared_topk_indices is not None:
            md.shared_topk_indices.copy_(state[-1])
        md._group_remap_batched.clear()

    @torch.inference_mode()
    def run_layers(self, md, layer_ids, rows, cached, seed, nreq: int = 1):
        return self.run_seg(md, layer_ids, [(rows, c) for c in self.cached(cached, nreq)], seed)

    @torch.inference_mode()
    def run_seg(self, md, layer_ids, segs, seed, latent_row_stride: int = 0):
        """Run ``_forward_dsa_attn`` of ``layer_ids`` on request-major segments.

        ``latent_row_stride`` > 576 passes ``latent_cache`` as a row-strided
        view (outside the fused rope domain).
        """
        q, ckv, k_pe, qr, hidden, pos = self.inputs_seg(segs, seed)
        n = q.shape[0]
        outs = []
        for idx in layer_ids:
            mla = self.layers[idx]
            inter = []
            if mla.mqa.indexer is not None:
                inter = list(mla.mqa.indexer.pre_indexer_proj(qr, hidden, pos))
            latent = torch.cat([ckv, k_pe], dim=-1)
            if latent_row_stride:
                buf = torch.zeros(n, latent_row_stride, dtype=latent.dtype, device=self.dev)
                buf[:, : KV_LORA + ROPE].copy_(latent)
                latent = buf[:, : KV_LORA + ROPE]
            out = torch.empty(n, HEADS * V, dtype=torch.bfloat16, device=self.dev)
            with dispatch.step(dispatch.Phase.DECODE):
                _forward_dsa_attn(
                    mla, q.clone(), ckv.clone(), k_pe.clone(), latent, inter, pos, md, out
                )
            outs.append(out)
        torch.cuda.synchronize()
        return outs

    @torch.inference_mode()
    def run_proj(self, md, layer_ids, segs, seed):
        """Run ``forward_sparse_attn`` (projections + attention) of ``layer_ids`` from hidden states."""
        _, _, _, _, hidden, pos = self.inputs_seg(segs, seed)
        outs = []
        for idx in layer_ids:
            out = torch.empty(hidden.shape[0], HEADS * V, dtype=torch.bfloat16, device=self.dev)
            with dispatch.step(dispatch.Phase.DECODE):
                forward_sparse_attn(self.layers[idx], pos, hidden, md, [out])
            outs.append(out)
        torch.cuda.synchronize()
        return outs

    @torch.inference_mode()
    def indexer_k(self, mla, rows, cached, seed, nreq=1):
        """Indexer K inputs of one layer call, as ``_forward_dsa_attn`` slices them."""
        _, _, _, qr, hidden, pos = self.inputs(rows, cached, seed, nreq)
        _, k_fp8, k_scale, _, _ = mla.mqa.indexer.pre_indexer_proj(qr, hidden, pos)
        return k_fp8[: rows * nreq], k_scale[: rows * nreq]

    @staticmethod
    def _topk_set_diff(a, b):
        """Per-row number of entries in one selection but not the other (order ignored)."""
        diffs = []
        for ra, rb in zip(a, b):
            sa, sb = set(ra.tolist()), set(rb.tolist())
            diffs.append(len(sa ^ sb))
        return diffs

    @staticmethod
    def _state_equal(i, a, b, is_topk):
        # The native GVR top-k emits its selection in a run-to-run varying order,
        # so the top-k reuse buffer is compared as a per-row set.
        if is_topk:
            return torch.equal(torch.sort(a, dim=-1).values, torch.sort(b, dim=-1).values)
        return torch.equal(a, b)

    def _compare_captures(self):
        """Per-layer FP8 query / scheduler / scale buffers and global indices vs the rope_kv_append-off runs.

        A buffer whose two rope_kv_append-off runs agree must be byte-identical with rope_kv_append;
        buffers that differ between the rope_kv_append-off runs (selection-dependent
        global indices under a nondeterministic top-k) are counted and
        reported instead.
        """
        (rope_a, idx_a), (rope_b, idx_b), (rope_w, idx_w) = self.captures
        assert len(rope_a) == len(rope_b) == len(rope_w) > 0, (len(rope_a), len(rope_w))
        assert len(idx_a) == len(idx_b) == len(idx_w) > 0, (len(idx_a), len(idx_w))
        names = ("quant_q", "cu_q", "cu_kv", "counter", "bmm1", "bmm2")
        report = {"rope_exact": 0, "rope_native_nondet": 0, "idx_exact": 0, "idx_native_nondet": 0}
        for i, (a, b, w) in enumerate(zip(rope_a, rope_b, rope_w)):
            for name, x, y, z in zip(names, a, b, w):
                xb, yb, zb = (t.view(torch.uint8) for t in (x, y, z))
                if torch.equal(xb, yb):
                    assert torch.equal(xb, zb), (
                        f"rope call {i}: {name} differs from the rope_kv_append-off run"
                    )
                    report["rope_exact"] += 1
                else:
                    report["rope_native_nondet"] += 1
        # The native top-k emits each row's selection in a run-to-run varying order, so global
        # indices are compared per row as sets when the two rope_kv_append-off runs differ only in
        # order.
        report.update(idx_set_exact=0)
        for i, (a, b, w) in enumerate(zip(idx_a, idx_b, idx_w)):
            if torch.equal(a, b):
                assert torch.equal(a, w), (
                    f"remap call {i}: global indices differ from the rope_kv_append-off run"
                )
                report["idx_exact"] += 1
                continue
            sa, sb, sw = (torch.sort(t, dim=-1).values for t in (a, b, w))
            if torch.equal(sa, sb):
                assert torch.equal(sa, sw), (
                    f"remap call {i}: global index sets differ from rope_kv_append-off"
                )
                report["idx_set_exact"] += 1
            else:
                report["idx_native_nondet"] += 1
        print("capture comparison:", report)
        self.capture_report = report

    def compare(self, md, step, base: str = "", capture: bool = False):
        """Run ``step`` twice with features ``base`` and once with ``base`` + rope_kv_append from the same state.

        Always: cache pools byte-identical to native, and every rope_kv_append bridge call
        equal to the native remap/copy of the rows it was given (per-call
        oracle). When the two native runs select the same top-k sets, rope_kv_append must
        select the same sets too and its sparse-MLA output must be exact (or
        within the native run-to-run spread). When the native selection itself
        differs between runs (characterized and printed), the selection-
        dependent outputs cannot be compared run to run; the per-call oracle
        and the pools still are.
        """
        state0 = self.state(md)
        runs = []
        self.engagement = []
        self.bridge_checks = []
        self.captures = []
        rope_append_features = ",".join([f for f in base.split(",") if f] + ["rope_kv_append"])
        for features in (base, base, rope_append_features):
            self.restore(md, state0)
            set_features(features)
            eng = Engagement(capture=capture)
            try:
                outs = step()
            finally:
                eng.close()
            self.engagement.append(eng.counts)
            self.bridge_checks.append(eng.bridge_checks)
            self.captures.append((eng.rope_outputs, eng.global_indices))
            runs.append((outs, self.state(md)))
        set_features("")
        if capture:
            self._compare_captures()
        print("engagement native/native/rope_kv_append:", self.engagement)
        assert self.bridge_checks[2] == self.engagement[2].get("rope_append_bridge_kernel", 0)
        (out_n, st_n), (out_n2, st_n2), (out_w, st_w) = runs
        topk_idx = len(st_n) - 1 if md.shared_topk_indices is not None else -1
        for i, (a, b, c) in enumerate(zip(st_n, st_n2, st_w)):
            if i == topk_idx:
                continue
            assert torch.equal(a, b), f"native cache state {i} is not deterministic"
            if not torch.equal(a, c):
                raise AssertionError(f"cache state {i}: {(a != c).sum().item()} bytes differ")
        assert any(not torch.equal(a, b) for a, b in zip(state0, st_n)), "native step wrote nothing"
        self.native_topk_diff = None
        if topk_idx >= 0:
            self.native_topk_diff = self._topk_set_diff(st_n[topk_idx], st_n2[topk_idx])
            if any(self.native_topk_diff):
                nz = [(r, d) for r, d in enumerate(self.native_topk_diff) if d]
                print(
                    f"native top-k selection differs between native runs (row, |symmetric diff|): {nz}"
                )
                return None
            rope_append_diff = self._topk_set_diff(st_n[topk_idx], st_w[topk_idx])
            assert not any(rope_append_diff), (
                f"rope_kv_append run selected different top-k sets: {rope_append_diff}"
            )
        report = []
        for i, (a, a2, b) in enumerate(zip(out_n, out_n2, out_w)):
            native_spread = (a.float() - a2.float()).abs().max().item()
            rope_append_diff = (a.float() - b.float()).abs().max().item()
            rope_append_diff2 = (a2.float() - b.float()).abs().max().item()
            report.append((native_spread, min(rope_append_diff, rope_append_diff2)))
            if native_spread == 0.0:
                assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"output {i} differs"
            else:
                assert min(rope_append_diff, rope_append_diff2) <= 2 * native_spread, (
                    i,
                    native_spread,
                    rope_append_diff,
                )
        print("output native-spread / rope_kv_append-diff per layer:", report)
        return report

    def shutdown(self):
        for r in self.reqs:
            self.kvm.free_resources(r)
        self.kvm.shutdown()
