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
"""draft_absorbed_query inside the MTP Eagle draft loop, through the target's attention and worker.

Modules: the built-in ``DeepseekV32Attention`` (adopted as the target's ``GlmMLA``) of the
full-indexer target
layer 6 (stand-in for the target verify forward) and of the MTP layer 78,
both with real GLM-5.2 weights at TP8 rank 0; FP8 KV cache (V1
``DSACacheManager`` holding layers 6 and 78); synthetic activations. Two
requests with a 2,600-token context are prefilled natively through both
layers.

One decode iteration runs as the GEN CUDA graph runs it, as a decode step:

1. ``on_update_kv_lens()`` (the runner's ``_preprocess_inputs``);
2. target verify: layer 6 on ``B x 6`` rows (MTP verify layout);
3. ``_prepare_attn_metadata_for_spec_dec``, then the draft loop of the target's
   adopted worker (``GlmMtpWorker._forward_linear_draft_loop`` on an ``MTPEagleWorker``),
   then ``_restore_attn_metadata_from_spec_dec``. The loop is the real
   one: it marks the draft loop and the accepted counts for the indexer,
   calls ``set_skip_topk(i > 0)``, gathers each request's last accepted row,
   advances positions, sets ``seq_lens`` to 1 and updates ``kv_lens_cuda``.
   Its ``draft_model`` is a stub whose ``mtp_layers[0]`` runs the native
   layer-78 attention (``MLA.forward``'s custom-op path, or ``forward_impl``)
   on the rows the loop hands it and returns an attention-independent next
   hidden state (a fixed column rotation), and whose ``shared_head`` returns
   ``hidden[:, :1024]`` as logits. So draft step 0 runs layer 78 on
   ``B x 6`` rows and steps 1-4 on ``B`` rows, exactly the draft-loop shapes.

Asserted:

* gating through the target's ``_absorbed_query_packet``: ``draft_absorbed_query`` engages only layer
  78 (all five draft steps, rows ``[6B, B, B, B, B]``), ``absorbed_query`` only layer 6,
  ``absorbed_query,draft_absorbed_query`` both, none of them nothing; layers that are not engaged
  produce caches and attention inputs byte-identical to native;
* indexer reuse semantics: in every run (native and draft_absorbed_query) draft step 0 scores
  the top-k (``in_mtp_draft_loop`` set, ``indexer_skip_topk`` clear) and
  steps 1-4 reuse it (both set), and the top-k of steps 1-4 equals, row for
  row and in order, the step-0 top-k row of each request's last accepted
  token;
* cache writes: per draft step the indexer keys handed to
  ``_update_k_cache`` (``k_fp8``, ``k_scale``) and, after the iteration,
  every cache (latent and indexer-key caches of layers 6 and 78) are
  bit-identical to native, and exactly the same slots changed;
* MLA inputs per call (absorbed query, FP8 ``quant_q_buffer``, BMM scales)
  and indexer inputs per call (FP8 query, head weights) bit-identical to
  native; the top-k and sparse-index key sets of every call equal native's;
* attention outputs per call within the native run-to-run noise bound (as
  in the absorbed_query integration test);
* determinism: a repeated draft_absorbed_query iteration reproduces caches and recorded inputs
  bit for bit;
* CUDA graph: the whole iteration (target layer + five draft steps) captured
  in one graph, replayed 1, 1 and 20 times with fresh hidden states and
  fresh accepted counts, reproduces the eager draft_absorbed_query iteration bit for bit and
  compares with native as above; a captured native graph reproduces eager
  native;
* a negative control (zeroed absorbed query on layer 78) is detected.

Every draft step launches the full fused-indexer kernel and writes every
packet output, including the indexer query inputs that the reusing steps 1-4
do not read.
"""

import inspect
import os
import sys
from types import SimpleNamespace

import absorbed_query_fixture
import pytest
import test_absorbed_query_attention_integration as aq_integration
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import plan_testlib  # noqa: E402

TARGET = absorbed_query_fixture.FULL_LAYER
MTP = 78
LAYERS = (TARGET, MTP)
CTX = 2600
MAX_REQS = 2
DRAFT_LEN = 5
VERIFY_ROWS = DRAFT_LEN + 1
STUB_VOCAB = 1024
# Native iterations per comparison. The output bound is twice the largest
# native repeat difference, so it needs enough repeats to sample the native
# sparse attention's run-to-run spread (graph replays included).
NATIVE_RUNS = int(os.environ.get("GLM_MOE_DSA_DRAFT_ABSORBED_QUERY_IT_NATIVE_RUNS", "9"))
# Accepted counts per request (including the target-sampled token, 1..6).
ACCEPTED = {1: [3], 2: [6, 1]}


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]


def _set_features(features: str, form: str = "auto"):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    query_proj.SERVED_FORM = form
    plan_testlib.select(dispatch, absorbed_query_fixture.with_base_features(features))


class _LaunchLog:
    """Records ``(layer, rows)`` of every fused-indexer absorbed_query launch."""

    def __enter__(self):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
            query_proj,
        )

        self.cls = query_proj.AbsorbedQueryProjection
        self.orig = self.cls.launch_indexer
        self.calls = []
        log, orig = self.calls, self.orig

        def launch_indexer(runner, kva, *args, **kwargs):
            log.append((runner.mla.layer_idx, kva.shape[0]))
            return orig(runner, kva, *args, **kwargs)

        self.cls.launch_indexer = launch_indexer
        return self

    def __exit__(self, *exc):
        self.cls.launch_indexer = self.orig
        return False


class _Recorder:
    """Per-call records of what the native attention consumes, per layer:
    lists indexed by call (one call for layer 6, five for layer 78).

    Wraps ``mla_rope_generation`` (absorbed query, ``quant_q_buffer``, BMM
    scales), ``sparse_attn_predict`` (converted indices), the indexer's
    ``_update_k_cache`` (keys appended to the indexer cache) and
    ``sparse_attn_indexer`` (logical top-k, indexer query inputs and the
    draft-loop flags it sees). Inside a capture the clones are graph-pool
    tensors that every replay refreshes; the flags are host values seen at
    capture."""

    def __init__(self, attn_by_layer):
        self.attn = attn_by_layer
        self.data = {layer: {} for layer in attn_by_layer}
        self._installed = []

    def _wrap(self, obj, name, post):
        orig = getattr(obj, name)
        sig = inspect.signature(orig)

        def hook(*args, **kwargs):
            out = orig(*args, **kwargs)
            post(sig.bind(*args, **kwargs).arguments, out)
            return out

        setattr(obj, name, hook)
        self._installed.append((obj, name))

    def __enter__(self):
        for layer, attn in self.attn.items():
            rec = self.data[layer]

            def add(key, value, rec=rec):
                rec.setdefault(key, []).append(value)

            def rope_post(a, _out, add=add, kv_lora=attn.kv_lora_rank):
                add("absorbed_q", a["fused_q"][..., :kv_lora].clone())
                for key, arg in (
                    ("quant_q", "quant_q_buffer"),
                    ("bmm1_scale", "mla_bmm1_scale"),
                    ("bmm2_scale", "mla_bmm2_scale"),
                ):
                    if a.get(arg) is not None:
                        add(key, a[arg].clone())

            def predict_post(_a, out, add=add):
                if isinstance(out, tuple) and out[0] is not None:
                    add("sparse_indices", out[0].clone())

            def kcache_post(a, _out, add=add):
                add("index_k_fp8", a["k_fp8"].clone())
                add("index_k_scale", a["k_scale"].clone())

            def topk_post(a, out, add=add):
                meta = a["metadata"]
                add("topk", out.clone())
                add("indexer_q_fp8", a["q_fp8"].clone())
                add("indexer_weights", a["weights"].clone())
                add("flags", (bool(meta.in_mtp_draft_loop), bool(meta.indexer_skip_topk)))

            self._wrap(attn.mqa, "mla_rope_generation", rope_post)
            self._wrap(attn.mqa, "sparse_attn_predict", predict_post)
            self._wrap(attn.mqa.indexer, "_update_k_cache", kcache_post)
            self._wrap(attn.mqa.indexer, "sparse_attn_indexer", topk_post)
        return self

    def __exit__(self, *exc):
        for obj, name in self._installed:
            delattr(obj, name)
        self._installed.clear()
        return False

    def snapshot(self):
        return {
            layer: {
                key: [v.clone() if torch.is_tensor(v) else v for v in vals]
                for key, vals in rec.items()
            }
            for layer, rec in self.data.items()
        }


class _Engagement:
    """Counts the absorbed_query package entry points the absorbed-query packet calls:
    ``(entry, layer, rows, engaged)`` per call."""

    def __enter__(self):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (  # noqa: E501, F401
            absorbed_query,
        )

        self.registry = dispatch._REGISTRY["absorbed_query"]
        self.saved = dict(self.registry)
        self.calls = []
        for name in ("dsa_proj", "mtp_dsa_proj"):
            fn = self.saved[name]

            def wrapped(mla, position_ids, hidden_states, attn_metadata, fn=fn, name=name):
                packet = fn(mla, position_ids, hidden_states, attn_metadata)
                self.calls.append((name, mla.layer_idx, hidden_states.shape[0], packet is not None))
                return packet

            self.registry[name] = wrapped
        return self

    def __exit__(self, *exc):
        self.registry.update(self.saved)
        return False


class _StubShared:
    def __call__(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        return hidden_states[:, :STUB_VOCAB].float()


class _StubMTPLayer:
    """``mtp_layers[0]``: the native layer-78 attention on the loop's rows;
    the next hidden state does not depend on the attention output, so native
    and draft_absorbed_query runs see identical inputs at every draft step."""

    def __init__(self, harness, path, outputs):
        self.h, self.path, self.outputs = harness, path, outputs
        self.shared_head = _StubShared()

    def __call__(
        self,
        input_ids,
        position_ids,
        hidden_states,
        embed_tokens,
        attn_metadata,
        all_rank_num_tokens=None,
        spec_metadata=None,
        **kwargs,
    ):
        out = self.h.forward(MTP, position_ids, hidden_states, attn_metadata, self.path)
        self.outputs.append(out.clone())
        return torch.roll(hidden_states, shifts=97, dims=-1)


class _StubDraftModel:
    def __init__(self, mtp_layer):
        self.mtp_layers = [mtp_layer]
        self.embed_tokens = None
        self.lm_head = None


class _Harness:
    def __init__(self):
        import tensorrt_llm.bindings
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.spec_worker import (  # noqa: E501
            GlmMtpWorker,
        )
        from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
        from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
        from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
        from tensorrt_llm.mapping import Mapping
        from tensorrt_llm.models.modeling_utils import QuantConfig
        from tensorrt_llm.quantization.mode import QuantAlgo

        self.KVCacheParams = KVCacheParams
        self.model_config = absorbed_query_fixture.build_model_config(0)
        pretrained = self.model_config.pretrained_config
        sparse = self.model_config.sparse_attention_config
        self.attn = {
            layer: absorbed_query_fixture.build_native_attention(layer, 0, self.model_config)
            for layer in LAYERS
        }
        assert self.attn[MTP].mqa.indexer.mtp_index_share
        self.mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_model_config = ModelConfig(
            mapping=self.mapping,
            sparse_attention_config=sparse,
            pretrained_config=pretrained,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        )
        self.kvm = DSACacheManager(
            KvCacheConfig(
                dtype="fp8",
                max_tokens=MAX_REQS * (CTX + 128),
                enable_block_reuse=False,
                event_buffer_max_size=0,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=len(LAYERS),
            layer_mask=[i in LAYERS for i in range(MTP + 1)],
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=64,
            max_seq_len=CTX + 128,
            max_batch_size=MAX_REQS,
            mapping=self.mapping,
            dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
            sparse_attention_config=sparse,
            model_config=kv_model_config,
            pretrained_config=pretrained,
            vocab_size=pretrained.vocab_size,
            max_num_tokens=MAX_REQS * CTX,
        )
        self.req_ids = list(range(MAX_REQS))
        for rid in self.req_ids:
            # CTX + 10 tokens (6 verify slots, up to 4 more draft slots) fit
            # the blocks of CTX + 1 tokens (64 per block).
            self.kvm.add_dummy_requests(
                request_ids=[rid], token_nums=[CTX + 1], is_gen=False, prepare_resource=True
            )
        sparse_params = sparse.to_sparse_params(pretrained_config=pretrained)
        self.Metadata = get_attention_backend("TRTLLM", sparse_params=sparse_params).Metadata
        self.meta_params = sparse.to_sparse_metadata_params(pretrained_config=pretrained)
        assert self.meta_params.mtp_index_share
        self.gamma = {layer: absorbed_query_fixture.load_input_layernorm(layer) for layer in LAYERS}
        self.worker = GlmMtpWorker.adopt(
            MTPEagleWorker(MTPDecodingConfig(max_draft_len=5), mapping=self.mapping)
        )
        assert self.worker.is_mtp_eagle
        self.batch_indices = torch.arange(MAX_REQS, dtype=torch.int32, device="cuda")

        # Native prefill of both requests through both layers.
        _set_features("")
        hidden = absorbed_query_fixture.make_hidden(
            MAX_REQS * CTX, seed=11, gamma=self.gamma[TARGET]
        )
        positions = (
            torch.arange(CTX, dtype=torch.int32, device="cuda").repeat(MAX_REQS).unsqueeze(0)
        )
        ctx_meta = self.Metadata(
            seq_lens=torch.tensor([CTX] * MAX_REQS, dtype=torch.int),
            request_ids=self.req_ids,
            max_num_requests=MAX_REQS,
            num_contexts=MAX_REQS,
            prompt_lens=[CTX] * MAX_REQS,
            max_num_tokens=MAX_REQS * CTX,
            kv_cache_manager=self.kvm,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * MAX_REQS),
            mapping=self.mapping,
            sparse_metadata_params=self.meta_params,
        )
        ctx_meta.prepare()
        ctx_meta.on_update_kv_lens()
        for layer in LAYERS:
            self.forward(layer, positions, hidden, ctx_meta, "forward_impl")
        torch.cuda.synchronize()

    # -- inputs and metadata -----------------------------------------------

    def iteration_inputs(self, nreq, seed, accepted=None):
        rows = nreq * VERIFY_ROWS
        accepted = ACCEPTED[nreq] if accepted is None else accepted
        return SimpleNamespace(
            hidden_t=absorbed_query_fixture.make_hidden(rows, seed=seed, gamma=self.gamma[TARGET]),
            pos_t=(CTX + torch.arange(VERIFY_ROWS, dtype=torch.int32, device="cuda"))
            .repeat(nreq)
            .unsqueeze(0),
            hidden_d=absorbed_query_fixture.make_hidden(rows, seed=seed + 1, gamma=self.gamma[MTP]),
            num_accepted=torch.tensor(accepted, dtype=torch.int32, device="cuda"),
            input_ids=torch.arange(rows, dtype=torch.int32, device="cuda"),
        )

    def gen_metadata(self, nreq, cuda_graph=False):
        """Verify-step metadata as the runner builds it (see the absorbed_query
        integration test): ``nreq`` requests x 6 tokens, MTP verify width
        from ``update_spec_dec_param``, then ``prepare()``."""
        req_ids = self.req_ids[:nreq]
        kv_params = self.KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[CTX] * nreq)
        meta = self.Metadata(
            seq_lens=torch.tensor([VERIFY_ROWS] * nreq, dtype=torch.int),
            request_ids=req_ids,
            max_num_requests=nreq,
            num_contexts=0,
            prompt_lens=[CTX] * nreq,
            max_num_tokens=nreq * VERIFY_ROWS,
            kv_cache_manager=self.kvm,
            kv_cache_params=kv_params,
            mapping=self.mapping,
            sparse_metadata_params=self.meta_params,
        )
        if cuda_graph:
            from tensorrt_llm._torch.memory_buffer_utils import Buffers

            meta = meta.create_cuda_graph_metadata(nreq, False, DRAFT_LEN, Buffers())
            assert meta.is_cuda_graph
            meta.request_ids = req_ids
            meta.prompt_lens = [CTX] * nreq
            meta.num_contexts = 0
            meta.kv_cache_params = kv_params
        meta.update_spec_dec_param(
            batch_size=nreq,
            is_spec_decoding_enabled=True,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=DRAFT_LEN,
            max_total_draft_tokens=DRAFT_LEN,
        )
        meta.prepare()
        return meta

    # -- forwards ----------------------------------------------------------

    def forward(self, layer, positions, hidden, meta, path):
        attn = self.attn[layer]
        if path == "forward_impl":
            out = attn._create_outputs(hidden, meta)
            attn.forward_impl(positions, hidden, meta, attn_output=out)
            return out[0]
        import weakref

        from tensorrt_llm._torch.utils import model_extra_attrs

        extra = self.model_config.extra_attrs
        extra["attention_metadata"] = weakref.ref(meta)
        with model_extra_attrs(extra):
            out = [torch.ops.trtllm.create_mla_outputs(hidden, attn.layer_idx_str)]
            attn._forward_custom_op(hidden, positions, out, None)
        return out[0]

    def iteration(self, meta, it, path):
        """Target verify (layer 6) and the draft loop (layer 78), as the GEN graph
        runs one decode iteration (a decode step). Returns the attention
        outputs per layer and call."""
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
            dispatch,
        )

        with dispatch.step(dispatch.Phase.DECODE):
            return self._iteration(meta, it, path)

    def _iteration(self, meta, it, path):
        nreq = it.num_accepted.shape[0]
        meta.on_update_kv_lens()
        outs = {TARGET: [], MTP: []}
        outs[TARGET].append(self.forward(TARGET, it.pos_t, it.hidden_t, meta, path).clone())
        draft_model = _StubDraftModel(_StubMTPLayer(self, path, outs[MTP]))
        spec_metadata = SimpleNamespace(
            runtime_draft_len=DRAFT_LEN,
            batch_indices_cuda=self.batch_indices[:nreq],
            all_rank_num_tokens=None,
            subseq_all_rank_num_tokens=None,
            wants_advanced_draft_sampling=False,
            max_num_requests=nreq,
        )
        self.worker._prepare_attn_metadata_for_spec_dec(meta)
        try:
            inputs = {
                "input_ids": it.input_ids,
                "position_ids": it.pos_t.squeeze(0),
                "hidden_states": it.hidden_d,
                "attn_metadata": meta,
                "spec_metadata": spec_metadata,
            }
            outs["tokens"] = self.worker._forward_linear_draft_loop(
                inputs, meta, spec_metadata, draft_model, None, 0, nreq, it.num_accepted, None
            )
        finally:
            self.worker._restore_attn_metadata_from_spec_dec(meta)
            meta.use_spec_decoding = True
        return outs

    def run(self, nreq, it, path, meta=None):
        """One eager iteration with the current features: outputs, records,
        entry-point engagement."""
        meta = meta if meta is not None else self.gen_metadata(nreq)
        with _Recorder(self.attn) as rec, _Engagement() as eng:
            outs = self.iteration(meta, it, path)
        torch.cuda.synchronize()
        return outs, rec.snapshot(), eng.calls

    # -- caches ------------------------------------------------------------

    def caches(self):
        bufs = {}
        for layer in LAYERS:
            bufs[f"latent{layer}"] = self.kvm.get_buffers(layer)
            bufs[f"index_k{layer}"] = self.kvm.get_indexer_k_cache_buffers(layer)
        return bufs

    def snapshot(self):
        return {k: v.clone() for k, v in self.caches().items()}

    def restore(self, snap):
        for k, v in self.caches().items():
            v.copy_(snap[k])
        torch.cuda.synchronize()

    def reset_runners(self):
        for attn in self.attn.values():
            attn._absorbed_query = None
            attn._draft_absorbed_query = None


@pytest.fixture(scope="module")
def harness():
    with torch.inference_mode():
        h = _Harness()
    yield h
    _set_features("")


# -- comparison helpers -----------------------------------------------------


def _expected_slots(nreq, accepted):
    """Changed token slots per layer after one iteration: six verify slots
    per request on both layers; on the MTP layer steps 1-4 append at
    positions ``accepted .. accepted + 3`` after the context."""
    mtp = sum(len(set(range(VERIFY_ROWS)) | set(range(a, a + DRAFT_LEN - 1))) for a in accepted)
    return {TARGET: nreq * VERIFY_ROWS, MTP: mtp}


def _compare_caches(snap, ref_cache, cand_cache, expected, report, tag):
    """Changed slots equal the reference's (and the expected count) and
    every byte equals the reference's."""
    fails = []
    for key in cand_cache:
        ref, cand = ref_cache[key], cand_cache[key]
        layer = int(key.replace("latent", "").replace("index_k", ""))
        n_diff = int(aq_integration._changed_bytes(ref, cand).sum())
        report[f"{tag}:{key}_bytes_differing"] = n_diff
        if key.startswith("latent"):
            slots = aq_integration._changed_latent_slots(snap[key], cand)
            ref_slots = aq_integration._changed_latent_slots(snap[key], ref)
        else:
            slots = aq_integration._changed_index_k_tokens(snap[key], cand)
            ref_slots = aq_integration._changed_index_k_tokens(snap[key], ref)
        if slots != ref_slots or len(ref_slots) != expected[layer]:
            fails.append(
                f"{tag}: {key} changed {len(slots)} slots vs "
                f"reference {len(ref_slots)} (expected "
                f"{expected[layer]})"
            )
        if n_diff:
            fails.append(f"{tag}: {key} differs in {n_diff} bytes")
    return fails


def _check_reuse(rec, accepted, report, tag):
    """Indexer reuse semantics of one run: layer 6 outside the draft loop;
    layer 78 scores at step 0 and reuses at steps 1-4, the reused rows being
    the step-0 rows of each request's last accepted token, in order."""
    fails = []
    if rec[TARGET]["flags"] != [(False, False)]:
        fails.append(f"{tag}: target flags {rec[TARGET]['flags']}")
    flags = rec[MTP]["flags"]
    want = [(True, False)] + [(True, True)] * (DRAFT_LEN - 1)
    if flags != want:
        fails.append(f"{tag}: draft-loop flags {flags}, expected {want}")
    topk = rec[MTP]["topk"]
    if len(topk) != DRAFT_LEN:
        return fails + [f"{tag}: {len(topk)} layer-78 top-k calls"]
    rows = torch.tensor(
        [r * VERIFY_ROWS + min(max(a - 1, 0), DRAFT_LEN) for r, a in enumerate(accepted)],
        device=topk[0].device,
    )
    stash = topk[0][rows]
    for step in range(1, DRAFT_LEN):
        if not torch.equal(topk[step], stash):
            fails.append(f"{tag}: step {step} top-k is not the step-0 last-accepted row")
    report[f"{tag}:reuse_checked_steps"] = DRAFT_LEN - 1
    return fails


def _same_sets(a, b):
    return torch.equal(a.sort(dim=-1).values, b.sort(dim=-1).values)


def _compare_records(ref, cand, report, tag, layers=LAYERS):
    """Per call: MLA inputs (absorbed query, FP8 query, BMM scales), indexer
    inputs (FP8 query, head weights) and appended indexer keys bitwise; top-k
    and sparse-index key sets equal."""
    fails = []
    for layer in layers:
        r, c = ref[layer], cand[layer]
        if set(r) != set(c) or "absorbed_q" not in r:
            return [f"{tag}: layer {layer} recorded {sorted(c)} vs {sorted(r)}"]
        for key in sorted(r):
            if len(r[key]) != len(c[key]):
                fails.append(f"{tag}: layer {layer} {key}: {len(c[key])} calls vs {len(r[key])}")
        if fails:
            return fails
        for i in range(len(r["absorbed_q"])):
            name = f"{tag}:L{layer}.call{i}"
            for key in (
                "absorbed_q",
                "quant_q",
                "bmm1_scale",
                "bmm2_scale",
                "indexer_q_fp8",
                "indexer_weights",
                "index_k_fp8",
                "index_k_scale",
            ):
                if key not in r:
                    continue
                n_diff = int(
                    (aq_integration._bytes(r[key][i]) != aq_integration._bytes(c[key][i])).sum()
                )
                report[f"{name}.{key}_bytes_differing"] = n_diff
                if n_diff:
                    fails.append(f"{name}: {key} differs in {n_diff} bytes")
            for key in ("topk", "sparse_indices"):
                same = _same_sets(r[key][i], c[key][i])
                report[f"{name}.{key}_same_sets"] = same
                if not same:
                    fails.append(f"{name}: {key} selects different keys")
    return fails


def _output_bounds(native_runs):
    """Per (layer, call) bounds from repeated native iterations."""
    ref = native_runs[0]
    bounds = {}
    for layer in LAYERS:
        for i in range(len(ref[layer])):
            base = ref[layer][i].float()
            noise = max(
                [(run[layer][i].float() - base).abs().max().item() for run in native_runs[1:]]
                + [0.0]
            )
            rel = max(
                [aq_integration._rel_rms(run[layer][i], ref[layer][i]) for run in native_runs[1:]]
                + [0.0]
            )
            ulp = aq_integration._bf16_ulp(base.abs().max().item())
            bounds[(layer, i)] = (
                2.0 * max(noise, ulp),
                max(2.0 * rel, aq_integration.REL_RMS_FLOOR),
                noise,
            )
    return bounds


def _compare_outputs(ref, cand, bounds, report, tag):
    fails = []
    for (layer, i), (max_bound, rel_bound, noise) in bounds.items():
        diff = (cand[layer][i].float() - ref[layer][i].float()).abs().max().item()
        rel = aq_integration._rel_rms(cand[layer][i], ref[layer][i])
        report[f"{tag}:L{layer}.call{i}.out_max_diff"] = diff
        report[f"{tag}:L{layer}.call{i}.out_rel_rms"] = rel
        report[f"L{layer}.call{i}.native_repeat_max_diff"] = noise
        if diff > max_bound or rel > rel_bound:
            fails.append(
                f"{tag}: L{layer} call {i} output max diff "
                f"{diff:.3e} (bound {max_bound:.3e}), relRMS "
                f"{rel:.3e} (bound {rel_bound:.3e})"
            )
    return fails


def _native_runs(h, nreq, it, path, snap, repeats=NATIVE_RUNS):
    """Repeated native iterations from the same snapshot; the native
    iteration must itself be repeatable in caches and recorded inputs."""
    _set_features("")
    h.reset_runners()
    outs, recs, cache, fails = [], [], None, []
    accepted = it.num_accepted.tolist()
    expected = _expected_slots(nreq, accepted)
    for i in range(repeats):
        h.restore(snap)
        out, rec, calls = h.run(nreq, it, "custom_op" if path is None else path)
        assert calls == [], f"native run called absorbed_query entry points: {calls}"
        outs.append(out)
        recs.append(rec)
        report = {}
        fails += _check_reuse(rec, accepted, report, f"native{i}")
        if i == 0:
            cache = h.snapshot()
        else:
            fails += _compare_caches(
                snap, cache, h.snapshot(), expected, report, f"native_repeat{i}"
            )
            fails += _compare_records(recs[0], rec, report, f"native_repeat{i}")
    h.restore(snap)
    assert not fails, f"native iteration is not repeatable: {fails}"
    return outs, recs, cache


def _expected_calls(features, nreq):
    """Entry-point calls the absorbed-query packet makes in one iteration."""
    selected = set(features.split(","))
    target_query, draft_query = "absorbed_query" in selected, "draft_absorbed_query" in selected
    target_rows, steps = nreq * VERIFY_ROWS, [nreq * VERIFY_ROWS] + [nreq] * (DRAFT_LEN - 1)
    calls = []
    if target_query:
        calls.append(("dsa_proj", TARGET, target_rows, True))
    elif draft_query:
        calls.append(("mtp_dsa_proj", TARGET, target_rows, False))
    for rows in steps:
        if target_query:
            calls.append(("dsa_proj", MTP, rows, False))
        if draft_query:
            calls.append(("mtp_dsa_proj", MTP, rows, True))
    return calls


def _compare_to_native(snap, native, cand, accepted, nreq, bounds, report, tag):
    """Caches and recorded inputs bit-identical to native, the indexer reuse
    semantics, and outputs within the native run-to-run bound."""
    native_out, native_rec, native_cache = native
    out, rec, cache = cand
    fails = _compare_caches(snap, native_cache, cache, _expected_slots(nreq, accepted), report, tag)
    fails += _compare_records(native_rec, rec, report, tag)
    fails += _check_reuse(rec, accepted, report, tag)
    fails += _compare_outputs(native_out, out, bounds, report, tag)
    return fails


def _expected_launches(nreq):
    """Fused-indexer launches of layer 78 in one iteration (draft_absorbed_query only)."""
    return [(MTP, nreq * VERIFY_ROWS)] + [(MTP, nreq)] * (DRAFT_LEN - 1)


# -- tests ------------------------------------------------------------------

FEATURE_SETS = {
    "none": "",
    "absorbed_query": "absorbed_query",
    "draft_absorbed_query": "draft_absorbed_query",
    "absorbed_query_draft_query": "absorbed_query,draft_absorbed_query",
}


@pytest.mark.parametrize("features", list(FEATURE_SETS))
@pytest.mark.parametrize("nreq", [1, 2])
@torch.inference_mode()
def test_feature_gating_through_adapter(harness, features, nreq):
    """Which layers each selection engages, through the absorbed-query packet,
    and native agreement of the whole iteration."""
    h = harness
    sel = FEATURE_SETS[features]
    it = h.iteration_inputs(nreq, seed=12)
    accepted = it.num_accepted.tolist()
    snap = h.snapshot()
    try:
        native_outs, native_recs, native_cache = _native_runs(h, nreq, it, "custom_op", snap)
        bounds = _output_bounds(native_outs)
        _set_features(sel)
        h.reset_runners()
        h.restore(snap)
        out, rec, calls = h.run(nreq, it, "custom_op")
        cache = h.snapshot()
        expected = _expected_calls(sel, nreq)
        assert calls == expected, f"engagement {calls} != {expected}"
        engaged = {layer for _, layer, _, ok in calls if ok}
        report = {}
        fails = _compare_to_native(
            snap,
            (native_outs[0], native_recs[0], native_cache),
            (out, rec, cache),
            accepted,
            nreq,
            bounds,
            report,
            f"{features}",
        )
        print(f"features={sel!r} nreq={nreq} engaged={sorted(engaged)} {report}")
        assert not fails, fails
    finally:
        h.restore(snap)
        _set_features("")
        h.reset_runners()


@pytest.mark.parametrize("path", ["custom_op", "forward_impl"])
@pytest.mark.parametrize("form", ["auto", "preabsorbed", "native_q"])
@pytest.mark.parametrize("nreq", [1, 2])
@torch.inference_mode()
def test_draft_query_draft_loop_matches_native(harness, nreq, form, path):
    """draft_absorbed_query alone (layer 78 on all five draft steps) against native: caches,
    MLA inputs, indexer keys and reuse, outputs; then determinism (the same
    iteration again reproduces caches and records bit for bit)."""
    h = harness
    it = h.iteration_inputs(nreq, seed=20 + nreq)
    accepted = it.num_accepted.tolist()
    snap = h.snapshot()
    try:
        native_outs, native_recs, native_cache = _native_runs(h, nreq, it, path, snap)
        bounds = _output_bounds(native_outs)
        _set_features("draft_absorbed_query", form)
        h.reset_runners()
        h.restore(snap)
        with _LaunchLog() as launches:
            out, rec, calls = h.run(nreq, it, path)
        cache = h.snapshot()
        assert calls == _expected_calls("draft_absorbed_query", nreq), calls
        assert launches.calls == _expected_launches(nreq), launches.calls
        report = {}
        fails = _compare_to_native(
            snap,
            (native_outs[0], native_recs[0], native_cache),
            (out, rec, cache),
            accepted,
            nreq,
            bounds,
            report,
            "draft_absorbed_query",
        )
        h.restore(snap)
        _, rec_again, _ = h.run(nreq, it, path)
        fails += _compare_caches(
            snap, cache, h.snapshot(), _expected_slots(nreq, accepted), report, "draft_query_again"
        )
        fails += _compare_records(rec, rec_again, report, "draft_query_again")
        print(f"nreq={nreq} form={form} path={path} {report}")
        assert not fails, fails
    finally:
        h.restore(snap)
        _set_features("")
        h.reset_runners()


GRAPH_SETS = {
    "w5only": ("draft_absorbed_query",),
    "nativeonly": ("native",),
    "both": ("draft_absorbed_query", "native"),
}
# Diagnostics: restrict the graph cases (comma lists).
GRAPH_NREQ = [
    int(x)
    for x in os.environ.get("GLM_MOE_DSA_DRAFT_ABSORBED_QUERY_IT_GRAPH_NREQ", "1,2").split(",")
]
GRAPH_SET_NAMES = os.environ.get(
    "GLM_MOE_DSA_DRAFT_ABSORBED_QUERY_IT_GRAPH_SETS", ",".join(GRAPH_SETS)
).split(",")


def _tensor_ptrs(meta):
    """data_ptr of every CUDA tensor attribute of the metadata (what a
    captured graph may have baked in), as the graph runner's strict buffer
    check records them."""
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

    return CUDAGraphRunner._snapshot_graph_tensor_ptrs(meta)


def _moved(before, after):
    return sorted(k for k, v in before.items() if after.get(k) != v)


@pytest.mark.parametrize("graph_set", GRAPH_SET_NAMES)
@pytest.mark.parametrize("form", ["auto", "native_q"])
@pytest.mark.parametrize("nreq", GRAPH_NREQ)
@torch.inference_mode()
def test_draft_query_draft_loop_cuda_graph(harness, nreq, form, graph_set):
    """The whole iteration (target layer + native draft loop over layer 78)
    captured in one CUDA graph, as the GEN graph runner does: warm-up of
    every configuration first, then capture on ``torch.cuda.graph``'s stream
    into one shared pool. Replays 1, 1 and 20 times with fresh hidden states
    and fresh accepted counts must reproduce the eager iteration of the same
    selection bit for bit (caches, records) and compare with eager native as
    the eager test does."""
    h = harness
    graphs = {}
    snap = h.snapshot()
    try:
        _graph_checks(h, nreq, form, snap, graphs, GRAPH_SETS[graph_set])
    except BaseException as exc:
        print(f"graph test failure: {exc!r}", flush=True)
        raise
    finally:
        graphs.clear()
        h.reset_runners()
        h.restore(snap)
        _set_features("")


def _graph_checks(h, nreq, form, snap, graphs, wanted):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    path = "custom_op"
    features = {"draft_absorbed_query": "draft_absorbed_query", "native": ""}
    configs = [name for name in ("draft_absorbed_query", "native") if name in wanted]
    warm = {}
    for name in configs:
        _set_features(features[name], form)
        h.reset_runners()
        meta = h.gen_metadata(nreq, cuda_graph=True)
        static = h.iteration_inputs(nreq, seed=200)
        with with_multi_stream(True):
            for _ in range(2):
                # Host-side per-step preparation, as before every replay
                # (the draft loop moved kv_lens_cuda on the device).
                meta.prepare()
                h.iteration(meta, static, path)
        torch.cuda.synchronize()
        runners = {
            layer: (h.attn[layer]._absorbed_query, h.attn[layer]._draft_absorbed_query)
            for layer in LAYERS
        }
        if name == "draft_absorbed_query":
            assert runners[MTP][1] not in (None, False), "draft_absorbed_query did not engage"
        warm[name] = (meta, static, runners)
    h.restore(snap)
    pool = None
    for name in configs:
        meta, static, runners = warm[name]
        _set_features(features[name], form)
        for layer, (r_absorbed_query, r_draft_query) in runners.items():
            h.attn[layer]._absorbed_query, h.attn[layer]._draft_absorbed_query = (
                r_absorbed_query,
                r_draft_query,
            )
        graph = torch.cuda.CUDAGraph()
        ptrs_before = _tensor_ptrs(meta)
        with (
            with_multi_stream(True),
            _Recorder(h.attn) as rec,
            _Engagement() as eng,
            _LaunchLog() as launches,
        ):
            with torch.cuda.graph(graph, pool=pool):
                outs = h.iteration(meta, static, path)
        pool = graph.pool()
        ptrs_captured = _tensor_ptrs(meta)
        print(
            f"[graph test] {name}: metadata tensors rebound during "
            f"capture: {_moved(ptrs_before, ptrs_captured)}; new: "
            f"{sorted(set(ptrs_captured) - set(ptrs_before))}",
            flush=True,
        )
        assert eng.calls == _expected_calls(features[name], nreq), eng.calls
        if name == "draft_absorbed_query":
            assert launches.calls == _expected_launches(nreq), launches.calls
        graphs[name] = (graph, meta, static, outs, rec, runners, ptrs_captured)
        torch.cuda.synchronize()
        print(f"[graph test] captured {name}", flush=True)
    h.reset_runners()

    report, fails = {}, []
    trials = [(301, 1, None), (302, 1, [1] * nreq), (303, 20, [VERIFY_ROWS] * nreq)]
    for trial, (seed, replays, accepted) in enumerate(trials):
        it = h.iteration_inputs(nreq, seed=seed, accepted=accepted)
        accepted = it.num_accepted.tolist()
        expected = _expected_slots(nreq, accepted)
        native_outs, native_recs, native_cache = _native_runs(h, nreq, it, path, snap)
        bounds = _output_bounds(native_outs)
        _set_features("draft_absorbed_query", form)
        h.reset_runners()
        h.restore(snap)
        draft_query_eager = h.run(nreq, it, path)
        draft_query_eager_cache = h.snapshot()
        _set_features("")
        h.reset_runners()
        tag = f"t{trial}x{replays}"
        for name, (graph, meta, static, outs, rec, runners, ptrs_captured) in graphs.items():
            h.restore(snap)
            meta.prepare()
            moved = _moved(ptrs_captured, _tensor_ptrs(meta))
            print(
                f"[graph test] trial {trial} {name}: metadata tensors "
                f"rebound by prepare() since capture: {moved}",
                flush=True,
            )
            for key in ("hidden_t", "pos_t", "hidden_d", "num_accepted", "input_ids"):
                getattr(static, key).copy_(getattr(it, key))
            for i in range(replays):
                if i:
                    # The captured draft loop advances kv_lens_cuda on the
                    # device; like the runner before every step, rebuild the
                    # step's metadata from the host first (the replays
                    # rewrite the same slots with the same bytes).
                    meta.prepare()
                graph.replay()
            torch.cuda.synchronize()
            print(f"[graph test] trial {trial} {name}: replayed x{replays}", flush=True)
            g_out = {layer: [o.clone() for o in outs[layer]] for layer in LAYERS}
            g_rec, g_cache = rec.snapshot(), h.snapshot()
            fails += _check_reuse(g_rec, accepted, report, f"{tag}:{name}")
            if name == "draft_absorbed_query":
                fails += _compare_caches(
                    snap,
                    draft_query_eager_cache,
                    g_cache,
                    expected,
                    report,
                    f"{tag}:draft_query_graph_vs_draft_query_eager",
                )
                fails += _compare_records(
                    draft_query_eager[1],
                    g_rec,
                    report,
                    f"{tag}:draft_query_graph_vs_draft_query_eager",
                )
                fails += _compare_to_native(
                    snap,
                    (native_outs[0], native_recs[0], native_cache),
                    (g_out, g_rec, g_cache),
                    accepted,
                    nreq,
                    bounds,
                    report,
                    f"{tag}:draft_query_graph_vs_native",
                )
            else:
                fails += _compare_caches(
                    snap, native_cache, g_cache, expected, report, f"{tag}:native_graph_vs_eager"
                )
                fails += _compare_records(
                    native_recs[0], g_rec, report, f"{tag}:native_graph_vs_eager"
                )
                fails += _compare_outputs(
                    native_outs[0], g_out, bounds, report, f"{tag}:native_graph_vs_eager"
                )
            for r_absorbed_query, r_draft_query in runners.values():
                for runner in (r_absorbed_query, r_draft_query):
                    if runner not in (None, False):
                        assert int(runner.sync.abs().sum()) == 0
    print(f"graph nreq={nreq} form={form} graphs={wanted} {report}")
    assert not fails, fails


@torch.inference_mode()
def test_checks_detect_a_wrong_mtp_query(harness, monkeypatch):
    """Negative control: draft_absorbed_query packets whose absorbed query is zeroed must fail
    the absorbed-query / quant_q checks on layer 78."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    h = harness
    nreq, path = 1, "custom_op"
    it = h.iteration_inputs(nreq, seed=12)
    snap = h.snapshot()
    orig_project = query_proj.AbsorbedQueryProjection.project

    def broken_project(self, *args, **kwargs):
        packet = orig_project(self, *args, **kwargs)
        if packet.fused_q is not None:
            packet.fused_q[..., : self.mla.kv_lora_rank].zero_()
        return packet

    try:
        native_outs, native_recs, native_cache = _native_runs(h, nreq, it, path, snap)
        monkeypatch.setattr(query_proj.AbsorbedQueryProjection, "project", broken_project)
        _set_features("draft_absorbed_query", "preabsorbed")
        h.reset_runners()
        h.restore(snap)
        _, bad_rec, calls = h.run(nreq, it, path)
        assert calls == _expected_calls("draft_absorbed_query", nreq)
        report = {}
        fails = _compare_records(native_recs[0], bad_rec, report, "bad")
        print(f"negative control {fails}")
        for step in range(DRAFT_LEN):
            assert any(f"L{MTP}.call{step}: absorbed_q" in f for f in fails), fails
            assert any(f"L{MTP}.call{step}: quant_q" in f for f in fails), fails
        assert not any(f"L{TARGET}." in f for f in fails), fails
    finally:
        monkeypatch.undo()
        h.restore(snap)
        _set_features("")
        h.reset_runners()
