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
"""Stock producer -> absorbed_query -> native DSA attention, through the target's attention.

The attention modules are the target's adopted ones (``GlmMLA``, built by
``absorbed_query_fixture.build_native_attention``); every generation step runs as a decode
step (``dispatch.Phase.DECODE``) and the prefill as a prefill step. Real
GLM-5.2 weights of the full-indexer layer 6 and the shared layer 7 at TP8
rank 0; synthetic activations; FP8 KV cache (V1 ``DSACacheManager``).

Six requests with a 2,600-token context are natively prefilled through both
layers (the indexer then selects 2,048 of 2,600 keys). A generation step
then presents six query rows in one of two layouts: six requests x one token,
or one request x six tokens (the served MTP verify layout, K = 5).

Both call paths are exercised: ``forward_impl`` (``forward_sparse_attn``)
and ``custom_op`` (``MLA.forward``'s ``_forward_custom_op`` with the
registered metadata, which the model uses). Every step, eager or captured,
first calls ``on_update_kv_lens()`` on the metadata, as the decoder
runner's ``_forward_step`` does through ``_preprocess_inputs``.

Checks against the same step with absorbed_query off (all asserted unless noted):

* only the new-token slots of every cache (latent caches of both layers,
  indexer-key cache of layer 6) change, and every byte equals native;
* the MLA query after native RoPE/quantization is bit-identical: the
  absorbed half ``fused_q[..., :512]`` and the FP8 ``quant_q_buffer``
  handed to the attention kernel, and the BMM scales written next to them
  (recorded by wrapping the native ``mla_rope_generation``; in the FP8-KV
  path the RoPE half of ``fused_q`` is not written, the rotated q-rope goes
  only to ``quant_q_buffer``);
* the indexer inputs (``q_fp8``, ``weights``) are bit-identical to native,
  and the indexer's logical top-k (layer 6) and the converted sparse indices
  of both layers select the same key sets (rows compared after sorting;
  exact order and, on a failure, the per-row overlap reported);
* the attention outputs: the native sparse attention is not bitwise
  run-to-run deterministic, so the outputs are bounded by its measured
  repeat difference over ``NATIVE_RUNS`` native runs of the same step:
  ``max|absorbed_query - native| <= 2 * max(native repeat max diff, one BF16 ulp of
  max|native|)`` and ``relRMS(absorbed_query - native) <= max(2 * native repeat relRMS,
  2**-10)``. This is a secondary check: the negative control shows that a
  zeroed absorbed query moves the shared layer's outputs by only a few
  times that noise, while the bit-identity checks of the attention inputs
  above flag it directly;
* absorbed_query itself is deterministic: a repeated absorbed_query step, 20 more absorbed_query
  steps and every CUDA-graph replay reproduce the caches and recorded attention inputs of
  the eager absorbed_query step bit for bit.

The graph test captures the adopted layer forward (absorbed_query, ``_forward_dsa_attn``
and the native cache append) for CUDA-graph generation metadata, replays it
once and 20 times with fresh hidden states, and compares each replay with an
eager native step and with a captured native graph.
"""

import inspect
import math
import os
import sys

import absorbed_query_fixture
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import plan_testlib  # noqa: E402

CTX = 2600
REQS = 6
LAYERS = (absorbed_query_fixture.FULL_LAYER, absorbed_query_fixture.SHARED_LAYER)
# layout name -> (requests, query tokens per request)
LAYOUTS = {"6req_x1tok": (6, 1), "1req_x6tok": (1, 6)}
REL_RMS_FLOOR = 2.0**-10


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]


def _set_absorbed_query(enabled: bool, form: str = "preabsorbed"):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    query_proj.SERVED_FORM = form
    plan_testlib.select(
        dispatch, absorbed_query_fixture.with_base_features("absorbed_query" if enabled else "")
    )


class _Recorder:
    """Records, per layer, what the native attention consumes.

    Wraps the native ``mla_rope_generation`` (post-RoPE ``fused_q``, FP8
    ``quant_q_buffer``, BMM scales), the indexer's ``sparse_attn_indexer``
    (logical top-k) and ``sparse_attn_predict`` (converted indices). Inside
    a CUDA-graph capture the clones are graph-pool tensors that every replay
    refreshes. When the installed package registers rope_kv_append's fused replacement
    of ``mla_rope_generation`` (integration images), that entry point is
    wrapped as well and records the same tensors.
    """

    def __init__(self, attn_by_layer):
        self.attn = attn_by_layer
        self.data = {layer: {} for layer in attn_by_layer}
        self._installed = []
        self._rope_append = None

    @staticmethod
    def _rope_record(rec, kv_lora, a):
        # The FP8-KV path writes the rotated q-rope only into
        # quant_q_buffer; fused_q[..., kv_lora:] stays unwritten.
        rec["absorbed_q"] = a["fused_q"][..., :kv_lora].clone()
        for key, arg in (
            ("quant_q", "quant_q_buffer"),
            ("bmm1_scale", "mla_bmm1_scale"),
            ("bmm2_scale", "mla_bmm2_scale"),
        ):
            if a.get(arg) is not None:
                rec[key] = a[arg].clone()

    def _wrap_rope_append_rope(self):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

        try:
            impl = dispatch.get_impl("rope_kv_append", "mla_rope_generation")
        except (ImportError, KeyError, ValueError):
            return
        sig = inspect.signature(impl)
        by_mqa = {
            id(attn.mqa): (self.data[layer], attn.kv_lora_rank) for layer, attn in self.attn.items()
        }

        def hook(*args, **kwargs):
            out = impl(*args, **kwargs)
            a = sig.bind(*args, **kwargs).arguments
            hit = by_mqa.get(id(a["mqa"]))
            if hit is not None:
                self._rope_record(hit[0], hit[1], a)
                # Marks the call as served by rope_kv_append (compared like any record).
                hit[0]["rope_append_rope"] = torch.ones(
                    1, dtype=torch.int32, device=a["fused_q"].device
                )
            return out

        dispatch._REGISTRY["rope_kv_append"]["mla_rope_generation"] = hook
        self._rope_append = (dispatch, impl)

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

            def rope_post(a, _out, rec=rec, kv_lora=attn.kv_lora_rank):
                self._rope_record(rec, kv_lora, a)

            def predict_post(_a, out, rec=rec):
                if isinstance(out, tuple) and out[0] is not None:
                    rec["sparse_indices"] = out[0].clone()

            self._wrap(attn.mqa, "mla_rope_generation", rope_post)
            self._wrap(attn.mqa, "sparse_attn_predict", predict_post)
            if attn.mqa.indexer is not None:

                def topk_post(a, out, rec=rec):
                    rec["topk"] = out.clone()
                    rec["indexer_q_fp8"] = a["q_fp8"].clone()
                    rec["indexer_weights"] = a["weights"].clone()

                self._wrap(attn.mqa.indexer, "sparse_attn_indexer", topk_post)
        self._wrap_rope_append_rope()
        return self

    def __exit__(self, *exc):
        for obj, name in self._installed:
            delattr(obj, name)
        self._installed.clear()
        if self._rope_append is not None:
            dispatch, impl = self._rope_append
            dispatch._REGISTRY["rope_kv_append"]["mla_rope_generation"] = impl
            self._rope_append = None
        return False

    def snapshot(self):
        return {layer: {k: v.clone() for k, v in rec.items()} for layer, rec in self.data.items()}


class _Harness:
    def __init__(self):
        import tensorrt_llm.bindings
        from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
        from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig
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
                max_tokens=REQS * (CTX + 128),
                enable_block_reuse=False,
                event_buffer_max_size=0,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=max(LAYERS) + 1,
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=64,
            max_seq_len=CTX + 128,
            max_batch_size=REQS,
            mapping=self.mapping,
            dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
            sparse_attention_config=sparse,
            model_config=kv_model_config,
            pretrained_config=pretrained,
            vocab_size=pretrained.vocab_size,
            max_num_tokens=REQS * CTX,
        )
        self.req_ids = list(range(REQS))
        for rid in self.req_ids:
            # CTX + 6 new tokens fit the blocks of CTX + 1 tokens (64/block).
            self.kvm.add_dummy_requests(
                request_ids=[rid], token_nums=[CTX + 1], is_gen=False, prepare_resource=True
            )
        sparse_params = sparse.to_sparse_params(pretrained_config=pretrained)
        self.Metadata = get_attention_backend("TRTLLM", sparse_params=sparse_params).Metadata
        self.meta_params = sparse.to_sparse_metadata_params(pretrained_config=pretrained)
        self.gamma = absorbed_query_fixture.load_input_layernorm(LAYERS[0])

        # Native prefill through both layers fills the latent and indexer
        # caches the way the stock producers do.
        _set_absorbed_query(False)
        hidden = absorbed_query_fixture.make_hidden(REQS * CTX, seed=11, gamma=self.gamma)
        positions = torch.arange(CTX, dtype=torch.int32, device="cuda").repeat(REQS).unsqueeze(0)
        ctx_meta = self.Metadata(
            seq_lens=torch.tensor([CTX] * REQS, dtype=torch.int),
            request_ids=self.req_ids,
            max_num_requests=REQS,
            num_contexts=REQS,
            prompt_lens=[CTX] * REQS,
            max_num_tokens=REQS * CTX,
            kv_cache_manager=self.kvm,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * REQS),
            mapping=self.mapping,
            sparse_metadata_params=self.meta_params,
        )
        ctx_meta.prepare()
        self.step(ctx_meta, positions, hidden, "forward_impl")
        torch.cuda.synchronize()

    # -- generation inputs -------------------------------------------------

    def gen_inputs(self, layout, seed):
        nreq, ntok = LAYOUTS[layout]
        hidden = absorbed_query_fixture.make_hidden(nreq * ntok, seed=seed, gamma=self.gamma)
        pos = (CTX + torch.arange(ntok, dtype=torch.int32, device="cuda")).repeat(nreq).unsqueeze(0)
        return hidden, pos

    def gen_metadata(self, layout, cuda_graph=False):
        """Generation metadata as the runner builds it: graph metadata from
        ``create_cuda_graph_metadata`` with a buffer registry (reserved,
        graph-stable buffers, as the graph runner passes its own), MTP verify
        width from ``update_spec_dec_param``, then ``prepare()``."""
        nreq, ntok = LAYOUTS[layout]
        req_ids = self.req_ids[:nreq]
        kv_params = self.KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[CTX] * nreq)
        meta = self.Metadata(
            seq_lens=torch.tensor([ntok] * nreq, dtype=torch.int),
            request_ids=req_ids,
            max_num_requests=nreq,
            num_contexts=0,
            prompt_lens=[CTX] * nreq,
            max_num_tokens=nreq * ntok,
            kv_cache_manager=self.kvm,
            kv_cache_params=kv_params,
            mapping=self.mapping,
            sparse_metadata_params=self.meta_params,
        )
        if cuda_graph:
            from tensorrt_llm._torch.memory_buffer_utils import Buffers

            meta = meta.create_cuda_graph_metadata(nreq, False, ntok - 1, Buffers())
            assert meta.is_cuda_graph
            meta.request_ids = req_ids
            meta.prompt_lens = [CTX] * nreq
            meta.num_contexts = 0
            meta.kv_cache_params = kv_params
        if ntok > 1:
            meta.update_spec_dec_param(
                batch_size=nreq,
                is_spec_decoding_enabled=True,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=ntok - 1,
                max_total_draft_tokens=ntok - 1,
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
        # The model's path: MLA.forward with register_to_config goes through
        # _forward_custom_op and the registered metadata.
        import weakref

        from tensorrt_llm._torch.utils import model_extra_attrs

        extra = self.model_config.extra_attrs
        extra["attention_metadata"] = weakref.ref(meta)
        with model_extra_attrs(extra):
            out = [torch.ops.trtllm.create_mla_outputs(hidden, attn.layer_idx_str)]
            attn._forward_custom_op(hidden, positions, out, None)
        return out[0]

    def step(self, meta, positions, hidden, path):
        """One forward step of both layers, as the decoder runner's
        ``_forward_step`` runs it (eager or captured): ``_preprocess_inputs``
        first refreshes the metadata derived from the KV lengths
        (``on_update_kv_lens``), then the model forward, under the step's phase (a
        generation-only step is a decode step)."""
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
            dispatch,
        )

        phase = dispatch.Phase.DECODE if meta.num_contexts == 0 else dispatch.Phase.PREFILL
        meta.on_update_kv_lens()
        with dispatch.step(phase):
            return {layer: self.forward(layer, positions, hidden, meta, path) for layer in LAYERS}

    def gen_step(self, layout, path, hidden, pos, meta=None):
        meta = meta if meta is not None else self.gen_metadata(layout)
        outs = {layer: out.clone() for layer, out in self.step(meta, pos, hidden, path).items()}
        torch.cuda.synchronize()
        return outs

    def recorded_step(self, layout, path, hidden, pos):
        with _Recorder(self.attn) as rec:
            out = self.gen_step(layout, path, hidden, pos)
        return out, rec.snapshot()

    # -- caches ------------------------------------------------------------

    def caches(self):
        bufs = {f"latent{layer}": self.kvm.get_buffers(layer) for layer in LAYERS}
        bufs[f"index_k{LAYERS[0]}"] = self.kvm.get_indexer_k_cache_buffers(LAYERS[0])
        return bufs

    def snapshot(self):
        return {k: v.clone() for k, v in self.caches().items()}

    def restore(self, snap):
        for k, v in self.caches().items():
            v.copy_(snap[k])
        torch.cuda.synchronize()

    def reset_absorbed_query_runners(self):
        for layer in LAYERS:
            self.attn[layer]._absorbed_query = None


@pytest.fixture(scope="module")
def harness():
    with torch.inference_mode():
        h = _Harness()
    yield h
    _set_absorbed_query(False)


# -- comparison helpers -----------------------------------------------------


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def _changed_bytes(before, after):
    return _bytes(before) != _bytes(after)


def _changed_latent_slots(before, after):
    """(block, token) pairs whose bytes changed in a latent cache buffer
    ``[blocks, kv_factor, tokens_per_block, heads, head_dim]``."""
    diff = before.view(torch.uint8) != after.view(torch.uint8)
    diff = diff.any(dim=4).any(dim=3).any(dim=1)
    return {tuple(x) for x in diff.nonzero().tolist()}


def _index_k_tokens(buf):
    """Indexer-key cache blocks ``[64 x 128 FP8 | 64 x FP32 scale]`` as
    (codes [blocks, 64, 128] uint8, scales [blocks, 64] float32)."""
    raw = buf.contiguous().view(torch.uint8).reshape(buf.shape[0], -1)
    tokens = buf.shape[1]
    data = raw[:, : tokens * 128].reshape(-1, tokens, 128)
    scale = raw[:, tokens * 128 : tokens * 132].contiguous().view(torch.float32).reshape(-1, tokens)
    return data, scale


def _changed_index_k_tokens(before, after):
    db, sb = _index_k_tokens(before)
    da, sa = _index_k_tokens(after)
    changed = (db != da).any(dim=-1) | (_bytes(sb) != _bytes(sa)).view(
        sb.shape[0], sb.shape[1], 4
    ).any(dim=-1)
    return {tuple(x) for x in changed.nonzero().tolist()}


def _compare_caches(snap, ref_cache, cand_cache, n_new_tokens, report, tag):
    """Only the new-token slots may differ from the pre-step snapshot, and
    the candidate's bytes must equal the reference's."""
    fails = []
    for key in cand_cache:
        ref, cand = ref_cache[key], cand_cache[key]
        n_diff = int(_changed_bytes(ref, cand).sum())
        report[f"{tag}:{key}_bytes_differing"] = n_diff
        if key.startswith("latent"):
            slots = _changed_latent_slots(snap[key], cand)
            ref_slots = _changed_latent_slots(snap[key], ref)
        else:
            slots = _changed_index_k_tokens(snap[key], cand)
            ref_slots = _changed_index_k_tokens(snap[key], ref)
        if len(slots) != n_new_tokens or slots != ref_slots:
            fails.append(
                f"{tag}: {key} changed slots {sorted(slots)} vs reference {sorted(ref_slots)}"
            )
        if n_diff:
            fails.append(f"{tag}: {key} differs in {n_diff} bytes")
    return fails


def _compare_records(ref, cand, report, tag):
    """Bit equality of the recorded MLA attention inputs and indexer inputs;
    top-k / sparse indices must select equal key sets (row-wise sorted; the
    per-row overlap is reported when they do not)."""
    fails = []
    for layer in LAYERS:
        r, c = ref[layer], cand[layer]
        if set(r) != set(c) or "absorbed_q" not in r:
            fails.append(f"{tag}: layer {layer} recorded {sorted(c)} vs {sorted(r)}")
            return fails
        for key in sorted(r):
            if key in ("topk", "sparse_indices"):
                rs = r[key].sort(dim=-1).values
                cs = c[key].sort(dim=-1).values
                report[f"{tag}:L{layer}.{key}_same_order"] = torch.equal(r[key], c[key])
                if torch.equal(rs, cs):
                    continue
                overlap = min(
                    len(set(x) & set(y)) / max(len(set(x)), 1)
                    for x, y in zip(rs.tolist(), cs.tolist())
                )
                n_rows = int((rs != cs).any(dim=-1).sum())
                report[f"{tag}:L{layer}.{key}_rows_differing"] = n_rows
                report[f"{tag}:L{layer}.{key}_min_row_overlap"] = overlap
                fails.append(f"{tag}: layer {layer} {key} selects different keys in {n_rows} rows")
            else:
                n_diff = int((_bytes(r[key]) != _bytes(c[key])).sum())
                report[f"{tag}:L{layer}.{key}_bytes_differing"] = n_diff
                if n_diff:
                    fails.append(f"{tag}: layer {layer} {key} differs in {n_diff} bytes")
    return fails


def _rel_rms(x, ref):
    return (
        (x.double() - ref.double()).pow(2).mean().sqrt()
        / ref.double().pow(2).mean().sqrt().clamp_min(1e-30)
    ).item()


def _bf16_ulp(value):
    return 2.0 ** (math.floor(math.log2(max(value, 2.0**-126))) - 7)


def _output_bounds(native_runs):
    """Per-layer bounds from repeated native runs of the same step."""
    ref = native_runs[0]
    bounds = {}
    for layer in LAYERS:
        noise = max(
            [
                (run[layer].float() - ref[layer].float()).abs().max().item()
                for run in native_runs[1:]
            ]
            + [0.0]
        )
        rel = max([_rel_rms(run[layer], ref[layer]) for run in native_runs[1:]] + [0.0])
        ulp = _bf16_ulp(ref[layer].float().abs().max().item())
        bounds[layer] = (2.0 * max(noise, ulp), max(2.0 * rel, REL_RMS_FLOOR), noise, rel)
    return bounds


def _compare_outputs(ref, cand, bounds, report, tag):
    fails = []
    for layer in LAYERS:
        max_bound, rel_bound, noise, _ = bounds[layer]
        diff = (cand[layer].float() - ref[layer].float()).abs().max().item()
        rel = _rel_rms(cand[layer], ref[layer])
        report[f"{tag}:L{layer}.out_max_diff"] = diff
        report[f"{tag}:L{layer}.out_rel_rms"] = rel
        report[f"L{layer}.native_repeat_max_diff"] = noise
        if diff > max_bound or rel > rel_bound:
            fails.append(
                f"{tag}: layer {layer} output max diff {diff:.3e} "
                f"(bound {max_bound:.3e}), relRMS {rel:.3e} (bound "
                f"{rel_bound:.3e})"
            )
    return fails


# Native runs per step used to estimate the attention's run-to-run noise
# (one reference and four repeats).
NATIVE_RUNS = 5


def _native_runs(h, layout, path, hidden, pos, snap, repeats=NATIVE_RUNS):
    """Repeated native steps from the same cache snapshot: outputs, the
    recorded attention inputs and the caches after the first run. Native
    caches and recorded inputs must themselves be repeatable."""
    _set_absorbed_query(False)
    outs, recs, cache, fails = [], [], None, []
    for i in range(repeats):
        h.restore(snap)
        out, rec = h.recorded_step(layout, path, hidden, pos)
        outs.append(out)
        recs.append(rec)
        if i == 0:
            cache = h.snapshot()
        else:
            report = {}
            fails += _compare_caches(
                snap, cache, h.snapshot(), math.prod(LAYOUTS[layout]), report, f"native_repeat{i}"
            )
            fails += _compare_records(recs[0], rec, report, f"native_repeat{i}")
    h.restore(snap)
    assert not fails, f"native step is not repeatable: {fails}"
    return outs, recs, cache


# -- tests ------------------------------------------------------------------


def _compare_absorbed_query_to_native(snap, native, candidate, n_new, bounds, report, tag):
    """absorbed_query step vs native step: caches, MLA and indexer inputs bitwise, equal
    top-k key sets, noise-bounded outputs."""
    native_out, native_rec, native_cache = native
    absorbed_query_out, absorbed_query_rec, absorbed_query_cache = candidate
    fails = _compare_caches(snap, native_cache, absorbed_query_cache, n_new, report, tag)
    fails += _compare_records(native_rec, absorbed_query_rec, report, tag)
    fails += _compare_outputs(native_out, absorbed_query_out, bounds, report, tag)
    return fails


@pytest.mark.parametrize("path", ["custom_op", "forward_impl"])
@pytest.mark.parametrize("form", ["auto", "preabsorbed", "native_q"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_absorbed_query_into_native_attention(harness, layout, form, path):
    h = harness
    n_new = math.prod(LAYOUTS[layout])
    hidden, pos = h.gen_inputs(layout, seed=12)
    snap = h.snapshot()
    try:
        native_outs, native_recs, native_cache = _native_runs(h, layout, path, hidden, pos, snap)
        bounds = _output_bounds(native_outs)

        _set_absorbed_query(True, form)
        h.reset_absorbed_query_runners()
        absorbed_query_out, absorbed_query_rec = h.recorded_step(layout, path, hidden, pos)
        for layer in LAYERS:
            runner = h.attn[layer]._absorbed_query
            assert runner not in (None, False), f"absorbed_query did not engage on {layer}"
        absorbed_query_cache = h.snapshot()

        report, fails = {}, []
        fails += _compare_absorbed_query_to_native(
            snap,
            (native_outs[0], native_recs[0], native_cache),
            (absorbed_query_out, absorbed_query_rec, absorbed_query_cache),
            n_new,
            bounds,
            report,
            "absorbed_query",
        )

        # absorbed_query is deterministic: the same step again from the snapshot ...
        h.restore(snap)
        absorbed_query_again, absorbed_query_rec_again = h.recorded_step(layout, path, hidden, pos)
        fails += _compare_caches(
            snap, absorbed_query_cache, h.snapshot(), n_new, report, "absorbed_query_again"
        )
        fails += _compare_records(
            absorbed_query_rec, absorbed_query_rec_again, report, "absorbed_query_again"
        )
        # ... and 20 more steps rewrite the same slots with the same bytes.
        for _ in range(20):
            again = h.gen_step(layout, path, hidden, pos)
        fails += _compare_caches(
            snap, absorbed_query_cache, h.snapshot(), n_new, report, "absorbed_query_repeat20"
        )
        fails += _compare_outputs(native_outs[0], again, bounds, report, "absorbed_query_repeat20")
        print(f"layout={layout} form={form} path={path} {report}")
        assert not fails, fails
    finally:
        h.restore(snap)
        _set_absorbed_query(False)


# Graphs captured by the CUDA-graph test: absorbed_query alone, native alone, or both in
# one process sharing one memory pool (as the graph runner keeps several
# graphs).
GRAPH_SETS = {
    "p1only": ("absorbed_query",),
    "nativeonly": ("native",),
    "both": ("absorbed_query", "native"),
}


@pytest.mark.parametrize("graph_set", list(GRAPH_SETS))
@pytest.mark.parametrize("form", ["auto", "preabsorbed", "native_q"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_absorbed_query_cuda_graph_replay_into_native_attention(harness, layout, form, graph_set):
    """Captured absorbed_query + native attention + native cache append, replayed once
    and 20 times with fresh inputs. Each absorbed_query replay must reproduce the eager
    absorbed_query step bit for bit (caches, recorded attention inputs) and compare with
    the eager native step as the eager test does; a captured native graph
    must reproduce the eager native step bit for bit.

    Capture follows the graph runner: every configuration is warmed up on
    the current stream first, then all are captured on ``torch.cuda.graph``'s
    own capture stream into one shared memory pool."""
    h = harness
    n_new = math.prod(LAYOUTS[layout])
    path = "custom_op"
    snap = h.snapshot()
    graphs = {}
    try:
        _graph_replay_checks(h, layout, form, n_new, path, snap, graphs, GRAPH_SETS[graph_set])
    except BaseException as exc:
        # Report before the graphs are destroyed: a CUDA error raised while
        # tearing them down would otherwise hide the original failure.
        print(f"graph test failure: {exc!r}", flush=True)
        raise
    finally:
        graphs.clear()
        h.reset_absorbed_query_runners()
        h.restore(snap)
        _set_absorbed_query(False)


def _graph_replay_checks(h, layout, form, n_new, path, snap, graphs, wanted):
    """Body of the CUDA-graph test; ``graphs`` is owned by the caller."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    configs = [
        (name, name == "absorbed_query") for name in ("absorbed_query", "native") if name in wanted
    ]
    # Diagnostics only: GLM_MOE_DSA_ABSORBED_QUERY_IT_MULTI_STREAM=0 warms up and captures without
    # the aux-stream overlap the graph runner enables.
    multi_stream = os.environ.get("GLM_MOE_DSA_ABSORBED_QUERY_IT_MULTI_STREAM", "1") != "0"

    def step(meta, static_pos, static_hidden):
        # The runner's step: on_update_kv_lens() is captured too, so the
        # graph recomputes the graph-temporary metadata (indexer scan
        # lengths, slot mappings) instead of addressing tensors that a later
        # prepare() rebinds.
        return h.step(meta, static_pos, static_hidden, path)

    # Pass 1: warm up every configuration (graph metadata, runners, JIT).
    warm = {}
    for name, enabled in configs:
        _set_absorbed_query(enabled, form)
        h.reset_absorbed_query_runners()
        meta = h.gen_metadata(layout, cuda_graph=True)
        static_hidden, static_pos = h.gen_inputs(layout, seed=200)
        with with_multi_stream(multi_stream):
            for _ in range(2):
                step(meta, static_pos, static_hidden)
        torch.cuda.synchronize()
        print(f"[graph test] warmed up {name}", flush=True)
        # The captured graph addresses its runners' counters: keep them.
        runners = [h.attn[layer]._absorbed_query for layer in LAYERS]
        if enabled:
            for layer, runner in zip(LAYERS, runners):
                assert runner not in (None, False), f"absorbed_query did not engage on {layer}"
        warm[name] = (meta, static_hidden, static_pos, runners)
    h.restore(snap)
    # Pass 2: capture into one shared pool.
    pool = None
    for name, enabled in configs:
        meta, static_hidden, static_pos, runners = warm[name]
        _set_absorbed_query(enabled, form)
        for layer, runner in zip(LAYERS, runners):
            h.attn[layer]._absorbed_query = runner
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(multi_stream), _Recorder(h.attn) as rec:
            with torch.cuda.graph(graph, pool=pool):
                outs = step(meta, static_pos, static_hidden)
        pool = graph.pool()
        graphs[name] = (graph, meta, static_hidden, outs, rec, runners)
        torch.cuda.synchronize()
        print(f"[graph test] captured {name}", flush=True)
    # Eager absorbed_query steps below use their own runners.
    h.reset_absorbed_query_runners()

    report, fails = {}, []
    for trial, (seed, replays) in enumerate([(301, 1), (302, 1), (303, 20)]):
        hidden, pos = h.gen_inputs(layout, seed=seed)
        native_outs, native_recs, native_cache = _native_runs(h, layout, path, hidden, pos, snap)
        bounds = _output_bounds(native_outs)
        _set_absorbed_query(True, form)
        h.restore(snap)
        absorbed_query_eager = h.recorded_step(layout, path, hidden, pos)
        absorbed_query_eager_cache = h.snapshot()
        _set_absorbed_query(False)
        print(f"[graph test] trial {trial}: eager steps done", flush=True)
        replayed = {}
        for name, (graph, meta, static_hidden, outs, rec, runners) in graphs.items():
            h.restore(snap)
            meta.prepare()  # host-side per-step preparation (no realloc)
            static_hidden.copy_(hidden)
            for _ in range(replays):
                graph.replay()
            torch.cuda.synchronize()
            print(f"[graph test] trial {trial}: {name} x{replays} replayed", flush=True)
            replayed[name] = (
                {layer: outs[layer].clone() for layer in LAYERS},
                rec.snapshot(),
                h.snapshot(),
            )
        tag = f"t{trial}x{replays}"
        if "absorbed_query" in replayed:
            absorbed_query_out, absorbed_query_rec, absorbed_query_cache = replayed[
                "absorbed_query"
            ]
            # Replays reproduce the eager absorbed_query step bit for bit ...
            fails += _compare_caches(
                snap,
                absorbed_query_eager_cache,
                absorbed_query_cache,
                n_new,
                report,
                f"{tag}:absorbed_query_graph_vs_absorbed_query_eager",
            )
            fails += _compare_records(
                absorbed_query_eager[1],
                absorbed_query_rec,
                report,
                f"{tag}:absorbed_query_graph_vs_absorbed_query_eager",
            )
            # ... and compare with native as in the eager test.
            fails += _compare_absorbed_query_to_native(
                snap,
                (native_outs[0], native_recs[0], native_cache),
                (absorbed_query_out, absorbed_query_rec, absorbed_query_cache),
                n_new,
                bounds,
                report,
                f"{tag}:absorbed_query_graph_vs_native",
            )
        if "native" in replayed:
            g_out, g_rec, g_cache = replayed["native"]
            fails += _compare_caches(
                snap, native_cache, g_cache, n_new, report, f"{tag}:native_graph_vs_eager"
            )
            fails += _compare_records(native_recs[0], g_rec, report, f"{tag}:native_graph_vs_eager")
            fails += _compare_outputs(
                native_outs[0], g_out, bounds, report, f"{tag}:native_graph_vs_eager"
            )
        absorbed_query_runners = graphs["absorbed_query"][5] if "absorbed_query" in graphs else []
        for runner in absorbed_query_runners + [h.attn[layer]._absorbed_query for layer in LAYERS]:
            if runner not in (None, False):
                assert int(runner.sync.abs().sum()) == 0
    print(f"graph layout={layout} form={form} graphs={wanted} {report}")
    assert not fails, fails


@torch.inference_mode()
def test_checks_detect_a_wrong_preabsorbed_query(harness, monkeypatch):
    """Negative control: an absorbed_query packet whose absorbed query is left unwritten
    (zeroed) must fail both the absorbed-query / quant_q checks and the
    output bound."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    h = harness
    layout, path = "6req_x1tok", "forward_impl"
    hidden, pos = h.gen_inputs(layout, seed=12)
    snap = h.snapshot()
    orig_project = query_proj.AbsorbedQueryProjection.project

    def broken_project(self, *args, **kwargs):
        packet = orig_project(self, *args, **kwargs)
        if packet.fused_q is not None:
            packet.fused_q[..., : self.mla.kv_lora_rank].zero_()
        return packet

    try:
        native_outs, native_recs, _ = _native_runs(h, layout, path, hidden, pos, snap)
        bounds = _output_bounds(native_outs)
        monkeypatch.setattr(query_proj.AbsorbedQueryProjection, "project", broken_project)
        _set_absorbed_query(True, "preabsorbed")
        h.reset_absorbed_query_runners()
        bad_out, bad_rec = h.recorded_step(layout, path, hidden, pos)
        report = {}
        rec_fails = _compare_records(native_recs[0], bad_rec, report, "bad")
        out_fails = _compare_outputs(native_outs[0], bad_out, bounds, report, "bad")
        print(f"negative control {report}")
        assert any("absorbed_q" in f for f in rec_fails), rec_fails
        assert any("quant_q" in f for f in rec_fails), rec_fails
        assert len(out_fails) == len(LAYERS), out_fails
    finally:
        monkeypatch.undo()
        h.restore(snap)
        _set_absorbed_query(False)
        h.reset_absorbed_query_runners()
