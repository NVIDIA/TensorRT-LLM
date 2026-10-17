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
"""sparse_mla_decode against native on real GLM-5.2 layer-0 attention inputs (see ``sparse_mla_real_fixture``).

The context latents are written by the native context step and the generation
rows by the native ``mla_rope_generation``. Two sources of the top-k:

* ``test_sparse_mla_real_layer0``: native indexer projections + a reference FP32
  logit evaluation and exact top-k (not the native top-k kernels);
* ``test_sparse_mla_real_layer0_native_indexer``: the unmodified native indexer
  (native indexer K cache from the context and generation rows, native paged
  MQA logits and decode top-k) under the served GEN indexer configuration
  and under the CUDA radix top-k; native and sparse_mla_decode consume the same native
  selection. This test also checks the premise of sparse_mla_decode's one deliberate
  difference (no ``-1`` inside a row's causal length) on the native top-k, on
  the global indices sparse_mla_decode receives, and on the native grouped remap of the same
  selection. With ``rope_kv_append`` selected, rope_kv_append's fused top-k bridge replaces the
  native remap and produces the indices sparse_mla_decode reads: every generation call of the bridge
  must then equal the native remap of the same local top-k bit for bit, emit no
  ``-1`` below a row's causal length, and be what sparse_mla_decode received.

The same error budget as ``test_sparse_mla_sparse_mla.py`` applies.
"""

import pytest
import torch
from sparse_mla_harness import (
    DsaGenHarness,
    capture_rope_append_bridge,
    check_budget,
    compare,
    error_stats,
    negative_inside_causal,
    oracle_for,
    record,
    rope_append_bridge_report,
    rope_append_selected,
)
from sparse_mla_real_fixture import (
    checkpoint_dir,
    indexer_projections,
    indexer_topk,
    layer_weight_names,
    load_indexer_weights,
    load_tensors,
    project_layer0,
    real_token_ids,
    recent_window_topk,
    reference_indexer_logits,
    selection_vs_kernel_scores,
    selection_vs_reference,
    topk_contract,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available()
    or checkpoint_dir() is None,
    reason="needs an SM10x GPU, the sparse_mla_decode kernel and the GLM-5.2 NVFP4 checkpoint",
)

SEQ_LEN_Q = 6
NATIVE_INDEXER_CONTEXTS = [100, 2043, 8192, 32768, 131072]
# Long contexts run in their own test (and process). The native context attention step of this
# harness faults at these lengths, before any generation-phase code runs (one 491K-token step,
# or 64K-token chunks of a 391K context), so these cases write only the
# native indexer K cache for the context tokens (native ``_update_k_cache``, 64K-token chunks)
# and keep random latent rows in the pool: the selection is the native indexer's on real
# indexer keys and queries, the attended latents are synthetic.
NATIVE_INDEXER_LONG_CONTEXTS = [391000, 491061]
CONTEXT_CHUNK_TOKENS = 65536
# Indexer kernels of the served GEN role and the exact
# CUDA radix decode top-k (indexerTopK.cu) that the default configuration selects.
NATIVE_INDEXER_CONFIGS = {
    "tp8_decode": dict(
        enable_heuristic_topk=True, use_cute_dsl_paged_mqa_logits=True, use_cute_dsl_topk=True
    ),
    "cuda_radix": dict(
        enable_heuristic_topk=False, use_cute_dsl_paged_mqa_logits=False, use_cute_dsl_topk=False
    ),
    # Diagnostic only: the served CuTe-DSL paged MQA logits with the exact CUDA radix top-k.
    "dsl_logits_cuda_radix": dict(
        enable_heuristic_topk=False, use_cute_dsl_paged_mqa_logits=True, use_cute_dsl_topk=False
    ),
}


@pytest.fixture(scope="module")
def layer0():
    ckpt = checkpoint_dir()
    max_tokens = max(NATIVE_INDEXER_CONTEXTS + NATIVE_INDEXER_LONG_CONTEXTS) + SEQ_LEN_Q
    weights = load_tensors(ckpt, layer_weight_names(0))
    ids = real_token_ids(ckpt, max_tokens)
    inputs = project_layer0(weights, ids)
    yield weights, inputs
    del weights, inputs
    torch.cuda.empty_cache()


@pytest.mark.parametrize("ctx_len", [32768, 131072])
def test_sparse_mla_real_layer0(layer0, ctx_len):
    weights, inputs = layer0
    harness = DsaGenHarness(
        [ctx_len], seq_len_q=SEQ_LEN_Q, fill_random=False, defer_generation=True
    )
    try:
        load_indexer_weights(harness.layers[0].indexer, weights)
        harness.run_context(
            0,
            inputs.fused_q[:ctx_len],
            inputs.q_pe[:ctx_len],
            inputs.latent[:ctx_len],
            recent_window_topk(ctx_len),
        )
        harness.start_generation()
        rows = slice(ctx_len, ctx_len + SEQ_LEN_Q)
        step = harness.make_step_from(
            0, inputs.fused_q[rows], inputs.q_pe[rows], inputs.latent[rows]
        )
        topk = indexer_topk(harness.layers[0].indexer, inputs, ctx_len, SEQ_LEN_Q)
        # Selection structure: overlap between consecutive MTP rows and spread over the context.
        sets = [set(topk[r].tolist()) for r in range(SEQ_LEN_Q)]
        overlap = [len(sets[r] & sets[r - 1]) / 2048 for r in range(1, SEQ_LEN_Q)]
        recent = [sum(1 for x in sets[r] if x >= ctx_len - 2048) / 2048 for r in range(SEQ_LEN_Q)]
        result = compare(
            harness,
            step,
            topk,
            "real_layer0_reference_topk",
            ctx_len=ctx_len,
            mtp_row_overlap=overlap,
            recent_2048_fraction=recent,
        )
        # Softmax peakedness of the real scores, per row and head (exact FP64 model).
        global_idx = result["global_idx"]
        q = step.quant_q.view(SEQ_LEN_Q, 8, 576).view(torch.float8_e4m3fn).double()
        pool = harness.pool_rows().view(torch.uint8)
        top_prob = []
        for r in range(SEQ_LEN_Q):
            keys = pool[global_idx[r][global_idx[r] >= 0].long()].view(torch.float8_e4m3fn).double()
            probs = torch.softmax(q[r] @ keys.T * float(step.bmm1_scale[0]), dim=-1)
            top_prob.append(probs.max(dim=-1).values.mean().item())
        record(
            "sparse_mla_correctness.jsonl",
            "real_layer0_scores",
            {"ctx_len": ctx_len, "mean_max_prob_per_row": top_prob},
        )
        assert error_stats(result["sparse_mla_decode"].double(), result["oracle"])["nonfinite"] == 0
    finally:
        harness.shutdown()


@pytest.mark.parametrize("ctx_len", NATIVE_INDEXER_CONTEXTS)
@pytest.mark.parametrize("config", ["tp8_decode", "cuda_radix"])
def test_sparse_mla_real_layer0_native_indexer(layer0, config, ctx_len):
    _run_native_indexer_case(layer0, config, ctx_len)


@pytest.mark.parametrize("ctx_len", NATIVE_INDEXER_LONG_CONTEXTS)
@pytest.mark.parametrize("config", ["tp8_decode", "cuda_radix"])
def test_sparse_mla_real_layer0_native_indexer_long(layer0, config, ctx_len):
    _run_native_indexer_case(
        layer0, config, ctx_len, chunk_tokens=CONTEXT_CHUNK_TOKENS, real_latents=False
    )


@pytest.mark.parametrize(
    "config,seq_len_q,mtp",
    [
        ("tp8_decode", 1, 0),
        ("tp8_decode", 4, 3),
        ("dsl_logits_cuda_radix", 6, 5),
        ("dsl_logits_cuda_radix", 1, 0),
    ],
)
def test_sparse_mla_real_layer0_native_indexer_variants(layer0, config, seq_len_q, mtp):
    """Diagnostic variants at 32K: verify-window width and the logits / top-k kernel pair."""
    _run_native_indexer_case(layer0, config, 32768, seq_len_q=seq_len_q, mtp=mtp)


def _run_native_indexer_case(
    layer0, config, ctx_len, chunk_tokens=None, real_latents=True, seq_len_q=SEQ_LEN_Q, mtp=None
):
    weights, inputs = layer0
    harness = DsaGenHarness(
        [ctx_len],
        seq_len_q=seq_len_q,
        seed=17,
        fill_random=not real_latents,
        defer_generation=True,
        sparse_overrides=NATIVE_INDEXER_CONFIGS[config],
        mtp_draft_len=seq_len_q - 1 if mtp is None else mtp,
    )
    try:
        indexer = harness.layers[0].indexer
        load_indexer_weights(indexer, weights)
        ctx_proj = indexer_projections(indexer, inputs, 0, ctx_len)
        harness.run_context(
            0,
            inputs.fused_q[:ctx_len],
            inputs.q_pe[:ctx_len],
            inputs.latent[:ctx_len],
            recent_window_topk(ctx_len) if real_latents else None,
            indexer_k=(ctx_proj[1], ctx_proj[2]),
            chunk_tokens=chunk_tokens,
            attention=real_latents,
        )
        del ctx_proj
        harness.start_generation()
        rows = slice(ctx_len, ctx_len + seq_len_q)
        step = harness.make_step_from(
            0, inputs.fused_q[rows], inputs.q_pe[rows], inputs.latent[rows]
        )
        gen_proj = indexer_projections(indexer, inputs, ctx_len, ctx_len + seq_len_q)
        harness.update_indexer_cache(0, gen_proj[1], gen_proj[2])

        harness.select_library(0, use_sparse_mla=False)
        with capture_rope_append_bridge() as bridge_native:
            native, topk_native = harness.forward_native_indexer(0, step, gen_proj)
        native = native.clone()
        native_topk_inputs = harness.last_topk_inputs
        assert "SparseMlaDecodeFmha" not in harness.selected_library(0)
        harness.select_library(0, use_sparse_mla=True)
        with (
            harness.capture_sparse_mla_inputs() as sparse_mla_calls,
            capture_rope_append_bridge() as bridge_sparse_mla,
        ):
            sparse_out, topk_sparse_mla = harness.forward_native_indexer(0, step, gen_proj)
        sparse_out = sparse_out.clone()
        torch.cuda.synchronize()
        assert harness.selected_library(0) == ["SparseMlaDecodeFmha"], harness.selected_library(0)
        assert len(sparse_mla_calls) == 1, len(sparse_mla_calls)
        received = sparse_mla_calls[0]

        contract = topk_contract(topk_native, ctx_len)
        reference_logits = reference_indexer_logits(indexer, inputs, ctx_len, seq_len_q)
        agreement = []
        for r in range(seq_len_q):
            native_ids = set(x for x in topk_native[r].tolist() if x >= 0)
            count = min(2048, reference_logits[r].numel())
            reference_ids = set(reference_logits[r].topk(count).indices.tolist())
            agreement.append(len(native_ids & reference_ids) / max(len(reference_ids), 1))
        same_selection = torch.equal(topk_native, topk_sparse_mla)
        selection = {
            "seq_len_q": seq_len_q,
            "max_draft_tokens": harness.metadata.max_draft_tokens,
            "dsl_expand": [
                bool(harness.metadata.expand_for_dsl),
                harness.metadata.dsl_expand_factor,
                harness.metadata.dsl_atom,
            ],
            "decode_top_k": indexer.top_k.decode_implementation.value,
            "self_sampling": bool(getattr(indexer, "_use_self_sampling_topk", False)),
            "same_selection_in_both_runs": same_selection,
            "rows": contract,
            "agreement_with_reference_topk": agreement,
            "native_selection_under_reference_logits": selection_vs_reference(
                topk_native, reference_logits
            ),
            "native_selection_vs_kernel_scores": (
                selection_vs_kernel_scores(topk_native, native_topk_inputs, reference_logits)
                if native_topk_inputs is not None
                else None
            ),
        }
        global_native = harness.global_indices(0, topk_native)
        oracle = oracle_for(harness, step, global_native)
        # Producer contract at sparse_mla_decode's input: the global indices it received (the native
        # remap of its run's selection) and the native grouped remap of the native run's selection.
        grouped_native = harness.grouped_global_indices(0, topk_native)
        received_negative = negative_inside_causal(
            received["indices"], received["seq_lens_kv"], received["seq_len_q"]
        )
        grouped_negative = negative_inside_causal(
            grouped_native, received["seq_lens_kv"], received["seq_len_q"]
        )
        selection["sparse_mla_input"] = {
            "negative_below_causal": received_negative.tolist(),
            "equals_remap_of_its_selection": torch.equal(
                received["indices"], harness.global_indices(0, topk_sparse_mla)
            ),
            "seq_len_q": received["seq_len_q"],
        }
        selection["grouped_remap"] = {
            "equals_per_layer_remap": torch.equal(grouped_native, global_native),
            "negative_below_causal": grouped_negative.tolist(),
        }
        # rope_kv_append's bridge (when selected) is the producer of the indices of both runs.
        rope_append_bridge = {
            "selected": rope_append_selected(),
            "calls_native_run": len(bridge_native),
            "calls_sparse_mla_run": len(bridge_sparse_mla),
            "calls": [
                rope_append_bridge_report(call, received["seq_lens_kv"], received["seq_len_q"])
                for call in bridge_native + bridge_sparse_mla
            ],
            "sparse_mla_received_bridge_output": (
                bool(torch.equal(received["indices"], bridge_sparse_mla[0]["global"]))
                if bridge_sparse_mla
                else None
            ),
        }
        selection["rope_append_bridge"] = rope_append_bridge
        if same_selection:
            check_budget(
                harness,
                native,
                sparse_out,
                global_native,
                oracle,
                "real_layer0_native_indexer",
                config=config,
                ctx_len=ctx_len,
                context_chunk_tokens=chunk_tokens,
                real_context_latents=real_latents,
                selection=selection,
            )
        else:
            # The native selection is not deterministic run to run. sparse_mla_decode is checked
            # twice: on the exact selection the native run consumed (the native indexer output
            # tensor fed to sparse_mla_decode through the fixed-top-k path) and, below, on the
            # selection of its own run.
            selection["sparse_mla_run_selection_overlap"] = [
                len(set(topk_native[r].tolist()) & set(topk_sparse_mla[r].tolist()))
                / max(len(set(topk_native[r].tolist())), 1)
                for r in range(seq_len_q)
            ]
            sparse_mla_same = harness.forward(0, step, topk_native).clone()
            assert harness.selected_library(0) == ["SparseMlaDecodeFmha"]
            check_budget(
                harness,
                native,
                sparse_mla_same,
                global_native,
                oracle,
                "real_layer0_native_indexer_same_selection",
                config=config,
                ctx_len=ctx_len,
                context_chunk_tokens=chunk_tokens,
                real_context_latents=real_latents,
                selection=selection,
            )
            global_sparse_mla = harness.global_indices(0, topk_sparse_mla)
            native_err = error_stats(native.double(), oracle)
            sparse_mla_err = error_stats(
                sparse_out.double(), oracle_for(harness, step, global_sparse_mla)
            )
            record(
                "sparse_mla_correctness.jsonl",
                "real_layer0_native_indexer",
                {
                    "config": config,
                    "ctx_len": ctx_len,
                    "context_chunk_tokens": chunk_tokens,
                    "real_context_latents": real_latents,
                    "selection": selection,
                    "native_vs_oracle": native_err,
                    "sparse_mla_vs_oracle": sparse_mla_err,
                },
            )
            assert sparse_mla_err["nonfinite"] == native_err["nonfinite"] == 0
            assert sparse_mla_err["nrms"] <= 1.25 * native_err["nrms"] + 2e-4, (
                sparse_mla_err,
                native_err,
            )
        # The premise of sparse_mla_decode's one deliberate difference from native: the native
        # indexer emits no -1 inside a row's causal length, and neither remap creates one.
        assert all(row["negative_below_causal"] == 0 for row in contract), contract
        assert int(received_negative.sum()) == 0, selection["sparse_mla_input"]
        assert selection["sparse_mla_input"]["equals_remap_of_its_selection"]
        assert selection["grouped_remap"]["equals_per_layer_remap"]
        assert int(grouped_negative.sum()) == 0, selection["grouped_remap"]
        if rope_append_bridge["selected"]:
            # The bridge served the generation call of both runs (counted per call: the
            # logged rope_kv_append/bridge decision is the first one, the declined context step).
            assert (
                rope_append_bridge["calls_native_run"] == 1
                and rope_append_bridge["calls_sparse_mla_run"] == 1
            ), rope_append_bridge
            for call in rope_append_bridge["calls"]:
                assert call["equals_native_remap"], rope_append_bridge
                assert sum(call["negative_below_causal"]) == 0, rope_append_bridge
            assert rope_append_bridge["sparse_mla_received_bridge_output"], rope_append_bridge
        else:
            assert not bridge_native and not bridge_sparse_mla
    finally:
        harness.shutdown()
