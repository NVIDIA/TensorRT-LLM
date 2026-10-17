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
"""Real GLM-5.2 attention layers through ``GlmMLA.forward``, for sparse_mla_decode composition tests.

Run as a script in a fresh interpreter: ``python sparse_mla_stack_case.py <output.pt>`` with the
fast paths to run in ``GLM_MOE_DSA_TEST_FAST_PATHS`` (``plan_testlib.FAST_PATHS_ENV``). It uses
the attention_output_tail tests' ``real_mla_harness.AttentionStack``: the TP8 rank-0 shards of two
consecutive real target layers (default 2, which runs its own
indexer, and 3, which reuses layer 2's top-k), a native prefill of
``GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_CTX`` real-token positions, and the MTP verify step of one request
(6 rows) with the generation metadata the decoder runner builds (CUDA-graph
metadata, ``update_spec_dec_param``, ``prepare``). The layers are the target's adopted ones and
the verify step is a decode step, so ``GlmMLA.forward`` runs through ``register_to_config`` (the
target's DSA attention op) as in the model and each package at the MLA / DSA call sites runs
where the model runs it:
absorbed_query (full-indexer layer) and shared_indexer_absorbed_query (shared-indexer layer) produce
the query and latent row, rope_kv_append the RoPE / FP8 query / latent append, the indexer-K scatter
and the top-k bridge, the attention_output_tail (attention_output_tail_prefetch /
attention_output_into_hop schedules) consumes the attention output as its programmatic dependent,
and sparse_mla_decode computes the attention. The process stands in for one of eight TP ranks
(o_proj all-reduce off).

``GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_TOPK=fixed`` (default) replaces the full layer's indexer
selection by a given local top-k (deterministic indices; the shared layer reuses it through
the shared top-k buffer); ``indexer`` runs the native indexer.

Saved (``torch.save``):

* ``inputs``: an eager step per input set; per layer the FMHA inputs (FP8
  query bytes, bmm scales, global indices, KV lengths, attended pool rows),
  cloned before the attention, plus the latent and module outputs;
* ``pdl``: the same steps with no recording kernel anywhere, so the kernel
  after the attention is the one the model runs there;
* ``serialized``: the same steps with a device synchronization right after
  each sparse_mla_decode launch (no programmatic overlap with the consumer);
* ``graph``: replays of one captured step over alternating input sets;
* ``kernels``: CUPTI records of two eager steps and of graph replays: kernel
  counts and, per attention kernel, its predecessor and successor on the
  stream and whether they overlapped it;
* ``engagement``: per-layer counts of absorbed_query / shared_indexer_absorbed_query packets,
  rope_kv_append fused RoPE calls and generation bridge calls, the attention_output_tail counters
  and config, sparse_mla_decode's dispatch accounting, and the logged capability decisions;
* ``rope_append_bridge``: per generation call of rope_kv_append's bridge in the ``inputs`` steps,
  whether its output equals the native remap of the same local top-k, the
  ``-1`` entries it emitted below a row's causal length, and whether the
  attention received exactly that output.
"""

import dataclasses
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import Mock

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "attention_output_tail"))
sys.path.insert(0, str(_HERE.parent))

import plan_testlib  # noqa: E402
import real_mla_harness  # noqa: E402
import torch  # noqa: E402
from sparse_mla_harness import (  # noqa: E402
    SPARSE_MLA_KERNEL,
    attention_neighbours,
    is_native_sparse_fmha,
    kernel_trace,
    rope_append_bridge_report,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E402
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E402, E501
    sparse_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode.sparse_mla import (  # noqa: E402, E501
    SparseMlaDecodeFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha  # noqa: E402
from tensorrt_llm._torch.distributed import AllReduceParams  # noqa: E402

ROWS = 6
TOP_K = 2048
HEAD_DIM = 576


def _selected(feature_id: str) -> bool:
    return feature_id in dispatch.SELECTOR_PATHS and dispatch.is_enabled(feature_id)


class Engagement:
    """Per-call counters of the producer packages, read at their registered entry points."""

    def __init__(self):
        self.counts = defaultdict(int)
        self.bridge = []
        self.record_bridge = False
        if _selected("absorbed_query"):
            for name, label in (
                ("dsa_proj", "absorbed_query"),
                ("shared_dsa_proj", "shared_indexer_absorbed_query"),
            ):
                self._count_packets(name, label)
        if _selected("rope_kv_append"):
            rope = dispatch.get_impl("rope_kv_append", "mla_rope_generation")

            def rope_hook(mqa, *args, **kwargs):
                self.counts[f"rope_append_rope/layer{mqa.layer_idx}"] += 1
                return rope(mqa, *args, **kwargs)

            dispatch._REGISTRY["rope_kv_append"]["mla_rope_generation"] = rope_hook
            bridge = dispatch.get_impl("rope_kv_append", "remap_topk_to_global")

            def bridge_hook(backend, topk, metadata, packet):
                if metadata.num_contexts != 0:
                    return bridge(backend, topk, metadata, packet)
                self.counts[f"rope_append_bridge/layer{backend.layer_idx}"] += 1
                entry = None
                if self.record_bridge:
                    source = topk if packet.source is None else packet.source
                    entry = {
                        "layer": backend.layer_idx,
                        "local_layer": backend.get_local_layer_idx(metadata),
                        "topk": source.clone(),
                        "metadata": metadata,
                    }
                out = bridge(backend, topk, metadata, packet)
                if entry is not None:
                    entry["global"] = out
                    self.bridge.append(entry)
                return out

            dispatch._REGISTRY["rope_kv_append"]["remap_topk_to_global"] = bridge_hook

    def _count_packets(self, name, label):
        original = dispatch.get_impl("absorbed_query", name)

        def hook(mla, *args, **kwargs):
            packet = original(mla, *args, **kwargs)
            if packet is not None:
                self.counts[f"{label}/layer{mla.layer_idx}"] += 1
            return packet

        dispatch._REGISTRY["absorbed_query"][name] = hook


class FmhaCapture:
    """Records each attention call: library, layer, output and (optionally) input clones.

    Only the input clones add kernels, and they run before the attention; the
    output is kept by reference (which also keeps its memory from being reused)
    and copied after the step, so nothing is inserted between the attention
    kernel and its consumer.
    """

    def __init__(self):
        self.calls = []
        self.clone_inputs = False
        self.sync_after_sparse_decode = False
        for cls in (SparseMlaDecodeFmha, FallbackFmha):
            original = cls.forward

            def forward(fmha, q, k, v, md, args, _original=original, _cls=cls):
                sparse = args.sparse_runtime_params
                entry = {
                    "library": _cls.__name__,
                    "layer": fmha.attn.layer_idx,
                    "generation": md.num_contexts == 0,
                    "output": args.output,
                    "indices_ref": sparse.sparse_attn_indices,
                }
                if self.clone_inputs:
                    entry.update(
                        quant_q=args.quant_q_buffer.clone(),
                        bmm1=args.mla_bmm1_scale.clone(),
                        bmm2=args.mla_bmm2_scale.clone(),
                        indices=sparse.sparse_attn_indices.clone(),
                        kv_lens=md.kv_lens_cuda_runtime[md.num_contexts :].clone(),
                    )
                _original(fmha, q, k, v, md, args)
                if self.sync_after_sparse_decode and _cls is SparseMlaDecodeFmha:
                    torch.cuda.synchronize()
                self.calls.append(entry)

            cls.forward = forward


def local_topk(ctx_len: int, seed: int) -> torch.Tensor:
    """Distinct local ids per row: the 32 most recent visible tokens plus random older ones."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    topk = torch.full((ROWS, TOP_K), -1, dtype=torch.int32)
    for row in range(ROWS):
        visible = ctx_len + row + 1
        recent = torch.arange(visible - 32, visible)
        rest = torch.randperm(visible - 32, generator=gen)[: TOP_K - 32]
        topk[row] = torch.cat([recent, rest]).to(torch.int32)
    return topk.cuda()


def summarize_kernels(kernels, steps, layers):
    names = Counter(k["name"] for k in kernels)
    neighbours = attention_neighbours(kernels)
    sparse_kernels = [a for a in neighbours if a["kernel"] == "sparse_mla_decode"]
    return {
        "steps": steps,
        "layers": layers,
        "sparse_mla_decode": sum(n for name, n in names.items() if SPARSE_MLA_KERNEL in name),
        "native_sparse_fmha": sum(n for name, n in names.items() if is_native_sparse_fmha(name)),
        "rope_append_bridge": sum(n for name, n in names.items() if "topkBridgeKernel" in name),
        "rope_append_rope": sum(
            n for name, n in names.items() if "ropeAppendScatterKernel" in name
        ),
        "native_remap": sum(n for name, n in names.items() if "convertReqIndexToGlobal" in name),
        "top_kernels": [[name[:120], n] for name, n in names.most_common(48)],
        "attention": neighbours,
        "successors": Counter(a["successor"] for a in neighbours).most_common(),
        "predecessors": Counter(a["predecessor"] for a in neighbours).most_common(),
        "sparse_mla_successor_early": sum(a["successor_early"] for a in sparse_kernels),
        "sparse_mla_started_before_predecessor_end": sum(
            a["started_before_predecessor_end"] for a in sparse_kernels
        ),
        "sparse_mla_duration_us": sorted(a["duration_us"] for a in sparse_kernels),
        "sparse_mla_successor_lead_us": sorted(
            a["successor_lead_us"] for a in sparse_kernels if a["successor_lead_us"] is not None
        ),
        "w4_predecessor_lead_us": sorted(
            a["predecessor_lead_us"] for a in sparse_kernels if a["predecessor_lead_us"] is not None
        ),
    }


def drive(
    stack, engagement, layers, ctx, replays, profiled_replays, topk_mode, features, out_path
) -> None:
    """Run, record and save every step of one process (see the module docstring)."""
    # The prefill detached every tail; attach what the adoption built (None: native tail).
    for layer, attn in stack.attn.items():
        attn._attn_tail = stack.attn_tail_tails[layer]
    full_layer = layers[0]
    indexer = stack.attn[full_layer].mqa.indexer
    assert indexer is not None, f"layer {full_layer} must run its own indexer"
    shared_layers = [layer for layer in layers if stack.attn[layer].mqa.indexer is None]

    hidden = [{layer: stack.gen_inputs(layer, step=s) for layer in layers} for s in (0, 1)]
    topks = [local_topk(ctx, 77), local_topk(ctx, 78)]
    static_hidden = {layer: hidden[0][layer].clone() for layer in layers}
    static_topk = topks[0].clone()
    if topk_mode == "fixed":
        indexer.forward_from_projected = Mock(return_value=static_topk)
    pos = stack.position_ids()
    no_ar = AllReduceParams(enable_allreduce=False)
    pool_rows = stack.kv_cache_manager.get_unique_primary_pool().view(torch.uint8)
    pool_rows = pool_rows.reshape(-1, HEAD_DIM)
    capture = FmhaCapture()

    def load(index):
        for layer in layers:
            static_hidden[layer].copy_(hidden[index][layer])
        static_topk.copy_(topks[index])

    def step(index):
        load(index)
        with stack.scope():
            return {
                layer: stack.forward(layer, static_hidden[layer], pos, no_ar) for layer in layers
            }

    def generation_calls(start):
        calls = [c for c in capture.calls[start:] if c["generation"]]
        assert [c["layer"] for c in calls] == layers, [c["layer"] for c in calls]
        return calls

    # Warm-up (JIT compilation, autotuning, lazy buffers) with both input sets.
    for index in (0, 1):
        step(index)
    torch.cuda.synchronize()

    result = {"features": features, "topk_mode": topk_mode, "ctx": ctx, "layers": layers}
    result["shared_layers"] = shared_layers
    # Eager steps with the FMHA inputs recorded (clones before the attention).
    capture.clone_inputs = True
    engagement.record_bridge = True
    result["inputs"] = []
    bridge_pairs = []
    for index in (0, 1):
        start, bridge_start = len(capture.calls), len(engagement.bridge)
        outs = step(index)
        torch.cuda.synchronize()
        per_layer = {}
        for call in generation_calls(start):
            idx = call["indices"]
            per_layer[call["layer"]] = {
                "library": call["library"],
                "quant_q": call["quant_q"].cpu(),
                "bmm1": call["bmm1"].cpu(),
                "bmm2": call["bmm2"].cpu(),
                "indices": idx.cpu(),
                "kv_lens": call["kv_lens"].cpu(),
                "latent": call["output"].clone().cpu(),
                "module_output": outs[call["layer"]].clone().cpu(),
                "rows": pool_rows[idx.long().clamp_min(0).reshape(-1)]
                .reshape(ROWS, TOP_K, -1)
                .cpu(),
            }
            for entry in engagement.bridge[bridge_start:]:
                if entry["layer"] == call["layer"]:
                    bridge_pairs.append((index, entry, idx, call["kv_lens"]))
        result["inputs"].append({"input": index, "layers": per_layer})
    capture.clone_inputs = False
    engagement.record_bridge = False

    def plain_steps(label):
        records = []
        for index in (0, 1):
            start = len(capture.calls)
            outs = step(index)
            torch.cuda.synchronize()
            records.append(
                {
                    "input": index,
                    "layers": {
                        c["layer"]: {
                            "library": c["library"],
                            "latent": c["output"].clone().cpu(),
                            "indices": c["indices_ref"].clone().cpu(),
                            "module_output": outs[c["layer"]].clone().cpu(),
                        }
                        for c in generation_calls(start)
                    },
                }
            )
        result[label] = records

    plain_steps("pdl")
    capture.sync_after_sparse_decode = True
    plain_steps("serialized")
    capture.sync_after_sparse_decode = False

    # One captured verify step (both layers), replayed over alternating input sets.
    step(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = len(capture.calls)
    with stack.scope():
        with torch.cuda.graph(graph):
            graph_outs = {
                layer: stack.forward(layer, static_hidden[layer], pos, no_ar) for layer in layers
            }
    graph_calls = generation_calls(start)

    def replay(index):
        load(index)
        graph.replay()

    result["graph"] = []
    for it in range(replays):
        index = it % 2
        replay(index)
        torch.cuda.synchronize()
        result["graph"].append(
            {
                "input": index,
                "layers": {
                    c["layer"]: {
                        "latent": c["output"].clone().cpu(),
                        "indices": c["indices_ref"].clone().cpu(),
                        "module_output": graph_outs[c["layer"]].clone().cpu(),
                    }
                    for c in graph_calls
                },
            }
        )

    # CUPTI kernel records: two eager steps without recording kernels, then graph replays.
    eager_kernels = kernel_trace(lambda: [step(i) for i in (0, 1)])
    graph_kernels = kernel_trace(lambda: [replay(i % 2) for i in range(profiled_replays)])
    result["kernels"] = {
        "eager": summarize_kernels(eager_kernels, 2, len(layers)),
        "graph": summarize_kernels(graph_kernels, profiled_replays, len(layers)),
    }

    # rope_kv_append bridge contract, checked after every run so the native remap used as the
    # reference cannot change the state of the steps above.
    result["rope_append_bridge"] = []
    for index, entry, received, kv_lens in bridge_pairs:
        report = rope_append_bridge_report(entry, kv_lens, ROWS)
        report["input"] = index
        report["attention_received_bridge_output"] = bool(
            torch.equal(received, entry["global"].to(received.device))
        )
        result["rope_append_bridge"].append(report)

    tails = {}
    for layer in layers:
        tail = stack.attn_tail_tails[layer]
        tails[layer] = (
            None
            if tail is None
            else {
                "engaged_calls": tail.engaged_calls,
                "declined_calls": tail.declined_calls,
                "static_reason": tail.static_reason,
                "into_hop_published_calls": tail.into_hop_published_calls,
                "config": dataclasses.asdict(tail.config),
            }
        )
    sparse_decode_accounting = sparse_mla.engagement_snapshot()
    result["engagement"] = {
        "counts": dict(engagement.counts),
        "attn_tail_tails": tails,
        "sparse_mla_decode": {
            "static": {str(k): v for k, v in sparse_decode_accounting["static"].items()},
            "engaged_calls": {
                str(k): v for k, v in sparse_decode_accounting["engaged_calls"].items()
            },
            "declined_selections": sparse_decode_accounting["declined_selections"],
        },
        "capability": {f"{f}/{key}": v for (f, key), v in dispatch._CAPABILITY_LOGGED.items()},
    }
    result["libraries"] = sorted({c["library"] for c in capture.calls if c["generation"]})
    torch.save(result, out_path)
    summary = {
        "features": features,
        "topk_mode": topk_mode,
        "libraries": result["libraries"],
        "engagement": result["engagement"],
        "kernels": {
            kind: {k: v for k, v in s.items() if k not in ("attention", "top_kernels")}
            for kind, s in result["kernels"].items()
        },
        "rope_append_bridge": result["rope_append_bridge"],
    }
    print("GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_SUMMARY " + json.dumps(summary, default=str))


def main(out_path: str) -> None:
    plan_testlib.select_from_env(dispatch)
    ctx = int(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_CTX", "32768"))
    layers = [
        int(x)
        for x in os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_LAYERS", "2,3").split(",")
    ]
    replays = int(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_REPLAYS", "8"))
    profiled_replays = int(
        os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_PROFILED_REPLAYS", "6")
    )
    topk_mode = os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_TOPK", "fixed")
    assert topk_mode in ("fixed", "indexer"), topk_mode
    ckpt = os.environ["GLM52_NVFP4_CKPT"]
    torch.manual_seed(0)
    features = sorted(dispatch.enabled_features())
    engagement = Engagement()

    stack = real_mla_harness.AttentionStack(ckpt, layers, tp_rank=0, ctx=ctx, rows=ROWS)
    # GEN steps run under inference mode, as the model engine runs them (absorbed_query's
    # indexer projection updates views of its outputs in place).
    with torch.inference_mode():
        drive(
            stack, engagement, layers, ctx, replays, profiled_replays, topk_mode, features, out_path
        )
    stack.shutdown()


if __name__ == "__main__":
    main(sys.argv[1])
