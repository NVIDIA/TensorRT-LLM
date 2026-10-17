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
"""sparse_mla_decode between engaged producers (absorbed_query, shared_indexer_absorbed_query,
rope_kv_append) and its PDL consumer (the attention_output_tail).

Each package set runs ``sparse_mla_stack_case.py`` in a fresh interpreter, without and
with ``sparse_mla_decode``, on real GLM-5.2 layers at TP8 rank 0: layer 2 runs its own indexer
(absorbed_query's layer), layer 3 reuses layer 2's top-k (shared_indexer_absorbed_query's layer).
``MLA.forward`` runs the MTP verify step of one request (6 rows) as the model does. Asserted:

* engagement, counted per call: absorbed_query returns its packet on the full layer and
  shared_indexer_absorbed_query on the shared layer; rope_kv_append's fused RoPE and top-k bridge
  serve every generation call; the attention_output_tail takes every call of both layers;
  sparse_mla_decode runs on both layers;
* the producers do not depend on the attention library: the FMHA inputs (FP8
  query bytes, bmm scales, KV lengths, global indices and the pool rows they
  address) are bit-identical without and with sparse_mla_decode;
* sparse_mla_decode's latent output is within the frozen error budget against the FP64 oracle
  on both layers, and the module outputs (after the attention_output_tail) agree with the
  native-attention run;
* sparse_mla_decode triggers its programmatic dependents before its gathers complete; the
  attention_output_tail is that dependent. With sparse_mla_decode, the eager steps (no
  recording kernels), the steps with a device synchronization right after each sparse_mla_decode
  launch, and the replays of a captured step must give bit-identical latent and module outputs;
* CUPTI: one ``sparseMlaDecodeKernel`` per DSA layer per step (eager and graph)
  and no native sparse FMHA kernel with sparse_mla_decode; none without it; in graph replays
  the kernel after sparse_mla_decode on its stream is attention_output_tail's W_UV kernel and it
  becomes resident before sparse_mla_decode ends;
* with rope_kv_append, every generation call of its bridge equals the native remap of the
  same local top-k bit for bit, emits no ``-1`` below a row's causal length, and
  is what the attention received.

Neighbour sets: native, ``absorbed_query,shared_indexer_absorbed_query,attention_output_tail``
(attention_output_tail default schedule) and the packages of the served selector that act at
these call sites: absorbed_query, shared_indexer_absorbed_query, rope_kv_append,
attention_output_tail, post_attention_hop, attention_output_tail_prefetch and
attention_output_into_hop (with post_attention_hop and attention_output_into_hop the tail takes
``INTO_HOP_SPARSE_DECODE_SCHEDULE`` behind sparse_mla_decode, which streams 32 ``w_o`` rows per
CTA before its PDL wait). The last set also runs with the unmodified native indexer.

Runs with ``GLM_MOE_DSA_SPARSE_MLA_DECODE_COMPOSITION=1`` on an integration build that contains the
packages, with the GLM-5.2 checkpoint.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from sparse_mla_harness import error_stats, oracle_attention, record

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plan_testlib  # noqa: E402
from utils.llm_data import llm_models_root

_CKPT = os.environ.get(
    "GLM52_NVFP4_CKPT",
    os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4"),
)
_CASE = Path(__file__).resolve().parent / "sparse_mla_stack_case.py"
_ATTN_TAIL_HARNESS = (
    Path(__file__).resolve().parent.parent / "attention_output_tail" / "real_mla_harness.py"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_COMPOSITION") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available()
    or not os.path.isfile(os.path.join(_CKPT, "config.json"))
    or not _ATTN_TAIL_HARNESS.is_file(),
    reason="set GLM_MOE_DSA_SPARSE_MLA_DECODE_COMPOSITION=1 on an SM10x integration build with the checkpoint",
)

SERVED_SITE_SET = (
    "absorbed_query,shared_indexer_absorbed_query,rope_kv_append,"
    "attention_output_tail,post_attention_hop,attention_output_tail_prefetch,attention_output_into_hop"
)
STACK_SETS = [
    ("", "fixed"),
    ("absorbed_query,shared_indexer_absorbed_query,attention_output_tail", "fixed"),
    (SERVED_SITE_SET, "fixed"),
    (SERVED_SITE_SET, "indexer"),
]
ROWS = 6
TOP_K = 2048
FULL_LAYER, SHARED_LAYER = 2, 3
TAIL_FEATURES = (
    "attention_output_tail",
    "attention_output_tail_prefetch",
    "attention_output_into_hop",
)


def _run(features: str, topk_mode: str, out_dir: Path) -> dict:
    out = out_dir / f"stack_{features.replace(',', '_') or 'native'}_{topk_mode}.pt"
    env = dict(os.environ)
    env[plan_testlib.FAST_PATHS_ENV] = features
    env["GLM_MOE_DSA_SPARSE_MLA_DECODE_STACK_TOPK"] = topk_mode
    env["GLM52_NVFP4_CKPT"] = _CKPT
    env["TLLM_LOG_LEVEL"] = "INFO"
    env.setdefault("TRTLLM_ENABLE_PDL", "1")
    proc = subprocess.run(
        [sys.executable, str(_CASE), str(out)], env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout[-8000:] + proc.stderr[-8000:]
    result = torch.load(out, weights_only=False)
    log = (proc.stdout + proc.stderr).splitlines()
    result["capability_log"] = sorted(
        {line[line.index("[glm_moe_dsa]") :] for line in log if "[glm_moe_dsa]" in line}
    )
    return result


def _oracle(entry: dict) -> torch.Tensor:
    rows = entry["rows"].cuda().reshape(ROWS * TOP_K, -1)
    indices = entry["indices"].cuda()
    local = torch.arange(ROWS * TOP_K, device="cuda", dtype=torch.int32).view(ROWS, TOP_K)
    remapped = torch.where(indices >= 0, local, torch.full_like(local, -1))
    kv_len = int(entry["kv_lens"][0])
    limits = [kv_len - (ROWS - 1 - r) for r in range(ROWS)]
    return oracle_attention(
        entry["quant_q"].cuda(),
        rows,
        remapped,
        float(entry["bmm1"][0]),
        float(entry["bmm2"][0]),
        list_limit=limits,
    )


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _check_engagement(run: dict, selected: set, with_sparse_mla: bool) -> dict:
    eng = run["engagement"]
    counts = eng["counts"]
    eager_steps = 2 + 2 + 2 + 1  # inputs, pdl, serialized, before capture (warm-up excluded)
    checked = {}
    if "absorbed_query" in selected:
        checked["absorbed_query_full"] = counts.get(f"absorbed_query/layer{FULL_LAYER}", 0)
        assert checked["absorbed_query_full"] >= eager_steps, counts
    if "shared_indexer_absorbed_query" in selected:
        checked["shared_query_shared"] = counts.get(
            f"shared_indexer_absorbed_query/layer{SHARED_LAYER}", 0
        )
        assert checked["shared_query_shared"] >= eager_steps, counts
        # The shared layer takes shared_indexer_absorbed_query's packet, not absorbed_query's.
        assert counts.get(f"absorbed_query/layer{SHARED_LAYER}", 0) == 0, counts
    if "rope_kv_append" in selected:
        for layer in (FULL_LAYER, SHARED_LAYER):
            assert counts.get(f"rope_append_rope/layer{layer}", 0) >= eager_steps, counts
            assert counts.get(f"rope_append_bridge/layer{layer}", 0) >= eager_steps, counts
    if selected & set(TAIL_FEATURES):
        for layer in (FULL_LAYER, SHARED_LAYER):
            tail = eng["attn_tail_tails"][layer]
            assert tail is not None and tail["static_reason"] is None, tail
            assert tail["engaged_calls"] >= eager_steps, tail
            checked[f"attn_tail_layer{layer}"] = {
                "engaged_calls": tail["engaged_calls"],
                "early_rows": tail["config"]["early_rows"],
                "oproj_grid": tail["config"]["oproj_grid"],
            }
    sparse_mla_calls = eng["sparse_mla_decode"]["engaged_calls"]
    if with_sparse_mla:
        for layer in (FULL_LAYER, SHARED_LAYER):
            assert sparse_mla_calls.get(str(layer), 0) >= eager_steps, eng["sparse_mla_decode"]
    else:
        assert not sparse_mla_calls, eng["sparse_mla_decode"]
    checked["sparse_mla_engaged_calls"] = sparse_mla_calls
    return checked


def _replay_check(run: dict, layers) -> dict:
    """Compare every plain / serialized / graph step with the recorded eager step of its input."""
    reference = {e["input"]: e["layers"] for e in run["inputs"]}
    mismatches, index_changes, compared = [], [], 0
    for kind in ("pdl", "serialized", "graph"):
        for i, entry in enumerate(run[kind]):
            ref = reference[entry["input"]]
            for layer in layers:
                got = entry["layers"][layer]
                if not _same(got["indices"], ref[layer]["indices"]):
                    index_changes.append((kind, i, layer))
                for key in ("latent", "module_output"):
                    compared += 1
                    if not _same(got[key], ref[layer][key]):
                        mismatches.append((kind, i, layer, key))
    return {"compared": compared, "mismatches": mismatches, "index_changes": index_changes}


@pytest.mark.parametrize("neighbours,topk_mode", STACK_SETS)
def test_sparse_mla_stack_with_engaged_neighbours(neighbours, topk_mode, tmp_path):
    out_dir = Path(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_RESULTS", tmp_path))
    selected = {f for f in neighbours.split(",") if f}
    with_sparse_mla = ",".join(sorted(selected) + ["sparse_mla_decode"])
    base = _run(neighbours, topk_mode, out_dir)
    cand = _run(with_sparse_mla, topk_mode, out_dir)
    layers = cand["layers"]
    assert layers == [FULL_LAYER, SHARED_LAYER] and cand["shared_layers"] == [SHARED_LAYER]
    assert base["libraries"] == ["FallbackFmha"], base["libraries"]
    assert cand["libraries"] == ["SparseMlaDecodeFmha"], cand["libraries"]

    summary = {
        "neighbours": neighbours or "native",
        "topk_mode": topk_mode,
        "ctx": cand["ctx"],
        "trtllm_enable_pdl": os.environ.get("TRTLLM_ENABLE_PDL", "1"),
        "engagement_without_sparse_mla": _check_engagement(base, selected, False),
        "engagement_with_sparse_mla": _check_engagement(cand, selected, True),
        "capability_log_with_sparse_mla": cand["capability_log"],
        "steps": [],
    }
    for b, c in zip(base["inputs"], cand["inputs"]):
        for layer in layers:
            bl, cl = b["layers"][layer], c["layers"][layer]
            # The producers do not depend on the attention library.
            for key in ("quant_q", "bmm1", "bmm2", "kv_lens"):
                assert _same(bl[key], cl[key]), (neighbours, layer, key)
            same_order = _same(bl["indices"], cl["indices"])
            same_sets = torch.equal(
                bl["indices"].sort(dim=-1).values, cl["indices"].sort(dim=-1).values
            )
            if topk_mode == "fixed":
                assert same_order and _same(bl["rows"], cl["rows"]), (neighbours, layer)
            assert not bool((cl["indices"] < 0).any()), "a -1 index below the causal length"
            native_err = error_stats(bl["latent"].cuda().double(), _oracle(bl))
            sparse_mla_err = error_stats(cl["latent"].cuda().double(), _oracle(cl))
            module_diff = error_stats(cl["module_output"].double(), bl["module_output"].double())
            summary["steps"].append(
                {
                    "input": c["input"],
                    "layer": layer,
                    "indices_same_order": same_order,
                    "indices_same_sets": same_sets,
                    "native_latent_vs_oracle": native_err,
                    "sparse_mla_latent_vs_oracle": sparse_mla_err,
                    "module_output_sparse_mla_vs_native": module_diff,
                }
            )
            assert sparse_mla_err["nonfinite"] == native_err["nonfinite"] == 0
            assert sparse_mla_err["nrms"] <= 1.25 * native_err["nrms"] + 2e-4, (
                layer,
                sparse_mla_err,
                native_err,
            )
            assert sparse_mla_err["max_abs"] <= 2.0 * native_err["max_abs"] + 1e-3, (
                layer,
                sparse_mla_err,
            )
            if same_sets:
                assert module_diff["nonfinite"] == 0 and module_diff["nrms"] < 0.05, module_diff

    summary["replays_without_sparse_mla"] = _replay_check(base, layers)
    summary["replays_with_sparse_mla"] = _replay_check(cand, layers)
    summary["kernels_without_sparse_mla"] = {
        k: {f: v for f, v in s.items() if f != "attention"} for k, s in base["kernels"].items()
    }
    summary["kernels_with_sparse_mla"] = {
        k: {f: v for f, v in s.items() if f != "attention"} for k, s in cand["kernels"].items()
    }
    summary["rope_append_bridge_without_sparse_mla"] = base["rope_append_bridge"]
    summary["rope_append_bridge_with_sparse_mla"] = cand["rope_append_bridge"]
    record(
        "sparse_mla_stack_composition.jsonl", "stack_composition", json.loads(json.dumps(summary))
    )

    # sparse_mla_decode is deterministic: with it, every plain, serialized and replayed step
    # reproduces the recorded eager step bit for bit when it attended the same indices (always with
    # a fixed top-k; the native indexer may reorder or change its selection from step to step).
    replays = summary["replays_with_sparse_mla"]
    if topk_mode == "fixed":
        assert not replays["mismatches"], replays
    else:
        changed = {(k, i, layer) for k, i, layer in replays["index_changes"]}
        stable = [m for m in replays["mismatches"] if tuple(m[:3]) not in changed]
        assert not stable, replays

    # One attention kernel per DSA layer per step; the right library in each run.
    for run, sparse_mla_expected in ((base, False), (cand, True)):
        for kind, stats in run["kernels"].items():
            calls = stats["steps"] * stats["layers"]
            assert stats["sparse_mla_decode"] == (calls if sparse_mla_expected else 0), (
                kind,
                stats["sparse_mla_decode"],
                calls,
            )
            assert stats["native_sparse_fmha"] == (0 if sparse_mla_expected else calls), (
                kind,
                stats["native_sparse_fmha"],
                calls,
            )
            if "rope_kv_append" in selected:
                assert stats["rope_append_bridge"] >= stats["steps"], (
                    kind,
                    stats["rope_append_bridge"],
                )
                if topk_mode == "fixed":
                    assert stats["native_remap"] == 0, (kind, stats["native_remap"])
    # The attention_output_tail is sparse_mla_decode's programmatic dependent in the captured graph
    # and becomes resident well before sparse_mla_decode ends (sparse_mla_decode triggers before its
    # gathers complete). Without PDL the tail starts only after sparse_mla_decode completes (CUPTI
    # may still show a sub-microsecond overlap of the end and start stamps of back-to-back kernels,
    # so the lead is compared, not its sign).
    if selected & set(TAIL_FEATURES):
        graph = cand["kernels"]["graph"]
        sparse_mla_records = [a for a in graph["attention"] if a["kernel"] == "sparse_mla_decode"]
        assert all("_wuv_kernel" in (a["successor"] or "") for a in sparse_mla_records), graph[
            "successors"
        ]
        leads = graph["sparse_mla_successor_lead_us"]
        median_lead = leads[len(leads) // 2]
        if os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1":
            assert graph["sparse_mla_successor_early"] == len(sparse_mla_records), graph[
                "sparse_mla_successor_early"
            ]
            assert median_lead >= 1.0, leads
        else:
            assert median_lead < 0.5, leads

    # rope_kv_append's bridge as the producer of the indices: equal to the native remap, no -1 below
    # a row's causal length, and what the attention received.
    if "rope_kv_append" in selected:
        for run in (base, cand):
            reports = run["rope_append_bridge"]
            assert len(reports) == 2 * len(layers), reports
            for report in reports:
                assert report["equals_native_remap"], report
                assert sum(report["negative_below_causal"]) == 0, report
                assert report["attention_received_bridge_output"], report
