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
"""sparse_mla_decode composed with the packages around it at the MLA / DSA call sites.

Each package set runs ``sparse_mla_case.py`` in a fresh interpreter, which installs the
plan of that set, on the same deterministic layer and inputs:

* producers (absorbed_query, absorbed_query + shared_indexer_absorbed_query, rope_kv_append) feed
  the query and indices; for every set ``X`` the FMHA inputs of ``X`` and ``X + sparse_mla_decode``
  must be bit-identical, and the latent output of each library is checked against an FP64 oracle
  with the frozen error budget;
* the consumer (attention_output_tail; attention_output_tail_prefetch where integrated) may start
  before sparse_mla_decode finishes (sparse_mla_decode triggers its dependents before its gathers
  complete): within each run the eager step, the same step with a device synchronization right after
  the sparse_mla_decode launch, and replays of a captured graph over alternating inputs must give
  bit-identical latent and module outputs whenever they attended the same
  global indices.

Every set runs with a fixed generation top-k (deterministic indices, so every
replay is compared bit for bit); sets with rope_kv_append also run with the unmodified
indexer, so that rope_kv_append's top-k bridge, where it engages, produces the indices
sparse_mla_decode reads (the native indexer may return its selection in another order from run
to run, so replays are compared only where the indices match). The capability
decisions and reasons the packages log are recorded.

Runs only where the installed build integrates the packages (an integration
image); a package that declines the layer is reported, not failed.
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

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_COMPOSITION") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available(),
    reason="set GLM_MOE_DSA_SPARSE_MLA_DECODE_COMPOSITION=1 on an SM10x integration build with the "
    "sparse_mla_decode kernel",
)

_CASE = Path(__file__).resolve().parent / "sparse_mla_case.py"
# Producers of the query/indices (absorbed_query, absorbed_query + shared_indexer_absorbed_query,
# rope_kv_append RoPE/append and top-k bridge), the consumer tail (attention_output_tail,
# attention_output_tail + attention_output_tail_prefetch) and one set with all of them; (set, top-k
# mode).
NEIGHBOUR_SETS = [
    ("", "fixed"),
    ("absorbed_query", "fixed"),
    ("absorbed_query,shared_indexer_absorbed_query", "fixed"),
    ("rope_kv_append", "fixed"),
    ("rope_kv_append", "indexer"),
    ("attention_output_tail", "fixed"),
    ("attention_output_tail,attention_output_tail_prefetch", "fixed"),
    ("absorbed_query,shared_indexer_absorbed_query,rope_kv_append,attention_output_tail", "fixed"),
    (
        "absorbed_query,shared_indexer_absorbed_query,rope_kv_append,attention_output_tail",
        "indexer",
    ),
]
ROWS = 6
TOP_K = 2048


def _run(features: str, topk_mode: str, out_dir: Path) -> dict:
    out = out_dir / f"mla_{features.replace(',', '_') or 'native'}_{topk_mode}.pt"
    env = dict(os.environ)
    env[plan_testlib.FAST_PATHS_ENV] = features
    env["GLM_MOE_DSA_SPARSE_MLA_DECODE_MLA_TOPK"] = topk_mode
    env["TLLM_LOG_LEVEL"] = "INFO"
    proc = subprocess.run(
        [sys.executable, str(_CASE), str(out)], env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-6000:]
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


@pytest.mark.parametrize("neighbours,topk_mode", NEIGHBOUR_SETS)
def test_sparse_mla_with_neighbours(neighbours, topk_mode, tmp_path):
    out_dir = Path(os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_RESULTS", tmp_path))
    with_sparse_mla = ",".join([f for f in neighbours.split(",") if f] + ["sparse_mla_decode"])
    base = _run(neighbours, topk_mode, out_dir)
    cand = _run(with_sparse_mla, topk_mode, out_dir)
    assert base["libraries"] == ["FallbackFmha"], base["libraries"]
    assert cand["libraries"] == ["SparseMlaDecodeFmha"], cand["libraries"]

    summary = {
        "neighbours": neighbours or "native",
        "topk_mode": cand["topk_mode"],
        "capability_without_sparse_mla": base["capability"],
        "capability_with_sparse_mla": cand["capability"],
        "capability_log_with_sparse_mla": cand["capability_log"],
        "steps": [],
    }
    assert base["topk_mode"] == cand["topk_mode"]
    for b, c in zip(base["eager"], cand["eager"]):
        # The producers do not depend on the consumer: identical FMHA inputs.
        for key in ("quant_q", "bmm1", "bmm2", "kv_lens"):
            assert _same(b[key], c[key]), (neighbours, key)
        same_order = _same(b["indices"], c["indices"])
        if not same_order:
            # Only an unmodified indexer may return the same selection in another order.
            assert cand["topk_mode"] == "indexer", neighbours
            assert torch.equal(b["indices"].sort(dim=-1).values, c["indices"].sort(dim=-1).values)
        else:
            assert _same(b["rows"], c["rows"]), (neighbours, "rows")
        assert not bool((c["indices"] < 0).any()), "an index below the causal length is -1"
        oracle = _oracle(c)
        native_err = error_stats(b["latent"].cuda().double(), oracle)
        sparse_mla_err = error_stats(c["latent"].cuda().double(), oracle)
        module_diff = error_stats(c["module_output"].double(), b["module_output"].double())
        summary["steps"].append(
            {
                "input": c["input"],
                "indices_same_order": same_order,
                "native_latent_vs_oracle": native_err,
                "sparse_mla_latent_vs_oracle": sparse_mla_err,
                "module_output_sparse_mla_vs_native": module_diff,
            }
        )
        assert sparse_mla_err["nonfinite"] == native_err["nonfinite"] == 0
        assert sparse_mla_err["nrms"] <= 1.25 * native_err["nrms"] + 2e-4, (
            sparse_mla_err,
            native_err,
        )
        assert sparse_mla_err["max_abs"] <= 2.0 * native_err["max_abs"] + 1e-3, (
            sparse_mla_err,
            native_err,
        )
        assert module_diff["nonfinite"] == 0 and module_diff["nrms"] < 0.05, module_diff

    # Programmatic overlap with the next kernel and graph replay change nothing.
    for run, label in ((base, "without_sparse_mla"), (cand, "with_sparse_mla")):
        eager = {e["input"]: e for e in run["eager"]}
        mismatches = []
        index_changes = []
        for kind in ("serialized", "graph"):
            for i, entry in enumerate(run[kind]):
                ref = eager[entry["input"]]
                if not _same(entry["indices"], ref["indices"]):
                    index_changes.append((kind, i))
                    continue
                for key in ("latent", "module_output"):
                    if not _same(entry[key], ref[key]):
                        mismatches.append((kind, i, key))
        summary[f"replay_mismatches_{label}"] = mismatches
        summary[f"replay_index_changes_{label}"] = index_changes
    record(
        "sparse_mla_composition.jsonl",
        "mla_composition",
        json.loads(json.dumps(summary, default=str)),
    )
    # sparse_mla_decode is deterministic, so with it every replay and the serialized step that
    # attended the same indices must match exactly; a fixed top-k never changes the indices.
    assert not summary["replay_mismatches_with_sparse_mla"], summary[
        "replay_mismatches_with_sparse_mla"
    ]
    if cand["topk_mode"] == "fixed":
        assert not summary["replay_index_changes_with_sparse_mla"], summary[
            "replay_index_changes_with_sparse_mla"
        ]
