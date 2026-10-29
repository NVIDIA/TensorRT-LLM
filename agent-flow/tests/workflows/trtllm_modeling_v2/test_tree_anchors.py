# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchors: what the workflow reads off the engine's tree, found where it moved or reported missing."""

from __future__ import annotations

import copy
import dataclasses
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art, tree_anchors
from agent_flow.workflows.trtllm_modeling_v2.spec import lint

from .test_intake_lint import _GOOD

ANCHORS = tree_anchors.ANCHORS
_CONFIG = {
    "architectures": ["ToyForCausalLM"],
    "text_config": {
        "head_dim": 192,
        "num_attention_heads": 8,
        "hidden_size": 1536,
        "sparse_config": {"attention_impl": "sparse_gqa", "topk": 512},
    },
}
_FMHA_RULE = 'return {false, "HeadDimV must be 32, 64, 80, 128, or a multiple of 128"};\n'


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_a_directory_anchor_is_its_hint_while_the_hint_exists(tmp_path):
    hint = ANCHORS["attention_backends"].hint
    _write(tmp_path, f"{hint}/sparse.py", "def f(): ...\n")
    found = tree_anchors.resolve(tmp_path, "attention_backends")
    assert (found.paths, found.via) == ((hint,), "hint")


def test_a_moved_directory_is_found_by_what_it_defines_and_scanned_there(tmp_path):
    _write(
        tmp_path,
        "tensorrt_llm/_torch/attn/impl/trtllm.py",
        "class TrtllmAttention(AttentionBackend[M]):\n    pass\n",
    )
    _write(
        tmp_path,
        "tensorrt_llm/_torch/attn/impl/sparse/qsa_kernels.py",
        "def triton_qsa_paged_sparse_gqa(q):\n    pass\n",
    )
    _write(tmp_path, "tensorrt_llm/_torch/other.py", "backend = AttentionBackend\n")
    found = tree_anchors.resolve(tmp_path, "attention_backends")
    assert (found.paths, found.via) == (("tensorrt_llm/_torch/attn/impl",), "search")
    leads = prior_art.find_prior_art(tmp_path, _CONFIG).candidates
    assert [c.path for c in leads] == ["tensorrt_llm/_torch/attn/impl/sparse/qsa_kernels.py"]


def test_a_moved_definition_is_read_where_it_went_with_its_line(tmp_path):
    _write(
        tmp_path,
        "tensorrt_llm/_torch/loader/kv.py",
        '"""KV."""\n_VALID_KV_CACHE_DTYPES = ("fp8", "auto")\n',
    )
    moved = (("fp8", "auto"), "tensorrt_llm/_torch/loader/kv.py:2")
    assert prior_art.engine_kv_dtypes(tmp_path) == moved
    # A file left at the hint that no longer defines it is not taken for it.
    _write(tmp_path, ANCHORS["kv_cache_dtypes"].hint, "# moved to loader/kv.py\n")
    assert prior_art.engine_kv_dtypes(tmp_path) == moved


def test_a_rule_the_tree_states_nowhere_claims_no_limit_and_the_lint_says_so(tmp_path):
    (tmp_path / "cpp").mkdir()
    assert prior_art.fmha_rule(tmp_path) is None
    assert prior_art.engine_constraints(_CONFIG, tmp_path) == []
    missing = [
        f
        for f in lint.lint_task(copy.deepcopy(_GOOD), repo=tmp_path).findings
        if f.rule == "tree-anchor-missing"
    ]
    assert {f.severity for f in missing} == {lint.INFO}
    fmha = next(f for f in missing if "FMHA" in f.message)
    assert ANCHORS["fmha_head_dims"].hint in fmha.message and "not named" in fmha.message
    # Moved, the rule is read where it is now, and cited there.
    _write(tmp_path, "cpp/kernels/fmha/v2/kernelTable.h", _FMHA_RULE)
    (note,) = prior_art.engine_constraints(_CONFIG, tmp_path)
    assert "head dim 192" in note and "cpp/kernels/fmha/v2/kernelTable.h:1" in note


def test_the_map_says_where_each_anchor_is_moved_or_missing(tmp_path):
    perf = ANCHORS["perf_tests"].hint
    _write(tmp_path, f"{perf}/test_perf.py", "class PerfTestConfig:\n    pass\n")
    _write(tmp_path, "tests/integration/lists/l0_b200.yml", "l0_b200: []\n")
    text = tree_anchors.render_map(tmp_path)
    assert f"`{perf}`. Find:" in text
    assert "`tests/integration/lists` (moved from" in text
    assert "**the KV-cache dtypes the PyTorch loader accepts**: missing" in text


def test_preflight_writes_the_tree_map_and_notes_what_moved_or_is_missing(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    repo = tmp_path / "repo"
    _write(repo, ANCHORS["kv_cache_dtypes"].hint, '_VALID_KV_CACHE_DTYPES = ("auto",)\n')
    workflow = SimpleNamespace(task_data={"trtllm_repo_path": str(repo)}, workspace=tmp_path)
    note = bootstrap._take_tree_map(workflow)
    written = (tmp_path / bootstrap.TREE_MAP_FILENAME).read_text()
    assert written.startswith("# Where things are in this checkout")
    assert note.startswith("tree map (missing: ") and "KV-cache" not in note
    # No checkout, no map.
    workflow.task_data = {}
    assert bootstrap._take_tree_map(workflow) == ""
    assert not (tmp_path / bootstrap.TREE_MAP_FILENAME).exists()


def test_every_anchor_is_found_by_what_identifies_it_in_this_checkout():
    """Inside a TensorRT-LLM checkout, a search that ignores each hint finds what the hint points at.

    The search is what a run relies on once the tree moves, so it is what this
    holds. A hint this tree lacks (an older or newer layout) only warns.
    """
    root = Path(__file__).resolve().parents[4]
    if not (root / "tensorrt_llm").is_dir():
        pytest.skip("not inside a TensorRT-LLM checkout")
    unfindable = []
    for key, anchor in ANCHORS.items():
        here = tree_anchors.resolve(root, anchor)
        if here.via != "hint":
            where = f"found at {list(here.paths[:3])}" if here.found else "missing"
            warnings.warn(f"anchor {key}: no {anchor.hint} in this tree, {where}", stacklevel=1)
            continue
        moved = dataclasses.replace(anchor, hint=f"{anchor.hint}.moved")
        found = tree_anchors.resolve(root, moved).paths
        hint = anchor.hint
        if not any(
            p == hint or p.startswith(hint + "/") or hint.startswith(p + "/") for p in found
        ):
            unfindable.append(f"{key}: a search found {list(found[:3])}")
    assert not unfindable, f"anchors a search cannot find by what identifies them: {unfindable}"
