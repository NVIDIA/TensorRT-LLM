# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the survey reads first, and the review that holds its draft to the tree's prior art."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.operators import prior_art, survey_review, tree_anchors
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import (
    CatalogEntry,
    Inventory,
    format_surface,
    surface_names,
)

FMHA_HEADER = tree_anchors.ANCHORS["fmha_head_dims"].hint
_CONFIG = {"text_config": {"sparse_config": {"attention_impl": "sparse_gqa", "topk": 512}}}

_LEAD = prior_art.Candidate(
    path="tensorrt_llm/_torch/attention/backends/sparse/qsa/kernels.py",
    kind="attention backend",
    matched=("gqa", "sparse", "topk"),
    score=9.0,
    names=("triton_qsa_paged_sparse_gqa",),
)
_FOUND = prior_art.PriorArt(words=("gqa", "sparse", "topk"), candidates=[_LEAD])
_LIMIT = prior_art.EngineLimit(
    "`text_config.head_dim` = 192: the trtllm-gen FMHA has no kernel for head dim 192. Pad.",
    "192",
    ("256", "native"),
)


def _inventory() -> Inventory:
    # Fifty ops of a target whose name sorts first, then the one this checkpoint wants.
    entries = [
        CatalogEntry(path=f"moe/router_{i:02d}.py", scope="aaa_family/aaa_target")
        for i in range(50)
    ]
    entries.append(
        CatalogEntry(path="attention/qsa_paged_sparse_gqa.py", scope="zzz_family/zzz_target")
    )
    return Inventory(arch="sm_103", catalog=entries)


def test_the_surface_lists_what_the_checkpoint_shares_first():
    inventory = _inventory()
    wanted = "zzz_family/zzz_target/ops/attention/qsa_paged_sparse_gqa"
    # Alphabetically the op sorts last, past the cut.
    assert wanted not in format_surface(inventory, limit=40)
    rank = prior_art.ranker(_CONFIG, surface_names(inventory))
    ranked = format_surface(inventory, limit=40, rank=rank)
    assert wanted in ranked and "most first" in ranked
    assert ranked.index(wanted) < ranked.index("aaa_family/aaa_target/ops/moe/router_00")


def test_the_survey_reads_the_leads_and_the_limits():
    block = prior_art.survey_block(_FOUND, [_LIMIT])
    assert block.startswith(prior_art.PRIOR_ART_HEADING)
    assert "head dim 192" in block and "qsa/kernels.py" in block
    assert prior_art.survey_block(None, []) == ""


def _gap(**row):
    return {
        "step": "sparse GQA attention over the selected regions",
        "values": {"topk": 512},
        "state": "GAP",
        "tier": 5,
        "tried": "no trtllm op or _torch/modules callable serves it",
        **row,
    }


def test_a_gap_that_writes_new_code_beside_a_lead_it_does_not_name_is_a_finding():
    findings = survey_review.unconsidered_leads({"steps": [_gap()]}, _FOUND)
    assert len(findings) == 1 and "qsa/kernels.py" in findings[0] and "tier 5" in findings[0]
    # Named with its reason, or reused at a cheaper rung, it stands.
    named = _gap(tried="qsa/kernels.py assumes power-of-two heads and a different index layout")
    assert survey_review.unconsidered_leads({"steps": [named]}, _FOUND) == []
    reused = _gap(tier=2, candidate="triton_qsa_paged_sparse_gqa", tried="")
    assert survey_review.unconsidered_leads({"steps": [reused]}, _FOUND) == []
    # A step that shares one word with the lead is not held to it.
    other = _gap(step="sparse MoE routing over experts", values={})
    assert survey_review.unconsidered_leads({"steps": [other]}, _FOUND) == []


def test_a_plan_states_its_design_for_each_engine_limit():
    assert survey_review.unaddressed_limits("Head dim 192: zero-pad to 256.", [_LIMIT]) == []
    assert survey_review.unaddressed_limits("- head_dim 192 needs a native kernel", [_LIMIT]) == []
    # The values apart do not say it: the proxy width is not the attention's way out.
    apart = "Attention head dim is 192.\nThe indexer proxy dim is 256."
    findings = survey_review.unaddressed_limits(apart, [_LIMIT])
    assert len(findings) == 1 and "192" in findings[0]


def test_the_review_runs_once_and_says_what_to_redraft():
    findings = survey_review.review_survey({"steps": [_gap()]}, "", _FOUND, [_LIMIT])
    assert len(findings) == 2
    review = survey_review.render_review(findings)
    assert review.startswith("## Survey review: re-draft") and "runs once" in review


def test_preflight_writes_the_prior_art_every_role_can_read(tmp_path):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    repo = tmp_path / "repo"
    backend = repo / "tensorrt_llm/_torch/attention/backends/sparse/qsa"
    backend.mkdir(parents=True)
    (backend / "kernels.py").write_text("def triton_qsa_paged_sparse_gqa(q):\n    pass\n")
    header = repo / FMHA_HEADER
    header.parent.mkdir(parents=True)
    header.write_text('"HeadDimV must be 32, 64, 80, 128, or a multiple of 128"\n')
    config = {"text_config": {**_CONFIG["text_config"], "head_dim": 192}}
    workflow = SimpleNamespace(task_data={"trtllm_repo_path": str(repo)}, workspace=tmp_path)
    bootstrap._take_prior_art(workflow, config)
    written = (tmp_path / bootstrap.PRIOR_ART_FILENAME).read_text()
    assert "qsa/kernels.py" in written and "head dim 192" in written
    assert workflow._prior_art_block.startswith(prior_art.PRIOR_ART_HEADING)
    assert [limit.subject for limit in workflow._engine_limits] == ["192"]
    # A checkpoint this tree knows nothing about leaves neither the file nor the section.
    bootstrap._take_prior_art(workflow, {})
    assert workflow._prior_art_block == ""
    assert not (tmp_path / bootstrap.PRIOR_ART_FILENAME).exists()
