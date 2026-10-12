# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kernel-source registry, held to the tree it describes.

Same contract as the runtime map: pointers resolve, no transcribed code, and
the claims that matter are checked rather than trusted. The one addition is a
coverage check in both directions --- a registration file the registry does
not know about is a kernel source an agent will meet and cannot classify.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_flow.workflows.staircase_onboard.runtime import (
    KERNEL_SOURCES_PATH,
    RuntimeIndexError,
    format_kernel_sources,
    load_kernel_sources,
    source_for_stem,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CUSTOM_OPS = REPO_ROOT / "tensorrt_llm/_torch/custom_ops"
in_checkout = pytest.mark.skipif(
    not CUSTOM_OPS.is_dir(), reason="not inside a TensorRT-LLM checkout"
)

#: Registration mechanisms rather than kernel sources. An op attributed to
#: one of these has a real provenance elsewhere, and the registry says so in
#: prose rather than inventing an entry for it.
NOT_A_SOURCE = {"__init__", "fast_custom_op", "cutedsl_matmul_heuristics"}


@pytest.fixture(scope="module")
def sources():
    return load_kernel_sources()


def test_every_source_declares_what_certifying_one_requires(sources):
    """Provenance decides the shape of a receipt; that is the point of the file."""
    for source in sources.values():
        assert source.tier in (1, 2, 3, 4), f"{source.name}: tier={source.tier}"
        assert len(source.availability.split()) >= 5, f"{source.name}: availability too thin"
        assert len(source.certifying.split()) >= 5, f"{source.name}: certifying too thin"
        assert source.pointers, f"{source.name}: points nowhere"


@in_checkout
def test_every_path_resolves(sources):
    missing = [
        f"{source.name}: {path}"
        for source in sources.values()
        for path in source.pointers
        if not (REPO_ROOT / path).exists()
    ]
    assert not missing, "kernel_sources.yaml points at paths that do not exist:\n  " + "\n  ".join(
        missing
    )


@in_checkout
def test_every_registration_file_in_the_tree_is_classified(sources):
    """The direction that actually rots.

    A new `custom_ops/<something>_custom_ops.py` lands and nothing tells an
    agent what kind of kernel it holds. This fails when that happens.
    """
    known = {Path(path).stem for source in sources.values() for path in source.registers}
    unclassified = sorted(
        py.stem
        for py in CUSTOM_OPS.glob("*.py")
        if py.stem not in known and py.stem not in NOT_A_SOURCE
    )
    assert not unclassified, (
        "custom_ops files no kernel source claims --- an agent meeting an op "
        f"from one of these cannot classify it: {unclassified}"
    )


@in_checkout
def test_the_two_registration_planes_are_both_covered(sources):
    """Missing either plane makes attribution silently wrong for ~a third of ops."""
    registers = {p for s in sources.values() for p in s.registers}
    assert any("custom_ops" in p for p in registers), "the Python plane is unrepresented"
    assert any("cpp/tensorrt_llm/thop" in p for p in registers), "the C++ plane is unrepresented"


@in_checkout
def test_attribution_resolves_real_catalog_ops_to_real_sources(sources):
    """End to end against the ops the catalog actually wraps.

    This is the join the smith phase depends on: an op name, discovered on
    the host, resolved to the provenance that decides its receipt.
    """
    import yaml

    index = REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml"
    if not index.is_file():
        pytest.skip("catalog index not present")

    registrations = {}
    for py in sorted(CUSTOM_OPS.glob("*.py")):
        for name in re.findall(r'"trtllm::([a-zA-Z0-9_]+)', py.read_text(encoding="utf-8")):
            registrations.setdefault(name, py.stem)

    entries = yaml.safe_load(index.read_text(encoding="utf-8"))["entries"]
    ops = [
        e["impl"].rsplit(".", 1)[-1]
        for e in entries
        if str(e.get("impl", "")).startswith("torch.ops.trtllm.")
    ]
    assert ops, "no trtllm ops in the catalog index"

    unresolved = []
    for op in ops:
        stem = registrations.get(op, "cpp_thop")
        if stem in NOT_A_SOURCE:
            continue  # a registration mechanism; provenance is elsewhere
        if source_for_stem(stem, sources) is None:
            unresolved.append(f"{op} (registered by {stem})")
    assert not unresolved, "catalog ops whose provenance cannot be resolved:\n  " + "\n  ".join(
        unresolved
    )


def test_the_cpp_plane_is_the_conclusion_for_an_unclaimed_op(sources):
    """Not a fallback: an op no Python file registers IS a C++ op."""
    assert source_for_stem("cpp_thop", sources) is not None
    assert source_for_stem("cpp_custom_ops", sources).name == "cpp_thop"


def test_flashinfer_is_marked_as_the_conditional_one(sources):
    """The only source whose absence is an environment fact, not a build one."""
    availability = sources["flashinfer"].availability
    assert "IS_FLASHINFER_AVAILABLE" in availability
    assert "version" in sources["flashinfer"].certifying  # no version pins


def test_vendored_triton_is_outside_the_op_plane(sources):
    """It is called as Python, so an op inventory cannot see it at all."""
    triton = sources["vendored_triton"]
    assert not triton.registers
    assert triton.tier == 2
    assert "invisible to an op" in " ".join(triton.summary.split())


def test_the_registry_carries_no_code():
    text = KERNEL_SOURCES_PATH.read_text(encoding="utf-8")
    assert "```" not in text
    from .test_skills import BARE_LINE_REFERENCE, LINE_CITATION

    assert not LINE_CITATION.findall(text) + BARE_LINE_REFERENCE.findall(text)


def test_schema_rejects_a_source_that_points_nowhere(tmp_path):
    bad = tmp_path / "k.yaml"
    bad.write_text(
        "x:\n  summary: s\n  tier: 1\n  availability: a b c d e\n  certifying: a b c d e\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeIndexError, match="at least one of"):
        load_kernel_sources(bad)


def test_the_rendered_block_explains_why_provenance_matters(sources):
    block = format_kernel_sources(sources)
    assert "registration planes" in block
    assert "certifying one" in block
    assert "tier 2" in block  # vendored_triton renders its tier


def test_the_registry_reaches_the_smith(tmp_path):
    """The role that decides where a computation comes from.

    A smith is dispatched with a tier; the registry turns "tier 1" into "and
    here is what certifying a CuTe DSL kernel actually requires".
    """
    from agent_flow.workflows.staircase_onboard.vocabulary import QueueItem

    from .test_workflow import _workflow

    workflow = _workflow(tmp_path)
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    workflow._kernel_sources_block = format_kernel_sources(load_kernel_sources())
    prompt = workflow._smith_prompt(QueueItem(n=1, mode="onboard", target="x", tier=1))

    assert "Where kernels come from" in prompt
    assert "cute_dsl" in prompt and "flashinfer" in prompt
    assert "registration planes" in prompt


def test_a_run_without_a_survey_still_dispatches(tmp_path):
    """The block defaults empty; a smith must not depend on it existing."""
    from agent_flow.workflows.staircase_onboard.vocabulary import QueueItem

    from .test_workflow import _workflow

    workflow = _workflow(tmp_path)
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    assert workflow._smith_prompt(QueueItem(n=1, mode="onboard", target="x", tier=1))
