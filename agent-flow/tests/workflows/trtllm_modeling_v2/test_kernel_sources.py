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

from agent_flow.workflows.trtllm_modeling_v2.onboard.runtime_map import (
    KERNEL_SOURCES_PATH,
    RuntimeIndexError,
    format_kernel_sources,
    load_kernel_sources,
    source_for_stem,
)
from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import (
    _VALID_TIERS,
    KERNEL_LIBRARIES,
    TIER_KERNEL_LIBRARY,
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
        assert source.tier in _VALID_TIERS, f"{source.name}: tier={source.tier}"
        assert len(source.availability.split()) >= 5, f"{source.name}: availability too thin"
        assert len(source.certifying.split()) >= 5, f"{source.name}: certifying too thin"
        assert source.pointers or source.package, f"{source.name}: points nowhere"


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
    unclassified = sorted(
        py.stem
        for py in CUSTOM_OPS.glob("*.py")
        if py.stem not in NOT_A_SOURCE
        and source_for_stem(py.stem, sources, text=py.read_text(encoding="utf-8")) is None
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
    """End to end against the ops the catalog and the targets' ``ops/`` actually wrap.

    This is the join the smith phase depends on: an op name, discovered on
    the host, resolved to the provenance that decides its receipt.
    """
    modeling_v2 = REPO_ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2"
    entry_files = [
        *modeling_v2.glob("catalog/*/*.py"),
        *modeling_v2.glob("models/*/*/ops/*/*.py"),
    ]
    if not entry_files:
        pytest.skip("no modeling_v2 catalog in this checkout")

    registrations = {}
    texts = {}
    for py in sorted(CUSTOM_OPS.glob("*.py")):
        texts[py.stem] = py.read_text(encoding="utf-8")
        for name in re.findall(r'"trtllm::([a-zA-Z0-9_]+)', texts[py.stem]):
            registrations.setdefault(name, py.stem)

    ops = sorted(
        {
            op
            for py in entry_files
            for op in re.findall(
                r"torch\.ops\.trtllm\.([a-zA-Z0-9_]+)", py.read_text(encoding="utf-8")
            )
        }
    )
    assert ops, "no trtllm ops in the catalog entries"

    unresolved = []
    for op in ops:
        stem = registrations.get(op, "cpp_thop")
        if stem in NOT_A_SOURCE:
            continue  # a registration mechanism; provenance is elsewhere
        if source_for_stem(stem, sources, text=texts.get(stem)) is None:
            unresolved.append(f"{op} (registered by {stem})")
    assert not unresolved, "catalog ops whose provenance cannot be resolved:\n  " + "\n  ".join(
        unresolved
    )


_TRITON_OP_FILE = """
import torch
import triton


@triton.jit
def _kernel(x_ptr):
    pass


@torch.library.custom_op("trtllm::new_triton_op", mutates_args=())
def new_triton_op(x: torch.Tensor) -> torch.Tensor:
    return x
"""


def test_a_triton_custom_op_file_no_path_names_is_claimed_by_content(sources):
    """A model's onboarding adds these one op at a time, under names nobody predicted.

    If the registry had to name each one, every such addition would fail this
    suite until the registry was edited to match the model's tree.
    """
    source = source_for_stem("new_triton_op", sources, text=_TRITON_OP_FILE)
    assert source is not None and source.name == "triton_custom_op"


def test_a_file_with_only_one_marker_is_not_claimed(sources):
    plain_op = _TRITON_OP_FILE.replace("@triton.jit", "")
    assert source_for_stem("new_plain_op", sources, text=plain_op) is None


def test_a_named_path_wins_over_content(sources):
    """Content is the fallback for unnamed files, never a reclassification."""
    source = source_for_stem("cute_dsl_custom_ops", sources, text=_TRITON_OP_FILE)
    assert source.name == "cute_dsl"


def test_the_rendered_block_names_the_content_markers(sources):
    rendered = sources["triton_custom_op"].render()
    assert "`@triton.jit` and `torch.library.custom_op`" in rendered


def test_schema_rejects_content_markers_that_are_not_strings(tmp_path):
    bad = tmp_path / "k.yaml"
    bad.write_text(
        "x:\n  summary: s\n  tier: 1\n  availability: a b c d e\n  certifying: a b c d e\n"
        "  registers: [a.py]\n  claims_files_containing: [1]\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeIndexError, match="claims_files_containing"):
        load_kernel_sources(bad)


def test_the_cpp_plane_is_the_conclusion_for_an_unclaimed_op(sources):
    """Not a fallback: an op no Python file registers IS a C++ op."""
    assert source_for_stem("cpp_thop", sources) is not None
    assert source_for_stem("cpp_custom_ops", sources).name == "cpp_thop"


def test_flashinfer_is_marked_as_the_conditional_one(sources):
    """The only source whose absence is an environment fact, not a build one."""
    availability = sources["flashinfer"].availability
    assert "IS_FLASHINFER_AVAILABLE" in availability
    assert "version" in sources["flashinfer"].certifying  # no version pins


def test_flashinfer_has_a_direct_api_source_on_the_library_rung(sources):
    """FlashInfer kernels no trtllm op wraps are a source of their own.

    Without it the only FlashInfer source would be the trtllm-wrapped one, so a
    smith would have no provenance to name for `flashinfer.<module>.<name>` and
    the library's unwrapped kernels would read as nothing the engine supplies.
    """
    source = sources["flashinfer_api"]
    assert source.tier == TIER_KERNEL_LIBRARY
    assert source.package == "flashinfer-python"
    assert "backend" in source.certifying and "version" in source.certifying
    assert "`flashinfer-python`, pinned in `requirements.txt`" in source.render()
    assert {s.package.split("-")[0] for s in sources.values() if s.package} <= set(KERNEL_LIBRARIES)


@in_checkout
def test_every_package_source_is_pinned_by_the_engine(sources):
    """The library rung reuses what the engine ships; it never adds a dependency."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pinned = {
        re.split(r"[=<>~!\[; ]", line.strip(), maxsplit=1)[0].lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    }
    unpinned = [s.name for s in sources.values() if s.package and s.package.lower() not in pinned]
    assert not unpinned, f"package sources the engine does not depend on: {unpinned}"


def test_schema_accepts_a_package_in_place_of_tree_paths(tmp_path):
    good = tmp_path / "k.yaml"
    good.write_text(
        "lib:\n  summary: s\n  tier: 3\n  availability: a b c d e\n  certifying: a b c d e\n"
        "  package: some-lib\n",
        encoding="utf-8",
    )
    assert load_kernel_sources(good)["lib"].package == "some-lib"
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "lib:\n  summary: s\n  tier: 3\n  availability: a b c d e\n  certifying: a b c d e\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeIndexError, match="'registers', 'implementation' or 'package'"):
        load_kernel_sources(bad)


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
    from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem

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
    from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem

    from .test_workflow import _workflow

    workflow = _workflow(tmp_path)
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    assert workflow._smith_prompt(QueueItem(n=1, mode="onboard", target="x", tier=1))
