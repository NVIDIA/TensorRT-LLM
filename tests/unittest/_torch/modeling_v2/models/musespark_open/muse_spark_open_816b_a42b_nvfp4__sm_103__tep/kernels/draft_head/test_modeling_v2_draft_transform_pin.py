# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The built-in DFlash forward's statements the draft head's deferred logit transform relies on.

The draft head (``kernels/draft_head/head.py``) hands the drafter's bf16 shard
logits through the drafter's ``apply_draft_logit_transform`` untransformed and
applies the transform after its top-k, in ``ShardedDraftHeadDFlashWorker``'s
``_refine_block_logits`` and selector. That is the built-in computation only
while nothing between the transform call and ``_refine_block_logits`` reads the
logits, and only while ``_refine_block_logits`` runs before the selector and
the sampler. This test reads ``DFlashWorker._forward_impl``'s source (no
import, no device) and fails when that sequence changes, so a change to it is
reviewed against the draft head instead of silently reaching a reader with
untransformed logits.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

#: The statements from the drafter's logits processor to the selector, as ``ast.unparse`` prints them.
SEQUENCE = (
    "gen_logits = draft_model.logits_processor(gen_hidden_states, draft_model.lm_head, "
    "attn_metadata, True)",
    "apply_logit_transform = getattr(draft_model, 'apply_draft_logit_transform', None)",
    "if apply_logit_transform is not None:\n    gen_logits = apply_logit_transform(gen_logits)",
    "vocab_size = gen_logits.shape[-1]",
    "gen_logits = gen_logits.reshape(num_gens, K, vocab_size)",
    "gen_logits = self._refine_block_logits(draft_model, gen_logits, inputs, spec_metadata)",
    "has_selector = getattr(draft_model, 'has_candidate_selector', False)",
)


def _source(relative: str) -> str:
    spec = importlib.util.find_spec("tensorrt_llm")
    assert spec is not None and spec.submodule_search_locations, "tensorrt_llm is not importable"
    return (Path(next(iter(spec.submodule_search_locations))) / relative).read_text()


def _forward_impl() -> ast.FunctionDef:
    tree = ast.parse(_source("_torch/speculative/dflash.py"))
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker")
    return next(
        n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_impl"
    )


def _bodies(node: ast.AST):
    for child in ast.walk(node):
        for name in ("body", "orelse"):
            body = getattr(child, name, None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                yield body


def test_nothing_reads_the_draft_logits_between_the_transform_and_refine():
    forward = _forward_impl()
    found = []
    for body in _bodies(forward):
        printed = [ast.unparse(s) for s in body]
        if SEQUENCE[0] in printed:
            start = printed.index(SEQUENCE[0])
            found.append(tuple(printed[start : start + len(SEQUENCE)]))
    assert len(found) == 1, (
        "the drafter's logits processor call is not where the draft head expects it"
    )
    assert found[0] == SEQUENCE, (
        "DFlashWorker._forward_impl changed between the drafter's logits processor and "
        "_refine_block_logits; check the draft head's deferred transform against it:\n"
        + "\n".join(found[0])
    )


def test_refine_runs_before_the_selector_and_the_samplers():
    forward = _forward_impl()
    calls = [
        (node.lineno, node.func.attr)
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr
        in (
            "_refine_block_logits",
            "_apply_dflash2_selector",
            "_sample_dflash2_block",
            "sample_draft_tokens",
        )
    ]
    names = [name for _, name in sorted(calls)]
    assert names.count("_refine_block_logits") == 1, names
    assert names[0] == "_refine_block_logits", names


def test_the_transform_has_one_caller():
    """The deferring transform is consulted only by that call."""
    root = Path(next(iter(importlib.util.find_spec("tensorrt_llm").submodule_search_locations)))
    callers = sorted(
        str(path.relative_to(root))
        for path in (root / "_torch").rglob("*.py")
        if "apply_draft_logit_transform" in path.read_text(encoding="utf-8", errors="ignore")
        and "_experimental" not in path.parts
    )
    assert callers == ["_torch/models/modeling_dflash.py", "_torch/speculative/dflash.py"], callers


#: This target's own sources, relative to the ``tensorrt_llm`` package.
TARGET = (
    "_torch/_experimental/modeling_v2/models/musespark_open/"
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)


def test_the_target_composes_no_other_reader_of_deferred_logits():
    """Only the draft head and the fold take deferred logits; only the draft head wraps the processor.

    A worker class of another fast path composed above the draft head that
    defined ``_refine_block_logits`` or ``_apply_dflash2_selector`` would see the
    deferred bf16 logits first (the draft head then stops deferring for it:
    ``head._defers_transform``), and a fast path that replaced the drafter's
    logits processor would end the deferral without a word. Read off the
    target's sources, so a new one of either is reviewed against the draft head.
    """
    root = Path(next(iter(importlib.util.find_spec("tensorrt_llm").submodule_search_locations)))
    owners = {"_refine_block_logits": [], "_apply_dflash2_selector": [], "processor": []}
    for path in sorted((root / TARGET).rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(root / TARGET))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name in owners:
                owners[node.name].append(rel)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == "logits_processor"
                for target in node.targets
            ):
                owners["processor"].append(rel)
    assert owners["_refine_block_logits"] == ["kernels/draft_head/head.py"], owners
    assert sorted(owners["_apply_dflash2_selector"]) == [
        "kernels/draft_head/head.py",
        "kernels/verify_fold/fold.py",
    ], owners
    assert owners["processor"] == ["kernels/draft_head/head.py"], owners
