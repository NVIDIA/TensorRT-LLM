# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The phase predicate and the Target base class.

Import-only and allocation-free: this runs on a CI machine with no GPU, like
the rest of the claims suite. What it pins down is the one decision the whole
layer rests on -- which target a step goes to -- and the fact that a Target
cannot be instantiated without answering for both of its abstract methods.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._target import Phase, Target, phase_of


@pytest.mark.parametrize(
    "num_contexts, expected",
    [
        (0, Phase.DECODE),
        (1, Phase.PREFILL),
        (8, Phase.PREFILL),
    ],
    ids=["pure-decode", "one-context", "mixed-batch"],
)
def test_phase_is_decided_by_num_contexts_alone(num_contexts, expected):
    """num_contexts == 0 is the whole predicate.

    A mixed batch -- context requests *and* generation requests in one step --
    goes to PREFILL, which is why PREFILL is the general case and DECODE is the
    specialization. See the design document, section 2.2.
    """
    assert phase_of(SimpleNamespace(num_contexts=num_contexts)) is expected


def test_phase_predicate_reads_nothing_but_num_contexts():
    """Anything else it touched would be a field a decode-only CUDA graph is
    not guaranteed to carry.

    Recorded rather than inferred: a metadata stand-in that logs every
    attribute read, so the assertion names the exact read set instead of
    resting on "no exception was raised".
    """
    reads = []

    class Recording:
        num_contexts = 0

        def __getattribute__(self, name):
            reads.append(name)
            return object.__getattribute__(self, name)

    phase_of(Recording())
    assert reads == ["num_contexts"]


def test_a_target_cannot_skip_either_abstract_method():
    class Incomplete(Target):
        def step_args(self, md):
            return {}

    with pytest.raises(TypeError, match="forward"):
        Incomplete(core=object())


def test_a_target_keeps_the_core_by_reference():
    """By reference, not by snapshot: the core replaces `_rope` after a
    contract check regrows it, and a target that captured operands at
    construction would hold the stale table. See the design document, 2.3.
    """

    class Concrete(Target):
        def step_args(self, md):
            return {}

        def forward(self, attn_metadata, *args, **kwargs):
            return self.core.marker

    core = SimpleNamespace(marker="first")
    target = Concrete(core=core)
    core.marker = "second"
    assert target.forward(None) == "second"


def test_gpt_oss_core_declares_both_phases():
    """The table the dispatcher indexes. A core that shipped one target would
    route half its steps into a KeyError, which is worth catching without a GPU.
    """
    import ast
    from pathlib import Path

    import tensorrt_llm._torch._experimental.modeling_v2 as _mv2

    source = (
        Path(_mv2.__file__).resolve().parent
        / "models/gpt_oss/gpt_oss_120b__sm_103__tp1/modeling.py"
    ).read_text()
    classes = {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)}
    assert {"PrefillTarget", "DecodeTarget"} <= classes


def test_decode_step_args_state_the_zero_rather_than_reading_it():
    """The one place the two gpt_oss targets actually differ.

    Decode's routing already guarantees no context rows, so it says so instead
    of reading the field back. That is what makes the source gate in
    test_modeling_v2_claims.py meaningful on this target rather than vacuous.
    """
    import ast
    from pathlib import Path

    import tensorrt_llm._torch._experimental.modeling_v2 as _mv2

    source = (
        Path(_mv2.__file__).resolve().parent
        / "models/gpt_oss/gpt_oss_120b__sm_103__tp1/modeling.py"
    ).read_text()
    decode = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.ClassDef) and n.name == "DecodeTarget"
    )
    literals = {
        kw.arg: kw.value.value
        for call in ast.walk(decode)
        if isinstance(call, ast.Call)
        for kw in call.keywords
        if kw.arg in ("num_contexts", "num_ctx_tokens") and isinstance(kw.value, ast.Constant)
    }
    assert literals == {"num_contexts": 0, "num_ctx_tokens": 0}
