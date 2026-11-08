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
"""GPU-free sanity guards for the in-process NeMo-Skills evaluators.

These do NOT load a model or need `nemo_skills` installed -- they pin the wiring
and pure-Python logic (class contract, CLI options, reasoning strip, aggregation,
judgement parsers) so refactors that break the eval fail fast in plain CI. The
heavyweight accuracy runs live behind `trtllm-eval <bench>` (see
examples/trtllm-eval/README.md), not here.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.evaluate.nemo_skills_eval import (
    AALCR,
    HLE,
    ArenaHard,
    GPQANemoSkills,
    IFBench,
    NemoSkillsEvaluator,
    SciCode,
)


def _fake_output(text, token_ids=(1, 2, 3)):
    """Minimal stand-in for a RequestOutput (only what _generation_text reads)."""
    return SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=list(token_ids))])


def _bare(cls):
    """Instance without running __init__ (which requires nemo_skills + a model)."""
    return cls.__new__(cls)


# --- Class contract: dataset / eval_type / prompt / reasoning-strip ----------
def test_evaluator_class_contract():
    expected = {
        # cls: (DATASET, EVAL_TYPE, PROMPT_CONFIG, STRIP_THINK)
        GPQANemoSkills: ("gpqa", "multichoice", "eval/aai/mcq-4choices", False),
        IFBench: ("ifbench", "ifbench", "generic/default", True),
        SciCode: ("scicode", "scicode", "eval/scicode/background", True),
        HLE: ("hle", "hle", "generic/hle", True),
        AALCR: ("aalcr", "aalcr", "generic/default", True),
        ArenaHard: ("arena-hard", "arena", "generic/default", True),
    }
    for cls, (dataset, eval_type, prompt, strip) in expected.items():
        assert cls.DATASET == dataset
        assert cls.EVAL_TYPE == eval_type
        assert cls.PROMPT_CONFIG == prompt
        assert cls.STRIP_THINK is strip, f"{cls.__name__}.STRIP_THINK"
    # SciCode grades the AA 80-problem (dev+test) split by name.
    assert SciCode.GENERATION_KEY == "generation"
    assert IFBench.GENERATION_KEY == "response"


# --- CLI commands + the sampling options are wired ---------------------------
def test_commands_registered_and_options():
    for cls, name in [
        (GPQANemoSkills, "gpqa_ns"),
        (IFBench, "ifbench"),
        (SciCode, "scicode_ns"),
        (HLE, "hle_aa"),
        (AALCR, "aa_lcr"),
        (ArenaHard, "arena_hard_aa"),
    ]:
        assert cls.command.name == name
        params = {p.name for p in cls.command.params}
        # The knobs the cross-check config relies on must stay exposed.
        for opt in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "num_repeats",
            "split",
            "chat_template_kwargs",
            "max_output_length",
        ):
            assert opt in params, f"{name} missing --{opt}"


# --- The SciCode reasoning-strip fix: grade the post-</think> answer ---------
def test_generation_text_strip_think():
    text = (
        "reason: draft here\n```python\ndef f(): return 0  # DRAFT\n```\n"
        "</think>\n\nAnswer:\n```python\ndef f(): return 1  # FINAL\n```"
    )
    # SciCode / IFBench / judges strip the reasoning; GPQA keeps the full text.
    assert "FINAL" in _bare(SciCode)._generation_text(_fake_output(text))
    assert "DRAFT" not in _bare(SciCode)._generation_text(_fake_output(text))
    kept = _bare(GPQANemoSkills)._generation_text(_fake_output(text))
    assert "</think>" in kept and "DRAFT" in kept


def test_generation_text_no_think_tag_passthrough():
    # No </think> -> STRIP_THINK is a no-op (text returned as-is, stripped).
    out = _bare(SciCode)._generation_text(_fake_output("Answer: A"))
    assert out == "Answer: A"


# --- Aggregation logic (pure, static) ----------------------------------------
def test_score_correctness():
    graded = [
        {"symbolic_correct": True, "predicted_answer": "A"},
        {"symbolic_correct": False, "predicted_answer": "B"},
        {"symbolic_correct": False, "predicted_answer": None},
    ]
    acc, no_answer, correct, n = NemoSkillsEvaluator._score_correctness(graded)
    assert (correct, n, no_answer) == (1, 3, 1)
    assert acc == pytest.approx(100.0 / 3)


def test_aggregate_repeats_pass_at_1():
    ev = _bare(GPQANemoSkills)  # EVAL_TYPE=multichoice, DATASET=gpqa (class attrs)
    ev.num_repeats = 2
    # q0: 2/2 correct; q1: 1/2 correct  ->  pass@1 avg-of-2 = (1.0 + 0.5)/2 = 75%.
    graded = [
        {"symbolic_correct": True, "predicted_answer": "A"},
        {"symbolic_correct": True, "predicted_answer": "A"},
        {"symbolic_correct": True, "predicted_answer": "B"},
        {"symbolic_correct": False, "predicted_answer": "C"},
    ]
    q_indices = [0, 0, 1, 1]
    assert ev._aggregate_repeats(graded, q_indices) == pytest.approx(75.0)


def test_aggregate_scicode():
    def status(*states):
        return {"eval_status": [{"process_status": s} for s in states]}

    graded = [
        status("completed", "completed"),  # problem fully solved
        status("completed", "error"),
    ]  # partially solved
    acc = SciCode._aggregate_scicode(graded)
    assert acc == pytest.approx(50.0)  # problem_accuracy: 1/2 problems


def test_aggregate_ifbench_strict_prompt_level():
    graded = [
        {  # all strict instructions followed -> strict prompt-level pass
            "strict_eval": {"follow_instruction_list": [True, True]},
            "loose_eval": {"follow_instruction_list": [True, True]},
        },
        {  # one strict instruction failed -> strict prompt-level fail
            "strict_eval": {"follow_instruction_list": [True, False]},
            "loose_eval": {"follow_instruction_list": [True, True]},
        },
    ]
    assert IFBench._aggregate_ifbench(graded) == pytest.approx(50.0)


# --- Judgement parsers (free-form / self-judge benches) ----------------------
def test_hle_judgement_parser():
    is_ok = _bare(HLE)._is_correct_judgement
    assert is_ok("reasoning...\nJudgement: yes") is True
    assert is_ok("**Judgement**: no") is False
    assert is_ok(None) is False


def test_aalcr_judgement_parser():
    is_ok = _bare(AALCR)._is_correct_judgement
    assert is_ok("CORRECT") is True
    assert is_ok("INCORRECT") is False
    assert is_ok("") is False


def test_arena_judge_score():
    assert ArenaHard._judge_score("verdict [[A>>B]]") == "A>>B"
    assert ArenaHard._judge_score("[[A>B]] then [[B>A]]") is None  # conflicting
    assert ArenaHard._judge_score("no verdict") is None


# --- The py3.12 scipy shim keeps its leading newline (needs nemo_skills) ------
def test_scicode_grader_shim_leading_newline():
    pytest.importorskip("nemo_skills", reason="nemo_skills not installed; shim guard skipped")
    import nemo_skills.evaluation.evaluator.scicode as sc

    original = sc.eval_prefix
    original_shim = getattr(sc, "_trtllm_simps_shim", False)
    try:
        # Force a fresh application of the shim onto the original prefix.
        sc._trtllm_simps_shim = False
        SciCode._patch_grader_for_py312()
        # The grader runs `full_generation + eval_prefix`; a missing leading
        # newline glues the shim onto the solution's last line -> SyntaxError.
        assert sc.eval_prefix.startswith("\n"), (
            "scicode eval_prefix must start with a newline after the simps shim"
        )
        assert "simps" in sc.eval_prefix and "simpson" in sc.eval_prefix
    finally:
        sc.eval_prefix = original
        sc._trtllm_simps_shim = original_shim
