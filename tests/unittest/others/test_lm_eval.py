# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the lm-evaluation-harness adjacent helpers.

Consolidates the pure-Python tests that guard:

* ``tensorrt_llm.evaluate.lm_eval_tasks.aime.utils`` — answer extraction /
  normalization helpers mirrored from upstream ``aime24`` / ``aime25``.
* ``LmEvalWrapper._get_sampling_params`` — the ``sampling_override`` flag
  that lets model-card recipes override task-yaml gen_kwargs.
* ``MultimodalLmEvalWrapper.apply_chat_template`` — the interleaved
  ``content_parts`` construction for multi-image OPENAI prompts.
* ``CoVoST2._normalize_prediction`` / ``_extract_translation`` — BLEU
  pre-processing for the HF AST transcribe+translate prompt.
* ``tensorrt_llm.evaluate.lm_eval_tasks.mmmu_pro.utils`` —
  ``parse_multi_choice_response`` reverse-scan and the
  ``MMMU_PRO_PROMPT_MODE`` env switch.
* ``LmEvalWrapper._log_spec_stats`` — the ``TLLM_EVAL_SPEC_STATS``-gated
  speculative-decoding acceptance-length (AL) corpus summary,
  iteration-weighted to match ``bench/dataclasses/reporting.py``.
* ``LmEvalWrapper._generate_until_windowed`` — the
  ``TLLM_EVAL_MAX_IN_FLIGHT`` submission window: in-flight cap,
  submission-order results under out-of-order completion, and fail-fast
  propagation of request errors.
* End-to-end: lm-eval's real ``evaluate()`` loop (real ``ConfigurableTask``,
  filters, aggregation) driven through ``LmEvalWrapper`` over a mocked LLM,
  for both the final score and the partial-score running estimates.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.evaluate.covost2 import CoVoST2
from tensorrt_llm.evaluate.lm_eval import (
    LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER,
    MAX_IN_FLIGHT_ENV_VAR,
    LmEvalWrapper,
    MultimodalLmEvalWrapper,
    _override_stop_strings,
)
from tensorrt_llm.evaluate.lm_eval_tasks.aime.utils import (
    is_equiv,
    last_boxed_only_string,
    process_results,
    remove_boxed,
    strip_string,
)
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


# ===========================================================================
# AIME utils — last_boxed_only_string / remove_boxed / is_equiv / strip_string
# ===========================================================================
#
# The helpers are mirrored from lm-evaluation-harness so the behaviour must
# stay byte-compatible with upstream ``aime24`` / ``aime25`` scoring.


def test_last_boxed_only_string_plain():
    r"""Canonical \boxed{N} is returned verbatim including prefix and brace."""
    assert last_boxed_only_string("The answer is \\boxed{42}.") == "\\boxed{42}"


def test_last_boxed_only_string_nested_braces():
    r"""Nested braces (\boxed{\frac{1}{2}}) are balanced and preserved."""
    s = "Final: \\boxed{\\frac{1}{2}}"
    assert last_boxed_only_string(s) == "\\boxed{\\frac{1}{2}}"


def test_last_boxed_only_string_takes_last():
    r"""Rightmost \boxed wins; AIME outputs often restate candidates first."""
    s = "First guess \\boxed{7}, but actually \\boxed{42}."
    assert last_boxed_only_string(s) == "\\boxed{42}"


def test_last_boxed_only_string_no_boxed_returns_none():
    r"""No \boxed and no \fbox returns None so $...$ fallback can run."""
    assert last_boxed_only_string("The answer is 42.") is None


def test_last_boxed_only_string_space_variant():
    r"""\boxed N (space, no braces) returns up to the terminating $."""
    s = "Answer: \\boxed 42$ end."
    out = last_boxed_only_string(s)
    assert out is not None
    assert out.startswith("\\boxed ")
    assert "42" in out


def test_remove_boxed_strips_wrapper():
    assert remove_boxed("\\boxed{42}") == "42"


def test_remove_boxed_preserves_nested_latex():
    assert remove_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_is_equiv_integer_exact():
    assert is_equiv("42", "42") is True


def test_is_equiv_whitespace_tolerant():
    """Leading, trailing, and internal whitespace is normalized away."""
    assert is_equiv(" 42 ", "42") is True
    assert is_equiv("4 2", "42") is True


def test_is_equiv_frac_sugar_expands():
    r"""\frac12 (compact sugar) compares equal to \frac{1}{2}."""
    assert is_equiv("\\frac12", "\\frac{1}{2}") is True


def test_is_equiv_distinct_values():
    assert is_equiv("42", "43") is False


def test_strip_string_kills_spaces_and_newlines():
    assert strip_string("  4\n2 ") == "42"


def test_strip_string_drops_leading_varname():
    """``x=42`` style prefixes are dropped when the LHS is short (<=2 chars)."""
    assert strip_string("k=42") == "42"


def _aime_doc(answer) -> dict:
    """AIME doc with MathArena lowercase ``answer`` field."""
    return {"problem_idx": 1, "problem": "...", "answer": answer}


def test_process_results_boxed_correct():
    doc = _aime_doc(42)
    results = ["Working... therefore \\boxed{42}."]
    assert process_results(doc, results) == {"exact_match": 1}


def test_process_results_boxed_wrong():
    doc = _aime_doc(42)
    results = ["Working... therefore \\boxed{7}."]
    assert process_results(doc, results) == {"exact_match": 0}


def test_process_results_boxed_overrides_dollar_delimited():
    r"""When both $...$ and \boxed{} appear, \boxed{} wins."""
    doc = _aime_doc(42)
    # The $...$ span captures 'decoy=7' but \boxed{42} must override it.
    results = ["Candidate $decoy=7$ but final \\boxed{42}."]
    assert process_results(doc, results) == {"exact_match": 1}


def test_process_results_dollar_fallback_when_no_boxed():
    r"""Without \boxed, fall back to content between first and last $."""
    doc = _aime_doc(42)
    results = ["The answer is $42$."]
    assert process_results(doc, results) == {"exact_match": 1}


def test_process_results_answer_key_case_insensitive():
    """``answer`` key match is case-insensitive against dataset schema drift."""
    doc = {"problem": "...", "Answer": 42}
    results = ["\\boxed{42}"]
    assert process_results(doc, results) == {"exact_match": 1}


# ===========================================================================
# LmEvalWrapper sampling — sampling_override flag
# ===========================================================================
#
# Exercises the ``sampling_override`` flag introduced so that model-card
# sampling recipes (e.g. Gemma4 26B: temperature=1.0 / top_p=0.95 / top_k=64)
# can override the greedy defaults baked into lm-eval-harness task YAMLs.


def _make_lm_eval_wrapper(
    sampling_params: SamplingParams | None = None,
    sampling_override: bool = False,
) -> LmEvalWrapper:
    """Build a wrapper with a fake llm.

    We only exercise ``_get_sampling_params`` which doesn't touch the llm
    object.
    """
    fake_llm = MagicMock()
    fake_llm.tokenizer = MagicMock()
    return LmEvalWrapper(
        fake_llm,
        sampling_params=sampling_params,
        sampling_override=sampling_override,
    )


def test_greedy_default_from_task_yaml():
    """Default (no sampling override): task yaml gen_kwargs win.

    Mirrors the original behaviour: lm-eval GPQA yaml sets temperature=0.0 to
    force greedy, and that has to keep working even when the caller supplies
    a default SamplingParams with temperature=0 from the CLI.
    """
    sp = SamplingParams(max_tokens=256)  # default temperature=0
    wrapper = _make_lm_eval_wrapper(sampling_params=sp, sampling_override=False)
    gen_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "until": ["</s>"],
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 0.0
    assert out.top_p == 1.0
    assert out.stop == ["</s>"]


def test_sampling_override_cli_wins_on_temperature_and_top_p():
    """sampling_override=True: CLI sampling params win over yaml.

    CLI temperature / top_p / top_k must NOT be clobbered by the task yaml's
    greedy gen_kwargs when the override flag is on.
    """
    sp = SamplingParams(
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        seed=1234,
    )
    wrapper = _make_lm_eval_wrapper(sampling_params=sp, sampling_override=True)
    # Task yaml tries to force greedy
    gen_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "until": ["</s>"],
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 1.0  # CLI wins
    assert out.top_p == 0.95  # CLI wins
    assert out.top_k == 64  # preserved from CLI
    assert out.seed == 1234  # preserved from CLI
    # stop tokens from task yaml are still respected
    assert out.stop == ["</s>"]


def test_sampling_override_still_respects_max_tokens_from_yaml():
    """sampling_override only touches temperature / top_p.

    max_gen_toks from the yaml (if any) must still take precedence so
    per-task output budgets behave as documented.
    """
    sp = SamplingParams(
        max_tokens=256,  # CLI default
        temperature=1.0,
        top_p=0.95,
    )
    wrapper = _make_lm_eval_wrapper(sampling_params=sp, sampling_override=True)
    gen_kwargs = {
        "temperature": 0.0,
        "max_gen_toks": 512,  # task-specific cap
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 1.0  # CLI wins
    assert out.max_tokens == 512  # task yaml wins


def test_sampling_override_no_cli_falls_back_to_yaml():
    """No-override path keeps the pre-existing behaviour.

    If the CLI doesn't supply any sampling knobs, the wrapper falls back to
    the task yaml's gen_kwargs populating SamplingParams.
    """
    wrapper = _make_lm_eval_wrapper(sampling_params=None, sampling_override=False)
    gen_kwargs = {
        "temperature": 0.0,
        "max_gen_toks": 256,
        "until": ["</s>"],
    }
    out = wrapper._get_sampling_params(dict(gen_kwargs))
    assert out.temperature == 0.0
    assert out.max_tokens == 256
    assert out.stop == ["</s>"]


# ===========================================================================
# Multimodal wrapper interleave — apply_chat_template content_parts
# ===========================================================================
#
# Guards the MMMU Pro multi-image regression: without interleaved content,
# multi-image prompts (``"Consider <image 1>. What does <image 2> show?"``)
# lose answer-grounding because all images get bulk-prepended before the
# text. The wrapper now produces an interleaved content_parts list for
# OPENAI-format chat templates so ``_build_openai_content`` emits a
# correctly-ordered OpenAI content list.


# Interleaving is opt-in per model: the wrapper reads
# ``MULTIMODAL_PLACEHOLDER_REGISTRY.get_interleave_placeholders(model_type)``
# at construction, and models that don't opt in keep the historical
# strip-and-bulk-insert behaviour. These tests drive that flag directly
# instead of naming an opted-in model, because which models are registered
# varies with the installed transformers version — keying on a real model
# name would make the tests environment-dependent. ``interleave=False``
# (the default here) matches an unregistered model such as ``gemma3``.
def _make_multimodal_wrapper(
    model_type: str = "gemma3",
    interleave: bool = False,
) -> MultimodalLmEvalWrapper:
    fake_llm = MagicMock()
    fake_llm.tokenizer = MagicMock()
    fake_llm.input_processor = MagicMock()
    fake_llm.input_processor.processor = MagicMock()
    with (
        patch.object(MultimodalLmEvalWrapper, "_get_model_type", return_value=model_type),
        patch.object(
            MULTIMODAL_PLACEHOLDER_REGISTRY,
            "get_interleave_placeholders",
            return_value=interleave,
        ),
    ):
        return MultimodalLmEvalWrapper(
            fake_llm,
            sampling_params=None,
            streaming=False,
            model_type=model_type,
        )


def _call_apply(wrapper, text: str, *, content_format: ContentFormat):
    """Run apply_chat_template against a stubbed trtllm_apply_chat_template.

    Returns the conversation dict that was built.  The real HF chat
    template requires an actual tokenizer; we only care about the
    conversation structure the wrapper constructs before it hands off.
    """
    chat_history = [{"role": "user", "content": text}]
    captured = {}

    def _fake_trtllm_apply(**kwargs):
        captured.update(kwargs)
        return "<stub>"

    with (
        patch(
            "tensorrt_llm.evaluate.lm_eval.resolve_hf_chat_template",
            return_value="<stub-template>",
        ),
        patch(
            "tensorrt_llm.evaluate.lm_eval._resolve_content_format",
            return_value=content_format,
        ),
        patch("tensorrt_llm.evaluate.lm_eval.trtllm_apply_chat_template", _fake_trtllm_apply),
    ):
        wrapper.apply_chat_template(chat_history)

    assert captured, "trtllm_apply_chat_template was not invoked"
    convs = captured["conversation"]
    assert len(convs) == 1
    return convs[0]


def test_not_opted_in_model_does_not_interleave():
    """A model that does not opt in keeps the historical bulk-insert path.

    content_parts stays absent so the existing BEFORE_TEXT default keeps working.
    """
    wrapper = _make_multimodal_wrapper()
    text = f"What is in {LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER}?"
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    assert conv.get("content_parts") is None
    assert LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER not in conv["content"]


def test_multi_image_openai_builds_content_parts():
    """Multi-image + OPENAI template carries the original interleaving in content_parts.

    ``_build_openai_content`` then emits media entries at the correct positions.
    """
    wrapper = _make_multimodal_wrapper(interleave=True)
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"Consider {ph}. What does {ph} show?"
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    parts = conv.get("content_parts")
    assert parts is not None, "expected interleaved content_parts for multi-image OPENAI prompt"

    # Expected: ["Consider ", image, ". What does ", image, " show?"]
    kinds = [("text" if isinstance(p, str) else p["type"]) for p in parts]
    assert kinds == ["text", "image", "text", "image", "text"]
    # image parts keep an ascending media_index so downstream code can
    # correlate them with the images list.
    media_parts = [p for p in parts if isinstance(p, dict)]
    assert [p["media_index"] for p in media_parts] == [0, 1]


def test_multi_image_string_format_not_opted_in_uses_placeholders():
    """STRING-format templates on a non-opted-in model use flat placeholders.

    Placeholders are inserted into the flat text via
    ``add_multimodal_placeholders`` instead, so ``content_parts`` stays absent.
    """
    wrapper = _make_multimodal_wrapper()
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"{ph} vs {ph}: what changed?"
    with patch(
        "tensorrt_llm.evaluate.lm_eval.add_multimodal_placeholders",
        return_value="<placeholders><placeholders> vs : what changed?",
    ):
        conv = _call_apply(wrapper, text, content_format=ContentFormat.STRING)
    assert conv.get("content_parts") is None


def test_trailing_text_after_last_image_preserved():
    """Text that follows the last image must be preserved verbatim.

    Otherwise the question suffix ('Answer:') is dropped before it reaches
    the model.
    """
    wrapper = _make_multimodal_wrapper(interleave=True)
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"Compare {ph} with {ph}. Answer with a letter."
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    parts = conv["content_parts"]
    # Last part must be the trailing text.
    assert isinstance(parts[-1], str)
    assert parts[-1].endswith("Answer with a letter.")


def test_leading_image_no_empty_text_segment():
    """Leading ``<image>`` placeholders do not emit an empty-string text part.

    content_parts must begin with the image entry itself.
    """
    wrapper = _make_multimodal_wrapper(interleave=True)
    ph = LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER
    text = f"{ph} {ph} Answer?"
    conv = _call_apply(wrapper, text, content_format=ContentFormat.OPENAI)
    parts = conv["content_parts"]
    assert parts, "expected non-empty content_parts"
    assert isinstance(parts[0], dict) and parts[0]["type"] == "image"
    # Empty text segments must not be inserted.  Whitespace-only segments
    # (e.g. " " between two adjacent ``<image>`` placeholders) are preserved
    # because they faithfully reflect the user's prompt.
    assert all((not isinstance(p, str)) or p != "" for p in parts)


# ===========================================================================
# CoVoST normalizer — _normalize_prediction / _extract_translation
# ===========================================================================
#
# Gemma4 instruct occasionally prepends ``Translation:`` (or wraps outputs in
# quotes) even when told "respond with only the translation, no other text".
# The normalizer strips those wrappers so BLEU 1-gram precision matches the
# raw reference text format, closing a sizable portion of the zh-CN→en gap.


def test_strip_common_prefixes_case_insensitive():
    norm = CoVoST2._normalize_prediction
    assert norm("Translation: Hello world") == "Hello world"
    assert norm("translation: Hello world") == "Hello world"
    assert norm("TRANSLATION: Hello world") == "Hello world"
    assert norm("Translated: Bonjour") == "Bonjour"
    assert norm("The translation is: Guten Tag") == "Guten Tag"
    assert norm("English translation: See you.") == "See you."
    assert norm("Here is the translation: Adiós") == "Adiós"


def test_strip_outer_quotes():
    norm = CoVoST2._normalize_prediction
    assert norm('"Hello world."') == "Hello world."
    assert norm("'Bonjour.'") == "Bonjour."
    # Smart quotes (U+201C U+201D)
    assert norm("“Hello.”") == "Hello."
    # Only strip quotes when the whole string is quoted.
    assert norm('She said "hi" to me.') == 'She said "hi" to me.'


def test_preserves_unprefixed_text():
    norm = CoVoST2._normalize_prediction
    assert norm("Hello world") == "Hello world"
    assert norm("  Hello world  ") == "Hello world"


def test_strip_composite_prefix_plus_quotes():
    """Prefix strip must run before the quote strip.

    Gemma4 sometimes emits ``Translation: "Hello world."``.
    """
    norm = CoVoST2._normalize_prediction
    assert norm('Translation: "Hello world."') == "Hello world."


def test_preserves_internal_colons():
    """Only strip the prefix at the very start."""
    norm = CoVoST2._normalize_prediction
    # "he said: hi" should not match "translation:" so it's preserved as-is.
    assert norm("he said: hi") == "he said: hi"


def test_strip_bom_and_zero_width():
    """Strip leading BOM / zero-width spaces.

    These occasionally appear on Unicode-heavy decode paths.
    """
    norm = CoVoST2._normalize_prediction
    assert norm("﻿Hello") == "Hello"
    assert norm("\u200bHello") == "Hello"


# The HF AST prompt ("transcribe, then translate") instructs the model to
# output the transcription first and then ``"{TARGET_LANGUAGE}: <translation>"``.
# BLEU must score the translation only — not the transcription — so we look
# for the language-name marker and return the text after it.  Falls back to
# the generic normalizer when the marker is missing (model disobeyed the
# format, thinking-mode chain-of-thought, empty output, etc.).


def test_extract_translation_basic_ast_format():
    """Standard HF AST response: transcription, then 'TARGET: translation'."""
    extract = CoVoST2._extract_translation
    response = "Hello world\n\nChinese: 你好世界"
    assert extract(response, "Chinese") == "你好世界"


def test_extract_translation_marker_case_insensitive():
    """Language-name matching ignores case — models lowercase occasionally."""
    extract = CoVoST2._extract_translation
    assert extract("Hola\n\nenglish: Hello", "English") == "Hello"
    assert extract("Hola\n\nENGLISH: Hello", "English") == "Hello"


def test_extract_translation_picks_last_marker():
    """Last marker wins under multiple occurrences.

    Thinking chains and self-correction lines can mention the target
    language multiple times — the final occurrence is the canonical
    translation.
    """
    extract = CoVoST2._extract_translation
    response = (
        "Thinking: the speaker says hello\n"
        "Chinese: 错误的翻译\n"
        "\n"
        "Actually let me retry.\n"
        "Chinese: 你好"
    )
    assert extract(response, "Chinese") == "你好"


def test_extract_translation_falls_back_to_normalize_when_no_marker():
    """If the model ignored the format, fall back to generic normalization."""
    extract = CoVoST2._extract_translation
    # Plain response without the AST marker.
    assert extract("Translation: Hello world", "Chinese") == "Hello world"


def test_extract_translation_stops_at_double_newline():
    """Translation region ends at the next double-newline.

    Trailing chain-of-thought after the translation must not be
    included in the BLEU input.
    """
    extract = CoVoST2._extract_translation
    response = "Hola\n\nEnglish: Hello\n\nAdditional explanation goes here."
    assert extract(response, "English") == "Hello"


def test_extract_translation_empty_input():
    """Empty or None-like response shouldn't crash — return empty string."""
    extract = CoVoST2._extract_translation
    assert extract("", "English") == ""


def test_extract_translation_normalizes_after_marker():
    """Extracted segment still runs through _normalize_prediction.

    Leading quotes and prefixes get stripped on the translation side too.
    """
    extract = CoVoST2._extract_translation
    response = 'Hola\n\nEnglish: "Hello world."'
    assert extract(response, "English") == "Hello world."


def test_prompt_text_uses_hf_ast_format():
    """Regression: CoVoST prompt must use the HF AST transcribe+translate form.

    Documented in the Gemma4 model card.  The old 'translate only' form
    under-performed substantially on non-Latin source languages because
    the model had no transcription step to ground the translation on.
    """
    cov = object.__new__(CoVoST2)
    cov.src_name = "English"
    cov.tgt_name = "Chinese"
    prompt = cov._prompt_text()
    # HF AST structure: transcribe + translate, with explicit marker line.
    assert "Transcribe" in prompt
    assert "translate" in prompt.lower()
    assert "Chinese:" in prompt  # target-language marker that _extract_translation keys off
    assert "English" in prompt  # source language


# ===========================================================================
# MMMU Pro parser — parse_multi_choice_response + MMMU_PRO_PROMPT_MODE
# ===========================================================================
#
# Guards two fixes:
#
# 1. ``_ANSWER_RE`` + reverse-scan in ``parse_multi_choice_response``: the
#    default MMMU parser scanned forward, which caused CoT / thinking-mode
#    outputs to pick up an earlier-appearing letter (e.g. from "option A is
#    wrong because...") instead of the final ``Answer: X`` line.  The new
#    reverse scan walks lines bottom-up and returns the first regex match,
#    so the canonical final-answer line wins.
#
# 2. ``MMMU_PRO_PROMPT_MODE`` env variable: switches the prompt suffix
#    between the MMMU-Benchmark's ``direct/standard`` template (default)
#    and ``cot/standard`` (opt-in).  The latter adds +10-25 pp on smaller
#    models by asking for "Answer: $LETTER" on the final line.


# Reload the module under test whenever the env flips, since the suffix is
# captured at import time.
def _reload_mmmu_pro_utils(mode: str | None):
    if mode is None:
        os.environ.pop("MMMU_PRO_PROMPT_MODE", None)
    else:
        os.environ["MMMU_PRO_PROMPT_MODE"] = mode
    from tensorrt_llm.evaluate.lm_eval_tasks.mmmu_pro import utils

    importlib.reload(utils)
    return utils


def test_cot_final_answer_line_wins():
    """The final 'Answer: X' line wins over earlier letters in the chain.

    This is the main reason thinking-mode went from 51% to 76% on 26B —
    the forward scanner was latching onto a random "A" inside the reasoning
    before ever reaching the final answer line.
    """
    utils = _reload_mmmu_pro_utils(None)
    resp = (
        "Let me think step by step.\n"
        "Option A is wrong because foo.\n"
        "Option B is wrong because bar.\n"
        "Option C is correct.\n"
        "Answer: C"
    )
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "C"


def test_final_answer_with_parentheses():
    """Models sometimes emit 'Answer: (C)' — regex tolerates parens."""
    utils = _reload_mmmu_pro_utils(None)
    resp = "Reasoning...\nAnswer: (C)"
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "C"


def test_final_answer_case_insensitive():
    """The regex is case-insensitive for the 'answer' keyword."""
    utils = _reload_mmmu_pro_utils(None)
    resp = "Thinking...\nanswer: D"
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "D"


def test_final_answer_keyword_is():
    """'Answer is X' form (without colon) also matches."""
    utils = _reload_mmmu_pro_utils(None)
    resp = "The answer is B."
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "B"


def test_final_answer_letter_out_of_choice_set_is_ignored():
    """Out-of-set regex match must not short-circuit the parser.

    If the final-answer regex matches a letter outside all_choices, the
    parser must fall back to the legacy scan instead of returning an
    invalid letter.
    """
    utils = _reload_mmmu_pro_utils(None)
    resp = "I think the answer is Z.\nBut actually A"
    # Only A/B/C/D are valid — Z must not win.
    result = utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {})
    assert result in {"A", "B", "C", "D"}  # Some valid letter, not Z.


def test_no_final_answer_falls_back_to_legacy_scan():
    """Fallback path: responses without 'Answer: X' use the legacy scan.

    We keep the upstream MMMU parser intact so non-CoT responses still
    get a best-effort letter match.
    """
    utils = _reload_mmmu_pro_utils(None)
    resp = "I choose (B) because it matches."
    # Legacy scan should still pick a letter.
    result = utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {})
    assert result in {"A", "B", "C", "D"}


def test_reverse_scan_picks_last_answer_line_across_multiple():
    """Last of several 'Answer: X' lines wins.

    Matches how the model typically self-corrects: it writes an initial
    guess, then a correction, and the final line is authoritative.
    """
    utils = _reload_mmmu_pro_utils(None)
    resp = "Answer: A\nWait, let me reconsider.\nAnswer: B"
    assert utils.parse_multi_choice_response(resp, ["A", "B", "C", "D"], {}) == "B"


def test_default_mode_is_direct():
    """Unset env => direct/standard suffix (backward-compatible default)."""
    utils = _reload_mmmu_pro_utils(None)
    assert utils._MODE == "direct"
    assert "letter" in utils._PROMPT_SUFFIX.lower()
    assert "step by step" not in utils._PROMPT_SUFFIX.lower()


def test_mode_cot_switches_suffix():
    """MMMU_PRO_PROMPT_MODE=cot => cot/standard suffix (think step-by-step).

    This is the suffix the HF Gemma4 blog numbers appear to use — it adds
    the 'Answer: $LETTER' final-line instruction, which pairs with the
    reverse-scan parser above.
    """
    utils = _reload_mmmu_pro_utils("cot")
    assert utils._MODE == "cot"
    assert "step by step" in utils._PROMPT_SUFFIX.lower()
    assert "answer: $letter" in utils._PROMPT_SUFFIX.lower()


def test_mode_unknown_value_defaults_to_direct():
    """Unrecognized values => fall back to direct (defensive)."""
    utils = _reload_mmmu_pro_utils("something-else")
    assert utils._MODE == "something-else"
    # Anything not 'cot' picks the direct suffix.
    assert "letter" in utils._PROMPT_SUFFIX.lower()
    assert "step by step" not in utils._PROMPT_SUFFIX.lower()


def test_mode_cot_included_in_example_format():
    """MULTI_CHOICE_EXAMPLE_FORMAT must embed the cot suffix when mode=cot."""
    utils = _reload_mmmu_pro_utils("cot")
    try:
        assert "step by step" in utils.MULTI_CHOICE_EXAMPLE_FORMAT.lower()
    finally:
        # Restore module state for other tests running in the same session.
        _reload_mmmu_pro_utils(None)


# ===========================================================================
# _RunningScoreTracker — partial score estimates during generate_until
# ===========================================================================
#
# Enabled via TLLM_EVAL_PARTIAL_SCORES_EVERY; scores each completed response
# with the owning task's filters + process_results on a throwaway instance
# copy, and must never disturb the real instance or fail the eval.


class _FakeEnsemble:
    """Minimal stand-in for lm_eval.api.filter.FilterEnsemble."""

    def __init__(self, name):
        self.name = name

    def apply(self, instances):
        for inst in instances:
            # Trivial "take_first" pipeline.
            inst.filtered_resps[self.name] = inst.resps[0]


class _FakeTask:
    def __init__(self):
        self._filters = [_FakeEnsemble("strict-match")]

    def process_results(self, doc, results):
        # Mirror lm-eval's ConfigurableTask.process_results: results is a list
        # (one entry per repeat/request), and the prediction is results[0].
        return {"exact_match": float(results[0] == doc["answer"])}


class _FakeInstance:
    def __init__(self, task_name, doc):
        self.task_name = task_name
        self.doc = doc
        self.resps = []
        self.filtered_resps = {}


def _make_tracker(interval=2):
    from tensorrt_llm.evaluate.lm_eval import _RunningScoreTracker

    return _RunningScoreTracker({"fake_task": _FakeTask()}, interval)


def test_running_score_tracker_aggregates_mean():
    """Running estimate is the mean of per-sample metric values."""
    tracker = _make_tracker()
    docs = [{"answer": "42"}, {"answer": "7"}, {"answer": "1"}]
    responses = ["42", "0", "1"]  # right, wrong, right
    for doc, text in zip(docs, responses):
        tracker.update(_FakeInstance("fake_task", doc), text)
    assert not tracker.disabled
    key = "fake_task,exact_match,strict-match"
    assert tracker.metric_counts[key] == 3
    assert tracker.metric_sums[key] == 2.0


def test_running_score_tracker_does_not_mutate_instance():
    """The real instance stays untouched — the harness fills it in later."""
    tracker = _make_tracker()
    instance = _FakeInstance("fake_task", {"answer": "42"})
    tracker.update(instance, "42")
    assert instance.resps == []
    assert instance.filtered_resps == {}


def test_running_score_tracker_unknown_task_disables():
    """Any scoring failure permanently disables the tracker, never raises."""
    tracker = _make_tracker()
    tracker.update(_FakeInstance("unknown_task", {"answer": "42"}), "42")
    assert tracker.disabled
    # Subsequent updates and logging are silent no-ops.
    tracker.update(_FakeInstance("fake_task", {"answer": "42"}), "42")
    assert not tracker.metric_counts
    tracker.maybe_log(10, 100)


def test_running_score_tracker_logs_on_interval():
    """maybe_log emits at every `interval` responses and at completion."""
    tracker = _make_tracker(interval=2)
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        tracker.update(_FakeInstance("fake_task", {"answer": "1"}), "1")
        tracker.maybe_log(1, 3)  # off-interval, not final -> no log
        mock_logger.info.assert_not_called()
        tracker.update(_FakeInstance("fake_task", {"answer": "1"}), "0")
        tracker.maybe_log(2, 3)  # on-interval -> logs
        assert mock_logger.info.call_count == 1
        tracker.update(_FakeInstance("fake_task", {"answer": "1"}), "1")
        tracker.maybe_log(3, 3)  # final response -> logs
        assert mock_logger.info.call_count == 2
    message = mock_logger.info.call_args[0][0]
    assert "2/3" not in message  # latest call reports 3/3
    assert "3/3" in message
    assert "fake_task,exact_match,strict-match" in message
    # 2 of 3 correct -> ~66.67 on the 0~100 scale.
    assert "66.67" in message


def test_running_score_tracker_process_results_list_convention():
    """process_results receives a list, not a bare string (regression for GSM8K bug).

    lm-eval's ConfigurableTask.process_results does ``result = results[0]`` to
    extract the prediction from the list of per-repeat responses.  If the tracker
    passes the filtered_resp string directly instead of wrapping it in a list,
    ``results[0]`` silently returns the *first character* of the string, causing
    multi-digit answers to score as misses (~26% on GSM8K) while single-digit
    answers accidentally match.
    """
    tracker = _make_tracker()
    # Use a multi-digit answer so the first-character bug is observable:
    # "42" would produce results[0]=="4" if the list wrap were missing.
    doc = {"answer": "42"}
    tracker.update(_FakeInstance("fake_task", doc), "42")
    assert not tracker.disabled
    key = "fake_task,exact_match,strict-match"
    assert tracker.metric_sums[key] == 1.0, (
        "multi-digit answer scored as miss — process_results likely received "
        "a bare string so results[0] returned only the first character"
    )


def test_running_score_tracker_task_groups_flattened():
    """Nested task_dict groups resolve to their leaf tasks."""
    from tensorrt_llm.evaluate.lm_eval import _RunningScoreTracker

    tracker = _RunningScoreTracker({"group": {"fake_task": _FakeTask()}}, 1)
    tracker.update(_FakeInstance("fake_task", {"answer": "42"}), "42")
    assert not tracker.disabled
    assert tracker.metric_counts["fake_task,exact_match,strict-match"] == 1


def test_running_score_tracker_separate_keys_per_task():
    """Two tasks with the same metric/filter don't mix their running estimates."""
    from tensorrt_llm.evaluate.lm_eval import _RunningScoreTracker

    task_a = _FakeTask()
    task_b = _FakeTask()
    tracker = _RunningScoreTracker({"task_a": task_a, "task_b": task_b}, 999)
    tracker.update(_FakeInstance("task_a", {"answer": "x"}), "x")  # correct
    tracker.update(_FakeInstance("task_b", {"answer": "x"}), "y")  # wrong
    assert not tracker.disabled
    assert tracker.metric_sums["task_a,exact_match,strict-match"] == 1.0
    assert tracker.metric_sums["task_b,exact_match,strict-match"] == 0.0


# ===========================================================================
# _parse_partial_scores_env — env-var parsing
# ===========================================================================


def test_parse_partial_scores_env_positive(monkeypatch):
    """A positive integer returns that interval."""
    from tensorrt_llm.evaluate.lm_eval import PARTIAL_SCORES_ENV_VAR, _parse_partial_scores_env

    monkeypatch.setenv(PARTIAL_SCORES_ENV_VAR, "100")
    assert _parse_partial_scores_env() == 100


def test_parse_partial_scores_env_zero_disables(monkeypatch):
    """Zero disables partial scoring (returns None)."""
    from tensorrt_llm.evaluate.lm_eval import PARTIAL_SCORES_ENV_VAR, _parse_partial_scores_env

    monkeypatch.setenv(PARTIAL_SCORES_ENV_VAR, "0")
    assert _parse_partial_scores_env() is None


def test_parse_partial_scores_env_negative_disables(monkeypatch):
    """Negative values disable partial scoring (returns None)."""
    from tensorrt_llm.evaluate.lm_eval import PARTIAL_SCORES_ENV_VAR, _parse_partial_scores_env

    monkeypatch.setenv(PARTIAL_SCORES_ENV_VAR, "-5")
    assert _parse_partial_scores_env() is None


def test_parse_partial_scores_env_invalid_raises(monkeypatch):
    """A non-integer value raises ValueError."""
    from tensorrt_llm.evaluate.lm_eval import PARTIAL_SCORES_ENV_VAR, _parse_partial_scores_env

    monkeypatch.setenv(PARTIAL_SCORES_ENV_VAR, "abc")
    with pytest.raises(ValueError, match=PARTIAL_SCORES_ENV_VAR):
        _parse_partial_scores_env()


def test_parse_partial_scores_env_unset_returns_none(monkeypatch):
    """Unset env var returns None."""
    from tensorrt_llm.evaluate.lm_eval import PARTIAL_SCORES_ENV_VAR, _parse_partial_scores_env

    monkeypatch.delenv(PARTIAL_SCORES_ENV_VAR, raising=False)
    assert _parse_partial_scores_env() is None


# ===========================================================================
# LmEvalWrapper.generate_until — partial scorer invocation
# ===========================================================================


def test_generate_until_invokes_partial_scorer():
    """generate_until calls scorer.update and scorer.maybe_log for each response."""
    from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper, _RunningScoreTracker

    fake_output = MagicMock()
    fake_output.result.return_value.outputs = [MagicMock(text="42")]
    fake_llm = MagicMock()
    fake_llm.generate_async.return_value = fake_output

    wrapper = LmEvalWrapper(
        llm=fake_llm,
        partial_scores_every=1,
        partial_scoring_task_dict={"fake_task": _FakeTask()},
    )

    fake_request = MagicMock()
    fake_request.args = ("hello world", {})
    fake_request.task_name = "fake_task"
    fake_request.doc = {"answer": "42"}

    with (
        patch.object(_RunningScoreTracker, "update") as mock_update,
        patch.object(_RunningScoreTracker, "maybe_log") as mock_log,
    ):
        wrapper.generate_until([fake_request], disable_tqdm=True)

    mock_update.assert_called_once_with(fake_request, "42")
    mock_log.assert_called_once_with(1, 1)


# ===========================================================================
# TLLM_EVAL_SPEC_STATS — speculative-decoding AL/AR stats
# ===========================================================================
#
# AL (acceptance length) is iteration-weighted (total decoded tokens / total
# decode iterations) to agree with the repo's canonical definition in
# ``bench/dataclasses/reporting.py``. AR (acceptance rate) is the corpus
# ratio of accepted to drafted tokens summed over per-request
# ``request_perf_metrics.speculative_decoding`` counters, which
# ``_get_sampling_params`` requests via ``return_perf_metrics=True`` when
# the stats are enabled.


def _make_spec_output(
    tokens_per_iter: float | None = None,
    decoding_iter: int | None = 1,
    accepted_drafted: tuple[int, int] | None = None,
) -> MagicMock:
    """Fake RequestOutput with an optional per-request AL sample.

    ``tokens_per_iter`` as None models a request without speculative
    metrics (non-spec-dec run, or a response that never reported them).
    ``decoding_iter`` is the AL aggregation weight (decode iterations the
    request ran); None models a result that never populated it.
    ``accepted_drafted`` populates the request_perf_metrics
    speculative_decoding counters the AR line reads; None models a request
    without perf metrics (return_perf_metrics off, or dropped).
    """
    output = MagicMock()
    output.avg_decoded_tokens_per_iter = tokens_per_iter
    output.decoding_iter = decoding_iter
    completion = MagicMock()
    if accepted_drafted is not None:
        accepted, drafted = accepted_drafted
        spec_dec = MagicMock()
        spec_dec.total_accepted_draft_tokens = accepted
        spec_dec.total_draft_tokens = drafted
        completion.request_perf_metrics.speculative_decoding = spec_dec
    else:
        completion.request_perf_metrics = None
    output.outputs = [completion]
    return output


def test_spec_stats_env_unset_disables(monkeypatch):
    """Unset env leaves the feature off."""
    from tensorrt_llm.evaluate.lm_eval import SPEC_STATS_ENV_VAR

    monkeypatch.delenv(SPEC_STATS_ENV_VAR, raising=False)
    wrapper = _make_lm_eval_wrapper()
    assert wrapper.spec_stats is False


def test_spec_stats_env_enabled(monkeypatch):
    """TLLM_EVAL_SPEC_STATS=1 turns the feature on."""
    from tensorrt_llm.evaluate.lm_eval import SPEC_STATS_ENV_VAR

    monkeypatch.setenv(SPEC_STATS_ENV_VAR, "1")
    wrapper = _make_lm_eval_wrapper()
    assert wrapper.spec_stats is True


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_spec_stats_env_non_one_values_disable(monkeypatch, value):
    """Only the literal "1" enables the feature."""
    from tensorrt_llm.evaluate.lm_eval import SPEC_STATS_ENV_VAR

    monkeypatch.setenv(SPEC_STATS_ENV_VAR, value)
    wrapper = _make_lm_eval_wrapper()
    assert wrapper.spec_stats is False


def test_log_spec_stats_reports_al_mean_min_max():
    """AL over equal-weight requests equals the plain mean; min/max/n present."""
    wrapper = _make_lm_eval_wrapper()
    outputs = [
        _make_spec_output(tokens_per_iter=2.0, decoding_iter=5),
        _make_spec_output(tokens_per_iter=4.0, decoding_iter=5),
    ]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    assert mock_logger.info.call_count == 1
    al_message = mock_logger.info.call_args[0][0]
    assert "AL" in al_message
    assert "3.000" in al_message  # equal weights -> mean of 2.0 and 4.0
    assert "min 2.000" in al_message
    assert "max 4.000" in al_message
    assert "n=2" in al_message


def test_log_spec_stats_weights_by_decode_iterations():
    """AL matches reporting.py's token-level mean: weighted by decode iterations.

    (2.0 tok/iter over 1 iter) + (4.0 tok/iter over 3 iters) = 14 decoded
    tokens over 4 iterations = 3.5 — NOT the unweighted mean 3.0, which
    would bias the result toward short requests (see the explicit rationale
    in ``bench/dataclasses/reporting.py``).
    """
    wrapper = _make_lm_eval_wrapper()
    outputs = [
        _make_spec_output(tokens_per_iter=2.0, decoding_iter=1),
        _make_spec_output(tokens_per_iter=4.0, decoding_iter=3),
    ]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    message = mock_logger.info.call_args[0][0]
    assert "3.500" in message
    # min/max stay per-request values, unweighted.
    assert "min 2.000" in message
    assert "max 4.000" in message


def test_log_spec_stats_missing_decoding_iter_falls_back_to_weight_one():
    """Requests without a usable decoding_iter contribute with weight 1."""
    wrapper = _make_lm_eval_wrapper()
    outputs = [
        _make_spec_output(tokens_per_iter=2.0, decoding_iter=None),
        _make_spec_output(tokens_per_iter=4.0, decoding_iter=0),
    ]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    message = mock_logger.info.call_args[0][0]
    assert "3.000" in message  # both fall back to weight 1 -> plain mean


def test_log_spec_stats_skips_requests_without_metrics():
    """Requests lacking spec metrics are excluded, not counted as zero."""
    wrapper = _make_lm_eval_wrapper()
    outputs = [
        _make_spec_output(tokens_per_iter=3.0),
        _make_spec_output(),  # no metrics (e.g. dropped by the engine)
    ]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    assert mock_logger.info.call_count == 1
    message = mock_logger.info.call_args[0][0]
    assert "3.000" in message
    assert "n=1" in message


def test_log_spec_stats_silent_on_non_spec_run():
    """A run with no speculative metrics at all logs nothing."""
    wrapper = _make_lm_eval_wrapper()
    outputs = [_make_spec_output(), _make_spec_output()]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    mock_logger.info.assert_not_called()


def test_log_spec_stats_reports_corpus_ar():
    """AR sums accepted/drafted over requests with populated counters."""
    wrapper = _make_lm_eval_wrapper()
    outputs = [
        _make_spec_output(tokens_per_iter=2.0, accepted_drafted=(30, 40)),
        _make_spec_output(tokens_per_iter=2.0, accepted_drafted=(3, 4)),
        _make_spec_output(tokens_per_iter=2.0),  # no perf metrics
    ]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    assert mock_logger.info.call_count == 2  # AL line + AR line
    ar_message = mock_logger.info.call_args_list[1][0][0]
    assert "AR" in ar_message
    assert "33/44" in ar_message
    assert "75.0%" in ar_message


def test_log_spec_stats_skips_ar_without_drafted_tokens():
    """Zeroed counters (no drafting) produce no AR line, only AL."""
    wrapper = _make_lm_eval_wrapper()
    outputs = [
        _make_spec_output(tokens_per_iter=2.0, accepted_drafted=(0, 0)),
    ]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    assert mock_logger.info.call_count == 1
    assert "AL" in mock_logger.info.call_args[0][0]


def test_log_spec_stats_ar_without_al_samples():
    """AR still reports when no request exposed avg_decoded_tokens_per_iter."""
    wrapper = _make_lm_eval_wrapper()
    outputs = [_make_spec_output(accepted_drafted=(1, 2))]
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        wrapper._log_spec_stats(outputs)
    assert mock_logger.info.call_count == 1
    ar_message = mock_logger.info.call_args[0][0]
    assert "AR" in ar_message
    assert "1/2" in ar_message


def test_spec_stats_enables_return_perf_metrics(monkeypatch):
    """TLLM_EVAL_SPEC_STATS=1 opts sampling params into perf metrics (AR)."""
    from tensorrt_llm.evaluate.lm_eval import SPEC_STATS_ENV_VAR

    monkeypatch.setenv(SPEC_STATS_ENV_VAR, "1")
    wrapper = _make_lm_eval_wrapper()
    out = wrapper._get_sampling_params({"max_gen_toks": 32})
    assert out.return_perf_metrics is True


def test_no_spec_stats_leaves_return_perf_metrics_off(monkeypatch):
    """Without the env var, perf metrics stay off (zero overhead default)."""
    from tensorrt_llm.evaluate.lm_eval import SPEC_STATS_ENV_VAR

    monkeypatch.delenv(SPEC_STATS_ENV_VAR, raising=False)
    wrapper = _make_lm_eval_wrapper()
    out = wrapper._get_sampling_params({"max_gen_toks": 32})
    assert out.return_perf_metrics is False


def test_generate_until_logs_spec_stats_when_enabled(monkeypatch):
    """generate_until forwards the collected outputs to _log_spec_stats."""
    from tensorrt_llm.evaluate.lm_eval import SPEC_STATS_ENV_VAR, LmEvalWrapper

    monkeypatch.setenv(SPEC_STATS_ENV_VAR, "1")
    fake_result = MagicMock()
    fake_result.outputs = [MagicMock(text="42")]
    fake_output = MagicMock()
    fake_output.result.return_value = fake_result
    fake_llm = MagicMock()
    fake_llm.generate_async.return_value = fake_output

    wrapper = LmEvalWrapper(llm=fake_llm)
    fake_request = MagicMock()
    fake_request.args = ("hello world", {})

    with patch.object(LmEvalWrapper, "_log_spec_stats") as mock_stats:
        wrapper.generate_until([fake_request], disable_tqdm=True)

    mock_stats.assert_called_once_with([fake_result])


def test_generate_until_skips_spec_stats_when_disabled(monkeypatch):
    """Without the env var, generate_until never touches _log_spec_stats."""
    from tensorrt_llm.evaluate.lm_eval import SPEC_STATS_ENV_VAR, LmEvalWrapper

    monkeypatch.delenv(SPEC_STATS_ENV_VAR, raising=False)
    fake_output = MagicMock()
    fake_output.result.return_value.outputs = [MagicMock(text="42")]
    fake_llm = MagicMock()
    fake_llm.generate_async.return_value = fake_output

    wrapper = LmEvalWrapper(llm=fake_llm)
    fake_request = MagicMock()
    fake_request.args = ("hello world", {})

    with patch.object(LmEvalWrapper, "_log_spec_stats") as mock_stats:
        wrapper.generate_until([fake_request], disable_tqdm=True)

    mock_stats.assert_not_called()


# ===========================================================================
# TLLM_EVAL_MAX_IN_FLIGHT — windowed generate_until
# ===========================================================================
#
# The windowed path caps concurrently in-flight requests at W, tops the
# window up as responses complete, and collects outputs into an
# index-addressed list. The correctness property the whole design exists to
# preserve is SUBMISSION-ORDER RESULTS under arbitrary completion order;
# the liveness property is that a failed request propagates promptly
# instead of deadlocking behind other in-flight waiters.


class _FakeAsyncOutput:
    """Async handle whose blocking .result() is supplied by the test."""

    def __init__(self, result_fn):
        self._result_fn = result_fn

    def result(self):
        return self._result_fn()


def _text_result(text: str) -> MagicMock:
    result = MagicMock()
    result.outputs = [MagicMock(text=text)]
    return result


def _make_windowed_llm(events, error_idx=None):
    """LLM whose request i blocks until events[i] is set, then yields resp-i.

    Returns the fake llm and the (mutated) list of submitted request indices,
    so tests can observe how far submission has progressed.
    """
    submitted = []
    llm = MagicMock()
    llm.tokenizer = MagicMock()

    def generate_async(prompt, sampling_params=None, streaming=False):
        idx = len(submitted)
        submitted.append(idx)

        def _result():
            assert events[idx].wait(timeout=30), f"request {idx} never released"
            if error_idx is not None and idx == error_idx:
                raise RuntimeError(f"request {idx} failed")
            return _text_result(f"resp-{idx}")

        return _FakeAsyncOutput(_result)

    llm.generate_async = generate_async
    return llm, submitted


def _make_requests(n: int) -> list:
    requests = []
    for i in range(n):
        request = MagicMock()
        request.args = (f"prompt-{i}", {})
        requests.append(request)
    return requests


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_windowed_caps_in_flight_and_preserves_order(monkeypatch):
    """At most W requests in flight, and results follow submission order.

    Request 1 is completed before request 0 (out-of-order completion); the
    window tops up with request 2 only after that completion, and the
    returned list is still resp-0..resp-4 in submission order.
    """
    monkeypatch.setenv(MAX_IN_FLIGHT_ENV_VAR, "2")
    total = 5
    events = [threading.Event() for _ in range(total)]
    llm, submitted = _make_windowed_llm(events)
    wrapper = LmEvalWrapper(llm=llm)
    assert wrapper.max_in_flight == 2

    returned = []
    worker = threading.Thread(
        target=lambda: returned.append(
            wrapper.generate_until(_make_requests(total), disable_tqdm=True)
        )
    )
    worker.start()
    try:
        # Only the first W requests are submitted while none have completed.
        assert _wait_until(lambda: len(submitted) == 2)
        time.sleep(0.05)
        assert len(submitted) == 2, "window overshot max_in_flight"
        # Completing request 1 (out of order) tops the window up by one.
        events[1].set()
        assert _wait_until(lambda: len(submitted) == 3)
        time.sleep(0.05)
        assert len(submitted) == 3
    finally:
        for event in events:
            event.set()
        worker.join(timeout=30)
    assert not worker.is_alive()
    assert returned and returned[0] == [f"resp-{i}" for i in range(total)]


def test_windowed_failed_request_raises_without_waiting(monkeypatch):
    """A failed request propagates while another request is still in flight.

    Regression guard for the deadlock the review called out: a blocking
    pool shutdown would join every other outstanding waiter with no
    cancellation or timeout, so if any of them never resolved the exception
    could not escape and the eval hung instead of failing.
    """
    monkeypatch.setenv(MAX_IN_FLIGHT_ENV_VAR, "2")
    events = [threading.Event() for _ in range(2)]
    llm, _ = _make_windowed_llm(events, error_idx=0)
    wrapper = LmEvalWrapper(llm=llm)
    events[0].set()  # request 0 fails immediately; request 1 stays blocked
    try:
        with pytest.raises(RuntimeError, match="request 0 failed"):
            wrapper.generate_until(_make_requests(2), disable_tqdm=True)
        # The exception escaped while request 1 had not resolved.
        assert not events[1].is_set()
    finally:
        events[1].set()  # release the lingering waiter thread


def test_windowed_window_larger_than_request_count(monkeypatch):
    """W >= len(requests) submits each request exactly once and stays ordered."""
    monkeypatch.setenv(MAX_IN_FLIGHT_ENV_VAR, "64")
    total = 3
    events = [threading.Event() for _ in range(total)]
    for event in events:
        event.set()
    llm, submitted = _make_windowed_llm(events)
    wrapper = LmEvalWrapper(llm=llm)
    result = wrapper.generate_until(_make_requests(total), disable_tqdm=True)
    assert result == [f"resp-{i}" for i in range(total)]
    assert submitted == list(range(total))


def test_windowed_empty_request_list(monkeypatch):
    """Zero requests short-circuit without creating a thread pool."""
    monkeypatch.setenv(MAX_IN_FLIGHT_ENV_VAR, "2")
    llm, _ = _make_windowed_llm([])
    wrapper = LmEvalWrapper(llm=llm)
    assert wrapper.generate_until([], disable_tqdm=True) == []


def test_windowed_invokes_partial_scorer(monkeypatch):
    """The windowed path feeds every completion to the partial scorer."""
    from tensorrt_llm.evaluate.lm_eval import _RunningScoreTracker

    monkeypatch.setenv(MAX_IN_FLIGHT_ENV_VAR, "2")
    total = 3
    events = [threading.Event() for _ in range(total)]
    for event in events:
        event.set()
    llm, _ = _make_windowed_llm(events)
    wrapper = LmEvalWrapper(
        llm=llm,
        partial_scores_every=1,
        partial_scoring_task_dict={"fake_task": _FakeTask()},
    )
    with (
        patch.object(_RunningScoreTracker, "update") as mock_update,
        patch.object(_RunningScoreTracker, "maybe_log") as mock_log,
    ):
        wrapper.generate_until(_make_requests(total), disable_tqdm=True)
    assert mock_update.call_count == total
    assert mock_log.call_count == total


# ===========================================================================
# MultimodalLmEvalWrapper.generate_until — partial scorer wiring
# ===========================================================================


def test_multimodal_generate_until_invokes_partial_scorer():
    """The multimodal override scores the post-processed text lm-eval sees."""
    from tensorrt_llm.evaluate.lm_eval import _RunningScoreTracker

    fake_output = MagicMock()
    fake_output.result.return_value.outputs = [MagicMock(text="<think>reasoning</think>42")]
    fake_llm = MagicMock()
    fake_llm.tokenizer = MagicMock()
    fake_llm.input_processor = MagicMock()
    fake_llm.generate_async.return_value = fake_output

    with patch.object(MultimodalLmEvalWrapper, "_get_model_type", return_value="gemma3"):
        wrapper = MultimodalLmEvalWrapper(
            fake_llm,
            sampling_params=None,
            model_type="gemma3",
            post_process_fn=lambda s: s.split("</think>")[-1],
            partial_scores_every=1,
            partial_scoring_task_dict={"fake_task": _FakeTask()},
        )

    fake_request = MagicMock()
    fake_request.args = ("prompt", {}, {"visual": [MagicMock()]})

    with (
        patch(
            "tensorrt_llm.evaluate.lm_eval.prompt_inputs",
            side_effect=lambda p: {"prompt": p},
        ),
        patch(
            "tensorrt_llm.evaluate.lm_eval.convert_image_mode",
            side_effect=lambda img, mode: img,
        ),
        patch.object(_RunningScoreTracker, "update") as mock_update,
        patch.object(_RunningScoreTracker, "maybe_log") as mock_log,
    ):
        results = wrapper.generate_until([fake_request], disable_tqdm=True)

    assert results == ["42"]
    # The scorer must see the post-processed text, not the raw output.
    mock_update.assert_called_once_with(fake_request, "42")
    mock_log.assert_called_once_with(1, 1)


# ===========================================================================
# End-to-end: real lm-eval evaluator over a mocked LLM
# ===========================================================================
#
# Runs lm_eval.evaluator.evaluate() — real ConfigurableTask, real filter
# pipeline, real metric aggregation — against LmEvalWrapper wrapping a
# mocked LLM that returns canned responses. This exercises the exact
# calling conventions between the harness and the wrapper (instance
# shapes, filtered_resps, the process_results list convention) that pure
# unit mocks can get subtly wrong; the GSM8K first-character bug above
# survived precisely because nothing ran the real harness loop.


_E2E_DOCS = [
    {"question": "2+2?", "answer": "4"},
    {"question": "3+4?", "answer": "7"},
    {"question": "5+6?", "answer": "11"},
    {"question": "10-3?", "answer": "7"},
]
# Model answers: 3 correct, 1 wrong ("12" != "11") -> exact_match 0.75.
_E2E_RESPONSES = [
    "The answer is 4.",
    "The answer is 7.",
    "The answer is 12.",
    "The answer is 7.",
]


def _toy_task():
    """A real generate_until ConfigurableTask over an in-memory dataset."""
    import datasets
    from lm_eval.api.task import ConfigurableTask

    return ConfigurableTask(
        config={
            "task": "toy_arith",
            "custom_dataset": lambda **kwargs: datasets.DatasetDict(
                {"test": datasets.Dataset.from_list(_E2E_DOCS)}
            ),
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "Q: {{question}}\nA:",
            "doc_to_target": "{{answer}}",
            "generation_kwargs": {"until": ["\n"], "do_sample": False},
            "filter_list": [
                {
                    "name": "strict-match",
                    "filter": [
                        {"function": "regex", "regex_pattern": r"(-?[0-9]+)"},
                        {"function": "take_first"},
                    ],
                }
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
        }
    )


def _canned_llm(responses):
    """LLM whose generate_async yields the canned texts in submission order."""
    llm = MagicMock()
    llm.tokenizer = MagicMock()
    response_iter = iter(responses)

    def generate_async(prompt, sampling_params=None, streaming=False):
        text = next(response_iter)
        output = MagicMock()
        output.result.return_value = _text_result(text)
        return output

    llm.generate_async = generate_async
    return llm


def test_e2e_harness_final_score_over_mocked_llm():
    """The real lm-eval evaluator scores canned responses correctly."""
    from lm_eval.evaluator import evaluate

    task = _toy_task()
    wrapper = LmEvalWrapper(llm=_canned_llm(_E2E_RESPONSES))
    results = evaluate(
        lm=wrapper,
        task_dict={"toy_arith": task},
        bootstrap_iters=0,
        log_samples=False,
    )
    score = results["results"]["toy_arith"]["exact_match,strict-match"]
    assert score == pytest.approx(0.75)


def test_e2e_partial_scores_match_final_score():
    """Partial-score estimates over the full corpus agree with the harness.

    Uses the REAL task's filters and process_results inside
    _RunningScoreTracker (no fakes), so a calling-convention mismatch
    between the tracker and lm-eval internals disables the tracker and
    fails this test.
    """
    from lm_eval.evaluator import evaluate

    task = _toy_task()
    task_dict = {"toy_arith": task}
    wrapper = LmEvalWrapper(
        llm=_canned_llm(_E2E_RESPONSES),
        partial_scores_every=2,
        partial_scoring_task_dict=task_dict,
    )
    with patch("tensorrt_llm.evaluate.lm_eval.logger") as mock_logger:
        results = evaluate(
            lm=wrapper,
            task_dict=task_dict,
            bootstrap_iters=0,
            log_samples=False,
        )
    messages = [call.args[0] for call in mock_logger.info.call_args_list]
    assert not any("Partial scoring disabled" in m for m in messages), (
        "tracker was disabled by a scoring failure against the real task"
    )
    partial = [m for m in messages if "Partial scores" in m]
    # interval=2 over 4 responses -> logs at 2/4 and 4/4.
    assert len(partial) == 2
    assert "2/4" in partial[0]
    assert "4/4" in partial[1]
    # The final running estimate agrees with the true score (0~100 scale).
    assert "75.00" in partial[1]
    score = results["results"]["toy_arith"]["exact_match,strict-match"]
    assert score == pytest.approx(0.75)


def test_e2e_windowed_matches_final_score(monkeypatch):
    """The windowed path produces the same harness score as submit-all.

    Windowing must be a pure scheduling change — outputs are collected in
    submission order, so the score is identical to the default path.
    """
    from lm_eval.evaluator import evaluate

    monkeypatch.setenv(MAX_IN_FLIGHT_ENV_VAR, "2")
    task = _toy_task()
    wrapper = LmEvalWrapper(llm=_canned_llm(_E2E_RESPONSES))
    assert wrapper.max_in_flight == 2
    results = evaluate(
        lm=wrapper,
        task_dict={"toy_arith": task},
        bootstrap_iters=0,
        log_samples=False,
    )
    score = results["results"]["toy_arith"]["exact_match,strict-match"]
    assert score == pytest.approx(0.75)


# ===========================================================================
# post_processing — Kimi K3 channel-structured MMMU answer extraction
# ===========================================================================
#
# Kimi K3 emits a channel-structured chat format whose reasoning ends with
# ``<|close|>think<|sep|>`` (NOT ``</think>``) and whose final answer lives in a
# ``<|open|>response<|sep|> X <|close|>response<|sep|>`` channel. The K2.5
# strip_thinking() path keys on ``</think>`` and therefore cannot see the
# answer, silently dropping ~6-7 MMMU points even when the model is correct
# (observed on the real checkpoint: full mmmu_val 67.67 -> 74.11 by parsing the
# channel alone). These tests pin the new extractor and guard that the K2.5
# path is unchanged.

from tensorrt_llm.evaluate.post_processing import (  # noqa: E402
    extract_kimi_k3_mmmu_answer,
    strip_thinking,
    strip_thinking_and_extract_mmmu_answer,
)


def _k3_output(thinking: str, answer: str) -> str:
    """Build a well-formed Kimi K3 channel-structured output string."""
    return (
        f"{thinking}<|close|>think<|sep|>"
        f"<|open|>response<|sep|>{answer}<|close|>response<|sep|>"
        f"<|close|>message<|sep|>"
    )


def test_k3_channel_bare_letter():
    """Answer is a bare option letter inside the response channel."""
    out = _k3_output("Reasoning about the options... I'll go with C.", "C")
    assert extract_kimi_k3_mmmu_answer(out) == "C"


def test_k3_channel_real_samples_recovered():
    """Real committed mmmu_val samples the old parser scored wrong.

    The channel holds the correct letter: accounting doc_id 7->C, 12->D, 21->A.
    """
    assert (
        extract_kimi_k3_mmmu_answer(
            _k3_output("...Total debits adjusted = 126,925. Final: C.", "C")
        )
        == "C"
    )
    assert (
        extract_kimi_k3_mmmu_answer(_k3_output("...Ending = 0. Option D. Final just D.", "D"))
        == "D"
    )
    assert (
        extract_kimi_k3_mmmu_answer(
            _k3_output("...commonly known by that name, so True. (A).", "A")
        )
        == "A"
    )


def test_k3_channel_parenthesized_answer():
    """Channel content ``(C) Photos 2 & 3`` reduces to the option letter."""
    out = _k3_output("Both use negative space.", "(C) Photos 2 & 3")
    assert extract_kimi_k3_mmmu_answer(out) == "C"


def test_k3_channel_answer_is_phrase():
    """Channel content ``The answer is (D).`` reduces to the option letter."""
    out = _k3_output("Long derivation here.", "The answer is (D).")
    assert extract_kimi_k3_mmmu_answer(out) == "D"


def test_k3_truncated_after_channel_open():
    """Truncation right after the channel opened still yields the letter.

    The regex boundary falls back to end-of-text for the unterminated span.
    """
    out = (
        "Some reasoning.<|close|>think<|sep|>"
        "<|open|>response<|sep|>B"
    )  # cut off before <|close|>response
    assert extract_kimi_k3_mmmu_answer(out) == "B"


def test_k3_last_channel_wins():
    """When multiple response channels are present, the final one is the answer."""
    out = (
        "r1<|close|>think<|sep|><|open|>response<|sep|>A<|close|>response<|sep|>"
        "<|open|>response<|sep|>E<|close|>response<|sep|><|close|>message<|sep|>"
    )
    assert extract_kimi_k3_mmmu_answer(out) == "E"


def test_k3_no_channel_bare_letter_falls_back():
    """Short direct answers with no channel go through the K2.5 fallback."""
    assert extract_kimi_k3_mmmu_answer("C") == "C"
    assert extract_kimi_k3_mmmu_answer("Answer: (B)") == "B"


def test_k3_no_channel_truncated_thinking_does_not_crash():
    """Thinking truncated before the channel opened (finish=length).

    No channel to parse -> fall back; must not raise and must not fabricate.
    """
    truncated = (
        "Let me reason step by step about this very long problem "
        "that never reaches a final answer channel " * 20
    )
    # Should not raise; returns whatever the fallback cascade yields (the model
    # genuinely did not emit an answer, so the exact value is not asserted).
    out = extract_kimi_k3_mmmu_answer(truncated)
    assert isinstance(out, str)


def test_k3_empty_input():
    assert extract_kimi_k3_mmmu_answer("") == ""


def test_k3_scrubs_residual_special_tokens():
    """Residual ``<|...|>`` tokens inside a channel span are scrubbed.

    They must not be returned as part of the answer.
    """
    out = (
        "t<|close|>think<|sep|><|open|>response<|sep|>"
        "<|reserved|>D<|close|>response<|sep|><|close|>message<|sep|>"
    )
    assert extract_kimi_k3_mmmu_answer(out) == "D"


def test_k2_5_strip_thinking_path_unchanged():
    """Guard: the new K3 extractor must not alter the K2.5 </think> behavior."""
    k25 = "<think>chain of thought here</think>Answer: (C)"
    # K2.5 path still extracts C directly.
    assert strip_thinking_and_extract_mmmu_answer(k25) == "C"
    # strip_thinking still returns content after the last </think>.
    assert strip_thinking(k25) == "Answer: (C)"
    # And the K3 extractor, given a </think> blob with no K3 response channel,
    # defers to the K2.5 cascade and returns the same answer.
    assert extract_kimi_k3_mmmu_answer(k25) == "C"


def test_k3_channel_extracting_to_nothing_falls_back_to_cascade():
    """A channel whose content extracts to nothing must not mask the fallback.

    "** **" survives the special-token scrub as non-empty text, but the
    cascade's markdown-bold stripping reduces it to "" — the extractor must
    keep scanning and recover the letter from the reasoning text instead of
    returning the empty string.
    """
    out = _k3_output("Elimination shows the answer is (B).", "** **")
    assert extract_kimi_k3_mmmu_answer(out) == "B"


# ===========================================================================
# _override_stop_strings — CLI override of the task yaml's ``until`` list
# ===========================================================================


class _FakeTaskConfig:
    """Minimal stand-in for lm-eval's ``TaskConfig``."""

    def __init__(self, generation_kwargs):
        self.generation_kwargs = generation_kwargs


class _FakeStopStringTask:
    """Minimal stand-in for lm-eval's ``ConfigurableTask``."""

    def __init__(self, generation_kwargs=None):
        self.config = _FakeTaskConfig(generation_kwargs)

    def set_config(self, key, value):
        setattr(self.config, key, value)


def test_override_stop_strings_replaces_until():
    """The yaml's ``until`` list is replaced, not appended to."""
    task = _FakeStopStringTask({"until": ["Question:", "</s>"]})
    _override_stop_strings(task, ["<|im_end|>"])
    assert task.config.generation_kwargs["until"] == ["<|im_end|>"]


def test_override_stop_strings_preserves_other_gen_kwargs():
    """Only ``until`` changes; max_gen_toks / do_sample survive."""
    task = _FakeStopStringTask({"until": ["Question:"], "max_gen_toks": 256, "do_sample": False})
    _override_stop_strings(task, ["</s>"])
    assert task.config.generation_kwargs == {
        "until": ["</s>"],
        "max_gen_toks": 256,
        "do_sample": False,
    }


def test_override_stop_strings_on_task_without_generation_kwargs():
    """A task yaml with no gen_kwargs at all still gets an ``until`` list."""
    task = _FakeStopStringTask(None)
    _override_stop_strings(task, ["</s>"])
    assert task.config.generation_kwargs == {"until": ["</s>"]}


def test_override_stop_strings_empty_list_disables_stopping():
    """An explicit empty list is honored, i.e. generate to max_gen_toks."""
    task = _FakeStopStringTask({"until": ["Question:"]})
    _override_stop_strings(task, [])
    assert task.config.generation_kwargs["until"] == []


def test_override_stop_strings_does_not_alias_caller_list():
    """The stored list is a copy; later caller mutation must not leak in."""
    stop_strings = ["</s>"]
    task = _FakeStopStringTask({"until": ["Question:"]})
    _override_stop_strings(task, stop_strings)
    stop_strings.append("<|im_end|>")
    assert task.config.generation_kwargs["until"] == ["</s>"]


def test_override_stop_strings_does_not_mutate_original_gen_kwargs():
    """The task's original gen_kwargs dict object is left untouched."""
    original = {"until": ["Question:"], "max_gen_toks": 256}
    task = _FakeStopStringTask(original)
    _override_stop_strings(task, ["</s>"])
    assert original == {"until": ["Question:"], "max_gen_toks": 256}
