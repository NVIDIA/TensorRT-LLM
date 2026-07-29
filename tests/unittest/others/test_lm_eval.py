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
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

from tensorrt_llm.evaluate.covost2 import CoVoST2
from tensorrt_llm.evaluate.lm_eval import (
    LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER,
    LmEvalWrapper,
    MultimodalLmEvalWrapper,
)
from tensorrt_llm.evaluate.lm_eval_tasks.aime.utils import (
    is_equiv,
    last_boxed_only_string,
    process_results,
    remove_boxed,
    strip_string,
)
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.sampling_params import SamplingParams

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


# Uses ``gemma3`` by default because it is always registered regardless of
# transformers version; the wrapper's interleave logic itself is generic.
def _make_multimodal_wrapper(model_type: str = "gemma3") -> MultimodalLmEvalWrapper:
    fake_llm = MagicMock()
    fake_llm.tokenizer = MagicMock()
    fake_llm.input_processor = MagicMock()
    fake_llm.input_processor.processor = MagicMock()
    with patch.object(MultimodalLmEvalWrapper, "_get_model_type", return_value=model_type):
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


def test_single_image_does_not_interleave():
    """Single-image prompts never need interleaving.

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
    wrapper = _make_multimodal_wrapper()
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


def test_multi_image_string_format_skips_interleave():
    """STRING-format chat templates skip the interleaving path.

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
    wrapper = _make_multimodal_wrapper()
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
    wrapper = _make_multimodal_wrapper()
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
