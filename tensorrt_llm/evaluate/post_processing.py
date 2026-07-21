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
r"""Post-processing utilities for thinking-model outputs in lm-eval evaluations.

These helpers compensate for limitations in lm-evaluation-harness's default
answer-extraction regex on multiple-choice benchmarks (notably MMMU). The
default regex is too strict and fails on common formats produced by reasoning
models, e.g. ``Answer: (B)``, ``**Answer: B**``, ``D) Phytoplasma``,
``(A) hierarchical scale``, ``\\boxed{42}``.

This is not specific to TRT-LLM. The same gap is observable when running
Kimi K2.5 through other engines (e.g. vLLM) under the same lm-eval pipeline:
on MMMU val (900 samples), 14-17% of the model's correct answers are scored
as wrong purely because lm-eval's default extractor cannot parse them.

The post-processor below is therefore an adapter layer over lm-eval's MMMU
support, not a model-specific patch. It is intended as an opt-in hook on
``MultimodalLmEvalWrapper``, so the base wrapper stays unchanged for non-
thinking models and runs with the appropriate cleanup for thinking models
such as Kimi K2.5.
"""

import re


def strip_thinking(text: str) -> str:
    """Strip ``<think>...</think>`` reasoning blocks from raw model output.

    Kimi K2.5 produces chain-of-thought reasoning inside ``<think>`` tags
    during offline inference. Returns the content after the last
    ``</think>`` tag.

    If no ``</think>`` is present (e.g. ``finish_reason=length`` truncated
    the thinking block), discards everything from ``<think>`` onward to
    avoid feeding partial reasoning into downstream extraction.
    """
    try:
        idx = text.rindex("</think>") + len("</think>")
        return text[idx:].strip()
    except ValueError:
        # No </think> found — likely finish=length, thinking never completed.
        # Discard everything from <think> onward.
        think_start = text.find("<think>")
        if think_start != -1:
            return text[:think_start].strip()
        return text.strip()


def extract_mmmu_answer(content: str) -> str:
    r"""Extract the final letter answer from cleaned MMMU response text.

    MMMU is a multiple-choice benchmark with options A-E. lm-eval's default
    regex misses many common answer formats produced by reasoning models;
    this cascade recovers them.

    Extraction priority (first match wins):
      1. ``Answer: (B)`` / ``answer is B`` explicit pattern
      2. Leading option letter: ``(A) True``, ``C\\nExplanation: ...``
      3. Short text (<=50 chars) after cleanup — already a clean answer
      4. Long text — search from the end for the final answer:
         a. Last ``\\boxed{...}`` (LaTeX)
         b. Trailing option letter ``(B)`` near end of text
         c. ``(or ...)`` / ``(approximately ...)`` trimming
         d. Last line as fallback
    """
    if not content:
        return ""

    answer = content

    # --- Step 1: "Answer: X" / "answer is X" explicit pattern ---
    # Covers: "Answer: (B)", "**Answer: B**", "The answer is (C)",
    #         "The answer is (C) 50.4 kip.", "The answer is (E)**."
    m = re.search(r"\b[Aa]nswer\s*(?:is|:)\s*\(?\s*([A-Ea-e])\s*\)?", answer)
    if m:
        return m.group(1).upper()

    # --- Step 2: strip markdown bold globally ---
    # **C** → C,  **(B)** → (B),  **$1,249** → $1,249
    answer = re.sub(r"\*\*([^*]+)\*\*", r"\1", answer)

    # --- Step 2b: leading option letter followed by text/newline ---
    # Covers: "(A) True", "(B) No", "(D) synthetic",
    #         "(C) B harmonic minor scale:", "(E) Physiological ...",
    #         "C\n\nExplanation: ..."
    m_leading = re.match(r"^\(?([A-Ea-e])\)?\s*(?:[.:\n\r]|\s+[A-Z]|\s*$)", answer)
    if m_leading:
        return m_leading.group(1).upper()

    # --- Step 3: short text — already a clean answer ---
    # Covers: single letters, short numbers, short phrases, LaTeX
    if len(answer) <= 50:
        # Normalize bare option letters: "(A)" → "A", "c" → "C"
        m_letter = re.fullmatch(r"\(?([A-Za-z])\)?\.?", answer.strip())
        if m_letter:
            return m_letter.group(1).upper()
        return answer.strip()

    # --- Step 4: long text — answer is near the end ---

    # 4a: LaTeX \boxed{...} — take the last occurrence
    boxed = re.findall(r"\\boxed\{([^}]+)\}", answer)
    if boxed:
        return boxed[-1].strip()

    # 4b: trailing option letter — last (X) where X is A-E
    #     e.g. "...therefore\n\n(B)" or "...\n**(C)**"
    #     Search in the last 200 chars for a standalone option letter
    tail = answer[-200:]
    m_tail_option = re.search(r"\(?([A-E])\)?[.\s*]*$", tail)
    if m_tail_option:
        return m_tail_option.group(1).upper()

    # 4c: trim "(or ...)" / "(approximately ...)" / "(accepting ...)"
    #     e.g. "$527.89 million (or approximately $528 million)"
    trimmed = re.sub(
        r"\s*\((?:or|approximately|accepting|and|about)\b[^)]*\)\s*$",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    if len(trimmed) <= 50 and trimmed:
        return trimmed

    # 4d: last line as fallback
    #     Model pattern: explanation paragraphs, then final answer on last line
    last_line = answer.rstrip().rsplit("\n", 1)[-1].strip()
    # Clean up the last line
    last_line = re.sub(r"\*\*([^*]+)\*\*", r"\1", last_line)
    if last_line and len(last_line) <= 100:
        # Check if it's an option letter
        m_ll = re.fullmatch(r"\(?([A-Za-z])\)?\.?", last_line.strip())
        if m_ll:
            return m_ll.group(1).upper()
        return last_line

    # 4e: nothing worked — return full text for lm-eval's parser
    return answer


def strip_thinking_and_extract_mmmu_answer(text: str) -> str:
    """Strip ``<think>...</think>`` then extract an MMMU letter answer.

    Composition of :func:`strip_thinking` (Step 0) and
    :func:`extract_mmmu_answer` (Steps 1-4). This is the offline
    counterpart to the online reasoning-parser path used by
    ``trtllm-serve``: offline lm-eval scoring needs a single string,
    not the ``(reasoning, content)`` split, plus benchmark-specific
    answer extraction.
    """
    return extract_mmmu_answer(strip_thinking(text))


# --- Inkling typed-content channel extraction --------------------------------
# Inkling does not wrap reasoning in ``<think>`` text tags. Instead it emits a
# sequence of typed content blocks delimited by SPECIAL TOKENS, e.g.:
#     <|content_thinking|>reasoning<|end_message|>
#     <|message_model|><|content_text|>visible answer<|end_message|>
#     <|content_model_end_sampling|>
# Reasoning must be routed out and only the ``<|content_text|>`` (visible)
# channel scored — exactly what SGLang's ``InklingDetector`` /
# ``--reasoning-parser inkling`` does online. For offline lm-eval scoring we
# need the generation detokenized WITHOUT ``skip_special_tokens`` so these
# markers survive; ``extract_inkling_content`` then returns only the visible
# content-text so GSM8K/MMLU flexible-extract scores the answer, not the
# chain-of-thought (whose trailing numbers otherwise poison last-number
# extraction, e.g. "**5 cars** ... first 15 minutes" -> wrongly extracts 15).
_INK_CONTENT_THINKING = "<|content_thinking|>"
_INK_CONTENT_TEXT = "<|content_text|>"
_INK_END_MESSAGE = "<|end_message|>"
_INK_CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"
# Any special token that opens a new (non content-text) block or closes one; a
# content-text run ends at the first of these.
_INK_CONTROL_TOKENS = (
    _INK_CONTENT_THINKING,
    _INK_CONTENT_TEXT,
    _INK_END_MESSAGE,
    _INK_CONTENT_MODEL_END_SAMPLING,
    "<|message_model|>",
    "<|message_system|>",
    "<|message_user|>",
    "<|message_tool|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_xml|>",
)
_INK_CONTROL_RE = re.compile("|".join(re.escape(t) for t in _INK_CONTROL_TOKENS))


def extract_inkling_content(text: str) -> str:
    """Return only the visible ``<|content_text|>`` channel from Inkling output.

    Mirrors SGLang's ``InklingDetector``: ``<|content_thinking|>`` blocks are
    reasoning (dropped) and ``<|content_text|>`` blocks are visible content
    (kept). Concatenates all content-text runs and returns them stripped.

    Requires the generation to be detokenized with ``skip_special_tokens=False``
    so the channel markers are present. If no Inkling markers are found (e.g.
    special tokens were skipped, or a non-Inkling model), the input is returned
    unchanged so behavior for every other model/benchmark is untouched.
    """
    if _INK_CONTENT_TEXT not in text and _INK_CONTENT_THINKING not in text:
        return text

    content_parts: list[str] = []
    kind = None  # None | "content" | "reasoning" | "other"
    pos = 0
    for m in _INK_CONTROL_RE.finditer(text):
        segment = text[pos:m.start()]
        if kind == "content" and segment:
            content_parts.append(segment)
        token = m.group(0)
        pos = m.end()
        if token == _INK_CONTENT_TEXT:
            kind = "content"
        elif token == _INK_CONTENT_THINKING:
            kind = "reasoning"
        else:
            # <|end_message|>, <|content_model_end_sampling|>, any <|message_*|>
            # header, or a tool/xml content marker -> close the current block.
            kind = "other"
    # Trailing text after the last control token (e.g. generation stopped at the
    # sampling-end token mid content-text, so there is no closing marker).
    if kind == "content" and pos < len(text):
        content_parts.append(text[pos:])

    # Mirror SGLang's ``InklingDetector``: the visible channel is the
    # concatenation of ``<|content_text|>`` runs ONLY. When Inkling markers are
    # present but no content-text was emitted (e.g. the generation looped or was
    # truncated inside the ``<|content_thinking|>`` block and never produced an
    # answer), SGLang routes everything to ``reasoning_text`` and returns an empty
    # ``normal_text``. We must return the empty visible content here too: falling
    # back to the stripped reasoning text would let a truncated / looping
    # chain-of-thought be scored as if it were the model's answer (its trailing
    # number would be harvested by GSM8K/MMLU flexible-extract) — exactly the
    # failure the reasoning channel is meant to exclude.
    return "".join(content_parts).strip()
