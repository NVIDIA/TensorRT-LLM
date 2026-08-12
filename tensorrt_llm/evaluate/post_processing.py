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
