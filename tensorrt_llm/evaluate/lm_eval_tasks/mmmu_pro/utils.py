# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Answer-parsing / scoring logic adapted from MMMU
# (https://github.com/MMMU-Benchmark/MMMU) and the MMMU lm-evaluation-harness
# task, with minor adjustments to support MMMU Pro's 10-option multiple-choice
# format and its uniform (all multiple-choice) question type.
import ast
import os
import random
import re

import numpy as np

random.seed(42)


# The prompt suffix is switched by setting MMMU_PRO_PROMPT_MODE=cot in the
# environment before launching ``trtllm-eval mmmu_pro``.  "direct" mirrors
# the MMMU-Benchmark's ``direct/standard`` template; "cot" mirrors the
# ``cot/standard`` template and is the one the Gemma4 HF blog numbers
# appear to use (close to +10-25 pp headline on MMMU Pro, especially for
# smaller / MoE models that benefit the most from chain-of-thought).
# Source: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml
_PROMPT_SUFFIX_DIRECT = "Answer with the option's letter from the given choices directly."
_PROMPT_SUFFIX_COT = (
    "Answer the preceding multiple choice question. The last line of "
    "your response should be of the following format: 'Answer: $LETTER' "
    "(without quotes) where LETTER is one of options. Think step by step "
    "before answering."
)

_MODE = os.environ.get("MMMU_PRO_PROMPT_MODE", "direct").lower()
_PROMPT_SUFFIX = _PROMPT_SUFFIX_COT if _MODE == "cot" else _PROMPT_SUFFIX_DIRECT

MULTI_CHOICE_EXAMPLE_FORMAT = "{}\n\n{}\n\n" + _PROMPT_SUFFIX


START_CHR = "A"


def doc_to_image(doc):
    """Return the list of image PIL objects for a single document.

    MMMU Pro questions embed ``<image {i}>`` placeholders (up to 7); this
    function collects the corresponding ``image_{i}`` fields in the order
    referenced in the question so they align with the expanded placeholders
    in ``doc_to_text``.  Empty ``image_{i}`` fields are skipped.
    """
    input_text = _doc_to_text(doc)
    image_placeholders = [
        img.replace(" ", "_").replace("<", "").replace(">", "")
        for img in re.findall("<image [1-7]>", input_text)
    ]
    visuals = [doc[img] for img in image_placeholders if doc.get(img) is not None]
    return visuals


def doc_to_text(doc):
    prompt = _doc_to_text(doc)
    for i in range(1, 8):
        prompt = prompt.replace(f"<image {i}>", "<image>")
    return prompt


def _doc_to_text(doc):
    """Build the textual prompt with the original ``<image {i}>`` tags intact."""
    choices_str = ""
    for i, choice in enumerate(ast.literal_eval(doc["options"])):
        choices_str += f"\n({chr(ord(START_CHR) + i)}) {choice}"
    choices_str = choices_str.lstrip()
    return MULTI_CHOICE_EXAMPLE_FORMAT.format(doc["question"], choices_str)


def process_results(doc, results):
    option_strs = ast.literal_eval(doc["options"])
    option_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    all_choices = option_letters[: len(option_strs)]
    index2ans = {index: ans for index, ans in zip(option_letters, option_strs)}

    pred = parse_multi_choice_response(results[0], all_choices, index2ans)
    is_correct = doc["answer"] == pred
    return {"acc": float(is_correct)}


# ----------- Answer parsing (from MMMU/MMMU) -----------
_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+)?answer\s*(?:is\s*)?[:\-]?\s*\(?\s*([A-J])\s*\)?",
    re.IGNORECASE,
)


def parse_multi_choice_response(response, all_choices, index2ans):
    # CoT prompts ask for "Answer: $LETTER" on the last line — pick that up
    # first so we don't accidentally match an earlier letter that appears
    # inside the reasoning.  Walk the response lines in reverse so the
    # LAST matching letter wins (final answer lines typically come last).
    valid_set = set(all_choices)
    for line in reversed(response.splitlines()):
        m = _ANSWER_RE.search(line)
        if m and m.group(1).upper() in valid_set:
            return m.group(1).upper()

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[int(np.argmax(start_indexes))]
    else:
        pred_index = candidates[0]

    return pred_index
