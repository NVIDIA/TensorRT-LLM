# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Word statistics over plain text."""

import re
from collections import Counter

WORD = re.compile(r"[a-z']+")


def words(text):
    """Return the words of text in order, lowercased: runs of letters and apostrophes."""
    return WORD.findall(text)


def word_counts(text):
    """Return a Counter of the words of text."""
    return Counter(words(text))


def top_words(text, n):
    """Return the n most frequent words as (word, count) pairs, ties in alphabetical order."""
    return sorted(word_counts(text).items(), key=lambda item: (-item[1], item[0]))[:n]


def average_word_length(text):
    """Return the mean length of the words of text, or 0.0 when it has none."""
    found = words(text)
    if not found:
        return 0.0
    return sum(len(word) for word in found) / len(found)
