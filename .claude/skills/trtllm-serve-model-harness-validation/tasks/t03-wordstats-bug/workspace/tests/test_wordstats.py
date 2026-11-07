# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

from wordstats import average_word_length, top_words, word_counts, words


class WordsTest(unittest.TestCase):
    def test_words_are_lowercased(self):
        self.assertEqual(words("The Cat sat."), ["the", "cat", "sat"])

    def test_apostrophes_stay_inside_words(self):
        self.assertEqual(words("it's fine"), ["it's", "fine"])


class CountsTest(unittest.TestCase):
    def test_counts_ignore_case(self):
        self.assertEqual(word_counts("Red red RED blue")["red"], 3)

    def test_top_words_break_ties_alphabetically(self):
        self.assertEqual(top_words("b a c b a", 2), [("a", 2), ("b", 2)])


class AverageTest(unittest.TestCase):
    def test_average_word_length(self):
        self.assertEqual(average_word_length("ab abcd"), 3.0)

    def test_text_without_words(self):
        self.assertEqual(average_word_length("... !"), 0.0)


if __name__ == "__main__":
    unittest.main()
