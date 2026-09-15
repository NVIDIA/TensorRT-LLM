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
"""``Logger.log`` %-formats printf-style call sites instead of joining them.

Historically ``Logger.log`` space-joined its arguments, but well over a
hundred call sites across the tree are written in the stdlib-logging style
``logger.warning("fell back to %s because %s", choice, reason)``.  Those lines
printed the format string LITERALLY with the values appended -- mangling
exactly the log lines someone reads while debugging the situation the line
describes.  These tests pin the fix: a leading format string with conversion
specifiers and matching arguments is %-formatted; everything else keeps the
historical join, and a mismatched format never raises out of a log call.

CPU only; the logger touches no GPU.
"""

import logging

import pytest

from tensorrt_llm.logger import Logger, logger


@pytest.fixture()
def emitted():
    """Capture the fully formatted records the TRT-LLM logger emits."""
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.DEBUG)
    logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        logger._logger.removeHandler(handler)


def test_percent_style_call_site_is_formatted(emitted):
    logger.error("fell back to %s because %s", "choiceA", "reasonB")
    assert len(emitted) == 1
    assert "fell back to choiceA because reasonB" in emitted[0]
    assert "%s" not in emitted[0]


def test_numeric_and_precision_specifiers(emitted):
    logger.error("took %.2f s over %d blocks", 1.2345, 7)
    assert "took 1.23 s over 7 blocks" in emitted[0]


def test_mapping_argument_feeds_named_fields(emitted):
    logger.error("window %(window)d demoted to %(target)s", {"window": 2048, "target": "V1"})
    assert "window 2048 demoted to V1" in emitted[0]


def test_plain_multi_argument_join_is_unchanged(emitted):
    logger.error("loaded", 3, "shards")
    assert "loaded 3 shards" in emitted[0]


def test_single_string_with_specifier_is_untouched(emitted):
    # No arguments: nothing to format with; the text must survive verbatim.
    logger.error("literal %s stays")
    assert "literal %s stays" in emitted[0]


def test_escaped_percent_is_not_a_specifier(emitted):
    logger.error("100%% done", "extra")
    assert "100%% done extra" in emitted[0]


def test_mismatched_arguments_fall_back_to_join(emitted):
    # Too few and too many arguments: never raise from inside a log call.
    logger.error("%s and %s", "only-one")
    logger.error("%s", "one", "two")
    assert "%s and %s only-one" in emitted[0]
    assert "%s one two" in emitted[1]


def test_non_string_first_argument_joins(emitted):
    logger.error(404, "not found")
    assert "404 not found" in emitted[0]


@pytest.mark.parametrize(
    "msg,expected",
    [
        (("plain",), None),
        (("no specifier here", "arg"), None),
        (("%s", "x"), "x"),
        (("%(k)r", {"k": "v"}), "'v'"),
        (("%d%%", 5), "5%"),
        (("%ld", 7), "7"),
    ],
)
def test_percent_format_shapes(msg, expected):
    assert Logger._percent_format(msg) == expected
