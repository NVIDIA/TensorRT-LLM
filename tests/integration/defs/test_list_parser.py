# -*- coding: utf-8 -*-
"""Logic for parsing test lists."""

import array
import json
import os
import re
import time
import traceback
from collections import defaultdict
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence

import pytest
from mako.template import Template

from .trt_test_alternative import print_info, print_warning

# from misc.reorder_venv_tests import reorder_tests

kVALID_TEST_LIST_MARKERS = ["XFAIL", "SKIP", "UNSTABLE", "TIMEOUT"]
record_invalid_tests = True

kSTRIP_PARENS_PAT = re.compile(r'\((.*?)\)')


def strip_parens(s):
    """Strips the outer parens from the given string or returns None if not
       found.  Does not check for balanced parentheses."""
    m = kSTRIP_PARENS_PAT.match(s)
    return m.group(1) if m else None


def preprocess_test_list_lines(test_list, lines, mako_opts={}):
    """Apply mako (https://www.makotemplates.org/) preprocessing if requested to the test list before parsing it.

    Args:
         test_list: Name of the test list, used for debugging
         lines:  The lines of the raw test list file
         mako_opts: JSON object containing key-value pairs to forward to Mako

     Returns:
         The processed lines if successful. Raises a RuntimeError on failure.
    """
    mako_tmpl_text = "".join(lines)

    try:
        #assert isinstance(mako_opts, dict)

        def applyMarkerIf(marker, cond, reason):
            if cond:
                return "{} ({})".format(marker, reason)
            else:
                return ""

        def applyMarkerIfPrefix(marker, prefix, reason):
            return applyMarkerIf(marker, prefix == mako_opts["test_prefix"],
                                 reason)

        mako_opts["skipIf"] = lambda c, r: applyMarkerIf("SKIP", c, r)
        mako_opts["xfailIf"] = lambda c, r: applyMarkerIf("XFAIL", c, r)
        mako_opts["unstableIf"] = lambda c, r: applyMarkerIf("UNSTABLE", c, r)

        mako_opts["skipIfPrefix"] = lambda pfx, r: applyMarkerIfPrefix(
            "SKIP", pfx, r)
        mako_opts["xfailIfPrefix"] = lambda pfx, r: applyMarkerIfPrefix(
            "XFAIL", pfx, r)
        mako_opts["unstableIfPrefix"] = lambda pfx, r: applyMarkerIfPrefix(
            "UNSTABLE", pfx, r)

        template = Template(mako_tmpl_text)
        new_text = template.render(**mako_opts)
        if isinstance(new_text, bytes):
            new_text = new_text.decode()
        lines = new_text.splitlines()

        # Strip extra whitespace characters from test names before returning them
        lines = [line.strip() for line in lines]
    except Exception:
        raise RuntimeError("Mako preprocessing of file {} failed: {}".format(
            test_list, traceback.format_exc()))

    return lines


def parse_test_list_lines(test_list, lines, test_prefix, convert_unittest=True):
    """Parses the lines of a test list. Test names returned contain all values within square brackets. Does not process
    each test id value.

       Args:
            test_list: Name of the test list, used for debugging
            lines:  The lines of the test list file
            test_prefix: The value of the --test-prefix option, or None if this option isn't set.
       Returns:
            A tuple (test_names, test_name_to_marker_dict).
            test_names: List of test names parsed from the test list file,
                        ordered by their appearance in the list.
            test_name_to_marker_dict: Dictionary mapping test names to a tuple (parsed test marker, reason string).
    """

    def parse_test_name(s):
        if s.startswith("full:"):
            s = s.lstrip("full:")
            if test_prefix:
                if test_prefix.split('-')[0] in s:
                    s = s.replace(test_prefix.split('-')[0], test_prefix)
            return s
        elif test_prefix:
            return "/".join([test_prefix, s])
        else:
            return s

    def parse_test_line(enumerated_line):
        lineno, line = enumerated_line
        lineno += 1
        # Strip comments and whitespace
        line = line.partition("#")[0].strip()
        if len(line) == 0:
            return (None, None, None)

        # test_name [MARKER] [REASON]
        test_name = line
        marker = None
        reason = None
        timeout = None
        for tmp_marker in kVALID_TEST_LIST_MARKERS:
            if f" {tmp_marker}" in line:
                test_name, marker, reason_raw = line.partition(f" {tmp_marker}")
                test_name = test_name.strip()
                marker = marker.strip()
                if marker == "TIMEOUT":
                    # Extract timeout value from parentheses
                    timeout = strip_parens(reason_raw.strip())
                    print_info(f"Timeout setting for {test_name}: {timeout}")
                    if not timeout or not timeout.isdigit():
                        raise ValueError(
                            f'{test_list}:{lineno}: Invalid syntax for TIMEOUT value: "{reason_raw}". '
                            "Expected a numeric value in parentheses.")
                    timeout = int(timeout) * 60
                elif len(reason_raw) > 0:
                    reason = strip_parens(reason_raw.strip())
                    if not reason:
                        raise ValueError(
                            ('{}:{}: Invalid syntax for reason: "{}". '
                             "Did you forget to add parentheses?").format(
                                 test_list, lineno, reason_raw))
                break

        if convert_unittest:
            # extract full:XXX/ prefix
            full_prefix = ""
            match = re.match(r'(full:.*?/)(.+)', test_name)
            if match:
                full_prefix = match.group(1)
                test_name = match.group(2)

            # convert unittest to actual test name
            if test_name.startswith("unittest/"):
                test_name = f"test_unittests.py::test_unittests_v2[{test_name}]"

            # combine back
            test_name = full_prefix + test_name

        test_name = parse_test_name(test_name)

        return (test_name, marker, reason, timeout)

    parsed_test_list = map(parse_test_line, enumerate(lines))
    parsed_test_list = list(filter(lambda x: x[0] is not None,
                                   parsed_test_list))
    test_names = [x[0] for x in parsed_test_list]
    test_name_to_marker_dict = {
        x[0]: (x[1], x[2], x[3])
        for x in parsed_test_list
    }

    return (test_names, test_name_to_marker_dict)


def parse_test_list(test_list, test_prefix):
    with open(test_list, "r") as f:
        lines = f.readlines()

    lines = preprocess_test_list_lines(test_list, lines)
    return parse_test_list_lines(test_list, lines, test_prefix)


def split_test_name_into_components(test_name):
    """
    Splits a fully-qualified test name with file name into components.

    Args:
        test_name (str): A test name (with or without parameters).

    Returns:
        Tuple[str, str, Tuple[str]]: A tuple containing:
        - The name of the file containing the test (if found)
        - The name of the test function
        - The parameters of the test as a single string
    """
    params = ()
    test_file, sep, test_basename = test_name.partition("::")

    if not sep:
        test_basename = test_file
        test_file = ""

    test_id_params, _, params = test_basename.partition("[")

    if params:
        params = params.rstrip("]")

    return test_file, test_id_params, params


def join_test_name_components(test_file: str, test_function: str,
                              test_params: str):
    """Performs the inverse of split_test_name_into_components()."""
    name = ""
    if test_file:
        name += f"{test_file}::"
    if test_function:
        name += test_function
    if test_params:
        name += f"[{test_params}]"
    return name


# Global cache for storing previously computed edit distances
_edit_distance_cache: "dict[tuple[str,str], int]" = {}


def edit_distance(s0: str, s1: str):
    """Compute the Levenshtein edit distance (https://en.wikipedia.org/wiki/Levenshtein_distance) between two strings."""

    if s0 == s1:
        return 0

    # Ensure that s0 <= s1, since edit_distance(s0, s1) == edit_distance(s1, s0)
    if s1 < s0:
        s0, s1 = s1, s0

    m, n = len(s0), len(s1)

    if not s1:
        return n
    if not s0:
        return m

    if (s0, s1) not in _edit_distance_cache:
        prev_ed = array.array("l", range(n + 1))
        cur_ed = array.array("l", [0] * (n + 1))

        for i in range(m):
            cur_ed[0] = i + 1
            for j in range(n):
                del_cost = prev_ed[j + 1] + 1
                ins_cost = cur_ed[j] + 1
                sub_cost = prev_ed[j] if s0[i] == s1[j] else prev_ed[j] + 1
                cur_ed[j + 1] = min(del_cost, ins_cost, sub_cost)
            cur_ed, prev_ed = prev_ed, cur_ed

        _edit_distance_cache[(s0, s1)] = prev_ed[n]

    return _edit_distance_cache[(s0, s1)]


def strip_prefix(test_prefix, test_name):
    """Strips the test prefix (as provided by the --test-prefix option) from the given test name, if applicable."""

    if test_prefix is None:
        return test_name

    if test_name.startswith(test_prefix):
        _, _, test_name = test_name.partition(test_prefix)

    # Strip away any leading slashes left over after removing the test prefix
    test_name = test_name.lstrip("/")

    return test_name


def get_test_name_corrections(
        test_names: Sequence[str],
        items: Sequence[Any],
        test_prefix: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Provided for backwards compatibility with tests which rely on this interface.  Use get_test_name_corrections_v2 instead."""

    all_valid_tests = set(
        strip_prefix(test_prefix, test.nodeid) for test in items)
    no_pfx_test_names = set(
        strip_prefix(test_prefix, test_name) for test_name in test_names)

    corr = get_test_name_corrections_v2(no_pfx_test_names, all_valid_tests,
                                        TestCorrectionMode.EXACT_MATCH)

    ret = {}
    for test_name, corrections in corr.items():
        if corrections:
            ret[test_name] = corrections[0]
        else:
            ret[test_name] = None

    return ret


class TestCorrectionMode:
    """Enum describing the different test correction modes supported by get_test_name_corrections_v2."""

    EXACT_MATCH = 0  # Filters specify exact matches
    SUBSTRING = 1  # Filters specify substring matches
    REGEX = 2  # Filters specify regex matches


def get_test_name_corrections_v2(
        test_filters: "set[str]", valid_test_names: "set[str]",
        mode: TestCorrectionMode) -> "dict[str, list[str]]":
    """
    Given a set of user-provided test filter names and set of valid test names, suggests
    corrections for any invalid tests.

    Expects test_filters and valid_test_names to be fully-qualified, e.g.
    `test_foo.py::test_foo[bar-baz]`.

    Args:
        test_filters (set[str]): Test name filters that may or may not be valid.
        valid_test_names (set[str]): Known valid test names.
        mode (TestCorrectionMode): How the test name filters are used to filter tests.  Used to determine which filters are invalid.

    Returns:
        dict[str, list[str]]: A mapping of invalid test names to a list of candidate corrections.
    """

    print_info("Checking for invalid test name filters and corrections")
    corrections_start_time = time.time()

    invalid_filters = set()

    compiled_regex_map = {}

    if mode == TestCorrectionMode.REGEX:
        # Pre-compile the regular expressions to save time.
        compiled_regex_map = {r: re.compile(r) for r in test_filters}

    if mode == TestCorrectionMode.EXACT_MATCH:
        invalid_filters = test_filters - valid_test_names
    elif mode in (TestCorrectionMode.SUBSTRING, TestCorrectionMode.REGEX):
        if mode == TestCorrectionMode.SUBSTRING:
            matches_func = lambda f, t: f in t
        else:
            matches_func = lambda f, t: compiled_regex_map[f].search(
                t) is not None
        invalid_filters = set(test_filters)
        for v in valid_test_names:
            found_filters = set()
            for f in invalid_filters:
                if matches_func(f, v):
                    found_filters.add(f)
            invalid_filters -= found_filters
    else:
        raise NotImplementedError()

    # Bail out of corrections if there are too many invalid filters, as it becomes prohibitively slow to check every invalid filter.
    MAX_INVALID_FILTER_THRESHOLD = 50

    # Maximum number of candidates to check across all filters.  If corrections
    # are computed, the number of candidates allowed per filter is then in the
    # range [MAX_NB_CANDIDATES / MAX_INVALID_FILTER_THRESHOLD, MAX_NB_CANDIDATES].
    #
    # Too many candidates can significantly impact the algorithm's runtime, so keep this number constrained.
    MAX_NB_CANDIDATES = 10000

    nb_invalid_filters = len(invalid_filters)
    nb_candidates_per_filter = (MAX_NB_CANDIDATES //
                                nb_invalid_filters if nb_invalid_filters else 0)

    def build_valid_test_buckets() -> "dict[str,dict[str,set[str]]]":
        """Bucket valid tests by file and test name to save time when looking for matches."""
        valid_test_buckets = defaultdict(lambda: defaultdict(set))
        for valid_test in valid_test_names:
            (
                valid_test_file,
                valid_test_name,
                valid_test_params,
            ) = split_test_name_into_components(valid_test)
            valid_test_buckets[valid_test_file][valid_test_name].add(
                valid_test_params)
        return valid_test_buckets  # pyright: ignore

    def suggest_correction(valid_test_buckets, test):
        """Attempt to find corrections for a given invalid test name.
        Works well for exact matches and probably substrings, but likely won't work well for regex patterns.
        """
        test_file, test_name, test_params = split_test_name_into_components(
            test)

        # Maximum number of corrections to suggest per invalid filter
        MAX_NB_CORRECTIONS = 3

        # Maximum edit distance for a test name for which we consider this a possible match
        MAX_TEST_NAME_EDIT_DISTANCE = 6

        # Maximum edit distance for a test parameter for which we consider this a possible match
        # Be more generous here than for test names, since it's easier to get the parameter naming wrong.
        MAX_TEST_PARAMETER_EDIT_DISTANCE = 12

        candidates: "list[tuple[tuple[str, int], str]]" = []

        # Only consider candidates in the same file as the invalid test filter as it is prohibitively expensive
        # to check every test in every file.
        tests_in_same_file = valid_test_buckets[test_file]
        # First add tests with identical file and test name (if any) to candidates list.
        # Arbitrarily prune the list if there are too many candidates.
        params_with_same_name = tests_in_same_file[test_name]
        candidates.extend(
            ((test_name, 0), p)
            for p in sorted(params_with_same_name)[:nb_candidates_per_filter])
        if len(candidates) < min(nb_candidates_per_filter, MAX_NB_CORRECTIONS):
            # Next, include tests with identical file, but differing test name.
            # Prioritize test names with smaller edit distance.
            # We can skip this step if we already have at least MAX_NB_CORRECTIONS
            # candidates as the only candidates selected will have identical test names (see PHP
            # discussion below).
            ctn = sorted(
                ((c, edit_distance(test_name, c))
                 for c in tests_in_same_file if c != test_name),
                key=lambda x: x[1],
            )
            for cname, cedit_distance in ctn:
                if cedit_distance > MAX_TEST_NAME_EDIT_DISTANCE:
                    continue
                cparams_bucket = tests_in_same_file[cname]
                for cparam in cparams_bucket:
                    if len(candidates) >= nb_candidates_per_filter:
                        break
                    candidates.append(((cname, cedit_distance), cparam))

        # Prune candidates based on the pigeonhole principle.  Let N = MAX_NB_CORRECTIONS.
        # If we already have N or more candidates in the list (sorted by test
        # name edit distance) then any candidate with edit distance worse than
        # the Nth candidate will not be selected and hence can be pruned from
        # the list.
        #
        # For example, given the below list of scores (N=5):
        #
        # [ 2 3 4 4 4 4 4 4 5 5 6 6 ]
        #
        # The 5th candidate has score 4, so we need to consider all candidates with score
        # 4 or below.  No candidate with score 5 or 6 will be selected (based on PHP)
        # and can be ignored.
        if len(candidates) >= MAX_NB_CORRECTIONS:
            threshold_candidate = candidates[MAX_NB_CORRECTIONS - 1]
            threshold_name_edit_distance = threshold_candidate[0][1]
            candidates = [
                c for c in candidates if c[0][1] <= threshold_name_edit_distance
            ]

        # For the remaining candidates, compute parameter edit distance and the final match score.
        possible_matches = []
        for ((cname, cedit_distance)), cparam in candidates:
            param_edit_distance = edit_distance(cparam, test_params)
            if param_edit_distance > MAX_TEST_PARAMETER_EDIT_DISTANCE:
                continue
            match_score = (cedit_distance, param_edit_distance)
            ctest_name = join_test_name_components(test_file, cname, cparam)
            possible_matches.append((ctest_name, match_score))

        return list(m[0] for m in sorted(
            possible_matches, key=lambda m: m[1]))[:MAX_NB_CORRECTIONS]

    print_info(
        f"Computing corrections for {nb_invalid_filters} invalid filters")

    if nb_invalid_filters > MAX_INVALID_FILTER_THRESHOLD:
        print_info(
            f"Bailing out of corrections as there are more than {MAX_INVALID_FILTER_THRESHOLD} invalid filters."
        )
        # Just return the invalid filters, with no suggested corrections.
        ret = {f: [] for f in invalid_filters}
    elif nb_invalid_filters > 0:
        valid_test_buckets = build_valid_test_buckets()
        ret = {
            f: suggest_correction(valid_test_buckets, f)
            for f in invalid_filters
        }
    else:
        ret = {}

    corrections_dt = time.time() - corrections_start_time
    print_info(
        f"Finished checking for corrections in {corrections_dt:.3f} seconds.")
    return ret


def apply_test_list_corrections(test_list,
                                corrections,
                                items,
                                test_prefix=None):
    """
    Attempt to correct invalid test names in a test list.

    Args:
        test_list (str): The path to a test list.
        corrections (Dict[str, str]): A mapping of invalid test names to valid tests, as returned by
                get_test_name_corrections().
        test_prefix (Optional[str]): The value of the --test-prefix option, or None if this option isn't set.
    """
    print_info("Applying corrections to: {}".format(test_list))
    with open(test_list, "r") as f:
        contents = f.read()

    for invalid_test, correction_list in corrections.items():
        if correction_list:
            correction = correction_list[0]
            if test_prefix:
                # Strip the test prefix from the correction and invalid test
                correction = correction[len(f"{test_prefix}/"):]
                invalid_test = invalid_test[len(f"{test_prefix}/"):]
            print_info(f"Correcting {invalid_test} to {correction}")
            contents = contents.replace(invalid_test, correction)
        else:
            print_info(
                "Could not automatically correct: {}".format(invalid_test))

    # We don't want to correct the test list automatically in L0 tests
    #with open(test_list, "w") as f:
    #    f.write(contents)

    # Clear the items list to prevent pytest from listing collected tests
    items.clear()

    raise pytest.UsageError(
        "Exiting early since --apply-test-list-correction was specified.")


def generate_correction_error_message(corrections: Dict[str, List[str]],
                                      prefix: Optional[str]) -> Optional[str]:
    """Returns a string error message reporting any corrections, or None if there are no corrections to report."""
    if not corrections:
        return None
    ret = dedent("""
        !!!!! INVALID TEST NAME FILTERS !!!!!

        Some filter strings do not correspond to any known test, and will be ignored.
        Please correct the test filters to use valid test names.

        Hint: If you encounter this in automation, this is likely due to a bad test name in test-db.
        Please correct the test name in test db configuration yaml file.

        Below are the invalid filter strings, as well as any suggested corrections that identified.

    """)

    for filter in sorted(corrections):
        corr_list = corrections[filter]

        ret += f"- {strip_prefix(prefix, filter)}\n"
        for l in corr_list:
            ret += f"  - correction: {strip_prefix(prefix, l)}\n"

    ret += "\n"

    return ret


def handle_corrections(corrections, test_prefix):
    corr_err_msg = generate_correction_error_message(corrections,
                                                     prefix=test_prefix)
    if corr_err_msg is None:
        return

    for l in corr_err_msg.splitlines():
        print(l)


def record_invalid_tests(output_file, corrections):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        invalid_tests = {"invalid": list(corrections.keys())}
        json.dump(invalid_tests, f)


def parse_and_validate_test_list(
    test_list,
    config,
    items,
    check_for_corrections,
):
    test_prefix = config.getoption("--test-prefix")
    test_names, test_name_to_marker_dict = parse_test_list(
        test_list, test_prefix)

    if check_for_corrections:
        corrections = get_test_name_corrections_v2(
            set(test_names),
            set(it.nodeid for it in items),
            TestCorrectionMode.EXACT_MATCH,
        )

        apply_test_list_correction = config.getoption(
            "--apply-test-list-correction")
        if apply_test_list_correction and corrections:
            apply_test_list_corrections(test_list, corrections, items,
                                        test_prefix)

        output_dir = config.getoption("--output-dir")
        if record_invalid_tests and corrections:
            record_invalid_tests(
                os.path.join(output_dir, "invalid_tests.json"),
                corrections,
            )

        handle_corrections(corrections, test_prefix)

    return test_names, test_name_to_marker_dict


def modify_by_test_list(test_list, items, config):
    """Filter out tests based on the test names specified by the given test_list.  Also
    ensure the test order matches the order specified by the test_list, and add any
    custom markers specified by the test_list."""
    all_test_names = []
    full_test_name_to_marker_dict = {}

    test_names, test_name_to_marker_dict = parse_and_validate_test_list(
        test_list,
        config,
        items,
        check_for_corrections=True,
    )
    all_test_names.extend(test_names)
    full_test_name_to_marker_dict.update(test_name_to_marker_dict)

    found_items = {}
    deselected = []

    # Figure out which items have names specified by the filter
    for item in items:
        if item.nodeid in full_test_name_to_marker_dict:
            found_items[item.nodeid] = item
        else:
            deselected.append(item)

    # Construct a list of tests based on the ordering given in the file
    selected = []
    for name in all_test_names:
        if name in found_items:
            item = found_items[name]
            selected.append(item)
            # Also update the item based on the marker specified in the file
            marker, reason, timeout = full_test_name_to_marker_dict[name]
            if marker:
                if marker == "TIMEOUT" and timeout:
                    item.add_marker(pytest.mark.timeout(timeout))
                else:
                    mark_func = getattr(pytest.mark, marker.lower())
                    mark = mark_func(reason=reason)
                    item.add_marker(mark)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def apply_waives(waives_file, items, config):
    """Apply waives based on the waive state specified by the given waives_file."""

    # Corrections don't make sense for the waives file as it specifies global negative
    # filters that may or may not be applicable to the current platform (i.e., the test names
    # being waived may not be generated on the current platform).
    ret = parse_and_validate_test_list(
        waives_file,
        config,
        items,
        check_for_corrections=False,
    )
    if not ret:
        return
    _, test_name_to_marker_dict = ret

    # For each item in the list, apply waives if a waive entry exists
    for item in items:
        if item.nodeid in test_name_to_marker_dict:
            marker, reason, _ = test_name_to_marker_dict[item.nodeid]
            if marker:
                mark_func = getattr(pytest.mark, marker.lower())
                mark = mark_func(reason=reason)
                item.add_marker(mark)


def uniquify_test_items(items):
    nodeid_set = set()
    duplication_set = set()
    items_unique = []
    for item in items:
        if item.nodeid not in nodeid_set:
            items_unique.append(item)
            nodeid_set.add(item.nodeid)
        else:
            duplication_set.add(item.nodeid)

    if duplication_set:
        print_warning("Test item duplication: " +
                      ",".join(list(duplication_set)))

    items[:] = items_unique
