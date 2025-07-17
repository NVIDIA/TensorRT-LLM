"""Lookup Jenkins stage names for integration tests and vice versa.

This helper parses ``jenkins/L0_Test.groovy`` and the YAML files under
``tests/integration/test_lists/test-db`` to provide a bidirectional mapping
between test names and Jenkins stage names. When ``--tests`` or ``--test-list``
options are used, each value is treated as a substring pattern. Any test whose
fully qualified name contains the pattern will be matched. If the pattern
corresponds exactly to a test name, it naturally matches that test as well.

Example usage::

   python scripts/test_to_stage_mapping.py --tests \\
       "triton_server/test_triton.py::test_gpt_ib_ptuning[gpt-ib-ptuning]"
   python scripts/test_to_stage_mapping.py --tests gpt_ib_ptuning
   python scripts/test_to_stage_mapping.py --stages \\
       A100X-Triton-Post-Merge-1

Tests can also be provided via ``--test-list`` pointing to either a plain text
file or a YAML list file. Quote individual test names on the command line so
the shell does not interpret ``[`` and ``]`` characters.
"""

import argparse
import os
import re
from collections import defaultdict
from glob import glob
from typing import List

import yaml


def _load_tests_file(path: str) -> List[str]:
    tests: List[str] = []
    yaml_mode = path.endswith('.yml') or path.endswith('.yaml')
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if yaml_mode:
                if line.startswith('- '):
                    tests.append(line[2:].strip())
            else:
                tests.append(line)
    return tests


# Regex to parse Jenkins stage configurations from Groovy files
# Matches patterns like: "Stage-Name": ["platform", "yaml_file", split_id, split_count, gpu_count]
#
# Pattern breakdown:
#   "(?P<stage>[^"]+)"     - Captures stage name in quotes (group 'stage')
#   \s*:\s*               - Matches colon with optional whitespace
#   \[                    - Matches opening bracket
#   "[^"]+"              - Matches platform string in quotes (ignored)
#   ,\s*                 - Matches comma with optional whitespace
#   "(?P<yml>[^"]+)"     - Captures yaml filename in quotes (group 'yml')
#   (?:,\s*\d+)*         - Matches zero or more comma-separated numbers (split_id, split_count, gpu_count)
#   \s*\]                - Matches closing bracket with optional whitespace
_STAGE_RE = re.compile(
    r'"(?P<stage>[^"]+)"\s*:\s*\["[^"]+",\s*"(?P<yml>[^"]+)"(?:,\s*\d+)*\s*\]')


def _extract_terms(entry):
    """Extract terms from either direct 'terms' or 'condition.terms'."""
    terms = entry.get('terms', {})
    if not terms:
        terms = entry.get('condition', {}).get('terms', {})
    return terms


class StageQuery:

    def __init__(self, groovy_path: str, test_db_dir: str):
        self.stage_to_yaml, self.yaml_to_stages = self._parse_stage_mapping(
            groovy_path)
        self.test_map, self.yaml_stage_tests = self._parse_tests(test_db_dir)
        # Build dynamic backend mapping from discovered data
        self._backend_keywords = self._discover_backend_keywords()

    @staticmethod
    def _parse_stage_mapping(path):
        stage_to_yaml = {}
        yaml_to_stages = defaultdict(list)
        with open(path, 'r') as f:
            for line in f:
                m = _STAGE_RE.search(line)
                if m:
                    stage = m.group('stage')
                    yml = m.group('yml') + '.yml'
                    stage_to_yaml[stage] = yml
                    yaml_to_stages[yml].append(stage)
        return stage_to_yaml, yaml_to_stages

    def _parse_tests(self, db_dir):
        """Parse tests from YAML files, supporting both .yml and .yaml."""
        test_map = defaultdict(list)
        yaml_stage_tests = defaultdict(lambda: defaultdict(list))

        yaml_files = (glob(os.path.join(db_dir, '*.yml')) +
                      glob(os.path.join(db_dir, '*.yaml')))

        for path in yaml_files:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            for key, entries in data.items():
                if key == 'version' or entries is None:
                    continue
                for entry in entries:
                    terms = _extract_terms(entry)

                    stage = terms.get('stage')
                    if stage is None:
                        continue

                    backend = terms.get('backend', '')  # Default to empty

                    tests = entry.get('tests', [])
                    yml = os.path.basename(path)
                    for t in tests:
                        test_map[t].append((yml, stage, backend))
                        yaml_stage_tests[yml][stage].append(t)
        return test_map, yaml_stage_tests

    def _discover_backend_keywords(self):
        """Discover backend keywords from existing data dynamically."""
        backend_keywords = {}

        # Collect all backends from test data
        all_backends = set()
        for mappings in self.test_map.values():
            for yml, stage_type, backend in mappings:
                if backend and backend.strip():
                    all_backends.add(backend.strip().lower())

        # Map backends to their likely stage name keywords
        for backend in all_backends:
            backend_keywords[backend] = backend.upper()

        # Add common variations/aliases
        aliases = {
            'tensorrt': ['TENSORRT', 'TRT'],
            'pytorch': ['PYTORCH', 'TORCH'],
            'cpp': ['CPP', 'C++'],
            'triton': ['TRITON']
        }

        for backend, keywords in aliases.items():
            if backend in backend_keywords:
                backend_keywords[backend] = keywords

        return backend_keywords

    def search_tests(self, pattern: str):
        parts = pattern.split()
        result = []
        for test in self.test_map:
            name = test.lower()
            if all(p.lower() in name for p in parts):
                result.append(test)
        return result

    def tests_to_stages(self, tests):
        result = set()
        for t in tests:
            for yml, stage_type, backend in self.test_map.get(t, []):
                for s in self.yaml_to_stages.get(yml, []):
                    if stage_type == 'post_merge' and 'Post-Merge' not in s:
                        continue
                    if stage_type == 'pre_merge' and 'Post-Merge' in s:
                        continue

                    # Filter by backend if specified
                    if backend and backend != '':
                        backend_keywords = self._backend_keywords.get(
                            backend.lower(), [backend.upper()])
                        if isinstance(backend_keywords, str):
                            backend_keywords = [backend_keywords]

                        if not any(keyword in s.upper()
                                   for keyword in backend_keywords):
                            continue

                    result.add(s)
        return sorted(result)

    def stages_to_tests(self, stages):
        result = set()
        for s in stages:
            yml = self.stage_to_yaml.get(s)
            if not yml:
                continue
            stage_type = 'post_merge' if 'Post-Merge' in s else 'pre_merge'

            # Determine expected backend dynamically from stage name
            expected_backend = None
            stage_upper = s.upper()
            for backend, keywords in self._backend_keywords.items():
                if isinstance(keywords, str):
                    keywords = [keywords]
                if any(keyword in stage_upper for keyword in keywords):
                    expected_backend = backend
                    break

            # Get all tests for yml/stage_type, then filter by backend
            all_tests = self.yaml_stage_tests.get(yml, {}).get(stage_type, [])
            for test in all_tests:
                # Check if test's backend matches stage's expected backend
                test_mappings = self.test_map.get(test, [])
                for test_yml, test_stage, test_backend in test_mappings:
                    if (test_yml == yml and test_stage == stage_type
                            and (expected_backend is None
                                 or test_backend == expected_backend)):
                        result.add(test)
                        break
        return sorted(result)


def main():
    parser = argparse.ArgumentParser(
        description='Map Jenkins stages to tests and vice versa.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--tests',
        nargs='+',
        help='One or more test name patterns to resolve to Jenkins stages')
    group.add_argument(
        '--test-list',
        help=('File with test name patterns, either newline separated '
              'or a YAML list'))
    group.add_argument('--stages',
                       nargs='+',
                       help='List of stage names to look up')
    parser.add_argument('--repo-root',
                        default=os.path.dirname(os.path.dirname(__file__)),
                        help='Path to repository root')
    args = parser.parse_args()

    groovy = os.path.join(args.repo_root, 'jenkins', 'L0_Test.groovy')
    db_dir = os.path.join(args.repo_root, 'tests', 'integration', 'test_lists',
                          'test-db')
    query = StageQuery(groovy, db_dir)

    if args.tests or args.test_list:
        patterns = []
        if args.tests:
            patterns.extend(args.tests)
        if args.test_list:
            patterns.extend(_load_tests_file(args.test_list))

        collected = []
        for pat in patterns:
            collected.extend(query.search_tests(pat))
        tests = sorted(set(collected))
        stages = query.tests_to_stages(tests)
        for s in stages:
            print(s)
    else:
        tests = query.stages_to_tests(args.stages)
        for t in tests:
            print(t)


if __name__ == '__main__':
    main()
