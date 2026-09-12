"""Lookup Jenkins stage names for integration tests and vice versa.

This helper reads ``jenkins/scripts/test_stage_configs.json`` -- the single
source of truth for sharded L0 test stage families, also consumed by
``jenkins/L0_Test.groovy`` (via ``loadStageConfigSpecs`` /
``buildStageConfigsFromSpecs``) -- and the YAML files under
``tests/integration/test_lists/test-db`` to provide a bidirectional mapping
between test names and Jenkins stage names. When ``--tests`` or ``--test-list``
options are used, each value is treated as a substring pattern. Any test whose
fully qualified name contains the pattern will be matched. If the pattern
corresponds exactly to a test name, it naturally matches that test as well.

Example usage::

   python scripts/test_to_stage_mapping.py --tests \\
       "unittest/_torch/sampler/test_beam_search.py"
   python scripts/test_to_stage_mapping.py --tests test_beam_search
   python scripts/test_to_stage_mapping.py --stages \\
       A100X-PyTorch-Post-Merge-1

Stage names passed to ``--stages`` must exactly match a stage name generated
from ``jenkins/scripts/test_stage_configs.json``. Unrecognized names, and
known names that map to no tests, are reported on stderr.

For every stage family, the number of generated shards ("<name>-<k>") is
sized from ``tests/integration/defs/.test_durations`` (~2h/shard) instead of
the JSON's hardcoded ``splits`` value whenever durations are available; see
``_dynamic_split_counts``. Sizing reuses ``block_matches_stage`` /
``derive_mako_from_stage`` from ``jenkins/scripts/cbts/blocks.py`` to
determine each family's actual (mako-filtered) test set -- the same matching
``trt-test-db`` performs at runtime via
``jenkins/L0_Test.groovy::getMakoArgsFromStageName`` -- so families that
share a ``testDB`` YAML with siblings (and are told apart only by stage-name
keyword filtering, e.g. ``DGX_B200-CPP`` vs. ``DGX_B200-PyTorch``) are still
sized correctly instead of being skipped.

``jenkins/L0_Test.groovy``'s ``computeDynamicSplitCounts`` shells out to this
script (via ``--emit-splits``) to size the Jenkins stage maps the same way,
rather than reimplementing this matching logic a second time in Groovy.

Tests can also be provided via ``--test-list`` pointing to either a plain text
file or a YAML list file. Quote individual test names on the command line so
the shell does not interpret ``[`` and ``]`` characters.
"""

import argparse
import difflib
import json
import math
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import List

import yaml

# jenkins/scripts/cbts/blocks.py is a sibling script tree, not an installed
# package -- add it to sys.path the same way jenkins/scripts/cbts/main.py
# does for its own sibling imports.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent / 'jenkins' / 'scripts' /
        'cbts'))
from blocks import block_matches_stage  # noqa: E402
from blocks import Stage, YAMLIndex, derive_mako_from_stage

# Target wall-time per pytest-split shard, and a safety ceiling against a
# corrupted/stale .test_durations blowing up shard counts.
_TARGET_SHARD_SECONDS = 2 * 60 * 60
_DYNAMIC_SPLITS_CAP = 20


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


def _extract_terms(entry):
    """Extract terms from either direct 'terms' or 'condition.terms'."""
    terms = entry.get('terms', {})
    if not terms:
        terms = entry.get('condition', {}).get('terms', {})
    return terms


def _phase_of(spec_name: str) -> str:
    return 'post_merge' if 'Post-Merge' in spec_name else 'pre_merge'


def _load_durations(durations_path):
    """Load pytest-split `.test_durations` ({node_id: seconds}); {} if absent/unparsable."""
    try:
        with open(durations_path, 'r') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, (int, float))}


def _dynamic_split_counts(specs, yaml_index, durations):
    """Sizes every stage family from its actual (mako-filtered) test set.

    For each spec, a synthetic '<name>-1' shard name is fed to
    derive_mako_from_stage() (jenkins/scripts/cbts/blocks.py) -- the '-1'
    suffix is needed only so trailing-dash keyword patterns like '-PyTorch-'
    or '-Ray-' can match; every shard of a family shares the same filtering
    regardless of shard index. block_matches_stage() then selects, from the
    family's testDB YAML, only the blocks whose condition the derived mako
    actually satisfies -- the same matching trt-test-db performs at runtime
    via jenkins/L0_Test.groovy::getMakoArgsFromStageName. This correctly
    sizes families that share a testDB with siblings and are told apart only
    by stage-name keyword filtering (e.g. DGX_B200-CPP vs. DGX_B200-PyTorch),
    which a naive "sum every block tagged with this phase" estimate would
    over-count for every sibling. No-op (returns {}) when durations are
    absent.
    """
    if not durations:
        return {}
    avg = sum(durations.values()) / len(durations)
    blocks_by_testdb = defaultdict(list)
    for block in yaml_index.blocks:
        blocks_by_testdb[block.yaml_stem].append(block)

    overrides = {}
    for spec in specs:
        mako = derive_mako_from_stage(f"{spec['name']}-1")
        stage = Stage(name=spec['name'],
                      yaml_stem=spec['testDB'],
                      cpu_arch='',
                      split_id=1,
                      total_splits=spec['splits'],
                      mako=mako)
        tests = set()
        for block in blocks_by_testdb.get(spec['testDB'], []):
            if block_matches_stage(block, stage):
                tests.update(block.tests)
        if not tests:
            continue
        seconds = sum(durations.get(t, avg) for t in tests)
        k = max(
            1,
            min(math.ceil(seconds / _TARGET_SHARD_SECONDS),
                _DYNAMIC_SPLITS_CAP))
        overrides[spec['name']] = k
    return overrides


class StageQuery:

    def __init__(self,
                 stage_config_path: str,
                 test_db_dir: str,
                 durations_path: str = None):
        self.test_map, self.yaml_stage_tests = self._parse_tests(test_db_dir)
        yaml_index = YAMLIndex.load(Path(test_db_dir))
        durations = _load_durations(durations_path) if durations_path else {}
        self.stage_to_yaml, self.yaml_to_stages = self._parse_stage_mapping(
            stage_config_path, yaml_index, durations)
        # Build dynamic backend mapping from discovered data
        self._backend_keywords = self._discover_backend_keywords()

    @staticmethod
    def _parse_stage_mapping(stage_config_path, yaml_index, durations):
        """Expands jenkins/scripts/test_stage_configs.json.

        the single source of truth for sharded L0 test stage families, also
        consumed by jenkins/L0_Test.groovy (via loadStageConfigSpecs /
        buildStageConfigsFromSpecs) -- into stage <-> YAML mappings.
        """
        stage_to_yaml = {}
        yaml_to_stages = defaultdict(list)
        for stage, yml in StageQuery._expand_json_specs(stage_config_path,
                                                        yaml_index, durations):
            stage_to_yaml[stage] = yml
            yaml_to_stages[yml].append(stage)
        return stage_to_yaml, yaml_to_stages

    @staticmethod
    def _expand_json_specs(json_path, yaml_index, durations):
        """Mirrors buildStageConfigsFromSpecs in jenkins/L0_Test.groovy.

        for each stage family spec, yields (stageName-k, yamlFile.yml) for k
        in 1..splits, where splits is dynamically sized from `.test_durations`
        when eligible (see _dynamic_split_counts), else the JSON's hardcoded
        fallback value.
        """
        with open(json_path, 'r') as f:
            specs = json.load(f)['configs']
        overrides = _dynamic_split_counts(specs, yaml_index, durations)
        for spec in specs:
            yml = spec['testDB'] + '.yml'
            splits = overrides.get(spec['name'], spec['splits'])
            for k in range(1, splits + 1):
                yield f"{spec['name']}-{k}", yml

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

    def suggest_stages(self, stage):
        """Return live stage names spelled similarly to an unknown one."""
        return difflib.get_close_matches(stage, self.stage_to_yaml)

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
    group.add_argument(
        '--emit-splits',
        action='store_true',
        help=('Print dynamically-sized shard counts for every family in '
              'jenkins/scripts/test_stage_configs.json as JSON '
              '({name: k}, only families with an override) on stdout. '
              'Used by jenkins/L0_Test.groovy::computeDynamicSplitCounts so '
              'the split-sizing logic (and its blocks.py dependency) lives '
              'in one place instead of a second Groovy port. Diagnostic '
              'lines go to stderr. Prints "{}" when .test_durations is '
              'absent/unparsable -- callers should keep their hardcoded '
              "splits in that case."))
    parser.add_argument('--repo-root',
                        default=os.path.dirname(os.path.dirname(__file__)),
                        help='Path to repository root')
    args = parser.parse_args()

    stage_config = os.path.join(args.repo_root, 'jenkins', 'scripts',
                                'test_stage_configs.json')
    db_dir = os.path.join(args.repo_root, 'tests', 'integration', 'test_lists',
                          'test-db')
    durations_path = os.path.join(args.repo_root, 'tests', 'integration',
                                  'defs', '.test_durations')

    if args.emit_splits:
        with open(stage_config, 'r') as f:
            specs = json.load(f)['configs']
        yaml_index = YAMLIndex.load(Path(db_dir))
        durations = _load_durations(durations_path)
        overrides = _dynamic_split_counts(specs, yaml_index, durations)
        for spec in specs:
            k = overrides.get(spec['name'])
            if k is not None and k != spec['splits']:
                sys.stderr.write(
                    f"computeDynamicSplitCounts: {spec['name']} -> {k} "
                    f"shard(s) (was {spec['splits']})\n")
        print(json.dumps(overrides))
        return

    query = StageQuery(stage_config, db_dir, durations_path)

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
        unknown = [s for s in args.stages if s not in query.stage_to_yaml]
        for s in unknown:
            sys.stderr.write(f'unknown stage: {s}\n')
            for hint in query.suggest_stages(s):
                sys.stderr.write(f'  did you mean: {hint}\n')
        # Report each known-but-empty stage on its own, so it stays visible when
        # another requested stage does contribute tests, instead of printing
        # nothing at all for it.
        empty = [
            s for s in args.stages
            if s not in unknown and not query.stages_to_tests([s])
        ]
        tests = query.stages_to_tests(args.stages)
        for t in tests:
            print(t)
        if empty:
            sys.stderr.write(f'no tests mapped to: {", ".join(empty)}\n')


if __name__ == '__main__':
    main()
