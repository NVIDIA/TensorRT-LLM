import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest

__extra_import_path__ = ["~/scripts"]
from test_to_stage_mapping import StageQuery

pytestmark = pytest.mark.cpu_only

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
GROOVY = os.path.join(REPO_ROOT, 'jenkins', 'L0_Test.groovy')
DB_DIR = os.path.join(REPO_ROOT, 'tests', 'integration', 'test_lists',
                      'test-db')

# Sampling configuration
MAX_SAMPLES = 10  # Small number for efficient testing
MIN_PATTERN_LENGTH = 3  # Minimum length for search patterns


def _stage_backed_tests(stage_query: StageQuery) -> list[str]:
    """Return tests from YAML files that are wired to a Jenkins stage."""
    return sorted(test for test, mappings in stage_query.test_map.items()
                  if all(yml in stage_query.yaml_to_stages
                         for yml, _stage, _backend in mappings))


def test_stage_backed_tests_exclude_mixed_mappings() -> None:
    """A test in any unwired YAML is not a live-stage sampling candidate."""
    stage_query = SimpleNamespace(
        test_map={
            'mixed': [('l0_wired.yml', 'pre_merge', 'pytorch'),
                      ('perf.yml', 'post_merge', 'pytorch')],
            'wired': [('l0_wired.yml', 'pre_merge', 'pytorch')],
        },
        yaml_to_stages={'l0_wired.yml': ['L0-PyTorch']},
    )

    assert _stage_backed_tests(stage_query) == ['wired']


@pytest.fixture(scope="module")
def stage_query():
    """Fixture that provides a StageQuery instance."""
    return StageQuery(GROOVY, DB_DIR)


@pytest.fixture(scope="module")
def sample_test_cases(stage_query: StageQuery) -> list[str]:
    """Fixture that samples tests backed by a live Jenkins stage."""
    all_tests = _stage_backed_tests(stage_query)
    if not all_tests:
        raise RuntimeError(
            "No tests are backed by a live Jenkins stage. Check that the "
            "Groovy stage map and test database reference the same YAML files.")

    # Return up to MAX_SAMPLES tests randomly selected
    if len(all_tests) <= MAX_SAMPLES:
        return all_tests

    return random.Random(0).sample(all_tests, MAX_SAMPLES)


@pytest.fixture(scope="module")
def sample_stages(stage_query: StageQuery) -> list[str]:
    """Fixture that provides sample stages from actual data."""
    all_stages = sorted(stage_query.stage_to_yaml)
    if not all_stages:
        raise RuntimeError(
            "No stages found in stage mapping. This indicates a configuration "
            "issue - either the Jenkins L0_Test.groovy file is not being "
            "parsed correctly or the regex pattern for stage matching needs "
            "to be updated. Please check that the groovy file exists and "
            "contains stage definitions in the expected format.")

    # Return up to MAX_SAMPLES stages randomly selected
    if len(all_stages) <= MAX_SAMPLES:
        return all_stages

    return random.Random(0).sample(all_stages, MAX_SAMPLES)


def test_data_availability(stage_query):
    """Test that we have basic data to work with."""
    assert stage_query.stage_to_yaml, "No stages found in Groovy file"
    assert stage_query.test_map, "No tests found in YAML files"

    # Display summary info
    print(f"\nTotal tests available: {len(stage_query.test_map)}")
    print(f"Total stages available: {len(stage_query.stage_to_yaml)}")
    print(f"Max samples configured: {MAX_SAMPLES}")


def test_all_stage_backed_tests_map(stage_query: StageQuery) -> None:
    """Every test in a Jenkins-wired YAML must resolve to a live stage."""
    stage_backed_tests = _stage_backed_tests(stage_query)
    assert stage_backed_tests, "No tests are backed by a live Jenkins stage"

    unmapped = [
        test for test in stage_backed_tests
        if not stage_query.tests_to_stages([test])
    ]
    assert not unmapped, \
        f"Stage-backed tests should map to at least one stage: {unmapped}"


def test_documented_stage_examples_are_live(stage_query):
    """Documented --stages examples must name stages that still exist in CI."""
    sources = [
        os.path.join(REPO_ROOT, 'docs', 'source', 'developer-guide',
                     'ci-overview.md'),
        os.path.join(SCRIPTS_DIR, 'test_to_stage_mapping.py'),
    ]

    checked = 0
    for path in sources:
        with open(path, 'r') as f:
            text = f.read()

        # Unwrap backslash continuations so wrapped commands read as one line,
        # then take the stage names each documented invocation passes.
        unwrapped = re.sub(r'\\+\s*\n\s*', ' ', text)
        for example in re.findall(r'test_to_stage_mapping\.py.*?--stages(.*)',
                                  unwrapped):
            # Brackets are matched, not skipped, so a stale ``[Post-Merge]``
            # spelling fails here instead of slipping through as two tokens.
            for name in re.findall(r'[\w\[\].-]+', example):
                if name.startswith('-'):
                    break  # start of the next option, not a stage name
                why = (f'{os.path.basename(path)} documents --stages {name}, '
                       'which ')
                assert name in stage_query.stage_to_yaml, \
                    why + f'is not a stage in {os.path.basename(GROOVY)}'
                assert stage_query.stages_to_tests([name]), \
                    why + 'maps to no tests'
                checked += 1

    assert checked, 'Found no documented --stages examples to validate'


def test_unknown_stage_reports_a_diagnostic(stage_query):
    """An unresolvable stage name must not be silently swallowed."""
    bogus = 'A100X-Triton-Post-Merge-1'
    assert bogus not in stage_query.stage_to_yaml
    assert 'A100X-PyTorch-Post-Merge-1' in stage_query.suggest_stages(bogus), \
        f"Expected the live A100X stage to be suggested for '{bogus}'"

    script = os.path.join(SCRIPTS_DIR, 'test_to_stage_mapping.py')
    proc = subprocess.run([sys.executable, script, '--stages', bogus],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    assert not proc.stdout.strip(), 'Unknown stage should map to no tests'
    assert f'unknown stage: {bogus}' in proc.stderr.decode()


def test_known_stage_without_tests_is_reported(tmp_path):
    """A known stage that runs no tests stays visible alongside other stages."""
    # No live stage is currently empty, so build a minimal repo whose
    # ``empty`` stage maps to a YAML holding only post_merge tests.
    (tmp_path / 'jenkins').mkdir()
    (tmp_path / 'jenkins' / 'L0_Test.groovy').write_text(
        '"Filled-PyTorch-1": ["x", "l0_filled", 1, 1],\n'
        '"Empty-PyTorch-1": ["x", "l0_empty", 1, 1],\n')
    db_dir = tmp_path / 'tests' / 'integration' / 'test_lists' / 'test-db'
    db_dir.mkdir(parents=True)
    for name, stage in (('l0_filled', 'pre_merge'), ('l0_empty', 'post_merge')):
        (db_dir / f'{name}.yml').write_text(
            f'version: 0.0.1\n{name}:\n- condition:\n    terms:\n'
            f'      stage: {stage}\n      backend: pytorch\n'
            f'  tests:\n  - unittest/{name}.py\n')

    script = os.path.join(SCRIPTS_DIR, 'test_to_stage_mapping.py')
    proc = subprocess.run([
        sys.executable, script, '--repo-root',
        str(tmp_path), '--stages', 'Empty-PyTorch-1', 'Filled-PyTorch-1'
    ],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    assert proc.stdout.decode().split() == ['unittest/l0_filled.py']
    assert 'no tests mapped to: Empty-PyTorch-1' in proc.stderr.decode()


@pytest.mark.parametrize("direction",
                         ["test_to_stage", "stage_to_test", "roundtrip"])
def test_bidirectional_mapping_consistency(stage_query, sample_test_cases,
                                           sample_stages, direction):
    """Test mapping consistency in both directions with roundtrip validation."""
    if direction == "test_to_stage":
        if not sample_test_cases:
            pytest.skip("No test cases available")

        for test_case in sample_test_cases:
            stages = stage_query.tests_to_stages([test_case])

            # Verify all returned stages are valid
            for stage in stages:
                assert stage in stage_query.stage_to_yaml, \
                    f"Invalid stage '{stage}' for test '{test_case}'"

            # Check mapping consistency: stage references should be valid
            mappings = stage_query.test_map[test_case]
            for yaml_file, stage_type, backend in mappings:
                assert yaml_file in stage_query.yaml_to_stages, \
                    f"Test {test_case} references invalid YAML {yaml_file}"

    elif direction == "stage_to_test":
        if not sample_stages:
            pytest.skip("No stages available")

        for stage in sample_stages:
            tests = stage_query.stages_to_tests([stage])
            # Verify returned tests are valid
            for test in tests:
                assert test in stage_query.test_map, \
                    f"Invalid test '{test}' for stage '{stage}'"

            # Check YAML consistency
            yaml_file = stage_query.stage_to_yaml[stage]
            assert yaml_file in stage_query.yaml_to_stages, \
                f"Stage {stage} references YAML {yaml_file} that doesn't exist"

    elif direction == "roundtrip":
        if not sample_test_cases:
            pytest.skip("No test cases available")

        for test_case in sample_test_cases:
            # Map test to stages
            stages = stage_query.tests_to_stages([test_case])
            if not stages:
                continue  # Skip tests that don't map to stages

            # Map stages back to tests
            back_mapped_tests = stage_query.stages_to_tests(stages)
            assert test_case in back_mapped_tests, \
                f"Roundtrip failed for '{test_case}'"


def test_search_functionality(stage_query, sample_test_cases):
    """Test search functionality using sample test cases."""
    if not sample_test_cases:
        pytest.skip("No test cases available")

    # Test with first sample only to keep it efficient
    test_case = sample_test_cases[0]

    # Extract search pattern from test name
    if '::' in test_case:
        # Use function name as search pattern
        pattern = test_case.split('::')[-1].split('[')[0]
    else:
        # Use file name as search pattern
        pattern = test_case.split('/')[-1].split('.')[0]

    if len(pattern) < MIN_PATTERN_LENGTH:
        pytest.skip(f"Pattern '{pattern}' too short")

    found_tests = stage_query.search_tests(pattern)
    assert test_case in found_tests, \
        f"Search for '{pattern}' should find '{test_case}'"


@pytest.mark.parametrize('file_format', ['txt', 'yml'])
def test_cli_functionality(tmp_path, stage_query, sample_test_cases,
                           file_format):
    """Test CLI functionality with sample data."""
    # Use the first sample that maps to at least one stage (some test-db
    # files, e.g. multi-node perf-sanity lists, have no L0 stage).
    test_case = next(
        (t for t in sample_test_cases if stage_query.tests_to_stages([t])),
        None)
    if test_case is None:
        pytest.skip("No sampled test maps to any stage")

    test_file = tmp_path / f'sample_tests.{file_format}'
    if file_format == 'txt':
        test_file.write_text(f'{test_case}\n')
    else:  # yml
        test_file.write_text(f'- {test_case}\n')

    script = os.path.join(SCRIPTS_DIR, 'test_to_stage_mapping.py')
    cmd = [sys.executable, script, '--test-list', str(test_file)]
    output = subprocess.check_output(cmd)
    lines = output.decode().strip().splitlines()

    # Should return at least one stage
    assert lines, f"No stages returned for test '{test_case}'"


def test_backend_filtering_consistency(stage_query):
    """Test that tests only map to stages matching their backend."""
    # Discover all backends and collect sample tests for each
    backend_to_tests = defaultdict(list)
    all_backends = set()

    for test_name, mappings in stage_query.test_map.items():
        for yml, stage_type, backend in mappings:
            if backend and backend.strip():  # Only consider non-empty backends
                backend_clean = backend.strip()
                all_backends.add(backend_clean)
                backend_to_tests[backend_clean].append(test_name)

    # Test each backend (limit samples for efficiency)
    for backend in sorted(all_backends):
        if not backend_to_tests[backend]:
            continue

        # Get sample tests for this backend (up to MAX_SAMPLES)
        sample_tests = backend_to_tests[backend][:MAX_SAMPLES]

        print(f"\nTesting backend '{backend}' with "
              f"{len(sample_tests)} sample tests")

        for test_name in sample_tests:
            stages = stage_query.tests_to_stages([test_name])

            if not stages:
                continue  # Skip tests that don't map to any stages

            # Check that test maps to at least one stage matching its backend
            found_matching_stage = False
            for stage in stages:
                # Check if stage name contains the backend identifier
                if backend.upper() in stage.upper():
                    found_matching_stage = True
                    break

            assert found_matching_stage, \
                f"Test '{test_name}' with backend '{backend}' should map to " \
                f"at least one stage containing '{backend.upper()}', " \
                f"but got stages: {stages}"

            # Check that test does NOT map to stages of backends it is not
            # declared under (tests may legitimately be listed under several
            # backends across test-db files).
            declared_backends = {
                b.strip()
                for _, _, b in stage_query.test_map[test_name]
                if b and b.strip()
            }
            other_backends = all_backends - declared_backends
            for stage in stages:
                stage_upper = stage.upper()
                for other_backend in other_backends:
                    other_upper = other_backend.upper()
                    if (other_upper in stage_upper
                            and backend.upper() not in stage_upper):
                        assert False, \
                            f"Test '{test_name}' with backend '{backend}' " \
                            f"incorrectly maps to '{other_backend}' " \
                            f"stage '{stage}'"

    # Test stage-to-tests mapping consistency
    for stage_name in list(stage_query.stage_to_yaml.keys())[:MAX_SAMPLES]:
        tests = stage_query.stages_to_tests([stage_name])

        # a stage should have at least one test
        assert tests, f"Stage '{stage_name}' has no tests"

        # Determine expected backend(s) from stage name
        stage_upper = stage_name.upper()
        expected_backends = set()
        for backend in all_backends:
            if backend.upper() in stage_upper:
                expected_backends.add(backend)

        assert expected_backends, \
            f"Stage '{stage_name}' must indicate a backend"

        # Sample a few tests from this stage
        sample_stage_tests = tests[:MAX_SAMPLES]

        for test_name in sample_stage_tests:
            assert test_name in stage_query.test_map, \
                f"Test '{test_name}' not found in test_map"

            # Get backends for this test
            test_backends = set()
            for yml, stage_type, backend in stage_query.test_map[test_name]:
                if backend and backend.strip():
                    test_backends.add(backend.strip())

            # If test has explicit backends, they should match stage backends
            if test_backends:
                common_backends = test_backends & expected_backends
                assert common_backends or not test_backends, \
                    f"Stage '{stage_name}' expects backends " \
                    f"{expected_backends} but contains test '{test_name}' " \
                    f"with backends {test_backends}"

    print(f"\nBackend filtering test completed for {len(all_backends)} "
          f"backends: {sorted(all_backends)}")
