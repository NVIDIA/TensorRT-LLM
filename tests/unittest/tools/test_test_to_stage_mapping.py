import os
import random
import subprocess
import sys
from collections import defaultdict

import pytest

# Add scripts directory to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
sys.path.insert(0, SCRIPTS_DIR)

from test_to_stage_mapping import StageQuery

GROOVY = os.path.join(REPO_ROOT, 'jenkins', 'L0_Test.groovy')
DB_DIR = os.path.join(REPO_ROOT, 'tests', 'integration', 'test_lists',
                      'test-db')

# Sampling configuration
MAX_SAMPLES = 10  # Small number for efficient testing
MIN_PATTERN_LENGTH = 3  # Minimum length for search patterns


@pytest.fixture(scope="module")
def stage_query():
    """Fixture that provides a StageQuery instance."""
    return StageQuery(GROOVY, DB_DIR)


@pytest.fixture(scope="module")
def sample_test_cases(stage_query):
    """Fixture that provides sample test cases from actual data."""
    random.seed(0)  # Ensure deterministic test results
    all_tests = list(stage_query.test_map.keys())
    if not all_tests:
        raise RuntimeError(
            "No tests found in test mapping. This indicates a configuration "
            "issue - either the test database YAML files are missing/empty "
            "or the StageQuery is not parsing them correctly. Please check "
            "that the test database directory exists and contains valid YAML "
            "files with test definitions.")

    # Return up to MAX_SAMPLES tests randomly selected
    if len(all_tests) <= MAX_SAMPLES:
        return all_tests

    return random.sample(all_tests, MAX_SAMPLES)


@pytest.fixture(scope="module")
def sample_stages(stage_query):
    """Fixture that provides sample stages from actual data."""
    random.seed(0)  # Ensure deterministic test results
    all_stages = list(stage_query.stage_to_yaml.keys())
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

    return random.sample(all_stages, MAX_SAMPLES)


def test_data_availability(stage_query):
    """Test that we have basic data to work with."""
    assert stage_query.stage_to_yaml, "No stages found in Groovy file"
    assert stage_query.test_map, "No tests found in YAML files"

    # Display summary info
    print(f"\nTotal tests available: {len(stage_query.test_map)}")
    print(f"Total stages available: {len(stage_query.stage_to_yaml)}")
    print(f"Max samples configured: {MAX_SAMPLES}")


@pytest.mark.skip(reason="https://nvbugs/5547275")
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
            assert stages, \
                f"Test '{test_case}' should map to at least one stage"

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
def test_cli_functionality(tmp_path, sample_test_cases, file_format):
    """Test CLI functionality with sample data."""
    if not sample_test_cases:
        pytest.skip("No test cases available")

    # Use only first sample for CLI test
    test_file = tmp_path / f'sample_tests.{file_format}'
    if file_format == 'txt':
        test_file.write_text(f'{sample_test_cases[0]}\n')
    else:  # yml
        test_file.write_text(f'- {sample_test_cases[0]}\n')

    script = os.path.join(SCRIPTS_DIR, 'test_to_stage_mapping.py')
    cmd = [sys.executable, script, '--test-list', str(test_file)]
    output = subprocess.check_output(cmd)
    lines = output.decode().strip().splitlines()

    # Should return at least one stage
    assert lines, f"No stages returned for test '{sample_test_cases[0]}'"


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

            # Check that test does NOT map to stages of other backends
            other_backends = all_backends - {backend}
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
