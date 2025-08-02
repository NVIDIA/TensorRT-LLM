#!/usr/bin/env python3
"""
End-to-end tests for assign_reviewers.py script.
Tests various scenarios without requiring GitHub API access or tokens.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import assign_reviewers
sys.path.insert(0, str(Path(__file__).parent.parent))
import assign_reviewers


class TestAssignReviewers(TestCase):
    """Test suite for the assign_reviewers.py script"""

    def setUp(self):
        """Set up test environment"""
        # Set required environment variables
        os.environ["PR_NUMBER"] = "123"
        os.environ["PR_AUTHOR"] = "test-author"
        os.environ["PER_MODULE_REVIEWER_LIMIT"] = "2"

        # Set up test data
        self.module_paths = {
            "cpp/": "Generic Runtime",
            "docs/": "Documentation",
        }

        self.module_owners = {
            "Generic Runtime": ["user1", "user2", "user3"],
            "Documentation": ["user9"],
            "Module1": ["owner1", "owner2"],
            "Module2": ["owner3", "owner4"],
            "Module3": [],  # No owners
        }

    def tearDown(self):
        """Clean up environment variables"""
        for var in ["PR_NUMBER", "PR_AUTHOR", "PER_MODULE_REVIEWER_LIMIT"]:
            if var in os.environ:
                del os.environ[var]

    def _mock_subprocess_run(self, *args, **kwargs):
        """Mock subprocess.run based on the command being executed"""
        cmd = args[0]
        cmd_str = ' '.join(cmd)  # Join command for easier matching

        # Mock response for getting changed files
        if "pr" in cmd and "view" in cmd and "files" in cmd:
            return MagicMock(stdout=self.mock_changed_files,
                             stderr="",
                             returncode=0)

        # Mock response for getting existing reviewers (users)
        elif "pr" in cmd and "view" in cmd and "reviewRequests" in cmd:
            # Check if it's asking for login (users) or name (teams)
            if "select(.login)" in cmd_str:
                return MagicMock(stdout=self.mock_existing_users,
                                 stderr="",
                                 returncode=0)
            elif "select(.name)" in cmd_str:
                return MagicMock(stdout=self.mock_existing_teams,
                                 stderr="",
                                 returncode=0)
            else:
                return MagicMock(stdout="", stderr="", returncode=0)

        # Mock response for assigning reviewers
        elif "pr" in cmd and "edit" in cmd:
            self.assign_reviewers_called = True
            self.assigned_reviewers = [
                cmd[i + 1] for i, arg in enumerate(cmd)
                if arg == "--add-reviewer"
            ]
            return MagicMock(stdout="", stderr="", returncode=0)

        return MagicMock(stdout="", stderr="", returncode=0)

    # ========== Unit Tests for Core Functions ==========

    def test_module_mapping_scenarios(self):
        """Test various module mapping scenarios with parametrized data"""
        test_cases = [
            # Basic mapping with unmapped files
            {
                "name":
                "basic_mapping",
                "files": [
                    "cpp/main.cpp", "cpp/utils.h", "docs/README.md",
                    "unknown/file.txt"
                ],
                "paths": {
                    "cpp/": "Generic Runtime",
                    "docs/": "Documentation"
                },
                "expected_modules": {"Generic Runtime", "Documentation"},
                "expected_unmapped": ["unknown/file.txt"]
            },
            # Most specific module matching
            {
                "name": "most_specific_single",
                "files": ["tensorrt_llm/_torch/models/bert.py"],
                "paths": {
                    "tensorrt_llm/": "LLM API/Workflow",
                    "tensorrt_llm/_torch/": "Torch Framework",
                    "tensorrt_llm/_torch/models/": "Torch Models"
                },
                "expected_modules": {"Torch Models"},
                "expected_unmapped": []
            },
            # Multiple files with overlapping paths
            {
                "name":
                "multiple_overlapping",
                "files": [
                    "tensorrt_llm/config.py", "tensorrt_llm/_torch/base.py",
                    "tensorrt_llm/_torch/models/gpt.py"
                ],
                "paths": {
                    "tensorrt_llm/": "LLM API/Workflow",
                    "tensorrt_llm/_torch/": "Torch Framework",
                    "tensorrt_llm/_torch/models/": "Torch Models"
                },
                "expected_modules":
                {"LLM API/Workflow", "Torch Framework", "Torch Models"},
                "expected_unmapped": []
            },
            # All files unmapped
            {
                "name": "all_unmapped",
                "files": ["unmapped/file1.txt", "another/file2.py"],
                "paths": {
                    "cpp/": "Generic Runtime"
                },
                "expected_modules": set(),
                "expected_unmapped": ["unmapped/file1.txt", "another/file2.py"]
            },
            # Exact file match priority
            {
                "name": "exact_file_match",
                "files": ["tests/integration/test_lists/waives.txt"],
                "paths": {
                    "tests/": "General Tests",
                    "tests/integration/": "Integration Tests",
                    "tests/integration/test_lists/": "Test Configuration",
                    "tests/integration/test_lists/waives.txt": "Test Waive List"
                },
                "expected_modules": {"Test Waive List"},
                "expected_unmapped": []
            }
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                modules, unmapped = assign_reviewers.map_modules(
                    case["files"], case["paths"])
                self.assertEqual(modules, case["expected_modules"],
                                 f"Failed for case: {case['name']}")
                self.assertEqual(set(unmapped), set(case["expected_unmapped"]),
                                 f"Failed for case: {case['name']}")

    def test_gather_reviewers_basic(self):
        """Test basic gather_reviewers functionality"""
        modules = {"Module1", "Module2", "Module3"}

        reviewers, module_assignments, modules_without_owners = assign_reviewers.gather_reviewers(
            modules, self.module_owners, per_module_limit=10)

        # Should get all unique reviewers from modules with owners
        expected = ["owner1", "owner2", "owner3", "owner4"]
        self.assertEqual(set(reviewers), set(expected))

        # Check module assignments
        self.assertEqual(set(module_assignments["Module1"]),
                         {"owner1", "owner2"})
        self.assertEqual(set(module_assignments["Module2"]),
                         {"owner3", "owner4"})
        self.assertEqual(module_assignments["Module3"], [])
        self.assertEqual(modules_without_owners, {"Module3"})

    def test_gather_reviewers_exclusions(self):
        """Test reviewer exclusion functionality"""
        modules = {"Module1", "Module2"}

        # Test PR author exclusion
        reviewers, module_assignments, _ = assign_reviewers.gather_reviewers(
            modules,
            self.module_owners,
            pr_author="owner1",
            per_module_limit=10)

        self.assertNotIn("owner1", reviewers)
        self.assertNotIn("owner1", module_assignments["Module1"])

        # Test existing reviewers exclusion
        existing = {"owner2", "owner3"}
        reviewers, module_assignments, _ = assign_reviewers.gather_reviewers(
            modules,
            self.module_owners,
            existing_reviewers=existing,
            per_module_limit=10)

        self.assertFalse(any(r in existing for r in reviewers))

    def test_per_module_reviewer_limit(self):
        """Test per-module reviewer limit functionality"""
        modules = {"Module1", "Module2"}
        module_owners = {
            "Module1": ["a", "b", "c", "d", "e"],  # 5 owners
            "Module2": ["f", "g", "h"],  # 3 owners
        }

        reviewers, module_assignments, _ = assign_reviewers.gather_reviewers(
            modules, module_owners, per_module_limit=2)

        # Each module should have at most 2 reviewers
        self.assertEqual(len(module_assignments["Module1"]), 2)
        self.assertEqual(len(module_assignments["Module2"]), 2)
        self.assertEqual(len(reviewers), 4)

        # Verify reviewers are from correct modules
        self.assertTrue(
            set(module_assignments["Module1"]).issubset(
                {"a", "b", "c", "d", "e"}))
        self.assertTrue(
            set(module_assignments["Module2"]).issubset({"f", "g", "h"}))

    def test_module_reviewer_overlap(self):
        """Test handling when reviewers own multiple modules"""
        modules = {"Module1", "Module2", "Module3"}
        module_owners = {
            "Module1": ["shared", "owner1"],
            "Module2": ["shared", "owner2"],
            "Module3": ["owner3"],
        }

        # Run multiple times to test randomness
        total_reviewers_counts = []
        for _ in range(10):
            reviewers, _, _ = assign_reviewers.gather_reviewers(
                modules, module_owners, per_module_limit=1)
            total_reviewers_counts.append(len(reviewers))

        # Should see both 2 and 3 reviewers due to random selection of 'shared'
        self.assertTrue(any(count == 2 for count in total_reviewers_counts))
        self.assertTrue(any(count == 3 for count in total_reviewers_counts))

    def test_module_coverage_edge_cases(self):
        """Test edge cases in module coverage"""
        module_owners = {
            "Module1": ["alice", "bob"],
            "Module2": ["bob"],  # Only bob owns this
            "Module3": ["charlie"],
        }

        # Case 1: PR author owns a module entirely
        modules = {"Module1", "Module2", "Module3"}
        reviewers, module_assignments, _ = assign_reviewers.gather_reviewers(
            modules, module_owners, pr_author="bob", per_module_limit=2)

        self.assertEqual(module_assignments["Module2"],
                         [])  # No eligible reviewers
        self.assertEqual(module_assignments["Module1"], ["alice"])
        self.assertEqual(module_assignments["Module3"], ["charlie"])

        # Case 2: All owners already assigned
        existing = {"alice", "charlie"}
        reviewers, module_assignments, _ = assign_reviewers.gather_reviewers(
            {"Module1", "Module3"},
            module_owners,
            pr_author="bob",
            existing_reviewers=existing,
            per_module_limit=2)

        self.assertEqual(len(reviewers), 0)
        self.assertEqual(module_assignments["Module1"], [])
        self.assertEqual(module_assignments["Module3"], [])

    # ========== Integration Tests ==========

    def _run_integration_test(self,
                              changed_files,
                              expected_reviewer_count=None,
                              expected_assigned=True,
                              pr_author=None,
                              existing_users="",
                              extra_assertions=None):
        """Helper method to run integration tests with common setup"""
        self.mock_changed_files = changed_files
        self.mock_existing_users = existing_users
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        if pr_author:
            os.environ["PR_AUTHOR"] = pr_author

        with patch('subprocess.run') as mock_run, \
             patch('assign_reviewers.load_json') as mock_load_json:

            mock_run.side_effect = self._mock_subprocess_run
            mock_load_json.side_effect = lambda path: (
                self.module_paths
                if "module-paths" in str(path) else self.module_owners)

            with patch('sys.argv', ['assign_reviewers.py']):
                assign_reviewers.main()

        self.assertEqual(self.assign_reviewers_called, expected_assigned)

        if expected_reviewer_count is not None and expected_assigned:
            self.assertEqual(len(self.assigned_reviewers),
                             expected_reviewer_count)

        if extra_assertions and expected_assigned:
            extra_assertions(self)

    def test_single_module_changed(self):
        """Test PR with files from a single module"""
        self._run_integration_test(
            changed_files="cpp/file1.cpp\ncpp/file2.h\n",
            expected_reviewer_count=2,
            extra_assertions=lambda self: self.assertTrue(
                all(r in ["user1", "user2", "user3"]
                    for r in self.assigned_reviewers)))

    def test_multiple_modules_changed(self):
        """Test PR with files from multiple modules"""
        self._run_integration_test(
            changed_files="cpp/file1.cpp\ndocs/README.md\n",
            expected_reviewer_count=
            3,  # 2 from Generic Runtime, 1 from Documentation
            extra_assertions=lambda self: self.assertTrue(
                all(r in ["user1", "user2", "user3", "user9"]
                    for r in self.assigned_reviewers)))

    def test_no_files_or_unmapped(self):
        """Test PR with no files or unmapped files"""
        # No files
        self._run_integration_test(changed_files="", expected_assigned=False)

        # Unmapped files
        self._run_integration_test(
            changed_files="unknown/file.txt\nrandom/path.py\n",
            expected_assigned=False)

    def test_pr_author_excluded(self):
        """Test that PR author is excluded from reviewers"""
        self._run_integration_test(
            changed_files="cpp/file1.cpp\n",
            pr_author="user2",
            expected_reviewer_count=2,
            extra_assertions=lambda self: self.assertNotIn(
                "user2", self.assigned_reviewers))

    def test_existing_reviewers_behavior(self):
        """Test behavior with existing reviewers"""
        # Should skip assignment when reviewers exist
        self._run_integration_test(
            changed_files="cpp/file1.cpp\n",
            existing_users="existing_user1\nexisting_user2\n",
            expected_assigned=False)

        # Force assign with existing reviewers
        self.mock_changed_files = "cpp/file1.cpp\n"
        self.mock_existing_users = "user1\n"
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        with patch('subprocess.run') as mock_run, \
             patch('assign_reviewers.load_json') as mock_load_json:

            mock_run.side_effect = self._mock_subprocess_run
            mock_load_json.side_effect = lambda path: (
                self.module_paths
                if "module-paths" in str(path) else self.module_owners)

            with patch('sys.argv', ['assign_reviewers.py', '--force-assign']):
                assign_reviewers.main()

        self.assertTrue(self.assign_reviewers_called)
        self.assertNotIn("user1", self.assigned_reviewers)

    def test_special_modes(self):
        """Test dry-run and error modes"""
        # Dry run mode
        import io
        from contextlib import redirect_stdout

        self.mock_changed_files = "cpp/file1.cpp\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        f = io.StringIO()
        with redirect_stdout(f):
            with patch('subprocess.run') as mock_run, \
                 patch('assign_reviewers.load_json') as mock_load_json:

                mock_run.side_effect = self._mock_subprocess_run
                mock_load_json.side_effect = lambda path: (
                    self.module_paths
                    if "module-paths" in str(path) else self.module_owners)

                with patch('sys.argv', ['assign_reviewers.py', '--dry-run']):
                    assign_reviewers.main()

        output = f.getvalue()
        self.assertIn("DRY RUN:", output)
        self.assertFalse(self.assign_reviewers_called)

    def test_error_handling(self):
        """Test various error handling scenarios"""
        # Subprocess error
        with patch('subprocess.run') as mock_run, \
             patch('assign_reviewers.load_json') as mock_load_json:

            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["gh", "pr", "view"])
            mock_load_json.side_effect = lambda path: self.module_paths

            with self.assertRaises(SystemExit) as cm:
                with patch('sys.argv', ['assign_reviewers.py']):
                    assign_reviewers.main()
            self.assertEqual(cm.exception.code, 1)

        # Missing JSON file
        with patch('assign_reviewers.load_json') as mock_load_json:
            mock_load_json.side_effect = FileNotFoundError(
                "module-paths.json not found")

            with self.assertRaises(FileNotFoundError):
                with patch('sys.argv', ['assign_reviewers.py']):
                    assign_reviewers.main()

        # Missing environment variable
        del os.environ["PR_NUMBER"]
        with self.assertRaises(KeyError):
            with patch('sys.argv', ['assign_reviewers.py']):
                assign_reviewers.main()

    def test_edge_cases_integration(self):
        """Test edge cases in full integration"""
        # Files with special characters
        self._run_integration_test(
            changed_files=
            "cpp/file with spaces.cpp\ncpp/file[brackets].h\ncpp/file@special.cpp\n",
            expected_reviewer_count=2)

        # Large reviewer pool
        large_module_owners = self.module_owners.copy()
        large_module_owners["Large Module"] = [f"user{i}" for i in range(20)]

        self.mock_changed_files = "large/file.cpp\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        with patch('subprocess.run') as mock_run, \
             patch('assign_reviewers.load_json') as mock_load_json:

            mock_run.side_effect = self._mock_subprocess_run
            mock_load_json.side_effect = lambda path: ({
                "large/": "Large Module"
            } if "module-paths" in str(path) else large_module_owners)

            with patch('sys.argv', ['assign_reviewers.py']):
                assign_reviewers.main()

        self.assertTrue(self.assign_reviewers_called)
        self.assertEqual(len(self.assigned_reviewers), 2)  # Per-module limit


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
