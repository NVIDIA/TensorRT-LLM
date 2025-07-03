#!/usr/bin/env python3
"""
End-to-end tests for assign_reviewers.py script.
Tests various scenarios without requiring GitHub API access or tokens.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path to import the script
sys.path.insert(0, str(Path(__file__).parent.parent))
import assign_reviewers


class TestAssignReviewers(unittest.TestCase):
    """Test suite for the assign_reviewers.py script"""

    def setUp(self):
        """Set up test fixtures"""
        # Sample module-paths.json data
        self.module_paths = {
            "cpp/": "Generic Runtime",
            "tensorrt_llm/": "LLM API/Workflow",
            "benchmarks/": "Performance",
            "docs/": "Documentation",
            "tensorrt_llm/_torch/": "Torch Framework"
        }

        # Sample module-owners.json data
        self.module_owners = {
            "Generic Runtime": ["user1", "user2", "user3"],
            "LLM API/Workflow": ["user4", "user5"],
            "Performance": ["user6", "user7", "user8"],
            "Documentation": ["user9"],
            "Torch Framework": ["user10", "user11"]
        }

        # Set required environment variables
        os.environ["PR_NUMBER"] = "123"
        os.environ["PR_AUTHOR"] = "test_author"
        os.environ["REVIEWER_LIMIT"] = "3"

    def tearDown(self):
        """Clean up environment variables"""
        for var in ["PR_NUMBER", "PR_AUTHOR", "REVIEWER_LIMIT"]:
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

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_single_module_changed(self, mock_run, mock_load_json):
        """Test PR with files from a single module"""
        # Setup mocks
        self.mock_changed_files = "cpp/file1.cpp\ncpp/file2.h\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify reviewers were assigned
        self.assertTrue(self.assign_reviewers_called)
        self.assertEqual(len(self.assigned_reviewers),
                         3)  # Should respect limit
        self.assertTrue(
            all(r in ["user1", "user2", "user3"]
                for r in self.assigned_reviewers))

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_multiple_modules_changed(self, mock_run, mock_load_json):
        """Test PR with files from multiple modules"""
        # Setup mocks
        self.mock_changed_files = "cpp/file1.cpp\ndocs/README.md\nbenchmarks/test.py\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify reviewers were assigned from multiple modules
        self.assertTrue(self.assign_reviewers_called)
        self.assertEqual(len(self.assigned_reviewers),
                         3)  # Should respect limit
        # Should have mix of reviewers from different modules
        all_possible = [
            "user1", "user2", "user3", "user6", "user7", "user8", "user9"
        ]
        self.assertTrue(all(r in all_possible for r in self.assigned_reviewers))

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_no_matching_module(self, mock_run, mock_load_json):
        """Test PR with files that don't match any module"""
        # Setup mocks
        self.mock_changed_files = "unknown/file.txt\nrandom/path.py\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify no reviewers were assigned
        self.assertFalse(self.assign_reviewers_called)

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_existing_reviewers_skip(self, mock_run, mock_load_json):
        """Test that assignment is skipped when reviewers already exist"""
        # Setup mocks
        self.mock_changed_files = "cpp/file1.cpp\n"
        self.mock_existing_users = "existing_user1\nexisting_user2\n"
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function (without force-assign)
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify no new reviewers were assigned
        self.assertFalse(self.assign_reviewers_called)

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_force_assign_with_existing(self, mock_run, mock_load_json):
        """Test force-assign flag with existing reviewers"""
        # Setup mocks
        self.mock_changed_files = "cpp/file1.cpp\n"
        self.mock_existing_users = "user1\n"  # user1 is already assigned
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run with force-assign flag
        with patch('sys.argv', ['assign_reviewers.py', '--force-assign']):
            assign_reviewers.main()

        # Verify reviewers were assigned, excluding already assigned ones
        self.assertTrue(self.assign_reviewers_called)
        self.assertNotIn("user1",
                         self.assigned_reviewers)  # Should not re-assign
        self.assertTrue(
            all(r in ["user2", "user3"] for r in self.assigned_reviewers))

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_pr_author_excluded(self, mock_run, mock_load_json):
        """Test that PR author is excluded from reviewers"""
        # Setup with PR author as a potential reviewer
        os.environ["PR_AUTHOR"] = "user2"

        self.mock_changed_files = "cpp/file1.cpp\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify author is not in assigned reviewers
        self.assertTrue(self.assign_reviewers_called)
        self.assertNotIn("user2", self.assigned_reviewers)

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_reviewer_limit_zero(self, mock_run, mock_load_json):
        """Test with reviewer limit set to 0 (no limit)"""
        os.environ["REVIEWER_LIMIT"] = "0"

        self.mock_changed_files = "cpp/file1.cpp\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify all reviewers were assigned (no limit)
        self.assertTrue(self.assign_reviewers_called)
        self.assertEqual(len(self.assigned_reviewers),
                         3)  # All from Generic Runtime

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_dry_run_mode(self, mock_run, mock_load_json):
        """Test dry-run mode doesn't execute commands"""
        self.mock_changed_files = "cpp/file1.cpp\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Capture printed output
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            with patch('sys.argv', ['assign_reviewers.py', '--dry-run']):
                assign_reviewers.main()

        output = f.getvalue()

        # Verify dry run message was printed and no actual assignment
        self.assertIn("DRY RUN:", output)
        self.assertIn("gh pr edit", output)
        self.assertFalse(self.assign_reviewers_called)

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_empty_pr_no_files(self, mock_run, mock_load_json):
        """Test PR with no changed files"""
        self.mock_changed_files = ""
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify no reviewers were assigned
        self.assertFalse(self.assign_reviewers_called)

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_subprocess_error_handling(self, mock_run, mock_load_json):
        """Test error handling when subprocess commands fail"""
        # Mock a subprocess error
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["gh", "pr", "view"])
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run should exit with error code
        with self.assertRaises(SystemExit) as cm:
            with patch('sys.argv', ['assign_reviewers.py']):
                assign_reviewers.main()

        self.assertEqual(cm.exception.code, 1)

    def test_map_modules_function(self):
        """Test the pure map_modules function"""
        changed_files = [
            "cpp/main.cpp", "cpp/utils.h", "docs/README.md", "unknown/file.txt"
        ]

        modules, unmapped_files = assign_reviewers.map_modules(
            changed_files, self.module_paths)

        self.assertEqual(modules, {"Generic Runtime", "Documentation"})
        self.assertEqual(unmapped_files, ["unknown/file.txt"])

    def test_gather_reviewers_function(self):
        """Test the pure gather_reviewers function"""
        modules = {"Generic Runtime", "Documentation"}

        # Test without exclusions
        reviewers, modules_without_owners = assign_reviewers.gather_reviewers(
            modules, self.module_owners)
        self.assertEqual(set(reviewers), {"user1", "user2", "user3", "user9"})
        self.assertEqual(modules_without_owners, set())

        # Test with author exclusion
        reviewers, modules_without_owners = assign_reviewers.gather_reviewers(
            modules, self.module_owners, pr_author="user1")
        self.assertEqual(set(reviewers), {"user2", "user3", "user9"})
        self.assertEqual(modules_without_owners, set())

        # Test with existing reviewers exclusion
        reviewers, modules_without_owners = assign_reviewers.gather_reviewers(
            modules, self.module_owners, existing_reviewers={"user2", "user9"})
        self.assertEqual(set(reviewers), {"user1", "user3"})
        self.assertEqual(modules_without_owners, set())

    def test_modules_without_owners(self):
        """Test modules that have no owners defined"""
        modules = {"Generic Runtime", "NonExistent Module"}

        reviewers, modules_without_owners = assign_reviewers.gather_reviewers(
            modules, self.module_owners)

        self.assertEqual(set(reviewers), {"user1", "user2", "user3"})
        self.assertEqual(modules_without_owners, {"NonExistent Module"})

    def test_all_files_unmapped(self):
        """Test when all files are unmapped"""
        changed_files = ["unmapped/file1.txt", "another/file2.py"]

        modules, unmapped_files = assign_reviewers.map_modules(
            changed_files, self.module_paths)

        self.assertEqual(modules, set())
        self.assertEqual(set(unmapped_files),
                         {"unmapped/file1.txt", "another/file2.py"})

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_module_with_no_owners(self, mock_run, mock_load_json):
        """Test module that has no owners defined"""
        # Add a module with no owners
        module_owners_with_empty = self.module_owners.copy()
        module_owners_with_empty["Empty Module"] = []

        self.mock_changed_files = "empty/file.txt\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: ({
            "empty/": "Empty Module"
        } if "module-paths" in str(path) else module_owners_with_empty)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify no reviewers were assigned
        self.assertFalse(self.assign_reviewers_called)

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_files_with_special_characters(self, mock_run, mock_load_json):
        """Test files with special characters in names"""
        self.mock_changed_files = "cpp/file with spaces.cpp\ncpp/file[brackets].h\ncpp/file@special.cpp\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: (
            self.module_paths
            if "module-paths" in str(path) else self.module_owners)

        # Run the main function
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify reviewers were assigned correctly despite special characters
        self.assertTrue(self.assign_reviewers_called)
        self.assertEqual(len(self.assigned_reviewers), 3)

    @patch('assign_reviewers.load_json')
    def test_json_file_not_found(self, mock_load_json):
        """Test handling of missing JSON configuration files"""
        mock_load_json.side_effect = FileNotFoundError(
            "module-paths.json not found")

        # Run should exit with error
        with self.assertRaises(FileNotFoundError):
            with patch('sys.argv', ['assign_reviewers.py']):
                assign_reviewers.main()

    @patch('assign_reviewers.load_json')
    @patch('subprocess.run')
    def test_large_reviewer_pool(self, mock_run, mock_load_json):
        """Test with a large number of potential reviewers"""
        # Create a module with many owners
        large_module_owners = self.module_owners.copy()
        large_module_owners["Large Module"] = [f"user{i}" for i in range(20)]

        self.mock_changed_files = "large/file.cpp\n"
        self.mock_existing_users = ""
        self.mock_existing_teams = ""
        self.assign_reviewers_called = False

        mock_run.side_effect = self._mock_subprocess_run
        mock_load_json.side_effect = lambda path: ({
            "large/": "Large Module"
        } if "module-paths" in str(path) else large_module_owners)

        # Run the main function with limit
        with patch('sys.argv', ['assign_reviewers.py']):
            assign_reviewers.main()

        # Verify only 3 reviewers were selected (respecting REVIEWER_LIMIT)
        self.assertTrue(self.assign_reviewers_called)
        self.assertEqual(len(self.assigned_reviewers), 3)
        self.assertTrue(
            all(r in [f"user{i}" for i in range(20)]
                for r in self.assigned_reviewers))

    @patch('subprocess.run')
    def test_missing_environment_variables(self, mock_run):
        """Test behavior when required environment variables are missing"""
        # Remove PR_NUMBER
        if "PR_NUMBER" in os.environ:
            del os.environ["PR_NUMBER"]

        # Should raise KeyError
        with self.assertRaises(KeyError):
            with patch('sys.argv', ['assign_reviewers.py']):
                assign_reviewers.main()


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
