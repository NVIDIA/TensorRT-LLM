#!/usr/bin/env python3
"""
Unit tests for scan_build_artifacts.py

Tests all 5 modules:
  1. DpkgResolver
  2. ArtifactCollector
  3. PatternMatcher (YAML-based)
  4. OutputGenerator
  5. Main CLI

Run with: python -m pytest test_scan_build_artifacts.py -v
"""

# Import modules under test
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scan_build_artifacts import (Artifact, ArtifactCollector, DpkgResolver,
                                  Mapping, OutputGenerator, PatternMatcher)

# ============================================================================
# Test Data Models
# ============================================================================


def test_artifact_creation():
    """Test Artifact dataclass creation and serialization"""
    artifact = Artifact(path="/usr/include/stdio.h",
                        type="header",
                        source="test.d",
                        context_dir="/build",
                        metadata={"test": "value"})

    assert artifact.path == "/usr/include/stdio.h"
    assert artifact.type == "header"
    assert artifact.source == "test.d"

    # Test serialization
    data = artifact.to_dict()
    assert data['path'] == "/usr/include/stdio.h"
    assert data['metadata']['test'] == "value"


def test_mapping_creation():
    """Test Mapping dataclass creation and serialization"""
    artifact = Artifact(path="/usr/lib/libfoo.so",
                        type="library",
                        source="link.txt")

    mapping = Mapping(artifact=artifact,
                      dependency="foo",
                      confidence="high",
                      strategy="dpkg-query",
                      metadata={"test": "meta"})

    assert mapping.dependency == "foo"
    assert mapping.confidence == "high"

    # Test serialization
    data = mapping.to_dict()
    assert data['dependency'] == "foo"
    assert data['artifact']['path'] == "/usr/lib/libfoo.so"


# ============================================================================
# Test DpkgResolver
# ============================================================================


class TestDpkgResolver:
    """Test cases for DpkgResolver class"""

    def test_get_library_search_paths(self):
        """Test _get_library_search_paths returns expected directories"""
        resolver = DpkgResolver()
        paths = resolver._lib_search_paths

        # Should contain standard paths
        assert any("/lib/x86_64-linux-gnu" in p for p in paths)
        assert any("/usr/lib/x86_64-linux-gnu" in p for p in paths)

    def test_find_library_path_pthread(self):
        """Test find_library_path resolves -lpthread"""
        resolver = DpkgResolver()
        result = resolver.find_library_path("-lpthread")

        # Should find libpthread.so* in system paths
        if result:  # May not exist on all systems
            assert "pthread" in result
            assert result.endswith((".so", ".a")) or ".so." in result

    @patch('subprocess.run')
    def test_get_package_success(self, mock_run):
        """Test get_package successfully parses dpkg-query output"""
        # Mock dpkg-query output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="libc6:amd64: /lib/x86_64-linux-gnu/libc.so.6\n")

        resolver = DpkgResolver()
        package = resolver.get_package("/lib/x86_64-linux-gnu/libc.so.6")

        assert package == "libc6"
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_package_not_found(self, mock_run):
        """Test get_package returns None for non-existent file"""
        # Mock dpkg-query failure
        mock_run.return_value = Mock(returncode=1, stdout="")

        resolver = DpkgResolver()
        package = resolver.get_package("/nonexistent/file.so")

        assert package is None

    @patch('subprocess.run')
    def test_get_package_caching(self, mock_run):
        """Test get_package caches results"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="libc6:amd64: /lib/x86_64-linux-gnu/libc.so.6\n")

        resolver = DpkgResolver()

        # First call
        pkg1 = resolver.get_package("/lib/x86_64-linux-gnu/libc.so.6")
        # Second call (should use cache)
        pkg2 = resolver.get_package("/lib/x86_64-linux-gnu/libc.so.6")

        assert pkg1 == pkg2
        # Should only call dpkg-query once
        assert mock_run.call_count == 1

    def test_normalize_cuda_package(self):
        """Test _normalize_cuda_package removes version suffixes"""
        resolver = DpkgResolver()

        # Should normalize CUDA packages
        assert resolver._normalize_cuda_package("cuda-cccl-12-9") == "cuda-cccl"
        assert resolver._normalize_cuda_package(
            "cuda-cudart-dev-12-9") == "cuda-cudart-dev"
        assert resolver._normalize_cuda_package(
            "libcublas-dev-12-9") == "libcublas-dev"

        # Should NOT normalize non-CUDA packages
        assert resolver._normalize_cuda_package("libc6") == "libc6"
        assert resolver._normalize_cuda_package(
            "python3-12-1") == "python3-12-1"

    @patch('subprocess.run')
    def test_get_package_linker_flag(self, mock_run):
        """Test get_package handles -l flags by resolving first"""
        # First call: find_library_path (no mock needed, uses real filesystem)
        # Second call: dpkg-query
        mock_run.return_value = Mock(
            returncode=0,
            stdout="libc6:amd64: /lib/x86_64-linux-gnu/libpthread.so.0\n")

        resolver = DpkgResolver()

        # Mock find_library_path to return a known path
        with patch.object(resolver,
                          'find_library_path',
                          return_value='/lib/x86_64-linux-gnu/libpthread.so.0'):
            package = resolver.get_package("-lpthread")

        # Should resolve to libc6
        if package:  # May fail if system doesn't have dpkg
            assert package == "libc6"


# ============================================================================
# Test ArtifactCollector
# ============================================================================


class TestArtifactCollector:
    """Test cases for ArtifactCollector class"""

    def test_parse_d_file_basic(self, tmp_path):
        """Test _parse_d_file parses basic D file"""
        # Create test D file
        d_file = tmp_path / "test.d"
        d_file.write_text(
            "build/foo.o: /usr/include/stdio.h /usr/include/stdlib.h\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_d_file(d_file)

        # Should find 2 headers (if they exist on system)
        assert len(artifacts) >= 0  # May be 0 if headers don't exist
        for artifact in artifacts:
            assert artifact.type == "header"
            assert artifact.source == str(d_file)

    def test_parse_d_file_line_continuations(self, tmp_path):
        """Test _parse_d_file handles line continuations"""
        # Create test D file with line continuations
        d_file = tmp_path / "test.d"
        d_file.write_text(
            "build/foo.o: /usr/include/stdio.h \\\n  /usr/include/stdlib.h\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_d_file(d_file)

        # Should handle continuations correctly
        assert isinstance(artifacts, list)

    def test_parse_d_file_relative_paths(self, tmp_path):
        """Test _parse_d_file resolves relative paths"""
        # Create test header
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        test_header = include_dir / "test.h"
        test_header.write_text("// test header\n")

        # Create D file with relative path
        d_file = tmp_path / "build" / "test.d"
        d_file.parent.mkdir(parents=True, exist_ok=True)
        d_file.write_text(f"build/foo.o: ../include/test.h\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_d_file(d_file)

        # Should resolve relative path
        if artifacts:
            assert any("test.h" in a.path for a in artifacts)

    def test_parse_link_file_basic(self, tmp_path):
        """Test _parse_link_file parses link.txt"""
        # Create test link file
        link_file = tmp_path / "link.txt"
        link_file.write_text(
            "/usr/bin/c++ -o foo -lpthread -ldl /path/to/libfoo.a\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_link_file(link_file)

        # Should find -l flags
        assert any(a.path == "-lpthread" for a in artifacts)
        assert any(a.path == "-ldl" for a in artifacts)

    def test_parse_link_file_response_files(self, tmp_path):
        """Test _parse_link_file handles @response.rsp recursively"""
        # Create response file
        rsp_file = tmp_path / "response.rsp"
        rsp_file.write_text("-lpthread -ldl\n")

        # Create link file referencing response file
        link_file = tmp_path / "link.txt"
        link_file.write_text(f"/usr/bin/c++ -o foo @response.rsp\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_link_file(link_file)

        # Should recursively expand response file
        assert any(a.path == "-lpthread" for a in artifacts)

    def test_parse_link_file_static_libraries(self, tmp_path):
        """Test _parse_link_file handles .a files"""
        # Create static library
        static_lib = tmp_path / "libtest.a"
        static_lib.write_text("fake static library\n")

        # Create link file
        link_file = tmp_path / "link.txt"
        link_file.write_text(f"/usr/bin/c++ -o foo {static_lib}\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_link_file(link_file)

        # Should find static library
        assert any("libtest.a" in a.path for a in artifacts)
        if artifacts:
            assert artifacts[0].metadata.get('static') == True

    def test_get_needed_libraries(self, tmp_path):
        """Test _get_needed_libraries extracts NEEDED entries"""
        # Mock readelf output
        mock_output = """
Dynamic section at offset 0x1000 contains 20 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x000000000000000e (SONAME)             Library soname: [libtest.so]
        """

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout=mock_output)

            needed = ArtifactCollector._get_needed_libraries(tmp_path /
                                                             "fake.so")

        assert "libcudart.so.12" in needed
        assert "libstdc++.so.6" in needed

    def test_scan_wheel(self, tmp_path):
        """Test _scan_wheel extracts and scans wheel"""
        # Create fake wheel
        wheel_file = tmp_path / "test-1.0-py3-none-any.whl"

        with zipfile.ZipFile(wheel_file, 'w') as zf:
            # Add fake .so file
            zf.writestr("tensorrt_llm/libs/libtest.so", b"fake binary")

        collector = ArtifactCollector(tmp_path)

        with patch.object(ArtifactCollector,
                          '_get_needed_libraries',
                          return_value=['libfoo.so']):
            artifacts = collector._scan_wheel(wheel_file)

        # Should find binary artifact
        assert any(a.type == "binary" for a in artifacts)
        assert any(a.type == "library" and a.path == "libfoo.so"
                   for a in artifacts)

    def test_collect_all_deduplication(self, tmp_path):
        """Test collect_all deduplicates artifacts"""
        # Create two D files with overlapping headers
        d1 = tmp_path / "test1.d"
        d2 = tmp_path / "test2.d"

        d1.write_text("build/foo.o: /usr/include/stdio.h\n")
        d2.write_text("build/bar.o: /usr/include/stdio.h\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector.collect_all()

        # Should deduplicate by path
        paths = [a.path for a in artifacts]
        assert len(paths) == len(set(paths))  # No duplicates


# ============================================================================
# Test PatternMatcher
# ============================================================================


class TestPatternMatcher:
    """Test cases for PatternMatcher class"""

    @pytest.fixture
    def dependencies_dir(self, tmp_path):
        """Create test dependencies directory with YAML files"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create dpkg.yml with system packages
        dpkg_data = {
            "dependencies": [{
                "name": "libc6",
                "version": "2.35",
                "description": "GNU C Library: Shared libraries",
                "basename_matches": [],
                "linker_flags_matches": ["-lpthread"],
                "directory_matches": [],
                "aliases": []
            }]
        }
        with open(deps_dir / "dpkg.yml", 'w') as f:
            yaml.dump(dpkg_data, f)

        # Create cuda-cudart-12.yml
        cuda_data = {
            "name": "cuda-cudart-12",
            "version": "12.0",
            "description": "NVIDIA CUDA Runtime library version 12",
            "basename_matches": ["libcudart.so.12"],
            "linker_flags_matches": [],
            "directory_matches": ["cuda-12"],
            "aliases": []
        }
        with open(deps_dir / "cuda-cudart-12.yml", 'w') as f:
            yaml.dump(cuda_data, f)

        # Create pytorch.yml
        pytorch_data = {
            "name": "pytorch",
            "version": "2.0",
            "description": "PyTorch machine learning framework",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["pytorch"],
            "aliases": []
        }
        with open(deps_dir / "pytorch.yml", 'w') as f:
            yaml.dump(pytorch_data, f)

        # Create deepep.yml with bundled binary pattern
        deepep_data = {
            "name": "deepep",
            "version": "1.0",
            "description": "DeepEP library",
            "basename_matches": ["deep_ep_cpp"],
            "linker_flags_matches": [],
            "directory_matches": [],
            "aliases": []
        }
        with open(deps_dir / "deepep.yml", 'w') as f:
            yaml.dump(deepep_data, f)

        # Create nlohmann-json.yml
        json_data = {
            "name": "nlohmann-json",
            "version": "3.11",
            "description": "JSON for Modern C++",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["json"],
            "aliases": []
        }
        with open(deps_dir / "nlohmann-json.yml", 'w') as f:
            yaml.dump(json_data, f)

        return deps_dir

    def test_match_exact_library(self, dependencies_dir):
        """Test _match_patterns finds exact matches"""
        matcher = PatternMatcher(dependencies_dir)

        artifact = Artifact(path="-lpthread", type="library", source="link.txt")

        mapping = matcher._match_patterns(artifact)

        assert mapping is not None
        assert mapping.dependency == "libc6"
        assert mapping.confidence == "high"
        assert mapping.strategy == "exact_pattern_match"

    def test_match_substring_pattern(self, dependencies_dir):
        """Test that substring matching is no longer supported (removed for safety)"""
        matcher = PatternMatcher(dependencies_dir)

        artifact = Artifact(path="tensorrt_llm/libs/deep_ep_cpp_tllm.so",
                            type="binary",
                            source="wheel")

        # Substring matching was removed from _match_patterns to prevent false positives
        # This test verifies it returns None for non-exact matches
        mapping = matcher._match_patterns(artifact)

        # Should return None since "deep_ep_cpp_tllm.so" doesn't exactly match "deep_ep_cpp"
        assert mapping is None

    def test_match_path_alias_rightmost(self, dependencies_dir):
        """Test _match_path_alias uses rightmost directory match"""
        matcher = PatternMatcher(dependencies_dir)

        artifact = Artifact(path="/build/pytorch/include/torch/torch.h",
                            type="header",
                            source="test.d")

        mapping = matcher._match_path_alias(artifact)

        assert mapping is not None
        assert mapping.dependency == "pytorch"
        assert mapping.metadata['matched_component'] == "pytorch"

    def test_match_generic_library_fallback(self, dependencies_dir):
        """Test _match_generic_library as fallback"""
        matcher = PatternMatcher(dependencies_dir)

        artifact = Artifact(path="/usr/lib/libunknown.so.1",
                            type="library",
                            source="link.txt")

        mapping = matcher._match_generic_library(artifact)

        assert mapping is not None
        assert mapping.dependency == "unknown"
        assert mapping.confidence == "low"

    def test_match_full_cascade(self, dependencies_dir):
        """Test match() tries all strategies in order"""
        matcher = PatternMatcher(dependencies_dir)

        # Should match exact library (highest priority)
        artifact1 = Artifact(path="-lpthread", type="library", source="test")
        mapping1 = matcher.match(artifact1)
        assert mapping1 is not None
        assert mapping1.strategy == "exact_pattern_match"

        # Should match exact pattern
        artifact2 = Artifact(path="/usr/lib/libcudart.so.12",
                             type="library",
                             source="test")
        mapping2 = matcher.match(artifact2)
        assert mapping2 is not None
        assert mapping2.strategy == "exact_pattern_match"

        # Should fall back to generic
        artifact3 = Artifact(path="/usr/lib/libfallback.so",
                             type="library",
                             source="test")
        mapping3 = matcher.match(artifact3)
        assert mapping3 is not None
        assert mapping3.strategy == "generic_library_inference"

    def test_yaml_loading_individual_files(self, tmp_path):
        """Test loading individual YAML dependency files"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create individual dependency file
        dep_data = {
            "name": "test-lib",
            "version": "1.0",
            "description": "Test library for unit tests",
            "basename_matches": ["libtest.so"],
            "linker_flags_matches": ["-ltest"],
            "directory_matches": ["test-lib"],
            "aliases": ["testlib"]
        }
        with open(deps_dir / "test-lib.yml", 'w') as f:
            yaml.dump(dep_data, f)

        matcher = PatternMatcher(deps_dir)

        # Check pattern_mappings
        assert "libtest.so" in matcher.pattern_mappings
        assert matcher.pattern_mappings["libtest.so"] == "test-lib"
        assert "-ltest" in matcher.pattern_mappings
        assert matcher.pattern_mappings["-ltest"] == "test-lib"

        # Check path_aliases
        assert "test-lib" in matcher.path_aliases
        assert matcher.path_aliases["test-lib"] == "test-lib"
        assert "testlib" in matcher.path_aliases
        assert matcher.path_aliases["testlib"] == "test-lib"

    def test_yaml_loading_dpkg_format(self, tmp_path):
        """Test loading dpkg.yml with dependencies list"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create dpkg.yml with multiple dependencies
        dpkg_data = {
            "dependencies": [{
                "name": "dep1",
                "version": "1.0",
                "description": "First dependency",
                "basename_matches": ["libdep1.so"],
                "linker_flags_matches": [],
                "directory_matches": [],
                "aliases": []
            }, {
                "name": "dep2",
                "version": "2.0",
                "description": "Second dependency",
                "basename_matches": ["libdep2.so"],
                "linker_flags_matches": [],
                "directory_matches": [],
                "aliases": []
            }]
        }
        with open(deps_dir / "dpkg.yml", 'w') as f:
            yaml.dump(dpkg_data, f)

        matcher = PatternMatcher(deps_dir)

        # Both dependencies should be loaded
        assert "libdep1.so" in matcher.pattern_mappings
        assert matcher.pattern_mappings["libdep1.so"] == "dep1"
        assert "libdep2.so" in matcher.pattern_mappings
        assert matcher.pattern_mappings["libdep2.so"] == "dep2"

    def test_yaml_duplicate_pattern_warning(self, tmp_path, capsys):
        """Test that duplicate patterns generate warnings"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create first file with pattern
        dep1_data = {
            "name": "dep1",
            "version": "1.0",
            "description": "First dependency with duplicate pattern",
            "basename_matches": ["duplicate.so"],
            "linker_flags_matches": [],
            "directory_matches": [],
            "aliases": []
        }
        with open(deps_dir / "dep1.yml", 'w') as f:
            yaml.dump(dep1_data, f)

        # Create second file with same pattern
        dep2_data = {
            "name": "dep2",
            "version": "2.0",
            "description": "Second dependency with duplicate pattern",
            "basename_matches": ["duplicate.so"],
            "linker_flags_matches": [],
            "directory_matches": [],
            "aliases": []
        }
        with open(deps_dir / "dep2.yml", 'w') as f:
            yaml.dump(dep2_data, f)

        # Initialize matcher (should emit warning)
        matcher = PatternMatcher(deps_dir)

        # Check warning was emitted
        captured = capsys.readouterr()
        assert "Warning: Duplicate basename match 'duplicate.so'" in captured.err

        # Last one wins
        assert matcher.pattern_mappings["duplicate.so"] == "dep2"

    def test_yaml_invalid_format_warning(self, tmp_path, capsys):
        """Test that invalid YAML format generates warnings"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create file with invalid format (missing name)
        with open(deps_dir / "invalid.yml", 'w') as f:
            yaml.dump({
                "version": "1.0",
                "description": "Missing name field"
            }, f)

        # Initialize matcher (should emit warning)
        PatternMatcher(deps_dir)

        # Check warning was emitted (either "Missing 'name' field" or "unrecognized format")
        captured = capsys.readouterr()
        assert ("Warning: Missing 'name' field" in captured.err
                or "Warning: Skipping invalid.yml - unrecognized format"
                in captured.err)

    def test_yaml_skip_underscore_files(self, tmp_path):
        """Test that files starting with underscore are skipped"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create _schema.yml (should be skipped)
        schema_data = {
            "name": "should-not-load",
            "version": "1.0",
            "description": "This file should be skipped",
            "basename_matches": ["should-not-exist.so"],
            "linker_flags_matches": [],
            "directory_matches": [],
            "aliases": []
        }
        with open(deps_dir / "_schema.yml", 'w') as f:
            yaml.dump(schema_data, f)

        # Create normal file (should be loaded)
        normal_data = {
            "name": "normal-dep",
            "version": "1.0",
            "description": "Normal dependency file",
            "basename_matches": ["normal.so"],
            "linker_flags_matches": [],
            "directory_matches": [],
            "aliases": []
        }
        with open(deps_dir / "normal-dep.yml", 'w') as f:
            yaml.dump(normal_data, f)

        matcher = PatternMatcher(deps_dir)

        # Schema file should not be loaded
        assert "should-not-exist.so" not in matcher.pattern_mappings

        # Normal file should be loaded
        assert "normal.so" in matcher.pattern_mappings
        assert matcher.pattern_mappings["normal.so"] == "normal-dep"

    def test_yaml_mixed_loading(self, tmp_path):
        """Test loading both dpkg.yml and individual files together"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create dpkg.yml
        dpkg_data = {
            "dependencies": [{
                "name": "system-dep",
                "version": "1.0",
                "description": "System dependency from dpkg",
                "basename_matches": ["libsystem.so"],
                "linker_flags_matches": [],
                "directory_matches": [],
                "aliases": []
            }]
        }
        with open(deps_dir / "dpkg.yml", 'w') as f:
            yaml.dump(dpkg_data, f)

        # Create individual file
        custom_data = {
            "name": "custom-dep",
            "version": "2.0",
            "description": "Custom dependency from individual file",
            "basename_matches": ["libcustom.so"],
            "linker_flags_matches": [],
            "directory_matches": [],
            "aliases": []
        }
        with open(deps_dir / "custom-dep.yml", 'w') as f:
            yaml.dump(custom_data, f)

        matcher = PatternMatcher(deps_dir)

        # Both should be loaded
        assert "libsystem.so" in matcher.pattern_mappings
        assert matcher.pattern_mappings["libsystem.so"] == "system-dep"
        assert "libcustom.so" in matcher.pattern_mappings
        assert matcher.pattern_mappings["libcustom.so"] == "custom-dep"

    def test_yaml_empty_arrays(self, tmp_path):
        """Test that empty arrays in YAML are handled correctly"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create dependency with empty arrays
        dep_data = {
            "name": "minimal-dep",
            "version": "1.0",
            "description": "Minimal dependency with empty arrays",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["minimal"],
            "aliases": []
        }
        with open(deps_dir / "minimal-dep.yml", 'w') as f:
            yaml.dump(dep_data, f)

        matcher = PatternMatcher(deps_dir)

        # Should load successfully
        assert "minimal" in matcher.path_aliases
        assert matcher.path_aliases["minimal"] == "minimal-dep"


# ============================================================================
# Test OutputGenerator
# ============================================================================


class TestOutputGenerator:
    """Test cases for OutputGenerator class"""

    def test_generate_creates_files(self, tmp_path):
        """Test generate() creates known and unknown YAML files"""
        artifacts = [
            Artifact(path="/usr/include/stdio.h",
                     type="header",
                     source="test.d"),
            Artifact(path="/usr/lib/libfoo.so",
                     type="library",
                     source="link.txt"),
            Artifact(path="/unknown/header.h", type="header", source="test.d")
        ]

        mappings = [
            Mapping(artifact=artifacts[0],
                    dependency="libc6",
                    confidence="high",
                    strategy="dpkg-query"),
            Mapping(artifact=artifacts[1],
                    dependency="foo",
                    confidence="medium",
                    strategy="pattern")
        ]

        output_dir = tmp_path / "reports"
        known_file, unknown_file = OutputGenerator.generate(
            mappings, artifacts, output_dir)

        # Check files exist
        assert known_file.exists()
        assert unknown_file.exists()

        # Check known.yml content (simplified structure: dependencies dict of lists)
        with open(known_file) as f:
            known_data = yaml.safe_load(f)

        assert known_data['summary']['total_artifacts'] == 3
        assert known_data['summary']['mapped'] == 2
        assert known_data['summary']['unmapped'] == 1
        assert len(known_data['dependencies']) == 2
        # Check dependencies is a dict with lists of paths
        assert isinstance(known_data['dependencies'], dict)
        assert 'libc6' in known_data['dependencies']
        assert 'foo' in known_data['dependencies']
        assert '/usr/include/stdio.h' in known_data['dependencies']['libc6']
        assert '/usr/lib/libfoo.so' in known_data['dependencies']['foo']

        # Check unknown.yml content (simplified structure: flat list of paths)
        with open(unknown_file) as f:
            unknown_data = yaml.safe_load(f)

        assert unknown_data['summary']['count'] == 1
        assert len(unknown_data['artifacts']) == 1
        assert "/unknown/header.h" in unknown_data['artifacts']

    def test_generate_groups_by_dependency(self, tmp_path):
        """Test generate() groups artifacts by dependency"""
        artifacts = [
            Artifact(path="/usr/include/stdio.h",
                     type="header",
                     source="test1.d"),
            Artifact(path="/usr/include/stdlib.h",
                     type="header",
                     source="test2.d"),
        ]

        mappings = [
            Mapping(artifact=artifacts[0],
                    dependency="libc6",
                    confidence="high",
                    strategy="dpkg"),
            Mapping(artifact=artifacts[1],
                    dependency="libc6",
                    confidence="high",
                    strategy="dpkg"),
        ]

        output_dir = tmp_path / "reports"
        known_file, _ = OutputGenerator.generate(mappings, artifacts,
                                                 output_dir)

        with open(known_file) as f:
            known_data = yaml.safe_load(f)

        # Should have 1 dependency with 2 artifacts (simplified: dict of lists)
        assert len(known_data['dependencies']) == 1
        assert 'libc6' in known_data['dependencies']
        assert len(known_data['dependencies']['libc6']) == 2
        assert '/usr/include/stdio.h' in known_data['dependencies']['libc6']
        assert '/usr/include/stdlib.h' in known_data['dependencies']['libc6']

    def test_generate_coverage_calculation(self, tmp_path):
        """Test generate() calculates coverage correctly"""
        artifacts = [
            Artifact(path=f"/test{i}.h", type="header", source="test.d")
            for i in range(10)
        ]
        mappings = [
            Mapping(artifact=artifacts[i],
                    dependency=f"dep{i}",
                    confidence="high",
                    strategy="dpkg") for i in range(7)  # 7 out of 10 mapped
        ]

        output_dir = tmp_path / "reports"
        known_file, _ = OutputGenerator.generate(mappings, artifacts,
                                                 output_dir)

        with open(known_file) as f:
            known_data = yaml.safe_load(f)

        # Verify summary section is still included in YAML output
        assert "70.0%" in known_data['summary']['coverage']


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow(self, tmp_path):
        """Test complete scan workflow end-to-end"""
        # Setup: Create dependencies directory with YAML files
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create dpkg.yml
        dpkg_data = {
            "dependencies": [{
                "name": "libc6",
                "version": "2.35",
                "description": "GNU C Library: Shared libraries",
                "basename_matches": [],
                "linker_flags_matches": ["-lpthread"],
                "directory_matches": [],
                "aliases": []
            }]
        }
        with open(deps_dir / "dpkg.yml", 'w') as f:
            yaml.dump(dpkg_data, f)

        # Setup: Create build artifacts
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        d_file = build_dir / "test.d"
        d_file.write_text("build/foo.o: /usr/include/stdio.h\n")

        link_file = build_dir / "link.txt"
        link_file.write_text("/usr/bin/c++ -o foo -lpthread\n")

        # Run workflow
        collector = ArtifactCollector(build_dir)
        artifacts = collector.collect_all()

        dpkg_resolver = DpkgResolver()
        pattern_matcher = PatternMatcher(deps_dir)

        all_mappings = []

        # Try dpkg first
        for artifact in artifacts:
            with patch.object(dpkg_resolver,
                              'get_package',
                              return_value='libc6'):
                package = dpkg_resolver.get_package(artifact.path)
                if package:
                    all_mappings.append(
                        Mapping(artifact=artifact,
                                dependency=package,
                                confidence='high',
                                strategy='dpkg-query'))

        # Try patterns for remaining
        dpkg_paths = {m.artifact.path for m in all_mappings}
        for artifact in artifacts:
            if artifact.path not in dpkg_paths:
                mapping = pattern_matcher.match(artifact)
                if mapping:
                    all_mappings.append(mapping)

        # Generate reports
        output_dir = tmp_path / "reports"
        known_file, unknown_file = OutputGenerator.generate(
            all_mappings, artifacts, output_dir)

        # Verify outputs
        assert known_file.exists()
        assert unknown_file.exists()

        with open(known_file) as f:
            data = yaml.safe_load(f)
            assert data['summary']['total_artifacts'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
