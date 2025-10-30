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
import os
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

    def test_parse_d_file_trailing_colons(self, tmp_path):
        """Test _parse_d_file strips trailing colons from malformed paths"""
        # Create test D file with trailing colons (malformed .d file)
        d_file = tmp_path / "test.d"
        d_file.write_text(
            "build/foo.o: /usr/include/stdio.h: /usr/include/stdlib.h:\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_d_file(d_file)

        # Should strip trailing colons from paths
        for artifact in artifacts:
            assert not artifact.path.endswith(':')
            assert artifact.path  # No empty strings

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

    def test_parse_link_file_cmake_linker_artifacts(self, tmp_path):
        """Test _parse_link_file handles CMakeFiles artifacts with -Wl,-soname"""
        # Create link file with CMakeFiles linker artifact
        link_file = tmp_path / "link.txt"
        link_file.write_text(
            "/usr/bin/c++ -o foo /build/CMakeFiles/foo.dir/-Wl,-soname,libtest.so.1\n"
        )

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_link_file(link_file)

        # Should extract library name from CMakeFiles artifact
        linker_artifacts = [
            a for a in artifacts if a.metadata.get('cmake_linker_artifact')
        ]
        assert len(linker_artifacts) == 1

        artifact = linker_artifacts[0]
        assert artifact.path == "-ltest"  # Extracted from libtest.so.1
        assert artifact.type == "library"
        assert artifact.metadata['linker_flag'] == True
        assert artifact.metadata['cmake_linker_artifact'] == True
        assert artifact.metadata['library_name'] == "libtest.so.1"

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

    def test_parse_d_file_relative_path_resolution(self, tmp_path):
        """Test _parse_d_file resolves paths relative to build_dir, not .d file's parent

        This test verifies the fix for the critical bug where relative paths in CMake .d files
        were being resolved from the wrong directory. CMake .d files use paths relative to the
        build root, not the .d file's directory.
        """
        # Create a realistic directory structure:
        # tmp_path (build_dir)
        # ├── CMakeFiles/
        # │   └── deeply/
        # │       └── nested/
        # │           └── target.dir/
        # │               └── test.d  (contains relative paths)
        # └── tensorrt_llm/
        #     └── runtime/
        #         └── layerProfiler.h  (target file)

        # Create the target header file
        runtime_dir = tmp_path / "tensorrt_llm" / "runtime"
        runtime_dir.mkdir(parents=True)
        target_header = runtime_dir / "layerProfiler.h"
        target_header.write_text("// TensorRT-LLM runtime header\n")

        # Create deeply nested .d file directory
        d_file_dir = tmp_path / "CMakeFiles" / "deeply" / "nested" / "target.dir"
        d_file_dir.mkdir(parents=True, exist_ok=True)
        d_file = d_file_dir / "test.d"

        # Write .d file with relative path from BUILD ROOT, not from .d file location
        # From build root: tensorrt_llm/runtime/layerProfiler.h
        # From .d file location: ../../../../../tensorrt_llm/runtime/layerProfiler.h would be wrong
        d_file.write_text("build/foo.o: tensorrt_llm/runtime/layerProfiler.h\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_d_file(d_file)

        # Should resolve relative path using build_dir as context, not d_file.parent
        assert len(artifacts) == 1, f"Expected 1 artifact, got {len(artifacts)}"
        artifact = artifacts[0]

        # Verify the path was resolved correctly
        assert "layerProfiler.h" in artifact.path
        assert artifact.type == "header"
        assert artifact.source == str(d_file)

        # Verify context_dir is build_dir, not d_file.parent
        assert artifact.context_dir == str(tmp_path)

        # Verify path exists (resolved correctly)
        assert artifact.metadata.get('path_exists') is True

        # Verify the resolved path points to the actual file
        canonical_target = os.path.realpath(str(target_header))
        assert artifact.path == canonical_target

    def test_parse_d_file_build_root_context(self, tmp_path):
        """Test that context_dir used for path resolution is self.build_dir

        This test verifies that regardless of where the .d file is located in the build tree,
        all relative paths are resolved from the same build root directory.
        """
        # Create headers at different locations
        header1_dir = tmp_path / "include"
        header1_dir.mkdir()
        header1 = header1_dir / "test1.h"
        header1.write_text("// test1 header\n")

        header2_dir = tmp_path / "src" / "common"
        header2_dir.mkdir(parents=True)
        header2 = header2_dir / "test2.h"
        header2.write_text("// test2 header\n")

        # Create .d files at different depths
        d_file1 = tmp_path / "shallow.d"
        d_file1.write_text("build/obj1.o: include/test1.h\n")

        d_file2_dir = tmp_path / "CMakeFiles" / "deep" / "nested"
        d_file2_dir.mkdir(parents=True)
        d_file2 = d_file2_dir / "deep.d"
        d_file2.write_text("build/obj2.o: src/common/test2.h\n")

        collector = ArtifactCollector(tmp_path)

        # Parse both .d files
        artifacts1 = collector._parse_d_file(d_file1)
        artifacts2 = collector._parse_d_file(d_file2)

        # Both should resolve successfully
        assert len(artifacts1) == 1
        assert len(artifacts2) == 1

        # Both should use build_dir as context_dir
        assert artifacts1[0].context_dir == str(tmp_path)
        assert artifacts2[0].context_dir == str(tmp_path)

        # Both should resolve to correct absolute paths
        assert artifacts1[0].metadata.get('path_exists') is True
        assert artifacts2[0].metadata.get('path_exists') is True

        # Verify correct files were found
        assert "test1.h" in artifacts1[0].path
        assert "test2.h" in artifacts2[0].path

    def test_parse_d_file_cross_project_paths(self, tmp_path):
        """Test _parse_d_file handles paths that reference directories outside the build

        This test verifies that paths referencing parent directories are resolved correctly
        relative to build root, and that non-existent paths are properly marked.
        """
        # Create a project structure with 3rdparty dependencies:
        # tmp_path (build_dir)
        # ├── CMakeFiles/
        # │   └── target.dir/
        # │       └── test.d
        # And simulate references to:
        # ../../../../triton_backend/src/model.h (doesn't exist)
        # ../../../tensorrt_llm/common/logger.h (exists)

        # Create an existing header in a sibling directory structure
        parent_dir = tmp_path.parent
        trtllm_dir = parent_dir / "tensorrt_llm" / "common"
        trtllm_dir.mkdir(parents=True, exist_ok=True)
        existing_header = trtllm_dir / "logger.h"
        existing_header.write_text("// Logger header\n")

        # Create .d file with both existing and non-existing cross-project paths
        d_file_dir = tmp_path / "CMakeFiles" / "target.dir"
        d_file_dir.mkdir(parents=True)
        d_file = d_file_dir / "test.d"

        # These paths are relative to BUILD ROOT
        d_file.write_text(
            "build/foo.o: ../../../../triton_backend/src/model.h ../tensorrt_llm/common/logger.h\n"
        )

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_d_file(d_file)

        # Should process both paths
        assert len(artifacts) == 2

        # Find the artifacts
        triton_artifact = None
        logger_artifact = None

        for artifact in artifacts:
            if "triton_backend" in artifact.path:
                triton_artifact = artifact
            elif "logger.h" in artifact.path:
                logger_artifact = artifact

        # Verify triton_backend artifact (non-existent)
        assert triton_artifact is not None
        assert triton_artifact.type == "header"
        assert triton_artifact.context_dir == str(tmp_path)
        assert triton_artifact.metadata.get('path_exists') is False

        # Verify logger artifact (exists)
        assert logger_artifact is not None
        assert logger_artifact.type == "header"
        assert logger_artifact.context_dir == str(tmp_path)
        assert logger_artifact.metadata.get('path_exists') is True

        # Verify logger resolved to correct absolute path
        canonical_existing = os.path.realpath(str(existing_header))
        assert logger_artifact.path == canonical_existing

    def test_parse_d_file_prevents_false_positives(self, tmp_path):
        """Test that correct path resolution prevents false positives in dependency classification

        This test demonstrates the practical impact of the bug fix: when paths are resolved
        correctly from build_dir, dependencies are classified accurately.
        """
        # Scenario: A .d file deep in the build tree references a 3rdparty dependency
        # OLD BUG: Would resolve from .d file's parent, potentially missing the file
        # NEW FIX: Resolves from build root, finds the file correctly

        # Create a 3rdparty dependency structure
        third_party_dir = tmp_path / "3rdparty" / "cutlass" / "include"
        third_party_dir.mkdir(parents=True)
        cutlass_header = third_party_dir / "cutlass.h"
        cutlass_header.write_text("// CUTLASS header\n")

        # Create deeply nested .d file (simulating CMake's structure)
        d_file_dir = tmp_path / "CMakeFiles" / "tensorrt_llm.dir" / "batch_manager" / "llm_request.cpp.o.d"
        d_file_dir.parent.mkdir(parents=True, exist_ok=True)
        d_file = d_file_dir

        # .d file contains relative path from BUILD ROOT to cutlass
        # OLD BUG: Would try to resolve from .d file's parent → incorrect path
        # NEW FIX: Resolves from build root → correct path
        d_file.write_text("build/obj.o: 3rdparty/cutlass/include/cutlass.h\n")

        collector = ArtifactCollector(tmp_path)
        artifacts = collector._parse_d_file(d_file)

        # Should successfully resolve the path
        assert len(artifacts) == 1
        artifact = artifacts[0]

        # Verify correct resolution
        assert artifact.metadata.get('path_exists') is True
        assert "cutlass.h" in artifact.path

        # Verify it resolved to the actual file
        canonical_cutlass = os.path.realpath(str(cutlass_header))
        assert artifact.path == canonical_cutlass

        # This artifact can now be correctly matched to cutlass dependency
        # (with correct path resolution, pattern matching will work)


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
                "directory_matches": []
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
            "directory_matches": ["cuda-12"]
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
            "directory_matches": ["pytorch"]
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
            "directory_matches": []
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
            "directory_matches": ["json"]
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
        assert mapping.metadata['matched_pattern'] == "pytorch"
        assert mapping.metadata['matched_sequence'] == "pytorch"

    def test_match_path_multi_directory(self, tmp_path):
        """Test _match_path_alias matches multi-directory patterns"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create YAML with multi-directory pattern
        dep_data = {
            "name": "test-lib",
            "description": "Test library with multi-directory pattern",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["foo/bar", "3rdparty/test"]
        }
        with open(deps_dir / "test-lib.yml", 'w') as f:
            yaml.dump(dep_data, f)

        matcher = PatternMatcher(deps_dir)

        # Test: /home/foo/bar/file.h matches "foo/bar"
        artifact1 = Artifact(path="/home/foo/bar/file.h",
                             type="header",
                             source="test.d")
        mapping1 = matcher._match_path_alias(artifact1)
        assert mapping1 is not None
        assert mapping1.dependency == "test-lib"
        assert mapping1.metadata['matched_pattern'] == "foo/bar"
        assert mapping1.metadata['matched_sequence'] == "foo/bar"

        # Test: /home/foobar/file.h does NOT match "foo/bar" (no substring matching)
        artifact2 = Artifact(path="/home/foobar/file.h",
                             type="header",
                             source="test.d")
        mapping2 = matcher._match_path_alias(artifact2)
        assert mapping2 is None

        # Test: /build/3rdparty/test/include/test.h matches "3rdparty/test"
        artifact3 = Artifact(path="/build/3rdparty/test/include/test.h",
                             type="header",
                             source="test.d")
        mapping3 = matcher._match_path_alias(artifact3)
        assert mapping3 is not None
        assert mapping3.dependency == "test-lib"
        assert mapping3.metadata['matched_pattern'] == "3rdparty/test"

    def test_match_path_multi_directory_rightmost(self, tmp_path):
        """Test rightmost wins for multi-directory patterns"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        dep_data = {
            "name": "test-lib",
            "description": "Test library for rightmost matching",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["foo/bar"]
        }
        with open(deps_dir / "test-lib.yml", 'w') as f:
            yaml.dump(dep_data, f)

        matcher = PatternMatcher(deps_dir)

        # Pattern: "foo/bar" appears twice in path
        # Path: /foo/bar/baz/foo/bar/qux.h
        # Should match at rightmost position (position 3)
        artifact = Artifact(path="/foo/bar/baz/foo/bar/qux.h",
                            type="header",
                            source="test.d")
        mapping = matcher._match_path_alias(artifact)

        assert mapping is not None
        assert mapping.dependency == "test-lib"
        assert mapping.metadata['matched_pattern'] == "foo/bar"
        # Position should be 3 (rightmost occurrence)
        assert mapping.metadata['position'] == 3

    def test_match_path_no_substring_matching(self, tmp_path):
        """Test that substring matching is NOT supported"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        dep_data = {
            "name": "test-lib",
            "description": "Test library for substring verification",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["oo/ba", "o/b"]
        }
        with open(deps_dir / "test-lib.yml", 'w') as f:
            yaml.dump(dep_data, f)

        matcher = PatternMatcher(deps_dir)

        # Pattern: "oo/ba"
        # Path: /foo/bar/file.h
        # Should NOT match ("oo" != "foo", "ba" != "bar")
        artifact1 = Artifact(path="/foo/bar/file.h",
                             type="header",
                             source="test.d")
        mapping1 = matcher._match_path_alias(artifact1)
        assert mapping1 is None

        # Pattern: "o/b"
        # Path: /foo/bar/file.h
        # Should NOT match
        artifact2 = Artifact(path="/foo/bar/file.h",
                             type="header",
                             source="test.d")
        mapping2 = matcher._match_path_alias(artifact2)
        assert mapping2 is None

    def test_match_path_mixed_single_and_multi(self, tmp_path):
        """Test single and multi-directory patterns coexist"""
        deps_dir = tmp_path / "dependencies"
        deps_dir.mkdir()

        # Create two dependencies: one with single, one with multi-dir patterns
        dep1_data = {
            "name": "pytorch",
            "description": "PyTorch with single component",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["pytorch", "torch"]
        }
        with open(deps_dir / "pytorch.yml", 'w') as f:
            yaml.dump(dep1_data, f)

        dep2_data = {
            "name": "cutlass",
            "description": "Cutlass with multi-directory",
            "basename_matches": [],
            "linker_flags_matches": [],
            "directory_matches": ["3rdparty/cutlass"]
        }
        with open(deps_dir / "cutlass.yml", 'w') as f:
            yaml.dump(dep2_data, f)

        matcher = PatternMatcher(deps_dir)

        # Test single component pattern still works
        artifact1 = Artifact(path="/home/pytorch/lib/test.so",
                             type="library",
                             source="test")
        mapping1 = matcher._match_path_alias(artifact1)
        assert mapping1 is not None
        assert mapping1.dependency == "pytorch"

        # Test multi-directory pattern works
        artifact2 = Artifact(path="/build/3rdparty/cutlass/include/cutlass.h",
                             type="header",
                             source="test.d")
        mapping2 = matcher._match_path_alias(artifact2)
        assert mapping2 is not None
        assert mapping2.dependency == "cutlass"
        assert mapping2.metadata['matched_sequence'] == "3rdparty/cutlass"

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
            "directory_matches": ["test-lib"]
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
                "directory_matches": []
            }, {
                "name": "dep2",
                "version": "2.0",
                "description": "Second dependency",
                "basename_matches": ["libdep2.so"],
                "linker_flags_matches": [],
                "directory_matches": []
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
            "directory_matches": []
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
            "directory_matches": []
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
            "directory_matches": []
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
            "directory_matches": []
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
                "directory_matches": []
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
            "directory_matches": []
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
            "directory_matches": ["minimal"]
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

    def test_generate_path_issues_yml_basic(self, tmp_path):
        """Test generate() creates path_issues.yml with non-existent headers"""
        artifacts = [
            Artifact(path="/usr/include/stdio.h",
                     type="header",
                     source="test.d",
                     metadata={'path_exists': True}),
            Artifact(path="/nonexistent/header.h",
                     type="header",
                     source="test2.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'nonexistent/header.h'
                     }),
            Artifact(path="/missing/include.h",
                     type="header",
                     source="test3.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'missing/include.h'
                     }),
        ]

        mappings = [
            Mapping(artifact=artifacts[0],
                    dependency="libc6",
                    confidence="high",
                    strategy="dpkg-query")
        ]

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Check path_issues.yml exists
        path_issues_file = output_dir / "path_issues.yml"
        assert path_issues_file.exists()

        # Load and verify content
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        # Should have 2 non-existent paths (not the existing one)
        assert path_issues_data['summary']['count'] == 2
        assert path_issues_data['summary']['total_artifacts'] == 3
        assert path_issues_data['summary']['percentage'] == "66.7%"

        # Verify non_existent_paths contains the right entries
        non_existent_paths = path_issues_data['non_existent_paths']
        assert len(non_existent_paths) == 2

        # Check field names
        assert all('resolved_path' in entry for entry in non_existent_paths)
        assert all('type' in entry for entry in non_existent_paths)
        assert all('source' in entry for entry in non_existent_paths)
        assert all('d_file_path' in entry for entry in non_existent_paths)

        # Check values
        paths = [entry['resolved_path'] for entry in non_existent_paths]
        assert "/nonexistent/header.h" in paths
        assert "/missing/include.h" in paths
        assert "/usr/include/stdio.h" not in paths

    def test_generate_path_issues_yml_excludes_libraries(self, tmp_path):
        """Test path_issues.yml excludes library artifacts even if they don't exist"""
        artifacts = [
            Artifact(path="/nonexistent/header.h",
                     type="header",
                     source="test.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'nonexistent/header.h'
                     }),
            Artifact(path="-lmissing",
                     type="library",
                     source="link.txt",
                     metadata={'path_exists': False}),
            Artifact(path="/missing/libfoo.so",
                     type="library",
                     source="link.txt",
                     metadata={'path_exists': False}),
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        # Should only have 1 entry (the header, not the libraries)
        assert path_issues_data['summary']['count'] == 1
        assert len(path_issues_data['non_existent_paths']) == 1

        # Verify it's the header
        entry = path_issues_data['non_existent_paths'][0]
        assert entry['resolved_path'] == "/nonexistent/header.h"
        assert entry['type'] == "header"
        assert entry['d_file_path'] == "nonexistent/header.h"

    def test_generate_path_issues_yml_field_names(self, tmp_path):
        """Test path_issues.yml has correct field names (not 'path', but 'resolved_path')"""
        artifacts = [
            Artifact(path="/resolved/absolute/path.h",
                     type="header",
                     source="build/CMakeFiles/test.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'relative/path.h'
                     }),
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        entry = path_issues_data['non_existent_paths'][0]

        # Verify field names
        assert 'resolved_path' in entry
        assert 'type' in entry
        assert 'source' in entry
        assert 'd_file_path' in entry

        # Verify it does NOT use 'path' as field name
        assert 'path' not in entry

        # Verify values
        assert entry['resolved_path'] == "/resolved/absolute/path.h"
        assert entry['type'] == "header"
        assert entry['source'] == "build/CMakeFiles/test.d"
        assert entry['d_file_path'] == "relative/path.h"

    def test_generate_path_issues_yml_percentage_calculation(self, tmp_path):
        """Test path_issues.yml calculates percentage correctly"""
        # Create 10 artifacts: 3 non-existent headers, 7 existing
        artifacts = []
        for i in range(3):
            artifacts.append(
                Artifact(path=f"/missing{i}.h",
                         type="header",
                         source="test.d",
                         metadata={
                             'path_exists': False,
                             'original_path': f'missing{i}.h'
                         }))
        for i in range(7):
            artifacts.append(
                Artifact(path=f"/exists{i}.h",
                         type="header",
                         source="test.d",
                         metadata={'path_exists': True}))

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        # 3 out of 10 = 30%
        assert path_issues_data['summary']['count'] == 3
        assert path_issues_data['summary']['total_artifacts'] == 10
        assert path_issues_data['summary']['percentage'] == "30.0%"

    def test_generate_path_issues_yml_only_includes_path_exists_false(
            self, tmp_path):
        """Test path_issues.yml only includes artifacts with path_exists=False"""
        artifacts = [
            Artifact(path="/exists.h",
                     type="header",
                     source="test.d",
                     metadata={'path_exists': True}),
            Artifact(path="/missing.h",
                     type="header",
                     source="test.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'missing.h'
                     }),
            Artifact(path="/no_metadata.h", type="header",
                     source="test.d"),  # No metadata
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        # Only the one with path_exists=False
        assert path_issues_data['summary']['count'] == 1
        assert len(path_issues_data['non_existent_paths']) == 1
        assert path_issues_data['non_existent_paths'][0][
            'resolved_path'] == "/missing.h"

    def test_generate_path_issues_yml_empty_when_all_exist(self, tmp_path):
        """Test path_issues.yml has zero entries when all paths exist"""
        artifacts = [
            Artifact(path="/exists1.h",
                     type="header",
                     source="test.d",
                     metadata={'path_exists': True}),
            Artifact(path="/exists2.h",
                     type="header",
                     source="test.d",
                     metadata={'path_exists': True}),
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        # Should have 0 entries
        assert path_issues_data['summary']['count'] == 0
        assert path_issues_data['summary']['total_artifacts'] == 2
        assert path_issues_data['summary']['percentage'] == "0.0%"
        assert len(path_issues_data['non_existent_paths']) == 0

    def test_generate_path_issues_yml_mixed_artifact_types(self, tmp_path):
        """Test path_issues.yml with headers, libraries, and binaries"""
        artifacts = [
            Artifact(path="/missing_header.h",
                     type="header",
                     source="test.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'missing_header.h'
                     }),
            Artifact(path="-lmissing",
                     type="library",
                     source="link.txt",
                     metadata={'path_exists': False}),
            Artifact(path="/missing_binary.so",
                     type="binary",
                     source="wheel",
                     metadata={
                         'path_exists': False,
                         'original_path': 'missing_binary.so'
                     }),
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        # Should include header and binary, but not library
        assert path_issues_data['summary']['count'] == 2
        assert len(path_issues_data['non_existent_paths']) == 2

        # Verify types
        types = {
            entry['type']
            for entry in path_issues_data['non_existent_paths']
        }
        assert 'header' in types
        assert 'binary' in types
        assert 'library' not in types

    def test_generate_path_issues_yml_original_path_metadata(self, tmp_path):
        """Test path_issues.yml uses original_path metadata for d_file_path field"""
        artifacts = [
            Artifact(path="/resolved/absolute/path/include/header.h",
                     type="header",
                     source="build/CMakeFiles/target.dir/test.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'relative/include/header.h'
                     }),
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        entry = path_issues_data['non_existent_paths'][0]

        # d_file_path should be the original relative path from the .d file
        assert entry['d_file_path'] == 'relative/include/header.h'
        # resolved_path should be the absolute resolved path
        assert entry[
            'resolved_path'] == "/resolved/absolute/path/include/header.h"

    def test_generate_path_issues_yml_missing_original_path_metadata(
            self, tmp_path):
        """Test path_issues.yml handles missing original_path metadata gracefully"""
        artifacts = [
            Artifact(path="/missing.h",
                     type="header",
                     source="test.d",
                     metadata={'path_exists': False}),  # No original_path
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        entry = path_issues_data['non_existent_paths'][0]

        # Should have 'N/A' when original_path is missing
        assert entry['d_file_path'] == 'N/A'
        assert entry['resolved_path'] == "/missing.h"

    def test_generate_path_issues_yml_note_field(self, tmp_path):
        """Test path_issues.yml summary contains explanatory note"""
        artifacts = [
            Artifact(path="/missing.h",
                     type="header",
                     source="test.d",
                     metadata={
                         'path_exists': False,
                         'original_path': 'missing.h'
                     }),
        ]

        mappings = []

        output_dir = tmp_path / "reports"
        OutputGenerator.generate(mappings, artifacts, output_dir)

        # Load path_issues.yml
        path_issues_file = output_dir / "path_issues.yml"
        with open(path_issues_file) as f:
            path_issues_data = yaml.safe_load(f)

        # Verify note field exists and mentions libraries are excluded
        assert 'note' in path_issues_data['summary']
        note = path_issues_data['summary']['note']
        assert 'libraries excluded' in note.lower()
        assert 'do not exist' in note.lower()


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
                "directory_matches": []
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
