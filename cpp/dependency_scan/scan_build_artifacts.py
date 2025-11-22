#!/usr/bin/env python3
"""
Minimal Build Artifact Scanner for TensorRT-LLM

Scans D files (headers), link.txt files (libraries), and wheels (binaries)
to generate a comprehensive dependency mapping report.

Resolution Strategy:
  PRIMARY: dpkg-query for system packages
  FALLBACK: YAML patterns from dependencies/ directory

Output:
  - known.yml: Successfully mapped artifacts grouped by dependency (paths only)
  - unknown.yml: Unmapped artifacts needing pattern additions (paths only)

Usage:
  python scan_build_artifacts.py --build-dir build/ --output-dir scan_output/
  python scan_build_artifacts.py --validate  # Validate YAML files
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

try:
    from jsonschema import ValidationError, validate
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception  # Fallback for type hints

# Configuration: Submodules directory location
# This path points to the 3rdparty submodules directory.
# Change this constant if dependencies move to a different location.
# Current: TRTLLM_ROOT/3rdparty
# Future: May change to ${CMAKE_BINARY_DIR}/_deps/
THIRDPARTY_ROOT = Path(__file__).parent.parent.parent / '3rdparty'

# ============================================================================
# MODULE 1: Data Models
# ============================================================================


@dataclass
class Artifact:
    """Represents a discovered build artifact (header, library, or binary)"""
    path: str  # Canonical resolved path
    type: str  # 'header', 'library', 'binary'
    source: str  # Which file discovered it (D file, link.txt, wheel)
    context_dir: Optional[str] = None  # For relative path resolution
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class Mapping:
    """Represents an artifact-to-dependency mapping"""
    artifact: Artifact
    dependency: str  # Canonical dependency name
    confidence: str  # 'high', 'medium', 'low'
    strategy: str  # Which resolution strategy succeeded
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['artifact'] = self.artifact.to_dict()
        return result


# ============================================================================
# MODULE 2: DpkgResolver (PRIMARY)
# ============================================================================


class DpkgResolver:
    """
    Resolves artifacts to packages using dpkg-query (system package manager).

    This is the PRIMARY resolution strategy for system-installed packages
    (glibc, libstdc++, gcc, cuda-dev, etc.).

    Algorithm:
      1. For absolute paths: dpkg-query -S <path>
      2. For -l flags: find_library_path() → dpkg-query -S <resolved_path>
      3. Parse output: "package:arch: /path/to/file"
      4. Cache results to avoid repeated queries
      5. Normalize package names (remove :arch suffix, handle cuda packages)
    """

    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}
        self._lib_search_paths = self._get_library_search_paths()

    def _get_library_search_paths(self) -> List[str]:
        """
        Get standard library search paths for resolving -l flags.

        Returns system library directories in priority order:
          - /lib/x86_64-linux-gnu
          - /usr/lib/x86_64-linux-gnu
          - /lib
          - /usr/lib
          - /usr/local/lib
        """
        paths = [
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
            "/lib",
            "/usr/lib",
            "/usr/local/lib",
        ]

        # Add LD_LIBRARY_PATH if set
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_library_path:
            paths.extend(ld_library_path.split(":"))

        return [p for p in paths if os.path.isdir(p)]

    def find_library_path(self, lib_name: str) -> Optional[str]:
        """
        Resolve linker flag (-lpthread) to actual library path.

        Algorithm:
          1. Strip -l prefix: "-lpthread" → "pthread"
          2. Try patterns: libpthread.so, libpthread.so.*, libpthread.a
          3. Search in standard library directories
          4. Return first match or None

        Examples:
          -lpthread → /lib/x86_64-linux-gnu/libpthread.so.0
          -lm → /lib/x86_64-linux-gnu/libm.so.6
          -lstdc++ → /usr/lib/x86_64-linux-gnu/libstdc++.so.6
        """
        if lib_name.startswith("-l"):
            lib_name = lib_name[2:]  # Remove -l prefix

        # Try different library name patterns
        patterns = [
            f"lib{lib_name}.so",
            f"lib{lib_name}.so.*",
            f"lib{lib_name}.a",
        ]

        for search_path in self._lib_search_paths:
            for pattern in patterns:
                # Use glob to match version suffixes
                import glob
                matches = glob.glob(os.path.join(search_path, pattern))
                if matches:
                    # Return first match (highest priority)
                    return matches[0]

        return None

    def get_package(self, file_path: str) -> Optional[str]:
        """
        Query dpkg for package owning the file.

        Algorithm:
          1. Check cache for previous result
          2. Handle -l flags: find_library_path() first
          3. Execute: dpkg-query -S <file_path>
          4. Parse output: "package:arch: /path/to/file"
          5. Extract package name, remove architecture suffix
          6. Normalize CUDA packages: cuda-cccl-12-9 → cuda-cccl
          7. Cache result and return

        Examples:
          /usr/include/c++/13/vector → libstdc++-13-dev
          -lpthread → libc6
          /usr/local/cuda-12.9/include/cuda.h → cuda-cudart-dev-12-9 → cuda-cudart-dev
        """
        # Check cache first
        if file_path in self._cache:
            return self._cache[file_path]

        # Handle linker flags
        if file_path.startswith("-l"):
            resolved_path = self.find_library_path(file_path)
            if not resolved_path:
                self._cache[file_path] = None
                return None
            file_path = resolved_path

        # Query dpkg
        try:
            result = subprocess.run(["dpkg-query", "-S", file_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=5)

            if result.returncode != 0:
                self._cache[file_path] = None
                return None

            # Parse output: "package:arch: /path/to/file"
            output = result.stdout.strip()
            if ":" in output:
                package_part = output.split(":", 1)[0]
                # Remove architecture suffix (package:amd64 → package)
                package = package_part.split(
                    ":")[0] if ":" in package_part else package_part

                # Normalize CUDA packages
                package = self._normalize_cuda_package(package)

                self._cache[file_path] = package
                return package

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        self._cache[file_path] = None
        return None

    @staticmethod
    def _normalize_cuda_package(package: str) -> str:
        """
        Normalize CUDA package names by removing version suffixes.

        Examples:
          cuda-cccl-12-9 → cuda-cccl
          cuda-cudart-dev-12-9 → cuda-cudart-dev
          libcublas-dev-12-9 → libcublas-dev
          libc6 → libc6 (no change)
        """
        # Pattern: package-name-##-# → package-name
        match = re.match(r"^(.+?)-(\d+)-(\d+)$", package)
        if match:
            base_name = match.group(1)
            # Only normalize if it looks like a CUDA/NVIDIA package
            if any(x in base_name for x in [
                    "cuda", "cublas", "curand", "cusolver", "cusparse",
                    "nvjitlink", "nvinfer"
            ]):
                return base_name

        return package


# ============================================================================
# MODULE 3: ArtifactCollector
# ============================================================================


class ArtifactCollector:
    """
    Collects artifacts from D files (headers), link.txt files (libraries), and wheels (binaries).

    Args:
        build_dir: Path to build directory to scan
    """

    def __init__(self, build_dir: Path):
        self.build_dir = build_dir

    def collect_all(self) -> List[Artifact]:
        """
        Collect all artifacts from build directory.

        Algorithm:
          1. Find all *.d files → parse headers
          2. Find all link.txt files → parse libraries
          3. Find all *.whl files → extract and scan binaries
          4. Return combined deduplicated list
        """
        artifacts = []

        # Collect from D files
        d_files = list(self.build_dir.rglob("*.d"))
        for d_file in d_files:
            artifacts.extend(self._parse_d_file(d_file))

        # Collect from link files
        link_files = list(self.build_dir.rglob("link.txt"))
        for link_file in link_files:
            artifacts.extend(self._parse_link_file(link_file))

        # Collect from wheels
        wheel_files = list(self.build_dir.rglob("*.whl"))
        for wheel_file in wheel_files:
            artifacts.extend(self._scan_wheel(wheel_file))

        # Deduplicate by path
        seen = set()
        unique_artifacts = []
        for artifact in artifacts:
            if artifact.path not in seen:
                seen.add(artifact.path)
                unique_artifacts.append(artifact)

        return unique_artifacts

    def _parse_d_file(self, d_file: Path) -> List[Artifact]:
        """
        Parse CMake dependency file (.d) to extract header dependencies.

        Algorithm:
          1. Read file content
          2. Handle line continuations (backslash at end)
          3. Split by whitespace to get all paths
          4. Skip first token (target: header1 header2 ...)
          5. Strip trailing colons from paths (handles malformed .d files)
          6. Resolve relative paths from depfile's parent directory
          7. Filter out non-existent paths
          8. Canonicalize with os.path.realpath()

        Malformed .d File Handling:
          Some CMake-generated .d files contain paths with trailing colons.
          Example: '/usr/include/stdc-predef.h:' (should be '/usr/include/stdc-predef.h')

          This is caused by incorrect formatting in CMake's dependency tracking.
          The parser strips trailing colons using rstrip(':') to handle these cases,
          preventing duplicate artifacts and improving accuracy.

        Example D file:
          ```
          build/foo.o: /usr/include/stdio.h \
            ../include/myheader.h \
            /usr/local/cuda/include/cuda.h
          ```
        """

        artifacts = []

        try:
            content = d_file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return artifacts

        # Handle line continuations
        content = content.replace("\\\n", " ").replace("\\\r\n", " ")

        # Split by whitespace
        tokens = content.split()

        # Skip first token (target:)
        if not tokens or not tokens[0].endswith(":"):
            return artifacts

        header_paths = tokens[1:]
        # CMake .d files use paths relative to the target's build directory (where Makefile is)
        # This is the parent of the CMakeFiles directory
        d_file_parts = d_file.parts
        if 'CMakeFiles' in d_file_parts:
            cmake_idx = d_file_parts.index('CMakeFiles')
            context_dir = Path(*d_file_parts[:cmake_idx])
        else:
            # Fallback to build_dir if no CMakeFiles in path
            context_dir = self.build_dir

        for header_path in header_paths:
            # Strip trailing colons from paths (malformed .d files)
            # Some .d files have malformed entries like '/usr/include/stdc-predef.h:'
            header_path = header_path.rstrip(':')
            if not header_path:
                continue

            # Store original relative path before joining (for 3rdparty resolution)
            original_header_path = header_path

            # Resolve relative paths
            if not os.path.isabs(header_path):
                header_path = os.path.join(context_dir, header_path)

            # Canonicalize path
            try:
                canonical_path = os.path.realpath(header_path)

                # If path doesn't exist and contains submodules dir pattern, try resolving from submodules directory
                submodules_pattern = f'{THIRDPARTY_ROOT.name}/'
                if not os.path.exists(
                        canonical_path
                ) and submodules_pattern in original_header_path:
                    # Extract the part starting from FIRST submodules dir pattern in ORIGINAL path (handles nested dirs)
                    # e.g., ../../../../3rdparty/xgrammar/3rdparty/picojson/picojson.h
                    # should extract: xgrammar/3rdparty/picojson/picojson.h
                    idx = original_header_path.find(submodules_pattern)
                    if idx != -1:
                        relative_part = original_header_path[
                            idx + len(submodules_pattern):]
                        # Resolve from THIRDPARTY_ROOT constant
                        alternative_path = THIRDPARTY_ROOT / relative_part
                        alternative_canonical = os.path.realpath(
                            str(alternative_path))

                        if os.path.exists(alternative_canonical):
                            canonical_path = alternative_canonical

                # If path doesn't exist and contains '_deps/', try resolving from build root
                if not os.path.exists(
                        canonical_path) and '_deps/' in header_path:
                    # Extract the part starting from '_deps/'
                    parts = header_path.split('_deps/')
                    if len(parts) >= 2:
                        # Find build root (go up from cpp/dependency_scan to cpp/build)
                        build_root = self.build_dir
                        alternative_path = build_root / '_deps' / parts[-1]
                        alternative_canonical = os.path.realpath(
                            str(alternative_path))

                        if os.path.exists(alternative_canonical):
                            canonical_path = alternative_canonical

                # If path doesn't exist, try searching for it within build directory
                # This handles cases like nvshmem-build/ or other CMake ExternalProject paths
                if not os.path.exists(canonical_path) and not os.path.isabs(
                        header_path):
                    # Extract base filename to search for
                    basename = os.path.basename(header_path)
                    # Try to find the file within the build directory
                    import subprocess
                    try:
                        result = subprocess.run([
                            'find',
                            str(self.build_dir), '-name', basename, '-type', 'f'
                        ],
                                                capture_output=True,
                                                text=True,
                                                timeout=5)
                        if result.returncode == 0 and result.stdout.strip():
                            matches = result.stdout.strip().split('\n')
                            # Try to find match with similar relative path structure
                            for match in matches:
                                if header_path in match or match.endswith(
                                        header_path):
                                    canonical_path = os.path.realpath(match)
                                    break
                            # If no exact match, use first match
                            if not os.path.exists(canonical_path) and matches:
                                canonical_path = os.path.realpath(matches[0])
                    except Exception:
                        pass

                # Include all headers (even non-existent) for complete coverage
                artifacts.append(
                    Artifact(path=canonical_path,
                             type='header',
                             source=str(d_file),
                             context_dir=str(context_dir),
                             metadata={
                                 'original_path': original_header_path,
                                 'path_exists': os.path.exists(canonical_path)
                             }))
            except Exception:
                continue

        return artifacts

    def _parse_link_file(self, link_file: Path) -> List[Artifact]:
        """
        Parse CMake link.txt file to extract library dependencies.

        Algorithm:
          1. Read file content (single line linker command)
          2. Split by whitespace
          3. Extract:
             a) -l flags (e.g., -lpthread)
             b) Absolute library paths (*.a, *.so)
             c) @response.rsp files → recursively expand
             d) CMakeFiles linker artifacts with embedded -Wl flags (special handling)
          4. Deduplicate and return

        CMakeFiles Linker Artifact Extraction:
          CMake generates special linker artifacts in CMakeFiles directories that
          encode library dependencies in the path itself.

          Pattern: /path/CMakeFiles/foo.dir/-Wl,-soname,libtest.so.1

          These paths contain embedded linker flags (-Wl,-soname) that specify
          the library's soname. The parser extracts the library name (libtest.so.1),
          strips the 'lib' prefix, and converts it to a linker flag (-ltest).

          This enables proper dependency mapping for internal build artifacts that
          would otherwise be unmapped.

        Example link.txt:
          ```
          /usr/bin/c++ ... -lpthread -ldl /path/to/libfoo.a @response.rsp
          ```
        """

        artifacts = []

        try:
            content = link_file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return artifacts

        tokens = content.split()
        context_dir = link_file.parent

        for token in tokens:
            # Handle response files (@response.rsp)
            if token.startswith("@"):
                rsp_file = Path(context_dir) / token[1:]
                if rsp_file.exists():
                    artifacts.extend(self._parse_link_file(rsp_file))
                continue

            # Handle CMakeFiles linker artifacts with embedded -Wl flags
            # Pattern: /path/CMakeFiles/foo.dir/-Wl,-soname,libbar.so
            # These encode library dependencies in the path itself
            if '/CMakeFiles/' in token and '/-Wl,' in token:
                # Extract library name from -Wl,-soname,libfoo.so
                match = re.search(r'-Wl,-soname,(.+)$', token)
                if match:
                    lib_name = match.group(1)
                    # Add as linker flag artifact for pattern matching
                    artifacts.append(
                        Artifact(
                            path=
                            f"-l{lib_name.replace('lib', '').split('.')[0]}",
                            type='library',
                            source=str(link_file),
                            context_dir=str(context_dir),
                            metadata={
                                'linker_flag': True,
                                'cmake_linker_artifact': True,
                                'original_token': token,
                                'library_name': lib_name
                            }))
                continue

            # Handle -l flags
            if token.startswith("-l"):
                artifacts.append(
                    Artifact(path=token,
                             type='library',
                             source=str(link_file),
                             context_dir=str(context_dir),
                             metadata={'linker_flag': True}))
                continue

            # Handle absolute library paths
            if token.endswith((".a", ".so")) or ".so." in token:
                # Resolve relative paths
                if not os.path.isabs(token):
                    token = os.path.join(context_dir, token)

                try:
                    canonical_path = os.path.realpath(token)
                    # Include all library paths (even non-existent) for complete coverage
                    artifacts.append(
                        Artifact(path=canonical_path,
                                 type='library',
                                 source=str(link_file),
                                 context_dir=str(context_dir),
                                 metadata={
                                     'static': token.endswith('.a'),
                                     'path_exists':
                                     os.path.exists(canonical_path)
                                 }))
                except Exception:
                    continue

        return artifacts

    def _scan_wheel(self, wheel_file: Path) -> List[Artifact]:
        """
        Extract wheel and scan for binary dependencies (.so files).

        Algorithm:
          1. Create temp directory
          2. Extract wheel (ZIP format)
          3. Find all *.so files
          4. For each .so:
             a) Run readelf -d to get NEEDED entries
             b) Extract required library names
          5. Cleanup temp directory
          6. Return binary artifacts with NEEDED metadata

        Example:
          tensorrt_llm-0.1.0-py3-none-any.whl contains:
            - tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so
            - Uses: libcudart.so.12, libnvinfer.so.10, libstdc++.so.6
        """
        artifacts = []

        # Create temp directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract wheel
                with zipfile.ZipFile(wheel_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find all .so files
                temp_path = Path(temp_dir)
                so_files = list(temp_path.rglob("*.so")) + list(
                    temp_path.rglob("*.so.*"))

                for so_file in so_files:
                    # Get NEEDED entries with readelf
                    needed_libs = self._get_needed_libraries(so_file)

                    # Create artifact for the .so file itself
                    artifacts.append(
                        Artifact(path=str(so_file.relative_to(temp_path)),
                                 type='binary',
                                 source=str(wheel_file),
                                 metadata={
                                     'wheel': wheel_file.name,
                                     'needed': needed_libs
                                 }))

                    # Create artifacts for NEEDED libraries
                    for needed_lib in needed_libs:
                        artifacts.append(
                            Artifact(path=needed_lib,
                                     type='library',
                                     source=str(wheel_file),
                                     metadata={
                                         'from_binary':
                                         str(so_file.relative_to(temp_path)),
                                         'dynamic_dependency':
                                         True
                                     }))

            except Exception:
                pass

        return artifacts

    @staticmethod
    def _get_needed_libraries(binary_path: Path) -> List[str]:
        """
        Extract NEEDED entries from ELF binary using readelf.

        Algorithm:
          1. Execute: readelf -d <binary>
          2. Parse output for lines containing "(NEEDED)"
          3. Extract library names from "Shared library: [libfoo.so]"
          4. Return list of library names

        Example output:
          ```
           0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.12]
           0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
          ```
        """
        needed = []

        try:
            result = subprocess.run(
                ["readelf", "-d", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=10)

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "(NEEDED)" in line and "Shared library:" in line:
                        # Extract library name between brackets
                        match = re.search(r'\[([^\]]+)\]', line)
                        if match:
                            needed.append(match.group(1))

        except Exception:
            pass

        return needed


# ============================================================================
# MODULE 4: PatternMatcher (FALLBACK)
# ============================================================================


class PatternMatcher:
    """
    Resolves artifacts using YAML files from dependencies/ directory (FALLBACK strategy for non-dpkg packages).

    Provides 3-tier resolution strategy:
      1. Exact pattern matching (basename_matches and linker_flags_matches)
      2. Path matching (directory_matches - rightmost match wins)
      3. Generic library name inference (fallback)

    YAML files are loaded from dependencies/ directory:
      - Individual dependency files (e.g., tensorrt-llm.yml)
      - Files with dependencies: list format (e.g., base.yml, cuda.yml)
      - All *.yml files except those starting with '_'
    """

    def __init__(self, metadata_dir: Path):
        """
        Initialize PatternMatcher by loading YAML files from dependencies/ directory.

        Args:
            metadata_dir: Path to directory containing YAML dependency files
        """
        self.pattern_mappings: Dict[str, str] = {}
        self.path_aliases: Dict[str, str] = {}
        self.known_names: Set[str] = set()  # Track all known dependency names
        self._schema = None
        self._duplicate_warnings: Set[str] = set()

        # Vendor directory magic strings (industry-standard patterns)
        self.vendor_patterns = [
            '3rdparty/', 'third-party/', 'thirdparty/', 'third_party/',
            'external/', 'externals/', 'vendor/', 'vendored/', 'deps/'
        ]

        # Load schema if available
        schema_file = metadata_dir / "_schema.yml"
        if schema_file.exists() and JSONSCHEMA_AVAILABLE:
            with open(schema_file, 'r') as f:
                self._schema = yaml.safe_load(f)

        # Load all YAML files
        self._load_yaml_files(metadata_dir)

    def _load_yaml_files(self, metadata_dir: Path):
        """
        Load all YAML files from dependencies/ directory.

        Algorithm:
          1. Find all *.yml files (except those starting with '_')
          2. Load each file and validate against schema
          3. Handle two formats:
             - Individual dependency files (with name, basename_matches, etc.)
             - Files with dependencies: list format (e.g., base.yml, cuda.yml)
          4. Merge all basename_matches/linker_flags_matches into pattern_mappings
          5. Merge all directory_matches into path_aliases
          6. Warn about validation errors and duplicates
        """
        yaml_files = sorted([
            f for f in metadata_dir.glob("*.yml") if not f.name.startswith("_")
        ])

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)

                # Handle files with dependencies list (e.g., dpkg.yml, cuda.yml)
                if "dependencies" in data and isinstance(
                        data["dependencies"], list):
                    for dep_data in data["dependencies"]:
                        self._process_dependency(dep_data, yaml_file)
                # Handle individual dependency files
                elif "name" in data:
                    self._process_dependency(data, yaml_file)
                else:
                    print(
                        f"Warning: Skipping {yaml_file.name} - unrecognized format",
                        file=sys.stderr)

            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse {yaml_file.name}: {e}",
                      file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error loading {yaml_file.name}: {e}",
                      file=sys.stderr)

    def _process_dependency(self, dep_data: Dict[str, Any], source_file: Path):
        """
        Process a single dependency definition and merge into internal structures.

        Args:
            dep_data: Dictionary containing dependency definition
            source_file: Path to YAML file being processed (for error messages)
        """
        # Validate against schema if available
        if self._schema and JSONSCHEMA_AVAILABLE:
            try:
                validate(instance=dep_data, schema=self._schema)
            except ValidationError as e:
                print(
                    f"Warning: Validation error in {source_file.name}: {e.message}",
                    file=sys.stderr)
                # Continue processing despite validation errors

        dependency_name = dep_data.get("name")
        if not dependency_name:
            print(f"Warning: Missing 'name' field in {source_file.name}",
                  file=sys.stderr)
            return

        # Merge basename_matches into pattern_mappings
        basename_matches = dep_data.get("basename_matches", [])
        for pattern in basename_matches:
            if pattern in self.pattern_mappings and pattern not in self._duplicate_warnings:
                print(
                    f"Warning: Duplicate basename match '{pattern}' found in {source_file.name} "
                    f"(previously mapped to '{self.pattern_mappings[pattern]}', now '{dependency_name}')",
                    file=sys.stderr)
                self._duplicate_warnings.add(pattern)
            self.pattern_mappings[pattern] = dependency_name

        # Merge linker_flags_matches into pattern_mappings
        linker_flags_matches = dep_data.get("linker_flags_matches", [])
        for flag in linker_flags_matches:
            if flag in self.pattern_mappings and flag not in self._duplicate_warnings:
                print(
                    f"Warning: Duplicate linker flag '{flag}' found in {source_file.name} "
                    f"(previously mapped to '{self.pattern_mappings[flag]}', now '{dependency_name}')",
                    file=sys.stderr)
                self._duplicate_warnings.add(flag)
            self.pattern_mappings[flag] = dependency_name

        # Merge directory_matches into path_aliases
        directory_matches = dep_data.get("directory_matches", [])
        for component in directory_matches:
            if component in self.path_aliases and component not in self._duplicate_warnings:
                print(
                    f"Warning: Duplicate path component '{component}' found in {source_file.name} "
                    f"(previously mapped to '{self.path_aliases[component]}', now '{dependency_name}')",
                    file=sys.stderr)
                self._duplicate_warnings.add(component)
            self.path_aliases[component] = dependency_name

        # Track known dependency names (for nested vendor detection)
        self.known_names.add(dependency_name.lower())
        for component in directory_matches:
            self.known_names.add(component.lower())

    def match(self, artifact: Artifact) -> Optional[Mapping]:
        """
        Match artifact using 3-tier strategy.

        Algorithm:
          1. Try pattern matching (basename_matches and linker_flags_matches - exact match only - highest confidence)
          2. Try path matching (directory_matches - rightmost directory wins)
          3. Try generic library name inference (lowest confidence)
          4. Return first match or None
        """
        # Strategy 1: Pattern matching (exact match only)
        result = self._match_patterns(artifact)
        if result:
            return result

        # Strategy 2: Path matching (directory_matches)
        result = self._match_path_alias(artifact)
        if result:
            return result

        # Strategy 3: Generic library name inference (fallback)
        result = self._match_generic_library(artifact)
        if result:
            return result

        return None

    def _match_patterns(self, artifact: Artifact) -> Optional[Mapping]:
        """
        Match using pattern_mappings dictionary (exact match only).

        Only performs exact matching against basename_matches from YAML files.
        Substring matching has been removed to prevent false positives.

        Algorithm:
          1. Try exact match on basename (e.g., "libcudart.so.12")
          2. Try exact match on full path (e.g., "-lpthread")
          3. Return mapped dependency with HIGH confidence

        Examples:
          -lpthread → libc6 (exact basename match)
          libcudart.so.12 → cuda-cudart-12 (exact basename match)

        Note: For partial path matching, use directory_matches in YAML files.
              Directory matches work on whole directory names (e.g., "fmt/" in path).
        """
        basename = os.path.basename(artifact.path)

        # Try exact match on basename
        if basename in self.pattern_mappings:
            return Mapping(artifact=artifact,
                           dependency=self.pattern_mappings[basename],
                           confidence='high',
                           strategy='exact_pattern_match',
                           metadata={'matched_key': basename})

        # Try exact match on full path (for -l flags)
        if artifact.path in self.pattern_mappings:
            return Mapping(artifact=artifact,
                           dependency=self.pattern_mappings[artifact.path],
                           confidence='high',
                           strategy='exact_pattern_match',
                           metadata={'matched_key': artifact.path})

        # Substring matching removed - too high risk for false positives
        # Use directory_matches instead for safe partial path matching

        return None

    def _match_path_alias(self, artifact: Artifact) -> Optional[Mapping]:
        """
        Match using path_aliases (supports single and multi-directory patterns).

        Algorithm:
          1. Split path by '/' to get directory components (filter empty strings)
          2. For each pattern in path_aliases:
             a. Normalize pattern (strip leading/trailing slashes)
             b. Split pattern into components
             c. Search for exact consecutive component sequence in path
             d. Rightmost match wins
          3. Return first match with MEDIUM confidence

        Examples:
          Single component:
            Pattern: "pytorch"
            /foo/bar/pytorch/include/torch/torch.h → pytorch (matches "pytorch")

          Multi-directory:
            Pattern: "foo/bar"
            /home/foo/bar/file.h → foo/bar (matches consecutive "foo", "bar")
            /home/foobar/file.h → NO MATCH ("foobar" != ["foo", "bar"])

          Rightmost wins:
            Pattern: "foo/bar"
            /foo/bar/baz/foo/bar/qux.h → matches at rightmost position
        """
        # Split path into components (filter out empty strings from leading slash)
        path_components = [c for c in artifact.path.split('/') if c]

        best_match = None
        best_position = -1  # Rightmost wins (highest position)

        for pattern, dependency in self.path_aliases.items():
            # Normalize pattern: strip slashes, split into components
            pattern_normalized = pattern.strip('/')

            # Handle empty pattern after stripping
            if not pattern_normalized:
                continue

            pattern_components = pattern_normalized.split('/')
            pattern_len = len(pattern_components)

            # Search for exact consecutive sequence match using sliding window
            # Iterate through all possible positions in the path
            for i in range(len(path_components) - pattern_len + 1):
                # Check if pattern_components match exactly at position i
                if path_components[i:i + pattern_len] == pattern_components:
                    # Found exact match at position i
                    # Keep rightmost match (highest i value wins)
                    if i > best_position:
                        best_position = i
                        best_match = Mapping(artifact=artifact,
                                             dependency=dependency,
                                             confidence='medium',
                                             strategy='path_alias',
                                             metadata={
                                                 'matched_pattern':
                                                 pattern,
                                                 'matched_sequence':
                                                 '/'.join(pattern_components),
                                                 'position':
                                                 i
                                             })

        return best_match

    def _match_generic_library(self, artifact: Artifact) -> Optional[Mapping]:
        """
        Generic library name inference (FALLBACK with LOW confidence).

        Algorithm:
          1. Check if library type
          2. Extract basename
          3. Strip lib prefix and .so/.a suffix
          4. Return as dependency with LOW confidence

        Examples:
          libfoobar.so → foobar (low confidence)
          libtest.so.1 → test (low confidence)
        """
        if artifact.type != 'library':
            return None

        basename = os.path.basename(artifact.path)

        # Try to extract library name
        match = re.match(r'^lib([a-zA-Z0-9_-]+)\.(?:so|a)(?:\.\d+)*$', basename)
        if match:
            return Mapping(artifact=artifact,
                           dependency=match.group(1),
                           confidence='low',
                           strategy='generic_library_inference',
                           metadata={'inferred_from': basename})

        return None

    def extract_vendor_components(self, path: str) -> List[tuple]:
        """
        Extract all vendor components from a path using magic strings.

        Args:
            path: Artifact path to scan

        Returns:
            List of (pattern, component) tuples for each vendor boundary found

        Example:
            "/3rdparty/xgrammar/3rdparty/picojson/file.h" →
            [("3rdparty/", "xgrammar"), ("3rdparty/", "picojson")]
        """
        components = []
        path_lower = path.lower()

        for pattern in self.vendor_patterns:
            idx = 0
            while True:
                idx = path_lower.find(pattern, idx)
                if idx == -1:
                    break

                # Extract component name after the pattern
                start = idx + len(pattern)
                end = path_lower.find('/', start)
                if end == -1:
                    end = len(path_lower)

                component = path[start:end]  # Use original case
                if component:  # Skip empty components
                    components.append((pattern, component))

                idx = end

        return components

    def find_unknown_vendor_boundaries(self,
                                       artifact: Artifact) -> Optional[str]:
        """
        Check if artifact contains any unknown vendor boundaries.

        Unified vendor boundary policy: ANY component following a vendor pattern
        (3rdparty/, vendor/, external/, etc.) MUST be in the known allowlist.

        Returns:
            Name of unknown vendor boundary component, or None if all are known

        Examples:
            Path: "/3rdparty/xgrammar/src/file.h"
            Components: ["xgrammar"]
            If "xgrammar" is known → returns None (OK)

            Path: "/3rdparty/unknown-lib/file.h"
            Components: ["unknown-lib"]
            If "unknown-lib" is NOT known → returns "unknown-lib" (REJECT)

            Path: "/3rdparty/xgrammar/3rdparty/picojson/file.h"
            Components: ["xgrammar", "picojson"]
            If "picojson" is NOT known → returns "picojson" (REJECT)
        """
        components = self.extract_vendor_components(artifact.path)

        # Check ALL vendor boundaries (rightmost has priority for detection)
        for pattern, component in reversed(components):
            component_lower = component.lower()
            if component_lower not in self.known_names:
                return component

        return None


# ============================================================================
# MODULE 5: OutputGenerator
# ============================================================================


class OutputGenerator:
    """
    Generates YAML reports for known and unknown artifacts.

    Output files:
      - known.yml: Successfully mapped artifacts grouped by dependency (paths only)
      - unknown.yml: Unmapped artifacts requiring pattern additions (paths only)
    """

    @staticmethod
    def generate(mappings: List[Mapping], artifacts: List[Artifact],
                 output_dir: Path):
        """
        Generate known.yml and unknown.yml with simplified structure (paths only).

        Algorithm:
          1. Create output directory if needed
          2. Separate mapped vs unmapped artifacts
          3. Group known artifacts by dependency (dict of lists)
          4. Sort dependencies by count (most artifacts first)
          5. Write YAML files with simplified structure
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Separate known vs unknown mappings
        # Artifacts mapped to "unknown" should be treated as truly unknown
        known_mappings = [m for m in mappings if m.dependency != 'unknown']
        unknown_mappings = [m for m in mappings if m.dependency == 'unknown']

        # Build mapping lookup (only for truly known)
        mapped_paths = {m.artifact.path for m in known_mappings}

        # Known artifacts - simplified structure (dependency -> list of paths)
        known = {}
        for mapping in known_mappings:
            dep = mapping.dependency
            if dep not in known:
                known[dep] = []
            known[dep].append(mapping.artifact.path)

        # Sort dependencies by count (most artifacts first)
        known_sorted = dict(
            sorted(known.items(), key=lambda x: len(x[1]), reverse=True))

        # Unknown artifacts - simplified structure (flat list of paths)
        unknown_paths = []

        # Add artifacts that weren't mapped at all
        for artifact in artifacts:
            if artifact.path not in mapped_paths and not any(
                    m.artifact.path == artifact.path for m in unknown_mappings):
                unknown_paths.append(artifact.path)

        # Add artifacts mapped to "unknown"
        for mapping in unknown_mappings:
            unknown_paths.append(mapping.artifact.path)

        # Write outputs
        known_file = output_dir / 'known.yml'
        unknown_file = output_dir / 'unknown.yml'

        with open(known_file, 'w') as f:
            yaml.dump(
                {
                    'summary': {
                        'total_artifacts':
                        len(artifacts),
                        'mapped':
                        len(known_mappings),
                        'unmapped':
                        len(unknown_paths),
                        'coverage':
                        f"{len(known_mappings) / len(artifacts) * 100:.1f}%"
                        if artifacts else "0%",
                        'unique_dependencies':
                        len(known)
                    },
                    'dependencies': known_sorted
                },
                f,
                default_flow_style=False,
                sort_keys=False)

        with open(unknown_file, 'w') as f:
            yaml.dump(
                {
                    'summary': {
                        'count':
                        len(unknown_paths),
                        'action_required':
                        'Add patterns to YAML files in dependencies/ for these artifacts'
                    },
                    'artifacts': unknown_paths
                },
                f,
                default_flow_style=False,
                sort_keys=False)

        # Generate path_issues.yml for non-existent paths
        path_issues_file = output_dir / 'path_issues.yml'
        non_existent_paths = []

        for artifact in artifacts:
            # Check if path_exists metadata is False
            # Exclude libraries since they don't have meaningful original_path metadata
            if (artifact.metadata
                    and not artifact.metadata.get('path_exists', True)
                    and artifact.type != 'library'):
                non_existent_paths.append({
                    'resolved_path':
                    artifact.path,
                    'type':
                    artifact.type,
                    'source':
                    artifact.source,
                    'd_file_path':
                    artifact.metadata.get('original_path', 'N/A')
                })

        with open(path_issues_file, 'w') as f:
            yaml.dump(
                {
                    'summary': {
                        'count':
                        len(non_existent_paths),
                        'total_artifacts':
                        len(artifacts),
                        'percentage':
                        f"{len(non_existent_paths) / len(artifacts) * 100:.1f}%"
                        if artifacts else "0%",
                        'note':
                        'These header paths were resolved from .d files but do not exist in the filesystem (libraries excluded)'
                    },
                    'non_existent_paths': non_existent_paths
                },
                f,
                default_flow_style=False,
                sort_keys=False)

        return known_file, unknown_file


# ============================================================================
# MODULE 6: Main Orchestration
# ============================================================================


def validate_yaml_files(metadata_dir: Path) -> bool:
    """
    Validate YAML files without running the scanner.

    Args:
        metadata_dir: Path to dependencies directory

    Returns:
        True if all files are valid, False otherwise
    """
    print("=" * 80)
    print("YAML Validation")
    print("=" * 80)
    print(f"Metadata directory: {metadata_dir}")
    print()

    # Check if jsonschema is available
    if not JSONSCHEMA_AVAILABLE:
        print("Warning: jsonschema not installed, skipping validation",
              file=sys.stderr)
        print("Install with: pip install jsonschema")
        return False

    # Load schema
    schema_file = metadata_dir / "_schema.yml"
    if not schema_file.exists():
        print(f"Error: Schema file not found: {schema_file}", file=sys.stderr)
        return False

    with open(schema_file, 'r') as f:
        schema = yaml.safe_load(f)

    # Validate all YAML files
    yaml_files = sorted(
        [f for f in metadata_dir.glob("*.yml") if not f.name.startswith("_")])
    total = 0
    valid = 0
    invalid = 0

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            # Handle files with dependencies list format (e.g., base.yml, cuda.yml)
            if "dependencies" in data and isinstance(data["dependencies"],
                                                     list):
                for dep_data in data["dependencies"]:
                    total += 1
                    try:
                        validate(instance=dep_data, schema=schema)
                        print(
                            f"✓ {yaml_file.name}:{dep_data.get('name', 'unknown')}"
                        )
                        valid += 1
                    except ValidationError as e:
                        print(
                            f"✗ {yaml_file.name}:{dep_data.get('name', 'unknown')}: {e.message}",
                            file=sys.stderr)
                        invalid += 1
            # Handle individual dependency files
            elif "name" in data:
                total += 1
                try:
                    validate(instance=data, schema=schema)
                    print(f"✓ {yaml_file.name}")
                    valid += 1
                except ValidationError as e:
                    print(f"✗ {yaml_file.name}: {e.message}", file=sys.stderr)
                    invalid += 1

        except yaml.YAMLError as e:
            print(f"✗ {yaml_file.name}: YAML parse error: {e}", file=sys.stderr)
            invalid += 1
        except Exception as e:
            print(f"✗ {yaml_file.name}: {e}", file=sys.stderr)
            invalid += 1

    print()
    print("=" * 80)
    print(f"Results: {valid}/{total} valid, {invalid}/{total} invalid")
    print("=" * 80)

    return invalid == 0


def main():
    """
    Main entry point for build artifact scanner.

    Algorithm:
      1. Parse command-line arguments
      2. Validate inputs (build-dir exists, dependencies/ exists)
      3. Collect artifacts using ArtifactCollector
      4. Resolve using DpkgResolver (PRIMARY)
      5. Resolve remaining using PatternMatcher (FALLBACK)
      6. Generate reports using OutputGenerator
      7. Print summary statistics
    """
    parser = argparse.ArgumentParser(
        description='Minimal Build Artifact Scanner for TensorRT-LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan default build directory
  python scan_build_artifacts.py

  # Scan custom build directory with custom output
  python scan_build_artifacts.py --build-dir build/Release --output-dir scan_output/

  # Validate YAML files without scanning
  python scan_build_artifacts.py --validate

  # Use custom dependencies directory
  python scan_build_artifacts.py --dependencies-dir custom_dependencies/
        """)

    parser.add_argument(
        '--build-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'build',
        help=
        'Build directory to scan for C++ artifacts (default: ../build/). Note: wheels are in ../../build/'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('scan_output'),
        help='Output directory for reports (default: scan_output/)')

    parser.add_argument(
        '--metadata-dir',
        type=Path,
        default=Path(__file__).parent / 'metadata',
        help=
        'Path to metadata directory containing YAML files (default: ./metadata/)'
    )

    parser.add_argument('--validate',
                        action='store_true',
                        help='Validate YAML files without running scanner')

    args = parser.parse_args()

    # Handle --validate flag
    if args.validate:
        success = validate_yaml_files(args.metadata_dir)
        sys.exit(0 if success else 1)

    # Validate inputs
    if not args.build_dir.exists():
        print(f"Error: Build directory not found: {args.build_dir}",
              file=sys.stderr)
        sys.exit(1)

    if not args.metadata_dir.exists():
        print(f"Error: Metadata directory not found: {args.metadata_dir}",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("TensorRT-LLM Build Artifact Scanner")
    print("=" * 80)
    print(f"Build directory: {args.build_dir}")
    print(f"Metadata directory: {args.metadata_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Step 1: Collect artifacts
    print("[1/4] Collecting artifacts...")
    collector = ArtifactCollector(args.build_dir)
    artifacts = collector.collect_all()
    print(f"  Found {len(artifacts)} unique artifacts")
    print(f"    - Headers: {sum(1 for a in artifacts if a.type == 'header')}")
    print(
        f"    - Libraries: {sum(1 for a in artifacts if a.type == 'library')}")
    print(f"    - Binaries: {sum(1 for a in artifacts if a.type == 'binary')}")
    print()

    # Step 2: Resolve with dpkg (PRIMARY)
    print("[2/4] Resolving with dpkg-query (PRIMARY strategy)...")
    dpkg_resolver = DpkgResolver()
    dpkg_mappings = []

    for artifact in artifacts:
        package = dpkg_resolver.get_package(artifact.path)
        if package:
            dpkg_mappings.append(
                Mapping(artifact=artifact,
                        dependency=package,
                        confidence='high',
                        strategy='dpkg-query',
                        metadata={'dpkg_package': package}))

    print(
        f"  Resolved {len(dpkg_mappings)} artifacts via dpkg ({len(dpkg_mappings) / len(artifacts) * 100:.1f}%)"
    )
    print()

    # Step 3: Resolve remaining with YAML patterns (FALLBACK)
    print("[3/4] Resolving remaining with YAML patterns (FALLBACK strategy)...")
    pattern_matcher = PatternMatcher(args.metadata_dir)
    pattern_mappings = []

    dpkg_resolved_paths = {m.artifact.path for m in dpkg_mappings}
    remaining_artifacts = [
        a for a in artifacts if a.path not in dpkg_resolved_paths
    ]

    for artifact in remaining_artifacts:
        mapping = pattern_matcher.match(artifact)
        if mapping:
            # Check for unknown vendor boundaries BEFORE accepting the mapping
            unknown_vendor = pattern_matcher.find_unknown_vendor_boundaries(
                artifact)
            if unknown_vendor:
                # Artifact has unknown vendor boundary - treat as unknown
                # Don't add to pattern_mappings (will be in unknown.yml)
                print(
                    f"  WARNING: Unknown vendor boundary '{unknown_vendor}' found in: {artifact.path}",
                    file=sys.stderr)
            else:
                pattern_mappings.append(mapping)

    print(
        f"  Resolved {len(pattern_mappings)} additional artifacts via patterns ({len(pattern_mappings) / len(artifacts) * 100:.1f}%)"
    )
    print()

    # Step 4: Generate reports
    print("[4/4] Generating reports...")
    all_mappings = dpkg_mappings + pattern_mappings
    known_file, unknown_file = OutputGenerator.generate(all_mappings, artifacts,
                                                        args.output_dir)

    print(f"  Reports written to:")
    print(f"    - {known_file}")
    print(f"    - {unknown_file}")
    print()

    # Summary
    # Separate known vs unknown (artifacts mapped to "unknown" are treated as unknown)
    known_mappings = [m for m in all_mappings if m.dependency != 'unknown']
    unknown_mappings = [m for m in all_mappings if m.dependency == 'unknown']

    total_known = len(known_mappings)
    total_unknown = len(artifacts) - total_known
    coverage = (total_known / len(artifacts) * 100) if artifacts else 0

    # Count dpkg/pattern strategies among known mappings only
    dpkg_known = sum(1 for m in dpkg_mappings if m.dependency != 'unknown')
    pattern_known = sum(1 for m in pattern_mappings
                        if m.dependency != 'unknown')

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total artifacts: {len(artifacts)}")
    print(
        f"  Mapped (dpkg): {dpkg_known} ({dpkg_known / len(artifacts) * 100:.1f}%)"
    )
    print(
        f"  Mapped (patterns): {pattern_known} ({pattern_known / len(artifacts) * 100:.1f}%)"
    )
    print(
        f"  Unknown: {total_unknown} ({total_unknown / len(artifacts) * 100:.1f}%)"
    )
    print(f"Coverage: {coverage:.1f}%")
    print()

    # Confidence breakdown (only for known mappings)
    high_conf = sum(1 for m in known_mappings if m.confidence == 'high')
    med_conf = sum(1 for m in known_mappings if m.confidence == 'medium')
    low_conf = sum(1 for m in known_mappings if m.confidence == 'low')

    if known_mappings:
        print("Confidence Distribution:")
        print(
            f"  High: {high_conf} ({high_conf / len(known_mappings) * 100:.1f}%)"
        )
        print(
            f"  Medium: {med_conf} ({med_conf / len(known_mappings) * 100:.1f}%)"
        )
        print(
            f"  Low: {low_conf} ({low_conf / len(known_mappings) * 100:.1f}%)")
    else:
        print("Confidence Distribution:")
        print("  High: 0")
        print("  Medium: 0")
        print("  Low: 0")
    print()

    if total_unknown > 0:
        print(f"ACTION REQUIRED: {total_unknown} artifacts unknown")
        if len(unknown_mappings) > 0:
            print(
                f"  {len(unknown_mappings)} artifacts matched generic fallback (need specific patterns)"
            )
        print(f"  Review {unknown_file}")
        print(f"  Add missing patterns to YAML files in {args.metadata_dir}")
        print(f"  Re-run scanner to improve coverage")
    else:
        print("SUCCESS: All artifacts mapped!")

    print("=" * 80)


if __name__ == '__main__':
    main()
