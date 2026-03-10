import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml
from _constants import COPYRIGHT_PATTERNS, LICENSE_PATTERNS
from _package_resolvers import _find_matching_files

try:
    from jsonschema import ValidationError, validate

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception  # type: ignore


VENDOR_MARKERS = [
    "3rdparty/",
    "third-party/",
    "thirdparty/",
    "third_party/",
    "external/",
    "externals/",
    "vendor/",
    "vendored/",
    "deps/",
    "ext/",
]
# _deps is not in VENDOR_MARKERS because it would interfere with vendor
# inference in match(). It is checked separately in has_unknown_vendor().


@dataclass
class DependencyMapping:
    """Maps a file to a dependency."""

    file_path: str
    dependency: str
    version: str = ""
    confidence: str = "high"
    strategy: str = ""
    raw_package: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatternMatcher:
    """Resolves files using YAML patterns and FetchContent data."""

    def __init__(self, metadata_dir: Path, fetch_content_json: Optional[Path] = None):
        self.patterns: Dict[str, str] = {}  # basename/flag -> dependency
        self.path_aliases: Dict[str, str] = {}  # path pattern -> dependency
        self.absolute_roots: Dict[
            str, str
        ] = {}  # dependency -> absolute path (from absolute directory_matches)
        self.known_names: Set[str] = set()
        self.fetch_content: Dict[str, Dict[str, Any]] = {}
        self._version_cache: Dict[str, str] = {}
        self._warnings: Set[str] = set()

        if (metadata_dir / "_schema.yml").exists() and JSONSCHEMA_AVAILABLE:
            with open(metadata_dir / "_schema.yml") as f:
                self._schema = yaml.safe_load(f)
        else:
            self._schema = None

        self._load_yaml(metadata_dir)
        self._load_fetch_content(
            fetch_content_json or Path(__file__).parents[3] / "3rdparty/fetch_content.json"
        )
        self._add_project_root_pattern()

    def _load_yaml(self, metadata_dir: Path):
        for yaml_file in sorted(metadata_dir.glob("*.yml")):
            if yaml_file.name.startswith("_"):
                continue
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                deps = data.get("dependencies", [data] if "name" in data else [])
                for dep in deps:
                    self._add_dependency(dep, yaml_file)
            except Exception as e:
                print(f"Warning: Error loading {yaml_file.name}: {e}", file=sys.stderr)

    def _add_dependency(self, dep: Dict, source: Path):
        if self._schema and JSONSCHEMA_AVAILABLE:
            try:
                validate(instance=dep, schema=self._schema)
            except ValidationError as e:
                print(f"Warning: Validation error in {source.name}: {e.message}", file=sys.stderr)

        name = dep.get("name")
        if not name:
            return

        for key in dep.get("basename_matches", []):
            if key in self.patterns and key not in self._warnings:
                print(f"Warning: Duplicate pattern '{key}' in {source.name}", file=sys.stderr)
                self._warnings.add(key)
            self.patterns[key] = name

        for comp in dep.get("directory_matches", []):
            if comp in self.path_aliases and comp not in self._warnings:
                print(f"Warning: Duplicate path '{comp}' in {source.name}", file=sys.stderr)
                self._warnings.add(comp)
            self.path_aliases[comp] = name
            self.known_names.add(comp.lower())
            # Absolute paths are used directly as source roots
            if comp.startswith("/"):
                self.absolute_roots[name] = comp

        self.known_names.add(name.lower())

    def _load_fetch_content(self, path: Path):
        try:
            with open(path) as f:
                data = json.load(f)
            for dep in data.get("dependencies", []):
                name = dep.get("name")
                if not name:
                    continue
                display = dep.get("display_name", name)
                self.path_aliases[f"_deps/{name}-src"] = display
                self.known_names.update(
                    [
                        display.lower(),
                        f"{name}-src".lower(),
                        f"{name}-build".lower(),
                        f"{name}-subbuild".lower(),
                    ]
                )
                self.fetch_content[display] = dep
        except Exception as e:
            print(f"Warning: Error loading fetch_content.json: {e}", file=sys.stderr)

    def _add_project_root_pattern(self):
        """Add the project root directory as a path alias for tensorrt-llm.

        This automatically maps files under the project directory to tensorrt-llm,
        which is in IGNORED_DEPS and will be filtered from attribution output.
        """
        # Get project root (4 levels up: scan -> attribution -> scripts -> root)
        project_root = Path(__file__).resolve().parents[3]
        root_name = project_root.name
        self.path_aliases[f"{root_name}/cpp"] = "tensorrt-llm"
        self.known_names.add(root_name.lower())

    def match(self, file_path: str) -> Optional[DependencyMapping]:
        """Match file to dependency using patterns, path aliases, or library inference."""
        basename = os.path.basename(file_path)

        # Exact pattern match
        for key in [basename, file_path]:
            if key in self.patterns:
                dep = self.patterns[key]
                return DependencyMapping(
                    file_path=file_path,
                    dependency=dep,
                    version=self._get_version(dep, file_path),
                    strategy="pattern_match",
                )

        # Path alias match (rightmost wins), including vendor inference
        parts = [p for p in file_path.split("/") if p]
        best_match, best_pos = None, -1

        for pattern, dep in self.path_aliases.items():
            pattern_parts = pattern.strip("/").split("/")
            for i in range(len(parts) - len(pattern_parts) + 1):
                if all(fnmatch(parts[i + j], pp) for j, pp in enumerate(pattern_parts)):
                    if i > best_pos:
                        best_pos = i
                        best_match = DependencyMapping(
                            file_path=file_path,
                            dependency=dep,
                            version=self._get_version(dep, file_path),
                            confidence="medium",
                            strategy="path_alias",
                        )

        # Vendor inference: check if any vendor marker appears deeper than best match
        # e.g., "deepgemm-src/third-party/cutlass/..." -> infer "cutlass"
        vendor_markers_no_slash = [m.rstrip("/") for m in VENDOR_MARKERS]
        for i, part in enumerate(parts):
            if part.lower() in vendor_markers_no_slash and i + 1 < len(parts):
                if i > best_pos:
                    inferred_dep = parts[i + 1]
                    if inferred_dep and not inferred_dep.startswith("."):
                        best_pos = i
                        best_match = DependencyMapping(
                            file_path=file_path,
                            dependency=inferred_dep,
                            version=self._get_git_version(file_path),
                            confidence="medium",
                            strategy="vendor_inference",
                        )

        if best_match:
            return best_match

        # Generic library inference (only for third-party libraries)
        lib_match = re.match(r"^lib([a-zA-Z0-9_-]+)\.(?:so|a)(?:\.\d+)*$", basename)
        if lib_match:
            return DependencyMapping(
                file_path=file_path,
                dependency=lib_match.group(1),
                confidence="low",
                strategy="library_inference",
            )

        return None

    def _get_version(self, dep: str, file_path: str) -> str:
        # FetchContent version
        if dep in self.fetch_content:
            return self.fetch_content[dep].get("git_tag", "")

        # Git submodule version
        if "/_deps/" in file_path:
            version = self._get_git_version(file_path)
            if version:
                return version

        # Special cases
        return self._get_special_version(dep, file_path)

    def _get_git_version(self, file_path: str) -> str:
        """Get version from git submodule."""
        current = os.path.dirname(file_path)
        while current and "/_deps/" in current:
            if current in self._version_cache:
                return self._version_cache[current]
            if os.path.isfile(os.path.join(current, ".git")):
                try:
                    result = subprocess.run(
                        ["git", "describe", "--tags", "--always"],
                        cwd=current,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    version = result.stdout.strip() if result.returncode == 0 else ""
                    self._version_cache[current] = version
                    return version
                except Exception:
                    self._version_cache[current] = ""
            current = os.path.dirname(current)
        return ""

    def _get_special_version(self, dep: str, file_path: str) -> str:
        """Extract version for special dependencies from header files."""
        cache_key = f"special:{dep}"
        if cache_key in self._version_cache:
            return self._version_cache[cache_key]

        version = ""
        configs = {
            "cpython": (r"/opt/python/(\d+\.\d+\.\d+)/", None, None),
            "tensorrt-llm": "any",
            "tensorrt": (
                r"(.*/tensorrt/include)/",
                "NvInferVersion.h",
                [(r"TRT_(\w+)_ENTERPRISE\s+(\d+)", ["MAJOR", "MINOR", "PATCH"])],
            ),
            "cuda": (
                r"(.*/cuda[^/]*/(?:targets/[^/]+/)?include)/",
                "cuda_runtime_api.h",
                [
                    (
                        r"CUDART_VERSION\s+(\d+)",
                        lambda m: f"{int(m.group(1)) // 1000}.{(int(m.group(1)) % 1000) // 10}",
                    )
                ],
            ),
            "ucx": (
                r"(.*/ucx/include)/",
                "ucp/api/ucp_version.h",
                [(r"UCP_API_(\w+)\s+(\d+)", ["MAJOR", "MINOR"])],
            ),
            "nixl": (
                r"(/opt/nvidia/nvda_nixl)/",
                "VERSION",
                [(r"(\S+)", lambda m: m.group(1))],
            ),
            "openmpi": (
                r"(/opt/hpcx)/",
                "VERSION",
                # Extract HPC-X version (e.g., "HPC-X v2.25.1" -> "hpcx-2.25.1")
                [(r"HPC-X v(\S+)", lambda m: f"hpcx-{m.group(1)}")],
            ),
            "nvshmem": (
                r"(.*/nvshmem-build)/",
                "NVSHMEMVersion.cmake",
                # Extract version from set(PACKAGE_VERSION "3.2.5.1")
                [(r'set\(PACKAGE_VERSION "([^"]+)"\)', lambda m: m.group(1))],
            ),
            "mooncake": (
                r"(/usr/local/Mooncake)/",
                "VERSION",
                [(r"(\S+)", lambda m: m.group(1))],
            ),
        }

        if dep not in configs:
            # Vendored dep - use parent version
            match = re.search(r"/_deps/([^/]+)-src/", file_path)
            if match:
                parent = match.group(1)
                if parent in self.fetch_content:
                    version = self.fetch_content[parent].get("git_tag", "")
            self._version_cache[cache_key] = version
            return version

        if isinstance(configs[dep], str):
            version = configs[dep]
            self._version_cache[cache_key] = version
            return version

        dir_pattern, header, extractors = configs[dep]

        if dir_pattern:
            path_match = re.search(dir_pattern, file_path)
            if path_match:
                if header is None:
                    # Version is in the path capture group (e.g., cpython)
                    version = path_match.group(1)
                elif extractors:
                    header_path = os.path.join(path_match.group(1), header)
                    if os.path.isfile(header_path):
                        try:
                            with open(header_path) as f:
                                content = f.read()
                            for pattern, processor in extractors:
                                if callable(processor):
                                    m = re.search(pattern, content)
                                    if m:
                                        version = processor(m)
                                else:
                                    vals = {}
                                    for m in re.finditer(pattern, content):
                                        vals[m.group(1)] = m.group(2)
                                    if all(k in vals for k in processor):
                                        version = ".".join(vals[k] for k in processor)
                        except Exception:
                            pass

        self._version_cache[cache_key] = version
        return version

    def has_unknown_vendor(self, file_path: str) -> bool:
        """Check if file is in an unknown vendor or FetchContent directory."""
        path_lower = file_path.lower()
        for marker in [*VENDOR_MARKERS, "_deps/"]:
            idx = 0
            while (idx := path_lower.find(marker, idx)) != -1:
                start = idx + len(marker)
                end = path_lower.find("/", start)
                component = path_lower[start : end if end != -1 else None]
                if component and component not in self.known_names:
                    return True
                idx = end if end != -1 else len(path_lower)
        return False

    def get_source_root(self, file_path: str, dependency: str = "") -> Optional[Path]:
        """Get source root for file (handles vendored deps).

        Supports multiple build system patterns:
        - FetchContent: /_deps/<name>-src/
        - ExternalProject: /<name>-build/ (e.g., nvshmem-build)
        - Container install paths via install_path in YAML metadata
        """
        # Check if dependency has an absolute path in directory_matches
        if dependency and dependency in self.absolute_roots:
            return Path(self.absolute_roots[dependency])

        # Try FetchContent pattern first: /_deps/<name>-src/
        match = re.search(r"/_deps/([^/]+)-src/", file_path)
        if match:
            root = Path(file_path[: match.end() - 1])
            remainder = file_path[match.end() :]
        else:
            # Try ExternalProject pattern: /<name>-build/
            match = re.search(r"/([^/]+-build)/", file_path)
            if match:
                root = Path(file_path[: match.end() - 1])
                remainder = file_path[match.end() :]
            else:
                return None

        remainder_lower = remainder.lower()

        # Check path_aliases that represent subdirectories (e.g., csrc/cutlass)
        # These take priority over generic vendor markers
        for pattern in self.path_aliases:
            if pattern.startswith("_deps/"):
                continue  # Skip FetchContent patterns
            pattern_lower = pattern.lower().rstrip("/")
            idx = remainder_lower.find(pattern_lower)
            if idx != -1:
                end_idx = idx + len(pattern_lower)
                # Ensure it's a directory boundary match
                if end_idx == len(remainder_lower) or remainder[end_idx] == "/":
                    vendor_root = root / remainder[:end_idx]
                    if vendor_root.is_dir():
                        return vendor_root

        # Check standard vendor markers
        for marker in VENDOR_MARKERS:
            idx = remainder_lower.find(marker.rstrip("/") + "/")
            if idx != -1:
                vendor_start = idx + len(marker)
                next_slash = remainder_lower.find("/", vendor_start)
                vendor_path = remainder[: next_slash if next_slash != -1 else len(remainder)]
                vendor_root = root / vendor_path
                if vendor_root.is_dir():
                    return vendor_root

        return root if root.is_dir() else None

    def get_license_path(self, file_path: str, dependency: str = "") -> str:
        root = self.get_source_root(file_path, dependency)
        return " OR ".join(_find_matching_files(str(root), LICENSE_PATTERNS)) if root else ""

    def get_copyright_files(self, file_path: str) -> str:
        root = self.get_source_root(file_path)
        return " OR ".join(_find_matching_files(str(root), COPYRIGHT_PATTERNS)) if root else ""

    def get_source_url(self, dep: str, version: str = "") -> str:
        data = self.fetch_content.get(dep, {})
        repo, tag = data.get("git_repository", ""), data.get("git_tag", "")
        if not repo or not tag:
            return ""
        # Only return URL if version matches (or no version specified)
        if version and tag != version:
            return ""
        repo = repo.replace("${github_base_url}", "https://github.com").rstrip(".git")
        return f"{repo}/tree/{tag}"
