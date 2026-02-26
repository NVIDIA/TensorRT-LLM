#!/usr/bin/env python3
"""Dependency Mapper for TensorRT-LLM.

Maps input files to their source dependencies using:
  PRIMARY: System package manager (dpkg-query or rpm)
  FALLBACK: YAML pattern files from metadata/ directory
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from _package_resolvers import DpkgResolver, PythonPackageResolver, Resolver, RpmResolver
from _pattern_matcher import VENDOR_MARKERS, DependencyMapping, PatternMatcher

# =============================================================================
# Constants
# =============================================================================

# Dependencies excluded from attribution (system/internal deps)
IGNORED_DEPS = {"glibc", "gcc", "linux", "tensorrt-llm"}


# =============================================================================
# Main API
# =============================================================================


class DependencyMapper:
    """Orchestrates file-to-dependency mapping using multiple resolvers."""

    def __init__(self, metadata_dir: Optional[Path] = None, use_system_resolver: bool = True):
        self.metadata_dir = metadata_dir or Path(__file__).parent / "metadata"

        # Initialize resolvers
        self.resolvers: List[Tuple[Resolver, str]] = []
        if use_system_resolver:
            if shutil.which("dpkg-query"):
                self.resolvers.append((DpkgResolver(), "dpkg-query"))
            elif shutil.which("rpm"):
                self.resolvers.append((RpmResolver(), "rpm-query"))
        self.resolvers.append((PythonPackageResolver(), "python-package"))

        self.pattern_matcher = (
            PatternMatcher(self.metadata_dir) if self.metadata_dir.exists() else None
        )

    def map_file(self, file_path: str) -> Optional[DependencyMapping]:
        """Map a single file to its dependency."""
        # Smart routing: check file patterns to pick the best resolver first
        # This avoids expensive dpkg-query calls for files that will be resolved
        # by other resolvers anyway

        # Python packages - try python resolver first
        if "/site-packages/" in file_path or "/dist-packages/" in file_path:
            for resolver, strategy in self.resolvers:
                if strategy == "python-package":
                    result = resolver.get_package(file_path)
                    if result:
                        return DependencyMapping(
                            file_path=file_path,
                            dependency=result[0],
                            version=result[1],
                            strategy=strategy,
                            raw_package=result[2],
                        )

        # FetchContent/vendored deps - try pattern matcher first
        if "/_deps/" in file_path or any(m in file_path for m in VENDOR_MARKERS):
            if self.pattern_matcher:
                mapping = self.pattern_matcher.match(file_path)
                if mapping and not self.pattern_matcher.has_unknown_vendor(file_path):
                    return mapping

        # Default path: try all resolvers in order
        for resolver, strategy in self.resolvers:
            result = resolver.get_package(file_path)
            if result:
                return DependencyMapping(
                    file_path=file_path,
                    dependency=result[0],
                    version=result[1],
                    strategy=strategy,
                    raw_package=result[2],
                )

        # Fallback to pattern matcher
        if self.pattern_matcher:
            mapping = self.pattern_matcher.match(file_path)
            if mapping and not self.pattern_matcher.has_unknown_vendor(file_path):
                return mapping

        return None

    def map_files(self, file_paths: List[str]) -> Tuple[List[DependencyMapping], List[str]]:
        """Map files to dependencies. Returns (mappings, unmapped_files)."""
        mappings, unmapped = [], []
        for path in file_paths:
            mapping = self.map_file(path)
            if mapping:
                mappings.append(mapping)
            else:
                unmapped.append(path)
        return mappings, unmapped

    def get_license_path(self, mapping: DependencyMapping, sample_file: str) -> str:
        """Get license path based on resolution strategy."""
        if mapping.strategy == "dpkg-query":
            resolver = next((r for r, s in self.resolvers if s == "dpkg-query"), None)
            return resolver.get_license_path(mapping.raw_package) if resolver else ""
        if mapping.strategy == "rpm-query":
            resolver = next((r for r, s in self.resolvers if s == "rpm-query"), None)
            return resolver.get_license_path(mapping.raw_package) if resolver else ""
        if mapping.strategy == "python-package":
            resolver = next((r for r, s in self.resolvers if s == "python-package"), None)
            return resolver.get_license_path(sample_file, mapping.dependency) if resolver else ""
        if self.pattern_matcher:
            return self.pattern_matcher.get_license_path(sample_file, mapping.dependency)
        return ""

    def get_attribution_path(self, mapping: DependencyMapping, sample_file: str) -> str:
        """Get attribution path (e.g., NOTICE files) based on resolution strategy."""
        if mapping.strategy == "python-package":
            resolver = next((r for r, s in self.resolvers if s == "python-package"), None)
            return (
                resolver.get_attribution_path(sample_file, mapping.dependency) if resolver else ""
            )
        return ""


def create_attribution_payloads(
    file_paths: List[str],
    output_dir: Path,
    metadata_dir: Optional[Path] = None,
    use_system_resolver: bool = True,
    existing_dependencies: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[Path, Path]:
    """Map files to dependencies and create attribution payload files.

    Returns:
        Tuple of (import_payload_path, file_mappings_path)
    """
    mapper = DependencyMapper(metadata_dir, use_system_resolver)
    mappings, unmapped = mapper.map_files(file_paths)
    existing = existing_dependencies or set()

    # Group by (dep, version)
    groups: Dict[Tuple[str, str], Dict] = {}

    def is_vendor(path: str) -> bool:
        m = re.search(r"/_deps/[^/]+-src/(.+)$", path)
        return m is not None and any(marker in m.group(1).lower() for marker in VENDOR_MARKERS)

    for m in mappings:
        key = (m.dependency, m.version)
        if key not in groups:
            groups[key] = {
                "files": [],
                "raw": m.raw_package,
                "strategy": m.strategy,
                "sample": m.file_path,
            }
        groups[key]["files"].append(m.file_path)
        # Prefer non-vendor sample
        if is_vendor(groups[key]["sample"]) and not is_vendor(m.file_path):
            groups[key]["sample"] = m.file_path

    # Build payloads, excluding ignored dependencies
    file_mappings = [
        {"dependency": dep, "version": ver, "files": g["files"]}
        for (dep, ver), g in groups.items()
        if dep not in IGNORED_DEPS
    ]
    if unmapped:
        file_mappings.append({"dependency": "unknown", "version": "", "files": unmapped})

    import_payload = []
    deps_to_process = [
        (k, v) for k, v in sorted(groups.items()) if k not in existing and k[0] not in IGNORED_DEPS
    ]
    for (dep, ver), g in deps_to_process:
        sample = g["sample"]
        mapping = DependencyMapping(sample, dep, ver, strategy=g["strategy"], raw_package=g["raw"])
        license_path = mapper.get_license_path(mapping, sample)
        attribution_path = mapper.get_attribution_path(mapping, sample)
        copyright_path = (
            mapper.pattern_matcher.get_copyright_files(sample) if mapper.pattern_matcher else ""
        )
        source_url = (
            mapper.pattern_matcher.get_source_url(dep, ver) if mapper.pattern_matcher else ""
        )

        import_payload.append(
            {
                "dependency": dep,
                "version": ver,
                "license": license_path,
                "copyright": copyright_path,
                "attribution": attribution_path,
                "source": source_url,
            }
        )

    # Write files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import_path = output_dir / "import_payload.json"
    mappings_path = output_dir / "file_mappings.json"
    with open(import_path, "w") as f:
        json.dump(import_payload, f, indent=2)
    with open(mappings_path, "w") as f:
        json.dump(file_mappings, f, indent=2)

    return import_path, mappings_path


def map_files_to_dependencies(
    file_paths: List[str],
    metadata_dir: Optional[Path] = None,
    use_system_resolver: bool = True,
) -> Tuple[List[DependencyMapping], List[str]]:
    """Map files to dependencies. Convenience function."""
    return DependencyMapper(metadata_dir, use_system_resolver).map_files(file_paths)


def filter_ignored_deps(
    file_paths: List[str],
    metadata_dir: Optional[Path] = None,
    use_system_resolver: bool = True,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Filter out files belonging to ignored dependencies and detect active versions.

    This function maps all input files to their dependencies, filters out files
    that belong to IGNORED_DEPS, and extracts version information for the rest.

    Args:
        file_paths: List of file paths to analyze
        metadata_dir: Path to YAML metadata directory (optional)
        use_system_resolver: Whether to use system package managers (dpkg/rpm)

    Returns:
        Tuple of:
        - List of file paths excluding those mapped to ignored dependencies
        - Dict mapping dependency name to list of version strings (excluding ignored deps)
    """
    mapper = DependencyMapper(metadata_dir, use_system_resolver)
    mappings, unmapped = mapper.map_files(file_paths)

    # Filter out files belonging to ignored deps, collect versions for others
    filtered_files: List[str] = []
    versions: Dict[str, Set[str]] = {}
    ignored_count = 0

    for m in mappings:
        if m.dependency in IGNORED_DEPS:
            ignored_count += 1
            continue
        filtered_files.append(m.file_path)
        if m.version:
            if m.dependency not in versions:
                versions[m.dependency] = set()
            versions[m.dependency].add(m.version)

    # Include unmapped files (they still need to be processed)
    filtered_files.extend(unmapped)

    active_versions = {dep: sorted(vers) for dep, vers in versions.items() if vers}
    return filtered_files, active_versions


def detect_active_versions(
    file_paths: List[str],
    metadata_dir: Optional[Path] = None,
    use_system_resolver: bool = True,
) -> Dict[str, List[str]]:
    """Detect active dependency versions from file paths.

    This function maps all input files to their dependencies and extracts
    the version information. Returns all unique versions detected for each
    dependency, supporting cases where multiple versions are used simultaneously
    (e.g., cutlass v4.2.1 and v4.3.0).

    Args:
        file_paths: List of file paths to analyze
        metadata_dir: Path to YAML metadata directory (optional)
        use_system_resolver: Whether to use system package managers (dpkg/rpm)

    Returns:
        Dict mapping dependency name to list of version strings.
        Dependencies without version information are excluded.
    """
    _, active_versions = filter_ignored_deps(file_paths, metadata_dir, use_system_resolver)
    return active_versions


def main():
    parser = argparse.ArgumentParser(description="Map files to their source dependencies")
    parser.add_argument("files", nargs="*", help="File paths to map")
    parser.add_argument("--input", "-i", type=Path, help="JSON file with file paths")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--metadata-dir", type=Path, default=Path(__file__).parent / "metadata")
    parser.add_argument("--no-system-resolver", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    file_paths = list(args.files)
    if args.input:
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(args.input) as f:
            data = json.load(f)
            file_paths.extend(data if isinstance(data, list) else data.get("files", []))

    if not file_paths:
        print("Error: No input files specified", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Mapping {len(file_paths)} files...", file=sys.stderr)

    mappings, unmapped = map_files_to_dependencies(
        file_paths, args.metadata_dir, not args.no_system_resolver
    )

    if not args.quiet:
        print(f"Mapped: {len(mappings)}, Unmapped: {len(unmapped)}", file=sys.stderr)

    output = {
        "summary": {
            "total_files": len(file_paths),
            "mapped": len(mappings),
            "unmapped": len(unmapped),
        },
        "mappings": [m.to_dict() for m in mappings],
        "unmapped": unmapped,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        if not args.quiet:
            print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
