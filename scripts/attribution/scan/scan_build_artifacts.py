#!/usr/bin/env python3
"""Build Input Scanner for TensorRT-LLM.

This is the main orchestration script that combines:
  - identify_build_inputs.py: Collects build artifacts (headers, libraries, binaries)
  - map_dependencies.py: Maps artifacts to their source dependencies

Output:
  - known.yml: Successfully mapped artifacts grouped by dependency
  - unknown.yml: Unmapped artifacts needing pattern additions
  - path_issues.yml: Artifacts with path resolution issues

Usage:
  python scan_build_artifacts.py --build-dir build/ --output-dir scan_output/
  python scan_build_artifacts.py --validate  # Validate YAML files

For collecting just the input files, use identify_build_inputs.py directly.
For mapping specific files to dependencies, use map_dependencies.py directly.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

try:
    from jsonschema import ValidationError, validate

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception  # type: ignore

# Import from the new modules
from identify_build_inputs import Artifact, collect_build_inputs
from map_dependencies import DependencyMapping, map_files_to_dependencies


def generate_reports(
    artifacts: List[Artifact],
    mappings: List[DependencyMapping],
    unmapped_paths: List[str],
    output_dir: Path,
):
    """Generate known.yml, unknown.yml, and path_issues.yml reports.

    Args:
        artifacts: All collected artifacts
        mappings: Successfully resolved mappings
        unmapped_paths: Paths that couldn't be mapped
        output_dir: Directory to write reports to

    Returns:
        Tuple of (known_file, unknown_file) paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group known artifacts by dependency
    dep_to_artifacts: Dict[str, List[str]] = {}
    dep_to_version_counts: Dict[str, Dict[str, int]] = {}

    for mapping in mappings:
        dep = mapping.dependency
        dep_to_artifacts.setdefault(dep, []).append(mapping.file_path)
        if mapping.version:
            dep_to_version_counts.setdefault(dep, {})[mapping.version] = (
                dep_to_version_counts.get(dep, {}).get(mapping.version, 0) + 1
            )

    # Build dependency objects with version info
    dep_objects: Dict[str, Dict[str, Any]] = {}
    for dep, artifacts_list in dep_to_artifacts.items():
        version = ""
        counts = dep_to_version_counts.get(dep, {})
        if counts:
            version = max(counts.items(), key=lambda kv: kv[1])[0]
        dep_objects[dep] = {
            "version": version,
            "artifacts": artifacts_list,
        }

    # Sort dependencies by artifact count (most artifacts first)
    known_sorted = dict(
        sorted(dep_objects.items(), key=lambda x: len(x[1]["artifacts"]), reverse=True)
    )

    # Write known.yml
    known_file = output_dir / "known.yml"
    with open(known_file, "w") as f:
        yaml.dump(
            {
                "summary": {
                    "total_artifacts": len(artifacts),
                    "mapped": len(mappings),
                    "unmapped": len(unmapped_paths),
                    "coverage": f"{len(mappings) / len(artifacts) * 100:.1f}%"
                    if artifacts
                    else "0%",
                    "unique_dependencies": len(known_sorted),
                },
                "dependencies": known_sorted,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    # Write unknown.yml
    unknown_file = output_dir / "unknown.yml"
    with open(unknown_file, "w") as f:
        yaml.dump(
            {
                "summary": {
                    "count": len(unmapped_paths),
                    "action_required": "Add patterns to YAML files in metadata/ for these artifacts",
                },
                "artifacts": unmapped_paths,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    # Generate path_issues.yml for non-existent paths
    path_issues_file = output_dir / "path_issues.yml"
    non_existent_paths = []

    for artifact in artifacts:
        if (
            artifact.metadata
            and not artifact.metadata.get("path_exists", True)
            and artifact.type != "library"
        ):
            non_existent_paths.append(
                {
                    "resolved_path": artifact.path,
                    "type": artifact.type,
                    "source": artifact.source,
                    "d_file_path": artifact.metadata.get("original_path", "N/A"),
                }
            )

    with open(path_issues_file, "w") as f:
        yaml.dump(
            {
                "summary": {
                    "count": len(non_existent_paths),
                    "total_artifacts": len(artifacts),
                    "percentage": f"{len(non_existent_paths) / len(artifacts) * 100:.1f}%"
                    if artifacts
                    else "0%",
                    "note": "These header paths were resolved from .d files but do not exist in the filesystem (libraries excluded)",  # noqa: E501
                },
                "non_existent_paths": non_existent_paths,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    return known_file, unknown_file


def validate_yaml_files(metadata_dir: Path) -> bool:
    """Validate YAML files without running the scanner.

    Args:
        metadata_dir: Path to metadata directory

    Returns:
        True if all files are valid, False otherwise
    """
    print("=" * 80)
    print("YAML Validation")
    print("=" * 80)
    print(f"Metadata directory: {metadata_dir}")
    print()

    if not JSONSCHEMA_AVAILABLE:
        print("Warning: jsonschema not installed, skipping validation", file=sys.stderr)
        print("Install with: pip install jsonschema")
        return False

    schema_file = metadata_dir / "_schema.yml"
    if not schema_file.exists():
        print(f"Error: Schema file not found: {schema_file}", file=sys.stderr)
        return False

    with open(schema_file, "r") as f:
        schema = yaml.safe_load(f)

    yaml_files = sorted([f for f in metadata_dir.glob("*.yml") if not f.name.startswith("_")])
    total = 0
    valid = 0
    invalid = 0

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            if "dependencies" in data and isinstance(data["dependencies"], list):
                for dep_data in data["dependencies"]:
                    total += 1
                    try:
                        validate(instance=dep_data, schema=schema)
                        print(f"✓ {yaml_file.name}:{dep_data.get('name', 'unknown')}")
                        valid += 1
                    except ValidationError as e:
                        print(
                            f"✗ {yaml_file.name}:{dep_data.get('name', 'unknown')}: {e.message}",
                            file=sys.stderr,
                        )
                        invalid += 1
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
    """Main entry point for build artifact scanner."""
    parser = argparse.ArgumentParser(
        description="Build Artifact Scanner for TensorRT-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan default build directory
  python scan_build_artifacts.py

  # Scan custom build directory with custom output
  python scan_build_artifacts.py --build-dir build/Release --output-dir scan_output/

  # Validate YAML files without scanning
  python scan_build_artifacts.py --validate

  # Use custom metadata directory
  python scan_build_artifacts.py --metadata-dir custom_metadata/
        """,
    )

    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parents[3] / "cpp" / "build",
        help="Build directory to scan for C++ artifacts (default: cpp/build/)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scan_output"),
        help="Output directory for reports (default: scan_output/)",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).parent / "metadata",
        help="Path to metadata directory containing YAML files (default: ./metadata/)",
    )

    parser.add_argument(
        "--validate", action="store_true", help="Validate YAML files without running scanner"
    )

    args = parser.parse_args()

    # Handle --validate flag
    if args.validate:
        success = validate_yaml_files(args.metadata_dir)
        sys.exit(0 if success else 1)

    # Validate inputs
    if not args.build_dir.exists():
        print(f"Error: Build directory not found: {args.build_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.metadata_dir.exists():
        print(f"Error: Metadata directory not found: {args.metadata_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("TensorRT-LLM Build Artifact Scanner")
    print("=" * 80)
    print(f"Build directory: {args.build_dir}")
    print(f"Metadata directory: {args.metadata_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Step 1: Collect artifacts using identify_build_inputs
    print("[1/3] Collecting artifacts...")
    artifacts = collect_build_inputs(args.build_dir)
    print(f"  Found {len(artifacts)} unique artifacts")
    print(f"    - Headers: {sum(1 for a in artifacts if a.type == 'header')}")
    print(f"    - Libraries: {sum(1 for a in artifacts if a.type == 'library')}")
    print(f"    - Binaries: {sum(1 for a in artifacts if a.type == 'binary')}")
    print()

    if len(artifacts) == 0:
        print("No artifacts found, exiting...")
        sys.exit(1)

    # Step 2: Map artifacts to dependencies using map_dependencies
    print("[2/3] Mapping artifacts to dependencies...")
    artifact_paths = [a.path for a in artifacts]
    mappings, unmapped = map_files_to_dependencies(
        artifact_paths,
        metadata_dir=args.metadata_dir,
        use_system_resolver=True,
    )

    # Count by strategy
    system_pkg_count = sum(1 for m in mappings if m.strategy in ("dpkg-query", "rpm-query"))
    pattern_count = len(mappings) - system_pkg_count

    print(f"  Resolved via system package manager: {system_pkg_count}")
    print(f"  Resolved via YAML patterns: {pattern_count}")
    print(f"  Unmapped: {len(unmapped)}")
    print()

    # Step 3: Generate reports
    print("[3/3] Generating reports...")
    known_file, unknown_file = generate_reports(artifacts, mappings, unmapped, args.output_dir)

    print("  Reports written to:")
    print(f"    - {known_file}")
    print(f"    - {unknown_file}")
    print(f"    - {args.output_dir / 'path_issues.yml'}")
    print()

    # Summary
    total_mapped = len(mappings)
    total_unmapped = len(unmapped)
    coverage = (total_mapped / len(artifacts) * 100) if artifacts else 0

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total artifacts: {len(artifacts)}")
    print(
        f"  Mapped (system pkg): {system_pkg_count} ({system_pkg_count / len(artifacts) * 100:.1f}%)"
    )
    print(f"  Mapped (patterns): {pattern_count} ({pattern_count / len(artifacts) * 100:.1f}%)")
    print(f"  Unmapped: {total_unmapped} ({total_unmapped / len(artifacts) * 100:.1f}%)")
    print(f"Coverage: {coverage:.1f}%")
    print()

    # Confidence breakdown
    high_conf = sum(1 for m in mappings if m.confidence == "high")
    med_conf = sum(1 for m in mappings if m.confidence == "medium")
    low_conf = sum(1 for m in mappings if m.confidence == "low")

    if mappings:
        print("Confidence Distribution:")
        print(f"  High: {high_conf} ({high_conf / len(mappings) * 100:.1f}%)")
        print(f"  Medium: {med_conf} ({med_conf / len(mappings) * 100:.1f}%)")
        print(f"  Low: {low_conf} ({low_conf / len(mappings) * 100:.1f}%)")
    else:
        print("Confidence Distribution:")
        print("  High: 0")
        print("  Medium: 0")
        print("  Low: 0")
    print()

    if total_unmapped > 0:
        print(f"ACTION REQUIRED: {total_unmapped} artifacts unmapped")
        print(f"  Review {unknown_file}")
        print(f"  Add missing patterns to YAML files in {args.metadata_dir}")
        print("  Re-run scanner to improve coverage")
    else:
        print("SUCCESS: All artifacts mapped!")

    print("=" * 80)


if __name__ == "__main__":
    main()
