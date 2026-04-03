"""Attribution workflow for TensorRT-LLM.

This script generates ATTRIBUTIONS.md and SBOM.json for wheel builds by:
1. Collecting input files from the build directory (headers, libraries)
2. Checking database coverage via trtllm-sbom core (dry-run)
3. For missing files: mapping to dependencies and creating payload templates
4. Attempting automatic import of new dependencies
5. Providing manual instructions if automatic steps fail

Usage:
    python attribute.py --build-dir /path/to/build

For documentation on the attribution system, see:
    scripts/attribution/README.md
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

_README_TEMPLATE = Path(__file__).parent / "attribution" / "README_TEMPLATE.txt"
_SBOM_DIR = Path(__file__).parent / "attribution" / "sbom"
_SBOM_DATA_DIR = Path(__file__).parent / "attribution" / "data"

# Add scripts/attribution/scan to path for imports
_SCAN_DIR = Path(__file__).parent / "attribution" / "scan"
if str(_SCAN_DIR) not in sys.path:
    sys.path.insert(0, str(_SCAN_DIR))

from identify_build_inputs import get_input_file_paths  # noqa: E402
from map_dependencies import create_attribution_payloads, filter_ignored_deps  # noqa: E402


def _write_json(path: Path, obj) -> None:
    """Write object as JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _get_sbom_core():
    """Import and return the trtllm_sbom.core module."""
    sbom_src = _SBOM_DIR / "src"
    if str(sbom_src) not in sys.path:
        sys.path.insert(0, str(sbom_src))
    return importlib.import_module("trtllm_sbom.core")


def _ensure_sbom_cli() -> bool:
    """Ensure the trtllm-sbom package is available, installing if necessary.

    Returns:
        True if package is available or was installed successfully, False otherwise.
    """
    try:
        _get_sbom_core()
        return True
    except ModuleNotFoundError:
        pass

    print("  'trtllm-sbom' package not found. Installing...")

    # Try pip install -e (works with standard pip)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(_SBOM_DIR)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            try:
                _get_sbom_core()
            except ModuleNotFoundError as e:
                print(f"  Installed package but import failed: {e}")
                return False
            print("  Successfully installed 'trtllm-sbom' package.")
            return True
        print(f"  pip install failed: {proc.stderr or proc.stdout}")
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  pip install failed: {e}")

    return False


def _get_existing_dependencies() -> set[tuple[str, str]]:
    """Query the trtllm-sbom database for existing dependencies.

    Returns:
        Set of (dependency, version) tuples that exist in the database.
        Returns empty set if the database cannot be queried.
    """
    existing: set[tuple[str, str]] = set()
    try:
        sbom_core = _get_sbom_core()
        for item in sbom_core.fetch_dependencies_list(_SBOM_DATA_DIR):
            dep = item["dependency"]
            ver = item["version"]
            existing.add((dep, ver))
    except (ModuleNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return existing


def _write_readme(out_dir: Path) -> None:
    """Write detailed README with instructions for filling in attribution data."""
    template = _README_TEMPLATE.read_text(encoding="utf-8")
    # Strip header (everything before the marker line)
    template_marker = "--- TEMPLATE START ---"
    template = template.split(template_marker, 1)[1].lstrip("\n")
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(template)


def attempt_generate_or_prepare_payloads(build_dir: Path, num_workers: int = 1) -> bool:
    """Implements the attribution workflow.

    Steps:
      1) Collect input files from build directory and detect actively used dependency versions
      2) Check database coverage via trtllm-sbom core (dry-run)
         - If all files covered: generate outputs and return success
      3) Map missing files to dependencies and create payload templates
      4) Attempt automatic import of dependencies and file mappings
         - If successful: retry generation
      5) Provide manual instructions if automatic steps fail

    Args:
        build_dir: Build directory used for dependency scan outputs
        num_workers: Number of parallel workers for file hashing (default: 1)

    Returns:
        True if attributions were generated successfully, False otherwise.
        Outputs are placed under build_dir/attribution/.
    """
    # Ensure trtllm-sbom package is available early so dependencies can import.
    if not _ensure_sbom_cli():
        print("Error: Failed to install 'trtllm-sbom' package. Please install manually:")
        print(f"  pip install -e {_SBOM_DIR}")
        return False

    build_dir = Path(build_dir)
    out_dir = build_dir / "attribution"
    attr_path = out_dir / "ATTRIBUTIONS.md"
    sbom_path = out_dir / "SBOM.json"
    missing_path = out_dir / "missing_files.json"
    active_versions_path = out_dir / "active_versions.json"

    # Step 1: Collect input files from build directory and detect actively used versions
    print("Step 1: Collecting input files from build directory...")
    all_files = get_input_file_paths(build_dir, num_workers=num_workers)
    if not all_files:
        print("No input files collected from build directory; skipping attribution generation.")
        return False
    print(f"  Found {len(all_files)} input files")

    # Step 1b: Filter out ignored deps and detect actively used versions
    print("  Filtering ignored dependencies and detecting versions...")
    files, active_versions = filter_ignored_deps(all_files, _SCAN_DIR / "metadata")
    print(f"  After filtering: {len(files)} files ({len(all_files) - len(files)} ignored)")
    files = [Path(p) for p in files]
    if active_versions:
        # Identical files may be present in multiple versions of the same dependency,
        # so assume we only have the versions noticed by map_dependencies.py.
        print(f"  Detected {len(active_versions)} active dependencies")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(active_versions_path, active_versions)

    sbom = _get_sbom_core()

    # Step 2: Check database coverage (if complete, generate outputs and return)
    print("\nStep 2: Checking for missing attribution data...")
    dry_run_ok, missing_files = sbom.run_generate(
        data_dir=_SBOM_DATA_DIR,
        files=files,
        json_path=None,
        active_versions_path=active_versions_path,
        dry_run=True,
        num_workers=num_workers,
        attr_out=None,
        sbom_out=None,
        missing_out=missing_path,
    )
    if dry_run_ok:
        # All files have attribution data - generate actual outputs
        print("  All files have attribution data")
        generate_ok, _ = sbom.run_generate(
            data_dir=_SBOM_DATA_DIR,
            files=files,
            json_path=None,
            active_versions_path=active_versions_path,
            dry_run=False,
            num_workers=num_workers,
            attr_out=attr_path,
            sbom_out=sbom_path,
            missing_out=None,
        )
        if generate_ok:
            print(f"\nAttributions generated: {attr_path}\nSBOM generated: {sbom_path}")
        else:
            print("'trtllm-sbom generate' failed.")
        return generate_ok

    # Step 3: Map missing files to dependencies and create payload templates
    print("\nStep 3: Mapping missing files to dependencies...")
    if not missing_files:
        print("  No missing files found in output, but dry-run failed.")
        return False

    # Query existing dependencies from the database to avoid re-adding them
    print("  Querying existing dependencies from the database...")
    existing_deps = _get_existing_dependencies()
    if existing_deps:
        print(f"  Found {len(existing_deps)} existing dependencies in the database")

    add_path, register_path = create_attribution_payloads(
        [str(f) for f in missing_files],
        output_dir=out_dir,
        metadata_dir=_SCAN_DIR / "metadata",
        existing_dependencies=existing_deps,
    )

    print(f"  Created {add_path.name}")
    print(f"  Created {register_path.name}")

    # Step 4: Attempt automatic import of dependencies and file mappings
    print("\nStep 4: Attempting automatic import...")

    import_success = False
    if add_path.exists():
        print("  Importing new dependencies to the database...")
        try:
            sbom.add_dependencies_from_json(_SBOM_DATA_DIR, add_path)
            print("  Successfully imported new dependencies to the database.")
            import_success = True
        except Exception as e:
            print(f"  Failed to import dependencies automatically: {e}")

    map_success = False
    if register_path.exists():
        try:
            sbom.register_files_from_json(_SBOM_DATA_DIR, register_path)
            print("  Successfully mapped files.")
            map_success = True
        except Exception as e:
            print(f"  Failed to map files automatically: {e}")

    # If automatic import/mapping was successful, try to generate again
    if import_success or map_success:
        print("\nRe-running attribution generation after automatic registration...")
        generate_retry_ok, _ = sbom.run_generate(
            data_dir=_SBOM_DATA_DIR,
            files=files,
            json_path=None,
            active_versions_path=active_versions_path,
            dry_run=False,
            num_workers=num_workers,
            attr_out=attr_path,
            sbom_out=sbom_path,
            missing_out=None,
        )
        if generate_retry_ok:
            print(f"Attributions generated successfully: {attr_path}\nSBOM generated: {sbom_path}")
            return True
        print("Attribution generation still failed after automatic registration.")

    # Step 5: Provide manual instructions
    _write_readme(out_dir)
    print(
        f"\nSome attribution data is missing. Please follow the instructions in:\n  {out_dir / 'README.txt'}\n"
    )
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run attribution workflow using dependency scan outputs."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Build directory used for dependency scan outputs",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers for file hashing (default: 1)",
    )
    args = parser.parse_args()
    success = attempt_generate_or_prepare_payloads(args.build_dir, num_workers=args.jobs)
    raise SystemExit(0 if success else 1)
