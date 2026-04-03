from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

from pydantic.type_adapter import TypeAdapter

from .models import (
    AddDependencyInput,
    DependencyMetadata,
    DependencyMetadataEntry,
    RegisterFilesInput,
)
from .storage import COPYRIGHT_TEXT_RE, DependencyDatabase, _extract_file_copyrights


def _dep_key(dependency: str, version: str) -> str:
    if "/" in dependency:
        raise ValueError(f"Dependency name cannot contain '/': {dependency}")
    if "/" in version:
        raise ValueError(f"Version cannot contain '/': {version}")
    return f"{dependency}/{version}"


def _dep_name_version(dep_key: str) -> tuple[str, str]:
    parts = dep_key.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid dependency key: {dep_key}. Expected format: <name>/<version>")
    return parts[0], parts[1]


def _warn_if_copyright_missing(
    database: DependencyDatabase, dep_key: str, meta: DependencyMetadataEntry
) -> None:
    if meta.copyright:
        return
    if not meta.license:
        return
    license_text = database.cas_get(meta.license) or ""
    if not COPYRIGHT_TEXT_RE.search(license_text):
        print(
            "Warning: No copyright notice is present in the license text for "
            f"'{dep_key}', and no explicit copyright was provided. Either add a "
            "copyright notice via the 'import' command or ensure every input file "
            "has a copyright header (SPDX-FileCopyrightText)."
        )


def _add_dependency(
    database: DependencyDatabase,
    dependency: str,
    version: str,
    license_path: Path,
    copyright_path: Path | None,
    attribution_path: Path | None,
    source: str | None,
    overwrite: bool = False,
) -> None:
    license_addr = database.cas_put(license_path.read_text(encoding="utf-8"))
    copyright_addr = (
        database.cas_put(copyright_path.read_text(encoding="utf-8")) if copyright_path else None
    )
    attribution_addr = (
        database.cas_put(attribution_path.read_text(encoding="utf-8")) if attribution_path else None
    )
    dep_key = _dep_key(dependency, version)
    meta = DependencyMetadataEntry(
        license=license_addr,
        copyright=copyright_addr,
        attribution=attribution_addr,
        source=source,
    )
    database.add_dependency_metadata(
        dep_key,
        meta,
        overwrite,
    )
    _warn_if_copyright_missing(database, dep_key, meta)


def add_dependencies_from_json(data_dir: Path, json_path: Path, overwrite: bool = False) -> None:
    objs = json.loads(json_path.read_text(encoding="utf-8"))
    database = DependencyDatabase(data_dir)
    errors: list[str] = []
    for obj in objs:
        dep_name = obj.get("dependency", "<unknown>")
        # Remove empty string fields (treating them as None)
        obj = {k: v for k, v in obj.items() if v != ""}
        try:
            item = AddDependencyInput.model_validate(obj)
            _add_dependency(
                database,
                item.dependency,
                item.version,
                item.license,
                item.copyright,
                item.attribution,
                item.source,
                overwrite,
            )
        except Exception as e:
            errors.append(f"{dep_name}: {e}")
    if errors:
        raise ValueError(f"Failed to add {len(errors)} dependencies:\n" + "\n".join(errors))


def add_dependency_from_args(
    data_dir: Path,
    dependency: str,
    version: str,
    license_path: Path,
    copyright_path: Path | None,
    attribution_path: Path | None,
    source: str | None,
    overwrite: bool = False,
) -> None:
    database = DependencyDatabase(data_dir)
    _add_dependency(
        database,
        dependency,
        version,
        license_path,
        copyright_path,
        attribution_path,
        source,
        overwrite,
    )


def register_files_from_json(data_dir: Path, json_path: Path) -> None:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    register_files_adapter = TypeAdapter(list[RegisterFilesInput])
    dependencies = register_files_adapter.validate_python(obj)
    for d in dependencies:
        register_files_from_args(data_dir, d.dependency, d.version, [Path(p) for p in d.files])


def register_files_from_args(
    data_dir: Path, dependency: str, version: str, files: Iterable[Path]
) -> None:
    database = DependencyDatabase(data_dir)
    files_list = list(files)
    try:
        database.register_dependency_files(_dep_key(dependency, version), files_list)
    except ValueError as e:
        missing: list[str] = []
        for f in files_list:
            try:
                content = Path(f).read_bytes().decode("utf-8", errors="ignore")
                notices = _extract_file_copyrights(content)
                if not notices:
                    missing.append(str(f))
            except (OSError, IOError):
                missing.append(str(f))
        if missing:
            print("The following files are missing copyright notices:")
            for m in missing:
                print(m)
        raise e


def generate_outputs(
    data_dir: Path,
    files: list[Path],
    dry_run: bool,
    active_versions: dict[str, list[str]] | None = None,
    num_workers: int = 1,
) -> Tuple[bool, list[Path], str, dict]:
    database = DependencyDatabase(data_dir)

    # Map input files to dependencies
    unmapped_files, dep_to_meta, dep_to_file_notices = database.map_files_to_dependencies(
        files, num_workers=num_workers
    )

    if unmapped_files:
        return False, unmapped_files, "", {}

    # In dry-run mode, just indicate success without generating content
    if dry_run:
        return True, [], "", {}

    # Filter dependencies based on active versions if provided
    # This removes stale license information when identical files exist in multiple dependency versions
    if active_versions:
        # First, count how many versions of each dependency are mapped
        dep_version_count: dict[str, int] = {}
        for dep in dep_to_meta.root:
            name, _ = _dep_name_version(dep)
            dep_version_count[name] = dep_version_count.get(name, 0) + 1

        filtered_meta: dict[str, DependencyMetadataEntry] = {}
        for dep, meta in dep_to_meta.root.items():
            name, version = _dep_name_version(dep)
            # Only filter if there are multiple versions of this dependency in the DB
            if dep_version_count.get(name, 0) > 1 and name in active_versions:
                # Include if version is in the list of active versions for this dependency
                if version in active_versions[name]:
                    filtered_meta[dep] = meta
            else:
                # Single version in DB or not in active_versions - include it
                filtered_meta[dep] = meta
        # Update dep_to_meta with filtered results
        dep_to_meta = DependencyMetadata(filtered_meta)

    # Build attribution text and minimal CycloneDX-like SBOM
    lines: list[str] = [
        "# Software Attributions",
        "",
        "This project uses the following libraries. Each library is licensed under the terms indicated below.",
        "",
    ]

    # Group dependencies by name and license text for combining
    grouped_deps: dict[tuple[str, str], list[tuple[str, DependencyMetadataEntry]]] = defaultdict(
        list
    )

    for dep, meta in sorted(dep_to_meta.root.items()):
        name, version = _dep_name_version(dep)
        # Get license text for grouping
        license_text = database.cas_get(meta.license) if meta.license else ""
        # Group by (name, license_text)
        grouped_deps[(name, license_text)].append((dep, meta))

    components: list[dict] = []
    transitive_attributions: dict[str, str] = {}

    for (name, license_text), dep_list in sorted(grouped_deps.items()):
        # Collect all transitive attributions (deduplicate by content)
        seen_attributions: set[str] = set()
        for dep, meta in dep_list:
            if meta.attribution:
                attrib_text = database.cas_get(meta.attribution)
                if attrib_text and attrib_text not in seen_attributions:
                    transitive_attributions[dep] = attrib_text
                    seen_attributions.add(attrib_text)

        # Add attribution text for this dependency/license combination
        lines.append(f"## {name}")
        lines.append("")

        # Collect all copyright notices from all versions (preserving order)
        all_notices: list[str] = []
        seen_notices: set[str] = set()
        for dep, meta in dep_list:
            ## Collect project-wide copyright notices
            if meta.copyright:
                copyright_text = database.cas_get(meta.copyright)
                for line in copyright_text.splitlines():
                    if line not in seen_notices:
                        seen_notices.add(line)
                        all_notices.append(line)
            ## Collect per-file copyright notices
            notices = dep_to_file_notices.get(dep, [])
            notices.sort()
            for n in notices:
                for line in n.splitlines():
                    if line not in seen_notices:
                        seen_notices.add(line)
                        all_notices.append(line)
        if all_notices:
            lines.append("### Authors")
            lines.append("")
            # Escape backticks to prevent breaking markdown code blocks
            escaped_notices = [n.replace("`", "\\`") for n in all_notices]
            lines.append("```")
            for n in escaped_notices:
                lines.append(n)
            lines.append("```")
            lines.append("")

        ## Collect license text (same for all in this group)
        lines.append("### License Text")
        lines.append("")
        lines.append("```")
        if license_text:
            # Escape backticks to prevent breaking markdown code blocks
            lines.extend(line.replace("`", "\\`") for line in license_text.splitlines())
        lines.append("```")
        lines.append("")

        # Collect SBOM data for each version
        for dep, meta in dep_list:
            _, version = _dep_name_version(dep)
            component: dict = {
                "type": "library",
                "name": name,
                "version": version,
            }
            if meta.source:
                component["externalReferences"] = [{"type": "vcs", "url": str(meta.source)}]
            components.append(component)

    # Append transitive attributions to the end of the attributions file
    for dep, attribution in sorted(transitive_attributions.items()):
        lines.append(f"# Attributions of {dep}")
        lines.append("")
        lines.append("```")
        # Escape backticks to prevent breaking markdown code blocks
        lines.extend(line.replace("`", "\\`") for line in attribution.splitlines())
        lines.append("```")
        lines.append("")

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": components,
    }
    return True, [], "\n".join(lines).strip() + "\n", sbom


def validate_database(data_dir: Path) -> tuple[bool, list[str]]:
    """Validate the internal state of the database.

    Returns:
        A tuple of (ok, problems) where ok is True if validation passed,
        and problems is a list of validation error messages.
    """
    res = DependencyDatabase(data_dir).validate_all()
    return res.ok, res.problems


def run_validate(data_dir: Path) -> bool:
    """Validate the database and print problems if any."""
    ok, problems = validate_database(data_dir)
    if not ok:
        for problem in problems:
            print(f"- {problem}")
    return ok


def fetch_dependencies_list(data_dir: Path) -> list[dict]:
    """Fetch the list of dependencies already in the database.

    Args:
        data_dir: Path to the data directory.

    Returns:
        List of {"dependency": name, "version": ver} dictionaries.
    """
    db = DependencyDatabase(data_dir)
    dependencies = []
    for key in sorted(db.meta_raw.keys()):
        name, version = _dep_name_version(key)
        dependencies.append({"dependency": name, "version": version})
    return dependencies


def format_dependencies_list(data_dir: Path, output_format: str) -> str:
    """Fetch the list of dependencies already in the database and format them for output.

    Args:
        data_dir: Path to the data directory.
        output_format: Either "json" or "text".

    Returns:
        Formatted string of dependencies.
    """
    dependencies = fetch_dependencies_list(data_dir)

    if output_format == "json":
        return json.dumps(dependencies, indent=2)
    else:
        if not dependencies:
            return "No dependencies registered."
        else:
            return "\n".join(f"{dep['dependency']}/{dep['version']}" for dep in dependencies)


def run_generate(
    data_dir: Path,
    files: list[Path],
    json_path: Path | None,
    active_versions_path: Path | None,
    dry_run: bool,
    num_workers: int,
    attr_out: Path | None,
    sbom_out: Path | None,
    missing_out: Path | None,
) -> tuple[bool, list[Path]]:
    """Run the generate workflow.

    Args:
        data_dir: Path to the data directory.
        files: List of input files.
        json_path: Optional path to JSON file containing input files.
        active_versions_path: Optional path to JSON file with active versions.
        dry_run: If True, only check coverage without generating output.
        num_workers: Number of parallel workers for file hashing.
        attr_out: Optional output path for attributions file.
        sbom_out: Optional output path for SBOM file.
        missing_out: Optional output path for missing files JSON.

    Returns:
        A tuple of (ok, missing_files) where ok is True if all files are mapped.
    """
    # Validate database state before generating outputs
    if not run_validate(data_dir):
        return False, []
    # Load input files from JSON if provided
    if json_path is not None:
        files = _load_files_from_json(json_path)
    # Load active versions if provided
    active_versions: dict[str, list[str]] | None = None
    if active_versions_path is not None:
        active_versions = _load_active_versions(active_versions_path)
    # Generate output data
    ok, missing, attr_text, sbom_json = generate_outputs(
        data_dir=data_dir,
        files=files,
        dry_run=dry_run,
        active_versions=active_versions,
        num_workers=num_workers,
    )
    # Write missing files to output file if provided
    if missing_out is not None:
        _write_missing_files(missing_out, missing)
    # If any files are missing, print them and return failure
    if not ok:
        print(
            "The following files are not associated with any dependency. "
            "Please use the 'map-files' command to associate them with a dependency."
        )
        for m in missing:
            print(str(m))
        return False, missing
    # If dry-run is enabled, return success without writing output
    if dry_run:
        return True, []
    # Write attribution text to output file if provided
    if attr_out is not None:
        _write_attribution_file(attr_out, attr_text)
    # Write SBOM to output file if provided
    if sbom_out is not None:
        _write_sbom_file(sbom_out, sbom_json)
    return True, []


def _load_files_from_json(json_path: Path) -> list[Path]:
    """Load a list of file paths from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return [Path(p) for p in json.load(f)]


def _load_active_versions(active_versions_path: Path) -> dict[str, list[str]] | None:
    """Load active versions from a JSON file if it exists."""
    if active_versions_path.exists():
        with open(active_versions_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_missing_files(missing_out: Path, missing: list[Path]) -> None:
    """Write list of missing files to a JSON file."""
    missing_out.parent.mkdir(parents=True, exist_ok=True)
    missing_out.write_text(json.dumps([str(p) for p in missing], indent=2), encoding="utf-8")


def _write_attribution_file(attr_out: Path, attr_text: str) -> None:
    """Write attribution text to a file."""
    attr_out.parent.mkdir(parents=True, exist_ok=True)
    attr_out.write_text(attr_text, encoding="utf-8")


def _write_sbom_file(sbom_out: Path, sbom_json: dict) -> None:
    """Write SBOM JSON to a file."""
    sbom_out.parent.mkdir(parents=True, exist_ok=True)
    sbom_out.write_text(json.dumps(sbom_json, indent=2), encoding="utf-8")
