import sys
from pathlib import Path

import click

from .core import (
    add_dependencies_from_json,
    add_dependency_from_args,
    format_dependencies_list,
    register_files_from_args,
    register_files_from_json,
    run_generate,
    run_validate,
)

DEFAULT_DATA_DIR = Path("scripts/attribution/data")


@click.group()
def main() -> None:
    """Dependency attribution management CLI."""


@main.command(
    name="import",
    help="Import dependencies with their license, copyright, attribution, and source code URL.",
)
@click.option("--data-dir", type=click.Path(path_type=Path), default=DEFAULT_DATA_DIR)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("-d", "dependency", type=str)
@click.option("-v", "version", type=str)
@click.option(
    "-l",
    "license_path",
    type=click.Path(path_type=Path),
    required=False,
    help="The license text for the dependency.",
)
@click.option(
    "-c",
    "copyright_path",
    type=click.Path(path_type=Path),
    required=False,
    help="The copyright text for the dependency. This is only needed if the license does not include a copyright notice and some header files do not have a copyright notices.",  # noqa: E501
)
@click.option(
    "-a",
    "attribution_path",
    type=click.Path(path_type=Path),
    required=False,
    help="The attribution text for the dependency. This is only needed if the dependency is statically linked and has dependencies.",  # noqa: E501
)
@click.option(
    "-s",
    "source",
    type=str,
    required=False,
    help="The source code URL for the dependency. This must contain a commit hash. Example: https://github.com/NVIDIA/TensorRT-LLM/commit/b87448b0...",
)
@click.option("-j", "json_path", type=click.Path(path_type=Path), required=False)
def import_deps(
    data_dir: Path,
    overwrite: bool,
    dependency: str | None,
    version: str | None,
    license_path: Path | None,
    copyright_path: Path | None,
    attribution_path: Path | None,
    source: str | None,
    json_path: Path | None,
) -> None:
    if json_path is not None:
        add_dependencies_from_json(data_dir, json_path, overwrite=overwrite)
        return
    if not all([dependency, version, license_path, source]):
        raise click.UsageError("Either provide -j JSON or all of: -d, -v, -l, -s.")
    add_dependency_from_args(
        data_dir=data_dir,
        dependency=dependency,
        version=version,
        license_path=license_path,
        copyright_path=copyright_path,
        attribution_path=attribution_path,
        source=source,
        overwrite=overwrite,
    )


@main.command(name="map-files", help="Map files to a dependency.")
@click.option("--data-dir", type=click.Path(path_type=Path), default=DEFAULT_DATA_DIR)
@click.option("-d", "dependency", type=str)
@click.option("-v", "version", type=str)
@click.option("-j", "json_path", type=click.Path(path_type=Path), required=False)
@click.argument("files", type=click.Path(path_type=Path), nargs=-1)
def map_files(
    data_dir: Path,
    dependency: str | None,
    version: str | None,
    json_path: Path | None,
    files: tuple[Path, ...],
) -> None:
    if json_path is not None:
        register_files_from_json(data_dir, json_path)
        return
    if not dependency or not version:
        raise click.UsageError("Either provide -j JSON or both -d and -v.")
    register_files_from_args(data_dir, dependency, version, list(files))


@main.command(help="Validate the internal state of the database.")
@click.option("--data-dir", type=click.Path(path_type=Path), default=DEFAULT_DATA_DIR)
def validate(data_dir: Path) -> None:
    if not run_validate(data_dir):
        sys.exit(1)


@main.command(name="list", help="List all dependencies in the database.")
@click.option("--data-dir", type=click.Path(path_type=Path), default=DEFAULT_DATA_DIR)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format: json or text (default: text)",
)
def list_dependencies(data_dir: Path, output_format: str) -> None:
    """List all dependencies currently registered in the database."""
    print(format_dependencies_list(data_dir, output_format))


@main.command(help="Generate attribution and SBOM files based on the given build input files.")
@click.option("--data-dir", type=click.Path(path_type=Path), default=DEFAULT_DATA_DIR)
@click.option("-a", "attr_out", type=click.Path(path_type=Path), required=False)
@click.option("-s", "sbom_out", type=click.Path(path_type=Path), required=False)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--missing-files", "missing_out", type=click.Path(path_type=Path), required=False)
@click.option(
    "--active-versions",
    "active_versions_path",
    type=click.Path(path_type=Path),
    required=False,
    help="JSON file mapping dependency names to lists of active versions. Only these versions are included.",
)
@click.option(
    "--jobs",
    "num_workers",
    type=int,
    default=1,
    help="Number of parallel workers for file hashing (default: 1).",
)
@click.option("--json", "json_path", type=click.Path(path_type=Path), required=False)
@click.argument("files", type=click.Path(path_type=Path), nargs=-1)
def generate(
    data_dir: Path,
    attr_out: Path | None,
    sbom_out: Path | None,
    dry_run: bool,
    missing_out: Path | None,
    active_versions_path: Path | None,
    num_workers: int,
    json_path: Path | None,
    files: tuple[Path, ...],
) -> None:
    ok, _ = run_generate(
        data_dir=data_dir,
        files=list(files),
        json_path=json_path,
        active_versions_path=active_versions_path,
        dry_run=dry_run,
        num_workers=num_workers,
        attr_out=attr_out,
        sbom_out=sbom_out,
        missing_out=missing_out,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
