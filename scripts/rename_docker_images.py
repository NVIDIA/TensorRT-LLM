#!/usr/bin/env python3
import argparse as _ap
import datetime as _dt
import pathlib as _pl
import subprocess as _sp

MERGE_REQUEST_GROOVY = "L0_MergeRequest.groovy"
IMAGE_MAPPING = {
    "LLM_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/devel:x86_64-devel-torch_skip",
    "LLM_SBSA_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/devel:sbsa-devel-torch_skip",
    "LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/devel:x86_64-rockylinux8-torch_skip-py310",
    "LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/devel:x86_64-rockylinux8-torch_skip-py312",
}

BUILD_GROOVY = "Build.groovy"

SRC_PATTERN = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/devel:x86_64-devel-torch_skip"
# [base_image_name]-[arch]-[os](-[python_version])-[trt_version]-[torch_install_type]-[stage]-[date]-[mr_id]
DST_IMAGE = "LLM_DOCKER_IMAGE"


def parse_arguments() -> _ap.Namespace:
    parser = _ap.ArgumentParser(
        description="Rename Docker images based on the given instructions.")
    parser.add_argument(
        'src_branch',
        type=str,
        help="The name of the source branch releasing the Docker image.")
    parser.add_argument(
        'src_build_id',
        type=int,
        help="The name of the source build id release the Docker image.")
    parser.add_argument(
        'dst_mr',
        type=int,
        help="The number of the merge request for the destination image.")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Simulate the rename process without making any changes.")
    return parser.parse_args()


def get_current_timestamp() -> str:
    """Get the current timestamp in YYYYMMDDhhmm format."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d%H%M")


def run_shell_command(command: str, dry_run: bool) -> None:
    """Run a shell command and display its output.

    Args:
        command (str): The shell command to execute.
    """
    print(command)
    if not dry_run:
        _sp.run(command, shell=True, check=True, text=True)


def find_script_directory() -> _pl.Path:
    """Find the directory containing this Python script.

    Returns:
        str: The absolute path of the script's directory.
    """
    return _pl.Path(__file__).resolve().parent


def extract_line_after_prefix(file_path: _pl.Path, prefix: str) -> str or None:
    """
    Extracts the line starting with a certain prefix from the given file.

    Args:
        file_path (Path): Path to the file.
        prefix (str): Prefix to search for.

    Returns:
        str: The line starting with the prefix, or None if no such line exists.
    """
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return None


def image_prefix(full_image_name: str) -> str:
    """
    Extracts the image prefix from a full image name.
    Args:
        full_image_name (str): Full image name.

    Returns:
        str: Image prefix.
    """
    image_no_quotes = full_image_name.strip('"')
    dash = '-'
    last_index = image_no_quotes.rfind(dash)  # Find the last occurrence
    if last_index == -1:
        raise ValueError("Invalid image name format")
    second_last_index = image_no_quotes.rfind(
        dash, 0, last_index)  # Look for the next occurrence before the last
    if second_last_index == -1:
        raise ValueError("Invalid image name format")
    return image_no_quotes[:second_last_index]


def rename_images(*,
                  src_branch: str,
                  src_build_id: int,
                  dst_mr: int,
                  dry_run: bool = False) -> None:
    print(
        f"Renaming images for branch {src_branch} and build id {src_build_id} to {dst_mr}"
    )
    if dry_run:
        print("Dry-run mode enabled. No actual changes will be made.")
    else:
        print("Renaming images...")

    timestamp = get_current_timestamp()
    src_branch_sanitized = src_branch.replace("/", "_")
    mr_groovy = find_script_directory(
    ).parent / "jenkins" / MERGE_REQUEST_GROOVY

    for dst_key, src_pattern in IMAGE_MAPPING.items():
        src_image = f"{src_pattern}-{src_branch_sanitized}-{src_build_id}"
        dst_pattern = image_prefix(
            extract_line_after_prefix(mr_groovy, dst_key + " = "))
        dst_image = f"{dst_pattern}-{timestamp}-{dst_mr}"
        run_shell_command(f"docker tag {src_image} {dst_image}", dry_run)
        run_shell_command(f"docker push {dst_image}", dry_run)


def main() -> None:
    args = parse_arguments()
    rename_images(**vars(args))


if __name__ == "__main__":
    main()
