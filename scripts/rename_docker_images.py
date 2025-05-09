#!/usr/bin/env python3
import argparse as _ap
import datetime as _dt
import os
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
    parser.add_argument(
        "--timestamp",
        type=str,
        required=False,
        help="The timestamp to use for the destination image name.")
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
    dash = '-'
    last_index = full_image_name.rfind(dash)  # Find the last occurrence
    if last_index == -1:
        raise ValueError("Invalid image name format")
    second_last_index = full_image_name.rfind(
        dash, 0, last_index)  # Look for the next occurrence before the last
    if second_last_index == -1:
        raise ValueError("Invalid image name format")
    return full_image_name[:second_last_index]


def find_and_replace_in_files(directory, file_extension: str,
                              search_string: str, replace_string: str,
                              dry_run: bool) -> None:
    """
    Perform find-and-replace in all files within a directory tree matching a specific extension.

    Args:
        directory (str or PathLike): Root directory of the search.
        file_extension (str): File extension to filter (e.g., ".txt").
        search_string (str): String to search for.
        replace_string (str): String to replace the search string with.
        dry_run (bool): Whether to perform the find-and-replace operation or not.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace strings in file content
                updated_content = content.replace(search_string, replace_string)
                if content != updated_content:
                    print(f"Updating {file_path}")
                    if not dry_run:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(updated_content)


def rename_images(*,
                  src_branch: str,
                  src_build_id: int,
                  dst_mr: int,
                  timestamp: str | None = None,
                  dry_run: bool = False) -> None:
    print(
        f"Renaming images for branch {src_branch} and build id {src_build_id} to {dst_mr}"
    )
    if dry_run:
        print("Dry-run mode enabled. No actual changes will be made.")
    else:
        print("Renaming images...")

    timestamp = timestamp or get_current_timestamp()
    src_branch_sanitized = src_branch.replace("/", "_")
    base_dir = find_script_directory().parent
    mr_groovy = base_dir / "jenkins" / MERGE_REQUEST_GROOVY

    for dst_key, src_pattern in IMAGE_MAPPING.items():
        print(f"Processing {dst_key} ...")
        src_image = f"{src_pattern}-{src_branch_sanitized}-{src_build_id}"
        dst_image_old = extract_line_after_prefix(mr_groovy,
                                                  dst_key + " = ").strip('"')
        dst_image = f"{image_prefix(dst_image_old)}-{timestamp}-{dst_mr}"
        run_shell_command(f"docker pull {src_image}", dry_run)
        run_shell_command(f"docker tag {src_image} {dst_image}", dry_run)
        run_shell_command(f"docker push {dst_image}", dry_run)
        find_and_replace_in_files(base_dir / "jenkins", ".groovy",
                                  dst_image_old, dst_image, dry_run)
        find_and_replace_in_files(base_dir / ".devcontainer", ".yaml",
                                  dst_image_old, dst_image, dry_run)


def main() -> None:
    args = parse_arguments()
    rename_images(**vars(args))


if __name__ == "__main__":
    main()
