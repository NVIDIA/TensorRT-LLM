#!/usr/bin/env python3
import argparse as _ap
import datetime as _dt
import os
import pathlib as _pl
import subprocess as _sp

CURRENT_TAG_FILE = "current_image_tags.properties"
IMAGE_MAPPING = {
    "LLM_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/__stage__:x86_64-__stage__-torch_skip",
    "LLM_SBSA_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/__stage__:sbsa-__stage__-torch_skip",
    "LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/__stage__:x86_64-rockylinux8-torch_skip-py310",
    "LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE":
    "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/__stage__:x86_64-rockylinux8-torch_skip-py312",
}


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
    parser.add_argument(
        "--stage",
        type=str,
        required=False,
        default="tritondevel",
        help=
        "The new stage part of the destination image name (default: tritondevel)."
    )
    return parser.parse_args()


def get_current_timestamp() -> str:
    """Get the current timestamp in YYYYMMDDhhmm format."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d%H%M")


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


def extract_line_after_prefix(file_path: _pl.Path, prefix: str) -> str | None:
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


def replace_text_between_dashes(input_string: str, dash_idx: int,
                                replacement: str) -> str:
    """
    Replace the text between the third and second '-' from the back in a string.

    Args:
        input_string (str): Original string.
        dash_idx (int): Index of the dash to start replacing (from the back).
        replacement (str): Replacement text.

    Returns:
        str: Updated string with the replacement applied.

    Raises:
        ValueError: If the input string does not have at least three dashes.
    """
    dash_indices = [i for i, char in enumerate(input_string) if char == '-']
    if len(dash_indices) < dash_idx:
        raise ValueError(f"Input string must have at least {dash_idx} dashes.")

    # Find the indices of third and second last dashes
    start_dash_idx = dash_indices[-dash_idx]
    stop_dash_idx = dash_indices[-dash_idx + 1]

    # Replace the segment between the dashes
    updated_string = (input_string[:start_dash_idx + 1] + replacement +
                      input_string[stop_dash_idx:])
    return updated_string


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
                  stage: str,
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
    current_tags_path = base_dir / "jenkins" / CURRENT_TAG_FILE

    for dst_key, src_pattern in IMAGE_MAPPING.items():
        print(f"Processing {dst_key} ...")
        src_image = f"{src_pattern}-{src_branch_sanitized}-{src_build_id}".replace(
            "__stage__", stage)
        dst_image_old = extract_line_after_prefix(current_tags_path,
                                                  dst_key + "=").strip('"')
        dst_image = replace_text_between_dashes(
            f"{image_prefix(dst_image_old)}-{timestamp}-{dst_mr}", 3, stage)

        run_shell_command(f"docker pull {src_image}", dry_run)
        run_shell_command(f"docker tag {src_image} {dst_image}", dry_run)
        run_shell_command(f"docker push {dst_image}", dry_run)
        find_and_replace_in_files(current_tags_path.parent,
                                  current_tags_path.name, dst_image_old,
                                  dst_image, dry_run)


def main() -> None:
    args = parse_arguments()
    rename_images(**vars(args))


if __name__ == "__main__":
    main()
