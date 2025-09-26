import argparse
import logging
import os
import re
import sys
import tempfile

import yaml


def replace_content(file_path,
                    replacement_dict,
                    inplace=False,
                    check_regex=None):
    """
    Replace content in a file based on a predefined dictionary using byte-level operations.
    If the replaced content contains a forbidden string (as a regex pattern), do not replace and return failure.
    """
    with open(file_path, 'rb') as file:
        original_content = file.read()
    content = original_content

    # Convert replacement_dict to bytes
    replacement_dict_bytes = {
        k.encode('utf-8'): v.encode('utf-8')
        for k, v in replacement_dict.items()
    }

    # Perform replacements
    for old_text, new_text in replacement_dict_bytes.items():
        content = content.replace(old_text, new_text)

    # Check if the replaced content contains the forbidden pattern
    if check_regex and check_regex.search(content):
        logging.error('%s: "%s" exist after replace' %
                      (file_path, check_regex.pattern.decode('utf-8')))
        return False

    if content == original_content:
        return True

    if inplace:
        # Create a temporary file in the same directory as the original file
        temp_dir = os.path.dirname(file_path)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir=temp_dir) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Replace the original file with the temporary file
        os.replace(temp_file_path, file_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description=
        "Replace content in files based on a YAML-defined dictionary.")
    parser.add_argument("file_paths",
                        nargs="+",
                        help="Paths to the files to be processed.")
    parser.add_argument(
        "--config",
        help=
        "Path to the YAML file containing the replacement dictionary and check pattern."
    )
    parser.add_argument("-i",
                        "--inplace",
                        action="store_true",
                        help="Write changes back to the file.")
    args = parser.parse_args()

    replacement_dict = {}
    check_pattern = None

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file) or {}
            replacement_dict = config.get("mapping", {})
            check_pattern = config.get("check")

    # Compile the regex pattern for bytes if provided
    check_regex = re.compile(
        check_pattern.encode('utf-8')) if check_pattern else None

    success = True
    for file_path in args.file_paths:
        file_success = replace_content(file_path=file_path,
                                       replacement_dict=replacement_dict,
                                       inplace=args.inplace,
                                       check_regex=check_regex)
        if not file_success:
            success = False

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
