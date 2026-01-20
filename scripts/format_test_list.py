#!/usr/bin/env python3
"""Normalize tabs and multiple spaces to single spaces in files."""
import argparse
import re
import sys


def normalize_whitespace(content: str) -> str:
    """Remove leading whitespace, replace tabs and multiple spaces with single spaces."""
    lines = content.splitlines(keepends=True)
    normalized_lines = []

    for line in lines:
        # Remove leading whitespace and tabs
        line = line.lstrip(' \t')
        # Replace tabs with single space
        line = line.replace('\t', ' ')
        # Replace multiple spaces with single space
        line = re.sub(r'  +', ' ', line)
        normalized_lines.append(line)

    return ''.join(normalized_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Normalize tabs and multiple spaces to single spaces')
    parser.add_argument('filenames', nargs='*', help='Filenames to fix')
    args = parser.parse_args()

    retval = 0
    for filename in args.filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            original_contents = f.read()

        normalized_contents = normalize_whitespace(original_contents)

        if original_contents != normalized_contents:
            print(f'Fixing {filename}')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(normalized_contents)
            retval = 1

    return retval


if __name__ == '__main__':
    sys.exit(main())
