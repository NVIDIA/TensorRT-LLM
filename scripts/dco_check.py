#!/usr/bin/env python3

import re
import sys


def commit_message_has_signoff(message):
    """
    Check if the commit message has a Signed-off-by line.

    Args:
        message (str): The commit message.

    Returns:
        bool: True if the message is valid, False otherwise.
    """
    for line in message.splitlines():
        if re.match(r'^Signed-off-by: .+ <.+>$', line):
            return True
    return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python dco_check.py <commit message filename>")
        sys.exit(1)

    # Read the commit message from the file passed as an argument by Git
    with open(sys.argv[1], 'r') as file:
        message = file.read().strip()

    # Validate the commit message
    if not commit_message_has_signoff(message):
        print(
            "The commit message does not contain a Signed-off-by line. Please review CONTRIBUTING.md for more details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
