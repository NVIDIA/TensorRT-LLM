import os
import re
import sys
from typing import List

# Matches a Markdown checklist item in the PR body.
# Expected format: "- [ ] Task description" or "* [x] Task description"
# Group 1 captures the checkbox state: ' ' (unchecked), 'x' or 'X' (checked).
# Group 2 captures the task content (the description of the checklist item).
TASK_PATTERN = re.compile(r'^\s*[-*]\s+\[( |x|X)\]\s*(.*)')


def find_all_tasks(pr_body: str) -> List[str]:
    """Return list of all task list items (both resolved and unresolved)."""
    tasks: List[str] = []
    for line in pr_body.splitlines():
        match = TASK_PATTERN.match(line)
        if match:
            tasks.append(match.group(0).strip())
    return tasks


def find_unresolved_tasks(pr_body: str) -> List[str]:
    """Return list of unresolved task list items.

    A task is considered resolved if it is checked (``[x]`` or ``[X]``)
    or if its text is struck through using ``~~`` markers.
    """
    unresolved: List[str] = []
    for line in pr_body.splitlines():
        match = TASK_PATTERN.match(line)
        if not match:
            continue
        state, content = match.groups()
        if state.lower() == 'x':
            continue
        # Check if the entire content is struck through
        if content.strip().startswith('~~') and content.strip().endswith('~~'):
            continue
        unresolved.append(match.group(0).strip())
    return unresolved


def check_pr_checklist_section(pr_body: str) -> tuple[bool, str]:
    """Check if the PR Checklist section exists with the required final checkbox.

    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if "## PR Checklist" header exists
    pr_checklist_pattern = re.compile(r'^##\s+PR\s+Checklist',
                                      re.IGNORECASE | re.MULTILINE)
    if not pr_checklist_pattern.search(pr_body):
        return False, "Missing '## PR Checklist' header. Please ensure you haven't removed the PR template section."

    # Check if the final checkbox exists (the one users must check)
    final_checkbox_pattern = re.compile(
        r'^\s*[-*]\s+\[( |x|X)\]\s+Please check this after reviewing the above items',
        re.MULTILINE)
    if not final_checkbox_pattern.search(pr_body):
        return False, "Missing the required final checkbox '- [ ] Please check this after reviewing the above items as appropriate for this PR.' Please ensure you haven't removed this from the PR template."

    return True, ""


def main() -> None:
    pr_body = os.environ.get("PR_BODY", "")
    enforce_checklist = os.environ.get("ENFORCE_PR_HAS_CHECKLIST",
                                       "false").lower() == "true"

    # Always check for PR Checklist section when enforcement is enabled
    if enforce_checklist:
        is_valid, error_msg = check_pr_checklist_section(pr_body)
        if not is_valid:
            print(f"Error: {error_msg}")
            sys.exit(1)

    all_tasks = find_all_tasks(pr_body)
    unresolved = find_unresolved_tasks(pr_body)

    # Check if we need to enforce the presence of at least one checklist item
    if enforce_checklist and not all_tasks:
        print(
            "Error: PR body must contain at least one checklist item when ENFORCE_PR_HAS_CHECKLIST is enabled."
        )
        print(
            "Expected format: - [ ] Task description or * [ ] Task description")
        sys.exit(1)

    # If we have tasks, check if any are unresolved
    if unresolved:
        print("Unresolved checklist items found:")
        for item in unresolved:
            print(f"{item}")
        sys.exit(1)

    if all_tasks:
        print("All checklist items resolved.")
    else:
        print("No checklist items found in PR body.")


if __name__ == "__main__":
    main()
