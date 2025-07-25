import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def get_pr_changed_files(pr_number: str) -> list[str]:
    """Get files changed in PR using GitHub CLI (more reliable than git diff)"""
    result = subprocess.run(
        [
            "gh", "pr", "view", pr_number, "--json", "files", "--jq",
            ".files[].path"
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_existing_reviewers(pr_number: str) -> tuple[set[str], set[str]]:
    """Get currently assigned reviewers (users and teams) for a PR"""
    try:
        # Get user reviewers
        user_result = subprocess.run(
            [
                "gh", "pr", "view", pr_number, "--json", "reviewRequests",
                "--jq",
                "(.reviewRequests // []) | .[] | select(.login) | .login"
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        user_reviewers = {
            line.strip()
            for line in user_result.stdout.splitlines() if line.strip()
        }

        # Get team reviewers
        team_result = subprocess.run(
            [
                "gh", "pr", "view", pr_number, "--json", "reviewRequests",
                "--jq", "(.reviewRequests // []) | .[] | select(.name) | .name"
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        team_reviewers = {
            line.strip()
            for line in team_result.stdout.splitlines() if line.strip()
        }

        return user_reviewers, team_reviewers
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fetch existing reviewers: {e}")
        return set(), set()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_modules(changed_files: list[str],
                module_paths: dict[str, str]) -> tuple[set[str], list[str]]:
    """Map changed files to modules using MOST SPECIFIC (longest) prefix match"""
    modules: set[str] = set()
    unmapped_files: list[str] = []

    for file in changed_files:
        # Find ALL matching prefixes
        matches = []
        for prefix, module in module_paths.items():
            if file.startswith(prefix):
                matches.append((len(prefix), prefix, module))

        if matches:
            # Sort by prefix length (descending) to get most specific first
            matches.sort(reverse=True)
            most_specific_module = matches[0][2]
            modules.add(most_specific_module)

            # Log if there were multiple matches (for debugging)
            if len(matches) > 1:
                matches[0][1]
                print(f"  File '{file}' has overlapping mappings:")
                for _, prefix, module in matches:
                    marker = "→" if module == most_specific_module else " "
                    print(f"    {marker} {prefix} -> {module}")
        else:
            unmapped_files.append(file)

    return modules, unmapped_files


def gather_reviewers(
    modules: set[str],
    module_owners: dict[str, list[str]],
    *,
    pr_author: str | None = None,
    existing_reviewers: set[str] | None = None,
    per_module_limit: int = 2
) -> tuple[list[str], dict[str, list[str]], set[str]]:
    """
    Gather reviewers ensuring each module gets representation.

    Args:
        modules: Set of module names that were touched
        module_owners: Dict mapping module names to lists of owners
        pr_author: PR author to exclude from reviewers
        existing_reviewers: Set of already assigned reviewers to exclude
        per_module_limit: Maximum reviewers to assign per module

    Returns:
        - List of all unique reviewers to assign
        - Dict mapping modules to their assigned reviewers
        - Set of modules without owners
    """
    all_reviewers: set[str] = set()
    module_assignments: dict[str, list[str]] = {}
    modules_without_owners: set[str] = set()

    for module in sorted(modules):  # Sort for consistent ordering
        owners = module_owners.get(module, [])
        if not owners:
            modules_without_owners.add(module)
            module_assignments[module] = []
            continue

        # Filter out PR author and existing reviewers
        eligible_owners = [
            o for o in owners if o != pr_author and (
                not existing_reviewers or o not in existing_reviewers)
        ]

        if not eligible_owners:
            # All owners are excluded
            print(
                f"  ⚠️  Module '{module}': All owners excluded (PR author or already assigned)"
            )
            module_assignments[module] = []
            continue

        # Sample up to per_module_limit reviewers for this module
        num_to_select = min(len(eligible_owners), per_module_limit)
        selected = random.sample(eligible_owners, num_to_select)

        module_assignments[module] = selected
        all_reviewers.update(selected)

    return sorted(all_reviewers), module_assignments, modules_without_owners


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign reviewers based on changed modules")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print the gh command instead of executing")
    parser.add_argument(
        "--force-assign",
        action="store_true",
        help=
        "Assign reviewers even if some already exist (default: only assign if no reviewers)"
    )
    args = parser.parse_args()

    pr_number = os.environ["PR_NUMBER"]
    per_module_limit = int(os.environ.get("PER_MODULE_REVIEWER_LIMIT", "2"))
    pr_author = os.environ.get("PR_AUTHOR")

    print(f"Testing PR #{pr_number} with author: {pr_author}")
    print(f"Per-module reviewer limit: {per_module_limit}")

    # Check existing reviewers
    existing_user_reviewers, existing_team_reviewers = get_existing_reviewers(
        pr_number)
    total_existing = len(existing_user_reviewers) + len(existing_team_reviewers)

    print(f"Existing user reviewers: {sorted(existing_user_reviewers)}")
    print(f"Existing team reviewers: {sorted(existing_team_reviewers)}")

    # Skip assignment if reviewers already exist (unless forced)
    if total_existing > 0 and not args.force_assign:
        print(
            f"✅ PR already has {total_existing} reviewer(s) assigned. Skipping auto-assignment."
        )
        print("   Use --force-assign to assign additional reviewers.")
        return

    try:
        changed_files = get_pr_changed_files(pr_number)
        print(f"Changed files: {changed_files}")

        module_paths = load_json(Path(".github") / "module-paths.json")
        module_owners = load_json(Path(".github") / "module-owners.json")

        modules, unmapped_files = map_modules(changed_files, module_paths)
        reviewers, module_assignments, modules_without_owners = gather_reviewers(
            modules,
            module_owners,
            pr_author=pr_author,
            existing_reviewers=
            existing_user_reviewers,  # Avoid re-assigning existing users
            per_module_limit=per_module_limit)

        print(f"\nChanged modules: {sorted(modules)}")

        # Show module-specific assignments
        if module_assignments:
            print("\nModule assignments:")
            for module, assigned in sorted(module_assignments.items()):
                if assigned:
                    print(f"  {module}: {assigned}")
                else:
                    print(f"  {module}: No eligible reviewers")

        print(f"\nFinal reviewers to assign: {reviewers}")

        # Provide detailed feedback about coverage gaps
        if unmapped_files:
            print(f"⚠️  Files with no module mapping: {unmapped_files}")
            print(
                f"   These files are not covered in .github/module-paths.json")
            print(
                f"   Consider adding appropriate module mappings for these paths."
            )

        if modules_without_owners:
            print(
                f"⚠️  Modules with no owners: {sorted(modules_without_owners)}")
            print(
                f"   These modules exist in module-paths.json but have no owners in module-owners.json"
            )
            print(f"   Consider adding owner assignments for these modules.")

        if reviewers:
            cmd = ["gh", "pr", "edit", pr_number]
            for reviewer in reviewers:
                cmd.extend(["--add-reviewer", reviewer])

            if args.dry_run:
                print(f"🔍 DRY RUN: {' '.join(cmd)}")
            else:
                try:
                    subprocess.run(cmd, check=True)
                    print(
                        f"✅ Successfully assigned {len(reviewers)} new reviewer(s)"
                    )
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to add reviewers: {e}", file=sys.stderr)
                    print(
                        "   This might be due to permissions or invalid usernames"
                    )
                    sys.exit(1)
        else:
            print("✅ No new reviewers to assign")

            # Explain why no reviewers were assigned
            if not modules and not unmapped_files:
                print("   Reason: No files were changed in this PR")
            elif not modules and unmapped_files:
                print(
                    "   Reason: All changed files are unmapped (no module coverage)"
                )
                print(
                    "   ➜ Action needed: Add module mappings to .github/module-paths.json"
                )
            elif modules and not reviewers:
                if modules_without_owners:
                    print("   Reason: Matched modules have no assigned owners")
                    print(
                        "   ➜ Action needed: Add owner assignments to .github/module-owners.json"
                    )
                else:
                    print(
                        "   Reason: All potential reviewers are already assigned or excluded"
                    )
            else:
                print(
                    "   Reason: Complex combination of mapping/ownership issues (see warnings above)"
                )

    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing PR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
