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
    """Map changed files to modules and return both modules and unmapped files"""
    modules: set[str] = set()
    unmapped_files: list[str] = []

    for file in changed_files:
        mapped = False
        for prefix, module in module_paths.items():
            if file.startswith(prefix):
                modules.add(module)
                mapped = True
                break

        if not mapped:
            unmapped_files.append(file)

    return modules, unmapped_files


def gather_reviewers(
        modules: set[str],
        module_owners: dict[str, list[str]],
        *,
        pr_author: str | None = None,
        existing_reviewers: set[str] | None = None
) -> tuple[list[str], set[str]]:
    """Gather reviewers and return both reviewers and modules without owners"""
    reviewers: set[str] = set()
    modules_without_owners: set[str] = set()

    for module in modules:
        owners = module_owners.get(module, [])
        if owners:
            reviewers.update(owners)
        else:
            modules_without_owners.add(module)

    if pr_author:
        reviewers.discard(pr_author)

    # Remove existing reviewers to avoid duplicate assignments
    if existing_reviewers:
        reviewers -= existing_reviewers

    return sorted(reviewers), modules_without_owners


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
    reviewer_limit = int(os.environ.get("REVIEWER_LIMIT", "0"))
    pr_author = os.environ.get("PR_AUTHOR")

    print(f"Testing PR #{pr_number} with author: {pr_author}")

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
        reviewers, modules_without_owners = gather_reviewers(
            modules,
            module_owners,
            pr_author=pr_author,
            existing_reviewers=
            existing_user_reviewers  # Avoid re-assigning existing users
        )

        if reviewer_limit and len(reviewers) > reviewer_limit:
            reviewers = random.sample(reviewers, reviewer_limit)

        print(f"Changed modules: {sorted(modules)}")
        print(f"Potential reviewers: {reviewers}")

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
