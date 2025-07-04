import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def run_gh_command(cmd: list[str]) -> str:
    """Run GitHub CLI command and return stdout"""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_pr_changed_files(pr_number: str) -> list[str]:
    """Get files changed in PR"""
    stdout = run_gh_command([
        "gh", "pr", "view", pr_number, "--json", "files", "--jq",
        ".files[].path"
    ])
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def get_existing_reviewers(pr_number: str) -> set[str]:
    """Get currently assigned user reviewers for a PR"""
    try:
        stdout = run_gh_command([
            "gh", "pr", "view", pr_number, "--json", "reviewRequests", "--jq",
            "(.reviewRequests // []) | .[] | select(.login) | .login"
        ])
        return {line.strip() for line in stdout.splitlines() if line.strip()}
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fetch existing reviewers: {e}")
        return set()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_codeowners() -> list[tuple[str, list[str]]]:
    """Parse CODEOWNERS file and return list of (pattern, owners) tuples"""
    codeowners_path = Path(".github/CODEOWNERS")
    if not codeowners_path.exists():
        return []

    patterns = []
    with open(codeowners_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    patterns.append((parts[0], parts[1:]))
    return patterns


def is_covered_by_codeowners(filepath: str,
                             patterns: list[tuple[str, list[str]]]) -> bool:
    """Check if a file is covered by CODEOWNERS"""
    for pattern, _ in patterns:
        if pattern.startswith("/"):
            pattern = pattern[1:]  # Remove leading slash
        if filepath.startswith(pattern):
            return True
    return False


def categorize_files(
        changed_files: list[str],
        module_paths: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Categorize files into CODEOWNERS, module-mapped, and unmapped"""
    codeowners_patterns = parse_codeowners()
    codeowners_files = []
    module_files = []
    unmapped_files = []

    for filepath in changed_files:
        if is_covered_by_codeowners(filepath, codeowners_patterns):
            codeowners_files.append(filepath)
        else:
            # Check module mapping
            mapped = False
            for prefix in module_paths:
                if filepath.startswith(prefix):
                    module_files.append(filepath)
                    mapped = True
                    break
            if not mapped:
                unmapped_files.append(filepath)

    return codeowners_files, module_files, unmapped_files


def get_modules_from_files(files: list[str],
                           module_paths: dict[str, str]) -> set[str]:
    """Get modules from list of files"""
    modules = set()
    for file in files:
        for prefix, module in module_paths.items():
            if file.startswith(prefix):
                modules.add(module)
                break
    return modules


def get_reviewers_for_modules(
        modules: set[str],
        module_owners: dict[str, list[str]],
        pr_author: str = None,
        existing: set[str] = None) -> tuple[list[str], set[str]]:
    """Get reviewers for modules, excluding author and existing reviewers"""
    reviewers = set()
    modules_without_owners = set()

    for module in modules:
        owners = module_owners.get(module, [])
        if owners:
            reviewers.update(owners)
        else:
            modules_without_owners.add(module)

    if pr_author:
        reviewers.discard(pr_author)
    if existing:
        reviewers -= existing

    return sorted(reviewers), modules_without_owners


def log_coverage(label: str, files: list[str], description: str):
    """Log file coverage information"""
    if files:
        print(f"✅ Files covered by {label}: {files}")
        print(f"   {description}")


def log_issues(modules_without_owners: set[str], unmapped_files: list[str]):
    """Log configuration issues"""
    if modules_without_owners:
        print(f"⚠️  Modules with no owners: {sorted(modules_without_owners)}")
        print("   Add owner assignments to .github/module-owners.json")

    if unmapped_files:
        print(f"⚠️  Files with no coverage: {unmapped_files}")
        print(
            "   Add mappings to .github/module-paths.json or .github/CODEOWNERS"
        )


def get_assignment_reason(codeowners_files: list[str], module_files: list[str],
                          unmapped_files: list[str], modules: set[str],
                          reviewers: list[str],
                          modules_without_owners: set[str]) -> str:
    """Determine why no reviewers were assigned"""
    if not codeowners_files and not module_files and not unmapped_files:
        return "No files were changed in this PR"
    elif codeowners_files and not module_files and not unmapped_files:
        return "All changed files are covered by CODEOWNERS (GitHub will handle reviewer assignment)"
    elif not modules and unmapped_files:
        return "All mappable files are unmapped → Add module mappings to .github/module-paths.json"
    elif modules and not reviewers:
        if modules_without_owners:
            return "Matched modules have no assigned owners → Add owners to .github/module-owners.json"
        else:
            return "All potential reviewers are already assigned or excluded"
    else:
        total_covered = len(codeowners_files) + len(module_files)
        total_files = len(codeowners_files) + len(module_files) + len(
            unmapped_files)
        return f"{total_covered}/{total_files} files have reviewer coverage"


def assign_reviewers_to_pr(pr_number: str, reviewers: list[str],
                           dry_run: bool) -> bool:
    """Assign reviewers to PR"""
    if not reviewers:
        return False

    cmd = ["gh", "pr", "edit", pr_number]
    for reviewer in reviewers:
        cmd.extend(["--add-reviewer", reviewer])

    if dry_run:
        print(f"🔍 DRY RUN: {' '.join(cmd)}")
        return True

    try:
        subprocess.run(cmd, check=True)
        print(
            f"✅ Successfully assigned {len(reviewers)} new reviewer(s) via auto-assignment"
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to add reviewers: {e}", file=sys.stderr)
        print("   This might be due to permissions or invalid usernames")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign reviewers based on changed modules")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print commands instead of executing")
    parser.add_argument("--force-assign",
                        action="store_true",
                        help="Assign reviewers even if some already exist")
    args = parser.parse_args()

    # Get environment variables
    pr_number = os.environ["PR_NUMBER"]
    reviewer_limit = int(os.environ.get("REVIEWER_LIMIT", "0"))
    pr_author = os.environ.get("PR_AUTHOR")

    print(f"Testing PR #{pr_number} with author: {pr_author}")

    try:
        # Check existing reviewers
        existing_reviewers = get_existing_reviewers(pr_number)
        print(f"Existing user reviewers: {sorted(existing_reviewers)}")

        # Skip assignment if reviewers already exist (unless forced)
        if existing_reviewers and not args.force_assign:
            print(
                f"✅ PR already has {len(existing_reviewers)} reviewer(s) assigned. Skipping auto-assignment."
            )
            print("   Use --force-assign to assign additional reviewers.")
            return

        # Get changed files and load configurations
        changed_files = get_pr_changed_files(pr_number)
        print(f"Changed files: {changed_files}")

        module_paths = load_json(Path(".github") / "module-paths.json")
        module_owners = load_json(Path(".github") / "module-owners.json")

        # Categorize files by coverage type
        codeowners_files, module_files, unmapped_files = categorize_files(
            changed_files, module_paths)

        # Get modules and reviewers for module-mapped files
        modules = get_modules_from_files(module_files, module_paths)
        reviewers, modules_without_owners = get_reviewers_for_modules(
            modules, module_owners, pr_author, existing_reviewers)

        # Apply reviewer limit
        if reviewer_limit and len(reviewers) > reviewer_limit:
            reviewers = random.sample(reviewers, reviewer_limit)

        print(f"Changed modules: {sorted(modules)}")
        print(f"Potential reviewers: {reviewers}")

        # Log coverage information
        log_coverage(
            "CODEOWNERS", codeowners_files,
            "These files will have reviewers automatically assigned by GitHub's CODEOWNERS mechanism"
        )
        log_coverage("module-paths.json", module_files,
                     "These files are handled by the auto-assignment system")
        log_issues(modules_without_owners, unmapped_files)

        # Assign reviewers or explain why not
        if assign_reviewers_to_pr(pr_number, reviewers, args.dry_run):
            return

        print("✅ No new reviewers to assign")
        reason = get_assignment_reason(codeowners_files, module_files,
                                       unmapped_files, modules, reviewers,
                                       modules_without_owners)
        print(f"   Reason: {reason}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing PR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
