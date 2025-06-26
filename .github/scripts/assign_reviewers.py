import argparse
import json
import os
import subprocess
from pathlib import Path


def run_git_diff(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_modules(changed_files: list[str], module_paths: dict[str,
                                                             str]) -> set[str]:
    modules: set[str] = set()
    for file in changed_files:
        for prefix, module in module_paths.items():
            if file.startswith(prefix):
                modules.add(module)
                break
    return modules


def gather_reviewers(modules: set[str],
                     module_owners: dict[str, list[str]],
                     *,
                     pr_author: str | None = None) -> list[str]:
    reviewers: set[str] = set()
    for module in modules:
        reviewers.update(module_owners.get(module, []))

    if pr_author:
        reviewers.discard(pr_author)

    return sorted(reviewers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign reviewers based on changed modules")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print the gh command instead of executing")
    args = parser.parse_args()

    base_sha = os.environ["BASE_SHA"]
    head_sha = os.environ["HEAD_SHA"]
    pr_number = os.environ["PR_NUMBER"]
    reviewer_limit = int(os.environ.get("REVIEWER_LIMIT", "0"))
    pr_author = os.environ.get("PR_AUTHOR")

    changed_files = run_git_diff(base_sha, head_sha)
    module_paths = load_json(Path(".github") / "module-paths.json")
    module_owners = load_json(Path(".github/workflows") / "module-owners.json")

    modules = map_modules(changed_files, module_paths)
    reviewers = gather_reviewers(modules, module_owners, pr_author=pr_author)

    if reviewer_limit:
        reviewers = reviewers[:reviewer_limit]

    if reviewers:
        cmd = ["gh", "pr", "edit", pr_number]
        for reviewer in reviewers:
            cmd.extend(["--add-reviewer", reviewer])
        if args.dry_run:
            print("Dry run:", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)

    print(f"Changed modules: {sorted(modules)}")
    print(f"Requested reviewers: {reviewers}")


if __name__ == "__main__":
    main()