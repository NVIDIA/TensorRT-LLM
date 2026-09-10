#!/usr/bin/env python3
"""Label pull requests by component, driven by .github/CODEOWNERS.

For each configured (CODEOWNERS team handle -> label) mapping, this resolves the
effective owners of every file a PR changes using CODEOWNERS last-match-wins
semantics, and applies the mapped label when any changed file is owned by that
team. Labels are only added, never removed.

Usage:
  # single PR (what the GitHub Action runs)
  label_component.py --pr 16143 --codeowners .github/CODEOWNERS
  # a few PRs
  label_component.py --pr 16143 16142
  # sweep every open PR (preview first with --dry-run)
  label_component.py --all-open --dry-run

The repo defaults to the upstream NVIDIA/TensorRT-LLM. CODEOWNERS is read from
--codeowners when given, otherwise fetched from the repo's default branch. The
token comes from GITHUB_TOKEN / GH_TOKEN, falling back to `gh auth token`.
"""

import argparse
import os
import re
import subprocess
import sys

import requests

GITHUB_API_URL = "https://api.github.com"
DEFAULT_REPO = "NVIDIA/TensorRT-LLM"

# CODEOWNERS team handle (lower-cased) -> label to apply.
# Extend this dict to cover more components.
COMPONENT_LABELS = {
    "@nvidia/trt-llm-torch-visual-gen-devs": "VisualGen",
}


# --- CODEOWNERS parsing / matching ---------------------------------------


def parse_codeowners(text):
    """Parse CODEOWNERS into an ordered list of (compiled_regex, owners)."""
    rules = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        pattern, owners = parts[0], [o.lower() for o in parts[1:]]
        rules.append((_pattern_to_regex(pattern), owners))
    return rules


def _pattern_to_regex(pattern):
    """Translate a CODEOWNERS (gitignore-style) pattern to a regex.

    '*' matches within a path segment, '**' crosses segments, and a directory
    pattern matches everything beneath it. All CODEOWNERS patterns here are
    root-anchored.
    """
    body = re.escape(pattern.strip("/"))
    body = body.replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.compile(rf"(?:{body})(?:/.*)?$")


def owners_for_path(path, rules):
    """Effective CODEOWNERS owners for a path (last matching rule wins)."""
    owners = []
    for regex, rule_owners in rules:
        if regex.match(path):
            owners = rule_owners
    return owners


def labels_for_files(files, rules, component_labels=COMPONENT_LABELS):
    labels = set()
    for path in files:
        owners = owners_for_path(path, rules)
        for team, label in component_labels.items():
            if team in owners:
                labels.add(label)
    return labels


# --- GitHub access -------------------------------------------------------


def resolve_token():
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("No token found: set GITHUB_TOKEN / GH_TOKEN, or run `gh auth login`.")


def make_session(token):
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "trtllm-label-component/1.0",
            "Authorization": f"token {token}",
        }
    )
    return session


def load_codeowners(session, repo, path):
    if path:
        with open(path, encoding="utf-8") as fh:
            return parse_codeowners(fh.read())
    r = session.get(
        f"{GITHUB_API_URL}/repos/{repo}/contents/.github/CODEOWNERS",
        headers={"Accept": "application/vnd.github.raw"},
        timeout=30,
    )
    r.raise_for_status()
    return parse_codeowners(r.text)


def iter_open_prs(session, repo, limit=None):
    """Yield (number, existing_labels) for open PRs; labels come free here."""
    page, seen = 1, 0
    while True:
        r = session.get(
            f"{GITHUB_API_URL}/repos/{repo}/pulls",
            params={
                "state": "open",
                "per_page": 100,
                "page": page,
                "sort": "created",
                "direction": "desc",
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            return
        for pr in batch:
            yield pr["number"], {lbl["name"] for lbl in pr.get("labels", [])}
            seen += 1
            if limit and seen >= limit:
                return
        page += 1


def get_changed_files(session, repo, pr_number):
    files, page = [], 1
    while True:
        r = session.get(
            f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}/files",
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        files.extend(f["filename"] for f in batch)
        page += 1
    return files


def get_pr_labels(session, repo, pr_number):
    r = session.get(f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}", timeout=30)
    r.raise_for_status()
    return {lbl["name"] for lbl in r.json().get("labels", [])}


def add_labels(session, repo, pr_number, labels):
    r = session.post(
        f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/labels",
        json={"labels": labels},
        timeout=30,
    )
    r.raise_for_status()


def process_pr(session, repo, pr_number, rules, existing_labels, dry_run):
    """Return the labels added (or that would be added). Empty if none."""
    files = get_changed_files(session, repo, pr_number)
    wanted = labels_for_files(files, rules)
    to_add = sorted(wanted - existing_labels)
    if not to_add:
        return []
    if dry_run:
        print(f"PR #{pr_number}: would add {to_add}")
    else:
        add_labels(session, repo, pr_number, to_add)
        print(f"PR #{pr_number}: added {to_add}")
    return to_add


# --- CLI -----------------------------------------------------------------


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default: {DEFAULT_REPO})")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--pr", type=int, nargs="+", metavar="N", help="label these PR number(s)")
    target.add_argument("--all-open", action="store_true", help="label every open PR in the repo")
    ap.add_argument(
        "--codeowners",
        metavar="PATH",
        help="local CODEOWNERS file; if omitted, fetched from the repo's default branch",
    )
    ap.add_argument(
        "--limit", type=int, metavar="N", help="with --all-open, cap the number of PRs scanned"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would change without adding labels"
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    session = make_session(resolve_token())
    rules = load_codeowners(session, args.repo, args.codeowners)

    if args.pr:
        targets = [(n, get_pr_labels(session, args.repo, n)) for n in args.pr]
    else:
        targets = iter_open_prs(session, args.repo, args.limit)

    scanned, labeled, failed = 0, 0, 0
    for number, existing in targets:
        scanned += 1
        # Isolate per-PR failures so one bad PR (transient 5xx, missing label)
        # does not abort an --all-open sweep.
        try:
            if process_pr(session, args.repo, number, rules, existing, args.dry_run):
                labeled += 1
        except requests.HTTPError as exc:
            failed += 1
            print(f"PR #{number}: failed ({exc})", file=sys.stderr)

    verb = "would be labeled" if args.dry_run else "labeled"
    print(f"Scanned {scanned} PR(s); {labeled} {verb}; {failed} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
