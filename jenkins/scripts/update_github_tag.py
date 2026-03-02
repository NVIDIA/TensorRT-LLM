#!/usr/bin/env python3
"""Update GitHub tag when all pre-merge tests pass.

This script checks whether a post-merge CI build should update the GitHub tag
'latest-ci-stable-commit-<branch>'. It validates downstream job durations,
analyzes failed stages, and creates/updates the tag if appropriate.

Note: Only supports post-merge builds (triggered by GitLab mirror).

Usage (called from Jenkins):
    python3 jenkins/scripts/update_github_tag.py \
        --build-result SUCCESS \
        --commit-sha <sha> \
        --target-branch main \
        --llm-repo <repo_url> \
        --downstream-durations '{"Build-x86_64": 30, ...}' \
        --jenkins-url <url> \
        --job-name <name> \
        --build-number <num> \
        --bot-root bot

    Environment variables:
        GITHUB_TOKEN: GitHub token for pushing tags
"""

import argparse
import json
import os
import subprocess
import sys

# Jobs to ignore when checking failures (e.g. docker image builds)
IGNORED_JOBS_FOR_TAG = ["BuildDockerImages"]

# Minimum required execution duration (minutes) for each required job.
# If a job runs shorter than this, it likely failed at startup.
MIN_JOB_DURATIONS = {
    "Build-x86_64": 10,
    "Build-SBSA": 10,
    "Test-x86_64-Single-GPU": 50,
    "Test-x86_64-Multi-GPU": 50,
    "Test-SBSA-Single-GPU": 40,
    "Test-SBSA-Multi-GPU": 40,
}


def log(msg):
    """Print a log message to stdout."""
    print(msg, flush=True)


def run_cmd(cmd, **kwargs):
    """Run a command and return the result."""
    log(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Step 1: Validate downstream job durations
# ---------------------------------------------------------------------------
def validate_downstream_job_durations(downstream_durations):
    """Validate that all required jobs ran with sufficient duration.

    Returns True if all required jobs are present and ran long enough.
    """
    log("=== Validating Required Job Execution (excludes queue time) ===")

    if not downstream_durations:
        log("❌ No downstream job data available")
        return False

    log(f"Checking {len(MIN_JOB_DURATIONS)} required job(s)...")

    issues = []
    for required_key, min_duration in MIN_JOB_DURATIONS.items():
        # Find matching job by substring match
        matched = None
        for actual_name, duration in downstream_durations.items():
            if required_key in actual_name or actual_name in required_key:
                matched = (actual_name, duration)
                break

        if matched is None:
            issues.append(f"{required_key}: Not executed (missing from downstream jobs)")
            log(f"❌ {required_key}: NOT FOUND - job was not executed")
            continue

        actual_name, actual_duration = matched
        actual_duration = float(actual_duration)

        if actual_duration < min_duration:
            issues.append(
                f"{required_key}: {actual_duration:.1f}min < {min_duration}min"
                " (likely startup failure)"
            )
            log(f"❌ {required_key}: Only {actual_duration:.1f}min (expected ≥{min_duration}min)")
        else:
            log(f"✓ {required_key}: {actual_duration:.1f}min")

    if issues:
        log("")
        log(f"❌ Validation FAILED - {len(issues)} issue(s) detected:")
        for issue in issues:
            log(f"  - {issue}")
        log("")
        log("Cannot update tag: Required jobs missing or failed too quickly")
        return False

    log(
        f"✓ All {len(MIN_JOB_DURATIONS)} required jobs executed successfully"
        " with sufficient duration"
    )
    return True


# ---------------------------------------------------------------------------
# Step 2: Get failed stages via bot's failures.py
# ---------------------------------------------------------------------------
def get_failed_stages(jenkins_url: str, job_name: str, build_number: str, bot_root: str):
    """Retrieve failed stages using the bot's failures.py script.

    Returns a list of failed stage names, or None on error.
    """
    failures_script = os.path.join(bot_root, "bin", "failures.py")
    if not os.path.isfile(failures_script):
        log(f"❌ Bot failures script not found: {failures_script}")
        return None

    cmd = [
        "python3",
        failures_script,
        "--jenkins-url",
        jenkins_url,
        "--build-path",
        job_name,
        "--build-number",
        build_number,
        "--json",
    ]

    result = run_cmd(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"ERROR: Failed to retrieve failed stages (exit code: {result.returncode})")
        if result.stderr:
            log(f"STDERR: {result.stderr}")
        if result.stdout:
            log(f"STDOUT: {result.stdout}")
        return None

    try:
        data = json.loads(result.stdout)
        failed_list = data.get("failed_stage_list", [])
    except (json.JSONDecodeError, KeyError) as exc:
        log(f"ERROR: Failed to parse failures output: {exc}")
        log(f"Raw output: {result.stdout}")
        return None

    log(f"Found {len(failed_list)} failed stage(s): {', '.join(failed_list)}")
    return failed_list


# ---------------------------------------------------------------------------
# Step 3: Check whether all failures are post-merge only
# ---------------------------------------------------------------------------
def are_all_failures_post_merge(failed_stage_list):
    """Return True if all relevant failures are from post-merge stages."""
    if not failed_stage_list:
        log("✓ No failed stages")
        return True

    # Filter out ignored jobs
    relevant = [
        s for s in failed_stage_list if not any(ignored in s for ignored in IGNORED_JOBS_FOR_TAG)
    ]

    if not relevant:
        log("✓ All failures are from ignored jobs")
        return True

    # Separate pre-merge and post-merge failures
    premerge = [s for s in relevant if "Post-Merge" not in s and "post-merge" not in s]
    postmerge_count = len(relevant) - len(premerge)

    log(
        f"Relevant failures: {len(relevant)} total"
        f" ({len(premerge)} pre-merge, {postmerge_count} post-merge)"
    )

    if not premerge:
        log(f"✓ Only post-merge failures: {', '.join(relevant)}")
        return True

    log(f"❌ Found pre-merge failures: {', '.join(premerge)}")
    return False


# ---------------------------------------------------------------------------
# Step 4: Create / update GitHub tag via REST API
# ---------------------------------------------------------------------------
GITHUB_REPO = "NVIDIA/TensorRT-LLM"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}"


def _github_api(method, endpoint, token, data=None):
    """Make a GitHub REST API request via curl.

    Returns (http_status_code: str, response_body: dict | None).
    """
    cmd = [
        "curl",
        "-s",
        "-X",
        method,
        f"{GITHUB_API_BASE}{endpoint}",
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "Accept: application/vnd.github.v3+json",
        "-w",
        "\\nHTTP_CODE:%{http_code}",
    ]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(data)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    code = output.split("HTTP_CODE:")[-1].strip() if "HTTP_CODE:" in output else "unknown"
    body = None
    try:
        body = json.loads(output.split("HTTP_CODE:")[0])
    except (json.JSONDecodeError, ValueError):
        pass

    return code, body


def create_github_tag(commit_sha, target_branch):
    """Create or update tag 'latest-ci-stable-commit-<branch>' via GitHub REST API.

    Requires GITHUB_TOKEN env var. Returns True on success.
    """
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        log("ERROR: GITHUB_TOKEN environment variable is not set")
        return False

    tag_name = f"latest-ci-stable-commit-{target_branch}"
    tag_message = f"Post-merge tests passed for branch {target_branch}"
    log(f"Creating/updating tag '{tag_name}' for commit {commit_sha}")

    # If the tag ref already exists, update it
    code, _ = _github_api("GET", f"/git/refs/tags/{tag_name}", github_token)
    if code == "200":
        code, _ = _github_api(
            "PATCH",
            f"/git/refs/tags/{tag_name}",
            github_token,
            {"sha": commit_sha, "force": True},
        )
        if code == "200":
            log(f"✓ Updated tag '{tag_name}' → {commit_sha}")
            return True
        log(f"ERROR: Failed to update tag ref (HTTP {code})")
        return False

    # Tag doesn't exist — create tag object + ref
    code, body = _github_api(
        "POST",
        "/git/tags",
        github_token,
        {"tag": tag_name, "message": tag_message, "object": commit_sha, "type": "commit"},
    )
    if code != "201" or not body or "sha" not in body:
        log(f"ERROR: Failed to create tag object (HTTP {code})")
        return False

    code, _ = _github_api(
        "POST",
        "/git/refs",
        github_token,
        {"ref": f"refs/tags/{tag_name}", "sha": body["sha"]},
    )
    if code == "201":
        log(f"✓ Created tag '{tag_name}' → {commit_sha}")
        return True

    log(f"ERROR: Failed to create tag ref (HTTP {code})")
    return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Update GitHub tag after CI build")
    parser.add_argument(
        "--build-result", required=True, help="Build result (SUCCESS, FAILURE, ...)"
    )
    parser.add_argument("--commit-sha", required=True, help="Git commit SHA")
    parser.add_argument("--target-branch", default="main", help="Target branch name")
    parser.add_argument("--llm-repo", required=True, help="LLM repo URL for cloning")
    parser.add_argument(
        "--downstream-durations", default="{}", help="JSON of job name -> duration (minutes)"
    )
    parser.add_argument("--jenkins-url", default="", help="Jenkins URL")
    parser.add_argument("--job-name", default="", help="Jenkins job name")
    parser.add_argument("--build-number", default="", help="Jenkins build number")
    parser.add_argument("--bot-root", default="bot", help="Path to bot checkout root")
    args = parser.parse_args()

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        log("ERROR: GITHUB_TOKEN environment variable is not set")
        return 1

    # Only support post-merge builds (triggered by GitLab mirror)
    if "PostMerge" not in args.job_name:
        log("ERROR: This script only supports post-merge builds")
        return 1

    log(f"=== GitHub Tag Update Check ({args.build_result}) ===")
    log("Post-merge build detected (GitLab mirror trigger)")

    # Fast path: SUCCESS → update tag directly
    if args.build_result == "SUCCESS":
        log("✓ Pipeline succeeded - updating tag")
        ok = create_github_tag(args.commit_sha, args.target_branch)
        return 0 if ok else 1

    # Slow path: Analyze failures
    log("Analyzing failures to determine if only post-merge tests failed...")

    # Parse downstream durations
    try:
        downstream_durations = json.loads(args.downstream_durations)
    except json.JSONDecodeError as exc:
        log(f"ERROR: Cannot parse downstream-durations JSON: {exc}")
        return 1

    # Step 1: Validate durations
    if not validate_downstream_job_durations(downstream_durations):
        return 1

    # Step 2: Get failed stages
    failed_stages = get_failed_stages(
        args.jenkins_url,
        args.job_name,
        args.build_number,
        args.bot_root,
    )
    if not failed_stages:
        log("❌ Cannot retrieve failure information - skipping tag update")
        return 1

    # Step 3: Check if only post-merge tests failed
    if not are_all_failures_post_merge(failed_stages):
        log(f"❌ Found pre-merge failures: {', '.join(failed_stages)}")
        return 1

    # All checks passed → create tag
    log("✓ Only post-merge failures detected - updating tag")
    ok = create_github_tag(args.commit_sha, args.target_branch)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
