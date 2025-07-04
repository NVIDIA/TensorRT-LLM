import os
import sys
from datetime import datetime, timedelta, timezone

import requests

GITHUB_API_URL = "https://api.github.com"
AUTO_LABEL_COMMUNITY_TOKEN = os.environ.get("AUTO_LABEL_COMMUNITY_TOKEN")
assert AUTO_LABEL_COMMUNITY_TOKEN, "AUTO_LABEL_COMMUNITY_TOKEN environment variable not set"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "PythonGitHubAction-Labeler/1.0",
    "Authorization": f"token {AUTO_LABEL_COMMUNITY_TOKEN}",
}


def check_user_membership(org: str, username: str) -> bool:
    """Checks if a user is a member of an organization using a direct API call."""
    url = f"{GITHUB_API_URL}/orgs/{org}/members/{username}"
    try:
        response = requests.get(url,
                                headers=HEADERS,
                                timeout=10,
                                allow_redirects=False)

        if response.status_code == 204:
            print(
                f"Membership check for '{username}' in '{org}': Positive (Status {response.status_code})."
            )
            return True
        elif response.status_code == 302:
            detail = (
                f"Cannot determine membership for '{username}' in '{org}' (Status 302). "
                f"The requester token does not have organization membership privileges. "
                f"This usually means the token is not associated with an org member."
            )
            print(detail)
            raise RuntimeError(detail)
        elif response.status_code == 404:
            print(
                f"Membership check for '{username}' in '{org}': Negative (Status {response.status_code})."
            )
            return False
        elif response.status_code == 403:
            error_message = "Details not parsable from JSON."
            try:
                error_message = response.json().get(
                    "message", "No specific message from API.")
            except requests.exceptions.JSONDecodeError:
                if response.text:
                    error_message = response.text
            detail = (
                f"Forbidden (403) checking membership for '{username}' in '{org}'. "
                f"Token permissions (e.g., 'read:org' scope) or org restrictions likely. API msg: {error_message}"
            )
            print(detail)
            raise RuntimeError(detail)
        else:
            print(
                f"Unexpected status {response.status_code} checking membership for '{username}' in '{org}'. Response: {response.text[:200]}"
            )
            response.raise_for_status()
            return False
    except requests.exceptions.Timeout:
        print(
            f"Timeout checking membership for '{username}' in '{org}'. Assuming not a member."
        )
        return False
    except requests.exceptions.RequestException as e:
        print(
            f"RequestException checking membership for '{username}' in '{org}': {e}. Assuming not a member."
        )
        return False


def add_label_to_pr(repo_owner: str, repo_name: str, pr_number: str,
                    label: str):
    """Adds a label to a pull request."""
    url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/issues/{pr_number}/labels"
    payload = {"labels": [label]}
    print(f"Attempting to add label. URL: {url}, Payload: {payload}")
    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        print(f"API Response Status Code: {response.status_code}")
        try:
            response_json = response.json()
            print(f"API Response JSON: {response_json}")
        except requests.exceptions.JSONDecodeError:
            print(f"API Response Text (not JSON): {response.text}")

        response.raise_for_status()
        print(f"Successfully added label '{label}' to PR #{pr_number}.")
    except requests.exceptions.RequestException as e:
        print(f"Error adding label '{label}' to PR #{pr_number}: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.content}")
        raise e


def get_recent_open_prs(repo_owner: str,
                        repo_name: str,
                        minutes_back: int = 65):
    """Get open PRs created or updated in the last N minutes."""
    cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)

    url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/pulls"
    params = {
        "state": "open",
        "sort": "updated",
        "direction": "desc",
        "per_page": 100
    }

    recent_prs = []
    page = 1

    try:
        while True:
            params["page"] = page
            response = requests.get(url,
                                    headers=HEADERS,
                                    params=params,
                                    timeout=30)
            response.raise_for_status()
            page_prs = response.json()

            if not page_prs:  # no more PRs
                break

            found_old_pr = False
            for pr in page_prs:
                created_at = datetime.strptime(
                    pr["created_at"],
                    "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                updated_at = datetime.strptime(
                    pr["updated_at"],
                    "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

                if created_at >= cutoff_time or updated_at >= cutoff_time:
                    recent_prs.append(pr)
                else:
                    # since sorted by updated desc, once we hit an old PR we can stop
                    found_old_pr = True
                    break

            if found_old_pr:
                break

            page += 1
            # safety limit to avoid infinite loops
            if page > 10:  # max 1000 PRs (100 * 10)
                print(
                    f"Warning: Hit pagination limit at page {page}, may have missed some PRs"
                )
                break

        print(
            f"Found {len(recent_prs)} PRs created/updated in the last {minutes_back} minutes (checked {page} pages)"
        )
        return recent_prs

    except requests.exceptions.RequestException as e:
        print(f"Error fetching PRs: {e}")
        raise


def main():
    """
    Main function to check user membership and apply community labels.

    Exit codes:
    0 - Success (user membership determined, appropriate action taken)
    1 - Failed to determine user membership (API permission issues)
    2 - Failed to add community label (labeling API issues)
    """
    repo_owner = os.environ.get("REPO_OWNER")
    assert repo_owner, "REPO_OWNER environment variable not set"
    repo_name = os.environ.get("REPO_NAME")
    assert repo_name, "REPO_NAME environment variable not set"
    community_label = os.environ.get("COMMUNITY_LABEL")
    assert community_label, "COMMUNITY_LABEL environment variable not set"
    time_window_minutes = int(os.environ.get("TIME_WINDOW_MINUTES"))

    print(
        f"Starting community PR labeling sweep for {repo_owner}/{repo_name}. Time window: {time_window_minutes} minutes."
    )

    try:
        recent_prs = get_recent_open_prs(repo_owner, repo_name,
                                         time_window_minutes)
    except requests.exceptions.RequestException:
        print("Failed to fetch recent PRs")
        sys.exit(1)

    processed_count = 0
    labeled_count = 0

    for pr in recent_prs:
        pr_number = pr["number"]
        pr_author = pr["user"]["login"]
        existing_labels = {label["name"] for label in pr["labels"]}

        if community_label in existing_labels:
            print(
                f"PR #{pr_number} by {pr_author} already has community label, skipping"
            )
            continue

        print(f"Processing PR #{pr_number} by {pr_author}")
        processed_count += 1

        try:
            is_member = check_user_membership("NVIDIA", pr_author)
        except RuntimeError as e:
            print(
                f"Critical error during NVIDIA membership check for '{pr_author}': {e}"
            )
            print("Continuing with next PR...")
            continue

        if not is_member:
            print(
                f"User '{pr_author}' is a community user. Adding label '{community_label}'."
            )
            try:
                add_label_to_pr(repo_owner, repo_name, str(pr_number),
                                community_label)
                labeled_count += 1
            except requests.exceptions.RequestException as e:
                print(f"Failed to add community label to PR #{pr_number}: {e}")
                # continue with other PRs instead of exiting
                continue
        else:
            print(f"User '{pr_author}' is an NVIDIA member. No label needed.")

    print(
        f"Sweep complete: processed {processed_count} PRs, labeled {labeled_count} as community"
    )


if __name__ == "__main__":
    main()
