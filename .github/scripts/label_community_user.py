import os

import requests

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
assert GITHUB_TOKEN, "GITHUB_TOKEN environment variable not set"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "PythonGitHubAction-Labeler/1.0",
    "Authorization": f"token {GITHUB_TOKEN}",
}


def check_user_membership(org: str, username: str) -> bool:
    """Checks if a user is a member of an organization using a direct API call."""
    url = f"{GITHUB_API_URL}/orgs/{org}/members/{username}"
    try:
        response = requests.get(url,
                                headers=HEADERS,
                                timeout=10,
                                allow_redirects=False)

        if response.status_code == 204 or response.status_code == 302:
            print(
                f"Membership check for '{username}' in '{org}': Positive (Status {response.status_code})."
            )
            return True
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


def main():
    pr_author = os.environ.get("PR_AUTHOR")
    assert pr_author, "PR_AUTHOR environment variable not set"
    pr_number = os.environ.get("PR_NUMBER")
    assert pr_number, "PR_NUMBER environment variable not set"
    repo_owner = os.environ.get("REPO_OWNER")
    assert repo_owner, "REPO_OWNER environment variable not set"
    repo_name = os.environ.get("REPO_NAME")
    assert repo_name, "REPO_NAME environment variable not set"
    community_label = os.environ.get("COMMUNITY_LABEL")
    assert community_label, "COMMUNITY_LABEL environment variable not set"

    print(
        f"Starting NVIDIA membership check for PR author '{pr_author}' on PR #{pr_number}."
    )

    try:
        is_member = check_user_membership("NVIDIA", pr_author)
    except RuntimeError as e:
        print(
            f"Critical error during NVIDIA membership check for '{pr_author}': {e}"
        )
        print("Halting script due to inability to determine membership status.")
        return

    print(
        f"User '{pr_author}' is determined to be an NVIDIA member: {is_member}")

    if not is_member:
        print(
            f"User '{pr_author}' is a community user. Adding label '{community_label}'."
        )
        add_label_to_pr(repo_owner, repo_name, pr_number, community_label)
    else:
        print(
            f"User '{pr_author}' is an NVIDIA member. No label will be added.")


if __name__ == "__main__":
    main()
