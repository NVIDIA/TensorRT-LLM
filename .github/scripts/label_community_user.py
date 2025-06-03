import os
import time

import requests

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
assert GITHUB_TOKEN, "GITHUB_TOKEN environment variable not set"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "PythonGitHubAction-Labeler/1.0",
    "Authorization": f"token {GITHUB_TOKEN}",
}


def get_nvidia_members() -> list[str]:
    """Fetches all NVIDIA organization members."""
    members = []
    page = 1
    per_page = 100

    while True:
        url = f"{GITHUB_API_URL}/orgs/NVIDIA/members?per_page={per_page}&page={page}"
        try:
            time.sleep(0.5)
            response = requests.get(url, headers=HEADERS)

            if response.status_code == 404:
                raise RuntimeError(
                    f"Organization 'NVIDIA' not found (404). Cannot fetch members."
                )
            elif response.status_code == 403:
                error_message = response.json().get(
                    "message", "") if response.content else ""
                raise RuntimeError(
                    f"Forbidden (403) when fetching members for 'NVIDIA'. "
                    f"This may be due to insufficient token permissions or rate limits. Details: {error_message}. Cannot fetch members."
                )

            response.raise_for_status()
            page_data = response.json()

            if not page_data:
                break

            for member_data in page_data:
                if isinstance(member_data, dict) and "login" in member_data:
                    members.append(member_data["login"].lower())

            if len(page_data) < per_page:
                break
            page += 1
        except Exception as e:
            print(f"Error fetching NVIDIA members: {e}")
            return []

    print(f"Successfully fetched {len(members)} members for 'NVIDIA'.")
    return members


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

    nvidia_members = get_nvidia_members()
    if not nvidia_members:
        print("Could not retrieve NVIDIA members list. Exiting.")
        return

    is_member = pr_author.lower() in nvidia_members
    print(f"User '{pr_author}' is a member of NVIDIA: {is_member}")

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
