import json
import sys
import time

import requests


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_license(result):
    license = result["license"]
    is_nvidia_proprietary = result["isProprietary"] and "nvidia" in license.lower()
    return result["isPermissive"] or is_nvidia_proprietary


def is_permissive(licenses: list, license_check_token: str) -> dict:
    """Checks permissiveness for a list of license IDs in a single API call.

    Returns a dict mapping each license ID to its permissiveness.
    On API failure, all licenses default to False (non-permissive).
    """
    if not licenses:
        return {}
    headers = {
        "Authorization": f"Bearer {license_check_token}",
        "Content-Type": "application/json",
    }
    for attempt in range(5):
        response = requests.post(
            "https://nspect.nvidia.com/pm/api/v1.0/public/osrb/license/status",
            headers=headers,
            data=json.dumps({"licenses": licenses}),
        )
        if response:
            resp = response.json()
            if resp["success"]:
                if len(resp["data"]) != len(licenses):
                    print("License API returned mismatched result count", file=sys.stderr)
                    continue
                return {result["license"]: check_license(result) for result in resp["data"]}
            print(json.dumps(resp), file=sys.stderr)
        else:
            print(f"HTTP {response.status_code}", file=sys.stderr)
        print(f"Check License attempt {attempt + 1} failed", file=sys.stderr)
        time.sleep(2**attempt)
    return {lic: False for lic in licenses}
