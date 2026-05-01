import json
import time

import requests


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_permissive(licenses, license_check_token):
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
        resp = response.json()
        if resp["success"]:
            return all(result["isPermissive"] for result in resp["data"])
        else:
            print(resp)
            print(f"Check License attempt {attempt + 1} failed")
            time.sleep(2**attempt)
    return False
