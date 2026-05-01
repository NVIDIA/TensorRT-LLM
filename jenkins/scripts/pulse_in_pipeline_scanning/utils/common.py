import json

import requests


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_permissive(licenses, license_check_token):
    headers = {
        "Authorization": f"Bearer {license_check_token}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        "https://nspect.nvidia.com/pm/api/v1.0/public/osrb/license/status",
        headers=headers,
        data=json.dumps({"licenses": licenses}),
    )
    has_non_permissive_license = False
    print(response)
    for check_result in response:
        if not check_result["isPermissive"]:
            has_non_permissive_license = True
            break
    return has_non_permissive_license
