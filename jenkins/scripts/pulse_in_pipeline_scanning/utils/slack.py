import os

import requests
from utils.es import get_dashboard_url


def post_slack_msg(build_number, branch, risk_detail):
    SLACK_WEBHOOK_URL = os.environ.get("TRTLLM_PLC_WEBHOOK")
    if not SLACK_WEBHOOK_URL:
        raise EnvironmentError("Error: Environment variable 'TRTLLM_PLC_WEBHOOK' is not set!")
    dependencyReport = f"New risk found from nightly scanning ({branch} branch)\n{risk_detail}"
    dashboard_link = get_dashboard_url(build_number, branch)
    slack_payload = {"report": dependencyReport, "dashboardUrl": dashboard_link}
    slack_resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=60)
    slack_resp.raise_for_status()
    return
