import os

import requests
from utils.es import get_dashboard_url

SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
SLACK_CHANNEL = "swdl-trtllm-plc"


def post_slack_msg(build_number, branch, risk_detail, build_url=""):
    """Post a scanning-risk report to the swdl-trtllm-plc channel as the Cora bot."""
    token = os.environ.get("CORA_SLACK_BOT_TOKEN")
    if not token:
        raise EnvironmentError("Error: Environment variable 'CORA_SLACK_BOT_TOKEN' is not set!")
    dashboard_link = get_dashboard_url(build_number, branch)
    lines = [f"New risk found from nightly scanning ({branch} branch)", risk_detail]
    if dashboard_link:
        lines.append(f"<{dashboard_link}|Dashboard>")
    if build_url:
        lines.append(f"<{build_url}|Build>")
    resp = requests.post(
        SLACK_POST_MESSAGE_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={"channel": SLACK_CHANNEL, "text": "\n".join(lines)},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"Slack chat.postMessage failed: {body.get('error')}")
