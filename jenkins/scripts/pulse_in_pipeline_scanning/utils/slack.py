import os
from datetime import datetime, timezone
from urllib.parse import quote

import requests


def post_slack_msg(build_number, branch, risk_detail):
    SLACK_WEBHOOK_URL = os.environ.get("TRTLLM_PLC_WEBHOOK")
    if not SLACK_WEBHOOK_URL:
        raise EnvironmentError("Error: Environment variable 'TRTLLM_PLC_WEBHOOK' is not set!")
    starttime = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    base = (
        "https://gpuwa.nvidia.com/kibana/s/tensorrt/app/dashboards"
        "#/view/f90d586c-553a-468e-b064-48e846e983a2"
    )
    start_iso = starttime.replace(tzinfo=None).isoformat()
    g = f"(filters:!(),refreshInterval:(pause:!t,value:60000),time:(from:'{start_iso}Z',to:now))"
    a = f"(query:(language:kuery,query:'s_build_number:{build_number} and s_branch:\"{branch}\"'))"
    dashboard_link = f"{base}?_g={quote(g)}&_a={quote(a)}"
    dependencyReport = f"New risk found from nightly scanning ({branch} branch)\n{risk_detail}"
    slack_payload = {"report": dependencyReport, "dashboardUrl": dashboard_link}
    slack_resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=60)
    slack_resp.raise_for_status()
