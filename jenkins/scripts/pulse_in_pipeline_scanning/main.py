import argparse
import os
from datetime import datetime, timezone
from urllib.parse import quote

import requests
from submit_report import (
    submit_container_licenses,
    submit_container_vulns,
    submit_source_code_licenses,
    submit_source_code_vulns,
)

TIMEOUT = 1000
DOC_TYPE = "_doc"
SIZE = 10000

# Slack configuration
# Required: TRTLLM_PLC_WEBHOOK      — Slack incoming webhook URL
# Required: TRTLLM_KIBANA_DASHBOARD — Kibana dashboard URL for this report
SLACK_WEBHOOK_URL = os.environ.get("TRTLLM_PLC_WEBHOOK")
KIBANA_DASHBOARD_URL = os.environ.get("TRTLLM_KIBANA_DASHBOARD")

# this json file will be generated from pulse in pipeline scanning
SOURCE_CODE_VULNERABILITY = "./nspect_scan_report.json"
SOURCE_CODE_SBOM = "./sbom_toupload.json"


parser = argparse.ArgumentParser()
parser.add_argument("--build-url", required=True, help="Jenkins build URL")
parser.add_argument("--build-number", required=True, help="Jenkins build number")
parser.add_argument("--branch", required=True, help="Branch name that passed to the pipeline")
parser.add_argument(
    "--report-directory", required=False, help="Directory where the reports located", default="./"
)
args = parser.parse_args()

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
ES_HEADERS = {"Content-Type": "application/json"}

NEWLY_REPORTED_DEPENDENCIES = {}


def post_slack_msg():
    starttime = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    base = (
        "https://gpuwa.nvidia.com/kibana/s/tensorrt/app/dashboards"
        "#/view/f90d586c-553a-468e-b064-48e846e983a2"
    )
    start_iso = starttime.replace(tzinfo=None).isoformat()
    g = f"(filters:!(),refreshInterval:(pause:!t,value:60000),time:(from:'{start_iso}Z',to:now))"
    a = (
        f"(query:(language:kuery,query:'s_build_number:{args.build_number}"
        f' and s_branch:"{args.branch}"\'))'
    )
    dashboard_link = f"{base}?_g={quote(g)}&_a={quote(a)}"
    hasNewDependencyReported = False
    dependencyReport = f"New TRTLLM dependency found from nightly scanning ({args.branch} branch)\n"
    for key in NEWLY_REPORTED_DEPENDENCIES:
        count_reported = NEWLY_REPORTED_DEPENDENCIES[key]
        dependencyReport += "\n- "
        if count_reported:
            hasNewDependencyReported = True
        else:
            continue
        match key:
            case "source_code_vulnerability":
                dependencyReport += f"{count_reported} new source code vulnerability reported"
            case "source_code_license":
                dependencyReport += f"{count_reported} new source code GPL license reported"
    if not hasNewDependencyReported:
        return
    slack_payload = {"report": dependencyReport, "dashboardUrl": dashboard_link}
    slack_resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=60)
    slack_resp.raise_for_status()


SUBMIT_KWARG = {
    "build_metadata": {
        "build_url": args.build_url,
        "build_number": args.build_number,
        "branch": args.branch,
    },
    "start_datetime": datetime.now(timezone.utc),
}

resp_source_vulns = submit_source_code_vulns(
    os.path.join(args.report_directory, "source_code/vulns.json"), **SUBMIT_KWARG
)

resp_source_licenses = submit_source_code_licenses(
    os.path.join(args.report_directory, "source_code/sbom.json"), **SUBMIT_KWARG
)

resp_container_vulns = submit_container_vulns(
    os.path.join(args.report_directory, "release_amd64_amd64/vulns.json"),
    os.path.join(args.report_directory, "base_amd64_amd64/vulns.json"),
    "amd64",
    **SUBMIT_KWARG,
)

resp_container_licenses = submit_container_licenses(
    os.path.join(args.report_directory, "release_amd64_amd64/licenses.json"),
    os.path.join(args.report_directory, "base_amd64_amd64/licenses.json"),
    "amd64",
    **SUBMIT_KWARG,
)
