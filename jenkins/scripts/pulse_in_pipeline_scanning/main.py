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
from utils.es import get_last_scan_results

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

NEW_RISKY_DEPENDENCIES = []

last_source_vulns = get_last_scan_results("source_code_vulnerability", args.branch)
new_source_vulns = submit_source_code_vulns(
    os.path.join(args.report_directory, "source_code/vulns.json"), last_source_vulns, **SUBMIT_KWARG
)
if len(new_source_vulns) > 0:
    NEW_RISKY_DEPENDENCIES.append(f"{len(new_source_vulns)} new source code vulnerability")


last_source_licenses = get_last_scan_results("source_code_license", args.branch)
new_source_licenses = submit_source_code_licenses(
    os.path.join(args.report_directory, "source_code/sbom.json"),
    last_source_licenses,
    **SUBMIT_KWARG,
)
if len(new_source_licenses) > 0:
    NEW_RISKY_DEPENDENCIES.append(
        f"{len(new_source_licenses)} new source code unpermissive license"
    )

last_container_vulns = get_last_scan_results("container_vulnerability", args.branch)
new_amd64_container_vulns = submit_container_vulns(
    os.path.join(args.report_directory, "release_amd64_amd64/vulns.json"),
    os.path.join(args.report_directory, "base_amd64_amd64/vulns.json"),
    "amd64",
    last_container_vulns,
    **SUBMIT_KWARG,
)
new_arm64_container_vulns = submit_container_vulns(
    os.path.join(args.report_directory, "release_arm64_arm64/vulns.json"),
    os.path.join(args.report_directory, "base_arm64_arm64/vulns.json"),
    "arm64",
    last_container_vulns,
    **SUBMIT_KWARG,
)
count_container_vulns = len(new_amd64_container_vulns + new_arm64_container_vulns)
if count_container_vulns > 0:
    NEW_RISKY_DEPENDENCIES.append(f"{count_container_vulns} new container vulnerability")

last_container_licneses = get_last_scan_results("container_license", args.branch)
new_amd64_container_licenses = submit_container_licenses(
    os.path.join(args.report_directory, "release_amd64_amd64/licenses.json"),
    os.path.join(args.report_directory, "base_amd64_amd64/licenses.json"),
    "amd64",
    last_container_licneses,
    **SUBMIT_KWARG,
)
new_arm64_container_licenses = submit_container_licenses(
    os.path.join(args.report_directory, "release_arm64_arm64/licenses.json"),
    os.path.join(args.report_directory, "base_arm64_arm64/licenses.json"),
    "arm64",
    last_container_licneses,
    **SUBMIT_KWARG,
)
count_container_licenses = len(new_amd64_container_licenses + new_arm64_container_licenses)
if count_container_licenses > 0:
    NEW_RISKY_DEPENDENCIES.append(f"{count_container_licenses} new container unpermissive license")

if NEW_RISKY_DEPENDENCIES:
    print(",".join(NEW_RISKY_DEPENDENCIES))
else:
    print("All Good")
