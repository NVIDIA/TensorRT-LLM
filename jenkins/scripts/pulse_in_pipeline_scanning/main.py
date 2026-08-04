import argparse
import json
import os
from datetime import datetime, timezone

from submit_report import (
    submit_container_licenses,
    submit_container_vulns,
    submit_source_code_licenses,
    submit_source_code_vulns,
)
from utils.es import get_dashboard_url, get_last_scan_results
from utils.slack import post_slack_msg

# this json file will be generated from pulse in pipeline scanning
SOURCE_CODE_VULNERABILITY = "./nspect_scan_report.json"
SOURCE_CODE_SBOM = "./sbom_toupload.json"

parser = argparse.ArgumentParser()
parser.add_argument("--build-url", required=True, help="Jenkins build URL")
parser.add_argument("--build-number", required=True, help="Jenkins build number")
parser.add_argument(
    "--ref", required=True, help="Branch name or commit ID that passed to the pipeline"
)
parser.add_argument(
    "--report-directory",
    required=False,
    help="Directory where the reports located",
    default="./scan_report",
)
parser.add_argument(
    "--scan-mode",
    required=True,
    help=(
        "If set to monitor, only newly introduced risk will be reported, "
        "if set to release, all risks will be reported"
    ),
)
args = parser.parse_args()

LICENSE_CHECK_TOKEN = os.environ.get("LICENSE_CHECK_TOKEN")
if not LICENSE_CHECK_TOKEN:
    raise EnvironmentError("Error: Environment variable 'LICENSE_CHECK_TOKEN' is not set!")

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
ES_HEADERS = {"Content-Type": "application/json"}


SUBMIT_KWARG = {
    "build_metadata": {
        "build_url": args.build_url,
        "build_number": args.build_number,
        "ref": args.ref,
    },
    "start_datetime": datetime.now(timezone.utc),
    "only_report_new_risk": args.scan_mode == "monitor",
}


def process_result():
    RISKY_DEPENDENCIES = []

    last_source_vulns = get_last_scan_results("source_code_vulnerability", args.ref)
    source_vulns = submit_source_code_vulns(
        os.path.join(args.report_directory, "source_code/vulns.json"),
        last_source_vulns,
        **SUBMIT_KWARG,
    )
    if len(source_vulns) > 0:
        RISKY_DEPENDENCIES.append(f"{len(source_vulns)} new source code vulnerability")

    last_source_licenses = get_last_scan_results("source_code_license", args.ref)
    source_licenses = submit_source_code_licenses(
        os.path.join(args.report_directory, "source_code/sbom.json"),
        last_source_licenses,
        **SUBMIT_KWARG,
        license_check_token=LICENSE_CHECK_TOKEN,
    )
    if source_licenses is None:
        RISKY_DEPENDENCIES.append("source code SBOM not found")
        sbom_missing = True
    else:
        sbom_missing = False
        if len(source_licenses) > 0:
            RISKY_DEPENDENCIES.append(
                f"{len(source_licenses)} new source code non-permissive license"
            )

    last_container_vulns = get_last_scan_results("container_vulnerability", args.ref)
    amd64_container_vulns = submit_container_vulns(
        os.path.join(args.report_directory, "release_amd64/vulns.json"),
        os.path.join(args.report_directory, "base_amd64/vulns.json"),
        "amd64",
        last_container_vulns,
        **SUBMIT_KWARG,
    )
    arm64_container_vulns = submit_container_vulns(
        os.path.join(args.report_directory, "release_arm64/vulns.json"),
        os.path.join(args.report_directory, "base_arm64/vulns.json"),
        "arm64",
        last_container_vulns,
        **SUBMIT_KWARG,
    )
    count_container_vulns = len(amd64_container_vulns + arm64_container_vulns)
    if count_container_vulns > 0:
        RISKY_DEPENDENCIES.append(f"{count_container_vulns} new container vulnerability")

    last_container_licenses = get_last_scan_results("container_license", args.ref)
    amd64_container_licenses = submit_container_licenses(
        os.path.join(args.report_directory, "release_amd64/licenses.json"),
        os.path.join(args.report_directory, "base_amd64/licenses.json"),
        "amd64",
        last_container_licenses,
        **SUBMIT_KWARG,
        license_check_token=LICENSE_CHECK_TOKEN,
    )
    arm64_container_licenses = submit_container_licenses(
        os.path.join(args.report_directory, "release_arm64/licenses.json"),
        os.path.join(args.report_directory, "base_arm64/licenses.json"),
        "arm64",
        last_container_licenses,
        **SUBMIT_KWARG,
        license_check_token=LICENSE_CHECK_TOKEN,
    )
    count_container_licenses = len(amd64_container_licenses + arm64_container_licenses)
    if count_container_licenses > 0:
        RISKY_DEPENDENCIES.append(
            f"{count_container_licenses} new container non-permissive license"
        )

    if RISKY_DEPENDENCIES:
        detail = ", ".join(RISKY_DEPENDENCIES)
        status = "unstable"
        if args.scan_mode == "monitor":
            post_slack_msg(args.build_number, args.ref, detail)
        if (
            args.scan_mode == "release"
            and not sbom_missing
            and count_container_licenses + len(source_licenses) == 0
        ):
            status = "success"

        return {
            "status": status,
            "detail": detail,
            "risks": RISKY_DEPENDENCIES,
            "dashboard_url": get_dashboard_url(args.build_number, args.ref),
        }
    else:
        return {"status": "success"}


if __name__ == "__main__":
    result = process_result()
    print(json.dumps(result))
