import argparse
import os
from datetime import datetime, timezone

from submit_report import (
    submit_container_licenses,
    submit_container_vulns,
    submit_source_code_licenses,
    submit_source_code_vulns,
)
from utils.es import get_last_scan_results
from utils.slack import post_slack_msg

# this json file will be generated from pulse in pipeline scanning
SOURCE_CODE_VULNERABILITY = "./nspect_scan_report.json"
SOURCE_CODE_SBOM = "./sbom_toupload.json"

parser = argparse.ArgumentParser()
parser.add_argument("--build-url", required=True, help="Jenkins build URL")
parser.add_argument("--build-number", required=True, help="Jenkins build number")
parser.add_argument("--branch", required=True, help="Branch name that passed to the pipeline")
parser.add_argument(
    "--report-directory",
    required=False,
    help="Directory where the reports located",
    default="./scan_report",
)
args = parser.parse_args()

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
ES_HEADERS = {"Content-Type": "application/json"}


SUBMIT_KWARG = {
    "build_metadata": {
        "build_url": args.build_url,
        "build_number": args.build_number,
        "branch": args.branch,
    },
    "start_datetime": datetime.now(timezone.utc),
}


def process_container_result():
    NEW_RISKY_DEPENDENCIES = []

    last_source_vulns = get_last_scan_results("source_code_vulnerability", args.branch)
    new_source_vulns = submit_source_code_vulns(
        os.path.join(args.report_directory, "source_code/vulns.json"),
        last_source_vulns,
        **SUBMIT_KWARG,
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
            f"{len(new_source_licenses)} new source code non-permissive license"
        )

    last_container_vulns = get_last_scan_results("container_vulnerability", args.branch)
    new_amd64_container_vulns = submit_container_vulns(
        os.path.join(args.report_directory, "release_amd64/vulns.json"),
        os.path.join(args.report_directory, "base_amd64/vulns.json"),
        "amd64",
        last_container_vulns,
        **SUBMIT_KWARG,
    )
    new_arm64_container_vulns = submit_container_vulns(
        os.path.join(args.report_directory, "release_arm64/vulns.json"),
        os.path.join(args.report_directory, "base_arm64/vulns.json"),
        "arm64",
        last_container_vulns,
        **SUBMIT_KWARG,
    )
    count_container_vulns = len(new_amd64_container_vulns + new_arm64_container_vulns)
    if count_container_vulns > 0:
        NEW_RISKY_DEPENDENCIES.append(f"{count_container_vulns} new container vulnerability")

    last_container_licenses = get_last_scan_results("container_license", args.branch)
    new_amd64_container_licenses = submit_container_licenses(
        os.path.join(args.report_directory, "release_amd64/licenses.json"),
        os.path.join(args.report_directory, "base_amd64/licenses.json"),
        "amd64",
        last_container_licenses,
        **SUBMIT_KWARG,
    )
    new_arm64_container_licenses = submit_container_licenses(
        os.path.join(args.report_directory, "release_arm64/licenses.json"),
        os.path.join(args.report_directory, "base_arm64/licenses.json"),
        "arm64",
        last_container_licenses,
        **SUBMIT_KWARG,
    )
    count_container_licenses = len(new_amd64_container_licenses + new_arm64_container_licenses)
    if count_container_licenses > 0:
        NEW_RISKY_DEPENDENCIES.append(
            f"{count_container_licenses} new container non-permissive license"
        )

    if NEW_RISKY_DEPENDENCIES:
        print(", ".join(NEW_RISKY_DEPENDENCIES))
        post_slack_msg(args.build_number, args.branch, ", ".join(NEW_RISKY_DEPENDENCIES))
    else:
        print("No new risk involved")


if __name__ == "__main__":
    process_container_result()
