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
from utils.es import get_dashboard_url
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
parser.add_argument(
    "--skip-source-code",
    action="store_true",
    default=False,
    help="Skip processing source code scanning results",
)
parser.add_argument(
    "--skip-container",
    action="store_true",
    default=False,
    help="Skip processing container scanning results",
)
args = parser.parse_args()

LICENSE_CHECK_TOKEN = os.environ.get("LICENSE_CHECK_TOKEN", "")
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
}


def _risks_to_preapproved_candidates(
    risks: list, scan_type: str, fallback_package_type: str | None = None
) -> list:
    """Extract preapproved index records from risk docs, deduplicating by (name, type)."""
    seen = set()
    candidates = []
    for doc in risks:
        name = doc.get("s_package_name")
        pkg_type = doc.get("s_package_type") or fallback_package_type
        key = (name, scan_type, pkg_type)
        if key in seen or not name:
            continue
        seen.add(key)
        candidates.append(
            {
                "scan_type": scan_type,
                "package_name": name,
                "package_version": doc.get("s_package_version"),
                "package_type": pkg_type,
            }
        )
    return candidates


def _license_docs_to_entries(docs: list, scan_type: str, license_info: dict | None = None) -> list:
    """Convert ES risk docs to output entries with dependency_name, license, is_permissive."""
    license_info = license_info or {}
    entries = []
    for doc in docs:
        pkg = doc.get("s_package_name", "unknown")
        info = license_info.get(pkg, {})
        entries.append(
            {
                "dependency_name": pkg,
                "license": doc.get("s_license_ids") or "Unknown",
                "corrected_license": doc.get("s_corrected_license") or "",
                "is_permissive": info.get("is_permissive", False),
                "is_nvidia_proprietary": info.get("is_nvidia_proprietary"),
                "scan_type": scan_type,
            }
        )
    return entries


def process_result():
    RISKY_DEPENDENCIES = []
    detected_licenses = []
    preapproved_candidates = []

    if not args.skip_source_code:
        source_vulns = submit_source_code_vulns(
            os.path.join(args.report_directory, "source_code/vulns.json"),
            **SUBMIT_KWARG,
        )
        if args.scan_mode != "release" and len(source_vulns) > 0:
            RISKY_DEPENDENCIES.append(f"{len(source_vulns)} new source code vulnerability")

        source_result = submit_source_code_licenses(
            os.path.join(args.report_directory, "source_code/sbom.json"),
            **SUBMIT_KWARG,
            license_check_token=LICENSE_CHECK_TOKEN,
        )
        if source_result is None:
            if args.scan_mode != "release":
                RISKY_DEPENDENCIES.append("source code SBOM not found")
        else:
            source_licenses, source_license_info = source_result
            source_entries = _license_docs_to_entries(
                source_licenses, "source_code", source_license_info
            )
            detected_licenses.extend(source_entries)
            preapproved_candidates.extend(
                _risks_to_preapproved_candidates(
                    source_licenses, "source_code_license", fallback_package_type="pypi"
                )
            )
            non_permissive = [e for e in source_entries if not e["is_permissive"]]
            if non_permissive:
                RISKY_DEPENDENCIES.append(
                    f"{len(non_permissive)} new source code non-permissive license"
                )

    if not args.skip_container:
        amd64_container_vulns = submit_container_vulns(
            os.path.join(args.report_directory, "release_amd64/vulns.json"),
            os.path.join(args.report_directory, "base_amd64/vulns.json"),
            "amd64",
            **SUBMIT_KWARG,
        )
        arm64_container_vulns = submit_container_vulns(
            os.path.join(args.report_directory, "release_arm64/vulns.json"),
            os.path.join(args.report_directory, "base_arm64/vulns.json"),
            "arm64",
            **SUBMIT_KWARG,
        )
        count_container_vulns = len(amd64_container_vulns) + len(arm64_container_vulns)
        if args.scan_mode != "release" and count_container_vulns > 0:
            RISKY_DEPENDENCIES.append(f"{count_container_vulns} new container vulnerability")

        amd64_container_licenses, amd64_license_info = submit_container_licenses(
            os.path.join(args.report_directory, "release_amd64/licenses.json"),
            os.path.join(args.report_directory, "base_amd64/licenses.json"),
            "amd64",
            **SUBMIT_KWARG,
            license_check_token=LICENSE_CHECK_TOKEN,
        )
        arm64_container_licenses, arm64_license_info = submit_container_licenses(
            os.path.join(args.report_directory, "release_arm64/licenses.json"),
            os.path.join(args.report_directory, "base_arm64/licenses.json"),
            "arm64",
            **SUBMIT_KWARG,
            license_check_token=LICENSE_CHECK_TOKEN,
        )
        container_license_entries = _license_docs_to_entries(
            amd64_container_licenses, "container_amd64", amd64_license_info
        ) + _license_docs_to_entries(
            arm64_container_licenses, "container_arm64", arm64_license_info
        )
        detected_licenses.extend(container_license_entries)
        preapproved_candidates.extend(
            _risks_to_preapproved_candidates(amd64_container_licenses, "container_license")
        )
        preapproved_candidates.extend(
            _risks_to_preapproved_candidates(arm64_container_licenses, "container_license")
        )
        non_permissive_container = [e for e in container_license_entries if not e["is_permissive"]]
        if non_permissive_container:
            RISKY_DEPENDENCIES.append(
                f"{len(non_permissive_container)} new container non-permissive license"
            )

    if RISKY_DEPENDENCIES:
        detail = ", ".join(RISKY_DEPENDENCIES)
        status = "unstable"
        if args.scan_mode == "monitor":
            post_slack_msg(args.build_number, args.ref, detail)

        result = {
            "status": status,
            "detail": detail,
            "risks": RISKY_DEPENDENCIES,
            "detected_licenses": detected_licenses,
            "dashboard_url": get_dashboard_url(args.build_number, args.ref),
        }
        if args.scan_mode == "release" and detected_licenses:
            result["needs_manual_review"] = True
            result["preapproved_candidates"] = preapproved_candidates
        return result
    else:
        return {"status": "success"}


if __name__ == "__main__":
    result = process_result()
    print(json.dumps(result))
