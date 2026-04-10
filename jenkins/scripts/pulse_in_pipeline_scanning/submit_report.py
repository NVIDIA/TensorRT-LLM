import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict

sys.path.append(os.path.abspath(".."))
from utils.common import load_json
from utils.es import es_post
from utils.report import diff_licenses, diff_vulns

ES_POST_URL = os.environ.get("TRTLLM_ES_POST_URL")
if not ES_POST_URL:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_POST_URL' is not set!")

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
GPL_LICENSE_PREFIXES = ("GPL", "LGPL", "GCC GPL")


class BuildMetadata(TypedDict):
    build_url: str
    build_number: str
    branch: str
    platform: NotRequired[str]


def safe(value, default=None):
    return value if value else default


def submit_source_code_vulns(
    input_file: str, last_scan_result: dict, build_metadata: BuildMetadata, start_datetime: datetime
):
    # Read scan report

    bulk_documents = []
    new_items = []
    vulns_path = Path(input_file)
    if vulns_path.exists():
        vulnerabilities = json.loads(vulns_path.read_text())

        for v in vulnerabilities:
            sev = v.get("Severity", "Low")
            if SEVERITY_RANK.get(sev, 0) <= 2:
                continue
            package_name = safe(v.get("Package Name"))
            package_version = safe(v.get("Package Version"))
            result_key = (package_name, package_version)
            is_new = result_key not in last_scan_result
            doc = {
                "ts_created": int(start_datetime.timestamp() * 1000),
                "s_type": "source_code_vulnerability",
                "s_run_date": start_datetime.strftime("%Y-%m-%d"),
                "s_build_url": build_metadata["build_url"],
                "s_build_number": build_metadata["build_number"],
                "s_branch": build_metadata["branch"],
                "s_severity": safe(v.get("Severity")),
                "s_package_name": package_name,
                "s_package_version": package_version,
                "s_cve": safe(v.get("Related Vuln")),
                "s_bdsa": safe(v.get("CVE ID")),
                "d_score": safe(v.get("Score")),
                "s_status": safe(v.get("Status")),
                "s_published_date": safe(v.get("Vulnerability Published Date")),
                "s_upgrade_short_term": safe(v.get("Upgrade-Guidance", {}).get("Short-Term")),
                "s_upgrade_long_term": safe(v.get("Upgrade-Guidance", {}).get("Long-Term")),
                "b_is_new": is_new,
            }
            if is_new:
                new_items.append(doc)

            bulk_documents.append(doc)

        vulnerability_count = len(bulk_documents)
        if vulnerability_count:
            _, errors = es_post(ES_POST_URL, bulk_documents)
            if errors:
                raise RuntimeError(
                    "Elasticsearch bulk indexing reported errors for vulnerability report."
                )
    else:
        print(f"Vulnerability result json not found, vulnerability reporting: {input_file}")

    return new_items


def submit_source_code_licenses(
    input_file: str, last_scan_result: dict, build_metadata: BuildMetadata, start_datetime: datetime
):
    sbom_documents = []
    new_items = []
    sbom_path = Path(input_file)
    if sbom_path.exists():
        sbom_data = json.loads(sbom_path.read_text())
        for component in sbom_data.get("components", []):
            license_ids = []
            package_name = component.get("name")
            package_version = component.get("version")
            for lic_entry in component.get("licenses", []):
                lic = lic_entry.get("license", {})
                license_ids.append(lic.get("id") or lic.get("name") or "")
            gpl_licenses = [lid for lid in license_ids if lid.startswith(GPL_LICENSE_PREFIXES)]
            if not gpl_licenses:
                continue
            result_key = (package_name, package_version)
            is_new = result_key not in last_scan_result
            purl = component.get("purl", "")
            supplier = component.get("supplier", {}).get("name", "") or ""
            doc = {
                "ts_created": int(start_datetime.timestamp() * 1000),
                "s_type": "source_code_license",
                "s_run_date": start_datetime.strftime("%Y-%m-%d"),
                "s_build_url": build_metadata["build_url"],
                "s_build_number": build_metadata["build_number"],
                "s_branch": build_metadata["branch"],
                "s_package_name": package_name,
                "s_package_version": package_version,
                "s_purl": purl,
                "s_supplier": supplier,
                "s_license_ids": ",".join(gpl_licenses),
                "s_bom_ref": component.get("bom-ref"),
                "s_component_type": component.get("type"),
                "b_is_new": is_new,
            }
            if is_new:
                new_items.append(doc)
            sbom_documents.append(doc)
        if sbom_documents:
            _, sbom_errors = es_post(ES_POST_URL, sbom_documents)
            if sbom_errors:
                raise RuntimeError(
                    "Elasticsearch bulk indexing reported errors for SBOM license report."
                )
    else:
        print(f"SBOM file not found, skipping GPL/LGPL license reporting: {input_file}")

    return new_items


def submit_container_vulns(
    input_file: str,
    base_input_file: str,
    platform: str,
    last_scan_result: dict,
    build_metadata: BuildMetadata,
    start_datetime: datetime,
):
    release_data = load_json(input_file)
    base_data = load_json(base_input_file)
    trtllm_deps = diff_vulns(input_file)

    docs = []
    new_items = []
    release_image = release_data.get("image_tag", "")
    base_image = base_data.get("image_tag", "")
    for v in trtllm_deps:
        package_name = v.get("package_name")
        package_version = v.get("package_version")
        result_key = (package_name, package_version)
        is_new = result_key not in last_scan_result
        doc = {
            "ts_created": int(start_datetime.timestamp() * 1000),
            "s_type": "container_vulnerability",
            "s_run_date": start_datetime.strftime("%Y-%m-%d"),
            "s_build_url": build_metadata["build_url"],
            "s_build_number": build_metadata["build_number"],
            "s_branch": build_metadata["branch"],
            "s_platform": platform,
            "s_release_image": release_image,
            "s_base_image": base_image,
            "s_severity": v.get("severity"),
            "s_package_name": package_name,
            "s_package_version": package_version,
            "s_cve": v.get("vuln"),
            "s_package_paths": ",".join(v.get("package_paths", [])),
            "b_is_new": is_new,
        }
        if is_new:
            new_items.append(doc)
        docs.append(doc)
    if docs:
        _, errors = es_post(ES_POST_URL, docs)
        if errors:
            raise RuntimeError(
                f"Elasticsearch indexing errors for container vulnerability ({release_image})."
            )
    else:
        print(f"No High/Critical container vulnerabilities in {release_image}.")

    return new_items


def submit_container_licenses(
    input_file: str,
    base_input_file: str,
    platform: str,
    last_scan_result: dict,
    build_metadata: BuildMetadata,
    start_datetime: datetime,
):
    release_data = load_json(input_file)
    base_data = load_json(base_input_file)
    trtllm_deps = diff_licenses(input_file, base_input_file)

    docs = []
    new_items = []
    release_image = release_data.get("image_tag", "")
    base_image = base_data.get("image_tag", "")
    for v in trtllm_deps:
        package_name = v.get("package_name")
        package_version = v.get("package_version")
        result_key = (package_name, package_version)
        is_new = result_key not in last_scan_result
        doc = {
            "ts_created": int(start_datetime.timestamp() * 1000),
            "s_type": "container_license",
            "s_run_date": start_datetime.strftime("%Y-%m-%d"),
            "s_build_url": build_metadata["build_url"],
            "s_build_number": build_metadata["build_number"],
            "s_branch": build_metadata["branch"],
            "s_platform": platform,
            "s_release_image": release_image,
            "s_base_image": base_image,
            "s_package_name": package_name,
            "s_package_version": package_version,
            "s_package_type": v.get("type"),
            "s_license_ids": ",".join(v.get("licenses", [])),
            "b_is_new": is_new,
        }
        if is_new:
            new_items.append(doc)
        docs.append(doc)
    if docs:
        _, errors = es_post(ES_POST_URL, docs)
        if errors:
            raise RuntimeError(
                f"Elasticsearch indexing errors for container licenses ({release_image})."
            )
    else:
        print(f"No non-permissive licenses in {release_image}.")

    return new_items
