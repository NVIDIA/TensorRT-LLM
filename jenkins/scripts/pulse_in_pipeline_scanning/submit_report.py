import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict

sys.path.append(os.path.abspath(".."))
from utils.common import is_permissive, load_json
from utils.es import es_post, get_triaged_deps, save_triage_records
from utils.report import diff_licenses, get_preapproved_deps_map, get_vulns, is_preapproved
from utils.triage import call_triage_agent, extract_ticket_refs, format_risks_for_agent

ES_POST_URL = os.environ.get("TRTLLM_ES_POST_URL", "")
if not ES_POST_URL:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_POST_URL' is not set!")

SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


class BuildMetadata(TypedDict):
    build_url: str
    build_number: str
    ref: str
    platform: NotRequired[str]


def safe(value, default=None):
    return value if value else default


def _run_triage(risk_docs: list, scan_type: str, branch: str, ts_created: int) -> dict:
    """Call triage agent for untriaged risk_docs; persist ticket records; return {pkg: ticket_url}."""
    if not risk_docs:
        return {}
    print(
        format_risks_for_agent(
            [],
            risk_docs,
        )
    )
    is_vuln = "vulnerability" in scan_type
    agent_resp = call_triage_agent(
        format_risks_for_agent(
            risk_docs if is_vuln else [],
            risk_docs if not is_vuln else [],
        )
    )
    print(agent_resp)
    if not agent_resp:
        return {}
    ticket_refs = extract_ticket_refs(agent_resp)
    new_tickets = {}
    for item in ticket_refs.get("vulnerability", []):
        pkg, url = item.get("dependency_name"), item.get("ticket_url")
        if pkg and url:
            new_tickets[pkg] = url
    license_ticket = ticket_refs.get("license")
    if license_ticket and license_ticket.get("ticket_url"):
        url = license_ticket["ticket_url"]
        for pkg in license_ticket.get("dependencies") or []:
            new_tickets[pkg.get("name")] = url
    if new_tickets:
        save_triage_records(
            ES_POST_URL,
            scan_type,
            branch,
            ts_created,
            [{"package_name": k, "ticket_url": v} for k, v in new_tickets.items()],
        )
    return new_tickets


def submit_source_code_vulns(
    input_file: str,
    build_metadata: BuildMetadata,
    start_datetime: datetime,
) -> list:
    """Triage untriaged source-code vulnerabilities, save all docs to ES, return untriaged risks."""
    SCAN_TYPE = "source_code_vulnerability"
    triaged_deps = get_triaged_deps(SCAN_TYPE, build_metadata["ref"])
    ts = int(start_datetime.timestamp() * 1000)

    bulk_documents = []
    risks_to_report = []
    vulns_path = Path(input_file)
    if vulns_path.exists():
        vulnerabilities = json.loads(vulns_path.read_text())

        for v in vulnerabilities:
            sev = v.get("Severity", "Low")
            if SEVERITY_RANK.get(sev, 0) <= 2:
                continue
            package_name = safe(v.get("Package Name"))
            package_version = safe(v.get("Package Version"))
            doc = {
                "ts_created": ts,
                "s_type": SCAN_TYPE,
                "s_run_date": start_datetime.strftime("%Y-%m-%d"),
                "s_build_url": build_metadata["build_url"],
                "s_build_number": build_metadata["build_number"],
                "s_branch": build_metadata["ref"],
                "s_severity": safe(v.get("Severity")),
                "s_package_name": package_name,
                "s_package_version": package_version,
                "s_cve": safe(v.get("Related Vuln", "N/A")),
                "s_bdsa": safe(v.get("CVE ID"), "N/A"),
                "d_score": safe(v.get("Score")),
                "s_status": safe(v.get("Status")),
                "s_published_date": safe(v.get("Vulnerability Published Date")),
                "s_package_fix_version": (
                    safe(v.get("Upgrade-Guidance", {}).get("Long-Term"))
                    or safe(v.get("Upgrade-Guidance", {}).get("Short-Term"))
                    or "N/A"
                ),
                "s_license_ids": "N/A",
                "s_ticket_url": triaged_deps.get(package_name, "N/A"),
            }
            if package_name not in triaged_deps:
                risks_to_report.append(doc)
            bulk_documents.append(doc)
        if risks_to_report:
            new_tickets = _run_triage(risks_to_report, SCAN_TYPE, build_metadata["ref"], ts)
            print(new_tickets)
            for doc in bulk_documents:
                if doc["s_package_name"] in new_tickets:
                    doc["s_ticket_url"] = new_tickets[doc["s_package_name"]]

        if bulk_documents:
            _, errors = es_post(ES_POST_URL, bulk_documents)
            if errors:
                raise RuntimeError(
                    "Elasticsearch bulk indexing reported errors for vulnerability report."
                )
    else:
        print(
            f"Vulnerability result json not found, vulnerability reporting: {input_file}",
            file=sys.stderr,
        )

    return risks_to_report


def submit_source_code_licenses(
    input_file: str,
    build_metadata: BuildMetadata,
    start_datetime: datetime,
    license_check_token: str,
) -> list | None:
    """Triage untriaged source-code license risks, save all docs to ES, return untriaged risks.

    Returns None if the SBOM file is missing.
    """
    SCAN_TYPE = "source_code_license"
    triaged_deps = get_triaged_deps(SCAN_TYPE, build_metadata["ref"])
    ts = int(start_datetime.timestamp() * 1000)

    map_preapproved = get_preapproved_deps_map(SCAN_TYPE)
    sbom_documents = []
    risks_to_report = []
    sbom_path = Path(input_file)
    if sbom_path.exists():
        sbom_data = json.loads(sbom_path.read_text())
        components = sbom_data.get("components", [])

        # Build {license_id: [components]} and call the API once for all licenses
        license_to_components = {}
        for component in components:
            for lic_entry in component.get("licenses", []):
                lic = lic_entry.get("license", {})
                lid = lic.get("id") or lic.get("name") or ""
                license_to_components.setdefault(lid, []).append(component)

        permissiveness = is_permissive(list(license_to_components.keys()), license_check_token)

        non_permissive_pkgs = {
            c.get("name")
            for lid, perm in permissiveness.items()
            if not perm
            for c in license_to_components[lid]
        } | {c.get("name") for c in license_to_components.get("", [])}

        for component in components:
            package_name = component.get("name")
            package_version = component.get("version")
            component_licenses = component.get("licenses", [])
            if component_licenses and package_name not in non_permissive_pkgs:
                continue
            license_ids = [
                lic_entry.get("license", {}).get("id")
                or lic_entry.get("license", {}).get("name")
                or ""
                for lic_entry in component_licenses
            ]
            purl = component.get("purl", "")
            supplier = component.get("supplier", {}).get("name", "") or ""
            doc = {
                "ts_created": ts,
                "s_type": SCAN_TYPE,
                "s_run_date": start_datetime.strftime("%Y-%m-%d"),
                "s_build_url": build_metadata["build_url"],
                "s_build_number": build_metadata["build_number"],
                "s_branch": build_metadata["ref"],
                "s_package_name": package_name,
                "s_package_version": package_version,
                "s_purl": purl,
                "s_supplier": supplier,
                "s_package_fix_version": "N/A",
                "s_cve": "N/A",
                "s_bdsa": "N/A",
                "s_license_ids": ",".join(license_ids),
                "s_bom_ref": component.get("bom-ref"),
                "s_component_type": component.get("type"),
                "s_ticket_url": triaged_deps.get(package_name, "N/A"),
            }
            if package_name not in triaged_deps and not is_preapproved(
                map_preapproved, package_name, (component.get("type") or "unknown").lower()
            ):
                risks_to_report.append(doc)
            sbom_documents.append(doc)

        if risks_to_report:
            new_tickets = _run_triage(risks_to_report, SCAN_TYPE, build_metadata["ref"], ts)
            print(new_tickets)
            for doc in sbom_documents:
                if doc["s_package_name"] in new_tickets:
                    doc["s_ticket_url"] = new_tickets[doc["s_package_name"]]

        if sbom_documents:
            _, sbom_errors = es_post(ES_POST_URL, sbom_documents)
            if sbom_errors:
                raise RuntimeError(
                    "Elasticsearch bulk indexing reported errors for SBOM license report."
                )
    else:
        print(
            f"SBOM file not found, skipping GPL/LGPL license reporting: {input_file}",
            file=sys.stderr,
        )
        return None

    return risks_to_report


def submit_container_vulns(
    input_file: str,
    base_input_file: str,
    platform: str,
    build_metadata: BuildMetadata,
    start_datetime: datetime,
) -> list:
    """Triage untriaged container vulnerabilities, save all docs to ES, return untriaged risks."""
    SCAN_TYPE = "container_vulnerability"
    triaged_deps = get_triaged_deps(SCAN_TYPE, build_metadata["ref"])
    ts = int(start_datetime.timestamp() * 1000)

    release_data = load_json(input_file)
    base_data = load_json(base_input_file)
    trtllm_deps = get_vulns(input_file)

    docs = []
    risks_to_report = []
    release_image = release_data.get("image_tag", "")
    base_image = base_data.get("image_tag", "")
    for v in trtllm_deps:
        package_name = v.get("package_name")
        package_version = v.get("package_version")
        doc = {
            "ts_created": ts,
            "s_type": SCAN_TYPE,
            "s_run_date": start_datetime.strftime("%Y-%m-%d"),
            "s_build_url": build_metadata["build_url"],
            "s_build_number": build_metadata["build_number"],
            "s_branch": build_metadata["ref"],
            "s_platform": platform,
            "s_release_image": release_image,
            "s_base_image": base_image,
            "s_severity": v.get("severity"),
            "s_package_name": package_name,
            "s_package_version": package_version,
            "s_package_fix_version": v.get("fix") or "N/A",
            "s_license_ids": "N/A",
            "s_cve": v.get("vuln"),
            "s_bdsa": "N/A",
            "s_cve_url": v.get("url"),
            "s_package_paths": ",".join(v.get("package_paths", [])),
            "s_ticket_url": triaged_deps.get(package_name, ""),
        }
        if package_name not in triaged_deps:
            risks_to_report.append(doc)
        docs.append(doc)
    if risks_to_report:
        new_tickets = _run_triage(risks_to_report, SCAN_TYPE, build_metadata["ref"], ts)
        for doc in docs:
            if doc["s_package_name"] in new_tickets:
                doc["s_ticket_url"] = new_tickets[doc["s_package_name"]]

    if docs:
        _, errors = es_post(ES_POST_URL, docs)
        if errors:
            raise RuntimeError(
                f"Elasticsearch indexing errors for container vulnerability ({release_image})."
            )
    else:
        print(f"No High/Critical container vulnerabilities in {release_image}.", file=sys.stderr)

    return risks_to_report


def submit_container_licenses(
    input_file: str,
    base_input_file: str,
    platform: str,
    build_metadata: BuildMetadata,
    start_datetime: datetime,
    license_check_token: str,
) -> list:
    """Triage untriaged container license risks, save all docs to ES, return untriaged risks."""
    SCAN_TYPE = "container_license"
    triaged_deps = get_triaged_deps(SCAN_TYPE, build_metadata["ref"])
    ts = int(start_datetime.timestamp() * 1000)

    release_data = load_json(input_file)
    base_data = load_json(base_input_file)
    trtllm_deps = diff_licenses(SCAN_TYPE, input_file, base_input_file)

    map_preapproved = get_preapproved_deps_map(SCAN_TYPE)

    docs = []
    risks_to_report = []
    release_image = release_data.get("image_tag", "")
    base_image = base_data.get("image_tag", "")
    # Build {license_id: [deps]} and call the API once for all detected licenses
    license_to_deps = {}
    for v in trtllm_deps:
        for lid in v.get("licenses", []):
            license_to_deps.setdefault(lid, []).append(v)

    permissiveness = is_permissive(list(license_to_deps.keys()), license_check_token)

    non_permissive_pkgs = {
        dep.get("package")
        for lid, perm in permissiveness.items()
        if not perm
        for dep in license_to_deps[lid]
    }

    for v in trtllm_deps:
        package_name = v.get("package")
        package_version = v.get("version")
        license_ids = v.get("licenses", [])
        if license_ids and package_name not in non_permissive_pkgs:
            continue
        doc = {
            "ts_created": ts,
            "s_type": SCAN_TYPE,
            "s_run_date": start_datetime.strftime("%Y-%m-%d"),
            "s_build_url": build_metadata["build_url"],
            "s_build_number": build_metadata["build_number"],
            "s_branch": build_metadata["ref"],
            "s_platform": platform,
            "s_release_image": release_image,
            "s_base_image": base_image,
            "s_package_name": package_name,
            "s_package_version": package_version,
            "s_package_type": v.get("type"),
            "s_cve": "N/A",
            "s_bdsa": "N/A",
            "s_package_fix_version": "N/A",
            "s_license_ids": ",".join(license_ids),
            "s_ticket_url": triaged_deps.get(package_name, ""),
        }
        if package_name not in triaged_deps and not is_preapproved(
            map_preapproved, package_name, (v.get("type") or "unknown").lower()
        ):
            risks_to_report.append(doc)
        docs.append(doc)

    if risks_to_report:
        new_tickets = _run_triage(risks_to_report, SCAN_TYPE, build_metadata["ref"], ts)
        for doc in docs:
            if doc["s_package_name"] in new_tickets:
                doc["s_ticket_url"] = new_tickets[doc["s_package_name"]]

    if docs:
        _, errors = es_post(ES_POST_URL, docs)
        if errors:
            raise RuntimeError(
                f"Elasticsearch indexing errors for container licenses ({release_image})."
            )
    else:
        print(f"No non-permissive licenses in {release_image}.", file=sys.stderr)

    return risks_to_report
