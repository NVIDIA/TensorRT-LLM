import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from elasticsearch import Elasticsearch, RequestsHttpConnection

# Define config
ES_POST_URL = os.environ.get("TRTLLM_ES_POST_URL")
ES_QUERY_URL = os.environ.get("TRTLLM_ES_QUERY_URL")
ES_INDEX_BASE = os.environ.get("TRTLLM_ES_INDEX_BASE") or ""
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

GPL_LICENSE_PREFIXES = ("GPL", "LGPL", "GCC GPL")

parser = argparse.ArgumentParser()
parser.add_argument("--build-url", required=True, help="Jenkins build URL")
parser.add_argument("--build-number", required=True, help="Jenkins build number")
parser.add_argument("--branch", required=True, help="Branch name that passed to the pipeline")
args = parser.parse_args()

NEWLY_REPORTED_DEPENDENCIES = {}

if not ES_POST_URL:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_POST_URL' is not set!")
if not ES_QUERY_URL:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_QUERY_URL' is not set!")
if not ES_INDEX_BASE:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_INDEX_BASE' is not set!")
if not SLACK_WEBHOOK_URL:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_PLC_WEBHOOK' is not set!")

ES_CLIENT = Elasticsearch(
    timeout=TIMEOUT, hosts=ES_QUERY_URL, connection_class=RequestsHttpConnection
)

NOW = datetime.now(timezone.utc)
SEVERITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
ES_HEADERS = {"Content-Type": "application/json"}


def safe(value, default=None):
    return value if value else default


def es_post(url, documents, headers):
    """POST a list of documents to an Elasticsearch index and return (indexed, errors)."""
    if not documents:
        return 0, False
    resp = requests.post(
        url.rstrip("/"),
        data=json.dumps(documents),
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    indexed = sum(
        1
        for item in result.get("items", [])
        if item.get("index", {}).get("result") in ("created", "updated")
    )
    errors = result.get("errors", False)
    if errors:
        failed = [
            item["index"] for item in result.get("items", []) if item.get("index", {}).get("error")
        ]
        print(f"Indexing errors ({len(failed)}):")
        for f in failed:
            print(f"  {f.get('_id')}: {f.get('error', {}).get('reason')}")
    return indexed, errors


def get_last_scan_results(report_type: str):
    data = ES_CLIENT.search(
        index=ES_INDEX_BASE + "-*",
        body={
            "size": 0,  # only the latest
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"s_type": report_type}},
                        {"term": {"s_branch": args.branch}},
                    ]
                }
            },
            "aggs": {"latest_ts": {"max": {"field": "ts_created"}}},
        },
    )
    if "aggregations" not in data or not data["aggregations"]["latest_ts"]["value"]:
        return {}
    latest_ts = data["aggregations"]["latest_ts"]["value"]
    scroll_size = 1000
    docs = []

    resp = ES_CLIENT.search(
        index=ES_INDEX_BASE + "-*",
        body={
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"s_type": report_type}},
                        {"term": {"s_branch": args.branch}},
                        {"term": {"ts_created": int(latest_ts)}},
                    ]
                }
            },
        },
        size=scroll_size,
        scroll="2m",
    )

    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]
    docs.extend(hits)

    while len(hits) > 0:
        resp = ES_CLIENT.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
        docs.extend(hits)

    detected_dependencies = {}
    for doc in docs:
        package_name = doc["_source"]["s_package_name"]
        if package_name not in detected_dependencies:
            detected_dependencies[package_name] = 1
        else:
            detected_dependencies[package_name] += 1
    return detected_dependencies


def process_vulnerability(input_file: str):
    # Read scan report
    vulnerabilities = json.loads(Path(input_file).read_text())
    map_dependencies_last_report = get_last_scan_results("source_code_vulnerability")

    bulk_documents = []

    for v in vulnerabilities:
        sev = v.get("Severity", "Low")
        if SEVERITY_RANK.get(sev, 0) <= 2:
            continue

        doc = {
            "ts_created": int(NOW.timestamp() * 1000),
            "s_type": "source_code_vulnerability",
            "s_run_date": NOW.strftime("%Y-%m-%d"),
            "s_build_url": args.build_url,
            "s_build_number": args.build_number,
            "s_branch": args.branch,
            "s_severity": safe(v.get("Severity")),
            "s_package_name": safe(v.get("Package Name")),
            "s_package_version": safe(v.get("Package Version")),
            "s_cve": safe(v.get("Related Vuln")),
            "s_bdsa": safe(v.get("CVE ID")),
            "d_score": safe(v.get("Score")),
            "s_status": safe(v.get("Status")),
            "s_published_date": safe(v.get("Vulnerability Published Date")),
            "s_upgrade_short_term": safe(v.get("Upgrade-Guidance", {}).get("Short-Term")),
            "s_upgrade_long_term": safe(v.get("Upgrade-Guidance", {}).get("Long-Term")),
        }

        # Elasticsearch bulk API: action metadata line + document line
        bulk_documents.append(doc)

    vulnerability_count = len(bulk_documents)
    if vulnerability_count:
        _, errors = es_post(ES_POST_URL, bulk_documents, ES_HEADERS)
        if errors:
            raise RuntimeError(
                "Elasticsearch bulk indexing reported errors for vulnerability report."
            )
    else:
        print("No High/Critical vulnerabilities found.")
    count_new_vulnerability = 0
    for doc in bulk_documents:
        dependency_name = doc["s_package_name"]
        if dependency_name not in map_dependencies_last_report:
            count_new_vulnerability += 1
    NEWLY_REPORTED_DEPENDENCIES["source_code_vulnerability"] = count_new_vulnerability


def process_sbom(input_file: str):
    sbom_documents = []
    sbom_path = Path(input_file)
    map_dependencies_last_report = get_last_scan_results("source_code_license")
    if sbom_path.exists():
        sbom_data = json.loads(sbom_path.read_text())
        for component in sbom_data.get("components", []):
            license_ids = []
            for lic_entry in component.get("licenses", []):
                lic = lic_entry.get("license", {})
                license_ids.append(lic.get("id") or lic.get("name") or "")
            gpl_licenses = [lid for lid in license_ids if lid.startswith(GPL_LICENSE_PREFIXES)]
            if not gpl_licenses:
                continue
            purl = component.get("purl", "")
            supplier = component.get("supplier", {}).get("name", "") or ""
            sbom_documents.append(
                {
                    "ts_created": int(NOW.timestamp() * 1000),
                    "s_type": "source_code_license",
                    "s_run_date": NOW.strftime("%Y-%m-%d"),
                    "s_build_url": args.build_url,
                    "s_build_number": args.build_number,
                    "s_branch": args.branch,
                    "s_package_name": component.get("name"),
                    "s_package_version": component.get("version"),
                    "s_purl": purl,
                    "s_supplier": supplier,
                    "s_license_ids": ",".join(gpl_licenses),
                    "s_bom_ref": component.get("bom-ref"),
                    "s_component_type": component.get("type"),
                }
            )
        _, sbom_errors = es_post(ES_POST_URL, sbom_documents, ES_HEADERS)
        if sbom_errors:
            raise RuntimeError(
                "Elasticsearch bulk indexing reported errors for SBOM license report."
            )
    else:
        print(f"SBOM file not found, skipping GPL/LGPL license reporting: {input_file}")
    count_new_vulnerability = 0
    for doc in sbom_documents:
        dependency_name = doc["s_package_name"]
        if dependency_name not in map_dependencies_last_report:
            count_new_vulnerability += 1
    NEWLY_REPORTED_DEPENDENCIES["source_code_license"] = count_new_vulnerability


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


process_vulnerability(SOURCE_CODE_VULNERABILITY)
process_sbom(SOURCE_CODE_SBOM)
post_slack_msg()
