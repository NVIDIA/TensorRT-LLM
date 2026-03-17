# Copyright 2026, NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Submit container scan diff reports (High/Critical vulns + non-permissive licenses)
# to Elasticsearch. Reads the diff JSON files produced by process_scan_report.py.
#
# Usage:
#   python3 submit_container_scan_report.py \
#       --diff-dir <dir> --build-url <url> --build-number <n> --branch <branch>

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from elasticsearch import Elasticsearch, RequestsHttpConnection

ES_POST_URL = os.environ.get("TRTLLM_ES_POST_URL")
ES_QUERY_URL = os.environ.get("TRTLLM_ES_QUERY_URL")
ES_INDEX_BASE = os.environ.get("TRTLLM_ES_INDEX_BASE") or ""
ES_API_KEY = os.environ.get("TRTLLM_ES_API_KEY")
SLACK_WEBHOOK_URL = os.environ.get("TRTLLM_PLC_WEBHOOK")
TIMEOUT = 1000

parser = argparse.ArgumentParser(description="Submit container scan diff reports to Elasticsearch.")
parser.add_argument(
    "--diff-dir",
    required=True,
    help="Directory containing vuln_diff_*.json and license_diff_*.json",
)
parser.add_argument("--build-url", required=True, help="Jenkins build URL")
parser.add_argument("--build-number", required=True, help="Jenkins build number")
parser.add_argument("--branch", required=True, help="Branch name")
args = parser.parse_args()

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
ES_HEADERS = {"Content-Type": "application/json"}
if ES_API_KEY:
    ES_HEADERS["Authorization"] = f"ApiKey {ES_API_KEY}"

NEWLY_REPORTED = {}


def es_post(url, documents, headers):
    """POST a list of documents to an Elasticsearch index and return (indexed, errors)."""
    if not documents:
        return 0, False
    resp = requests.post(
        url.rstrip("/"),
        data=json.dumps(documents),
        headers={**headers, "Content-Type": "application/x-ndjson"},
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


def get_last_scan_packages(report_type):
    data = ES_CLIENT.search(
        index=ES_INDEX_BASE + "-*",
        body={
            "size": 0,
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
        return set()
    latest_ts = data["aggregations"]["latest_ts"]["value"]

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
        size=1000,
        scroll="2m",
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]
    docs = list(hits)
    while hits:
        resp = ES_CLIENT.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
        docs.extend(hits)

    return {doc["_source"]["s_package_name"] for doc in docs}


def process_vuln_diff(platform, diff_data, last_packages):
    docs = []
    release_image = diff_data.get("release_image", "")
    base_image = diff_data.get("base_image", "")
    for v in diff_data.get("added_in_release", []):
        docs.append(
            {
                "ts_created": int(NOW.timestamp() * 1000),
                "s_type": "container_vulnerability",
                "s_run_date": NOW.strftime("%Y-%m-%d"),
                "s_build_url": args.build_url,
                "s_build_number": args.build_number,
                "s_branch": args.branch,
                "s_platform": platform,
                "s_release_image": release_image,
                "s_base_image": base_image,
                "s_severity": v.get("severity"),
                "s_package_name": v.get("package_name"),
                "s_package_version": v.get("package_version"),
                "s_cve": v.get("vuln"),
                "s_package_paths": ",".join(v.get("package_paths", [])),
            }
        )
    if docs:
        _, errors = es_post(ES_POST_URL, docs, ES_HEADERS)
        if errors:
            raise RuntimeError(
                f"Elasticsearch indexing errors for container vuln diff ({platform})."
            )
    else:
        print(f"[{platform}] No High/Critical container vulnerabilities in diff.")

    new_count = sum(1 for d in docs if d["s_package_name"] not in last_packages)
    NEWLY_REPORTED[f"container_vulnerability_{platform}"] = new_count
    print(f"[{platform}] Uploaded {len(docs)} container vulnerability records ({new_count} new).")


def process_license_diff(platform, diff_data, last_packages):
    docs = []
    release_image = diff_data.get("release_image", "")
    base_image = diff_data.get("base_image", "")
    for e in diff_data.get("added_in_release", []):
        docs.append(
            {
                "ts_created": int(NOW.timestamp() * 1000),
                "s_type": "container_license",
                "s_run_date": NOW.strftime("%Y-%m-%d"),
                "s_build_url": args.build_url,
                "s_build_number": args.build_number,
                "s_branch": args.branch,
                "s_platform": platform,
                "s_release_image": release_image,
                "s_base_image": base_image,
                "s_package_name": e.get("package"),
                "s_package_version": e.get("version"),
                "s_package_type": e.get("type"),
                "s_license_ids": ",".join(e.get("licenses", [])),
            }
        )
    if docs:
        _, errors = es_post(ES_POST_URL, docs, ES_HEADERS)
        if errors:
            raise RuntimeError(
                f"Elasticsearch indexing errors for container license diff ({platform})."
            )
    else:
        print(f"[{platform}] No non-permissive container licenses in diff.")

    new_count = sum(1 for d in docs if d["s_package_name"] not in last_packages)
    NEWLY_REPORTED[f"container_license_{platform}"] = new_count
    print(f"[{platform}] Uploaded {len(docs)} container license records ({new_count} new).")


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

    lines = []
    for key, count in NEWLY_REPORTED.items():
        if not count:
            continue
        platform = key.split("_")[-1]
        if "vulnerability" in key:
            lines.append(f"- {count} new container High/Critical vulnerabilities ({platform})")
        elif "license" in key:
            lines.append(f"- {count} new container non-permissive licenses ({platform})")

    if not lines:
        return

    report = f"New TRTLLM container scan findings ({args.branch} branch)\n" + "\n".join(lines)
    slack_resp = requests.post(
        SLACK_WEBHOOK_URL,
        json={"report": report, "dashboardUrl": dashboard_link},
        timeout=60,
    )
    slack_resp.raise_for_status()


diff_dir = Path(args.diff_dir)
for platform in ["amd64", "arm64"]:
    vuln_path = diff_dir / f"vuln_diff_{platform}.json"
    license_path = diff_dir / f"license_diff_{platform}.json"

    if vuln_path.exists():
        last_vuln_pkgs = get_last_scan_packages("container_vulnerability")
        process_vuln_diff(platform, json.loads(vuln_path.read_text()), last_vuln_pkgs)
    else:
        print(f"[{platform}] Vuln diff file not found, skipping: {vuln_path}")

    if license_path.exists():
        last_lic_pkgs = get_last_scan_packages("container_license")
        process_license_diff(platform, json.loads(license_path.read_text()), last_lic_pkgs)
    else:
        print(f"[{platform}] License diff file not found, skipping: {license_path}")

post_slack_msg()
