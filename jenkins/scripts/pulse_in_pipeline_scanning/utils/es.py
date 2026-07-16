import json
import os
import sys
from datetime import datetime, timezone
from urllib.parse import quote

import requests
from elasticsearch import Elasticsearch, RequestsHttpConnection

ES_QUERY_URL = os.environ.get("TRTLLM_ES_QUERY_URL")
ES_INDEX_BASE = os.environ.get("TRTLLM_ES_INDEX_BASE") or ""
ES_INDEX_PREAPPROVED_BASE = os.environ.get("TRTLLM_ES_INDEX_PREAPPROVED_BASE") or ""

if not ES_QUERY_URL:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_QUERY_URL' is not set!")
if not ES_INDEX_BASE:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_INDEX_BASE' is not set!")
if not ES_INDEX_PREAPPROVED_BASE:
    raise EnvironmentError(
        "Error: Environment variable 'TRTLLM_ES_INDEX_PREAPPROVED_BASE' is not set!"
    )

TIMEOUT = 1000
ES_CLIENT = Elasticsearch(
    timeout=TIMEOUT, hosts=ES_QUERY_URL, connection_class=RequestsHttpConnection
)


def es_post(url, documents):
    """POST a list of documents to an Elasticsearch index and return (indexed, errors)."""
    if not documents:
        return 0, False
    resp = requests.post(
        url.rstrip("/"),
        data=json.dumps(documents),
        headers={"Content-Type": "application/json"},
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
        print(f"Indexing errors ({len(failed)}):", file=sys.stderr)
        for f in failed:
            print(f"  {f.get('_id')}: {f.get('error', {}).get('reason')}", file=sys.stderr)
    return indexed, errors


def get_latest_license_preapproved_container_deps(scan_type: str):
    data = ES_CLIENT.search(
        index=ES_INDEX_PREAPPROVED_BASE + "-*",
        body={
            "size": 1,
            "sort": [{"ts_created": "desc"}],
            "query": {
                "bool": {
                    "should": [
                        {"match": {"s_scan_type": scan_type}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "_source": ["nested_preapproved_deps"],
        },
    )
    data_source = data["hits"]["hits"][0]["_source"]
    if not data_source:
        return []
    return data_source["nested_preapproved_deps"]


def get_triaged_deps(scan_type: str, branch: str) -> dict:
    """Return {package_name: ticket_url} for all packages that have a triage_record."""
    try:
        resp = ES_CLIENT.search(
            index=ES_INDEX_BASE + "-*",
            body={
                "size": 10000,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"s_type": "triage_record"}},
                            {"term": {"s_scan_type": scan_type}},
                            {"term": {"s_branch": branch}},
                        ]
                    }
                },
                "_source": ["s_package_name", "s_ticket_url"],
            },
        )
    except Exception as exc:
        print(f"Failed to query triaged deps for {scan_type}: {exc}", file=sys.stderr)
        return {}
    result = {}
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        pkg = src.get("s_package_name")
        ticket = src.get("s_ticket_url")
        if pkg and ticket:
            result[pkg] = ticket
    return result


def save_triage_records(
    post_url: str,
    scan_type: str,
    branch: str,
    ts_created: int,
    records: list,
) -> None:
    """Persist (package_name, ticket_url) triage records so future runs can skip re-triage.

    Each record dict must have 'package_name' and 'ticket_url' keys.
    """
    docs = [
        {
            "s_type": "triage_record",
            "s_scan_type": scan_type,
            "s_branch": branch,
            "ts_created": ts_created,
            "s_package_name": rec["package_name"],
            "s_ticket_url": rec["ticket_url"],
        }
        for rec in records
        if rec.get("package_name") and rec.get("ticket_url")
    ]
    if not docs:
        return
    _, errors = es_post(post_url, docs)
    if errors:
        print(f"Failed to save some triage records for {scan_type}", file=sys.stderr)


def get_dashboard_url(build_number, branch):
    starttime = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    base = (
        "https://gpuwa.nvidia.com/kibana/s/tensorrt/app/dashboards"
        "#/view/4969f302-2d26-4a4f-bc80-3b69c4626945"
    )
    start_iso = starttime.replace(tzinfo=None).isoformat()
    g = f"(filters:!(),refreshInterval:(pause:!t,value:60000),time:(from:'{start_iso}Z',to:now))"
    a = f"(query:(language:kuery,query:'s_build_number:{build_number} and s_branch:\"{branch}\"'))"
    dashboard_link = f"{base}?_g={quote(g)}&_a={quote(a)}"
    return dashboard_link
