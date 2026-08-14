import json
import os
import sys
from datetime import datetime, timezone
from urllib.parse import quote

import requests
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import scan as es_scan

ES_QUERY_URL = os.environ.get("TRTLLM_ES_QUERY_URL")
ES_INDEX_BASE = os.environ.get("TRTLLM_ES_INDEX_BASE") or ""
ES_INDEX_PREAPPROVED_BASE = "df-swdl-tensorrt-infra-plc-pre-approve"
ES_PREAPPROVED_POST_URL = os.environ.get("TRTLLM_ES_PREAPPROVED_POST_URL", "")

if not ES_QUERY_URL:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_QUERY_URL' is not set!")
if not ES_INDEX_BASE:
    raise EnvironmentError("Error: Environment variable 'TRTLLM_ES_INDEX_BASE' is not set!")

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


def get_preapproved_deps(scan_type: str) -> list[dict]:
    """Return preapproved dependency records for the given scan type.

    Queries all individual records stored by post_preapproved_deps and returns
    them as a list of dicts with s_package_name and s_package_type.
    """
    query = {"query": {"match": {"s_scan_type": scan_type}}}
    try:
        return [
            hit["_source"]
            for hit in es_scan(
                ES_CLIENT, index=ES_INDEX_PREAPPROVED_BASE + "-*", query=query, size=1000
            )
            if hit.get("_source")
        ]
    except Exception as exc:
        print(f"Failed to query preapproved deps for {scan_type}: {exc}", file=sys.stderr)
        return []


def get_triaged_deps(scan_type: str, branch: str, container: str = "") -> dict:
    """Return {package_name: ticket_url} for all packages that have a triage_record.

    When ``container`` is provided, only records scoped to that container image are returned.
    """
    filters = [
        {"term": {"s_type": "triage_record"}},
        {"term": {"s_scan_type": scan_type}},
        {"term": {"s_branch": branch}},
    ]
    if container:
        filters.append({"term": {"s_release_image": container}})
    try:
        resp = ES_CLIENT.search(
            index=ES_INDEX_BASE + "-*",
            body={
                "size": 10000,
                "query": {"bool": {"filter": filters}},
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

    Each record dict must have 'package_name' and 'ticket_url' keys, and optionally 'container'
    for records scoped to a specific container image.
    """
    docs = [
        {
            "s_type": "triage_record",
            "s_scan_type": scan_type,
            "s_branch": branch,
            "ts_created": ts_created,
            "s_package_name": rec["package_name"],
            "s_ticket_url": rec["ticket_url"],
            **({"s_release_image": rec["container"]} if rec.get("container") else {}),
        }
        for rec in records
        if rec.get("package_name") and rec.get("ticket_url")
    ]
    if not docs:
        return
    _, errors = es_post(post_url, docs)
    if errors:
        print(f"Failed to save some triage records for {scan_type}", file=sys.stderr)


def post_preapproved_deps(risk_docs: list[dict], fallback_package_type: str | None = None) -> bool:
    """POST each risk doc as an individual preapproved record to the preapproved index.

    Reads s_type, s_package_name, s_package_version, and s_package_type from each doc.
    When s_package_type is absent, fallback_package_type is used (e.g. "pypi" for source code).
    Returns True if all records were indexed successfully.
    """
    if not ES_PREAPPROVED_POST_URL:
        print(
            "TRTLLM_ES_PREAPPROVED_POST_URL not set, skipping preapproved indexing",
            file=sys.stderr,
        )
        return False
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_ok = True
    for doc in risk_docs:
        record = {
            "ts_created": ts,
            "s_scan_type": doc.get("s_type"),
            "s_package_name": doc.get("s_package_name"),
            "s_package_version": doc.get("s_package_version") or None,
            "s_package_type": doc.get("s_package_type") or fallback_package_type or None,
        }
        try:
            resp = requests.post(
                ES_PREAPPROVED_POST_URL.rstrip("/"),
                data=json.dumps(record),
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            outcome = result.get("status", "").lower()
            if outcome not in ("created", "updated"):
                print(
                    f"Unexpected preapproved result for {record['s_package_name']}: {outcome}",
                    file=sys.stderr,
                )
                all_ok = False
            else:
                print(f"Preapproved indexed: {record['s_package_name']} -> {outcome}")
        except Exception as exc:
            print(
                f"Failed to index preapproved dep {record['s_package_name']}: {exc}",
                file=sys.stderr,
            )
            all_ok = False
    return all_ok


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
