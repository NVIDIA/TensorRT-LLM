import json
import os

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
        print(f"Indexing errors ({len(failed)}):")
        for f in failed:
            print(f"  {f.get('_id')}: {f.get('error', {}).get('reason')}")
    return indexed, errors


def get_last_scan_results(report_type: str, branch: str):
    data = ES_CLIENT.search(
        index=ES_INDEX_BASE + "-*",
        body={
            "size": 0,  # only the latest
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"s_type": report_type}},
                        {"term": {"s_branch": branch}},
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
                        {"term": {"s_branch": branch}},
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
        package_version = doc["_source"]["s_package_version"]
        if (package_name, package_version) not in detected_dependencies:
            detected_dependencies[(package_name, package_version)] = 1
        else:
            detected_dependencies[(package_name, package_version)] += 1
    return detected_dependencies


def get_latest_license_preapproved_container_deps():
    data = ES_CLIENT.search(
        index=ES_INDEX_PREAPPROVED_BASE + "-*",
        body={"size": 1, "sort": [{"ts_created": "desc"}], "_source": ["preapproved_deps"]},
    )
    data_source = data["hits"]["hits"][0]["_source"]
    if not data_source:
        return []
    return data_source["preapproved_deps"]
