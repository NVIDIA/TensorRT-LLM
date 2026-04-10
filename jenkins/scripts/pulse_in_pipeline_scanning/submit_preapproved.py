import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ES_POST_URL = os.environ.get("TRTLLM_ES_PREAPPROVED_POST_URL")
if not ES_POST_URL:
    raise EnvironmentError("Environment variable 'TRTLLM_ES_PREAPPROVED_POST_URL' is not set.")


def es_post(url: str, documents: list) -> tuple[int, bool]:
    """POST documents to an Elasticsearch bulk endpoint."""
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


def load_preapproved(csv_path: str) -> list[dict]:
    """Parse pre_approved.csv and return a list of ES documents."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    preapproved_dependencies = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            package_name = row[0].strip() if len(row) > 0 else ""
            package_version = row[1].strip() if len(row) > 1 else ""
            if not package_name:
                continue
            preapproved_dependencies.append(
                {
                    "s_package_name": package_name,
                    "s_package_version": package_version or None,
                }
            )
    return [{"ts_created": ts, "preapproved_deps": preapproved_dependencies}]


def main():
    parser = argparse.ArgumentParser(
        description="Upload pre-approved license list to Elasticsearch."
    )
    parser.add_argument(
        "--csv",
        default="./pre_approved.csv",
        help="Path to pre_approved.csv (default: %(default)s)",
    )
    args = parser.parse_args()

    docs = load_preapproved(args.csv)
    print(f"Loaded {len(docs)} pre-approved entries from {args.csv}")
    print(docs)

    indexed, errors = es_post(ES_POST_URL, docs)
    if errors:
        print("ERROR: Elasticsearch reported indexing errors.", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully indexed {indexed} documents.")


if __name__ == "__main__":
    main()
