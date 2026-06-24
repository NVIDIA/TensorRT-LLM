# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import argparse
import csv
import glob
import json
import os
import time

# Default CSV columns from pytest-csv:
#   id, module, name, file, doc, markers, status, message, duration
STATUS_COLUMN = 6
DURATION_COLUMN = 8

OPENSEARCH_INDEX = "df-swdl-trtllm-infra-ci-prod-test_info-*"


def query_opensearch_durations(url, credentials, days):
    import requests

    since_ms = int((time.time() - days * 86400) * 1000)
    search_url = f"{url.rstrip('/')}/{OPENSEARCH_INDEX}/_search"

    auth = None
    if credentials and ':' in credentials:
        user, password = credentials.split(':', 1)
        auth = (user, password)

    test_durations = {}
    after_key = None
    page = 0

    while True:
        composite_agg = {
            "size": 1000,
            "sources": [{
                "test_name": {
                    "terms": {
                        "field": "s_test_name"
                    }
                }
            }]
        }
        if after_key is not None:
            composite_agg["after"] = after_key

        query = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [{
                        "term": {
                            "s_status": "PASSED"
                        }
                    }, {
                        "range": {
                            "ts_created": {
                                "gte": since_ms
                            }
                        }
                    }],
                    "must_not": [{
                        "term": {
                            "s_test_name": "Stage Failed"
                        }
                    }]
                }
            },
            "aggs": {
                "by_test": {
                    "composite": composite_agg,
                    "aggs": {
                        "avg_duration_ms": {
                            "avg": {
                                "field": "l_e2e_time_ms"
                            }
                        }
                    }
                }
            }
        }

        response = requests.post(search_url, json=query, auth=auth, timeout=60)
        response.raise_for_status()
        data = response.json()

        buckets = data["aggregations"]["by_test"]["buckets"]
        page += 1
        print(f"  Page {page}: got {len(buckets)} buckets")

        for bucket in buckets:
            test_name = bucket["key"]["test_name"]
            avg_ms = bucket["avg_duration_ms"]["value"]
            if avg_ms is not None:
                test_durations[test_name] = avg_ms / 1000.0

        after_key = data["aggregations"]["by_test"].get("after_key")
        if not after_key or len(buckets) == 0:
            break

    return test_durations


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate test duration file.")
parser.add_argument(
    "--duration-file",
    type=str,
    default="new_test_duration.json",
    help="Path to the output duration file (default: new_test_duration.json)")
parser.add_argument(
    "--cluster",
    type=str,
    default=None,
    help="Cluster name (e.g. 'aws_dfw').  When set, writes "
    "tests/integration/defs/.test_durations_<cluster> relative to the "
    "repo root instead of --duration-file.")
parser.add_argument("--from-opensearch",
                    action="store_true",
                    default=False,
                    help="Query OpenSearch instead of reading local CSV files.")
parser.add_argument(
    "--days",
    type=int,
    default=3,
    help="Number of days to look back in OpenSearch (default: 3).")
parser.add_argument(
    "--opensearch-url",
    type=str,
    default=None,
    help=
    "OpenSearch base URL. Defaults to OPEN_SEARCH_DB_BASE_URL env var or http://gpuwa.nvidia.com/opensearch."
)
parser.add_argument(
    "--opensearch-credentials",
    type=str,
    default=None,
    help=
    "OpenSearch credentials as 'user:password'. Defaults to OPEN_SEARCH_DB_CREDENTIALS env var."
)
args = parser.parse_args()

# Resolve output path
if args.cluster:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NEW_TEST_DURATION = os.path.join(_repo_root, "tests", "integration", "defs",
                                     f".test_durations_{args.cluster}")
else:
    NEW_TEST_DURATION = args.duration_file

if args.from_opensearch:
    opensearch_url = (args.opensearch_url
                      or os.environ.get("OPEN_SEARCH_DB_BASE_URL")
                      or "http://gpuwa.nvidia.com/opensearch")
    opensearch_creds = (args.opensearch_credentials
                        or os.environ.get("OPEN_SEARCH_DB_CREDENTIALS") or "")

    print(
        f"Querying OpenSearch at {opensearch_url} for last {args.days} day(s)..."
    )
    test_durations = query_opensearch_durations(opensearch_url,
                                                opensearch_creds, args.days)

    with open(NEW_TEST_DURATION, 'w') as file:
        json.dump(test_durations, file, indent=3)

    print("\nSummary:")
    print(f"  OpenSearch URL         : {opensearch_url}")
    print(f"  Days looked back       : {args.days}")
    print(f"  Unique tests in output : {len(test_durations)}")
    print(f"  Output written to      : {NEW_TEST_DURATION}")

else:
    # CSV mode: read from local report.csv files produced by the CI run
    TEST_RESULTS_DIR = os.getcwd()
    all_csv_files = sorted(
        glob.glob(os.path.join(TEST_RESULTS_DIR, '*/report.csv')))

    print(f"TEST_RESULTS_DIR: {TEST_RESULTS_DIR}")
    print(f"Found {len(all_csv_files)} CSV report file(s)")

    test_durations = {}
    passed_count = 0
    skipped_count = 0

    for report_csv in all_csv_files:
        print(f"Processing {report_csv}...")
        try:
            with open(report_csv, 'r', newline='') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if len(row) <= max(STATUS_COLUMN, DURATION_COLUMN):
                        continue

                    status = row[STATUS_COLUMN].strip()
                    if status != 'passed':
                        skipped_count += 1
                        continue

                    test_id = row[0].strip()
                    duration_str = row[DURATION_COLUMN].strip()

                    # Remove stage name prefix (everything up to first '/')
                    test_name = test_id.split('/', 1)[-1]

                    try:
                        duration = float(duration_str)
                    except ValueError:
                        # Fall back to last column if the fixed index doesn't work
                        try:
                            duration = float(row[-1].strip())
                        except ValueError:
                            print(
                                f"  Warning: Could not parse duration for {test_name}. Skipping."
                            )
                            continue

                    test_durations[test_name] = duration
                    passed_count += 1
        except (OSError, csv.Error) as e:
            print(f"  Warning: Failed to process {report_csv}: {e}")

    with open(NEW_TEST_DURATION, 'w') as file:
        json.dump(test_durations, file, indent=3)

    print("\nSummary:")
    print(f"  CSV files processed    : {len(all_csv_files)}")
    print(f"  Passed rows collected  : {passed_count}")
    print(f"  Non-passed rows skipped: {skipped_count}")
    print(f"  Unique tests in output : {len(test_durations)}")
    print(f"  Output written to      : {NEW_TEST_DURATION}")
