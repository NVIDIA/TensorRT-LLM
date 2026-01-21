#!/usr/bin/env python3

import argparse
import json
import re
import sys
import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

sys.path.insert(0, sys.path[0] + "/..")
from open_search_db import OpenSearchDB

QUERY_LOOKBACK_DAYS = 90
MAX_QUERY_SIZE = 3000
MAX_TEST_CASES_PER_MSG = 5
POST_SLACK_MSG_RETRY_TIMES = 5


def _parse_value(value):
    value = value.strip()
    if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in ("'", '"')):
        return value[1:-1]
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _split_and_clauses(text):
    return [
        part.strip() for part in re.split(r"\s+AND\s+", text, flags=re.IGNORECASE) if part.strip()
    ]


def _parse_assignments(text):
    clauses = _split_and_clauses(text)
    if not clauses:
        return None, "No fields provided"
    result = {}
    for clause in clauses:
        if "=" not in clause:
            return None, f"Invalid clause (missing '='): {clause}"
        key, value = clause.split("=", 1)
        key = key.strip()
        if not key:
            return None, f"Invalid clause (empty field name): {clause}"
        result[key] = _parse_value(value)
    return result, None


def parse_update_operation(operation):
    match = re.match(
        r"^\s*UPDATE\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?\s*$", operation, flags=re.IGNORECASE
    )
    if not match:
        return None, None, "Invalid UPDATE operation format"
    set_text = match.group(1).strip()
    where_text = match.group(2).strip() if match.group(2) else ""
    set_values, error = _parse_assignments(set_text)
    if error:
        return None, None, f"Invalid SET clause: {error}"
    where_values = {}
    if match.group(2) is not None:
        if not where_text:
            return None, None, "Invalid WHERE clause: empty scope"
        where_values, error = _parse_assignments(where_text)
        if error:
            return None, None, f"Invalid WHERE clause: {error}"
    return set_values, where_values, None


def update_perf_data_fields(data_list, set_values):
    updated_list = []
    for data in data_list:
        updated_data = data.copy()
        for key, value in set_values.items():
            updated_data[key] = value
        updated_list.append(updated_data)
    return updated_list


def post_perf_data(data_list, project_name):
    if not data_list:
        print(f"No data to post to {project_name}")
        return False
    try:
        print(f"Ready to post {len(data_list)} data to {project_name}")
        return OpenSearchDB.postToOpenSearchDB(data_list, project_name)
    except Exception as e:
        print(f"Failed to post data to {project_name}, error: {e}")
        return False


def get_regression_data_by_job_id(data_list, query_job_number):
    """Returns a dict with job_id as key and list of regression data as value.

    Only returns the latest query_job_number jobs.
    """
    if data_list is None or len(data_list) == 0:
        return {}

    # Group data by job_id
    job_data_dict = {}
    for data in data_list:
        job_id = data.get("s_job_id", "")
        if job_id == "":
            continue
        if job_id not in job_data_dict:
            job_data_dict[job_id] = []
        job_data_dict[job_id].append(data)

    # Sort job_ids by the latest ts_created in each group (descending)
    def get_latest_timestamp(job_id):
        timestamps = [d.get("ts_created", 0) for d in job_data_dict[job_id]]
        return max(timestamps) if timestamps else 0

    sorted_job_ids = sorted(job_data_dict.keys(), key=get_latest_timestamp, reverse=True)

    # Only keep the latest query_job_number jobs
    latest_job_ids = sorted_job_ids[:query_job_number]

    result = {}
    for job_id in latest_job_ids:
        result[job_id] = job_data_dict[job_id]

    return result


def process_regression_message(regression_dict):
    """Process regression data into message chunks.

    Returns a list of messages, each containing at most MAX_TEST_CASES_PER_MSG test cases.
    """
    if not regression_dict:
        return []

    # Flatten all test cases into a list with (job_id, idx, data) tuples
    all_test_cases = []
    for job_id, data_list in regression_dict.items():
        sorted_data_list = sorted(data_list, key=lambda x: x.get("s_test_case_name", ""))
        for idx, data in enumerate(sorted_data_list, start=1):
            all_test_cases.append((job_id, idx, data))

    # Split into chunks of MAX_TEST_CASES_PER_MSG
    chunks = []
    for i in range(0, len(all_test_cases), MAX_TEST_CASES_PER_MSG):
        chunks.append(all_test_cases[i : i + MAX_TEST_CASES_PER_MSG])

    # Build messages for each chunk
    messages = []
    for chunk in chunks:
        msg_parts = []
        current_job_id = None
        for job_id, idx, data in chunk:
            # Add job header when switching to a new job_id
            if job_id != current_job_id:
                if msg_parts:
                    msg_parts.append("\n")
                job_header = f"*LLM/main/L0_PostMerge/{job_id}:*\n"
                msg_parts.append(job_header)
                current_job_id = job_id

            test_case_name = data.get("s_test_case_name", "N/A")
            regression_info = data.get("s_regression_info", "N/A")
            msg_parts.append(f"*REGRESSION TEST CASE {idx}: {test_case_name}*\n")
            for part in regression_info.split(","):
                part = part.strip()
                if part and "baseline_id" not in part:
                    msg_parts.append(f"  {part}\n")

        msg = "".join(msg_parts).strip()
        messages.append(msg)

    return messages


def send_regression_message(messages, channel_id, bot_token):
    """Send regression messages to Slack channel(s).

    channel_id can be a single ID or multiple IDs separated by commas.
    """
    if not messages:
        print("No regression data to send")
        return

    if channel_id and bot_token:
        channel_ids = [cid.strip() for cid in channel_id.split(",") if cid.strip()]
        for cid in channel_ids:
            for msg in messages:
                send_message(msg, cid, bot_token)
    else:
        print("Slack channel_id or bot_token not provided, printing message:")
        for i, msg in enumerate(messages, start=1):
            print(f"--- Message {i} ---")
            print(msg)


def send_message(msg, channel_id, bot_token):
    """Send message to Slack channel using slack_sdk."""
    client = WebClient(token=bot_token)

    attachments = [
        {
            "title": "Perf Sanity Regression Report",
            "color": "#ff0000",
            "text": msg,
        }
    ]

    for attempt in range(1, POST_SLACK_MSG_RETRY_TIMES + 1):
        try:
            result = client.chat_postMessage(
                channel=channel_id,
                attachments=attachments,
            )
            assert result["ok"] is True, json.dumps(result.data)
            print(f"Message sent successfully to channel {channel_id}")
            return
        except SlackApiError as e:
            print(
                f"Attempt {attempt}/{POST_SLACK_MSG_RETRY_TIMES}: Error sending message to Slack: {e}"
            )
        except Exception as e:
            print(f"Attempt {attempt}/{POST_SLACK_MSG_RETRY_TIMES}: Unexpected error: {e}")

        if attempt < POST_SLACK_MSG_RETRY_TIMES:
            time.sleep(1)

    print(
        f"Failed to send message to channel {channel_id} after {POST_SLACK_MSG_RETRY_TIMES} attempts"
    )


def main():
    parser = argparse.ArgumentParser(description="Perf Sanity Triage Script")
    parser.add_argument("--project_name", type=str, required=True, help="OpenSearch project name")
    parser.add_argument("--operation", type=str, required=True, help="Operation to perform")
    parser.add_argument(
        "--channel_id",
        type=str,
        default="",
        help="Slack channel ID(s), comma-separated for multiple channels",
    )
    parser.add_argument("--bot_token", type=str, default="", help="Slack bot token")
    parser.add_argument(
        "--query_job_number", type=int, default=1, help="Number of latest jobs to query"
    )

    args = parser.parse_args()

    print(f"Project Name: {args.project_name}")
    print(f"Operation: {args.operation}")
    print(f"Channel ID: {args.channel_id}")
    print(f"Bot Token: {'***' if args.bot_token else 'Not provided'}")
    print(f"Query Job Number: {args.query_job_number}")

    if args.operation == "SLACK BOT SENDS MESSAGE":
        last_days = QUERY_LOOKBACK_DAYS
        must_clauses = [
            {"term": {"b_is_valid": True}},
            {"term": {"b_is_post_merge": True}},
            {"term": {"b_is_regression": True}},
            {"term": {"b_is_baseline": False}},
            {
                "range": {
                    "ts_created": {
                        "gte": int(time.time() - 24 * 3600 * last_days)
                        // (24 * 3600)
                        * 24
                        * 3600
                        * 1000,
                    }
                }
            },
        ]
        data_list = OpenSearchDB.queryPerfDataFromOpenSearchDB(
            args.project_name, must_clauses, size=MAX_QUERY_SIZE
        )
        if data_list is None:
            print("Failed to query regression data")
            return

        regression_dict = get_regression_data_by_job_id(data_list, args.query_job_number)
        messages = process_regression_message(regression_dict)
        send_regression_message(messages, args.channel_id, args.bot_token)
    elif args.operation.strip().upper().startswith("UPDATE"):
        set_values, where_values, error = parse_update_operation(args.operation)
        if error:
            print(error)
            return

        must_clauses = []
        for key, value in where_values.items():
            must_clauses.append({"term": {key: value}})

        data_list = OpenSearchDB.queryPerfDataFromOpenSearchDB(
            args.project_name, must_clauses, size=MAX_QUERY_SIZE
        )
        if data_list is None:
            print("Failed to query data for update")
            return
        if len(data_list) == 0:
            print("No data matched the update scope")
            return

        updated_data_list = update_perf_data_fields(data_list, set_values)
        if not post_perf_data(updated_data_list, args.project_name):
            print("Failed to post updated data")
            return
        print(f"Updated {len(updated_data_list)} entries successfully")
    else:
        print(f"Unknown operation: {args.operation}")


if __name__ == "__main__":
    main()
