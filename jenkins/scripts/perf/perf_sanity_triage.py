#!/usr/bin/env python3

import argparse
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, sys.path[0] + "/..")
from open_search_db import OpenSearchDB

MAX_QUERY_SIZE = 3000

# Comparison operators (order matters: >= before >, <= before <, != before =)
COMPARISON_OPERATORS = [">=", "<=", "!=", ">", "<", "="]
COMPARISON_ALLOWED_PREFIXES = ("d_", "l_")
COMPARISON_ALLOWED_FIELDS = ("ts_created",)


def _parse_date_string(date_str):
    """Convert date string like 'Feb 18, 2026 @ 22:32:02.960' to millisecond timestamp.

    All date strings are interpreted as UTC to ensure consistent timestamps
    across different environments/timezones.
    """
    date_str = date_str.strip()
    # Try format: "Feb 18, 2026 @ 22:32:02.960"
    try:
        dt = datetime.strptime(date_str, "%b %d, %Y @ %H:%M:%S.%f")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass
    # Try format: "Feb 18, 2026 @ 22:32:02"
    try:
        dt = datetime.strptime(date_str, "%b %d, %Y @ %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass
    # Try format: "2026/02/18"
    try:
        dt = datetime.strptime(date_str, "%Y/%m/%d")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass
    raise ValueError(f"Unable to parse date string: {date_str}")


def _can_use_comparison_operator(field_name):
    """Check if a field can use comparison operators (>, <, >=, <=)."""
    if field_name in COMPARISON_ALLOWED_FIELDS:
        return True
    if field_name.startswith(COMPARISON_ALLOWED_PREFIXES):
        return True
    return False


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


def _parse_where_clauses(text):
    """Parse WHERE clauses supporting =, >, <, >=, <= operators.

    Returns a list of tuples: (field_name, operator, value)
    Only d_*, l_*, and ts_created fields can use comparison operators.
    """
    clauses = _split_and_clauses(text)
    if not clauses:
        return None, "No fields provided"

    result = []
    for clause in clauses:
        # Match: field_name <operator> value
        # Using regex to find operator right after field name, avoiding false matches in values
        m = re.match(r"^\s*(\w+)\s*(>=|<=|!=|>|<|=)\s*(.*)", clause)
        if not m:
            return None, f"Invalid clause (missing operator): {clause}"

        key = m.group(1).strip()
        found_op = m.group(2)
        value = _parse_value(m.group(3))

        if not key:
            return None, f"Invalid clause (empty field name): {clause}"

        # Check if comparison operator is allowed for this field
        # != is allowed for all fields, but >, <, >=, <= are restricted
        if found_op not in ("=", "!=") and not _can_use_comparison_operator(key):
            return None, (
                f"Comparison operator '{found_op}' not allowed for field '{key}'. "
                f"Only fields starting with 'd_', 'l_', or field 'ts_created' can use >, <, >=, <= operators."
            )

        # Convert date string to timestamp for ts_created
        if key == "ts_created" and isinstance(value, str):
            try:
                value = _parse_date_string(value)
            except ValueError as e:
                return None, str(e)

        result.append((key, found_op, value))

    return result, None


def _build_opensearch_clause(field, operator, value):
    """Build OpenSearch query clause from field, operator, and value.

    Returns a tuple (clause_type, clause) where clause_type is "must" or "must_not".
    """
    if operator == "=":
        return ("must", {"term": {field: value}})

    if operator == "!=":
        return ("must_not", {"term": {field: value}})

    op_map = {
        ">": "gt",
        "<": "lt",
        ">=": "gte",
        "<=": "lte",
    }
    return ("must", {"range": {field: {op_map[operator]: value}}})


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
    where_clauses = []
    if match.group(2) is not None:
        if not where_text:
            return None, None, "Invalid WHERE clause: empty scope"
        where_clauses, error = _parse_where_clauses(where_text)
        if error:
            return None, None, f"Invalid WHERE clause: {error}"
    return set_values, where_clauses, None


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


def main():
    parser = argparse.ArgumentParser(description="Perf Sanity Data Update Script")
    parser.add_argument("--project_name", type=str, required=True, help="OpenSearch project name")
    parser.add_argument("--commands", type=str, required=True, help="UPDATE commands, one per line")

    args = parser.parse_args()

    print(f"Project Name: {args.project_name}")

    commands = [line.strip() for line in args.commands.strip().splitlines() if line.strip()]
    if not commands:
        print("No commands provided")
        return

    print(f"Number of commands: {len(commands)}")

    for i, command in enumerate(commands, start=1):
        print(f"\n--- Command {i}: {command} ---")
        if not command.strip().upper().startswith("UPDATE"):
            print(f"Skipping non-UPDATE command: {command}")
            continue

        set_values, where_clauses, error = parse_update_operation(command)
        if error:
            print(f"Error: {error}")
            continue

        must_clauses = []
        must_not_clauses = []
        for field, operator, value in where_clauses:
            clause_type, clause = _build_opensearch_clause(field, operator, value)
            if clause_type == "must":
                must_clauses.append(clause)
            else:
                must_not_clauses.append(clause)

        data_list = OpenSearchDB.queryPerfDataFromOpenSearchDB(
            args.project_name, must_clauses, size=MAX_QUERY_SIZE, must_not_clauses=must_not_clauses
        )
        if data_list is None:
            print("Failed to query data for update")
            continue
        if len(data_list) == 0:
            print("No data matched the update scope")
            continue

        updated_data_list = update_perf_data_fields(data_list, set_values)
        if not post_perf_data(updated_data_list, args.project_name):
            print("Failed to post updated data")
            continue
        print(f"Updated {len(updated_data_list)} entries successfully")


if __name__ == "__main__":
    main()
