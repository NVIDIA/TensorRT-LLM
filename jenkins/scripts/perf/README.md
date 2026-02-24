# Perf Sanity Triage

This directory contains `perf_sanity_triage.py`, a helper script for querying
and updating perf sanity data in OpenSearch, and for sending regression
summaries to Slack.

## Basic Usage

This script is run by the Jenkins pipeline. Inputs are configured in `jenkins/runPerfSanityTriage.groovy`:

- `BRANCH`: repo branch to checkout
- `OPEN_SEARCH_PROJECT_NAME`: OpenSearch project name
- `OPERATION`: operation to perform (see Operations below)
- `QUERY_JOB_NUMBER`: number of latest jobs to query (OPERATION = "SLACK BOT SENDS MESSAGE" only)
- `SLACK_CHANNEL_ID`: Slack channel IDs (OPERATION = "SLACK BOT SENDS MESSAGE" only)
- `SLACK_BOT_TOKEN`: Slack bot token (OPERATION = "SLACK BOT SENDS MESSAGE" only)

## Operations

### 1) `SLACK BOT SENDS MESSAGE`

Queries regression data (post-merge only) and sends a formatted summary to
Slack. The query filters for:

- `b_is_valid = true`
- `b_is_post_merge = true`
- `b_is_regression = true`
- `b_is_baseline = false`

**Format**

```
SLACK BOT SENDS MESSAGE
```

### 2) `UPDATE SET ... (WHERE ...)`

Updates fields on existing perf records that match a query scope and posts the
updated documents back to OpenSearch.

**Operators**

- SET clause: Only `=` is supported.
- WHERE clause: Supports `=`, `!=`, `>`, `<`, `>=`, `<=` operators.
- `=` and `!=` operators are allowed for all fields.
- `>`, `<`, `>=`, `<=` operators are only allowed for `ts_created` field (timestamp) or fields starting with `d_` (double type) or `l_` (integer type).

**ts_created Date Formats**

The `ts_created` field accepts date strings in the following formats:
- `'Feb 18, 2026 @ 22:32:02.960'` (with milliseconds)
- `'Feb 18, 2026 @ 22:32:02'` (without milliseconds)
- `'2026/02/18'` (date only)

**Note:** All date strings are interpreted as UTC for consistent timestamp conversion across different environments.

**Examples**

```
UPDATE SET b_is_valid=false WHERE s_test_case_name='test1'
UPDATE SET b_is_valid=false WHERE s_gpu_type!='H100'
UPDATE SET b_is_valid=false WHERE d_latency > 100.5 AND l_count >= 10
UPDATE SET b_is_valid=false WHERE ts_created <= 'Feb 18, 2026 @ 22:32:02.960' AND s_test_case_name='test1'
```
