# Perf Sanity Triage

This directory contains `perf_sanity_triage.py`, a helper script for querying
and updating perf sanity data in OpenSearch, and for sending regression
summaries to Slack.

## Basic Usage

This script is run by the Jenkins pipeline:
https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/TRTLLM-Perf/job/PerfSanityTriage/

Inputs are configured in `jenkins/runPerfSanityTriage.groovy`:

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

**Format**

```
UPDATE SET <field>=<value> [AND <field>=<value> ...] [WHERE <field>=<value> [AND <field>=<value> ...]]
```
