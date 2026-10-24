You are executing ONE role turn for the `trtllm-modeling-v2` orchestrator: role **$role** ($what),
request `$request_id`. The standing instructions for every turn (working root, running GPU work,
the hard rules inside a session, handing back) are in `$instructions`; read them first.
$agent_clause
**Working root: `$root`.** Every relative path below is relative to it.

## Load your role

Read IN FULL, in this order:

1. `$system_prompt_file` --- your **system prompt**, binding for this turn.
2. `$prompt_file` --- your **user prompt** for this turn.
$history_clause
The run's skills are at `$skills`; read the ones your role prompt names.

## Do the work

Write every artifact to the exact path the user prompt names$decision_clause

Every spool job you submit is named `$job_prefix-<what>`:

```bash
export MV2FLOW_JOB_PREFIX=$job_prefix
```

`spoolctl.py cancel <name>` is the only way to stop a job.

## Hand the turn back

1. Write your final message to `$result_file`
2. THEN write the single character `1` to `$done_file`

Never write `.done` first, and never while a job you started is still running. A turn that ends
without `.done` hangs the run until its timeout ($timeout).
