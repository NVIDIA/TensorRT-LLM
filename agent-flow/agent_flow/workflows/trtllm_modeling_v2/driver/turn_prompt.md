You are executing ONE role turn for the `trtllm-modeling-v2` orchestrator, which is running as a
plain Python process that cannot authenticate a child agent on this host. It has written your
turn to disk and is blocked waiting for your answer. You are the transport and nothing else: you
do the role's work, you do not second-guess the orchestrator.

**Working root: `$root`.** Every relative path in the files below is relative to that directory.
Use absolute paths or `git -C`; do not `cd` into subdirectories of the repository.

Your role for this turn is **$role** ($what), request `$request_id`.
$agent_clause
## Step 1 --- load your role

Read these two files IN FULL before doing anything else:

1. `$system_prompt_file`
   --- your **system prompt**. Every instruction in it is binding on you for this turn, exactly
   as if it had been installed as your system prompt.
2. `$prompt_file`
   --- your **user prompt** for this turn.
$history_clause
The run's skills are at `$skills`. They are not auto-loaded for you, so read the ones your role
prompt names as ordinary files. The run's standing instructions are in `$instructions`.

## Step 2 --- do the work

Do exactly what the user prompt asks, writing every artifact to the exact path it names$decision_clause

**Every spool job you submit is named `$job_prefix-<what>` (or `free-$job_prefix-<what>`).**
Export it before you submit anything, and the submit script enforces it:

```bash
export MV2FLOW_JOB_PREFIX=$job_prefix
```

A name outside your prefix is refused, so you can never overwrite another turn's results.
Never kill, `pkill` or time out a process yourself; `spoolctl.sh cancel <name>` is the only way to
stop a job.
$embargo_clause
## Step 3 --- hand the turn back

This is how the orchestrator unblocks, so do it last and do it exactly:

1. Write your final message --- the prose you would have ended the turn with --- to
   `$result_file`
2. THEN write the single character `1` to
   `$done_file`

Order matters: `.done` is the completion signal, so a reader that sees it must be able to trust
the result beside it is complete. Never write `.done` first. Do not hand back while a job you
started is still running: wait for it, or cancel it and say so.

If you are blocked, still write both files: put the blocker in the result file and, if the role
prompt defines a decision value for that case, use it. A turn that ends without `.done` hangs the
run until its timeout ($timeout).

Report back in three lines: the decision you wrote, the artifacts you created, and anything the
orchestrator should know that did not fit in the decision file.
