# Operating a run: a guide for the agent session that watches it

A `trtllm-modeling-v2` run is watched by an outer agent session: the one that started it. The
flow has no operator of its own and nothing to configure for one. The workers solve what they
can; what a worker cannot solve reaches you as a blocked task and its question; what you cannot solve you put to
a human, as one question. This guide is the whole protocol. It names no cluster, account or
path: what your site adds (queue limits, which QoS to switch to, where anchors come from) is
your operations notes' to say.

## The loop

```
trtllm-modeling-v2 monitor --workspace <ws> wait
```

It blocks until something needs you (at most nine minutes, then `TIMEOUT`) and prints the
event on its first line, then the run's status. Act on the first word, then wait again.
Never cancel a job the run submitted and never run a replacement check beside it: a cancelled
check proves nothing and is run again.

| First word | What it means | What you do |
|---|---|---|
| `TASK_BLOCKED <task> <file>` | a task is blocked; every task that does not depend on it goes on. The line says who blocked it, what was observed, what it asks and what it holds; `<file>` is the receipt its worker left | read the line and `<file>`, then one `resolve` verb on `<task>` (below) |
| `RUN_BLOCKED <file>` | a final pass did not hold and no task owns the failure; `<file>` is the run's `journal.md`, whose last `[final]` lines say what each requested check and the final tests came to. The controller stays fifteen minutes for an answer, then exits 76 | read the line and `<file>`, then `resolve retry run` (the pass again, or with `--task T` to send a task back) or `change-requirement run` |
| `TASK_ASKS <task> <file>` | a task is done and still asks: a doubt its work did not wait for (a requirement that looks wrong). It holds nothing | read the line and `<file>`. A question whose answer changes nothing takes no verb: say the ruling with `feedback` and pass over the question with `wait --ignore <task>`. A requirement that is mistaken is `change-requirement`, which does not send the task back. `retry` is its work again |
| `TURN_LONG <task> <file>` | an agent has been in one turn for an hour; `<file>` is the turn's activity log (`<ws>/turns/<turn>.activity.log`). It holds nothing and judges nothing: a long turn may be honest work. It comes last, after every other event | read the end of `<file>`: is the worker still on what its task is for? If not, `resolve skip <task>` or `resolve retry <task> --note ...` ends the turn (below). Either way `wait --ignore <turn>` passes over it; the line names `<turn>` |
| `POOL_SUBMIT_FAILED` | a holder was asked for and did not come: the scheduler refused it, or did not answer. The line has the scheduler's own text | put right what it names if it is yours (an account, a partition, a limit), then `resolve pool-retry pool --note ...`; `--close-unknown SEQ` first closes a submission record you verified no job exists for. `wait --ignore POOL_SUBMIT_FAILED` passes over it while you have it in hand |
| `POOL_CAPACITY` | the spec's numbers do not allow the holder the work needs: a wall no holder can fit, or the bound of node-hours (`environment.slurm.max_node_hours`, 48 unless stated). At the bound nothing more is asked for; what runs finishes, and once no holder is left the controller exits 76 | `pool status --workspace <ws>` says what was asked for. If the run deserves it: `resolve pool-adjust pool --set <key>=<value> --note ...` (`max_node_hours=<more>` for the bound; a controller that runs takes it on its next pass, one that has exited is started again with the same command). If not: `pool stop --workspace <ws>` |
| `POOL_DEVICE_HELD` | devices are held and serve nothing: memory on a free device that no job of the pool accounts for, or a holder that runs and whose watchers never came up. Nothing is killed. On Slurm a holder that ready work waits on for such memory is given back and asked for again without its nodes, three times in a run | if it still stands, look at the node: the memory will not go by waiting. `wait --ignore POOL_DEVICE_HELD` passes over it |
| `ORCHESTRATOR_EXITED` | the controller is gone; the detail says how | exit 0: the run closed; read the closing report's first line and its section "The final pass" (each requested check: passed, failed, not verified or not run, and under which key; the final tests with each failing case's result at the base) (`delivered`; `partial`: every task ended, and a task was skipped or a check could not be verified, each listed; `delivery-failed`: what kept it is listed). Exit 76 (`escalated`): everything left waited for an answer and none came. The controller stays fifteen minutes for one before it leaves, and its journal says until when (`[loop] everything left waits for an answer ... takes one until HH:MM:SSZ`): an answer given in that time is taken by the same controller (`resolve` says `left for the running controller`) and the run goes on in the job it has. After it: resolve what is open, then run the same command again. Exit 75: it stopped for the limit of the job it runs in; the same command, run again, goes on (the detail has it) |
| `TIMEOUT` | nothing new | wait again |

`monitor --workspace <ws> wait --ignore <task>` passes over a blocked task you already have in
hand (one you asked a human about; `run` and a pool event's name are taken too) and returns
the next thing. The workspace is named before
`wait`; `$MV2FLOW_WORKSPACE` names it when the option is left out.

The flow submits no job for its controller. Where a site's script runs the command inside
a job, that is another node than the host a dry run was typed on, and its first minutes are
the proof: an `ORCHESTRATOR_EXITED` soon after the start is a start that did not start.
Read the last section of `<ws>/journal.md` (`preflight failed` or `agent probe failed`, with
the reason) before you go by the job's state in the queue.

## A blocked task

A blocked task is its own record. The graph (`<ws>/graph.yaml`) says it is `blocked`, on
which attempt, how that attempt ended, who blocked it (a worker's blocked receipt, a check
or a used attempt budget, a landing) and as what, and what it asks. Its receipt
(`<ws>/tasks/<task>.<attempt>.receipt.yaml`) is what its worker left: what was `observed`
against what was `expected`, the `command` that reproduces it and its `result` line, what
was `tried` and `ruled_out`, the `next_question`, and the pin. `monitor --workspace <ws>
status` lists everything that waits for an answer, one line each. There is no other record
and nothing has an id: a block is named by its task.

Some kinds, and what each asks of you:

| Kind | What happened | What usually answers it |
|---|---|---|
| `blocked` | the worker's receipt says it cannot go on, with its reproducer and its question | whatever its question needs: `retry` with what you found, `edit-env`, `split` |
| `attempts_exhausted` | the task's attempts are spent; `observed` has the last failure | `retry` (one more, with a note), `split`, or a changed requirement when the failure is the requirement's |
| `write_scope` | its files are outside what the run may commit, or its commits reached outside its scope | `allow-shared --path` |
| `skeleton_failed` | the shared package initializers its files need could not be committed; `observed` has git's reason | fix the cause it names, then `retry`: it is asked for again |
| `final_check_failed` | the final pass measured a requested check on the frozen candidate and it failed; its task is blocked on it | `retry` the task that must repair it (the repair is a new candidate, and the final pass runs on that one), or a changed requirement when the failure is the requirement's |
| `final_check_not_run` | a requested check left no result on the candidate (its job was lost, or it measured another tree) | fix what the line names, then `retry run`: the final pass asks again |
| `final_tests_not_run` | the final tests' job ended without a result for a selected test and no case failed (the driver's own traceback, a job that never started); the question quotes the job's last line | look at the environment first (the job's log in the spool), then `retry run`: the final pass runs again on the target as it is, and no task is sent back |
| `final_tests_failed` | the final tests ran once on the frozen candidate and a case that passed at the base fails, or a new case fails; `observed` lists each with its result on the candidate and at the base. A case that fails at the base too is reported and fails nothing | `retry run --task` the task that must repair it: the repair is a new candidate, and the final tests run on that one |
| `not_delivered` | every requested check passed on the candidate and its delivery files do not check out | read which file, then `retry run --task` the task that owns it |

A check that ran, failed nothing and could not verify what it is for blocks nothing: the
work lands, its task is done and lists the check as not verified, `onboarded` and the closing
report say so, and the run closes `partial`. A check that failed, or that did not run, lands
nothing.

An answer is for the attempt its task is blocked on now. One that reopens another task
(`retry <task> --task T`) answers that task too, if it was blocked. An answer to an attempt
its task has moved on from changes nothing, and says so.

A worker's `next_question` begins with the word that names what it thinks answers it
(`allow-shared --path ...:`, `change-requirement:`, `retry --task ...:`, or `feedback:` for a
ruling that changes nothing, which is no verb of `resolve`), then its question. That is the
worker's suggestion, made from inside one task: check it against the evidence, then run that
verb or a better one. A verb answers one kind of question, and a suggested verb may be the
wrong kind:

| The question asks for | What answers it |
|---|---|
| a ruling that changes no requirement, from a task that is blocked | `retry <task> --note "<the ruling>"`: one more attempt with it, at no cost to its attempts |
| a ruling that changes no requirement, from a task that is done (`TASK_ASKS`) | no verb. `feedback --workspace <ws> "<the ruling>"` puts it where every later turn reads it, and `wait --ignore <task>` passes over the question. `retry` would run the task's work again (a whole survey, for a survey), and `change-requirement` with nothing to change is refused |
| a row of the survey that is wrong or lacks a cell | `split <task> --rows rows.yaml`: write the row as it should be, one row is enough. `change-requirement` changes the spec, never a row |
| a requirement of the spec that is mistaken (an anchor, an option, a deployment's tests) | `change-requirement` with `--set key=value`, or `--deployment D --benchmark B --anchor V` |

Read the evidence the file names before you choose. Separate three things: the environment
or the pool (a temporary directory, a cache size, a dataset mount, a queue limit, a broken
node); the work (a task too large for one turn, a fix that belongs in shared code, a cell
nobody needs); and a requirement that is itself mistaken (an anchor measured another way, a
protocol that does not fit the model). Do not repeat what `ruled_out` already lists.

## The verbs

```
trtllm-modeling-v2 resolve --workspace <ws> <verb> <task|run|pool> ...
```

A verb names what it answers: a blocked task, `run` (a final pass that did not hold), or
`pool`. `retry` and `skip` also name a task that is running: you are the one who sees
that a turn has left what its task is for, and you need not wait for it to stop talking.
The controller ends the turn (the agent's process, as at the turn's own limit) and then
applies the verb. Nothing of the ended turn lands: `skip` leaves its tree as it is,
unmerged; `retry` starts the next attempt in that same tree, with your note. The ended
attempt costs the task none of its attempts. A job the worker's own check had queued is
not withdrawn: it runs to its end and its result is under the pin it measured. A task
whose turn is over and whose checks run has no turn to end: the answer waits for the
checks. Every verb takes its reason (`--note`, or `--reason`). With no controller running,
this command takes the workspace lock and applies the verb itself. With one running it
leaves the answer beside the task's receipt (`<ws>/tasks/<name>.answer.yaml`); the
controller applies it on its next pass, says in that file what became of it, and this
command prints that. A later answer to the same block replaces one not applied yet. Exit 0
applied, 1 not applied (with why), 3 left for the controller and not applied yet. An answer
is applied whole or not at all: one that is refused (a key that cannot be set, a split that
makes a cycle) changes nothing and records nothing.

| Verb | Arguments | What it does |
|---|---|---|
| `retry` | `<task>` or `run`; `--note`, `[--task T]` | one more attempt, past the budget if need be (an attempt that ended in its worker's own question, once you answer it with any verb, costs the task none of its attempts); the note is appended to the notes file every worker reads first. `--task` reopens another task (the one whose code must change) and the blocked one waits for it, also where that task is the child of a split |
| `edit-env` | `<task>` or `run`; `--set key=value ...`, `--note` | changes the environment in the spec (`environment.env.<NAME>`, a deployment's `env` or `timeout`, a cache size) and retries |
| `split` | `<task>`; `--rows rows.yaml`, `--note` | the task becomes an ordinary join of the typed rows you wrote (each: `id`, `kind`, `outcome`, `required`; an `operator` row also `target` and `mode`); each row is a task with its own card. A row may state its check's job (`check: {gpus: N, wall: SECONDS}`): one row that restates the task with a longer `wall` is how a check that was cut at its wall gets more time. A row with no check of its own says `check: {id: ""}` |
| `allow-shared` | `<task>`; `--path <repository path>`, `--note` | widens the task's write scope (and the run's commit policy when it must) and retries; the tests shared code it changes reaches run once, in the final pass |
| `skip` | `<task>`; `--reason`, `[--nice-to-have]` | the task is not done and is reported `NOT DONE`. With `--nice-to-have` a required task is made one that was only nice to have (a row nobody asked for): what depended on it goes on, and the run is not held to it. What depended on a required task does not start: nothing produced what it needs. Skip those too, or resolve them another way; left alone, the run closes `partial` with them listed. What depended on a task that was only nice to have goes on and is told so |
| `change-requirement` | `<task>` or `run`; `--deployment D --benchmark B --anchor V [--unit U]` for an anchor, `--set key=value ...` for any other requirement (a floor or tolerance, a protocol, an option, the Modeling V2 requirement); `--reason` | changes the requirement and records it. An anchor is changed where the run froze it (a run freezes its anchors at its first start, so a spec key alone moves none; `--anchor none` takes one away). The checks it judged are judged again: a pass by the old requirement no longer counts, and the session is queued for a new load. One key of an options file is `--set deployments.<d>.options.<key>=<value>` (`<d>.ctx.options` or `<d>.gen.options` for a side): it is set in that file, so the file goes on saying what runs |
| `pool-retry` | `pool`; `--note`, `[--close-unknown SEQ]` | asks for the holder again now; `--close-unknown` first closes a submission record you verified no job exists for |
| `pool-adjust` | `pool`; `--set <key>=<value> ...`, `--note` | changes the pool's values in the spec and has the pool take them. On Slurm the keys are `environment.slurm`'s, by their own names: `qos`, `time` (written `04:00:00`: a value is read as YAML, where `4:00:00` is a number), `partition`; `max_node_hours` (the bound a `POOL_CAPACITY` names). |

No verb is forbidden and no value is refused. A requirement that is mistaken, must change or
rests on wrong information may be changed: the first principle is to onboard early. What the
flow owes in return is the record. Every change of the spec is appended to
`<ws>/spec_changes.jsonl`, and every changed anchor to `<ws>/anchor_changes.jsonl`, with what
changed, from what, to what and why; the closing report (`<ws>/delivery_report.md`) lists each
beside the result, so `onboarded` always says under which requirements.
The file the user wrote is never rewritten: a later start puts the recorded changes back over it.

`trtllm-modeling-v2 feedback --workspace <ws> "<note>"` appends to `<ws>/operator_notes.md`
without resolving anything. Every worker's card tells it to read that file first and again
before its receipt. Use it for what you learned that every later turn should know.

## What a worker was told, and how to answer it

A worker's card (the start of `<ws>/turns/<n>-<role>.prompt.md`) names its outcome, what is
required and what is optional, what to search for, reference code to look for, its write
scope and its one check. It names no place in the engine: the tree moves, and a weak model
follows a path it is given rather than search. Answer in the same terms.

| The worker says | What it means | What usually answers it |
|---|---|---|
| a name its card gave was not found, after wider searches | the engine renamed or moved what the row assumed | search yourself (`git grep` in its tree); then `retry` with a note that names what the thing is called now and what it does. Say the commit when you name a file: a location is evidence about one commit |
| an optional line failed | nothing: optional lines never block, and belong under `not_done` | if it blocked on one all the same, `retry` with a note saying so |
| a required cell fails and the first deployment does not need it | the row asked for more than the spec owes | `retry` with a note naming the cells to leave out, or `split` the row |
| it needs a file another row owns | the rows were cut too fine, or one is missing | `split` with the missing row, or `allow-shared --path` |
| the engine's Python is not in its shell | expected: it runs in jobs of the pool | nothing; if its job imported the installed engine instead of its tree, the note is "`--tree <your tree>`" |
| its row needs native code changed | every tree links the one native build the run started from, and no run rebuilds it | make the change and rebuild the base yourself, start the run again from it, then `retry` with a note that says what is built now |

Rows you write for `split` are read by a worker as a card: an `outcome`, what is `required`
and what is `optional`, and `leads`: names to search for with why each may fit. Only
`writes` holds paths: the files the row owns.

## When the engine's tree has moved

At each start the run writes `<ws>/tree_map.md`: what each thing it reads off the engine was
found by, where, and at which commit. The journal says `tree map (missing: ...)` when a
definition found nothing in this checkout. A missing one costs what it is read for (the
engine's own model or a limit is not named before the survey); it stops nothing. The
engine renamed the class, the registration or the message the definition looks for: find the
new name, tell the workers in a note, and report it so that the definition
(`operators/tree_anchors.py`) is updated.

## Putting a question to a human

When you cannot choose (a requirement only its owner can judge, an allocation you are not
entitled to, two fixes with different costs), ask in your own conversation, once, with:

1. **Observed against expected**, verbatim from the block's line and its receipt: the failing line or score,
   and the contract or floor it missed.
2. **What was ruled out**, by the worker and by you, each with its evidence.
3. **The smallest reproducing command** (the receipt's `command`, or a narrower one you
   ran), with the pin it ran on.
4. **The choices**, each as the exact `resolve` command it would run and its consequence:
   what is measured again, what is reported not done, what requirement changes.

Then `monitor --workspace <ws> wait --ignore <task>` while you wait for the answer: the rest of
the run goes on.
