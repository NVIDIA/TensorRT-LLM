# trtllm-modeling-v2

Brings one HuggingFace checkpoint to a gated **ModelingV2** target: a flat,
self-contained forward assembled only from catalog entries, trusted through
accuracy gates rather than through shared abstractions. It can then tune the
target's serving performance without making it less correct.

This package is the agent half of modeling_v2. The code half is in tree at
`tensorrt_llm/_torch/_experimental/modeling_v2/`: the catalog, the targets, the
router and their tests. This package builds them: a Python orchestrator, a
planner agent that writes a task graph, worker agents on one prompt, the
contract's gates, and the compute-session substrate the gates run on.

## Quick start

```bash
trtllm-modeling-v2 --task task.yaml --workspace workspace/trtllm-modeling-v2 --dry-run
trtllm-modeling-v2 --task task.yaml --workspace workspace/trtllm-modeling-v2
```

[`task.example.yaml`](task.example.yaml) documents every key of the spec.
`--dry-run` runs preflight, resolves the launch environment and stops before
the first agent turn: the cheapest way to learn whether this host can host
this run. Re-running the same command resumes; `--clean` starts over. Launch
from a plain terminal or a tmux session, not from inside a Claude Code
session; from Codex, see [user guide §8](docs/user-guide.md#from-a-codex-session).

Or let the run derive its task from the checkpoint and the cluster's
`site.yaml` (the fast path, [INTAKE.md](INTAKE.md#2-the-fast-path)):

```bash
trtllm-modeling-v2 --checkpoint /models/newmodel --site site.yaml --workspace W \
    --goal "mtp disagg@tp1/tep4 tp1 tep4 dep4 isl=8192 osl=1024"
```

[docs/user-guide.md](docs/user-guide.md) takes an operator from a checkpoint to
a delivered target. **When something breaks, read
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) before you debug the target:** it is
keyed on the verbatim error text, and every entry there read as a model defect
while it was happening, and every one was the environment.

## Documents

| document | covers |
|---|---|
| [docs/user-guide.md](docs/user-guide.md) | operating a run: prerequisites, the spec, compute sessions, starting, watching and steering a run, tuning, standalone gates |
| [docs/design.md](docs/design.md) | how and why it works: the run's lifecycle, the contract and its gates, the task graph, integration, the substrate, transports, tuning, failure handling |
| [docs/developer-guide.md](docs/developer-guide.md) | changing the workflow: package layout, the workspace on disk, control flow, recipes, testing, conventions |
| [docs/mechanisms.md](docs/mechanisms.md) | each mechanism, the failure it exists for, its switch and default, and whether a GPU run has exercised it |
| [docs/examples.md](docs/examples.md) | one run followed end to end, with the artifacts it writes |
| [docs/results.md](docs/results.md) | measured onboarding and tuning results |
| [INTAKE.md](INTAKE.md) | Step 1: the user-facing `task.yaml`, `site.yaml`, the fast path, `intake derive\|check\|confirm\|compile` |
| [task.example.yaml](task.example.yaml) | every key of the resolved spec, with its default |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | symptoms keyed on the verbatim error, with causes and fixes |
| [BUILD_AND_TEST.md](BUILD_AND_TEST.md) | building TensorRT-LLM and running tests at CPU, 1-GPU, multi-GPU and multi-node scale |
| [LESSONS.md](LESSONS.md) | what real runs got wrong, and the mechanism that prevents each |
| [examples/](examples/) | one model's values for the generic features, for operators; a run is never handed them |

## How a run works

```
preflight  ->  planned run  ->  delivery
```

**Preflight** is Python. Before any agent turn it checks the spec, the
checkpoint, the host and the launch path, resolves the one launch prefix every
role's prompt carries, and writes `contract.yaml`: every check the run owes.
The stock-path cold start that proves the host generates a token runs beside
the first turns.

**The planned run.** A planner agent writes the task graph (`graph.yaml`);
nothing fixes stages or their order. The orchestrator, a Python loop, starts
tasks as capacity and their files allow, each in its own git worktree, rebases
and fast-forwards each finished task into the checkout, runs the contract
checks the task claims on a read-only snapshot of the commit, and wakes the
planner when something happens. Each task kind runs as one role:

| kind | role | does |
|---|---|---|
| `survey` | `surveyor` | maps the checkpoint's computation steps onto the catalog (`vocabulary.yaml`); each queue row becomes a kernel task |
| `kernel` | `kernel_writer` | one catalog entry, or a `promote` of local variants into the catalog |
| `assemble`, `feature`, `integrate`, `investigate` | `assembler` | the target's core, one feature, a merge, a diagnosis |
| `tune` | `tuner` turns under Python | the tune campaign, on the devices the build work leaves idle |
| `check` | none | contract checks the orchestrator runs with no agent turn |

**Checks are exit codes.** Every gate runs with `TRTLLM_MODELING_V2=require`:
boot and accuracy for each unit at its members, acceptance for a speculator,
and the witnesses, envelope, serve and logit gates where the spec owes them
([design.md §5](docs/design.md#5-the-contract-and-checks)). An agent reports
that it finished; it never reports that a check passed.

**Delivery.** The first commit on which the first unit's boot and accuracy
hold at its narrowest member gets the branch `<branch>-bringup`; the first on
which every required check holds, with every landed shared change through a
regression check, gets `<branch>-correct`, the correctness milestone;
`<branch>-best` names the commit the most cells hold on at once. Delivery then
checks open accuracy findings, the delivered files and tests, and the contract
on the target as it is. The tune campaign never holds correctness back: its
best branch is reported, never integrated.

## Flags

| flag | default | |
|---|---|---|
| `--task` | `--task` or `--checkpoint` | the resolved spec, or an intake `task.yaml` with `--site`. Validated before any agent is constructed |
| `--checkpoint` | `--task` or `--checkpoint` | a HuggingFace checkpoint: derive the task from its `config.json`, `--site` and `--goal` into `<workspace>/task.intake.yaml` and run it; its intake checks run in phase 0 |
| `--site` | none | the cluster's `site.yaml`: required with `--checkpoint`; with an intake `--task`, the two compile into `<workspace>/task.resolved.yaml` |
| `--goal` | none | with `--checkpoint`: features, members and workload in one line ([INTAKE.md](INTAKE.md#2-the-fast-path)); `-<feature>` leaves out one the checkpoint carries (write `--goal=-mtp ...` when the line starts with `-`) |
| `--confirm` [BY] | not confirmed | with `--checkpoint`: record that BY (default `$USER`) confirms the derived task. The run does not wait for it; an edit after the run started is refused until confirmed |
| `--workspace` | `workspace/trtllm-modeling-v2` | the run's records; on the shared filesystem under `session_spool` |
| `--dry-run` | off | preflight, then stop before the first agent turn |
| `--clean` | off | delete the checkpoint and the run's records, and start a new run |
| `--max-parallel-tasks` | `8` | agents working at once, the planner's slot among them; slots go by priority (`infra/execution/agent_gate.py`) |
| `--revision-cap` | `6` | graph revisions without a contract check turning green before the run closes `planner-stalled` |
| `--event-window` SECONDS | `60` | how long a task that finished with its checks green waits to reach the planner with the events after it; a failure wakes the planner at once |
| `--no-task-worktrees` | off | tasks edit the checkout directly instead of each in its own worktree |
| `--agent-transport` | `inprocess` | `inprocess` runs each role turn here; `external` writes it to `<workspace>/dispatch/` for a driver ([below](#running-where-the-orchestrator-cannot-authenticate-an-agent)) |
| `--agent-turn-timeout` | per role | seconds one role turn may take, for every role. Without it: assembler 16 h, tuner 8 h, kernel writer 6 h, surveyor 4 h, planner 2 h |
| `--no-primers` | off | start every planner, kernel writer and assembler turn afresh instead of as a fork of its role's briefing session (`run/primers.py`) |
| `--manage-sessions` | off | submit, renew and release the run's compute sessions with `sbatch` from this host (`infra/substrate/managed.py`) |
| `--wait-for-spool` MINUTES | `0` | under `session_spool`, wait up to MINUTES for the spool to appear before validating the task |
| `--skip-substrate-gate` | off | skip preflight's stock-path cold start, and with it the three failures it catches; for an architecture the stock path does not know, declare `environment.substrate_checkpoint` instead |
| `--stage` | none | `tune`: the tune campaign alone, on a target onboarded elsewhere or a closed run |

Subcommands:

| command | does |
|---|---|
| `trtllm-modeling-v2 tune --task task.yaml --workspace W` | the tune campaign alone, as `--stage tune` ([user guide §11](docs/user-guide.md#11-tune)) |
| `trtllm-modeling-v2 gate <name> --task task.yaml [--unit U] [--member M] [--commit C \| --tree T]` | one gate or witness, exactly as the orchestrator runs it ([user guide §12](docs/user-guide.md#12-running-one-gate-standalone)) |
| `trtllm-modeling-v2 tests {affected,compare} ...` | the test-result cache: the tests a change can affect, and the regressions against a base ([design.md §6.12](docs/design.md#612-the-test-cache)) |
| `trtllm-modeling-v2 spool {submit,launch,wait,status,cancel} ...` | jobs in a compute session's spool; the same program as `infra/substrate/spoolctl.py`, which a shell without the CLI runs by path |
| `trtllm-modeling-v2 intake {derive,check,confirm,compile} ...` | Step 1 ([INTAKE.md](INTAKE.md)) |
| `trtllm-modeling-v2 session {render,status,requests,submission,allocate,ack,watchdog} ...` | compute sessions ([below](#execution-substrate)) |
| `trtllm-modeling-v2 metrics --workspace W` | write the run's `metrics.json` now and print it |
| `trtllm-modeling-v2 feedback --workspace W "<message>"` (or `--file F`) | a message the planner of a running run reads first in its next turn; it steers the work and relaxes no check |
| `trtllm-modeling-v2 driver --workspace W {wait,pending,status,claim,release,prompt,answer,instructions,reload}` | the operator loop (`wait --sessions-only` for an in-process run), the external transport's turn hand-out, and `reload`, which loads changed flow code into the running orchestrator |

## Running where the orchestrator cannot authenticate an agent

By default (`--agent-transport inprocess`) the orchestrator starts one agent
CLI per turn, and preflight proves that each agent can take a turn here
([user guide §8](docs/user-guide.md#8-step-4-start-the-run)). On a sandboxed
host the spawned CLI cannot write its credential lock, and the turn fails with
`EROFS ... credentials.lock` reported as `stop_reason='stop_sequence'` and
**zero tokens**: it looks like a model refusal, but no agent started. A
credential store copied to a writable home shares a single-use refresh token
with the parent session, and one of the two loses within minutes.

Use `--agent-transport external`: an already-authenticated session runs the
turns. It moves the turn and nothing else (`infra/external/`). The
orchestrator writes `dispatch/<id>.system.md`, `dispatch/<id>.prompt.md` and
`dispatch/<id>.request.json`, then waits until a driver writes
`dispatch/<id>.result.md` and then `dispatch/<id>.done`. The driver is trusted
with nothing: every decision is validated from the decision file against the
role's closed enum, and every verdict comes from a gate the orchestrator ran.
What the driver must get right is that **the role's system prompt travels with
the turn**: a driver that delivers only the prompt runs a different agent, and
its answer still looks plausible. When `task.yaml` routes a role, the request
names its agent (`agent`). The driver's commands are in
[user guide §8](docs/user-guide.md#external-transport).

## Execution substrate

`environment.kind` is `local` (this host, the devices preflight claims) or
`session_spool`. On a Slurm/pyxis cluster a run stands on **compute
sessions**: allocations whose main process is the spool scheduler
(`infra/substrate/scheduler.py`). A job is a script renamed into
`<spool>/pending/`; its receipt is `done/<name>.rc`, written last, beside the
verdict in `done/<name>.json`. Declare the sessions in `environment.sessions`
and render each entry script; never edit one by hand:

```bash
trtllm-modeling-v2 session render     --task task.yaml --name gates-1   # writes <spool>/session_entry.sh
trtllm-modeling-v2 session submission --task task.yaml --name gates-1   # prints the call that starts it
trtllm-modeling-v2 session status     --task task.yaml                  # heartbeat, end and queue of each
```

Agents and operators submit with `spoolctl.py submit --gpus N|all --wall 2h
--tree <checkout> <name> '<cmd>'`. `--tree` runs the job from that checkout,
first on `PYTHONPATH`, and a result whose ranks imported another checkout is
refused (`reason=mixed_checkout`). A job sent to a spool that will not start
it soon gets a `warning:` naming the spools of the same kind that can;
`--pick` sends it to the one that can start it soonest. Stop a job with
`spoolctl.py cancel <name>`, never by signalling its processes: under **the
safe-kill rule** the scheduler never signals a process that carries the
allocation's MPI context, since that can take the whole allocation down.

A session with a `time_limit` asks for a successor before it ends. On a run
whose turns run in process, the operator answers those requests, and notices
the orchestrator stop, with `trtllm-modeling-v2 driver --workspace W wait
--sessions-only` ([user guide §6](docs/user-guide.md#6-step-2-compute-sessions)).
How the scheduler admits, places, limits and recovers jobs is in
[design.md §8](docs/design.md#8-execution-substrate); multi-node sessions are
in [infra/scripts/mn/README.md](infra/scripts/mn/README.md).

## Tests

```bash
cd agent-flow && pytest tests/workflows/trtllm_modeling_v2/
```

Every test runs on CPU, with no model, GPU or cluster;
[developer-guide.md §5](docs/developer-guide.md#5-testing) says how the tests
fake the outside world and which ones read the TensorRT-LLM checkout.
