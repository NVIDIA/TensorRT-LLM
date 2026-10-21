# Multi-node launches on a multi-node compute session (optional)

A run that needs ranks on more than one node declares a session with `nodes: 2` (or more):

```yaml
environment:
  sessions:
    - name: mn
      spool: /shared/run/spool-mn
      role: gates
      nodes: 2
      gpus_per_node: 4
      time_limit: "12:00:00"
```

`trtllm-modeling-v2 session render --task task.yaml --name mn` writes `<spool>/session_entry.sh`.
Submit the allocation with `--nodes=2 --gpus-per-node=4` (and `--segment=2` where the cluster uses
block topology) and **no** `--ntasks-per-node`, so the batch step runs the entry once per node, each
copy in its own container. `trtllm-modeling-v2 session submission --task task.yaml --name mn` prints
the broker call. Each copy becomes a spool scheduler:

| node | spool | used for |
|---|---|---|
| 0 | `<spool>` | the session's spool: submit here |
| k | `<spool>-n<k>` | node k's spool: hold jobs and the remote MPI daemons |

Each node registers `<spool>/hosts/<hostname>`, which is how the launcher finds the other nodes.

## Launching across the nodes

```bash
spoolctl.py submit --spool <spool> --gpus all --tree <checkout> <name> \
  'bash <scripts>/mn/mn_mpirun.sh 4 python <script> <args>'
```

- **Rank layout.** `mn_mpirun.sh <ranks-per-node> <cmd>` starts `nodes x ranks-per-node` ranks:
  ranks 0-3 on node 0, 4-7 on node 1. A rank's GPU is `OMPI_COMM_WORLD_LOCAL_RANK`.
- **Devices.** The job holds node 0 (`--gpus all`) and takes each other node through a whole-node
  `mn-hold-*` job in that node's spool, so it never runs beside other work on either node. The hold
  ends when the launcher exits, or when the launching job's receipt appears.
- **Remote daemons.** There is no sshd and no srun in the container. PRRTE starts each remote
  `prted` through `mn_node_agent.sh`, which queues it as a `free-prted-*` job in the remote node's
  spool. Leave `mn-hold-*` and `free-prted-*` jobs alone; they end with the launch.
- **Working directory and tree.** Every rank starts in the job's working directory
  (`--wdir $PWD`), and ranks import from it ahead of `PYTHONPATH`; use `--tree`. The resolve-hook
  variables are forwarded, so the provenance check covers every rank on every node.
- **Environment.** Every rank-hosting process inherits its node's session environment. On top of
  it every `TRTLLM_*`, `TLLM_*`, `NCCL_*`, `UCX_*` variable, `PYTHONPATH` and the names in
  `MN_FORWARD` are forwarded with `-x`. `MV2FLOW_MN_ENV` names a file every one of them sources.
- **Output.** Each rank writes `<spool>/ranks/<job>-<time>/rank.<job>.<rank>.{out,err}`; output
  forwarded through `mpirun` from a remote node is slow (about 15 KB/s measured). The job log ends
  with rank 0's stdout, a key-line summary of every rank and the tail of every rank's stderr.
  `MN_RANK_OUTPUT=none` routes everything through `mpirun` instead.
- **Deadline.** Inside a spool job the job's own wall limit governs; outside one, `MN_TIMEOUT`
  (default 7200 s). Stopping the launcher is safe: it and its ranks carry none of the allocation's
  Slurm/PMIx context.
- **Tests that start their own `mpirun`** stay on one node. `mn_local_mpirun.sh <ranks> <cmd>`
  applies the same rules on a single node.

## Rules

- Never kill a job or a rank by hand. `spoolctl.py cancel <name>` stops a job under the safe-kill
  rule; otherwise let it end or let its wall limit end it.
- `pkill`/`kill` by pattern is never allowed: each node's scheduler is that node's allocation.
- A multi-node TP group larger than a node's GPU count all-reduces through NCCL; the IPC
  all-reduce workspace needs `tp_size <= gpus_per_node`.
