#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch one MPI job across the nodes of a multi-node compute session.
#
#   spoolctl.py submit --spool <session spool> --gpus all --tree <checkout> <name> \
#       'bash <scripts>/mn/mn_mpirun.sh <ranks-per-node> <command...>'
#
# Run it as a whole-node job in the session's own spool (node 0). It takes
# every other node's devices through a whole-node "hold" job in that node's
# spool, then starts `mpirun` with ranks 0..P-1 on node 0, P..2P-1 on node 1,
# and so on. The container has no sshd and no srun, so PRRTE starts its remote
# daemons through mn_node_agent.sh, which hands each one to the remote node's
# spool scheduler.
#
# Environment, all optional:
#   MN_NODES_EXTRA   how many other nodes to use (default 1)
#   MN_FORWARD       extra variable names to forward to every rank (space separated);
#                    TRTLLM_* TLLM_* NCCL_* UCX_* PYTHONPATH, the resolve-hook variables
#                    and CUDA_LAUNCH_BLOCKING are always forwarded
#   MN_RANK_OUTPUT   directory for per-rank stdout/stderr (default <spool>/ranks/<job>-<time>);
#                    `none` routes everything through mpirun instead (slow from remote nodes)
#   MN_TIMEOUT       seconds before mpirun is stopped; default 0 inside a spool job (the job's
#                    wall limit governs), 7200 outside one
#   MN_KEY_PATTERN   extra regex for the per-rank key-line summary
#   MN_MPIRUN_ARGS   extra mpirun arguments
#   MV2FLOW_MN_ENV a file every rank-hosting process sources first
#
# Every rank starts in the caller's working directory (`--wdir $PWD`): MPI
# ranks import from the working directory ahead of PYTHONPATH, so run it with
# --tree (or `cd <checkout> &&`).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PPN=${1:?usage: mn_mpirun.sh <ranks-per-node> <command...>}
shift
BASE=${MV2FLOW_MN_BASE_SPOOL:-${MV2FLOW_SPOOL:-}}
[ -n "$BASE" ] || { echo "mn_mpirun: run this inside a spool job of a multi-node session" >&2; exit 64; }
HOSTS="$BASE/hosts"
export MV2FLOW_MN_BASE_SPOOL=$BASE

for v in $(env | sed -n 's/^\(\(PMI\|PMIX\|OMPI\|SLURM\|PRTE\|PALS\|MPI_LOCALRANKID\|MPI_LOCALNRANKS\)[A-Za-z0-9_]*\)=.*/\1/p'); do
  unset "$v"
done
ulimit -c 0
# shellcheck disable=SC1090
[ -n "${MV2FLOW_MN_ENV:-}" ] && source "$MV2FLOW_MN_ENV"

self=$(hostname)
others=()
for f in "$HOSTS"/*; do
  [ -f "$f" ] || continue
  h=$(basename "$f")
  [ "$h" = "$self" ] && continue
  hb="$(cat "$f")/watcher.heartbeat"
  if [ -f "$hb" ] && [ $(( $(date +%s) - $(stat -c %Y "$hb") )) -le 60 ]; then
    others+=("$h")
  fi
done
if [ "${#others[@]}" -lt "${MN_NODES_EXTRA:-1}" ]; then
  echo "mn_mpirun: ${#others[@]} live node(s) registered in $HOSTS besides $self; need ${MN_NODES_EXTRA:-1}" >&2
  exit 70
fi
NODES=("$self" "${others[@]:0:${MN_NODES_EXTRA:-1}}")

# Hold every other node's devices for the whole run. The hold ends when this
# script exits, or -- if it is stopped without running its trap -- when this
# job's own receipt appears.
PARENT_RC="${MV2FLOW_SPOOL:-$BASE}/done/${MV2FLOW_SPOOL_JOB:-none}.rc"
held=()
release() { for f in "${held[@]}"; do touch "$f.release"; done; }
trap release EXIT
for h in "${NODES[@]:1}"; do
  sp=$(cat "$HOSTS/$h")
  mkdir -p "$sp/hold" "$sp/pending"
  name="mn-hold-$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM"
  flag="$sp/hold/$name"
  {
    echo "#!/bin/bash"
    echo "#spool: gpus=all wall=0${MV2FLOW_JOB_PREFIX:+ owner=$MV2FLOW_JOB_PREFIX}"
    echo "touch '$flag.held'"
    echo "while [ ! -f '$flag.release' ] && [ ! -f '$PARENT_RC' ]; do sleep 2; done"
    echo "rm -f '$flag.held' '$flag.release'"
  } > "$sp/pending/.$name.sh.tmp"
  mv "$sp/pending/.$name.sh.tmp" "$sp/pending/$name.sh"
  held+=("$flag")
  echo "mn_mpirun: $(date -u +%H:%M:%SZ) waiting for node $h's devices ($name)"
  while [ ! -f "$flag.held" ]; do sleep 2; done
done
hostlist=$(printf '%s:'"$PPN"',' "${NODES[@]}")
hostlist=${hostlist%,}
NP=$(( PPN * ${#NODES[@]} ))

fwd=()
for v in $(env | sed -n 's/^\(\(TRTLLM_\|TLLM_\|NCCL_\|UCX_\)[A-Za-z0-9_]*\)=.*/\1/p') \
    PYTHONPATH PYTHONNOUSERSITE CUDA_LAUNCH_BLOCKING MV2FLOW_TREE MV2FLOW_RESOLVE_LOG \
    MV2FLOW_RESOLVE_MODULES MV2FLOW_SPOOL_JOB ${MN_FORWARD:-}; do
  [ -n "${!v+x}" ] && fwd+=(-x "$v")
done

if [ -z "${MN_RANK_OUTPUT+x}" ]; then
  MN_RANK_OUTPUT="$BASE/ranks/${MV2FLOW_SPOOL_JOB:-mn}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[ "$MN_RANK_OUTPUT" = none ] && MN_RANK_OUTPUT=
out=()
if [ -n "$MN_RANK_OUTPUT" ]; then
  mkdir -p "$MN_RANK_OUTPUT"
  out=(--output "file=$MN_RANK_OUTPUT/rank:nocopy")
fi

if [ -n "${MV2FLOW_SPOOL_JOB:-}" ]; then DEFAULT_TIMEOUT=0; else DEFAULT_TIMEOUT=7200; fi
TO=()
[ "${MN_TIMEOUT:-$DEFAULT_TIMEOUT}" != 0 ] && TO=(timeout -k 60 "${MN_TIMEOUT:-$DEFAULT_TIMEOUT}")

echo "mn_mpirun: $(date -u +%H:%M:%SZ) np=$NP hosts=$hostlist forward=${fwd[*]:-none}"
# shellcheck disable=SC2086
"${TO[@]}" mpirun -np "$NP" --host "$hostlist" --map-by slot --bind-to none --oversubscribe \
  --allow-run-as-root --wdir "$PWD" \
  --prtemca plm ssh --prtemca plm_ssh_agent "$HERE/mn_node_agent.sh" --prtemca prte_tmpdir_base /tmp \
  "${out[@]}" ${MN_MPIRUN_ARGS:-} "${fwd[@]}" "$@"
rc=$?
[ "$rc" = 124 ] && echo "mn_mpirun: TIMED OUT after ${MN_TIMEOUT:-$DEFAULT_TIMEOUT}s -- the job was stopped, not failed"
if [ -n "$MN_RANK_OUTPUT" ]; then
  echo "mn_mpirun: rc=$rc $(date -u +%H:%M:%SZ); per-rank output under $MN_RANK_OUTPUT"
  for f in "$MN_RANK_OUTPUT"/*.0.out; do
    [ -f "$f" ] && { echo "--- $f"; cat "$f"; }
  done
  echo "--- key lines of every rank"
  grep -h -E "\[mv2flow-resolve\]|Traceback|Error${MN_KEY_PATTERN:+|$MN_KEY_PATTERN}" \
      "$MN_RANK_OUTPUT"/*.out "$MN_RANK_OUTPUT"/*.err 2>/dev/null \
    | grep -v "UCP worker\|uct_iface\|pml_ucx" | sort | uniq -c | cut -c1-300 | head -80
  for f in "$MN_RANK_OUTPUT"/*.err; do
    [ -f "$f" ] && { echo "--- tail $f"; tail -3 "$f" | cut -c1-300; }
  done
fi
exit $rc
