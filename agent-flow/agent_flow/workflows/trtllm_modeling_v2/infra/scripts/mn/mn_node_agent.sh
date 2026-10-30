#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PRRTE `plm_ssh_agent` for a multi-node compute session: `mn_node_agent.sh <host> <cmd...>`.
#
# PRRTE starts one `prted` daemon per remote node by calling its agent with a
# host and a shell command, exactly as it would call `ssh`. There is no sshd and
# no srun in the container, so this agent hands the command to the target
# node's spool scheduler (the session entry maps each hostname to its spool
# under <session spool>/hosts/) and blocks until it ends, which is what an ssh
# session does. The daemon then connects back to mpirun and starts that node's
# ranks itself.
#
# The job holds no device (the node's devices are held by mn_mpirun.sh's hold
# job) and has no wall limit (it ends with mpirun).
set -u
BASE=${MV2FLOW_MN_BASE_SPOOL:?mn_node_agent: MV2FLOW_MN_BASE_SPOOL is not set; run it under mn_mpirun.sh}
host=${1:?usage: mn_node_agent.sh <host> <command...>}
shift
# PRRTE may pass options before the host; skip anything that starts with '-'.
while [[ "$host" == -* ]]; do host=${1:?}; shift; done
short=${host%%.*}
map="$BASE/hosts/$short"
[ -f "$map" ] || map="$BASE/hosts/$host"
if [ ! -f "$map" ]; then echo "mn_node_agent: no spool registered for host $host under $BASE/hosts" >&2; exit 255; fi
spool=$(cat "$map")
name="free-prted-$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM"
mkdir -p "$spool/pending"
tmp="$spool/pending/.$name.sh.tmp"
{
  echo "#!/bin/bash"
  echo "#spool: gpus=0 wall=0 resolve=off${MV2FLOW_JOB_PREFIX:+ owner=$MV2FLOW_JOB_PREFIX}"
  echo 'ulimit -c 0'
  [ -n "${MV2FLOW_MN_ENV:-}" ] && echo "source '$MV2FLOW_MN_ENV'"
  echo "echo \"mn_node_agent: prted on \$(hostname) at \$(date -u +%H:%M:%SZ)\""
  printf '%s\n' "$*"
} > "$tmp"
mv "$tmp" "$spool/pending/$name.sh"
echo "mn_node_agent: queued $name on $host ($spool)" >&2
while [ ! -f "$spool/done/$name.rc" ]; do sleep 1; done
exit "$(cat "$spool/done/$name.rc")"
